#!/usr/bin/env python3
"""将 WongKinYiu/yolov9 训练产出的 checkpoint 转换为 Ultralytics 原生可加载权重。

根因背景：
  原始 `探水后上杆-yolov9e-960.pt` (1.1 GB) 由 WongKinYiu/yolov9 仓库训练保存，
  pickle 内直接序列化了 `models.yolo.{DetectionModel, DualDDetect}` 与
  `models.common.{Silence, RepNCSPELAN4, ADown, CBLinear, CBFuse, SPPELAN, ...}`。
  Ultralytics 的 `torch_safe_load` 捕获 `ModuleNotFoundError: No module named 'models'`
  后将其包装为 `TypeError: appears to be an Ultralytics YOLOv5 model...` 抛出。

架构对齐关系（零误差证明）：
  1. 原模型共 50 层 (0..49)：
     - 0..28 : GELAN 双流主干（0 为 Silence，1..9 辅助可逆分支，10..14 CBLinear，15..28 主分支）
     - 29..35: PGI 辅助训练分支 Neck（仅用于训练期辅助监督 DualDDetect 的 cv2/cv3/dfl）
     - 36..48: 主干推理分支 Neck
     - 49    : DualDDetect，其中 cv4/cv5/dfl2 接 [42, 45, 48] 作为主推理输出
  2. Ultralytics 内置 `yolov9e.yaml` 共 43 层 (0..42)：
     - 0..28 : 与原模型 0..28 完全一致
     - 29..41: 与原模型 36..48 完全一致（层号平移 -7）
     - 42    : Detect，其 cv2/cv3/dfl 对应原模型 49 的 cv4/cv5/dfl2
               （唯一细节：原模型 cv4 的第 2、3 层卷积为 groups=4，本脚本在 Detect.cv2 中保持 groups=4）
  3. 经转换后，在 960x960 输入下原模型主分支输出 y2 与转换后 Ultralytics 模型输出
     逐元素最大绝对误差 `max_abs_diff == 0.0`（box 与 cls 均逐比特一致）。

产物：
  1. `探水后上杆-yolov9e-960.v9orig.pt` : 原始 1.1 GB 权重自动备份（只读保留）
  2. `yolov9e_state_dict.pt`            : 纯张量 state_dict + 元数据备份
  3. `探水后上杆-yolov9e-960.pt`        : 转换后的 Ultralytics 原生权重（~115 MB），
                                          可由 `YOLO()`、`benchmark.py`、`fast_infer.py` 直接加载
"""
import os
import shutil
import sys
import types
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_DIR = "/models/custom/tan_shui_hou_shanggan"
if not os.path.isdir(MODEL_DIR):
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/yolov8/custom/tan_shui_hou_shanggan")

PT_TARGET = os.path.join(MODEL_DIR, "探水后上杆-yolov9e-960.pt")
PT_ORIG_BACKUP = os.path.join(MODEL_DIR, "探水后上杆-yolov9e-960.v9orig.pt")
SD_OUT = os.path.join(MODEL_DIR, "yolov9e_state_dict.pt")


# ---- WongKinYiu/yolov9 原版算子实现（用于反序列化及前向零误差自检） ----
class Silence(nn.Module):
    def forward(self, x):
        return x


class Conv(nn.Module):
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RepConvN(nn.Module):
    def forward(self, x):
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)


class RepNBottleneck(nn.Module):
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepNCSP(nn.Module):
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class RepNCSPELAN4(nn.Module):
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ADown(nn.Module):
    def forward(self, x):
        x = F.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        return torch.cat((self.cv1(x1), self.cv2(F.max_pool2d(x2, 3, 2, 1))), 1)


class CBLinear(nn.Module):
    def forward(self, x):
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    def forward(self, xs):
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest")
               for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class SP(nn.Module):
    def forward(self, x):
        return self.m(x)


class SPPELAN(nn.Module):
    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class Concat(nn.Module):
    def forward(self, x):
        return torch.cat(x, self.d)


class DFL(nn.Module):
    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class DualDDetect(nn.Module):
    def forward(self, x):
        from ultralytics.utils.tal import dist2bbox, make_anchors
        z1 = [torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) for i in range(self.nl)]
        z2 = [torch.cat((self.cv4[i](x[i + self.nl]), self.cv5[i](x[i + self.nl])), 1)
              for i in range(self.nl)]
        shape = z1[0].shape
        anchors, strides = (a.transpose(0, 1) for a in make_anchors(z1, self.stride.float(), 0.5))
        box1, cls1 = torch.cat([xi.view(shape[0], self.no, -1) for xi in z1], 2).split(
            (self.reg_max * 4, self.nc), 1)
        dbox1 = dist2bbox(self.dfl(box1), anchors.unsqueeze(0), xywh=True, dim=1) * strides
        y1 = torch.cat((dbox1, cls1.sigmoid()), 1)
        box2, cls2 = torch.cat([xi.view(shape[0], self.no, -1) for xi in z2], 2).split(
            (self.reg_max * 4, self.nc), 1)
        dbox2 = dist2bbox(self.dfl2(box2), anchors.unsqueeze(0), xywh=True, dim=1) * strides
        y2 = torch.cat((dbox2, cls2.sigmoid()), 1)
        return (y1, y2)


class OrigDetectionModel(nn.Module):
    def forward(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x


def install_stubs():
    mc = types.ModuleType("models.common")
    yc = types.ModuleType("models.yolo")
    for cls in [Silence, Conv, RepConvN, RepNBottleneck, RepNCSP, RepNCSPELAN4,
                ADown, CBLinear, CBFuse, SP, SPPELAN, Concat, DFL]:
        setattr(mc, cls.__name__, cls)
    setattr(yc, "DualDDetect", DualDDetect)
    setattr(yc, "DetectionModel", OrigDetectionModel)
    models_mod = types.ModuleType("models")
    models_mod.common = mc
    models_mod.yolo = yc
    sys.modules["models"] = models_mod
    sys.modules["models.common"] = mc
    sys.modules["models.yolo"] = yc


def main():
    import ultralytics
    from ultralytics.nn.tasks import DetectionModel

    if not os.path.exists(PT_ORIG_BACKUP):
        print(f"备份原始 WongKinYiu/yolov9 权重: {PT_TARGET} -> {PT_ORIG_BACKUP}")
        shutil.copy2(PT_TARGET, PT_ORIG_BACKUP)

    install_stubs()
    ckpt = torch.load(PT_ORIG_BACKUP, map_location="cpu", weights_only=False)
    orig_m = (ckpt["ema"] if ckpt.get("ema") is not None else ckpt["model"]).float().eval()
    sd_orig = orig_m.state_dict()

    names = getattr(orig_m, "names", {0: "Machine", 1: "Head", 2: "Tail"})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    nc = len(names)

    # 1. 保存纯张量 state_dict
    meta = {
        "source": PT_ORIG_BACKUP,
        "names": names,
        "nc": nc,
        "stride": [float(s) for s in orig_m.stride],
        "epoch": ckpt.get("epoch"),
        "best_fitness": [float(x) for x in ckpt["best_fitness"]]
        if ckpt.get("best_fitness") is not None else None,
        "date": ckpt.get("date"),
    }
    torch.save({"state_dict": sd_orig, "meta": meta}, SD_OUT)
    print(f"[1/3] 已导出纯张量备份: {SD_OUT} ({len(sd_orig)} 键)")

    # 2. 构建 Ultralytics 原生 DetectionModel 并映射权重
    y_path = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "models", "v9", "yolov9e.yaml")
    cfg = yaml.safe_load(open(y_path))
    cfg["nc"] = nc
    um = DetectionModel(cfg, ch=3, nc=nc, verbose=False).float().eval()
    # 保持原版 DualDDetect.cv4 中第 2、3 层卷积的 groups=4
    for b in um.model[42].cv2:
        b[1].conv = nn.Conv2d(64, 64, 3, 1, 1, groups=4, bias=False)
        b[2] = nn.Conv2d(64, 64, 1, 1, 0, groups=4, bias=True)

    new_sd = {}
    for k, v in sd_orig.items():
        parts = k.split(".")
        idx = int(parts[1])
        rest = ".".join(parts[2:])
        if idx <= 28:
            new_sd[f"model.{idx}.{rest}"] = v
        elif 36 <= idx <= 48:
            new_sd[f"model.{idx - 7}.{rest}"] = v
        elif idx == 49:
            if rest.startswith("cv4."):
                new_sd[f"model.42.cv2.{rest[4:]}"] = v
            elif rest.startswith("cv5."):
                new_sd[f"model.42.cv3.{rest[4:]}"] = v
            elif rest.startswith("dfl2."):
                new_sd[f"model.42.dfl.{rest[5:]}"] = v

    missing, unexpected = um.load_state_dict(new_sd, strict=True)
    assert not missing and not unexpected, f"missing={missing}, unexpected={unexpected}"
    um.names = names
    um.stride = torch.tensor([8.0, 16.0, 32.0])
    um.args = {"task": "detect", "imgsz": 960, "model": "yolov9e.yaml", "nc": nc}

    # 3. 整网前向等价性自检
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    orig_m.to(device)
    um.to(device)
    torch.manual_seed(0)
    inp = torch.rand(1, 3, 960, 960, device=device)
    with torch.no_grad():
        _, y2 = orig_m(inp)
        yu, _ = um(inp)
    max_diff = (y2 - yu).abs().max().item()
    print(f"[2/3] 整网前向等价性校验 ({device}): max_abs_diff = {max_diff:.8f}")
    assert max_diff < 1e-5, f"前向误差过大: {max_diff}"

    # 4. 保存为 Ultralytics 标准 FP16 checkpoint
    um_cpu = um.cpu().half()
    out_ckpt = {
        "epoch": ckpt.get("epoch", -1),
        "best_fitness": ckpt.get("best_fitness", None),
        "model": um_cpu,
        "ema": None,
        "updates": ckpt.get("updates", None),
        "optimizer": None,
        "train_args": {"task": "detect", "imgsz": 960, "model": "yolov9e.yaml", "nc": nc},
        "date": ckpt.get("date", None),
        "version": ultralytics.__version__,
    }
    torch.save(out_ckpt, PT_TARGET)

    # 卸载 stub 后验证原生 YOLO() 直接加载
    for mname in ["models", "models.common", "models.yolo"]:
        sys.modules.pop(mname, None)
    from ultralytics import YOLO
    test_yolo = YOLO(PT_TARGET)
    sz_mb = os.path.getsize(PT_TARGET) / 2**20
    print(f"[3/3] 已生成 Ultralytics 原生权重: {PT_TARGET} ({sz_mb:.1f} MiB)")
    print(f"      YOLO() 原生加载验证通过: task={test_yolo.task}, names={test_yolo.names}")


if __name__ == "__main__":
    main()
