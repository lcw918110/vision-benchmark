#!/usr/bin/env python3
"""可追溯的同卷评测：Ultralytics detect/obb；结果协议供自动比较和后续界面使用。"""
import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 'vision-benchmark/run-v1'
SUPPORTED = {'detect', 'obb'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_dataset(root):
    """对实际图像和标注内容取摘要，不依赖 manifest 中的声明。"""
    root = Path(root)
    images, labels = root / 'images', root / 'labels'
    if not images.is_dir() or not labels.is_dir():
        raise ValueError('数据集必须包含 images 和 labels 目录')
    paths = sorted(p for p in images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError('数据集没有可评测图片')
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError('不同图片使用相同文件名，标注无法唯一对应')
    h = hashlib.sha256()
    for image in paths:
        label = labels / (image.stem + '.txt')
        if not label.is_file():
            raise ValueError(f'标注缺失（背景图请提供空标注文件）: {label}')
        for path in (image, label):
            h.update(path.relative_to(root).as_posix().encode('utf-8') + b'\0')
            h.update(bytes.fromhex(sha256(path)))
    return {'sha256': h.hexdigest(), 'images': len(paths)}


def canonical_names(names):
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ValueError('类别名称必须是非空列表或映射')
    try:
        result = {str(int(k)): str(v).strip() for k, v in names.items()}
    except (ValueError, TypeError) as e:
        raise ValueError('类别 ID 必须是整数') from e
    if len(result) != len(names) or any(not v for v in result.values()):
        raise ValueError('类别 ID 重复或类别名称为空')
    return dict(sorted(result.items(), key=lambda x: int(x[0])))


def check_names(dataset_names, model_names):
    if canonical_names(dataset_names) != canonical_names(model_names):
        raise ValueError('模型与数据集类别 ID/名称不完全一致；目前不支持自动类别映射')


def comparable(a, b):
    """缺少任一关键字段即拒绝比较，不猜测历史结果的口径。"""
    reasons = []
    if a.get('schema') != SCHEMA or b.get('schema') != SCHEMA:
        reasons.append('结果协议不支持')
    if a.get('status') != 'success' or b.get('status') != 'success':
        reasons.append('存在未成功的运行')
    for record in (a, b):
        if not isinstance(record.get('model'), dict) or not record['model'].get('weights_sha256'):
            reasons.append('缺少模型权重摘要')
        metrics = record.get('metrics')
        if not isinstance(metrics, dict) or any(not isinstance(metrics.get(k), (int, float)) or isinstance(metrics.get(k), bool) or not math.isfinite(metrics[k]) or not 0 <= metrics[k] <= 1 for k in ('mAP50', 'mAP50_95')):
            reasons.append('缺少有效的精度指标')
    for section, keys in {
        'dataset': ('sha256', 'images', 'names'),
        'protocol': ('task', 'adapter', 'imgsz', 'batch', 'precision', 'confidence', 'nms_iou', 'max_det', 'split', 'device', 'metric_version'),
        'environment': ('python', 'ultralytics', 'torch', 'cuda', 'gpu'),
        'provenance': ('evaluator_sha256',),
    }.items():
        for key in keys:
            left, right = a.get(section), b.get(section)
            x = left.get(key) if isinstance(left, dict) else None
            y = right.get(key) if isinstance(right, dict) else None
            if x is None or y is None or x != y:
                reasons.append(f'{section}.{key} 缺失或不一致')
    return reasons


def assert_output_inside_project(path):
    """宿主运行时禁止评测输出写入项目外；容器仅允许 /output 映射。"""
    path = Path(path).resolve()
    roots = [Path('/output').resolve()] if Path('/configs/benchmark_v1.py').exists() else [Path(__file__).resolve().parent.parent]
    if not any(path == root or root in path.parents for root in roots):
        raise ValueError(f'输出路径必须位于项目输出目录: {path}')
    return path


def git_revision():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True, timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment(device):
    import torch
    import ultralytics
    cpu = str(device).lower() == 'cpu'
    if not cpu and (not str(device).isdigit() or not torch.cuda.is_available()):
        raise ValueError('请使用单个 CUDA 设备编号，或指定 cpu')
    gpu = 'cpu' if cpu else torch.cuda.get_device_name(int(device))
    return {'python': platform.python_version(), 'ultralytics': ultralytics.__version__,
            'torch': torch.__version__, 'cuda': torch.version.cuda or 'none', 'gpu': gpu}


def run(args):
    record = {'schema': SCHEMA, 'run_id': uuid.uuid4().hex, 'status': 'running',
              'started_at': datetime.now(timezone.utc).isoformat(), 'dataset': {}, 'model': {},
              'protocol': {'task': args.task, 'adapter': 'ultralytics-yolo', 'imgsz': args.imgsz,
                           'batch': args.batch, 'precision': args.precision, 'confidence': args.conf,
                           'nms_iou': args.iou, 'max_det': args.max_det, 'split': 'val', 'device': args.device,
                           'metric_version': 'ultralytics-box-val-v1'},
              'environment': {}, 'metrics': {}, 'warnings': [
                  '未验证训练数据泄漏；需独立审查模型来源。',
                  'P/R 是 Ultralytics 验证输出，不宣称为固定工作点指标。',
                  '计时由 Ultralytics 报告，不含完整业务流水线。'],
              'provenance': {'git_revision': git_revision(), 'evaluator_sha256': sha256(__file__)}}
    out = assert_output_inside_project(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / (record['run_id'] + '.json')
    started = time.monotonic()
    try:
        if args.task not in SUPPORTED:
            raise ValueError('仅支持 detect 和 obb；其他任务尚未实现')
        if args.imgsz <= 0 or args.batch <= 0 or args.max_det <= 0 or not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
            raise ValueError('输入尺寸、批量、最大检测数或阈值无效')
        import yaml
        from ultralytics import YOLO
        root = Path(args.dataset).resolve()
        data = yaml.safe_load((root / 'data.yaml').read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('data.yaml 格式无效')
        # 保证 Ultralytics 使用的路径和实际哈希的数据一致。
        if data.get('val') != 'images':
            raise ValueError('当前仅支持 val: images 的标准集')
        record['dataset'] = dict(fingerprint_dataset(root), name=root.name,
                                 names=canonical_names(data.get('names')), yaml_sha256=sha256(root / 'data.yaml'))
        weight = Path(args.weights).resolve(strict=True)
        record['model'] = {'name': args.name or weight.stem, 'weights_sha256': sha256(weight), 'weights_path': str(weight)}
        record['environment'] = environment(args.device)
        model = YOLO(str(weight))
        if model.task != args.task:
            raise ValueError(f'模型任务 {model.task} 与指定任务 {args.task} 不一致')
        check_names(record['dataset']['names'], model.names)
        # 创建运行专属 yaml，避免原文件的 path 指向其他样本。
        run_yaml = out / (record['run_id'] + '.yaml')
        run_yaml.write_text(yaml.safe_dump({'path': str(root), 'train': 'images', 'val': 'images',
                                             'names': {int(k): v for k, v in record['dataset']['names'].items()}}, allow_unicode=True), encoding='utf-8')
        try:
            result = model.val(data=str(run_yaml), imgsz=args.imgsz, batch=args.batch, half=args.precision == 'fp16',
                               conf=args.conf, iou=args.iou, max_det=args.max_det, device=args.device,
                               workers=0, plots=False, save=False, verbose=False,
                               project=str(out), name=record['run_id'], exist_ok=True)
        finally:
            run_yaml.unlink(missing_ok=True)
        box = result.box
        if len(box.ap_class_index) == 0:
            raise ValueError('数据集中没有可用于计算 AP 的标注，不能将无样本结果记为零')
        per_class = {}
        for i, cid in enumerate(box.ap_class_index):
            per_class[str(int(cid))] = {'ap50': float(box.ap50[i]), 'ap50_95': float(box.ap[i]),
                                        'precision': float(box.p[i]), 'recall': float(box.r[i])}
        record['metrics'] = {'mAP50': float(box.map50), 'mAP50_95': float(box.map),
                             'precision': float(box.mp), 'recall': float(box.mr),
                             'per_class': per_class, 'speed_ms_per_image': dict(result.speed)}
        record['status'] = 'success'
    except Exception as exc:
        record['status'] = 'failed'
        record['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    record['duration_s'] = round(time.monotonic() - started, 3)
    record['finished_at'] = datetime.now(timezone.utc).isoformat()
    try:
        text = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)
    except (ValueError, TypeError) as exc:
        record['status'] = 'failed'
        record['metrics'] = {}
        record['error'] = {'type': type(exc).__name__, 'message': '评测结果无法序列化为有效 JSON: ' + str(exc)}
        text = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)
    result_path.write_text(text, encoding='utf-8')
    print(f"{record['status']}: {result_path}")
    return 0 if record['status'] == 'success' else 1


def compare(args):
    records = [json.loads(Path(p).read_text(encoding='utf-8')) for p in args.results]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError('结果文件必须包含 JSON 对象')
    if len(records) < 2:
        raise ValueError('至少需要两个结果')
    errors = []
    for i, record in enumerate(records[1:], 1):
        reasons = comparable(records[0], record)
        if reasons:
            errors.append({'index': i, 'run_id': record.get('run_id'), 'reasons': reasons})
    report = {'schema': 'vision-benchmark/comparison-v1', 'comparable': not errors,
              'reasons': errors, 'run_ids': [r.get('run_id') for r in records],
              'candidates': [{'run_id': r.get('run_id'), 'model': r.get('model'),
                              'metrics': r.get('metrics')} for r in records] if not errors else []}
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.out:
        dest = assert_output_inside_project(args.out)
        if dest.resolve() in {Path(p).resolve() for p in args.results}:
            raise ValueError('比较输出不能覆盖输入结果')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text + '\n', encoding='utf-8')
    print(text)
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--dataset', required=True, help='容器内标准集目录')
    p.add_argument('--weights', required=True, help='容器内模型权重')
    p.add_argument('--name', default='')
    p.add_argument('--task', choices=sorted(SUPPORTED), required=True)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    p.add_argument('--conf', type=float, default=0.001, help='AP 验证置信度阈值')
    p.add_argument('--iou', type=float, default=0.7, help='NMS IoU，不是匹配 IoU')
    p.add_argument('--max-det', type=int, default=300)
    p.add_argument('--device', default='0')
    p.add_argument('--out', default='/output/benchmark/runs')
    p = sub.add_parser('compare')
    p.add_argument('results', nargs='+')
    p.add_argument('--out', default='', help='将比较结果保存为 JSON 文件')
    args = parser.parse_args()
    try:
        return run(args) if args.command == 'run' else compare(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f'错误: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
