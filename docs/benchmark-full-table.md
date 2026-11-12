# 矿山视觉算法横向评测整表台账（A～Z 共 26 列 × 45 行）

> **说明**：本文件为项目标准归档文档 `docs/benchmark-full-table.md`，由 `scripts/05-sync-benchmark-table.py` 从 `configs/registry.py` 与 `output/benchmark/arena/*.json` 自动同步生成。同一业务数据集共享的列已自动纵向合并（`rowspan`），仅按标签独立的列（行号、标签、类别ID、标注对象数、单类AP）逐行展开。配套指标规范见 [`docs/metrics-and-conventions.md`](./metrics-and-conventions.md)，标准测试集详情见 [`docs/dataset-standard-report.md`](./dataset-standard-report.md)。

<table>
  <thead>
    <tr>
      <th>行号</th>
      <th>数据集名称</th>
      <th>标注方式</th>
      <th>数据集路径</th>
      <th>标签</th>
      <th>标签Yolo类别</th>
      <th>共: 1602511 张</th>
      <th>标注对象: 4670797 个</th>
      <th>yolo版本</th>
      <th>权重</th>
      <th>本项目权重地址(宿主机)</th>
      <th>容器内权重路径</th>
      <th>具体算法架构</th>
      <th>默认推理分辨率</th>
      <th>权重体积(MB)</th>
      <th>参数量 / 计算量(GFLOPs)</th>
      <th>标准测试集路径(张数)</th>
      <th>标准集精度 mAP50/50-95 (FP16/INT8)</th>
      <th>标准集查准率P / 召回率R</th>
      <th>单类AP50 / AP50-95(对应D列标签)</th>
      <th>官方基线FPS(b1/FP32)</th>
      <th>优化最佳FPS(配置/加速比)</th>
      <th>纯GPU前向极限FPS(FP16/b16)</th>
      <th>实测状态与备注</th>
      <th>分尺度召回率(小/中/大目标)</th>
      <th>误报率FAR(背景图误报统计)</th>
      <th>单帧P99延迟 / C++静态图加速</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>2</strong></td>
      <td rowspan="3"><strong>人安全帽</strong></td>
      <td rowspan="3">目标检测</td>
      <td rowspan="3">/home/jykj/Dataset/Hbb/PersonHelmet</td>
      <td>人</td>
      <td>2</td>
      <td rowspan="3">309422</td>
      <td>790421</td>
      <td rowspan="3">yolo11x</td>
      <td rowspan="3">models/yolov8/custom/person/yolo11x_960_best.pt</td>
      <td rowspan="3">models/yolov8/custom/person/yolo11x_960_best.pt</td>
      <td rowspan="3">/models/custom/person/yolo11x_960_best.pt</td>
      <td rowspan="3">YOLO11x</td>
      <td rowspan="3">960</td>
      <td rowspan="3">109.1 MB</td>
      <td rowspan="3">56.84M / 437.4 GFLOPs@960 (194.4@640)</td>
      <td rowspan="3">data/benchmark/standard/PersonHelmet (600张)</td>
      <td rowspan="3">0.9778 / 0.7625 (FP16:0.9778 | INT8:0.9717)</td>
      <td rowspan="3">0.9622 / 0.9372</td>
      <td>0.9782 / 0.8121 (P=0.9439, R=0.9358)</td>
      <td rowspan="3">20.2 FPS</td>
      <td rowspan="3">103.0 FPS (FP16/b16, 5.10x)</td>
      <td rowspan="3">125.7 FPS</td>
      <td rowspan="3">已拿真实数据跑通；覆盖330矿区，FP16逐框校验0漏检</td>
      <td rowspan="3">小:0.909 / 中:0.964 / 大:0.984</td>
      <td rowspan="3">4.98% (14框/243背景图)</td>
      <td rowspan="3">b1 P99=27.5ms | C++ JIT b1: 61.8 FPS (2.01x)</td>
    </tr>
    <tr>
      <td><strong>3</strong></td>
      <td>戴安全帽</td>
      <td>3</td>
      <td>278960</td>
      <td>0.9712 / 0.7075 (P=0.9527, R=0.9215)</td>
    </tr>
    <tr>
      <td><strong>4</strong></td>
      <td>未戴安全帽</td>
      <td>8</td>
      <td>300719</td>
      <td>0.9840 / 0.7681 (P=0.9899, R=0.9542)</td>
    </tr>
    <tr>
      <td><strong>5</strong></td>
      <td rowspan="2"><strong>人睡觉</strong></td>
      <td rowspan="2">目标检测</td>
      <td rowspan="2">/home/jykj/Dataset/Hbb/PersonSleep</td>
      <td>人睡觉</td>
      <td>0</td>
      <td rowspan="2">31216</td>
      <td>6224</td>
      <td rowspan="2">yolov8m</td>
      <td rowspan="2">models/yolov8/custom/20261030-99人睡觉yolov8m/20261030-99.pt</td>
      <td rowspan="2">models/yolov8/custom/20261030-99人睡觉yolov8m/20261030-99.pt</td>
      <td rowspan="2">/models/custom/20261030-99人睡觉yolov8m/20261030-99.pt</td>
      <td rowspan="2">YOLOv8m</td>
      <td rowspan="2">640</td>
      <td rowspan="2">49.6 MB</td>
      <td rowspan="2">25.84M / 78.7 GFLOPs@640</td>
      <td rowspan="2">data/benchmark/standard/PersonSleep (500张)</td>
      <td rowspan="2">0.9939 / 0.8852 (FP16:0.9939 | INT8:0.9938)</td>
      <td rowspan="2">0.9898 / 0.9909</td>
      <td>0.9936 / 0.8845 (P=0.9855, R=0.9865)</td>
      <td rowspan="2">27.2 FPS</td>
      <td rowspan="2">259.2 FPS (FP16/b16, 9.53x)</td>
      <td rowspan="2">756.4 FPS</td>
      <td rowspan="2">已拿真实数据跑通；覆盖21矿区(100%)，1080p视频流实测45.0 FPS，FP16校验0漏检</td>
      <td rowspan="2">小:0.995 / 中:0.997 / 大:1.000</td>
      <td rowspan="2">1.61% (3框/23背景图)</td>
      <td rowspan="2">b1 P99=13.4ms | C++ JIT b1: 116.6 FPS (1.9x)</td>
    </tr>
    <tr>
      <td><strong>6</strong></td>
      <td>人</td>
      <td>2</td>
      <td>44514</td>
      <td>0.9942 / 0.8858 (P=0.9942, R=0.9953)</td>
    </tr>
    <tr>
      <td><strong>7</strong></td>
      <td rowspan="4"><strong>人工作服</strong></td>
      <td rowspan="4">目标检测</td>
      <td rowspan="4">/home/jykj/Dataset/Hbb/WorkclothesHelmet</td>
      <td>穿工作服的人</td>
      <td>0</td>
      <td rowspan="4">5158</td>
      <td>4730</td>
      <td rowspan="4">yolov8m</td>
      <td rowspan="4">models/yolov8/custom/20260806-99工作服安全帽yolov8m-v1.0/20260806-99.pt</td>
      <td rowspan="4">models/yolov8/custom/20260806-99工作服安全帽yolov8m-v1.0/20260806-99.pt</td>
      <td rowspan="4">/models/custom/20260806-99工作服安全帽yolov8m-v1.0/20260806-99.pt</td>
      <td rowspan="4">YOLOv8m</td>
      <td rowspan="4">640</td>
      <td rowspan="4">49.6 MB</td>
      <td rowspan="4">25.84M / 78.7 GFLOPs@640</td>
      <td rowspan="4">data/benchmark/standard/WorkclothesHelmet (500张)</td>
      <td rowspan="4">0.9890 / 0.8689 (FP16:0.9891 | INT8:0.9877)</td>
      <td rowspan="4">0.9854 / 0.9799</td>
      <td>0.9947 / 0.9412 (P=0.9922, R=0.9867)</td>
      <td rowspan="4">25.0 FPS</td>
      <td rowspan="4">210.5 FPS (FP16/b16, 8.42x)</td>
      <td rowspan="4">758.2 FPS</td>
      <td rowspan="4">已拿真实数据跑通；覆盖13矿区(100%)，FP16逐框校验0漏检</td>
      <td rowspan="4">小:0.974 / 中:0.995 / 大:0.996</td>
      <td rowspan="4">2.65% (0框/2背景图)</td>
      <td rowspan="4">b1 P99=13.5ms | C++ JIT b1: 116.6 FPS (1.9x)</td>
    </tr>
    <tr>
      <td><strong>8</strong></td>
      <td>未穿工作服</td>
      <td>2</td>
      <td>3935</td>
      <td>0.9936 / 0.9277 (P=0.9867, R=0.9912)</td>
    </tr>
    <tr>
      <td><strong>9</strong></td>
      <td>戴安全帽</td>
      <td>3</td>
      <td>5944</td>
      <td>0.9834 / 0.8002 (P=0.9808, R=0.9685)</td>
    </tr>
    <tr>
      <td><strong>10</strong></td>
      <td>未戴安全帽</td>
      <td>8</td>
      <td>2107</td>
      <td>0.9841 / 0.8066 (P=0.9820, R=0.9731)</td>
    </tr>
    <tr>
      <td><strong>11</strong></td>
      <td rowspan="8"><strong>人乘猴车</strong></td>
      <td rowspan="8">目标检测</td>
      <td rowspan="8">/home/jykj/Dataset/Hbb/RenChengHouChe</td>
      <td>人乘猴车</td>
      <td>0</td>
      <td rowspan="8">8386</td>
      <td>2467</td>
      <td rowspan="8">无</td>
      <td rowspan="8">无</td>
      <td rowspan="8">无</td>
      <td rowspan="8">无</td>
      <td rowspan="8">待交付权重</td>
      <td rowspan="8">—</td>
      <td rowspan="8">—</td>
      <td rowspan="8">—</td>
      <td rowspan="8">data/benchmark/standard/RenChengHouChe (500张)</td>
      <td rowspan="8">—</td>
      <td rowspan="8">—</td>
      <td>—</td>
      <td rowspan="8">—</td>
      <td rowspan="8">—</td>
      <td rowspan="8">—</td>
      <td rowspan="8">全量8386张数据预检0损坏，标准测试集(500张，全类别覆盖)已就绪，待上游交付.pt权重</td>
      <td rowspan="8">—</td>
      <td rowspan="8">—</td>
      <td rowspan="8">—</td>
    </tr>
    <tr>
      <td><strong>12</strong></td>
      <td>睡觉</td>
      <td>1</td>
      <td>828</td>
      <td>—</td>
    </tr>
    <tr>
      <td><strong>13</strong></td>
      <td>人</td>
      <td>2</td>
      <td>11571</td>
      <td>—</td>
    </tr>
    <tr>
      <td><strong>14</strong></td>
      <td>戴安全帽</td>
      <td>3</td>
      <td>8098</td>
      <td>—</td>
    </tr>
    <tr>
      <td><strong>15</strong></td>
      <td>超长异物</td>
      <td>4</td>
      <td>596</td>
      <td>—</td>
    </tr>
    <tr>
      <td><strong>16</strong></td>
      <td>脚蹬地</td>
      <td>5</td>
      <td>3832</td>
      <td>—</td>
    </tr>
    <tr>
      <td><strong>17</strong></td>
      <td>未戴安全帽</td>
      <td>8</td>
      <td>675</td>
      <td>—</td>
    </tr>
    <tr>
      <td><strong>18</strong></td>
      <td>打电话</td>
      <td>9</td>
      <td>584</td>
      <td>—</td>
    </tr>
    <tr>
      <td><strong>19</strong></td>
      <td rowspan="8"><strong>皮带原版</strong></td>
      <td rowspan="8">目标检测</td>
      <td rowspan="8">/home/jykj/Dataset/Hbb/PiDaiYuanBan</td>
      <td>无煤</td>
      <td>0</td>
      <td rowspan="8">166048</td>
      <td>62406</td>
      <td rowspan="8">yolov11m</td>
      <td rowspan="8">models/yolov8/custom/belt/yolov11m_belt.pt</td>
      <td rowspan="8">models/yolov8/custom/belt/yolov11m_belt.pt</td>
      <td rowspan="8">/models/custom/belt/yolov11m_belt.pt</td>
      <td rowspan="8">YOLO11m</td>
      <td rowspan="8">640</td>
      <td rowspan="8">38.6 MB</td>
      <td rowspan="8">20.04M / 67.7 GFLOPs@640</td>
      <td rowspan="8">data/benchmark/standard/PiDaiYuanBan (600张)</td>
      <td rowspan="8">0.9866 / 0.9145 (FP16:0.9866 | INT8:0.9850)</td>
      <td rowspan="8">0.9760 / 0.9693</td>
      <td>0.9948 / 0.9629 (P=0.9914, R=0.9944)</td>
      <td rowspan="8">24.7 FPS</td>
      <td rowspan="8">218.4 FPS (FP16/b16, 8.84x)</td>
      <td rowspan="8">735.3 FPS</td>
      <td rowspan="8">已拿真实数据跑通；覆盖272矿区(99.6%)及全部8类(含水煤/未堆煤/堆煤)，FP16校验0漏检</td>
      <td rowspan="8">小:0.985 / 中:0.984 / 大:0.992</td>
      <td rowspan="8">2.57% (0框/0背景图)</td>
      <td rowspan="8">b1 P99=19.1ms | C++ JIT b1: 90.8 FPS (1.59x)</td>
    </tr>
    <tr>
      <td><strong>20</strong></td>
      <td>有煤</td>
      <td>1</td>
      <td>96893</td>
      <td>0.9949 / 0.9361 (P=0.9956, R=1.0000)</td>
    </tr>
    <tr>
      <td><strong>21</strong></td>
      <td>水煤</td>
      <td>2</td>
      <td>181</td>
      <td>0.9950 / 0.9770 (P=0.9567, R=1.0000)</td>
    </tr>
    <tr>
      <td><strong>22</strong></td>
      <td>托辊</td>
      <td>3</td>
      <td>806412</td>
      <td>0.9948 / 0.9417 (P=0.9890, R=0.9900)</td>
    </tr>
    <tr>
      <td><strong>23</strong></td>
      <td>大块煤</td>
      <td>4</td>
      <td>53677</td>
      <td>0.9552 / 0.7244 (P=0.9244, R=0.8468)</td>
    </tr>
    <tr>
      <td><strong>24</strong></td>
      <td>异物</td>
      <td>5</td>
      <td>36796</td>
      <td>0.9684 / 0.7981 (P=0.9878, R=0.9231)</td>
    </tr>
    <tr>
      <td><strong>25</strong></td>
      <td>未堆煤</td>
      <td>6</td>
      <td>4899</td>
      <td>0.9950 / 0.9950 (P=0.9630, R=1.0000)</td>
    </tr>
    <tr>
      <td><strong>26</strong></td>
      <td>堆煤</td>
      <td>7</td>
      <td>10159</td>
      <td>0.9950 / 0.9808 (P=1.0000, R=1.0000)</td>
    </tr>
    <tr>
      <td><strong>27</strong></td>
      <td rowspan="3"><strong>护帮板</strong></td>
      <td rowspan="3">目标检测</td>
      <td rowspan="3">/home/jykj/Dataset/Hbb/HuBangBan</td>
      <td>人</td>
      <td>2</td>
      <td rowspan="3">14121</td>
      <td>1525</td>
      <td rowspan="3">无</td>
      <td rowspan="3">无</td>
      <td rowspan="3">无</td>
      <td rowspan="3">无</td>
      <td rowspan="3">待交付权重</td>
      <td rowspan="3">—</td>
      <td rowspan="3">—</td>
      <td rowspan="3">—</td>
      <td rowspan="3">data/benchmark/standard/HuBangBan (500张)</td>
      <td rowspan="3">—</td>
      <td rowspan="3">—</td>
      <td>—</td>
      <td rowspan="3">—</td>
      <td rowspan="3">—</td>
      <td rowspan="3">—</td>
      <td rowspan="3">全量14121张数据预检0损坏，标准测试集(500张，全类别覆盖)已就绪，待上游交付.pt权重</td>
      <td rowspan="3">—</td>
      <td rowspan="3">—</td>
      <td rowspan="3">—</td>
    </tr>
    <tr>
      <td><strong>28</strong></td>
      <td>护帮板</td>
      <td>3</td>
      <td>43956</td>
      <td>—</td>
    </tr>
    <tr>
      <td><strong>29</strong></td>
      <td>滚筒</td>
      <td>4</td>
      <td>2717</td>
      <td>—</td>
    </tr>
    <tr>
      <td><strong>30</strong></td>
      <td rowspan="3"><strong>探水后上杆钻机</strong></td>
      <td rowspan="3">目标检测</td>
      <td rowspan="3">/home/jykj/Dataset/Hbb/TanShuiHouShangGan</td>
      <td>机器</td>
      <td>0</td>
      <td rowspan="3">704199</td>
      <td>655326</td>
      <td rowspan="3">yolov9</td>
      <td rowspan="3">models/yolov8/custom/tan_shui_hou_shanggan/探水后上杆-yolov9e-960.pt</td>
      <td rowspan="3">models/yolov8/custom/tan_shui_hou_shanggan/探水后上杆-yolov9e-960.pt</td>
      <td rowspan="3">/models/custom/tan_shui_hou_shanggan/探水后上杆-yolov9e-960.pt</td>
      <td rowspan="3">YOLOv9e</td>
      <td rowspan="3">960</td>
      <td rowspan="3">111.8 MB</td>
      <td rowspan="3">57.29M / 424.3 GFLOPs@960 (188.6@640)</td>
      <td rowspan="3">data/benchmark/standard/TanShuiHouShangGan (500张)</td>
      <td rowspan="3">0.8263 / 0.6334 (FP16:0.8263 | INT8:0.8196)</td>
      <td rowspan="3">0.9235 / 0.7816</td>
      <td>0.8834 / 0.7283 (P=0.9971, R=0.8254)</td>
      <td rowspan="3">14.5 FPS</td>
      <td rowspan="3">119.4 FPS (FP16/b16, 8.23x)</td>
      <td rowspan="3">90.9 FPS</td>
      <td rowspan="3">已拿真实数据跑通；覆盖58矿区(100%)及堵头类，YOLOv9e DualDDetect已通过自定义钩子无损加载</td>
      <td rowspan="3">小:0.348 / 中:0.696 / 大:0.780</td>
      <td rowspan="3">6.52% (0框/2背景图)</td>
      <td rowspan="3">b1 P99=53.2ms | C++ JIT b1: 39.6 FPS (1.65x)</td>
    </tr>
    <tr>
      <td><strong>31</strong></td>
      <td>堵头</td>
      <td>1</td>
      <td>235126</td>
      <td>0.9836 / 0.7800 (P=0.8420, R=1.0000)</td>
    </tr>
    <tr>
      <td><strong>32</strong></td>
      <td>机尾</td>
      <td>2</td>
      <td>581444</td>
      <td>0.6120 / 0.3918 (P=0.9314, R=0.5193)</td>
    </tr>
    <tr>
      <td><strong>33</strong></td>
      <td rowspan="2"><strong>风门开关</strong></td>
      <td rowspan="2">目标检测</td>
      <td rowspan="2">/home/jykj/Dataset/Hbb/FengMenKaiGuan</td>
      <td>风门关</td>
      <td>0</td>
      <td rowspan="2">19409</td>
      <td>11270</td>
      <td rowspan="2">yolo11</td>
      <td rowspan="2">models/yolov8/custom/20260923-99风门开关yolov11m-v1.4/20260923-99.pt</td>
      <td rowspan="2">models/yolov8/custom/20260923-99风门开关yolov11m-v1.4/20260923-99.pt</td>
      <td rowspan="2">/models/custom/20260923-99风门开关yolov11m-v1.4/20260923-99.pt</td>
      <td rowspan="2">YOLO11m</td>
      <td rowspan="2">640</td>
      <td rowspan="2">38.6 MB</td>
      <td rowspan="2">20.03M / 67.6 GFLOPs@640</td>
      <td rowspan="2">data/benchmark/standard/FengMenKaiGuan (500张)</td>
      <td rowspan="2">0.9947 / 0.9895 (FP16:0.9947 | INT8:0.9938)</td>
      <td rowspan="2">0.9949 / 0.9959</td>
      <td>0.9944 / 0.9850 (P=0.9916, R=1.0000)</td>
      <td rowspan="2">25.0 FPS</td>
      <td rowspan="2">274.2 FPS (FP32/b16, 10.97x)</td>
      <td rowspan="2">728.9 FPS</td>
      <td rowspan="2">已拿真实数据跑通；覆盖10矿区(100%)，FP16逐框校验0漏检</td>
      <td rowspan="2">小:无样本 / 中:0.972 / 大:1.000</td>
      <td rowspan="2">0.61% (1框/11背景图)</td>
      <td rowspan="2">b1 P99=17.6ms | C++ JIT b1: 90.8 FPS (1.59x)</td>
    </tr>
    <tr>
      <td><strong>34</strong></td>
      <td>风门开</td>
      <td>1</td>
      <td>6918</td>
      <td>0.9950 / 0.9940 (P=0.9981, R=0.9919)</td>
    </tr>
    <tr>
      <td><strong>35</strong></td>
      <td rowspan="3"><strong>传感器</strong></td>
      <td rowspan="3">目标检测</td>
      <td rowspan="3">/home/jykj/Dataset/Hbb/ChuanGanQi</td>
      <td>传感器</td>
      <td>0</td>
      <td rowspan="3">2299</td>
      <td>2602</td>
      <td rowspan="3">yolo11</td>
      <td rowspan="3">models/yolov8/custom/20260904-传感器yolov11m-v1.5/20260904-99.pt</td>
      <td rowspan="3">models/yolov8/custom/20260904-传感器yolov11m-v1.5/20260904-99.pt</td>
      <td rowspan="3">/models/custom/20260904-传感器yolov11m-v1.5/20260904-99.pt</td>
      <td rowspan="3">YOLO11m</td>
      <td rowspan="3">640</td>
      <td rowspan="3">38.6 MB</td>
      <td rowspan="3">20.03M / 67.7 GFLOPs@640</td>
      <td rowspan="3">data/benchmark/standard/ChuanGanQi (400张)</td>
      <td rowspan="3">0.9950 / 0.9660 (FP16:0.9950 | INT8:0.9950)</td>
      <td rowspan="3">0.9984 / 1.0000</td>
      <td>0.9950 / 0.9899 (P=0.9993, R=1.0000)</td>
      <td rowspan="3">25.4 FPS</td>
      <td rowspan="3">236.7 FPS (FP32/b16, 9.32x)</td>
      <td rowspan="3">719.9 FPS</td>
      <td rowspan="3">已拿真实数据跑通；覆盖8矿区(100%)及传感器架子类，FP16逐框校验0漏检</td>
      <td rowspan="3">小:1.000 / 中:1.000 / 大:1.000</td>
      <td rowspan="3">0.08% (0框/0背景图)</td>
      <td rowspan="3">b1 P99=21.7ms | C++ JIT b1: 90.8 FPS (1.59x)</td>
    </tr>
    <tr>
      <td><strong>36</strong></td>
      <td>传感器进气口</td>
      <td>1</td>
      <td>2072</td>
      <td>0.9950 / 0.9132 (P=0.9998, R=1.0000)</td>
    </tr>
    <tr>
      <td><strong>37</strong></td>
      <td>传感器架子</td>
      <td>2</td>
      <td>1517</td>
      <td>0.9950 / 0.9950 (P=0.9961, R=1.0000)</td>
    </tr>
    <tr>
      <td><strong>38</strong></td>
      <td rowspan="5"><strong>卡车空满载</strong></td>
      <td rowspan="5">目标检测</td>
      <td rowspan="5">/home/jykj/Dataset/Hbb/ChanYunXiaoMeiZhong</td>
      <td>卡车空载</td>
      <td>0</td>
      <td rowspan="5">77074</td>
      <td>28203</td>
      <td rowspan="5">无</td>
      <td rowspan="5">无</td>
      <td rowspan="5">无</td>
      <td rowspan="5">无</td>
      <td rowspan="5">待交付权重</td>
      <td rowspan="5">—</td>
      <td rowspan="5">—</td>
      <td rowspan="5">—</td>
      <td rowspan="5">data/benchmark/standard/ChanYunXiaoMeiZhong (500张)</td>
      <td rowspan="5">—</td>
      <td rowspan="5">—</td>
      <td>—</td>
      <td rowspan="5">—</td>
      <td rowspan="5">—</td>
      <td rowspan="5">—</td>
      <td rowspan="5">全量77076张数据预检0损坏，标准测试集(500张，全类别覆盖)已就绪，待上游交付.pt权重</td>
      <td rowspan="5">—</td>
      <td rowspan="5">—</td>
      <td rowspan="5">—</td>
    </tr>
    <tr>
      <td><strong>39</strong></td>
      <td>卡车满载</td>
      <td>1</td>
      <td>952</td>
      <td>—</td>
    </tr>
    <tr>
      <td><strong>40</strong></td>
      <td>车辆车斗遮盖</td>
      <td>3</td>
      <td>19786</td>
      <td>—</td>
    </tr>
    <tr>
      <td><strong>41</strong></td>
      <td>块煤</td>
      <td>6</td>
      <td>5349</td>
      <td>—</td>
    </tr>
    <tr>
      <td><strong>42</strong></td>
      <td>沫煤</td>
      <td>7</td>
      <td>6106</td>
      <td>—</td>
    </tr>
    <tr>
      <td><strong>43</strong></td>
      <td rowspan="3"><strong>探水前上杆</strong></td>
      <td rowspan="3">目标旋转</td>
      <td rowspan="3">/home/jykj/Dataset/Obb/TS-Front_V2</td>
      <td>无杆</td>
      <td>0</td>
      <td rowspan="3">265179</td>
      <td>108087</td>
      <td rowspan="3">yolov8</td>
      <td rowspan="3">models/yolov8/custom/tan_shui_qian_shanggan/探水前上杆-v8l-obb.pt</td>
      <td rowspan="3">models/yolov8/custom/tan_shui_qian_shanggan/探水前上杆-v8l-obb.pt</td>
      <td rowspan="3">/models/custom/tan_shui_qian_shanggan/探水前上杆-v8l-obb.pt</td>
      <td rowspan="3">YOLOv8l-OBB</td>
      <td rowspan="3">640</td>
      <td rowspan="3">85.2 MB</td>
      <td rowspan="3">44.46M / 168.5 GFLOPs@640</td>
      <td rowspan="3">data/benchmark/standard/TS_Front_V2 (600张)</td>
      <td rowspan="3">0.9949 / 0.9731 (FP16:0.9949 | INT8:0.9949)</td>
      <td rowspan="3">0.9966 / 0.9962</td>
      <td>0.9950 / 0.9681 (P=0.9968, R=1.0000)</td>
      <td rowspan="3">40.6 FPS (val b16)</td>
      <td rowspan="3">40.6 FPS (OBB标准流水线)</td>
      <td rowspan="3">445.4 FPS</td>
      <td rowspan="3">已拿真实数据跑通；覆盖4矿区(100%)，OBB旋转框模型(r_IoU评测)，FP16校验0漏检</td>
      <td rowspan="3">小:无样本 / 中:1.000 / 大:0.997</td>
      <td rowspan="3">1.36% (0框/0背景图)</td>
      <td rowspan="3">b1 P99=17.4ms | C++ JIT b1: 69.9 FPS (1.53x)</td>
    </tr>
    <tr>
      <td><strong>44</strong></td>
      <td>有杆</td>
      <td>1</td>
      <td>157463</td>
      <td>0.9950 / 0.9736 (P=1.0000, R=0.9982)</td>
    </tr>
    <tr>
      <td><strong>45</strong></td>
      <td>机器</td>
      <td>2</td>
      <td>262750</td>
      <td>0.9946 / 0.9775 (P=0.9929, R=0.9905)</td>
    </tr>
  </tbody>
</table>
