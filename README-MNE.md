# MNE 学习入口

使用项目虚拟环境（已安装 MNE），保持现有 eeg_viewer.py 负责录制。

## 分析最新录制

```powershell
.\.venv\Scripts\python.exe eeg_mne.py
```

脚本默认读取 recordings 中最新的 eeg_*.csv。每次生成独立的 analysis 子目录，包含：

- report.html：MNE 报告，在浏览器中打开。
- waveforms.png：双通道输出与佩戴/连续性标记。
- device_metrics.png：设备 θ/α/β 百分比、评分及接收间隔。
- spectrum.png、spectrum.csv：MNE Welch 归一化频谱。
- summary.json：分析条件、统计和 MNE 版本。

也可指定文件：

```powershell
.\.venv\Scripts\python.exe eeg_mne.py recordings\eeg_20261026_112234_613404.csv
```

## 在明确假设下学习 Hz 频谱和 MNE 浏览器

```powershell
.\.venv\Scripts\python.exe eeg_mne.py --fs 50 --browse
```

50 Hz 是调用者指定的输出采样假设，不是测量确认值，也不是硬件的 250 Hz。
此模式另外导出 assumed_rate_raw.fif，保存采样率假设与 BAD_wear/BAD_boundary 注释。
MNE 浏览器的秒轴同样基于该假设，不能与主机接收时间混为一谈。
未提供电压标定，通道使用 misc 类型；设备波段百分比和评分不是 EEG 电压。
不指定 --fs 时，不创建虚构秒轴的 Raw 对象。

## 选项

- --source raw：分析 CSV 中厂家命名的原始 ADC 列；默认 filtered。
- --wear-ok 1：默认把状态 1 作为佩戴正常，基于现有记录。
- --browse：打开 MNE 波形浏览器，需要 --fs。

质量处理按佩戴异常、模 256 的序号不连续及超过 1 秒的接收间隔分段。
频谱用各连续段中 128 点、50% 重叠的窗口计算，再按窗口平均。
不跨边界拼接信号；少于 128 点的片段不参与频谱。
这些规则不能自动识别或清除眨眼、咬牙和移动伪迹。

## 建议学习顺序

1. 打开报告，看原始输出和设备指标的对应关系。
2. 阅读 eeg_mne.py 中 load_recording、quality、compute_spectrum。
3. 运行测试，观察已知 10 Hz 模拟信号在指定采样率和归一化坐标下的位置。
4. 学习 make_raw 的通道、采样率和 annotations，再做按任务分段比较。

```powershell
.\.venv\Scripts\python.exe -m unittest test_eeg_mne -v
```
