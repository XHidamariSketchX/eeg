# PyBCI 接入

## 启动实机学习

先关闭旧版监视窗口，再启动更新后的采集器：

```powershell
.\.venv\Scripts\python.exe eeg_viewer.py
```

连接 COM3。打开第二个终端启动训练界面：

```powershell
.\.venv\Scripts\python.exe eeg_pybci.py
```

1. 等待“已连接”。监视器提供 Gemini-Metrics 指标流，训练界面提供 Gemini-TaskMarkers 标记流。
2. 点击“录制静坐试次”：2 秒准备，然后保持睁眼静坐 8 秒。
3. 点击“录制心算试次”：2 秒准备，然后默算连续减 7，保持姿势 8 秒。
4. 交替进行，每类至少 6 个有效试次。被跳过的试次不计数。
5. 点击“训练完成 → 保存并预测”。模型此后冻结，显示任务类别预测。

6 次只是让演示模型开始工作的最低数量，不代表科学验证充分。
预测是实验任务分类，不是“专注百分比”或情绪诊断。
没有随机拆分相邻窗口得到的准确率；应另录多轮或跨日期的有标签数据做独立验证。

## 不连接硬件试用

```powershell
.\.venv\Scripts\python.exe eeg_pybci.py --demo
```

此模式内置模拟指标源，不需要打开监视器。模拟源按按钮生成可分开的指标，
只验证软件流程，绝不代表真实分类效果。不要同时启动其他同名 Demo 指标流。

## 保存与加载

每次会话保存在 bci_sessions 的独立目录：

- metrics.csv：所有收到的指标、佩戴状态、包序号和 LSL 接收时间。
- events.jsonl：准备阶段、任务开始标记、试次结束、预测结果等。
- training_features.npz：有效试次的 12 维特征及类别编号。
- labels.json：类别编号和有效次数。
- model.joblib：点击切换预测后保存的模型、标准化器、标签和窗口配置。

如果还要保存完整 21 列数据，点击监视器中的“开始录制”；它仍保存到 recordings。
LSL marker 使用主机时钟，不能校正设备内部未知的算法延迟。

加载自己生成且可信的模型：

```powershell
.\.venv\Scripts\python.exe eeg_pybci.py --model bci_sessions\实际会话目录\model.joblib
```

实机模型用实机流，模拟模型必须配合 --demo，不混用。
训练软件关闭会停止线程并保存已写入数据；监视器可继续工作。

## 特征与质量规则

只使用设备 theta/alpha/beta 百分比的均值、标准差、中位数、首尾差，共 12 维。
不使用频域算法，不使用设备专注度当作标签，不假设硬件 250 Hz 等于输出采样率。
每次标记对应一个完整试次，不切成重叠训练样本。
不规则 LSL 流名义速率为 0；PyBCI 接收器按 LSL 时间截取窗口。

以下片段跳过：佩戴状态非 1、非有限数值、百分比越界、包序号不连续、
相邻数据间隔超过 0.5 秒、窗口有效时间覆盖不足 75%。
这些规则不能识别全部动作伪迹，训练时仍应保持姿势一致。

## 本地 PyBCI 适配版本

已安装 .vendor/pybci-main 中保留 MIT 许可的源码版本。
上游 https://github.com/LMBooth/pybci ，获取日期 2026-10-26。
本地适配不代表上游正式发布：

- 修复发行包只包含顶层 pybci 包、缺失 Utils 等子模块的问题。
- 兼容 pylsl 新版 resolve_streams。
- sklearn 路径不强制安装 TensorFlow/PyTorch；Torch 改为需要时导入。
- 无效特征片段允许返回 None，跳过后才增加有效试次计数。
- 只在新训练样本到达后重新拟合；移除随机窗口划分准确率。
- 降低线程空转频率，缩短停止等待，每次查询仅返回一份响应。

重新安装本项目使用的版本（从项目根目录运行）：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-pybci.txt
```

不要用普通 pip install --upgrade pybci-package 覆盖本地修复。
当前只验证 sklearn + 自定义指标特征这一条路径。

测试：

```powershell
.\.venv\Scripts\python.exe -m unittest test_bci_features -v
.\.venv\Scripts\python.exe test_pybci_integration.py
```
