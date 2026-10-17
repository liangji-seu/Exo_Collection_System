# 超声区域移动播放器

这是一个独立、只读的历史 session 播放器。它按 session 内记录的真实时间回放超声深度曲线，估计相隔一段时间的局部移动方向。它不连接采集设备、不训练模型、不修改 session，也不开网络服务。

## 启动

双击 `run_us_motion_viewer.bat`，再选择 session 文件夹或其中的 `ultrasound.bin`。也可在本目录运行：

```powershell
E:\miniconda\envs\EXO\python.exe run_us_motion_viewer.py --session E:\path\to\session --channel 2 --time 5.6
```

需要加载完成后自动播放时加上 `--play`；默认加载后暂停在所选时间。

程序检查 `PySide6`、`pyqtgraph`、`numpy` 和 `scipy`，但不会自动安装任何依赖。双击启动失败时窗口会保留错误信息。

## 界面含义

- 主图黑色实线只显示当前帧的原始 ADC 信号，横轴为 0–996 原始深度采样点，纵轴为 0–255 ADC 值。原始帧缺失时会明确显示不可用，不会用预处理曲线代替。
- 区域色带位于主图下方，并与主图使用同一深度坐标和像素映射；每个色块约 49 个采样点：蓝色 `←` 表示向浅层，橙色 `→` 表示向深层，黑色 `≈` 表示接近零，灰色 `?` 表示未知。
- “未知”表示没有足够历史、帧无效或证据不足。尤其在 session 开头，缺少历史不能解释为静止。
- 播放按 wall clock 和记录时间推进，支持 0.25×、0.5×、1×、2×、4×。计算较慢时跳到最新应显示的帧，不排队追赶旧帧。
- 拖动时间条、逐帧、换通道或换历史间隔会暂停播放并重新计算当前画面。

“导出截图”和“导出 CSV”只写入用户在保存对话框中选择的位置。CSV 保留 957 点的方向分析结果，包含逐深度方向、质量筛选、支持度和区域标签；它是方向分析输出，不是原始 ADC 导出。

## 自动核验

下面的命令使用合成数据离屏完成加载、播放/暂停、拖动、换通道和关闭，并保存一张预览图：

```powershell
E:\miniconda\envs\EXO\python.exe run_us_motion_viewer.py --smoke-test --screenshot us_motion_viewer_preview.png
```

在命令中再加 `--session E:\path\to\session`，会用真实核心和指定 session 执行同一套自动互动核验。
