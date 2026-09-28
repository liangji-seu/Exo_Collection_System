# US / 48 段特征瀑布图

双击 `run_us_feature_viewer.bat`，选择包含超声文件的 session 文件夹即可。

界面左侧四行是各通道 48 段特征，右侧四行是对应原始 ADC。沿用 Data Studio 的灰度瀑布组件，10 秒窗口从左向右循环填充，50 ms 刷新，按真实采集时间回放。两侧各自固定灰度范围，亮度不可跨两侧作数值比较。

也可以在 `Exo_Collection_System` 目录执行：

```powershell
E:\miniconda\envs\EXO\python.exe run_us_feature_viewer.py
```

也可以直接传入 session 文件夹（其中应有 `ultrasound.bin`、`ultrasound.idx`、`ultrasound.meta.json`）：

```powershell
E:\miniconda\envs\EXO\python.exe run_us_feature_viewer.py E:\path\to\session
```

左侧严格按 `payload[2:-1]`（997 点）转 `float32`，逐帧减中位数、取绝对值、`uniform_filter1d(size=20, mode="nearest")`、`log1p(max(..., 0))`，截取 `[20:977]` 后 `np.array_split(..., 48)` 逐段均值。左图显示这 48 维原值，不做标准化或模型推断。右图显示同一批帧的原始 ADC payload。

raw Ethernet 每个通道独立到达，播放时按各通道第 k 个 packet 配成一帧，时间取四通道到达时间的最大值；这表示回放对齐顺序，不声称硬件物理同步。空格播放/暂停，左右方向键或按钮逐帧查看，进度条可定位。
