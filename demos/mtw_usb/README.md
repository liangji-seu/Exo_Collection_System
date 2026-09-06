# MTw 三颗 USB 有线采集 demo

独立于 Collector，不修改现有无线适配器。按完整设备 ID 查找直接连接的 MTw，逐颗设置并读回 100 Hz，分别写 CSV。每次运行建立新的时间命名输出目录。

## 明天怎么试

1. 关闭 Exo Collector、MT Manager 等可能占用设备的程序。拔掉 Awinda Dongle/Station 的电脑 USB 连接，使用能传数据的 Micro-USB 线，将三颗 MTw 直接连到电脑。也可经可靠供电的 USB Hub 连接。不要仅把传感器插在充电座上当作直连验证。
2. 编辑本目录 `config.json` 的设备 ID。默认沿用 102 数据的配置：

   | 位置 | ID |
   |---|---|
   | left_leg（左腿） | 10B42626 |
   | right_leg（右腿） | 10B4260D |
   | pelvis（骨盆） | 10B4261F |

   必须填写完整 ID，不能只写末三位；大小写均可。三颗 ID 都留在 `sensors` 里即可——程序会自动匹配当前实际连上的子集：只插一颗，就只采集这一颗并在控制台/`summary.json` 里标出缺了哪两颗。
3. 第一次若当前 Python 没有 SDK，双击 `install_sdk.bat`。它使用工程已有的 `Exo_data_capture_system/MT SDK/Python` 下匹配 wheel 离线安装，不下载、不升级固件。安装到同一 EXO Python 环境；已有同版本时 pip 不重复安装。
4. 可先双击 `scan_devices.bat`，看是否能发现目标 ID。扫描成功仅证明找到端口，尚未证明能输出数据。
5. 双击 `run_demo.bat`。默认录 60 秒，期间可 Ctrl+C 停止保存。控制台每两秒显示各颗累计行数与计数跳缺，结束显示接收频率估计。
6. 检查本目录 `output/日期_时间_编号/`。明天把这个完整输出文件夹留给后续诊断。

启动器优先使用 `E:\miniconda\envs\EXO\python.exe`，路径不存在时才使用 PATH 中的 python；也可以设置环境变量 `MTW_DEMO_PYTHON` 指定自己的 Python 完整路径。现有本地 SDK wheel 包含 Python 3.8–3.11，推荐 Python 3.11 x64。不要用系统 Python 3.14 来装 cp311 wheel。

不使用启动器时，在已配置 SDK 的终端执行：

```powershell
python run_demo.py --scan-only
python run_demo.py
python run_demo.py --config config.json
```

## 功能与判定

- 自动匹配：扫描到的 MTw 按配置 ID 对上几颗就采几颗（1–3 颗），并在控制台与 `summary.json` 的 `matched_sensor_names` / `missing_sensor_names` 标明；只有一颗都不匹配时才报错。多余/陌生 ID 忽略，不会用它替代缺的设备。
- 打开后要求 SDK 的 `connectivityState` 为 `XCS_PluggedIn`；不启用无线，也不会无线回退。
- 采用本地官方 `Examples/xda_matlab/example_mtw.m` 的 MTw 配置方式：`gotoConfig`、`XSO_Orientation | XSO_Calibrate`、`supportedUpdateRates`、`setUpdateRate(100)`、`gotoMeasurement`。不直接套用 MTi 的输出配置方式。
- 不支持 100 Hz 或读回不是 100 Hz 时退出，不偷偷改为 80 Hz。
- 各颗独立保存所有回调数据，不要求三颗同编号同时到齐，不插值、不补零、不用另一颗的数据填缺失。
- 加速度、角速度、磁场、欧拉角、四元数均输出。缺某类信号时保留该包、相应数据为 NaN，标记 has_calibrated / has_orientation。
- 超过默认 10 秒无包，或一直无完整校准/姿态数据，报错并停止。队列溢出也明确失败，已保存数据保留。

## 输出

- `left_leg_ID.csv`、`right_leg_ID.csv`、`pelvis_ID.csv`：各传感器原始解码数据。
- `summary.json`：设备型号、USB 端口、采样率读回、行数、信号完整性、接收频率估计、计数缺口和错误。
- `run.log`：操作过程；`error_traceback.txt`：出现异常时的详细信息。
- `config_snapshot.json`：此次配置。

`FINISHED` 只表示正常跑完，不是“同步与信号质量全部通过”。首先检查每颗 `reported_rate_hz=100`、接收频率是否接近 100、完整信号行数是否接近总行数、计数缺口和重复是否为 0。一分钟测试通常每颗约 6000 行，顺序启动和停止可能使行数略有差别。

接收频率按首尾回调到达时间估计，不是独立硬件采样率认证。计数统计从首个实际计数开始，不把初始大编号算作缺失；处理 16 位回绕。若没有计数器则报告 null，不能解释为零丢包；出现重置或乱序时，缺口数字只作线索，需查看原始计数。

## 时间与三颗同步

CSV 同时记录：设备 packet_counter、设备 sample_time_fine（存在时保留 SDK 原始值）、主机回调到达时间 host_monotonic_ns、审计时间 host_utc_ns。elapsed_host_s 为共同主机起点的相对到达时间，含配置启动阶段，不是传感器采样时间。

**三路 USB 并不保证三颗同步采样。** 此 demo 用于验证有线读取和数据完整性，明确保存 `cross_device_hardware_sync_verified=false`。不能把相同包编号直接认为同一时刻，也不能仅凭主机到达时间证明精确同步。正式用于双髋分析前，要验证三颗时间偏移、漂移及与力台/动捕的同步。

本 demo 保存 CSV 解码数据，不产生原厂 `.mtb` 日志。开始后需要记录原厂日志/同步功能时，应另外扩展，不要把本 demo 输出直接替换原始 Collector 的 imu.h5。

## 常见情况

- 扫描不到任何 MTw：先确认数据线、USB 驱动、完整 ID 和软件占用；若只识别到接收器，仍不是传感器直连。
- 识别了设备但不是 XCS_PluggedIn：程序会拒绝采集，保留日志用于确认 SDK/固件行为。
- 配置成功但没有姿态：不要把这一轮当作成功，检查 SDK 版本和 MTw 软件滤波输出。完整错误在输出目录。
- 当前开发机 EXO 环境起初缺少 `xsensdeviceapi`，已提供离线安装入口；未在本轮改动该环境。

已执行离线模拟测试，覆盖三颗独立保存、单颗自动匹配、缺失 ID（全部不匹配）、有线类型门禁、100 Hz 门禁、队列溢出、无数据超时、计数回绕和重复配置。**尚未接真实三颗 USB MTw 验证，明天的接线测试是硬件验收。**
