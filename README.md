# Exo Collection System

外骨骼多模态数据采集系统的第二版实现。一个仓库提供两个桌面应用：

- **Exo Collector**：设备检查、工况锁定、采集、实时预览与 Trial 最终化；
- **Exo Data Studio**：本地数据树、统计、质量审核、离线回放和人工离线上传入口。

两者共享 `exo_collection` 核心包。架构和数据契约以 [ARCHITECTURE.md](ARCHITECTURE.md) 为准。

## 日常运行（推荐，无需命令行参数）

完成系统 Python 初始化后，项目根目录提供两个不需要任何命令行参数的 Python 启动脚本：

- `run_collector.py`：启动采集端；
- `run_data_studio.py`：启动数据管理端。

在 PowerShell 中直接使用系统 CPython 3.11 运行：

```powershell
python run_collector.py
python run_data_studio.py
```

完成打包后也可直接双击 `dist\ExoCollector.exe` 或 `dist\ExoDataStudio.exe`。设备、路径和上传参数都在 UI 中配置并按当前 Windows 用户持久化，无需为正常使用添加命令行参数。

首次启动后，在 UI 的“数据根目录”处点击“选择…”指定数据目录。该选择会保存在当前 Windows 用户的应用设置中，后续启动自动作为默认目录；需要更换时仍在 UI 中重新选择。Collector 与 Data Studio 应使用同一个数据根目录。

## Collector 现场采集流程

1. 选择项目 `F — 正式` 或 `T — 测试`（默认为更安全的 `T`），并输入三位受试者编码，例如 `001`。数据分别进入数据根目录下的 `F` 或 `T` 分区，实际关联仍由 UUID + Manifest 确定。
2. 在“设备连接”中点击蓝色下划线的超声、IMU、电机编码器或同步脉冲名称，分别设置该模态的设备参数；保存后立即写入当前 Windows 用户设置，后续启动自动恢复。随后分别点击各行“连接”，或使用“全部连接”。每个模态收到首批真实帧/样本后才显示 `READY`，并立即驱动右侧对应预览窗口。
3. 连接/预览阶段不创建 Trial、Manifest、HDF5 或超声二进制原始文件。四个必需模态都 `READY` 后“开始 Trial”才可用；点击后系统会先释放预览进程对设备的独占，再由独立 Collector Worker 开始原始写盘。
4. Trial Worker 打开写盘 gate 后立即进入“采集中”；同步状态独立显示为“等待同步”。若收到合格同步上升沿，则以该边沿建立正式 `t0`；否则以写盘 gate 的主机单调时钟作为 `t0`。同步前原始数据仍保留供审计。
5. 实验完成时人工点击“受控停止”，无需预先设置采集秒数。Writer 完成 flush、校验和最终化后才会发布 Manifest。
6. 若停止前从未收到合格触发，系统记录 `NOT_RECEIVED / OPTIONAL` 并正常最终化，不产生告警。只有已启用模态断流、设备故障、序列缺口、队列溢出或写盘完整性失败才保留 `.recording` 数据并进入失败/恢复流程。

每个成功 Trial 都会生成 `manifest.json`、`quality_report.json`、`device_status.csv`、`sync_check.csv`、完整边沿/脉宽/间隔/时钟映射审计 `sync_manifest.json`、两张质控预览图和 `warnings.txt`，并纳入 Manifest Artifact 和 SHA-256 校验。实际使用的 `config/quality_rules/default.json` 与 `config/storage.json` 会冻结到 `derived/quality_rules_snapshot.json`；其算法版本和文件 SHA-256 同时写入统计、质量报告、配置快照及 Manifest Artifact。

Collector 和 Data Studio 在项目/安装目录的 `log\` 中为每次启动创建一份独立 UTF-8 系统日志，文件名包含启动时间和进程 PID，例如 `ExoCollector_20260718_130501_123456_pid1234.log`。UI 中的“打开日志目录”可直接定位。单次日志达到 10 MiB 时会滚动；每个 Trial 另有自己的 `logs/trial.jsonl`。常见密码、token、secret 和 key 字段在写入主日志前会被脱敏。

质量等级不会再因为“没有生成异常”就自动成为 A。A 要求所有必需结构规则确实执行并通过，包括正式时间窗内各已启用必需模态存在以及 sequence/丢批检查。同步触发和依赖同步的时钟映射属于可选对齐证据，未收到时明确记录为 `UNASSESSED`，不降低质量等级。尚无真实硬件校准依据的超声饱和、IMU/编码器量程与跳变、时钟残差等阈值也默认记录为 `UNASSESSED`，不会伪造硬件阈值或误报硬失败；取得校准依据后可在质量规则配置中填写阈值及 `calibration_reference`。

## Data Studio 数据管理流程

Data Studio 使用 Manifest 与 SQLite Catalog 建立 `Project → Subject → Session → Trial → Artifact` 数据树。可按项目、受试者、Session、工况、日期、质量等级和关键词筛选，并查看工况覆盖率、重复轮次、最终化/待恢复/中止、待审核和待上传统计；当前清单可导出为 CSV 与 JSON。

离线回放使用统一时间游标显示四通道超声、IMU、编码器和同步事件。质量复核写入独立、带 SHA-256 锚点和哈希链的追加式审核记录，不修改原始文件。测力台、动作捕捉等外部文件通过“外部模态导入”复制到独立附录目录，保留源文件校验值、脉冲边沿和外部时钟映射；没有厂商协议时仅支持通用文件与显式列映射，不猜测数据语义。

恢复、全量统计、校验、外部导入与 SSH/SCP 上传均在后台 Worker 中执行。Collector 活动锁存在或不可安全解析时，Data Studio 会保守进入轻量模式并禁用这些重任务。上传只能人工触发，且只接受已最终化 Trial；密码和私钥口令仅通过进程内匿名管道传给上传 Worker，不写配置、日志、SQLite 或命令行。

## Windows 开发环境

新电脑首次使用时，先安装 **64 位 CPython 3.11** 和 Git。若需要真实硬件版本，先解压并执行同级 `SDK_Transfer` 中最新的
`Exo_Hardware_Runtime_Windows_Python311_x64_v3.zip` 安装脚本，再安装 Npcap
（勾选 WinPcap API-compatible Mode）。然后在项目根目录只需执行：

```powershell
python first_time_setup_and_build.py
```

该零参数脚本不创建虚拟环境，会把应用、测试、打包和开源硬件依赖安装到当前 Windows 用户的系统 Python，检查 Xsens/Scapy/pyserial/Npcap，运行全量测试，然后生成两个 EXE。日常手动运行测试可使用：

```powershell
python -m pytest
```

## 从源码运行（开发调试）

不使用虚拟环境，也无需给应用传参数：

```powershell
python run_collector.py
python run_data_studio.py
```

## 编译打包 Windows 可执行文件

项目根目录的 `build_exe.py` 会依次构建两个应用：

```powershell
python build_exe.py
```

构建前建议先运行上述完整测试。EXE 位于：

```text
dist\ExoCollector.exe
dist\ExoDataStudio.exe
```

构建成功只表明模拟设备、界面、存储、真实 Adapter 的依赖收集和冻结进程边界已通过检查。真实超声、Xsens IMU 和 Teensy 编码器仍必须在实验室设备上分别完成连接、持续预览、Trial 切换和长时间压力验收；不得因单元测试或打包通过就宣称物理硬件已验收。

采集期间不要运行全盘校验、回放或上传；Data Studio 检测到 Collector
活动租约后会自动进入轻量模式。`.recording` Trial 只能通过显式恢复流程检查，
不会被 Data Studio 当成已最终化数据打开。

## 真实设备环境与 UI 配置

真实设备不使用命令行参数，也没有全局“设备配置”表单。Collector“设备连接”
表格中的四个模态名称就是各自的设置入口：超声设置 Raw Ethernet/Npcap 网卡与
标称帧率，并可在后台扫描目标帧；IMU 设置 Awinda 信道、采样率和按躯干/左腿/
右腿顺序的 3 个 MTw ID；编码器设置 Teensy 串口、波特率、VID/PID 与标称采样率；
同步脉冲设置当前台架模拟信号参数。每次保存只更新对应模态，不会覆盖其他设备，
并立即写入当前 Windows 用户的 QSettings，以后启动默认沿用。密码和凭据不写入
这些设置。

源码环境需先安装通用硬件依赖：

```powershell
python -m pip install -e ".[hardware]"
```

Xsens Python 绑定由 MT SDK 提供，不从 PyPI 猜测安装。当前旧系统中已有与
Python 3.11 x64 匹配的官方 wheel，可在两个项目共存时执行：

```powershell
python -m pip install `
  "..\Exo_data_capture_system\MT SDK\Python\x64\xsensdeviceapi-2025.2.0-cp311-none-win_amd64.whl"
```

该 Xsens 2025.2 wheel 使用 NumPy 1.x ABI，所以本项目明确约束 `numpy>=1.26,<2`。
不要单独升级到 NumPy 2.x，否则 `xsensdeviceapi` 会在导入阶段因 ABI 不匹配失败。

Raw Ethernet 超声依赖 Scapy；Windows 还必须安装 Npcap，并在 Npcap
安装器中勾选 `WinPcap API-compatible Mode`。正式打包真实设备版本时，
构建用 Python 必须已安装 hardware 依赖和 Xsens wheel，PyInstaller 才会
收集 `scapy`、`xsensdeviceapi` 和 `serial` 等可选硬件模块。Npcap 是系统
驱动，不会被 PyInstaller 打进 EXE。

> 当前 `hardware` Profile 的超声、三台 IMU 和电机编码器为真实适配；
> `sync_pulse` 仍是台架模拟信号。这一模式用于验证三类设备接入，不得宣称为
> 测力台/动捕正式同步已完成。
