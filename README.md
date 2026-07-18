# Exo Collection System

外骨骼多模态数据采集系统的第二版实现。一个仓库提供两个桌面应用：

- **Exo Collector**：设备检查、工况锁定、采集、实时预览与 Trial 最终化；
- **Exo Data Studio**：本地数据树、统计、质量审核、离线回放和人工离线上传入口。

两者共享 `exo_collection` 核心包。架构和数据契约以 [ARCHITECTURE.md](ARCHITECTURE.md) 为准。

## 日常运行（推荐，无需命令行参数）

完成一次 Windows 打包后，直接双击项目根目录中的脚本：

- `Run_ExoCollector.cmd`：启动采集端；
- `Run_ExoDataStudio.cmd`：启动数据管理端。

也可以在 PowerShell 中运行相同脚本：

```powershell
.\Run_ExoCollector.cmd
.\Run_ExoDataStudio.cmd
```

脚本会根据自身位置寻找 `dist` 中的程序，因此从其他工作目录运行、或项目路径中包含空格和中文时也不需要修改路径。它们不会向桌面应用传入任何命令行参数。

成品版 `Run_*.cmd` 会异步启动 GUI；如需查看源码日志并让脚本返回应用退出码，请使用下文的 `Run_*_From_Source.cmd`。

首次启动后，在 UI 的“数据根目录”处点击“选择…”指定数据目录。该选择会保存在当前 Windows 用户的应用设置中，后续启动自动作为默认目录；需要更换时仍在 UI 中重新选择。Collector 与 Data Studio 应使用同一个数据根目录。

从 `release/ExoCollectionSystem-<version>-windows-x64.zip` 部署到其他电脑时，
先完整解压 ZIP，再双击包内的两个 `Run_*.cmd`。不要只单独拷贝某一个
EXE。发布包内的 `BUILD_MANIFEST.json` 记录 Git 来源、构建环境、验证状态和全部
运行文件的 SHA-256；同目录的 `.zip.sha256` 是整个 ZIP 的外部校验值，发布归档时
应同时保留。

## Collector 现场采集流程

1. 选择项目 `F — 正式` 或 `T — 测试`（默认为更安全的 `T`），并输入三位受试者编码，例如 `001`。数据分别进入数据根目录下的 `F` 或 `T` 分区，实际关联仍由 UUID + Manifest 确定。
2. 点击“设备预检 / 连接”。四个必需模态全部为 `READY` 后，“开始 Trial”才可用。当前里程碑预检的是可替换的模拟设备配置；真实厂商协议尚未接入。
3. 点击“开始 Trial”后，系统先 arm 设备并显示“等待同步”。首个合格同步脉冲上升沿才建立正式 `t0`，状态随后进入“采集中”。触发前原始数据仍保留供审计。
4. 实验完成时人工点击“受控停止”，无需预先设置采集秒数。Writer 完成 flush、校验和最终化后才会发布 Manifest。
5. 若停止前从未收到合格触发，系统会显式标记失败，保留 `.recording` 数据和恢复信息，不会把它伪装成正常的已同步 Trial。

每个成功 Trial 都会生成 `manifest.json`、`quality_report.json`、`device_status.csv`、`sync_check.csv`、完整边沿/脉宽/间隔/时钟映射审计 `sync_manifest.json`、两张质控预览图和 `warnings.txt`，并纳入 Manifest Artifact 和 SHA-256 校验。实际使用的 `config/quality_rules/default.json` 与 `config/storage.json` 会冻结到 `derived/quality_rules_snapshot.json`；其算法版本和文件 SHA-256 同时写入统计、质量报告、配置快照及 Manifest Artifact。

质量等级不会再因为“没有生成异常”就自动成为 A。A 要求所有必需结构规则确实执行并通过，包括正式时间窗内各必需模态存在、sequence/丢批检查、同步触发和时钟映射证据。尚无真实硬件校准依据的超声饱和、IMU/编码器量程与跳变、时钟残差等阈值默认明确记录为 `UNASSESSED`，不会伪造硬件阈值或误报硬失败；取得校准依据后可在质量规则配置中填写阈值及 `calibration_reference`。

## Data Studio 数据管理流程

Data Studio 使用 Manifest 与 SQLite Catalog 建立 `Project → Subject → Session → Trial → Artifact` 数据树。可按项目、受试者、Session、工况、日期、质量等级和关键词筛选，并查看工况覆盖率、重复轮次、最终化/待恢复/中止、待审核和待上传统计；当前清单可导出为 CSV 与 JSON。

离线回放使用统一时间游标显示四通道超声、IMU、编码器和同步事件。质量复核写入独立、带 SHA-256 锚点和哈希链的追加式审核记录，不修改原始文件。测力台、动作捕捉等外部文件通过“外部模态导入”复制到独立附录目录，保留源文件校验值、脉冲边沿和外部时钟映射；没有厂商协议时仅支持通用文件与显式列映射，不猜测数据语义。

恢复、全量统计、校验、外部导入与 SSH/SCP 上传均在后台 Worker 中执行。Collector 活动锁存在或不可安全解析时，Data Studio 会保守进入轻量模式并禁用这些重任务。上传只能人工触发，且只接受已最终化 Trial；密码和私钥口令仅通过进程内匿名管道传给上传 Worker，不写配置、日志、SQLite 或命令行。

## Windows 开发环境

**路径约束**：项目目录（以及所有上级目录）的路径中不能包含 Windows
保留设备名（如 `NUL`、`CON`、`PRN`、`AUX`、`COM1`-`COM9`、`LPT1`-`LPT9`）
作为单独的路径段。所有 `.cmd` 启动脚本在入口处会检测并拒绝此类路径，
因为系统可能将路径段中的设备名当作 I/O 设备处理，导致工作目录解析异常。

新电脑首次使用时，先安装 **64 位 Python 3.11**（包含 Windows `py` launcher）
和 Git，并从远程仓库取得干净的 checkout，然后在项目根目录双击或运行：

```powershell
.\First_Time_Setup.cmd
```

该入口会检查 Python 3.11，创建缺失的 `.venv`，安装项目及开发/打包
依赖，然后执行完整测试、构建两个 EXE、冻结应用 smoke check 和单一 ZIP
发布包生成。`First_Time_Setup_And_Build.cmd` 作为旧名兼容入口仍然保留。
若 `.venv` 已存在，脚本只校验并复用它；版本不是 Python 3.11 或环境不完整时会
停止并提示，绝不会自动删除或覆盖已有环境。若软件源暂时不可用，但现有环境的
完整依赖和当前仓库 editable 路径均通过严格检查，脚本会警告后复用该环境，
且不会改动用户的代理设置。

如需手工配置，执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,packaging]"
```

运行测试：

```powershell
python -m pytest
```

## 从源码运行（开发调试）

无需激活虚拟环境，也无需给应用传参数：

```powershell
.\Run_ExoCollector_From_Source.cmd
.\Run_ExoDataStudio_From_Source.cmd
```

源码脚本会检查 `.venv`、Python 3.11 和必要依赖，并在缺少环境时显示可直接执行的修复命令。源码运行时会保留终端窗口，便于查看调试输出。

## 编译打包 Windows 可执行文件

推荐使用项目根目录的统一入口：

```powershell
.\Build_Windows.cmd
```

该入口内部调用 `packaging\build_windows.ps1`。如需直接调用 PowerShell 脚本：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\packaging\build_windows.ps1
```

构建脚本会检查 64 位 Windows Python 3.11 和 PyInstaller，然后默认严格执行：

1. 完整 `pytest` 测试；
2. 构建 Collector 与 Data Studio；
3. 对冻结 GUI 运行 offscreen 检查，其中 Collector 会通过独立进程完成设备预检，
   Data Studio 会等待 Catalog 和管理/annex 索引独立进程均完成；
4. 用冻结后的 Collector 运行一次短模拟 Trial，验证 Windows `spawn` Worker；
5. 生成单一 ZIP bundle、覆盖全部运行文件的 SHA-256 清单和整个 ZIP 的
   `.zip.sha256` 外部校验文件。

默认发布构建要求 Git 工作树干净，并在构建结束前再次确认 HEAD 和工作树没有
变化，避免把未提交源码错误归因给某个 commit。仅调试构建可直接调用 PowerShell
脚本并显式添加 `-AllowDirtyWorkingTree`；该产物会在清单中标记为 dirty，不能视为
可从所列 commit 单独复现的正式发布。

中间 EXE 位于：

```text
dist\ExoCollector.exe
dist\ExoDataStudio.exe
```

可分发的主产物位于：

```text
release\ExoCollectionSystem-{version}-windows-x64.zip
release\ExoCollectionSystem-{version}-windows-x64.zip.sha256
```

ZIP 内同时包含两个应用、启动脚本、使用说明和 `BUILD_MANIFEST.json`；构建脚本会
重新打开 ZIP，逐条核对路径、大小和 SHA-256 后才报告成功。
如果电脑已安装 Inno Setup 6，同一构建还会根据
`packaging\installer\ExoCollectionSystem.iss` 额外生成将两个应用同时安装的
`release\ExoCollectionSystem-{version}-Setup.exe`；没有 Inno Setup 时 ZIP 仍是完整发布包。

构建成功只表明模拟设备、界面、存储和冻结进程边界已通过检查。真实厂商 SDK、
真实数据协议和现场服务器参数尚未提供，不得因打包通过就宣称真实硬件已可用。

采集期间不要运行全盘校验、回放或上传；Data Studio 检测到 Collector
活动租约后会自动进入轻量模式。`.recording` Trial 只能通过显式恢复流程检查，
不会被 Data Studio 当成已最终化数据打开。

## 真实设备环境与 UI 配置

真实设备不使用命令行参数。在 Collector 的“设备配置”中选择“真实超声 + 3×IMU +
电机编码器”，再点击“真实设备设置…”填写 Elonxi SDK 目录、可选设备 IP、
Awinda 信道/采样率/3 个 MTw ID，以及 Teensy 串口参数。选择会写入当前
Windows 用户的 QSettings，以后启动默认沿用。密码和凭据不在该设置中。

源码环境需先安装通用硬件依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[hardware]"
```

Xsens Python 绑定由 MT SDK 提供，不从 PyPI 猜测安装。当前旧系统中已有与
Python 3.11 x64 匹配的官方 wheel，可在两个项目共存时执行：

```powershell
.\.venv\Scripts\python.exe -m pip install `
  "..\Exo_data_capture_system\MT SDK\Python\x64\xsensdeviceapi-2025.2.0-cp311-none-win_amd64.whl"
```

该 Xsens 2025.2 wheel 使用 NumPy 1.x ABI，所以本项目明确约束 `numpy>=1.26,<2`。
不要单独升级到 NumPy 2.x，否则 `xsensdeviceapi` 会在导入阶段因 ABI 不匹配失败。

Elonxi 设置选择的目录必须包含 `Elonxi_SDK.dll`。正式打包真实设备版本时，
必须在构建用的 `.venv` 中先安装上述硬件依赖和 Xsens wheel，PyInstaller
才会收集相应模块。

> 当前 `hardware` Profile 的超声、三台 IMU 和电机编码器为真实适配；
> `sync_pulse` 仍是台架模拟信号。这一模式用于验证三类设备接入，不得宣称为
> 测力台/动捕正式同步已完成。
