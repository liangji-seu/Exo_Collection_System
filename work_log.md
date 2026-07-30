# Exo Collection System — 工作交接报告

> 生成时间：2026-07-30
> 最后提交：`38d318f` — feat(collector): add human prompt-label events during Trial recording
> 分支：`main`（无未提交修改）

---

## 2026-07-30 — XING/Nokov 动捕 Marker + EMG 在线采集（已实现，待现场验证）

- 新增 `mocap` 与 `emg` 两个原生采集模态，真实 Adapter 分别使用官方 SDK 的
  `PySetDataCallback` 和 `PySetAnalogChFunc`。
- MarkerSet 在连接时读取并冻结名称/顺序，写入 `raw/mocap.h5`，数据形状为
  `(sample, marker, xyz)`、单位为 mm。
- EMG Analog 子帧转为 `(sample, channel)` 并写入 `raw/emg.h5`；通道数不一致
  直接报告设备故障。
- Collector 增加两行独立连接/设置/健康状态，以及 Marker Z 曲线和最多 16 通道
  EMG 实时预览；预览区域改为可滚动布局。
- 模拟 Profile 同步增加动捕与 EMG 模拟源，可在无现场设备时验证完整预览与写盘。
- PyInstaller 在构建环境已安装 `nokov` 时收集其 Python 模块与厂商 DLL。
- 待现场验证：实际 Seeker IP、MarkerSet 描述读取、EMG 通道数/单位/实际采样率、
  双 Client 长时间运行稳定性。

---

## 1. 项目概览

**Exo Collection System** 是外骨骼多模态数据采集系统的第二版实现。整个项目为一个 Git 仓库，发布一个安装包，但向工作人员提供两个职责不同的桌面应用：

| 应用 | 职责 |
|------|------|
| **Exo Collector** | 设备连接 → 工况选择 → Trial 采集 → 实时预览 → 设备告警 → 停止与最终化 |
| **Exo Data Studio** | 本地数据树浏览 → 统计 → 质量审核 → 离线多模态回放 → SSH/SCP 离线上传 |

目标平台：**Windows 11**，语言：**Python 3.11**，GUI：**PySide6 + pyqtgraph**。

### 核心设计原则

- **Trial 是最小完整数据单元**（Project → Subject → Session → Trial）
- **原始数据不可变**（写盘后不得原地修改）
- **采集、预览、管理、上传解耦**（多进程架构）
- **UUID 是真实主键**，可读文件名仅用于界面显示
- **Local-first**：采集不依赖网络，服务器不可用不能影响现场采集
- **单仓库、双桌面应用**，共享 `exo_collection` 核心包

---

## 2. 架构概要

### 2.1 运行时进程模型

| 进程 | 归属 | 职责 |
|------|------|------|
| `collector-ui` | Exo Collector | 主 UI 进程，不直接写原始数据 |
| `collector-core` | Exo Collector | Session/Trial 状态机、设备编排 |
| `device-worker-*` | Exo Collector | 各厂商 SDK 独立进程 |
| `writer-*` | Exo Collector | 按模态隔离写盘 |
| `catalog-worker` | Exo Collector | Trial 完成后的索引和统计入库 |
| `studio-ui` | Exo Data Studio | 主 UI 进程 |
| `analysis-worker` | Exo Data Studio | 大文件扫描、统计重算 |
| `playback-worker` | Exo Data Studio | 回放缓存和索引读取 |
| `transfer-worker` | Exo Data Studio | SSH/SCP 上传和远端校验 |

### 2.2 Trial 状态机

```
IDLE → PREPARING → READY → WAITING_SYNC → RECORDING → STOPPING → FINALIZING → FINALIZED
                                                   ↘ ABORTED / RECOVERABLE
```

- 现场 Trial 不预设固定采集时长，人工点击开始/停止
- 首个合格同步上升沿建立正式 t0；未收到同步脉冲记为 `NOT_RECEIVED / OPTIONAL`
- 只有 FINALIZED 或 ABORTED 的 Trial 可进入上传列表

### 2.3 数据存储规范

```
dataset_root/
  T|F_BASE|F_STEADY|F_TRANSIENT/     ← 项目代码分区
    subject_code/                      ← 三位可读编码
      session_uuid/
        session.json
        trials/
          trial_uuid/
            manifest.json
            raw/
              ultrasound.bin           ← 分块二进制
              imu.h5                   ← HDF5
              encoder.h5               ← HDF5
              sync_pulse.h5            ← HDF5
              prompt_labels.jsonl      ← NDJSON（仅有人工标签时）
            derived/ preview/ alignment.json statistics.json quality_rules_snapshot.json
            reports/ quality_report.json device_status.csv sync_check.csv sync_manifest.json warnings.txt
            logs/ trial.jsonl
            checksums.sha256
  external_annexes/
```

### 2.4 时间同步模型

- 公共时间基准：`time.perf_counter_ns()` 主机单调时钟
- UTC 用于文件命名和跨机器审计
- 设备时钟映射：`t_global_ns = a * t_device + b`
- 同步脉冲：独立 `SyncPulseAdapter`，同时保存原始波形和检测事件

---

## 3. 源码结构

```
Exo_Collection_System/
├── ARCHITECTURE.md              ← 架构设计权威文档（必读）
├── README.md                    ← 日常运行说明
├── pyproject.toml               ← 项目元数据与依赖声明
├── build_exe.py                 ← PyInstaller 打包脚本
├── run_collector.py             ← Collector 启动脚本
├── run_data_studio.py           ← Data Studio 启动脚本
├── first_time_setup_and_build.py ← 零参数环境安装+测试+构建
├── config/                      ← 配置文件（Pydantic 校验）
│   ├── app.json
│   ├── storage.json
│   ├── devices/simulated.json, hardware.json
│   ├── protocols/default.json
│   └── quality_rules/default.json
├── schemas/                     ← Manifest JSON Schema（1.0/1.1/1.2）
├── packaging/                   ← PyInstaller .spec 文件
├── src/exo_collection/          ← ★ 核心 Python 包
│   ├── domain/                  ← 领域模型 & 状态机
│   │   ├── models.py            ← Project/Subject/Session/Trial/Artifact
│   │   ├── states.py            ← TrialState 枚举与状态机
│   │   ├── events.py            ← 领域事件
│   │   ├── project_codes.py     ← T/F_BASE/F_STEADY/F_TRANSIENT 项目代码
│   │   └── prompt_labels.py     ← [NEW] 人工 Prompt 标签模型
│   ├── acquisition/             ← 采集管线
│   │   ├── buffers.py           ← 共享内存预览缓冲区
│   │   ├── messages.py          ← WorkerEvent 消息类型
│   │   ├── workers.py           ← CollectorWorker 多进程封装
│   │   ├── preview.py           ← 预览引擎
│   │   ├── recording_stream.py  ← 录制流端点
│   │   └── stream_proxy.py      ← 流代理
│   ├── adapters/                ← 设备适配器
│   │   ├── base.py              ← ModalityAdapter 协议
│   │   ├── ultrasound/          ← 超声（elonxi/raw_ethernet/simulated）
│   │   ├── imu/                 ← IMU（xsens_awinda/simulated）
│   │   ├── encoder/             ← 编码器（teensy_serial/simulated）
│   │   └── sync_pulse/          ← 同步脉冲（simulated）
│   ├── writers/                 ← 写盘器
│   │   ├── binary_block.py      ← 超声分块二进制 Writer
│   │   ├── block_binary_process.py
│   │   └── hdf5_signal.py       ← IMU/编码器 HDF5 Writer
│   ├── orchestration/           ← 编排层
│   │   ├── models.py            ← TrialRunRequest/TrialRunResult
│   │   ├── simulated.py         ← 模拟 Trial 完整流程（核心引擎）
│   │   └── cli.py               ← CLI 入口
│   ├── storage/                 ← 存储层
│   │   ├── manifest.py          ← Manifest Pydantic 模型
│   │   ├── layout.py            ← 目录布局与路径解析
│   │   ├── package.py           ← Trial 最终化打包
│   │   ├── recovery.py / recovery_manager.py ← 崩溃恢复
│   │   ├── checksum.py          ← SHA-256 校验
│   │   └── activity.py          ← 采集活动锁
│   ├── catalog/                 ← SQLite Catalog
│   │   ├── models.py            ← SQLAlchemy ORM
│   │   ├── db.py                ← 数据库连接管理
│   │   ├── repositories.py      ← 查询仓库
│   │   └── migrations/          ← Alembic 迁移
│   ├── apps/                    ← 桌面应用
│   │   ├── collector/           ← Exo Collector
│   │   │   ├── main.py          ← QApplication 入口
│   │   │   ├── window.py        ← 主窗口（采集 UI）
│   │   │   ├── device_preview.py
│   │   │   ├── device_settings.py
│   │   │   ├── preflight.py     ← 采集前检查
│   │   │   └── theme.py
│   │   └── data_studio/         ← Exo Data Studio
│   │       ├── main.py
│   │       ├── window.py        ← 主窗口（管理 UI）
│   │       ├── local_dialogs.py ← 回放/统计对话框
│   │       ├── local_tools.py   ← 回放/质检/统计工具函数
│   │       ├── upload.py / upload_dialog.py
│   │       ├── credential_store.py ← Windows 凭据管理器
│   │       ├── recovery_dialog.py / recovery_service.py
│   │       ├── management.py / management_dialog.py
│   │       ├── service.py
│   │       ├── process_workers.py
│   │       └── external_import_dialog.py / external_import_worker.py
│   ├── timing/                  ← 时间系统
│   │   ├── clock.py, clock_model.py, pulse_detector.py, alignment.py
│   ├── quality/                 ← 质量引擎
│   │   ├── engine.py, config.py
│   ├── protocols/               ← 工况协议
│   │   └── models.py
│   ├── readers/                 ← 二进制读取器
│   │   └── binary_block.py
│   ├── configuration/           ← 配置管理
│   │   ├── app_settings.py, device_profiles.py, adapter_registry.py
│   ├── external/                ← 外部文件导入
│   │   └── importer.py
│   ├── reporting/               ← 报告生成
│   │   └── preview_png.py
│   └── logging_setup.py
└── tests/                       ← 测试
    ├── unit/                    ← 单元测试（~40 个测试文件）
    └── integration/             ← 集成测试（4 个测试文件）
```

---

## 4. 当前开发阶段

按照 ARCHITECTURE.md 的五阶段规划：

| 阶段 | 内容 | 状态 |
|------|------|------|
| **阶段 0** | 数据契约冻结（Manifest Schema、二进制格式、时间模型、接口） | ✅ 完成 |
| **阶段 1** | 稳定采集核心（模拟设备、状态机、Adapter、多进程写盘、实时预览、崩溃恢复） | ✅ 基本完成 |
| **阶段 2** | 本地数据管理（SQLite Catalog、数据树、统计、多模态回放、质量报告） | ✅ 基本完成 |
| **阶段 3** | 外部同步（模拟脉冲、脉冲检测、外部文件导入、时间映射） | 🔶 同步脉冲基础完成，外部导入框架就绪 |
| **阶段 4** | 离线上传（SSH/SCP、远端校验、指数同步、凭据管理） | ✅ 基本完成 |
| **阶段 5** | 现场验证与冻结 | 🔜 待开始 |

### 已实现的核心功能

- ✅ 四模态模拟/真实采集（超声、IMU、编码器、同步脉冲）
- ✅ `hardware` extras profile（真实超声 Raw Ethernet、Xsens Awinda IMU、Teensy 编码器）
- ✅ Trial 完整生命周期管理（含崩溃恢复 `.recording` → `RECOVERABLE`）
- ✅ SQLite Catalog + Alembic 迁移
- ✅ 实时预览（IMU/编码器环形曲线、超声四通道 A-scan）
- ✅ 离线多模态回放（含超声瀑布图 + 当前帧 + A-scan）
- ✅ 质量引擎（版本化规则、PASS/FAIL/WARNING/UNASSESSED）
- ✅ SSH/SCP 离线上传 + 包指纹同步索引 + Windows 凭据管理器
- ✅ Data Studio 活动锁检测与轻量模式
- ✅ 半透明 toast 通知
- ✅ 项目代码分类筛选（T/F_BASE/F_STEADY/F_TRANSIENT）
- ✅ 2026 实验工况协议
- ✅ 编码器五项指标预览与五倍放大
- ✅ 人工 Prompt 标签（受试者 `<` / 工作人员 `>`）—— **本次提交的核心内容**

---

## 5. 最近 30 条提交历史（2026年7月）

```
9416c0d feat(collector): categorize projects and filter conditions
ce7b4a7 feat(protocol): add 2026 experimental conditions
a87fabb fix(collector): tune encoder preview to five-times zoom
4e19fcf fix(collector): enlarge encoder joint angle preview
579561d fix(collector): overlay encoder metrics per motor
eb38f94 feat(collector): preview all encoder measurements
b2b4215 docs(encoder): document frozen position feedback diagnosis
c009461 feat(encoder): support AK80 Teensy 35-byte status frames
d9ef76f feat(data-studio): sync cloud status on startup
01c4d81 style(data-studio): tint Trial rows by sync status
c5c0762 fix(data-studio): refresh verified upload status
59b49aa style(data-studio): add Trial status lights and modality badges
7419345 feat(data-studio): add one-click indexed cloud sync
7acdc8a fix(data-studio): select Python with SSH runtime
9ca3221 feat(data-studio): securely remember SSH passwords
7ba4727 feat(data-studio): sync and visualize remote dataset status
e8ff4d5 feat(upload): mirror local data hierarchy additively
b6f4df9 fix(data-studio): use fixed ADC range for A-scans
d263aad style(data-studio): use thin antialiased A-scan lines
a5414b3 fix(data-studio): show A-scans only in combined playback
956c083 fix(data-studio): make per-channel A-scans visible
f6dd31f fix(data-studio): place A-scan below each waterfall
dfb1d9d feat(data-studio): show current ultrasound frame in playback
57ddb3a feat(data-studio): add combined multimodal playback tab
a7ae2f5 fix(data-studio): render continuous ultrasound waterfall
b8832d9 fix(data-studio): orient ultrasound waterfall by time
13bafc0 fix(data-studio): keep playback controls visible
0847754 feat(data-studio): rebuild tree playback and diagnostics
143fe8a fix(data-studio): 修复 .exo/ 重构引入的路径解析 + 重写离线回放界面
d67c25d feat: semi-transparent toast notifications
```

最近一个月的工作主线：
- **编码器升级**（AK80 Teensy 35字节帧、五项指标预览、五倍放大、电机覆盖标注）
- **Data Studio 完善**（超声瀑布回放、A-scan 可视化、组合回放、离线回放重写）
- **云端同步**（上传状态可视化、启动同步、一键索引同步、SSH 凭据管理）
- **项目与工况**（项目分类筛选、2026 实验工况）
- **人工标签**（Prompt Label 事件捕获、持久化、回放可视化）

---

## 6. 本次提交详细说明（Prompt Labels 功能）

### 6.1 功能概述

在 Trial 录制期间，操作者可以通过键盘按键标记人工事件：

- **`<` 键**：受试者标签（受试者发出信号/状态变化的时刻）
- **`>` 键**：工作人员标签（操作者做出干预/注意的时刻）

### 6.2 修改的文件（16 个文件，+1217 / -35）

**新增文件：**
- [prompt_labels.py](src/exo_collection/domain/prompt_labels.py) — 领域模型：`PromptLabelEvent`（Pydantic BaseModel）、`PromptLabelSource`（SUBJECT/OPERATOR 枚举）、`load_prompt_label_events()` 读取器。Schema 版本 1.0.0，通过 `extra="forbid"` 严格校验。
- [test_prompt_labels.py](tests/unit/test_prompt_labels.py) — 单元测试：验证序列连续性、时间单调性、字段校验。

**修改的核心模块：**

| 文件 | 变更要点 |
|------|---------|
| [messages.py](src/exo_collection/acquisition/messages.py) | 新增 `WorkerEventType.PROMPT_LABEL` |
| [workers.py](src/exo_collection/acquisition/workers.py) | `CollectorWorker` 增加 `_prompt_labels` 独立有界队列（maxsize=256）、`record_prompt_label()` 方法；Worker 进程入口新增 `prompt_label_queue` 参数 |
| [window.py](src/exo_collection/apps/collector/window.py) | **最大变更**（+250 行）：全局 `eventFilter` 捕获 `<` / `>` 按键（排除自动重复）；`_capture_prompt_label()` 调用 Worker；`RingTrace.mark_current()` 在环形缓冲上绘制细红竖线；设备健康表增加两行显示标签计数 |
| [simulated.py](src/exo_collection/orchestration/simulated.py) | **第二大变更**（+170 行）：`_PromptLabelWriter` 类实现持久化 NDJSON 流（每事件 fsync）；`drain_prompt_events()` 在主事件循环中轮询标签队列；边界检查（写盘前/停止后按键丢弃）；写入 Manifest Artifact；空文件清理；`WorkerEventType.PROMPT_LABEL` 回传 |
| [local_tools.py](src/exo_collection/apps/data_studio/local_tools.py) | `TrialPlayback` 数据类增加 `prompt_labels` 字段；`load_trial_playback()` 从 `raw/prompt_labels.jsonl` 加载标签并按 `formal_t0_ns` 转换为 Trial 相对时间 |
| [local_dialogs.py](src/exo_collection/apps/data_studio/local_dialogs.py) | `_SweepWaterfallPlot` 和 `_SweepSignalPlot` 增加 `_prompt_lines` 标记线管理；`_update_prompt_marker_lines()` 在固定窗口循环中渲染虚线（受试者）/实线（工作人员）红竖线；换圈时逐列覆盖而非整屏清空 |
| [repositories.py](src/exo_collection/catalog/repositories.py) | Catalog 数据树统计排除 `prompt_label` 模态 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 新增 §7.4 "人工 Prompt 标签" 章节；数据目录规范增加 `prompt_labels.jsonl`；预览/回放说明更新 |
| [README.md](README.md) | Collector 流程增加第 7 步；回放说明更新；Trial 产物列表增加 `prompt_labels.jsonl` |
| [.gitignore](.gitignore) | 新增 `log/` 目录忽略规则 |

### 6.3 架构决策

- 标签**不是** Modality Adapter，不参与设备 READY 判定或模态丢失质检
- 标签队列为**独立有界控制队列**（maxsize=256），不与超声/IMU/编码器的高吞吐数据队列混用
- 每个事件写入后立即 `fsync`，确保崩溃后不丢已确认标签
- 若 Trial 全程无标签，不发布空 Artifact（Writer 在 close 时删除空文件）
- 按键必须发生在 `RECORDING` 或 `WAITING_SYNC` 状态期间；自动重复按键被忽略
- 停止边界后的标签不进入 Trial

---

## 7. 技术栈

| 领域 | 技术选型 |
|------|---------|
| Python | 3.11（64-bit），NumPy >=1.26, <2 |
| GUI | PySide6 + pyqtgraph |
| 数据模型 | Pydantic >=2.7, <3 |
| 结构化存储 | HDF5（h5py >=3.10）、分块二进制（NumPy memoryview） |
| 本地索引 | SQLite + SQLAlchemy >=2.0 + Alembic |
| 进程通信 | `multiprocessing` Queue/Pipe + shared memory |
| SSH/SCP | Paramiko + scp |
| 打包 | PyInstaller（两个 spec 文件 → ExoCollector.exe / ExoDataStudio.exe） |
| 测试 | pytest + hypothesis |
| 硬件依赖 | pyserial, scapy, pythonnet（可选 `[hardware]` extras） |

---

## 8. 日常开发命令

```powershell
# 环境初始化
python first_time_setup_and_build.py

# 启动应用（无需命令行参数）
python run_collector.py
python run_data_studio.py

# 运行全部测试
python -m pytest

# 只运行单元测试
python -m pytest tests/unit/

# 只运行集成测试
python -m pytest tests/integration/

# 编译打包
python build_exe.py

# 安装硬件依赖（仅真实设备开发）
python -m pip install -e ".[hardware]"
```

---

## 9. 关键注意事项

1. **Xsens NumPy ABI 兼容性**：`numpy` 必须保持 `>=1.26,<2`，因为 Xsens MT SDK 2025.2 wheel 使用 NumPy 1.x ABI，升级到 2.x 会导致导入失败。

2. **两个应用边界**：
   - Collector 是 `.recording`、`.partial` 和当前 Trial 原始数据的唯一写入方
   - Data Studio 默认只读取 FINALIZED/ABORTED/RECOVERABLE Trial
   - Data Studio 检测到 Collector 活动锁后进入轻量模式（禁用大文件回放、全盘统计、上传）

3. **SQLite** 启用 WAL + `busy_timeout` + 短事务；Collector 写，Data Studio 主要读。

4. **密码安全**：SSH 密码仅通过 Windows 凭据管理器加密保存，禁止写入 JSON、日志、SQLite、Manifest 或命令行参数。

5. **Trial 现场流程**：先选择项目代码和受试者 → 连接设备 → 等待 READY → 点击开始 → 自由时长采集 → 人工点击停止 → 自动最终化。不预设固定采集时长。

6. **数据恢复**：程序启动时扫描 `.recording` 和 `.partial` 目录，验证完整块和 CRC → 截断不完整尾块 → 重建索引 → 生成 RECOVERABLE Trial。

---

## 10. 待办与后续方向

- **阶段 5 现场验证**：长时间压力测试、真实设备故障注入、安装包环境检查
- **测力台/动作捕捉**：外部模态导入器已有框架，需结合具体设备协议完善
- **编码器 AK80**：当前适配 35 字节 V3 固件帧格式，后续固件版本需验证兼容性
- **超声压缩**：当前 `none`，若实际吞吐不满足需求可增加轻量块压缩
- **SFTP 分块续传**：当前 SCP 对大文件无断点续传，网络不稳定场景可增加 SFTP 后端

---

## 11. Collector 可停靠预览工作区

- 右侧固定纵向滚动布局改为 PySide6 原生 `QDockWidget` 工作区。
- 超声、IMU、编码器、动捕、EMG 和测力台各自注册独立面板；支持拖动、改变大小、浮动、关闭、重新添加、默认平铺和放大预览区。
- 面板按模态键动态绑定预览事件。关闭仅隐藏视图，不停止 Adapter、预览进程或记录支路。
- 停靠位置、浮动状态和显隐状态通过共享 `QSettings` 保存，应用重启后恢复。
- 动捕改为显示全部 Marker 最新 XYZ 的表格；IPC 循环曲线仍仅携带前 8 个 Marker，避免无界增长。
- 测力台面板已接受 `fx/fy/fz/mx/my/mz` 通用预览格式；真实测力台 Adapter 尚未接入。
- Collector 相关核心测试 151 项通过；完整套件在当前 Windows 未启用长路径的环境中仍有既有深层 Trial 临时路径失败。
