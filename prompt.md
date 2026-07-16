# DeepSeek 续开发总指令：完成 Exo Collection System 第一可运行里程碑

你现在接手的是一个**已经完成绝大部分实现、但尚未完成最终收尾、提交、重建和推送**的 Windows/Python 桌面项目。不要重新搭空目录，不要推倒重写，不要把现有实现替换成演示壳子。你必须从当前工作树继续，先审计现状，再修复剩余问题，最后完成测试、端到端验证、Windows 打包、分阶段 Git 提交和 GitHub 推送。

本文是执行合同。除非代码和测试给出更强的事实证据，否则不要擅自改变这里定义的数据语义、安全边界和验收标准。

---

## 1. 角色、目标和完成定义

你的角色是本项目的主开发者、测试负责人和发布负责人。你的目标不是“给建议”，而是完成当前能够在缺少真实厂商 SDK 的条件下完成的所有功能，并交付可运行、可测试、可打包、可审计的第一里程碑。

项目根目录：

```text
D:\Master\2_学业\my_project\2_外骨骼课题\1_exo_数据采集方案\Exo_Collection_System
```

完成必须同时满足：

1. 当前工作树中的已有实现全部审查完毕，不丢失任何用户或前任开发者改动。
2. 所有源码能够编译，完整测试稳定通过，不存在 quiet 模式测试挂起和遗留 Python/Worker 进程。
3. Collector 能通过四种模拟设备完成真实的多进程 Trial：设备预检、arm、等待同步、正式触发、人工/测试停止、写盘、质控、最终化和 Catalog 入库。
4. Data Studio 能扫描 Catalog/Manifest、显示层级树、筛选和统计、回放、质检审核、恢复、外部模态导入以及人工离线 SSH/SCP 上传。
5. 两个桌面应用正常启动，日常用户不需要提供任何命令行参数；数据根目录从 UI 选择并持久化。
6. 默认发布构建在干净 Git 工作树上运行，完成测试、两个 EXE、冻结进程 smoke、ZIP 完整性验证和发布清单；若安装了 Inno Setup，再生成安装器。
7. 所有改动按架构、核心、Collector、Data Studio、发布合理拆分提交。
8. `main` 推送到：

```text
git@github.com:liangji-seu/Exo_Collection_System.git
```

9. 真实硬件协议缺失的内容必须明确为“接口已完成、真机未验证”，绝不伪造厂商实现或性能结论。

只有源码、测试、源码 smoke、冻结 EXE smoke、发布包检查、Git 提交和 push 全部完成后，才允许报告“完成”。

---

## 2. 首要保护规则：当前工作树不能丢

当前仓库存在大量有意保留且尚未提交的改动。严禁执行：

```text
git reset --hard
git checkout -- <file>
git restore <file>
git clean -fd
```

也不要删除、覆盖或回退任何不理解的现有修改。所有文件操作必须限定在项目目录内；递归删除构建目录前必须解析并验证绝对路径，拒绝穿越 junction/reparse point。

开始工作后首先执行并保存输出：

```powershell
Set-Location 'D:\Master\2_学业\my_project\2_外骨骼课题\1_exo_数据采集方案\Exo_Collection_System'
git status --short
git diff --stat
git diff --check
git log --oneline --decorate -12
git remote -v
```

当前已知 Git 基线：

- 当前分支：`main`
- 当前本地 HEAD：`18a3c97 Add one-click first-time Windows setup`
- 远端 `origin/main` 当前已知位置：`82d0c6f`
- 本地在本轮大改之前已经领先远端 3 个提交。
- 当前工作树约有 44 个已跟踪文件被修改、33 个未跟踪入口；数字可能因本 `prompt.md` 和后续修复略有变化。
- 这些大量改动都是本轮功能开发，不能清理。

架构基线和此前的单仓库双应用架构提交已经存在：

- `58ea84b Add initial system architecture design`
- `0a26b2e Document dual desktop application architecture`

---

## 3. 必须先完整阅读的文件

按以下顺序完整阅读，不要只读摘要：

1. `ARCHITECTURE.md`
2. `README.md`
3. `pyproject.toml`
4. `schemas/README.md`
5. `schemas/manifest-v1.0.0.json`
6. `schemas/manifest-v1.1.0.json`
7. `config/devices/simulated.json`
8. `config/protocols/default.json`
9. `config/quality_rules/default.json`
10. `config/storage.json`
11. `src/exo_collection/domain/models.py`
12. `src/exo_collection/domain/states.py`
13. `src/exo_collection/storage/manifest.py`
14. `src/exo_collection/storage/layout.py`
15. `src/exo_collection/storage/activity.py`
16. `src/exo_collection/storage/recovery_manager.py`
17. `src/exo_collection/catalog/repositories.py`
18. `src/exo_collection/orchestration/models.py`
19. `src/exo_collection/orchestration/simulated.py`
20. `src/exo_collection/acquisition/workers.py`
21. `src/exo_collection/apps/collector/preflight.py`
22. `src/exo_collection/apps/collector/window.py`
23. `src/exo_collection/apps/collector/main.py`
24. `src/exo_collection/apps/data_studio/window.py`
25. `src/exo_collection/apps/data_studio/main.py`
26. `src/exo_collection/apps/data_studio/management.py`
27. `src/exo_collection/apps/data_studio/local_tools.py`
28. `src/exo_collection/apps/data_studio/quality_reviews.py`
29. `src/exo_collection/apps/data_studio/recovery_service.py`
30. `src/exo_collection/apps/data_studio/upload.py`
31. `src/exo_collection/external/importer.py`
32. `packaging/build_windows.ps1`
33. `packaging/collector.spec`
34. `packaging/data_studio.spec`
35. `packaging/installer/ExoCollectionSystem.iss`
36. `tests/unit/test_release_packaging.py`
37. `tests/unit/test_collector_ui.py`
38. `tests/unit/test_data_studio.py`
39. `tests/integration/test_simulated_trial.py`

随后使用 `rg --files src tests` 和 `rg -n` 检查所有模块、测试和引用，不要漏读因动态 import 或 Windows `spawn` 才会使用的文件。

---

## 4. 当前真实开发状态

### 4.1 已经完成并有测试覆盖的主体功能

以下功能已经存在于当前工作树。你的任务是审查、修复和最终验证，不要另写一套平行实现。

#### 核心与存储

- Project、Subject、Session、Trial、Artifact、Condition 领域模型。
- Trial 状态机，包含 `WAITING_SYNC`、`RECOVERABLE`、`ABORTED` 和 `FINALIZED`。
- Manifest Pydantic 模型，当前版本 `1.1.0`，向后兼容读取 `1.0.0`。
- `1.1.0` 强制 `project_code` 为 `F/T`，`subject_code` 为三位 ASCII 数字；公开 JSON Schema 与运行时约束应一致。
- Catalog 对同一 `trial_uuid` 的 project/subject/session/condition/repeat 身份冲突实行事务内 fail-closed，不得覆盖原记录。
- `F/T/001/session_uuid/trials/trial_uuid` 的可读目录布局；UUID 和 Manifest 仍是权威关联，不能用文件名猜关系。
- `.recording/.partial/.aborted/.building` 按 Windows 大小写不敏感语义识别，同时不能误伤 `trial.partial.backup` 等名称中段。
- 临时文件、同卷原子发布、SHA-256、Catalog 重建和基础恢复。
- Windows 安全路径校验：绝对路径、`..`、ADS 冒号、非法字符、控制字符、尾随点/空格、保留设备名等必须拒绝。
- 活动锁和心跳；最近损坏的活动锁必须 fail closed，Data Studio 进入轻量模式；Windows PID 探测不得误杀被检测进程。

#### 采集与同步

- 统一 Modality Adapter、Writer、WorkerEvent、DeviceHealth 接口。
- 可配置的超声、IMU、编码器、模拟同步脉冲四种模拟设备。
- 超声独立 Writer 进程；原始队列有界且不能静默丢弃；预览队列允许丢弃。
- `SharedPreviewBuffer` 使用 `multiprocessing.shared_memory` 传递降采样预览，大数组不经普通 Queue 重复序列化。
- Collector 正常 UI 不包含固定采集时长，也不要求操作者。
- 点击开始后先进入 `WAITING_SYNC`；所有 Writer 可以保留触发前原始数据，但首个合格同步上升沿才建立正式 `t0` 并进入 `RECORDING`。
- 现场 Trial 由用户人工点击受控停止；`duration_s` 只允许内部 smoke/自动测试使用，且从正式 `t0` 计算。
- 没有合格触发时不得生成看似正常的 FINALIZED Manifest；保留 `.recording`、Journal、原始波形和失败报告供恢复/中止审计。
- Adapter 在最后一次健康轮询之后、停止之前发生故障的竞态已经增加停止后汇总检查；致命故障必须保留为 RECOVERABLE。
- 慢 Writer 导致原始队列满、HDF5 写盘异常、父进程死亡、无 UI 消费等边界已有测试。

#### Collector UI

- 项目下拉框：`F — 正式`、`T — 测试`，默认 `T`。
- 受试者编码默认 `001`，严格三位数字。
- 数据根目录通过 UI 选择，并通过共享 QSettings 持久化；Collector 与 Data Studio 使用同一设置命名空间。
- 设备预检已经迁移到独立 `spawn` 进程，不应在 GUI 线程调用模拟器、真实 SDK 或 fsync 探测。
- 开始按钮必须由四个关键模态 READY 和有效受试者编码共同门控。
- 预检执行真实的模拟 Adapter 生命周期、原始批次、同步上升沿、可写性、磁盘空间和 1 MiB fsync 写探针；未配置真实吞吐阈值时只展示测量值，不能假装通过真实最大速率验证。
- 顶部总状态：未连接、等待同步、可采集、采集中、保存中、失败。
- 设备表：连接/健康、样本或帧数、实际速率、丢包、队列、最近更新时间。
- 同步面板：状态、trigger 计数、首触发、质量。
- 四通道 A-mode 预览：当前 A-scan、最近 8 秒灰度瀑布、峰值深度趋势、峰值强度趋势；切换通道不能混淆历史。
- IMU、Encoder 实时曲线和同步/状态/告警时间线。
- 全零等格式级异常可直接报警；弱信号、边界、滑移和依赖真实标定的阈值必须明确显示 `UNASSESSED`。
- 实验元数据对话框：身高、体重、腿长、性别、年龄、肌肉、左右侧、近中远位置、4 通道映射、固定方式、绑带压力描述、是否重贴、跑台速度、助力、负载、坡度和 Trial 备注。
- 元数据已按 `(project_code, subject_code)` 隔离，受试者切换不能串数据；工况切换应清除 measured condition 和 trial notes；一次性备注/重贴字段在 Trial 后不能静默沿用。必须复核这一实现是否完整。
- WorkerEvent 必须校验 `trial_uuid`，其他 Trial 的事件不能改变当前 Trial UI 或最终状态。
- 关闭流程应执行有界的受控停止，然后 `terminate_for_recovery`/kill 回退，绝不能让卡住的 SDK/Worker 使窗口永远无法关闭；强制终止必须保留可恢复语义。

#### Trial 自动质控输出

每个成功 Trial 当前应包含：

```text
manifest.json
checksums.sha256
raw/ultrasound.bin
raw/ultrasound.meta.json
raw/ultrasound.idx
raw/imu.h5
raw/encoder.h5
raw/sync_pulse.h5
derived/configuration_snapshot.json
derived/statistics.json
derived/quality_rules_snapshot.json
reports/quality_report.json
reports/device_status.csv
reports/sync_check.csv
reports/sync_manifest.json
reports/us_quality_preview.png
reports/imu_encoder_preview.png
reports/warnings.txt
logs/trial.jsonl
```

所有 Manifest Artifact 必须使用 UUID 关联 Trial，保存大小、SHA-256、来源 Artifact UUID、媒体类型、派生算法/规则版本等，不能只靠文件名。

质量规则使用严格、版本化、`extra=forbid` 的 Pydantic 配置。每条规则输出：

```text
PASS / FAIL / WARNING / UNASSESSED
```

质量 A 只允许在所有必需结构规则确实执行并通过时产生；空规则集或跳过规则不能产生 A。没有真实硬件依据的量程、饱和、跳变、SNR、边界和滑移阈值必须保持 `UNASSESSED`，不能写虚构常量。需要复核 `warnings.txt` 和 UI 是否足够醒目地提示存在 UNASSESSED 规则，避免用户把结构完整 A 误解为“所有硬件信号质量均已验证”。可以增加明确提示，但不要把未标定阈值伪装成 FAIL。

#### Data Studio

- 后台扫描 Manifest/SQLite，显示 `Project → Subject → Session → Trial → Artifact`。
- 管理筛选：项目、受试者、Session、工况、质量、起止日期、全文搜索；过滤后保留父层级。
- 基础统计和管理摘要：Trial 数、总时长、Artifact 字节、工况覆盖率、重复轮次、FINALIZED、待恢复、ABORTED、待质检、待上传、sidecar 错误。
- 当前筛选清单原子导出 CSV + JSON，禁止写入最终化 Trial 或 Annex 内。
- External Annex/External File 作为 Trial 子节点展示，包含 VERIFIED/INVALID、映射质量、锚点、大小和错误 tooltip。
- 离线回放使用共享时间游标，支持 play/pause/seek、0.25～4x、四通道 US、IMU、Encoder、同步事件；读取必须有界，不能一次加载整份大型超声。
- 全量统计、SHA 校验和质量审核。
- 人工质量审核使用独立、追加式、以 Manifest SHA-256 锚定且哈希链连接的 sidecar；不得改写原 Manifest/原始文件；并发追加必须有跨进程锁。
- 外部测力台/动捕/其他文件导入到：

```text
dataset_root/external_annexes/<trial_uuid>/<annex_uuid>/
```

  保留源文件精确字节和 SHA-256；支持手工脉冲或 CSV 列；单脉冲只估 offset，多脉冲拟合 affine drift + residual；通过 `.building` 临时目录和原子发布，不改写已最终化 Trial。
- 恢复流程只读发现 `.recording`，拒绝活动包、reparse point、symlink、hardlink alias、中部 CRC 损坏；只有证据完整且人工确认才可最终化，否则可追加式审计后原子改名 `.aborted`。
- 人工离线 SSH/SCP 上传在独立 `spawn` 进程执行；只接受 FINALIZED；采集期间禁止；首次主机指纹必须确认并保存；逐文件远端 SHA-256 验证后原子发布；支持取消、重试和审计。
- 密码和私钥口令只通过进程创建后的匿名 Pipe 发送；不得进入配置、日志、SQLite、命令行参数、对象 repr 或审核 JSON。
- Data Studio 发现 Collector 活动或不可安全解析的活动锁时进入轻量模式，禁用回放、全盘统计、校验、审核、导入、恢复、管理扫描、导出和上传。
- 已打开 RecoveryDialog 后若 Collector 开始采集，必须立即切换只读/阻止操作；后端 AcquisitionLock 仍是最终授权边界。
- 关闭 Data Studio 时应有界终止上传和所有进程 Worker；QRunnable 无法强杀时窗口保持存活、异步重试，不能在任务仍运行时销毁对象或让应用挂死。

#### Windows 发布

- `Run_ExoCollector.cmd` 与 `Run_ExoDataStudio.cmd`：双击运行成品 EXE，不向应用传日常参数。
- `Run_ExoCollector_From_Source.cmd` 与 `Run_ExoDataStudio_From_Source.cmd`：源码调试入口。
- `First_Time_Setup.cmd`：新电脑零参数初始化、安装依赖、测试、构建和打包。
- `Build_Windows.cmd`：默认严格发布构建。
- PyInstaller 两个 spec 收集 Windows `spawn` 动态导入边界、Paramiko/SCP、Adapter、Writer、QC、reporting、Data Studio tools、external、recovery、management。
- 发布脚本已有/应有：干净 Git 门禁、显式 `-AllowDirtyWorkingTree` 开发开关、当前 checkout 路径验证、reparse point 删除防护、全 payload SHA、ZIP 重新打开逐项验证、ZIP sidecar SHA、可选 Inno、构建 provenance。
- 默认发布必须是干净树。dirty 开关只能用于开发探测，`BUILD_MANIFEST.json` 和 Trial 运行时 provenance 都必须如实标记 `+dirty` 或 `+provenance-unavailable`。

### 4.2 当前验证结果

重置前后已经得到这些事实：

- `compileall`：通过。
- 完整测试收集数：`331`。
- `pytest -vv -x`：`331 passed in 45.90s`。
- Collector UI/异步预检源码 smoke：通过。
- Collector 独立进程模拟 Trial smoke：通过。
- Data Studio Catalog/management 源码 smoke：通过。
- `git diff --check`：没有 whitespace error，只有 Windows LF→CRLF 提示。
- 发布静态测试 `tests/unit/test_release_packaging.py` 已包含 8 项并在 verbose 全测中通过。

### 4.3 当前最重要的未解决问题

不要忽略下面的问题：

1. 两次运行：

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest -q
```

  都超过 150～240 秒未退出，并留下 pytest 父/子 Python 进程；而 `pytest -vv -x` 同一套 331 项在约 46 秒通过。这是**真实的退出/时序竞态或遗留进程问题**，必须查清，不能因为 verbose 通过就忽略。
2. 之前挂起的具体进程命令行为 `.venv\Scripts\python.exe -m pytest -q` 及其系统 Python 子进程。接手时先检查是否仍有本项目 pytest 遗留；只终止命令行明确属于本项目测试的进程，不要无差别杀掉所有 Python。
3. UI 生命周期和发布可追溯性子任务在额度中断前写入了大量代码，但没有完成正式人工收尾报告。虽然现有 331 项 verbose 测试通过，仍需逐行审查，防止半截实现。
4. 当前所有本轮功能改动尚未提交。
5. 当前 `dist/*.exe` 和 `release/*.zip` 是旧构建，`BUILD_MANIFEST.json` 指向旧 HEAD：

```text
18a3c97487a6a42c75122da14fe84ccfdde55787
built_at_utc: 2026-07-15T17:22:59.8500515Z
```

  它们不包含当前最新功能，绝不能交付。
6. 默认 `packaging/build_windows.ps1` 现在要求干净 Git 树；因此必须先审计并分阶段提交，再运行正式构建。
7. 最新代码尚未执行最终 PyInstaller 双 EXE 构建和冻结 smoke。
8. 最新提交尚未 push 到远端。

---

## 5. 不可更改的核心设计

### 5.1 Trial、身份和目录

- Trial 是最小完整采集单元，一个 Trial 只能有一个明确工况。
- Project、Subject、Session、Trial、Artifact 均使用 UUID 作为真实主键。
- `F/T` 和 `001` 只是可读分区，不是关系推断依据。
- Project UUID 应按 F/T 稳定派生；Subject UUID 应在 Project UUID 命名空间下按三位编码稳定派生；显式提供的 UUID 不得被重写。
- Session UUID 在一次应用会话和同一数据根/项目/受试者上下文中复用，切换上下文产生新 Session。
- Catalog 扫描时校验 Manifest 的 project/subject/session/trial 与目录上下文，防止复制到错误目录的合法 Manifest 覆盖 Catalog。
- 最终化数据不可通过 Data Studio 原地修改。人工审核、上传审计和外部附录必须使用 sidecar/annex。

目标目录：

```text
dataset_root/
  F|T/
    001/
      <session_uuid>/
        session.json
        trials/
          <trial_uuid>.recording/    # 采集中/待恢复
          <trial_uuid>/              # FINALIZED
  external_annexes/
    <trial_uuid>/
      <annex_uuid>/
  .upload-audit/
  .quality-reviews/
  catalog.sqlite3
```

不要采用 `S001_WALK...h5` 文件名作为关联机制。可读显示可以存在，但关联必须来自 UUID + Manifest。

### 5.2 状态机

合法主路径：

```text
IDLE
→ PREPARING
→ READY
→ WAITING_SYNC
→ RECORDING
→ STOPPING
→ FINALIZING
→ FINALIZED
```

失败路径至少包括：

```text
PREPARING → FAILED
WAITING_SYNC → RECOVERABLE
RECORDING → ABORTED/RECOVERABLE
FINALIZING → RECOVERABLE
RECOVERABLE → FINALIZED（证据完整且人工确认）
RECOVERABLE → ABORTED（人工确认保留）
```

只有 Orchestrator 可以改变状态，UI 只能发命令和展示事件。

### 5.3 时间语义

- 采集主时间使用 `time.monotonic_ns()`/主机单调时钟。
- UTC 只用于审计展示和跨系统记录，不能驱动持续时间或实时调度。
- 所有模态保留：source sequence、sample/frame index、device timestamp、host monotonic timestamp、host UTC timestamp。
- Writer arm 后可以保存 pre-trigger 原始数据。
- 首个通过双阈值迟滞、最小脉宽和去抖规则的同步上升沿建立正式 `t0`。
- 正式 duration 是 `stop_host_monotonic_ns - t0`，不是设备启动到停止。
- 所有同步边沿、脉宽、间隔、正式窗口归属和 clock mapping 写入 `sync_manifest.json`。
- 单外部脉冲只能拟合 offset；多脉冲才允许拟合 drift/scale。

### 5.4 超声二进制格式

超声原始数据绝对不能保存 CSV。

当前固定块头：

```text
magic = b"EXOUSBLK"
format_version = 1
struct = <8sHHQQQQqQQII
```

字段顺序：

```text
magic
format_version
header_size
sequence
first_sample_index
sample_count
payload_nbytes
device_timestamp
host_monotonic_ns
host_utc_ns
flags
payload_crc32
```

要求：

- 显式 little-endian、固定大小、无本机对齐。
- payload 是连续 C-order NumPy 数组。
- dtype 和单样本 shape 记录在 `ultrasound.meta.json`。
- 每块 CRC32 必须验证。
- sequence 重复/倒退、sample range 重叠、截断 header/payload、中部 CRC 错误必须显式报错。
- 文件尾部的截断块可以通过恢复流程截断；中部损坏绝不能“跳过后继续”或静默截断。
- `.idx` 是可重建派生索引，原 `.bin` 才是事实来源。
- index magic `b"EXOIDX01"`、version 1；index entry 为 `<QQQQ>`。
- Reader 默认只读；索引重建必须显式请求，不能打开文件就偷偷修改。

### 5.5 IMU/Encoder/Sync HDF5

使用 `exo-hdf5-signal` 1.0.0，所有 dataset 同步扩展，至少包含：

```text
/samples/data
/samples/sample_index
/samples/device_time
/samples/host_monotonic_ns
/samples/host_utc_ns
/samples/source_sequence
/events/discontinuities
/events/records
/metadata/channels
/metadata/units
/metadata/device_json
/metadata/trial_json
/metadata/clock_model_json
```

规则：

- 单 Writer 所有权；不同线程/进程误用要拒绝。
- chunked append。
- 正常 flush/close 后才设置 `closed_cleanly=true`。
- 异常上下文必须留下 `closed_cleanly=false` 或明确异常，不能伪装成功。
- HDF5 内嵌 Trial UUID、F/T、受试者编码、Session UUID、工况、实验元数据、软件/时钟策略等写盘前已知事实；最终化后不得回写原始 HDF5。

### 5.6 进程和背压

- UI 线程只做界面更新和轻量验证。
- 预检、原始采集、超声写盘、统计、回放、恢复、外部导入、管理扫描和上传按现有设计放进独立 Worker。
- 原始数据队列有界，满时必须触发致命错误和可恢复 Trial，绝不能静默丢原始数据。
- 预览/健康遥测可丢弃和覆盖；状态/终态事件不可静默丢失。
- 共享内存 marker 的 generation、长度、x/value 对齐必须验证；读取到新 generation 时不得把旧 marker 元数据和新数组拼接。
- Windows `spawn` target 必须是模块顶层可 pickle 函数，并被 PyInstaller spec 显式收集。
- 所有 terminate 后都要有有界 join；仍存活则 kill；只有退出后才能 close Process/Pipe/Queue/shared memory。
- 强杀采集进程后数据语义是 RECOVERABLE，不是 FINALIZED。

---

## 6. 第一优先级：定位并修复 `pytest -q` 挂起

这是接手后的第一项代码工作。不要直接开始提交或打包。

### 6.1 清理并确认测试环境

先只查看属于本项目的 Python 命令行：

```powershell
Get-CimInstance Win32_Process |
  Where-Object {
    $_.Name -eq 'python.exe' -and
    $_.CommandLine -like '*Exo_Collection_System*pytest*'
  } |
  Select-Object ProcessId,ParentProcessId,CommandLine
```

若确实是此前遗留的本项目 pytest，再按具体 PID 终止；不要执行 `Stop-Process -Name python`。

### 6.2 稳定复现

分别运行并记录：

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -vv -x
.\.venv\Scripts\python.exe -m pytest
```

每次运行前后记录 Python 子进程。使用合理外部超时，但不要把超时当通过。

### 6.3 缩小范围

按文件组二分，优先检查最近新增生命周期测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/unit/test_collector_preflight.py
.\.venv\Scripts\python.exe -m pytest -q tests/unit/test_collector_ui.py
.\.venv\Scripts\python.exe -m pytest -q tests/unit/test_data_studio.py
.\.venv\Scripts\python.exe -m pytest -q tests/unit/test_offline_upload.py
.\.venv\Scripts\python.exe -m pytest -q tests/unit/test_data_studio_management_ui.py
.\.venv\Scripts\python.exe -m pytest -q tests/integration/test_simulated_trial.py
```

然后组合运行，找出只有快速 quiet 时序才触发的竞态。

### 6.4 重点怀疑点

逐项审计：

1. `CollectorPreflightWorker.close()` 是否在 Queue feeder 尚未结束时调用阻塞的 `join_thread()`。
2. `CollectorWindow.closeEvent()` 是否在 fake Worker/真实 Worker 的 `is_alive`、`join`、`close` 之间有重入或 QTimer 重试竞态。
3. `CollectorWorker.terminate_for_recovery()` 是否总是关闭 SharedMemory、Queue、Process handle；异常路径是否遗漏。
4. Data Studio `closeEvent()` 的 `_shutdown_retry_pending` 是否可能让一个隐藏窗口永远存活。
5. QThreadPool/QRunnable 是否仍持有 QObject signal target，导致 QApplication/pytest session 不退出。
6. RecoveryDialog 的活动计时器是否在对话框销毁后仍运行。
7. `UploadWorkerHandle.terminate_for_shutdown()` 和 fake process 测试是否留下 Queue/Pipe/Process。
8. `multiprocessing.Queue.cancel_join_thread()`、`close()`、`join_thread()` 是否由正确进程调用。
9. 测试是否创建 `QApplication`、窗口或 QTimer 后没有 `close/deleteLater/processEvents`。
10. `pytest -q` 更快导致前一个 Worker 还未真正退出时下一个测试启动，产生全局进程或 Qt 状态冲突。

### 6.5 修复标准

- 不允许用增加几十秒 sleep 掩盖竞态。
- 关闭必须基于明确状态、终态事件和进程存活检查。
- 所有测试无论成功、异常、start failure、cancel、强杀，都必须在 `finally` 中释放资源。
- 全测 `pytest -q`、`pytest -vv -x`、默认 `pytest` 各至少连续通过 2 次。
- 每次结束后无属于本项目的 pytest、collector-core、writer、preflight、Data Studio process worker 遗留。
- 增加能够在修复前失败、修复后稳定通过的回归测试。

---

## 7. 第二优先级：审查两个中断子任务留下的实现

### 7.1 Collector/UI 生命周期

重点文件：

```text
src/exo_collection/apps/collector/preflight.py
src/exo_collection/apps/collector/window.py
src/exo_collection/apps/collector/main.py
src/exo_collection/acquisition/workers.py
tests/unit/test_collector_preflight.py
tests/unit/test_collector_ui.py
```

必须确认：

- 预检 Worker 真实使用 `multiprocessing.get_context("spawn")`。
- GUI 点击预检立即返回事件循环，按钮显示进行中，不冻结界面。
- 预检完成、失败、进程异常退出、Queue feeder 延迟、窗口关闭和 start failure 都能有界回收。
- 预检结果属于当前数据根目录；用户切换根目录后旧结果不能继续放行。
- 预检进程退出后 UI 文案应表达“预检通过/可采集”，不要声称一个已经关闭的模拟器仍保持物理连接。
- 真正开始 Trial 时 Orchestrator 会重新连接、准备和启动设备；失败则不产生正常 Trial。
- 预检测得写速率不等于真实超声最大负载证明；阈值为空时 UI 必须写“待真实设备最大速率确定”。
- WorkerEvent 的 `trial_uuid` 对当前 Worker 事件必须一致。对于 legacy fake event 若字段为空，只允许在测试/兼容路径按明确规则接受，其他 Trial 的 terminal 必须拒绝并报警。
- closeEvent 第一次先受控停止，超时后强制 recovery；窗口不能永远卡住，也不能在 Writer 正常 flush 前过早 kill。
- `main._run_ui` 和 Window close 逻辑不要重复产生两个互相竞争的 shutdown 状态机。
- Collector smoke 应真正等待异步预检完成，失败或超时返回非零；冻结 EXE 也要覆盖这个 spawn target。

元数据规则：

- `(F,001)` 与 `(T,001)` 是两个不同身份缓存。
- 001 切换 002：不能继承 001 的人口学、探头、实测工况或备注；切回 001 可以恢复该身份的长期元数据，但不要恢复已清除的一次性备注。
- 工况从 WALK 切换 STAND：人口学和探头位置可以保留，`measured_condition`、`trial_notes` 必须清空并在 UI 提示。
- 每个 Trial 结束后清空 `trial_notes` 和 `probe_reapplied`；是否保留速度/助力等必须与工况规则一致并有测试。
- 不要把元数据悄悄写入全局配置；当前只要求应用会话内安全默认和 Trial 快照。

另外审查 `TrialRunRequest.subject_group` 当前默认值。如果仍是 `"simulated"`，Collector 的正式 F 项目可能被错误标记为 simulated group。模拟设备来源应该记录在设备/配置 provenance，不应污染受试者分组。将交互式默认改为 `None` 或明确的 `not_recorded`，并更新兼容测试；不要伪造受试者分组。

### 7.2 Data Studio 生命周期

重点文件：

```text
src/exo_collection/apps/data_studio/window.py
src/exo_collection/apps/data_studio/main.py
src/exo_collection/apps/data_studio/recovery_dialog.py
src/exo_collection/apps/data_studio/process_workers.py
src/exo_collection/apps/data_studio/upload.py
tests/unit/test_data_studio.py
tests/unit/test_offline_upload.py
```

必须确认：

- 损坏活动锁 sentinel（例如 pid=0）在 banner 中显示“活动锁不可安全解析，保守进入轻量模式”，不得显示“Collector PID 0 正在采集”。
- 已打开 RecoveryDialog 后出现活动锁，操作按钮立即禁用，并显示原因；即使 UI 漏掉，后端仍拒绝。
- 上传取消检查覆盖单个大 SCP 文件内部和远端 SHA 读取循环。
- Collector 在上传中开始后，上传立即停止、清理 staging、写无凭据失败审计。
- upload terminate 后若仍存活必须 kill；kill 后仍存活必须显式错误，不能假装 close 成功。
- Data Studio 主窗口关闭不会销毁仍有 QRunnable 的对象；使用异步关闭重试而不是 GUI 线程长阻塞。
- 管理 refresh/summary/export、playback/checksum、external import、recovery、upload 每类 Worker 的 start failure、正常完成、异常、取消、关闭均有资源回收测试。
- Data Studio frozen `--smoke-test` 必须等待 Catalog refresh 和至少一次 management refresh spawn 完成，不能只在管理子进程 import 前就退出。

### 7.3 发布和 provenance

重点文件：

```text
packaging/build_windows.ps1
packaging/collector.spec
packaging/data_studio.spec
packaging/installer/ExoCollectionSystem.iss
packaging/installer/README.md
packaging/installer/README_START_HERE.txt
scripts/first_time_setup_and_build.ps1
tests/unit/test_release_packaging.py
```

逐项确认：

1. 默认构建要求 Git 可用、HEAD 可解析、工作树干净。
2. `-AllowDirtyWorkingTree` 只能显式调用，不能由 `Build_Windows.cmd` 默认传入。
3. 构建前后 HEAD 不变；默认 clean build 期间工作树不能变脏。
4. `.venv` 导入的 `exo_collection.__file__` 必须位于当前 checkout 的 `src`，防止串到另一个仓库。
5. spec 嵌入统一的 build-info；两个 EXE 的 commit/dirty/version 一致。
6. 源码 Trial `_git_commit()`：clean 返回 40 位 SHA；dirty 返回 `<sha>+dirty`；无 Git 返回 `<sha-or-unknown>+provenance-unavailable`。不要让 dirty Trial 看似 clean。
7. `BUILD_MANIFEST.json` 记录版本、commit、dirty/provenance、UTC、目标、硬件支持边界和所有 bundle payload 的 size/SHA。
8. `BUILD_MANIFEST.json` 至少覆盖两个 EXE、两个 `Run_*.cmd`、使用说明；Manifest 自身通过 ZIP sidecar hash 或外部校验覆盖。
9. ZIP 创建后重新打开，拒绝重复、意外或缺失条目，逐项核对长度和 SHA。
10. 生成 `ExoCollectionSystem-0.1.0-windows-x64.zip.sha256` 或等价 sidecar，并验证其内容。
11. 递归删除旧 stage/ZIP/Setup 前验证绝对路径位于 ProjectRoot 且路径链无 reparse point。
12. Inno 构建前删除旧目标，构建后验证新文件时间/存在；安装包携带同一 BUILD_MANIFEST 和 README。
13. 没安装 Inno Setup 时 ZIP 仍是完整交付物，构建成功；不能把缺少可选安装器当失败。
14. ZIP 解压结构保持：顶层 bundle 文件夹内有 `Run_*.cmd`，其 `dist/*.exe` 相对路径正确。
15. Unicode 和空格项目路径可用。
16. PowerShell 5.1 兼容；写 JSON/文本时避免 BOM 引发解析差异。
17. PyInstaller spec 显式收集新异步 preflight target、Collector writers/adapters/QC/reporting、Data Studio management/external/recovery/upload/Paramiko。

---

## 8. 端到端验收矩阵

完成修复后，建立一个新的临时数据根，不使用旧 `runtime_data` 判断成功。

### 8.1 源码 Collector E2E

运行内部 smoke：

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m exo_collection.apps.collector.main --smoke-test
.\.venv\Scripts\python.exe -m exo_collection.apps.collector.main --collect-smoke-test --duration 0.4
```

内部 smoke 参数只用于开发验证；日常 `Run_ExoCollector*.cmd` 仍不传参数。

还要使用 API/测试执行一个 F/001 Trial 和一个 T/001 Trial，检查：

- 路径分别以 `F/001`、`T/001` 开头。
- Project/Subject UUID 稳定且 F/T 不同。
- 事件顺序包含 WAITING_SYNC → RECORDING。
- `timing.start_host_monotonic_ns` 等于首合格 trigger。
- pre-trigger duration 大于 0。
- 超声 4 通道、dtype/shape 正确。
- CRC、index、HDF5 clean close、checksums 全通过。
- 所有规定 QC 文件存在且进入 Manifest Artifact。
- Catalog 中层级和统计正确。
- 无 `.partial/.recording/.building` 遗留。

再运行失败 E2E：

- 禁止同步 trigger，人工停止；必须 RECOVERABLE，无 final Manifest。
- 注入原始队列饱和；必须 RECOVERABLE/显式失败。
- 注入 HDF5 OSError；必须保留恢复数据。
- 在停止边界注入 Adapter fault；不得 FINALIZED。
- 强杀 Worker；不得发布正常 Trial。

### 8.2 源码 Data Studio E2E

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m exo_collection.apps.data_studio.main --smoke-test
```

检查：

- Catalog refresh 和 management refresh 均完成。
- 数据树层级完整，Artifact 大小/SHA/模态显示。
- F/T 和受试者/工况/质量/日期/关键词筛选组合正确。
- 覆盖率和重复轮次正确。
- CSV/JSON 导出在 Trial 外，原子写入。
- 回放数据有界，4 通道切换和统一时间轴正确。
- 人工审核 sidecar 不改变 Manifest SHA。
- 附件导入不改变 Trial 任一字节。
- 活动锁出现时重任务按钮禁用，后端也拒绝。

### 8.3 测试稳定性

依次执行：

```powershell
.\.venv\Scripts\python.exe -m compileall -q src tests
git diff --check
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -vv -x
.\.venv\Scripts\python.exe -m pytest
```

验收：

- 测试数不得少于当前 331；若合理重构删除测试，必须解释并提供等价覆盖，不能为了通过而删测试。
- 四次完整运行全部在合理时间内退出。
- 失败/超时时退出码非零。
- 运行后无本项目遗留 Python/Worker。
- 不允许 `pytest.mark.skip`、`xfail` 或放宽断言来隐藏剩余问题。

---

## 9. Git 分阶段提交方案

只有所有源码测试稳定后才开始提交。提交前逐文件看 diff，确保没有生成物、凭据、临时路径和个人服务器参数。

建议提交顺序如下。实际文件依赖可小幅调整，但每个提交必须主题明确、可审查。

### 提交 1：架构契约更新

建议消息：

```text
Document synchronized F/T acquisition and immutable annex workflow
```

主要文件：

```text
ARCHITECTURE.md
schemas/README.md
```

### 提交 2：核心模型、时间、存储、质控与恢复

建议消息：

```text
Implement synchronized immutable multimodal trial pipeline
```

包含：

```text
config/devices/simulated.json
config/quality_rules/default.json
schemas/manifest-v1.1.0.json
src/exo_collection/acquisition/
src/exo_collection/adapters/sync_pulse/
src/exo_collection/domain/
src/exo_collection/orchestration/
src/exo_collection/quality/
src/exo_collection/reporting/
src/exo_collection/storage/
src/exo_collection/writers/
src/exo_collection/catalog/repositories.py
对应 core/unit/integration tests
```

注意 Manifest 1.0 Schema 不能被修改；先后计算 SHA-256，确认 diff 为空。

### 提交 3：Collector 工作流与实时预览

建议消息：

```text
Complete Collector preflight sync workflow and A-mode quality preview
```

包含：

```text
src/exo_collection/apps/collector/
tests/unit/test_collector_preflight.py
tests/unit/test_collector_ui.py
tests/integration/test_experiment_metadata_persistence.py
```

### 提交 4：Data Studio 管理、回放、审核、导入、恢复和上传

建议消息：

```text
Complete Data Studio review recovery import and offline transfer tools
```

包含：

```text
src/exo_collection/apps/data_studio/
src/exo_collection/external/
对应 Data Studio、management、local tools、external、recovery、upload tests
```

### 提交 5：Windows 发布与使用文档

建议消息：

```text
Harden reproducible Windows dual-application release workflow
```

包含：

```text
.gitignore
README.md
Build_Windows.cmd
First_Time_Setup.cmd
First_Time_Setup_And_Build.cmd
scripts/first_time_setup_and_build.ps1
packaging/
tests/unit/test_release_packaging.py
prompt.md
```

每次提交后执行：

```powershell
git status --short
git show --stat --oneline HEAD
```

最终提交后必须：

```powershell
git status --short
```

输出为空，构建目录和 release 目录应由 `.gitignore` 忽略。

---

## 10. 正式 Windows 构建与发布验证

正式构建必须在所有提交完成、工作树干净后运行：

```powershell
.\Build_Windows.cmd
```

或：

```powershell
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass `
  -File .\packaging\build_windows.ps1
```

不要在正式构建中使用：

```text
-SkipTests
-SkipSmokeChecks
-AllowDirtyWorkingTree
```

正式脚本应完成：

1. 完整 pytest。
2. 构建 `dist/ExoCollector.exe`。
3. 构建 `dist/ExoDataStudio.exe`。
4. Collector frozen UI + 异步预检 smoke。
5. Collector frozen spawned Trial smoke。
6. Data Studio frozen Catalog + management spawn smoke。
7. 创建完整 ZIP bundle。
8. 逐项重新打开 ZIP 验证大小/SHA。
9. 生成并验证 ZIP SHA sidecar。
10. 若 Inno 可用，创建并验证 Setup.exe；否则明确输出 ZIP 是完整交付物。

构建后检查：

```powershell
Get-Item .\dist\ExoCollector.exe
Get-Item .\dist\ExoDataStudio.exe
Get-Item .\release\ExoCollectionSystem-0.1.0-windows-x64.zip
Get-FileHash .\dist\ExoCollector.exe -Algorithm SHA256
Get-FileHash .\dist\ExoDataStudio.exe -Algorithm SHA256
Get-FileHash .\release\ExoCollectionSystem-0.1.0-windows-x64.zip -Algorithm SHA256
Get-Content .\release\ExoCollectionSystem-0.1.0-windows-x64\BUILD_MANIFEST.json
```

确认 `BUILD_MANIFEST.json` 中：

- git commit 等于最终 HEAD。
- dirty 为 false。
- 两个 EXE 和 bundle 文件 hashes 与实际一致。
- `hardware_support` 明确是模拟设备和可替换接口，不宣称真机。

然后从 ZIP 解压到另一个包含中文和空格的临时路径，双击/调用包内两个 `Run_*.cmd` 做一次启动验证；不要只验证项目根目录的 dist。

---

## 11. 推送远端

构建成功且工作树仍干净后：

```powershell
git fetch origin
git status --short
git log --oneline --decorate --graph --all -20
git rev-list --left-right --count origin/main...main
```

如果远端没有新的未知提交，执行：

```powershell
git push -u origin main
```

不要 force push。若远端出现本地没有的提交，停止 push，先报告并审查；不要擅自 rebase/merge 覆盖用户历史。

push 后验证：

```powershell
git status --short
git branch -vv
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

本地 HEAD 与远端 main SHA 必须一致。

---

## 12. 真实硬件和科研质量边界

下面这些不能凭模拟器“完成”：

- 超声厂商 SDK、真实帧协议、设备序列号/固件读取。
- IMU/编码器真实串口、CAN、EtherCAT 或其他协议。
- 测力台/动捕真实触发电气接口和每模态独立 trigger 回读。
- 真实服务器地址、账号策略和目录权限。
- 真实超声最大吞吐、30/60 分钟 soak、断线恢复和现场磁盘压力。
- 弱信号、边界不清、探头滑移、SNR、饱和、IMU/编码器量程/跳变的可信阈值。

当前里程碑要完成的是：

- 稳定 Adapter/Writer/Worker 接口。
- 严格 Schema、时间语义、存储格式和恢复机制。
- 模拟器和可注入故障。
- UI 和后台进程边界。
- `UNASSESSED` 机制和标定引用字段。
- 外部文件的通用、显式列映射和同步拟合。

最终报告必须明确：模拟环境通过不等于真实硬件通过。不要写“真实设备已支持”“已验证长期稳定性”“自动识别探头滑移”等没有证据的话。

---

## 13. 工程纪律和禁止事项

- 不创建微服务、Web 后端或云端自动上传。
- 不在采集时自动 SSH/SCP。
- 不把原始超声改成 CSV。
- 不在 UI 线程做设备 SDK、fsync、大文件扫描、回放、统计、校验或网络。
- 不用无限 Queue。
- 不静默丢原始数据。
- 不在 FINALIZED Trial 内写人工审核、上传状态或外部 Annex。
- 不把密码写入配置、日志、SQLite、命令行、Git 或测试快照。
- 不为真实硬件虚构采样率、阈值、协议字段或服务器默认值。
- 不用 sleep 掩盖进程/线程竞态。
- 不删除失败/边界测试来让测试变绿。
- 不修改已发布的 `manifest-v1.0.0.json`。
- 不在没有完整测试和冻结 smoke 的情况下报告完成。
- 不 push 构建产物到 Git；`dist/`、`build/`、`release/` 保持忽略。发布包留在本地交付路径。

---

## 14. 持续进度汇报格式

执行过程中每完成一个阶段就汇报，不能长时间无信息。每次只写事实：

```text
阶段：
完成：
修改文件：
测试命令：
结果：
发现的问题：
下一步：
```

若测试失败，先报告具体失败和根因，再修；不要只说“正在优化”。

---

## 15. 最终交付报告模板

最终报告必须至少包含：

1. 功能完成清单：Collector、Data Studio、存储/同步/质控、恢复/附件/上传。
2. 完整测试次数、每次测试数和耗时。
3. quiet 挂起问题的根因、修复文件和回归测试。
4. 源码 smoke 结果。
5. 冻结 EXE smoke 结果。
6. 最终 Git 提交列表。
7. 最终 HEAD 和远端 main SHA。
8. EXE 路径、ZIP 路径、Setup 路径（若生成）。
9. EXE 和 ZIP SHA-256。
10. 真实硬件尚缺的资料与未验证边界。

不要只写“全部完成”。所有结论必须能由命令输出、测试、Manifest、BUILD_MANIFEST 或 Git SHA 复核。

现在开始：先完整阅读架构和当前 diff，清理并定位 `pytest -q` 挂起，然后按本指令逐阶段完成，直到最终构建和 push 真正成功。
