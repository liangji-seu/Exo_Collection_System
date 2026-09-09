# Exo Data Studio（数据管理端）技术文档

> 入口：`run_data_studio.py` → `src/exo_collection/apps/data_studio/main.py` → `DataStudioWindow`
> 模块目录：`src/exo_collection/apps/data_studio/`（约 16000 行，全项目最大模块）
> 定位：采集完成之后的**全生命周期数据管理**：浏览/检索、离线回放、质量审核、校验、归档、上传远端、恢复、外部导入与数据同步。

---

## 1. 概述

Data Studio 与采集端 Exo Collector 构成双桌面程序架构，二者共享 `exo_collection` 核心包。Collector 只负责采集落盘；Data Studio 负责之后的一切管理动作。其核心设计原则是：**管理端绝不改写已 FINALIZED 的原始采集数据**——所有写操作（人工审核、上传索引、外部 annex、归档）都通过追加式文件或独立目录完成，原 trial 目录只读。

入口链：`run_data_studio.py`（含 SSH 依赖探测，缺 Paramiko/SCP 时自动切换到 `py -3.11`）→ `main.py` → `window.py`（主窗口 3078 行，全部菜单/面板/对话框编排）。

与其他三模块的关系：

- **collector**：只消费其产物（trial 目录 + `manifest.json` + `catalog.sqlite3`），通过共享 QSettings 命名空间读写同一 `data_root`，通过 `.exo/.collector-active.json` 活动锁做轻量互斥。
- **calculate**：**不 import opensim**，只通过 `SharedAppSettings.opensim_python_executable` 定位子环境，或直接读 calculate 已落盘的 `derived/opensim/run_*/result.json` + `viewer/*.npy` 做展示/QC。
- **process**：读取 process 生成的 `ground_truth.csv` 与 XINGYING 同步侧车文件。

---

## 2. 架构设计

### 2.1 进程与线程模型（重工具隔离出 GUI 进程）

- **GUI 线程 + `QThreadPool(max 1)`**：只读、本地、纯 Python/NumPy 的工具（`FullStatistics`、`verify_trial_checksums` 等）作为 `QRunnable` 串行执行。
- **`DataStudioProcessWorker`**（[process_workers.py:102-211](src/exo_collection/apps/data_studio/process_workers.py)）：`multiprocessing` **spawn** 子进程，`ProcessOperation ∈ {catalog_refresh, playback, checksum, management_refresh, management_summary, management_export}`；子进程在干净解释器内 import 工具实现，单结果经 `Queue(maxsize=1)` 返回。
- **`UploadWorkerHandle`**（[upload.py:2378-2522](src/exo_collection/apps/data_studio/upload.py)）：spawn + **两条管道**（双向 command 管道 + 单工 event 管道）；**密钥经管道在进程启动后发送，绝不放进 argv**（防进程列表泄露）。
- **`ExternalImportWorker`** 与 **`RecoveryBackgroundService`**：均为 spawn 子进程 + 队列。

### 2.2 模块内部结构

| 文件 | 职责 |
|---|---|
| [window.py](src/exo_collection/apps/data_studio/window.py)（3078 行） | `DataStudioWindow` 主窗口，菜单、面板、对话框与工具调度中枢 |
| [service.py](src/exo_collection/apps/data_studio/service.py) | 服务层（绑定后台服务与 UI） |
| [management.py](src/exo_collection/apps/data_studio/management.py)（1648 行） | 纯后端（不依赖 Qt）：trial 管理记录、过滤、覆盖度/状态汇总、清单导出、上传审计校验 |
| [upload.py](src/exo_collection/apps/data_studio/upload.py)（2550 行） | SSH/SCP 上传引擎、远端扫描、远端下载、上传 worker handle |
| [local_tools.py](src/exo_collection/apps/data_studio/local_tools.py)（2153 行） | 只读本地分析工具集（回放加载、统计、校验、质控审计、artifact 检查） |
| [qc_report.py](src/exo_collection/apps/data_studio/qc_report.py)（603 行） | OpenSim 结果的步态 QC（足跟触地、归一化步态周期、GRF 同步） |
| [global_preview.py](src/exo_collection/apps/data_studio/global_preview.py)（196 行） | ground_truth.csv 全时间轴静态预览 |
| [opensim_overlay.py](src/exo_collection/apps/data_studio/opensim_overlay.py)（145 行） | OpenSim 验证三联图数据加载（Fz/髋角/髋力矩） |
| [gait_baseline.py](src/exo_collection/apps/data_studio/gait_baseline.py)（318 行） | Winter 髋力矩样条 LUT + IMU pitch 步态相位估计 |
| [fullscreen_viewer.py](src/exo_collection/apps/data_studio/fullscreen_viewer.py)（592 行） | 全屏回放工作区（多 dock 面板 + mocap 3D 骨骼） |
| [plots.py](src/exo_collection/apps/data_studio/plots.py) | `TimeSeriesPlot` 示波器式扫描图 |
| [local_dialogs.py](src/exo_collection/apps/data_studio/local_dialogs.py)（1390 行） | 只读工具结果对话框（统计/校验/质控/回放） |
| [data_view.py](src/exo_collection/apps/data_studio/data_view.py) | `DataViewWidget` 渲染 artifact 检查结果 |
| [sync_data.py](src/exo_collection/apps/data_studio/sync_data.py) | XINGYING `.cap` → `.c3d/.txt` 同步侧车文件 |
| [quality_reviews.py](src/exo_collection/apps/data_studio/quality_reviews.py) | 追加式、哈希链的人工审核记录 |
| [credential_store.py](src/exo_collection/apps/data_studio/credential_store.py) | Windows 凭据管理器封装 |
| [recovery_service.py](src/exo_collection/apps/data_studio/recovery_service.py) | 恢复后台服务 |
| upload_dialog.py / external_import_dialog.py / recovery_dialog.py / management_dialog.py | 各配置/进度对话框 |

### 2.3 关键类与协作

- **`DataStudioWindow`**：编排中枢，持有各 worker/service，把菜单动作分发到 `LocalToolTask`（QRunnable）、`DataStudioProcessWorker`、`UploadWorkerHandle`、`ExternalImportWorker`、`RecoveryBackgroundService`。
- **`TrialManagementRecord` / `TrialFilter` / `ManagementIndex`**（[management.py:93-146](src/exo_collection/apps/data_studio/management.py)）：管理视图数据模型与过滤（`TrialFilter` 为 Pydantic frozen、`extra=forbid`）。
- **`OfflineUploadRequest`**：上传请求 frozen dataclass，`password/private_key/passphrase` 字段 `repr=False, compare=False`，避免泄露到日志与相等比较。
- **`ParamikoScpSession`**（[upload.py:930-1204](src/exo_collection/apps/data_studio/upload.py)）：封装 paramiko SSH + scp；**只做 SCP 传输 + SFTP 的 mkdir/read/verify，从不执行远端命令**。
- **`SshScpTrialUploader`**：上传状态机主体。

### 2.4 共享模块角色

- `storage/`：layout（目录）、manifest（契约）、checksum（SHA-256）、package（归档）、subject_lock（受试者锁）、activity（collector 活动锁）、recovery（恢复）。
- `catalog/`：db（SQLite WAL + 迁移）、models（ORM）、repositories（仓储）。
- `external/importer.py`：外部数据导入（append-only annex + 时钟对齐）。
- `readers/binary_block.py`：超声/EMG 二进制块格式读取。
- `quality/`：config 加载规则、engine 执行评级。
- `timing/clock_model.py`：设备时钟仿射重建。

---

## 3. 业务需求

### 3.1 功能清单

1. **数据管理**：trial 浏览/检索/过滤（受试者、天、项目、工况、session、等级）、覆盖度统计（`compute_subject_coverage`）、数据集状态汇总（`summarize_dataset_states`）、清单导出（CSV+JSON）。
2. **归档**：trial 打包成归档单元，生成 `checksums.sha256`。
3. **回放**：全模态离线回放（超声瀑布/A-scan、IMU、EMG、编码器、同步脉冲、动捕、力矩、人工标签），固定循环窗口。
4. **QC 报告**：自动质控审计 + OpenSim 步态 QC + 人工审核链。
5. **上传**：SSH/SCP 上传到远端、选择性上传、进度展示、远端状态扫描、远端下载。
6. **恢复**：扫描/修复/最终化/丢弃 4 类恢复操作。
7. **外部导入**：外部测力台/动捕数据附到 trial 作为 annex，做时钟对齐。
8. **数据同步**：XINGYING `.cap` 对应的 `.c3d/.txt` 侧车文件同步。

### 3.2 完整工作流

- **浏览 → 回放 → 审核 → 上传**：刷新 catalog → 选中 trial → 打开回放/统计/校验/质控 → 人工追加审核 → 上传远端。
- **上传**：收集端点+凭据 → `build_upload_plan`（本地哈希 + manifest 校验）→ `SshScpTrialUploader.upload`（VALIDATING→CONNECTING→staging→UPLOADING→VERIFYING→PUBLISHING）→ 更新远端同步索引 → 写审计。
- **恢复**：`RecoveryDialog` 100ms 轮询 + 证据门控（有证据才点亮「安全修复尾块 / 确认 FINALIZED / 保留数据并 ABORTED」）→ 提交 `RecoveryBackgroundService`。
- **外部导入**：`ExternalImportDialog` 选来源/模态/脉冲映射 → `import_external_artifact` → 写入 `external_annexes/` annex + 对齐映射。

### 3.3 UI 结构

- **主窗口**：菜单驱动的单窗口多面板，带 catalog 树/列表 + 详情面板。
- **对话框**：`OfflineUploadDialog`（端点+凭据）、`SelectiveUploadDialog`、`UploadProgressDialog`、`ExternalImportDialog`、`RecoveryDialog`、`ManagementSummaryDialog`（覆盖度 + 状态两 tab）、`FullStatisticsDialog` / `ChecksumDialog` / `QualityAuditDialog` / `PlaybackDialog`。
- **独立窗口**：`FullscreenViewer`（多 dock：mocap 3D / 超声瀑布 / IMU / EMG / 编码器 / 力矩 / 验证）、`GlobalPreviewWindow`、`GaitQCReportWindow`。

---

## 4. 数据规格与格式

### 4.1 目录布局与 Catalog

**数据文件树**（五级：受试者 / 第几天 / 主工况 / 详细工况 / session）：

```text
{data_root}/
└── {subject}/                          # 受试者，如 103
    └── d{day}/                         # 第几天（采集天次），如 d1
        └── {project}/                  # 主工况，如 F_STEADY
            └── {condition}/            # 详细工况，如 WALK_0P6_EXO
                └── session{repeat}_{YYYYmmdd_HHMMSS}/   # session，如 session1_20260908_055822
                    ├── .exo/
                    │   ├── manifest.json                # 采集契约（schema 1.2.0，含四级 UUID）
                    │   └── checksums.sha256             # 每文件 SHA-256
                    ├── (设备原始数据：超声 / IMU / EMG / mocap / 测力台…)
                    └── derived/opensim/run_*/           # 解算产物（viewer/*.npy、result.json）
```

- 未发布 session 目录带后缀 `.recording/.partial/.aborted/.building`，FINALIZED 后去掉。
- **Catalog**：SQLite `.exo/catalog.sqlite3`，WAL、`busy_timeout 5000`、`foreign_keys ON`、Alembic 迁移 + `.migrate.lock` 文件锁（Windows `msvcrt.locking`）。trial 真实主键是 UUID，人类可读名仅展示。
- 外部 annex：`external_annexes/<trial_uuid>/<annex_uuid>/`。
- 审核链：`.studio-records/quality-reviews/{trial_uuid}/`，文件名 `{timestamp}-{review_uuid}-{digest}.json`。
- 上传审计：`.upload-audit/`。

### 4.2 远端目录与同步索引

- 远端目录：`remote_workdir/{subject}/d{day}/{project}/{condition}/session{repeat}_{ts}/`（[upload.py:578-603](src/exo_collection/apps/data_studio/upload.py) `build_remote_trial_directory` 精确镜像本地相对路径，含「第几天」层级）。`validate_remote_directory()` 要求绝对 POSIX 路径，每段只允许 `[A-Za-z0-9._-]`。
- 同步索引：远端 `data/.exo/exo_sync_index.json`（schema `exo.remote-sync-index/v1`）+ 本地 `data/.exo/exo_sync_cache.json`（`exo.local-sync-cache/v1`）。状态同步只读并对比这两个小索引，不通过网络重算大文件 SHA-256。
- 包指纹：`_package_fingerprint()` = 对 `relpath\0size\0sha256\n` 按 relpath 排序后整体 SHA-256，作为上传去重/合并单元指纹。

---

## 5. 计算方法与实现细节

### 5.1 归档与校验

- **打包**：把 trial 所有 artifact 打包为归档单元，生成每文件 SHA-256 的 `checksums.sha256`。
- **上传前校验**：`build_upload_plan()` 对每个打包文件做哈希，并验证 manifest 声明的 artifact 的 size + sha256。
- **本地校验**：`verify_trial_checksums()`；`_sha256_file_idle()` 每 4MB 块重新检查 `_require_idle`（轻量模式下让位 collector）。

### 5.2 上传（SSH/SCP）协议与状态机

**凭据与信任**：

- 密码存 Windows 凭据管理器（`SERVICE_NAME="ExoCollectionSystem.DataStudio.SSH"`，account = `sha256(host:port:username)` 十六进制，blob UTF-16-LE，`CRED_PERSIST_LOCAL_MACHINE`）。
- 私钥路径与 `remember_password` 存 QSettings，**明文不含任何密钥**。
- 主机指纹用 `ConfirmFirstUsePolicy` + `~/.ssh/known_hosts` 与 `~/.exo_collection_system/known_hosts` 双重信任来源。

**状态机**（[upload.py:1246-1564](src/exo_collection/apps/data_studio/upload.py)）：

```
VALIDATING → CONNECTING → 远端 staging（.{uuid}.partial-{uuid4}）
    → UPLOADING（20Hz 心跳 guard + 4Hz 进度报告）
    → VERIFYING（远端逐文件 SHA-256）
    → PUBLISHING（staging rename 成最终目录名）
    → 更新同步索引 → 写审计
```

逐文件状态：`NOT_SELECTED / QUEUED / TRANSFERRING / TRANSFERRED / VERIFYING / VERIFIED / FAILED`。

**断点 / 合并 / 恢复**：`_merge_existing_trial()` 对远端已存在 trial 做**追加式合并，绝不覆盖冲突**；`RemoteDatasetStatusScanner` 把远端 trial 分类为 `UPLOADED/NOT_UPLOADED/PARTIAL/CONFLICT`；`RemoteDatasetDownloader` 做 git-pull 式镜像，用指纹跳过未变更文件。**核心安全承诺：所有远端文件在 SHA-256 校验通过前都视为不完整；云端校验通过后绝不自动删除本地数据。**

**守卫**：`_guard()` 在 collector 活动锁存在时抛 `COLLECTOR_ACTIVE`，避免上传与采集抢 IO。

### 5.3 回放与可视化

- **`load_trial_playback()`**（[local_tools.py:1211-1427](src/exo_collection/apps/data_studio/local_tools.py)）：有界/降采样加载（`max_signal_points=4000`、`max_ultrasound_frames=4000`、`max_ultrasound_depth_points=1000`）。加载超声（`.bin` + `.meta.json` + `.idx`）、IMU/编码器/同步脉冲（`.h5`）、EMG（块二进制）、mocap（`.h5`）、力矩（`ground_truth.csv`）、人工标签（`.jsonl`）。
- **`_read_hdf5_signal()`**：检查 `closed_cleanly`、`_unwrap_device_clock` 解卷绕设备钟、`fit_affine_clock` 重建均匀时间轴、计算 `break_before` 断点掩码（画图断开毛刺）。
- **`_read_hdf5_imu_sensors()`**：解耦 IMU 独立流（`alignment_mode=="none_independent_streams"` 或 >2 个负时间差）→ 每传感器独立序列。
- **回放 UI**：固定 10s 循环窗口、50ms QTimer（20FPS）、速度 0.25–4.0×；`_SweepWaterfallPlot` row-major 图像（列=时间左→右、行=深度 0..999）；`_SweepSignalPlot` 用 `break_before` + `dt` 判定「正常前向步进 ≤3×nominal_step」来决定是否连线，避免换行/跨周期毛刺。
- **`FullscreenViewer`**：多 dock 面板 + `Mocap3DCanvas`（含 `_BONE_SEGMENTS` 骨骼、正交投影、反转 Nokov x 轴）。

### 5.4 QC 报告（足跟触地、归一化步态周期、GRF 同步）

**计算函数**（[qc_report.py](src/exo_collection/apps/data_studio/qc_report.py)）：

- **`detect_heel_strikes()`**（[qc_report.py:240-266](src/exo_collection/apps/data_studio/qc_report.py)）：阈值默认 `max(0.03*peak, 20.0)` N、`min_gap_s=0.5`、`frame_rate=100`；检测 **Fz 上升沿（初始着地）而非峰值**。
- **`normalize_gait_cycles()`**：把每个 `[HS_i, HS_{i+1}]` 重采样到 0–100%（101 点），`np.nanmean`/`np.nanstd` 得 `(x, mean, std)`。
- **`representative_cycle()`**：取时长中位数的步态周期作为代表周期。
- **`load_gait_qc()`**：读 `result.json` + `viewer/time_s.npy` + `viewer/grf.npy`（形状 `(n,2,3)`、`cop_order=[right,left]`、**y 分量为竖直**，故右脚 Fz=`grf[:,0,1]`）+ `viewer/moments.npy`（`(n,6)`，列 0–5 = 左右髋屈/膝角/踝角）。IMU pitch 从 `ground_truth.csv` 读并插值对齐到 viewer 时间轴。

**报告窗口 `GaitQCReportWindow`**（[qc_report.py:328-602](src/exo_collection/apps/data_studio/qc_report.py)）包含：

- **头部文本**：QC 状态徽标（绿 PASS / 红 FAIL / 琥珀 WARN）、滤波信息（marker/GRF cutoff Hz）、帧数/帧率、周期数、run 目录路径。
- **7 个图**（每个图标题标注其内容，导出 PNG 时逐图可辨识）：

| # | 图标题 | 内容 |
|---|---|---|
| 1 | GRF 同步 · 右 Fz | 右 Fz 与右髋力矩上下两个 x 联动子图，heel strike 竖虚线标注 |
| 2 | 右髋力矩 | 右髋力矩曲线 |
| 3 | 髋关节力矩真值 · 右髋(橙) 左髋(绿) | ground_truth.csv 左右髋力矩 |
| 4 | IMU 姿态角（右腿）· pitch(蓝) roll(绿) yaw(红) | 右腿 IMU 三姿态角 |
| 5 | 归一化步态周期 · 力矩 mean±std · 右髋(橙) 左髋(绿) | 0–100% 步态周期力矩均值±标准差 |
| 6 | 归一化步态周期 · 角度 mean±std · 右髋(橙) 右膝(蓝) 右踝(绿) | 0–100% 步态周期角度均值±标准差 |
| 7 | 左右髋对比 · 右(橙) 左(绿) | 同一归一化轴上右/左髋力矩 mean±std |

颜色约定：右橙 `#F28E2B`、左绿 `#59A14F`、pitch 蓝 `#1F77B4`、roll 绿 `#2CA02C`、yaw 红 `#D62728`。

- **IK QC（文本）**：静态 RMS/max 标「未采集」；动态 marker RMS/max + 最差标记列表（按 max_cm 降序前 8）。
- **ID QC（文本）**：pelvis 残差力 RMS/p95、残差力矩 RMS/p95。
- **PNG 导出**：`graphics.grab().save(path, "PNG")`（去掉 ImageExporter，规避 PySide6 6.11 + pyqtgraph 0.14 的离屏抓取偶发 segfault）。

### 5.5 步态相位估计与基线（`gait_baseline.py`）

- **`estimate_gait_phase()`**：Hann 平滑 + pitch 峰值（=足跟触地，取极大）+ 非极大抑制（`min_sep_s=0.6`）+ 线性相位插值 + `max_stride_factor=1.8`（停顿→NaN）。
- **`build_hip_baseline()`**：Winter 髋力矩样条 LUT（`hip_torque_lut.csv`）。
- **`moment_imu_offset_s()`**：从 `manifest.json` 的 `c3d_t0_host_monotonic_ns` 算力矩与 IMU 的时偏。

### 5.6 外部数据导入（`external/importer.py`）

外部数据 append-only 落在 `external_annexes/<trial_uuid>/<annex_uuid>/`，含 `annex_manifest.json`、`alignment/mapping.json`、`checksums.sha256`；时钟对齐用 `align_shared_pulses` 仿射拟合 `t_global_ns = a×t_external + b`。已 FINALIZED 的 Trial Manifest 和原始 Artifact 不因事后导入而改写。

### 5.7 恢复机制与 subject lock

- **恢复**：`RecoveryBackgroundService` 调 `discover_recoverable_trials` / `repair_recording_directory` / `finalize_prepared_recording` / `abort_recording_preserving_data`。
- **subject lock**：`.exo/subject-locks/{code}.json` —— **手动操作锁（存在即锁，无租约）**，与自动的 collector activity 锁（心跳租约，stale 5s）不同。
- **轻量模式**：读 `.exo/.collector-active.json`（心跳租约），collector 活跃时暂停重工具。

---

## 6. 输出产物

- **QC 报告 PNG**：导出 `GaitQCReportWindow` 整图（含 7 个标注图 + QC 头部 + IK/ID 文本），逐图标题可辨识。
- **清单导出**：CSV + JSON（管理汇总）。
- **校验报告**：逐文件 SHA-256 结果。
- **人工审核链**：追加式哈希链 JSON（`QualityReviewRecord.previous_record_sha256` 锚定 manifest sha256）。
- **上传审计**：`.upload-audit/`（带 `AUDIT_SECRET_GUARD` 校验，密钥字段不落审计）。
- **远端同步索引**：`exo_sync_index.json` + 本地 `exo_sync_cache.json`。
- **外部 annex**：`annex_manifest.json` + `alignment/mapping.json` + `checksums.sha256`。

---

## 7. 与其他模块的接口与数据流

```
Collector 落盘 → Data Studio（管理/回放/QC/上传/恢复/导入/同步）
                       │
                       ├─► 上传 → 远端服务器（SSH/SCP + SHA-256 校验 + 同步索引）
                       ├─► 读 calculate 结果（result.json + viewer/*.npy）→ QC 报告
                       └─► 读 process 结果（ground_truth.csv）→ 全局预览/力矩展示
```

- **消费 collector**：只读 trial 目录 + `catalog.sqlite3`；`.exo/.collector-active.json` 活动锁实现轻量互斥；`_guard()` 在 collector 活跃时拒绝上传。
- **触发/展示 calculate**：不 import opensim；读 `derived/opensim/run_*/result.json` + `viewer/grf.npy`、`viewer/moments.npy`、`viewer/time_s.npy`。
- **展示 process**：读 `ground_truth.csv`（髋力矩 + IMU 姿态）、XINGYING 同步侧车。

---

## 8. 关键文件清单

| 文件 | 一句话职责 |
|---|---|
| [window.py](src/exo_collection/apps/data_studio/window.py) | 主窗口，全模块菜单/面板/对话框与工具调度中枢 |
| [upload.py](src/exo_collection/apps/data_studio/upload.py) | SSH/SCP 上传引擎（状态机、同步索引、远端扫描/下载、审计） |
| [management.py](src/exo_collection/apps/data_studio/management.py) | 非 Qt 后端：trial 管理、覆盖度/状态汇总、清单导出 |
| [local_tools.py](src/exo_collection/apps/data_studio/local_tools.py) | 只读本地分析：回放加载、统计、SHA-256 校验、质控审计 |
| [qc_report.py](src/exo_collection/apps/data_studio/qc_report.py) | OpenSim 步态 QC：足跟触地、归一化步态周期 mean±std、GRF 同步、报告窗口 |
| [fullscreen_viewer.py](src/exo_collection/apps/data_studio/fullscreen_viewer.py) | 全屏多 dock 回放工作区 + mocap 3D 骨骼画布 |
| [local_dialogs.py](src/exo_collection/apps/data_studio/local_dialogs.py) | 回放/统计/校验/质控四类结果对话框及扫描绘图控件 |
| [recovery_dialog.py](src/exo_collection/apps/data_studio/recovery_dialog.py) + [recovery_service.py](src/exo_collection/apps/data_studio/recovery_service.py) | 证据门控恢复对话框 + 恢复后台服务 |
| [process_workers.py](src/exo_collection/apps/data_studio/process_workers.py) | spawn 子进程 worker 框架与操作枚举 |
| [quality_reviews.py](src/exo_collection/apps/data_studio/quality_reviews.py) | 追加式哈希链人工审核记录 |
| [credential_store.py](src/exo_collection/apps/data_studio/credential_store.py) | Windows 凭据管理器封装（SSH 密码） |
| [gait_baseline.py](src/exo_collection/apps/data_studio/gait_baseline.py) | Winter 髋力矩 LUT + IMU pitch 步态相位估计 |
| [opensim_overlay.py](src/exo_collection/apps/data_studio/opensim_overlay.py) | OpenSim 验证三联图数据加载 |
| [global_preview.py](src/exo_collection/apps/data_studio/global_preview.py) | ground_truth.csv 全时间轴真值预览 |
| [sync_data.py](src/exo_collection/apps/data_studio/sync_data.py) | XINGYING `.cap` → `.c3d/.txt` 侧车同步 |
| [external_import_worker.py](src/exo_collection/apps/data_studio/external_import_worker.py) + [external_import_dialog.py](src/exo_collection/apps/data_studio/external_import_dialog.py) | 外部数据导入子进程与配置 UI |
| [upload_dialog.py](src/exo_collection/apps/data_studio/upload_dialog.py) | 上传端点/凭据/进度/选择性上传对话框 |
| [app_settings.py](src/exo_collection/configuration/app_settings.py) | 双应用共享 QSettings（data_root、上传端点、OpenSim 子环境） |
| [clock_model.py](src/exo_collection/timing/clock_model.py) | 设备时钟仿射重建 |
