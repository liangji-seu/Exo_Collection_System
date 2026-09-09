# Exo Process（批量解算端）技术文档

> 入口：`run_process.py` → `src/exo_collection/apps/process/main.py`
> 模块目录：`src/exo_collection/apps/process/`（约 1000 行，四个模块中最轻量）
> 定位：按受试者**批量**产出关节力矩训练真值，复用 Exo Calculate 的解算管线。

---

## 1. 概述

Process 是「按受试者批量解算」的一键工具。Calculate 把「自动同步 → 预处理 → OpenSim 解算 → 导出真值」四步做成需要逐 session 交互确认的流程；Process 把它自动化：用户选定**受试者 + 一个静态标定 session**，程序就自动遍历该受试者所有未解算的动态 session，逐个跺脚同步、解算，并把 `ground_truth.csv` 直接写进各 session 目录。

核心约定（[window.py:2-5](src/exo_collection/apps/process/window.py) 模块 docstring）：

- 原始 C3D / TXT / HDF5 只读，绝不改写；
- 解算中间产物写 `derived/opensim/run_*/`；
- 真值 `ground_truth.csv` 是给下游训练直接按 session 读取的最终产物。

### 与 Calculate 的关系

| 维度 | 复用（直接 import） | 覆盖/重写 |
|---|---|---|
| 说明 | session 发现与静态推荐（`discovery.py`）、`SessionRecord/SessionFiles` 数据契约（`models.py`）、OpenSim 子环境发现（`opensim_env.py`）、真值对齐与写 CSV（`ground_truth.py`）、QC 溯源 sidecar（`ground_truth_status.py`）、viewer 加载（`viewer.py`）、pipeline 路径挂载（`_pipeline.py`） | Calculate 的交互式多 Worker 编排（`SyncWorker`/`PrepWorker`/`OpenSimProcessWorker`）被 [batch.py](src/exo_collection/apps/process/batch.py) 里的单一 **`solve_one_session` 纯函数**取代 |

`solve_one_session` 不 import Qt、不 import opensim，是可在 UI 外直接单测的纯 Python 函数（[batch.py:1-7](src/exo_collection/apps/process/batch.py)）。两者的真值产物格式**完全一致**（同一个 `write_ground_truth_csv`），因此「批量 vs 单次」互认状态、互为补充。

---

## 2. 架构设计

### 2.1 进程与线程模型

```
ProcessWindow（UI 线程，所有 Qt 对象）
   └─ QThreadPool.globalInstance()
        └─ BatchWorker.run()（唯一后台线程，串行遍历 session）
             └─ solve_one_session(session, ...)   ← 纯函数，EXO 环境
                  ├─ run_auto_sync(...)           ← 自动跺脚同步（EXO 环境，纯 numpy）
                  ├─ prepare_session(...)         ← 预处理（EXO 环境）
                  ├─ _run_opensim(...)            ← subprocess.Popen 起 OpenSim 子进程
                  └─ _export_ground_truth(...)    ← 读 viewer 力矩 + IMU，写 CSV
```

- **OpenSim 子进程**（[batch.py:81](src/exo_collection/apps/process/batch.py)）：`subprocess.Popen([opensim_python, "scripts/process_session.py", "--manifest", ..., "--cancel-file", ...])`，`cwd` 切到 `process_session.py` 所在目录。子进程把 `stage / log / result / cancelled / error` 事件以 **JSON-Lines** 逐行打到 stdout，父进程逐行解析（[batch.py:128-146](src/exo_collection/apps/process/batch.py)）。OpenSim 只在子进程 import，主 EXO 进程绝不 import opensim。
- **进度回传**：子进程事件 → `_run_opensim` 的 `progress` 回调 → `BatchWorker.signals.progress.emit`（跨线程 queued）→ UI `_append_log`。
- **取消**：`BatchWorker._cancel_requested` 标志 + 每个 session 写 `run_dir/cancel.flag` 文件。`_run_opensim` 起一个 daemon 线程（[batch.py:115-126](src/exo_collection/apps/process/batch.py)）在 stdout 长时间静默时也轮询 `cancel_check` 补写取消文件，子进程在 stage 边界检查该文件（[process_session.py:169-171](opensim_joint_moment_pipeline/scripts/process_session.py)）。

> 注意：Process 版**没有** Calculate 里的 `QTimer` terminate/kill 兜底（对比 [workers.py:613-614](src/exo_collection/apps/calculate/workers.py) 的 `singleShot(5000/8000)`）。若子进程卡在单个 OpenSim tool 内部且没有 stage 边界，取消只能等它自然退出，无法强制终止。

### 2.2 模块内部结构

| 文件 | 职责 |
|---|---|
| [main.py](src/exo_collection/apps/process/main.py) | CLI 入口，构造 `QApplication` + `ProcessWindow`，支持 `--smoke-test` 离屏冒烟 |
| [window.py](src/exo_collection/apps/process/window.py) | `ProcessWindow` 主窗口：只做 UI 组合与信号接线，解算逻辑全部在 batch |
| [batch.py](src/exo_collection/apps/process/batch.py) | 核心：`solve_one_session` 纯函数 + `BatchWorker`（唯一接触 Qt 的薄壳）+ `SessionSolveState` 状态枚举 + `session_solve_status`/`erase_ground_truth` 纯函数 |
| [__init__.py](src/exo_collection/apps/process/__init__.py) | 模块 docstring |

### 2.3 关键类与协作

- **`ProcessWindow`**（[window.py:59](src/exo_collection/apps/process/window.py)）：持有 `QThreadPool.globalInstance()`、`BatchWorker` 引用、`_item_by_session_name` 映射。信号链 `BatchWorker.signals.* → _on_session_started/_finished/_failed/_all_done`，更新树节点配色、进度条与日志。
- **`BatchWorker(QRunnable)`**（[batch.py:372](src/exo_collection/apps/process/batch.py)）：串行喂 session 队列；`_patient_info()`（[batch.py:401](src/exo_collection/apps/process/batch.py)）从 Gaitway TXT 头读体重/身高，失败回退默认值。
- **`SessionSolveState`**（[batch.py:28](src/exo_collection/apps/process/batch.py)）：展示层状态机，与 Calculate 的 `ProcessingState`（[models.py:17](src/exo_collection/apps/calculate/models.py)）**独立**，按「缺文件 / 未解算 / QC 结论 / 失败 / 取消」分类。

### 2.4 依赖的共享模块

- `storage/layout.py`：递归发现 `manifest.json`（`iter_finalized_manifest_paths`）；
- `apps/data_studio/sync_data.py`：读取 XINGYING capture 名，配对 `.c3d`/`.txt`；
- `configuration/app_settings.py`：`SharedAppSettings`（`data_root`、`opensim_python_executable` 持久化）；
- `logging_setup.py`：`setup_process_logging` / `process_log_path`（`log/ExoProcess_<ts>_pid<pid>.log`）。

---

## 3. 业务需求

### 3.1 功能清单

1. **选受试者**：顶部受试者下拉（[window.py:96-99](src/exo_collection/apps/process/window.py)），来自 `discover_sessions` 扫描出的去重 `subject_code`。
2. **选静态标定**：静态标定下拉（[window.py:102-106](src/exo_collection/apps/process/window.py)），默认按 `recommend_static_for_subject` 自动推荐；「显示其他受试者静态标定」复选框可跨 subject 手选静态模型。
3. **状态树总览**：三列 `QTreeWidget`（名称 / 齐全状态 / 解算状态），按工况分组着色（[window.py:141-149](src/exo_collection/apps/process/window.py)、`_rebuild_tree:258`）。
4. **批量解算**：只解算 `UNSOLVED` 的 session；勾选「覆盖已解算」后也重算 `SOLVED/QC_PASS/QC_WARN/QC_FAIL`。
5. **进度与日志**：`QProgressBar`（按 target 数计）+ `QPlainTextEdit` 滚动日志。
6. **取消**：协作式取消，等当前 session 结束。
7. **擦除真值**：删除数据根下所有 `ground_truth.csv` 及其 `.qc.json`，使 session 回退「未解算」（二次确认，[window.py:404-427](src/exo_collection/apps/process/window.py)）。

### 3.2 操作流程

启动 → 扫描数据根 → 选受试者 → 自动推荐静态标定 → 检查树中待解算 session → 点「批量解算」→ 逐 session 解算并实时刷新状态/进度 → 完成后看各行 QC 结论；或先「擦除真值」再重算。

### 3.3 UI 结构

单窗口（无分页），垂直三段：顶部两行（受试者 + 静态标定 + OpenSim 环境标签 / 覆盖 + 批量 + 取消 + 擦除按钮）、中部状态树、底部进度条 + 日志。窗口标题「Exo Process —— 批量解算」，`showMaximized`。

---

## 4. 数据规格与格式

### 4.1 输入（每个动态 session）

| 文件 | 内容 | 来源 |
|---|---|---|
| `<capture>.c3d` | 标记点 + 力（由 XINGYING 动捕系统导出） | Collector 采集 |
| `<capture>.txt` | Gaitway 测力台 ASCII（含坡度、体重/身高头部信息） | Collector 采集 |
| `mocap.h5` | XING/Nokov 标记点，`(frame, marker, xyz)` 毫米，100 Hz | Collector 采集 |
| `imu.h5` | Xsens MTw IMU，右腿 12 通道（`slice(0,12)`），nominal 100 Hz | Collector 采集 |

输入校验（[batch.py:229-234](src/exo_collection/apps/process/batch.py)）：`has_dynamic_inputs`（c3d + txt + mocap.h5 + imu.h5 齐全）且静态 C3D 存在，否则 `ValueError`。

### 4.2 输出

| 文件 | 位置 | 内容 |
|---|---|---|
| `ground_truth.csv` | `<session_dir>/ground_truth.csv` | 表头 `time_s + imu_*12 通道 + 力矩列`；每一行 = 一个 C3D 时刻的右腿 IMU 特征（12 通道）+ 该时刻关节力矩真值，IMU 已线性插值到力矩采样点，可直接 `pandas.read_csv` 训练 |
| `ground_truth.qc.json` | `<session_dir>/ground_truth.qc.json` | QC 溯源 sidecar（`qc_status` + `run_dir` + csv size/mtime） |
| 解算中间产物 | `<session_dir>/derived/opensim/run_<ts>_<shortid>/` | `static.trc`(19 marker)、`dynamic.trc`(15 marker)、`grf.mot`、`external_loads.xml`、`support_mask.npy`、模型副本、`manifest.json`、`result.json`、`viewer/*.npy` |

真值 CSV 的 IMU 通道顺序由 [ground_truth.py:20-25](src/exo_collection/apps/calculate/ground_truth.py) 的 `IMU_CHANNEL_NAMES` 定义（12 通道）；力矩列来自 viewer 的 `moments.npy`（100 Hz，C3D 时间）。

---

## 5. 计算方法与实现细节

### 5.1 单 session 完整处理步骤（`solve_one_session`，[batch.py:208-314](src/exo_collection/apps/process/batch.py)）

1. **输入校验**：动态输入齐全 + 静态 C3D 存在。
2. **自动同步**（[batch.py:237-257](src/exo_collection/apps/process/batch.py)）：`run_auto_sync(c3d, mocap_h5, imu_h5, txt)`（[sync.py:77](opensim_joint_moment_pipeline/pipeline/synchronization/sync.py)）：
   - C3D ↔ mocap.h5 标记点精确匹配；
   - 主机单调时钟把 IMU 映射到 C3D 时间；
   - 右腿 IMU 加速度包络与 Gaitway 总垂直力包络配对跺脚峰（3 次）；
   - 输出 `gaitway_offset_s` 与 `c3d_start_in_mocap_h5_frame`。
   - `StompSyncError`（峰不足 3 对）→ `SyncFailed`；`confidence != "HIGH"` → 抛 `SyncFailed` 跳过本 session。
   - **关键点**（[sync.py:168-170](opensim_joint_moment_pipeline/pipeline/synchronization/sync.py)）：即使跺脚 MAD 极小，只要 C3D↔mocap.h5 匹配非 exact+unique，置信度也会被压到 MEDIUM，从而被 Process 拒收——这类 session 需回 Calculate 人工复核。
3. **预处理**（[batch.py:259-283](src/exo_collection/apps/process/batch.py)）：在 EXO 环境直接调 `prepare_session`（[prep_session.py:71](opensim_joint_moment_pipeline/pipeline/opensim_io/prep_session.py)）：读坡度、修正力/COP 方向与位置（后沿转轴 + 左右分解）、生成 `static.trc` / `dynamic.trc` / `grf.mot` / `external_loads.xml` / `support_mask.npy` / 模型副本 / `manifest.json`，写到 `_new_run_directory`（[batch.py:53](src/exo_collection/apps/process/batch.py)）生成的新 run 目录。
4. **OpenSim 子进程**（[batch.py:285-293](src/exo_collection/apps/process/batch.py)）：`_run_opensim` 启动 `process_session.py`，子进程执行 Scale → 两遍静态 marker refine 标定 → 动态 IK → 双侧 GRF ID → QC，返回 `result` 事件（含 `viewer_dir`、`qc`、`moments`）。
5. **导出真值**（[batch.py:295-307](src/exo_collection/apps/process/batch.py)）：`_export_ground_truth`（[batch.py:161](src/exo_collection/apps/process/batch.py)）：
   - 读 viewer 的 `moments`（100 Hz C3D 时间）；
   - `read_host_monotonic_ns(mocap_h5)[start_frame]` 得 C3D 零点主机时钟；
   - `find_imu_sensor(imu_h5, side="right")` + `imu_sensor_on_c3d_time(..., axis_slice=slice(0,12))` 取右腿 IMU 12 通道；
   - `align_ground_truth` 把 IMU 线性插值到力矩时间轴（[ground_truth.py:28](src/exo_collection/apps/calculate/ground_truth.py)）；
   - `write_ground_truth_csv` 写 CSV；`write_ground_truth_qc` 写 `.qc.json`。

### 5.2 批量调度实现

- **队列**：`_on_batch_clicked`（[window.py:347-357](src/exo_collection/apps/process/window.py)）先按 `session_solve_status` 筛选 targets（默认只 `UNSOLVED`，覆盖模式加已解算四态），构造 `BatchWorker(targets, static, opensim_python, generic_model)`。
- **逐个串行**：`BatchWorker.run()` 的 `for record in self._sessions`（[batch.py:427](src/exo_collection/apps/process/batch.py)）严格串行，无并行。
- **失败处理**：单个 session 的 `SyncFailed` 或其它异常只 `failed += 1` 并继续下一 session（[batch.py:455-467](src/exo_collection/apps/process/batch.py)）；只有 `SolveCancelled` 才 `break`。
- **断点续跑**：无显式检查点。天然断点 = `session_solve_status` 判定：已导出 `ground_truth.csv` 的 session 自动归入「已解算」，下次运行自然跳过；「覆盖已解算」可强制重算。

### 5.3 与 Calculate 单次解算的参数差异

| 参数 | Process（batch.py） | Calculate |
|---|---|---|
| 同步方式 | 仅 `AUTO_HIGH`，非 HIGH 跳过（[batch.py:252-254](src/exo_collection/apps/process/batch.py)） | 自动 / 人工配对 / 专家强制，MEDIUM/LOW 可确认 |
| `marker_cutoff_hz` | 固定 `6.0`（[batch.py:218](src/exo_collection/apps/process/batch.py)） | 可配置（默认 6.0） |
| `grf_cutoff_hz` | 固定 `20.0`（[batch.py:219](src/exo_collection/apps/process/batch.py)） | 可配置（默认 20.0） |
| `opensim_x_sign / z_sign` | 固定 `-1.0 / -1.0`（[batch.py:220-221](src/exo_collection/apps/process/batch.py)） | 可配置 |
| `analysis_time_range_s` / `static_time_range_s` | 不传（`None` → 自动稳态检测 / 自动静态窗） | 专家可手定 |
| `marker_adjustment_expert_confirmed` | 不传（默认 False） | 可配置 |
| `mass_kg / height_m` | 每 session 从 TXT 头 `read_gaitway_patient_info` 读，缺省 `75.0 / 1.75`（[batch.py:401-421](src/exo_collection/apps/process/batch.py)） | 预填 UI 可改 |
| 同步 quality 块 | `{**sync, "method":"AUTO_HIGH"}`（[batch.py:281](src/exo_collection/apps/process/batch.py)） | `_sync_quality()` 从 `_sync` 组装更全字段 |
| QC 判定 | 直接用子进程 `payload["qc"]["status"]`（[batch.py:302-306](src/exo_collection/apps/process/batch.py)） | 先取子进程 qc，缺失时本地兜底 `evaluate_qc` |

### 5.4 关键常量 / 阈值 / 默认值

| 项 | 值 | 位置 |
|---|---|---|
| marker 低通 | `6.0 Hz` | [batch.py:218](src/exo_collection/apps/process/batch.py) |
| GRF 抗混叠低通 | `20.0 Hz` | [batch.py:219](src/exo_collection/apps/process/batch.py) |
| 力符号 | `opensim_x_sign=-1.0`、`opensim_z_sign=-1.0` | [batch.py:220-221](src/exo_collection/apps/process/batch.py) |
| 人体默认 | `mass_kg=75.0`、`height_m=1.75` | [batch.py:417/420](src/exo_collection/apps/process/batch.py) |
| IMU 通道 | `axis_slice=slice(0, 12)` | [batch.py](src/exo_collection/apps/process/batch.py) |
| 跺脚峰 prominence | `0.05` | [sync.py:85](opensim_joint_moment_pipeline/pipeline/synchronization/sync.py) |
| 力阈值 | `force_threshold_N=50.0` | [prep_session.py:91](opensim_joint_moment_pipeline/pipeline/opensim_io/prep_session.py) |
| 模型 | `pipeline_root()/data/models/gait2392/gait2392_simbody.osim` | [window.py:185](src/exo_collection/apps/process/window.py) |
| 解算脚本 | `pipeline_root()/scripts/process_session.py` | [batch.py:290](src/exo_collection/apps/process/batch.py) |
| 取消退出码 | `_CANCEL_EXIT=2` | [process_session.py:67](opensim_joint_moment_pipeline/scripts/process_session.py) |

---

## 6. 输出产物

### 6.1 真值 CSV（训练数据）

`ground_truth.csv` 表头：`time_s, imu_*（12 通道）, hip_flexion_r, hip_flexion_l, knee_angle_r, knee_angle_l, ankle_angle_r, ankle_angle_l`（列名与力矩来源一致）。每一行对应一个 C3D 时刻（100 Hz），IMU（nominal 100 Hz 但非严格等间隔）已线性插值到该时刻。这是外骨骼力矩估计算法的训练输入。

### 6.2 QC 溯源 sidecar

`ground_truth.qc.json` 由 [ground_truth_status.py](src/exo_collection/apps/calculate/ground_truth_status.py) 写入，记录 `qc_status`（PASS/WARN/FAIL）、`run_dir`、CSV 大小与 mtime。Process 和 Calculate 的「已解算 / QC 结论」状态都从这个 sidecar 读取，因此两端互认。

### 6.3 QC 结果

QC 结论由 OpenSim 子进程内的 `pipeline/qc/evaluate.py`（版本化规则）综合判定，结果 `PASS / WARN / FAIL` 直接取 `payload["qc"]["status"]`。判定规则与 Calculate 完全一致（详见 [calculate.md](calculate.md) 的 QC 章节），包括同步可信度、静态 marker 调整分级、marker 重投影 RMS、ID 残差力、左右力覆盖等。

---

## 7. 与其他模块的接口与数据流

```
Collector 采集（C3D/TXT/mocap.h5/imu.h5）
        │
        ▼
Process.solve_one_session ──复用──► opensim_joint_moment_pipeline（同步/预处理/OpenSim/QC）
        │                                    │
        │ 写 ground_truth.csv + .qc.json      │ 写 derived/opensim/run_*/
        ▼                                    ▼
   <session_dir>/ground_truth.csv     <session_dir>/derived/opensim/run_*/
        │                                    │
        ▼                                    ▼
   下游训练直接读取            data_studio 全局预览 / QC 报告（读 viewer/*.npy + result.json）
```

- **与 data_studio**：只复用其存储布局发现（`storage.layout`、`sync_data.load_cap_names`），不写回 Data Studio 的展示产物；二者通过共享 `data_root` 与 `SharedAppSettings` 互通。Data Studio 的「全局预览 / QC 报告」直接读 Process 解算出的 `viewer/*.npy` + `result.json`。
- **与 calculate**：平行入口，共享同一套 `opensim_joint_moment_pipeline` 与真值导出逻辑。Process 跳过的「非 HIGH 同步」session，回 Calculate 人工复核/专家强制后单独解算并导出真值，产物格式一致。

---

## 8. 关键文件清单

**Process 模块自身**

- [batch.py](src/exo_collection/apps/process/batch.py) — 核心：`solve_one_session` 四步流水线、`_run_opensim` 子进程 + 取消 watch 线程、`_export_ground_truth`、`BatchWorker` 串行调度、`SessionSolveState`、`session_solve_status` / `erase_ground_truth`。
- [window.py](src/exo_collection/apps/process/window.py) — `ProcessWindow` 主窗口：受试者/静态/覆盖选择、状态树着色、进度/日志、批量按钮接线。
- [main.py](src/exo_collection/apps/process/main.py) — CLI 入口，`QApplication` + 窗口 + smoke-test。

**复用/依赖的 Calculate 关键文件**

- [_pipeline.py](src/exo_collection/apps/calculate/_pipeline.py) — `pipeline_root()` / `ensure_pipeline_on_path()`，把 `opensim_joint_moment_pipeline` 挂到 `sys.path`。
- [discovery.py](src/exo_collection/apps/calculate/discovery.py) — `discover_sessions` / `recommend_static_for_subject` / `check_inputs`。
- [models.py](src/exo_collection/apps/calculate/models.py) — `SessionRecord` / `SessionFiles` / `is_stand` / `ProcessingConfig` 契约。
- [opensim_env.py](src/exo_collection/apps/calculate/opensim_env.py) — OpenSim 子环境发现与默认选择。
- [ground_truth.py](src/exo_collection/apps/calculate/ground_truth.py) — `align_ground_truth` / `write_ground_truth_csv` / `IMU_CHANNEL_NAMES`。
- [ground_truth_status.py](src/exo_collection/apps/calculate/ground_truth_status.py) — `write_ground_truth_qc` / `read_ground_truth_qc` 溯源 sidecar。
- [viewer.py](src/exo_collection/apps/calculate/viewer.py) — `load_viewer_data`（真值力矩来源）。

**复用的 opensim_joint_moment_pipeline 关键文件**

- [sync.py](opensim_joint_moment_pipeline/pipeline/synchronization/sync.py) — `run_auto_sync`（跺脚同步）、`StompSyncError`、`save_sync_calibration`。
- [prep_session.py](opensim_joint_moment_pipeline/pipeline/opensim_io/prep_session.py) — `prepare_session`（预处理 + manifest）。
- [process_session.py](opensim_joint_moment_pipeline/scripts/process_session.py) — OpenSim 子进程入口（Scale/标定/IK/ID/QC + JSON-Lines 事件协议）。
- [ascii_export.py](opensim_joint_moment_pipeline/pipeline/gaitway/ascii_export.py) — `read_gaitway_patient_info`（体重/身高读取）。
- [clock.py](opensim_joint_moment_pipeline/pipeline/synchronization/clock.py) — `find_imu_sensor` / `imu_sensor_on_c3d_time` / `read_host_monotonic_ns`（IMU→C3D 时间映射）。
