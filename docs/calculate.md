# Exo Calculate（标定 / 同步 / 解算 / 回放端）技术文档

> 入口：`run_calculate.py` → `src/exo_collection/apps/calculate/main.py` → `CalculateWindow`
> 模块目录：`src/exo_collection/apps/calculate/`（约 4700 行）+ 子进程解算包 `opensim_joint_moment_pipeline/`
> 定位：把 Collector 落盘的四路原始数据（C3D 动捕、Gaitway 测力台 TXT、mocap.h5、imu.h5）加工成**生物力学关节力矩真值**（`ground_truth.csv`），是采集到模型训练的关键中间环节。

---

## 1. 概述

Calculate 是「标定 → 同步 → OpenSim 解算 → 关节力矩回放 → QC」四合一桌面端。核心价值在于把**异源、异时钟、异坐标系**的原始数据，通过时间同步与 OpenSim 骨骼动力学，统一到关节力矩真值这一最终产物上。

与其他三模块的关系：

- **collector**：产出输入。Calculate 只读 Collector 落盘的 `mocap.h5` / `imu.h5`（[discovery.py:104-105](src/exo_collection/apps/calculate/discovery.py)）以及 XINGYING 外部导出的 `.c3d` / `.txt`，不写回采集数据。
- **process**：批量化复用同一 pipeline（见 [process.md](process.md)）。
- **data_studio**：展示结果。`qc_report.py` 读 Calculate 产出的 `result.json` + `viewer/*.npy` 做步态周期级质控。

---

## 2. 架构设计

### 2.1 双 Python 环境边界（核心架构约束）

主进程（EXO 环境：numpy 1.x + ezc3d + PySide6）**永不 `import opensim`**；opensim 只在受控子进程（numpy 2.x + opensim 4.6）中 import。三处硬边界：

1. 路径发现与校验在子进程里 `import opensim`（[opensim_env.py:62-80](src/exo_collection/apps/calculate/opensim_env.py)，执行 `"import opensim; print(opensim.GetVersionAndDate())"`），结果只返回版本号字符串，不污染主进程。
2. 子进程编排 [process_session.py:42](opensim_joint_moment_pipeline/scripts/process_session.py) 顶部 `import opensim as osim`，由父进程 `subprocess.Popen` 启动。
3. 派生数据用纯 NumPy 导出（`export_viewer.py`），回放页只读 `.npy`，不接触 opensim 模型。

| 环境 | 用途 | 关键依赖 |
|---|---|---|
| `Exo`（主进程） | 跑 `run_calculate.py` | numpy 1.x、ezc3d、h5py、scipy、pandas、PySide6、pyqtgraph |
| `opensim`（子进程） | Scale / IK / ID 解算 | numpy 2.x、opensim 4.6 |

### 2.2 进程与线程模型

- **主进程**：`QThreadPool.globalInstance()` 跑 `QRunnable` Worker，所有大文件读取/子进程调用都在后台线程，不阻塞 Qt 主线程。
- **OpenSim 子进程**：`subprocess.Popen`，`stdout=PIPE`、`stderr=STDOUT`、`bufsize=1`、`text=True`、`errors="replace"`（[workers.py:379-388](src/exo_collection/apps/calculate/workers.py)）。子进程以 **JSON-Lines** 协议输出事件 `start/stage/log/result/cancelled/error`（[process_session.py:12-18](opensim_joint_moment_pipeline/scripts/process_session.py)），父进程逐行解析。
- **协作式取消**：父进程写 `run_dir/cancel.flag`，子进程在 Scale/IK/ID 之间 `check_cancel()` 检查并抛 `_Cancelled` 输出 `cancelled` 事件；父进程 `cancel()` 置位后 `QTimer.singleShot(5000, terminate)`、`(8000, kill)` 兜底（[window.py:612-614](src/exo_collection/apps/calculate/window.py)）。**取消意图优先于退出码**：`_finalize_outcome` 在 `cancel_requested or cancelled` 时一律判 `cancelled` 而非 `failed`。
- **串 Session 防护**：`OperationContext` 携带自增 `operation_id`，过期回调被 `_is_current` 丢弃；运行期间锁定数据根/Session 选择；`closeEvent` 关窗前强制终止子进程防悬空回调。

### 2.3 模块内部结构

**UI 层（`apps/calculate/`）**

| 文件 | 职责 |
|---|---|
| [main.py](src/exo_collection/apps/calculate/main.py) | CLI 入口，`--smoke-test` 离屏冒烟，`freeze_support()` |
| [_pipeline.py](src/exo_collection/apps/calculate/_pipeline.py) | `pipeline_root()` 定位仓库根（冻结时走 `sys._MEIPASS`），`ensure_pipeline_on_path()` 把 `opensim_joint_moment_pipeline` 挂上 `sys.path` |
| [operation.py](src/exo_collection/apps/calculate/operation.py) | `OperationContext` 不可变操作令牌 |
| [models.py](src/exo_collection/apps/calculate/models.py) | 不可变 DTO 与状态机枚举（`ProcessingState` / `SessionRecord` / `SyncResult` / `ProcessingConfig`） |
| [controller.py](src/exo_collection/apps/calculate/controller.py) | 核心状态机 `CalculateController(QObject)` |
| [discovery.py](src/exo_collection/apps/calculate/discovery.py) | Session 发现、静态推荐、只读输入检查 |
| [history.py](src/exo_collection/apps/calculate/history.py) | 历史 run 管理与 STALE 判定 |
| [opensim_env.py](src/exo_collection/apps/calculate/opensim_env.py) | opensim 子环境自动发现/校验 |
| [workers.py](src/exo_collection/apps/calculate/workers.py) | 4 个后台 Worker + 子进程 JSON-Lines 解析 |
| [window.py](src/exo_collection/apps/calculate/window.py) | 主窗口，编排 PrepWorker→OpenSimProcessWorker 流水线、协作式取消 |
| session_selector.py / sync_view.py / processing_view.py / viewer.py | 四个页面 |
| [ground_truth.py](src/exo_collection/apps/calculate/ground_truth.py) / [ground_truth_status.py](src/exo_collection/apps/calculate/ground_truth_status.py) | 真值 CSV 导出与 QC 溯源 sidecar |

**子进程 pipeline（`opensim_joint_moment_pipeline/`）**：`pipeline/` 下按领域分 `c3d/`（读 C3D）、`gaitway/`（测力台 ASCII）、`gait/`（步态检测）、`synchronization/`（同步链）、`opensim_io/`（TRC/GRF/Scale/IK/ID/viewer）、`qc/`（版本化 QC）。顶层 `transforms.py`（坐标变换）、`filtering.py`（滤波）。编排脚本在 `scripts/`。

### 2.4 关键类与协作

`CalculateController` 是唯一状态权威，信号**值传递**（UI 只读、回写必须走方法）。状态迁移刻意区分「进程退出码 0」与「QC PASS」：`COMPLETED_QC_PASS/WARN/FAIL` 只反映生物力学 QC，与子进程异常退出 `FAILED` 无关。

---

## 3. 业务需求

### 3.1 功能清单

1. **Session 发现与静态推荐**：递归扫描 `.exo/manifest.json`，受试者→动态工况→静态标定三级下拉，自动绑定该受试者最近的 STAND 静态试次。
2. **只读输入检查**：核对四输入齐全性、C3D 采样率/时长/HH19 marker 数（动态 15 点、静态 19 点）、Gaitway 双侧列、H5 采样率。
3. **自动同步**：跺脚（stomp）对齐四路时钟。
4. **人工标定**：同屏点选 IMU/Gaitway 跺脚峰、≥3 对单调峰对求中位 offset。
5. **专家强制 offset**：无峰证据时直接输入 offset。
6. **解算**：PrepWorker（EXO 环境预处理）→ OpenSimProcessWorker（opensim 环境 Scale→标定→IK→ID）。
7. **回放**：3D 骨架 + marker + GRF/COP 箭头 + 力矩曲线 + IMU 曲线，墙钟基准播放。
8. **导出真值**：IMU 12 通道 + 关节力矩对齐写 `ground_truth.csv`。
9. **历史 run 管理**：列出 `run_*`、判定 STALE。
10. **OpenSim 子环境发现/校验**。

### 3.2 完整工作流

选受试者 → 自动绑定静态 → 选动态工况 → 检查输入 → 自动同步（或人工/专家）→ 确认同步 → 配置参数（体重/身高/滤波/区间）→ 开始解算（prep → opensim 子进程）→ QC 结论 → 自动载入回放页 → 导出真值。

### 3.3 UI 结构

`CalculateWindow` 用 `QSplitter` 分左（SessionSelector + 输入报告）右（`QTabWidget` 三页）：
- **session_selector**：三下拉 + 「检查输入」。
- **sync_view**：`QStackedWidget` 自动页/人工页；人工页两个 `pg.PlotWidget`（IMU 冲击包络 + Gaitway Fz）+ offset 微调滑块。
- **processing_view**：OpenSim 子环境（浏览/自动发现/校验）、解算参数、区间勾选、进度日志。
- **viewer**：`Scene3DCanvas`（自绘 QPainter 正交投影，无 OpenGL）+ `MomentCurvesWidget` + `IMUCurvesWidget`，统一时间游标。

### 3.4 QC 语义（关键约定）

**「进程退出码 0」≠「QC 通过」**。子进程退出码：`0`=正常跑完、`2`=取消（`_CANCEL_EXIT=2`）、`1`=异常。QC 结论 `PASS/WARN/FAIL` 由 `result.json` 的 `qc.status` 单独判定，状态机据此映射，而非依据退出码。

---

## 4. 数据规格与格式

### 4.1 输入

| 文件 | 内容 | 来源 |
|---|---|---|
| `<capture>.c3d` | 标记点 + 测力台 analog（XINGYING 动捕导出，含 subject 前缀、虚拟 marker、静态副本） | XINGYING 外部导出 |
| `<capture>.txt` | Gaitway 测力台 tab 分隔 ASCII（含左右分解力/COP + `Grade (%)` 坡度 + 受试者体重/身高信息） | XINGYING 外部导出 |
| `mocap.h5` | XING/Nokov 标记点，`(frame, marker, xyz)` 毫米，100 Hz，含 `samples/host_monotonic_ns` | Collector 落盘 |
| `imu.h5` | Xsens MTw IMU，右腿 12 通道（acc/gyr/mag/roll/pitch/yaw），nominal 100 Hz | Collector 落盘 |

Session 元数据来自 Collector 写的 `.exo/manifest.json`。

### 4.2 输出（写回 Session 的 `derived/`）

原始数据只读；派生结果写 `derived/opensim/run_YYYYMMDD_HHMMSS_ffff/`：

- `sync_calibration.json`：同步结论 + 四输入 SHA-256 指纹（schema 1.1.0）
- `manifest.json`：prep 摘要 + 输入指纹 + 处理参数（schema 1.0.0）
- `static.trc`（19 点）/ `dynamic.trc`（15 点）/ `grf.mot` / `external_loads.xml` / `support_mask.npy`
- `gait2392_simbody.osim`（模型副本）、`hh19_scaledOnly.osim`、`hh19_static_calibrated.osim`、`hh19_static_calibrated_ik.mot` / `_id.mot`
- `result.json`：`files` / `marker_qc` / `id_qc` / `moments` / `qc` / `viewer`
- `qc_report.json`：版本化 QC（与 result.json/界面同一结论）
- `viewer/`：7 个 `.npy` + `viewer_meta.json`（离线回放）
- 真值 `ground_truth.csv` + `.qc.json` 写回 **session 根目录**

**viewer/*.npy 数据契约**（data_studio 消费）：

| 文件 | 形状 | 说明 |
|---|---|---|
| `time_s.npy` | `(n,)` | 时间轴（100 Hz） |
| `grf.npy` | `(n,2,3)` | `[right,left]`，分量 `[x,y,z]`，**y 轴（index 1）为竖直方向** |
| `moments.npy` | `(n,6)` | 列序 `hip_flexion_r/l, knee_angle_r/l, ankle_angle_r/l`（力矩 N·m） |
| `model_markers.npy` / `experimental_markers.npy` | — | 模型 / 实验 marker 轨迹 |
| `body_origins.npy` / `cop.npy` | — | 刚体质心原点 / 压力中心 |

---

## 5. 计算方法与实现细节（核心）

### 5.1 自动时间同步（四源对齐）

同步链由 `sync.py:run_auto_sync`（[sync.py:77-204](opensim_joint_moment_pipeline/pipeline/synchronization/sync.py)）编排，分三段：

**(a) C3D ↔ mocap.h5 逐帧精确匹配**（`c3d_h5.py:match_c3d_to_h5`）
- C3D 与 mocap.h5 由同一 SDK 录制、保存同一批浮点样本，因此能找到「C3D 第 0 帧 = H5 第几帧」的精确起点（RMS ≈ 0 mm）。
- 阈值：`_EXACT_RMS_MM=1e-4`、`_UNIQUE_GAP_MM=1.0`、`_MIN_UNIQUE_MARKERS=3`、`_MIN_UNIQUE_OVERLAP=10`（[c3d_h5.py:19-23](opensim_joint_moment_pipeline/pipeline/synchronization/c3d_h5.py)）。
- RMS 只在两边均有效的坐标上计算，分母是有效坐标数，绝不把无效值当 0。
- **唯一性复合判定**：不能只靠 `rms<=1e-4`（周期动作会让两个不同起点都近似零 RMS）；次优必须取「距离 ≥ 2 帧的真正不同位置」候选，再按 RMS 差距 + 公共 marker 数 + 重叠长度判定。

**(b) 主机单调时钟**（`clock.py`）
- 两个 H5 的样本都带 `samples/host_monotonic_ns`（同一主机单调时钟），用**绝对时钟相减**而非各自首帧归零。uint64 先转 int64 再差分防溢出。
- `clock_health` 检查单调性与间断：按全跨度估 nominal period，只把 `≥2×nominal` 的间隔计为丢包 gap，不计调度抖动。
- `find_imu_sensor` 按 `metadata/device.preview_labels` 定位右腿 IMU；`imu_sensor_on_c3d_time` 取单传感器后丢弃该传感器为 NaN 的行（imu.h5 按「包」存储、行属于不同传感器）。

**(c) IMU ↔ Gaitway 跺脚对齐**（`stomp.py`）
- 对 IMU 加速度模长 / Gaitway 总垂直力做 **2 Hz 高通包络**（`highpass_envelope`），突出冲击、抑制站姿漂移。
- 峰检测在**归一化包络**上用 `find_peaks(prominence)`，幅值评分保留原始值（`_normalize` 只抬下限不裁上限，避免稀疏冲击被 clip 平顶丢主峰）。峰间距用真实时间轴判定（`_MIN_PEAK_DISTANCE_S=0.45`），兼容 IMU 掉帧。
- `_stomp_burst` 选「采集开头、正式走路之前」的爆发段，核心是**时间结构 + 真实冲击**：以包络 P25 为安静基线，要求首峰为「硬跺脚」（≥ `_IMPACT_RATIO=15×` 基线）、前置静默 `_PRE_QUIET_S=2.0s`、段内峰同量级（≥ `_MIN_BURST_AMP_RATIO=0.05×` 最大峰）、间隔变异系数 ≤ `_MAX_INTERVAL_CV=0.50`、段后隔离 `_ISOLATION_GAP_S=2.0s`。阈值集中 [stomp.py:27-41](opensim_joint_moment_pipeline/pipeline/synchronization/stomp.py)。
- `_align_sequences` 在近恒定 offset 下单调 1:1 配对（容忍漏检/多检一峰），取对数最多、MAD 最小的解。
- `_fit_affine` 拟合一阶漂移 `gaitway = a·imu + b`，**跨度 <5s 时返回 UNASSESSED（None）**，不伪造接近 0 的漂移。
- 置信度：`HIGH`（≥3 对且 MAD≤0.05s）/ `MEDIUM`（MAD≤0.20s）/ `LOW`。**C3D↔H5 匹配非 exact+unique 时，即使跺脚 MAD 再小也降为 MEDIUM**（[sync.py:168-170](opensim_joint_moment_pipeline/pipeline/synchronization/sync.py)）。

最终 `gaitway_offset_s = median_offset + final_adjustment_ms/1000`，时间方向约定 `t_gaitway = t_c3d + offset`，写入 `sync_calibration.json`（含四输入 SHA-256、方法、峰对/offset/MAD/confidence）。输入变化会使旧标定失效。

### 5.2 OpenSim 解算链（Scale → 静态标定 → 动态 IK → 双侧 ID）

由 [process_session.py:138-316](opensim_joint_moment_pipeline/scripts/process_session.py) 驱动，输入是 `prep_session.py` 写的 `manifest.json`。

**预处理 `prepare_session`（EXO 环境，不 import opensim）**（[prep_session.py:71-243](opensim_joint_moment_pipeline/pipeline/opensim_io/prep_session.py)）：
- 读静态/动态 C3D + Gaitway ASCII，选静态稳定窗口（手动优先，否则自动滑窗）。
- 稳态分析区间（手动优先，否则 `detect_steady_walking` 自动）。
- `build_trc` 写 static.trc（19 点）/ dynamic.trc（15 点，剔除 medial 点）到 OpenSim ground 帧；`build_bilateral_grf` 写 grf.mot 与 external_loads.xml + support_mask.npy；拷贝 gait2392 模型；写 manifest.json。

**子进程内（opensim 环境）**：

1. **Scale**：`add_hh19_markers` 给通用模型加 19 个 HH19 marker（parent_frame 必须填**绝对路径** `/bodyset/...`，否则 printToXML 后重载报错）。ScaleTool：ModelScaler 用静态 TRC 按测量缩放段长/质量，**MarkerPlacer 关闭（`<apply>false</apply>`）**——因其内部 model 写在 Windows 上抛 `Xml::writeToFile` 空错误，而默认 HH19 偏移缩放后已吻合到 RMS≈0.05mm。
2. **静态 marker 标定（两遍 refine）**：`_refine_marker_locations` 在静态 trial 上逐帧 realize 位形，把实验 marker 位置变换到 parent frame，取**中位**作为新位置，clamp 到 `max_adjustment_m`（第一遍 0.15m，第二遍 0.04m）。第一遍结果进入 marker 调整分级。
3. **动态 IK**：用动态 TRC 里实际存在的 15 个 marker 建 task，锁定 subtalar/mtp + 腰椎坐标。IK 约束权重 20 / 精度 1e-5。
4. **双侧 GRF ID**：指定 ExternalLoads（左右 calcn），坐标低通 6 Hz。
5. **QC 指标**：`_marker_qc` 算逐帧 marker RMS；`_result_qc` 算左右髋力矩 p95（Nm/kg）+ pelvis 残差力 RMS/p95；`_moment_summary` 只在分析区间内统计。
6. **导出 viewer**：`export_viewer_data` 写 `viewer/*.npy` + `viewer_meta.json`。

**关键约束**：OpenSim 只吃 ASCII **相对**路径，含中文绝对路径打不开，故所有 OpenSim 消费文件写进 run 目录，setup XML 只用裸文件名，并 `os.chdir(out)` 使各 Tool 的相对路径基准重合。

### 5.3 测力台方向 / COP 变换（`transforms.py` + `ascii_export.py`）

- **三坐标系**：mocap 全局（mm，+X 左/+Y 后/+Z 上）、OpenSim ground（m，+X 前/+Y 上/+Z 右）、测力台 native（XINGYING 通道名与标定轴**相反**：`Fx1/COPx1`=侧向、`Fy1/COPy1`=行走向）。
- 常量：`R_MOCAP_TO_OPENSIM` 旋转矩阵、`O_MOCAP_MM=[-5078.5,2086.5,-251.0]` 平移参考（使台面 → OpenSim Y=0）、`REAR_AXIS_MOCAP=[814.8,0.0,2.1]` 固定后沿轴。
- `force_plate_native_to_opensim`：重排 native→标定局部帧 `(walk,lat,up)=(fy,fx,fz)` → `R_fp2mocap` 旋转 → `R_mocap2osim` 旋转 → `ground_on_foot=True` 时整体取反（地对脚）。
- `build_bilateral_grf` 用 **Gaitway 原生左右分解**（免单支撑拆分）：水平力符号 `opensim_x_sign/z_sign`（默认 -1,-1）作用于**旋转前的 native 轴**；力和 COP 绕固定后沿轴旋转坡度（角度 `-atan(grade/100)`）。
- **抗混叠**：1000Hz→100Hz 前先对每个接触段独立零相位低通（`lowpass_segmented`），避免跨接触边界振铃与直接降采样混叠。
- 身高 `>3.0m` 视为厘米 ÷100。

### 5.4 关节力矩真值导出（`ground_truth.py`）

`align_ground_truth` 把右腿 IMU 12 通道线性插值到力矩时间轴（两者都已映射到 C3D 时间），`np.unique` 去重保证 `np.interp` 单调，只保留 IMU 覆盖范围内的力矩帧。`write_ground_truth_csv` 写表头 `time_s + 12 imu_* + moment_*`。`ground_truth_status.py` 写 `.qc.json` 溯源 sidecar（qc_status/run_dir/csv size/mtime），读时校验 size+mtime 防止被后续 run 误标。

### 5.5 QC 规则（`qc/evaluate.py`，版本化 schema 1.1.0）

`evaluate_qc` 消费 marker 误差、ID 残余、同步质量、力覆盖、marker 调整分级，输出 `PASS/WARN/FAIL` + 逐项 checks + summary。判定优先级 `FAIL > WARN > PASS > INFO`。

| 检查 | 阈值 | 出处 |
|---|---|---|
| marker RMS 均值 | WARN 2.0 cm / FAIL 4.0 cm | [evaluate.py:39-40](opensim_joint_moment_pipeline/pipeline/qc/evaluate.py) |
| marker RMS p95 / 单点峰值 p95 | WARN 6.0 cm | `:41` |
| 残余力 RMS（占体重比） | WARN 15% / FAIL 30% | `:42-43` |
| 残余力 p95 | WARN 40% | `:44` |
| 同步公共 marker 数 | ≥3 | `:47` |
| 跺脚 MAD | WARN 0.05 s | `:48` |
| 时钟间断 | WARN 5 次 | `:49` |
| 力有效覆盖 | WARN 80% / FAIL 50% | `:52-53` |
| 静态 marker 调整 | REMIND 30 mm / WARN 50 mm / FAIL 80 mm | `:57-59` |

特殊规则：

- **专家强制 offset（`method==EXPERT_FORCED`）整份 QC 最多 WARN，绝不 PASS**。
- C3D↔H5 不唯一且无人工确认 → FAIL；不唯一但人工确认 → WARN。
- 跺脚峰对 <3 且非专家强制 → FAIL（普通模式禁止解算）。
- marker 调整：解剖点 >80mm 默认 BLOCK→FAIL，但若**最终动态 marker 拟合通过**（rms_mean_cm≤2.0）则降为 WARN；技术跟踪点（L/R Thigh/Shank）>80mm 仅提示。
- 关节力矩**不设硬阈值**判定 PASS/FAIL，只在 INFO 检查展示量级。

### 5.6 静态稳定窗口自动选择（`static_window.py`）

在静态 trial 里挑「完整度最高、速度最低」的连续 2.5s（`DEFAULT_TARGET_DURATION_S=2.5`、`MIN=2.0`、`EDGE_TRIM=0.5`）。评分 `score = completeness_weight×valid_frac - mean_velocity_mm_s`，`completeness_weight=1000` 使一个缺失 marker 的损失（52.6）远大于速度差异，故完整度优先。trial 过短退回「去首尾整段」并如实报告缺失，绝不伪造窗口。

### 5.7 稳态分析区间自动检测（`steady_state.py`）

用单脚 Fz 上升沿（脚跟触地）作事件，稳健接触阈值用 **p95 而非 max**；周期稳定掩码 `|period-median| ≤ 0.25×median`（`cycle_jitter_frac=0.25`）；取最长「≥3 个周期」（`min_cycles=3`）的连续稳定步段；首尾各让半周期并夹到 1.0s 边缘保护带内（`edge_guard_s=1.0`）。失败时返回 `method="fallback"` 并报告原因。

### 5.8 历史 run 管理与 STALE_INPUTS（`history.py`）

`list_history_runs` 读 `run_*/manifest.json` + `result.json` + `cancel.flag` 判定状态（completed/cancelled/failed）。STALE 判定对照 `manifest["inputs"]` 指纹：**size+mtime_ns 一致视为未变（跳过全文件哈希）**，任一不一致才重算 SHA-256 终判。旧 run 一律只读不写。

### 5.9 关键常量 / 默认值速查

| 项 | 值 | 位置 |
|---|---|---|
| marker 低通 | `6.0 Hz` | [models.py:203-214](src/exo_collection/apps/calculate/models.py) |
| GRF 抗混叠低通 | `20.0 Hz` | 同上 / [ascii_export.py:155](opensim_joint_moment_pipeline/pipeline/gaitway/ascii_export.py) |
| 力符号 | `opensim_x_sign=-1.0`、`opensim_z_sign=-1.0` | [models.py](src/exo_collection/apps/calculate/models.py) |
| 体重 / 身高默认 | `75 kg` / `1.75 m`（UI 范围 20-300 / 1.0-2.5） | [processing_view.py:80-99](src/exo_collection/apps/calculate/processing_view.py) |
| ID 坐标低通 | `6 Hz` | [run_opensim.py:244](opensim_joint_moment_pipeline/pipeline/opensim_io/run_opensim.py) |
| IK 约束权重 / 精度 | `20` / `1e-5` | [run_opensim.py:213-214](opensim_joint_moment_pipeline/pipeline/opensim_io/run_opensim.py) |
| HH19 权重 | 骨盆/足=10、膝踝成对=5、Thigh/Shank=1 | [hh19_markers.py:51-58](opensim_joint_moment_pipeline/pipeline/opensim_io/hh19_markers.py) |
| 模型 | `pipeline_root()/data/models/gait2392/gait2392_simbody.osim` | — |
| 取消退出码 | `_CANCEL_EXIT=2` | [process_session.py:67](opensim_joint_moment_pipeline/scripts/process_session.py) |

---

## 6. 输出产物

### 6.1 QC 计算结果

- `result.json` 的 `qc.status` 为 `PASS / WARN / FAIL` 三态之一（[evaluate.py](opensim_joint_moment_pipeline/pipeline/qc/evaluate.py) 综合判定）。
- `qc_report.json` 保存版本化 QC（`QC_SCHEMA_VERSION`），与 result.json、界面显示**同一结论**。
- 逐项 checks 含：同步可信度（公共 marker 数/RMS/唯一性、峰对数/MAD/置信度、是否专家强制）、静态 marker 调整分级、marker 重投影 RMS、ID 残差力（RMS/p95 占体重比）、左右力有效覆盖比例。
- 必需指标缺失或 NaN 一律判 `FAIL` 并在摘要标「缺失」。

### 6.2 回放与导出

- **3D 回放**：骨架 + 19 模型 marker + 15 实验 marker + 左右 COP/GRF 箭头，与髋力矩（左右）曲线同步游标（读 `viewer/*.npy`）。
- **真值 CSV**：`ground_truth.csv`，每一行 = 一个 C3D 时刻的右腿 IMU 12 通道特征 + 该时刻关节力矩，供训练 `pandas.read_csv` 直接读取。
- **QC 溯源 sidecar**：`ground_truth.qc.json`。

---

## 7. 与其他模块的接口与数据流

```
Collector 落盘（mocap.h5 / imu.h5）  +  XINGYING 导出（.c3d / .txt）
        │
        ▼
Calculate: 自动同步 → 预处理 → OpenSim 子进程 → QC → 真值导出
        │
        ├─► derived/opensim/run_*/（sync_calibration.json / trc / grf.mot / result.json / viewer/*.npy / qc_report.json）
        ├─► session_dir/ground_truth.csv + .qc.json（训练真值）
        ▼
data_studio（读 result.json + viewer/*.npy 渲染 QC 报告）
process（复用同一 pipeline 批量解算）
```

- **collector** → calculate：只读 `mocap.h5` / `imu.h5`（含 `host_monotonic_ns`）+ XINGYING `.c3d` / `.txt`。
- **calculate** → process：共享 `_pipeline.ensure_pipeline_on_path` 与整个 `opensim_joint_moment_pipeline`，`solve_one_session` 复刻四步。
- **calculate** → data_studio：`qc_report.py:load_gait_qc` 读 `result.json` + `viewer/*.npy` + IK/ID `.mot` 做步态质控报告（heel strike、归一化步态周期、GRF 同步）。

---

## 8. 关键文件清单

| 文件 | 一句话职责 |
|---|---|
| [process_session.py](opensim_joint_moment_pipeline/scripts/process_session.py) | OpenSim 子进程主编排：Scale→两遍静态标定→动态 IK→双侧 ID→QC→导出 viewer，JSON-Lines 事件协议，退出码 0/1/2 |
| [stomp.py](opensim_joint_moment_pipeline/pipeline/synchronization/stomp.py) | 跺脚冲击配对核心：高通包络、峰检测、爆发段选择、单调配对、MAD/置信度/仿射漂移 |
| [evaluate.py](opensim_joint_moment_pipeline/pipeline/qc/evaluate.py) | 版本化 QC 评估：五类带阈值判定 |
| [sync.py](opensim_joint_moment_pipeline/pipeline/synchronization/sync.py) | 自动同步编排 + sync_calibration.json 落盘 |
| [transforms.py](opensim_joint_moment_pipeline/pipeline/transforms.py) | mocap/测力台/OpenSim 三坐标系变换 + 坡度后沿旋转 |
| [ascii_export.py](opensim_joint_moment_pipeline/pipeline/gaitway/ascii_export.py) | Gaitway ASCII 解析 + 双侧 GRF 构建（方向/COP 变换 + 分段抗混叠） |
| [prep_session.py](opensim_joint_moment_pipeline/pipeline/opensim_io/prep_session.py) | EXO 环境预处理：静态窗口/稳态区间/TRC/GRF/manifest |
| [window.py](src/exo_collection/apps/calculate/window.py) | 主窗口编排：PrepWorker→OpenSim 子进程流水线、协作式取消、QC 映射、真值导出 |
| [controller.py](src/exo_collection/apps/calculate/controller.py) | 状态机：同步方法判定、operation token、QC 状态迁移 |
| [workers.py](src/exo_collection/apps/calculate/workers.py) | 4 个后台 Worker + 子进程 JSON-Lines 解析 + 取消终局判定 |
| [run_precision_opensim.py](opensim_joint_moment_pipeline/scripts/run_precision_opensim.py) | marker 精定位（中位 clamp）+ marker/ID QC 指标 |
| [run_opensim.py](opensim_joint_moment_pipeline/pipeline/opensim_io/run_opensim.py) | HH19 marker 添加 + Scale/IK/ID setup XML 生成 |
| [c3d_h5.py](opensim_joint_moment_pipeline/pipeline/synchronization/c3d_h5.py) | C3D↔mocap.h5 逐帧精确匹配 + 唯一性复合判定 |
| [export_viewer.py](opensim_joint_moment_pipeline/pipeline/opensim_io/export_viewer.py) | 解算结果导出为 viewer/*.npy |
| [models.py](src/exo_collection/apps/calculate/models.py) | 不可变 DTO 与状态枚举 |
| [discovery.py](src/exo_collection/apps/calculate/discovery.py) | Session 发现、静态推荐、只读输入检查 |
| [history.py](src/exo_collection/apps/calculate/history.py) | 历史 run 列表 + STALE_INPUTS 指纹判定 |
| [static_window.py](opensim_joint_moment_pipeline/pipeline/opensim_io/static_window.py) / [steady_state.py](opensim_joint_moment_pipeline/pipeline/gait/steady_state.py) | 静态稳定窗口 / 稳态分析区间自动选择 |
| [ground_truth.py](src/exo_collection/apps/calculate/ground_truth.py) | IMU 12 通道 + 关节力矩对齐导出 |
