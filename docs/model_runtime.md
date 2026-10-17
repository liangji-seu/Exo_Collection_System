# Exo 在线测试（模型推理运行时）技术文档

> 入口：`run.py` →「在线测试」→ `run_model.py` → `src/exo_collection/apps/model_runtime/main.py` → `ModelRuntimeWindow`
> 模块目录：`src/exo_collection/apps/model_runtime/`（24 个文件，约 8000 行）
> 定位：把已训练的 IMU 力矩估计 TCN 模型**在线部署**，经中层助力处理后，把期望力矩经**反馈门控**下发到电机，并复用采集端基础设施做在线录制。
> 姊妹文档：审核结论见 [model_runtime_在线测试推理框架审核总结.md](model_runtime_在线测试推理框架审核总结.md)

---

## 1. 概述

「在线测试」是外骨骼系统的**实时推理运行时**（PySide6 / Windows 11 / Python 3.11）。它不负责训练模型，也不负责离线分析，只做一件事：**把采集端送来的实时数据，因果地变成电机力矩指令，安全地下发。**

它在 Collector（采集端）的数据采集/存储底座上，加装了一条五段式实时推理链：

```
订阅环形缓冲 → 因果特征 → 可插拔模型后端 → 中层助力 → 反馈门控电机输出
```

核心设计原则：

- **采集 / 推理 / 输出三层解耦**：采集复用 Collector 的预览子进程；推理跑在独立 `QThread`；电机串口服务跑在独立线程。三者互不阻塞，任何一层故障都不阻塞另两层的安全关停。
- **因果性零妥协**：特征装配只取「≤ 当前网格点」的最近样本（previous-sample），不插值、不补零、不重复首帧、不用未来样本，样本年龄上限 50 ms。训练与在线推理的通道顺序/归一化语义严格一致。
- **单一契约 `ModelSpec`**：模型怎么加载、输入怎么装配、输出怎么映射，全部由一份 `model_spec.json` 声明（Pydantic `extra="forbid"`，坏配置加载即失败，不会跑到会话中途才暴露）。
- **安全第一**：输出直接驱动电机，因此使能必须经底层反馈确认、任何异常先归零再失能、扭矩限幅、看门狗、故障首沿锁存、急停路径层层设防。
- **统一主机单调时钟**：全链路只用 `time.perf_counter_ns()` 的 `host_monotonic_ns`，与采集端同一时间轴。

本模块与其余模块的关系：

- **collector**：直接复用其 `ModalityPreviewProcessHandle`（预览子进程）、`recording_stream`（原始数据流）、`domain.events`（事件类型）、`writers`（HDF5/Binary 落盘）与 UI 主题/设置。在线录制产物与采集端同格式。
- **run_online_replay.py**：只读离线回放本模块录制的在线会话，**从不 import 电机代码**。
- **calculate / data_studio / process**：不直接耦合；本模块仅复用共享层 `configuration` / `domain`。

---

## 2. 架构设计

### 2.1 进程与线程模型

```
Qt GUI 主线程（ModelRuntimeWindow：连接、模型/输出/故障生命周期、录制上下文）
   ├─ 每模态 ModalityPreviewProcessHandle 子进程（复用 Collector，会话期持久持有硬件）
   │     └─ 预览事件 → 信号预览面板；begin_recording 后 → RecordingStreamEndpoint
   ├─ InferenceController（QThread，推理线程：订阅→特征→模型→力矩→辅助）
   ├─ model_loader（daemon 线程，后台加载模型；generation 计数淘汰陈旧结果）
   ├─ SerialTorqueOutput（reader + sender 两条 daemon 线程；串口读写）
   └─ 录制单写线程（OnlineRecording → _SessionWriter）
```

关键设计：

- **推理不碰串口、电机不碰模型**：`InferenceController` 只产出 `TorqueCommand`，通过 `set_torque_output(output)` 注入 `TorqueOutput` 抽象（`MockTorqueOutput` 或 `SerialTorqueOutput`），运行时可在两者间切换（`attach_motor` / `detach_motor`）。
- **全部用 `RLock` / `threading.Event` / `generation` 计数防重入与防陈旧回调**：模型加载完成、推理预测、故障事件到达 UI 时都校验 sender/generation，晚到的旧结果不会覆盖新状态。
- **使能是「一次性请求」**：`poll_output` 每次 Run/Connect 边界最多消费一次使能意图，失败不自动重试，避免「自动重新使能」的风险。

### 2.2 模块内部结构

| 文件 | 职责 |
|---|---|
| [main.py](src/exo_collection/apps/model_runtime/main.py) | CLI 入口；`--smoke-test`（离屏 + simulated 设备 + demo 模型 + 30s 超时）；DPI 感知 |
| [window.py](src/exo_collection/apps/model_runtime/window.py)（2143 行） | `ModelRuntimeWindow` 主窗口：连接/模型/输出/故障生命周期、电机串口选择与轮询、录制上下文 |
| [spec.py](src/exo_collection/apps/model_runtime/spec.py) | `ModelSpec` 声明式契约（Pydantic，`schema_version="1.0.0"`，`extra="forbid"` + 交叉校验） |
| [model_catalog.py](src/exo_collection/apps/model_runtime/model_catalog.py) | 扫描 `runtime_data/models/*/model_spec.json`，只读元数据，不 import 模型 |
| [model_loader.py](src/exo_collection/apps/model_runtime/model_loader.py) | 后台线程加载模型，generation 淘汰陈旧加载结果 |
| [model_bindings.py](src/exo_collection/apps/model_runtime/model_bindings.py) | 把设备槽位里的 IMU ID 绑定到 spec 的右腿/骨盆 ID |
| [backends.py](src/exo_collection/apps/model_runtime/backends.py) | 四种后端运行时：torch / torchscript / callable / demo |
| [subscription.py](src/exo_collection/apps/model_runtime/subscription.py) | `SubscriptionHub`：多模态环形缓冲，按 `host_monotonic_ns` 对齐（ROS2 topic 的类比物） |
| [features.py](src/exo_collection/apps/model_runtime/features.py) | 通用特征抽取（零阶保持 + 通道扁平 + 归一化），`FeatureExtractor` ABC |
| [imu_tcn.py](src/exo_collection/apps/model_runtime/imu_tcn.py) | IMU TCN 专用严格因果输入装配（12×N BTC），`ImuTcnFeatureExtractor` |
| [inference.py](src/exo_collection/apps/model_runtime/inference.py) | `InferenceController(QThread)`：订阅→特征→模型→力矩主循环，`_tick_attempt` 单次重试 |
| [model_adapter.py](src/exo_collection/apps/model_runtime/model_adapter.py) | 输入/输出适配器：契约校验、单/双输出 → 左右力矩映射、`right_to_both_same_sign` 复制 |
| [assistance.py](src/exo_collection/apps/model_runtime/assistance.py) | `AssistancePipeline`：一阶因果低通 + 增益/体重 + 时延，产出期望 Nm |
| [torque.py](src/exo_collection/apps/model_runtime/torque.py) | `TorqueOutput` 协议 + `MockTorqueOutput` + CRC-8/控制帧构造 + `clamp_torque` |
| [motor_service.py](src/exo_collection/apps/model_runtime/motor_service.py)（939 行） | `SerialTorqueOutput`：串口双线程、状态帧解析、反馈门控使能、故障首沿锁存 |
| [serial_ports.py](src/exo_collection/apps/model_runtime/serial_ports.py) | 只枚举串口、识别 Teensy（VID:PID），**绝不开/写设备** |
| [run_coordinator.py](src/exo_collection/apps/model_runtime/run_coordinator.py) | `RunCoordinator`：运行/输出状态机、电机挂接/脱挂、使能一次请求 |
| [startup.py](src/exo_collection/apps/model_runtime/startup.py) | 可取消的「加载→连接→预热→启动」阶段协调器 |
| [recording.py](src/exo_collection/apps/model_runtime/recording.py) | `OnlineRecording`：按使能边界录制，复用 Collector 存储格式 |
| [session_log.py](src/exo_collection/apps/model_runtime/session_log.py) | 追加式运行审计 jsonl |
| [diagnostics.py](src/exo_collection/apps/model_runtime/diagnostics.py) | 有界、best-effort 应用诊断（RotatingFileHandler） |
| [control_widgets.py](src/exo_collection/apps/model_runtime/control_widgets.py) | 参数控件 + 电机串口选择（构造即不碰设备） |
| [preview_panels.py](src/exo_collection/apps/model_runtime/preview_panels.py) | 「模型输出」面板（预测/期望/实际下发三态同图） |

### 2.3 端到端数据流

```
设备/模拟器
  → ModalityPreviewProcessHandle 子进程（复用 Collector，会话期持久持有硬件）
       ├─ 预览事件 → SignalPreviewMixin 信号预览面板
       └─ begin_recording 后 → RecordingStreamEndpoint.queue
            → SubscriptionHub（每模态一个环形缓冲，按 host_monotonic_ns 对齐；
               二维 sample_shape 用 NaN 槽位保留各物理设备（如两台 IMU）的独立到达时刻）
                 → FeatureExtractor（IMU TCN：右腿 6 轴 + 骨盆 6 轴 → [12, N] float32 BTC）
                      → ModelRuntime.predict → 平坦 [out] float32 向量
                           → ModelOutputAdapter.map_output（单标量→右髋；成对→左右）
                                → AssistancePipeline（Nm/kg×体重 → 一阶低通 → 增益 → 时延 → 期望 Nm）
                                     → TorqueOutput（Mock 记录 / Serial 下发）
                                          → 串口控制帧（限幅 + CRC-8）→ 电机
```

**时间语义**：特征对齐是「每个网格点取 ≤ 该点的最近样本，且样本年龄 ≤ 50 ms」。两路独立 IMU 走同一条水位线（`min(right, pelvis, now)` 对齐到控制率整数倍），任何一路没有新样本都不会伪造出新网格。宁可空窗等数据，也绝不插值或重复。

### 2.4 关键类与协作

- **`ModelRuntimeWindow`**：聚合 `RunCoordinator`（运行/输出状态机）、`InferenceController`（推理线程）、`ModelLoader`、`OnlineRecording`、`Diagnostics`、`SessionLog`、`SubscriptionHub`、`TorqueOutput`（Mock 或 Serial）与各预览面板。`_run_prediction` 走可取消的「加载→连接→预热→启动」阶段链。
- **`InferenceController(QThread)`**：`run()` 主循环 = `_tick`；`_tick_attempt` 做一次有上限的过期/新鲜度/重试处理，产出 `prediction`/`torque_commanded`/`fault` 等 Qt 信号。`_fault_stop` 先归零再失能。
- **`ModelInputAdapter` / `ModelOutputAdapter`**：预检契约 → 抽特征（带新鲜度看门狗）→ `predict` → 映射输出 → 交给 `AssistancePipeline`。
- **`SubscriptionHub`**：`register_endpoint` 登记每个 `RecordingStreamEndpoint`，`drain`/`window`/`window_for_device`/`latest_for_device` 供特征层按设备取窗。
- **`SerialTorqueOutput`**：`_feed_bytes_locked` 状态帧解析（CRC/格式错误计数、非有限反馈即故障）；使能确认靠「反馈 `state==2` 且 `error==0` 且晚于使能请求代际」。
- **`RunCoordinator`**：`OutputState` 状态机（见 §3.4），`attach_motor`/`detach_motor` 在 Mock 显示输出与真实电机之间切换；`poll_output` 每边界最多消费一次使能意图。

### 2.5 复用 Collector 的资产

| 复用的采集端资产 | 位置 |
|---|---|
| `ModalityPreviewProcessHandle`、`ProfileModalityAdapterFactory`、`PreviewWorkspace` | `collector.device_preview` |
| `SignalPreviewMixin`、`ModalityStatusStrip`、`COLLECTOR_STYLESHEET` | `collector.*` |
| `load_device_profile`、`SharedAppSettings`（共享 QSettings 命名空间） | `configuration` |
| `RecordingStreamEndpoint` / `RecordedRawEvent` / `RecordingBoundary` | `acquisition.recording_stream` |
| `SampleBatch` / `FrameBatch` / `SyncPulseEvent` | `domain.events` |
| `Hdf5SignalWriter` / `BlockBinaryWriter` / `TrialLayout` / `finalize_trial_package` | `writers` / `storage` |
| 设备适配器：IMU simulated、超声 raw_ethernet、encoder teensy_serial（CRC-8） | `adapters` |

**收益**：在线录制落盘格式与采集端完全一致（`raw/imu.h5`、`raw/encoder.h5`、`.exo/manifest.json`、`raw/online_events.jsonl`），所以 `run_online_replay.py` 和 `read_online_recording()` 能用 Collector 的读取器做样本级回放与分析。

---

## 3. 业务需求与关键契约

### 3.1 功能清单

1. **在线推理**：连接设备 → 订阅实时数据 → 因果特征 → 模型 → 力矩。
2. **可插拔模型后端**：`torch` / `torchscript` / `callable` / `demo` 四选一（见 §3.2）。
3. **中层助力**：Nm/kg 输出 × 体重 → 一阶因果低通 → 增益 → 时延 → 期望 Nm → 左右分配。
4. **电机输出**：`MockTorqueOutput`（记录不下发）与 `SerialTorqueOutput`（真实下发，反馈门控使能）。
5. **在线录制**：按使能边界录制，与采集端同格式，可用 `run_online_replay.py` 回放。
6. **故障生命周期**：故障首沿锁存 → 归零失能 → 回收连接 → 弹窗 + 日志入口，需显式重新运行/连接。
7. **诊断/审计**：`Diagnostics`（有界轮转）+ `SessionLog`（运行审计）+ 在线事件 jsonl。

### 3.2 模型部署契约（接入新模型只看这里）

`model_catalog.discover_local_models` 扫描 `runtime_data/models/*/model_spec.json`。一个部署目录最小包含：

```
runtime_data/models/<model_name>/
  ├─ model_spec.json        # 唯一契约，schema_version="1.0.0"
  ├─ model_display.json     # 可选：显示名/描述（缺省由 spec 推导）
  └─ 权重/模块文件           # 按 backend 决定：final_model.ts / *.pt / model.py
```

**四种后端**：

| backend | 权重/模块 | 说明 |
|---|---|---|
| `torch` | `.pt` + `module.py::class` | 从源码文件 import 类再恢复权重（原型） |
| `torchscript` | `.ts` | `torch.jit.load`，自包含，**不 import 训练代码**（生产 IMU TCN） |
| `callable` | `module.py::class` | 任意暴露 `predict(ndarray)` 的类，零依赖（快速原型） |
| `demo` | 无 | 内置确定性标量，供框架/E2E 测试 |

**模型推理统一契约**：`predict(features: np.ndarray) -> np.ndarray`，输入 `[C, T]` float32（`input_layout="btc"`），输出平坦 `[out_features]` float32。

**`ModelSpec` 交叉校验（坏配置加载即失败）**：启用 `imu_tcn` 时强制 `backend=torchscript`、`input_layout=btc`、`unit=Nm/kg`、窗口 `window_s×control_rate_hz` 必须为正整数、外部归一化必须为空（scaler 已内嵌模型）、只允许右髋单输出。

### 3.3 IMU TCN 通道契约（训练侧与在线侧必须一致）

输入 `[B, T, 12]`（btc），仅帧数随模型不同：

- **前 6 列 = 右大腿 IMU**：`acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z`
- **后 6 列 = 骨盆 IMU**：`acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z`
- **不得**输入左腿、磁力计或姿态角；通道顺序不可变。
- 模型内已嵌入训练域 12 通道 mean/std，**外部不得二次归一化**。
- 输出为右髋 Nm/kg。

`ImuTcnFeatureExtractor.extract` 只取每台设备前 6 个字段（`right[:, :6]` + `pelvis[:, :6]`），与训练侧严格对应——这是移植正确性的核心保障点。

当前部署的三个 TorchScript 模型（本体 `IMUTorqueTCN`，`model.py`）：

| 目录 | 控制率/窗口 | 扭矩限幅 |
|---|---|---|
| `imu_right_pelvis_torque_v1_20261010` | 20 Hz / 40 帧 | 0 Nm（默认禁助力） |
| `imu_right_pelvis_torque_v2_20261016` | 20 Hz / 40 帧 | 18 Nm |
| `imu_right_pelvis_torque_100hz_rf94_20261016` | 100 Hz / 94 帧 | 18 Nm |

### 3.4 输出状态机（安全核心）

```
DETACHED（电机未连接）
   → DISARMED（已连接 · 保持失能）
        → ARM_INTENT（运行/连接边界产生一次使能意图）
             → WAITING_CONFIRMATION（已发一条 0 扭矩 + 使能帧）
                  → ENABLED（收到底层新 state=2 且 error=0 的确认帧）
                  → FAULT（确认超时 / 反馈异常）
DISARMED / ENABLED / ARM_INTENT 任一 → FAULT（底层故障、看门狗、串口错误）
```

使能门控条件（`poll_output.validate()`）：真实输入（非 simulated）→ 电机已连接 → 反馈新鲜（年龄 ≤ `feedback_timeout_ns`）→ `state ∈ {0,2}` 且 `error==0`。确认帧必须**晚于使能请求代际**，不能用请求前的旧帧确认。

### 3.5 电机串口协议（控制帧 / 状态帧）

- 帧头/尾：`HEAD_CONTROL=0xAA` / `HEAD_ENABLE=0xBB` / `HEAD_STATUS=0xCC` / `FRAME_TAIL=0x55`；使能命令 `ENABLE_CMD=0xFC` / `DISABLE_CMD=0xFD`；`SEND_HZ=200`。
- 状态帧 `STATUS_FORMAT="<BHBBffffffIBB"`：`head + seq + state + error + 6×float32 + device_time + crc8 + tail`。
- CRC-8（poly 0x07），与采集端 `teensy_serial.calc_crc8` 逐字节一致。
- 固件错误码：**253 = 至少一侧电机超过 200ms 无合法 CAN 反馈**，**254 = PC 控制指令超时**（详见 [电机253故障核查_20261016.txt](电机253故障核查_20261016.txt)）。

### 3.6 安全分层防御

1. **输出状态机**：使能一次性请求 + 反馈确认（§3.4）。
2. **多层归零**：`zero_on_error`（任何推理异常先 `zero()` 再 `set_enabled(False)`）；`enable_required`（Mock 与 Serial 都强制）；急停按钮独立路径。
3. **硬件限幅**：`clamp_torque` 把非有限值强置 0 再夹到 `[-limit, +limit]`；`build_control_frame` 对限幅值做 **float32 保守向下舍入**（防止固件解码值超过操作者设定阈值）；`torque_limit_nm ≤ TORQUE_MAX_NM`。
4. **故障首沿锁存**：同一故障只做一次归零/失能（防串口写风暴），审计事件缓存在 `_deferred_record_events`，物理归零完成后才落盘。
5. **看门狗**：推理 `watchdog_s`（默认 0.25s）、命令超时、反馈新鲜度、使能确认超时、50ms 源年龄。
6. **真实硬件禁止模拟数据**：`smoke_test` / 非 hardware profile / 描述器未显式 `simulated=false` 任一成立即拒绝连接真实电机或使能。

---

## 4. 运行与测试

### 4.1 运行

```bash
# 正常启动（EXO 环境）
E:\miniconda\envs\EXO\python.exe run_model.py

# 或从统一启动器 run.py 选「在线测试」

# 冒烟测试（离屏 + simulated 设备 + demo 模型，30s 内自动结束，不碰硬件）
E:\miniconda\envs\EXO\python.exe run_model.py --smoke-test
```

依赖环境：EXO conda 环境（PySide6 + numpy + torch + pydantic + pyserial）。注意本模块与 `opensim` 环境分离（见仓库记忆）。

### 4.2 测试

- 单元测试：`tests/unit/test_model_runtime_*.py`（30 个，覆盖 spec/后端/订阅/特征/推理/电机服务/故障生命周期等）。
- 集成测试：`tests/integration/test_model_runtime_*.py`（覆盖 100Hz 真后端调度、致命故障策略、实时加载、一键启动、输出切换、饱和 IMU、simulated 全链路）。

```bash
QT_QPA_PLATFORM=offscreen PYTHONIOENCODING=utf-8 /e/miniconda/envs/EXO/python -m pytest \
  tests/unit/test_model_runtime_*.py tests/integration/test_model_runtime_*.py -q
```

### 4.3 运行产物（本地，已被 `runtime_data/` gitignore）

- `runtime_data/model_runtime_diagnostics/online_*.jsonl`：每进程一次的诊断日志（`Diagnostics`）。
- `runtime_data/model_runtime_logs/*.jsonl`：运行审计（`SessionLog`）。
- `runtime_data/online_recordings/`：在线录制会话（同 Collector 格式）。

---

## 5. 扩展点

| 需求 | 改哪里 |
|---|---|
| 接入新模型 | 新建 `runtime_data/models/<name>/model_spec.json` + 权重/模块，按 §3.2 契约 |
| 新增后端类型 | `backends.py` 增加 `build_model_runtime` 分支 + `spec.py` 的 `Literal` 枚举 |
| 新增输入模态/特征 | `features.py` 实现 `FeatureExtractor` 子类 + `inference._default_extractor` 注册 |
| 新输出映射 | `model_adapter.py::map_output_values` + `AssistancePipeline` |
| 新电机协议 | `torque.py` 增加 `TorqueOutput` 实现（沿用 `clamp_torque` / CRC） |
| 新录制字段 | `recording.py::_SessionWriter` + `read_online_recording` 同步 |

---

## 6. 一句话总结

「在线测试」= Collector 数据底座 + 「订阅环形缓冲 → 因果特征 → 可插拔模型 → 中层助力 → 反馈门控电机输出」五段式实时推理链，以 `ModelSpec` 单一 JSON 契约贯穿「移植—校验—加载—运行」全过程，安全设计（使能确认、故障锁存、限幅、看门狗、因果对齐）达到可驱动真实电机的工程强度。
