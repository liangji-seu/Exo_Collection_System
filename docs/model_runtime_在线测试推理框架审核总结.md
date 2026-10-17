# 在线测试推理框架（model_runtime）通读审核与总结

> 审核对象：`src/exo_collection/apps/model_runtime/`（24 个源文件，约 8000 行）
> 启动链：`run.py` →「在线测试」→ `run_model.py` → `exo_collection.apps.model_runtime.main`
> 审核日期：2026-10-16
> 结论：**架构分层清晰、安全设计扎实、与采集端充分复用；主要风险集中在超大单文件、故障码文案重复、离线回读全量载入三处，均不阻塞使用。**

---

## 一、定位与启动链

`run.py` 是统一启动器，`_MODULES` 里注册了「在线测试」项：

| 启动器显示名 | 说明 | 入口脚本 |
|---|---|---|
| 在线测试 | IMU 力矩估计 · 辅助控制 · 电机反馈 | `run_model.py` |

`run.py` 用 `subprocess.Popen([sys.executable, script])` 在独立子进程里启动目标脚本（沿用当前 EXO 解释器）。`run_model.py` 与 `run_collector.py` 结构完全一致——只是往 `sys.path` 里插 `src` 后调用各自 app 的 `main()`：

```python
# run_model.py（与 run_collector.py 同构）
SOURCE_ROOT = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SOURCE_ROOT))
from exo_collection.apps.model_runtime.main import main
```

`main.py` 负责：解析 `--smoke-test` 参数 → Windows DPI 感知 → 建 `QApplication` → 建 `ModelRuntimeWindow` → 进入事件循环。smoke 模式用临时 QSettings 强制 `simulated` 设备、离屏渲染、不碰硬件，并有 30 秒超时兜底。

**「借鉴 run_collector」的确切含义**：采集端（`run_collector.py` 背后）已经有一套成熟的「设备预览子进程 + 原始数据流」基础设施，本框架直接复用，没有另起炉灶。具体复用见 §四。

---

## 二、模块地图

按职责分层的 24 个文件：

| 文件 | 行数 | 职责 |
|---|---|---|
| `main.py` | 127 | CLI 入口、DPI、smoke 驱动 |
| `window.py` | 2143 | 主窗口：连接/模型/输出/故障生命周期/录制上下文 |
| `spec.py` | 230 | `ModelSpec` 严格 JSON 契约（`extra="forbid"`） |
| `model_catalog.py` | 92 | 扫描 `runtime_data/models/*/model_spec.json`，不导入模型 |
| `model_loader.py` | 112 | 后台 daemon 线程加载模型，generation 淘汰陈旧结果 |
| `model_bindings.py` | 38 | 把设备槽位里的 IMU ID 绑到 spec 的右腿/骨盆 ID |
| `backends.py` | 421 | 四种后端：torch / torchscript / callable / demo |
| `subscription.py` | 495 | `SubscriptionHub`：多模态环形缓冲（ROS2 topic 的类比物） |
| `features.py` | 184 | 通用特征抽取（零阶保持 + 通道扁平 + 归一化） |
| `imu_tcn.py` | 294 | IMU TCN 专用严格因果输入装配（12×N） |
| `inference.py` | 583 | `InferenceController` QThread：订阅→特征→模型→力矩 |
| `model_adapter.py` | 240 | 输入/输出适配器：契约校验、单/双输出→左右力矩映射 |
| `assistance.py` | 114 | 中层：一阶因果低通 + 增益/体重 + 时延 |
| `torque.py` | 294 | `TorqueOutput` 协议 + `MockTorqueOutput` + CRC-8/帧构造 |
| `motor_service.py` | 939 | `SerialTorqueOutput`：串口双线程 + 反馈门控使能 + 故障锁存 |
| `serial_ports.py` | 48 | 只枚举串口，绝不开/写设备 |
| `run_coordinator.py` | 276 | 运行/输出状态机、电机挂接/脱挂、使能一次请求 |
| `startup.py` | 72 | 可取消的「加载→连接→预热→启动」阶段协调器 |
| `recording.py` | 513 | `OnlineRecording`：按使能边界录制，复用 Collector 存储格式 |
| `session_log.py` | 41 | 追加式运行审计 jsonl |
| `diagnostics.py` | 113 | 有界、best-effort 应用诊断（RotatingFileHandler） |
| `control_widgets.py` | 260 | 参数控件 + 电机串口选择（构造即不碰设备） |
| `preview_panels.py` | 403 | 「模型输出」面板（预测/期望/实际下发三态同图） |
| `__init__.py` | 10 | 导出 `ModelSpec`/`load_model_spec` |

---

## 三、端到端数据流

```
设备/模拟器
  → ModalityPreviewProcessHandle 子进程（复用 Collector）
     └ 预览事件 → 信号预览面板；begin_recording 后
     └ RecordingStreamEndpoint.queue → SubscriptionHub
          └ 环形缓冲（按 host_monotonic_ns 对齐，NaN 槽位保留各物理设备到达时刻）
               → FeatureExtractor（IMU TCN：12 通道 × N 帧，严格因果、无插值/补零/前填）
                    → ModelRuntime.predict → 平坦 [out] 向量
                         → ModelOutputAdapter.map_output（单标量→右髋，或成对→左右）
                              → AssistancePipeline（Nm/kg×体重 → 低通 → 增益 → 延时）
                                   → TorqueOutput（Mock 记录 / Serial 下发）
```

**统一时间语义**：全链路只用 `time.perf_counter_ns()` 的 `host_monotonic_ns`。特征对齐是「每个网格点取 ≤ 该点的最近样本，且样本年龄 ≤ 50ms」——宁可空窗等数据，也**绝不插值、绝不重复首帧、绝不用未来样本**。这一点贯穿 `imu_tcn.py`、`features.py`、`subscription.py` 三处，是实现「因果、可复现」的关键。

---

## 四、复用 Collector 的部分（借了什么）

| 复用的采集端资产 | 位置 |
|---|---|
| `ModalityPreviewProcessHandle`（每模态一个预览子进程） | `collector.device_preview` |
| `ProfileModalityAdapterFactory`、`load_device_profile` | `collector.device_preview` / `configuration` |
| `PreviewWorkspace`、`SignalPreviewMixin`、`ModalityStatusStrip` | `collector.*` |
| `COLLECTOR_STYLESHEET`（界面主题） | `collector.theme` |
| `SharedAppSettings`（与采集端共享 QSettings 命名空间） | `configuration` |
| 原始数据流：`RecordingStreamEndpoint`/`RecordedRawEvent`/`RecordingBoundary` | `acquisition.recording_stream` |
| 事件类型：`SampleBatch`/`FrameBatch`/`SyncPulseEvent` | `domain.events` |
| 存储：`Hdf5SignalWriter`/`BlockBinaryWriter`/`TrialLayout`/`finalize_trial_package` | `writers.*` / `storage.*` |
| 设备适配器：IMU simulated、超声 raw_ethernet、encoder teensy_serial（CRC-8） | `adapters.*` |

**收益**：在线录制落盘格式与采集端完全一致（`raw/imu.h5`、`raw/encoder.h5`、`.exo/manifest.json`、`raw/online_events.jsonl`），所以 `run_online_replay.py` 和 `read_online_recording()` 能直接用 Collector 的读取器做样本级回放与分析，无需两套存储。

---

## 五、模型移植与部署机制

### 5.1 部署目录契约

`model_catalog.discover_local_models` 扫描 `runtime_data/models/*/model_spec.json`，一个部署目录最小包含：

- `model_spec.json` —— 唯一契约（`schema_version: 1.0.0`）
- `model_display.json` —— 可选显示名/描述（缺省由 spec 推导）
- 权重/模块文件 —— 按 backend 决定（`final_model.ts` / `.pt` / `model.py`）

### 5.2 四种后端

| backend | 权重 | 用途 |
|---|---|---|
| `torch` | `.pt` + `module.py::class` | 从源码文件加载类，恢复权重（主后端/原型） |
| `torchscript` | `.ts` | 加载自包含 `torch.jit` 产物，**不 import 训练代码**（生产 IMU TCN） |
| `callable` | `module.py::class` | 任意暴露 `predict(ndarray)` 的类，零依赖（快速原型） |
| `demo` | 无 | 内置确定性标量（`amplitude*tanh(mean/scale)`），供框架/E2E 测试 |

关键设计：**spec 严格校验**（`extra="forbid"` + 交叉校验）。启用 `imu_tcn` 时强制 `backend=torchscript`、`input_layout=btc`、`unit=Nm/kg`、窗口 `window_s×control_rate_hz` 必须为正整数、外部归一化必须为空（scaler 已内嵌模型）、只允许右髋输出。坏 spec 在加载时即报错，不会跑到在线会话中途才暴露。

### 5.3 当前部署的三个模型

| 目录 | 控制率/窗口 | 阈值 | 说明 |
|---|---|---|---|
| `imu_right_pelvis_torque_v1_20261010` | 20 Hz / 40 帧 | 0 Nm | 首版，默认禁助力 |
| `imu_right_pelvis_torque_v2_20261016` | 20 Hz / 40 帧 | 18 Nm | 6 Hz 滤波标签重训 |
| `imu_right_pelvis_torque_100hz_rf94_20261016` | 100 Hz / 94 帧 | 18 Nm | 最新，RF94 种子，全 103/104/201/202 训练 |

模型本体是 `IMUTorqueTCN`（`model.py`）：`Conv1d` 投影 → 4 个膨胀因果残差块（dilation 1/2/4/8，`LayerNorm` 只在通道维、保持因果）→ 取最后时刻 → `Linear` 输出右髋 Nm/kg。

**输入通道契约（三者一致，仅帧数不同）**：`[B, T, 12]`，前 6 列为右大腿 IMU（acc_x/y/z + gyr_x/y/z），后 6 列为骨盆 IMU。**不得**输入左腿、磁力计或姿态角，通道顺序不可变；模型内已嵌入训练域 12 通道 mean/std，外部不得二次归一化。这与 `imu_tcn.py::extract` 里「只取每台设备前 6 个字段、`concatenate(right[:, :6], pelvis[:, :6])`」严格对应，是移植正确性的核心保障点。

---

## 六、安全设计（本框架最重的一块）

因为输出直接驱动电机，安全是第一优先级。分层防御如下：

1. **输出状态机**（`run_coordinator.py`）：`DETACHED → DISARMED → ARM_INTENT → WAITING_CONFIRMATION → ENABLED / FAULT`。使能是**一次性请求**——先经静态校验（真实输入/方向/阈值门）→ 发一条 0 扭矩指令 + 使能帧 → **必须等到底层回 `state=2` 且 error=0 的新状态帧才确认使能**，且该帧必须晚于请求代际（不能用请求前的旧帧确认）。失败不自动重试，避免「自动重新使能」。

2. **多层扭矩归零**：`zero_on_error`（任何推理异常 → `_fault_stop` → 先 `zero()` 再 `set_enabled(False)`，物理安全优先于日志/队列检查）；`enable_required`（Mock 和 Serial 都强制）；急停按钮单独一条 `emergency_stop` 路径。

3. **硬件级限幅与协议安全**（`torque.py` / `motor_service.py`）：
   - `build_control_frame` 把限幅值 echo 给固件前，做 **float32 保守向下舍入**（0.1 这类十进制向上舍入时，逐步减 1 个 ULP），保证固件解码值绝不超过操作者设定的阈值。
   - `clamp_torque` 把所有非有限值强置 0，再夹到 `[-limit, +limit]`。
   - 串口状态帧 CRC-8 校验（与 `teensy_serial.calc_crc8` 逐字节一致），非有限反馈帧进故障锁存而绝不上报为可用反馈。

4. **反馈门控 + 看门狗**：命令超时（`command_timeout_s`）、反馈新鲜度（`feedback_timeout_s`）、使能确认超时，任一触发即锁存故障并归零失能。推理层另有 `watchdog_s`（默认 0.25 s）与 50 ms 源年龄双重检查。

5. **故障锁存与首沿动作**（`motor_service._latch_fault_locked`）：同一种故障只做一次归零/失能（防串口写风暴），先把审计事件缓存在 `_deferred_record_events`、物理归零完成后才落盘。

6. **真实硬件禁止模拟数据**（`window._require_real_inputs`）：`smoke_test`/非 `hardware` profile/描述器未显式 `simulated=false` 任一成立即拒绝连接真实电机或使能。

7. **线程纪律**：推理 `QThread`、模型加载 daemon 线程、串口 reader/sender 双线程、录制单写线程；全部用 `RLock`/`threading.Event`/`generation` 计数防重入与防陈旧回调（`_dispatch_controller`、`_on_model_loaded`、`_fault_incident` 都对 sender/generation 做校验）。

---

## 七、审核发现

### 7.1 做得好的（保留）

- **单一契约 `ModelSpec`** + `extra="forbid"`，坏配置加载即失败，契约内交叉校验覆盖到位（IMU TCN 的组合约束尤其完整）。
- **复用而非复制**：采集/存储/预览全走 Collector 既有资产，在线录制与采集数据同格式、同读取器。
- **因果性零妥协**：特征装配严格 previous-sample、50 ms 年龄上限、不插值不补齐，训练与在线推理的通道顺序/归一化语义一致。
- **防御深度**：从 spec 校验到 float32 ULP 舍入到故障首沿锁存，层层兜底。
- **可观测性**：`Diagnostics`（有界轮转）+ `SessionLog`（运行审计）+ `online_events.jsonl`（逐帧/逐命令）+ 回放/分析工具齐备。
- **测试覆盖**：`tests/unit/test_model_runtime_*.py` 30 个 + `tests/integration/test_model_runtime_*.py` 8 个，覆盖 spec/后端/订阅/特征/推理/电机服务/故障生命周期/100Hz 真后端调度等。

### 7.2 问题与风险

| 级别 | 问题 | 位置 | 说明 |
|---|---|---|---|
| 中 | **离线回读全量 `read_text`** | `recording.py:509` | `read_online_recording()` 用 `read_text().splitlines()` 一次性读入整份 `online_events.jsonl`；而 `run_online_replay.py:53` 明确注释「Never read_text this file」（因为含大模型张量）。长会话 + 100Hz 时该函数可能内存吃紧，且两处对同一文件的处理策略不一致。 |
| 低 | **故障码 253/254 文案重复 3 处** | `run_coordinator.py:_motor_fault_reason`、`window.py:_incident_message`、`window.py:_friendly_fault` | 固件特定错误码的解释分散三处，后续改文案/加新码容易漏。 |
| 低 | **`_tick_attempt` 复杂度** | `inference.py:367`（已标 `noqa: C901`） | 单个方法约 160 行，含重试/新鲜度/过期/IMU 特判多条分支。逻辑正确，但可读性与可维护性偏弱。 |
| 低 | **`window.py` 单文件 2143 行** | `window.py` | 主窗口承担 UI + 连接管理 + 故障生命周期 + 模型加载 + 录制上下文 + 电机轮询，职责过重。 |
| 低 | **`_transient_input_error` 字符串匹配** | `window.py:103` | 用子串匹配判断传输故障是否可重试；保守（配置/依赖/契约错误不重试），但匹配词表可能漏判/误判。 |
| 提示 | **100Hz 对源速率 ≥ 控制率的硬约束** | `imu_tcn.py:76/155` | `source_rate_hz + 1e-6 < control_rate_hz` 即报错。真实 Xsens 名义 100Hz 实际可能略低于 100Hz，存在误报可能；当前 1e-6 容差已很小，若实机遇「IMU source rate below control rate」可优先排查此点。 |
| 提示 | **`complete.json` 含非标准 JSON `Infinity`** | 训练产物 | `best_val_mse: Infinity` 非严格 JSON（Python `json.loads` 能解析，其它解析器可能失败）。非框架代码问题，但建议训练导出时序列化为 `null`。 |

### 7.3 建议（按性价比排序）

1. **`read_online_recording` 的 events 读取改流式**（对齐 `run_online_replay.py` 的做法，逐行 `json.loads` + 过滤），消除长会话内存风险与两处策略不一致。
2. **抽一个 `_motor_fault_message(code, fault)` 集中 253/254（及未来固件码）文案**，三处调用改为引用。
3. **拆分 `_tick_attempt`**：把「过期判断 + 重试」抽成独立方法，降低圈复杂度、便于单独测试。
4. **`_transient_input_error` 考虑结构化**：优先按异常类型/errno/winerror 判定，字符串子串仅作兜底。
5. `window.py` 的超大类拆分与 `complete.json` 导出修正属长期重构/训练侧改进，非阻塞。

---

## 八、一句话总结

「在线测试」框架在 Collector 的数据采集/存储底座上，加装了「订阅环形缓冲 → 因果特征 → 可插拔模型后端 → 中层助力 → 反馈门控电机输出」五段式实时推理链，以 `ModelSpec` 单一 JSON 契约贯穿「移植—校验—加载—运行」全过程，安全设计（使能确认、故障锁存、限幅、看门狗、因果对齐）达到可驱动真实电机的工程强度；当前遗留问题集中在可维护性与离线回读内存两处，不影响在线运行正确性。
