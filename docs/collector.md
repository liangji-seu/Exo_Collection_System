# Exo Collector（采集端）技术文档

> 入口：`run_collector.py` → `src/exo_collection/apps/collector/main.py` → `CollectorWindow`
> 模块目录：`src/exo_collection/apps/collector/`（约 10000 行，含 5253 行的中心窗口）+ 共享层 `acquisition/` `adapters/` `timing/` `storage/` `quality/` 等
> 定位：多模态数据的**生产者**，把异构设备在统一主时钟上对齐采样，落盘为不可变、带版本化 Manifest 的 Trial 包。

---

## 1. 概述

Collector 是外骨骼多模态数据采集系统的**采集端桌面应用**（PySide6 / Windows 11 / Python 3.11）。职责边界：把超声、IMU、电机编码器、动作捕捉、肌电、测力台、同步脉冲等多个异构设备，在**同一条主机单调时钟**上对齐采样，落盘成一个不可变的 Trial 包，并以 UUID 为四层主键（Project / Subject / Session / Trial）。

核心设计原则：

- **Trial 是最小完整数据单元**：一次连续记录 = 一个 Trial，对应一个确定工况。
- **原始数据不可变**：落盘 publish 后不得原地改写（`.recording → .partial → os.replace` 生命周期保证原子发布）。
- **完整不等于同步**：不同设备用不同采样时钟，系统同时保存设备计数、设备时间与主机单调时间，并记录各时钟域到公共时间轴的映射。
- **采集、预览、写盘解耦**：UI 只消费降采样预览，UI 卡顿不阻塞采集；写盘是独立子进程。

Collector **不产生 `.c3d` / `.txt`**——这两类由 XINGYING 动捕系统外部导出，Collector 只把 `.cap` 采集名写入 `raw/xingying_trigger.jsonl` 与 `.exo/sync_manifest.json`，由 Data Studio 事后同名拷回。

---

## 2. 架构设计

### 2.1 进程与线程模型（多进程 spawn，控制面/数据面分离）

```
Qt GUI 主线程（UI + 事件泵）
   ├─ 每模态 preview worker 子进程（spawn，会话期持久持有真实硬件）
   │     └─ ProfileModalityAdapterFactory → ModalityAdapter + RecordingStreamProducer
   ├─ CollectorWorker 子进程（collector-core-<uuid8>，磁盘写盘者，跑 run_trial 全编排）
   ├─ preflight 子进程（连接/采样/写盘探针）
   └─ 适配器内部线程（厂商 SDK 回调线程 + 消费线程）
         ├─ 有界 loss-intolerant 原始队列（溢出即 FAULTED）
         └─ 128 容量 lossy 控制队列
```

关键设计：**录制不重连硬件**。录制期经 `StreamProxyAdapter` 挂到持久化 preview worker 的原始 IPC 端点，由 `RecordingCommand`（START/STOP/SHUTDOWN）经控制管道下发，避免「开始写盘」时设备断开重连。

- 控制面传输开始/停止/状态/配置等小消息；数据面传输超声帧和传感器批次，两者用独立队列，防止高吞吐数据阻塞停止命令。
- 大块采样数据用 `multiprocessing.shared_memory` 共享，避免大数组反复序列化。

### 2.2 模块内部结构

| 文件 | 职责 |
|---|---|
| [main.py](src/exo_collection/apps/collector/main.py) | CLI 入口；`--smoke-test` / `--collect-smoke-test` / `--duration`；Windows DPI Per-Monitor-v2；受控关停泵事件循环 |
| [window.py](src/exo_collection/apps/collector/window.py)（5253 行） | 中心 `CollectorWindow(QMainWindow)`：全部采集编排、状态机、健康判定、XINGYING 集成 |
| [device_preview.py](src/exo_collection/apps/collector/device_preview.py) | 每模态 preview 子进程模型：`_preview_runner_target`、`RecordingStreamProducer`、`ModalityPreviewProcessHandle` |
| [preflight.py](src/exo_collection/apps/collector/preflight.py) | `CollectorPreflightWorker`：连接/采样/校验描述符/写盘探针 |
| [device_settings.py](src/exo_collection/apps/collector/device_settings.py) | 每模态参数对话框与 `_validated_override`（Pydantic 校验） |
| [xingying_remote.py](src/exo_collection/apps/collector/xingying_remote.py) | `XingYingRemoteCapture`（UDP XML 7060 控制）/ `XingYingRemoteTrigger`（7061 触发监听） |
| [xingying_recording.py](src/exo_collection/apps/collector/xingying_recording.py) | 采集状态面板（跑马灯） |
| [button_marker.py](src/exo_collection/apps/collector/button_marker.py) | `WH_KEYBOARD_LL` 键盘钩子（`,` 打标签，`.` 启停） |
| sync_filename.py / elapsed_timer.py / status_overview.py / preview_workspace.py / theme.py | 文件名栏、秒表、五色模态状态条、可停靠预览工作区、主题 |

### 2.3 关键类与协作

- **`CollectorWindow`** 聚合：`CollectorWorker`（写盘子进程句柄）、每模态 `ModalityPreviewProcessHandle`、`PreflightWorkerHandle`、`RingTrace`、`DismissibleToastLabel`、各设置/元数据对话框、`ModalityStatusStrip`、`PreviewWorkspace`、`ElapsedTimerPanel`、`SyncFilenameBar`、`ButtonMarkerListener`。
- **`CollectorWorker`**（[workers.py:184-192](src/exo_collection/acquisition/workers.py)）：UI 侧句柄，拥有一个 spawn 的 `collector-core` 子进程 + 每模态 `SharedPreviewBuffer` + 有界队列。
- **`ModalityAdapter`（Protocol）→ `QueuedSimulatedAdapter` / `QueuedHardwareAdapter` → 各具体设备适配器**（xsens_awinda、noraxon、teensy_serial、gaitway_tcp、elonxi、raw_ethernet、xing_nokov 等）。
- **`StreamProxyAdapter`**：录制期不重新连硬件的代理适配器，从持久化 preview worker 的 `RecordingStreamEndpoint` 读原始事件。

### 2.4 共享模块角色

| 共享模块 | 角色 |
|---|---|
| `acquisition/` | workers（子进程句柄与队列）、messages（事件类型）、recording_stream / stream_proxy（录制桥）、preview（预览打包）、buffers（共享内存预览缓冲） |
| `adapters/` | 16 个适配器字符串常量 → 具体类（adapter_registry.py），硬件/仿真统一生命周期 |
| `timing/` | clock（主时钟）、clock_model（仿射映射）、pulse_detector（滞回脉冲检测）、alignment（共享脉冲对齐） |
| `storage/` | layout（目录）、manifest（契约）、package、recovery*、checksum、subject_lock、activity（采集锁） |
| `domain/` | models / states（状态机）/ events / project_codes / prompt_labels / xingying_trigger |
| `configuration/` | app_settings、device_profiles、adapter_registry |
| `quality/` | engine（评级）/ config（规则加载） |
| `writers/` `readers/` | 落盘格式实现 |

---

## 3. 业务需求

### 3.1 功能清单

1. **以 Trial 为最小单元的采集**：一次试验 = 一次受控同步录制，产出唯一 trial 目录。
2. **多模态并发采集**：`MODALITIES=(ultrasound, imu, encoder, mocap, emg)`，硬件档额外可选 `force_plate`。
3. **设备参数配置 + 预检**：每模态设置对话框，preflight 连接/采样/写盘探针。
4. **实时预览**：五块状态条 + 可停靠预览工作区，预览经共享内存推送。
5. **启停控制**：界面按钮 + 全局键盘钩子（`,`/`.`）。
6. **同步脉冲**：模拟/硬件同步脉冲波形，用于跨设备对齐（`sync_pulse` 模态 + `PulseDetector`）。
7. **提示标签**：被试 `<`、操作者 `>`、按键 `,` 三种来源，写 `raw/prompt_labels.jsonl`。
8. **XINGYING 动捕远程采集**：UDP XML 控制 `.cap` 采集，采集名 `{subject}_{condition}_r{repeat}_{uuid8}`。
9. **试验元数据**：受试者身高体重腿长、超声探头贴放、测量条件。
10. **静态标定**：`STATIC_CALIB` 条件码 + 标定点缺失检测。
11. **受控停止 / 中止 / 恢复**：受控停止超时（默认 30s）、故障中止、启动恢复与磁盘恢复。
12. **健康监控与故障分类**：掉帧率/卡死/失联判定。
13. **质量评级与报告**：A/B/C/INVALID 四级。
14. **受试者冻结锁**：Data Studio 可锁某受试者阻止 Collector 写入。

### 3.2 工作流与状态机

启动 → preflight（设备就绪 + 磁盘空间 + 写吞吐）→ 连接各模态预览 → 填写试验元数据 → 开始录制（同步等待窗）→ 采集（预览 + 写盘 + 健康监控）→ 受控停止 → 停止 → finalize（写 manifest + checksum + 原子重命名 publish）→（Data Studio 同步 c3d/txt）→ Calculate/Process 解算。

Trial 状态机（[domain/states.py:40-65](src/exo_collection/domain/states.py)）：

```
IDLE → PREPARING → READY → WAITING_SYNC → RECORDING → STOPPING → FINALIZING → FINALIZED
                                    （含 FAILED / ABORTED / RECOVERABLE 分支）
```

现场 Trial **不预设固定时长**：点击开始后所有 Writer 立即记录原始数据；首个合格同步上升沿建立正式 `t0`，未收到则回退到写盘 gate 的主机单调时钟。未收到同步脉冲记录为 `NOT_RECEIVED / OPTIONAL`，不产生告警、不阻止最终化。

### 3.3 UI 结构

`CollectorWindow` 主窗：顶部 `ModalityStatusStrip`（五色方块：off 灰 / transition 黄 / ok 绿 / bad 红 / recording 录制红）、`PreviewWorkspace` 可停靠面板、`SyncFilenameBar`、`ElapsedTimerPanel`、模态设置/元数据对话框、预检进度、XINGYING 采集状态面板。

---

## 4. 数据规格与格式（重点）

### 4.1 设备与采样率规格

**真实硬件档**（[config/devices/hardware.json](config/devices/hardware.json)）：

| 模态 | 适配器 | 采样率/帧率 | 通道与单位 | 落盘格式 |
|---|---|---|---|---|
| 超声 Raw Ethernet | `RawEthernetUltrasoundAdapter` | **20 Hz** 帧率 | 4 通道 × 1000 samples，uint8 a.u. | block_binary `.bin` |
| IMU Xsens MTw USB | `XsensMtwUsbImuAdapter` | **100 Hz**（3 台直连） | 12（acc m/s² / gyr rad/s / mag a.u. / RPY deg） | hdf5 `.h5` |
| 编码器 Teensy | `TeensySerialEncoderAdapter` | **200 Hz**（固件） | 6（左右 pos rad / vel rad/s / torque N·m） | hdf5 `.h5` |
| 动捕 XING/Nokov | `XingNokovMocapAdapter` | **100 Hz** | 每 marker xyz mm（缺 marker = 9.0e7/NaN） | hdf5 `mocap.h5` |
| 测力台 XING/Nokov | `XingNokovForcePlateAdapter` | **100 Hz** | 6（Fx/Fy/Fz/Mx/My/Mz，N / N·m） | hdf5 `force_plate.h5` |
| EMG Noraxon | `NoraxonEmgAdapter` | **4000 Hz** | 每肌肉 `channel+unit_id`，µV | block_binary `.bin` |

**设备协议细节**（README 实测约定）：

- **Teensy 编码器**：匹配 AK80-9 V3 固件的 **35 字节小端状态帧**：帧头 `0xCC`、帧尾 `0x55`、CRC-8 多项式 `0x07`（覆盖 `bytes[1:33]`）。以 200 Hz 记录左右电机位置/速度/基于 Iq 估算的扭矩；AK80 原始 CAN 反馈 50 Hz，相邻帧可能含重复电机反馈值。串口留空时按 `VID/PID 16C0:0483` 自动发现并排除蓝牙虚拟串口。
- **XING/Nokov**：`mocap` 用 `PySetDataCallback` 取 MarkerSet（`(frame, marker, xyz)` 毫米）；`emg` 用 `PySetAnalogChFunc` 取 Analog Channel 子帧（`(sample, channel)`）；`force_plate` 复用 mocap 帧的 `Analogdata` 提取 6 模拟通道。三模态来自**同一 Nokov 服务器广播**（`server_ip=10.1.1.198`），按 `stream_kind` 注册不同 SDK 回调。EMG 配置通道数必须与 Seeker 输出一致，不一致置 FAULT。
- **超声 Raw Ethernet**：Scapy 抓包（需 Npcap WinPcap 兼容模式），每帧 1 事件（4 帧 = 1 A-line）。
- **Xsens**：MT SDK 2025.2 wheel，NumPy 1.x ABI（约束 `numpy>=1.26,<2`）。

**其它已实现适配器**（未进当前硬件档，或为可选/仿真）：IMU Awinda 200 Hz（`xsens_awinda.py`，按 `PacketCounter` wrap 65536 对齐 ≤3 台）、Gaitway TCP 测力台 1000 Hz（可选 100–2000）、Elonxi 超声 20 Hz（pythonnet 载 SDK DLL）、Nokov EMG 1000 Hz、同步脉冲（仅仿真 2000 Hz）。

**模拟设备档**（[config/devices/simulated.json](config/devices/simulated.json)）：超声 20 Hz/4ch/1000 uint16、IMU 200 Hz/3 台、编码器 100 Hz、mocap 100 Hz/12 markers、emg 1000 Hz/8 通道。

### 4.2 落盘文件格式

**HDF5 信号**（`writers/hdf5_signal.py`，`exo-hdf5-signal` v1.0.0，chunk_rows=1024）：

```
/samples/data（float32）
/samples/sample_index
/samples/device_time
/samples/host_monotonic_ns
/samples/host_utc_ns
/samples/source_sequence
/events/discontinuities
/metadata/{channels, units, device, trial, clock_model}
```

用于 imu / encoder / mocap / sync_pulse / force_plate（及部分 emg）。

**二进制块**（`writers/binary_block.py`，`EXOUSBLK` v1）：64 字节头（magic/version/sequence/first_sample_index/sample_count/payload_nbytes/device_timestamp/host_monotonic_ns/host_utc_ns/flags/payload_crc32），负载为 C-order NumPy 数组，dtype/shape 在伴生 `*.meta.json`，索引 `EXOIDX01`。CRC=zlib.crc32。用于超声。索引可重建，损坏时顺序扫描 `.bin` 恢复。

**Gaitway 双流**（`writers/gaitway_packet.py`）：`gaitway_raw.bin`（逐包 U16 自帧）+ `gaitway_type1.csv` / `type2.csv` + `gaitway_meta.json` + 日志。

**NDJSON 追加式**：`raw/prompt_labels.jsonl`（人工标签）、`raw/xingying_trigger.jsonl`（动捕触发）。

**校验和**：独立 `checksums.sha256`（不入 manifest）。

### 4.3 目录布局与生命周期

```
{data_root}/{subject}/d{day}/{project}/{condition}/session{repeat}_{YYYYmmdd_HHMMSS}/
    .exo/
        manifest.json
        checksums.sha256
        sync_manifest.json
    raw/
        ultrasound.bin / ultrasound.meta.json / ultrasound.idx
        imu.h5 / encoder.h5 / mocap.h5 / emg.bin / force_plate.h5
        prompt_labels.jsonl / xingying_trigger.jsonl
    derived/  reports/  logs/
```

簿记统一在隐藏子目录 `.exo/`。生命周期后缀（大小写不敏感）：录制目录 `…session{ts}.recording` → 各产物先写 `.partial` → `publish_partial` 原子 `os.replace` → `finalize_directory` 单次 `os.replace` 发布。`iter_finalized_manifest_paths` 只返回**已发布** manifest，绝不进入 `.recording` 目录。

> 示例（真实数据根）：`Exo_Collection_Data/103/d1/F_STEADY/WALK_0P6_EXO/session1_20260908_055822`（103=受试者、d1=第 1 天、F_STEADY=项目分区、WALK_0P6_EXO=工况、session1_...=第 1 次重复）。

### 4.4 Manifest 契约

`manifest.json`（`storage/manifest.py`，Pydantic `extra="forbid"`，schema **1.2.0**，向后兼容 1.0.0/1.1.0）。顶层字段：`manifest_uuid / project_* / subject_* / session_uuid / trial_uuid / state / condition / timing / software / configuration / devices / modalities / artifacts / clock_and_alignment / quality / abnormal_termination / external_artifacts / upload_records`。

- `subject_code` 恒 3 位数字；`project_code ∈ {T, F_BASE, F_STEADY, F_TRANSIENT}`（遗留 `F` 兼容读取）。
- 每个 Artifact 记录 `modality / kind / media_type / relative_path / size_bytes / sha256 / finalized_at_utc`。
- 时钟域 `ClockDomainManifest`（kind: host_monotonic/device_tick/device_timestamp/external）+ 时钟映射 `ClockMapping`（`t_target_ns = scale_a × t_source + offset_b_ns`，含 anchor_count / residuals）。
- 发布态仅 `{FINALIZED, ABORTED, RECOVERABLE}`。

---

## 5. 计算方法与实现细节

### 5.1 多设备时钟同步（三层机制）

**(a) 主时钟源**（[timing/clock.py](src/exo_collection/timing/clock.py)）：`HostClock.monotonic_ns() = time.perf_counter_ns()` 为对齐权威（不受校时跳变影响），`utc_ns() = time.time_ns()` 仅审计；`read()` 先采 monotonic 再 UTC。

**(b) 同步脉冲波形检测**（[timing/pulse_detector.py](src/exo_collection/timing/pulse_detector.py)）：`PulseDetector` 是**流式滞回状态机**，四态 `low / rise_candidate / high / fall_candidate`。阈值 `high_threshold=2.5, low_threshold=1.0, min_pulse_width_ns=1ms, debounce_ns=0.5ms`。上升沿确认条件：`>= high` 后跌回 `<= low` 且宽度 `>= 1ms`，或持续高位达 `1ms`。每条 `SyncPulseEvent` 携带 `pulse_id=source_device:NNNNNN`、边沿类型、跨界采样索引、`host_monotonic_ns`、幅度、宽度、置信度。置信度 = 幅度裕度（相对滞回带）+ 宽度裕度（相对最小脉宽）的混合。

**(c) 每设备仿射时钟映射（事后拟合）**：`t_global_ns = a × t_device + b`。
- `fit_affine_clock`（[clock_model.py:79-151](src/exo_collection/timing/clock_model.py)）：单锚点 → `a=1, b=target-source`（offset-only，算法 `single-anchor-offset-1.0.0`）；多锚点 → 中心化最小二乘 `a=dot(Δsrc,Δdst)/dot(Δsrc,Δsrc)`，拒绝非正 a（`affine-least-squares-1.0.0`）。残差统计含 count/mean/rms/std/p95/max。
- `DeviceClockMapper`：**流式逐样本映射**——用设备计数器（如 Xsens `PacketCounter`，`wrap_mod=65536, warmup=16`）重构均匀时间戳，尺度固定为标称周期，偏移为 `arrival_ns - period×unwrapped_counter` 的滑动均值、warmup 后冻结。
- 采集期记录锚点 `(device_ts, host_monotonic_ns)`，停止后每模态锚点裁剪到 2000 点再 `fit_affine_clock`，产出 `ClockMapping` 写 manifest。写盘期间 clock_model 先置 `unfitted_during_acquisition`，映射在收数据**之后**才拟合。

**(d) 正式时间线 / t0 选取**：首个合格**上升沿**设为 `first_trigger_host_monotonic_ns`，并把 `recording_deadline` 重锚到 `trigger + duration_s`；`formal_start` = 首触发（有）否则回退到录制门 `start_token`。**同步是可选项**——缺脉冲不挂起 worker，t0 回退到录制门。

XINGYING `.cap` 对齐：去重后的 `XingYingTriggerEvent` 注册 `clock_domain="xingying_capture_clock"`（EXTERNAL）的 offset-only `AlignmentRecord`，把 frame 0 映射到采集起始 host monotonic。

### 5.2 缓冲 / 背压 / 预览降采样

- **共享内存预览缓冲** `SharedPreviewBuffer`（[buffers.py](src/exo_collection/acquisition/buffers.py)）：`multiprocessing.shared_memory` 单写多读，**seqlock 序列锁**无锁交换；头 3×uint64 `[generation, valid length, host_monotonic_ns]`，负载 float32。**溢出 = 有损降采样**：`size > capacity` 时 `np.linspace` 均匀抽点。
- **有界队列与背压**（[workers.py:35-61](src/exo_collection/acquisition/workers.py)）：
  - 遥测队列（PREVIEW/HEALTH/METRIC/SYNC）`maxsize=256`，满则静默丢弃（lossy）。
  - 控制队列（STATE + COMPLETED/FAILED/ALERT/PROMPT_LABEL）`maxsize=32`，阻塞可靠。
  - prompt/xingying 触发队列 `maxsize=256`，满抛 `RuntimeError`。
- **录制原始队列 loss-intolerant**：`_put_lossless` 满即抛 `RecordingStreamOverflow`，原始数据绝不被静默丢弃。
- **适配器原始队列溢出即致命**：`_publish_raw` 满 → `RawQueueOverflowError` → 状态 FAULTED。
- **预览降采样**：录制期预览限到 ~15 Hz；超声 >2000 点重采样到 512；`MAX_PREVIEW_POINTS=4096`、`SIGNAL_RING_CAPACITY=1000`、`EMG_PREVIEW_RING_CAPACITY=16000`。

### 5.3 健康监控与故障分类

健康轮询 0.5s。阈值（[window.py:248-255](src/exo_collection/apps/collector/window.py)）：

| 阈值 | 值 |
|---|---|
| `HEALTH_FAULT_STREAK_THRESHOLD` | 3（连续 3 次坏判故障） |
| `HEALTH_DROP_RATE_FATAL` | 0.01 |
| `HEALTH_DROP_COUNT_FATAL_MIN` | 10 |
| `HEALTH_DATA_STALE_AFTER_S` | 3.0 |

`HealthSnapshot` 暴露队列深度/丢弃/间隙与 `queue_utilization`。

### 5.4 质量评级（`quality/engine.py`）

A/B/C/INVALID 四级。硬 FAIL 集合 `{DISK_SPACE_PREFLIGHT, FORMAL_RECORDING_WINDOW, REQUIRED_MODALITY_FORMAL_DATA, SYNC_RISING_EDGE_COUNT, FIRST_SYNC_TRIGGER}` → INVALID；否则任一 FAIL → C、任一 WARNING → B、全 A 级必需项 PASS → A。阈值来自 JSON（[quality_rules/default.json](config/quality_rules/default.json)、[storage.json](config/storage.json)），**非硬编码**。

质量规则要点：`required_modalities=[ultrasound, imu, encoder, mocap, emg, sync_pulse]`；`maximum_sequence_gaps=0`、`maximum_dropped_batches=0`；同步 `minimum_rising_edges=1`、`minimum_complete_pulses=1`、`minimum_mapping_anchors=2`、`pulse_width_ns.minimum=1ms`。无真实硬件校准依据的饱和/量程/跳变阈值默认为 `UNASSESSED`，不伪造硬件阈值。

### 5.5 预检规则（`preflight.py`）

连接/准备/采样每适配器，校验描述符（modality/device_id/clock_domain/channels-vs-units），写盘探针 fsync。默认 `minimum_free_space_gib=2.0`、`write_probe_mib=1.0`、`timeout_s=1.25`。`ready` 要求可写 + 空间 + 吞吐 + 全设备 READY/OPTIONAL_UNAVAILABLE。

### 5.6 关键常量速查

| 项 | 值 |
|---|---|
| 受控停止超时 | `DEFAULT_CONTROLLED_STOP_TIMEOUT_S=30.0` |
| 预览队列 / 录制队列 | 128 / 2048 |
| 共享缓冲 | 超声 8192、imu/encoder/sync_pulse 4096 |
| 采集锁 | `AcquisitionLock` stale 5.0s / 心跳 1.0s |

---

## 6. 输出产物

每个成功 Trial 生成：

- `manifest.json`（版本化契约，含 UUID/工况/时序/设备/artifact/时钟映射/质量摘要）
- `quality_report.json`（A/B/C/INVALID 评级 + 逐项 PASS/WARNING/FAIL/UNASSESSED + 观测值/阈值）
- `device_status.csv`（设备状态审计）
- `sync_check.csv` + `sync_manifest.json`（完整边沿/脉宽/间隔/时钟映射审计）
- **两张质控预览图**（`reporting/preview_png.py` 生成）
- `warnings.txt`
- `raw/prompt_labels.jsonl`（存在人工标签时）
- `checksums.sha256`（逐文件 SHA-256）

---

## 7. 与其他模块的接口与数据流

```
Collector 落盘（manifest.json + raw/* + checksums.sha256 + catalog.sqlite3）
        │
        ├─► Data Studio：扫描 manifest 建 Catalog、上传、回放、QC（只读）
        ├─► Calculate：读 c3d/txt/mocap.h5/imu.h5 做自动同步 + OpenSim 解算
        └─► Process：复用 Calculate 管线批量解算
```

- **→ Data Studio**：只读 trial 目录 + `.exo/catalog.sqlite3`；通过 `.exo/.collector-active.json` 活动锁（心跳租约，stale 5s）实现轻量互斥；`sync_data.py` 读 `.cap` 名把 XINGYING 导出的 `.c3d/.txt` 按同名拷回 trial 目录（含 take-index `_001` 后缀去尾）。
- **→ Calculate**：`discovery.py` 通过 `iter_finalized_manifest_paths` 定位，读 `mocap.h5` / `imu.h5` 的 `samples/host_monotonic_ns` 对齐，读 `metadata/device` 求采样率。
- 下游以 **`host_monotonic_ns`** 为统一时间基准。

---

## 8. 关键文件清单

| 文件 | 一句话职责 |
|---|---|
| [window.py](src/exo_collection/apps/collector/window.py) | Collector 中心窗，采集编排/健康判定/XINGYING 集成（5253 行） |
| [device_preview.py](src/exo_collection/apps/collector/device_preview.py) | 每模态预览子进程 + 录制流桥接 |
| [preflight.py](src/exo_collection/apps/collector/preflight.py) | 预检 worker 与写盘探针 |
| [device_settings.py](src/exo_collection/apps/collector/device_settings.py) | 每模态参数对话框与 Pydantic 校验 |
| [xingying_remote.py](src/exo_collection/apps/collector/xingying_remote.py) | XINGYING UDP XML 远程采集/触发 |
| [button_marker.py](src/exo_collection/apps/collector/button_marker.py) | `WH_KEYBOARD_LL` 全局热键打标签/启停 |
| [workers.py](src/exo_collection/acquisition/workers.py) | `CollectorWorker` 子进程句柄、有界队列与背压策略 |
| [buffers.py](src/exo_collection/acquisition/buffers.py) | seqlock 共享内存预览缓冲 |
| [recording_stream.py](src/exo_collection/acquisition/recording_stream.py) + [stream_proxy.py](src/exo_collection/acquisition/stream_proxy.py) | 录制桥（命令/边界/代理适配器） |
| [clock_model.py](src/exo_collection/timing/clock_model.py) + [pulse_detector.py](src/exo_collection/timing/pulse_detector.py) + [alignment.py](src/exo_collection/timing/alignment.py) | 时钟模型 / 滞回脉冲检测 / 共享脉冲对齐 |
| [base.py](src/exo_collection/adapters/base.py) + [hardware_base.py](src/exo_collection/adapters/hardware_base.py) | 统一适配器生命周期与线程/队列模型 |
| [xsens_mtw_usb.py](src/exo_collection/adapters/imu/xsens_mtw_usb.py) / [noraxon.py](src/exo_collection/adapters/emg/noraxon.py) / [teensy_serial.py](src/exo_collection/adapters/encoder/teensy_serial.py) / [gaitway_tcp.py](src/exo_collection/adapters/force_plate/gaitway_tcp.py) / [xing_nokov.py](src/exo_collection/adapters/xing_nokov.py) | 各设备协议实现 |
| [layout.py](src/exo_collection/storage/layout.py) + [manifest.py](src/exo_collection/storage/manifest.py) | 目录规范与 manifest 契约（Schema 1.2.0） |
| [package.py](src/exo_collection/storage/package.py) + [checksum.py](src/exo_collection/storage/checksum.py) | publish / 校验和 / 原子发布 |
| [recovery.py](src/exo_collection/storage/recovery.py) + [recovery_manager.py](src/exo_collection/storage/recovery_manager.py) | 磁盘/试验级恢复 |
| [activity.py](src/exo_collection/storage/activity.py) + [subject_lock.py](src/exo_collection/storage/subject_lock.py) | 采集锁与受试者冻结锁 |
| [device_profiles.py](src/exo_collection/configuration/device_profiles.py) + [adapter_registry.py](src/exo_collection/configuration/adapter_registry.py) | 设备档默认值与适配器注册表 |
| [engine.py](src/exo_collection/quality/engine.py) + [config.py](src/exo_collection/quality/config.py) | 质量规则与 A/B/C/INVALID 评级 |
| [hdf5_signal.py](src/exo_collection/writers/hdf5_signal.py) + [binary_block.py](src/exo_collection/writers/binary_block.py) + [gaitway_packet.py](src/exo_collection/writers/gaitway_packet.py) | 三种落盘格式 |
| [states.py](src/exo_collection/domain/states.py) + [events.py](src/exo_collection/domain/events.py) + [prompt_labels.py](src/exo_collection/domain/prompt_labels.py) + [xingying_trigger.py](src/exo_collection/domain/xingying_trigger.py) | 状态机 / 事件 / 标签 / 触发 |
