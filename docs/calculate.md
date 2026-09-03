# Exo Calculate 使用说明

外骨骼关节力矩离线解算端：**标定 → 同步 → OpenSim 解算 → 关节力矩回放 → QC**。

`run_calculate.py` 启动后打开的是与 Collector / Data Studio 并列的第三个桌面应用
（源码位于 `src/exo_collection/apps/calculate/`，入口 `main.py`）。

## 它能做什么

1. 选择受试者 + 静态标定 Session + 动态试验 Session；
2. 自动时间同步四种数据源：C3D、`mocap.h5` 标记点、IMU、Gaitway 测力台；
3. 自动同步失败 / 不确定时，在「同步标定」页人工目视校准；
4. OpenSim 子进程跑完 Scale → 静态 marker 标定 → 动态 IK → 双侧 GRF ID；
5. 3D 回放（骨架 + 19 个模型 marker + 15 个实验 marker + 左右 COP/GRF 箭头）与
   髋关节力矩（左右）曲线同步游标；
6. 逐项 QC 检查表；
7. 自动选择静态稳定窗口与稳态分析区间（可人工覆盖），并把「同步可信度 / 力覆盖 /
   静态 marker 调整量」一并纳入最终 QC；
8. 历史 run 管理：列出、回放、输入变化标 `STALE_INPUTS`，旧 run 只读不覆盖。

## 环境前置

需要**两个独立 Python 环境**（不能合并）：

| 环境 | 用途 | 关键依赖 |
|---|---|---|
| `Exo`（主进程） | 运行 `run_calculate.py` / `ExoCalculate.exe` | numpy 1.x、ezc3d、h5py、scipy、pandas、PySide6、pyqtgraph |
| `opensim`（子进程） | Scale / IK / ID 解算 | numpy 2.x、opensim 4.6 |

主进程**绝不 import opensim**；OpenSim 环境只在「解算」页被选择 / 校验，由
`OpenSimProcessWorker` 起子进程执行 `process_session.py`。子进程通过 JSON-Lines
把进度回传给主界面。

## 从源码运行

```powershell
python run_calculate.py          # 或 /e/miniconda/envs/Exo/python.exe run_calculate.py
```

冒烟测试（离屏建窗后即退出，不计算）：

```powershell
python run_calculate.py --smoke-test
```

## 操作流程

1. **选择数据根目录**（左上角 Session 选择器）。动态 Session 与静态标定 Session
   分别选择；静态 Session 用于缩放模型（条件名含 `STAND` 或类似站位被识别为静态）。

2. **输入检查**：点击「检查输入」，只读扫描 C3D / TXT / HDF5 是否齐全、HH19 标记点
   数量、Gaitway 双侧力列是否存在。不齐全会列出问题清单。

3. **同步标定**：点「自动同步」。高可信（HIGH）自动进入 `SYNC_CONFIRMED`；中/低
   可信（MEDIUM/LOW）自动同步只进入 `SYNC_NEEDS_REVIEW`，必须点「确认该同步」才能
   解算。人工模式至少 3 对单调峰；专家模式（直接输 offset）需二次确认、最终 QC
   最多 WARN。结果连同四输入 SHA-256 写入 `derived/opensim/sync_calibration.json`。

4. **解算**：先在「解算」页配置 OpenSim 子环境 `python.exe`，再填体重、身高、
   marker 低通、GRF 抗混叠低通（默认 20 Hz）。分析区间与静态窗口默认自动检测，
   取消勾选可手动指定。点「开始解算」前必须已通过同步门禁（`SYNC_CONFIRMED`）。
   进度以日志流式显示；「取消」是协作式的——先请求子进程自行退出，超时再强制终止，
   最终状态记为 `CANCELLED` 而非 `FAILED`。

5. **QC 与回放**：解算结束自动跳「回放」页，显示 3D 骨架 + 髋力矩曲线；「解算」页
   显示逐项 QC 检查表。历史 run 在「解算」页底部列出：可「载入回放」任意已完成 run；
   输入文件变化会标 `STALE_INPUTS`；未完成 / 取消的 run 明确标识且只读。

## QC 语义（重要）

**「进程退出码 0」≠「QC 通过」**。这两个是不同概念：

- 退出码 0 只表示 `process_session.py` 正常跑完，未崩溃、未取消；
- QC 结论由 `pipeline/qc/evaluate.py`（版本化规则，`QC_SCHEMA_VERSION`）综合判定，
  结果只有 `PASS / WARN / FAIL`；`result.json`、`qc_report.json` 与界面显示同一结论。

最终 QC 至少检查：

- **同步可信度**（C3D↔H5 公共 marker 数 / RMS / 唯一性、时钟单调性、峰对数 / MAD /
  置信度、是否专家强制 offset）；专家强制 offset 最多 WARN，绝不 PASS；
- **静态 marker 调整分级**：< 30 mm 常规、30–50 mm 提醒、50–80 mm 警告、> 80 mm
  默认阻止、需专家确认（确认后降为 WARN）；
- **marker 重投影 RMS**：均值 < 2 cm 通过、2–4 cm 警告、> 4 cm 失败；
- **ID 残差力**：< 15 % 体重通过、15–30 % 警告、> 30 % 失败；
- **左右力有效覆盖比例**。

必需指标缺失或为 NaN 一律判 `FAIL` 并在摘要标「缺失」。

## 数据安全

- 所有原始 C3D / TXT / HDF5 只读，绝不改写；
- 派生结果只写到 Session 的 `derived/`；
- 每次解算新建递增 run 目录（`derived/opensim/run_*/`），**不覆盖旧 run**；
- 取消 / 失败不会删除已写出的中间产物，方便审计；
- 每个 run 的 `manifest.json` 记录输入文件指纹（路径 / 大小 / mtime / SHA-256）；
  历史 run 据此判定 `STALE_INPUTS`，旧 run 只读、不覆盖、不自动删除。

## 编译打包

`build_exe.py` 现会依次构建三个应用：

```powershell
python build_exe.py
```

```text
dist\ExoCollector.exe
dist\ExoDataStudio.exe
dist\ExoCalculate.exe
```

`packaging/calculate.spec` 把 `opensim_joint_moment_pipeline` 的 `pipeline/` 与
`scripts/` 源码、以及 `data/models/gait2392/gait2392_simbody.osim` 打进 bundle
（排除 `outputs/` 与 `__pycache__` / `opensim.log`）。冻结后 `_pipeline.pipeline_root()`
改从 `sys._MEIPASS` 定位这些文件；`pipeline` 通过 `ensure_pipeline_on_path()` 在
运行时加入 `sys.path`。

**运行时仍依赖 OpenSim 子环境**：即使打包成 EXE，Scale / IK / ID 仍由用户在
「解算」页选定的 opensim 环境 `python.exe` 执行，主 EXE 本身不含 opensim。

> 说明：打包脚本与 spec 已就位，但 `ExoCalculate.exe` 的 PyInstaller 构建尚未在
> 本机跑通验证。构建需在**装有 PyInstaller 且含 ezc3d / scipy / pandas / PySide6 /
> pyqtgraph** 的 Python 环境中执行（建议用 Exo 环境补装 `pyinstaller`）。若
> `ezc3d` 缺 hook，可能需要按 PyInstaller 报错补一个 `hook-ezc3d.py`。构建通过
> 后仍需在实验室用真实 003 数据做一次端到端验收（同步 → 解算 → 回放 → QC）。
