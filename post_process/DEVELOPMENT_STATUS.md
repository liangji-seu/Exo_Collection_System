# DEVELOPMENT_STATUS —— post_process 开发记录

> 记录当前开发进度、已确认的事实、踩过的坑与下一步。最后更新：2026-09-01。

## 1. 目标

把 NOKOV / XINGYING 导出的 C3D 解析成 marker + force plate，经坐标/单位/时间
同步，走 `Scale → IK → ExternalLoads → ID`，最终导出**左右髋关节净力矩**，作为
US / IMU / encoder 预测的监督标签。**完全脚本化，无 OpenSim GUI。**

核心原则（prompt.md 硬性要求，贯穿全部代码）：
- 未经实测/确认的信息一律显式 `BLOCKING`，绝不静默猜测、伪造数据；
- 上游能完成的 stage 继续完成（部分执行），下游依赖缺失时直接终止并报「缺什么/为什么/怎么测」；
- 严禁假设默认坐标系、静默交换 XYZ、假设力方向符号、假设左右脚 GRF 已存在。

## 2. 当前状态（Stage 1 完成）

| 阶段 | 状态 |
| --- | --- |
| C3D inspection / marker / force 提取 | ✅ READY（真实数据验证通过）|
| 坐标变换框架（点 vs 向量）+ 单位/滤波/同步 | ✅ READY（有单测）|
| TRC / GRF.mot / ExternalLoads 写出 | ✅ READY（框架，写盘验证）|
| Force→Mocap 变换 | 🔴 BLOCKING（缺标定矩阵）|
| Mocap→OpenSim 变换 | 🔴 BLOCKING（缺轴向确认）|
| Scale / IK / ID | 🔴 BLOCKING（缺模型 + 质量 + 静态标定）|

测试：**16/16 通过**（`pytest -q`）。

## 3. 已确认的关键事实

### 3.1 C3D 数据本身（ezc3d 1.7.2 实测）
- `c3d['data']['points']` = `(4, n_points, n_frames)` = `(x, y, z, residual)`；
  `c3d['data']['analogs']` = `(n_subframes, n_channels, n_frames)`。
  ezc3d **已应用** POINT:SCALE / ANALOG:SCALE，读出来就是物理值（mm / N）。
- XINGYING 用 `SUBJECTS:USES_PREFIXES=1` 把 marker 分成 `*_static`（静态标定）
  与 `*_dynamic`（动态步态）。
- 虚拟 marker 两种前缀：`V_`（V_Mid_ASIS）与 `V.`（V.Sacral）。
- 测力台 analog 是原始台面输出 `Fx/Fy/Fz + COPx/COPy + Tz`（自由力矩），
  **不是**六维力 Mx/My/Mz。

### 3.2 模型 = HH19（Helen Hayes / Newington CGM）下半身
- 静态 18 真实点（含 4 个 `Knee.Medial`/`Ankle.Medial`）+ 18 虚拟 = 36 点；
- 动态 14 真实点（去掉 4 个 Medial）+ 14 虚拟 = 28 点；
- 整文件 64 点，分类 real=32 / suspected_virtual=32 / unknown=0。

### 3.3 真实 trial `100_WALK_STEADY_0P75_r1_71e7a7c21.c3d`（Nokov · XINGYING 4.7.0.7953）
- point rate 100 Hz · mm · 64 点 · 4044 帧 ≈ 40.44 s；
- analog rate 100 Hz · 6 通道 `[Fx1, Fy1, Fz1, COPx1, COPy1, Tz1]`；
- analog 单位 `['N','N','N','mm','mm','Nmm']`，scale `[-1,-1,-1,1,1,-1]`（含符号翻转）；
- **GRF_MODE = `TOTAL_ONLY`**（只有合力，无左右脚分解）；
- 力台 1：type=1，通道 1–6，CORNERS(mm) `x=[±800] y=[390,-390,-390,390] z=0`，
  ORIGIN `[0,0,0]`，CAL_MATRIX 空；
- 时间同步 ratio=1.0 整数倍；
- **内嵌 static subject 全空**（36 点 missing=100%，有效帧=0）——见 §4.1。

## 4. 踩过的坑

### 4.1 ezc3d 打不开含中文的绝对路径（已修）
- 现象：`ezc3d.c3d()` 传**含非 ASCII（中文）的绝对路径** → `OSError: iostream stream error`；
  传**相对 ASCII 路径**正常。
- 根因：ezc3d C++ 层（ifstream）在 Windows 上对非 ASCII 绝对路径处理有缺陷。
- 尝试过的失败方案：`GetShortPathNameW` 返回的仍是原中文路径（该卷 8.3 短名已禁用）。
- 修复：`src/postprocess/c3d/reader.py` 的 `_ezc3d_path()` 三级降级：
  1. 路径本身 ASCII → 原样返回；
  2. 含非 ASCII → `os.path.relpath` 转相对当前工作目录的 ASCII 路径；
  3. 跨盘仍不行 → 复制到临时 ASCII 目录读取。
- 影响：用户的整棵目录树含中文（`e:\...\2_学业\...\2_外骨骼课题\...`），此修复必需。

### 4.2 conda run 传多行 `-c` 脚本失败
- `conda run -n Exo python -c "多行"` → `NotImplementedError: Support for scripts where
  arguments contain newlines not implemented`。
- 解决：写脚本文件或用单行 `-c`；直接调 `E:/miniconda/envs/Exo/python.exe` + `PYTHONIOENCODING=utf-8`
  （避免 GBK 打印中文报 UnicodeEncodeError）。

### 4.3 测试断言错误（已修）
- `test_trc_writer`：TRC 是 **5 行头部**（PathFileType / DataRate 标签 / DataRate 值 /
  marker 名 / 坐标标签）+ 数据行，原断言按 4 头部行写错。
- `test_grf_writer`：GRF.mot 头部 **6 行**（name/version/nRows/nColumns/inDegrees/endheader），
  列名行在第 7 行（索引 6），原断言取 `lines[5]`（=endheader）错。
- `test_c3d_parser`：`marker_class_counts["real"]` 是**整文件**（static 18 + dynamic 14 = 32），
  原断言按 dynamic-only 的 14 写错。

## 5. 目录结构

```
post_process/
  environment_check.py           依赖 + OpenSim 绑定检测
  requirements.txt
  conftest.py
  DEVELOPMENT_STATUS.md          本文件
  configs/pipeline_template.yaml 全字段模板（TODO/BLOCKING 显式暴露）
  configs/subject_template.yaml
  models/README.md               gait2392 / marker set 缺失说明
  docs/MISSING_INFORMATION.md    36 项 BLOCKING/TODO 汇总
  src/postprocess/
    blocking.py                  BLOCKING 状态机 + stage 依赖链
    c3d/reader.py · inspect_c3d.py · extract_forces.py · extract_markers.py
    preprocessing/units.py · coordinate_transform.py · filtering.py
                 synchronization.py · marker_processing.py · grf_processing.py
    opensim_io/write_trc.py · write_grf_mot.py · write_external_loads.py
              write_scale_setup.py · write_ik_setup.py · write_id_setup.py · read_sto.py
    opensim_pipeline/_bindings.py · scale.py · inverse_kinematics.py · inverse_dynamics.py
    validation/validate_config.py · validate_transform.py · validate_units.py
              validate_sync.py · validate_markers.py · validate_grf.py
    qc/marker_qc.py · force_qc.py · ik_qc.py · id_qc.py · generate_report.py
    export/export_hip_moment.py
  scripts/inspect_trial.py · run_pipeline.py · batch_process.py
  tests/（6 个测试文件）
  outputs/                       运行时产物（inspection/intermediate/opensim/results/qc）
```

## 6. 关键 BLOCKING（详见 docs/MISSING_INFORMATION.md）

**致命项（不补齐无法跑 ID）：**
1. **静态标定 trial** —— 当前这份动态 trial 内嵌 static subject 全 0，Scale 无静态数据。
   需一份独立采集的静止站立 c3d（含有效 Medial 点 + V.Sacral）。
2. **左右脚 GRF 分解** —— GRF_MODE=`TOTAL_ONLY`，双支撑期无法唯一确定左右 external load，
   双侧 ID 必须 BLOCKING（除非另有左右脚板数据）。
3. **gait2392 通用模型**（官方 osim，不自行下载）。
4. **受试者质量**（Scale 惯量 + ID 必需）。
5. **标定矩阵**：`R_forceplate_to_mocap` + `t_forceplate_origin_in_mocap`（正交性待校验）。

**次优先级：**
6. mocap 全局与 OpenSim ground 的轴对应（`R_mocap_to_opensim`，Z-up→Y-up）。
7. **力方向符号约定**（ground→foot 还是 foot→ground）——analog scale 已含 `-1` 翻转，
   但物理符号必须实测确认。
8. HH19 → gait2392 的 marker mapping + 静态 scaling measurement 定义。

**非阻塞 TODO：** 滤波 cutoff、IK 权重、坐标锁定、ID 低通。

## 7. 下一步

1. 拿到独立静态标定 trial → 验证 Scale 输入；
2. 确认左右脚 GRF 是否能分开采集 → 决定双侧 ID 可行性；
3. 确认 mocap 轴向 + 力方向符号；
4. 提供 gait2392 模型 + 受试者质量；
5. 安装 OpenSim 绑定（`conda install -c opensim-org opensim`，专用 conda 环境）。

## 8. 运行命令

```bash
cd Exo_Collection_System/post_process

# 环境检测
E:/miniconda/envs/Exo/python.exe environment_check.py

# 全量测试
E:/miniconda/envs/Exo/python.exe -m pytest -q

# 单 trial inspection
E:/miniconda/envs/Exo/python.exe scripts/inspect_trial.py \
  --c3d ../../c3d_test/c3d_test/100_WALK_STEADY_0P75_r1_71e7a7c21.c3d \
  --out outputs/demo/inspection

# 带 config 跑 preflight（可视化 BLOCKING 依赖）
E:/miniconda/envs/Exo/python.exe scripts/run_pipeline.py --config configs/pipeline_template.yaml
```
