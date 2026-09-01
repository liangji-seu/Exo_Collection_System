# opensim_pipeline —— 单块跑步机测力台下髋关节力矩解算流水线

完全脚本化、无需 OpenSim GUI。第一阶段目标：把 pipeline 搭到
「只差真实标定参数即可运行」，并用现有 C3D **自动识别可靠的单支撑区间**。

本目录是 `Exo_Collection_System` 的独立离线分析模块。把受试者 C3D 放入
`data/c3d/`，复制并修改一份 YAML 配置后运行；`data/c3d/` 与 `outputs/` 默认不提交，
避免把受试者数据和大体积中间结果推送到远程仓库。

## 数据链

```
NOKOV/XINGYING C3D
   ├── 静态 trial ──> Scale（BLOCKING：缺模型/OpenSim）
   └── 动态 WALK ──> 单支撑检测 ──> TRC/GRF ──> ID（BLOCKING）
```

## 目录

```
opensim_joint_moment_pipeline/
├── data/c3d/              真实 C3D（已拷贝）
├── configs/subject_001.yaml
├── pipeline/              包（c3d / gait / opensim_io / blocking / pipeline）
├── scripts/               inspect_trial.py · run_pipeline.py
├── tests/
└── outputs/               运行时产物
```

## 运行

**两个环境分开**（见 [[opensim-python-env-setup]]：numpy ABI 冲突，不能同环境）：
- `EXO`：读 C3D / 检查 / 单支撑检测 / 写 TRC·GRF·segments（`numpy 1.26.4` + ezc3d）
- `opensim`：Scale / IK / ID（`numpy 2.4` + opensim 4.6）

```bash
# 上游（EXO 环境）
E:/miniconda/envs/Exo/python.exe environment_check.py

# 单 trial inspection
E:/miniconda/envs/Exo/python.exe scripts/inspect_trial.py \
  --c3d data/c3d/001_WALK_LEVEL_r1_5432165b1.c3d --out outputs/inspection

# 全 pipeline（preflight + inspection + 单支撑检测 + TRC）
E:/miniconda/envs/Exo/python.exe scripts/run_pipeline.py \
  --config configs/subject_001.yaml --out outputs/subject_001/walk_level

# 测试
E:/miniconda/envs/Exo/python.exe -m pytest -q

# 下游（opensim 环境，Scale/IK/ID 待标定矩阵后接入）
E:/miniconda/envs/opensim/python.exe -c "import opensim; print(opensim.GetVersionAndDate())"
```

安装 opensim 环境：`conda create -n opensim -c opensim-org -c conda-forge opensim=4.6 python=3.11 -y`

## 001 精度修正版

精度配置在 `configs/subject_001_precision.yaml`。当前参数来自本次数据的质量审计：

- marker 低通 6 Hz、GRF 低通 20 Hz，均为零相位滤波；
- 测力台信号相对 marker 提前 160 ms（100–170 ms 网格搜索中全身残余力最低）；
- OpenSim 前向和左右向水平力反号，竖直力方向保持不变；
- 只分析 12–30 s，并从每段单支撑的两端各去掉 150 ms；
- 迭代修正模型 marker 位置后重新运行 IK 和 ID。

```bash
# 预处理（EXO 环境）
E:/miniconda/envs/Exo/python.exe scripts/prep_opensim.py \
  --config configs/subject_001_precision.yaml \
  --out outputs/subject_001/walk_level/opensim_precision

# 生成初始缩放模型和 IK（OpenSim 环境）
E:/miniconda/envs/opensim/python.exe scripts/run_opensim.py \
  --manifest outputs/subject_001/walk_level/opensim_precision/manifest.json

# Marker 精修与最终 IK/ID（OpenSim 环境）
E:/miniconda/envs/opensim/python.exe scripts/run_precision_opensim.py \
  --manifest outputs/subject_001/walk_level/opensim_precision/manifest.json \
  --seed-model outputs/subject_001/walk_level/opensim_precision/hh19_scaledOnly.osim \
  --seed-ik outputs/subject_001/walk_level/opensim_precision/hh19_ik.mot
```

主要结果为 `hh19_precision_id.mot` 和 `precision_report.json`。由于只有一块 total-force
跑台，双支撑仍无法可靠分力，因此这些结果只代表单支撑中段，不应扩展成完整步态周期的
ground truth。

### HH19 Marker 数量

- 静态试验：19 个真实 Marker；
- 动态试验：15 个真实 Marker（骨盆 3 个，左右腿各 6 个）；
- 动态时摘除的 4 个点是左右膝内侧和踝内侧 Marker。

最终 `hh19_precision_refined_2.osim` 始终保留 19 个模型 Marker。动态 IK 用实际存在的
15 个实验 Marker 驱动模型；另外 4 个模型点可由模型姿态计算出来，但不能冒充动态实测点。
`scripts/export_opensim_overlay.py` 可导出 19 个模型 Marker 和 OpenSim 骨架轨迹，供 3-D
实验点/模型点叠加质检。

## 绝对禁止（prompt2 §27）

1. 双支撑阶段 50/50 分力
2. 根据 COP 简单比例分左右力后称为 ground truth
3. 把 total GRF 同时施加给两只脚
4. 把 total GRF 全部施加给某一脚贯穿整个步态周期
5. 没有标定矩阵就硬跑 ID
6. 默认 Z-up → Y-up（垂直轴从数据推导）
7. 默认 force sign
8. 默认 V.Sacral 一定要删（作为配置项，默认 AUTO 保留）
9. 把 XINGYING joint center 直接当 OpenSim IK marker
10. 为了跑通伪造缺失实验参数

## 已知坑

- **marker 前缀**：C3D 里 marker 前缀是 `100_no_exo_*`（XINGYING 残留的 subject 名），
  与 `subject.id="001"` 无关。代码里按前缀动态剥离，不受影响。
- **单块测力台**：`GRF_MODE = TOTAL_ONLY`，双支撑无法分解左右力，第一版严格 mask。
- **垂直轴**：不硬编码 Z-up，由静态 trial 的骨盆点 vs 足点自动推导。

## 当前依赖状态

- gait2392 `.osim` 与配套 Scale/IK XML ✅ 已随模块保存
- ~~forceplate→mocap 变换~~ ✅ 已定（三点标定，COP 原点=右下角）
- ~~mocap→opensim 变换~~ ✅ 已定（R=[[0,-1,0],[0,0,1],[-1,0,0]]）
- ~~力方向符号~~ ✅ 已定（C3D=脚对台面，接 ID 前 Fx/Fy/Fz/Tz 取反）
- ~~OpenSim Python 绑定~~ ✅ 已装（独立 `opensim` 环境，4.6 / numpy 2.4）
