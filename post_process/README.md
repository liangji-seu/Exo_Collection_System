# post_process —— C3D → OpenSim 逆动力学流水线

把 NOKOV / XINGYING 导出的 C3D 解析成 marker + force plate，经坐标/单位/时间
同步处理，走 `Scale → IK → ExternalLoads → ID`，最终导出**左右髋关节净力矩**，
作为后续 US / IMU / encoder 预测的监督标签。

**完全脚本化，无 OpenSim GUI。**

## 核心原则

凡是尚未实测/确认的信息（标定矩阵、坐标轴、左右脚 GRF、受试者质量、模型路径、
力方向符号……）一律显式标成 `BLOCKING`：

- 上游能完成的 stage 继续完成（**部分执行**）；
- 下游依赖缺失时**直接终止该步骤**，并明确报"缺什么 / 为什么 / 怎么测"；
- **绝不静默猜测、伪造数据、假设默认坐标系/符号。**

## 目录结构

```
post_process/
  environment_check.py           依赖与 OpenSim 绑定检测
  requirements.txt
  configs/
    pipeline_template.yaml       全字段模板（TODO/BLOCKING 显式暴露）
    subject_template.yaml        每受试者一份
  models/                        gait2392 模型与 marker set（用户提供）
  src/postprocess/
    blocking.py                  BLOCKING 状态系统 + stage 依赖链
    c3d/                         reader / inspect / extract（仅 ezc3d+numpy）
    preprocessing/               units / filtering / synchronization /
                                 coordinate_transform / marker / grf
    opensim_io/                  TRC / GRF.mot / ExternalLoads / Setup / read_sto
    opensim_pipeline/            Scale / IK / ID wrapper（OpenSim 绑定）
    validation/                  config / transform / units / sync / marker / grf
    qc/                          marker / force / ik / id QC + report
    export/                      export_hip_moment
  scripts/
    inspect_trial.py             C3D inspection（无 OpenSim 也能跑）
    run_pipeline.py              preflight + 部分执行
    batch_process.py             批量 inspection
  tests/
  outputs/                       运行时产物（inspection/intermediate/opensim/results/qc）
```

## 快速开始

```bash
# 1. 环境检测（重点看 OpenSim 绑定是否可用）
python environment_check.py

# 2. 跑测试（不依赖 OpenSim，也不依赖标定矩阵）
pytest -q

# 3. 对单个 trial 做 inspection（当前阶段核心，无需 OpenSim/标定）
python scripts/inspect_trial.py --c3d path/to/trial.c3d --out outputs/demo/inspection

# 4. 带 config 跑 preflight（把 BLOCKING 依赖可视化）
python scripts/run_pipeline.py --config configs/pipeline_template.yaml
```

## 当前状态（Stage 1）

| 阶段 | 状态 |
| --- | --- |
| C3D inspection | READY（可用真实 c3d 验证） |
| Marker / force 提取 | READY |
| 坐标变换框架（点 vs 向量） | READY（有单测） |
| TRC / GRF.mot / ExternalLoads 写出 | READY（框架） |
| Force→Mocap 变换 | BLOCKING（待标定） |
| Mocap→OpenSim 变换 | BLOCKING（待确认轴） |
| Scale / IK / ID | BLOCKING（缺模型 + 质量 + 标定） |

具体还缺什么见 [docs/MISSING_INFORMATION.md](docs/MISSING_INFORMATION.md)。
