# MISSING_INFORMATION —— 待补充的实验信息

所有未实测/未确认的信息汇总。`run_pipeline.py` 的 preflight 会把这些标成
`BLOCKING`，对应 stage 暂停，直到填实。**优先级**：BLOCKING > TODO > TODO_RECOMMENDED。

## A. 受试者信息
1. **[BLOCKING]** subject mass（kg）—— Scale 缩放惯量 + ID 必需。
2. **[RECOMMENDED]** subject height（m）—— 部分缩放策略用。

## B. 静态 trial
3. **[BLOCKING]** 对应受试者的 static C3D。需确认含 `Knee.Medial` / `Ankle.Medial`
   等 HH 标定点。

   > **实测发现（2026-08-31）**：当前这份 `100_WALK_STEADY_0P75_r1_*.c3d` 是**动态
   > trial**，其内嵌的 `100_no_exo_static` subject 全部 36 个 marker 均为
   > `missing=100% / 有效帧=0`（纯占位、全 0/NaN）。**Scale 无法用它做静态标定**，
   > 需要一份**独立采集的静态标定 trial**（受试者静止站立、含 4 个 Medial 点 +
   > V.Sacral 有效数据的 c3d）。

## C. Generic model
4. **[BLOCKING]** `gait2392` generic osim 路径（官方文件，不自行下载）。

## D. HH19 → OpenSim MarkerSet
5. **[BLOCKING]** 最终 marker mapping（HH19 → gait2392 marker 名）。
6. **[BLOCKING]** `V.Sacral` 定义 + 是否用于 IK。
7. **[BLOCKING]** HH 静态 scaling measurement 定义（用哪些 marker 定骨段长度）。

## E. NOKOV 全局坐标系
8. **[BLOCKING]** +X / +Y / +Z 各指向哪里；哪个是 vertical / forward / left-right；
   单位 mm 还是 m。

## F. Gaitway 本地坐标系
9. **[BLOCKING]** +X / +Y / +Z 实际物理方向；origin 位置；COP 参考平面；
   COP 单位；moment 单位。

## G. Gaitway → NOKOV 标定
10. **[BLOCKING]** `R_forceplate_to_mocap`（旋转）。
11. **[BLOCKING]** `t_forceplate_origin_in_mocap`（平移）。
    最终存成 `forceplate_calibration.yaml/json`，并做正交性校验（R.T@R≈I，det≈+1）。

## H. NOKOV → OpenSim
12. **[BLOCKING]** mocap global 与 OpenSim model ground 的轴对应 →
    `R_mocap_to_opensim`（Z-up→Y-up 等）。

## I. GRF
13. **[BLOCKING]** C3D 实际保存的 GRF 通道：`TOTAL_ONLY` 还是 `LEFT/RIGHT`。
    （当前实测 = **TOTAL_ONLY**，双支撑阶段无法唯一确定左右 external load。）
14. **[BLOCKING]** Force convention：`ground→foot` 还是 `foot→ground`。
15. **[BLOCKING]** 左右脚 channel mapping。
16. **[BLOCKING]** 左右脚 COP。
17. **[BLOCKING]** 左右脚 Tz / free moment 是否存在。

## J. Treadmill 坡度
18. **[BLOCKING FOR SLOPE]** 坡度改变时 Gaitway frame 是否随台面旋转。
19. **[BLOCKING FOR SLOPE]** GRF 是否由软件自动转入固定坐标。
20. **[BLOCKING FOR SLOPE]** forceplate origin 是否随坡度变化。
21. **[BLOCKING FOR SLOPE]** 每个 trial 的 slope angle。

## K. Synchronization
22. **[BLOCKING]** C3D marker 与 GRF 是否硬件同步。
23. **[BLOCKING]** 是否存在 frame offset。
24. **[BLOCKING]** point rate。
25. **[BLOCKING]** force/analog rate。

## L. Filtering
26. **[TODO]** marker cutoff（Hz）。
27. **[TODO]** GRF cutoff（Hz）。

## M. OpenSim settings
28. **[TODO]** IK marker weights（当前 DEFAULT_INITIAL_VALUE）。
29. **[TODO]** final model coordinate 选择。
30. **[TODO]** 是否锁定不需要的自由度。
31. **[TODO]** ID low-pass settings。

## N. Exoskeleton（穿戴外骨骼的 trial）
32. **[BLOCKING FOR STRICT ID]** 外骨骼质量。
33. **[BLOCKING FOR STRICT ID]** 各部分质量分布。
34. **[BLOCKING FOR STRICT ID]** 惯量。
35. **[BLOCKING FOR STRICT ID]** 人-外骨骼交互力/力矩。
36. **[BLOCKING FOR STRICT ID]** 是否有主动助力。
    → 第一版优先处理 **NO-EXOSKELETON** trial。

---

## OpenSim 安装（补充说明）

OpenSim 4.x Python 绑定是**官方 conda 包**，不是 `pip install opensim`：

```bash
conda install -c opensim-org opensim
```

- 绑定与特定 Python 版本绑定，通常需要**专用 conda 环境**；
- 无绑定时，C3D inspection / 预处理 / TRC / MOT 写出仍可开发与测试，
  只有 Scale/IK/ID 报 BLOCKING。
