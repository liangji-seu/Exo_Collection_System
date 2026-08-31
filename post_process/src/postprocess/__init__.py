"""post_process —— C3D → OpenSim 逆动力学流水线（完全脚本化，无 GUI）。

包结构：
- ``c3d``            C3D 解析 / inspection / marker & force 提取
- ``preprocessing``  单位、滤波、同步、坐标变换、marker/GRF 处理
- ``opensim_io``     TRC / GRF.mot / ExternalLoads / Setup 文件写出
- ``opensim_pipeline`` Scale / IK / ID 的 OpenSim 绑定 wrapper
- ``validation``     配置 / 变换 / 单位 / 同步 校验
- ``qc``             marker / force / ik / id 质量控制与报告
- ``export``         髋关节净力矩导出

设计原则（见 docs/MISSING_INFORMATION.md 与根 prompt）：
凡是未实测的标定矩阵、坐标轴、左右脚 GRF、受试者质量等，一律显式暴露为
BLOCKING，绝不静默猜测或伪造，也绝不让下游计算"看着能跑但结果是错的"。
"""

__version__ = "0.1.0"
