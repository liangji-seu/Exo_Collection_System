"""Exo Calculate —— 标定 / 同步 / OpenSim 解算 / 回放的独立离线程序。

与 Exo Collector、Exo Data Studio 并列的第三个桌面入口。本包只负责 UI 与
任务编排；所有数值计算复用 ``opensim_joint_moment_pipeline``，数据树与
Manifest 复用 ``exo_collection.storage`` / ``exo_collection.domain``。
"""
