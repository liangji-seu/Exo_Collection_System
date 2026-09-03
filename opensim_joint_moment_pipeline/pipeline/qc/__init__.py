"""版本化生物力学 QC 规则。

独立于 OpenSim，只消费 ``process_session.py`` 产出的 ``marker_qc`` / ``id_qc``
数值，输出结构化的 PASS / WARN / FAIL 结论与逐项检查表。关键约定：**「进程退出码
0」与「QC PASS」是两回事**——本模块只依据结果数值判定生物力学可接受性。

对外入口：:func:`evaluate_qc`。
"""

from .evaluate import QC_SCHEMA_VERSION, evaluate_qc

__all__ = ["QC_SCHEMA_VERSION", "evaluate_qc"]
