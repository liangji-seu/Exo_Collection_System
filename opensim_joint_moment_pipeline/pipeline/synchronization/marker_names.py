"""Marker 名称规范化：让 C3D label 与 mocap.h5 marker 名互相对齐。

同一动捕 SDK 里，C3D 的 marker 名带 subject 前缀和冒号分隔符
（``003_no_exo_dynamic:R.ASIS``），而 mocap.h5 用斜杠分隔符
（``003_no_exo_dynamic/R.ASIS``）。短名（``R.ASIS``）才是跨设备稳定的
标识；前缀（subject 名）会随试验与导出变化，不能参与匹配。

命名规则只做「剥离最后一个 ``:`` / ``/`` 之前的前缀」这一件事，不做任何
改写（大小写、点、下划线都保留），因为虚拟 marker（``V_R.Hip_JC``、
``V.Sacral``）本身就带点与下划线，改写会破坏唯一性。
"""

from __future__ import annotations

from typing import Iterable


def normalize_marker_name(name: str) -> str:
    """剥离 subject 前缀，返回稳定的短名。

    ``003_no_exo_dynamic:R.ASIS`` → ``R.ASIS``
    ``003_no_exo_dynamic/R.ASIS`` → ``R.ASIS``
    ``R.ASIS`` → ``R.ASIS``
    """
    text = str(name)
    short = text.rsplit(":", 1)[-1].rsplit("/", 1)[-1]
    return short.strip()


def build_marker_index(
    c3d_labels: Iterable[str],
    h5_names: Iterable[str],
) -> dict[str, tuple[int, int]]:
    """把 C3D 与 H5 中「短名相同」的 marker 配对，返回 ``{short: (c3d_idx, h5_idx)}``。

    两边都有重名时取第一个；短名只在两边都存在时才收录。调用方可以只取
    HH19 交集，避免虚拟 marker 干扰精确匹配。
    """
    h5_index: dict[str, int] = {}
    for idx, name in enumerate(h5_names):
        short = normalize_marker_name(name)
        h5_index.setdefault(short, idx)

    result: dict[str, tuple[int, int]] = {}
    for idx, label in enumerate(c3d_labels):
        short = normalize_marker_name(label)
        if short in h5_index and short not in result:
            result[short] = (idx, h5_index[short])
    return result


__all__ = ["build_marker_index", "normalize_marker_name"]
