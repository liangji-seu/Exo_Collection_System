"""Collector 期次 / 主工况 / 详细工况 三层分组配置。

结构（仅采集端分组，不写入 manifest / 数据）：:

    期次（可增删）
      └─ 主工况（固定 4 类：基础 / 稳态 / 非稳态 / 特殊）
           └─ 详细工况（每类下面的具体工况，可设「是否增加穿戴版本」）

4 类主工况对应协议的 ``condition_level``（``TEST`` 的随意测试 / 静态标定归入「基础」）。
详细工况用**基础工况码**（去掉 ``_NOEXO`` / ``_EXO`` 后缀）标识，``wear`` 表示是否在
非穿戴基础上额外纳入穿戴版本。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from exo_collection.protocols import load_default_protocol

PHASE_CONFIG_SCHEMA_VERSION = 3

# 主工况固定 4 类，顺序即下拉/标签页顺序。
MAIN_CATEGORIES: tuple[dict[str, str], ...] = (
    {"key": "BASELINE", "name": "基础"},
    {"key": "STEADY_STATE", "name": "稳态"},
    {"key": "TRANSIENT", "name": "非稳态"},
    {"key": "SPECIAL", "name": "特殊"},
)
MAIN_CATEGORY_KEYS: tuple[str, ...] = tuple(cat["key"] for cat in MAIN_CATEGORIES)
MAIN_CATEGORY_NAMES: dict[str, str] = {cat["key"]: cat["name"] for cat in MAIN_CATEGORIES}

# ``TEST`` 级（随意测试 / 静态标定）并入「基础」。
CATEGORY_BY_LEVEL: dict[str, str] = {
    "TEST": "BASELINE",
    "BASELINE": "BASELINE",
    "STEADY_STATE": "STEADY_STATE",
    "TRANSIENT": "TRANSIENT",
    "SPECIAL": "SPECIAL",
}


def _base_code(code: str) -> str:
    """把完整工况码还原为基础码（去掉 ``_NOEXO`` / ``_EXO`` 后缀）。"""
    if code.endswith("_NOEXO"):
        return code[: -len("_NOEXO")]
    if code.endswith("_EXO"):
        return code[: -len("_EXO")]
    return code


def _base_name(name: str) -> str:
    """把工况中文名还原为基础名（去掉「（不穿戴）/（穿戴）」后缀）。"""
    for suffix in ("（不穿戴）", "（穿戴）"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


@lru_cache(maxsize=1)
def base_catalog() -> dict[str, dict[str, Any]]:
    """从默认协议推导基础工况目录。

    返回 ``{基础码: {"name", "category", "has_noexo", "has_exo"}}``，
    顺序沿用协议定义顺序。
    """

    catalog: dict[str, dict[str, Any]] = {}
    for condition in load_default_protocol().conditions:
        code = condition.condition_code
        base = _base_code(code)
        level = str(condition.condition_level or "").strip().upper()
        exo = (condition.parameters or {}).get("exo")
        entry = catalog.get(base)
        if entry is None:
            entry = {
                "name": _base_name(condition.condition_name),
                "category": CATEGORY_BY_LEVEL.get(level, level),
                "has_noexo": False,
                "has_exo": False,
            }
            catalog[base] = entry
        if exo is False:
            entry["has_noexo"] = True
        elif exo is True:
            entry["has_exo"] = True
    return catalog


def expand_category_details(details: Any) -> list[str]:
    """把某主工况的详细工况展开为协议完整工况码列表。

    规则：
    - 成对工况（有 NOEXO+EXO）：``wear=false`` → ``[CODE_NOEXO]``，
      ``wear=true`` → ``[CODE_NOEXO, CODE_EXO]``；
    - 仅穿戴工况（如 ``SQUAT_STD``）：恒 ``[CODE_EXO]``；
    - 无穿戴工况（随意测试 / 静态标定）：恒 ``[CODE]``。
    """

    catalog = base_catalog()
    result: list[str] = []
    for detail in details if isinstance(details, (list, tuple)) else ():
        if not isinstance(detail, dict):
            continue
        base = str(detail.get("code") or "").strip().upper()
        info = catalog.get(base)
        if info is None:
            continue
        wear = bool(detail.get("wear"))
        if info["has_noexo"]:
            result.append(f"{base}_NOEXO")
        if info["has_exo"] and (wear or not info["has_noexo"]):
            result.append(f"{base}_EXO")
        if not info["has_noexo"] and not info["has_exo"]:
            result.append(base)
    return result


# ── 默认种子（基础码）────────────────────────────────────────────────

def _slope_base_matrix() -> tuple[str, ...]:
    """爬坡矩阵基础码：1.5°/2.5°/3.5°/5° × 0.6/0.8/1.0/1.2 m/s。"""
    return tuple(
        f"DWALK_{slope}D_{speed}"
        for slope in ("1P5", "2P5", "3P5", "5")
        for speed in ("0P6", "0P8", "1P0", "1P2")
    )


def _terrain_matrix() -> tuple[str, ...]:
    """二期详细地形矩阵基础码：平地 + 0.5°~9.5°坡 × 0.6/0.8/1.0/1.2 m/s。"""
    level = tuple(f"DWALK_{speed}" for speed in ("0P6", "0P8", "1P0", "1P2"))
    slopes = tuple(
        f"DWALK_{slope}D_{speed}"
        for slope in (
            "0P5",
            "1P5",
            "2P5",
            "3P5",
            "4P5",
            "5",
            "5P5",
            "6P5",
            "7P5",
            "8P5",
            "9P5",
        )
        for speed in ("0P6", "0P8", "1P0", "1P2")
    )
    return level + slopes


# 第一期 = 新增工况（平地走四档 + 爬坡矩阵 + 匀加减速 + 特殊工况）。
PHASE_1_BASELINE: tuple[str, ...] = ("FREE_TEST", "STATIC_CALIB")
PHASE_1_STEADY: tuple[str, ...] = (
    "WALK_0P6",
    "WALK_1P0",
    "DWALK_0P8",
    "DWALK_1P2",
) + _slope_base_matrix()
PHASE_1_TRANSIENT: tuple[str, ...] = ("ACCEL_DECEL",)
PHASE_1_SPECIAL: tuple[str, ...] = (
    "SQUAT",
    "SIT_RISE",
    "STANCE_PERTURB",
    "SLOW_SQUAT",
    "LIFT_LOAD",
    "BACKWARD_FALL",
)

# 第二期 = 原有标准工况 + 详细地形矩阵 + 特殊工况。
PHASE_2_BASELINE: tuple[str, ...] = (
    "FREE_TEST",
    "STATIC_CALIB",
    "START_STAND_30S",
    "START_WALK_1P0_30S",
    "END_STAND_30S",
    "END_WALK_1P0_30S",
)
PHASE_2_STEADY: tuple[str, ...] = (
    "WALK_0P6",
    "WALK_1P0",
    "WALK_1P4",
    "WALK_2P5D_0P6",
    "WALK_2P5D_1P0",
    "WALK_5D_0P6",
    "WALK_5D_1P0",
) + _terrain_matrix()
PHASE_2_TRANSIENT: tuple[str, ...] = (
    "SQUAT_STD",
    "SPEED_RAMP",
    "START_LEFT",
    "START_RIGHT",
    "STOP_LEFT",
    "STOP_RIGHT",
)
PHASE_2_SPECIAL: tuple[str, ...] = (
    "CONST_ACCEL",
    "SLOPE_RAMP",
    "INTENT_FORWARD",
    "INTENT_BACKWARD",
    "INTENT_LATERAL",
    "INTENT_TEST",
    "FREE_ACTIVE_FORWARD",
    "FREE_ACTIVE_LATERAL",
    "FREE_ACTIVE_BACKWARD",
    "FREE_PASSIVE_FORWARD",
    "FREE_PASSIVE_LATERAL",
    "FREE_PASSIVE_BACKWARD",
    "FIXED_ACTIVE_FORWARD",
    "FIXED_ACTIVE_LATERAL",
    "FIXED_ACTIVE_BACKWARD",
    "FIXED_PASSIVE_FORWARD",
    "FIXED_PASSIVE_LATERAL",
    "FIXED_PASSIVE_BACKWARD",
    "ACTIVE_CONSTRAINED_HIP_FLEX",
    "ACTIVE_CONSTRAINED_HIP_EXT",
    "ACTIVE_CONSTRAINED_KNEE_FLEX",
    "ACTIVE_CONSTRAINED_KNEE_EXT",
    "ACTIVE_CONSTRAINED_ABDUCT",
    "ACTIVE_FREE_SLIGHT_HIP_FLEX",
    "ACTIVE_FREE_SLIGHT_HIP_EXT",
    "ACTIVE_FREE_SLIGHT_KNEE_FLEX",
    "ACTIVE_FREE_SLIGHT_KNEE_EXT",
    "ACTIVE_FREE_SLIGHT_ABDUCT",
    "ACTIVE_FREE_LARGE_HIP_FLEX",
    "ACTIVE_FREE_LARGE_HIP_EXT",
    "ACTIVE_FREE_LARGE_KNEE_FLEX",
    "ACTIVE_FREE_LARGE_KNEE_EXT",
    "ACTIVE_FREE_LARGE_ABDUCT",
    "PASSIVE_CONSTRAINED_HIP_FLEX",
    "PASSIVE_CONSTRAINED_HIP_EXT",
    "PASSIVE_CONSTRAINED_KNEE_FLEX",
    "PASSIVE_CONSTRAINED_KNEE_EXT",
    "PASSIVE_CONSTRAINED_ABDUCT",
    "PASSIVE_FREE_SLIGHT_HIP_FLEX",
    "PASSIVE_FREE_SLIGHT_HIP_EXT",
    "PASSIVE_FREE_SLIGHT_KNEE_FLEX",
    "PASSIVE_FREE_SLIGHT_KNEE_EXT",
    "PASSIVE_FREE_SLIGHT_ABDUCT",
    "PASSIVE_FREE_LARGE_HIP_FLEX",
    "PASSIVE_FREE_LARGE_HIP_EXT",
    "PASSIVE_FREE_LARGE_KNEE_FLEX",
    "PASSIVE_FREE_LARGE_KNEE_EXT",
    "PASSIVE_FREE_LARGE_ABDUCT",
)


# 第三期 = 站姿/坐姿 关节工况（主动/被动 × 约束/轻微/明显位移 × 关节）。
PHASE_3_SPECIAL: tuple[str, ...] = (
    "STAND_ACTIVE_CONSTRAINED_HIP_FLEX",
    "STAND_ACTIVE_CONSTRAINED_HIP_EXT",
    "STAND_ACTIVE_CONSTRAINED_KNEE_FLEX",
    "STAND_ACTIVE_CONSTRAINED_KNEE_EXT",
    "STAND_ACTIVE_CONSTRAINED_ABDUCT",
    "STAND_ACTIVE_FREE_SLIGHT_HIP_FLEX",
    "STAND_ACTIVE_FREE_SLIGHT_HIP_EXT",
    "STAND_ACTIVE_FREE_SLIGHT_KNEE_FLEX",
    "STAND_ACTIVE_FREE_SLIGHT_KNEE_EXT",
    "STAND_ACTIVE_FREE_SLIGHT_ABDUCT",
    "STAND_ACTIVE_FREE_LARGE_HIP_FLEX",
    "STAND_ACTIVE_FREE_LARGE_HIP_EXT",
    "STAND_ACTIVE_FREE_LARGE_KNEE_FLEX",
    "STAND_ACTIVE_FREE_LARGE_KNEE_EXT",
    "STAND_ACTIVE_FREE_LARGE_ABDUCT",
    "STAND_PASSIVE_CONSTRAINED_HIP_FLEX",
    "STAND_PASSIVE_CONSTRAINED_HIP_EXT",
    "STAND_PASSIVE_CONSTRAINED_KNEE_FLEX",
    "STAND_PASSIVE_CONSTRAINED_KNEE_EXT",
    "STAND_PASSIVE_CONSTRAINED_ABDUCT",
    "STAND_PASSIVE_FREE_SLIGHT_HIP_FLEX",
    "STAND_PASSIVE_FREE_SLIGHT_HIP_EXT",
    "STAND_PASSIVE_FREE_SLIGHT_KNEE_FLEX",
    "STAND_PASSIVE_FREE_SLIGHT_KNEE_EXT",
    "STAND_PASSIVE_FREE_SLIGHT_ABDUCT",
    "STAND_PASSIVE_FREE_LARGE_HIP_FLEX",
    "STAND_PASSIVE_FREE_LARGE_HIP_EXT",
    "STAND_PASSIVE_FREE_LARGE_KNEE_FLEX",
    "STAND_PASSIVE_FREE_LARGE_KNEE_EXT",
    "STAND_PASSIVE_FREE_LARGE_ABDUCT",
    "SIT_ACTIVE_CONSTRAINED_HIP_FLEX",
    "SIT_ACTIVE_CONSTRAINED_KNEE_FLEX",
    "SIT_ACTIVE_CONSTRAINED_KNEE_EXT",
    "SIT_ACTIVE_CONSTRAINED_ABDUCT",
    "SIT_ACTIVE_FREE_SLIGHT_HIP_FLEX",
    "SIT_ACTIVE_FREE_SLIGHT_KNEE_FLEX",
    "SIT_ACTIVE_FREE_SLIGHT_KNEE_EXT",
    "SIT_ACTIVE_FREE_SLIGHT_ABDUCT",
    "SIT_ACTIVE_FREE_LARGE_HIP_FLEX",
    "SIT_ACTIVE_FREE_LARGE_KNEE_FLEX",
    "SIT_ACTIVE_FREE_LARGE_KNEE_EXT",
    "SIT_ACTIVE_FREE_LARGE_ABDUCT",
    "SIT_PASSIVE_CONSTRAINED_HIP_FLEX",
    "SIT_PASSIVE_CONSTRAINED_KNEE_FLEX",
    "SIT_PASSIVE_CONSTRAINED_KNEE_EXT",
    "SIT_PASSIVE_CONSTRAINED_ABDUCT",
    "SIT_PASSIVE_FREE_SLIGHT_HIP_FLEX",
    "SIT_PASSIVE_FREE_SLIGHT_KNEE_FLEX",
    "SIT_PASSIVE_FREE_SLIGHT_KNEE_EXT",
    "SIT_PASSIVE_FREE_SLIGHT_ABDUCT",
    "SIT_PASSIVE_FREE_LARGE_HIP_FLEX",
    "SIT_PASSIVE_FREE_LARGE_KNEE_FLEX",
    "SIT_PASSIVE_FREE_LARGE_KNEE_EXT",
    "SIT_PASSIVE_FREE_LARGE_ABDUCT",
)


def _seed_detail(code: str, catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """种子详细工况：无穿戴工况不带 wear，其余默认「增加穿戴」。"""
    info = catalog[code]
    if info["has_noexo"] or info["has_exo"]:
        return {"code": code, "wear": True}
    return {"code": code}


def _make_phase(
    name: str,
    baseline: tuple[str, ...],
    steady: tuple[str, ...],
    transient: tuple[str, ...],
    special: tuple[str, ...],
) -> dict[str, Any]:
    catalog = base_catalog()
    return {
        "name": name,
        "categories": {
            "BASELINE": {
                "name": "基础",
                "details": [_seed_detail(code, catalog) for code in baseline],
            },
            "STEADY_STATE": {
                "name": "稳态",
                "details": [_seed_detail(code, catalog) for code in steady],
            },
            "TRANSIENT": {
                "name": "非稳态",
                "details": [_seed_detail(code, catalog) for code in transient],
            },
            "SPECIAL": {
                "name": "特殊",
                "details": [_seed_detail(code, catalog) for code in special],
            },
        },
    }


def default_phase_config() -> dict[str, Any]:
    """首次无配置时的默认期次：第一期=新增工况，第二期=标准工况+详细地形矩阵，第三期=站姿/坐姿关节工况。"""
    return {
        "schema_version": PHASE_CONFIG_SCHEMA_VERSION,
        "phases": [
            _make_phase(
                "第一期",
                PHASE_1_BASELINE,
                PHASE_1_STEADY,
                PHASE_1_TRANSIENT,
                PHASE_1_SPECIAL,
            ),
            _make_phase(
                "第二期",
                PHASE_2_BASELINE,
                PHASE_2_STEADY,
                PHASE_2_TRANSIENT,
                PHASE_2_SPECIAL,
            ),
            _make_phase(
                "第三期",
                (),
                (),
                (),
                PHASE_3_SPECIAL,
            ),
        ],
    }


# ── 规范化 ───────────────────────────────────────────────────────────


def _unique_details(
    details_raw: Any, catalog: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """去重、丢弃未知基础码，规范 wear 语义，保留输入顺序。"""
    result: list[dict[str, Any]] = []
    if not isinstance(details_raw, (list, tuple)):
        return result
    seen: set[str] = set()
    for detail in details_raw:
        if not isinstance(detail, dict):
            continue
        code = str(detail.get("code") or "").strip().upper()
        info = catalog.get(code)
        if info is None or code in seen:
            continue
        seen.add(code)
        if info["has_noexo"]:
            normalized = {"code": code, "wear": bool(detail.get("wear"))}
        elif info["has_exo"]:
            normalized = {"code": code, "wear": True}
        else:
            normalized = {"code": code}
        result.append(normalized)
    return result


def _details_from_codes(
    codes: Any, catalog: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """兼容旧版 ``codes`` 列表形状：按 level 归入主工况并合并 NOEXO/EXO。"""
    categories: dict[str, dict[str, Any]] = {
        key: {"name": MAIN_CATEGORY_NAMES[key], "details": []}
        for key in MAIN_CATEGORY_KEYS
    }
    if not isinstance(codes, (list, tuple)):
        return categories
    order: list[tuple[str, str]] = []
    has_exo_seen: dict[tuple[str, str], bool] = {}
    for raw_code in codes:
        code = str(raw_code).strip().upper()
        base = _base_code(code)
        info = catalog.get(base)
        if info is None:
            continue
        key = (info["category"], base)
        if key not in has_exo_seen:
            order.append(key)
            has_exo_seen[key] = False
        if code.endswith("_EXO"):
            has_exo_seen[key] = True

    for category, base in order:
        info = catalog[base]
        detail: dict[str, Any] = {"code": base}
        if info["has_noexo"]:
            detail["wear"] = has_exo_seen[(category, base)]
        elif info["has_exo"]:
            detail["wear"] = True
        categories[category]["details"].append(detail)
    return categories


def normalize_phase_config(raw: Any) -> dict[str, Any]:
    """把外部 / 持久化输入规范成三层期次配置；非法或为空时回退默认种子。"""
    default = default_phase_config()
    if not isinstance(raw, dict):
        return default
    phases_raw = raw.get("phases")
    if not isinstance(phases_raw, list) or not phases_raw:
        return default

    catalog = base_catalog()
    phases: list[dict[str, Any]] = []
    for item in phases_raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        categories_raw = item.get("categories")
        if isinstance(categories_raw, dict):
            categories: dict[str, dict[str, Any]] = {}
            for key, cat in categories_raw.items():
                key = str(key).strip().upper()
                if key not in MAIN_CATEGORY_KEYS or not isinstance(cat, dict):
                    continue
                categories[key] = {
                    "name": MAIN_CATEGORY_NAMES[key],
                    "details": _unique_details(cat.get("details"), catalog),
                }
            for key in MAIN_CATEGORY_KEYS:
                categories.setdefault(
                    key, {"name": MAIN_CATEGORY_NAMES[key], "details": []}
                )
        else:
            categories = _details_from_codes(item.get("codes"), catalog)
        if not categories:
            continue
        phases.append({"name": name, "categories": categories})

    if not phases:
        return default
    return {"schema_version": PHASE_CONFIG_SCHEMA_VERSION, "phases": phases}


__all__ = [
    "CATEGORY_BY_LEVEL",
    "MAIN_CATEGORIES",
    "MAIN_CATEGORY_KEYS",
    "MAIN_CATEGORY_NAMES",
    "PHASE_1_BASELINE",
    "PHASE_1_SPECIAL",
    "PHASE_1_STEADY",
    "PHASE_1_TRANSIENT",
    "PHASE_2_BASELINE",
    "PHASE_2_SPECIAL",
    "PHASE_3_SPECIAL",
    "PHASE_2_STEADY",
    "PHASE_2_TRANSIENT",
    "PHASE_CONFIG_SCHEMA_VERSION",
    "base_catalog",
    "default_phase_config",
    "expand_category_details",
    "normalize_phase_config",
]
