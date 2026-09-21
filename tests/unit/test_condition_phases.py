from __future__ import annotations

from exo_collection.domain.condition_phases import (
    MAIN_CATEGORY_KEYS,
    base_catalog,
    default_phase_config,
    expand_category_details,
    normalize_phase_config,
)
from exo_collection.protocols import load_default_protocol


def _expanded_codes(phase: dict) -> set[str]:
    codes: set[str] = set()
    for category in phase["categories"].values():
        codes.update(expand_category_details(category["details"]))
    return codes


def test_default_phase_config_partitions_protocol() -> None:
    cfg = default_phase_config()
    phases = cfg["phases"]
    assert len(phases) == 2
    p1, p2 = phases
    assert p1["name"] == "第一期"
    assert p2["name"] == "第二期"

    # 每期都固定 4 类主工况。
    for phase in phases:
        assert list(phase["categories"].keys()) == list(MAIN_CATEGORY_KEYS)

    # 展开为协议完整码后：第一期 56 码，第二期 222 码。
    assert len(_expanded_codes(p1)) == 56
    assert len(_expanded_codes(p2)) == 222

    protocol_codes = {c.condition_code for c in load_default_protocol().conditions}
    union = _expanded_codes(p1) | _expanded_codes(p2)
    assert union == protocol_codes
    assert len(union) == 236


def test_level_walking_four_speeds_shared_across_phases() -> None:
    cfg = default_phase_config()
    p1, p2 = cfg["phases"]

    # 平地走四档：0.6/1.0 用 WALK，0.8/1.2 用 DWALK，均归入第一期稳态。
    steady = p1["categories"]["STEADY_STATE"]
    steady_codes = set(expand_category_details(steady["details"]))
    for code in (
        "WALK_0P6_NOEXO",
        "WALK_0P6_EXO",
        "WALK_1P0_NOEXO",
        "WALK_1P0_EXO",
        "DWALK_0P8_NOEXO",
        "DWALK_0P8_EXO",
        "DWALK_1P2_NOEXO",
        "DWALK_1P2_EXO",
    ):
        assert code in steady_codes

    # 0.6/1.0 同时属于第二期稳态（跨期共享）。
    p2_steady = set(expand_category_details(p2["categories"]["STEADY_STATE"]["details"]))
    assert "WALK_0P6_NOEXO" in p2_steady
    assert "WALK_1P0_NOEXO" in p2_steady

    # 第一期稳态 DWALK 类工况：0.8/1.2 平地走 4 条 + 爬坡矩阵 32 条 = 36 条。
    dwalk_codes = [code for code in steady_codes if code.startswith("DWALK_")]
    assert len(dwalk_codes) == 36
    # 爬坡矩阵 32 条：DWALK code 中含 "D_" 坡度标识（0.8/1.2 平地走不含）。
    slope_codes = [code for code in dwalk_codes if "D_" in code]
    assert len(slope_codes) == 32


def test_expand_category_details_applies_wear_rules() -> None:
    # 成对工况（有 NOEXO+EXO）：wear=false → 仅不穿戴；wear=true → 不穿戴 + 穿戴。
    assert expand_category_details([{"code": "WALK_1P0", "wear": False}]) == [
        "WALK_1P0_NOEXO"
    ]
    assert expand_category_details([{"code": "WALK_1P0", "wear": True}]) == [
        "WALK_1P0_NOEXO",
        "WALK_1P0_EXO",
    ]
    # 仅有穿戴的工况：恒穿戴，忽略 wear。
    assert expand_category_details([{"code": "SQUAT_STD", "wear": False}]) == [
        "SQUAT_STD_EXO"
    ]
    # 无穿戴工况（随意测试 / 静态标定）：恒原码。
    assert expand_category_details([{"code": "FREE_TEST"}]) == ["FREE_TEST"]
    # 未知基础码被丢弃。
    assert expand_category_details([{"code": "NOT_A_CODE"}]) == []


def test_base_catalog_derives_base_codes() -> None:
    catalog = base_catalog()
    # 成对工况：不穿戴 + 穿戴。
    assert catalog["WALK_1P0"]["has_noexo"] is True
    assert catalog["WALK_1P0"]["has_exo"] is True
    assert catalog["WALK_1P0"]["category"] == "STEADY_STATE"
    # TEST 级（随意测试 / 静态标定）并入基础，且无穿戴标志。
    assert catalog["FREE_TEST"]["category"] == "BASELINE"
    assert catalog["FREE_TEST"]["has_noexo"] is False
    assert catalog["FREE_TEST"]["has_exo"] is False
    # 仅有穿戴的工况。
    assert catalog["SQUAT_STD"]["has_noexo"] is False
    assert catalog["SQUAT_STD"]["has_exo"] is True


def test_normalize_drops_unknown_codes_and_dedupes() -> None:
    raw = {
        "phases": [
            {
                "name": "X",
                "categories": {
                    "BASELINE": {
                        "name": "基础",
                        "details": [
                            {"code": "FREE_TEST"},
                            {"code": "FREE_TEST"},
                            {"code": "NOT_A_CODE"},
                        ],
                    },
                },
            },
            {"name": "", "categories": {}},
        ]
    }
    cfg = normalize_phase_config(raw)
    phases = cfg["phases"]
    assert len(phases) == 1  # 空名期次被丢弃
    assert phases[0]["name"] == "X"
    assert phases[0]["categories"]["BASELINE"]["details"] == [
        {"code": "FREE_TEST"}
    ]


def test_normalize_merges_legacy_codes_shape() -> None:
    raw = {
        "phases": [
            {
                "name": "X",
                "codes": ["FREE_TEST", "WALK_0P6_NOEXO", "WALK_0P6_EXO"],
            },
        ]
    }
    cfg = normalize_phase_config(raw)
    phase = cfg["phases"][0]
    assert phase["name"] == "X"
    # 随意测试归入基础；WALK_0P6 成对出现 → 归入稳态且 wear=True。
    assert phase["categories"]["BASELINE"]["details"] == [{"code": "FREE_TEST"}]
    assert phase["categories"]["STEADY_STATE"]["details"] == [
        {"code": "WALK_0P6", "wear": True}
    ]


def test_normalize_invalid_returns_default() -> None:
    default = default_phase_config()
    assert normalize_phase_config(None)["phases"] == default["phases"]
    assert normalize_phase_config([])["phases"] == default["phases"]
    assert normalize_phase_config({"phases": []})["phases"] == default["phases"]
