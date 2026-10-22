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
    assert len(phases) == 6
    p1, p2, p3, p4, p5, p6 = phases
    assert p1["name"] == "第一期"
    assert p2["name"] == "第二期"
    assert p3["name"] == "第三期"
    assert p4["name"] == "第四期"
    assert p5["name"] == "第五期"
    assert p6["name"] == "第六期"

    # 每期都固定 4 类主工况。
    for phase in phases:
        assert list(phase["categories"].keys()) == list(MAIN_CATEGORY_KEYS)

    # 展开为协议完整码后：第一期 56 码，第二期 222 码，第三期 108 码，第四期 62 码，第五期 16 码，第六期 38 码（只不穿戴）。
    assert len(_expanded_codes(p1)) == 56
    assert len(_expanded_codes(p2)) == 222
    assert len(_expanded_codes(p3)) == 108
    assert len(_expanded_codes(p4)) == 62
    assert len(_expanded_codes(p5)) == 16
    assert len(_expanded_codes(p6)) == 38
    # 第六期只做不穿戴：展开后不应出现 _EXO。
    assert all(code.endswith("_NOEXO") for code in _expanded_codes(p6))

    protocol_codes = {c.condition_code for c in load_default_protocol().conditions}
    union = _expanded_codes(p1) | _expanded_codes(p2) | _expanded_codes(p3) | _expanded_codes(p4) | _expanded_codes(p5) | _expanded_codes(p6)
    assert union == protocol_codes
    assert len(union) == 441


def test_load_details_ordered_by_terrain_speed_load() -> None:
    # 负重类工况统一按 地形 → 速度 → 负重 排序；第四期与第六期共用同一批基础码，
    # 因此两期的详细工况顺序一致（同地形、同速度下按负重升序）。
    params_by_code = {
        condition.condition_code: (condition.parameters or {})
        for condition in load_default_protocol().conditions
    }

    def _sort_key(code: str) -> tuple[float, float, float]:
        params = params_by_code[f"{code}_NOEXO"]
        return (
            float(params.get("slope_deg") or 0),
            float(params.get("speed_mps") or 0),
            float(params["load_kg"]),
        )

    phases = default_phase_config()["phases"]
    for index in (3, 5):
        phase = phases[index]
        for category in ("STEADY_STATE", "TRANSIENT"):
            codes = [detail["code"] for detail in phase["categories"][category]["details"]]
            keys = [_sort_key(code) for code in codes]
            assert keys == sorted(keys), (phase["name"], category)

    # 平地 0.6 m/s 先走完该期全部负重档，再进 1.0 m/s。
    p4_steady = [d["code"] for d in phases[3]["categories"]["STEADY_STATE"]["details"]]
    assert p4_steady[:4] == [
        "LWALK_0P6_0KG",
        "LWALK_0P6_2P5KG",
        "LWALK_0P6_5KG",
        "LWALK_0P6_10KG",
    ]
    p6_steady = [d["code"] for d in phases[5]["categories"]["STEADY_STATE"]["details"]]
    assert p6_steady[:5] == [
        "LWALK_0P6_0KG",
        "LWALK_0P6_5KG",
        "LWALK_0P6_10KG",
        "LWALK_0P6_15KG",
        "LWALK_0P6_20KG",
    ]
    assert p6_steady[-1] == "LWALK_7P5D_0P6_20KG"


def test_level_walking_four_speeds_shared_across_phases() -> None:
    cfg = default_phase_config()
    p1, p2, *_ = cfg["phases"]

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
