---
name: add-condition
description: 给 Exo_Collection_System 协议新增详细工况（default.json + 期次种子 + 4 处测试计数同步）。当用户要求「新增/加几个工况、特殊工况、稳态/非稳态工况」时使用。
---

# 新增工况（add-condition）

往采集方案里加「详细工况」的完整流程。核心事实：**`config/protocols/default.json` 是唯一数据源**，其余代码（期次种子、目录推导）都从它派生；加一次工况 = 改 1 个数据文件 + 1 个种子文件 + 4 处测试计数。

## 架构速览

三层分组（只影响采集端 UI，不写入 manifest / 数据）：

```
期次（第一期 / 第二期）
  └─ 主工况（固定 4 类：基础 / 稳态 / 非稳态 / 特殊）
       └─ 详细工况（每条一个「基础码」，可选 NOEXO/EXO 成对）
```

- **完整码** = `基础码 + _NOEXO/_EXO` 后缀（如 `FREE_ACTIVE_FORWARD_NOEXO`）。协议里存的是完整码，每个基础码对应 0/1/2 条完整码。
- **基础码** = 去掉后缀（如 `FREE_ACTIVE_FORWARD`），用于期次种子里填 `details[].code`。
- `condition_level` 决定归入哪类主工况：`TEST`→基础、`BASELINE`→基础、`STEADY_STATE`→稳态、`TRANSIENT`→非稳态、`SPECIAL`→特殊（映射见 `condition_phases.py` 的 `CATEGORY_BY_LEVEL`）。
- `parameters.exo` 决定是否成对：`false`→有 `_NOEXO`，`true`→有 `_EXO`。两者都出现→成对工况。

## 步骤

### 1. 确定命名（先和用户对齐）

- 基础码：`UPPER_SNAKE_CASE`（如 `无约束-主动-向前` → `FREE_ACTIVE_FORWARD`）。
  已有约定：无约束=`FREE`、固定约束=`FIXED`、主动=`ACTIVE`、被动=`PASSIVE`、
  向前=`FORWARD`、向外=`LATERAL`、向后=`BACKWARD`、意图=`INTENT`。
- 中文名基础名不带穿戴后缀，完整码名 = `基础名（不穿戴）` / `基础名（穿戴）`。
- `condition_level`：按工况性质选 SPECIAL / STEADY_STATE / TRANSIENT / BASELINE。
- 成对 or 不成对：默认（有 NOEXO 版本时）成对；仅有穿戴（如 `SQUAT_STD`）只有 `_EXO`；随意测试/静态标定（TEST）不成对。
- 所属期次：第一期=新增工况，第二期=标准+地形矩阵+特殊。同一基础码可跨期（如 `WALK_0P6` 两期都在）。

### 2. 编辑 `config/protocols/default.json`

在对应 `conditions` 数组里加条目。成对工况加两条（NOEXO + EXO）：

```json
{
  "condition_code": "FREE_ACTIVE_FORWARD_NOEXO",
  "condition_name": "无约束-主动-向前（不穿戴）",
  "condition_level": "SPECIAL",
  "parameters": { "category": "special_free_active_forward", "exo": false }
},
{
  "condition_code": "FREE_ACTIVE_FORWARD_EXO",
  "condition_name": "无约束-主动-向前（穿戴）",
  "condition_level": "SPECIAL",
  "parameters": { "category": "special_free_active_forward", "exo": true }
}
```

- `category` 命名：`special_` + 小写基础码（用下划线，如 `special_free_active_forward`）。
- `description` / `recommended_trial_count` 可选；动作定义留空就省略 `description`。
- 协议顶层 `schema_version`/`protocol_version` 保持不动（历史几次新增均未改，`test_protocols.py` 断言 `protocol_version == "1.3.0"`，改了会挂）。

### 3. 编辑 `src/exo_collection/domain/condition_phases.py`

把基础码加进对应期次的对应主工况元组（`PHASE_1_*` / `PHASE_2_*`），例如加进第二期特殊：

```python
PHASE_2_SPECIAL: tuple[str, ...] = (
    ...,
    "FREE_ACTIVE_FORWARD", "FREE_ACTIVE_LATERAL", "FREE_ACTIVE_BACKWARD",
    ...
)
```

**只填基础码**，不要带 `_NOEXO/_EXO`。`_make_phase` 会自动 `_seed_detail`（成对工况默认 `wear=True`，即同时纳入穿戴+不穿戴）。

### 4. 同步 4 处测试计数

顺序任意，但**一处都不能漏**（漏了测试就红）：

| 文件 | 位置 / 断言 | 改法 |
|---|---|---|
| `tests/unit/test_protocols.py` | `len(protocol.conditions) == 176` + 显式 code 集合 | 总数 +N，集合里补每个新完整码 |
| `tests/unit/test_condition_phases.py` | `_expanded_codes(p2) == 162`、`union == 176` | 对应期展开数 +N、并集数 +N |
| `tests/unit/test_app_settings.py` | `expanded_counts == [56, 162]` | 第二项 +N（注释里的「第二期 162」也改） |
| `tests/unit/test_collector_ui.py` | `test_condition_combo_exposes_phase_conditions` 里对应期×主工况的 `count() == N` | 改成新的完整码数 |

**计数公式**（成对工况最常见）：每加 1 个成对基础码（NOEXO+EXO）→
协议总数 **+2**、所在期的展开数 **+2**、并集数 **+2**、collector UI 该期该主工况 combo 计数 **+2**。
仅穿戴 / 无穿戴工况 → 各 +1。

**不确定就现算**，用这段脚本打印所有该填的数（在仓库根目录、EXO 环境跑）：

```bash
/e/miniconda/envs/EXO/python -c "from exo_collection.domain.condition_phases import default_phase_config, expand_category_details; from exo_collection.protocols import load_default_protocol; cfg=default_phase_config(); [print(p['name'], sum(len(expand_category_details(c['details'])) for c in p['categories'].values()), {k: len(expand_category_details(c['details'])) for k,c in p['categories'].items()}) for p in cfg['phases']]; print('total', len(load_default_protocol().conditions))"
```

### 5. 跑测试验证

```bash
cd "E:/1_Master/2_学业/my_project/2_外骨骼课题/1_exo_数据采集方案/Exo_Collection_System"
QT_QPA_PLATFORM=offscreen PYTHONIOENCODING=utf-8 /e/miniconda/envs/EXO/python -m pytest \
  tests/unit/test_protocols.py tests/unit/test_condition_phases.py tests/unit/test_app_settings.py \
  "tests/unit/test_collector_ui.py::test_condition_combo_exposes_phase_conditions" \
  --basetemp="E:/t" -p no:cacheprovider -q
```

## 坑（务必注意）

1. **`test_collector_ui.py` 用默认种子**（空 QSettings → `default_phase_config`），所以只要动了默认种子，它的期×主工况 combo 计数就会跟着变。上次加「意图」工况时漏改这一处，测试静默红了很久才被发现——**每次改完 4 处计数都要全跑一遍上面那条命令**。
2. **不要整跑 `test_collector_ui.py` 全套**：它大量时序/子进程 mock 测试，极易挂起（>600s）。只跑上面点名的那个用例即可。
3. **用户已有 run_collector 不会自动出现新工况**：本地 QSettings 存的是旧格式（legacy `codes` 形状）期次配置，新增工况要用户在「工况设置…」对话框里「→ 纳入」，或直接改其存储的 `collector/condition_phases_json`。
4. 基础码拼错或没加进 `condition_phases.py` 的期次元组 → `expand_category_details` 会静默丢弃（`catalog.get(base)` 为 None），测试计数会不匹配，报的是计数错而不是拼写错，先查这里。

## 相关文件

- 数据源：[config/protocols/default.json](../../config/protocols/default.json)
- 种子/展开：[condition_phases.py](../../src/exo_collection/domain/condition_phases.py)
- 协议加载：[protocols.py](../../src/exo_collection/protocols.py)
