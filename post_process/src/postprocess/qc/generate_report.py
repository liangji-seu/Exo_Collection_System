"""QC 报告汇总：把各子 QC dict 组装成 qc_report.{json,md}。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_qc_report(sections: dict[str, Any], out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "qc_report.json").write_text(
        json.dumps(sections, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "qc_report.md").write_text(_to_markdown(sections), encoding="utf-8")
    return sections


def _to_markdown(sections: dict[str, Any]) -> str:
    lines = ["# QC Report", ""]
    for name, content in sections.items():
        lines.append(f"## {name}")
        lines.append("")
        if isinstance(content, dict):
            for k, v in content.items():
                lines.append(f"- **{k}**: {_fmt(v)}")
        else:
            lines.append(str(content))
        lines.append("")
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    if isinstance(v, dict):
        return ", ".join(f"{k}={_fmt(vv)}" for k, vv in v.items())
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


__all__ = ["generate_qc_report"]
