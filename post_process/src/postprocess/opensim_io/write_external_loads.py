"""ExternalLoads XML 写出。

每个 ExternalForce 把一个力台的数据施加到对应 foot body 上。foot body 名字
（calcn_r / calcn_l 等）**必须从 scaled model 自动解析**，这里只负责按给定的
body 名生成 XML，不硬编码。
"""

from __future__ import annotations

from pathlib import Path


def write_external_loads(
    path: str | Path,
    feet: list[dict],      # [{"identifier": 1, "body": "calcn_r", "datafile": "grf.mot"}]
) -> None:
    if not feet:
        raise ValueError("feet 不能为空")

    objects = []
    for f in feet:
        ident = f["identifier"]
        body = f["body"]
        datafile = f.get("datafile", "grf.mot")
        objects.append(
            "    <ExternalForce name=\"ExternalForce{ident}\">\n"
            f"      <applied_to_body>{body}</applied_to_body>\n"
            "      <force_expressed_in_body>ground</force_expressed_in_body>\n"
            "      <point_expressed_in_body>ground</point_expressed_in_body>\n"
            f"      <force_identifier>{ident}</force_identifier>\n"
            f"      <point_identifier>{ident}</point_identifier>\n"
            f"      <torque_identifier>{ident}</torque_identifier>\n"
            f"      <datafile>{datafile}</datafile>\n"
            "    </ExternalForce>"
        )

    datafile = feet[0].get("datafile", "grf.mot")
    xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<OpenSimDocument Version=\"40000\">\n"
        "  <ExternalLoads name=\"external_loads\">\n"
        "    <objects>\n"
        + "\n".join(objects)
        + "\n    </objects>\n"
        f"    <datafile>{datafile}</datafile>\n"
        "  </ExternalLoads>\n"
        "</OpenSimDocument>\n"
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(xml, encoding="utf-8")


__all__ = ["write_external_loads"]
