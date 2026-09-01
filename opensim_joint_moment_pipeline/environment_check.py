"""环境依赖检测：确认哪些 stage 能跑、哪些 BLOCKING。"""

from __future__ import annotations


def main() -> int:
    deps = {
        "numpy": "数值",
        "scipy": "滤波/插值",
        "pandas": "表格",
        "ezc3d": "C3D 读取",
        "yaml": "配置",
        "matplotlib": "QC 绘图",
        "opensim": "Scale/IK/ID（可选，缺则 BLOCKING）",
    }
    print("=" * 52)
    print("Environment check")
    print("=" * 52)
    missing_hard = []
    for mod, why in deps.items():
        try:
            __import__(mod)
            status = "OK"
        except ImportError:
            status = "MISSING"
            if mod != "opensim":
                missing_hard.append(mod)
        print(f"  [{status:<7}] {mod:<12} {why}")

    print()
    if missing_hard:
        print(f"硬缺失（上游无法运行）: {', '.join(missing_hard)}")
        return 1
    print("上游依赖齐全。")
    print("OpenSim（Scale/IK/ID）装在独立 `opensim` 环境（numpy 2.x），勿在此环境安装：")
    print("  conda create -n opensim -c opensim-org -c conda-forge opensim=4.6 python=3.11 -y")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
