"""一键启动 Exo Process（批量解算端）。

复用 Exo Calculate 的解算管线（自动同步 / 预处理 / OpenSim / 真值导出），但按
受试者批量处理：指定受试者 + 静态标定后，逐 session 自动跺脚同步、解算，并把
力矩真值 CSV 写进各 session 目录；也可按受试者/dX 导出逐 session 指标和汇总
统计的数据质量报告 PNG。
"""

from pathlib import Path
import sys


# A fresh clone can run directly from ``src`` after system dependencies are
# installed; it does not depend on an editable install or virtual environment.
SOURCE_ROOT = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from exo_collection.apps.process.main import main

if __name__ == "__main__":
    raise SystemExit(main())
