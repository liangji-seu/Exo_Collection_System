"""一键启动 Exo Collector（采集端）。"""

from pathlib import Path
import sys


# A fresh clone can run directly from ``src`` after system dependencies are
# installed; it does not depend on an editable install or virtual environment.
SOURCE_ROOT = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from exo_collection.apps.collector.main import main

if __name__ == "__main__":
    raise SystemExit(main())
