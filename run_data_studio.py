"""一键启动 Exo Data Studio（数据管理端）。"""

from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from exo_collection.apps.data_studio.main import main

if __name__ == "__main__":
    raise SystemExit(main())
