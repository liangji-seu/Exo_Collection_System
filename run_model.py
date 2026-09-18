"""一键启动在线测试模块（本地模型实时推理 + Mock 扭矩输出）。"""

from pathlib import Path
import sys


# A fresh clone can run directly from ``src`` after system dependencies are
# installed; it does not depend on an editable install or virtual environment.
SOURCE_ROOT = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from exo_collection.apps.model_runtime.main import main

if __name__ == "__main__":
    raise SystemExit(main())
