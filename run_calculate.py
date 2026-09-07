"""一键启动 Exo Calculate（标定 / 同步 / 解算 / 回放端）。

解算由 pipeline.opensim_io.prep_session 驱动：读取 Gaitway 实际坡度，
先转换原始力方向，再将力和 COP 绕零坡度标定的后沿轴旋转。
修改后需新建解算运行；历史结果不会自动改写。
"""

from pathlib import Path
import sys


# A fresh clone can run directly from ``src`` after system dependencies are
# installed; it does not depend on an editable install or virtual environment.
SOURCE_ROOT = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from exo_collection.apps.calculate.main import main

if __name__ == "__main__":
    raise SystemExit(main())
