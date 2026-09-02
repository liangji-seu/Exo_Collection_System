"""一键执行 gaitway-3D 测力台现场自检（Type I + Type II）。

不启动正式实验；连接 gaitway TCP（默认 49500），请求 Type I + Type II，
采集约 15 秒后输出 ``gaitway_test_report.json`` 与 ``gaitway_test_plot.png``。

用法::

    python run_gaitway_test.py [--host 127.0.0.1] [--port 49500] [--seconds 15]

退出码 0 表示自检通过（Type I + Type II 均收到且校验通过），1 表示需检查。
"""

from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from exo_collection.adapters.force_plate.gaitway_test import main

if __name__ == "__main__":
    raise SystemExit(main())
