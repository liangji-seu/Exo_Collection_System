"""pytest 入口：把 src/ 加进 sys.path，使 postprocess 包可导入。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
