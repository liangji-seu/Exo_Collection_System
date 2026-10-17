"""Launcher for the ultrasound region annotator."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from us_region_annotator import main


if __name__ == "__main__":
    raise SystemExit(main())
