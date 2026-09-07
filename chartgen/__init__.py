"""Generate Clone Hero charts from audio.

Importing this package puts the vendored EasyChartGenerator on sys.path: it is
a loose script (kept for the --reducer easygen comparison), not an installable
dependency.
"""
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parent.parent / "vendor"
for _path in (VENDOR / "EasyChartGenerator" / "EasyChartGenerator",):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
