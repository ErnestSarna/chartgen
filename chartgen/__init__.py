"""Generate Clone Hero charts from audio.

Importing this package puts the vendored trees on sys.path: audio2chart uses
absolute imports (`from inference.engine import ...`) and EasyChartGenerator is a
loose script, so neither is installable as a normal dependency.
"""
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parent.parent / "vendor"
for _path in (VENDOR / "audio2chart", VENDOR / "EasyChartGenerator" / "EasyChartGenerator"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
