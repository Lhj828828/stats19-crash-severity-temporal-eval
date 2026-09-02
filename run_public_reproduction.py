"""Cross-platform entry point for the public STATS19 reconstruction."""

from pathlib import Path
import runpy
import sys


code_dir = Path(__file__).resolve().parent / "code"
sys.path.insert(0, str(code_dir))
runpy.run_path(
    str(code_dir / "public_reproduction.py"),
    run_name="__main__",
)
