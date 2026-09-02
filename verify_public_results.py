"""Cross-platform entry point for compact public-result verification."""

from pathlib import Path
import runpy
import sys


code_dir = Path(__file__).resolve().parent / "code"
sys.path.insert(0, str(code_dir))
runpy.run_path(
    str(code_dir / "public_result_verifier.py"),
    run_name="__main__",
)
