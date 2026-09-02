"""Cross-platform entry point for public data download and verification."""

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parent / "code" / "public_data.py"),
    run_name="__main__",
)
