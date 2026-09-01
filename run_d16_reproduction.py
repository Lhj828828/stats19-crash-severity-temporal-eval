"""Cross-platform entry point for the D16 isolated reproduction."""

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parent / "code" / "d16_reproduce.py"),
    run_name="__main__",
)
