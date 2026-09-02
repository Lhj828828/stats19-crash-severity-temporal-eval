"""Cross-platform entry point for the public CAS replication."""

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parent / "code" / "cas_public_reproduction.py"),
    run_name="__main__",
)
