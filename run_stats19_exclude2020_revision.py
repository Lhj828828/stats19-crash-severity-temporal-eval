"""Run the isolated 15-feature exclusion-2020 sensitivity analysis."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from stats19_exclude2020_revision import main

if __name__ == "__main__":
    main()
