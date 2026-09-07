"""Full analytical entry point for the isolated 15-feature STATS19 revision."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from stats19_feature_revision_full import main

if __name__ == "__main__":
    main()
