"""Preparation-only entry point for the post-review 15-feature revision."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from stats19_feature_revision import main

if __name__ == "__main__":
    main()
