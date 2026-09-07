"""Run the separately archived, post-review random-reference correction."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from random_reference_revision import main

if __name__ == "__main__":
    main()
