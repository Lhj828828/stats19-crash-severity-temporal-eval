"""Post hoc QWK standardization using frozen predictions, without model fitting."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from qwk_prevalence_sensitivity import main

if __name__ == "__main__":
    main()
