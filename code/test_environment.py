"""Check the STATS19 paper project's Python environment."""

from __future__ import annotations

import importlib
import platform
import sys
from datetime import datetime
from pathlib import Path


PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "lightgbm": "lightgbm",
    "statsmodels": "statsmodels",
    "shap": "shap",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "openpyxl": "openpyxl",
}


def main() -> int:
    project_dir = Path(__file__).resolve().parents[1]
    output_file = project_dir / "logs" / "environment_info.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "STATS19 paper project environment",
        f"Checked at: {datetime.now().isoformat(timespec='seconds')}",
        f"Python: {sys.version.split()[0]}",
        f"Executable: {sys.executable}",
        f"Platform: {platform.platform()}",
        "",
        "Packages:",
    ]

    failed = []
    for display_name, import_name in PACKAGES.items():
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "unknown")
            line = f"{display_name}: {version}"
            print(line)
            lines.append(line)
        except Exception as exc:  # Keep the error visible for diagnosis.
            line = f"{display_name}: FAILED ({exc})"
            print(line)
            lines.append(line)
            failed.append(display_name)

    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        del OrderedModel
        print("OrderedModel: OK")
        lines.append("OrderedModel: OK")
    except Exception as exc:
        line = f"OrderedModel: FAILED ({exc})"
        print(line)
        lines.append(line)
        failed.append("OrderedModel")

    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nEnvironment report: {output_file}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
