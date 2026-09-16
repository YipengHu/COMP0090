"""Execute notebooks and script entry points without changing source files."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def main():
    os.environ["MPLBACKEND"] = "Agg"
    for path in sorted(ROOT.glob("tia_*/activity.ipynb")):
        start = time.monotonic()
        print(f"Running {path.parent.name} ...", flush=True)
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        NotebookClient(
            notebook,
            timeout=300,
            kernel_name="comp0090",
            resources={"metadata": {"path": str(path.parent)}},
        ).execute()
        print(f"  Passed ({time.monotonic() - start:.1f}s)", flush=True)
    with tempfile.TemporaryDirectory(prefix="comp0090-results-") as directory:
        for name in ("tia_02_generalisation", "tia_05_pretraining"):
            print(f"Running {name}/activity.py ...", flush=True)
            output = Path(directory) / f"{name}.json"
            subprocess.run(
                [sys.executable, str(ROOT / name / "activity.py"), "--output", str(output)],
                cwd=directory, check=True, timeout=300,
            )
            results = json.loads(output.read_text())
            assert results["seed"] == 7
            if name == "tia_02_generalisation":
                random = next(row for row in results["capacity"]
                              if row["width"] == 64 and row["labels"] == "random")
                assert random["train_acc"] > .95 and random["test_acc"] < .20
                assert results["shift"]["in_distribution"] > results["shift"]["rotated"] + .05
            else:
                scores = results["scores_by_count"]
                assert scores["5"]["pretrained-96"] > scores["5"]["random"] + .15
                assert scores["20"]["pretrained-96"] > scores["5"]["pretrained-96"]
    print("All notebooks and scripts passed, including their numerical checks.")


if __name__ == "__main__":
    main()
