# Compatibility validation

Validation performed on **16 September 2026**. The local machine runs Ubuntu 22.04.5 under WSL2, x86_64. All experiments used CPU execution and the pins in [requirements.txt](../requirements.txt).

## Executed locally

| Python | PyTorch | Result |
|---|---|---|
| 3.10.21 | 2.8.0+cpu | Fresh setup, environment check, `pip check`, all six notebooks and both scripts passed. |
| 3.11.16 | 2.8.0+cpu | Fresh setup, environment check, `pip check`, all six notebooks and both scripts passed. |
| 3.12.14 | 2.8.0+cpu | Fresh setup, environment check, `pip check`, all six notebooks and both scripts passed. |
| 3.11.16 | 2.2.2+cpu | All six notebooks and both scripts passed with the older PyTorch API used on Intel Macs. This Linux run does not validate the macOS binary. |

The default notebook suite took approximately 22 seconds on the local machine; slower laptops may take several minutes. Every notebook ran in a fresh kernel from its own directory with its numerical assertions enabled. The script checks ran from outside the repository directory and verified the saved JSON results.

Additional checks on Python 3.11: Jupyter started successfully and advertised the `Python (COMP0090)` kernel using this repository's `.venv`; rerunning setup reused the environment; both scripts completed with `--seed 8 --full` and separate output files. Python 3.9 was rejected with instructions to choose a supported interpreter.

## Pending native execution

Compatible wheels were verified from PyPI metadata for every directly pinned requirement on **macOS 12+**, both **arm64** and **x86_64**, with Python **3.10, 3.11 and 3.12**. The Linux aarch64 PyTorch 2.8.0 CPU wheel was also confirmed in the official wheel index. Wheel availability does not establish runtime correctness or resolve every transitive dependency.

The [GitHub Actions workflow](../.github/workflows/tutorials.yml) installs the environment and executes the complete validation on:

- Linux x86_64: `ubuntu-24.04`.
- Linux aarch64: `ubuntu-24.04-arm`.
- Apple Silicon macOS: `macos-15`.
- Intel macOS: `macos-15-intel`.

Each platform is configured for Python 3.10, 3.11 and 3.12. These CI jobs have **not been run as part of this local validation**. macOS and Linux ARM runtime results remain pending until the workflow runs on GitHub. The workflow prints the resolved package versions to retain a record of transitive dependencies.

To repeat locally after setup:

```bash
source .venv/bin/activate
python scripts/check_environment.py
python -m pip check
python scripts/validate_tutorials.py
```
