# Setup and troubleshooting

## Install Python

Use Python 3.11 for a new installation. An existing Python 3.10 or 3.12 is also suitable. The pinned NumPy and Intel Mac PyTorch packages need this Python range; installing the newest Python is not the course setup. The macOS target is **macOS 12 (Monterey) or later**; SciPy's Apple Silicon wheels require macOS 12.

- **macOS:** install the **macOS 64-bit universal2 installer** from the [Python 3.11.9 release page](https://www.python.org/downloads/release/python-3119/) (the final 3.11 release with a python.org macOS installer). Reopen Terminal and run `python3.11 --version`, then `python3.11 setup_env.py`. On Apple Silicon, use a native terminal and native Python rather than Rosetta. If you already use a package manager, its Python 3.11 or 3.12 is also suitable.
- **Ubuntu 22.04 / 24.04:** run `sudo apt update` then `sudo apt install python3 python3-venv`. Their default Python versions are 3.10 and 3.12 respectively. Then run `python3 setup_env.py` from this repository.
- **Other Linux distributions or no administrator access:** use an available Python 3.10–3.12 with its `venv`/`ensurepip` support. If your system Python is newer, [uv's Python installer](https://docs.astral.sh/uv/guides/install-python/) can install Python in your account: after installing uv, run `uv python install 3.11`, then `uv run --python 3.11 --no-project python setup_env.py`. Normal study sessions still use `source .venv/bin/activate`.

Keep your operating system's Python in place. The setup script creates a separate `.venv` in this repository and does not change the older `comp0090/` environment if you have one.

## What gets installed

The course pins are in [requirements.txt](../requirements.txt). The numerical stack is NumPy **1.26.4**, SciPy **1.13.1**, Matplotlib **3.9.4** and scikit-learn **1.6.1**. Jupyter Notebook is **7.5.7**. These pins fix the packages used directly by the activities and key numerical dependencies; this is not a lock of every transitive Jupyter dependency.

| Platform | PyTorch | Installation source |
|---|---|---|
| macOS, Apple Silicon (arm64) | 2.8.0 | PyPI |
| macOS, Intel (x86_64) | 2.2.2 | PyPI |
| Linux, x86_64 or aarch64 | 2.8.0 CPU build | Official PyTorch CPU wheel index |

PyTorch 2.2.x was the [last release series with official Intel macOS builds](https://pytorch.org/blog/pytorch2-2/). NumPy stays at 1.26.4 for compatibility with that build. Modern Linux wheels require a glibc-based distribution; Ubuntu 22.04/24.04 are suitable targets. Alpine/musl and 32-bit systems are outside this setup. See the [official PyTorch installation commands](https://pytorch.org/get-started/previous-versions/) for platform details.

The default exercises use CPU tensors even if your computer has a GPU. Optional extensions using torchvision or downloaded models are outside the tested core environment; do the core activity first.

## Check your installation

Setup runs this automatically. To repeat it after activation:

```bash
python scripts/check_environment.py
python -m pip check
```

The check prints the interpreter and versions, then exercises PyTorch gradients, conversion to NumPy, the packaged dataset and plotting. Success ends with `CPU, autograd, NumPy conversion, packaged dataset and plotting: OK`.

To check the complete activities, including their numerical assertions:

```bash
python scripts/validate_tutorials.py
```

This runs each notebook in a fresh kernel and checks both script commands. It does not write execution results back into the notebooks. Allow several minutes on a slower laptop. Small experiments use one CPU thread to avoid the overhead of large thread pools.

### Verification record

See [validation.md](validation.md) for the configurations actually exercised and their results. The [CI workflow](../.github/workflows/tutorials.yml) repeats installation and execution on Linux and both Mac architectures with Python 3.10–3.12. A workflow definition is not evidence of a passing run: consult its results before claiming macOS runtime validation.

## Common problems

| Symptom | What to do |
|---|---|
| `python3: command not found`, or Python is outside 3.10–3.12 | Install a compatible Python above. Use its full command, e.g. `python3.11 setup_env.py`. |
| `No module named venv` or `ensurepip is not available` | On Ubuntu/Debian, install `python3-venv` for the system interpreter. For a separately installed version, install the matching package, e.g. `python3.11-venv`. Rename an incomplete `.venv` and rerun setup. |
| `.venv is incomplete` or uses a different Python | Close Jupyter, run `deactivate` if active, rename `.venv` (e.g. to `.venv-backup`), then rerun setup with the intended Python. Environments should be recreated after moving the repository or changing computers. |
| `No matching distribution found` | Check `python3 --version` and `python3 -c "import platform; print(platform.system(), platform.machine())"`. Use a supported Python/architecture. On Apple Silicon, `x86_64` means this Python is using Intel emulation. |
| Download fails or reports a certificate error | Check internet access. On macOS with the python.org installer, run **Install Certificates.command** in its Applications/Python folder. On a university network, use the institution's proxy/certificate instructions, then rerun setup. |
| `ModuleNotFoundError` in a notebook | Start Jupyter with `.venv` active and choose **Python (COMP0090)**. A cell containing `import sys; print(sys.executable)` should point inside this repository's `.venv`. |
| `No module named activity` in notebook 2 or 5 | Open the notebook in its original activity directory; keep its sibling `activity.py` file. Start Jupyter from the repository as instructed. |
| An edit to `activity.py` has no effect in a notebook | Restart that notebook's kernel and run all cells; Python caches imported modules. |
| Jupyter does not open a browser | Copy the local URL printed in the terminal into your browser. For a remote Linux machine, run `python -m notebook --no-browser` and use your usual SSH port forwarding. |
| A numerical assertion fails after experimentation | Restart the kernel, restore the default seed/settings and run all cells. The assertions describe the reference experiment; changing the seed or training budget may change its outcome. |

For a manual install, create and activate a `venv` with a supported Python. **Linux only:** first run `python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu`. On both macOS and Linux, run `python -m pip install --only-binary=:all: -r requirements.txt`, then `python -m ipykernel install --sys-prefix --name comp0090 --display-name "Python (COMP0090)"` and the environment check above. The setup script does these steps for you.
