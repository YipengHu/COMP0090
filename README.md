# COMP0090: Introduction to Deep Learning

Theory-in-Action (TiA) activities connect ideas to observable behaviour through small experiments. All core activities run on a CPU, use generated or packaged data, and need no dataset or model downloads after setup.

## Start here: macOS and Linux

Download or clone this repository, then open a terminal **in the COMP0090 directory** (the directory containing this README). An internet connection is needed for setup.

Use **Python 3.11** if installing Python for this course. Python 3.10 and 3.12 are also accepted; Python 3.13 and later are outside the pinned environment. Check your version:

```bash
python3 --version
```

If Python is missing or the version is outside that range, follow the short [Python installation instructions](docs/setup.md#install-python) first.

```bash
# Once: create .venv, install the course versions, and check the installation.
python3 setup_env.py

# Each study session: activate the environment, then launch Jupyter.
source .venv/bin/activate
python -m notebook
```

If your compatible interpreter is named `python3.11`, use `python3.11 setup_env.py` for the first command. Setup automatically selects the appropriate PyTorch build for Apple Silicon, Intel Macs or Linux. You do not need CUDA, a GPU or Conda.

Jupyter opens in your browser. Open `tia_01_depth/activity.ipynb`, select **Python (COMP0090)** if asked for a kernel, and run the cells from top to bottom. To start again with clean state, use **Kernel → Restart Kernel and Run All Cells**. Small differences in numerical results between computers are normal.

Activities 2 and 5 also have a terminal workflow. After activation, run:

```bash
python tia_02_generalisation/activity.py
python tia_05_pretraining/activity.py
```

Each script prints its results and saves a JSON file under `outputs/`. See the activity guides below for predictions, exercises and commands to compare runs. The scripts and their companion notebooks share experiment functions.

When returning later, open a terminal in this directory and repeat only the activation and launch commands. Stop Jupyter with **Ctrl-C** in its terminal; leave the environment with `deactivate`.

If anything fails, see [setup, troubleshooting and verification](docs/setup.md). New to the software? See [Tools and tutorials](docs/tools.md). References are collected in the [Reading list](docs/reading.md).

## Activities

| TiA | Title | Format | Investigation |
|---|---|---|---|
| 1 | Training Deep Networks | [Notebook](tia_01_depth/activity.ipynb) | Initialisation, signals, gradients and residual paths. |
| 2 | Generalisation vs Memorisation | [Script guide](tia_02_generalisation/README.md) · [Notebook](tia_02_generalisation/activity.ipynb) | Capacity, random labels, regularisation and distribution shift. |
| 3 | Inductive Bias and Equivariance | [Notebook](tia_03_inductive_bias/activity.ipynb) | Locality, shared weights and translation. |
| 4 | Self-Supervised Representations | [Notebook](tia_04_self_supervision/activity.ipynb) | Contrastive learning and augmentation choices. |
| 5 | Pretraining and Transfer | [Script guide](tia_05_pretraining/README.md) · [Notebook](tia_05_pretraining/activity.ipynb) | Frozen features, label efficiency and model width. |
| 6 | Diffusion from Noise to Data | [Notebook](tia_06_diffusion/activity.ipynb) | Forward noise, exact scores and reverse sampling. |
