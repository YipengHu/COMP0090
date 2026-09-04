# COMP0090: Introduction to Deep Learning

Theory-in-Action (TiA) activities for COMP0090 module. The notebooks use self-contained experiments to connect ideas to observable behaviour.

## Start here

Run the following commands from the repository root. Python 3.10–3.12 is recommended.

```bash
# Set up the environment once
python -m venv comp0090
source comp0090/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

# Start Jupyter Notebook
jupyter notebook
```

Jupyter will open a file browser in your web browser. For example, select `tia_01_depth`, then open `activity.ipynb`. Run its cells from top to bottom before moving to `tia_02_generalisation`.

For later study sessions, return to the repository root, reactivate the environment, and restart Jupyter:

```bash
source comp0090/bin/activate       # Windows PowerShell: .\comp0090\Scripts\Activate.ps1
jupyter notebook
```

New to any of the software? See [Tools and tutorials](docs/tools.md). Papers cited by the activities, broader background, and selected modern examples are collected in the [Reading list](docs/reading.md).

## Activities

| TiA | Title | Description |
|---|---|---|
| [1](tia_01_depth/activity.ipynb) | Training Deep Networks | Explore how initialisation and residual paths affect signals and gradients in deep networks. |
| [2](tia_02_generalisation/activity.ipynb) | Generalisation vs Memorisation | Test how capacity, regularisation and distribution shift affect fitting and generalisation. |
| [3](tia_03_inductive_bias/activity.ipynb) | Inductive Bias and Equivariance | Compare MLPs and CNNs to see how locality and equivariance shape sample efficiency. |
| [4](tia_04_self_supervision/activity.ipynb) | Self-Supervised Representations | Examine how contrastive learning and augmentations shape representations without labels. |
| [5](tia_05_pretraining/activity.ipynb) | Pretraining and Transfer | Measure how pretrained features and model scale affect transfer with limited labels. |
| [6](tia_06_diffusion/activity.ipynb) | Diffusion from Noise to Data | Visualise forward noise, score functions and reverse diffusion sampling. |
