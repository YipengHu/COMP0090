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

| TiA | Question | Main idea |
|---|---|---|
| [1](tia_01_depth/activity.ipynb) | Why are deep networks difficult to train? | Signal and gradient propagation, initialisation, residual paths |
| [2](tia_02_generalisation/activity.ipynb) | How can a model both generalise and memorise? | Random labels, capacity, regularisation, distribution shift |
| [3](tia_03_inductive_bias/activity.ipynb) | What does inductive bias buy us? | MLP/CNN comparison, translation, equivariance, mismatched bias |
| [4](tia_04_self_supervision/activity.ipynb) | Can useful representations emerge without labels? | Tiny contrastive learning, representation geometry, linear probes |
| [5](tia_05_pretraining/activity.ipynb) | What does pretraining give us? | Frozen features, transfer, labelled-data efficiency, model scale |
| [6](tia_06_diffusion/activity.ipynb) | How does a generative model turn noise into data? | Forward noise, scores, reverse diffusion, sampling trade-offs |
