# COMP0090: Introduction to Deep Learning

Theory-in-Action (TiA) activities for a theory-led deep learning module. The notebooks use small, self-contained experiments to connect mathematical claims to observable behaviour.

## Setup

Python 3.9–3.12 is supported.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
jupyter lab
```

Open the TiAs in numerical order and run each notebook from top to bottom. Work through one activity at a time: predict an outcome before running a cell, compare the result with the prediction, and answer the explanation prompts before moving on. The default path is CPU-friendly, deterministic, and requires no downloaded data, papers, model weights, or network access.

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

Shared teaching utilities live in `comp0090/`; activity-specific reasoning remains visible in each notebook. Papers are linked rather than distributed.
