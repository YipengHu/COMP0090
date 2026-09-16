# TiA 5 — Pretraining and Transfer

This activity follows a common development workflow: train a model, extract frozen features, compare downstream results and save the experiment configuration. The [companion notebook](activity.ipynb) explains linear probing mathematically and uses the same functions.

## Predict, then run

Predict whether a pretrained encoder will make digit classes easier to separate with a linear classifier when only five labels per class are available. Also predict whether a wider encoder must always be better.

From the repository directory, after [setup](../README.md):

```bash
source .venv/bin/activate
python tia_05_pretraining/activity.py
```

The script pretrains three small encoders on packaged 8×8 digits, then compares raw pixels, random features and pretrained features. Pretraining examples, downstream training examples and downstream test examples are disjoint. This is a controlled same-domain example of transferring a representation to new examples, not evidence of transfer to a new domain.

Read the rows for 5, 20 and 40 labels per class and the model parameter counts. Each accuracy lies between 0 and 1. The script writes `outputs/pretraining.json`, recording the seed, training epochs and results. It runs on CPU without model downloads or a plotting window.

```bash
python tia_05_pretraining/activity.py --seed 8 --output outputs/pretraining-seed8.json
python tia_05_pretraining/activity.py --full --output outputs/pretraining-longer.json
```

`--full` uses 60 rather than 25 pretraining epochs. The default output file is replaced on each run; use `--output` to preserve comparisons. Use `--help` to list the options.

## Read and change the program

Open [activity.py](activity.py). `Encoder` defines a representation and classification head; `pretrain` updates their parameters; `extract_features` computes representations without gradients; `probe_features` fits only a linear classifier. Scaling is fitted on downstream training examples only. `main` ties these steps to the command line and saves results.

1. Run the default configuration and compare random and pretrained features at five labels per class. Explain what the probe measures.
2. Add width 48 to the model loop in `main`. Predict its parameter count and performance, then save a new result file.
3. Change the label counts in `probe_features` to `(2, 5, 10, 20, 40)`. Explain the trend and any departures from it. The function rejects counts that exceed available examples.
4. Repeat with `--full` or another seed, changing one setting at a time. Explain why model width alone cannot establish a universal scaling law.
5. Describe how fine-tuning would differ from this frozen-feature experiment.

Expect the wider pretrained encoder to outperform random features in the default low-label experiment. Raw pixels are a useful baseline and may remain competitive. Exact scores and rankings can vary between settings and computers.

The notebook's optional modern-model extension needs additional packages and model downloads. It is separate from this self-contained activity.
