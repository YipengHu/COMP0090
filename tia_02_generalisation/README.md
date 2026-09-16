# TiA 2 — Generalisation vs Memorisation

This activity uses a normal Python program: read its functions, run an experiment in the terminal, inspect saved results, change one setting and rerun. The [companion notebook](activity.ipynb) supplies the mathematical notation and calls the same functions.

## Predict, then run

Before running, predict whether a wider neural network can fit random labels, and whether fitting them should improve accuracy on genuine test labels. Training accuracy measures fit to the examples seen by the optimiser; test accuracy measures performance on held-out examples.

From the repository directory, after [setup](../README.md):

```bash
source .venv/bin/activate
python tia_02_generalisation/activity.py
```

The script uses scikit-learn's packaged 8×8 digits, trains small MLPs on 600 examples, and prints training and test accuracy for genuine and randomly permuted labels. It then tests rotation and weight decay. All data is local and the script needs no plotting window.

Results are saved to `outputs/generalisation.json`. Running the same command replaces that result file. Preserve a comparison by choosing another output path:

```bash
python tia_02_generalisation/activity.py --seed 8 --output outputs/generalisation-seed8.json
python tia_02_generalisation/activity.py --full --output outputs/generalisation-longer.json
```

`--full` increases the L-BFGS iteration budget from 300 to 800. Run `python tia_02_generalisation/activity.py --help` for all options. If the optimiser exhausts its budget, reported scores describe that budget rather than a guaranteed optimum.

## Read and change the program

Open [activity.py](activity.py) in your editor. `load_data` separates training and testing; `fit_model` fits one network; `capacity_experiment` compares widths using the same label permutation; `shift_experiment` rotates the held-out images. `main` reads command-line options and saves the result. Importing the file does not run training because of the `if __name__ == "__main__"` guard.

1. Run the default experiment and record the width-64 genuine/random-label results. Expect random labels to fit well while test accuracy remains near chance (10%).
2. Change the widths in `capacity_experiment` to include 16. Predict the result, rerun to a new output file, and compare training and test accuracy.
3. Compare genuine and rotated test accuracy. Explain what changed about the evaluation distribution.
4. Compare weight decay values. Explain why regularisation need not improve every run on this small clean dataset.
5. Repeat with another seed. Describe which conclusions survive and which numbers move.

Parameter counts include weights and biases. Exact numerical results can vary slightly by platform. Successful interpolation does not guarantee generalisation; this small experiment does not establish a complete theory of either.

**Reading:** [Zhang et al., Understanding Deep Learning Requires Rethinking Generalization](https://openreview.net/forum?id=Sy8gdB9xx).
