"""TiA 2: compare memorisation, generalisation and distribution shift on CPU.

Run from the repository: python tia_02_generalisation/activity.py
Read README.md in this directory for predictions and exercises.
"""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits


def load_data(seed=7):
    """Use the packaged digits data; keep a separate, stratified test set."""
    x, y = load_digits(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x / 16, y, test_size=0.35, stratify=y, random_state=seed
    )
    return x_train[:600], x_test, y_train[:600], y_test


def fit_model(data, width, labels, alpha=0.0, max_iter=300, seed=7):
    x_train, x_test, _, y_test = data
    model = MLPClassifier(
        (width,), alpha=alpha, solver="lbfgs", max_iter=max_iter, random_state=seed
    )
    # A fixed optimisation budget can finish before convergence. Report scores
    # at that budget; do not treat them as proof of the best attainable fit.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, labels)
    return (
        model,
        accuracy_score(labels, model.predict(x_train)),
        accuracy_score(y_test, model.predict(x_test)),
    )


def capacity_experiment(data, seed=7, max_iter=300):
    """Use the same random-label permutation at every model width."""
    labels = data[2]
    random_labels = np.random.default_rng(seed).permutation(labels)
    rows = []
    for width in [8, 32, 64]:
        for name, targets in [("true", labels), ("random", random_labels)]:
            model, train, test = fit_model(data, width, targets, max_iter=max_iter, seed=seed)
            parameters = sum(w.size for w in model.coefs_ + model.intercepts_)
            rows.append((width, name, parameters, train, test))
    return rows


def shift_experiment(data, seed=7, max_iter=300):
    _, x_test, y_train, y_test = data
    model, _, accuracy = fit_model(data, 128, y_train, 1e-2, max_iter, seed)
    # A 90-degree rotation changes the test distribution without a download.
    rotated = np.rot90(x_test.reshape(-1, 8, 8), axes=(1, 2)).reshape(-1, 64)
    return {"in_distribution": accuracy,
            "rotated": accuracy_score(y_test, model.predict(rotated))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--full", action="store_true", help="Use 800 rather than 300 optimiser iterations.")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs/generalisation.json",
                        help="JSON results file (rerunning replaces this file).")
    args = parser.parse_args()
    max_iter = 800 if args.full else 300

    with threadpool_limits(limits=1):
        data = load_data(args.seed)
        rows = capacity_experiment(data, args.seed, max_iter)
        print("width labels parameters train_acc test_acc")
        for width, labels, parameters, train, test in rows:
            print(f"{width:5} {labels:6} {parameters:10} {train:.3f} {test:.3f}")
        shift = shift_experiment(data, args.seed, max_iter)
        print("Distribution shift:", shift)
        regularisation = []
        for alpha in [0, 1e-3, 1e-1, 1]:
            _, train, test = fit_model(data, 128, data[2], alpha, max_iter, args.seed)
            regularisation.append({"alpha": alpha, "train_acc": train, "test_acc": test})
            print(f"weight_decay={alpha:g}: train={train:.3f}, test={test:.3f}")

    results = {
        "seed": args.seed, "max_iter": max_iter,
        "capacity": [dict(zip(("width", "labels", "parameters", "train_acc", "test_acc"), row))
                     for row in rows],
        "shift": shift, "regularisation": regularisation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
