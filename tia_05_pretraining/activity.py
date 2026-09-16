"""TiA 5: pretrain an encoder, freeze its features, then fit linear probes.

Run from the repository: python tia_05_pretraining/activity.py
Read README.md in this directory for predictions and exercises.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


class Encoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(64, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU()
        )
        self.head = nn.Linear(width, 10)

    def forward(self, x, features=False):
        representation = self.features(x)
        return representation if features else self.head(representation)


def load_data(seed=7):
    x, y = load_digits(return_X_y=True)
    x = (x / 16).astype("float32")
    pretrain_idx, transfer_idx = train_test_split(
        np.arange(len(y)), test_size=0.5, stratify=y, random_state=seed
    )
    return x, y, pretrain_idx, transfer_idx


def pretrain(x, y, indices, width, epochs=25):
    model = Encoder(width)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    inputs = torch.tensor(x[indices])
    targets = torch.tensor(y[indices], dtype=torch.long)
    for _ in range(epochs):
        optimiser.zero_grad()
        loss = nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        optimiser.step()
    return model


def extract_features(x, models):
    """Freeze inference: downstream probes cannot update the encoder."""
    random_model = Encoder(32)
    inputs = torch.tensor(x)
    with torch.no_grad():
        features = {"pixels": x, "random": random_model(inputs, True).numpy()}
        features.update({f"pretrained-{width}": model(inputs, True).numpy()
                         for width, model in models.items()})
    return features


def probe_features(features, y, transfer_idx, seed=7, counts=(5, 20, 40)):
    # Pretraining, probe training and probe testing use disjoint examples.
    train, test = train_test_split(
        transfer_idx, test_size=0.5, stratify=y[transfer_idx], random_state=seed
    )
    available = min(np.sum(y[train] == label) for label in range(10))
    scores_by_count = {}
    for count in counts:
        if not 1 <= count <= available:
            raise ValueError(f"Requested {count} labels/class; this split supports 1–{available}.")
        chosen = np.concatenate([train[y[train] == label][:count] for label in range(10)])
        scores_by_count[count] = {
            name: make_pipeline(StandardScaler(), LogisticRegression(max_iter=600))
            .fit(values[chosen], y[chosen]).score(values[test], y[test])
            for name, values in features.items()
        }
    return scores_by_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--full", action="store_true", help="Use 60 rather than 25 pretraining epochs.")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs/pretraining.json",
                        help="JSON results file (rerunning replaces this file).")
    args = parser.parse_args()
    epochs = 60 if args.full else 25
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    with threadpool_limits(limits=1):
        x, y, pretrain_idx, transfer_idx = load_data(args.seed)
        models = {}
        for width in [8, 32, 96]:
            print(f"Pretraining width={width} for {epochs} epochs ...", flush=True)
            models[width] = pretrain(x, y, pretrain_idx, width, epochs)
        features = extract_features(x, models)
        scores = probe_features(features, y, transfer_idx, args.seed)
        parameters = {width: sum(p.numel() for p in model.parameters())
                      for width, model in models.items()}
        for count, values in scores.items():
            print(f"labels/class={count}", {name: round(value, 3) for name, value in values.items()})
        print("Parameter counts:", parameters)

    results = {"seed": args.seed, "epochs": epochs, "scores_by_count": scores,
               "parameter_counts": parameters}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
