"""Check versions and a small CPU/autograd/NumPy round trip before class."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import sys

from packaging.requirements import Requirement
from packaging.version import Version


def main():
    print(f"Python {platform.python_version()} | {platform.system()} {platform.machine()}")
    print(f"Interpreter: {sys.executable}")
    errors = []
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        errors.append("Use Python 3.10–3.12.")
    requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
    for line in requirements.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            installed = version(requirement.name)
            print(f"  {requirement.name}: {installed}")
            if Version(installed) not in requirement.specifier:
                errors.append(f"{requirement.name}: expected {requirement.specifier}, got {installed}")
        except PackageNotFoundError:
            errors.append(f"{requirement.name} is missing")
    if errors:
        sys.exit("\n".join(errors) + "\nRun python3 setup_env.py from the repository directory.")

    import numpy as np
    import torch
    from sklearn.datasets import load_digits
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.set_num_threads(1)
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    x.square().sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [2.0, 4.0])
    assert load_digits().data.shape == (1797, 64)
    fig, ax = plt.subplots()
    ax.plot(x.detach().numpy())
    fig.canvas.draw()
    plt.close(fig)
    print("CPU, autograd, NumPy conversion, packaged dataset and plotting: OK")


if __name__ == "__main__":
    main()
