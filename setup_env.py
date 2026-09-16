"""Create the course environment: python3 setup_env.py (Python 3.10–3.12)."""

import platform
from pathlib import Path
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parent
ENV = ROOT / ".venv"


def run(*command):
    subprocess.run([str(part) for part in command], check=True, cwd=ROOT)


def main():
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        sys.exit(
            "Use Python 3.10, 3.11 or 3.12 (3.11 recommended).\n"
            "For example: python3.11 setup_env.py\n"
            "See docs/setup.md for installation instructions."
        )
    if sys.platform not in {"darwin", "linux"} or platform.machine() not in {
        "x86_64", "arm64", "aarch64"
    }:
        sys.exit("This installer targets 64-bit macOS and Linux. See docs/setup.md.")
    if sys.platform == "darwin" and int(platform.mac_ver()[0].split(".")[0]) < 12:
        sys.exit("The course environment targets macOS 12 or later. See docs/setup.md.")

    python = ENV / "bin" / "python"
    if ENV.exists():
        # Reuse only a working environment with the requested Python version.
        # Never overwrite a student's existing environment or recreate it in place.
        try:
            version = subprocess.check_output(
                [str(python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            sys.exit(".venv is incomplete. Rename it, then rerun this command.")
        if version != f"{sys.version_info.major}.{sys.version_info.minor}":
            sys.exit(".venv uses a different Python version. Rename it, then rerun this command.")
    else:
        print(f"Creating {ENV} with Python {platform.python_version()}", flush=True)
        try:
            venv.EnvBuilder(with_pip=True).create(ENV)
        except (OSError, subprocess.CalledProcessError) as error:
            sys.exit(
                f"Could not create the environment: {error}\n"
                "On Ubuntu/Debian, install the venv package for your Python version\n"
                "(for the system Python: sudo apt install python3-venv).\n"
                "Rename the incomplete .venv directory before trying again. See docs/setup.md."
            )

    run(python, "-m", "pip", "install", "--upgrade", "pip==26.0.1")
    if sys.platform == "linux":
        # A dedicated install avoids pulling in large CUDA packages from PyPI.
        run(python, "-m", "pip", "install", "--only-binary=:all:",
            "torch==2.8.0", "--index-url", "https://download.pytorch.org/whl/cpu")
    run(python, "-m", "pip", "install", "--only-binary=:all:", "-r", ROOT / "requirements.txt")
    run(python, "-m", "pip", "check")
    # Register inside this environment, so the course kernel cannot point at an
    # unrelated global/Conda Python and does not modify the user's other kernels.
    run(python, "-m", "ipykernel", "install", "--sys-prefix",
        "--name", "comp0090", "--display-name", "Python (COMP0090)")
    run(python, ROOT / "scripts" / "check_environment.py")
    print("\nReady. From the repository directory, run:\n"
          "  source .venv/bin/activate\n"
          "  python -m notebook", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(f"\nSetup stopped (exit {error.returncode}). See docs/setup.md for help; "
                 "after fixing the problem, rerun the same setup command.")
