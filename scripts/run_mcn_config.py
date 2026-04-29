#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcn.config_io import flatten_cli_args, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an MCN experiment from a YAML/JSON config.")
    parser.add_argument("config", help="Path to a config file under configs/ or elsewhere.")
    args, overrides = parser.parse_known_args()

    config = load_config(args.config)
    script = ROOT / "scripts" / "train_mcn_synthetic.py" if "task" in config.get("data", {}) else ROOT / "scripts" / "train_mcn_toy.py"
    command = [sys.executable, str(script), *flatten_cli_args(config), *overrides]
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
