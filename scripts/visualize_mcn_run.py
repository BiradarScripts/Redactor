#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcn.visualization import load_jsonl, plot_compute_profile_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate aggregate MCN plots from a completed run directory.")
    parser.add_argument("run_dir", help="Directory containing compute_profile.jsonl.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = load_jsonl(run_dir / "compute_profile.jsonl")
    plot_compute_profile_png(rows, run_dir / "plots" / "compute_profile.png")
    print(f"plots={run_dir / 'plots'}")


if __name__ == "__main__":
    main()
