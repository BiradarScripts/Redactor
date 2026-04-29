#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_mcn_toy.py"


def run_variant(args, suite_dir: Path, variant: str, extra: list[str]) -> dict:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--d-model",
        str(args.d_model),
        "--n-cells",
        str(args.n_cells),
        "--n-seed-cells",
        str(args.n_seed_cells),
        "--n-dev-steps",
        str(args.n_dev_steps),
        "--n-exec-steps",
        str(args.n_exec_steps),
        "--eval-items",
        str(args.eval_items),
        "--prediction-items",
        str(args.prediction_items),
        "--seed",
        str(args.seed),
        "--split",
        args.split,
        "--random-test-fraction",
        str(args.random_test_fraction),
        "--device",
        args.device,
        "--output-dir",
        str(suite_dir),
        "--run-name",
        variant,
        *extra,
    ]
    if args.save_checkpoint:
        command.append("--save-checkpoint")
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

    matches = sorted(suite_dir.glob(f"{variant}-*/summary.json"))
    if not matches:
        raise FileNotFoundError(f"no summary.json found for {variant} under {suite_dir}")
    with matches[-1].open(encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Run baseline, MCN, and MCN+CGB toy-SCAN experiments.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-cells", type=int, default=24)
    parser.add_argument("--n-seed-cells", type=int, default=6)
    parser.add_argument("--n-dev-steps", type=int, default=6)
    parser.add_argument("--n-exec-steps", type=int, default=2)
    parser.add_argument("--eval-items", type=int, default=128)
    parser.add_argument("--prediction-items", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split", choices=["jump", "random"], default="jump")
    parser.add_argument("--random-test-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", default="runs/mcn_suite")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--skip-cgb", action="store_true")
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    suite_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    suite_dir = output_dir / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    summaries = {
        "transformer": run_variant(args, suite_dir, "transformer", ["--model", "transformer"]),
        "mcn": run_variant(args, suite_dir, "mcn", ["--model", "mcn"]),
    }
    if args.include_baselines:
        for variant in ["baseline", "universal", "moe", "ponder", "deq"]:
            summaries[variant] = run_variant(args, suite_dir, variant, ["--model", variant])
    if not args.skip_cgb:
        summaries["mcn-cgb"] = run_variant(args, suite_dir, "mcn-cgb", ["--model", "mcn", "--use-cgb"])
    if args.include_ablations:
        ablations = {
            "mcn-no-prolif": ["--model", "mcn", "--disable-proliferation"],
            "mcn-no-prune": ["--model", "mcn", "--disable-pruning"],
            "mcn-no-diff": ["--model", "mcn", "--disable-differentiation"],
            "mcn-fixed-steps": ["--model", "mcn", "--disable-halting"],
            "mcn-random-dev": ["--model", "mcn", "--random-development"],
            "mcn-template-chain": ["--model", "mcn", "--template-adjacency", "chain"],
        }
        for variant, extra in ablations.items():
            summaries[variant] = run_variant(args, suite_dir, variant, extra)

    suite_summary = {"suite_dir": str(suite_dir), "runs": summaries}
    (suite_dir / "suite_summary.json").write_text(json.dumps(suite_summary, indent=2, sort_keys=True) + "\n")
    print(f"suite_artifacts={suite_dir}")


if __name__ == "__main__":
    main()
