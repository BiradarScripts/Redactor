#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOY_SCRIPT = ROOT / "scripts" / "train_mcn_toy.py"
SYNTH_SCRIPT = ROOT / "scripts" / "train_mcn_synthetic.py"


def run(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def latest_summary(root: Path, run_name: str) -> dict:
    matches = sorted(root.glob(f"{run_name}-*/summary.json"))
    if not matches:
        raise FileNotFoundError(f"no summary.json found under {root} for {run_name}")
    with matches[-1].open(encoding="utf-8") as handle:
        return json.load(handle)


def toy_command(args, suite_dir: Path, split: str, run_name: str, extra: list[str]) -> list[str]:
    return [
        sys.executable,
        str(TOY_SCRIPT),
        "--split",
        split,
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
        "--device",
        args.device,
        "--output-dir",
        str(suite_dir / "compositional" / split),
        "--run-name",
        run_name,
        *extra,
    ]


def synthetic_command(args, suite_dir: Path, task: str, run_name: str, extra: list[str]) -> list[str]:
    return [
        sys.executable,
        str(SYNTH_SCRIPT),
        "--task",
        task,
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
        "--device",
        args.device,
        "--output-dir",
        str(suite_dir / "synthetic"),
        "--run-name",
        run_name,
        *extra,
    ]


def main():
    parser = argparse.ArgumentParser(description="Run the local-scale MCN roadmap completion suite.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-cells", type=int, default=16)
    parser.add_argument("--n-seed-cells", type=int, default=4)
    parser.add_argument("--n-dev-steps", type=int, default=4)
    parser.add_argument("--n-exec-steps", type=int, default=2)
    parser.add_argument("--eval-items", type=int, default=128)
    parser.add_argument("--prediction-items", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="runs/mcn_roadmap")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--skip-cgb", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--minimal", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.quick:
        args.epochs = min(args.epochs, 1)
        args.eval_items = min(args.eval_items, 32)
        args.prediction_items = min(args.prediction_items, 8)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    suite_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    suite_dir = output_dir / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}

    toy_runs = [
        ("random", "transformer", ["--model", "transformer"]),
        ("random", "mcn", ["--model", "mcn"]),
        ("jump", "mcn-hard", ["--model", "mcn"]),
    ]
    if not args.minimal:
        toy_runs.extend(
            [
                ("random", "universal", ["--model", "universal"]),
                ("random", "baseline-gru", ["--model", "baseline"]),
                ("random", "moe", ["--model", "moe"]),
                ("random", "ponder", ["--model", "ponder"]),
                ("random", "deq", ["--model", "deq"]),
                ("random", "mcn-no-prolif", ["--model", "mcn", "--disable-proliferation"]),
                ("random", "mcn-no-prune", ["--model", "mcn", "--disable-pruning"]),
                ("random", "mcn-no-diff", ["--model", "mcn", "--disable-differentiation"]),
                ("random", "mcn-fixed-steps", ["--model", "mcn", "--disable-halting"]),
                ("random", "mcn-random-dev", ["--model", "mcn", "--random-development"]),
                ("random", "mcn-template-chain", ["--model", "mcn", "--template-adjacency", "chain"]),
            ]
        )
    if not args.skip_cgb:
        toy_runs.append(("random", "mcn-cgb", ["--model", "mcn", "--use-cgb"]))
        if not args.minimal:
            toy_runs.append(("jump", "mcn-cgb-hard", ["--model", "mcn", "--use-cgb"]))
    for split, name, extra in toy_runs:
        root = suite_dir / "compositional" / split
        run(toy_command(args, suite_dir, split, name, extra))
        summaries[f"compositional/{split}/{name}"] = latest_summary(root, name)

    synthetic_runs = [
        ("binding", "transformer", ["--model", "transformer"]),
        ("binding", "mcn", ["--model", "mcn"]),
        ("length", "mcn", ["--model", "mcn"]),
        ("length", "mcn-no-prolif", ["--model", "mcn", "--disable-proliferation"]),
        ("next_token", "mcn", ["--model", "mcn"]),
        ("scientific", "mcn", ["--model", "mcn"]),
    ]
    if not args.minimal:
        synthetic_runs.extend(
            [
                ("binding", "universal", ["--model", "universal"]),
                ("binding", "baseline-gru", ["--model", "baseline"]),
                ("binding", "moe", ["--model", "moe"]),
                ("binding", "ponder", ["--model", "ponder"]),
                ("binding", "deq", ["--model", "deq"]),
                ("clevr_binding", "transformer", ["--model", "transformer"]),
                ("clevr_binding", "mcn", ["--model", "mcn"]),
                ("length", "mcn-no-prune", ["--model", "mcn", "--disable-pruning"]),
                ("length", "mcn-no-diff", ["--model", "mcn", "--disable-differentiation"]),
                ("length", "mcn-fixed-steps", ["--model", "mcn", "--disable-halting"]),
                ("length", "mcn-random-dev", ["--model", "mcn", "--random-development"]),
                ("length", "mcn-template-chain", ["--model", "mcn", "--template-adjacency", "chain"]),
                ("arithmetic", "mcn", ["--model", "mcn"]),
                ("adaptive_compute", "mcn", ["--model", "mcn"]),
            ]
        )
    if not args.skip_cgb:
        synthetic_runs.append(("binding", "mcn-cgb", ["--model", "mcn", "--use-cgb"]))
        if not args.minimal:
            synthetic_runs.append(("clevr_binding", "mcn-cgb", ["--model", "mcn", "--use-cgb"]))
    for task, name, extra in synthetic_runs:
        root = suite_dir / "synthetic" / task
        run(synthetic_command(args, suite_dir, task, name, extra))
        summaries[f"synthetic/{task}/{name}"] = latest_summary(root, name)

    summary = {
        "suite_dir": str(suite_dir),
        "scope": "local-scale implementation of every MCN roadmap track",
        "minimal": args.minimal,
        "quick": args.quick,
        "runs": summaries,
    }
    (suite_dir / "roadmap_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"roadmap_artifacts={suite_dir}")


if __name__ == "__main__":
    main()
