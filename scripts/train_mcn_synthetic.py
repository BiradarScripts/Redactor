#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcn.experiment import append_jsonl, core_profile_rows, graph_to_dot, profile_summary, write_csv, write_json
from mcn.model import (
    DEQClassifierBaseline,
    FixedClassifierBaseline,
    MCNConfig,
    MCNClassifier,
    MoEClassifierBaseline,
    PonderClassifierBaseline,
    TransformerClassifierBaseline,
    UniversalClassifierBaseline,
    mcn_regularization,
)
from mcn.synthetic_tasks import TASK_BUILDERS, build_classification_vocabs, collate_classification
from mcn.visualization import (
    plot_adjacency_heatmap_png,
    plot_compute_profile_png,
    plot_development_trace_png,
    plot_graph_png,
    plot_operation_distribution_png,
)


def shuffled_batches(items, batch_size: int, seed: int):
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start : start + batch_size]


@torch.no_grad()
def evaluate(model, examples, vocab, label_to_id, label_names, device, batch_size: int, max_items: int | None):
    model.eval()
    subset = examples[:max_items] if max_items else examples
    correct = 0
    rows = []
    for batch_items in shuffled_batches(subset, batch_size, 0):
        batch = collate_classification(batch_items, vocab, label_to_id)
        tokens = batch["tokens"].to(device)
        labels = batch["labels"].to(device)
        output = model(tokens)
        logits = output[0] if isinstance(output, tuple) else output
        pred = logits.argmax(dim=-1)
        correct += pred.eq(labels).sum().item()
        for idx, example in enumerate(batch_items):
            rows.append(
                {
                    "tokens": list(example.tokens),
                    "expected": example.label,
                    "predicted": label_names[int(pred[idx].cpu())],
                    "exact": bool(pred[idx].cpu().item() == label_to_id[example.label]),
                }
            )
    return correct / max(len(subset), 1), rows


@torch.no_grad()
def classification_loss(model, examples, vocab, label_to_id, device, batch_size: int, max_items: int | None) -> float:
    model.eval()
    subset = examples[:max_items] if max_items else examples
    total_loss = 0.0
    total_examples = 0
    for start in range(0, len(subset), batch_size):
        batch_items = subset[start : start + batch_size]
        batch = collate_classification(batch_items, vocab, label_to_id)
        tokens = batch["tokens"].to(device)
        labels = batch["labels"].to(device)
        output = model(tokens)
        logits = output[0] if isinstance(output, tuple) else output
        loss = F.cross_entropy(logits, labels, reduction="sum")
        total_loss += float(loss.detach().cpu())
        total_examples += len(batch_items)
    return total_loss / max(total_examples, 1)


def run_dir_for(args) -> Path:
    name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = args.model
    if args.model == "mcn" and args.use_cgb:
        suffix = "mcn-cgb"
    if args.model == "mcn" and args.disable_proliferation:
        suffix = "mcn-no-prolif"
    if args.model == "mcn" and args.disable_pruning:
        suffix = "mcn-no-prune"
    if args.model == "mcn" and args.disable_differentiation:
        suffix = "mcn-no-diff"
    if args.model == "mcn" and args.disable_halting:
        suffix = "mcn-fixed-steps"
    if args.model == "mcn" and args.random_development:
        suffix = "mcn-random-dev"
    if args.model == "mcn" and args.template_adjacency != "learned":
        suffix = f"mcn-template-{args.template_adjacency}"
    base = Path(args.output_dir)
    if not base.is_absolute():
        base = ROOT / base
    return base / args.task / f"{name}-{suffix}"


def build_model(args, vocab_size: int, n_classes: int, pad_id: int):
    config = MCNConfig(
        d_model=args.d_model,
        n_cells=args.n_cells,
        n_seed_cells=args.n_seed_cells,
        n_dev_steps=args.n_dev_steps,
        n_exec_steps=args.n_exec_steps,
        d_signal=args.d_signal or max(16, args.d_model // 2),
        op_temperature=args.op_temperature,
        hard_ops=args.hard_ops,
        min_active_gate=args.min_active_gate,
        enable_proliferation=not args.disable_proliferation,
        enable_pruning=not args.disable_pruning,
        enable_differentiation=not args.disable_differentiation,
        enable_halting=not args.disable_halting,
        random_development=args.random_development,
        template_adjacency=args.template_adjacency,
        use_cgb=args.use_cgb,
        clifford_p=args.clifford_p,
        max_seq_len=args.max_seq_len,
    )
    if args.model == "transformer":
        return TransformerClassifierBaseline(vocab_size, n_classes, config, pad_id=pad_id), config
    if args.model == "baseline":
        return FixedClassifierBaseline(vocab_size, n_classes, config, pad_id=pad_id), config
    if args.model == "universal":
        return UniversalClassifierBaseline(vocab_size, n_classes, config, pad_id=pad_id), config
    if args.model == "moe":
        return MoEClassifierBaseline(vocab_size, n_classes, config, pad_id=pad_id), config
    if args.model == "ponder":
        return PonderClassifierBaseline(vocab_size, n_classes, config, pad_id=pad_id), config
    if args.model == "deq":
        return DEQClassifierBaseline(vocab_size, n_classes, config, pad_id=pad_id), config
    return MCNClassifier(vocab_size, n_classes, config, pad_id=pad_id), config


def main():
    parser = argparse.ArgumentParser(description="Train MCN classifiers on local roadmap synthetic benchmarks.")
    parser.add_argument("--task", choices=sorted(TASK_BUILDERS), default="binding")
    parser.add_argument(
        "--model",
        choices=["mcn", "transformer", "baseline", "universal", "moe", "ponder", "deq"],
        default="mcn",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-signal", type=int, default=None)
    parser.add_argument("--n-cells", type=int, default=16)
    parser.add_argument("--n-seed-cells", type=int, default=4)
    parser.add_argument("--n-dev-steps", type=int, default=4)
    parser.add_argument("--n-exec-steps", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--op-temperature", type=float, default=1.0)
    parser.add_argument("--hard-ops", action="store_true")
    parser.add_argument("--min-active-gate", type=float, default=0.0)
    parser.add_argument("--use-cgb", action="store_true")
    parser.add_argument("--clifford-p", type=int, default=3)
    parser.add_argument("--disable-proliferation", action="store_true")
    parser.add_argument("--disable-pruning", action="store_true")
    parser.add_argument("--disable-differentiation", action="store_true")
    parser.add_argument("--disable-halting", action="store_true")
    parser.add_argument("--random-development", action="store_true")
    parser.add_argument("--template-adjacency", choices=["learned", "dense", "chain", "star"], default="learned")
    parser.add_argument("--eval-items", type=int, default=0)
    parser.add_argument("--prediction-items", type=int, default=64)
    parser.add_argument("--graph-edge-threshold", type=float, default=0.15)
    parser.add_argument("--output-dir", default="runs/mcn_synthetic")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train, test = TASK_BUILDERS[args.task](args.seed)
    vocab, label_to_id, label_names = build_classification_vocabs([*train, *test])
    model, config = build_model(args, len(vocab), len(label_names), vocab.pad_id)
    model = model.to(args.device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    run_dir = run_dir_for(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "args": vars(args),
        "mcn_config": asdict(config),
        "dataset": {"task": args.task, "train_examples": len(train), "test_examples": len(test)},
        "vocab": vocab.itos,
        "labels": label_names,
    }
    write_json(run_dir / "config.json", config_payload)
    write_json(run_dir / "config.yaml", config_payload)

    metric_names = [
        "active_cells",
        "occupied_cells",
        "spawned_cells",
        "edge_density",
        "development_stability",
        "op_entropy",
        "cell_type_entropy",
        "adjacency_entropy",
        "halt_probability",
        "development_steps",
        "estimated_flops",
        "compute_proxy",
    ]
    history = []
    best_test_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        # Curriculum: ramp morphogenic freedom over training
        progress = epoch / args.epochs
        if isinstance(model, MCNClassifier):
            model.core.set_curriculum_progress(progress)
        model.train()
        total_loss = 0.0
        total_examples = 0
        metric_totals = {name: 0.0 for name in metric_names}
        metric_batches = 0
        for batch_items in shuffled_batches(train, args.batch_size, args.seed + epoch):
            batch = collate_classification(batch_items, vocab, label_to_id)
            tokens = batch["tokens"].to(args.device)
            labels = batch["labels"].to(args.device)
            output = model(tokens)
            if isinstance(output, tuple):
                logits, core_out = output
            else:
                logits, core_out = output, None
            loss = F.cross_entropy(logits, labels)
            if core_out is not None:
                loss = loss + mcn_regularization(core_out)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * len(batch_items)
            total_examples += len(batch_items)
            if core_out is not None:
                for name in metric_names:
                    metric_totals[name] += float(core_out.metrics[name].detach().cpu())
                metric_batches += 1

        scheduler.step()

        train_acc, _ = evaluate(model, train, vocab, label_to_id, label_names, args.device, args.batch_size, args.eval_items)
        test_acc, _ = evaluate(model, test, vocab, label_to_id, label_names, args.device, args.batch_size, args.eval_items)
        test_loss = classification_loss(model, test, vocab, label_to_id, args.device, args.batch_size, args.eval_items)
        row = {
            "epoch": epoch,
            "model": args.model,
            "task": args.task,
            "loss": total_loss / max(total_examples, 1),
            "val_loss": test_loss,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
        }
        if metric_batches:
            row.update({name: value / metric_batches for name, value in metric_totals.items()})
        history.append(row)
        append_jsonl(run_dir / "history.jsonl", row)
        checkpoint = {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "vocab": vocab.itos,
            "labels": label_names,
            "epoch": epoch,
            "row": row,
        }
        torch.save(checkpoint, run_dir / "checkpoint_last.pt")
        if test_acc >= best_test_accuracy:
            best_test_accuracy = test_acc
            torch.save(checkpoint, run_dir / "checkpoint_best.pt")
        metric_text = ""
        if metric_batches:
            metric_text = (
                f" active_cells={row['active_cells']:.2f} occupied_cells={row['occupied_cells']:.2f}"
                f" edge_density={row['edge_density']:.3f} halt={row['halt_probability']:.3f}"
            )
        print(
            f"epoch={epoch} loss={row['loss']:.4f} train_accuracy={train_acc:.3f} "
            f"test_accuracy={test_acc:.3f}{metric_text}"
        )

    _, train_predictions = evaluate(
        model,
        train,
        vocab,
        label_to_id,
        label_names,
        args.device,
        args.batch_size,
        args.prediction_items,
    )
    _, test_predictions = evaluate(
        model,
        test,
        vocab,
        label_to_id,
        label_names,
        args.device,
        args.batch_size,
        args.prediction_items,
    )
    for row in train_predictions:
        append_jsonl(run_dir / "predictions.jsonl", {"split": "train", **row})
    for row in test_predictions:
        append_jsonl(run_dir / "predictions.jsonl", {"split": "test", **row})

    profile_rows = []
    if isinstance(model, MCNClassifier):
        model.eval()
        for split_name, items in [("train", train[: args.prediction_items]), ("test", test[: args.prediction_items])]:
            offset = 0
            for batch_items in [items[start : start + args.batch_size] for start in range(0, len(items), args.batch_size)]:
                if not batch_items:
                    continue
                batch = collate_classification(batch_items, vocab, label_to_id)
                with torch.no_grad():
                    _, core_out = model(batch["tokens"].to(args.device))
                rows = core_profile_rows(core_out, split_name, offset)
                profile_rows.extend(rows)
                for row in rows:
                    append_jsonl(run_dir / "compute_profile.jsonl", row)
                offset += len(batch_items)
        write_json(run_dir / "compute_profile.json", profile_summary(profile_rows))
        try:
            plot_compute_profile_png(profile_rows, run_dir / "plots" / "compute_profile.png")
        except RuntimeError as exc:
            print(f"visualization_skipped={exc}")

    write_csv(run_dir / "history.csv", history)
    write_csv(run_dir / "train_loss.csv", [{"epoch": row["epoch"], "train_loss": row["loss"]} for row in history])
    write_csv(run_dir / "val_loss.csv", [{"epoch": row["epoch"], "val_loss": row["val_loss"]} for row in history])
    write_csv(
        run_dir / "exact_match.csv",
        [
            {"epoch": row["epoch"], "train_accuracy": row["train_accuracy"], "test_accuracy": row["test_accuracy"]}
            for row in history
        ],
    )
    write_csv(
        run_dir / "active_cells.csv",
        [
            {"epoch": row["epoch"], "active_cells": row.get("active_cells", ""), "occupied_cells": row.get("occupied_cells", "")}
            for row in history
        ],
    )
    write_csv(
        run_dir / "graph_density.csv",
        [
            {"epoch": row["epoch"], "edge_density": row.get("edge_density", ""), "soft_edges": row.get("soft_edges", "")}
            for row in history
        ],
    )
    summary = {
        "run_dir": str(run_dir),
        "final": history[-1] if history else {},
        "best_test_accuracy": max((row["test_accuracy"] for row in history), default=0.0),
        "prediction_items": len(train_predictions) + len(test_predictions),
        "compute_profile": profile_summary(profile_rows) if profile_rows else {},
    }
    write_json(run_dir / "summary.json", summary)

    if isinstance(model, MCNClassifier):
        old_trace = model.config.save_development_trace
        model.config.save_development_trace = True
        batch = collate_classification([test[0]], vocab, label_to_id)
        model.eval()
        with torch.no_grad():
            _, core_out = model(batch["tokens"].to(args.device))
        model.config.save_development_trace = old_trace
        dot = graph_to_dot(
            core_out,
            [op.__class__.__name__ for op in model.core.ops],
            edge_threshold=args.graph_edge_threshold,
        )
        (run_dir / "sample_graph.dot").write_text(dot, encoding="utf-8")
        graphs_dir = run_dir / "graphs"
        plots_dir = run_dir / "plots"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        (graphs_dir / "sample_graph.dot").write_text(dot, encoding="utf-8")
        op_names = [op.__class__.__name__ for op in model.core.ops]
        op_probs = torch.softmax(core_out.op_logits[0].detach().cpu(), dim=-1).mean(dim=0)
        write_csv(
            run_dir / "operation_distribution.csv",
            [{"operation": name, "probability": float(prob)} for name, prob in zip(op_names, op_probs)],
        )
        try:
            plot_graph_png(core_out, op_names, graphs_dir / "sample_graph.png", edge_threshold=args.graph_edge_threshold)
            plot_adjacency_heatmap_png(core_out, plots_dir / "adjacency_heatmap.png")
            plot_operation_distribution_png(core_out, op_names, plots_dir / "operation_distribution.png")
            plot_development_trace_png(core_out, plots_dir / "development_trace.png")
        except RuntimeError as exc:
            print(f"visualization_skipped={exc}")

    print(f"artifacts={run_dir}")


if __name__ == "__main__":
    main()
