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

from mcn.experiment import (
    append_jsonl,
    core_profile_rows,
    exact_match,
    graph_to_dot,
    prediction_rows,
    profile_summary,
    write_csv,
    write_json,
)
from mcn.model import (
    DEQSeq2SeqBaseline,
    FixedSeq2SeqBaseline,
    MCNConfig,
    MCNSeq2Seq,
    MoESeq2SeqBaseline,
    PonderSeq2SeqBaseline,
    TransformerSeq2SeqBaseline,
    UniversalSeq2SeqBaseline,
    mcn_regularization,
)
from mcn.toy_scan import build_vocabs, collate_examples, make_jump_composition_split, make_random_split
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


def build_model(args, src_vocab_size: int, tgt_vocab_size: int, pad_id: int):
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
        return TransformerSeq2SeqBaseline(src_vocab_size, tgt_vocab_size, config, pad_id=pad_id), config
    if args.model == "baseline":
        return FixedSeq2SeqBaseline(src_vocab_size, tgt_vocab_size, config, pad_id=pad_id), config
    if args.model == "universal":
        return UniversalSeq2SeqBaseline(src_vocab_size, tgt_vocab_size, config, pad_id=pad_id), config
    if args.model == "moe":
        return MoESeq2SeqBaseline(src_vocab_size, tgt_vocab_size, config, pad_id=pad_id), config
    if args.model == "ponder":
        return PonderSeq2SeqBaseline(src_vocab_size, tgt_vocab_size, config, pad_id=pad_id), config
    if args.model == "deq":
        return DEQSeq2SeqBaseline(src_vocab_size, tgt_vocab_size, config, pad_id=pad_id), config
    return MCNSeq2Seq(src_vocab_size, tgt_vocab_size, config, pad_id=pad_id), config


@torch.no_grad()
def sequence_loss(model, examples, src_vocab, tgt_vocab, device: str, batch_size: int, max_items: int | None) -> float:
    model.eval()
    subset = examples[:max_items] if max_items else examples
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(subset), batch_size):
        batch_items = subset[start : start + batch_size]
        batch = collate_examples(batch_items, src_vocab, tgt_vocab)
        src = batch["src"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        tgt_out = batch["tgt_out"].to(device)
        output = model(src, tgt_in)
        logits = output[0] if isinstance(output, tuple) else output
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tgt_out.reshape(-1),
            ignore_index=tgt_vocab.pad_id,
            reduction="sum",
        )
        nonpad = tgt_out.ne(tgt_vocab.pad_id).sum().item()
        total_loss += float(loss.detach().cpu())
        total_tokens += nonpad
    return total_loss / max(total_tokens, 1)


def run_dir_for(args) -> Path:
    name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = args.model
    if args.model == "mcn" and args.use_cgb:
        suffix = "mcn-cgb"
    elif args.model == "mcn" and args.disable_proliferation:
        suffix = "mcn-no-prolif"
    elif args.model == "mcn" and args.disable_pruning:
        suffix = "mcn-no-prune"
    elif args.model == "mcn" and args.disable_differentiation:
        suffix = "mcn-no-diff"
    elif args.model == "mcn" and args.disable_halting:
        suffix = "mcn-fixed-steps"
    elif args.model == "mcn" and args.random_development:
        suffix = "mcn-random-dev"
    elif args.model == "mcn" and args.template_adjacency != "learned":
        suffix = f"mcn-template-{args.template_adjacency}"
    base = Path(args.output_dir)
    if not base.is_absolute():
        base = ROOT / base
    return base / f"{name}-{suffix}"


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate MCN v0.1 on a toy SCAN-like split.")
    parser.add_argument(
        "--model",
        choices=["mcn", "transformer", "baseline", "universal", "moe", "ponder", "deq"],
        default="mcn",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-signal", type=int, default=None)
    parser.add_argument("--n-cells", type=int, default=24)
    parser.add_argument("--n-seed-cells", type=int, default=6)
    parser.add_argument("--n-dev-steps", type=int, default=6)
    parser.add_argument("--n-exec-steps", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--op-temperature", type=float, default=1.0)
    parser.add_argument("--hard-ops", action="store_true")
    parser.add_argument("--min-active-gate", type=float, default=0.0)
    parser.add_argument("--split", choices=["jump", "random"], default="jump")
    parser.add_argument("--random-test-fraction", type=float, default=0.2)
    parser.add_argument("--use-cgb", action="store_true")
    parser.add_argument("--clifford-p", type=int, default=3)
    parser.add_argument("--disable-proliferation", action="store_true")
    parser.add_argument("--disable-pruning", action="store_true")
    parser.add_argument("--disable-differentiation", action="store_true")
    parser.add_argument("--disable-halting", action="store_true")
    parser.add_argument("--random-development", action="store_true")
    parser.add_argument("--template-adjacency", choices=["learned", "dense", "chain", "star"], default="learned")
    parser.add_argument("--eval-items", type=int, default=128)
    parser.add_argument("--prediction-items", type=int, default=64)
    parser.add_argument("--graph-edge-threshold", type=float, default=0.15)
    parser.add_argument("--output-dir", default="runs/mcn_toy")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.split == "random":
        train, test = make_random_split(args.seed, args.random_test_fraction)
        test_name = "random_test"
    else:
        train, test = make_jump_composition_split(args.seed)
        test_name = "jump_composition"
    src_vocab, tgt_vocab = build_vocabs([*train, *test])
    model, config = build_model(args, len(src_vocab), len(tgt_vocab), src_vocab.pad_id)
    model = model.to(args.device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    run_dir = run_dir_for(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "args": vars(args),
        "mcn_config": asdict(config),
        "dataset": {"split": args.split, "train_examples": len(train), f"{test_name}_examples": len(test)},
        "src_vocab": src_vocab.itos,
        "tgt_vocab": tgt_vocab.itos,
    }
    write_json(run_dir / "config.json", config_payload)
    write_json(run_dir / "config.yaml", config_payload)

    history: list[dict[str, float | int | str]] = []
    best_test_exact = -1.0
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

    for epoch in range(1, args.epochs + 1):
        # Curriculum: ramp morphogenic freedom over training
        progress = epoch / args.epochs
        if isinstance(model, MCNSeq2Seq):
            model.core.set_curriculum_progress(progress)
        model.train()
        total_loss = 0.0
        total_tokens = 0
        metric_totals = {name: 0.0 for name in metric_names}
        metric_batches = 0

        for batch_items in shuffled_batches(train, args.batch_size, args.seed + epoch):
            batch = collate_examples(batch_items, src_vocab, tgt_vocab)
            src = batch["src"].to(args.device)
            tgt_in = batch["tgt_in"].to(args.device)
            tgt_out = batch["tgt_out"].to(args.device)

            output = model(src, tgt_in)
            if isinstance(output, tuple):
                logits, core_out = output
            else:
                logits, core_out = output, None

            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tgt_out.reshape(-1),
                ignore_index=tgt_vocab.pad_id,
            )
            if core_out is not None:
                loss = loss + mcn_regularization(core_out)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            nonpad = tgt_out.ne(tgt_vocab.pad_id).sum().item()
            total_loss += loss.item() * nonpad
            total_tokens += nonpad
            if core_out is not None:
                for name in metric_names:
                    metric_totals[name] += float(core_out.metrics[name].detach().cpu())
                metric_batches += 1

        scheduler.step()

        train_exact = exact_match(
            model,
            train,
            src_vocab,
            tgt_vocab,
            args.device,
            max_items=args.eval_items,
        )
        test_exact = exact_match(
            model,
            test,
            src_vocab,
            tgt_vocab,
            args.device,
            max_items=args.eval_items,
        )
        test_loss = sequence_loss(model, test, src_vocab, tgt_vocab, args.device, args.batch_size, args.eval_items)
        row: dict[str, float | int | str] = {
            "epoch": epoch,
            "model": args.model,
            "loss": total_loss / max(total_tokens, 1),
            "val_loss": test_loss,
            "train_exact": train_exact,
            f"{test_name}_exact": test_exact,
        }
        if metric_batches:
            row.update({name: value / metric_batches for name, value in metric_totals.items()})
        history.append(row)
        append_jsonl(run_dir / "history.jsonl", row)
        checkpoint = {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "src_vocab": src_vocab.itos,
            "tgt_vocab": tgt_vocab.itos,
            "epoch": epoch,
            "row": row,
        }
        torch.save(checkpoint, run_dir / "checkpoint_last.pt")
        if test_exact >= best_test_exact:
            best_test_exact = test_exact
            torch.save(checkpoint, run_dir / "checkpoint_best.pt")

        metric_text = ""
        if metric_batches:
            metric_text = (
                f" active_cells={row['active_cells']:.2f} occupied_cells={row['occupied_cells']:.2f}"
                f" spawned={row['spawned_cells']:.2f} edge_density={row['edge_density']:.3f}"
                f" halt={row['halt_probability']:.3f}"
            )
        print(
            f"epoch={epoch} loss={row['loss']:.4f} train_exact={train_exact:.3f} "
            f"{test_name}_exact={test_exact:.3f}{metric_text}"
        )

    predictions = [
        *prediction_rows(model, train, "train", src_vocab, tgt_vocab, args.device, args.prediction_items),
        *prediction_rows(model, test, test_name, src_vocab, tgt_vocab, args.device, args.prediction_items),
    ]
    for row in predictions:
        append_jsonl(run_dir / "predictions.jsonl", row)

    profile_rows = []
    if isinstance(model, MCNSeq2Seq):
        model.eval()
        for split_name, items in [("train", train[: args.prediction_items]), (test_name, test[: args.prediction_items])]:
            offset = 0
            for batch_items in [items[start : start + args.batch_size] for start in range(0, len(items), args.batch_size)]:
                if not batch_items:
                    continue
                batch = collate_examples(batch_items, src_vocab, tgt_vocab)
                with torch.no_grad():
                    _, _, _, core_out = model.encode(batch["src"].to(args.device))
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

    test_metric = f"{test_name}_exact"
    write_csv(run_dir / "history.csv", history)
    write_csv(run_dir / "train_loss.csv", [{"epoch": row["epoch"], "train_loss": row["loss"]} for row in history])
    write_csv(run_dir / "val_loss.csv", [{"epoch": row["epoch"], "val_loss": row["val_loss"]} for row in history])
    write_csv(
        run_dir / "exact_match.csv",
        [
            {"epoch": row["epoch"], "train_exact": row["train_exact"], test_metric: row[test_metric]}
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
    final_summary = {
        "run_dir": str(run_dir),
        "final": history[-1] if history else {},
        f"best_{test_name}_exact": max((float(row[test_metric]) for row in history), default=0.0),
        "prediction_items": len(predictions),
        "compute_profile": profile_summary(profile_rows) if profile_rows else {},
    }
    write_json(run_dir / "summary.json", final_summary)

    if isinstance(model, MCNSeq2Seq):
        old_trace = model.config.save_development_trace
        model.config.save_development_trace = True
        batch = collate_examples([test[0]], src_vocab, tgt_vocab)
        model.eval()
        with torch.no_grad():
            _, _, _, core_out = model.encode(batch["src"].to(args.device))
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

    if args.save_checkpoint:
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": asdict(config),
                "src_vocab": src_vocab.itos,
                "tgt_vocab": tgt_vocab.itos,
                "summary": final_summary,
            },
            run_dir / "checkpoint.pt",
        )

    print(f"artifacts={run_dir}")


if __name__ == "__main__":
    main()
