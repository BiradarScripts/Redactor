from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any

import torch

from .model import MCNCoreOutput, MCNSeq2Seq
from .toy_scan import ToyScanExample, Vocab, collate_examples


def batches(items: list[ToyScanExample], batch_size: int) -> list[list[ToyScanExample]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def metrics_to_floats(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in metrics.items()}


def summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "min": 0.0, "p25": 0.0, "mean": 0.0, "p50": 0.0, "p75": 0.0, "max": 0.0}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "count": float(tensor.numel()),
        "min": float(tensor.min()),
        "p25": float(torch.quantile(tensor, 0.25)),
        "mean": float(tensor.mean()),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p75": float(torch.quantile(tensor, 0.75)),
        "max": float(tensor.max()),
    }


def core_profile_rows(core_out: MCNCoreOutput, split: str, offset: int = 0) -> list[dict[str, float | int | str]]:
    gates = core_out.active_gates.detach().cpu().squeeze(-1)
    adjacency = core_out.adjacency.detach().cpu()
    logits = core_out.op_logits.detach().cpu()
    occupancy = core_out.occupancy.detach().cpu().squeeze(-1) if core_out.occupancy is not None else gates
    active_cells = gates.sum(dim=-1)
    occupied_cells = occupancy.sum(dim=-1)
    soft_edges = adjacency.sum(dim=(-2, -1))
    op_entropy = torch.distributions.Categorical(logits=logits).entropy().mean(dim=-1)
    d_model = float(core_out.cell_states.shape[-1])
    n_ops = float(core_out.op_logits.shape[-1])
    steps = float(core_out.metrics.get("development_steps", torch.tensor(1.0)).detach().cpu())
    estimated_flops = occupied_cells * d_model * d_model * steps + active_cells * n_ops * d_model * d_model + soft_edges * d_model
    rows = []
    for idx in range(gates.shape[0]):
        rows.append(
            {
                "split": split,
                "index": offset + idx,
                "active_cells": float(active_cells[idx]),
                "occupied_cells": float(occupied_cells[idx]),
                "soft_edges": float(soft_edges[idx]),
                "edge_density": float(soft_edges[idx] / max(float(active_cells[idx] ** 2), 1.0)),
                "op_entropy": float(op_entropy[idx]),
                "estimated_flops": float(estimated_flops[idx]),
            }
        )
    return rows


def profile_summary(rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, float]]:
    metric_names = ["active_cells", "occupied_cells", "soft_edges", "edge_density", "op_entropy", "estimated_flops"]
    return {name: summary_stats([float(row[name]) for row in rows]) for name in metric_names}


def decode_prediction(row: torch.Tensor, vocab: Vocab) -> list[str]:
    decoded: list[str] = []
    for idx in row.detach().cpu().tolist():
        if idx == vocab.eos_id:
            break
        if idx != vocab.pad_id:
            decoded.append(vocab.itos[idx])
    return decoded


@torch.no_grad()
def exact_match(
    model: torch.nn.Module,
    examples: list[ToyScanExample],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: str,
    batch_size: int = 32,
    max_items: int | None = None,
    max_len: int = 20,
) -> float:
    model.eval()
    subset = examples[:max_items] if max_items else examples
    exact = 0
    total = 0
    for batch_items in batches(subset, batch_size):
        batch = collate_examples(batch_items, src_vocab, tgt_vocab)
        pred = model.generate(
            batch["src"].to(device),
            tgt_vocab.bos_id,
            tgt_vocab.eos_id,
            max_len=max_len,
        ).cpu()
        for row, example in zip(pred, batch_items):
            exact += tuple(decode_prediction(row, tgt_vocab)) == example.actions
            total += 1
    return exact / max(total, 1)


@torch.no_grad()
def prediction_rows(
    model: torch.nn.Module,
    examples: list[ToyScanExample],
    split: str,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: str,
    max_items: int = 64,
    max_len: int = 20,
) -> list[dict[str, Any]]:
    model.eval()
    subset = examples[:max_items]
    rows: list[dict[str, Any]] = []
    for batch_items in batches(subset, 32):
        batch = collate_examples(batch_items, src_vocab, tgt_vocab)
        pred = model.generate(
            batch["src"].to(device),
            tgt_vocab.bos_id,
            tgt_vocab.eos_id,
            max_len=max_len,
        ).cpu()
        for row, example in zip(pred, batch_items):
            predicted = decode_prediction(row, tgt_vocab)
            rows.append(
                {
                    "split": split,
                    "command": example.command,
                    "expected": list(example.actions),
                    "predicted": predicted,
                    "exact": predicted == list(example.actions),
                }
            )
    return rows


def graph_to_dot(
    core_out: MCNCoreOutput,
    op_names: list[str],
    sample_index: int = 0,
    node_threshold: float = 0.05,
    edge_threshold: float = 0.15,
) -> str:
    gates = core_out.active_gates[sample_index].squeeze(-1).detach().cpu()
    adjacency = core_out.adjacency[sample_index].detach().cpu()
    op_ids = core_out.op_logits[sample_index].argmax(dim=-1).detach().cpu()
    lines = [
        "digraph MCN {",
        "  rankdir=LR;",
        '  graph [fontname="Helvetica"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fillcolor="#eaf2ff"];',
        '  edge [fontname="Helvetica", color="#5b6f94"];',
    ]
    active_nodes = [idx for idx, gate in enumerate(gates.tolist()) if gate >= node_threshold]
    for idx in active_nodes:
        op = op_names[int(op_ids[idx])] if int(op_ids[idx]) < len(op_names) else f"op_{int(op_ids[idx])}"
        lines.append(f'  c{idx} [label="cell {idx}\\n{op}\\ngate={gates[idx]:.2f}"];')
    for src in active_nodes:
        for dst in active_nodes:
            weight = float(adjacency[src, dst])
            if weight >= edge_threshold:
                lines.append(f'  c{src} -> c{dst} [label="{weight:.2f}"];')
    lines.append("}")
    return "\n".join(lines) + "\n"


@torch.no_grad()
def mcn_graph_dot(
    model: MCNSeq2Seq,
    example: ToyScanExample,
    src_vocab: Vocab,
    device: str,
    edge_threshold: float = 0.15,
) -> str:
    model.eval()
    batch = collate_examples([example], src_vocab, Vocab(example.actions))
    _, _, _, core_out = model.encode(batch["src"].to(device))
    return graph_to_dot(
        core_out,
        [op.__class__.__name__ for op in model.core.ops],
        edge_threshold=edge_threshold,
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
