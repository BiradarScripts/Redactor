from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import torch

from .model import MCNCoreOutput


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for PNG visualizations") from exc


def plot_graph_png(
    core_out: MCNCoreOutput,
    op_names: list[str],
    path: Path,
    sample_index: int = 0,
    node_threshold: float = 0.05,
    edge_threshold: float = 0.15,
) -> None:
    plt = _load_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)

    gates = core_out.active_gates[sample_index].detach().cpu().squeeze(-1)
    adjacency = core_out.adjacency[sample_index].detach().cpu()
    op_ids = core_out.op_logits[sample_index].argmax(dim=-1).detach().cpu()
    active_nodes = [idx for idx, gate in enumerate(gates.tolist()) if gate >= node_threshold]
    if not active_nodes:
        active_nodes = [int(gates.argmax())]

    n_cells = adjacency.shape[0]
    angles = torch.linspace(0, 2 * math.pi, n_cells + 1)[:-1]
    points = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("MCN Grown Graph")
    cmap = plt.get_cmap("tab10")

    max_edge = float(adjacency.max().clamp_min(1e-6))
    for src in active_nodes:
        for dst in active_nodes:
            weight = float(adjacency[src, dst])
            if weight < edge_threshold:
                continue
            start = points[src]
            end = points[dst]
            ax.plot(
                [float(start[0]), float(end[0])],
                [float(start[1]), float(end[1])],
                color="#4a5f7f",
                alpha=0.15 + 0.55 * weight / max_edge,
                linewidth=0.5 + 2.5 * weight / max_edge,
                zorder=1,
            )

    for idx in active_nodes:
        point = points[idx]
        op_id = int(op_ids[idx])
        op_name = op_names[op_id] if op_id < len(op_names) else f"op_{op_id}"
        gate = float(gates[idx])
        ax.scatter(
            [float(point[0])],
            [float(point[1])],
            s=160 + 700 * gate,
            color=cmap(op_id % 10),
            edgecolor="#1f2937",
            linewidth=1.0,
            zorder=2,
        )
        ax.text(
            float(point[0]) * 1.12,
            float(point[1]) * 1.12,
            f"{idx}\n{op_name}\n{gate:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color="#111827",
            zorder=3,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_development_trace_png(
    core_out: MCNCoreOutput,
    path: Path,
    sample_index: int = 0,
) -> None:
    if not core_out.development_trace:
        return
    plt = _load_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)

    trace = core_out.development_trace
    gates = [step[sample_index].squeeze(-1) for step in trace["gates"]]
    adjacency = [step[sample_index] for step in trace["adjacency"]]
    halt = [float(step[sample_index]) for step in trace["halt_probs"]]
    active_cells = [float(gate.sum()) for gate in gates]
    graph_density = [
        float(adj.sum() / max(float((gate > 0.05).sum().pow(2)), 1.0))
        for adj, gate in zip(adjacency, gates)
    ]

    steps = list(range(1, len(active_cells) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(steps, active_cells, marker="o", color="#2563eb")
    axes[0].set_title("Active Cells")
    axes[0].set_xlabel("development step")
    axes[0].set_ylabel("soft count")

    axes[1].plot(steps, graph_density, marker="o", color="#059669")
    axes[1].set_title("Graph Density")
    axes[1].set_xlabel("development step")

    axes[2].plot(steps, halt, marker="o", color="#dc2626")
    axes[2].set_title("Halting Probability")
    axes[2].set_xlabel("development step")
    axes[2].set_ylim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_adjacency_heatmap_png(
    core_out: MCNCoreOutput,
    path: Path,
    sample_index: int = 0,
) -> None:
    plt = _load_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    adjacency = core_out.adjacency[sample_index].detach().cpu()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(adjacency, cmap="viridis", vmin=0.0, vmax=float(adjacency.max().clamp_min(1e-6)))
    ax.set_title("Learned Adjacency")
    ax.set_xlabel("target cell")
    ax.set_ylabel("source cell")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_operation_distribution_png(
    core_out: MCNCoreOutput,
    op_names: list[str],
    path: Path,
    sample_index: int = 0,
) -> None:
    plt = _load_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    probs = torch.softmax(core_out.op_logits[sample_index].detach().cpu(), dim=-1).mean(dim=0)
    labels = [name.replace("Op", "") for name in op_names]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 4))
    ax.bar(labels, probs.tolist(), color="#2563eb")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("mean cell probability")
    ax.set_title("Operation Distribution")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_compute_profile_png(rows: Iterable[dict[str, object]], path: Path) -> None:
    plt = _load_pyplot()
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    complexity = list(range(len(rows)))
    active = [float(row["active_cells"]) for row in rows]
    density = [float(row["edge_density"]) for row in rows]
    flops = [float(row["estimated_flops"]) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].scatter(complexity, active, s=18, color="#2563eb")
    axes[0].set_title("Active Cells")
    axes[0].set_xlabel("profile row")

    axes[1].scatter(complexity, density, s=18, color="#059669")
    axes[1].set_title("Graph Density")
    axes[1].set_xlabel("profile row")

    axes[2].scatter(active, flops, s=18, color="#7c3aed")
    axes[2].set_title("Compute Proxy")
    axes[2].set_xlabel("active cells")
    axes[2].set_ylabel("estimated FLOPs")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
