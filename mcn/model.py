from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .clifford import CliffordAlgebra


@dataclass
class MCNConfig:
    d_model: int = 128
    n_cells: int = 32
    n_seed_cells: int = 6
    n_dev_steps: int = 8
    n_exec_steps: int = 2
    d_signal: int = 64
    dropout: float = 0.1
    op_temperature: float = 1.0
    hard_ops: bool = False
    halt_threshold: float = 0.95
    min_active_gate: float = 0.0
    enable_proliferation: bool = True
    enable_pruning: bool = True
    enable_differentiation: bool = True
    enable_halting: bool = True
    random_development: bool = False
    template_adjacency: str = "learned"
    use_cgb: bool = False
    clifford_p: int = 3
    max_seq_len: int = 64
    save_development_trace: bool = False


@dataclass
class MCNCoreOutput:
    pooled: torch.Tensor
    cell_states: torch.Tensor
    adjacency: torch.Tensor
    active_gates: torch.Tensor
    op_logits: torch.Tensor
    halt_prob: torch.Tensor
    metrics: dict[str, torch.Tensor]
    occupancy: torch.Tensor | None = None
    cell_type_logits: torch.Tensor | None = None
    development_trace: dict[str, list[torch.Tensor]] | None = None


class IdentityOp(nn.Module):
    def forward(self, states: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        return states


class ZeroOp(nn.Module):
    def forward(self, states: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        return torch.zeros_like(states)


class LinearOp(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, states: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.proj(states)


class GatedMLPOp(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.up = nn.Linear(d_model, d_model * 2)
        self.down = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        value, gate = self.up(states).chunk(2, dim=-1)
        return self.down(self.dropout(F.gelu(value) * torch.sigmoid(gate)))


class InputAttentionOp(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.q(states)
        k = self.k(memory)
        v = self.v(memory)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        if memory_mask is not None:
            scores = scores.masked_fill(~memory_mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        return self.out(self.dropout(torch.matmul(weights, v)))


class CGBInputAttentionOp(nn.Module):
    """Geometric-product attention over input memory.

    Query/key projections are interpreted as multivectors in Cl(p, 0). The
    scalar grade of Q*K forms the attention logit.  Higher grades (bivectors,
    etc.) encode relational binding structure and are projected back into the
    output so that compositional information is forwarded through the network.
    """

    def __init__(self, d_model: int, dropout: float, clifford_p: int):
        super().__init__()
        self.algebra = CliffordAlgebra(clifford_p)
        if d_model % self.algebra.n_blades != 0:
            raise ValueError("d_model must be divisible by 2 ** clifford_p for CGB attention")
        self.channels = d_model // self.algebra.n_blades
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.binding_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _as_multivectors(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(*tensor.shape[:-1], self.channels, self.algebra.n_blades)

    def forward(self, states: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self._as_multivectors(self.q(states))
        k = self._as_multivectors(self.k(memory))
        q_pair = q.unsqueeze(2)
        k_pair = k.unsqueeze(1)
        bound = self.algebra.geometric_product(q_pair, k_pair)
        scores = self.algebra.scalar_part(bound).mean(dim=-1) / math.sqrt(self.channels)
        if memory_mask is not None:
            scores = scores.masked_fill(~memory_mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        attended = self.out(self.dropout(torch.matmul(weights, self.v(memory))))
        # Forward higher-grade binding: attention-weighted bound multivectors
        # with scalar grade zeroed out to isolate relational structure.
        binding = (weights.unsqueeze(-1).unsqueeze(-1) * bound).sum(dim=2)
        binding[..., 0] = 0.0  # remove scalar grade
        binding_flat = binding.flatten(-2, -1)
        return attended + self.binding_proj(binding_flat)


class MorphogenicCore(nn.Module):
    def __init__(self, config: MCNConfig):
        super().__init__()
        if config.n_cells <= 0:
            raise ValueError("n_cells must be positive")
        if config.n_seed_cells <= 0:
            raise ValueError("n_seed_cells must be positive")
        if config.n_exec_steps <= 0:
            raise ValueError("n_exec_steps must be positive")
        if config.template_adjacency not in {"learned", "dense", "chain", "star"}:
            raise ValueError("template_adjacency must be one of: learned, dense, chain, star")
        self.config = config
        self.n_seed_cells = min(config.n_seed_cells, config.n_cells)
        self.cell_init = nn.Parameter(torch.randn(config.n_cells, config.d_model) * 0.02)
        self.seed_to_cell = nn.Linear(config.d_model, config.d_model)
        self.signal_net = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_signal),
            nn.GELU(),
        )
        self.connect_q = nn.Linear(config.d_model, config.d_signal)
        self.connect_k = nn.Linear(config.d_model, config.d_signal)
        self.update_net = nn.Sequential(
            nn.Linear(config.d_model + config.d_signal + config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.state_norm = nn.LayerNorm(config.d_model)
        self.prune_gate = nn.Linear(config.d_model, 1)
        nn.init.constant_(self.prune_gate.bias, 1.0)
        self.halt_net = nn.Linear(config.d_model, 1)
        self.spawn_q = nn.Linear(config.d_model, config.d_signal)
        self.spawn_k = nn.Linear(config.d_model, config.d_signal)
        self.spawn_state = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.spawn_gate = nn.Linear(config.d_model * 3, 1)
        nn.init.constant_(self.spawn_gate.bias, -2.0)
        self.execution_update = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.execution_norm = nn.LayerNorm(config.d_model)

        ops: list[nn.Module] = [
            IdentityOp(),
            ZeroOp(),
            LinearOp(config.d_model),
            GatedMLPOp(config.d_model, config.dropout),
            InputAttentionOp(config.d_model, config.dropout),
        ]
        if config.use_cgb:
            ops.append(CGBInputAttentionOp(config.d_model, config.dropout, config.clifford_p))
        self.ops = nn.ModuleList(ops)
        self.op_logits = nn.Linear(config.d_model, len(self.ops))
        self.cell_type_logits = nn.Linear(config.d_model, max(2, len(self.ops)))
        self._curriculum_progress = 1.0

    def _operation_weights(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.config.enable_differentiation:
            return logits.new_full(logits.shape, 1.0 / logits.shape[-1])

        if self.training:
            return F.gumbel_softmax(
                logits,
                tau=self.config.op_temperature,
                hard=False,
                dim=-1,
            )

        if self.config.hard_ops:
            indices = logits.argmax(dim=-1)
            return F.one_hot(indices, num_classes=logits.shape[-1]).to(dtype=logits.dtype)

        return F.softmax(logits / self.config.op_temperature, dim=-1)

    def _initial_occupancy(self, batch: int, reference: torch.Tensor) -> torch.Tensor:
        occupancy = reference.new_zeros(batch, self.config.n_cells, 1)
        occupancy[:, : self.n_seed_cells] = 1.0
        return occupancy

    def set_curriculum_progress(self, progress: float) -> None:
        """Set training curriculum progress (0.0 = start, 1.0 = end of training)."""
        self._curriculum_progress = max(0.0, min(1.0, progress))

    @property
    def curriculum_freedom(self) -> float:
        """Morphogenic freedom factor: 0 during first 10%, linearly ramps to 1 by 30%."""
        p = self._curriculum_progress
        if p < 0.1:
            return 0.0
        return min(1.0, (p - 0.1) / 0.2)

    def _template_adjacency(self, states: torch.Tensor) -> torch.Tensor | None:
        mode = self.config.template_adjacency
        if mode == "learned":
            return None

        batch, n_cells, _ = states.shape
        template = states.new_zeros(batch, n_cells, n_cells)
        if mode == "dense":
            template.fill_(1.0)
        elif mode == "chain":
            idx = torch.arange(n_cells, device=states.device)
            template[:, idx, idx] = 1.0
            if n_cells > 1:
                template[:, idx[:-1], idx[1:]] = 1.0
                template[:, idx[1:], idx[:-1]] = 1.0
        elif mode == "star":
            template[:, 0, :] = 1.0
            template[:, :, 0] = 1.0
        return template

    def _proliferate(
        self,
        states: torch.Tensor,
        occupancy: torch.Tensor,
        seed_cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.config.enable_proliferation or self.n_seed_cells == self.config.n_cells:
            return states, occupancy, occupancy.new_zeros(occupancy.shape)

        source_strength = occupancy * torch.sigmoid(self.prune_gate(states))
        q = self.spawn_q(states)
        k = self.spawn_k(states)
        compatibility_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        compatibility = torch.sigmoid(compatibility_logits)

        parent_logits = compatibility_logits + torch.log(source_strength.squeeze(-1).clamp_min(1e-6)).unsqueeze(-1)
        parent_weights = F.softmax(parent_logits, dim=1)
        child_states = torch.bmm(parent_weights.transpose(1, 2), self.spawn_state(states))

        source_mass = source_strength.sum(dim=1, keepdim=True).clamp_min(1e-6)
        spawn_pressure = torch.bmm(compatibility.transpose(1, 2), source_strength) / source_mass
        spawn_features = torch.cat([states, child_states, seed_cell.expand_as(states)], dim=-1)
        spawn_prob = torch.sigmoid(self.spawn_gate(spawn_features)) * spawn_pressure * (1.0 - occupancy)

        states = states + spawn_prob * (child_states - states)
        occupancy = occupancy + (1.0 - occupancy) * spawn_prob
        return states, occupancy, spawn_prob

    def forward(
        self,
        seed: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ) -> MCNCoreOutput:
        batch = seed.shape[0]
        seed_cell = self.seed_to_cell(seed).unsqueeze(1)
        states = seed_cell + self.cell_init.unsqueeze(0)
        occupancy = self._initial_occupancy(batch, states)
        gates = occupancy
        adjacency = states.new_zeros(batch, self.config.n_cells, self.config.n_cells)
        halt_prob = states.new_zeros(batch)
        prev_states = states
        spawn_prob = states.new_zeros(batch, self.config.n_cells, 1)
        steps_taken = 0
        freedom = self.curriculum_freedom
        trace: dict[str, list[torch.Tensor]] | None = None
        if self.config.save_development_trace:
            trace = {"cell_states": [], "adjacency": [], "op_logits": [], "gates": [], "halt_probs": []}

        for _ in range(self.config.n_dev_steps):
            steps_taken += 1
            prev_states = states
            learned_signals = self.signal_net(states)
            signals = torch.randn_like(learned_signals) if self.config.random_development else learned_signals
            q = self.connect_q(states)
            k = self.connect_k(states)
            learned_adjacency = torch.sigmoid(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1]))
            if self.config.random_development:
                adjacency = torch.rand_like(learned_adjacency)
            else:
                adjacency = learned_adjacency
            template = self._template_adjacency(states)
            if template is not None:
                adjacency = 0.5 * adjacency + 0.5 * template
            if freedom < 1.0:
                cur_template = states.new_ones(batch, self.config.n_cells, self.config.n_cells)
                cur_template = cur_template / self.config.n_cells
                adjacency = freedom * adjacency + (1.0 - freedom) * cur_template
            adjacency = adjacency * gates * gates.transpose(1, 2)
            received = torch.bmm(adjacency, signals) / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            update = self.update_net(torch.cat([states, received, seed_cell.expand_as(states)], dim=-1))
            states = self.state_norm(states + update * gates)
            if freedom < 1.0:
                pre_states, pre_occ = states.clone(), occupancy.clone()
            states, occupancy, spawn_prob = self._proliferate(states, occupancy, seed_cell)
            if freedom < 1.0:
                states = freedom * states + (1.0 - freedom) * pre_states
                occupancy = freedom * occupancy + (1.0 - freedom) * pre_occ
                spawn_prob = spawn_prob * freedom
            if self.config.enable_pruning:
                gate_prob = torch.sigmoid(self.prune_gate(states))
                if self.config.min_active_gate > 0.0:
                    floor = min(max(float(self.config.min_active_gate), 0.0), 1.0)
                    gate_prob = floor + (1.0 - floor) * gate_prob
                raw_gates = occupancy * gate_prob
                gates = freedom * raw_gates + (1.0 - freedom) * occupancy if freedom < 1.0 else raw_gates
            else:
                gates = occupancy
            halt_prob = torch.sigmoid(self.halt_net((states * gates).sum(1) / gates.sum(1).clamp_min(1e-6))).squeeze(-1)
            if trace is not None:
                step_logits = self.op_logits(states)
                if not self.config.enable_differentiation:
                    step_logits = torch.zeros_like(step_logits)
                trace["cell_states"].append(states.detach().cpu())
                trace["adjacency"].append(adjacency.detach().cpu())
                trace["op_logits"].append(step_logits.detach().cpu())
                trace["gates"].append(gates.detach().cpu())
                trace["halt_probs"].append(halt_prob.detach().cpu())
            if (
                self.config.enable_halting
                and not self.training
                and bool((halt_prob > self.config.halt_threshold).all())
            ):
                break

        logits = self.op_logits(states)
        if not self.config.enable_differentiation:
            logits = torch.zeros_like(logits)
        cell_type_logits = self.cell_type_logits(states)
        op_weights = self._operation_weights(logits)
        if freedom < 1.0:
            uniform = logits.new_full(op_weights.shape, 1.0 / op_weights.shape[-1])
            op_weights = freedom * op_weights + (1.0 - freedom) * uniform
        executed = states
        graph_messages = states.new_zeros(states.shape)
        for _ in range(self.config.n_exec_steps):
            op_outputs = torch.stack([op(executed, memory, memory_mask) for op in self.ops], dim=-2)
            mixed_ops = (op_outputs * op_weights.unsqueeze(-1)).sum(dim=-2)
            graph_messages = torch.bmm(adjacency, mixed_ops) / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            executed = self.execution_norm(
                executed + self.execution_update(torch.cat([mixed_ops, graph_messages], dim=-1)) * gates
            )
        active = gates.clamp_min(1e-6)
        pooled = (executed * active).sum(dim=1) / active.sum(dim=1).clamp_min(1e-6)
        active_cells = gates.squeeze(-1).sum(dim=-1)
        occupied_cells = occupancy.squeeze(-1).sum(dim=-1)
        soft_edges = adjacency.sum(dim=(-2, -1))
        d_model = states.new_tensor(float(self.config.d_model))
        d_signal = states.new_tensor(float(self.config.d_signal))
        n_ops = states.new_tensor(float(len(self.ops)))
        development_flops = (
            occupied_cells * d_model * d_signal
            + soft_edges * d_signal
            + occupied_cells * d_model * d_model
        ) * float(steps_taken)
        execution_flops = (active_cells * n_ops * d_model * d_model + soft_edges * d_model) * float(
            self.config.n_exec_steps
        )
        estimated_flops = development_flops + execution_flops
        compute_proxy = active_cells + soft_edges / self.config.n_cells + (occupied_cells - self.n_seed_cells).clamp_min(0)
        edge_probs = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        adjacency_entropy = -(edge_probs * edge_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()

        metrics = {
            "active_cells": active_cells.mean(),
            "occupied_cells": occupied_cells.mean(),
            "spawned_cells": (occupied_cells - self.n_seed_cells).clamp_min(0).mean(),
            "spawn_pressure": spawn_prob.mean(),
            "edge_density": (soft_edges / active_cells.pow(2).clamp_min(1.0)).mean(),
            "soft_edges": soft_edges.mean(),
            "development_stability": (states - prev_states).pow(2).mean(),
            "op_entropy": torch.distributions.Categorical(logits=logits).entropy().mean(),
            "cell_type_entropy": torch.distributions.Categorical(logits=cell_type_logits).entropy().mean(),
            "adjacency_entropy": adjacency_entropy,
            "halt_probability": halt_prob.mean(),
            "development_steps": halt_prob.new_tensor(float(steps_taken)),
            "estimated_flops": estimated_flops.mean(),
            "compute_proxy": compute_proxy.mean(),
        }
        return MCNCoreOutput(
            pooled,
            states,
            adjacency,
            gates,
            logits,
            halt_prob,
            metrics,
            occupancy=occupancy,
            cell_type_logits=cell_type_logits,
            development_trace=trace,
        )


class MCNClassifier(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int, config: MCNConfig, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.config = config
        self.token_emb = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.randn(config.max_seq_len, config.d_model) * 0.02)
        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.core = MorphogenicCore(config)
        self.readout = nn.Linear(config.d_model, n_classes)

    def encode(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tokens.shape[1] > self.config.max_seq_len:
            raise ValueError(f"sequence length {tokens.shape[1]} exceeds max_seq_len={self.config.max_seq_len}")
        mask = tokens.ne(self.pad_id)
        positions = self.pos_emb[: tokens.shape[1]].unsqueeze(0)
        memory = self.encoder_norm(self.token_emb(tokens) + positions)
        masked = memory * mask.unsqueeze(-1)
        seed = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return seed, memory, mask

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, MCNCoreOutput]:
        seed, memory, mask = self.encode(tokens)
        core_out = self.core(seed, memory, mask)
        return self.readout(core_out.pooled), core_out


def _n_heads(d_model: int) -> int:
    for heads in (8, 4, 2):
        if d_model % heads == 0:
            return heads
    return 1


class UniversalSequenceEncoder(nn.Module):
    """Shared-depth Transformer encoder, a compact Universal Transformer proxy."""

    def __init__(self, vocab_size: int, config: MCNConfig, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.config = config
        self.token_emb = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.randn(config.max_seq_len, config.d_model) * 0.02)
        self.layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=_n_heads(config.d_model),
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tokens.shape[1] > self.config.max_seq_len:
            raise ValueError(f"sequence length {tokens.shape[1]} exceeds max_seq_len={self.config.max_seq_len}")
        mask = tokens.ne(self.pad_id)
        memory = self.token_emb(tokens) + self.pos_emb[: tokens.shape[1]].unsqueeze(0)
        for _ in range(self.config.n_dev_steps):
            memory = self.layer(memory, src_key_padding_mask=~mask)
        memory = self.norm(memory)
        seed = (memory * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return seed, memory, mask


class TransformerSequenceEncoder(nn.Module):
    """Standard fixed-topology Transformer encoder baseline."""

    def __init__(self, vocab_size: int, config: MCNConfig, pad_id: int = 0, n_layers: int = 2):
        super().__init__()
        self.pad_id = pad_id
        self.config = config
        self.token_emb = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.randn(config.max_seq_len, config.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=_n_heads(config.d_model),
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tokens.shape[1] > self.config.max_seq_len:
            raise ValueError(f"sequence length {tokens.shape[1]} exceeds max_seq_len={self.config.max_seq_len}")
        mask = tokens.ne(self.pad_id)
        memory = self.token_emb(tokens) + self.pos_emb[: tokens.shape[1]].unsqueeze(0)
        memory = self.encoder(memory, src_key_padding_mask=~mask)
        memory = self.norm(memory)
        seed = (memory * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return seed, memory, mask


class MoEBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float, n_experts: int = 4):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 2, d_model),
                )
                for _ in range(n_experts)
            ]
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.router(states), dim=-1)
        expert_outputs = torch.stack([expert(states) for expert in self.experts], dim=-2)
        return (expert_outputs * weights.unsqueeze(-1)).sum(dim=-2)


class PonderPooler(nn.Module):
    """Small differentiable adaptive-depth pooler inspired by PonderNet."""

    def __init__(self, d_model: int, n_steps: int):
        super().__init__()
        self.n_steps = n_steps
        self.update = nn.GRUCell(d_model, d_model)
        self.halt = nn.Linear(d_model, 1)

    def forward(self, seed: torch.Tensor) -> torch.Tensor:
        hidden = seed
        remainder = seed.new_ones(seed.shape[0], 1)
        pooled = torch.zeros_like(seed)
        for step in range(self.n_steps):
            hidden = self.update(seed, hidden)
            halt_prob = torch.sigmoid(self.halt(hidden))
            if step == self.n_steps - 1:
                weight = remainder
            else:
                weight = torch.minimum(halt_prob, remainder)
            pooled = pooled + weight * hidden
            remainder = (remainder - weight).clamp_min(0.0)
        return pooled


class DEQPooler(nn.Module):
    """Fixed-point iteration proxy for Deep Equilibrium style baselines."""

    def __init__(self, d_model: int, n_steps: int):
        super().__init__()
        self.n_steps = n_steps
        self.f = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.Tanh(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, seed: torch.Tensor) -> torch.Tensor:
        hidden = seed
        for _ in range(self.n_steps):
            hidden = self.norm(hidden + self.f(torch.cat([hidden, seed], dim=-1)))
        return hidden


class FixedClassifierBaseline(nn.Module):
    """Fixed-topology classifier baseline with the same encoder scale."""

    def __init__(self, vocab_size: int, n_classes: int, config: MCNConfig, pad_id: int = 0):
        super().__init__()
        if config.d_model % 2 != 0:
            raise ValueError("d_model must be even for the bidirectional GRU encoder")
        self.pad_id = pad_id
        self.config = config
        self.token_emb = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.randn(config.max_seq_len, config.d_model) * 0.02)
        self.encoder = nn.GRU(config.d_model, config.d_model // 2, batch_first=True, bidirectional=True)
        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.readout = nn.Linear(config.d_model, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] > self.config.max_seq_len:
            raise ValueError(f"sequence length {tokens.shape[1]} exceeds max_seq_len={self.config.max_seq_len}")
        mask = tokens.ne(self.pad_id)
        positions = self.pos_emb[: tokens.shape[1]].unsqueeze(0)
        memory, _ = self.encoder(self.token_emb(tokens) + positions)
        memory = self.encoder_norm(memory)
        pooled = (memory * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.readout(pooled)


class TransformerClassifierBaseline(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int, config: MCNConfig, pad_id: int = 0):
        super().__init__()
        self.encoder = TransformerSequenceEncoder(vocab_size, config, pad_id=pad_id)
        self.readout = nn.Linear(config.d_model, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seed, _, _ = self.encoder(tokens)
        return self.readout(seed)


class UniversalClassifierBaseline(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int, config: MCNConfig, pad_id: int = 0):
        super().__init__()
        self.encoder = UniversalSequenceEncoder(vocab_size, config, pad_id=pad_id)
        self.readout = nn.Linear(config.d_model, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seed, _, _ = self.encoder(tokens)
        return self.readout(seed)


class MoEClassifierBaseline(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int, config: MCNConfig, pad_id: int = 0):
        super().__init__()
        if config.d_model % 2 != 0:
            raise ValueError("d_model must be even for the bidirectional GRU encoder")
        self.pad_id = pad_id
        self.config = config
        self.token_emb = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.randn(config.max_seq_len, config.d_model) * 0.02)
        self.encoder = nn.GRU(config.d_model, config.d_model // 2, batch_first=True, bidirectional=True)
        self.moe = MoEBlock(config.d_model, config.dropout)
        self.norm = nn.LayerNorm(config.d_model)
        self.readout = nn.Linear(config.d_model, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] > self.config.max_seq_len:
            raise ValueError(f"sequence length {tokens.shape[1]} exceeds max_seq_len={self.config.max_seq_len}")
        mask = tokens.ne(self.pad_id)
        memory, _ = self.encoder(self.token_emb(tokens) + self.pos_emb[: tokens.shape[1]].unsqueeze(0))
        memory = self.norm(memory + self.moe(memory))
        pooled = (memory * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.readout(pooled)


class PonderClassifierBaseline(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int, config: MCNConfig, pad_id: int = 0):
        super().__init__()
        self.base = FixedClassifierBaseline(vocab_size, n_classes, config, pad_id=pad_id)
        self.pooler = PonderPooler(config.d_model, config.n_dev_steps)
        self.base.readout = nn.Identity()
        self.readout = nn.Linear(config.d_model, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seed = self.base(tokens)
        return self.readout(self.pooler(seed))


class DEQClassifierBaseline(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int, config: MCNConfig, pad_id: int = 0):
        super().__init__()
        self.base = FixedClassifierBaseline(vocab_size, n_classes, config, pad_id=pad_id)
        self.pooler = DEQPooler(config.d_model, config.n_dev_steps)
        self.base.readout = nn.Identity()
        self.readout = nn.Linear(config.d_model, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seed = self.base(tokens)
        return self.readout(self.pooler(seed))


class MCNSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, config: MCNConfig, pad_id: int = 0):
        super().__init__()
        if config.d_model % 2 != 0:
            raise ValueError("d_model must be even for the bidirectional GRU encoder")
        self.pad_id = pad_id
        self.config = config
        self.src_emb = nn.Embedding(src_vocab_size, config.d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, config.d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.randn(config.max_seq_len, config.d_model) * 0.02)
        self.encoder = nn.GRU(config.d_model, config.d_model // 2, batch_first=True, bidirectional=True)
        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.core = MorphogenicCore(config)
        self.decoder_init = nn.Linear(config.d_model, config.d_model)
        self.decoder = nn.GRU(config.d_model, config.d_model, batch_first=True)
        self.decoder_query = nn.Linear(config.d_model, config.d_model)
        self.decoder_fuse = nn.Linear(config.d_model * 3, config.d_model)
        self.output = nn.Linear(config.d_model, tgt_vocab_size)

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, MCNCoreOutput]:
        if src.shape[1] > self.config.max_seq_len:
            raise ValueError(f"sequence length {src.shape[1]} exceeds max_seq_len={self.config.max_seq_len}")
        mask = src.ne(self.pad_id)
        positions = self.pos_emb[: src.shape[1]].unsqueeze(0)
        embedded = self.src_emb(src) + positions
        memory, _ = self.encoder(embedded)
        memory = self.encoder_norm(memory)
        seed = (memory * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        core_out = self.core(seed, memory, mask)
        return core_out.pooled, memory, mask, core_out

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> tuple[torch.Tensor, MCNCoreOutput]:
        graph_state, memory, mask, core_out = self.encode(src)
        hidden = torch.tanh(self.decoder_init(graph_state)).unsqueeze(0)
        decoded, _ = self.decoder(self.tgt_emb(tgt_in), hidden)
        context = self._decoder_context(decoded, memory, mask)
        fused = torch.tanh(self.decoder_fuse(torch.cat([decoded, context, graph_state.unsqueeze(1).expand_as(decoded)], dim=-1)))
        return self.output(fused), core_out

    def _decoder_context(self, decoded: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(self.decoder_query(decoded), memory.transpose(-1, -2)) / math.sqrt(decoded.shape[-1])
        scores = scores.masked_fill(~mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, memory)

    @torch.no_grad()
    def generate(self, src: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 32) -> torch.Tensor:
        self.eval()
        graph_state, memory, mask, _ = self.encode(src)
        hidden = torch.tanh(self.decoder_init(graph_state)).unsqueeze(0)
        token = torch.full((src.shape[0], 1), bos_id, dtype=torch.long, device=src.device)
        outputs = []
        for _ in range(max_len):
            decoded, hidden = self.decoder(self.tgt_emb(token[:, -1:]), hidden)
            context = self._decoder_context(decoded, memory, mask)
            fused = torch.tanh(
                self.decoder_fuse(torch.cat([decoded, context, graph_state.unsqueeze(1).expand_as(decoded)], dim=-1))
            )
            next_token = self.output(fused[:, -1]).argmax(dim=-1, keepdim=True)
            outputs.append(next_token)
            token = torch.cat([token, next_token], dim=1)
            if bool(next_token.eq(eos_id).all()):
                break
        return torch.cat(outputs, dim=1) if outputs else token[:, :0]


class TransformerSeq2SeqBaseline(nn.Module):
    """Standard encoder-decoder Transformer baseline for SCAN-style sequence tasks."""

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, config: MCNConfig, pad_id: int = 0, n_layers: int = 2):
        super().__init__()
        self.pad_id = pad_id
        self.config = config
        self.src_emb = nn.Embedding(src_vocab_size, config.d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, config.d_model, padding_idx=pad_id)
        self.src_pos_emb = nn.Parameter(torch.randn(config.max_seq_len, config.d_model) * 0.02)
        self.tgt_pos_emb = nn.Parameter(torch.randn(config.max_seq_len, config.d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=_n_heads(config.d_model),
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=_n_heads(config.d_model),
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, tgt_vocab_size)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if src.shape[1] > self.config.max_seq_len:
            raise ValueError(f"sequence length {src.shape[1]} exceeds max_seq_len={self.config.max_seq_len}")
        src_mask = src.ne(self.pad_id)
        memory = self.src_emb(src) + self.src_pos_emb[: src.shape[1]].unsqueeze(0)
        memory = self.encoder(memory, src_key_padding_mask=~src_mask)
        return self.norm(memory), src_mask

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        if tgt_in.shape[1] > self.config.max_seq_len:
            raise ValueError(f"target length {tgt_in.shape[1]} exceeds max_seq_len={self.config.max_seq_len}")
        memory, src_mask = self.encode(src)
        tgt_mask = tgt_in.ne(self.pad_id)
        target = self.tgt_emb(tgt_in) + self.tgt_pos_emb[: tgt_in.shape[1]].unsqueeze(0)
        decoded = self.decoder(
            target,
            memory,
            tgt_mask=self._causal_mask(tgt_in.shape[1], tgt_in.device),
            tgt_key_padding_mask=~tgt_mask,
            memory_key_padding_mask=~src_mask,
        )
        return self.output(decoded)

    @torch.no_grad()
    def generate(self, src: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 32) -> torch.Tensor:
        self.eval()
        memory, src_mask = self.encode(src)
        tokens = torch.full((src.shape[0], 1), bos_id, dtype=torch.long, device=src.device)
        outputs = []
        for _ in range(max_len):
            if tokens.shape[1] > self.config.max_seq_len:
                break
            target = self.tgt_emb(tokens) + self.tgt_pos_emb[: tokens.shape[1]].unsqueeze(0)
            decoded = self.decoder(
                target,
                memory,
                tgt_mask=self._causal_mask(tokens.shape[1], tokens.device),
                memory_key_padding_mask=~src_mask,
            )
            next_token = self.output(decoded[:, -1]).argmax(dim=-1, keepdim=True)
            outputs.append(next_token)
            tokens = torch.cat([tokens, next_token], dim=1)
            if bool(next_token.eq(eos_id).all()):
                break
        return torch.cat(outputs, dim=1) if outputs else tokens[:, :0]


class FixedSeq2SeqBaseline(nn.Module):
    """Fixed-topology encoder-decoder baseline for compositional splits."""

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, config: MCNConfig, pad_id: int = 0):
        super().__init__()
        if config.d_model % 2 != 0:
            raise ValueError("d_model must be even for the bidirectional GRU encoder")
        self.pad_id = pad_id
        self.config = config
        self.src_emb = nn.Embedding(src_vocab_size, config.d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, config.d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.randn(config.max_seq_len, config.d_model) * 0.02)
        self.encoder = nn.GRU(config.d_model, config.d_model // 2, batch_first=True, bidirectional=True)
        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.decoder_init = nn.Linear(config.d_model, config.d_model)
        self.decoder = nn.GRU(config.d_model, config.d_model, batch_first=True)
        self.decoder_query = nn.Linear(config.d_model, config.d_model)
        self.decoder_fuse = nn.Linear(config.d_model * 2, config.d_model)
        self.output = nn.Linear(config.d_model, tgt_vocab_size)

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if src.shape[1] > self.config.max_seq_len:
            raise ValueError(f"sequence length {src.shape[1]} exceeds max_seq_len={self.config.max_seq_len}")
        mask = src.ne(self.pad_id)
        positions = self.pos_emb[: src.shape[1]].unsqueeze(0)
        memory, _ = self.encoder(self.src_emb(src) + positions)
        memory = self.encoder_norm(memory)
        seed = (memory * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return seed, memory, mask

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        seed, memory, mask = self.encode(src)
        hidden = torch.tanh(self.decoder_init(seed)).unsqueeze(0)
        decoded, _ = self.decoder(self.tgt_emb(tgt_in), hidden)
        context = self._decoder_context(decoded, memory, mask)
        fused = torch.tanh(self.decoder_fuse(torch.cat([decoded, context], dim=-1)))
        return self.output(fused)

    def _decoder_context(self, decoded: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(self.decoder_query(decoded), memory.transpose(-1, -2)) / math.sqrt(decoded.shape[-1])
        scores = scores.masked_fill(~mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, memory)

    @torch.no_grad()
    def generate(self, src: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 32) -> torch.Tensor:
        self.eval()
        seed, memory, mask = self.encode(src)
        hidden = torch.tanh(self.decoder_init(seed)).unsqueeze(0)
        token = torch.full((src.shape[0], 1), bos_id, dtype=torch.long, device=src.device)
        outputs = []
        for _ in range(max_len):
            decoded, hidden = self.decoder(self.tgt_emb(token[:, -1:]), hidden)
            context = self._decoder_context(decoded, memory, mask)
            fused = torch.tanh(self.decoder_fuse(torch.cat([decoded, context], dim=-1)))
            next_token = self.output(fused[:, -1]).argmax(dim=-1, keepdim=True)
            outputs.append(next_token)
            token = torch.cat([token, next_token], dim=1)
            if bool(next_token.eq(eos_id).all()):
                break
        return torch.cat(outputs, dim=1) if outputs else token[:, :0]


class UniversalSeq2SeqBaseline(FixedSeq2SeqBaseline):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, config: MCNConfig, pad_id: int = 0):
        super().__init__(src_vocab_size, tgt_vocab_size, config, pad_id=pad_id)
        self.universal_encoder = UniversalSequenceEncoder(src_vocab_size, config, pad_id=pad_id)

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.universal_encoder(src)


class MoESeq2SeqBaseline(FixedSeq2SeqBaseline):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, config: MCNConfig, pad_id: int = 0):
        super().__init__(src_vocab_size, tgt_vocab_size, config, pad_id=pad_id)
        self.moe = MoEBlock(config.d_model, config.dropout)

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if src.shape[1] > self.config.max_seq_len:
            raise ValueError(f"sequence length {src.shape[1]} exceeds max_seq_len={self.config.max_seq_len}")
        mask = src.ne(self.pad_id)
        positions = self.pos_emb[: src.shape[1]].unsqueeze(0)
        memory, _ = self.encoder(self.src_emb(src) + positions)
        memory = self.encoder_norm(memory + self.moe(memory))
        seed = (memory * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return seed, memory, mask


class PonderSeq2SeqBaseline(FixedSeq2SeqBaseline):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, config: MCNConfig, pad_id: int = 0):
        super().__init__(src_vocab_size, tgt_vocab_size, config, pad_id=pad_id)
        self.pooler = PonderPooler(config.d_model, config.n_dev_steps)

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seed, memory, mask = super().encode(src)
        return self.pooler(seed), memory, mask


class DEQSeq2SeqBaseline(FixedSeq2SeqBaseline):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, config: MCNConfig, pad_id: int = 0):
        super().__init__(src_vocab_size, tgt_vocab_size, config, pad_id=pad_id)
        self.pooler = DEQPooler(config.d_model, config.n_dev_steps)

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seed, memory, mask = super().encode(src)
        return self.pooler(seed), memory, mask


def mcn_regularization(
    core_out: MCNCoreOutput,
    compute_weight: float = 1e-3,
    stability_weight: float = 1e-2,
    diversity_weight: float = 1e-4,
    sparsity_weight: float = 1e-3,
) -> torch.Tensor:
    n_cells = float(core_out.adjacency.shape[-1])
    edge_cost = core_out.metrics.get("soft_edges", core_out.metrics["edge_density"] * n_cells * n_cells)
    compute = core_out.metrics["active_cells"] / n_cells
    compute = compute + edge_cost / max(n_cells * n_cells, 1.0)
    compute = compute + core_out.metrics.get("spawned_cells", core_out.metrics["active_cells"].new_tensor(0.0)) / n_cells
    stability = core_out.metrics["development_stability"]
    diversity = -core_out.metrics["op_entropy"]
    sparsity = core_out.adjacency.abs().mean()
    return (
        compute_weight * compute
        + stability_weight * stability
        + diversity_weight * diversity
        + sparsity_weight * sparsity
    )
