# Morphogenic Computation Network Prototype

This repository now includes a runnable MCN v0.1 research prototype under `mcn/`.

It implements the practical first phase of the proposal:

- Fixed-cap developmental substrate with learned cell states.
- Learned proliferation into dormant cell slots, starting from a small seed cell set.
- Differentiation through operation logits.
- Learned soft connectivity between cells.
- Apoptosis through continuous prune gates.
- Development halting signal for inference-time early stop.
- Graph execution over the grown adjacency matrix, so the learned topology affects the final representation.
- Multiple graph-execution steps over the grown topology.
- Sparse compute, stability, diversity, and connectivity regularization.
- Optional development traces for plotting active cells, graph density, and halting probability across developmental steps.
- Optional Causal Geometric Binding (CGB) attention using a Euclidean Clifford algebra `Cl(p, 0)`, with higher-grade (bivector+) binding forwarded through the network.
- Cosine LR schedule with eta_min at 1% of base LR.
- Curriculum warmup: first 10% of training uses constrained morphogenesis (dense template adjacency, no proliferation/pruning/differentiation), ramping to full freedom by 30%.
- A toy SCAN-like compositional split and train/evaluate script.
- Standard Transformer, fixed GRU, Universal Transformer-style, MoE, Ponder-style, and DEQ-style baselines for local comparison.
- Ablation switches for proliferation, pruning/apoptosis, differentiation, halting, learned development signals, and template graph bias.
- JSONL experiment logs, CSV curves, prediction dumps, compute profiles, checkpoints, DOT graph exports, and PNG visualizations for submission artifacts.

## Files

```text
mcn/clifford.py       Dense Clifford algebra and geometric product
mcn/experiment.py     Evaluation, prediction logging, and graph export helpers
mcn/config_io.py      YAML/JSON config loading and CLI flattening
mcn/model.py          Morphogenic core, classifier, seq2seq model, CGB attention
mcn/synthetic_tasks.py Local binding, CLEVR-style, length, arithmetic, adaptive-compute, next-token, and scientific tasks
mcn/toy_scan.py       Small SCAN-like compositional dataset generator
mcn/visualization.py  Graph, heatmap, operation, trace, and compute-profile PNG plotting
configs/*.yaml        Reproducible MCN and baseline configs
scripts/train_mcn_toy.py
scripts/train_mcn_synthetic.py
scripts/run_mcn_config.py
scripts/run_mcn_suite.py
scripts/run_mcn_roadmap.py
scripts/visualize_mcn_run.py
tests/test_mcn_*.py
```

## Run A Smoke Training Loop

```bash
pip install -r requirements-mcn.txt
python scripts/train_mcn_toy.py --epochs 2 --d-model 64 --n-cells 16 --n-seed-cells 4 --n-dev-steps 4
```

To enable geometric-product attention:

```bash
python scripts/train_mcn_toy.py --epochs 2 --use-cgb
```

To train the standard Transformer baseline:

```bash
python scripts/train_mcn_toy.py --model transformer --epochs 2 --d-model 64
```

Other local baselines:

```bash
python scripts/train_mcn_toy.py --model baseline --epochs 2
python scripts/train_mcn_toy.py --model universal --epochs 2
python scripts/train_mcn_toy.py --model moe --epochs 2
python scripts/train_mcn_toy.py --model ponder --epochs 2
python scripts/train_mcn_toy.py --model deq --epochs 2
```

To run from a config:

```bash
python scripts/run_mcn_config.py configs/mcn_scan.yaml --epochs 2 --d-model 64
python scripts/run_mcn_config.py configs/transformer_baseline.yaml --epochs 2 --d-model 64
```

Important ablations:

```bash
python scripts/train_mcn_toy.py --disable-proliferation --epochs 2
python scripts/train_mcn_toy.py --disable-pruning --epochs 2
python scripts/train_mcn_toy.py --disable-differentiation --epochs 2
python scripts/train_mcn_toy.py --disable-halting --epochs 2
python scripts/train_mcn_toy.py --random-development --epochs 2
python scripts/train_mcn_toy.py --template-adjacency chain --epochs 2
```

To run a comparable suite:

```bash
python scripts/run_mcn_suite.py --epochs 30 --eval-items 0 --include-baselines --include-ablations --run-name final-suite
```

The default split is the hard SCAN-style `jump` composition holdout. For a sanity-learning split where every primitive appears compositionally in both train and test, use:

```bash
python scripts/run_mcn_suite.py --split random --epochs 30 --eval-items 0 --run-name sanity-suite
```

Each run writes artifacts under `runs/mcn_toy/<timestamp>-<variant>/`:

```text
config.json        Arguments, MCN config, vocabularies, and split sizes
config.yaml        YAML-compatible config snapshot
history.csv        Per-epoch scalar metrics
history.jsonl      Per-epoch loss, exact match, and MCN topology metrics
train_loss.csv     Training loss curve
val_loss.csv       Validation loss curve
exact_match.csv    Exact-match or accuracy curve
active_cells.csv   Active and occupied cell curve
graph_density.csv  Edge density and soft-edge curve
operation_distribution.csv  Final sample operation mix
predictions.jsonl  Command-level expected/predicted outputs
compute_profile.json  Per-example adaptive-compute summary for MCN runs
compute_profile.jsonl Per-example active cells, edge density, entropy, and FLOPs proxy
sample_graph.dot   Graphviz DOT export of one developed MCN graph
graphs/sample_graph.png  Rendered final grown graph
plots/*.png        Adjacency, operation, development trace, and compute plots
summary.json       Final and best exact-match metrics for the selected split
checkpoint_best.pt Best checkpoint by held-out exact match or accuracy
checkpoint_last.pt Last checkpoint
```

## What This Is

This is a working, differentiable MCN prototype that exercises the full path:

```text
input tokens -> seed embedding -> morphogenic development -> graph execution over learned adjacency -> decoded output
```

It is intentionally small enough to run locally and to test. It is not a claim that the 1B-parameter scaling roadmap is solved.

## Key Design Choices

- The implementation keeps a fixed maximum number of tensor slots for GPU-friendly batching, but only the seed slots start occupied. Proliferation is implemented as learned, differentiable activation of dormant slots.
- Dynamic graph choices are represented as differentiable soft adjacency, prune gates, and Gumbel-Softmax operation mixtures during training. Evaluation uses deterministic softmax routing by default, or hard argmax routing when `hard_ops=True`.
- After differentiation chooses per-cell operations, each cell receives messages from its learned adjacency neighborhood for `n_exec_steps` before pooling. This makes the grown graph a causal part of the executed computation, not just an interpretability byproduct.
- CGB is implemented as an attention operation: query/key projections are interpreted as multivectors, the scalar grade of the geometric product gives the attention logit, and higher-grade components (bivectors, trivectors) encoding relational binding structure are projected and added to the output, forwarding compositional information through the network.
- Both training scripts use a cosine annealing LR schedule (eta_min = 1% of base LR) and curriculum warmup. During the first 10% of training, morphogenic freedom is fully constrained (dense template adjacency, no proliferation, no pruning, uniform operation weights). Freedom linearly ramps from 0 to 1 between 10% and 30% of training, reaching full morphogenic freedom for the remaining 70%.
- The toy split holds out composed uses of `jump`, creating a small compositional generalization check similar in spirit to SCAN.

## Useful Metrics Exposed By The Core

- `active_cells`
- `occupied_cells`
- `spawned_cells`
- `spawn_pressure`
- `edge_density`
- `soft_edges`
- `development_stability`
- `op_entropy`
- `cell_type_entropy`
- `adjacency_entropy`
- `halt_probability`
- `development_steps`
- `estimated_flops`
- `compute_proxy`

These can be logged during experiments to see whether the architecture is actually adapting its computation.

## Submission Checklist

```bash
python -m pytest -q
python scripts/run_mcn_suite.py --split random --epochs 30 --eval-items 0 --run-name sanity-suite
python scripts/run_mcn_suite.py --split jump --epochs 30 --eval-items 0 --run-name final-hard-jump
```

For the report, cite `suite_summary.json` for side-by-side final numbers, each run's `history.jsonl` for learning curves, `predictions.jsonl` for qualitative examples, and `sample_graph.dot` for a visual explanation of the grown computation graph.

## Roadmap Coverage

The original 24-month plan is represented here as a complete local-scale implementation suite. It does not claim billion-parameter training, but every research track has runnable code, metrics, and artifacts:

| Roadmap item | Local implementation |
|:---|:---|
| MCN v0.1 proof of concept | `mcn/model.py`, `scripts/train_mcn_toy.py` |
| Differentiation, pruning, proliferation, halting | `MorphogenicCore` metrics in every MCN run |
| CGB attention | `CGBInputAttentionOp`, `--use-cgb` |
| Compositional generalization | `--split jump` toy SCAN-style holdout |
| Sanity/generalization split | `--split random` toy SCAN split |
| Transformer baseline | `--model transformer` |
| Fixed GRU baseline | `FixedSeq2SeqBaseline`, `FixedClassifierBaseline` |
| Universal Transformer baseline | `--model universal` |
| MoE baseline | `--model moe` |
| PonderNet-style baseline | `--model ponder` |
| DEQ-style baseline | `--model deq` |
| Ablation: no proliferation | `--disable-proliferation` |
| Ablation: no apoptosis/pruning | `--disable-pruning` |
| Ablation: no differentiation | `--disable-differentiation` |
| Ablation: fixed development steps | `--disable-halting` |
| Ablation: random development signals | `--random-development` |
| Ablation: template initialization/bias | `--template-adjacency chain` |
| Binding benchmark | `scripts/train_mcn_synthetic.py --task binding` |
| CLEVR-style binding benchmark | `scripts/train_mcn_synthetic.py --task clevr_binding` |
| Length generalization | `scripts/train_mcn_synthetic.py --task length` |
| Arithmetic length generalization | `scripts/train_mcn_synthetic.py --task arithmetic` |
| Adaptive compute easy/hard mix | `scripts/train_mcn_synthetic.py --task adaptive_compute` |
| Small language-modeling surrogate | `scripts/train_mcn_synthetic.py --task next_token` |
| Scientific-computing surrogate | `scripts/train_mcn_synthetic.py --task scientific` |
| Compute/FLOPs distribution artifacts | `compute_profile.json`, `compute_profile.jsonl`, `plots/compute_profile.png` |
| Graph visualizations | `graphs/sample_graph.png`, `plots/adjacency_heatmap.png`, `plots/development_trace.png`, `plots/operation_distribution.png` |
| Full roadmap artifact suite | `scripts/run_mcn_roadmap.py` |

Run the local roadmap suite:

```bash
python scripts/run_mcn_roadmap.py --epochs 12 --run-name final-roadmap
```

For a fast plumbing check:

```bash
python scripts/run_mcn_roadmap.py --quick --run-name smoke-roadmap
```

Use `--minimal` for the smaller legacy matrix. The full default suite writes `roadmap_summary.json` plus per-run `summary.json`, CSV curves, `history.jsonl`, `predictions.jsonl`, `compute_profile.json`, `compute_profile.jsonl`, checkpoints, and MCN graph/plot artifacts.
