# SLAY: Spherical Linearized Attention with Yat-Kernel

A PyTorch implementation of spherical linearized attention mechanisms with various polynomial kernel approximations for efficient transformer models.

## Canonical JAX path

Reviewer-facing SLAY experiments now live in the pure JAX package
[`slay_jax/`](slay_jax/) with runnable scripts under
[`jax_experiments/`](jax_experiments/). Anchors and PRF projections are generated
once from explicit seeds and frozen, causal attention uses `jax.lax.scan`
without materializing prefix states, and timing separates XLA compilation from
synchronized steady-state execution.

See [`docs/JAX_CANONICAL_IMPLEMENTATION.md`](docs/JAX_CANONICAL_IMPLEMENTATION.md)
for the anchor contract, corrected synthetic-task protocols, profiling rules,
and local commands. The older PyTorch code remains a reproduction reference for
the submitted numbers.

## Overview

This repository contains implementations of:

1. **Exact Attention Mechanisms**
   - Standard softmax attention (O(L²))
   - Exact spherical Yat attention (O(L²))

2. **Linearized Attention Mechanisms** (O(L))
   - Performer (FAVOR+ with ReLU features)
   - Linear attention (ELU+1)
   - Cosformer
   - Random Fourier Features (RFF)

3. **Spherical Yat-Performer Variants** (O(L))
   - **Anchor** - Low-rank anchor features (recommended)
   - **Laplace-only** - PRF without polynomial factor
   - **Hadamard** - Shared ω Hadamard fusion
   - **TensorSketch** - FFT-based polynomial sketch
   - **Random Maclaurin** - Random projection features
   - **Nyström** - Nyström approximation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/slay.git
cd slay

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy
```

## Project Structure

```
slay/
├── src/
│   ├── attention/           # Attention implementations
│   │   ├── standard.py      # Standard softmax attention
│   │   ├── performer.py     # FAVOR+ linear attention
│   │   ├── linear.py        # ELU+1 linear attention
│   │   ├── cosformer.py     # Cosformer attention
│   │   ├── rff.py           # Random Fourier Features
│   │   ├── yat.py           # Exact Yat attention
│   │   ├── yat_spherical.py # Exact spherical Yat
│   │   ├── yat_performer.py # Linearized Yat (Hadamard)
│   │   ├── yat_performer_laplace.py  # Laplace-only PRF
│   │   ├── yat_performer_tensor.py   # TensorSketch
│   │   └── yat_performer_poly_alt.py # RM, Nyström, Anchor
│   ├── models/              # Model architectures
│   │   ├── gpt.py           # GPT-style decoder
│   │   └── blocks.py        # Transformer blocks
│   ├── activations.py       # Custom activation functions
│   ├── config.py            # Configuration classes
│   └── data.py              # Data utilities
├── tests/                   # Benchmarks and tests
│   ├── benchmark_scaling.py       # Latency/memory scaling
│   ├── benchmark_kernel_quality.py # Approximation quality
│   ├── benchmark_tasks.py         # 22 synthetic tasks (attention-only)
│   ├── ablation_poly_approx.py    # Polynomial ablation
│   ├── run_all_benchmarks.py      # Master benchmark runner
│   └── generate_paper_tables.py   # LaTeX table generator
├── docs/                    # Documentation
│   └── BENCHMARK_TASKS.md   # Full task suite description
├── tables/                  # Generated LaTeX tables
├── artifacts/               # Benchmark results (JSON)
├── main.py                  # Training script
└── main.tex                 # Paper source
```

## Attention Variants

| Name | Type | Complexity | Description |
|------|------|------------|-------------|
| `standard` | Exact | O(L²) | Standard softmax attention |
| `yat-spherical` | Exact | O(L²) | Exact spherical Yat kernel |
| `performer` | Linear | O(L) | FAVOR+ (ReLU features) |
| `linear` | Linear | O(L) | ELU+1 linear attention |
| `cosformer` | Linear | O(L) | Cosformer with cos reweighting |
| `yat-performer-anchor` | Linear | O(L) | **Recommended** - Anchor features |
| `yat-performer-laplace` | Linear | O(L) | Laplace-only (no poly factor) |
| `yat-performer` | Linear | O(L) | Hadamard fusion (shared ω) |
| `yat-performer-tensor` | Linear | O(L) | TensorSketch polynomial |
| `yat-performer-rm` | Linear | O(L) | Random Maclaurin polynomial |
| `yat-performer-nystrom` | Linear | O(L) | Nyström approximation |

## Benchmarks

### Quick Start

```bash
# Run all benchmarks (quick mode, ~10 min)
python tests/run_all_benchmarks.py --quick --device cuda

# Run all benchmarks (full mode, ~2-4 hours)
python tests/run_all_benchmarks.py --full --device cuda
```

### Individual Benchmarks

#### 1. Scaling Benchmark (Latency & Memory)

Measures attention latency and memory usage across sequence lengths.

```bash
# Forward-only
python tests/benchmark_scaling.py --device cuda

# Forward + backward
python tests/benchmark_scaling.py --device cuda --backward

# Quick mode (fewer sequence lengths)
python tests/benchmark_scaling.py --device cuda --quick

# Custom settings
python tests/benchmark_scaling.py --device cuda \
    --embed-dim 128 --n-heads 8 --batch-size 4 \
    --attentions standard yat-performer-anchor yat-performer-laplace
```

**Output:** `tables/latency_scaling_*.tex`, `tables/memory_scaling_*.tex`

#### 2. Kernel Approximation Quality

Measures how well linearized versions approximate the exact kernel.

```bash
# Basic run
python tests/benchmark_kernel_quality.py --device cuda

# With feature budget sweep
python tests/benchmark_kernel_quality.py --device cuda --sweep

# Custom settings
python tests/benchmark_kernel_quality.py --device cuda \
    --seq-len 512 --batch-size 16 --embed-dim 128
```

**Metrics:**
- Relative L2 error: `||approx - exact|| / ||exact||`
- Cosine similarity: `cos(approx, exact)`
- MSE: `mean((approx - exact)²)`

**Output:** `tables/kernel_quality.tex`, `artifacts/kernel_quality.json`

#### 3. Synthetic Task Benchmarks (22 Tasks)

Comprehensive suite testing attention quality on controlled learning tasks using **attention-only architectures** (no FFN, no LayerNorm) to isolate attention performance.

```bash
# All 22 tasks (3 seeds each)
python tests/benchmark_tasks.py --device cuda

# Quick mode (50 epochs, 1 seed)
python tests/benchmark_tasks.py --device cuda --quick

# Specific tasks
python tests/benchmark_tasks.py --device cuda --tasks copy sort retrieval

# By category
python tests/benchmark_tasks.py --device cuda --category memory

# Attention-critical tasks only
python tests/benchmark_tasks.py --device cuda --attention-only
```

**Model Architecture:**
```
Input → Embedding + PosEmbed → [Attention + Residual] × L → Linear Head → Output
```

**Task Categories (22 tasks):**

| Category | Tasks | Tests |
|----------|-------|-------|
| **Basic** | copy, sort, reverse | Information routing |
| **Memory** | retrieval, kv_recall, first_token, selective_copy | Sparse retrieval, associative memory |
| **Long-Range** | long_copy, distant_match, multihop | Long-range dependencies |
| **Reasoning** | stack, induction, pattern | State tracking, pattern matching |
| **Arithmetic** | counting, parity, addition, modular | Aggregation |
| **Pattern** | bigram, majority | Statistical patterns |
| **Robustness** | noisy_copy, compression | Noise filtering |
| **Aggregation** | histogram | Multi-class counting |

See [docs/BENCHMARK_TASKS.md](docs/BENCHMARK_TASKS.md) for full task descriptions.

**Output:** `tables/task_*.tex`, `artifacts/task_results.json`

#### 4. Polynomial Approximation Ablation

Compares different polynomial kernel approximation methods.

```bash
# Full sweep
python tests/ablation_poly_approx.py --device cuda --sweep

# Quick mode
python tests/ablation_poly_approx.py --device cuda --sweep --quick
```

**Output:** `tables/poly_ablation_sweep.tex`, `artifacts/poly_ablation_sweep.md`

### Generate Paper Tables

After running benchmarks, generate LaTeX tables:

```bash
python tests/generate_paper_tables.py
```

This reads from `artifacts/` and produces formatted tables in `tables/`.

## Usage Example

### Using Attention Modules Directly

```python
from src.attention import get_attention_class

# Get attention class by name
AttentionClass = get_attention_class('yat-performer-anchor')

# Create attention module
attn = AttentionClass(
    embed_dim=64,
    n_heads=4,
    num_prf_features=16,
    num_quadrature_nodes=2,
    poly_dim=32,
    epsilon=1e-6,
)

# Forward pass
import torch
x = torch.randn(2, 128, 64)  # (batch, seq_len, embed_dim)
out = attn(x)  # (2, 128, 64)
```

### Training a Model

Train a GPT-style model with configurable attention:

```bash
# Basic training with DeepSpeed (uses all available GPUs by default)
# To specify number of GPUs, use --num_gpus=X BEFORE main.py
deepspeed --num_gpus=8 main.py --attention standard --context-len 1024

# If 'deepspeed' command is not found, run as module:
python3 -m deepspeed.launcher.runner --num_gpus=8 main.py --attention standard --context-len 1024

# Train with Yat-Performer (linear complexity)
deepspeed --num_gpus=8 main.py --attention yat-performer-anchor --context-len 2048

# Full configuration example
deepspeed --num_gpus=8 main.py \
    --attention cosformer \
    --context-len 1024 \
    --embed-dim 768 \
    --n-layers 12 \
    --n-heads 12 \
    --lr 3e-4 \
    --warmup-steps 2000 \
    --total-steps 20000 \
    --dropout 0.1 \
    --batch-size 64 \
    --gradient-accumulation-steps 1
```

**Key Training Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--attention` | `performer` | Attention type (see table above) |
| `--context-len` | 1024 | Context window length |
| `--embed-dim` | 768 | Embedding dimension |
| `--n-layers` | 12 | Number of transformer layers |
| `--n-heads` | 12 | Number of attention heads |
| `--lr` | 3e-4 | Learning rate |
| `--warmup-steps` | 2000 | LR warmup steps (linear ramp) |
| `--total-steps` | 20000 | Total training steps |
| `--dropout` | 0.1 | Dropout rate (attention + MLP + embeddings) |
| `--batch-size` | 64 | Micro batch size per GPU |
| `--use-triton` | False | Use Triton-accelerated CUDA kernels |

**Training Features:**
- LR warmup with cosine decay (WarmupDecayLR)
- Dropout on attention, MLP, and embeddings
- Gradient checkpointing for memory efficiency
- FP16 mixed precision with dynamic loss scaling
- ZeRO-2 optimizer for distributed training
- Gradient clipping (1.0)

## Results

### Kernel Approximation Quality

| Method | Rel L2 ↓ | Cosine ↑ |
|--------|----------|----------|
| yat-performer-anchor | 0.54 | 0.84 |
| yat-performer-laplace | 0.55 | 0.83 |
| yat-performer | 0.79 | 0.72 |

### Scaling (Forward-only @ L=4096)

| Method | Latency | Memory | Complexity |
|--------|---------|--------|------------|
| standard | 18.7ms | 543MB | O(L²) |
| yat-spherical | 32.9ms | 1.0GB | O(L²) |
| yat-performer-laplace | 17.1ms | 56MB | O(L) |
| linear | 2.2ms | 62MB | O(L) |

## Citation

```bibtex
@article{slay2026,
  title={SLAY: Geometry-Aware Spherical Linearized Attention with Yat-Kernel},
  author={...},
  journal={...},
  year={2026}
}
```

## License

MIT License
