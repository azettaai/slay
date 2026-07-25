#!/usr/bin/env python3
"""Profile anchor SLAY cost versus feature width and render rebuttal figures."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from jax_experiments.benchmark_reviewer_jax import anchor_stage_breakdown, measure
from slay_jax.anchors import create_feature_state
from slay_jax.attention import (
    exact_spherical_yat_attention,
    streaming_slay_attention,
)


def parse_sweeps(poly_dims, prf_dims, fixed_poly, fixed_prf):
    configs = [
        {"sweep": "anchors P", "poly_dim": p, "prf_dim": fixed_prf}
        for p in poly_dims
    ]
    configs += [
        {"sweep": "PRF M", "poly_dim": fixed_poly, "prf_dim": m}
        for m in prf_dims
        if not (fixed_poly == m == fixed_prf)
    ]
    return configs


def plot_cost(rows, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for sweep, marker in (("anchors P", "o"), ("PRF M", "s")):
        selected = sorted(
            [row for row in rows if row["sweep"] == sweep],
            key=lambda row: row["feature_dim"],
        )
        axes[0].plot(
            [row["feature_dim"] for row in selected],
            [row["forward"]["steady_median_ms"] for row in selected],
            marker=marker,
            label=f"{sweep}, forward",
        )
        axes[0].plot(
            [row["feature_dim"] for row in selected],
            [row["forward_backward"]["steady_median_ms"] for row in selected],
            marker=marker,
            linestyle="--",
            label=f"{sweep}, forward+backward",
        )
        axes[1].plot(
            [row["feature_dim"] for row in selected],
            [
                row["forward_backward"]["compiler_memory"]["temp_size_in_bytes"]
                / 2**20
                for row in selected
            ],
            marker=marker,
            label=sweep,
        )
    axes[0].set(xlabel="Total feature width F=R×P×M", ylabel="Median latency (ms)")
    axes[1].set(xlabel="Total feature width F=R×P×M", ylabel="Compiler temporary memory (MiB)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_stages(rows, output):
    stage_labels = {
        "normalization_qk": "Q/K normalization",
        "anchor_polynomial_qk": "anchor projection + square",
        "laplace_prf_qk": "Laplace PRF",
        "tensor_fusion_qk_materialized_diagnostic": "P×M tensor fusion",
        "scan_from_precomputed_features_diagnostic": "causal recurrence",
    }
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    selected = sorted(rows, key=lambda row: row["feature_dim"])
    for key, label in stage_labels.items():
        axis.plot(
            [row["feature_dim"] for row in selected],
            [row["stages"][key]["steady_median_ms"] for row in selected],
            marker="o",
            label=label,
        )
    axis.set(
        xlabel="Total feature width F=R×P×M",
        ylabel="Isolated median stage time (ms)",
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_quality_cost(rows, output):
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    for sweep, marker in (("anchors P", "o"), ("PRF M", "s")):
        selected = sorted(
            [row for row in rows if row["sweep"] == sweep],
            key=lambda row: row["feature_dim"],
        )
        axis.plot(
            [row["forward"]["steady_median_ms"] for row in selected],
            [row["output_relative_l2"] for row in selected],
            marker=marker,
            label=sweep,
        )
        for index, row in enumerate(selected):
            if sweep == "PRF M" and index == len(selected) - 1:
                continue
            axis.annotate(
                f"F={row['feature_dim']}",
                (
                    row["forward"]["steady_median_ms"],
                    row["output_relative_l2"],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
    axis.set(
        xlabel="Forward median (ms)",
        ylabel="Relative L2 error vs exact spherical Yat",
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--quadrature", type=int, default=2)
    parser.add_argument("--poly-dims", nargs="+", type=int, default=[4, 8, 16, 32, 64])
    parser.add_argument("--prf-dims", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--fixed-poly", type=int, default=32)
    parser.add_argument("--fixed-prf", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, required=True)
    args = parser.parse_args()

    shape = (args.batch, args.heads, args.length, args.head_dim)
    q_key, k_key, v_key = jax.random.split(jax.random.key(args.seed), 3)
    q = jax.random.normal(q_key, shape)
    k = jax.random.normal(k_key, shape)
    v = jax.random.normal(v_key, shape)
    configs = parse_sweeps(
        args.poly_dims, args.prf_dims, args.fixed_poly, args.fixed_prf
    )
    state_start = time.perf_counter()
    full_state = create_feature_state(
        seed=args.seed,
        num_layers=1,
        num_heads=args.heads,
        head_dim=args.head_dim,
        poly_dim=max(config["poly_dim"] for config in configs),
        prf_dim=max(config["prf_dim"] for config in configs),
        num_quadrature=args.quadrature,
    )
    jax.block_until_ready(full_state)
    initialization_ms = 1000.0 * (time.perf_counter() - state_start)
    exact_output = exact_spherical_yat_attention(
        q, k, v, causal=True
    )
    jax.block_until_ready(exact_output)
    rows = []

    for config in configs:
        state = full_state._replace(
            anchors=full_state.anchors[:, : config["poly_dim"]],
            omega=full_state.omega[..., : config["prf_dim"]],
        )

        def forward(q_arg, k_arg, v_arg):
            return streaming_slay_attention(
                q_arg, k_arg, v_arg, state, variant="anchor"
            )

        def forward_backward(q_arg, k_arg, v_arg):
            return jax.value_and_grad(
                lambda qv, kv, vv: jnp.mean(jnp.square(forward(qv, kv, vv))),
                argnums=(0, 1, 2),
            )(q_arg, k_arg, v_arg)

        identity = jnp.eye(args.heads * args.head_dim)
        stages = anchor_stage_breakdown(
            q=q,
            k=k,
            v=v,
            state=state,
            output_weight=identity,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        approximate_output = forward(q, k, v)
        relative_l2 = jnp.linalg.norm(
            approximate_output - exact_output
        ) / jnp.linalg.norm(exact_output)
        row = config | {
            "feature_dim": args.quadrature
            * config["poly_dim"]
            * config["prf_dim"],
            "output_relative_l2": float(relative_l2),
            "forward": measure(
                forward,
                (q, k, v),
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "forward_backward": measure(
                forward_backward,
                (q, k, v),
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "stages": stages,
        }
        rows.append(row)
        print(
            f"{config['sweep']}: P={config['poly_dim']} M={config['prf_dim']} "
            f"F={row['feature_dim']} forward={row['forward']['steady_median_ms']:.3f}ms",
            flush=True,
        )

    payload = {
        "protocol": {
            key: value if not isinstance(value, Path) else str(value)
            for key, value in vars(args).items()
        },
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__,
        "one_time_random_state_initialization_ms": initialization_ms,
        "random_bank_policy": "one maximum-size frozen bank; smaller configurations use nested prefixes",
        "rows": rows,
        "stage_note": "isolated JIT timings are diagnostic and are not additive because the full graph may fuse operations",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    plot_cost(rows, args.plot_dir / "anchor-cost-vs-features.png")
    plot_stages(
        [row for row in rows if row["sweep"] == "anchors P"],
        args.plot_dir / "anchor-stage-time-vs-features.png",
    )
    plot_quality_cost(
        rows,
        args.plot_dir / "anchor-quality-vs-cost.png",
    )
    print(f"wrote {args.output} and plots in {args.plot_dir}")


if __name__ == "__main__":
    main()
