#!/usr/bin/env python3
"""Render clear accelerator figures from ``kaggle_anchor_profile.py`` JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def mib(metric):
    memory = metric.get("compiler_memory")
    return memory["temp_size_in_bytes"] / 2**20 if memory else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    rows = payload["rows"]
    lengths = sorted({row["length"] for row in rows})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, (latency_axis, memory_axis) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = plt.cm.viridis_r([index / max(len(lengths) - 1, 1) for index in range(len(lengths))])
    for color, length in zip(colors, lengths):
        selected = sorted(
            (row for row in rows if row["length"] == length),
            key=lambda row: row["feature_dim"],
        )
        widths = [row["feature_dim"] for row in selected]
        latency_axis.plot(
            widths,
            [
                row["end_to_end_forward"]["streaming_scan"]["median_ms"]
                for row in selected
            ],
            color=color,
            marker="o",
            label=f"scan, L={length}",
        )
        latency_axis.plot(
            widths,
            [
                row["end_to_end_forward"]["parallel_prefix"]["median_ms"]
                for row in selected
            ],
            color=color,
            marker="^",
            linestyle="--",
            label=f"parallel, L={length}",
        )
        memory_axis.plot(
            widths,
            [
                mib(row["end_to_end_forward"]["streaming_scan"])
                for row in selected
            ],
            color=color,
            marker="o",
            label=f"scan, L={length}",
        )
        memory_axis.plot(
            widths,
            [
                mib(row["end_to_end_forward"]["parallel_prefix"])
                for row in selected
            ],
            color=color,
            marker="^",
            linestyle="--",
            label=f"parallel, L={length}",
        )
    latency_axis.set(
        xlabel="Total feature width F=R×P×M",
        ylabel="End-to-end attention forward (ms)",
    )
    memory_axis.set(
        xlabel="Total feature width F=R×P×M",
        ylabel="Compiler temporary memory (MiB)",
    )
    for axis in (latency_axis, memory_axis):
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "p100-anchor-cost-vs-features.png", dpi=180)
    plt.close(fig)

    longest = max(lengths)
    selected = sorted(
        (row for row in rows if row["length"] == longest),
        key=lambda row: row["feature_dim"],
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    widths = [row["feature_dim"] for row in selected]
    for values, label, marker in (
        (
            [row["feature_construction"]["median_ms"] for row in selected],
            "anchor + PRF feature construction",
            "o",
        ),
        (
            [
                row["causal_recurrence"]["streaming_scan"]["median_ms"]
                for row in selected
            ],
            "causal recurrence: scan",
            "s",
        ),
        (
            [
                row["causal_recurrence"]["parallel_prefix"]["median_ms"]
                for row in selected
            ],
            "causal recurrence: parallel prefix",
            "^",
        ),
    ):
        axis.plot(widths, values, marker=marker, label=label)
    axis.set(
        xlabel="Total feature width F=R×P×M",
        ylabel="Isolated median stage time (ms)",
        title=f"P100 stage breakdown at context {longest}",
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "p100-anchor-stage-breakdown.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
