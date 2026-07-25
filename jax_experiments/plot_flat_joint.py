#!/usr/bin/env python3
"""Plot Cartesian versus flat-joint SLAY cost and fidelity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--fidelity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile = json.loads(args.profile.read_text())
    fidelity = json.loads(args.fidelity.read_text())

    lengths = [row["length"] for row in profile["rows"]]
    cartesian = [
        row["module_forward"]["parallel_prefix"]["median_ms"]
        for row in profile["rows"]
    ]
    flat = [
        row["flat_joint_module_forward"]["parallel_prefix"]["median_ms"]
        for row in profile["rows"]
    ]
    performer = [
        row["performer_module_forward"]["parallel_prefix"]["median_ms"]
        for row in profile["rows"]
    ]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(lengths, cartesian, "o-", label="Cartesian SLAY, F=1024")
    ax.plot(lengths, flat, "o-", label="Flat joint SLAY, J=64")
    ax.plot(lengths, performer, "o-", label="Performer, F=64")
    ax.set(xlabel="Context length", ylabel="Forward latency (ms)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "p100-flat-joint-latency.png", dpi=180)
    plt.close(fig)

    order = [
        "flat_joint_J32",
        "flat_joint_J64",
        "flat_joint_J128",
        "flat_joint_J256",
        "cartesian_P8_M4_R2",
        "cartesian_P32_M16_R2",
    ]
    labels, widths, means, stds = [], [], [], []
    for method in order:
        row = fidelity["summary"][method]
        labels.append(method.replace("flat_joint_", "flat ").replace("cartesian_", "cart. "))
        widths.append(row["features"])
        means.append(row["output_relative_frobenius_mean"])
        stds.append(row["output_relative_frobenius_std"])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for label, width, mean, std in zip(labels, widths, means, stds):
        ax.errorbar(width, mean, yerr=std, marker="o", capsize=3, label=label)
    ax.set_xscale("log", base=2)
    ax.set(xlabel="Recurrent feature width", ylabel="Relative output error")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "p100-flat-joint-fidelity.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
