#!/usr/bin/env python3
"""Audit the population target of squared-projection anchor features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument("--anchors", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    key = jax.random.key(args.seed)
    q_key, orthogonal_key, anchor_key = jax.random.split(key, 3)
    q = jax.random.normal(q_key, (args.dimension,))
    q /= jnp.linalg.norm(q)
    orthogonal = jax.random.normal(orthogonal_key, (args.dimension,))
    orthogonal -= jnp.dot(orthogonal, q) * q
    orthogonal /= jnp.linalg.norm(orthogonal)
    anchors = jax.random.normal(anchor_key, (args.anchors, args.dimension))
    anchors /= jnp.linalg.norm(anchors, axis=-1, keepdims=True)

    rows = []
    for cosine in (-1.0, -0.5, 0.0, 0.5, 1.0):
        k = cosine * q + np.sqrt(max(0.0, 1.0 - cosine**2)) * orthogonal
        empirical = jnp.mean(jnp.square(anchors @ q) * jnp.square(anchors @ k))
        population = (1.0 + 2.0 * cosine**2) / (
            args.dimension * (args.dimension + 2)
        )
        rescaled = args.dimension * (args.dimension + 2) * empirical / 3.0
        rows.append(
            {
                "cosine": cosine,
                "target_x_squared": cosine**2,
                "anchor_inner_product": float(empirical),
                "population_formula": population,
                "diagonal_rescaled_anchor": float(rescaled),
            }
        )

    output = {
        "dimension": args.dimension,
        "num_anchors": args.anchors,
        "identity": (
            "E_a[(a^Tq)^2(a^Tk)^2] = "
            "(||q||^2||k||^2 + 2(q^Tk)^2)/(d(d+2))"
        ),
        "conclusion": (
            "For unit q,k, diagonal rescaling targets (1+2x^2)/3, not x^2."
        ),
        "rows": rows,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/anchor_bias_audit.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
