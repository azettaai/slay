#!/usr/bin/env python3
"""Matched-seed kernel/output fidelity for exact factors and SLAY estimators."""

from __future__ import annotations

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def normalize(x):
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def normalized_attention(kernel, value):
    weights = kernel / jnp.maximum(kernel.sum(axis=-1, keepdims=True), 1e-8)
    return jnp.einsum("bhqs,bhsd->bhqd", weights, value)


def errors(estimate, target):
    delta = estimate - target
    return {
        "relative_frobenius": float(
            jnp.linalg.norm(delta) / jnp.maximum(jnp.linalg.norm(target), 1e-8)
        ),
        "mae": float(jnp.mean(jnp.abs(delta))),
        "max_abs": float(jnp.max(jnp.abs(delta))),
    }


def cartesian_features(x, key, p, m, r):
    anchor_key, omega_key = jax.random.split(key)
    _, heads, _, dimension = x.shape
    anchors = normalize(jax.random.normal(anchor_key, (p, dimension)))
    omega = jax.random.normal(omega_key, (r, heads, dimension, m))
    nodes_np, weights_np = np.polynomial.laguerre.laggauss(r)
    nodes = jnp.asarray(nodes_np / (2.0 + 1e-6), jnp.float32)
    weights = jnp.asarray(weights_np / (2.0 + 1e-6), jnp.float32)
    polynomial = jnp.square(jnp.einsum("bhtd,pd->bhtp", x, anchors))
    projection = jnp.einsum("bhtd,rhdm->rbhtm", x, omega)
    prf = (
        jnp.exp(
            jnp.clip(
                jnp.sqrt(2.0 * nodes[:, None, None, None, None]) * projection
                - nodes[:, None, None, None, None],
                -10.0,
                10.0,
            )
        )
        * jnp.sqrt(weights[:, None, None, None, None])
    )
    return jnp.einsum("bhtp,rbhtm->bhtrpm", polynomial, prf).reshape(
        *x.shape[:-1], -1
    )


def flat_features(x, key, width):
    anchor_key, omega_key, scale_key = jax.random.split(key, 3)
    _, heads, _, dimension = x.shape
    anchors = normalize(jax.random.normal(anchor_key, (width, dimension)))
    omega = jax.random.normal(omega_key, (heads, dimension, width))
    scales = jax.random.exponential(scale_key, (width,)) / (2.0 + 1e-6)
    polynomial = jnp.square(jnp.einsum("bhtd,jd->bhtj", x, anchors))
    projection = jnp.einsum("bhtd,hdj->bhtj", x, omega)
    return polynomial * jnp.exp(
        jnp.clip(
            jnp.sqrt(2.0 * scales[None, None, None, :]) * projection
            - scales[None, None, None, :],
            -10.0,
            10.0,
        )
    )


def main():
    length = int(os.environ.get("LENGTH", "256"))
    heads = int(os.environ.get("HEADS", "4"))
    dimension = int(os.environ.get("HEAD_DIM", "16"))
    seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
    flat_widths = [
        int(x) for x in os.environ.get("FLAT_WIDTHS", "32,64,128,256").split(",")
    ]
    cartesian = [(8, 4, 2), (32, 16, 2)]
    rows = []
    for seed in seeds:
        data_key, feature_key = jax.random.split(jax.random.key(seed))
        q_key, k_key, v_key = jax.random.split(data_key, 3)
        shape = (1, heads, length, dimension)
        q = normalize(jax.random.normal(q_key, shape))
        k = normalize(jax.random.normal(k_key, shape))
        value = jax.random.normal(v_key, shape)
        similarity = jnp.einsum("bhqd,bhsd->bhqs", q, k)
        causal = jnp.tril(jnp.ones((length, length), dtype=jnp.float32))
        polynomial = jnp.square(similarity) * causal
        proximity = (1.0 / (2.0 + 1e-6 - 2.0 * similarity)) * causal
        full = polynomial * proximity
        targets = {
            "polynomial_only": normalized_attention(polynomial, value),
            "proximity_only": normalized_attention(proximity, value),
            "exact_spherical_yat": normalized_attention(full, value),
        }
        rows.append(
            {
                "seed": seed,
                "method": "factor_difference",
                "features": None,
                "output_error_to_full": {
                    name: errors(output, targets["exact_spherical_yat"])
                    for name, output in targets.items()
                    if name != "exact_spherical_yat"
                },
            }
        )
        for p, m, r in cartesian:
            q_feature_key, k_feature_key = jax.random.split(feature_key)
            # Q and K must share the same frozen random state.
            qf = cartesian_features(q, feature_key, p, m, r)
            kf = cartesian_features(k, feature_key, p, m, r)
            kernel = jnp.einsum("bhqf,bhsf->bhqs", qf, kf) * causal
            output = normalized_attention(kernel, value)
            rows.append(
                {
                    "seed": seed,
                    "method": f"cartesian_P{p}_M{m}_R{r}",
                    "features": p * m * r,
                    "kernel_error": errors(kernel, full),
                    "output_error": errors(
                        output, targets["exact_spherical_yat"]
                    ),
                }
            )
        for width in flat_widths:
            qf = flat_features(q, feature_key, width)
            kf = flat_features(k, feature_key, width)
            kernel = jnp.einsum("bhqf,bhsf->bhqs", qf, kf) * causal
            output = normalized_attention(kernel, value)
            rows.append(
                {
                    "seed": seed,
                    "method": f"flat_joint_J{width}",
                    "features": width,
                    "kernel_error": errors(kernel, full),
                    "output_error": errors(
                        output, targets["exact_spherical_yat"]
                    ),
                }
            )
    summary = {}
    for method in sorted({row["method"] for row in rows if "output_error" in row}):
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            "features": selected[0]["features"],
            "output_relative_frobenius_mean": float(
                np.mean([r["output_error"]["relative_frobenius"] for r in selected])
            ),
            "output_relative_frobenius_std": float(
                np.std([r["output_error"]["relative_frobenius"] for r in selected])
            ),
        }
    result = {
        "environment": {
            "backend": jax.default_backend(),
            "device_kinds": [d.device_kind for d in jax.devices()],
            "length": length,
            "heads": heads,
            "head_dim": dimension,
            "seeds": seeds,
        },
        "summary": summary,
        "rows": rows,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/kernel_fidelity.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
