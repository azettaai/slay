#!/usr/bin/env python3
"""Self-contained Kaggle TPU/GPU profile for persistent-anchor SLAY."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def integers(name: str, default: str) -> list[int]:
    return [int(value) for value in os.environ.get(name, default).split(",")]


def normalize(x):
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def features(x, anchors, omega, nodes, weights):
    x = normalize(x)
    polynomial = jnp.square(jnp.einsum("bhtd,pd->bhtp", x, anchors))
    polynomial /= jnp.sqrt(anchors.shape[0])
    projection = jnp.einsum("bhtd,rhdm->rbhtm", x, omega)
    exponent = jnp.clip(
        jnp.sqrt(2.0 * nodes[:, None, None, None, None]) * projection
        - nodes[:, None, None, None, None],
        -10.0,
        10.0,
    )
    prf = jnp.exp(exponent) / jnp.sqrt(omega.shape[-1])
    prf *= jnp.sqrt(weights[:, None, None, None, None])
    fused = jnp.einsum("bhtp,rbhtm->bhtrpm", polynomial, prf)
    return fused.reshape(*x.shape[:-1], -1)


def flat_joint_features(x, anchors, omega, scales, denominator_constant):
    """Jointly sample anchor, Laplace scale, and PRF in J positive features."""
    x = normalize(x)
    polynomial = jnp.square(jnp.einsum("bhtd,jd->bhtj", x, anchors))
    projection = jnp.einsum("bhtd,hdj->bhtj", x, omega)
    exponent = jnp.clip(
        jnp.sqrt(2.0 * scales[None, None, None, :]) * projection
        - scales[None, None, None, :],
        -10.0,
        10.0,
    )
    return (
        polynomial
        * jnp.exp(exponent)
        / jnp.sqrt(denominator_constant * anchors.shape[0])
    )


def causal(qf, kf, value):
    qf, kf, value = (jnp.moveaxis(x, 2, 0) for x in (qf, kf, value))
    batch, heads, feature_dim = qf.shape[1:]
    value_dim = value.shape[-1]
    initial = (
        jnp.zeros((batch, heads, feature_dim, value_dim), jnp.float32),
        jnp.zeros((batch, heads, feature_dim), jnp.float32),
    )

    def step(carry, inputs):
        kv, ks = carry
        q_t, k_t, v_t = (x.astype(jnp.float32) for x in inputs)
        kv = kv + jnp.einsum("bhf,bhd->bhfd", k_t, v_t)
        ks = ks + k_t
        numerator = jnp.einsum("bhf,bhfd->bhd", q_t, kv)
        denominator = jnp.einsum("bhf,bhf->bh", q_t, ks)
        output = numerator / jnp.maximum(denominator[..., None], 1e-6)
        return (kv, ks), output

    _, output = jax.lax.scan(jax.checkpoint(step), initial, (qf, kf, value))
    return jnp.moveaxis(output, 0, 2)


def causal_parallel_prefix(qf, kf, value):
    """Fully parallel prefix reference; faster hardware may trade HBM for speed."""
    kv_prefix = jnp.cumsum(
        jnp.einsum("bhtf,bhtd->bhtfd", kf, value), axis=2
    )
    k_prefix = jnp.cumsum(kf, axis=2)
    numerator = jnp.einsum("bhtf,bhtfd->bhtd", qf, kv_prefix)
    denominator = jnp.einsum("bhtf,bhtf->bht", qf, k_prefix)
    return numerator / jnp.maximum(denominator[..., None], 1e-6)


def measure(fn, arguments, warmup, repeats):
    compiled = jax.jit(fn).lower(*arguments).compile()
    compile_memory = compiled.memory_analysis()
    for _ in range(warmup):
        jax.block_until_ready(compiled(*arguments))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*arguments))
        samples.append(1000.0 * (time.perf_counter() - start))
    return {
        "median_ms": float(np.median(samples)),
        "min_ms": float(np.min(samples)),
        "samples_ms": samples,
        "compiler_memory": (
            {
                key: int(getattr(compile_memory, key))
                for key in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "temp_size_in_bytes",
                )
            }
            if compile_memory is not None
            else None
        ),
    }


def main():
    lengths = integers("LENGTHS", "128,256")
    poly_dims = integers("POLY_DIMS", "8,32")
    prf_dim = int(os.environ.get("PRF_DIM", "16"))
    quadrature = int(os.environ.get("QUADRATURE", "2"))
    batch = int(os.environ.get("BATCH", "1"))
    heads = int(os.environ.get("HEADS", "4"))
    head_dim = int(os.environ.get("HEAD_DIM", "16"))
    warmup = int(os.environ.get("WARMUP", "1"))
    repeats = int(os.environ.get("REPEATS", "3"))
    seed = int(os.environ.get("SEED", "42"))
    profile_module = os.environ.get("PROFILE_MODULE", "0") == "1"
    profile_backward = os.environ.get("PROFILE_BACKWARD", "0") == "1"
    profile_performer = os.environ.get("PROFILE_PERFORMER", "0") == "1"
    profile_exact = os.environ.get("PROFILE_EXACT", "0") == "1"
    profile_paired = os.environ.get("PROFILE_PAIRED", "0") == "1"
    joint_features = int(os.environ.get("JOINT_FEATURES", "32"))
    performer_features = int(os.environ.get("PERFORMER_FEATURES", "64"))
    key = jax.random.key(seed)
    rows = []

    laguerre_nodes, laguerre_weights = np.polynomial.laguerre.laggauss(quadrature)
    nodes = jnp.asarray(laguerre_nodes / (2.0 + 1e-6), jnp.float32)
    weights = jnp.asarray(laguerre_weights / (2.0 + 1e-6), jnp.float32)

    for length in lengths:
        data_key, key = jax.random.split(key)
        q_key, k_key, v_key = jax.random.split(data_key, 3)
        shape = (batch, heads, length, head_dim)
        q = jax.random.normal(q_key, shape)
        k = jax.random.normal(k_key, shape)
        v = jax.random.normal(v_key, shape)
        for poly_dim in poly_dims:
            state_key, key = jax.random.split(key)
            anchor_key, omega_key = jax.random.split(state_key)
            anchors = jax.random.normal(anchor_key, (poly_dim, head_dim))
            anchors /= jnp.linalg.norm(anchors, axis=-1, keepdims=True)
            omega = jax.random.normal(
                omega_key, (quadrature, heads, head_dim, prf_dim)
            )
            flat_key, key = jax.random.split(key)
            flat_anchor_key, flat_omega_key, flat_scale_key = jax.random.split(
                flat_key, 3
            )
            flat_anchors = jax.random.normal(
                flat_anchor_key, (joint_features, head_dim)
            )
            flat_anchors /= jnp.linalg.norm(
                flat_anchors, axis=-1, keepdims=True
            )
            flat_omega = jax.random.normal(
                flat_omega_key, (heads, head_dim, joint_features)
            )
            denominator_constant = 2.0 + 1e-6
            flat_scales = (
                jax.random.exponential(flat_scale_key, (joint_features,))
                / denominator_constant
            )

            def forward_scan(q_arg, k_arg, v_arg):
                return causal(
                    features(q_arg, anchors, omega, nodes, weights),
                    features(k_arg, anchors, omega, nodes, weights),
                    v_arg,
                )

            def forward_parallel(q_arg, k_arg, v_arg):
                return causal_parallel_prefix(
                    features(q_arg, anchors, omega, nodes, weights),
                    features(k_arg, anchors, omega, nodes, weights),
                    v_arg,
                )

            def feature_only(q_arg, k_arg):
                return (
                    features(q_arg, anchors, omega, nodes, weights),
                    features(k_arg, anchors, omega, nodes, weights),
                )

            qf, kf = feature_only(q, k)
            jax.block_until_ready((qf, kf))
            recurrence = {
                "streaming_scan": measure(
                    causal, (qf, kf, v), warmup, repeats
                )
            }
            end_to_end = {
                "streaming_scan": measure(
                    forward_scan, (q, k, v), warmup, repeats
                )
            }
            if os.environ.get("PARALLEL_PREFIX", "1") == "1":
                recurrence["parallel_prefix"] = measure(
                    causal_parallel_prefix, (qf, kf, v), warmup, repeats
                )
                end_to_end["parallel_prefix"] = measure(
                    forward_parallel, (q, k, v), warmup, repeats
                )
            module_metrics = None
            backward_metrics = None
            performer_module_metrics = None
            performer_backward_metrics = None
            softmax_module_metrics = None
            softmax_backward_metrics = None
            exact_yat_module_metrics = None
            exact_yat_backward_metrics = None
            flat_module_metrics = None
            flat_backward_metrics = None
            if profile_module:
                parameter_key, key = jax.random.split(key)
                qkv_key, output_key = jax.random.split(parameter_key)
                embed_dim = heads * head_dim
                parameters = {
                    "qkv": jax.random.normal(
                        qkv_key, (embed_dim, 3 * embed_dim)
                    )
                    / jnp.sqrt(embed_dim),
                    "output": jax.random.normal(
                        output_key, (embed_dim, embed_dim)
                    )
                    / jnp.sqrt(embed_dim),
                }
                x = jax.random.normal(data_key, (batch, length, embed_dim))

                def module(params, inputs, recurrence_fn):
                    qkv = jnp.einsum("bte,ef->btf", inputs, params["qkv"])
                    q_value, k_value, v_value = jnp.split(qkv, 3, axis=-1)

                    def split_heads(value):
                        return value.reshape(
                            batch, length, heads, head_dim
                        ).transpose(0, 2, 1, 3)

                    attended = recurrence_fn(
                        features(
                            split_heads(q_value),
                            anchors,
                            omega,
                            nodes,
                            weights,
                        ),
                        features(
                            split_heads(k_value),
                            anchors,
                            omega,
                            nodes,
                            weights,
                        ),
                        split_heads(v_value),
                    )
                    merged = attended.transpose(0, 2, 1, 3).reshape(
                        batch, length, embed_dim
                    )
                    return jnp.einsum(
                        "bte,ef->btf", merged, params["output"]
                    )

                def module_scan(params, inputs):
                    return module(params, inputs, causal)

                def module_parallel(params, inputs):
                    return module(params, inputs, causal_parallel_prefix)

                def flat_module(params, inputs, recurrence_fn):
                    qkv = jnp.einsum("bte,ef->btf", inputs, params["qkv"])
                    q_value, k_value, v_value = jnp.split(qkv, 3, axis=-1)

                    def split_heads(value):
                        return value.reshape(
                            batch, length, heads, head_dim
                        ).transpose(0, 2, 1, 3)

                    def map_features(value):
                        return flat_joint_features(
                            split_heads(value),
                            flat_anchors,
                            flat_omega,
                            flat_scales,
                            denominator_constant,
                        )

                    attended = recurrence_fn(
                        map_features(q_value),
                        map_features(k_value),
                        split_heads(v_value),
                    )
                    merged = attended.transpose(0, 2, 1, 3).reshape(
                        batch, length, embed_dim
                    )
                    return jnp.einsum(
                        "bte,ef->btf", merged, params["output"]
                    )

                def flat_scan(params, inputs):
                    return flat_module(params, inputs, causal)

                def flat_parallel(params, inputs):
                    return flat_module(params, inputs, causal_parallel_prefix)

                module_metrics = {
                    "streaming_scan": measure(
                        module_scan, (parameters, x), warmup, repeats
                    ),
                    "parallel_prefix": measure(
                        module_parallel, (parameters, x), warmup, repeats
                    ),
                }
                if profile_paired:
                    flat_module_metrics = {
                        "streaming_scan": measure(
                            flat_scan, (parameters, x), warmup, repeats
                        ),
                        "parallel_prefix": measure(
                            flat_parallel, (parameters, x), warmup, repeats
                        ),
                    }
                if profile_backward:
                    def backward(fn, params, inputs):
                        return jax.value_and_grad(
                            lambda p, value: jnp.mean(jnp.square(fn(p, value))),
                            argnums=(0, 1),
                        )(params, inputs)

                    backward_metrics = {
                        "streaming_scan": measure(
                            lambda p, value: backward(module_scan, p, value),
                            (parameters, x),
                            warmup,
                            repeats,
                        ),
                        "parallel_prefix": measure(
                            lambda p, value: backward(module_parallel, p, value),
                            (parameters, x),
                            warmup,
                            repeats,
                        ),
                    }
                    if profile_paired:
                        flat_backward_metrics = {
                            "streaming_scan": measure(
                                lambda p, value: backward(
                                    flat_scan, p, value
                                ),
                                (parameters, x),
                                warmup,
                                repeats,
                            ),
                            "parallel_prefix": measure(
                                lambda p, value: backward(
                                    flat_parallel, p, value
                                ),
                                (parameters, x),
                                warmup,
                                repeats,
                            ),
                        }
                if profile_performer:
                    performer_key, key = jax.random.split(key)
                    performer_projection = jax.random.normal(
                        performer_key,
                        (heads, head_dim, performer_features),
                    )

                    def performer_module(params, inputs, recurrence_fn):
                        qkv = jnp.einsum(
                            "bte,ef->btf", inputs, params["qkv"]
                        )
                        q_value, k_value, v_value = jnp.split(
                            qkv, 3, axis=-1
                        )

                        def split_heads(value):
                            return value.reshape(
                                batch, length, heads, head_dim
                            ).transpose(0, 2, 1, 3)

                        q_heads = split_heads(q_value)
                        k_heads = split_heads(k_value)
                        v_heads = split_heads(v_value)
                        q_features = (
                            jax.nn.relu(
                                jnp.einsum(
                                    "bhtd,hdf->bhtf",
                                    q_heads,
                                    performer_projection,
                                )
                            )
                            + 1e-4
                        )
                        k_features = (
                            jax.nn.relu(
                                jnp.einsum(
                                    "bhtd,hdf->bhtf",
                                    k_heads,
                                    performer_projection,
                                )
                            )
                            + 1e-4
                        )
                        attended = recurrence_fn(
                            q_features, k_features, v_heads
                        )
                        merged = attended.transpose(0, 2, 1, 3).reshape(
                            batch, length, embed_dim
                        )
                        return jnp.einsum(
                            "bte,ef->btf", merged, params["output"]
                        )

                    def performer_scan(params, inputs):
                        return performer_module(params, inputs, causal)

                    def performer_parallel(params, inputs):
                        return performer_module(
                            params, inputs, causal_parallel_prefix
                        )

                    performer_module_metrics = {
                        "streaming_scan": measure(
                            performer_scan,
                            (parameters, x),
                            warmup,
                            repeats,
                        ),
                        "parallel_prefix": measure(
                            performer_parallel,
                            (parameters, x),
                            warmup,
                            repeats,
                        ),
                    }
                    if profile_backward:
                        performer_backward_metrics = {
                            "streaming_scan": measure(
                                lambda p, value: backward(
                                    performer_scan, p, value
                                ),
                                (parameters, x),
                                warmup,
                                repeats,
                            ),
                            "parallel_prefix": measure(
                                lambda p, value: backward(
                                    performer_parallel, p, value
                                ),
                                (parameters, x),
                                warmup,
                                repeats,
                            ),
                        }
                if profile_exact:
                    def exact_module(params, inputs, kind):
                        qkv = jnp.einsum(
                            "bte,ef->btf", inputs, params["qkv"]
                        )
                        q_value, k_value, v_value = jnp.split(
                            qkv, 3, axis=-1
                        )

                        def split_heads(value):
                            return value.reshape(
                                batch, length, heads, head_dim
                            ).transpose(0, 2, 1, 3)

                        q_heads = split_heads(q_value)
                        k_heads = split_heads(k_value)
                        v_heads = split_heads(v_value)
                        if kind == "softmax":
                            scores = jnp.einsum(
                                "bhtd,bhsd->bhts", q_heads, k_heads
                            ) / jnp.sqrt(head_dim)
                            scores = jnp.where(
                                jnp.tril(
                                    jnp.ones(
                                        (length, length), dtype=jnp.bool_
                                    )
                                ),
                                scores,
                                jnp.finfo(scores.dtype).min,
                            )
                            attention_weights = jax.nn.softmax(
                                scores, axis=-1
                            )
                        elif kind == "spherical_yat":
                            q_normalized = normalize(q_heads)
                            k_normalized = normalize(k_heads)
                            similarity = jnp.einsum(
                                "bhtd,bhsd->bhts",
                                q_normalized,
                                k_normalized,
                            )
                            attention_weights = jnp.square(similarity) / (
                                2.0 + 1e-6 - 2.0 * similarity
                            )
                            attention_weights = jnp.where(
                                jnp.tril(
                                    jnp.ones(
                                        (length, length), dtype=jnp.bool_
                                    )
                                ),
                                attention_weights,
                                0.0,
                            )
                            attention_weights /= jnp.maximum(
                                jnp.sum(
                                    attention_weights,
                                    axis=-1,
                                    keepdims=True,
                                ),
                                1e-6,
                            )
                        else:
                            raise ValueError(kind)
                        attended = jnp.einsum(
                            "bhts,bhsd->bhtd",
                            attention_weights,
                            v_heads,
                        )
                        merged = attended.transpose(0, 2, 1, 3).reshape(
                            batch, length, embed_dim
                        )
                        return jnp.einsum(
                            "bte,ef->btf", merged, params["output"]
                        )

                    softmax_forward = lambda p, value: exact_module(
                        p, value, "softmax"
                    )
                    exact_yat_forward = lambda p, value: exact_module(
                        p, value, "spherical_yat"
                    )
                    softmax_module_metrics = measure(
                        softmax_forward,
                        (parameters, x),
                        warmup,
                        repeats,
                    )
                    exact_yat_module_metrics = measure(
                        exact_yat_forward,
                        (parameters, x),
                        warmup,
                        repeats,
                    )
                    if profile_backward:
                        softmax_backward_metrics = measure(
                            lambda p, value: backward(
                                softmax_forward, p, value
                            ),
                            (parameters, x),
                            warmup,
                            repeats,
                        )
                        exact_yat_backward_metrics = measure(
                            lambda p, value: backward(
                                exact_yat_forward, p, value
                            ),
                            (parameters, x),
                            warmup,
                            repeats,
                        )
            row = {
                "length": length,
                "poly_dim": poly_dim,
                "prf_dim": prf_dim,
                "quadrature": quadrature,
                "feature_dim": quadrature * poly_dim * prf_dim,
                "feature_construction": measure(
                    feature_only, (q, k), warmup, repeats
                ),
                "causal_recurrence": recurrence,
                "end_to_end_forward": end_to_end,
                "module_forward": module_metrics,
                "module_forward_backward": backward_metrics,
                "performer_feature_dim": (
                    performer_features if profile_performer else None
                ),
                "performer_module_forward": performer_module_metrics,
                "performer_module_forward_backward": performer_backward_metrics,
                "softmax_module_forward": softmax_module_metrics,
                "softmax_module_forward_backward": softmax_backward_metrics,
                "exact_yat_module_forward": exact_yat_module_metrics,
                "exact_yat_module_forward_backward": exact_yat_backward_metrics,
                "flat_joint_feature_dim": (
                    joint_features if profile_paired else None
                ),
                "flat_joint_module_forward": flat_module_metrics,
                "flat_joint_module_forward_backward": flat_backward_metrics,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    metadata = {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "device_kinds": [device.device_kind for device in jax.devices()],
        "environment": {
            "lengths": lengths,
            "poly_dims": poly_dims,
            "prf_dim": prf_dim,
            "quadrature": quadrature,
            "batch": batch,
            "heads": heads,
            "head_dim": head_dim,
            "warmup": warmup,
            "repeats": repeats,
            "seed": seed,
            "profile_module": profile_module,
            "profile_backward": profile_backward,
            "profile_performer": profile_performer,
            "profile_exact": profile_exact,
            "profile_flat_joint": profile_paired,
            "joint_features": joint_features,
            "performer_features": performer_features,
        },
        "rows": rows,
        "note": "Feature and recurrence timings are isolated diagnostics and need not sum to the fused end-to-end graph.",
    }
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "anchor_profile.json").write_text(json.dumps(metadata, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for length in lengths:
        selected = sorted(
            (row for row in rows if row["length"] == length),
            key=lambda row: row["feature_dim"],
        )
        axes[0].plot(
            [row["feature_dim"] for row in selected],
            [
                row["end_to_end_forward"]["streaming_scan"]["median_ms"]
                for row in selected
            ],
            marker="o",
            label=f"L={length}",
        )
        if all(
            "parallel_prefix" in row["end_to_end_forward"] for row in selected
        ):
            axes[0].plot(
                [row["feature_dim"] for row in selected],
                [
                    row["end_to_end_forward"]["parallel_prefix"]["median_ms"]
                    for row in selected
                ],
                marker="^",
                linestyle="--",
                label=f"parallel, L={length}",
            )
        axes[1].plot(
            [row["feature_dim"] for row in selected],
            [row["feature_construction"]["median_ms"] for row in selected],
            marker="o",
            label=f"features, L={length}",
        )
        axes[1].plot(
            [row["feature_dim"] for row in selected],
            [
                row["causal_recurrence"]["streaming_scan"]["median_ms"]
                for row in selected
            ],
            marker="s",
            linestyle="--",
            label=f"recurrence, L={length}",
        )
        if all(
            "parallel_prefix" in row["causal_recurrence"] for row in selected
        ):
            axes[1].plot(
                [row["feature_dim"] for row in selected],
                [
                    row["causal_recurrence"]["parallel_prefix"]["median_ms"]
                    for row in selected
                ],
                marker="^",
                linestyle=":",
                label=f"parallel prefix, L={length}",
            )
    axes[0].set(xlabel="Total feature width F=R×P×M", ylabel="Forward median (ms)")
    axes[1].set(xlabel="Total feature width F=R×P×M", ylabel="Isolated median (ms)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "anchor_profile.png", dpi=180)
    print("wrote results/anchor_profile.json and results/anchor_profile.png")


if __name__ == "__main__":
    main()
