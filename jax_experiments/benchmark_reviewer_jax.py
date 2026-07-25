#!/usr/bin/env python3
"""Faithful, methodologically corrected JAX version of the reviewer benchmark.

The submitted mode reproduces the original feature budgets, including anchor
SLAY's R=2, P=32, M=16 (F=1024) versus Performer F=64. The matched mode uses
F=64 for both. Random features are initialized once and frozen. Compilation,
steady execution, backward, compiler memory, and isolated anchor stages are
reported separately.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from slay_jax.anchors import create_feature_state
from slay_jax.attention import (
    causal_linear_attention,
    exact_softmax_attention,
    exact_spherical_yat_attention,
    streaming_cosformer_attention,
    streaming_elementwise_attention,
    streaming_slay_attention,
)


METHODS = (
    "standard",
    "performer",
    "linear",
    "cosformer",
    "yat-spherical",
    "yat-performer-anchor",
    "yat-performer-laplace",
    "yat-performer",
)


def initialize_parameters(seed: int, embed_dim: int):
    qkv_key, output_key = jax.random.split(jax.random.PRNGKey(seed))
    return {
        "qkv": jax.random.normal(qkv_key, (embed_dim, 3 * embed_dim))
        / jnp.sqrt(embed_dim),
        "output": jax.random.normal(output_key, (embed_dim, embed_dim))
        / jnp.sqrt(embed_dim),
    }


def split_qkv(x, parameters, heads):
    batch, length, embed_dim = x.shape
    qkv = jnp.einsum("bte,ef->btf", x, parameters["qkv"])
    q, k, v = jnp.split(qkv, 3, axis=-1)
    head_dim = embed_dim // heads

    def reshape(array):
        return array.reshape(batch, length, heads, head_dim).transpose(0, 2, 1, 3)

    return reshape(q), reshape(k), reshape(v)


def module_forward(
    parameters,
    x,
    *,
    method,
    heads,
    anchor_state,
    compact_state,
    performer_projection,
):
    q, k, v = split_qkv(x, parameters, heads)
    if method == "standard":
        attended = exact_softmax_attention(q, k, v, causal=True)
    elif method == "performer":
        attended = streaming_elementwise_attention(
            q,
            k,
            v,
            kind="performer_relu",
            performer_projection=performer_projection,
        )
    elif method == "linear":
        attended = streaming_elementwise_attention(q, k, v, kind="linear_elu")
    elif method == "cosformer":
        attended = streaming_cosformer_attention(q, k, v)
    elif method == "yat-spherical":
        attended = exact_spherical_yat_attention(q, k, v, causal=True)
    elif method == "yat-performer-anchor":
        attended = streaming_slay_attention(q, k, v, anchor_state, variant="anchor")
    elif method == "yat-performer-laplace":
        attended = streaming_slay_attention(q, k, v, compact_state, variant="laplace")
    elif method == "yat-performer":
        attended = streaming_slay_attention(q, k, v, compact_state, variant="hadamard")
    else:
        raise ValueError(method)
    batch, _, length, _ = attended.shape
    merged = attended.transpose(0, 2, 1, 3).reshape(batch, length, -1)
    return jnp.einsum("bte,ef->btf", merged, parameters["output"])


def _memory_analysis(compiled):
    try:
        memory = compiled.memory_analysis()
        return {
            name: int(getattr(memory, name))
            for name in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
                "generated_code_size_in_bytes",
            )
            if hasattr(memory, name)
        }
    except Exception as error:
        return {"error": str(error)}


def measure(function, arguments, *, warmup, repeats):
    start = time.perf_counter()
    lowered = jax.jit(function).lower(*arguments)
    lowering_ms = 1000.0 * (time.perf_counter() - start)
    start = time.perf_counter()
    compiled = lowered.compile()
    compile_ms = 1000.0 * (time.perf_counter() - start)
    start = time.perf_counter()
    jax.block_until_ready(compiled(*arguments))
    first_execution_ms = 1000.0 * (time.perf_counter() - start)
    for _ in range(warmup):
        jax.block_until_ready(compiled(*arguments))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*arguments))
        samples.append(1000.0 * (time.perf_counter() - start))
    return {
        "lowering_ms": lowering_ms,
        "compile_ms": compile_ms,
        "first_execution_ms": first_execution_ms,
        "steady_median_ms": float(np.median(samples)),
        "steady_min_ms": float(np.min(samples)),
        "steady_samples_ms": samples,
        "compiler_memory": _memory_analysis(compiled),
    }


def feature_configuration(mode):
    if mode == "submitted":
        return {
            "anchor_poly_dim": 32,
            "anchor_prf_dim": 16,
            "quadrature": 2,
            "performer_features": 64,
        }
    if mode == "matched":
        return {
            "anchor_poly_dim": 8,
            "anchor_prf_dim": 4,
            "quadrature": 2,
            "performer_features": 64,
        }
    raise ValueError(mode)


def feature_dim(method, config, head_dim):
    return {
        "standard": None,
        "performer": config["performer_features"],
        "linear": head_dim,
        "cosformer": 2 * head_dim,
        "yat-spherical": None,
        "yat-performer-anchor": (
            config["quadrature"]
            * config["anchor_poly_dim"]
            * config["anchor_prf_dim"]
        ),
        "yat-performer-laplace": 2 * 32,
        "yat-performer": 2 * 32,
    }[method]


def analytical_memory(length, *, batch, heads, head_dim, features):
    fp32 = 4
    result = {
        "qkv_bytes": batch * heads * length * head_dim * 3 * fp32,
        "output_bytes": batch * heads * length * head_dim * fp32,
    }
    if features is None:
        result["quadratic_score_matrix_bytes"] = batch * heads * length * length * fp32
    else:
        result["recurrent_state_bytes"] = (
            batch * heads * features * (head_dim + 1) * fp32
        )
        result["materialized_qk_feature_pair_bytes_avoided_by_fused_scan"] = (
            2 * batch * heads * length * features * fp32
        )
    return result


def anchor_stage_breakdown(
    *,
    q,
    k,
    v,
    state,
    output_weight,
    warmup,
    repeats,
):
    anchors = state.anchors[0]
    omega = state.omega[0]
    nodes = state.nodes[:, None, None, None, None]
    weights = state.weights[:, None, None, None, None]

    def normalize_pair(q_arg, k_arg):
        norm = lambda x: x / jnp.maximum(
            jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-6
        )
        return norm(q_arg), norm(k_arg)

    q_norm, k_norm = normalize_pair(q, k)

    def polynomial_pair(q_arg, k_arg):
        feature = lambda x: (
            jnp.square(jnp.einsum("bhtd,pd->bhtp", x, anchors))
            / jnp.sqrt(anchors.shape[0])
        )
        return feature(q_arg), feature(k_arg)

    def prf_pair(q_arg, k_arg):
        def feature(x):
            projection = jnp.einsum("bhtd,rhdm->rbhtm", x, omega)
            exponent = jnp.clip(jnp.sqrt(2.0 * nodes) * projection - nodes, -10.0, 10.0)
            return (
                jnp.exp(exponent)
                / jnp.sqrt(omega.shape[-1])
                * jnp.sqrt(weights)
            )

        return feature(q_arg), feature(k_arg)

    q_poly, k_poly = polynomial_pair(q_norm, k_norm)
    q_prf, k_prf = prf_pair(q_norm, k_norm)

    def fusion_pair(qp, kp, qr, kr):
        def fuse(poly, prf):
            fused = jnp.einsum("bhtp,rbhtm->bhtrpm", poly, prf)
            return fused.reshape(*poly.shape[:-1], -1)

        return fuse(qp, qr), fuse(kp, kr)

    q_features, k_features = fusion_pair(q_poly, k_poly, q_prf, k_prf)
    attended = causal_linear_attention(q_features, k_features, v, remat_scan_body=True)
    batch, heads, length, head_dim = attended.shape
    merged = attended.transpose(0, 2, 1, 3).reshape(batch, length, heads * head_dim)

    stages = {
        "normalization_qk": (normalize_pair, (q, k)),
        "anchor_polynomial_qk": (polynomial_pair, (q_norm, k_norm)),
        "laplace_prf_qk": (prf_pair, (q_norm, k_norm)),
        "tensor_fusion_qk_materialized_diagnostic": (
            fusion_pair,
            (q_poly, k_poly, q_prf, k_prf),
        ),
        "scan_from_precomputed_features_diagnostic": (
            lambda qf, kf, value: causal_linear_attention(
                qf, kf, value, remat_scan_body=True
            ),
            (q_features, k_features, v),
        ),
        "output_projection": (
            lambda value, weight: jnp.einsum("bte,ef->btf", value, weight),
            (merged, output_weight),
        ),
    }
    return {
        name: measure(function, arguments, warmup=warmup, repeats=repeats)
        for name, (function, arguments) in stages.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", nargs="+", type=int, default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--modes", nargs="+", choices=["submitted", "matched"], default=["submitted", "matched"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-stage-breakdown", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/jax_reviewer_benchmark.json"))
    args = parser.parse_args()
    if args.embed_dim % args.heads:
        raise ValueError("embed_dim must be divisible by heads")
    head_dim = args.embed_dim // args.heads
    parameters = initialize_parameters(args.seed, args.embed_dim)
    results = []

    for mode in args.modes:
        config = feature_configuration(mode)
        anchor_state = create_feature_state(
            seed=args.seed,
            num_layers=1,
            num_heads=args.heads,
            head_dim=head_dim,
            poly_dim=config["anchor_poly_dim"],
            prf_dim=config["anchor_prf_dim"],
            num_quadrature=config["quadrature"],
        )
        compact_state = create_feature_state(
            seed=args.seed + 1,
            num_layers=1,
            num_heads=args.heads,
            head_dim=head_dim,
            poly_dim=1,
            prf_dim=32,
            num_quadrature=2,
        )
        performer_projection = jax.random.normal(
            jax.random.PRNGKey(args.seed + 2),
            (args.heads, head_dim, config["performer_features"]),
        )
        jax.block_until_ready((anchor_state, compact_state, performer_projection))

        for length in args.lengths:
            x = jax.random.normal(
                jax.random.PRNGKey(args.seed + length),
                (args.batch, length, args.embed_dim),
            )
            for method in args.methods:
                print(f"{mode}: {method} @ {length}", flush=True)

                def forward(current_parameters, current_x):
                    return module_forward(
                        current_parameters,
                        current_x,
                        method=method,
                        heads=args.heads,
                        anchor_state=anchor_state,
                        compact_state=compact_state,
                        performer_projection=performer_projection,
                    )

                def forward_backward(current_parameters, current_x):
                    def objective(p, value):
                        return jnp.mean(jnp.square(forward(p, value)))

                    return jax.value_and_grad(objective, argnums=(0, 1))(
                        current_parameters, current_x
                    )

                features = feature_dim(method, config, head_dim)
                entry = {
                    "mode": mode,
                    "method": method,
                    "length": length,
                    "feature_dim": features,
                    "forward": measure(
                        forward,
                        (parameters, x),
                        warmup=args.warmup,
                        repeats=args.repeats,
                    ),
                    "forward_backward": measure(
                        forward_backward,
                        (parameters, x),
                        warmup=args.warmup,
                        repeats=args.repeats,
                    ),
                    "analytical_memory": analytical_memory(
                        length,
                        batch=args.batch,
                        heads=args.heads,
                        head_dim=head_dim,
                        features=features,
                    ),
                }
                if (
                    method == "yat-performer-anchor"
                    and not args.skip_stage_breakdown
                ):
                    q, k, v = split_qkv(x, parameters, args.heads)
                    entry["isolated_anchor_stages"] = anchor_stage_breakdown(
                        q=q,
                        k=k,
                        v=v,
                        state=anchor_state,
                        output_weight=parameters["output"],
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                results.append(entry)

    payload = {
        "protocol": {
            "origin": "JAX reproduction of reviewer quick scaling benchmark",
            "batch": args.batch,
            "embed_dim": args.embed_dim,
            "heads": args.heads,
            "head_dim": head_dim,
            "dtype": "float32",
            "lengths": args.lengths,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "anchors": "iid Gaussian, unit-normalized, generated once and frozen",
            "gradient_policy": "fresh value_and_grad call; no accumulation",
            "timing_policy": "lowering, compilation, first execution, and synchronized steady state separated",
            "stage_note": "isolated materialized stages are diagnostic; canonical anchor execution fuses feature construction into lax.scan",
        },
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
