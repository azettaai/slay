#!/usr/bin/env python3
"""Length-scaling benchmark with synchronized JAX forward and backward timing."""

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
from slay_jax.attention import attention


def timed_samples(compiled, arguments, *, warmup: int, repeats: int):
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
    }


def compile_and_measure(function, arguments, *, warmup: int, repeats: int):
    start = time.perf_counter()
    lowered = jax.jit(function).lower(*arguments)
    lowering_ms = 1000.0 * (time.perf_counter() - start)
    start = time.perf_counter()
    compiled = lowered.compile()
    compile_ms = 1000.0 * (time.perf_counter() - start)
    measured = timed_samples(compiled, arguments, warmup=warmup, repeats=repeats)
    return {"lowering_ms": lowering_ms, "compile_ms": compile_ms, **measured}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument(
        "--attentions",
        nargs="+",
        default=["softmax", "linear_elu", "performer_relu", "slay"],
        choices=["softmax", "linear_elu", "performer_relu", "slay"],
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--performer-features", type=int, default=64)
    parser.add_argument("--poly-dim", type=int, default=8)
    parser.add_argument("--prf-dim", type=int, default=8)
    parser.add_argument("--quadrature", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("artifacts/jax_scaling_local.json"))
    args = parser.parse_args()

    anchor_start = time.perf_counter()
    feature_state = create_feature_state(
        seed=args.seed,
        num_layers=1,
        num_heads=args.heads,
        head_dim=args.head_dim,
        poly_dim=args.poly_dim,
        prf_dim=args.prf_dim,
        num_quadrature=args.quadrature,
    )
    jax.block_until_ready(feature_state)
    anchor_generation_cold_ms = 1000.0 * (time.perf_counter() - anchor_start)
    warm_anchor_start = time.perf_counter()
    warm_feature_state = create_feature_state(
        seed=args.seed + 10_000,
        num_layers=1,
        num_heads=args.heads,
        head_dim=args.head_dim,
        poly_dim=args.poly_dim,
        prf_dim=args.prf_dim,
        num_quadrature=args.quadrature,
    )
    jax.block_until_ready(warm_feature_state)
    anchor_generation_warm_ms = 1000.0 * (time.perf_counter() - warm_anchor_start)
    projection = jax.random.normal(
        jax.random.PRNGKey(args.seed + 1),
        (args.heads, args.head_dim, args.performer_features),
    )
    jax.block_until_ready(projection)

    results = []
    for length in args.lengths:
        shape = (args.batch, args.heads, length, args.head_dim)
        q_key, k_key, v_key = jax.random.split(jax.random.PRNGKey(length), 3)
        q = jax.random.normal(q_key, shape)
        k = jax.random.normal(k_key, shape)
        v = jax.random.normal(v_key, shape)
        for kind in args.attentions:
            print(f"{kind}, L={length}", flush=True)

            def forward(q_arg, k_arg, v_arg):
                return attention(
                    q_arg,
                    k_arg,
                    v_arg,
                    kind=kind,
                    causal=True,
                    feature_state=feature_state,
                    performer_projection=projection,
                    remat_scan_body=kind == "slay",
                )

            def forward_backward(q_arg, k_arg, v_arg):
                def scalar(q_inner, k_inner, v_inner):
                    return jnp.mean(jnp.square(forward(q_inner, k_inner, v_inner)))

                return jax.grad(scalar, argnums=(0, 1, 2))(q_arg, k_arg, v_arg)

            forward_result = compile_and_measure(
                forward, (q, k, v), warmup=args.warmup, repeats=args.repeats
            )
            backward_result = compile_and_measure(
                forward_backward, (q, k, v), warmup=args.warmup, repeats=args.repeats
            )
            feature_dim = {
                "softmax": None,
                "linear_elu": args.head_dim,
                "performer_relu": args.performer_features,
                "slay": args.quadrature * args.poly_dim * args.prf_dim,
            }[kind]
            state_elements_per_batch_head = (
                None if feature_dim is None else feature_dim * (args.head_dim + 1)
            )
            results.append(
                {
                    "attention": kind,
                    "length": length,
                    "feature_dim": feature_dim,
                    "causal_state_elements_per_batch_head": state_elements_per_batch_head,
                    "forward": forward_result,
                    "forward_backward": backward_result,
                }
            )

    payload = {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__,
        "shape": {
            "batch": args.batch,
            "heads": args.heads,
            "head_dim": args.head_dim,
        },
        "feature_config": {
            "performer_features": args.performer_features,
            "slay_poly_dim": args.poly_dim,
            "slay_prf_dim": args.prf_dim,
            "slay_quadrature": args.quadrature,
            "slay_feature_dim": args.quadrature * args.poly_dim * args.prf_dim,
        },
        "frozen_random_state": {
            "seed": args.seed,
            "anchor_generation_cold_ms_one_time": anchor_generation_cold_ms,
            "anchor_generation_warm_ms_one_time": anchor_generation_warm_ms,
            "regenerated_per_call": False,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
