#!/usr/bin/env python3
"""Separate JAX lowering/compile/first-run/steady-state costs by attention stage."""

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
from slay_jax.attention import attention, causal_linear_attention, slay_features


def _milliseconds(start: float) -> float:
    return 1000.0 * (time.perf_counter() - start)


def _ready(tree):
    return jax.block_until_ready(tree)


def measure_compilation(function, *args, repeats: int) -> dict:
    start = time.perf_counter()
    lowered = jax.jit(function).lower(*args)
    lowering_ms = _milliseconds(start)

    start = time.perf_counter()
    compiled = lowered.compile()
    compile_ms = _milliseconds(start)

    start = time.perf_counter()
    _ready(compiled(*args))
    first_execution_ms = _milliseconds(start)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ready(compiled(*args))
        samples.append(_milliseconds(start))

    result = {
        "lowering_ms": lowering_ms,
        "compile_ms": compile_ms,
        "first_execution_ms": first_execution_ms,
        "steady_state_median_ms": float(np.median(samples)),
        "steady_state_min_ms": float(np.min(samples)),
        "steady_state_samples_ms": samples,
    }
    try:
        result["cost_analysis"] = compiled.cost_analysis()
    except Exception as error:
        result["cost_analysis_error"] = str(error)
    try:
        memory = compiled.memory_analysis()
        result["memory_analysis"] = {
            field: int(getattr(memory, field))
            for field in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
                "generated_code_size_in_bytes",
            )
            if hasattr(memory, field)
        }
    except Exception as error:
        result["memory_analysis_error"] = str(error)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--poly-dim", type=int, default=8)
    parser.add_argument("--prf-dim", type=int, default=8)
    parser.add_argument("--quadrature", type=int, default=2)
    parser.add_argument("--performer-features", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("artifacts/jax_profile_local.json"))
    args = parser.parse_args()

    state = create_feature_state(
        seed=17,
        num_layers=1,
        num_heads=args.heads,
        head_dim=args.head_dim,
        poly_dim=args.poly_dim,
        prf_dim=args.prf_dim,
        num_quadrature=args.quadrature,
    )
    q_key, k_key, v_key, projection_key = jax.random.split(jax.random.PRNGKey(3), 4)
    shape = (args.batch, args.heads, args.length, args.head_dim)
    q = jax.random.normal(q_key, shape)
    k = jax.random.normal(k_key, shape)
    v = jax.random.normal(v_key, shape)
    projection = jax.random.normal(
        projection_key, (args.heads, args.head_dim, args.performer_features)
    )

    functions = {
        "softmax_causal": lambda q, k, v: attention(
            q, k, v, kind="softmax", causal=True
        ),
        "linear_elu_causal": lambda q, k, v: attention(
            q, k, v, kind="linear_elu", causal=True
        ),
        "performer_relu_causal": lambda q, k, v: attention(
            q,
            k,
            v,
            kind="performer_relu",
            causal=True,
            performer_projection=projection,
        ),
        "slay_causal": lambda q, k, v: attention(
            q, k, v, kind="slay", causal=True, feature_state=state
        ),
        "slay_feature_map_only": lambda q: slay_features(q, state, layer_index=0),
    }
    slay_feature_shape = (
        args.batch,
        args.heads,
        args.length,
        args.quadrature * args.poly_dim * args.prf_dim,
    )
    feature_key1, feature_key2 = jax.random.split(jax.random.PRNGKey(99))
    q_features = jax.random.uniform(feature_key1, slay_feature_shape) + 1e-4
    k_features = jax.random.uniform(feature_key2, slay_feature_shape) + 1e-4
    functions["slay_scan_only"] = lambda qf, kf, value: causal_linear_attention(
        qf, kf, value
    )

    results = {}
    for name, function in functions.items():
        call_args = (q,) if name == "slay_feature_map_only" else (
            (q_features, k_features, v) if name == "slay_scan_only" else (q, k, v)
        )
        print(f"profiling {name}...", flush=True)
        results[name] = measure_compilation(function, *call_args, repeats=args.repeats)

    payload = {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__,
        "shape": {
            "batch": args.batch,
            "heads": args.heads,
            "length": args.length,
            "head_dim": args.head_dim,
            "slay_feature_dim": args.quadrature * args.poly_dim * args.prf_dim,
            "performer_feature_dim": args.performer_features,
        },
        "anchor_generation": {
            "seed": 17,
            "generated_once_before_profiling": True,
            "included_in_timing": False,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
