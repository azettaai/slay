#!/usr/bin/env python3
"""Train small causal JAX models on non-leaky synthetic diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from slay_jax.anchors import create_feature_state
from slay_jax.attention import attention
from slay_jax.synthetic_tasks import TASKS, TaskBatch, masked_accuracy, masked_cross_entropy


@dataclass(frozen=True)
class ModelConfig:
    embed_dim: int = 32
    num_heads: int = 2
    num_layers: int = 2
    ff_dim: int = 64
    performer_features: int = 32
    poly_dim: int = 4
    prf_dim: int = 4
    num_quadrature: int = 1
    anchor_seed: int = 2026


def _normal(key, shape, fan_in):
    return jax.random.normal(key, shape) / jnp.sqrt(fan_in)


def initialize_model(key, *, vocab_size: int, max_length: int, config: ModelConfig):
    keys = iter(jax.random.split(key, 4 + 6 * config.num_layers))
    params = {
        "token_embedding": _normal(next(keys), (vocab_size, config.embed_dim), config.embed_dim),
        "position_embedding": _normal(next(keys), (max_length, config.embed_dim), config.embed_dim),
        "layers": [],
    }
    for _ in range(config.num_layers):
        params["layers"].append(
            {
                "norm1": jnp.ones((config.embed_dim,)),
                "qkv": _normal(
                    next(keys), (config.embed_dim, 3 * config.embed_dim), config.embed_dim
                ),
                "out": _normal(
                    next(keys), (config.embed_dim, config.embed_dim), config.embed_dim
                ),
                "norm2": jnp.ones((config.embed_dim,)),
                "ff1": _normal(next(keys), (config.embed_dim, config.ff_dim), config.embed_dim),
                "ff2": _normal(next(keys), (config.ff_dim, config.embed_dim), config.ff_dim),
            }
        )
    params["final_norm"] = jnp.ones((config.embed_dim,))
    params["head"] = _normal(next(keys), (config.embed_dim, vocab_size), config.embed_dim)

    head_dim = config.embed_dim // config.num_heads
    feature_state = create_feature_state(
        seed=config.anchor_seed,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        head_dim=head_dim,
        poly_dim=config.poly_dim,
        prf_dim=config.prf_dim,
        num_quadrature=config.num_quadrature,
    )
    projection_key = next(keys)
    performer_projection = jax.random.normal(
        projection_key,
        (
            config.num_layers,
            config.num_heads,
            head_dim,
            config.performer_features,
        ),
    )
    frozen = {
        "slay": feature_state,
        "performer_projection": performer_projection,
    }
    return params, frozen


def rms_norm(x, scale, eps=1e-6):
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps) * scale


def model_forward(params, frozen, inputs, *, attention_kind: str, config: ModelConfig):
    batch, length = inputs.shape
    hidden = params["token_embedding"][inputs] + params["position_embedding"][:length]
    head_dim = config.embed_dim // config.num_heads

    for layer_index, layer in enumerate(params["layers"]):
        residual = hidden
        normalized = rms_norm(hidden, layer["norm1"])
        qkv = jnp.einsum("bte,ef->btf", normalized, layer["qkv"])
        q, k, v = jnp.split(qkv, 3, axis=-1)

        def split_heads(x):
            return x.reshape(batch, length, config.num_heads, head_dim).transpose(0, 2, 1, 3)

        attended = attention(
            split_heads(q),
            split_heads(k),
            split_heads(v),
            kind=attention_kind,
            causal=True,
            feature_state=frozen["slay"],
            layer_index=layer_index,
            performer_projection=frozen["performer_projection"][layer_index],
            remat_scan_body=attention_kind == "slay",
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, config.embed_dim)
        hidden = residual + jnp.einsum("bte,ef->btf", attended, layer["out"])
        residual = hidden
        normalized = rms_norm(hidden, layer["norm2"])
        feedforward = jax.nn.gelu(jnp.einsum("bte,ef->btf", normalized, layer["ff1"]))
        hidden = residual + jnp.einsum("btf,fe->bte", feedforward, layer["ff2"])

    hidden = rms_norm(hidden, params["final_norm"])
    return jnp.einsum("bte,ev->btv", hidden, params["head"])


def run_one(
    *,
    task_name: str,
    attention_kind: str,
    config: ModelConfig,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    eval_batches: int,
) -> dict:
    task = TASKS[task_name]
    model_key, data_key = jax.random.split(jax.random.PRNGKey(seed))
    params, frozen = initialize_model(
        model_key,
        vocab_size=task.vocab_size,
        max_length=task.sequence_length,
        config=config,
    )
    optimizer = optax.adamw(learning_rate, weight_decay=1e-4)
    optimizer_state = optimizer.init(params)

    def loss_function(current_params, batch):
        logits = model_forward(
            current_params,
            frozen,
            batch.inputs,
            attention_kind=attention_kind,
            config=config,
        )
        return masked_cross_entropy(logits, batch)

    @jax.jit
    def train_step(current_params, current_optimizer_state, batch):
        loss, gradients = jax.value_and_grad(loss_function)(current_params, batch)
        updates, next_optimizer_state = optimizer.update(
            gradients, current_optimizer_state, current_params
        )
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_optimizer_state, loss

    training_losses = []
    step_times = []
    compile_and_first_step_ms = None
    for step in range(steps):
        data_key, batch_key = jax.random.split(data_key)
        batch = task.generator(batch_key, batch_size)
        start = time.perf_counter()
        params, optimizer_state, loss = train_step(params, optimizer_state, batch)
        jax.block_until_ready(loss)
        elapsed = 1000.0 * (time.perf_counter() - start)
        if step == 0:
            compile_and_first_step_ms = elapsed
        else:
            step_times.append(elapsed)
        training_losses.append(float(loss))

    accuracies = []
    for _ in range(eval_batches):
        data_key, batch_key = jax.random.split(data_key)
        batch = task.generator(batch_key, batch_size * 2)
        logits = model_forward(
            params,
            frozen,
            batch.inputs,
            attention_kind=attention_kind,
            config=config,
        )
        accuracies.append(float(masked_accuracy(logits, batch)))

    return {
        "task": task_name,
        "attention": attention_kind,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "chance_accuracy": task.chance_accuracy,
        "final_train_loss": training_losses[-1],
        "eval_accuracy_mean": float(np.mean(accuracies)),
        "eval_accuracy_std_across_batches": float(np.std(accuracies)),
        "compile_and_first_train_step_ms": compile_and_first_step_ms,
        "steady_train_step_median_ms": float(np.median(step_times)) if step_times else None,
        "loss_curve_every_25_steps": training_losses[::25],
        "solved_at_95_percent": float(np.mean(accuracies)) >= 0.95,
        "slay_feature_dim": (
            config.poly_dim * config.prf_dim * config.num_quadrature
            if attention_kind == "slay"
            else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "multi_query_associative_recall",
            "long_delayed_copy",
            "induction_recall",
        ],
        choices=sorted(TASKS),
    )
    parser.add_argument(
        "--attentions",
        nargs="+",
        default=["softmax", "linear_elu", "performer_relu", "slay"],
        choices=["softmax", "linear_elu", "performer_relu", "slay"],
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--poly-dim", type=int, default=4)
    parser.add_argument("--prf-dim", type=int, default=4)
    parser.add_argument("--quadrature", type=int, default=1)
    parser.add_argument("--performer-features", type=int, default=32)
    parser.add_argument("--anchor-seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("artifacts/jax_synthetic_local.json"))
    args = parser.parse_args()
    if args.embed_dim % args.num_heads:
        raise ValueError("embed_dim must be divisible by num_heads")

    config = ModelConfig(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=2 * args.embed_dim,
        poly_dim=args.poly_dim,
        prf_dim=args.prf_dim,
        num_quadrature=args.quadrature,
        performer_features=args.performer_features,
        anchor_seed=args.anchor_seed,
    )
    results = []
    for task_name in args.tasks:
        for attention_kind in args.attentions:
            print(f"training {attention_kind} on {task_name}...", flush=True)
            result = run_one(
                task_name=task_name,
                attention_kind=attention_kind,
                config=config,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                eval_batches=args.eval_batches,
            )
            results.append(result)
            print(
                f"  accuracy={result['eval_accuracy_mean']:.3f}, "
                f"loss={result['final_train_loss']:.4f}, "
                f"steady_step={result['steady_train_step_median_ms']:.2f} ms"
            )

    payload = {
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "model_config": asdict(config),
        "anchor_protocol": {
            "distribution": "iid standard Gaussian, L2-normalized to unit sphere",
            "scope": "independent per layer, shared across heads",
            "lifetime": "generated once before training and frozen",
            "seed": config.anchor_seed,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
