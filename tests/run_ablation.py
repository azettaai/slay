"""
ICML Ablation — AetherLM
Run ablation experiments sequentially.

Usage:
    python run_ablation.py --experiments all
    python run_ablation.py --experiments A3_yat_attn_s0 A8_full_aether_s0
    python run_ablation.py --list
"""

import argparse
import datetime as dt
import gc
import json
import os
import traceback
import time
from pathlib import Path

# ── Load .env if present (local runs) ────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass  # dotenv not installed; rely on env vars being set externally

# ── Kaggle secrets (only when running on Kaggle) ─────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    GITHUB_TOKEN = user_secrets.get_secret("GITHUB_TOKEN")
    os.environ["WANDB_API_KEY"] = user_secrets.get_secret("WANDB_API_KEY")
    os.environ["WANDB_ENTITY"] = user_secrets.get_secret("WANDB_ENTITY")
    os.environ["WANDB_PROJECT"] = user_secrets.get_secret("WANDB_PROJECT")
    os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")
except Exception:
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
    os.environ["WANDB_API_KEY"] = os.environ.get("WANDB_API_KEY", "")
    os.environ["WANDB_ENTITY"]  = os.environ.get("WANDB_ENTITY", "")
    os.environ["WANDB_PROJECT"] = os.environ.get("WANDB_PROJECT", "")
    os.environ["HF_TOKEN"]      = os.environ.get("HF_TOKEN", "")

import jax
import jax.numpy as jnp

n_devices = jax.device_count()
platform = jax.devices()[0].platform
print(f"JAX {jax.__version__}  |  {n_devices} devices  |  {platform}")
for d in jax.devices():
    print(f"  {d}")

# ══════════════════════════════════════════════════════════════════════
# Shared hyperparameters (same across all arms)
# ══════════════════════════════════════════════════════════════════════
EMBED_DIM       = 1536
NUM_HEADS       = 24
NUM_LAYERS      = 12
FF_DIM          = 6144
MAXLEN          = 1024
VOCAB_SIZE      = 32000
TOKENIZER       = "mistralai/Mistral-7B-v0.1"
TIE_EMBEDDINGS  = True
USE_ROPE        = False
USE_REMAT       = True
REMAT_POLICY    = "dots_saveable"
DROPOUT         = 0.0
PARAM_DTYPE     = "bfloat16"
ATTN_KERNEL     = "auto"

LR              = 3e-3
WARMUP          = 2000
END_LR          = 1e-5
OPT_TYPE        = "lamb"
WEIGHT_DECAY    = 0.01
BETA1           = 0.9
BETA2           = 0.999
EPS             = 1e-6
LABEL_SMOOTH    = 0.0
Z_LOSS          = 1e-4
MAX_GRAD_NORM   = 0.0
GRAD_DTYPE      = "bfloat16"

TRAIN_DATASET       = "HuggingFaceFW/fineweb-edu"
TRAIN_DATASET_CFG   = "sample-100BT"
VAL_DATASET         = "wikitext"
VAL_DATASET_CFG     = "wikitext-103-v1"
VAL_DATASET_SPLIT   = "validation"

BATCH_PER_DEVICE = 32
GRAD_ACCUM       = 1
GLOBAL_BATCH     = BATCH_PER_DEVICE * n_devices * GRAD_ACCUM
TOK_PER_STEP     = GLOBAL_BATCH * MAXLEN

CHINCHILLA_TOKENS = 2_621_440_000
STEPS = max(CHINCHILLA_TOKENS // TOK_PER_STEP, 1)

print(f"Global batch : {GLOBAL_BATCH}  ({TOK_PER_STEP:,} tok/step)")
print(f"Steps        : {STEPS:,}  ({STEPS * TOK_PER_STEP / 1e9:.1f}B tokens)")

# ══════════════════════════════════════════════════════════════════════
# Experiment arms registry
# (name, model_type, norm_type, mlp_type, seed, extra_overrides)
# ══════════════════════════════════════════════════════════════════════
ALL_EXPERIMENTS = {
    "A1_baseline_s0": (
        "gpt", "pre_rmsnorm", "gelu", 0, {}),
    "A2_yat_mlp_s0": (
        "gpt_yatnmn_pre_rmsnorm", "pre_rmsnorm", "yatnmn", 0, {}),
    "A3_yat_attn_s0": (
        "gpt_yat_attn_pre_rmsnorm", "pre_rmsnorm", "gelu", 0, {}),
    "A4_no_norm_s0": (
        "gpt", "none", "gelu", 0, {}),
    "A5_yat_both_s0": (
        "gpt_yatnmn_yat", "pre_rmsnorm", "yatnmn", 0, {}),
    "A6_yat_mlp_nf_s0": (
        "gpt_yatnmn", "none", "yatnmn", 0, {}),
    "A7_yat_attn_nf_s0": (
        "gpt_yat_attn", "none", "gelu", 0, {}),
    "A8_full_aether_s0": (
        "gpt_yatnmn_yat", "norm_free", "yatnmn", 0, {}),
    "A9_yat_performer_s0": (
        "gpt_yatnmn_yat_performer_v2", "norm_free", "yatnmn", 0,
        {"num_anchor_features": 8, "num_prf_features": 8}),
    "A10_gpt2_baseline_s0": (
        "gpt", "post_layernorm", "gelu", 0, {}),
    "A11_aether_post_ln_s0": (
        "gpt_yatnmn_yat", "post_layernorm", "yatnmn", 0, {}),
    "A12_yat_attn_perf_s0": (
        "gpt_yat_attn_performer_v2", "pre_rmsnorm", "gelu", 0,
        {"num_anchor_features": 8, "num_prf_features": 8}),
    "A13_performer_s0": (
        "gpt_performer", "pre_rmsnorm", "gelu", 0, {}),
    "A14_cosformer_s0": (
        "gpt_cosformer", "pre_rmsnorm", "gelu", 0, {}),
    "A15_aether_b128_s0": (
        "gpt_yatnmn_yat", "norm_free", "yatnmn", 0, {}),
}

# ══════════════════════════════════════════════════════════════════════
# Config builder
# ══════════════════════════════════════════════════════════════════════
from aetherlm.core.config import AetherConfig


def build_experiment_config(exp_name, model_type, norm_type, mlp_type, seed, extra):
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"icml_{exp_name}_kaggle_{ts}"

    return AetherConfig.from_dict({
        "training": {
            "mode": "causal",
            "batch_size": BATCH_PER_DEVICE,
            "learning_rate": LR,
            "max_steps": STEPS,
            "grad_accumulation_steps": GRAD_ACCUM,
            "label_smoothing": LABEL_SMOOTH,
            "z_loss_coeff": Z_LOSS,
            "grad_dtype": GRAD_DTYPE,
            "max_inflight_steps": 4,
            "ce_chunk_size": 256,
            "log_grad_norm": True,
            "log_param_norm": True,
        },
        "model": {
            "model_type": model_type,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "ff_dim": FF_DIM,
            "maxlen": MAXLEN,
            "vocab_size": VOCAB_SIZE,
            "tokenizer_name": TOKENIZER,
            "tie_embeddings": TIE_EMBEDDINGS,
            "dropout_rate": DROPOUT,
            "param_dtype": PARAM_DTYPE,
            "norm_type": norm_type,
            "mlp_type": mlp_type,
            "fused_yatnmn": True,
            "use_rope": USE_ROPE,
            "use_remat": USE_REMAT,
            "remat_policy": REMAT_POLICY,
            "attention_kernel": ATTN_KERNEL,
            "extra": {
                "num_recurrences": 1,
                "yatnmn_epsilon": 0.001,
                "yat_epsilon": 1.0,
                "yat_alpha": "learnable",
                "yat_normalization": "l1",
                "use_rope": True,
                "seed": seed,
                **extra,
            },
        },
        "optimizer": {
            "type": OPT_TYPE,
            "warmup_steps": WARMUP,
            "end_lr": END_LR,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "adam_beta1": BETA1,
            "adam_beta2": BETA2,
            "adam_eps": EPS,
            "schedule_type": "warmup_cosine",
        },
        "data": {
            "train_dataset": TRAIN_DATASET,
            "train_dataset_config": TRAIN_DATASET_CFG,
            "train_streaming": True,
            "packing": True,
            "tokenize_num_workers": 8,
            "prefetch_buffer": 32,
            "num_prefetch_workers": 2,
            "val_dataset": VAL_DATASET,
            "val_dataset_config": VAL_DATASET_CFG,
            "val_dataset_split": VAL_DATASET_SPLIT,
        },
        "checkpoint": {
            "dir": f"/kaggle/working/checkpoints/{run_name}",
            "interval": 1000,
            "zip_on_save": True,
        },
        "logging": {
            "run_name": run_name,
            "use_wandb": bool(os.environ.get("WANDB_API_KEY")),
            "wandb_entity": os.environ.get("WANDB_ENTITY", ""),
            "wandb_project": os.environ.get("WANDB_PROJECT", "icmlDelulu"),
            "log_interval": 20,
            "print_interval": 20,
            "eval_interval": STEPS,
            "eval_steps": 50,
            "watch_weights": True,
            "weight_watch_interval": 100,
            "log_weight_histograms": True,
            "enable_duckdb": True,
            "duckdb_path": f"{run_name}.duckdb",
            "duckdb_flush_every": 10,
            "interpret_interval": 500,
            "interpret_anomaly": True,
        },
        "tpu": {
            "precision": "bf16",
        },
    })


# ══════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════
from aetherlm.core.engine import AetherEngine
from aetherlm.data.causal import load_causal_datasets


def run_experiments(experiments: dict, output_path: str = "/kaggle/working/ablation_results.json"):
    results = []

    for i, (exp_name, (model_type, norm_type, mlp_type, seed, extra)) in enumerate(experiments.items()):
        print()
        print("=" * 70)
        print(f"[{i+1}/{len(experiments)}] {exp_name}")
        print(f"  model={model_type}  norm={norm_type}  mlp={mlp_type}  seed={seed}")
        print("=" * 70)

        t0 = time.time()
        status = "failed"
        final_loss = None
        final_val_loss = None

        try:
            config = build_experiment_config(exp_name, model_type, norm_type, mlp_type, seed, extra)

            engine = AetherEngine(config)
            engine.setup()

            dcfg = config.data
            train_ds, val_ds = load_causal_datasets(
                maxlen=MAXLEN,
                tokenizer_name=TOKENIZER,
                train_dataset_name=dcfg.train_dataset,
                train_dataset_config=dcfg.train_dataset_config,
                train_streaming=dcfg.train_streaming,
                val_dataset_name=dcfg.val_dataset,
                val_dataset_config=dcfg.val_dataset_config,
                val_dataset_split=dcfg.val_dataset_split,
            )

            engine.train(train_data=train_ds, val_data=val_ds, num_steps=STEPS)

            stats = engine.metrics.compute_statistics()
            final_loss = stats.get("best_train_loss")
            final_val_loss = stats.get("best_val_loss")
            status = "done"

            print(f"\n  DONE  loss={final_loss}  val_loss={final_val_loss}")

        except Exception as e:
            print(f"\n  FAILED: {e}")
            traceback.print_exc()

        elapsed = time.time() - t0
        results.append({
            "experiment": exp_name,
            "model_type": model_type,
            "norm_type": norm_type,
            "mlp_type": mlp_type,
            "seed": seed,
            "status": status,
            "train_loss": final_loss,
            "val_loss": final_val_loss,
            "elapsed_min": round(elapsed / 60, 1),
            "steps": STEPS,
            "global_batch": GLOBAL_BATCH,
        })

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        try:
            del engine, train_ds, val_ds, config
        except NameError:
            pass
        gc.collect()
        print(f"  Cleanup done. Elapsed: {elapsed/60:.1f} min")

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)

    print(f"\n{'Experiment':<28s} {'Status':>8s} {'Loss':>10s} {'Val Loss':>10s} {'Time':>8s}")
    print("-" * 70)
    for r in results:
        loss_str = f"{r['train_loss']:.4f}" if r["train_loss"] else "---"
        val_str = f"{r['val_loss']:.4f}" if r["val_loss"] else "---"
        print(f"{r['experiment']:<28s} {r['status']:>8s} {loss_str:>10s} {val_str:>10s} {r['elapsed_min']:>6.1f}m")

    n_ok = sum(1 for r in results if r["status"] == "done")
    print(f"\n{n_ok}/{len(results)} succeeded")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SLAY ablation experiments")
    parser.add_argument(
        "--experiments", nargs="+", default=["all"],
        help="Experiment name(s) to run, or 'all' to run every arm. Use --list to see available names.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print all available experiment names and exit.",
    )
    parser.add_argument(
        "--output", default="/kaggle/working/ablation_results.json",
        help="Path to write JSON results (default: /kaggle/working/ablation_results.json)",
    )
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for name, (mt, nt, mlp, seed, _) in ALL_EXPERIMENTS.items():
            print(f"  {name:<28s}  model={mt:<35s}  norm={nt:<14s}  mlp={mlp}  seed={seed}")
        exit(0)

    if args.experiments == ["all"]:
        selected = ALL_EXPERIMENTS
    else:
        unknown = [e for e in args.experiments if e not in ALL_EXPERIMENTS]
        if unknown:
            print(f"Unknown experiment(s): {unknown}")
            print("Run with --list to see available names.")
            exit(1)
        selected = {name: ALL_EXPERIMENTS[name] for name in args.experiments}

    print(f"{len(selected)} experiment(s) queued:")
    for name, (mt, nt, mlp, seed, _) in selected.items():
        print(f"  {name:<28s}  model={mt:<35s}  norm={nt:<14s}  mlp={mlp}  seed={seed}")

    run_experiments(selected, output_path=args.output)
