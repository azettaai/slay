#!/usr/bin/env python3
"""Launch the packaged SLAY synthetic suite with a fair attention-LR sweep."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def values(name, default):
    return [x for x in os.environ.get(name, default).split(",") if x]


def run(arguments):
    command = [sys.executable, "benchmark_synthetic.py", *arguments]
    print("RUN", " ".join(command), flush=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path("slay_bundle.zip").resolve())
    subprocess.run(command, check=True, env=environment)


def main():
    tasks = values(
        "TASKS",
        "induction_recall,long_delayed_copy,multi_query_associative_recall",
    )
    seeds = values("SEEDS", "0,1,2")
    multipliers = values("LR_MULTIPLIERS", "1,2,4")
    common = [
        "--tasks",
        *tasks,
        "--steps",
        os.environ.get("STEPS", "1000"),
        "--batch-size",
        os.environ.get("BATCH_SIZE", "64"),
        "--eval-batches",
        os.environ.get("EVAL_BATCHES", "8"),
        "--learning-rate",
        os.environ.get("LEARNING_RATE", "0.003"),
        "--embed-dim",
        os.environ.get("EMBED_DIM", "32"),
        "--num-heads",
        os.environ.get("NUM_HEADS", "2"),
        "--num-layers",
        os.environ.get("NUM_LAYERS", "2"),
        "--poly-dim",
        os.environ.get("POLY_DIM", "8"),
        "--prf-dim",
        os.environ.get("PRF_DIM", "4"),
        "--quadrature",
        os.environ.get("QUADRATURE", "2"),
        "--performer-features",
        os.environ.get("PERFORMER_FEATURES", "64"),
    ]
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    files = []
    for seed in seeds:
        baseline = output_dir / f"synthetic_seed{seed}_baselines.json"
        run(
            common
            + [
                "--attentions",
                "softmax",
                "performer_relu",
                "--seed",
                seed,
                "--output",
                str(baseline),
            ]
        )
        files.append(baseline)
        for multiplier in multipliers:
            output = output_dir / f"synthetic_seed{seed}_slay_lr{multiplier}.json"
            run(
                common
                + [
                    "--attentions",
                    "slay",
                    "--slay-attention-lr-multiplier",
                    multiplier,
                    "--seed",
                    seed,
                    "--output",
                    str(output),
                ]
            )
            files.append(output)
    payloads = [json.loads(path.read_text()) for path in files]
    combined = {
        "backend": payloads[0]["backend"],
        "jax_version": payloads[0]["jax_version"],
        "device_kinds": [device.device_kind for device in __import__("jax").devices()],
        "protocol": {
            "tasks": tasks,
            "seeds": seeds,
            "attention_lr_multipliers": multipliers,
            "equal_feature_budget": "SLAY R*P*M=64; Performer F=64",
            "selection": "report all attempted multipliers; no post-hoc hidden trials",
        },
        "results": [
            row for payload in payloads for row in payload["results"]
        ],
    }
    (output_dir / "synthetic_suite.json").write_text(
        json.dumps(combined, indent=2) + "\n"
    )
    print("wrote results/synthetic_suite.json", flush=True)


if __name__ == "__main__":
    main()
