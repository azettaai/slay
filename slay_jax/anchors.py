"""Deterministic, persistent random features used by SLAY.

Anchor contract
---------------
For each layer, anchors are sampled i.i.d. from a standard Gaussian in R^d,
L2-normalized onto the unit sphere, and then frozen.  They are shared across
heads within a layer, matching the original implementation.  The PRF projection
matrix is also generated once and frozen.  Both arrays are checkpointed; the
seed and policy are retained as metadata for exact reconstruction and audit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


ANCHOR_POLICY = "iid_gaussian_then_unit_l2_per_layer_shared_across_heads"
FEATURE_STATE_VERSION = 1


class SLAYFeatureState(NamedTuple):
    """Frozen arrays used by SLAY feature maps.

    Shapes:
      anchors: [num_layers, poly_dim, head_dim]
      omega:   [num_layers, num_quadrature, num_heads, head_dim, prf_dim]
      nodes:   [num_quadrature]
      weights: [num_quadrature]
    """

    anchors: jax.Array
    omega: jax.Array
    nodes: jax.Array
    weights: jax.Array


def _laguerre_rule(num_quadrature: int, denominator_constant: float) -> tuple[np.ndarray, np.ndarray]:
    if num_quadrature < 1:
        raise ValueError("num_quadrature must be positive")
    nodes, weights = np.polynomial.laguerre.laggauss(num_quadrature)
    return nodes / denominator_constant, weights / denominator_constant


def create_feature_state(
    *,
    seed: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    poly_dim: int,
    prf_dim: int,
    num_quadrature: int,
    epsilon: float = 1e-6,
    dtype: jnp.dtype = jnp.float32,
) -> SLAYFeatureState:
    """Generate SLAY random features exactly once from an explicit seed."""
    if min(num_layers, num_heads, head_dim, poly_dim, prf_dim) < 1:
        raise ValueError("all dimensions must be positive")

    key = jax.random.PRNGKey(seed)
    anchor_key, omega_key = jax.random.split(key)
    anchors = jax.random.normal(
        anchor_key, (num_layers, poly_dim, head_dim), dtype=dtype
    )
    anchors = anchors / jnp.maximum(
        jnp.linalg.norm(anchors, axis=-1, keepdims=True),
        jnp.finfo(dtype).tiny,
    )
    omega = jax.random.normal(
        omega_key,
        (num_layers, num_quadrature, num_heads, head_dim, prf_dim),
        dtype=dtype,
    )
    nodes, weights = _laguerre_rule(num_quadrature, 2.0 + epsilon)
    return SLAYFeatureState(
        anchors=anchors,
        omega=omega,
        nodes=jnp.asarray(nodes, dtype=dtype),
        weights=jnp.asarray(weights, dtype=dtype),
    )


def save_feature_state(
    path: str | Path,
    state: SLAYFeatureState,
    *,
    seed: int,
    epsilon: float,
) -> None:
    """Save frozen random features and generation metadata in one NPZ file."""
    path = Path(path)
    metadata = {
        "version": FEATURE_STATE_VERSION,
        "seed": int(seed),
        "epsilon": float(epsilon),
        "anchor_policy": ANCHOR_POLICY,
    }
    np.savez_compressed(
        path,
        anchors=np.asarray(state.anchors),
        omega=np.asarray(state.omega),
        nodes=np.asarray(state.nodes),
        weights=np.asarray(state.weights),
        metadata=np.asarray(json.dumps(metadata)),
    )


def load_feature_state(path: str | Path) -> tuple[SLAYFeatureState, dict]:
    """Restore frozen features without invoking any random generator."""
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata.get("version") != FEATURE_STATE_VERSION:
            raise ValueError(f"unsupported feature-state version: {metadata.get('version')}")
        state = SLAYFeatureState(
            anchors=jnp.asarray(data["anchors"]),
            omega=jnp.asarray(data["omega"]),
            nodes=jnp.asarray(data["nodes"]),
            weights=jnp.asarray(data["weights"]),
        )
    return state, metadata
