"""Canonical JAX implementation and diagnostic benchmarks for SLAY."""

from .anchors import (
    SLAYFeatureState,
    create_feature_state,
    load_feature_state,
    save_feature_state,
)
from .attention import (
    attention,
    causal_linear_attention,
    exact_softmax_attention,
    slay_features,
)

__all__ = [
    "SLAYFeatureState",
    "attention",
    "causal_linear_attention",
    "create_feature_state",
    "exact_softmax_attention",
    "load_feature_state",
    "save_feature_state",
    "slay_features",
]
