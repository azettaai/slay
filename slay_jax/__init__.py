"""Canonical JAX implementation and diagnostic benchmarks for SLAY."""

from .anchors import (
    FlatJointSLAYState,
    SLAYFeatureState,
    create_flat_joint_feature_state,
    create_feature_state,
    load_flat_joint_feature_state,
    load_feature_state,
    save_flat_joint_feature_state,
    save_feature_state,
)
from .attention import (
    attention,
    causal_linear_attention,
    chunked_causal_linear_attention,
    exact_softmax_attention,
    flat_joint_slay_features,
    parallel_prefix_causal_attention,
    parallel_anchor_slay_attention,
    slay_features,
    streaming_flat_joint_slay_attention,
)

__all__ = [
    "FlatJointSLAYState",
    "SLAYFeatureState",
    "attention",
    "causal_linear_attention",
    "chunked_causal_linear_attention",
    "create_flat_joint_feature_state",
    "create_feature_state",
    "exact_softmax_attention",
    "flat_joint_slay_features",
    "load_flat_joint_feature_state",
    "parallel_prefix_causal_attention",
    "parallel_anchor_slay_attention",
    "load_feature_state",
    "save_feature_state",
    "save_flat_joint_feature_state",
    "slay_features",
    "streaming_flat_joint_slay_attention",
]
