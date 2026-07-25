"""Pure JAX attention kernels used by benchmarks and task models."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .anchors import SLAYFeatureState


def _normalize(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def exact_softmax_attention(
    q: jax.Array, k: jax.Array, v: jax.Array, *, causal: bool
) -> jax.Array:
    """Exact softmax attention, used only as a quadratic systems baseline."""
    scale = q.shape[-1] ** -0.5
    scores = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale
    if causal:
        time = q.shape[-2]
        mask = jnp.tril(jnp.ones((time, time), dtype=jnp.bool_))
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhts,bhsd->bhtd", weights, v)


def causal_linear_attention(
    q_features: jax.Array,
    k_features: jax.Array,
    v: jax.Array,
    *,
    eps: float = 1e-6,
    remat_scan_body: bool = False,
) -> jax.Array:
    """Causal associative attention with O(T) scan memory.

    No [B,H,T,F,D] prefix tensor is materialized.
    """
    q_time = jnp.moveaxis(q_features, 2, 0)
    k_time = jnp.moveaxis(k_features, 2, 0)
    v_time = jnp.moveaxis(v, 2, 0)
    batch, heads, _, feature_dim = q_features.shape
    value_dim = v.shape[-1]
    accumulator_dtype = jnp.float32
    initial = (
        jnp.zeros((batch, heads, feature_dim, value_dim), dtype=accumulator_dtype),
        jnp.zeros((batch, heads, feature_dim), dtype=accumulator_dtype),
    )

    def step(carry, inputs):
        kv_state, k_state = carry
        q_t, k_t, v_t = inputs
        k_acc = k_t.astype(accumulator_dtype)
        v_acc = v_t.astype(accumulator_dtype)
        q_acc = q_t.astype(accumulator_dtype)
        kv_state = kv_state + jnp.einsum("bhf,bhd->bhfd", k_acc, v_acc)
        k_state = k_state + k_acc
        numerator = jnp.einsum("bhf,bhfd->bhd", q_acc, kv_state)
        denominator = jnp.einsum("bhf,bhf->bh", q_acc, k_state)
        output = numerator / jnp.maximum(denominator[..., None], eps)
        return (kv_state, k_state), output.astype(v.dtype)

    scan_step = jax.checkpoint(step) if remat_scan_body else step
    _, output = jax.lax.scan(scan_step, initial, (q_time, k_time, v_time))
    return jnp.moveaxis(output, 0, 2)


def bidirectional_linear_attention(
    q_features: jax.Array, k_features: jax.Array, v: jax.Array, *, eps: float = 1e-6
) -> jax.Array:
    kv = jnp.einsum("bhtf,bhtd->bhfd", k_features, v)
    k_sum = jnp.sum(k_features, axis=2)
    numerator = jnp.einsum("bhtf,bhfd->bhtd", q_features, kv)
    denominator = jnp.einsum("bhtf,bhf->bht", q_features, k_sum)
    return numerator / jnp.maximum(denominator[..., None], eps)


def elu_features(x: jax.Array) -> jax.Array:
    return jax.nn.elu(x) + 1.0


def relu_random_features(x: jax.Array, projection: jax.Array) -> jax.Array:
    """Positive random ReLU features; projection is [H,D,F]."""
    return jax.nn.relu(jnp.einsum("bhtd,hdf->bhtf", x, projection)) + 1e-4


@partial(jax.named_call, name="slay_feature_map")
def slay_features(
    x: jax.Array,
    state: SLAYFeatureState,
    *,
    layer_index: int,
    exp_clip: float = 10.0,
) -> jax.Array:
    """Compute anchor × Laplace-PRF tensor features [B,H,T,R*P*M]."""
    x_norm = _normalize(x)
    anchors = state.anchors[layer_index]
    omega = state.omega[layer_index]

    with jax.named_scope("anchor_polynomial_features"):
        anchor_projection = jnp.einsum("bhtd,pd->bhtp", x_norm, anchors)
        polynomial = jnp.square(anchor_projection) / jnp.sqrt(anchors.shape[0])

    with jax.named_scope("laplace_prf_features"):
        projection = jnp.einsum("bhtd,rhdm->rbhtm", x_norm, omega)
        nodes = state.nodes[:, None, None, None, None]
        exponent = jnp.clip(jnp.sqrt(2.0 * nodes) * projection - nodes, -exp_clip, exp_clip)
        prf = jnp.exp(exponent) / jnp.sqrt(omega.shape[-1])
        prf = prf * jnp.sqrt(state.weights[:, None, None, None, None])

    with jax.named_scope("tensor_feature_fusion"):
        fused = jnp.einsum("bhtp,rbhtm->bhtrpm", polynomial, prf)
        return fused.reshape(*x.shape[:-1], -1)


def attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    kind: str,
    causal: bool,
    feature_state: SLAYFeatureState | None = None,
    layer_index: int = 0,
    performer_projection: jax.Array | None = None,
    remat_scan_body: bool = False,
) -> jax.Array:
    """Dispatch a common-shape attention benchmark implementation."""
    if kind == "softmax":
        return exact_softmax_attention(q, k, v, causal=causal)
    if kind == "linear_elu":
        qf, kf = elu_features(q), elu_features(k)
    elif kind == "performer_relu":
        if performer_projection is None:
            raise ValueError("performer_projection is required")
        qf = relu_random_features(q, performer_projection)
        kf = relu_random_features(k, performer_projection)
    elif kind == "slay":
        if feature_state is None:
            raise ValueError("feature_state is required")
        qf = slay_features(q, feature_state, layer_index=layer_index)
        kf = slay_features(k, feature_state, layer_index=layer_index)
    else:
        raise ValueError(f"unknown attention kind: {kind}")

    if causal:
        return causal_linear_attention(
            qf, kf, v, remat_scan_body=remat_scan_body
        )
    return bidirectional_linear_attention(qf, kf, v)
