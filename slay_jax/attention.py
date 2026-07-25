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


def exact_spherical_yat_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    causal: bool,
    epsilon: float = 1e-6,
) -> jax.Array:
    """Exact kernel-normalized spherical Yat attention."""
    q_norm, k_norm = _normalize(q), _normalize(k)
    similarity = jnp.einsum("bhtd,bhsd->bhts", q_norm, k_norm)
    weights = jnp.square(similarity) / jnp.maximum(2.0 + epsilon - 2.0 * similarity, epsilon)
    if causal:
        time = q.shape[-2]
        weights = jnp.where(jnp.tril(jnp.ones((time, time), dtype=jnp.bool_)), weights, 0.0)
    numerator = jnp.einsum("bhts,bhsd->bhtd", weights, v)
    return numerator / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), epsilon)


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
        safe_denominator = jnp.where(
            jnp.abs(denominator) >= eps,
            denominator,
            jnp.where(denominator >= 0.0, eps, -eps),
        )
        output = numerator / safe_denominator[..., None]
        return (kv_state, k_state), output.astype(v.dtype)

    scan_step = jax.checkpoint(step) if remat_scan_body else step
    _, output = jax.lax.scan(scan_step, initial, (q_time, k_time, v_time))
    return jnp.moveaxis(output, 0, 2)


def chunked_causal_linear_attention(
    q_features: jax.Array,
    k_features: jax.Array,
    v: jax.Array,
    *,
    block_size: int = 128,
    eps: float = 1e-6,
) -> jax.Array:
    """Causal linear attention with parallel prefix work inside each block.

    This is algebraically identical to ``causal_linear_attention``. It keeps
    only the cross-block recurrent state while evaluating each block with
    vectorized cumulative sums, reducing sequential scan depth from T to
    ceil(T / block_size).
    """
    if block_size < 1:
        raise ValueError("block_size must be positive")
    batch, heads, length, feature_dim = q_features.shape
    value_dim = v.shape[-1]
    padded_length = ((length + block_size - 1) // block_size) * block_size
    padding = padded_length - length

    def pad_time(array):
        return jnp.pad(array, ((0, 0), (0, 0), (0, padding), (0, 0)))

    q_blocks = pad_time(q_features).reshape(
        batch, heads, -1, block_size, feature_dim
    )
    k_blocks = pad_time(k_features).reshape(
        batch, heads, -1, block_size, feature_dim
    )
    v_blocks = pad_time(v).reshape(batch, heads, -1, block_size, value_dim)
    q_blocks, k_blocks, v_blocks = (
        jnp.moveaxis(array, 2, 0) for array in (q_blocks, k_blocks, v_blocks)
    )
    initial = (
        jnp.zeros((batch, heads, feature_dim, value_dim), dtype=jnp.float32),
        jnp.zeros((batch, heads, feature_dim), dtype=jnp.float32),
    )

    def step(carry, inputs):
        kv_state, k_state = carry
        q_block, k_block, v_block = (
            array.astype(jnp.float32) for array in inputs
        )
        kv_increments = jnp.einsum("bhcf,bhcd->bhcfd", k_block, v_block)
        kv_prefix = jnp.cumsum(kv_increments, axis=2) + kv_state[:, :, None]
        k_prefix = jnp.cumsum(k_block, axis=2) + k_state[:, :, None]
        numerator = jnp.einsum("bhcf,bhcfd->bhcd", q_block, kv_prefix)
        denominator = jnp.einsum("bhcf,bhcf->bhc", q_block, k_prefix)
        safe_denominator = jnp.where(
            jnp.abs(denominator) >= eps,
            denominator,
            jnp.where(denominator >= 0.0, eps, -eps),
        )
        output = numerator / safe_denominator[..., None]
        return (kv_prefix[:, :, -1], k_prefix[:, :, -1]), output.astype(v.dtype)

    _, output_blocks = jax.lax.scan(step, initial, (q_blocks, k_blocks, v_blocks))
    output = jnp.moveaxis(output_blocks, 0, 2).reshape(
        batch, heads, padded_length, value_dim
    )
    return output[:, :, :length]


def parallel_prefix_causal_attention(
    q_features: jax.Array,
    k_features: jax.Array,
    v: jax.Array,
    *,
    eps: float = 1e-6,
) -> jax.Array:
    """Causal prefix evaluation that trades temporary memory for parallelism.

    This is algebraically identical to the streaming scan. It materializes the
    [B,H,T,F,D] prefix state, so it is only selected when accelerator profiling
    shows a useful speedup and the measured memory fits the target device.
    """
    kv_prefix = jnp.cumsum(
        jnp.einsum("bhtf,bhtd->bhtfd", k_features, v), axis=2
    )
    k_prefix = jnp.cumsum(k_features, axis=2)
    numerator = jnp.einsum("bhtf,bhtfd->bhtd", q_features, kv_prefix)
    denominator = jnp.einsum("bhtf,bhtf->bht", q_features, k_prefix)
    safe_denominator = jnp.where(
        jnp.abs(denominator) >= eps,
        denominator,
        jnp.where(denominator >= 0.0, eps, -eps),
    )
    return numerator / safe_denominator[..., None]


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


def _slay_token_features(
    x: jax.Array,
    state: SLAYFeatureState,
    *,
    layer_index: int,
    variant: str,
    epsilon: float,
    exp_clip: float = 10.0,
) -> jax.Array:
    """Feature map for one scan position, x=[B,H,D]."""
    x_norm = _normalize(x)
    omega = state.omega[layer_index]
    projection = jnp.einsum("bhd,rhdm->rbhm", x_norm, omega)
    nodes = state.nodes[:, None, None, None]
    exponent = jnp.clip(jnp.sqrt(2.0 * nodes) * projection - nodes, -exp_clip, exp_clip)
    prf = jnp.exp(exponent) / jnp.sqrt(omega.shape[-1])

    if variant == "anchor":
        anchors = state.anchors[layer_index]
        polynomial = jnp.square(jnp.einsum("bhd,pd->bhp", x_norm, anchors))
        polynomial = polynomial / jnp.sqrt(anchors.shape[0])
        weighted_prf = prf * jnp.sqrt(state.weights[:, None, None, None])
        fused = jnp.einsum("bhp,rbhm->bhrpm", polynomial, weighted_prf)
    elif variant == "laplace":
        constant = 2.0 + epsilon
        laplace_weights = state.weights * (constant * constant / 4.0)
        fused = prf * jnp.sqrt(laplace_weights[:, None, None, None])
        fused = jnp.transpose(fused, (1, 2, 0, 3))
    elif variant == "hadamard":
        polynomial = jnp.square(projection) / jnp.sqrt(omega.shape[-1])
        weighted_prf = prf * jnp.sqrt(state.weights[:, None, None, None])
        fused = polynomial * weighted_prf
        fused = jnp.transpose(fused, (1, 2, 0, 3))
    else:
        raise ValueError(f"unknown SLAY feature variant: {variant}")
    return fused.reshape(*x.shape[:-1], -1)


def streaming_causal_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    q_feature_fn,
    k_feature_fn,
    feature_dim: int,
    eps: float = 1e-6,
    remat_scan_body: bool = True,
) -> jax.Array:
    """Fuse feature construction into a recurrent scan.

    This avoids materializing [B,H,T,F] features or [B,H,T,F,D] prefix
    states. It is the canonical JAX causal path for large F.
    """
    q_time, k_time, v_time = (jnp.moveaxis(array, 2, 0) for array in (q, k, v))
    batch, heads, _, value_dim = v.shape
    initial = (
        jnp.zeros((batch, heads, feature_dim, value_dim), dtype=jnp.float32),
        jnp.zeros((batch, heads, feature_dim), dtype=jnp.float32),
    )

    def step(carry, inputs):
        kv_state, k_state = carry
        q_t, k_t, v_t = inputs
        q_feature = q_feature_fn(q_t).astype(jnp.float32)
        k_feature = k_feature_fn(k_t).astype(jnp.float32)
        value = v_t.astype(jnp.float32)
        kv_state = kv_state + jnp.einsum("bhf,bhd->bhfd", k_feature, value)
        k_state = k_state + k_feature
        numerator = jnp.einsum("bhf,bhfd->bhd", q_feature, kv_state)
        denominator = jnp.einsum("bhf,bhf->bh", q_feature, k_state)
        output = numerator / jnp.maximum(denominator[..., None], eps)
        return (kv_state, k_state), output.astype(v.dtype)

    body = jax.checkpoint(step) if remat_scan_body else step
    _, output = jax.lax.scan(body, initial, (q_time, k_time, v_time))
    return jnp.moveaxis(output, 0, 2)


def streaming_slay_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    state: SLAYFeatureState,
    *,
    layer_index: int = 0,
    variant: str = "anchor",
    epsilon: float = 1e-6,
    remat_scan_body: bool = True,
) -> jax.Array:
    if variant == "anchor":
        feature_dim = (
            state.anchors.shape[1] * state.omega.shape[1] * state.omega.shape[-1]
        )
    else:
        feature_dim = state.omega.shape[1] * state.omega.shape[-1]
    feature = partial(
        _slay_token_features,
        state=state,
        layer_index=layer_index,
        variant=variant,
        epsilon=epsilon,
    )
    return streaming_causal_attention(
        q,
        k,
        v,
        q_feature_fn=feature,
        k_feature_fn=feature,
        feature_dim=feature_dim,
        eps=epsilon,
        remat_scan_body=remat_scan_body,
    )


def blockwise_anchor_slay_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    state: SLAYFeatureState,
    *,
    layer_index: int = 0,
    block_size: int = 32,
    epsilon: float = 1e-6,
) -> jax.Array:
    """Exact submitted anchor feature map with block-parallel causal prefixes."""
    return chunked_causal_linear_attention(
        slay_features(q, state, layer_index=layer_index),
        slay_features(k, state, layer_index=layer_index),
        v,
        block_size=block_size,
        eps=epsilon,
    )


def parallel_anchor_slay_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    state: SLAYFeatureState,
    *,
    layer_index: int = 0,
    epsilon: float = 1e-6,
) -> jax.Array:
    """Submitted anchor map with a fully parallel, memory-heavy prefix."""
    return parallel_prefix_causal_attention(
        slay_features(q, state, layer_index=layer_index),
        slay_features(k, state, layer_index=layer_index),
        v,
        eps=epsilon,
    )


def streaming_elementwise_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    kind: str,
    performer_projection: jax.Array | None = None,
) -> jax.Array:
    """Streaming ELU, random-ReLU Performer, or Cosformer baseline."""
    if kind == "linear_elu":
        q_fn = k_fn = lambda token: jax.nn.elu(token) + 1.0
        feature_dim = q.shape[-1]
    elif kind == "performer_relu":
        if performer_projection is None:
            raise ValueError("performer_projection is required")
        feature_dim = performer_projection.shape[-1]
        q_fn = k_fn = lambda token: (
            jax.nn.relu(jnp.einsum("bhd,hdf->bhf", token, performer_projection))
            + 1e-4
        )
    else:
        raise ValueError(f"unsupported streaming baseline: {kind}")
    return streaming_causal_attention(
        q, k, v, q_feature_fn=q_fn, k_feature_fn=k_fn, feature_dim=feature_dim
    )


def streaming_cosformer_attention(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    length = q.shape[2]
    positions = jnp.arange(length, dtype=q.dtype)
    cosine = jnp.cos(jnp.pi * positions / (2.0 * length))
    sine = jnp.sin(jnp.pi * positions / (2.0 * length))
    q_time, k_time, v_time = (jnp.moveaxis(array, 2, 0) for array in (q, k, v))
    batch, heads, _, dim = q.shape
    initial = (
        jnp.zeros((batch, heads, 2 * dim, dim), dtype=jnp.float32),
        jnp.zeros((batch, heads, 2 * dim), dtype=jnp.float32),
    )

    def step(carry, inputs):
        kv_state, k_state = carry
        q_t, k_t, v_t, cos_t, sin_t = inputs
        q_relu, k_relu = jax.nn.relu(q_t), jax.nn.relu(k_t)
        qf = jnp.concatenate((q_relu * cos_t, q_relu * sin_t), axis=-1)
        kf = jnp.concatenate((k_relu * cos_t, k_relu * sin_t), axis=-1)
        kv_state = kv_state + jnp.einsum("bhf,bhd->bhfd", kf, v_t)
        k_state = k_state + kf
        output = jnp.einsum("bhf,bhfd->bhd", qf, kv_state)
        norm = jnp.einsum("bhf,bhf->bh", qf, k_state)
        return (kv_state, k_state), output / jnp.maximum(norm[..., None], 1e-6)

    _, output = jax.lax.scan(
        jax.checkpoint(step), initial, (q_time, k_time, v_time, cosine, sine)
    )
    return jnp.moveaxis(output, 0, 2)


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
