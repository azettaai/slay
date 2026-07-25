from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import unittest

from slay_jax.anchors import create_feature_state
from slay_jax.attention import (
    bidirectional_linear_attention,
    blockwise_anchor_slay_attention,
    causal_linear_attention,
    chunked_causal_linear_attention,
    parallel_prefix_causal_attention,
    parallel_anchor_slay_attention,
    slay_features,
    streaming_slay_attention,
)


class AttentionTest(unittest.TestCase):
    def test_scan_matches_materialized_causal_prefix(self):
        qf = jax.random.uniform(jax.random.PRNGKey(0), (2, 2, 7, 5)) + 0.1
        kf = jax.random.uniform(jax.random.PRNGKey(1), (2, 2, 7, 5)) + 0.1
        v = jax.random.normal(jax.random.PRNGKey(2), (2, 2, 7, 3))
        actual = causal_linear_attention(qf, kf, v)
        weights = jnp.einsum("bhtf,bhsf->bhts", qf, kf)
        weights = jnp.where(jnp.tril(jnp.ones((7, 7), dtype=bool)), weights, 0.0)
        expected = jnp.einsum("bhts,bhsd->bhtd", weights, v)
        expected /= jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-6)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_causal_attention_has_no_future_dependency(self):
        qf = jnp.ones((1, 1, 5, 3))
        kf = jnp.ones((1, 1, 5, 3))
        first_values = jnp.arange(5.0).reshape(1, 1, 5, 1)
        changed_future = first_values.at[:, :, 3:, :].set(1000.0)
        first = causal_linear_attention(qf, kf, first_values)
        second = causal_linear_attention(qf, kf, changed_future)
        np.testing.assert_allclose(first[:, :, :3], second[:, :, :3])

    def test_chunked_prefix_matches_token_scan_with_padding(self):
        qf = jax.random.uniform(jax.random.key(20), (2, 2, 11, 7)) + 0.1
        kf = jax.random.uniform(jax.random.key(21), (2, 2, 11, 7)) + 0.1
        v = jax.random.normal(jax.random.key(22), (2, 2, 11, 3))
        expected = causal_linear_attention(qf, kf, v)
        actual = chunked_causal_linear_attention(qf, kf, v, block_size=4)
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)

    def test_parallel_prefix_matches_token_scan(self):
        qf = jax.random.uniform(jax.random.key(23), (1, 2, 9, 6)) + 0.1
        kf = jax.random.uniform(jax.random.key(24), (1, 2, 9, 6)) + 0.1
        v = jax.random.normal(jax.random.key(25), (1, 2, 9, 4))
        expected = causal_linear_attention(qf, kf, v)
        actual = parallel_prefix_causal_attention(qf, kf, v)
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)

    def test_slay_feature_dimension_and_finiteness(self):
        state = create_feature_state(
            seed=4,
            num_layers=1,
            num_heads=2,
            head_dim=4,
            poly_dim=3,
            prf_dim=5,
            num_quadrature=2,
        )
        x = jax.random.normal(jax.random.PRNGKey(5), (2, 2, 6, 4))
        features = slay_features(x, state, layer_index=0)
        self.assertEqual(features.shape, (2, 2, 6, 30))
        self.assertTrue(bool(jnp.all(jnp.isfinite(features))))

    def test_streaming_slay_matches_materialized_features(self):
        state = create_feature_state(
            seed=9,
            num_layers=1,
            num_heads=2,
            head_dim=4,
            poly_dim=3,
            prf_dim=2,
            num_quadrature=2,
        )
        q = jax.random.normal(jax.random.PRNGKey(10), (1, 2, 6, 4))
        k = jax.random.normal(jax.random.PRNGKey(11), (1, 2, 6, 4))
        v = jax.random.normal(jax.random.PRNGKey(12), (1, 2, 6, 4))
        expected = causal_linear_attention(
            slay_features(q, state, layer_index=0),
            slay_features(k, state, layer_index=0),
            v,
        )
        actual = streaming_slay_attention(
            q, k, v, state, layer_index=0, variant="anchor"
        )
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)

        blocked = blockwise_anchor_slay_attention(
            q, k, v, state, layer_index=0, block_size=4
        )
        np.testing.assert_allclose(blocked, expected, rtol=3e-5, atol=3e-5)
        parallel = parallel_anchor_slay_attention(
            q, k, v, state, layer_index=0
        )
        np.testing.assert_allclose(parallel, expected, rtol=3e-5, atol=3e-5)

    def test_bidirectional_associative_form_matches_explicit_weights(self):
        qf = jax.random.uniform(jax.random.PRNGKey(6), (1, 2, 5, 4)) + 0.1
        kf = jax.random.uniform(jax.random.PRNGKey(7), (1, 2, 5, 4)) + 0.1
        v = jax.random.normal(jax.random.PRNGKey(8), (1, 2, 5, 3))
        actual = bidirectional_linear_attention(qf, kf, v)
        weights = jnp.einsum("bhtf,bhsf->bhts", qf, kf)
        expected = jnp.einsum("bhts,bhsd->bhtd", weights, v)
        expected /= jnp.sum(weights, axis=-1, keepdims=True)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
