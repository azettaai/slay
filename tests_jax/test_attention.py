from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import unittest

from slay_jax.anchors import create_feature_state
from slay_jax.attention import (
    bidirectional_linear_attention,
    causal_linear_attention,
    slay_features,
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
