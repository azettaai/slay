from __future__ import annotations

import numpy as np
import tempfile
import unittest
from pathlib import Path

from slay_jax.anchors import (
    ANCHOR_POLICY,
    create_feature_state,
    load_feature_state,
    save_feature_state,
)


def _state(seed: int):
    return create_feature_state(
        seed=seed,
        num_layers=2,
        num_heads=2,
        head_dim=4,
        poly_dim=5,
        prf_dim=3,
        num_quadrature=2,
    )


class AnchorStateTest(unittest.TestCase):
    def test_anchor_generation_is_deterministic_and_seeded(self):
        first = _state(17)
        repeat = _state(17)
        different = _state(18)
        np.testing.assert_array_equal(first.anchors, repeat.anchors)
        np.testing.assert_array_equal(first.omega, repeat.omega)
        self.assertFalse(np.array_equal(first.anchors, different.anchors))
        np.testing.assert_allclose(
            np.linalg.norm(np.asarray(first.anchors), axis=-1), 1.0, atol=1e-6
        )

    def test_feature_state_round_trip_does_not_regenerate(self):
        original = _state(23)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "frozen_features.npz"
            save_feature_state(checkpoint, original, seed=23, epsilon=1e-6)
            restored, metadata = load_feature_state(checkpoint)
        np.testing.assert_array_equal(original.anchors, restored.anchors)
        np.testing.assert_array_equal(original.omega, restored.omega)
        np.testing.assert_array_equal(original.nodes, restored.nodes)
        np.testing.assert_array_equal(original.weights, restored.weights)
        self.assertEqual(metadata["seed"], 23)
        self.assertEqual(metadata["anchor_policy"], ANCHOR_POLICY)


if __name__ == "__main__":
    unittest.main()
