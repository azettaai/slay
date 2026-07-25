from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import unittest

from slay_jax.synthetic_tasks import TASKS


class SyntheticTaskTest(unittest.TestCase):
    def test_task_shapes_ranges_and_nonempty_loss_mask(self):
        for task_name, spec in TASKS.items():
            with self.subTest(task=task_name):
                batch = spec.generator(jax.random.PRNGKey(0), 7)
                self.assertEqual(batch.inputs.shape, (7, spec.sequence_length))
                self.assertEqual(batch.targets.shape, batch.inputs.shape)
                self.assertEqual(batch.loss_mask.shape, batch.inputs.shape)
                self.assertGreaterEqual(int(jnp.min(batch.inputs)), 0)
                self.assertLess(int(jnp.max(batch.inputs)), spec.vocab_size)
                self.assertGreaterEqual(int(jnp.min(batch.targets)), 0)
                self.assertLess(int(jnp.max(batch.targets)), spec.vocab_size)
                self.assertGreater(float(jnp.sum(batch.loss_mask)), 0)

    def test_delayed_copy_does_not_expose_answers_at_scored_positions(self):
        spec = TASKS["delayed_copy"]
        batch = spec.generator(jax.random.PRNGKey(1), 16)
        scored_inputs = np.asarray(batch.inputs)[np.asarray(batch.loss_mask, dtype=bool)]
        scored_targets = np.asarray(batch.targets)[np.asarray(batch.loss_mask, dtype=bool)]
        self.assertTrue(np.all(scored_inputs == 9))  # MASK token
        self.assertTrue(np.any(scored_inputs != scored_targets))

    def test_associative_recall_answer_matches_queried_pair(self):
        batch = TASKS["associative_recall"].generator(jax.random.PRNGKey(2), 32)
        pairs = np.asarray(batch.inputs[:, :8]).reshape(32, 4, 2)
        query_keys = np.asarray(batch.inputs[:, -2])
        answers = np.asarray(batch.targets[:, -1])
        for row in range(32):
            matches = pairs[row, :, 0] == query_keys[row]
            self.assertEqual(matches.sum(), 1)
            self.assertEqual(pairs[row, matches, 1].item(), answers[row])

    def test_mqar_answers_match_each_queried_key(self):
        batch = TASKS["multi_query_associative_recall"].generator(
            jax.random.PRNGKey(3), 32
        )
        pairs = np.asarray(batch.inputs[:, :16]).reshape(32, 8, 2)
        query_keys = np.asarray(batch.inputs[:, 16:])
        answers = np.asarray(batch.targets[:, 16:])
        mask = np.asarray(batch.loss_mask[:, 16:])
        self.assertTrue(np.all(mask == 1))
        for row in range(32):
            lookup = dict(pairs[row])
            for query, answer in zip(query_keys[row], answers[row]):
                self.assertEqual(lookup[query], answer)


if __name__ == "__main__":
    unittest.main()
