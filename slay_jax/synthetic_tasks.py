"""Non-leaky causal synthetic tasks for attention diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class TaskBatch(NamedTuple):
    inputs: jax.Array
    targets: jax.Array
    loss_mask: jax.Array


@dataclass(frozen=True)
class TaskSpec:
    name: str
    sequence_length: int
    vocab_size: int
    chance_accuracy: float
    description: str
    generator: Callable[[jax.Array, int], TaskBatch]


def _masked_targets(inputs: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jnp.zeros_like(inputs), jnp.zeros_like(inputs, dtype=jnp.float32)


def delayed_copy(key: jax.Array, batch_size: int, *, data_length: int = 8, alphabet: int = 8) -> TaskBatch:
    """[data, SEP, MASK...] -> predict data only after SEP."""
    data = jax.random.randint(key, (batch_size, data_length), 0, alphabet)
    sep, mask_token = alphabet, alphabet + 1
    inputs = jnp.concatenate(
        [
            data,
            jnp.full((batch_size, 1), sep, dtype=jnp.int32),
            jnp.full((batch_size, data_length), mask_token, dtype=jnp.int32),
        ],
        axis=1,
    )
    targets, loss_mask = _masked_targets(inputs)
    targets = targets.at[:, data_length + 1 :].set(data)
    loss_mask = loss_mask.at[:, data_length + 1 :].set(1.0)
    return TaskBatch(inputs, targets, loss_mask)


def long_delayed_copy(
    key: jax.Array,
    batch_size: int,
    *,
    data_length: int = 8,
    delay_length: int = 32,
    alphabet: int = 8,
) -> TaskBatch:
    """Copy after random distractors, isolating long-range information retention."""
    data_key, distractor_key = jax.random.split(key)
    data = jax.random.randint(data_key, (batch_size, data_length), 0, alphabet)
    distractors = jax.random.randint(
        distractor_key, (batch_size, delay_length), 0, alphabet
    )
    sep, mask_token = alphabet, alphabet + 1
    inputs = jnp.concatenate(
        [
            data,
            distractors,
            jnp.full((batch_size, 1), sep, dtype=jnp.int32),
            jnp.full((batch_size, data_length), mask_token, dtype=jnp.int32),
        ],
        axis=1,
    )
    targets, loss_mask = _masked_targets(inputs)
    output_start = data_length + delay_length + 1
    targets = targets.at[:, output_start:].set(data)
    loss_mask = loss_mask.at[:, output_start:].set(1.0)
    return TaskBatch(inputs, targets, loss_mask)


def delayed_reverse(key: jax.Array, batch_size: int, *, data_length: int = 8, alphabet: int = 8) -> TaskBatch:
    batch = delayed_copy(key, batch_size, data_length=data_length, alphabet=alphabet)
    targets = batch.targets.at[:, data_length + 1 :].set(
        jnp.flip(batch.inputs[:, :data_length], axis=1)
    )
    return TaskBatch(batch.inputs, targets, batch.loss_mask)


def delayed_sort(key: jax.Array, batch_size: int, *, data_length: int = 8, alphabet: int = 8) -> TaskBatch:
    batch = delayed_copy(key, batch_size, data_length=data_length, alphabet=alphabet)
    targets = batch.targets.at[:, data_length + 1 :].set(
        jnp.sort(batch.inputs[:, :data_length], axis=1)
    )
    return TaskBatch(batch.inputs, targets, batch.loss_mask)


def associative_recall(
    key: jax.Array,
    batch_size: int,
    *,
    num_pairs: int = 4,
    num_keys: int = 8,
    num_values: int = 8,
) -> TaskBatch:
    """[k1,v1,...,QUERY,k_i,MASK] -> predict v_i at the last token."""
    if num_pairs > num_keys:
        raise ValueError("num_pairs cannot exceed num_keys")
    key_perm, key_values, key_query = jax.random.split(key, 3)
    row_keys = jax.random.split(key_perm, batch_size)
    keys = jax.vmap(lambda k: jax.random.permutation(k, num_keys)[:num_pairs])(row_keys)
    values = jax.random.randint(key_values, (batch_size, num_pairs), 0, num_values)
    values = values + num_keys
    query_indices = jax.random.randint(key_query, (batch_size,), 0, num_pairs)
    rows = jnp.arange(batch_size)
    query_keys = keys[rows, query_indices]
    answers = values[rows, query_indices]
    query_token = num_keys + num_values
    mask_token = query_token + 1
    pairs = jnp.stack([keys, values], axis=-1).reshape(batch_size, 2 * num_pairs)
    inputs = jnp.concatenate(
        [
            pairs,
            jnp.full((batch_size, 1), query_token, dtype=jnp.int32),
            query_keys[:, None],
            jnp.full((batch_size, 1), mask_token, dtype=jnp.int32),
        ],
        axis=1,
    )
    targets, loss_mask = _masked_targets(inputs)
    targets = targets.at[:, -1].set(answers)
    loss_mask = loss_mask.at[:, -1].set(1.0)
    return TaskBatch(inputs, targets, loss_mask)


def multi_query_associative_recall(
    key: jax.Array,
    batch_size: int,
    *,
    num_pairs: int = 8,
    num_queries: int = 4,
    num_keys: int = 16,
    num_values: int = 16,
) -> TaskBatch:
    """Multiple exact lookups in one sequence with loss only on query slots.

    This controlled local version places a key/value table first and then emits
    query keys. At each query-key position, the causal LM must predict the
    associated value. The later accelerator suite should also use the official
    Zoology generator, which varies query positions and gaps.
    """
    if num_pairs > num_keys:
        raise ValueError("num_pairs cannot exceed num_keys")
    permutation_key, value_key, query_key = jax.random.split(key, 3)
    row_keys = jax.random.split(permutation_key, batch_size)
    keys = jax.vmap(lambda k: jax.random.permutation(k, num_keys)[:num_pairs])(
        row_keys
    )
    values = (
        jax.random.randint(value_key, (batch_size, num_pairs), 0, num_values)
        + num_keys
    )
    query_indices = jax.random.randint(
        query_key, (batch_size, num_queries), 0, num_pairs
    )
    query_keys = jnp.take_along_axis(keys, query_indices, axis=1)
    answers = jnp.take_along_axis(values, query_indices, axis=1)
    pairs = jnp.stack([keys, values], axis=-1).reshape(batch_size, 2 * num_pairs)
    inputs = jnp.concatenate([pairs, query_keys], axis=1)
    targets, loss_mask = _masked_targets(inputs)
    query_positions = 2 * num_pairs + jnp.arange(num_queries)
    targets = targets.at[:, query_positions].set(answers)
    loss_mask = loss_mask.at[:, query_positions].set(1.0)
    return TaskBatch(inputs, targets, loss_mask)


def first_token_recall(
    key: jax.Array, batch_size: int, *, data_length: int = 16, alphabet: int = 8
) -> TaskBatch:
    """Remember the first token through distractors and answer a final query."""
    data = jax.random.randint(key, (batch_size, data_length), 0, alphabet)
    query_token, mask_token = alphabet, alphabet + 1
    inputs = jnp.concatenate(
        [
            data,
            jnp.full((batch_size, 1), query_token, dtype=jnp.int32),
            jnp.full((batch_size, 1), mask_token, dtype=jnp.int32),
        ],
        axis=1,
    )
    targets, loss_mask = _masked_targets(inputs)
    targets = targets.at[:, -1].set(data[:, 0])
    loss_mask = loss_mask.at[:, -1].set(1.0)
    return TaskBatch(inputs, targets, loss_mask)


def induction_recall(
    key: jax.Array, batch_size: int, *, distractor_length: int = 12, alphabet: int = 8
) -> TaskBatch:
    """A repeated marker must retrieve the token that followed its first use."""
    value_key, distractor_key = jax.random.split(key)
    values = jax.random.randint(value_key, (batch_size,), 0, alphabet)
    distractors = jax.random.randint(
        distractor_key, (batch_size, distractor_length), 0, alphabet
    )
    marker, mask_token = alphabet, alphabet + 1
    inputs = jnp.concatenate(
        [
            jnp.full((batch_size, 1), marker, dtype=jnp.int32),
            values[:, None],
            distractors,
            jnp.full((batch_size, 1), marker, dtype=jnp.int32),
            jnp.full((batch_size, 1), mask_token, dtype=jnp.int32),
        ],
        axis=1,
    )
    targets, loss_mask = _masked_targets(inputs)
    targets = targets.at[:, -1].set(values)
    loss_mask = loss_mask.at[:, -1].set(1.0)
    return TaskBatch(inputs, targets, loss_mask)


TASKS = {
    "delayed_copy": TaskSpec(
        "delayed_copy", 17, 10, 1 / 8, "Copy eight symbols after a separator.", delayed_copy
    ),
    "delayed_reverse": TaskSpec(
        "delayed_reverse", 17, 10, 1 / 8, "Reverse after a separator.", delayed_reverse
    ),
    "delayed_sort": TaskSpec(
        "delayed_sort", 17, 10, 1 / 8, "Sort after a separator.", delayed_sort
    ),
    "long_delayed_copy": TaskSpec(
        "long_delayed_copy",
        49,
        10,
        1 / 8,
        "Copy eight symbols after 32 random distractors.",
        long_delayed_copy,
    ),
    "associative_recall": TaskSpec(
        "associative_recall", 11, 18, 1 / 8, "Retrieve a value associated with a queried key.", associative_recall
    ),
    "multi_query_associative_recall": TaskSpec(
        "multi_query_associative_recall",
        20,
        32,
        1 / 16,
        "Retrieve four values from an eight-pair table.",
        multi_query_associative_recall,
    ),
    "first_token_recall": TaskSpec(
        "first_token_recall", 18, 10, 1 / 8, "Recall the first token after distractors.", first_token_recall
    ),
    "induction_recall": TaskSpec(
        "induction_recall", 16, 10, 1 / 8, "Retrieve the value following an earlier marker.", induction_recall
    ),
}


def masked_cross_entropy(logits: jax.Array, batch: TaskBatch) -> jax.Array:
    losses = -jax.nn.log_softmax(logits, axis=-1)
    token_losses = jnp.take_along_axis(losses, batch.targets[..., None], axis=-1)[..., 0]
    return jnp.sum(token_losses * batch.loss_mask) / jnp.maximum(jnp.sum(batch.loss_mask), 1.0)


def masked_accuracy(logits: jax.Array, batch: TaskBatch) -> jax.Array:
    correct = (jnp.argmax(logits, axis=-1) == batch.targets).astype(jnp.float32)
    return jnp.sum(correct * batch.loss_mask) / jnp.maximum(jnp.sum(batch.loss_mask), 1.0)
