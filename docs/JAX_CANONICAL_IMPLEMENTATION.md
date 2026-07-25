# Canonical JAX implementation

The reviewer-facing implementation is `slay_jax/`. The older PyTorch modules
remain useful only as a frozen reproduction reference.

## Anchor generation contract

For every attention layer, sample `P` vectors independently from
`N(0, I_d)`, normalize each vector to unit L2 norm, and share that bank across
heads in the layer. Generate it once from the explicit `anchor_seed`, freeze it
for training and evaluation, and save the arrays in the checkpoint. Never
generate anchors inside the forward pass.

`slay_jax.anchors.create_feature_state` implements this contract.
`save_feature_state` and `load_feature_state` provide a standalone auditable
checkpoint. The PRF projection matrix follows the same generate-once/freeze
policy.

The current feature width is

```text
F = num_quadrature * poly_dim * prf_dim.
```

This must be reported next to every latency and memory result. In particular, a
SLAY run at `R=2, P=32, M=16` has `F=1024` and is not a feature-matched
comparison to a Performer run at `F=64`.

## Synthetic diagnostics

`jax_experiments/benchmark_synthetic.py` uses causal tasks with explicit loss
masks. Copy, reverse, and sort place outputs after a separator; long delayed copy
adds random distractors; recall tasks score only masked answer slots. The local
MQAR task performs four lookups against eight key/value pairs in one sequence.
Therefore the input token at a scored position does not reveal its target.

For final MQAR results, run both this controlled generator and the official
Zoology generator, which varies query locations and gaps. The controlled version
is intended for fast debugging and factor-wise ablation, not as a renamed
replacement for the published benchmark.

Use at least three model/data seeds for paper numbers. A task is marked solved
only at 95% masked-token accuracy, but full mean, dispersion, learning curves,
feature width, and step time are retained.

## Profiling

`jax_experiments/profile_attention.py` reports separately:

1. JAX lowering time;
2. XLA compilation time;
3. first compiled execution;
4. synchronized steady-state execution;
5. compiled cost and memory analysis;
6. isolated SLAY feature-map and recurrent-scan stages.

Anchor generation happens before profiling and is not included in execution
timings. This matches inference/training reality because anchors are persistent.

Do not implement a Pallas kernel based on compile time. Consider Pallas only
after an accelerator trace shows that the steady-state scan or feature fusion
dominates, the feature budget is already matched, and XLA has failed to fuse the
relevant operations. Pallas cannot repair the algorithmic cost of an oversized
`F = R*P*M`.

## Local commands

```bash
python -m unittest discover -s tests_jax -v
python jax_experiments/profile_attention.py
python jax_experiments/benchmark_attention.py \
  --lengths 32 64 128 256 512
python jax_experiments/benchmark_synthetic.py \
  --tasks multi_query_associative_recall long_delayed_copy induction_recall \
  --attentions softmax linear_elu performer_relu slay
```
