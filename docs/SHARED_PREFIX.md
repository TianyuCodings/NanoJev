# Shared-prefix candidate encoding

Every candidate path of a question is `prefix + candidate_suffix`, where the prefix is
the state text plus the question text. That prefix is byte-identical for all candidates
of the question, so the reference encoder evaluates it once per candidate. This document
describes the shared-prefix encoder that evaluates each prefix once, states the
equivalence argument, and reports what it saves.

## What the reference encoder does

`predict_toy_decisions.prepare_examples` and `train_toy_decisions.load_examples` both
build paths as:

```python
prefix = encode("State:\n{state}\n") + encode("Question type: {type}\nQuestion:\n{...}\n")
paths  = [prefix + encode("Candidate:\n{text}\nDecision:") + [eos] for text in texts]
```

`DecisionModel.encode_leaves` concatenates those paths into one flat batch, pads them to
the widest path, and runs the backbone once. Every row therefore contains a full copy of
the prefix. For `K` candidates and a `P`-token prefix with suffix width `S`, one question
costs `K * (P + S)` token evaluations of which `K * P` are duplicate work.

## What the shared-prefix encoder does

`SharedPrefixEncoder.encode` splits the work in two stages:

1. **Prefix stage.** Encode the prefix alone as a single row with `use_cache=True` and
   keep the resulting keys and values.
2. **Suffix stage.** For each candidate, run only its suffix tokens. The suffix queries
   attend to the cached prefix plus their own causal suffix through an additive mask of
   shape `(rows, 1, S_max, P + S_max)`. Position ids continue at `P`, and `cache_position`
   is passed so the cache's own bookkeeping matches the positions. The mask is what
   constrains attention: transformers passes a supplied 4-D mask through verbatim and adds
   no causal safety net, which is why `assert_supported_attention` restricts the backend to
   eager or SDPA (Flash-Attention 2 rejects a float 4-D mask) and `assert_supported_cache`
   rejects static, sliding-window and hybrid caches whose in-place storage writes would
   disturb the shared prefix.

The prefix keys and values are handed to the model through a zero-stride `expand`, so the
broadcast itself does not copy them, and the prefix is never re-encoded. Candidate rows are
processed in optional chunks (`suffix_chunk`) so that the cache built during the suffix pass
stays bounded.

**Memory caveat.** The scan above is about compute. The KV cache is a separate matter:
`DynamicLayer.update` appends with `torch.cat(..., dim=-2)`, so the suffix stage materializes
`chunk` copies of the prefix keys and values alongside their suffixes — `O(chunk * (P + S))`,
not `O(P + chunk * S)`. Chunking bounds the candidate dimension, but the prefix is *not* held
once during attention. The saving is re-encoding work, not peak cache memory, and at small
`K` the added mask and the per-question prefill can outweigh it (see the timings below).

## Equivalence argument

For candidate `c`, the reference reads the final hidden state of the token sequence
`prefix + s_c` under a causal mask. The shared path presents the same sequence: stage one
produces the prefix keys/values, stage two appends `s_c`'s keys/values and lets query
position `P + j` attend to prefix positions `0..P-1` and suffix positions `0..j`. The keys,
values and query positions are identical, so the attention inputs are identical. Only the
reduction order inside the softmax differs.

The tests pin this on a real checkpoint rather than only on a toy network:

* `scripts/test_shared_prefix.py::RealCheckpointTests::test_real_model_equivalence_and_reduction`
  runs the real Qwen3-0.6B backbone in float32 and asserts a maximum absolute logit
  deviation below `1e-3`; the observed value is a few `1e-6`.
* `scripts/test_shared_prefix.py::TinyModelTests` covers mixed candidate counts and widths,
  Boolean single-path questions, chunked encoding, and gradient flow through the shared
  path.

A note on dtype. Precision changes the comparison, so the tests state what they measure:

* **float32** — the anchor. Deviation is a few `1e-6` (observed `1.19e-6` to `4.3e-6`).
* **bfloat16 with SDPA** — bit-identical when the attention implementation is held fixed.
* **bfloat16 autocast, which is what the service runs** — `0.02` absolute on logits of
  order `1`. This is bf16 softmax reduction-order noise, not a mask defect: the same model
  differs from *itself* by `5.5e-2` in bfloat16 when only the attention implementation
  changes. `test_bf16_autocast_matches_under_matched_attention` pins the relative bound,
  and a wrong mask moves logits by order `1`, so that test still fails on a real defect.

## Prefixes are shared across questions only on exact identity

Two questions can tokenize to equal prefixes, and repeated action sets across states are
common. The planner records an exact reuse (`covered_by`) only when the prefix tokens *and*
every candidate suffix are identical, which makes the cached keys and values the same
tensors; the reused question costs no prefix encode and no suffix encode. No approximate or
fuzzy prefix match is attempted, and `accounting()["reused_questions"]` reports how many
questions were served this way.

An authored `prefix_length` (the state-plus-question text) bounds the split, so the prefix
never absorbs candidate text. Without that declaration the whole observed common run is
shared and the group is reported under `unverified_split_questions`.

## What it saves

Reported by `SharedPrefixPlan.accounting()`, which depends only on the token layout and is
therefore hardware independent:

| Quantity | Reference | Shared |
| --- | --- | --- |
| Prefix token evaluations | `K * P` | `P` |
| Suffix token evaluations | `K * S_max` | `sum(len(s_c))` |
| Batch padding | every row padded to the widest path in the batch | none |

`scripts/benchmark_shared_prefix.py` measures the real checkpoint. On an Apple M-series GPU
with `sdpa` in float32, two states per candidate count. `accounting()` counts the tokens the
encoder actually evaluates, including chunk padding, and counts every supplied example in the
reference cost — an earlier revision summed unpadded suffixes and skipped reused examples,
which reported `1.96x` on a case whose true value is `1.01x`.

Two states with **different** state text; Qwen3-0.6B weights loaded strictly (the base
checkpoint stores `model.*` keys, so a naive `strict=False` load matches nothing and leaves
the backbone random — see the artifact's `corrections` field).

| Candidates | Leaf tokens reference → shared | Token reduction | Wall clock (MPS fp32, 5 repeats) | Speedup | Max logit drift |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 672 → 324 | 2.07× | 251 ms → 177 ms | 1.42× | 5.3e-06 |
| 16 | 2752 → 1012 | 2.72× | 1114 ms → 616 ms | 1.81× | 8.3e-06 |
| 64 | 11008 → 3700 | 2.98× | 2976 ms → 1049 ms | 2.84× | 8.9e-06 |

The full artifact is `results/shared_prefix_benchmark.json`.

### When it pays off

There is no single candidate-count threshold: the benefit depends on the candidate count, the
shared prefix length, the per-candidate suffix width and the chunk size, together. Measured
points from an independent review on the same machine (M1 Max, MPS, fp32, SDPA, released
weights loaded strictly, device-synchronised timing, alternating order, median of five):

| Workload | Reference | Shared | Ratio |
| --- | ---: | ---: | ---: |
| 2 candidates, 66-token prefix | 105 ms | 162 ms | **0.65x** |
| 4 candidates, 66-token prefix | 188 ms | 182 ms | 1.03x |
| 16 candidates, 66-token prefix | 645 ms | 442 ms | 1.46x |
| 64 candidates, 66-token prefix | 5400 ms | 1762 ms | 3.06x |
| 255 candidates, 66-token prefix | 8738 ms | 3586 ms | 2.44x |
| 16 candidates, 696-token prefix | 3820 ms | 635 ms | 6.02x |

Read as a rule of thumb rather than a threshold:

* **Candidate count sets the ceiling.** The prefix is re-encoded `K` times in the reference
  and once here, so the removable work grows with `K` while the per-question overhead does
  not. Single-digit `K` is where the overhead wins and the change is a regression.
* **Prefix length sets the payoff.** The same 16 candidates went from 1.46x to 6.02x when the
  prefix grew tenfold, because the re-encoded prefix is what is being removed. Long state
  text is the case this change exists for.
* **Chunking trades memory for speed.** Capping suffix rows reduced a 16-candidate case to
  `0.72x` in that review. Chunk only when the cache is the binding constraint.
* **Few-token candidates lose either way.** A Boolean question has one short path, so a batch
  of them measured `0.40x`. This change is for many-candidate Choice and Score questions.

Other caveats:

* Timings on this machine are noisy (MPS medians moved ~50% between runs). Token reduction is
  exact and hardware independent; wall clock is recorded, never promised. The review's numbers
  and this document's differ for that reason; both are single-machine observations.
* Token reduction is not wall-clock reduction: the suffix pass still attends over the cached
  prefix, so attention cost grows with prefix length even though the prefix is not re-encoded.
* The memory complexity claim was **wrong in an earlier revision of this document** and is
  corrected above: the suffix pass holds `O(chunk * (P + S))`. A module-boundary sample in the
  same review measured roughly 71 MB additional live tensors for the reference, 729 MB shared,
  and 201 MB shared with `suffix_chunk=4` at 16 candidates, so do not expect a memory win.
* Only eager and SDPA were measured, on CPU and MPS. The production path is CUDA bfloat16,
  which was not available. The default stays `prefix_sharing=False` for this reason.

## Using it

```python
from shared_prefix import SharedPrefixEncoder
encoder = SharedPrefixEncoder(model.backbone, pad_token_id=tokenizer.pad_token_id)
logits, valid = model(examples, tokenizer.pad_token_id, prefix_sharing=encoder)
```

`DecisionPredictor.predict(payload, prefix_sharing=True, suffix_chunk=None)` selects the
same path for the serving entry point, and the CLI exposes it as `--prefix-sharing` /
`--suffix-chunk`. The response reports `execution.prefix_sharing`, `execution.tree_attention`,
`execution.prefix_sharing_requested`, `execution.suffix_chunk` and `execution.sharing`.

Those flags report **what ran**, not what was requested. A batch whose questions have no
shareable prefix is handed back to the reference encoder inside `DecisionModel.forward`, and
in that case `prefix_sharing` stays `false` with `sharing.fallback_reason` set, even though
`prefix_sharing_requested` is `true`. The default remains `False`, so existing responses
keep their recorded `prefix_sharing: false` semantics.

The trainer still defaults to the reference encoder. `scripts/train_unified_games.py`
imports `DecisionModel` and calls `model(group, pad_token)`; that call site is unchanged and
keeps the reference layout. Enabling sharing during training is a separate decision that
changes the update path, so it is deliberately not part of this change.

## Batch semantics

With sharing enabled the encoder builds one prefix cache per question, so `batch_questions`
no longer partitions backbone work; the reported `forward_passes` is `1` regardless of that
limit. The reference path keeps its historical batching. Both paths produce the same
published probabilities, which
`scripts/test_shared_prefix_predict.py::PredictorParityTests::test_reference_path_honours_batch_questions_but_shared_path_does_not`
asserts.
