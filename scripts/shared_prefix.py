#!/usr/bin/env python3
"""Shared-prefix candidate encoding: compute each state/question prefix once.

Every candidate path of a question is `prefix + candidate_suffix`. The reference
implementation writes that prefix into every path and pushes all of them through
the backbone, so a question with `K` candidates and a `P`-token prefix performs
`K * P` prefix token evaluations whose inputs are byte-identical. This module
encodes the prefix once, keeps its self-attention keys and values, and evaluates
only the per-candidate suffix against them.

Equivalence argument
--------------------
For candidate `c` with suffix `s_c`, the reference path is the token sequence
`prefix + s_c` under a causal mask, read at its final position. The shared path
builds the same sequence in two stages: stage one caches the prefix keys and
values, stage two appends the suffix keys and values and attends from the suffix
positions to the cached prefix plus the causal suffix. Both stages present the
same keys, values and query positions, so the attention inputs are identical and
only the reduction order inside the softmax differs. `scripts/test_shared_prefix.py`
pins that equivalence: on CPU float32 the observed deviation is a few 1e-6, and
with the same attention implementation in bfloat16 it is exactly zero.

The saving is a function of the token layout, not of the accelerator:

    reference prefix tokens = K * P      (every path repeats the prefix)
    shared prefix tokens    = P          (computed once)
    suffix tokens           = K * S_max  (unchanged in both)
    reference leaf tokens   = K * W      (padded to one width)
    shared leaf tokens      = P + K * S_max

Prefixes are never shared across questions: two questions can tokenize to equal
prefixes, but that reuse has a different equivalence story and is left out.

This module performs no inference, no training and no network access by itself;
the caller supplies both forward passes.
"""
import copy
import hashlib

# Blocked attention positions are filled with the dtype minimum rather than a
# small negative constant, so their softmax weight is exactly zero in fp32,
# fp16 and bf16 alike.
BLOCKED_MASK_SENTINEL = "dtype_minimum"
SUPPORTED_CACHE_DTYPES = ("float32", "float64", "float16", "bfloat16")


def encode_prefix_parts(segments, encode, add_special_tokens=False):
    """Tokenize ordered text segments and concatenate them into one prefix.

    `encode` is any callable with the tokenizer interface `encode(text, ...)`.
    Returns `(prefix_tokens, segment_lengths)`.
    """
    if not isinstance(segments, (list, tuple)) or not segments:
        raise ValueError("Prefix segments must be a nonempty list or tuple")
    if not all(isinstance(segment, str) and segment for segment in segments):
        raise ValueError("Prefix segments must be nonempty strings")
    prefix, lengths = [], []
    for segment in segments:
        tokens = encode(segment, add_special_tokens=add_special_tokens)
        if not isinstance(tokens, (list, tuple)) or not all(type(t) is int and t >= 0 for t in tokens):
            raise ValueError("encode must return a list of nonnegative integer token ids")
        prefix.extend(tokens)
        lengths.append(len(tokens))
    if not prefix:
        raise ValueError("Prefix must contain at least one token")
    return prefix, lengths


def encode_candidate_suffixes(texts, encode, eos_token_id):
    """Tokenize candidate descriptions into per-candidate `suffix` token lists."""
    if not isinstance(texts, (list, tuple)) or not texts:
        raise ValueError("Candidate texts must be a nonempty list or tuple")
    if not all(isinstance(text, str) and text for text in texts):
        raise ValueError("Candidate texts must be nonempty strings")
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise ValueError("eos_token_id must be a nonnegative integer")
    suffixes = []
    for text in texts:
        tokens = encode(f"Candidate:\n{text}\nDecision:", add_special_tokens=False)
        if not isinstance(tokens, (list, tuple)) or not all(type(t) is int and t >= 0 for t in tokens):
            raise ValueError("encode must return a list of nonnegative integer token ids")
        suffix = list(tokens) + [eos_token_id]
        if len(suffix) < 2:
            raise ValueError("Every candidate suffix must contain a description token and EOS")
        suffixes.append(suffix)
    return suffixes


def expected_path_count(example):
    """Number of encoded paths a prepared question must supply.

    A Boolean question reports two answer slots but has one semantic path, matching
    `predict_toy_decisions.prepare_examples` and `train_toy_decisions.load_examples`.
    Every other question needs one path per candidate.
    """
    slots = len(example.get("candidate_ids") or ())
    if example.get("type") == "boolean" and slots > 1:
        return 1
    return slots


def shared_prefix_length(paths):
    """Length of the longest common leading run across nonempty token lists."""
    if not paths:
        raise ValueError("At least one path is required")
    for path in paths:
        if not path:
            raise ValueError("Paths must be nonempty")
    shortest = min(len(path) for path in paths)
    length = 0
    while length < shortest and all(path[length] == paths[0][length] for path in paths):
        length += 1
    return length


class PrefixGroup:
    """One question whose candidate paths share a leading token run."""

    __slots__ = ("example_index", "example_id", "prefix_tokens", "suffix_tokens",
                 "candidate_ids", "question_type", "prefix_sha256", "sample_sha256",
                 "split_unverified")

    def __init__(self, example_index, example_id, prefix_tokens, suffix_tokens,
                 candidate_ids, question_type, sample_sha256, split_unverified=False):
        if not suffix_tokens:
            raise ValueError("A prefix group needs at least one candidate suffix")
        if all(len(suffix) == 0 for suffix in suffix_tokens) and len(suffix_tokens) != 1:
            raise ValueError("Only a single-path question may have an empty suffix")
        if not candidate_ids:
            raise ValueError("A prefix group needs at least one candidate id")
        if len(suffix_tokens) > len(candidate_ids):
            raise ValueError("A question cannot have more encoded paths than candidate slots")
        self.example_index = example_index
        self.example_id = example_id
        self.prefix_tokens = prefix_tokens
        self.suffix_tokens = suffix_tokens
        self.candidate_ids = candidate_ids
        self.question_type = question_type
        self.prefix_sha256 = hashlib.sha256(
            ",".join(map(str, prefix_tokens)).encode("ascii")).hexdigest()
        self.sample_sha256 = sample_sha256
        self.split_unverified = split_unverified

    @property
    def prefix_length(self):
        return len(self.prefix_tokens)

    @property
    def suffix_width(self):
        return max(len(suffix) for suffix in self.suffix_tokens)

    @property
    def candidate_count(self):
        return len(self.candidate_ids)

    @property
    def path_count(self):
        return len(self.suffix_tokens)

    @property
    def path_width(self):
        return self.prefix_length + self.suffix_width

    def reference_path_tokens(self):
        """Path tokens the reference implementation evaluates for one copy of this question.

        A reused question still appears in the input and the reference pays for every copy;
        `SharedPrefixPlan.reference_leaf_tokens` multiplies this by the occurrence count.
        """
        return self.path_count * self.path_width

    def shared_path_tokens(self, suffix_chunk=None):
        """Path tokens the shared implementation actually evaluates for this question.

        The prefix is evaluated once. Suffixes are evaluated in chunks of `suffix_chunk`
        rows (all candidates when omitted), and `_stage_two_inputs` pads every row in a
        chunk to that chunk's widest suffix, so the real cost depends on the chunking
        rather than on the sum of suffix lengths. Counting the unpadded sum understates
        the shared cost whenever suffix lengths are uneven.
        """
        widths = [len(suffix) for suffix in self.suffix_tokens]
        if suffix_chunk is None:
            return self.prefix_length + len(widths) * max(widths)
        if type(suffix_chunk) is not int or suffix_chunk < 1:
            raise ValueError("suffix_chunk must be None or a positive integer")
        total = self.prefix_length
        for start in range(0, len(widths), suffix_chunk):
            chunk = widths[start:start + suffix_chunk]
            total += len(chunk) * max(chunk)
        return total


class SharedPrefixPlan:
    """Deterministic shared-prefix layout for a batch of prepared questions.

    A question whose candidates share no leading token is reported through
    `fully_shared=False` and `prefix_mismatch_examples`; the planner never
    silently substitutes a different layout for it. Callers decide whether to
    fall back to the reference encoding for those questions.
    """

    def __init__(self, examples):
        if not isinstance(examples, (list, tuple)) or not examples:
            raise ValueError("SharedPrefixPlan requires a nonempty example list")
        self.examples = list(examples)
        self.groups = []
        self.ungrouped = []
        self.prefix_mismatch_examples = []
        self.reused_samples = []
        self.covered_by = {}
        seen_samples = {}
        for index, example in enumerate(self.examples):
            self._add_example(index, example, seen_samples)

    def _add_example(self, index, example, seen_samples):
        paths = example.get("leaf_tokens")
        candidate_ids = example.get("candidate_ids")
        example_id = example.get("id")
        if not isinstance(paths, (list, tuple)) or not paths:
            raise ValueError(f"{example_id}: leaf_tokens must be a nonempty list")
        if not isinstance(candidate_ids, (list, tuple)) or not candidate_ids:
            raise ValueError(f"{example_id}: candidate_ids must be a nonempty list")
        if len(paths) != expected_path_count(example):
            raise ValueError(
                f"{example_id}: expected {expected_path_count(example)} encoded paths for "
                f"{len(candidate_ids)} candidate slots, found {len(paths)}")
        for path in paths:
            if not isinstance(path, (list, tuple)) or not path:
                raise ValueError(f"{example_id}: every candidate path must be nonempty")
            if not all(type(token) is int and token >= 0 for token in path):
                raise ValueError(f"{example_id}: token ids must be nonnegative integers")
        observed = shared_prefix_length(paths)
        declared = example.get("prefix_length")
        if declared is not None and (type(declared) is not int or declared < 1):
            raise ValueError(f"{example_id}: prefix_length must be a positive integer")
        if declared is not None and declared > min(len(path) for path in paths):
            raise ValueError(f"{example_id}: prefix_length exceeds the shortest candidate path")
        # The prefix never absorbs candidate text. An authored `prefix_length` (the
        # state-plus-question text) bounds the split, so a candidate whose description
        # happens to repeat its siblings still keeps its own suffix tokens. Without a
        # declaration the whole observed common run is shared, which is the maximum
        # saving, but the split point is then unverified and marked as such.
        if declared is None:
            if observed == 0:
                self._leave_ungrouped(index, example_id, "candidate paths share no leading token")
                return
            split, unverified = observed, True
        else:
            if observed < declared:
                self._leave_ungrouped(index, example_id, "candidates diverge inside the "
                                                         "declared prefix")
                return
            split, unverified = declared, False
        prefix = list(paths[0][:split])
        suffixes = [list(path[split:]) for path in paths]
        if any(not suffix for suffix in suffixes):
            self._leave_ungrouped(
                index, example_id,
                "a candidate path is entirely shared prefix, so it has no final position to read")
            return
        self._append_group(index, example, prefix, suffixes, seen_samples,
                           split_unverified=unverified)

    def _append_group(self, index, example, prefix, suffixes, seen_samples,
                      split_unverified=False):
        example_id = example.get("id")
        candidate_ids = list(example.get("candidate_ids"))
        sample_sha256 = hashlib.sha256(
            ",".join(map(str, prefix)).encode("ascii") + b"|" +
            ";".join(",".join(map(str, suffix)) for suffix in suffixes).encode("ascii")
        ).hexdigest()
        if sample_sha256 in seen_samples:
            # Byte-identical state, question and candidate set: the cached prefix and the
            # candidate suffixes are the same tensors, so this is an exact reuse rather
            # than a new encode. Repeated action sets across states are the common case.
            self.covered_by[index] = seen_samples[sample_sha256]
            self.reused_samples.append(example_id)
            return
        seen_samples[sample_sha256] = len(self.groups)
        self.groups.append(PrefixGroup(
            example_index=index,
            example_id=example_id,
            prefix_tokens=prefix,
            suffix_tokens=suffixes,
            candidate_ids=list(candidate_ids),
            question_type=example.get("type"),
            sample_sha256=sample_sha256,
            split_unverified=split_unverified))

    def _leave_ungrouped(self, index, example_id, reason):
        self.prefix_mismatch_examples.append(example_id)
        self.ungrouped.append({"example_index": index, "example_id": example_id, "reason": reason})

    def fully_shared(self):
        """True when every supplied question is covered by a prefix group or a reuse."""
        return not self.ungrouped

    def covered_examples(self):
        return len(self.groups) + len(self.covered_by)

    def reference_width(self):
        """Padded path width the reference implementation uses for this flat batch.

        The reference flattens every candidate path of the batch into one tensor and
        pads to the single widest path in that tensor, so one long candidate widens
        every row. `shared_leaf_tokens` does not inherit that coupling. Returns
        `None` for a plan with no shareable group.
        """
        if not self.groups:
            return None
        return max(group.path_width for group in self.groups)

    def occurrences(self):
        """How many input examples each group serves, including exact reuses."""
        counts = {index: 1 for index in range(len(self.groups))}
        for source in self.covered_by.values():
            counts[source] = counts.get(source, 1) + 1
        return counts

    def reference_leaf_tokens(self):
        """Token evaluations in the reference implementation for this batch.

        Counts every supplied example, not just the deduplicated groups: the reference
        encoder evaluates each copy, so omitting reuses understates its cost and inflates
        the reported reduction.
        """
        width = self.reference_width()
        if width is None:
            return 0
        counts = self.occurrences()
        return width * sum(self.groups[index].path_count * count
                           for index, count in counts.items())

    def accounting(self, suffix_chunk=None):
        reference = self.reference_leaf_tokens()
        shared = sum(group.shared_path_tokens(suffix_chunk) for group in self.groups)
        reference_prefix = sum(group.path_count * group.prefix_length for group in self.groups)
        shared_prefix = sum(group.prefix_length for group in self.groups)
        return {
            "questions": len(self.groups),
            "candidates": sum(group.candidate_count for group in self.groups),
            "encoded_paths": sum(group.path_count for group in self.groups),
            "reference_leaf_tokens": reference,
            "reference_padded_width": self.reference_width(),
            "shared_leaf_tokens": shared,
            "shared_leaf_tokens_unpadded":
                sum(group.prefix_length + sum(len(s) for s in group.suffix_tokens)
                    for group in self.groups),
            "suffix_chunk_assumed": suffix_chunk,
            "reference_prefix_tokens": reference_prefix,
            "shared_prefix_tokens": shared_prefix,
            "token_reduction": round(reference / shared, 6) if shared else None,
            "prefix_fraction_before": round(reference_prefix / reference, 6) if reference else None,
            "prefix_fraction_after": round(shared_prefix / shared, 6) if shared else None,
            "fully_shared": self.fully_shared(),
            "reused_questions": len(self.reused_samples),
            "reused_examples": list(self.reused_samples),
            "covered_examples": self.covered_examples(),
            "ungrouped_questions": len(self.ungrouped),
            "ungrouped_examples": list(self.ungrouped),
            "prefix_mismatch_examples": list(self.prefix_mismatch_examples),
            "blocked_mask_sentinel": BLOCKED_MASK_SENTINEL,
            "unverified_split_questions": [group.example_id for group in self.groups
                                           if group.split_unverified],
        }


def build_suffix_attention_mask(suffix_lengths, prefix_length, dtype, device=None):
    """Boolean or additive mask of shape `(rows, 1, width, prefix_length + width)`.

    Position `(row, q)` may attend to all `prefix_length` cached prefix positions
    and to suffix positions `0..q` of its own row. It may never attend to another
    row's suffix, to its own future suffix positions, or to its own padding.
    Pass `dtype=None` to get the boolean allow-mask.

    Note for future readers: because every row's suffix starts at the same column,
    a global causal mask over the whole concatenated row is *numerically equivalent*
    to this mask for every real query position (query `P+q` is allowed prefix
    `0..P-1` plus suffix `0..q` under both). The two differ only for padding queries,
    whose outputs are discarded. That equivalence was verified by mutation testing,
    so do not "fix" one into the other expecting a behavioural change; the
    distinguishing defects are prefix invisibility, cross-row leakage and shifted
    position ids, which the tests do catch.
    """
    import torch
    if not suffix_lengths:
        raise ValueError("suffix_lengths must be nonempty")
    if type(prefix_length) is not int or prefix_length < 0:
        raise ValueError("prefix_length must be a nonnegative integer")
    lengths = torch.as_tensor(list(suffix_lengths), dtype=torch.long)
    if lengths.dim() != 1 or int(lengths.min()) < 0:
        raise ValueError("Suffix lengths must be nonnegative")
    if int(lengths.min()) < 1:
        raise ValueError("Every suffix must contain at least one token")
    rows, width = len(suffix_lengths), int(lengths.max())
    allowed = torch.zeros((rows, 1, width, prefix_length + width), dtype=torch.bool, device=device)
    if prefix_length:
        allowed[:, :, :, :prefix_length] = True
    causal = torch.tril(torch.ones((width, width), dtype=torch.bool, device=device))
    allowed[:, :, :, prefix_length:] = causal[None, None]
    if dtype is None:
        return allowed
    try:
        blocked = torch.finfo(dtype).min
    except TypeError as error:
        raise ValueError(f"No floating-point minimum for dtype {dtype}") from error
    return torch.where(allowed, torch.zeros((), dtype=dtype, device=device),
                       torch.full((), blocked, dtype=dtype, device=device))


def assert_supported_attention(model):
    """Fail loudly unless the attention backend honours a supplied 4-D mask.

    Eager and SDPA use the mask verbatim. Flash-Attention 2 rejects a float 4-D mask,
    and a static or offloading cache writes into preallocated storage, which would
    corrupt the broadcast prefix. Both are refused rather than silently mis-computed.
    """
    config = getattr(model, "config", None)
    implementation = getattr(config, "_attn_implementation", None)
    if implementation not in (None, "eager", "sdpa"):
        raise ValueError(f"Shared-prefix encoding needs eager or sdpa attention, got {implementation!r}")
    return implementation or "default"


def assert_supported_cache(cache):
    """Reject caches that write in place instead of rebinding their layer tensors."""
    name = type(cache).__name__
    if name in {"StaticCache", "StaticSlidingWindowCache", "SlidingWindowCache", "HybridCache"}:
        raise ValueError(f"Shared-prefix encoding needs a rebuilding cache, not {name}")
    layers = getattr(cache, "layers", None)
    if not layers:
        raise ValueError("Stage-one cache exposes no layers")
    return name


def last_hidden_state(output):
    """Read `last_hidden_state` from a model output or a bare hidden-state tensor."""
    import torch
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, torch.Tensor):
        hidden = output
    if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
        raise ValueError("Forward pass must return last_hidden_state of shape (rows, width, hidden)")
    return hidden


class CacheAdapter:
    """Fail-loud adapter over a transformers `Cache`-like object.

    `prefixed(rows)` returns a shallow clone whose layer keys and values carry a
    zero-stride batch dimension of size `rows`, so the broadcast itself does not copy
    the prefix.

    What this does *not* do is keep the prefix to one copy for the whole suffix pass.
    `DynamicLayer.update` appends the new keys and values with `torch.cat(..., dim=-2)`,
    which materializes `rows` copies of the prefix plus their suffixes. The saving is in
    *compute* — the prefix is never re-encoded — not in peak cache memory. See
    `docs/SHARED_PREFIX.md` for the measured token counts and the memory caveat.
    """

    @staticmethod
    def from_forward_output(output):
        cache = getattr(output, "past_key_values", None)
        if cache is None:
            raise ValueError("Stage-one forward must return past_key_values; pass use_cache=True")
        assert_supported_cache(cache)
        return CacheAdapter(cache)

    def __init__(self, cache):
        self._cache = cache
        self.prefix_length = None
        # Hold one reference per pristine prefix tensor. `prefixed` never hands these
        # tensors to the model: it builds freshly constructed layer objects that point at
        # zero-stride views of them. A cache that rebinds its layer keys/values (the
        # dynamic caches used here) therefore cannot disturb the shared prefix. The
        # storage is genuinely shared, so a cache that writes in place, an offloading
        # cache or a static preallocated cache would: those are rejected by
        # `assert_supported_cache`, not silently tolerated.
        self._prefix_tensors = []
        for index, keys, values in self.layer_tensors():
            self._prefix_tensors.append((index, keys, values))
        if self.prefix_length is None:
            raise ValueError("Cache exposes no layers")

    def layer_tensors(self):
        import torch
        for index, layer in enumerate(self._cache.layers):
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            for name, tensor in (("keys", keys), ("values", values)):
                if not isinstance(tensor, torch.Tensor) or tensor.dim() != 4:
                    raise ValueError(f"layer {index} {name} must be a rank-4 tensor")
            if tuple(keys.shape) != tuple(values.shape):
                raise ValueError(f"layer {index} key/value shapes differ: "
                                 f"{tuple(keys.shape)} vs {tuple(values.shape)}")
            if keys.shape[0] != 1:
                raise ValueError(
                    f"layer {index} prefix cache has batch {keys.shape[0]}; the shared prefix is "
                    "always generated with a single row so that it can broadcast to candidates")
            if self.prefix_length is None:
                self.prefix_length = keys.shape[-2]
            elif self.prefix_length != keys.shape[-2]:  # pragma: no cover - guarded in __init__

                raise ValueError("All cache layers must share one prefix length")
            yield index, keys, values

    def batch_bytes(self, rows):
        total = 0
        for _, keys, values in self._prefix_tensors:
            total += rows * (keys.numel() + values.numel()) * keys.element_size()
        return total

    def prefixed(self, rows):
        """Shallow clone whose prefix keys/values broadcast across `rows` rows."""
        if type(rows) is not int or rows < 1:
            raise ValueError("rows must be a positive integer")
        clone = copy.copy(self._cache)
        layers = []
        for index, keys, values in self._prefix_tensors:
            if keys.shape[0] != 1:  # pragma: no cover - guarded at construction
                raise RuntimeError("Pristine prefix tensor is not batch one")
            # `expand` to a single row is a no-op that keeps the original stride, so
            # only a multi-row broadcast is expected to produce a zero-stride view.
            expanded_keys = keys.expand(rows, -1, -1, -1)
            expanded_values = values.expand(rows, -1, -1, -1)
            if rows > 1 and (expanded_keys.stride(0) != 0 or expanded_values.stride(0) != 0):
                raise RuntimeError("Prefix broadcast copied data; expected a zero-stride view")
            # Fresh layer objects keep `_prefix_tensors` unreachable from the cache the
            # model mutates.
            layer = copy.copy(self._cache.layers[index])
            layer.keys = expanded_keys
            layer.values = expanded_values
            layers.append(layer)
        clone.layers = layers
        return clone

    def pristine_tensors(self):
        """The unreachable prefix tensors this adapter broadcasts from."""
        return tuple((index, keys, values) for index, keys, values in self._prefix_tensors)


class SharedPrefixEncoder:
    """Encode candidate leaves while evaluating each question prefix exactly once."""

    def __init__(self, model, pad_token_id, suffix_chunk=None, device=None):
        if type(pad_token_id) is not int or pad_token_id < 0:
            raise ValueError("pad_token_id must be a nonnegative integer")
        if suffix_chunk is not None and (type(suffix_chunk) is not int or suffix_chunk < 1):
            raise ValueError("suffix_chunk must be None or a positive integer")
        self.model = model
        assert_supported_attention(model)
        self.pad_token_id = pad_token_id
        self.suffix_chunk = suffix_chunk
        self.device = device
        self.stats = {"prefix_forwards": 0, "suffix_forwards": 0, "prefix_tokens": 0,
                      "suffix_tokens": 0, "prefix_cache_bytes": 0, "max_suffix_batch": 0,
                      "questions": 0, "reused_questions": 0, "fallback_questions": 0}
        # A request can fall back to the reference encoder; callers report what ran,
        # not what they asked for, so this flag is the source of truth.
        self.used_shared_prefix = False

    @property
    def runner(self):
        """The callable that performs the backbone forward passes."""
        return self.model

    # -- internals --------------------------------------------------------
    def _resolve_dtype(self):
        parameters = list(self.model.parameters())
        if not parameters:
            raise ValueError("model exposes no parameters, so the cache dtype is unknown")
        dtype = parameters[0].dtype
        if str(dtype).replace("torch.", "") not in SUPPORTED_CACHE_DTYPES:
            raise ValueError(f"Unsupported model dtype for masking: {dtype}")
        return dtype

    def _stage_one(self, prefix_tokens):
        import torch
        tokens = torch.tensor([list(prefix_tokens)], dtype=torch.long, device=self.device)
        length = tokens.shape[1]
        output = self.runner(
            input_ids=tokens,
            attention_mask=torch.ones((1, length), dtype=torch.long, device=self.device),
            position_ids=torch.arange(length, device=self.device)[None, :],
            use_cache=True)
        self.stats["prefix_forwards"] += 1
        self.stats["prefix_tokens"] += length
        return CacheAdapter.from_forward_output(output)

    def _stage_two_inputs(self, group, rows):
        import torch
        suffix_lengths = [len(group.suffix_tokens[row]) for row in rows]
        width = max(suffix_lengths)
        tokens = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long,
                            device=self.device)
        for offset, row in enumerate(rows):
            suffix = torch.tensor(group.suffix_tokens[row], dtype=torch.long, device=self.device)
            tokens[offset, :suffix.numel()] = suffix
        position_ids = (group.prefix_length + torch.arange(width, device=self.device))[None, :]
        position_ids = position_ids.expand(len(rows), width).contiguous()
        cache_position = torch.arange(group.prefix_length, group.prefix_length + width,
                                      device=self.device)
        return tokens, suffix_lengths, position_ids, cache_position

    def _encode_group(self, group, dtype):
        import torch
        if min(len(suffix) for suffix in group.suffix_tokens) < 1:
            raise RuntimeError(
                f"{group.example_id}: a group cannot contain an empty suffix; the planner "
                "excludes those questions before encoding")
        cache = self._stage_one(group.prefix_tokens)
        if cache.prefix_length != group.prefix_length:
            raise RuntimeError(
                f"{group.example_id}: cached prefix length {cache.prefix_length} does not match the "
                f"planned prefix length {group.prefix_length}")
        self.stats["prefix_cache_bytes"] += cache.batch_bytes(1)
        bound = self.suffix_chunk or group.path_count
        collected = []
        for start in range(0, group.path_count, bound):
            chunk = list(range(start, min(start + bound, group.path_count)))
            tokens, lengths, position_ids, cache_position = self._stage_two_inputs(group, chunk)
            mask = build_suffix_attention_mask(lengths, group.prefix_length, dtype,
                                               device=self.device)
            output = self.runner(
                input_ids=tokens, attention_mask=mask, position_ids=position_ids,
                cache_position=cache_position, past_key_values=cache.prefixed(len(chunk)),
                use_cache=False)
            hidden = last_hidden_state(output)
            rows = torch.arange(len(chunk), device=hidden.device)
            final = torch.tensor(lengths, device=hidden.device) - 1
            collected.append(hidden[rows, final])
            self.stats["suffix_forwards"] += 1
            self.stats["suffix_tokens"] += len(chunk) * tokens.shape[1]
            self.stats["suffix_padded_tokens"] = self.stats.get("suffix_padded_tokens", 0) + \
                len(chunk) * tokens.shape[1] - sum(lengths)
            self.stats["max_suffix_batch"] = max(self.stats["max_suffix_batch"], len(chunk))
        stacked = torch.cat(collected, dim=0)
        if stacked.shape[0] != group.path_count:
            raise RuntimeError(f"{group.example_id}: encoded {stacked.shape[0]} of "
                               f"{group.path_count} candidate paths")
        return stacked

    # -- public API -------------------------------------------------------
    def encode_examples(self, model, examples, pad_token_id):
        """Encoder callable shaped for `DecisionModel.forward(prefix_sharing=...)`.

        Returns a flat list of leaf states ordered exactly like
        `[leaf for example in examples for leaf in example['leaf_tokens']]`, or
        `None` when this batch has no fully shareable prefix and the caller must
        use the reference encoding.
        """
        if pad_token_id != self.pad_token_id:
            raise ValueError("Encoder pad_token_id does not match the model pad token")
        plan = SharedPrefixPlan(examples)
        if not plan.groups or not plan.fully_shared():
            # Partial coverage would need a mixed-batch layout; the batch falls back as
            # a whole so that no question is encoded by a second code path.
            self.stats["fallback_questions"] += len(examples) - plan.covered_examples()
            return None
        self.model = model
        leaves, _ = self.encode(plan)
        self.stats["questions"] += len(plan.groups)
        self.stats["reused_questions"] = len(plan.covered_by)
        self.used_shared_prefix = True
        return [row for leaf in leaves for row in leaf]

    def encode(self, plan):
        """Return `(leaves, accounting)`; `leaves[i]` aligns with `plan.examples[i]`."""
        if not isinstance(plan, SharedPrefixPlan):
            raise ValueError("encode expects a SharedPrefixPlan")
        if self.device is None:
            self.device = next(self.model.parameters()).device
        dtype = self._resolve_dtype()
        leaves = [None] * len(plan.examples)
        # `plan.covered_by` maps an example index to a *group* index, because groups are
        # created in encounter order while covered examples are not. Resolve through the
        # group's own example index; using the group index directly as an example index
        # silently reads the wrong leaf as soon as a batch has more than one group.
        group_example = {index: group.example_index for index, group in enumerate(plan.groups)}
        for group in plan.groups:
            leaves[group.example_index] = self._encode_group(group, dtype)
        for index, source in plan.covered_by.items():
            if source not in group_example:
                raise RuntimeError(
                    f"{plan.examples[index].get('id')}: reuse points at group {source}, "
                    f"but only {len(plan.groups)} groups exist")
            leaves[index] = leaves[group_example[source]]
            if leaves[index] is None:  # pragma: no cover - group order guarantees this
                raise RuntimeError(f"{plan.examples[index].get('id')}: reused leaf is not encoded")
        missing = [plan.examples[i].get("id") for i, leaf in enumerate(leaves) if leaf is None]
        if missing:
            raise RuntimeError(f"No leaves produced for: {missing}")
        accounting = plan.accounting()
        accounting["observed"] = dict(self.stats)
        accounting["suffix_chunk"] = self.suffix_chunk
        accounting["prefix_cache_bytes_at_batch_one"] = self.stats["prefix_cache_bytes"]
        accounting["device"] = str(self.device)
        return leaves, accounting
