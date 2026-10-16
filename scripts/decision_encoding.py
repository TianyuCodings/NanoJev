#!/usr/bin/env python3
"""Canonical text encoding for decision questions, shared by trainer and inference.

`train_toy_decisions.load_examples` and `predict_toy_decisions.prepare_examples`
build the same candidate paths independently. Both write `state` + `question` into
every candidate path, and both must agree token for token, because a checkpoint
trained on one encoding cannot be served by the other. This module holds that
encoding in one place: `question_prefix_segments` produces the shared leading
text, `candidate_suffix_text` produces the per-candidate tail, and
`candidate_path_tokens` assembles the reference path.

The functions are pure string/token operations. They load no checkpoint, import
no PyTorch and make no network requests, so they are cheap to test exhaustively.
"""
BOOLEAN_IDS = ("false", "true")
BOOLEAN_TEXT = "The proposition is true."
CANDIDATE_TEMPLATE = "Candidate:\n{}\nDecision:"


def state_text(state):
    """Serialize a state exactly as both reference implementations do."""
    return state if isinstance(state, str) else str(state)


def question_candidate_texts(question):
    """Return `(candidate_ids, candidate_texts)` for one question.

    Choice ids are the supplied keys, ordered as the caller ordered them. Boolean
    has a single semantic path and reports it under `true`. Score ids are the
    level indices as strings, and levels are never given an injected ordinal.
    """
    if not isinstance(question, dict):
        raise ValueError("Question must be an object")
    kind = question.get("type")
    if kind == "boolean":
        return list(BOOLEAN_IDS), [BOOLEAN_TEXT]
    if kind == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("Choice questions require a nonempty criteria object")
        ids = list(criteria)
        return ids, [f"{key}: {criteria[key]}" for key in ids]
    if kind == "score":
        criteria = question.get("criteria")
        if not isinstance(criteria, (list, tuple)) or not criteria:
            raise ValueError("Score questions require a nonempty criteria list")
        return [str(index) for index in range(len(criteria))], list(criteria)
    raise ValueError(f"Unsupported question type: {kind!r}")


def question_prefix_segments(state, question):
    """Ordered text segments that form the shared prefix of every candidate path."""
    kind = question.get("type")
    if kind not in {"boolean", "choice", "score"}:
        raise ValueError(f"Unsupported question type: {kind!r}")
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions:
        raise ValueError("Question instructions must be a nonempty string")
    segments = [f"State:\n{state_text(state)}\n",
                f"Question type: {kind}\nQuestion:\n{instructions}\n"]
    criteria = question.get("criteria")
    if kind in {"choice", "score"}:
        expected = dict if kind == "choice" else (list, tuple)
        if criteria is not None and (not isinstance(criteria, expected) or not criteria):
            raise ValueError(f"{kind} criteria must be a nonempty "
                             f"{'object' if kind == 'choice' else 'list'}")
    if kind == "boolean":
        if criteria is not None and (not isinstance(criteria, dict)
                                     or set(criteria) - {"false", "true"}):
            raise ValueError("Boolean criteria may only contain false/true keys")
        criteria = criteria or {}
        for key, label in (("false", "False"), ("true", "True")):
            if key in criteria:
                value = criteria[key]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("This prototype requires textual Boolean criteria")
                segments[1] += f"{label} criterion: {value}\n"
    return segments


def candidate_suffix_text(text):
    return CANDIDATE_TEMPLATE.format(text)


def build_candidate_paths(encode, state, question, eos_token_id):
    """Return `(candidate_ids, candidate_texts, prefix_tokens, leaf_tokens)`.

    This is the single encoder used by both the trainer and the serving entry point,
    so a checkpoint can never be trained on one token layout and served with another.
    """
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise ValueError("eos_token_id must be a nonnegative integer")
    prefix_tokens = []
    for segment in question_prefix_segments(state, question):
        tokens = encode(segment, add_special_tokens=False)
        if not isinstance(tokens, (list, tuple)):
            raise ValueError("encode must return a token list")
        prefix_tokens.extend(tokens)
    if not prefix_tokens:
        raise ValueError("Encoded prefix is empty")
    candidate_ids, candidate_texts = question_candidate_texts(question)
    leaf_tokens = []
    for text in candidate_texts:
        suffix = encode(candidate_suffix_text(text), add_special_tokens=False)
        leaf_tokens.append(list(prefix_tokens) + list(suffix) + [eos_token_id])
    return candidate_ids, candidate_texts, prefix_tokens, leaf_tokens


def candidate_path_tokens(encode, state, question, eos_token_id):
    """Alias of `build_candidate_paths`, kept for callers that use the older name."""
    return build_candidate_paths(encode, state, question, eos_token_id)
