#!/usr/bin/env python3
"""One command that re-derives every shared-prefix claim on this machine.

The tests assert these facts; this script reproduces them and prints the numbers, so a
reviewer does not have to read test code to decide whether to believe them. It performs no
optimization, no training and no network access.

    python3 scripts/verify_shared_prefix.py
    python3 scripts/verify_shared_prefix.py --checkpoint-dir checkpoints/NanoJev-unified

Checks, in increasing order of strength:

1. **Ranking** — shared and reference paths agree on the argmax for every question.
2. **Magnitude** — the maximum absolute logit and probability deviation.
3. **Oracle** — the refactored `forward` is bit-identical to a verbatim copy of the released
   `618cea6` body, so the default serving path did not move.
4. **Accounting** — the structural token counts, which depend only on the token layout.

Set `NANOJEV_VERIFY_MUTATE` to inject a known defect and confirm the checks fail:
`leak`, `no_prefix`, `shifted_positions`.
"""
import argparse
import os
import sys

from shared_prefix_cli import (
    RELEASED_REVISION, RELEASED_WEIGHTS_SHA256, distinct_payload, prepare_inference,
    released_forward,
)


def inject(mutation, shared_prefix):
    """Install one known defect so the checks can be shown to fail on it."""
    import torch
    original_mask = shared_prefix.build_suffix_attention_mask
    original_stage = shared_prefix.SharedPrefixEncoder._stage_two_inputs

    def finalize(allowed, dtype, device):
        if dtype is None:
            return allowed
        return torch.where(allowed, torch.zeros((), dtype=dtype, device=device),
                           torch.full((), float(torch.finfo(dtype).min), dtype=dtype,
                                      device=device))

    if mutation == "leak":
        def patched(lengths, prefix_length, dtype, device=None):
            allowed = original_mask(lengths, prefix_length, None, device)
            allowed[:, :, :, prefix_length:] = True
            return finalize(allowed, dtype, device)
    elif mutation == "no_prefix":
        def patched(lengths, prefix_length, dtype, device=None):
            allowed = original_mask(lengths, prefix_length, None, device)
            if prefix_length:
                allowed[:, :, :, :prefix_length] = False
            return finalize(allowed, dtype, device)
    elif mutation == "shifted_positions":
        def patched(self, group, rows):
            tokens, lengths, _, cache_position = original_stage(self, group, rows)
            width = tokens.shape[1]
            positions = torch.arange(width, device=self.device)[None, :]
            return (tokens, lengths,
                    positions.expand(len(rows), width).contiguous(), cache_position)
    else:
        raise SystemExit(f"unknown mutation: {mutation}")
    if mutation == "shifted_positions":
        shared_prefix.SharedPrefixEncoder._stage_two_inputs = patched
    else:
        shared_prefix.build_suffix_attention_mask = patched


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir",
                        help="Released checkpoint; defaults to a local Qwen3-0.6B")
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args(argv)

    try:
        import torch
    except ImportError as error:
        raise SystemExit(f"torch required: {error}")

    setup = prepare_inference(args, torch)
    shared_prefix, predictor = setup["shared_prefix"], setup["predictor"]
    model, tokenizer = setup["model"], setup["tokenizer"]
    if setup["released"]:
        print(f"checkpoint: released {RELEASED_REVISION} (sha256 verified "
              f"{RELEASED_WEIGHTS_SHA256[:12]}...)")
    print(f"strict load: {setup['description']}")

    if os.environ.get("NANOJEV_VERIFY_MUTATE"):
        mutation = os.environ["NANOJEV_VERIFY_MUTATE"]
        inject(mutation, shared_prefix)
        print(f"*** injected defect: {mutation} (checks are expected to FAIL) ***")

    examples = predictor.prepare_examples(
        distinct_payload(args.candidates, args.states), tokenizer, 8192)
    plan = shared_prefix.SharedPrefixPlan(examples)
    accounting = plan.accounting()
    if accounting["reused_questions"]:
        raise SystemExit("payload states are not distinct, so the measurement would include "
                         "exact-sample reuse on top of prefix sharing")
    encoder = shared_prefix.SharedPrefixEncoder(model.backbone,
                                                pad_token_id=tokenizer.pad_token_id,
                                                device=torch.device("cpu"))
    with torch.inference_mode():
        reference_logits, reference_valid = model(examples, tokenizer.pad_token_id)
        shared_logits, shared_valid = model(examples, tokenizer.pad_token_id,
                                            prefix_sharing=encoder)
        oracle_logits, oracle_valid = released_forward(model, examples,
                                                       tokenizer.pad_token_id, torch)

    drift = (reference_logits - shared_logits).abs().max().item()
    oracle_drift = (reference_logits - oracle_logits).abs().max().item()
    argmax_equal = torch.equal(reference_logits.argmax(-1), shared_logits.argmax(-1))
    valid_equal = torch.equal(reference_valid, shared_valid)
    oracle_equal = (torch.equal(reference_logits, oracle_logits)
                    and torch.equal(reference_valid, oracle_valid))
    worst_probability = 0.0
    for example, left, right in zip(examples, reference_logits, shared_logits):
        k = len(example["candidate_ids"])
        worst_probability = max(worst_probability,
                                (left[:k].float().softmax(-1)
                                 - right[:k].float().softmax(-1)).abs().max().item())

    print()
    print(f"questions                     {len(examples)}")
    print(f"candidates                    {accounting['candidates']} "
          f"({accounting['encoded_paths']} encoded paths)")
    print(f"leaf tokens reference -> shared  {accounting['reference_leaf_tokens']} -> "
          f"{accounting['shared_leaf_tokens']}  ({accounting['token_reduction']}x)")
    print(f"prefix forwards               {encoder.stats['prefix_forwards']} "
          f"(reused questions {encoder.stats['reused_questions']})")
    print(f"1. argmax identical           {argmax_equal}")
    print(f"2. max logit deviation        {drift:.3e}  (tolerance {args.tolerance:g})")
    print(f"   max probability deviation  {worst_probability:.3e}")
    print(f"   valid mask bitwise equal   {valid_equal}")
    print(f"3. oracle bitwise identical   {oracle_equal}  (drift {oracle_drift:.3e})")

    failures = []
    if not argmax_equal:
        failures.append("decisions changed")
    if drift > args.tolerance:
        failures.append(f"logit deviation {drift:.3e} exceeds {args.tolerance:g}")
    if not valid_equal:
        failures.append("valid mask changed")
    if not oracle_equal:
        failures.append("default path differs from the released implementation")
    if failures:
        print("\nFAIL: " + "; ".join(failures))
        return 1
    print("\nPASS: shared-prefix encoding reproduces the reference path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
