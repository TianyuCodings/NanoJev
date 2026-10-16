#!/usr/bin/env python3
"""Measure what shared-prefix encoding actually saves, on a local checkpoint.

Two independent quantities are reported:

* **Structural token accounting** — computed from the encoder's real chunk layout, so it is
  hardware independent and includes the padding the encoder actually performs.
* **Wall clock and peak accelerator memory** — measured, therefore machine specific. The two
  do not track each other one-for-one: the suffix pass still attends over the cached prefix,
  so attention cost grows with prefix length even though the prefix is no longer re-encoded.

Read `docs/SHARED_PREFIX.md` before interpreting a result: the payoff depends on the candidate
count, the prefix length and the chunk size together, and small workloads regress.

Usage:

    python3 scripts/benchmark_shared_prefix.py --checkpoint-dir checkpoints/Qwen3-0.6B \\
        --output results/shared_prefix_benchmark.json --device cpu

The command loads a local checkpoint only, performs no optimization and makes no network
requests.
"""
import argparse
import contextlib
import json
import platform
import statistics
import time
from pathlib import Path

from shared_prefix_cli import distinct_payload, prepare_inference

DEFAULT_CANDIDATES = (4, 16, 64)


def timed(fn, repeats):
    fn()  # warm up
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), min(samples)


def autocast_context(torch, device, precision):
    """bf16 means autocast over float32 parameters, exactly as the service runs it."""
    if precision == "bf16":
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def peak_memory_bytes(torch, device):
    if device.type == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated(device))
    return None


def measure(torch, setup, device, candidates, states, repeats, max_length, precision,
            suffix_chunk):
    shared_prefix = setup["shared_prefix"]
    examples = setup["predictor"].prepare_examples(
        distinct_payload(candidates, states), setup["tokenizer"], max_length)
    plan = shared_prefix.SharedPrefixPlan(examples)
    accounting = plan.accounting(suffix_chunk)
    if not accounting["fully_shared"]:
        raise SystemExit("benchmark payload must have a shareable prefix for every question")
    if accounting["reused_questions"]:
        raise SystemExit("benchmark states must be distinct, otherwise the measurement "
                         "includes whole-question deduplication on top of prefix sharing")
    model, tokenizer = setup["model"], setup["tokenizer"]
    context = lambda: autocast_context(torch, device, precision)  # noqa: E731

    def reference():
        with torch.inference_mode(), context():
            return model(examples, tokenizer.pad_token_id)[0]

    def shared():
        encoder = shared_prefix.SharedPrefixEncoder(model.backbone,
                                                    pad_token_id=tokenizer.pad_token_id,
                                                    suffix_chunk=suffix_chunk, device=device)
        with torch.inference_mode(), context():
            return model(examples, tokenizer.pad_token_id, prefix_sharing=encoder)[0]

    reference_median, reference_best = timed(reference, repeats)
    shared_median, shared_best = timed(shared, repeats)
    with torch.inference_mode(), context():
        drift = (reference() - shared()).abs().max().item()
    return {
        "candidates": candidates,
        "states": states,
        "questions": len(examples),
        "prefix_length": plan.groups[0].prefix_length,
        "suffix_width": plan.groups[0].suffix_width,
        "reference_leaf_tokens": accounting["reference_leaf_tokens"],
        "shared_leaf_tokens": accounting["shared_leaf_tokens"],
        "shared_leaf_tokens_unpadded": accounting["shared_leaf_tokens_unpadded"],
        "token_reduction": accounting["token_reduction"],
        "reference_seconds_median": round(reference_median, 6),
        "shared_seconds_median": round(shared_median, 6),
        "reference_seconds_best": round(reference_best, 6),
        "shared_seconds_best": round(shared_best, 6),
        "wall_clock_speedup_median": round(reference_median / shared_median, 4),
        "max_abs_logit_drift": round(drift, 8),
        "repeats": repeats,
        "suffix_chunk": suffix_chunk,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--candidates", default=",".join(map(str, DEFAULT_CANDIDATES)))
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--suffix-chunk", type=int,
                        help="Cap suffix rows per forward pass; default uses all candidates")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    try:
        import torch
    except ImportError as error:
        raise SystemExit(f"torch required: {error}")

    setup = prepare_inference(args, torch)
    device = torch.device(args.device)
    setup["model"] = setup["model"].to(device=device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f"strict load: {setup['description']}")
    print(f"parameters pinned to float32; precision mode {args.precision}")

    rows = []
    for candidates in [int(value) for value in args.candidates.split(",")]:
        states = 1 if candidates >= 128 else args.states
        row = measure(torch, setup, device, candidates, states, args.repeats,
                      args.max_length, args.precision, args.suffix_chunk)
        row["peak_memory_bytes"] = peak_memory_bytes(torch, device)
        rows.append(row)
        print(json.dumps(row), flush=True)

    report = {
        "schema": "nanojev-shared-prefix-benchmark-v1",
        "device": str(device),
        "precision": args.precision,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "attention_implementation": setup["config"]._attn_implementation,
        "checkpoint": str(setup["weights"]),
        "checkpoint_sha256": setup["weights_sha256"],
        "checkpoint_is_released": setup["released"],
        "note": ("Structural token accounting is exact and hardware independent. Timings and "
                 "peak memory are this machine only. The payoff depends on candidate count, "
                 "prefix length and chunk size together; small candidate counts regress. See "
                 "docs/SHARED_PREFIX.md."),
        "rows": rows,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
