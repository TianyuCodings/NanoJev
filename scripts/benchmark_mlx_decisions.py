#!/usr/bin/env python3
"""Warm MLX NanoJev decision-inference benchmark.

The benchmark reports end-to-end predictor latency after model load. It includes
request assembly, candidate tokenization, one or more MLX backbone forwards,
head evaluation, and probability materialization, but excludes process startup
and network transport.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from typing import Any


def make_payload(questions: int, candidates: int) -> dict[str, Any]:
    criteria = {
        f"candidate_{index:03d}": f"Choose candidate {index} in the current decision context."
        for index in range(candidates)
    }
    return {
        "states": [
            {
                "id": f"benchmark-{questions}-{candidates}",
                "state": "A benchmark state with several mutually exclusive candidate actions.",
                "questions": {
                    f"q_{index:03d}": {
                        "type": "choice",
                        "instructions": "Which candidate is preferred under the stated context?",
                        "criteria": criteria,
                    }
                    for index in range(questions)
                },
            }
        ]
    }


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--decision-head", required=True)
    parser.add_argument("--questions", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--candidates", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batch-questions", type=int, default=0)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("warmup must be non-negative and repeats must be positive")
    if any(value <= 0 for value in args.questions + args.candidates):
        raise SystemExit("questions and candidates must be positive")

    from mlx_decisions import MLXDecisionPredictor

    predictor = MLXDecisionPredictor(args.model_dir, args.decision_head)
    rows = []
    for question_count in args.questions:
        for candidate_count in args.candidates:
            payload = make_payload(question_count, candidate_count)
            for _ in range(args.warmup):
                predictor.predict(payload, batch_questions=args.batch_questions)
            timings = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                result = predictor.predict(payload, batch_questions=args.batch_questions)
                elapsed = time.perf_counter() - started
                if result["execution"]["questions"] != question_count:
                    raise AssertionError("benchmark question count mismatch")
                timings.append(elapsed)
            median = statistics.median(timings)
            rows.append({
                "questions": question_count,
                "candidates_per_question": candidate_count,
                "candidate_paths": question_count * candidate_count,
                "forward_passes": result["execution"]["forward_passes"],
                "median_seconds": median,
                "p50_seconds": percentile(timings, 0.50),
                "p95_seconds": percentile(timings, 0.95),
                "questions_per_second": question_count / median,
                "candidate_paths_per_second": (question_count * candidate_count) / median,
                "samples": timings,
            })
    print(json.dumps({
        "backend": "MLX",
        "model_dir": args.model_dir,
        "decision_head": args.decision_head,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "batch_questions": args.batch_questions or "all",
        "scope": "warm inference; includes tokenization, tensor assembly, MLX forward, decision head, and probability materialization; excludes process startup and HTTP transport",
        "results": rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
