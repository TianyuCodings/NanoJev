#!/usr/bin/env python3
"""Contract and regression tests for NanoJev's MLX decision backend.

This test deliberately exercises all three public question types. With a
PyTorch/MPS reference JSON, it also checks that the MLX float16-backbone path
preserves the decision distribution within a configurable tolerance.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any



def contract_payload() -> dict[str, Any]:
    return {
        "states": [
            {
                "id": "mlx-contract",
                "state": "A navigation agent is at a junction. North is open, east is blocked, and the score is 7.",
                "questions": {
                    "is_safe": {
                        "type": "boolean",
                        "instructions": "Is the north route safe under the stated facts?",
                        "criteria": {
                            "false": "The north route is not safe.",
                            "true": "The north route is safe.",
                        },
                    },
                    "direction": {
                        "type": "choice",
                        "instructions": "Which direction should the agent choose?",
                        "criteria": {
                            "north": "Move north through the open route.",
                            "east": "Move east through the blocked route.",
                            "stay": "Remain at the current junction.",
                        },
                    },
                    "risk": {
                        "type": "score",
                        "instructions": "What is the risk level?",
                        "criteria": ["very low", "low", "medium", "high"],
                    },
                },
            },
            {
                "id": "mlx-contract-2",
                "state": "A fair coin is tossed once. The result has not been observed.",
                "questions": {
                    "coin": {
                        "type": "choice",
                        "instructions": "Which result occurred?",
                        "criteria": {
                            "heads": "The coin shows heads.",
                            "tails": "The coin shows tails.",
                        },
                    },
                },
            },
        ]
    }


def reverse_choice_order(payload: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(payload))
    criteria = result["states"][0]["questions"]["direction"]["criteria"]
    result["states"][0]["questions"]["direction"]["criteria"] = {
        key: criteria[key] for key in reversed(list(criteria))
    }
    return result


def answer_map(result: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (state["id"], qid): answer
        for state in result["states"]
        for qid, answer in state["answers"].items()
    }


def check_contract(result: dict[str, Any]) -> None:
    answers = answer_map(result)
    expected = {
        ("mlx-contract", "is_safe"),
        ("mlx-contract", "direction"),
        ("mlx-contract", "risk"),
        ("mlx-contract-2", "coin"),
    }
    if set(answers) != expected:
        raise AssertionError(f"unexpected answer keys: {set(answers)}")
    for key, answer in answers.items():
        probs = answer["probabilities"]
        if not probs or abs(math.fsum(probs.values()) - 1.0) > 1e-5:
            raise AssertionError(f"{key}: probabilities do not sum to one: {probs}")
        if not all(math.isfinite(float(value)) and 0 <= value <= 1 for value in probs.values()):
            raise AssertionError(f"{key}: invalid probabilities: {probs}")
    boolean = answers[("mlx-contract", "is_safe")]
    if boolean["type"] != "boolean" or not math.isclose(boolean["p_true"], boolean["probabilities"]["true"], abs_tol=1e-7):
        raise AssertionError(f"boolean output contract failed: {boolean}")
    choice = answers[("mlx-contract", "direction")]
    if choice["type"] != "choice" or choice["choice"] not in choice["probabilities"]:
        raise AssertionError(f"choice output contract failed: {choice}")
    score = answers[("mlx-contract", "risk")]
    if score["type"] != "score" or not 0 <= score["level"] < 4 or not 0 <= score["score"] <= 3:
        raise AssertionError(f"score output contract failed: {score}")


def compare_results(left: dict[str, Any], right: dict[str, Any], tolerance: float, label: str) -> float:
    a, b = answer_map(left), answer_map(right)
    if set(a) != set(b):
        raise AssertionError(f"{label}: answer keys differ")
    max_error = 0.0
    for key in a:
        pa, pb = a[key]["probabilities"], b[key]["probabilities"]
        if list(pa) != list(pb):
            raise AssertionError(f"{label} {key}: candidate order differs")
        error = max(abs(float(pa[c]) - float(pb[c])) for c in pa)
        max_error = max(max_error, error)
        if error > tolerance:
            raise AssertionError(f"{label} {key}: max probability error {error:.6g} > {tolerance:.6g}")
        if a[key].get("type") == "choice" and a[key]["choice"] != b[key]["choice"]:
            raise AssertionError(f"{label} {key}: argmax differs")
    return max_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--decision-head", required=True)
    parser.add_argument("--reference", type=Path, help="optional PyTorch/MPS result JSON")
    parser.add_argument("--max-error", type=float, default=0.03)
    args = parser.parse_args()
    if args.max_error < 0 or not math.isfinite(args.max_error):
        raise SystemExit("--max-error must be a finite non-negative number")

    from mlx_decisions import MLXDecisionPredictor
    predictor = MLXDecisionPredictor(args.model_dir, args.decision_head)
    payload = contract_payload()
    started = time.perf_counter()
    result = predictor.predict(payload)
    elapsed = time.perf_counter() - started
    check_contract(result)

    # A second call with one question per forward validates batching equivalence.
    split = predictor.predict(payload, batch_questions=1)
    batch_error = compare_results(result, split, args.max_error, "batching")

    # Choice probabilities are keyed by candidate ID, so reordering the offered
    # set must not change the probability attached to a semantic candidate.
    reversed_result = predictor.predict(reverse_choice_order(payload))
    original = answer_map(result)[("mlx-contract", "direction")]["probabilities"]
    reordered = answer_map(reversed_result)[("mlx-contract", "direction")]["probabilities"]
    order_error = max(abs(float(original[key]) - float(reordered[key])) for key in original)
    if order_error > args.max_error:
        raise AssertionError(f"candidate reorder error {order_error:.6g} > {args.max_error:.6g}")

    reference_error = None
    if args.reference:
        reference_error = compare_results(result, json.loads(args.reference.read_text(encoding="utf-8")), args.max_error, "PyTorch reference")

    report = {
        "status": "passed",
        "question_types": ["boolean", "choice", "score"],
        "questions": result["execution"]["questions"],
        "candidate_paths": result["execution"]["candidate_paths"],
        "batching_max_probability_error": batch_error,
        "candidate_reorder_max_probability_error": order_error,
        "pytorch_reference_max_probability_error": reference_error,
        "warm_inference_seconds": elapsed,
        "execution": result["execution"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
