#!/usr/bin/env python3
"""MLX implementation of NanoJev's candidate-set decision head.

The Qwen3 backbone is loaded through mlx-lm; the non-generative NanoJev head
(norm, scalar score, and optional set-attention block) is evaluated directly
with MLX arrays. This keeps NanoJev's decision semantics instead of turning it
into an ordinary text-generation prompt.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load

from predict_toy_decisions import (
    answer_from_probabilities,
    complete_question_batches,
    prepare_examples,
    read_json,
    validate_request,
)


class MLXDecisionPredictor:
    """Persistent MLX NanoJev inference object."""

    def __init__(self, model_dir: str | Path, decision_head: str | Path,
                 max_length: int | None = None, lazy: bool = False):
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.decision_head_path = Path(decision_head).expanduser().resolve()
        if not self.model_dir.is_dir():
            raise ValueError(f"MLX model directory does not exist: {self.model_dir}")
        if not self.decision_head_path.is_file():
            raise ValueError(f"NanoJev decision head does not exist: {self.decision_head_path}")

        self.backbone, self.tokenizer = load(str(self.model_dir), lazy=lazy)
        self.hidden_size = int(self.backbone.model.args.hidden_size)
        config = self.backbone.model.args
        run_config_path = self.decision_head_path.parent / "config.json"
        run_max_length = None
        if run_config_path.is_file():
            try:
                run_max_length = json.loads(run_config_path.read_text(encoding="utf-8")).get("max_length")
            except (OSError, ValueError, TypeError):
                run_max_length = None
        self.max_length = int(max_length or run_max_length or getattr(config, "max_position_embeddings", 2048))
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

        weights = mx.load(str(self.decision_head_path))
        self.set_head = "attention"
        if run_config_path.is_file():
            try:
                configured_head = json.loads(run_config_path.read_text(encoding="utf-8")).get("set_head", self.set_head)
            except (OSError, ValueError, TypeError):
                configured_head = self.set_head
            if configured_head not in {"none", "attention"}:
                raise ValueError(f"unsupported NanoJev set_head: {configured_head!r}")
            self.set_head = configured_head
        required = {"norm.weight", "norm.bias", "scalar.weight", "scalar.bias"}
        if self.set_head == "attention":
            required.update({
                "set_project.weight", "set_project.bias",
                "set_attention.in_proj_weight", "set_attention.in_proj_bias",
                "set_attention.out_proj.weight", "set_attention.out_proj.bias",
                "set_output.weight", "set_output.bias",
            })
        missing = sorted(required.difference(weights))
        if missing:
            raise ValueError(f"decision head is missing keys: {missing}")
        # MLX backbone is float16 for the Apple GPU path. Keep the small head in
        # float32 so its numerics match the recorded PyTorch implementation.
        self.head = {key: value.astype(mx.float32) for key, value in weights.items() if key in required}
        self.inference_calls = 0

    @staticmethod
    def _layer_norm(x: mx.array, weight: mx.array, bias: mx.array, eps: float = 1e-5) -> mx.array:
        mean = mx.mean(x, axis=-1, keepdims=True)
        variance = mx.mean((x - mean) * (x - mean), axis=-1, keepdims=True)
        return (x - mean) * mx.rsqrt(variance + eps) * weight + bias

    def _set_attention(self, u: mx.array, valid: mx.array) -> mx.array:
        """Match torch.nn.MultiheadAttention(128, 4, batch_first=True)."""
        qkv = mx.matmul(u, self.head["set_attention.in_proj_weight"].T)
        qkv = qkv + self.head["set_attention.in_proj_bias"]
        q, k, v = mx.split(qkv, 3, axis=-1)
        batch, width, dim = q.shape
        heads = 4
        head_dim = dim // heads
        q = q.reshape(batch, width, heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, width, heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, width, heads, head_dim).transpose(0, 2, 1, 3)
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
        # PyTorch key_padding_mask masks keys, not queries. Padded queries are
        # masked only after the output decision logits are assembled.
        scores = mx.where(valid[:, None, None, :], scores, mx.array(-1e9, dtype=scores.dtype))
        probs = mx.softmax(scores, axis=-1)
        mixed = mx.matmul(probs, v).transpose(0, 2, 1, 3).reshape(batch, width, dim)
        return mx.matmul(mixed, self.head["set_attention.out_proj.weight"].T) + self.head["set_attention.out_proj.bias"]

    def _forward_batch(self, examples: list[dict[str, Any]]) -> mx.array:
        if not examples:
            raise ValueError("at least one example is required")
        paths = [ids for example in examples for ids in example["leaf_tokens"]]
        if not paths:
            raise ValueError("examples contain no candidate paths")
        lengths = [len(ids) for ids in paths]
        width = max(lengths)
        if width > self.max_length:
            raise ValueError(f"input length {width} exceeds max_length {self.max_length}")
        pad_token = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        tokens = mx.full((len(paths), width), pad_token, dtype=mx.int32)
        for i, ids in enumerate(paths):
            tokens[i, :len(ids)] = mx.array(ids, dtype=mx.int32)
        # Calling the inner Qwen3Model returns final hidden states, not LM logits.
        hidden = self.backbone.model(tokens)
        lengths_array = mx.array(lengths, dtype=mx.int32)
        leaves = hidden[mx.arange(len(paths), dtype=mx.int32), lengths_array - 1]

        kmax = max(len(example["candidate_ids"]) for example in examples)
        groups = []
        valid_rows = []
        offset = 0
        for example in examples:
            n = len(example["leaf_tokens"])
            row = leaves[offset:offset + n]
            if n < kmax:
                row = mx.concatenate([row, mx.zeros((kmax - n, self.hidden_size), dtype=row.dtype)], axis=0)
            groups.append(row)
            valid_rows.append([True] * len(example["candidate_ids"]) + [False] * (kmax - len(example["candidate_ids"])))
            offset += n
        h = mx.stack(groups, axis=0).astype(mx.float32)
        valid = mx.array(valid_rows, dtype=mx.bool_)
        h = self._layer_norm(h, self.head["norm.weight"], self.head["norm.bias"])
        z = mx.matmul(h, self.head["scalar.weight"].T) + self.head["scalar.bias"]
        z = z[..., 0]

        choice_positions = [i for i, example in enumerate(examples) if example["type"] == "choice"]
        choice_delta = {}
        if self.set_head == "attention" and choice_positions:
            choice_idx = mx.array(choice_positions, dtype=mx.int32)
            choice_h = h[choice_idx]
            choice_valid = valid[choice_idx]
            log_k = mx.log(mx.sum(choice_valid.astype(mx.float32), axis=-1))[:, None, None]
            log_k = mx.broadcast_to(log_k, (len(choice_positions), kmax, 1))
            u = mx.concatenate([choice_h, log_k], axis=-1)
            u = mx.matmul(u, self.head["set_project.weight"].T) + self.head["set_project.bias"]
            mixed = self._set_attention(u, choice_valid)
            delta = mx.matmul(mx.tanh(u + mixed), self.head["set_output.weight"].T)
            delta = delta + self.head["set_output.bias"]
            delta = delta[..., 0]
            for rank, position in enumerate(choice_positions):
                choice_delta[position] = delta[rank]

        rows = []
        for i, example in enumerate(examples):
            row = z[i]
            if i in choice_delta:
                row = row + choice_delta[i]
            if example["type"] == "boolean":
                boolean = mx.stack([row[0] * 0, row[0]])
                if kmax > 2:
                    boolean = mx.concatenate([boolean, mx.zeros((kmax - 2,), dtype=boolean.dtype)])
                row = boolean
            rows.append(row)
        logits = mx.stack(rows, axis=0)
        logits = mx.where(valid, logits, mx.array(-1e9, dtype=logits.dtype))
        mx.eval(logits)
        return logits

    def predict(self, payload: dict[str, Any], batch_questions: int = 0,
                temperature: float = 1.0) -> dict[str, Any]:
        states = validate_request(payload)
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise ValueError("temperature must be a finite positive number")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a finite positive number")
        examples = prepare_examples(payload, self.tokenizer, self.max_length)
        batches = complete_question_batches(examples, batch_questions)
        outputs = {state["id"]: {"id": state["id"], "answers": {}} for state in states}
        self.inference_calls += 1
        for batch in batches:
            logits = self._forward_batch(batch)
            for row, example in enumerate(batch):
                scores = logits[row, :len(example["candidate_ids"])] / temperature
                probabilities = mx.softmax(scores, axis=-1).tolist()
                if not all(math.isfinite(float(value)) for value in probabilities):
                    raise ValueError("MLX model produced non-finite probabilities")
                outputs[example["state_id"]]["answers"][example["qid"]] = answer_from_probabilities(example, probabilities)
        return {
            "schema_version": "openjev-mlx-inference-v1",
            "checkpoint": {
                "directory": str(self.decision_head_path.parent),
                "base_model": "Qwen/Qwen3-0.6B",
                "set_head": self.set_head,
                "backbone_format": "MLX",
                "decision_head": str(self.decision_head_path),
            },
            "temperature": {
                "value": float(temperature),
                "fitted_by_this_command": False,
                "note": "explicit scalar applied; default 1 is not a calibration claim.",
            },
            "execution": {
                "device": "Apple GPU via MLX",
                "parameter_storage": "backbone=float16, decision_head=float32",
                "precision": "mixed",
                "forward_autocast": "not applicable",
                "states": len(states),
                "questions": len(examples),
                "candidate_paths": sum(len(ex["leaf_tokens"]) for ex in examples),
                "forward_passes": len(batches),
                "batch_questions_limit": batch_questions or "all",
                "autoregressive_decode_steps": 0,
                "prefix_sharing": False,
                "max_length": self.max_length,
                "network_model_calls": 0,
                "persistent_model_load_count": 1,
                "inference_call_index": self.inference_calls,
            },
            "states": list(outputs.values()),
        }


def predict(payload: dict[str, Any], model_dir: str | Path, decision_head: str | Path,
            temperature: float = 1.0, batch_questions: int = 0,
            max_length: int | None = None) -> dict[str, Any]:
    validate_request(payload)
    predictor = MLXDecisionPredictor(model_dir, decision_head, max_length=max_length)
    return predictor.predict(payload, batch_questions=batch_questions, temperature=temperature)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--decision-head", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-questions", type=int, default=0)
    parser.add_argument("--max-length", type=int)
    args = parser.parse_args()
    result = predict(read_json(args.input), args.model_dir, args.decision_head,
                     temperature=args.temperature, batch_questions=args.batch_questions,
                     max_length=args.max_length)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(json.dumps({"output": args.output, "execution": result["execution"]}, ensure_ascii=False))
    else:
        print(text, end="")
