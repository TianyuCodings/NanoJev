"""CPU-accelerated NanoJev inference: thread tuning + prefix sharing for high-candidate questions.

Numerically equivalent to predict_toy_decisions.py. Questions with more than
--threshold candidates encode the shared state+question prefix once and batch
the candidate suffixes through the cached KV, avoiding repeated backbone work.
"""
import argparse
import json
import math
import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

from predict_toy_decisions import (
    answer_from_probabilities,
    complete_question_batches,
    load_decision_model_class,
    local_checkpoint_files,
    prepare_examples,
    read_json,
    validate_request,
)

SHARED_THRESHOLD = 8  # 前缀共享在候选数 > 8 时开始优于直接 batch (实测交叉点)


class FastDecisionPredictor:
    """Persistent CPU predictor with prefix sharing. Loads weights once, reuses them across calls."""

    def __init__(self, checkpoint_dir, max_length=None, device_name="cpu", precision="fp32", threads=None):
        if precision not in {"fp32", "bf16"}:
            raise ValueError("precision 必须为 fp32 或 bf16")
        if threads is not None:
            torch.set_num_threads(threads)
        root, paths = local_checkpoint_files(checkpoint_dir)
        run_config = read_json(paths["run_config"])
        if not isinstance(run_config, dict) or run_config.get("set_head") not in {"none", "attention"}:
            raise ValueError("checkpoint config 缺少合法 set_head")

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

        device = torch.device(device_name)
        tokenizer = AutoTokenizer.from_pretrained(str(paths["tokenizer"]), local_files_only=True,
                                                 trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        body_config = AutoConfig.from_pretrained(str(paths["body_config"]), local_files_only=True,
                                                trust_remote_code=False)
        body_config.use_cache = True  # forward_shared 需要 past_key_values; 完整路径 forward 显式传 False
        limit = run_config.get("max_length", 512) if max_length is None else max_length
        if type(limit) is not int or limit <= 0:
            raise ValueError("max-length 必须为正整数")

        body = AutoModel.from_config(body_config, attn_implementation="sdpa", trust_remote_code=False).float()
        DecisionModel = load_decision_model_class()
        model = DecisionModel(body, run_config["set_head"])
        weights = load_file(str(paths["weights"]), device="cpu")
        model.load_state_dict(weights, strict=True)
        del weights
        model.to(device=device, dtype=torch.float32)
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.root = root
        self.run_config = run_config
        self.limit = limit
        self.device = device
        self.precision = precision
        self.inference_calls = 0

    def predict(self, payload, temperature=1.0, threshold=SHARED_THRESHOLD):
        states = validate_request(payload)
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature 必须为有限正数")
        examples = prepare_examples(payload, self.tokenizer, self.limit)
        outputs = {state["id"]: {"id": state["id"], "answers": {}} for state in states}
        model, pad_token = self.model, self.tokenizer.pad_token_id
        shared = [ex for ex in examples if len(ex["leaf_tokens"]) > threshold]
        batched = [ex for ex in examples if len(ex["leaf_tokens"]) <= threshold]
        self.inference_calls += 1
        with torch.inference_mode():
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.precision == "bf16"):
                for ex in shared:
                    logits, _ = model.forward_shared(ex, pad_token)
                    k = len(ex["candidate_ids"])
                    probs = (logits[0, :k].float() / temperature).softmax(-1).cpu().tolist()
                    outputs[ex["state_id"]]["answers"][ex["qid"]] = answer_from_probabilities(ex, probs)
                for batch in complete_question_batches(batched, 0):
                    logits, _ = model(batch, pad_token)
                    for ex, values in zip(batch, logits):
                        k = len(ex["candidate_ids"])
                        probs = (values[:k].float() / temperature).softmax(-1).cpu().tolist()
                        outputs[ex["state_id"]]["answers"][ex["qid"]] = answer_from_probabilities(ex, probs)
        return {
            "schema_version": "openjev-toy-inference-v1",
            "checkpoint": {"directory": str(self.root), "base_model": self.run_config.get("model"),
                           "base_revision": self.run_config.get("resolved_model_revision"),
                           "set_head": self.run_config["set_head"]},
            "temperature": {"value": float(temperature), "fitted_by_this_command": False,
                            "note": "显式应用给定标量；默认1不表示模型已校准。"},
            "execution": {"device": str(self.device), "precision": self.precision,
                          "states": len(states), "questions": len(examples),
                          "shared_prefix_questions": len(shared),
                          "batched_questions": len(batched),
                          "inference_call_index": self.inference_calls},
            "states": list(outputs.values()),
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--input", required=True, help="含states数组的JSON文件")
    p.add_argument("--output", required=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--threshold", type=int, default=SHARED_THRESHOLD,
                   help="候选数超过该值走前缀共享; 0 表示全部走前缀共享")
    p.add_argument("--threads", type=int, default=None, help="torch 线程数; 默认系统决定")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=None)
    a = p.parse_args()
    engine = FastDecisionPredictor(a.checkpoint_dir, max_length=a.max_length,
                                   device_name=a.device, precision=a.precision, threads=a.threads)
    payload = json.loads(Path(a.input).read_text(encoding="utf-8"))
    result = engine.predict(payload, temperature=a.temperature, threshold=a.threshold)
    Path(a.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": a.output, "execution": result["execution"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
