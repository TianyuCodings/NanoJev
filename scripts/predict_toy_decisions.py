#!/usr/bin/env python3
"""从本地 toy checkpoint 批量预测；无 teacher/gold、无下载、无自回归生成。"""
import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"JSON 含重复键：{key}")
        obj[key] = value
    return obj


def reject_nonfinite(value):
    raise ValueError(f"JSON 不允许非有限数值：{value}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                      parse_constant=reject_nonfinite)


def nonempty_text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_request(payload):
    if not isinstance(payload, dict) or set(payload) != {"states"}:
        raise ValueError("输入必须为且仅为 {\"states\": [...]}，不需要 teacher 或 gold")
    states = payload["states"]
    if not isinstance(states, list) or not states:
        raise ValueError("states 必须是非空数组")
    seen_ids = set()
    for state in states:
        if not isinstance(state, dict) or set(state) != {"id", "state", "questions"}:
            raise ValueError("每个 state 项必须只包含 id、state、questions")
        if not nonempty_text(state["id"]) or state["id"] in seen_ids:
            raise ValueError("state id 必须是唯一的非空字符串")
        seen_ids.add(state["id"])
        # 与trainer的 f-string 状态序列化一致；字符串最贴近本次训练数据。
        if not isinstance(state["state"], (str, dict, list)):
            raise ValueError("state 内容须为字符串、JSON对象或数组")
        if not state["state"]:
            raise ValueError("state 内容不得为空")
        questions = state["questions"]
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions 必须是非空对象")
        for qid, question in questions.items():
            if not nonempty_text(qid) or not isinstance(question, dict):
                raise ValueError("question ID 必须是非空字符串，内容必须为对象")
            if set(question) - {"type", "instructions", "criteria"}:
                raise ValueError(f"{state['id']}:{qid} 含不支持的 question 字段")
            typ = question.get("type")
            if typ not in {"boolean", "choice", "score"} or not nonempty_text(question.get("instructions")):
                raise ValueError(f"{state['id']}:{qid} 题型或 instructions 无效")
            if typ == "boolean":
                if "criteria" in question:
                    criteria = question["criteria"]
                    if not isinstance(criteria, dict) or set(criteria) - {"false", "true"}:
                        raise ValueError("Boolean criteria 只能是含 false 和/或 true 键的对象")
                    if not all(nonempty_text(value) for value in criteria.values()):
                        raise ValueError("Boolean criterion 必须是非空字符串")
            elif typ == "choice":
                criteria = question.get("criteria")
                if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                    raise ValueError("Choice criteria 必须是含 2–255 项的对象")
                if not all(nonempty_text(k) and nonempty_text(v) for k, v in criteria.items()):
                    raise ValueError("Choice 候选 ID 和语义描述必须是非空字符串")
            else:
                criteria = question.get("criteria")
                if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                    raise ValueError("Score criteria 必须是含 2–10 项的有序数组")
                if not all(nonempty_text(value) for value in criteria):
                    raise ValueError("Score 等级描述必须是非空字符串")
    return states


def prepare_examples(payload, tokenizer, max_length):
    """逐段encode、候选文本和EOS均精确遵循train_toy_decisions.load_examples。"""
    encoding = load_encoding_module()
    states = validate_request(payload)
    if type(max_length) is not int or max_length <= 0:
        raise ValueError("max_length 必须为正整数")
    if type(tokenizer.eos_token_id) is not int or tokenizer.eos_token_id < 0:
        raise ValueError("checkpoint tokenizer 必须有合法 eos_token_id")
    examples = []
    for row in states:
        for qid, q in row["questions"].items():
            typ = q["type"]
            # One canonical encoder for training and serving; see decision_encoding.py.
            ids, texts, prefix, leaves = encoding.build_candidate_paths(
                tokenizer.encode, row["state"], q, tokenizer.eos_token_id)
            largest = max(map(len, leaves))
            if largest > max_length:
                raise ValueError(f"{row['id']}:{qid} 候选路径为 {largest} token，超过 max_length={max_length}；未截断输入")
            examples.append({"id": f"{row['id']}:{qid}", "state_id": row["id"], "qid": qid,
                             "type": typ, "candidate_ids": ids, "candidate_texts": texts,
                             "prefix_length": len(prefix), "leaf_tokens": leaves})
    return examples


def complete_question_batches(examples, batch_questions=0):
    if type(batch_questions) is not int or batch_questions < 0:
        raise ValueError("batch_questions 必须为非负整数；0 表示全部问题一次前向")
    size = batch_questions or len(examples)
    if not examples:
        return []
    return [examples[i:i + size] for i in range(0, len(examples), size)]


def answer_from_probabilities(example, probabilities):
    ids = example["candidate_ids"]
    if len(probabilities) != len(ids) or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities):
        raise ValueError("模型产生了无效概率")
    if abs(math.fsum(probabilities) - 1.0) > 1e-5:
        raise ValueError("模型概率总和不为1")
    best = max(range(len(ids)), key=probabilities.__getitem__)
    result = {"type": example["type"], "probabilities": dict(zip(ids, probabilities))}
    if example["type"] == "boolean":
        result.update(p_true=probabilities[1], value=bool(best))
    elif example["type"] == "choice":
        result.update(choice=ids[best], value=ids[best])
    else:
        score = math.fsum(i * p for i, p in enumerate(probabilities))
        result.update(score=score, level=best, value=score)
    return result


def local_checkpoint_files(checkpoint_dir):
    root = Path(checkpoint_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("checkpoint-dir 必须是本地目录")
    paths = {"run_config": root / "config.json", "body_config": root / "backbone_config",
             "tokenizer": root / "tokenizer", "weights": root / "best.safetensors"}
    for label, path in paths.items():
        if not path.exists():
            raise ValueError(f"checkpoint 缺少 {label}: {path.name}")
    if not paths["run_config"].is_file() or not paths["weights"].is_file():
        raise ValueError("config.json 和 best.safetensors 必须是文件")
    if not paths["body_config"].is_dir() or not paths["tokenizer"].is_dir():
        raise ValueError("backbone_config 和 tokenizer 必须是目录")
    return root, paths


def load_sibling_module(module_name):
    """Load a sibling script by path without requiring it to be importable.

    Delayed on purpose: schema and tokenization checks must work on a machine without
    torch, and importing the trainer would execute its module level work.
    """
    path = Path(__file__).with_name(module_name + ".py")
    spec = importlib.util.spec_from_file_location("openjev_" + module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载本地模块：{module_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_encoding_module():
    """规范编码；与 trainer 共用同一份实现。"""
    return load_sibling_module("decision_encoding")


def load_decision_model_class():
    return load_sibling_module("train_toy_decisions").DecisionModel


def load_shared_prefix_module():
    """共享前缀编码器；schema 检查路径不需要它。"""
    return load_sibling_module("shared_prefix")


class DecisionPredictor:
    """本地持久推理对象：构造时加载一次权重，每次predict批量计算完整问题。"""

    def __init__(self, checkpoint_dir, max_length=None, device_name="cuda:0",
                 disable_native_triton=False, precision="bf16"):
        if precision not in {"fp32", "bf16"}:
            raise ValueError("precision 必须为 fp32 或 bf16")
        root, paths = local_checkpoint_files(checkpoint_dir)
        run_config = read_json(paths["run_config"])
        if not isinstance(run_config, dict) or run_config.get("set_head") not in {"none", "attention"}:
            raise ValueError("checkpoint config 缺少合法 set_head")
    
        # 只读本地checkpoint；不读取.env、不访问Hub、不下载原始模型权重。
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    
        if disable_native_triton:
            from torch._native import triton_utils
            triton_utils.deregister_op_overrides()
        device = torch.device(device_name)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("此原型推理入口需要可用CUDA设备；本命令未启用CPU或远程回退")
        torch.cuda.set_device(device)
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("当前CUDA设备不支持本checkpoint推理配置所需的BF16")
        torch.backends.cuda.matmul.allow_tf32 = False
    
        tokenizer = AutoTokenizer.from_pretrained(str(paths["tokenizer"]), local_files_only=True,
                                                 trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        body_config = AutoConfig.from_pretrained(str(paths["body_config"]), local_files_only=True,
                                                trust_remote_code=False)
        body_config.use_cache = False
        limit = run_config.get("max_length", 512) if max_length is None else max_length
        if type(limit) is not int or limit <= 0:
            raise ValueError("max-length 必须为正整数")
        context_limit = getattr(body_config, "max_position_embeddings", None)
        if isinstance(context_limit, int) and limit > context_limit:
            raise ValueError("max-length 超过backbone配置声明的上下文长度")
    
        # from_config 只构造结构；全部参数由best.safetensors加载，不用from_pretrained下载底座。
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
        self.disable_native_triton = disable_native_triton
        self.inference_calls = 0
        self._torch = torch

    def predict(self, payload, batch_questions=0, temperature=1.0, prefix_sharing=False,
                suffix_chunk=None):
        states = validate_request(payload)
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature 必须为有限正数")
        if type(prefix_sharing) is not bool:
            raise ValueError("prefix_sharing 必须为布尔值")
        if suffix_chunk is not None and (type(suffix_chunk) is not int or suffix_chunk < 1):
            raise ValueError("suffix_chunk 必须为 None 或正整数")
        torch = self._torch
        model, tokenizer = self.model, self.tokenizer
        root, run_config, limit = self.root, self.run_config, self.limit
        device, precision = self.device, self.precision
        disable_native_triton = self.disable_native_triton
        examples = prepare_examples(payload, tokenizer, limit)
        # With shared prefixes every question owns an independent prefix cache, so the
        # question limit no longer partitions the backbone work. The reference path keeps
        # its historical batching so that its reported forward_passes stay comparable.
        batches = [examples] if prefix_sharing else complete_question_batches(examples, batch_questions)
        self.inference_calls += 1
        model.eval()
        sharing = None
        if prefix_sharing:
            shared_prefix = load_shared_prefix_module()
            sharing = shared_prefix.SharedPrefixEncoder(
                model.backbone, pad_token_id=tokenizer.pad_token_id, device=device,
                suffix_chunk=suffix_chunk)
            SharedPrefixPlan = shared_prefix.SharedPrefixPlan
        outputs = {state["id"]: {"id": state["id"], "answers": {}} for state in states}
        with torch.inference_mode():
            for batch in batches:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                    logits, _ = model(batch, tokenizer.pad_token_id, prefix_sharing=sharing)
                for example, values in zip(batch, logits):
                    k = len(example["candidate_ids"])
                    scores = values[:k].float()
                    if not torch.isfinite(scores).all():
                        raise ValueError("模型产生非有限logits，未返回部分预测")
                    probabilities = (scores / temperature).softmax(-1).cpu().tolist()
                    outputs[example["state_id"]]["answers"][example["qid"]] = answer_from_probabilities(example, probabilities)
        # Report what ran, not what was requested: a batch with no shareable prefix falls
        # back to the reference encoder inside DecisionModel.forward.
        used_sharing = bool(sharing is not None and sharing.used_shared_prefix)
        sharing_report = {"requested": bool(prefix_sharing), "used": used_sharing,
                          "tree_attention": used_sharing,
                          "note": "候选路径共享 state 前缀；布尔与 Choice 的输入 token 与参考实现一致。"}
        if sharing is not None:
            plan = SharedPrefixPlan(examples)
            accounting = plan.accounting()
            sharing_report["accounting"] = accounting
            sharing_report["observed"] = dict(sharing.stats)
            sharing_report["plan_covers_all_questions"] = accounting["fully_shared"]
            if prefix_sharing and not used_sharing:
                sharing_report["fallback_reason"] = "本批次没有可共享前缀，已回退参考编码"
        return {
            "schema_version": "openjev-toy-inference-v1",
            "checkpoint": {"directory": str(root), "base_model": run_config.get("model"),
                           "base_revision": run_config.get("resolved_model_revision"), "set_head": run_config["set_head"]},
            "temperature": {"value": float(temperature), "fitted_by_this_command": False,
                            "note": "显式应用给定标量；默认1不表示模型已校准。"},
            "execution": {"device": str(device), "parameter_storage": "float32", "precision": precision,
                          "forward_autocast": "bfloat16" if precision == "bf16" else "disabled",
                          "states": len(states), "questions": len(examples),
                          "candidate_paths": sum(len(ex["leaf_tokens"]) for ex in examples),
                          "forward_passes": len(batches), "batch_questions_limit": batch_questions or "all",
                          "autoregressive_decode_steps": 0, "prefix_sharing": used_sharing,
                          "tree_attention": used_sharing,
                          "prefix_sharing_requested": bool(prefix_sharing),
                          "suffix_chunk": suffix_chunk,
                          "sharing": sharing_report,
                          "max_length": limit, "disable_native_triton": disable_native_triton,
                          "network_model_calls": 0, "persistent_model_load_count": 1,
                          "inference_call_index": self.inference_calls},
            "states": list(outputs.values()),
        }


def predict(payload, checkpoint_dir, temperature=1.0, batch_questions=0, max_length=None,
            device_name="cuda:0", disable_native_triton=False, precision="bf16",
            prefix_sharing=False, suffix_chunk=None):
    """兼容原一次性接口；连续调用请复用DecisionPredictor实例。"""
    # Fail on malformed input before loading a checkpoint, as in the original entry point.
    validate_request(payload)
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature 必须为有限正数")
    engine = DecisionPredictor(checkpoint_dir, max_length=max_length, device_name=device_name,
                               disable_native_triton=disable_native_triton, precision=precision)
    return engine.predict(payload, batch_questions=batch_questions, temperature=temperature,
                          prefix_sharing=prefix_sharing, suffix_chunk=suffix_chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--input", required=True, help="含states数组的JSON文件")
    parser.add_argument("--output", help="不设置时将完整结果输出到stdout")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-questions", type=int, default=0, help="0=全部问题一次前向；其他值按完整问题分批")
    parser.add_argument("--max-length", type=int, help="默认使用checkpoint训练配置；超长输入报错，不截断")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16",
                        help="bf16沿用训练评估默认；fp32关闭autocast用于数值参照")
    parser.add_argument("--disable-native-triton", action="store_true", help="沿用trainer的进程内ATen回退开关")
    parser.add_argument("--prefix-sharing", action="store_true",
                        help="每个 state/question 前缀只前向一次，候选仅前向自身后缀")
    parser.add_argument("--suffix-chunk", type=int,
                        help="共享前缀模式下每次后缀前向的候选行数上限；默认一次处理全部候选")
    args = parser.parse_args()
    try:
        result = predict(read_json(args.input), args.checkpoint_dir, args.temperature, args.batch_questions,
                         args.max_length, args.device, args.disable_native_triton, precision=args.precision,
                         prefix_sharing=args.prefix_sharing, suffix_chunk=args.suffix_chunk)
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
            print(json.dumps({"output": str(destination), "execution": result["execution"]}, ensure_ascii=False))
        else:
            print(text, end="")
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
