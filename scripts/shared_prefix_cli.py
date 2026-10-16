#!/usr/bin/env python3
"""Shared setup for the two shared-prefix command-line tools.

`benchmark_shared_prefix.py` measures the encoding and `verify_shared_prefix.py` asserts it
reproduces the reference path. Both need the same four things: sibling modules loaded as
top-level names, a checkpoint resolved and **strictly** loaded, a payload whose states are
genuinely distinct, and a verbatim copy of the released `forward` as an oracle. Keeping one
copy here means a loading mistake cannot be fixed in one tool and left in the other — which is
exactly how an earlier revision of the benchmark measured a randomly initialised backbone.

No network access, no training, no optimization.
"""
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASED_WEIGHTS_SHA256 = "f68c47d66998231b86b7e91b4ed5e82ae23acf104c8b7cd6d165c3ac7b7ffe1b"
RELEASED_REVISION = "unified-games-v1"


def load_sibling(name):
    """Load `scripts/<name>.py` as a top-level module, as the scripts expect."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    if name in sys.modules:
        return sys.modules[name]
    path = HERE / (name + ".py")
    if not path.is_file():
        raise SystemExit(f"missing sibling module: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_checkpoint_dir(explicit=None):
    """Find a checkpoint directory, preferring an explicit path.

    Honours `NANOJEV_CHECKPOINT_DIR`, then walks up from the checkout so a linked git
    worktree finds what only the primary checkout holds.
    """
    relative = Path("checkpoints") / "Qwen3-0.6B"
    candidates = [explicit, os.environ.get("NANOJEV_CHECKPOINT_DIR")]
    candidates += [d / relative for d in [HERE.parent, *list(HERE.parent.parents)[:3]]]
    for candidate in candidates:
        if candidate and (Path(candidate) / "tokenizer.json").is_file():
            return Path(candidate)
    return None


def load_tokenizer(AutoTokenizer, directory):
    """Load a tokenizer, normalizing one upstream export detail.

    transformers 4.51 wrote `extra_special_tokens` as a list; 4.57 expects a mapping and
    raises while reading it. The released `tokenizer.json` is byte-identical to the base
    model's, so only that metadata field is rewritten, in a temp copy of the config.
    """
    import json
    import shutil
    import tempfile
    config_path = Path(directory) / "tokenizer_config.json"
    if not isinstance(json.loads(config_path.read_text()).get("extra_special_tokens"), list):
        return AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    staging = Path(tempfile.mkdtemp(prefix="nanojev-tokenizer-"))
    for item in Path(directory).iterdir():
        if item.is_file():
            shutil.copy2(item, staging / item.name)
    config = json.loads((staging / "tokenizer_config.json").read_text())
    config["extra_special_tokens"] = {}
    (staging / "tokenizer_config.json").write_text(json.dumps(config, indent=2))
    return AutoTokenizer.from_pretrained(str(staging), local_files_only=True)


def load_model(trainer, AutoModel, config, backbone, weights):
    """Load a checkpoint into a full DecisionModel and refuse a silent partial load.

    Two key layouts exist. A decision checkpoint stores `backbone.*` plus decision-head
    weights; a base `model.safetensors` stores `model.*` and an `lm_head` that a bare
    `AutoModel` does not have. `load_state_dict(..., strict=False)` matches **zero** keys in
    the second case and leaves the backbone random, so the layout is resolved explicitly and
    leftovers are an error rather than a quietly weaker measurement.
    """
    import torch
    from safetensors.torch import load_file
    state = load_file(str(weights))
    if any(key.startswith("backbone.") for key in state):
        model = trainer.DecisionModel(backbone, "attention")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise SystemExit(f"decision checkpoint did not load cleanly: "
                             f"missing={missing[:4]} unexpected={unexpected[:4]}")
        return model, f"decision checkpoint {weights.name}: {len(state)} tensors"
    expected = set(backbone.state_dict())
    remapped, dropped = {}, []
    for key, tensor in state.items():
        candidate = key[len("model."):] if key.startswith("model.") else key
        if candidate in expected:
            remapped[candidate] = tensor
        else:
            dropped.append(key)
    if len(remapped) < 0.9 * len(expected):
        raise SystemExit(f"only {len(remapped)} of {len(expected)} backbone tensors mapped "
                         f"from {weights.name}; refusing a partial load")
    missing, unexpected = backbone.load_state_dict(remapped, strict=False)
    if missing or unexpected:
        raise SystemExit(f"backbone did not load cleanly: missing={missing[:4]} "
                         f"unexpected={unexpected[:4]}")
    return (trainer.DecisionModel(backbone, "attention"),
            f"base backbone {weights.name}: {len(remapped)} tensors "
            f"(ignored {len(dropped)} head tensors)")


def find_weights(root):
    for name in ("best.safetensors", "model.safetensors"):
        candidate = Path(root) / name
        if candidate.is_file():
            return candidate
    raise SystemExit(f"no weights under {root}")


def distinct_payload(candidates, states):
    """A payload whose states differ in **text**, not only in `id`.

    Varying only `id` makes every state tokenize identically, so the whole question becomes
    an exact cache reuse and a measurement silently reports prefix sharing plus
    whole-question deduplication.
    """
    criteria = {f"room_{i}": f"enter room {i} whose north side is "
                             f"{'open' if i % 2 else 'blocked'}"
                for i in range(candidates)}
    return {"states": [
        {"id": f"s{index}",
         "state": (f"Local map variant {index}: A is north of the exit, B is a wall, C is "
                   f"open. The agent stands in a corridor with {2 + index} untried exits."),
         "questions": {
             "action": {"type": "choice",
                        "instructions": "Which room should the agent enter?",
                        "criteria": criteria},
             "safe": {"type": "boolean", "instructions": "Is the chosen room safe?",
                      "criteria": {"false": "The room contains a wall.",
                                   "true": "The room is inside the maze and open."}}}}
        for index in range(states)]}


def released_forward(model, examples, pad_token, torch):
    """Verbatim body of `DecisionModel.forward` at commit 618cea6, used as an oracle."""
    F = torch.nn.functional
    paths = [ids for ex in examples for ids in ex["leaf_tokens"]]
    device = model.scalar.weight.device
    lengths = torch.tensor([len(ids) for ids in paths], device=device)
    width = int(lengths.max())
    tokens = torch.full((len(paths), width), pad_token, dtype=torch.long, device=device)
    for i, ids in enumerate(paths):
        tokens[i, :len(ids)] = torch.tensor(ids, device=device)
    attention = torch.arange(width, device=device)[None, :] < lengths[:, None]
    hidden = model.backbone(input_ids=tokens, attention_mask=attention,
                            use_cache=False).last_hidden_state
    leaves = hidden[torch.arange(len(paths), device=device), lengths - 1]
    kmax = max(len(ex["candidate_ids"]) for ex in examples)
    h = leaves.new_zeros((len(examples), kmax, leaves.shape[-1]))
    valid = torch.zeros((len(examples), kmax), dtype=torch.bool, device=device)
    offset = 0
    for i, ex in enumerate(examples):
        n = len(ex["leaf_tokens"])
        h[i, :n] = leaves[offset:offset + n]
        valid[i, :len(ex["candidate_ids"])] = True
        offset += n
    h = model.norm(h)
    z = model.scalar(h).squeeze(-1).float()
    choice = torch.tensor([i for i, ex in enumerate(examples) if ex["type"] == "choice"],
                          device=device)
    if model.set_head == "attention" and len(choice):
        log_k = valid[choice].sum(-1).float().log()[:, None, None].expand(-1, kmax, 1)
        u = model.set_project(torch.cat([h[choice], log_k.to(h.dtype)], dim=-1))
        mixed, _ = model.set_attention(u, u, u, key_padding_mask=~valid[choice],
                                        need_weights=False)
        delta = model.set_output(torch.tanh(u + mixed)).squeeze(-1).float()
        z = z.index_add(0, choice, delta)
    out = []
    for i, ex in enumerate(examples):
        if ex["type"] == "boolean":
            out.append(F.pad(torch.stack([z[i, 0] * 0, z[i, 0]]), (0, kmax - 2)))
        else:
            out.append(z[i])
    return torch.stack(out).masked_fill(~valid, -1e9), valid


def prepare_inference(args, torch, suffix_chunk=None):
    """Resolve checkpoint, tokenizer, weights and a strictly loaded DecisionModel."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    trainer = load_sibling("train_toy_decisions")
    shared_prefix = load_sibling("shared_prefix")
    predictor = load_sibling("predict_toy_decisions")

    explicit = getattr(args, "checkpoint_dir", None)
    if explicit:
        root = Path(explicit)
        tokenizer_dir = root / "tokenizer" if (root / "tokenizer").is_dir() else root
        backbone_dir = root / "backbone_config" if (root / "backbone_config").is_dir() else root
        for required in (tokenizer_dir, backbone_dir):
            if not required.exists():
                raise SystemExit(f"missing {required}")
    else:
        root = resolve_checkpoint_dir()
        if root is None:
            raise SystemExit("no local tokenizer; pass --checkpoint-dir or set "
                             "NANOJEV_CHECKPOINT_DIR")
        tokenizer_dir = backbone_dir = root

    tokenizer = load_tokenizer(AutoTokenizer, tokenizer_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(str(backbone_dir), local_files_only=True)
    config.use_cache = True
    config._attn_implementation = "sdpa"
    weights = find_weights(root)
    digest = sha256_file(weights)
    released = digest == RELEASED_WEIGHTS_SHA256 and (root / "backbone_config").is_dir()
    if released:
        # The released bundle declares its backbone separately; load the decision checkpoint.
        weights = Path(root) / "best.safetensors"
    model, description = load_model(trainer, AutoModel, config, AutoModel.from_config(config),
                                   weights)
    # A base checkpoint stores bf16 tensors while freshly built decision-head parameters are
    # float32; pin both to float32 and apply bf16 as autocast, which is what serving does.
    model = model.to(dtype=torch.float32)
    return {
        "trainer": trainer, "shared_prefix": shared_prefix, "predictor": predictor,
        "tokenizer": tokenizer, "model": model, "config": config, "root": root,
        "weights": weights, "weights_sha256": digest, "released": released,
        "description": description,
    }
