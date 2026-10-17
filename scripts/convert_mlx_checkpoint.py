#!/usr/bin/env python3
"""Convert a NanoJev checkpoint's Qwen3 backbone to an MLX model directory.

The NanoJev decision head stays in the original best.safetensors file and is
loaded by scripts/mlx_decisions.py. The generated MLX directory therefore
contains only the backbone, which is the format oMLX/ mlx-lm can discover.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from safetensors.numpy import load_file, save_file


def copy_tree_contents(source: Path, destination: Path) -> None:
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path,
                        help="NanoJev checkpoint containing best.safetensors/config.json")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="destination for the MLX backbone")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    weights_path = checkpoint / "best.safetensors"
    run_config_path = checkpoint / "config.json"
    body_config_path = checkpoint / "backbone_config" / "config.json"
    tokenizer_dir = checkpoint / "tokenizer"
    for path in (weights_path, run_config_path, body_config_path, tokenizer_dir):
        if not path.exists():
            raise SystemExit(f"missing NanoJev checkpoint file: {path}")
    if output.exists():
        if not args.force:
            raise SystemExit(f"output already exists; pass --force to replace: {output}")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nanojev-hf-", dir=str(output.parent)) as temp_name:
        hf_root = Path(temp_name)
        shutil.copy2(body_config_path, hf_root / "config.json")
        copy_tree_contents(tokenizer_dir, hf_root)
        # mlx-lm versions before the latest Transformers release expect the
        # legacy top-level rope_theta field rather than rope_parameters.
        body_config = json.loads((hf_root / "config.json").read_text(encoding="utf-8"))
        rope_parameters = body_config.pop("rope_parameters", None)
        if "rope_theta" not in body_config and isinstance(rope_parameters, dict):
            body_config["rope_theta"] = rope_parameters.get("rope_theta", 1000000)
        body_config["torch_dtype"] = "float32"
        (hf_root / "config.json").write_text(json.dumps(body_config, indent=2) + "\n", encoding="utf-8")

        raw = load_file(str(weights_path))
        backbone = {key[len("backbone."):]: value for key, value in raw.items()
                    if key.startswith("backbone.")}
        if not backbone:
            raise SystemExit("best.safetensors contains no backbone.* weights")
        state = {f"model.{key}": value for key, value in backbone.items()}
        # Qwen3-0.6B uses tied word embeddings; mlx-lm's CausalLM loader still
        # expects an explicit lm_head tensor in the temporary HF package.
        state["lm_head.weight"] = backbone["embed_tokens.weight"].copy()
        save_file(state, str(hf_root / "model.safetensors"), metadata={
            "source": "NanoJev backbone only; decision head remains in best.safetensors",
            "format": "pt",
        })

        command = [sys.executable, "-m", "mlx_lm", "convert",
                   "--hf-path", str(hf_root), "--mlx-path", str(output),
                   "--dtype", args.dtype]
        print("running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)

    # mlx-lm may regenerate tokenizer metadata. Preserve NanoJev's exact
    # tokenizer assets so candidate-path tokenization and EOS semantics stay
    # aligned with the PyTorch implementation. In particular, Qwen3 uses
    # <|im_end|> (151645) as EOS while <|endoftext|> (151643) is padding.
    copy_tree_contents(tokenizer_dir, output)
    source_tokenizer_config = tokenizer_dir / "tokenizer_config.json"
    if source_tokenizer_config.is_file():
        shutil.copy2(source_tokenizer_config, output / "tokenizer_config.json")
    source_chat_template = tokenizer_dir / "chat_template.jinja"
    if source_chat_template.is_file():
        shutil.copy2(source_chat_template, output / "chat_template.jinja")

    manifest = {
        "format": "nanojev-mlx-backbone-v1",
        "source_checkpoint": str(checkpoint),
        "source_best_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "decision_head": str(weights_path),
        "dtype": args.dtype,
        "note": "This directory contains the Qwen3 backbone only. Use scripts/mlx_decisions.py for NanoJev decision inference.",
    }
    (output / "nanojev_mlx_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "decision_head": str(weights_path), "dtype": args.dtype}, ensure_ascii=False))


if __name__ == "__main__":
    main()
