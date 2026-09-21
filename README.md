# NanoJev — A nano replica of [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

**English** | [简体中文](README.zh-CN.md)

**A 0.6B parallel decision model: states and questions in, complete probability distributions out. Zero output-token decoding.**

> **New project: [JevHarness](https://github.com/TianyuCodings/JevHarness)** — Let an LLM build task-specific decision harnesses with [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), with optional refinement using rewards and execution traces. Includes an interactive Pokémon demo.

[Play ViZDoom](https://nanojev-dev.tianyuchen99.chatgpt.site/?autoplay=1) · [Maze & Snake](https://nanojev-dev.tianyuchen99.chatgpt.site/side-by-side?autoplay=1#maze) · [Model](https://huggingface.co/C-Tianyu/NanoJev) · [Dataset](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data)

**Now playing ViZDoom:** one shared checkpoint handles Basic aiming and Predict Position's moving-target rocket shots, alongside Maze and Snake.

**4 tasks · 18,760 decision questions per data variant · 896 Predict Position expert episodes**

## What's new

**September 20, 2026 — One model, four games.**

- **ViZDoom Basic:** **128/128** test successes, compared with **56/128** for [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).
- **ViZDoom Predict Position:** **27/128** test successes, up from **11/128** before this round; the matched [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) run also scores **11/128**. The policy learns when to turn, wait and fire at a moving target.
- **16,333 ViZDoom questions** within an **18,760-question** mixed-task dataset per target variant, spanning train, dev, calibration, test and OOD.
- **One model, four games:** the same step-400 checkpoint also completes the 50×50 maze in **225 attempts** and collects **30 food items** during a full 256-step Snake run.

## Three models, side by side

Real browser replays of **[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), NanoJev and Untuned Qwen**. These animations loop automatically; click either one to open its interactive player. All four demos use the same current NanoJev checkpoint. The interactive development site currently requires access; all recordings can also be played locally using the commands below.

### ViZDoom Basic · Aim, then fire

[![NanoJev eliminates the target with one shot while Jev and Untuned Qwen fail, shown side by side on the same game clock](assets/basic_unified_autoplay.gif)](https://nanojev-dev.tianyuchen99.chatgpt.site/?autoplay=1)

Move into position, line up the target, fire. NanoJev eliminates the target with **one shot in 1.40 s**; [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) and Untuned Qwen each fire **19 shots** without an elimination before the deadline. The three panels share the same game clock and show original frames and action probabilities.

### Find the exit · 50×50 Maze

[![Jev, current NanoJev and Untuned Qwen explore the same 50×50 maze in the live three-panel viewer](assets/maze_unified_autoplay.gif)](https://nanojev-dev.tianyuchen99.chatgpt.site/side-by-side?autoplay=1#maze)

NanoJev reaches the exit in **225 attempts**, versus **2,738** for [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) and **4,726** for Untuned Qwen. Each system combines local safety probabilities with the same exploration code and remembered open paths.

[Play Snake](https://nanojev-dev.tianyuchen99.chatgpt.site/side-by-side?autoplay=1#snake) · [Play Predict Position](https://nanojev-dev.tianyuchen99.chatgpt.site/predict-position?autoplay=1)

## What NanoJev does

- **Parallel decisions:** batch independent states, questions and candidate paths in one backbone forward.
- **Dynamic candidates:** Choice returns a distribution over 2–255 supplied candidates using a shared scoring head.
- **Boolean and ordered scores:** predict a proposition's probability, or a distribution and expectation over 2–10 ordered levels.
- **Direct probabilities:** rank, select or sample actions without generating answer tokens.
- **One small backbone:** Qwen3-0.6B with decision heads, reused across all four game tasks and a persistent inference service.

Each request supplies a **state**, a **question** and its **candidates**. The backbone encodes candidate paths; shared heads produce the requested probabilities. Choice uses set attention and a softmax, Boolean uses a sigmoid, and Score returns a probability-weighted level.

## Held-out gameplay

Successful episodes on the complete **274-case test set**, using the same observation interface, candidate actions and seeded epsilon-greedy controller across systems:

| Model | Maze | Snake | Basic | Predict Position |
|---|---:|---:|---:|---:|
| **NanoJev** | **4/10** | **8/8** | **128/128** | **27/128** |
| [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | 7/10 | 8/8 | 56/128 | 11/128 |
| Untuned Qwen3-0.6B | 2/10 | 0/8 | 56/128 | 11/128 |

Test and OOD together contain **548 cases per model**. Every evaluated trajectory passes independent simulator replay. The large navigation showcases above use their displayed local-question and code-planning settings.

[Complete test and OOD results](docs/SONIC_PREDICT_POSITION_RESULTS.md) · [Training pipeline](docs/SONIC_PREDICT_POSITION.md)

## Dataset scale and model

**18,760 decision questions per target variant, including 16,333 ViZDoom questions.** The matched hard-target and soft-target variants cover the same questions across train, dev, calibration, test and OOD.

| Task | All five splits | Training split |
|---|---:|---:|
| **ViZDoom Predict Position** | **11,173** | **6,788** |
| **ViZDoom Basic** | **5,160** | **3,054** |
| Maze | 1,469 | 653 |
| Snake | 958 | 403 |
| **Total per variant** | **18,760** | **10,898** |

**Expert gameplay:** the package includes **896 Predict Position episodes with 17,498 recorded decisions**, including 512 episodes assigned to training. It also contains the original mixed-task inputs, hard/soft targets, evaluation trajectories and replay checks.

Of the 10,898 stored training questions, **10,893** pass the target-validity filter. Existing Maze, Snake and Basic splits are preserved.

### Release version and training run

**`unified-games-v1` packages the step-400 checkpoint from the `hard_lr1e5` training run.** Both names refer to the same selected model used across the four demos.

| Name | Meaning | When to use it |
|---|---|---|
| **`unified-games-v1`** | Hugging Face release tag identifying the matching model and dataset snapshots. | Download with `revision="unified-games-v1"`. |
| **`hard_lr1e5`** | Training experiment: hard (one-hot) action targets for Predict Position, backbone learning rate `1e-5`, decision-head learning rate `1e-4`. | Inspect training configs, logs and experiment comparisons. |

The shared model is trained with complete-question cross entropy. Updates mix Maze, Snake, Basic and Predict Position with weights **1/3, 1/3, 1/6, 1/6**.

**Hugging Face release complete:** the model and complete dataset are uploaded and verified as `unified-games-v1`. Both release tags resolve to their recorded snapshots; every uploaded file passes remote identity checks.

The model and dataset are public and can be downloaded without signing in.

## Quick start

```bash
git clone https://github.com/TianyuCodings/NanoJev.git
cd NanoJev
python -m pip install -r requirements-toy.txt huggingface_hub
```

Download the current checkpoint and data:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="C-Tianyu/NanoJev",
    revision="unified-games-v1",
    local_dir="checkpoints/NanoJev-unified",
    allow_patterns=["best.safetensors", "config.json", "tokenizer/*", "backbone_config/*"],
)
snapshot_download(
    repo_id="C-Tianyu/NanoJev-Data",
    repo_type="dataset",
    revision="unified-games-v1",
    local_dir="data/NanoJev-unified",
)
```

Start inference in a CUDA environment:

```bash
python scripts/serve_decisions.py \
  --checkpoint-dir checkpoints/NanoJev-unified \
  --web-root web --port 8765 --disable-native-triton
```

The service loads the model once. Send state/question batches to **`POST http://127.0.0.1:8765/api/evaluate`**.

To explore the recorded games locally:

```bash
python3 -m http.server 8080 --bind 127.0.0.1 --directory web
```

Open **http://127.0.0.1:8080/dev/?autoplay=1** for ViZDoom Basic or **http://127.0.0.1:8080/dev/side-by-side.html?autoplay=1#maze** for Maze.

## Development notes

[Release contents and reproduction](docs/UNIFIED_DEVELOPMENT_RELEASE.md) · [Input contract](docs/TYPESAFE_CONTRACT.md) · [Unified environments](docs/UNIFIED_GAMES.md) · [Atomic planning](docs/ATOMIC_PLANNING.md) · [Predict Position replay](docs/PREDICT_POSITION_DEMO.md) · [Shooting replay](docs/SHOOTING_DEMO.md)

## Apple Silicon / MLX inference

NanoJev is a non-generative decision model: the Qwen3 backbone produces candidate-path hidden states, then NanoJev's decision head emits the complete Boolean, Choice, or Score distribution. On Apple Silicon, the MLX backend keeps that decision head instead of routing the request through ordinary text generation.

Install the optional MLX dependencies in an arm64 Python environment:

```bash
python -m pip install -r requirements-mlx.txt
```

Convert a downloaded NanoJev checkpoint. The converter extracts only `backbone.*`, preserves the original tokenizer (including Qwen3's `<|im_end|>` EOS), and leaves `best.safetensors` as the decision-head file:

```bash
python scripts/convert_mlx_checkpoint.py \
  --checkpoint-dir checkpoints/local_atomic_seed17 \
  --output-dir omlx-backbone-mlx \
  --dtype float16
```

Run the MLX decision service:

```bash
python scripts/serve_decisions_mlx.py \
  --model-dir omlx-backbone-mlx \
  --decision-head checkpoints/local_atomic_seed17/best.safetensors \
  --web-root web --port 8765
```

The service exposes the same `GET /api/health` and `POST /api/evaluate` contract as the PyTorch service. It performs one non-autoregressive backbone pass, returns complete candidate distributions, and supports Boolean, Choice, and Score questions. The current Apple Silicon path stores the Qwen3 backbone in float16 and the small decision head in float32.

The converted `omlx-backbone-mlx` directory is also discoverable by oMLX as an ordinary Qwen3 model. Copy it below an oMLX model root as `nanojev-backbone` and verify it through oMLX's `/v1/models` or `/v1/chat/completions` endpoint. The NanoJev decision service currently loads the same MLX artifact directly; it does not ask oMLX's text-generation endpoint for hidden states, and ordinary oMLX `/v1/chat/completions` therefore remains a backbone smoke test rather than the NanoJev decision API. The structured decision endpoint is `scripts/serve_decisions_mlx.py` and `/api/evaluate`.

To verify native oMLX discovery separately from NanoJev decision inference:

```bash
python scripts/test_omlx_compat.py \
  --model-dir omlx-backbone-mlx \
  --omlx-cli /path/to/omlx
```

This smoke test checks that oMLX discovers `nanojev-backbone` and serves ordinary `/v1/chat/completions`. The structured NanoJev path remains `/api/evaluate`, because the public oMLX text API does not expose backbone hidden states or a custom decision-head hook.

A warm inference benchmark (excluding process startup and HTTP transport) is available:

```bash
python scripts/benchmark_mlx_decisions.py \
  --model-dir omlx-backbone-mlx \
  --decision-head checkpoints/local_atomic_seed17/best.safetensors \
  --questions 1 4 8 --candidates 2 4 8
```

On the development M5 Max run, 8 questions × 8 candidates completed in a median **78.55 ms** (64 candidate paths); the exact numbers depend on the Apple Silicon model, MLX version, and thermal state.

Run the local regression/contract test, optionally against a PyTorch/MPS reference result:

```bash
python scripts/test_mlx_decisions.py \
  --model-dir omlx-backbone-mlx \
  --decision-head checkpoints/local_atomic_seed17/best.safetensors
```


## Roadmap

- [x] One unified checkpoint for Maze, Snake and both shooting tasks.
- [x] 50×50 Maze, long Snake games and synchronized three-model browser replays.
- [x] Mixed-task SFT, reproducible data splits and independently replayed evaluation.
- [ ] RLCD post-training for broader long-horizon tasks.
- [ ] Shared-prefix inference and larger candidate batches.
- [ ] Broader shooting scenarios and structured input support.
