# NanoJev — A nano replica of [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

**English** | [简体中文](README.zh-CN.md)

**A 0.6B parallel decision model. States and questions in, complete probability distributions out—with zero output-token decoding.**

[Model](https://huggingface.co/C-Tianyu/NanoJev) · [Dataset](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data) · [Maze + Snake: open the decision arcade](web/arcade.html)

## Recorded showcase runs

Watch model judgments and shared code planning work together. Each game uses the same controller code across its three systems; the recordings preserve the actual actions, probabilities, and final outcomes.

### Find the exit: 50×50 maze

[![NanoJev finds the exit in a 50×50 maze, with recorded comparison results](assets/arcade_maze.gif)](assets/arcade_maze.mp4)

[Watch the MP4](assets/arcade_maze.mp4) · [Interactive replay](web/arcade.html)

The model judges four local directions. Code remembers collisions, explores untried edges, and repositions through verified open paths.

| System | Attempts | Collisions | Outcome |
|---|---:|---:|---|
| **NanoJev** | **244** | **36** | **Goal reached** |
| [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | 2,738 | 1,044 | Goal reached |
| Starting NanoJev | 171 | 43 | Goal reached |

Starting NanoJev is the earlier trained NanoJev checkpoint. The new NanoJev model uses matched local safety training.

### Keep growing: 12×12 Snake

[![NanoJev grows through a complete Snake run, with recorded comparison results](assets/arcade_snake.gif)](assets/arcade_snake.mp4)

[Watch the MP4](assets/arcade_snake.mp4) · [Interactive replay](web/arcade.html)

The common planner filters immediate collisions and finds static paths toward the visible food. The model breaks ties between the remaining actions; a single remaining action is a code-forced move. **Seed: 61005. Controller: greedy.**

| System | Food collected | Steps | Outcome |
|---|---:|---:|---|
| **NanoJev** | **27** | **256** | **Alive at horizon** |
| [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | 30 | 256 | Alive at horizon |
| Untuned Qwen3-0.6B | 25 | 211 | Trapped |

Untuned Qwen uses its original pretrained weights and native language-model head, conditioned on the offered A–D answer tokens.

[Recorded cases and replay verification](assets/arcade_data_manifest.json) · [Eight-case controller comparison](results/arcade_controller_comparison.json)

## Features

- **0.6B LLM backbone.** Qwen3-0.6B with decision heads for structured outputs.
- **Multiple states and questions in one forward.** Batch independent decisions together.
- **Dynamic Choice.** Supply **2–255 candidates** and receive a probability for every candidate.
- **Boolean decisions.** Receive the probability that a complete proposition is true.
- **Ordered Score.** Supply **2–10 levels** and receive the level distribution and expected score.
- **Complete distributions.** Use the same output for ranking, greedy selection, or probability sampling.
- **Zero output decoding.** Read decisions directly from a forward pass.
- **Persistent serving.** Load a checkpoint once and reuse it across requests.

Measured in the running service: **6 states · 18 questions · 44 candidate paths · 1 backbone forward**.

## Larger games and calibrated decisions

- **Full-size environments:** 8×8, 16×16, 32×32, and 50×50 mazes, four topologies, multiple positions per map, and configurable larger sizes.
- **Local judgments + code planning:** matched 5×5 observations, four parallel safety judgments, movement memory, and model-guided exploration.
- **Snake dynamics:** reproducible food generation, body growth, collision rules, tail movement, dynamic action candidates, and safety questions.
- **Probability learning:** observed-event datasets, CE/Brier training, paired proper-reward learning, exact gradient checks, and completed Qwen3-0.6B runs.
- **Verified evaluation:** map-separated data, frozen game cohorts, real model execution, and independent trajectory replay.

The local safety model reaches **77.84% accuracy on test questions** and **76.56% on 50×50 OOD questions**. The probability-learning pilot's paired proper-reward arm reaches **0.11844 test / 0.06202 OOD distribution error**, measured as the sum of squared differences from the simulator's event probabilities.

[Atomic planning](docs/ATOMIC_PLANNING.md) · [Scaled-game pipeline](docs/SCALED_GAMES.md) · [RLCD implementation and results](docs/RLCD_EXPERIMENT.md) · [Input contract](docs/TYPESAFE_CONTRACT.md) · [Game results](docs/DEVELOPMENT_RESULTS.md)

## Earlier 40-map navigation benchmark

**Controller: T=1 probability sampling.** The full benchmark contains 20 test maps and 20 OOD maps.

| System | 4×4 test | 6×6 OOD |
|---|---:|---:|
| **NanoJev** | **19/20 — 95%** | **18/20 — 90%** |
| [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | 20/20 — 100% | 19/20 — 95% |
| Untuned Qwen3-0.6B | 7/20 — 35% | 3/20 — 15% |

[Earlier comparison viewer](web/comparison.html) · [Complete benchmark results](research/nanojev_comparison_public.json)

## How it works

Each decision is defined by a **state**, a **question**, and its **candidate set**. Every candidate path carries the relevant input into the backbone. Shared decision heads return a distribution over the candidates supplied for that question.

Choice uses a shared scalar head and set attention. Boolean uses a single-path sigmoid. Score evaluates its ordered level descriptions and returns their probability-weighted expectation.

1. **Build queries.** Generate states, questions, candidate descriptions, and target distributions.
2. **Organize data.** Keep related maps, rules, and their variations in the same split.
3. **Train.** Initialize Qwen3-0.6B, warm up the decision heads, and train with complete-question distribution losses.
4. **Evaluate.** Measure probability quality and execute game controllers with recorded actions.
5. **Serve and visualize.** Reuse a persistent model endpoint and replay complete trajectories in the browser.

[Complete pipeline commands](research/pipeline_runbook.md)

## Quick start: decision arcade

The interactive replay runs with Python's built-in HTTP server:

```bash
git clone https://github.com/TianyuCodings/NanoJev.git
cd NanoJev
python3 -m http.server 8080 --bind 127.0.0.1 --directory web
```

Open **http://127.0.0.1:8080/arcade.html** to play the maze and Snake recordings, inspect decisions, and step through the actual trajectories. The earlier benchmark viewer remains at **http://127.0.0.1:8080/comparison.html**.

## Download the showcase models

| Use | Checkpoint in [C-Tianyu/NanoJev](https://huggingface.co/C-Tianyu/NanoJev/tree/main/variants) |
|---|---|
| **50×50 maze demo** | `variants/local_atomic_seed17` |
| **Snake demo** | `variants/games_gold_seed17` |
| Full-map comparison | `variants/games_api_seed17` |
| Calibrated-decision experiments | `variants/events_ce_seed17`, `variants/events_brier_seed17`, `variants/events_paired_seed17` |

```python
from pathlib import Path
from huggingface_hub import snapshot_download

variant = "local_atomic_seed17"  # Select "games_gold_seed17" for Snake.
snapshot = snapshot_download(
    repo_id="C-Tianyu/NanoJev",
    allow_patterns=[f"variants/{variant}/*"],
)
checkpoint_dir = Path(snapshot) / "variants" / variant
```

The [game data package](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data/tree/main/games_v4) contains the matching training splits, frozen evaluation inputs, and all six Snake controller recordings. [Download, verify, and reproduce the games](docs/GAME_RELEASE.md).

## Download and run the model

The [model](https://huggingface.co/C-Tianyu/NanoJev) and [dataset](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data) are public. Prepare a CUDA environment with the recorded [Python dependencies](requirements-toy.txt):

```bash
python -m pip install -r requirements-toy.txt
```

Download the base release checkpoint and dataset. The root checkpoint is the initialization model and the earlier navigation baseline:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="C-Tianyu/NanoJev", local_dir="checkpoints/NanoJev",
    allow_patterns=["best.safetensors", "config.json", "tokenizer/*", "backbone_config/*"],
)
snapshot_download(
    repo_id="C-Tianyu/NanoJev-Data", repo_type="dataset", local_dir="data/NanoJev",
)
```

Start the persistent service:

```bash
python scripts/serve_decisions.py \
  --checkpoint-dir checkpoints/NanoJev \
  --web-root web --port 8765
```

Open **http://127.0.0.1:8765**. The service loads the model once and accepts repeated batches through **`POST /api/evaluate`**.

The [pipeline runbook](research/pipeline_runbook.md) covers data generation, training, evaluation, checkpoint creation, and continuing from the downloaded model and data.

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

The service exposes the same `GET /api/health` and `POST /api/evaluate` contract as the PyTorch service. It performs one non-autoregressive backbone pass, returns complete candidate distributions, and supports Boolean, Choice, and Score questions. The current Apple Silicon path stores the Qwen3 backbone in float16 and the small decision head in float32. It is compatible with an oMLX-served backbone, but the decision head must still be called by `scripts/serve_decisions_mlx.py`; ordinary oMLX `/v1/chat/completions` is not the NanoJev decision API.

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

- [x] **Scale up data** — Add larger mazes, Snake, atomic questions, and observed-event datasets.
- [x] **Calibrated reward prototype** — Implement and test paired proper-reward learning with CE/Brier controls.
- [ ] **RLCD expansion** — Add broader semantic tasks, stochastic long-horizon events, and additional model seeds.
- [ ] **Structured input support** — Version the encoder for structured instructions, criteria, and the native Noul interface.
