# NanoJev — A nano replica of [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

**简体中文** | [English](README.md)

**一个 0.6B 并行决策模型：输入状态与问题，直接得到完整概率分布，无需生成答案 token。**

[模型](https://huggingface.co/C-Tianyu/NanoJev) · [数据集](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data)

**[打开在线三栏对照演示 →](https://nanojev.tianyuchen99.chatgpt.site)**

## 三个模型，同一场游戏

[![Jev、NanoJev 与原始 Qwen 并排探索迷宫](assets/side_by_side_maze.png)](https://nanojev.tianyuchen99.chatgpt.site/#maze)

[下载迷宫视频（MP4）](assets/side_by_side_maze.mp4) · 27 秒 · 1440 × 1120 · 30 fps

[打开贪吃蛇](https://nanojev.tianyuchen99.chatgpt.site/#snake) · [探索 50×50 迷宫](https://nanojev.tianyuchen99.chatgpt.site/#maze) · [真实来源与回放核验](assets/side_by_side_data_manifest.json)

独立的 ChatGPT Sites 网站以浅色三栏展示 **Jev、NanoJev 和原始 Qwen**。三个画面按同一环境步推进，已经结束的对局停留在真实终局。概率条展示产生当前画面的最后一次决策；各系统均包含共同的代码规划部分。

新版迷宫对照使用真实的原始 Qwen3-0.6B：**4,726 次尝试、2,044 次碰撞后到达目标**。下方旧迷宫视频继续保留原来的**起始 NanoJev** 对照与实测数字。

## 真实对局实录

看模型判断与代码规划共同完成任务。每个游戏的三组系统都使用相同控制代码，回放保留实际动作、概率和完整终局。

### 找到出口：50×50 迷宫

[![NanoJev、Jev 与起始 NanoJev 探索同一张 50×50 迷宫](assets/arcade_maze.gif)](assets/arcade_maze.mp4)

[观看 MP4](assets/arcade_maze.mp4) · [交互回放](web/arcade.html)

模型判断四个局部方向是否可通行；代码记住碰撞、探索未知边，并沿已经走通过的路径重新定位。

| 系统 | 行动尝试 | 碰撞 | 结果 |
|---|---:|---:|---|
| **NanoJev** | **244** | **36** | **到达目标** |
| [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | 2,738 | 1,044 | 到达目标 |
| 起始 NanoJev | 171 | 43 | 到达目标 |

起始 NanoJev 是此前已经训练的 NanoJev checkpoint；新版模型进行了与局部输入对应的安全判断训练。

### 持续成长：12×12 贪吃蛇

[![NanoJev、Jev 与原始 Qwen 使用共同规划器玩贪吃蛇](assets/arcade_snake.gif)](assets/arcade_snake.mp4)

[观看 MP4](assets/arcade_snake.mp4) · [交互回放](web/arcade.html)

共同规划器先排除立即碰撞的动作，再寻找通向当前食物的静态路径。模型在剩余候选之间选择；仅剩一个候选时由代码直接执行。**种子：61005；控制方式：贪心选择。**

| 系统 | 吃到食物 | 步数 | 结果 |
|---|---:|---:|---|
| **NanoJev** | **27** | **256** | **达到上限时仍存活** |
| [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | 30 | 256 | 达到上限时仍存活 |
| 原始 Qwen3-0.6B | 25 | 211 | 陷入死局 |

原始 Qwen 使用未经本项目微调的预训练权重和原生语言模型输出头，概率条件限定为所提供的 A–D 候选 token。

[案例、模型身份与轨迹核验记录](assets/arcade_data_manifest.json)

## 核心能力

| 能力 | 已实现 |
|---|---|
| 多状态并行 | 同一批处理多个独立环境状态 |
| 多问题并行 | 每个状态同时回答多个问题 |
| 动态候选 | Choice 每题支持 2–255 个候选，共享决策头 |
| 多种决策类型 | Choice 候选分布、Boolean 概率、Score 的 2–10 级分布及期望 |
| 直接输出概率 | 一次前向得到完整候选分布，无输出 token 解码 |
| 轻量底座 | Qwen3-0.6B，支持单卡训练与部署 |
| 持久推理服务 | 模型加载一次，复用权重处理后续请求 |

实际运行已验证：**6 个状态 · 18 个问题 · 44 条候选路径 · 1 次 backbone 前向**。[并行调用记录](research/parallel_example_v3.json)

## 更大的游戏与校准决策

- **完整环境：** 8×8、16×16、32×32、50×50 迷宫，四类拓扑，每图多个位置，并支持配置更大尺寸。
- **局部判断与代码规划：** 统一 5×5 局部观察、四方向并行安全判断、移动记忆及模型引导探索。
- **贪吃蛇规则：** 可复现食物生成、身体增长、碰撞与尾部移动、动态动作候选和安全问题。
- **概率学习：** 观测事件数据、CE/Brier 训练、成对适当奖励学习、精确梯度检查及已完成的 Qwen3-0.6B 训练。
- **完整评测：** 按地图划分数据、固定游戏集合、真实模型执行与独立轨迹重放核验。

局部安全模型的测试题准确率为 **77.84%**，50×50 OOD 题为 **76.56%**。概率学习先导中，成对适当奖励组的分布误差为 **测试 0.11844 / OOD 0.06202**；该误差是模型分布与模拟器事件概率之间的差值平方和。

[原子判断与规划](docs/ATOMIC_PLANNING.md) · [大规模游戏流程](docs/SCALED_GAMES.md) · [RLCD 实现与结果](docs/RLCD_EXPERIMENT.md) · [输入契约](docs/TYPESAFE_CONTRACT.md) · [游戏结果](docs/DEVELOPMENT_RESULTS.md)

## 此前完整 40 图导航评测

**控制方式：T=1 概率采样。** 包含 20 张 4×4 测试地图和 20 张 6×6 OOD 地图。

| 模型 | 4×4 测试地图 | 6×6 OOD 地图 |
|---|---:|---:|
| **NanoJev** | **19/20 · 95%** | **18/20 · 90%** |
| [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | 20/20 · 100% | 19/20 · 95% |
| 原始 Qwen3-0.6B | 7/20 · 35% | 3/20 · 15% |

[此前的对照回放](web/comparison.html) · [完整评测结果](research/nanojev_comparison_public.json)

## 实现流程

每个决策由**状态、问题和候选集合**定义。模型编码候选路径，再由共享决策头输出每题的完整分布。Choice 使用共享标量头与集合注意力；Boolean 使用单路径 sigmoid；Score 返回等级分布和概率加权期望。

1. **构建问题：** 生成状态、问题、候选描述和目标分布。
2. **组织数据：** 相关地图、规则及其变体保留在同一分区。
3. **训练模型：** 初始化 Qwen3-0.6B，预热决策头，再使用完整问题的分布损失训练。
4. **执行评测：** 测量概率质量，运行游戏控制器并记录真实动作。
5. **服务与可视化：** 复用持久模型接口，在浏览器重放完整轨迹。

[完整训练与运行手册（English）](research/pipeline_runbook.md)

## 快速体验三栏对照

交互回放只需 Python：

```bash
git clone https://github.com/TianyuCodings/NanoJev.git
cd NanoJev
python3 -m http.server 8080 --bind 127.0.0.1 --directory web
```

打开 **http://127.0.0.1:8080/side-by-side.html**，并排播放贪吃蛇与迷宫的三方实录。深色游戏厅保留在 **http://127.0.0.1:8080/arcade.html**，此前的导航对照页面位于 **http://127.0.0.1:8080/comparison.html**。

## 下载演示使用的模型

| 用途 | [模型仓库](https://huggingface.co/C-Tianyu/NanoJev/tree/main/variants)中的检查点 |
|---|---|
| **50×50 迷宫演示** | `variants/local_atomic_seed17` |
| **Snake 演示** | `variants/games_gold_seed17` |
| 整图问题对照 | `variants/games_api_seed17` |
| 校准决策实验 | `variants/events_ce_seed17`、`variants/events_brier_seed17`、`variants/events_paired_seed17` |

```python
from pathlib import Path
from huggingface_hub import snapshot_download

variant = "local_atomic_seed17"  # Snake 使用 "games_gold_seed17"。
snapshot = snapshot_download(
    repo_id="C-Tianyu/NanoJev",
    allow_patterns=[f"variants/{variant}/*"],
)
checkpoint_dir = Path(snapshot) / "variants" / variant
```

[游戏数据包](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data/tree/main/games_v4)包含匹配的训练分区、固定评测输入和全部六组 Snake 控制器实录。[下载、校验与复现命令](docs/GAME_RELEASE.md)。

## 下载并运行模型

模型和数据集均可公开下载。`best.safetensors` 已包含 Qwen3-0.6B 底座**和**决策头，推理时**不会**再下载 `Qwen/Qwen3-0.6B`。Hugging Face Hub 只负责把这份 checkpoint 拉下来；运行还需要 **PyTorch**、**transformers** 和 **safetensors**。

在兼容 CUDA 的环境中安装 [Python 依赖](requirements-toy.txt)：

```bash
python -m pip install -r requirements-toy.txt
```

下载基础 checkpoint 和数据集。根目录权重对应此前的导航版本，也是后续训练的初始化模型：

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

启动持久服务：

```bash
python scripts/serve_decisions.py \
  --checkpoint-dir checkpoints/NanoJev \
  --web-root web --port 8765
```

打开 **http://127.0.0.1:8765**，或向 **`POST /api/evaluate`** 发送批量请求。模型只加载一次，后续请求复用权重。

### Apple Silicon（Mac）

`serve_decisions.py` 与 `predict_toy_decisions.py` 默认 `--device auto`：有 CUDA 用 CUDA，否则用 MPS，再否则 CPU。推理不需要 NVIDIA。macOS 请安装官方 Mac 版 PyTorch（不要直接套 `requirements-toy.txt` 里的 CUDA 钉死版本）：

```bash
python -m pip install torch transformers safetensors huggingface_hub
```

随后使用与上面相同的 `snapshot_download` 和 `serve_decisions.py`。MPS 默认 fp32；在 M3 Max 上复现迷宫局部安全测试准确率为 **77.84%**（176 题）。训练脚本仍按 CUDA 编写。

[完整手册](research/pipeline_runbook.md)包含数据生成、训练、评测、checkpoint 创建，以及从下载模型和数据继续运行的命令。

## 路线图

- [x] **扩展数据：** 大迷宫、贪吃蛇、原子问题与观测事件数据集。
- [x] **校准奖励原型：** 实现并验证成对适当奖励学习，提供 CE/Brier 对照。
- [ ] **扩展 RLCD：** 更多语义任务、随机长程事件与模型种子。
- [ ] **结构化输入：** 为结构化 instructions、criteria 和原生 Noul 接口建立新版编码器。
