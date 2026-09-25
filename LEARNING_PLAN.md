# MiniMind 学习计划

> 这份计划基于当前仓库代码（`minimind-3` 主线）整理，目标是从「能跑起来」逐步走到「每一行都看得懂，并能自己改」。
> 整个仓库的核心 Python 代码约 4700 行，其中最关键的模型文件 `model/model_minimind.py` 只有 292 行，完全可以逐行精读。

---

## 0. 总览

### 0.1 仓库地图

| 目录/文件 | 行数 | 作用 | 学习优先级 |
|---|---|---|---|
| `model/model_minimind.py` | 292 | 模型结构：Config、RMSNorm、RoPE/YaRN、GQA Attention、SwiGLU FFN、MoE、KV Cache、generate | ★★★★★ |
| `dataset/lm_dataset.py` | 260 | 5 种数据集：Pretrain / SFT / DPO / RLAIF / AgentRL，含 loss mask 构造 | ★★★★★ |
| `trainer/trainer_utils.py` | 209 | 学习率调度、DDP 初始化、断点续训、模型初始化、SkipBatchSampler、奖励模型封装 | ★★★★ |
| `trainer/train_pretrain.py` | 163 | 预训练主循环（混合精度、梯度累积、梯度裁剪、保存） | ★★★★★ |
| `trainer/train_full_sft.py` | 164 | 全参数指令微调 | ★★★★★ |
| `model/model_lora.py` + `trainer/train_lora.py` | 65 + 176 | 手写 LoRA 及其训练 | ★★★★ |
| `trainer/train_distillation.py` | 238 | 知识蒸馏（CE + KL） | ★★★ |
| `trainer/train_dpo.py` | 218 | DPO 偏好优化 | ★★★★ |
| `trainer/train_ppo.py` | 447 | PPO（Actor + Critic + GAE + Reward Model） | ★★★★ |
| `trainer/train_grpo.py` | 326 | GRPO / CISPO | ★★★★ |
| `trainer/rollout_engine.py` | 224 | RL 采样引擎（Torch / SGLang） | ★★★ |
| `trainer/train_agent.py` | 485 | 多轮 Tool-Use 的 Agentic RL | ★★★ |
| `trainer/train_tokenizer.py` | 189 | BPE 分词器训练 | ★★ |
| `eval_llm.py` | 96 | 命令行对话测试 | ★★★ |
| `scripts/convert_model.py` | 144 | torch 权重 ↔ transformers 格式、合并 LoRA | ★★ |
| `scripts/serve_openai_api.py` / `web_demo.py` / `eval_toolcall.py` | — | OpenAI 兼容服务、Streamlit WebUI、工具调用测评 | ★★ |

### 0.2 训练流水线（全局视角）

```
train_tokenizer (可选，不建议重训)
        │
        ▼
train_pretrain  ──►  out/pretrain_768.pth          学「接下一个词」
        │
        ▼
train_full_sft  ──►  out/full_sft_768.pth          学「对话模板 / 思考标签 / 工具调用」
        │
        ├──► train_lora          领域适配（只训低秩矩阵）
        ├──► train_distillation  大模型教小模型
        ├──► train_dpo           偏好对齐（离线，无需采样）
        └──► train_ppo / train_grpo / train_agent   在线 RL（需要 rollout + 奖励）
        │
        ▼
eval_llm / scripts/*   推理、部署、转换
```

### 0.3 前置知识自检

开始前，确认你对以下内容至少「知道是什么」，不熟的在第 1 周补：

- Python：`argparse`、类与继承、列表推导、`__getitem__`
- PyTorch：`nn.Module`、`nn.Linear`、`nn.Embedding`、张量形状操作（`view` / `transpose` / `reshape` / `gather` / `scatter`）、`autograd`、`DataLoader`
- 数学：矩阵乘法、softmax、交叉熵、KL 散度、复数/旋转矩阵的基本概念
- 深度学习：Transformer（推荐先读 *Attention Is All You Need* 和 Jay Alammar 的 *The Illustrated Transformer*）

### 0.4 硬件与节奏建议

- **有单张 ≥12GB 的 GPU**：按计划完整复现 mini 数据（`pretrain_t2t_mini.jsonl` + `sft_t2t_mini.jsonl`），README 称 3090 上 SFT 1 epoch 约 2 小时。
- **只有 CPU / 小显存**：把模型缩小（例如 `--hidden_size 256 --num_hidden_layers 2`），数据截取前几千行（`head -n 5000 xxx.jsonl > tiny.jsonl`），目标是「看懂 loss 在下降」，而不是得到好模型。
- 默认节奏：**8 周，每周 8–10 小时**。每个阶段都有「阅读 → 动手 → 检验」三部分，检验题答不出就不要进入下一阶段。

---

## 阶段一（第 1 周）：环境搭建 & 先跑起来

**目标**：建立直觉 —— 先看到一个训练好的小模型能做什么，再去拆它。

### 阅读
- `README.md` 的「项目介绍」「快速开始」「数据介绍」「模型」四节。
- `requirements.txt`：注意 `transformers==4.57.6`、`torch` 被注释（需自行按 CUDA 版本安装）。

### 动手
1. 安装依赖：`pip install -r requirements.txt`，再安装匹配的 `torch`。
2. 下载官方权重（README 中 ModelScope / HuggingFace 链接，`minimind-3`），用 `python eval_llm.py --load_from ./minimind-3` 对话。
3. 试 `--open_thinking 1`、调 `--temperature` / `--top_p`，观察输出变化。
4. （可选）`streamlit run scripts/web_demo.py` 体验 WebUI 和 Tool Call。
5. 下载 `pretrain_t2t_mini.jsonl`、`sft_t2t_mini.jsonl` 到 `./dataset/`，用 `head -n 3` 看数据长什么样。

### 检验
- [ ] 说出 minimind-3 的参数量、层数、隐藏维度、词表大小（64M / 8 / 768 / 6400）。
- [ ] 解释为什么词表只用 6400（提示：embedding 和 lm_head 参数量 = vocab × hidden）。
- [ ] 手算：`tokenizer("白日依山尽")` 大约几个 token？用代码验证。

---

## 阶段二（第 2–3 周）：精读模型结构 `model/model_minimind.py` ⭐ 核心

**目标**：逐行读懂 292 行模型代码，能在纸上画出张量形状流。建议边读边在 Jupyter 里构造小张量逐函数调试。

### 2.1 配置 `MiniMindConfig`（L10–46）
- 默认值：`hidden_size=768`、`num_attention_heads=8`、`num_key_value_heads=4`、`head_dim=96`、`rope_theta=1e6`、`tie_word_embeddings=True`。
- `intermediate_size = ceil(hidden*π/64)*64` → 手算得 **2432**。思考：为什么是 π 倍而不是经典的 4 倍？（SwiGLU 有 3 个矩阵，约 8/3 倍可保持参数量相当；π≈3.14 是作者的取舍，并对齐到 64 的倍数方便硬件。）
- MoE 相关：`num_experts=4`、`num_experts_per_tok=1`、`router_aux_loss_coef=5e-4`。

### 2.2 RMSNorm（L51–62）
- 对比 LayerNorm：去掉了减均值和 bias，只做 `x / sqrt(mean(x²)+eps) * weight`。
- 注意 `.float()` 再 `.type_as(x)`：为什么归一化要在 fp32 下做？

### 2.3 RoPE 与 YaRN（L64–85）
- `precompute_freqs_cis`：频率 `1/θ^(2i/d)`，对每个位置 t 求 `cos(tθ_i)`、`sin(tθ_i)`。
- `apply_rotary_pos_emb` + `rotate_half`：这是「半维配对」的实现方式（与原论文相邻维度配对不同，但等价）。
- YaRN 部分：`ramp` 在高频维度不缩放、低频维度按 `factor` 缩放，中间线性过渡。
- 📖 配套阅读：RoFormer 论文、苏剑林《Transformer 升级之路》系列、YaRN 论文。
- 🧪 实验：验证 RoPE 的相对位置性质 —— 对同一对 q、k，平移相同位置后点积不变。

### 2.4 Attention（L93–133）—— 本文件最重要的部分
逐行标注形状（以 `bsz=2, seq=10` 为例）：
- `q_proj: 768 → 8×96`，`k_proj/v_proj: 768 → 4×96` → **GQA**（分组查询注意力），KV 头数是 Q 的一半。
- `q_norm / k_norm`：对每个头做 RMSNorm（**QK-Norm**，Qwen3 的做法，稳定训练）。
- KV Cache：`past_key_value` 在 `dim=1`（序列维）拼接。
- `repeat_kv`：把 4 个 KV 头复制成 8 个以匹配 Q。
- 两条路径：`F.scaled_dot_product_attention`（Flash）与手写 softmax 路径。思考：什么条件下会走手写路径？（有 KV cache 做增量解码 / 有 padding mask 时。）
- 手写路径的 causal mask：`triu(1)` 填 `-inf`，并且只作用在最后 `seq_len` 列（为什么？因为前面是 cache）。

### 2.5 FeedForward & MoE（L135–180）
- SwiGLU：`down(silu(gate(x)) * up(x))`。
- `MOEFeedForward`：
  - 路由：`softmax(gate(x))` → `topk`。
  - top-1 时的技巧：`top1 - top1.detach() + 1.0` —— 前向值恒为 1，梯度通过 straight-through 传回路由器。想清楚为什么这样做。
  - `y[0,0] += 0 * sum(p.sum() ...)`：给未被选中的专家制造「零梯度」，防止 DDP 报 unused parameters。
  - 负载均衡损失 `aux_loss = Σ(load_i × mean_score_i) × E × coef`（Switch Transformer 风格）。

### 2.6 Block / Model / CausalLM（L182–292）
- Pre-Norm 残差：`h = h + attn(norm(h))`，`h = h + mlp(norm(h))`。
- `MiniMindModel.forward`：`start_pos` 从 cache 长度推出，用来切 RoPE 表。
- `MiniMindForCausalLM`：
  - 权重共享：`embed_tokens.weight = lm_head.weight`。
  - loss：`logits[..., :-1]` 对 `labels[..., 1:]`，即 **模型内部做 shift**，`ignore_index=-100`。
  - `logits_to_keep`：推理时只算最后一个位置的 logits 省显存。
- `generate`：手写的采样循环 —— temperature → repetition penalty → top-k → top-p → multinomial / argmax，带 KV cache 与 `finished` 掩码。

### 动手练习
1. 实例化 `MiniMindForCausalLM(MiniMindConfig())`，打印每个子模块参数量，验证总量约 64M，并算出 embedding 占比。
2. 设 `use_moe=True`，用 `trainer_utils.get_model_params` 验证 `198M-A64M` 的含义。
3. 关掉 `flash_attn`，对比两条 attention 路径输出是否一致（`torch.allclose`）。
4. 验证 KV Cache 正确性：一次性 forward 整个序列 vs 逐 token 增量 forward，最后一个位置 logits 应一致。
5. **从零重写**：不看源码，自己写一个 `Attention` 类（含 GQA + RoPE + causal mask），与源码对比。

### 检验
- [ ] 画出一个 Block 的完整数据流图并标注每一步张量形状。
- [ ] 说明 GQA 相比 MHA 节省了什么（KV Cache 显存减半）。
- [ ] 解释 `tie_word_embeddings` 为什么对小模型尤其重要。
- [ ] 解释 top-p 实现里 `mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0` 这行的作用。

---

## 阶段三（第 4 周）：数据与预训练

**目标**：读懂「数据 → token → loss」的全过程，并亲手训练一个 pretrain 模型。

### 3.1 Tokenizer（`model/tokenizer.json`、`tokenizer_config.json`、`trainer/train_tokenizer.py`）
- 读 `train_tokenizer.py` 了解 ByteLevel BPE 的训练流程和特殊 token（`<|im_start|>`、`<|im_end|>`、`<think>`、`<tool_call>` 等）。
- 在 `tokenizer_config.json` 中找到 `chat_template`（Jinja），理解 `apply_chat_template` 如何把 `conversations` 拼成字符串，以及 `open_thinking`、`tools` 参数如何影响输出。
- README 说明「不建议重训 tokenizer」，理解原因即可，训练只做可选练习。

### 3.2 `PretrainDataset`（`dataset/lm_dataset.py` L40–58）
- `[bos] + tokens + [eos]` 再 pad 到 `max_length`，pad 位置 label = -100。
- 注意：这里 `labels == input_ids`（未 shift），shift 发生在模型 forward 里。

### 3.3 训练基础设施（`trainer/trainer_utils.py`）
- `get_lr`：余弦退火，从 `lr` 降到 `0.1·lr`（`0.1 + 0.45(1+cos)`），**没有 warmup**。
- `init_distributed_mode`：通过 `RANK` 环境变量判断是否 torchrun 启动。
- `lm_checkpoint`：保存 fp16 权重 + 单独的 resume 文件（含 optimizer / scaler / step / world_size），`os.replace` 原子写入；GPU 数变化时自动换算 step。
- `SkipBatchSampler`：续训时跳过已训练的 batch。
- `init_model`：权重命名约定 `out/{weight}_{hidden}{_moe}.pth`。

### 3.4 `trainer/train_pretrain.py`
按注释的 9 个步骤读：初始化 → 配置 → 混合精度（`autocast` + `GradScaler`，bf16 时 scaler 实际关闭）→ 日志（SwanLab）→ 模型/数据/优化器（AdamW）→ 续训 → `torch.compile` / DDP → 训练循环 → 清理。
- 梯度累积：`loss / accumulation_steps`，每 N 步才 `step()`。
- `loss = res.loss + res.aux_loss`：MoE 的均衡损失直接加进来。

### 动手
```bash
cd trainer
# 小规模验证（CPU 或小卡）
python train_pretrain.py --hidden_size 256 --num_hidden_layers 2 --batch_size 8 \
  --data_path ../dataset/tiny_pretrain.jsonl --epochs 1 --log_interval 10
# 正式（单卡）
python train_pretrain.py
# 多卡
torchrun --nproc_per_node 2 train_pretrain.py
```
- 训完后 `python eval_llm.py --weight pretrain`：预训练模型只会「续写」，不会「对话」—— 亲眼确认这一点。
- 打开 SFTDataset 里被注释掉的「调试打印」代码，对 Pretrain 数据写一个类似的打印，看 X→Y 对齐。

### 检验
- [ ] 为什么 pad 位置 label 要设为 -100？
- [ ] `accumulation_steps=8, batch_size=32` 的等效 batch 是多少？DDP 2 卡呢？
- [ ] 中断训练后用 `--from_resume 1` 恢复，验证 loss 曲线连续。
- [ ] 对照 `images/pretrain_loss.jpg`，你的 loss 大致在什么量级收敛？

---

## 阶段四（第 5 周）：SFT、LoRA、蒸馏

**目标**：理解「只对 assistant 回复计算 loss」这一 SFT 核心思想，以及参数高效微调。

### 4.1 `SFTDataset`（L61–122）
- `pre_processing_chat`：20% 概率随机插入 system prompt（数据增强）；工具数据不动。
- `post_processing_chat`：80% 概率删掉空的 `<think>\n\n</think>`，让模型同时学会「思考」和「不思考」两种模式（自适应思考的数据基础）。
- `generate_labels`：扫描 `<|im_start|>assistant\n` 到 `<|im_end|>\n` 之间的 token 才保留 label，其余全部 -100。
- 🧪 **强烈建议**：取消 L117–121 的调试打印注释，跑一条样本，逐 token 看哪些位置在算 loss。

### 4.2 `train_full_sft.py`
- 与 pretrain 几乎相同，区别：`from_weight='pretrain'`、学习率 `1e-5`（小 50 倍）、`max_seq_len=768`、数据集换为 `SFTDataset`。
- 训练后 `python eval_llm.py --weight full_sft`，对比 pretrain 模型的行为差异。

### 4.3 LoRA（`model/model_lora.py` + `trainer/train_lora.py`）
- `LoRA`：`B(A(x))`，A 高斯初始化、**B 零初始化**（保证训练开始时等价于原模型）。
- `apply_lora`：只给 `in_features == out_features` 的 Linear 挂 LoRA。🧪 练习：算一下在默认配置下是哪些层？（`q_proj` 768→768 和 `o_proj` 768→768；`k/v_proj` 768→384、MLP 都不是方阵，所以不挂。）
- 通过猴子补丁替换 `module.forward`，注意默认参数绑定 `layer1=original_forward` 防止闭包晚绑定 bug。
- `merge_lora`：`W' = W + B·A`，推理时零额外开销。
- 🧪 练习：修改 `apply_lora` 让它支持所有 Linear（需要处理 `in != out`，其实 LoRA 本身就支持），对比效果和可训练参数量。
- 用 `scripts/convert_model.py` 的 `convert_merge_base_lora` 合并权重。

### 4.4 知识蒸馏（`train_distillation.py`）
- `distillation_loss`：`KL(softmax(t/T) || softmax(s/T)) × T²`，为什么要乘 T²？
- 总 loss = `α·CE + (1-α)·KD`，且只在 assistant 位置（`loss_mask`）计算。
- 了解「黑盒蒸馏（用大模型生成的数据做 SFT）」与「白盒蒸馏（对齐 logits）」的区别 —— README 中 SFT 数据本身就含大量 Qwen3 蒸馏数据。

### 检验
- [ ] 解释为什么 SFT 不对 user 部分算 loss。
- [ ] LoRA rank=16 时，挂在一个 768×768 层上新增多少参数？占原层比例？
- [ ] 蒸馏时 teacher 与 student 的词表必须一致吗？为什么？

---

## 阶段五（第 6 周）：偏好对齐 —— DPO

**目标**：从一个具体的、无需采样的算法入手理解「对齐」。

### 阅读
- README「Ⅳ 强化学习」开篇与「👀 PO 算法的统一视角」：**所有 PO 算法 = 策略项 Φ(r, A) − 正则项 h(KL)**。后面所有 RL 代码都用这个框架去对照。
- `DPODataset`：chosen / rejected 各自构造 `x, y, mask`（这里在 Dataset 中就做了 shift）。
- `train_dpo.py`：
  - `logits_to_log_probs`：`log_softmax` 后 `gather` 取目标 token 概率。
  - `dpo_loss`：`-logsigmoid(β · [(logπ_c − logπ_r) − (logref_c − logref_r)])`。
  - batch 内前一半是 chosen、后一半是 rejected 的拼接技巧。
  - `ref_model` 冻结，只做前向。
- 📖 论文：*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*。

### 动手
- 基于 `full_sft` 跑 DPO；观察 `dpo_loss` 从 `ln2≈0.693` 开始下降的现象，思考为什么初始值是 ln2。
- 调 `--beta`（默认 0.15），观察模型偏离 SFT 的程度。

### 检验
- [ ] 从 RLHF 的 KL 约束目标推导出 DPO loss（纸笔完成）。
- [ ] DPO 为什么不需要奖励模型和采样？代价是什么（离线数据分布偏移）？

---

## 阶段六（第 7 周）：在线强化学习 —— PPO / GRPO / CISPO

**目标**：理解 rollout → reward → advantage → policy update 的完整循环。这是本仓库最难的部分，建议预留充足时间。

### 6.1 采样引擎 `trainer/rollout_engine.py`
- `RolloutResult` 数据结构：`output_ids / completion_ids / per_token_logps / completion_mask / prompt_lens`。
- `compute_per_token_logps`：如何只取 completion 部分的 logp。
- `TorchRolloutEngine`（直接调用 `model.generate`）与 `SGLangRolloutEngine`（HTTP 调外部推理服务，`update_policy` 时同步权重）的区别 —— 训推分离的最小实现。

### 6.2 奖励设计（`calculate_rewards`，PPO/GRPO 中）
- 外部奖励模型（默认 `internlm2-1_8b-reward`，`LMForRewardModel` 封装，分数裁剪到 [-3, 3]）。
- 规则奖励：长度区间、`<think>` 格式、`rep_penalty`（n-gram 重复惩罚）。
- 思考：规则奖励如何防止 reward hacking？

### 6.3 PPO（`train_ppo.py`）
- `CriticModel`：复用 MiniMind 主干，把 `lm_head` 换成输出标量的 value head。
- 奖励只加在最后一个 token 上，中间 token 奖励为 0。
- **GAE** 反向递推（`gamma`、`lam`），`returns = advantages + values`，再做 masked 标准化。
- 多轮 `ppo_update_iters` + minibatch；`approx_kl` 超过 `early_stop_kl` 提前停止（并用 `all_reduce` 同步防止 DDP 死锁）。
- 裁剪的 policy loss + 裁剪的 value loss + KL(ref) 惩罚。
- 📖 论文：PPO（Schulman 2017）、*Secrets of RLHF in LLMs Part I*。

### 6.4 GRPO / CISPO（`train_grpo.py`）
- 每个 prompt 采样 `num_generations=6` 个回答，组内 `(r − mean) / std` 作为优势 → **不需要 Critic**。
- KL 用 k3 估计：`exp(ref−π) − (ref−π) − 1`（恒非负、无偏）。
- `loss_type=grpo`：PPO 式 `min(ratio·A, clip(ratio)·A)`。
- `loss_type=cispo`（默认）：`clamp(ratio, max=ε_high).detach() · A · logπ` —— 裁剪的是重要性权重而不是梯度，所有 token 都保留梯度。
- 📖 论文：DeepSeekMath（GRPO）、MiniMax-M1（CISPO）。

### 动手
- 在 README 中找到 PPO / GRPO / CISPO 各自对应 `Φ`、`A`、`h(KL)` 的具体形式，填一张对照表。
- 用 `--debug_mode` 观察每步生成的样本与奖励。
- 对比 `images/ppo_loss.jpg` 与 `images/grpo_loss.jpg`，理解 reward、KL、回答长度曲线的走势。

### 检验
- [ ] PPO 需要同时加载几个模型？（actor、critic、ref、reward = 4 个）GRPO 呢？
- [ ] 如果一组 6 个回答奖励完全相同，GRPO 的优势是多少？这组样本还有梯度吗？
- [ ] GRPO 与 CISPO 的核心差别用一句话说清楚。

---

## 阶段七（第 8 周）：Tool Use、Agentic RL、部署

### 7.1 Tool Calling 数据与模板
- SFT 数据中 `tools`（system）、`tool_calls`（assistant）、`role=tool` 的格式。
- `scripts/eval_toolcall.py`：本地/API 两种后端，`parse_tool_calls` 解析 `<tool_call>...</tool_call>`。
- `trainer_utils.safe_math_eval`：用 AST 白名单代替 `eval` 的安全求值，值得作为安全编程范例阅读。

### 7.2 Agentic RL（`train_agent.py`）
- `rollout_single`：多轮循环 —— 模型生成 → 解析工具调用 → 执行工具 → 把 observation 拼回上下文 → 继续生成，最多 `max_turns` 轮。
- **关键细节**：observation（工具返回）部分的 `response_mask=0`、`old_logp=0`，即**只对模型自己生成的 token 算策略梯度**。用 marker 字符串切分模板以精确得到 observation 的 token 增量。
- 奖励：答案是否包含 `gt`（可验证奖励，RLVR）+ 格式 + 可选奖励模型。
- 了解 SGLang 训推分离的启动方式（README 7.4 节）。

### 7.3 部署与生态
- `scripts/convert_model.py`：torch `.pth` → transformers 目录（之后可用 `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` 加载）。
- `scripts/serve_openai_api.py`：FastAPI 实现 OpenAI 兼容接口，支持流式、`reasoning_content`、`tool_calls`。
- 用 `scripts/chat_api.py` 调自己起的服务；（可选）按 README 用 ollama / vllm / llama.cpp 部署。

### 检验
- [ ] 为什么 observation token 不能参与策略梯度？参与了会怎样？
- [ ] 把自己训的模型转成 transformers 格式并通过 OpenAI SDK 调用成功。

---

## 阶段八（持续）：综合项目 & 进阶方向

选 1–2 个做成完整小项目，这是检验是否真正掌握的最好方式：

1. **结构消融实验**：在同样数据/步数下对比 ① 去掉 QK-Norm ② MHA vs GQA ③ 不共享 embedding ④ 深而窄 vs 宽而浅（参考 README「模型配置」与 MobileLLM），画 loss 曲线写结论。
2. **加 warmup**：给 `get_lr` 加线性 warmup，对比训练初期稳定性。
3. **MoE 改进**：加入 shared expert（DeepSeek-MoE 风格）或 top-2 路由，观察 `aux_loss` 与专家负载分布。
4. **长度外推**：开启 `--inference_rope_scaling`（YaRN），测不同长度下 PPL，对照 `images/rope_ppl.png`。
5. **领域 LoRA**：自己构造一个小领域数据集（如自我认知、某专业问答），训练 LoRA 并合并部署。
6. **自定义奖励**：为 GRPO 设计一个可验证奖励（如数学题答案校验），观察训练曲线。
7. **测评**：按 README 在 C-Eval / CMMLU / OpenBookQA 上评测不同阶段的模型。
8. **扩展阅读**：作者的衍生项目 MiniMind-V（视觉）、MiniMind-O（多模态）、dLM（扩散语言模型）、Linear Attention（见 GitHub Discussions）。

---

## 附录 A：推荐阅读顺序（按代码行跳读）

1. `model/model_minimind.py` 全文
2. `dataset/lm_dataset.py` L40–122（Pretrain + SFT）
3. `trainer/trainer_utils.py` L20–160
4. `trainer/train_pretrain.py` 全文 → `train_full_sft.py`（只看 diff）
5. `model/model_lora.py` → `trainer/train_lora.py`
6. `trainer/train_distillation.py` L25–92
7. `dataset/lm_dataset.py` L125–196 → `trainer/train_dpo.py` L25–90
8. `trainer/rollout_engine.py` → `trainer/train_grpo.py` L31–150
9. `trainer/train_ppo.py` L36–280
10. `dataset/lm_dataset.py` L199–256 → `trainer/train_agent.py`
11. `eval_llm.py` → `scripts/*`

## 附录 B：论文清单

| 主题 | 论文 |
|---|---|
| Transformer | Attention Is All You Need (2017) |
| RMSNorm | Root Mean Square Layer Normalization (2019) |
| SwiGLU | GLU Variants Improve Transformer (2020) |
| RoPE | RoFormer (2021) |
| YaRN | YaRN: Efficient Context Window Extension (2023) |
| GQA | GQA: Training Generalized Multi-Query Transformer (2023) |
| MoE | Switch Transformers (2021)、DeepSeekMoE (2024) |
| 小模型结构 | MobileLLM (2024) |
| LoRA | LoRA: Low-Rank Adaptation (2021) |
| 蒸馏 | Distilling the Knowledge in a Neural Network (2015) |
| RLHF / PPO | InstructGPT (2022)、PPO (2017) |
| DPO | Direct Preference Optimization (2023) |
| GRPO | DeepSeekMath (2024) |
| CISPO | MiniMax-M1 (2025) |
| 参考架构 | Qwen3 Technical Report |

## 附录 C：每周自查表

| 周 | 阶段 | 产出物 |
|---|---|---|
| 1 | 环境 & 体验 | 能用官方权重对话；数据已下载 |
| 2–3 | 模型结构 | 形状流程图；自写 Attention；KV Cache 一致性验证 |
| 4 | 预训练 | 自己的 `pretrain_*.pth` 与 loss 曲线 |
| 5 | SFT / LoRA / 蒸馏 | 能对话的 `full_sft_*.pth`；一个 LoRA 权重 |
| 6 | DPO | DPO 推导笔记；DPO 模型 |
| 7 | PPO / GRPO | PO 算法对照表；一次 GRPO 训练记录 |
| 8 | Agent & 部署 | 多轮工具调用 demo；OpenAI 兼容服务 |
| 之后 | 综合项目 | 一份消融实验报告 |
