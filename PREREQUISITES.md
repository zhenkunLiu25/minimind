# MiniMind 前置知识速查手册

> 配合 [`LEARNING_PLAN.md`](./LEARNING_PLAN.md) 第 0.3 节「前置知识自检」使用。
> 定位是**复习**：给你「学过但记不清了」的知识点配上最小可运行代码和解释，并标出它在 MiniMind 源码中出现的位置（📍）。
>
> - 除标注「源码摘录」的两处外，所有 Python 代码块都**各自独立、可直接运行**（只依赖 `torch`，CPU 即可，已在 torch 2.x 上逐一验证），建议复制到 Jupyter 中边改边跑。
> - `# →` 后面是运行时的实际输出或形状。
> - 每章末尾有「自测」，答案见文末附录。

## 目录

- [第一部分 Python](#第一部分-python)
  - 1.1 argparse · 1.2 类、继承与 `super()` · 1.3 `*args / **kwargs` · 1.4 推导式 · 1.5 魔术方法与生成器 · 1.6 闭包与晚绑定陷阱 · 1.7 上下文管理器与装饰器 · 1.8 `getattr / hasattr / setattr` · 1.9 其它常见小语法
- [第二部分 PyTorch](#第二部分-pytorch)
  - 2.1 Tensor 基础与广播 · 2.2 `nn.Module` · 2.3 `nn.Linear` 与 `nn.Embedding` · 2.4 形状操作全家桶 · 2.5 索引类操作 `gather / scatter / topk / index_add_` · 2.6 autograd · 2.7 损失函数 · 2.8 优化器与学习率 · 2.9 Dataset / DataLoader / Sampler · 2.10 混合精度 · 2.11 保存与加载 · 2.12 分布式 DDP 速览
- [第三部分 数学](#第三部分-数学)
  - 3.1 矩阵乘法与形状 · 3.2 softmax 与温度 · 3.3 交叉熵、困惑度 · 3.4 KL 散度及其估计 · 3.5 旋转矩阵与复数（RoPE 的数学） · 3.6 sigmoid / logsigmoid · 3.7 标准化与方差 · 3.8 低秩分解
- [第四部分 深度学习与 Transformer](#第四部分-深度学习与-transformer)
  - 4.1 语言建模与 shift · 4.2 分词 BPE · 4.3 缩放点积注意力 · 4.4 多头：MHA / MQA / GQA · 4.5 位置编码 · 4.6 FFN 与激活函数 · 4.7 归一化：LayerNorm vs RMSNorm、Pre-Norm vs Post-Norm · 4.8 残差连接 · 4.9 **100 行从零实现一个迷你 GPT 并训练** · 4.10 KV Cache · 4.11 解码采样策略 · 4.12 训练技巧 · 4.13 MoE 入门
- [附录 自测题答案](#附录-自测题答案)

---

# 第一部分 Python

## 1.1 argparse：命令行参数

📍 每个 `trainer/train_*.py` 的 `if __name__ == "__main__":` 部分。

```python
import argparse

parser = argparse.ArgumentParser(description="MiniMind Pretraining")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--learning_rate", type=float, default=5e-4)
parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])   # 用 int 0/1 表示布尔
parser.add_argument("--use_wandb", action="store_true")                 # 出现即为 True
parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl")

# 真正运行时是 parser.parse_args()，这里传列表模拟命令行
args = parser.parse_args(["--batch_size", "8", "--use_moe", "1", "--use_wandb"])
print(args.batch_size, args.learning_rate, bool(args.use_moe), args.use_wandb)
# → 8 0.0005 True True
```

要点：
- `--learning_rate` 在代码里用 `args.learning_rate` 访问（连字符 `-` 会被转成下划线 `_`）。
- **不要写 `type=bool`**：`bool("0")` 为 `True`（非空字符串都为真）。所以 MiniMind 用 `type=int, choices=[0,1]` 或 `action="store_true"`。
- `choices` 限制合法取值，传错会直接报错退出。

## 1.2 类、继承与 `super()`

📍 `MiniMindConfig(PretrainedConfig)`、`MiniMindForCausalLM(PreTrainedModel, GenerationMixin)`、`CriticModel(MiniMindForCausalLM)`（`trainer/train_ppo.py`）。

```python
class Base:
    def __init__(self, name):
        self.name = name
    def forward(self, x):
        return x * 2
    def __call__(self, x):          # 让实例可以像函数一样被调用：obj(x)
        return self.forward(x)

class Child(Base):
    def __init__(self, name, extra):
        super().__init__(name)      # 先让父类完成它的初始化
        self.extra = extra
    def forward(self, x):           # 重写（override）父类方法
        return super().forward(x) + self.extra

c = Child("c", 1)
print(c(10), c.name, isinstance(c, Base))
# → 21 c True
```

要点：
- `super().__init__(...)` 必须调用，否则父类里设置的属性不存在。在 `nn.Module` 子类里**忘了调用它会直接报错**（见 2.2）。
- `nn.Module` 正是靠 `__call__` 去调用 `forward`，所以我们写 `model(x)` 而不是 `model.forward(x)`（`__call__` 还会顺带触发 hook）。
- MiniMind 的 `CriticModel` 继承了整个语言模型，只额外加了一个 `value_head`，并重写 `forward` 输出标量价值 —— 这就是继承复用的典型场景。

## 1.3 `*args / **kwargs` 与 `dict.get`

📍 `MiniMindConfig.__init__(self, hidden_size=768, ..., **kwargs)` 大量使用 `kwargs.get("xxx", 默认值)`。

```python
def config(hidden_size=768, **kwargs):
    vocab = kwargs.get("vocab_size", 6400)      # 没传就用默认值
    heads = kwargs.get("num_attention_heads", 8)
    return hidden_size, vocab, heads, kwargs

print(config(512, vocab_size=1000, foo="bar"))
# → (512, 1000, 8, {'vocab_size': 1000, 'foo': 'bar'})

def f(*args, **kwargs):
    return args, kwargs
print(f(1, 2, a=3))            # → ((1, 2), {'a': 3})
params = {"a": 1, "b": 2}
print(f(**params))             # 解包字典作为关键字参数 → ((), {'a': 1, 'b': 2})
```

## 1.4 推导式

📍 保存权重时 `{k: v.half().cpu() for k, v in state_dict.items()}`；加载 LoRA 时 `{(k[7:] if k.startswith('module.') else k): v ...}`。

```python
state_dict = {"module.layer.weight": 1.0, "module.layer.bias": 2.0, "head.weight": 3.0}

# 字典推导 + 条件表达式：去掉 DDP 包装带来的 "module." 前缀（len("module.") == 7）
clean = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
print(clean)       # → {'layer.weight': 1.0, 'layer.bias': 2.0, 'head.weight': 3.0}

# 列表推导 + 过滤
print([k for k in clean if "weight" in k])     # → ['layer.weight', 'head.weight']

# 生成器表达式（不建列表，直接给 sum 用）
print(sum(x * x for x in range(4)))            # → 14
```

## 1.5 魔术方法与生成器

📍 所有 `Dataset` 都实现 `__len__` 和 `__getitem__`；`SkipBatchSampler.__iter__` 用 `yield` 产生 batch。

```python
class MyDataset:
    def __init__(self, data):
        self.data = data
    def __len__(self):               # len(ds)
        return len(self.data)
    def __getitem__(self, i):        # ds[i]
        return self.data[i] * 10

ds = MyDataset([1, 2, 3])
print(len(ds), ds[1])                # → 3 20


class SkipBatchSampler:              # 精简自 trainer/trainer_utils.py
    def __init__(self, indices, batch_size, skip_batches=0):
        self.indices, self.bs, self.skip = indices, batch_size, skip_batches
    def __iter__(self):              # 含 yield 的函数 = 生成器，惰性地逐个产出
        batch, skipped = [], 0
        for idx in self.indices:
            batch.append(idx)
            if len(batch) == self.bs:
                if skipped < self.skip:          # 断点续训：丢弃已训练过的 batch
                    skipped += 1; batch = []; continue
                yield batch
                batch = []
        if batch and skipped >= self.skip:
            yield batch

print(list(SkipBatchSampler(range(10), 3, skip_batches=1)))
# → [[3, 4, 5], [6, 7, 8], [9]]
```

## 1.6 闭包与「晚绑定」陷阱

📍 `model/model_lora.py` 的 `apply_lora`：`def forward_with_lora(x, layer1=original_forward, layer2=lora)`。为什么要用默认参数？

```python
# ❌ 错误写法：闭包捕获的是「变量」而不是「值」，循环结束后所有函数看到的都是最后一个 i
funcs = []
for i in range(3):
    funcs.append(lambda x: x + i)
print([f(0) for f in funcs])        # → [2, 2, 2]

# ✅ 正确写法：用默认参数在「定义时」把当前值固定下来
funcs = []
for i in range(3):
    funcs.append(lambda x, i=i: x + i)
print([f(0) for f in funcs])        # → [0, 1, 2]
```

`apply_lora` 在循环中给每个 Linear 层替换 forward，如果不用默认参数绑定，所有层都会调用**最后一层**的原始 forward 和 LoRA —— 这是个非常隐蔽的 bug。

## 1.7 上下文管理器与装饰器

📍 `with autocast_ctx:`、`with torch.no_grad():`、`nullcontext()`、`@torch.inference_mode()`、`@dataclass`。

```python
from contextlib import contextmanager, nullcontext
import time

@contextmanager
def timer(name):
    t = time.time()
    yield                                   # with 块中的代码在这里执行
    print(f"{name}: {time.time() - t:.4f}s")

with timer("sum"):
    s = sum(range(10**6))

# nullcontext：什么也不做的上下文，用来统一写法
device_type = "cpu"
ctx = nullcontext() if device_type == "cpu" else timer("gpu")
with ctx:
    pass                                    # CPU 时不开混合精度，代码却不用写两套


def log_call(fn):                           # 装饰器 = 接收函数、返回新函数的函数
    def wrapper(*args, **kwargs):
        print("calling", fn.__name__)
        return fn(*args, **kwargs)
    return wrapper

@log_call                                   # 等价于 add = log_call(add)
def add(a, b):
    return a + b
print(add(1, 2))
# → calling add
# → 3
```

## 1.8 `getattr / hasattr / setattr`

📍 `raw_model = getattr(model, '_orig_mod', model)`（取出 `torch.compile` 包装前的模型）；`setattr(module, "lora", lora)`；`hasattr(module, 'lora')`。

```python
class M: pass
m = M()
setattr(m, "lora", "LoRA-module")               # 等价于 m.lora = ...
print(hasattr(m, "lora"), getattr(m, "lora"))   # → True LoRA-module
print(getattr(m, "_orig_mod", m) is m)          # 属性不存在就返回默认值 → True
```

## 1.9 其它常见小语法

```python
# (1) and 短路：train_sampler 为 None 时不会调用 set_epoch（📍 train_pretrain.py）
train_sampler = None
train_sampler and train_sampler.set_epoch(0)    # 什么也不发生

# (2) 条件表达式
use_moe = True
suffix = "_moe" if use_moe else ""
print(f"pretrain_768{suffix}.pth")              # → pretrain_768_moe.pth

# (3) slice 对象：📍 MiniMindForCausalLM.forward 中的 logits_to_keep
seq = list(range(10))
logits_to_keep = 1
s = slice(-logits_to_keep, None)                # 等价于 seq[-1:]
print(seq[s], seq[slice(-0, None)])             # → [9] [0, 1, ..., 9]（-0 == 0，即保留全部）

# (4) 多重赋值 / 解包
a, (b, c) = 1, (2, 3)
xq, xk = "q", "k"

# (5) zip / enumerate(start=)
for step, (x, y) in enumerate(zip("ab", "cd"), start=1):
    print(step, x, y)                           # → 1 a c / 2 b d
```

### 自测 1
1. 为什么 `parser.add_argument("--flag", type=bool)` 是错误写法？
2. 下面代码输出什么？`fs=[lambda: i for i in range(3)]; print([f() for f in fs])`
3. `getattr(model, '_orig_mod', model)` 在模型没被 `torch.compile` 时返回什么？

---

# 第二部分 PyTorch

## 2.1 Tensor 基础与广播

```python
import torch

x = torch.randn(2, 3)                           # 标准正态分布
print(x.shape, x.dtype, x.device)               # → torch.Size([2, 3]) torch.float32 cpu
print(torch.zeros(2, 2).long().dtype)           # → torch.int64（token id 用 long）
print(x.to(torch.bfloat16).dtype)               # → torch.bfloat16

# .item()：单元素张量 → Python 数字（打印 loss 时用）
print(torch.tensor(3.14).item())                # → 3.14

# 广播（broadcasting）：从最后一维对齐，维度为 1 或缺失的自动扩展
a = torch.ones(4, 1, 3)
b = torch.ones(5, 1)                            # 视为 (1, 5, 1)
print((a + b).shape)                            # → torch.Size([4, 5, 3])

# 📍 RMSNorm 的 weight 形状是 (hidden,)，与 (bsz, seq, hidden) 的 x 相乘时就靠广播
w = torch.ones(768); h = torch.randn(2, 10, 768)
print((w * h).shape)                            # → torch.Size([2, 10, 768])

# keepdim=True：保留被规约的维度，方便后续广播
m = h.pow(2).mean(-1, keepdim=True)
print(m.shape)                                  # → torch.Size([2, 10, 1])
```

## 2.2 `nn.Module`

📍 MiniMind 里所有组件（`RMSNorm`、`Attention`、`FeedForward`、`MiniMindBlock`……）都是 `nn.Module`。

```python
import torch
from torch import nn

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()                                  # 必须！否则注册子模块时报错
        self.fc = nn.Linear(4, 4)                           # 子模块：自动注册，参数自动收集
        self.scale = nn.Parameter(torch.ones(4))            # 可训练参数
        self.register_buffer("const", torch.arange(4.), persistent=False)  # 非参数状态
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])  # 列表要用 ModuleList
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.drop(self.fc(x)) * self.scale + self.const
        for b in self.blocks:
            x = b(x)
        return x

m = Tiny()
print(m(torch.randn(2, 4)).shape)                          # → torch.Size([2, 4])
print(sum(p.numel() for p in m.parameters()))              # → 64 (20 + 4 + 20*2)
print([n for n, _ in m.named_parameters()])
# → ['scale', 'fc.weight', 'fc.bias', 'blocks.0.weight', 'blocks.0.bias', 'blocks.1.weight', 'blocks.1.bias']
print(list(m.state_dict().keys()))                         # persistent=False 的 buffer 不会出现
print([n for n, mod in m.named_modules()])                 # → ['', 'fc', 'blocks', 'blocks.0', 'blocks.1', 'drop']

m.train(); m.eval()          # 切换训练 / 推理模式：影响 Dropout、BatchNorm 等行为
for p in m.fc.parameters():
    p.requires_grad = False  # 冻结参数（LoRA 训练就是冻结原模型，只训 LoRA）
```

要点：
- **参数（Parameter）** 会被优化器更新；**buffer** 跟着模型移动设备（`.to(device)`），但不训练。
- 📍 `MiniMindModel` 把 RoPE 的 `freqs_cos / freqs_sin` 注册成 `persistent=False` 的 buffer：它们可以随时由公式重算，不必保存进权重文件。
- 普通 Python `list` 装的子模块**不会被注册**（参数不会被优化器看到，也不会 `.to(device)`），必须用 `nn.ModuleList`。
- `named_modules()` 递归遍历所有子模块 —— `apply_lora` 就是用它找出所有 `nn.Linear`。

## 2.3 `nn.Linear` 与 `nn.Embedding`

```python
import torch
from torch import nn

fc = nn.Linear(768, 384, bias=False)
print(fc.weight.shape)                   # → torch.Size([384, 768])  注意是 [out, in]！
x = torch.randn(2, 10, 768)
y = fc(x)                                # y = x @ W^T (+ b)，只作用在最后一维
print(y.shape)                           # → torch.Size([2, 10, 384])
print(torch.allclose(y, x @ fc.weight.T))   # → True

emb = nn.Embedding(6400, 768)            # 本质是一张 6400×768 的查找表
ids = torch.tensor([[1, 5, 2]])
print(emb(ids).shape)                    # → torch.Size([1, 3, 768])
print(torch.equal(emb(ids)[0, 1], emb.weight[5]))   # 就是按行取 → True

# 📍 权重共享 tie_word_embeddings：输入 embedding 与输出 lm_head 用同一个矩阵
lm_head = nn.Linear(768, 6400, bias=False)
emb.weight = lm_head.weight              # 两者形状都是 [6400, 768]
print(emb.weight is lm_head.weight)      # → True
print(f"共享可省 {6400*768/1e6:.2f}M 参数")  # → 4.92M，对 64M 的模型很可观
```

## 2.4 形状操作全家桶

📍 `Attention.forward` 几乎用到了下面所有操作。

```python
import torch

bsz, seq, n_heads, head_dim = 2, 5, 8, 96
x = torch.randn(bsz, seq, n_heads * head_dim)          # (2, 5, 768)

# view：不复制数据，只改变「看待方式」；要求内存连续
q = x.view(bsz, seq, n_heads, head_dim)                # (2, 5, 8, 96)

# transpose：交换两个维度（不复制，但之后内存不再连续）
qt = q.transpose(1, 2)                                 # (2, 8, 5, 96)  把 head 维移到前面，方便批量矩阵乘
print(qt.is_contiguous())                              # → False
# qt.view(bsz, seq, -1) 会报错！需要 reshape（必要时自动复制）或 .contiguous().view
out = qt.transpose(1, 2).reshape(bsz, seq, -1)         # (2, 5, 768)  📍 注意力输出合并多头
print(out.shape)

# permute：任意重排多个维度
print(q.permute(0, 2, 3, 1).shape)                     # → torch.Size([2, 8, 96, 5])

# unsqueeze / squeeze：增加 / 删除大小为 1 的维度
cos = torch.randn(seq, head_dim)
print(cos.unsqueeze(1).shape)                          # → (5, 1, 96)  📍 apply_rotary_pos_emb 中与 (bsz, seq, heads, dim) 广播
print(torch.randn(3, 1).squeeze(-1).shape)             # → (3,)

# expand vs repeat：expand 不复制内存（只能扩展大小为 1 的维度），repeat 真复制
kv = torch.randn(bsz, seq, 4, head_dim)                # 4 个 KV 头
n_rep = 2
kv_rep = kv[:, :, :, None, :].expand(bsz, seq, 4, n_rep, head_dim).reshape(bsz, seq, 4 * n_rep, head_dim)
print(kv_rep.shape)                                    # → (2, 5, 8, 96)  📍 repeat_kv
print(torch.equal(kv_rep[:, :, 0], kv_rep[:, :, 1]))   # 相邻两个头是同一份 KV → True

# repeat_interleave：每个元素连续重复 n 次（📍 GRPO 中一个 prompt 生成多个回答）
print(torch.tensor([1, 2]).repeat_interleave(3))       # → tensor([1, 1, 1, 2, 2, 2])
print(torch.tensor([1, 2]).repeat(3))                  # → tensor([1, 2, 1, 2, 1, 2])

# cat：沿已有维度拼接（📍 KV Cache 在序列维拼接）；stack：新建一个维度
past_k = torch.randn(bsz, 3, 4, head_dim)
print(torch.cat([past_k, kv], dim=1).shape)            # → (2, 8, 4, 96)
print(torch.stack([torch.ones(3), torch.zeros(3)]).shape)  # → (2, 3)

# 负索引切片：📍 logits[..., :-1, :] 与 labels[..., 1:]
logits = torch.randn(2, 5, 100)
print(logits[..., :-1, :].shape, logits[:, -1, :].shape)   # → (2, 4, 100) (2, 100)
```

一句话记忆：**`view` 要连续、`reshape` 随便用；`transpose` 之后想 `view` 先 `contiguous()`；`expand` 省内存、`repeat` 真复制。**

## 2.5 索引类操作：`gather / scatter / topk / index_add_ / where / 布尔索引`

这几个是读 RL 代码（DPO / PPO / GRPO）时最容易卡住的。

```python
import torch
import torch.nn.functional as F

# ---------- gather：按索引「取」 ----------
# dim=2 时：out[b][t] = input[b][t][index[b][t]]
# 📍 train_dpo.py logits_to_log_probs：取出每个位置上「真实下一个 token」的对数概率
logits = torch.randn(2, 4, 10)                      # (batch, seq, vocab)
labels = torch.randint(0, 10, (2, 4))               # (batch, seq)
log_probs = F.log_softmax(logits, dim=-1)
per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
print(per_token.shape)                              # → (2, 4)
print(torch.isclose(per_token[1, 2], log_probs[1, 2, labels[1, 2]]))   # → True

# ---------- scatter：按索引「放」（gather 的逆操作） ----------
# dim=1 时：out[i][index[i][j]] = src[i][j]
z = torch.zeros(2, 5)
idx = torch.tensor([[0, 3], [1, 4]])
print(z.scatter(1, idx, torch.ones(2, 2)))
# → [[1,0,0,1,0],[0,1,0,0,1]]

# ---------- topk ----------
scores = torch.tensor([[0.1, 0.6, 0.3], [0.5, 0.2, 0.3]])
w, i = torch.topk(scores, k=1, dim=-1)
print(w.squeeze(), i.squeeze())                     # → tensor([0.6, 0.5]) tensor([1, 0])  📍 MoE 路由选专家

# ---------- index_add_：按行索引累加（原地） ----------
# 📍 MOEFeedForward：把各专家的输出加回对应 token 的位置
y = torch.zeros(4, 2)
token_idx = torch.tensor([0, 2])
y.index_add_(0, token_idx, torch.tensor([[1., 1.], [2., 2.]]))
print(y)                                            # → 第 0 行 [1,1]，第 2 行 [2,2]

# ---------- 布尔索引与 where ----------
t = torch.tensor([3., -1., 5.])
t[t < 0] = float("-inf")                            # 📍 top-k 采样：把非候选 token 的 logits 置为 -inf
print(t)                                            # → tensor([3., -inf, 5.])
print(torch.where(t > 4, t, torch.zeros_like(t)))   # → tensor([0., 0., 5.])

# ---------- 带 mask 的平均（RL 代码中到处都是） ----------
per_token_loss = torch.randn(2, 5)
mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]).float()   # 0 表示 padding
loss = ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()
print(loss.shape)                                   # → torch.Size([])  标量
```

## 2.6 autograd 自动求导

```python
import torch

w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
loss = (w * x - 1) ** 2          # d loss/dw = 2(wx-1)·x = 2·5·3 = 30
loss.backward()
print(w.grad)                    # → tensor(30.)

# ⚠️ 梯度默认「累加」而非覆盖 —— 这正是梯度累积的原理，也是每步要 zero_grad 的原因
loss = (w * x - 1) ** 2
loss.backward()
print(w.grad)                    # → tensor(60.)
w.grad = None                    # 等价于 optimizer.zero_grad(set_to_none=True)

# no_grad / inference_mode：不建计算图，省显存（推理、ref_model 前向时用）
with torch.no_grad():
    y = w * x
print(y.requires_grad)           # → False

# detach：切断梯度流，得到「值相同但不传梯度」的张量
a = w * 2
b = a.detach()
print(b.requires_grad)           # → False

# 📍 Straight-Through 技巧（MOEFeedForward 在 top-1 时使用）：
# 前向值恒等于 1.0，但反向时梯度照常流向 p
p = torch.tensor(0.7, requires_grad=True)
st = p - p.detach() + 1.0
print(st.item())                 # → 1.0
st.backward(); print(p.grad)     # → tensor(1.)

# 梯度裁剪：总范数超过阈值则等比缩小（📍 所有 trainer 都用 grad_clip=1.0）
w2 = torch.nn.Parameter(torch.ones(3))
(w2 * 100).sum().backward()
total = torch.nn.utils.clip_grad_norm_([w2], max_norm=1.0)
print(total, w2.grad.norm())     # 裁剪前范数 173.2，裁剪后 → 1.0
```

## 2.7 损失函数：`F.cross_entropy` 与 `ignore_index`

📍 `MiniMindForCausalLM.forward`：`F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)`

```python
import torch
import torch.nn.functional as F

vocab = 5
logits = torch.randn(2, 3, vocab)                   # (batch, seq, vocab)
labels = torch.tensor([[1, 2, -100], [4, -100, -100]])

# cross_entropy 要求 (N, C) 的 logits 和 (N,) 的类别下标，所以先展平
loss = F.cross_entropy(logits.view(-1, vocab), labels.view(-1), ignore_index=-100)

# 手工验证：-100 的位置不参与计算，只对剩下 3 个位置取平均
logp = F.log_softmax(logits, dim=-1)
manual = -(logp[0, 0, 1] + logp[0, 1, 2] + logp[1, 0, 4]) / 3
print(torch.isclose(loss, manual))                  # → True

# reduction='none' 得到逐 token 的 loss，自己再乘 mask（📍 train_distillation.py）
per_tok = F.cross_entropy(logits.view(-1, vocab), labels.view(-1).clamp(min=0), reduction="none")
print(per_tok.shape)                                # → torch.Size([6])
```

**-100 是 MiniMind 里最重要的约定**：Pretrain 把 padding 设为 -100，SFT 把「非 assistant 回复」全部设为 -100。

## 2.8 优化器与学习率

📍 `optim.AdamW(model.parameters(), lr=...)`，每步手动设置 `param_group['lr'] = get_lr(...)`。

```python
import math, torch
from torch import nn, optim

model = nn.Linear(4, 1)
opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

def get_lr(step, total, lr):     # 📍 trainer_utils.get_lr：余弦从 lr 衰减到 0.1·lr
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total)))

total = 100
for step in range(total):
    for g in opt.param_groups:   # 每个 param_group 可有独立 lr
        g["lr"] = get_lr(step, total, 5e-4)
    loss = model(torch.randn(8, 4)).pow(2).mean()
    loss.backward()
    opt.step()                   # 按梯度更新参数
    opt.zero_grad(set_to_none=True)

print([round(get_lr(s, total, 1.0), 3) for s in (0, 50, 100)])   # → [1.0, 0.55, 0.1]
```

AdamW 回顾：
- **Adam** 维护梯度的一阶矩 m（动量）和二阶矩 v（梯度平方的滑动平均），更新量 ≈ `lr · m̂ / (√v̂ + ε)`，每个参数自适应步长。
- **AdamW** 把 weight decay 从梯度里「解耦」出来，直接 `θ ← θ − lr·λ·θ`，这是训练 Transformer 的事实标准。
- Adam 需要额外存 m 和 v，**优化器状态 = 2 倍参数量**（fp32），这是显存大头之一，也是 resume 文件要保存 `optimizer.state_dict()` 的原因。

## 2.9 Dataset / DataLoader / Sampler

📍 `dataset/lm_dataset.py`、`DataLoader(train_ds, batch_sampler=..., num_workers=..., pin_memory=True)`

```python
import torch
from torch.utils.data import Dataset, DataLoader

class ToyLM(Dataset):
    def __init__(self, n=10, max_len=6):
        self.n, self.max_len = n, max_len
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        tokens = [1] + list(range(2, 2 + i % 4)) + [2]          # bos ... eos
        ids = tokens + [0] * (self.max_len - len(tokens))       # pad 到固定长度
        ids = torch.tensor(ids)
        labels = ids.clone(); labels[ids == 0] = -100           # 📍 PretrainDataset 同款
        return ids, labels

ds = ToyLM()
loader = DataLoader(ds, batch_size=4, shuffle=True)             # 自动把样本 stack 成 batch
for ids, labels in loader:
    print(ids.shape, labels.shape)                              # → (4, 6) (4, 6)
    break

# 自定义 batch_sampler（每次产出一个 batch 的下标列表）时不能再传 batch_size / shuffle
loader2 = DataLoader(ds, batch_sampler=[[0, 1], [2, 3]])
print([b[0].shape for b in loader2])                            # → [(2, 6), (2, 6)]
```

要点：
- `__getitem__` 返回元组 → batch 也是元组；返回 dict → batch 是 dict（📍 `DPODataset` 返回 dict）。
- 返回字符串时，DataLoader 会把它们收集成 `list[str]`（📍 `RLAIFDataset` 返回 `prompt` 字符串）。
- `num_workers>0` 用多进程预取数据；`pin_memory=True` 加速 CPU→GPU 拷贝。

## 2.10 混合精度

📍 `autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)`、`scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))`

| 格式 | 位数 | 指数位 | 尾数位 | 特点 |
|---|---|---|---|---|
| fp32 | 32 | 8 | 23 | 默认，精确 |
| fp16 | 16 | 5 | 10 | 范围小（最大约 65504），小梯度容易下溢为 0 → 需要 **GradScaler** |
| bf16 | 16 | 8 | 7 | 范围与 fp32 相同，精度低，**一般不需要 GradScaler**（Ampere 及以上 GPU 支持） |

```python
import torch
print(torch.tensor(1e-8, dtype=torch.float16))    # → 0.（下溢）
print(torch.tensor(1e-8, dtype=torch.bfloat16))   # → 1.0012e-08（没问题）
print(torch.tensor(70000., dtype=torch.float16))  # → inf（上溢）
```

标准写法（与 `train_pretrain.py` 一致）：
```python
import torch
from contextlib import nullcontext
model = torch.nn.Linear(4, 1); opt = torch.optim.AdamW(model.parameters())
device_type = "cuda" if torch.cuda.is_available() else "cpu"
ctx = nullcontext() if device_type == "cpu" else torch.autocast(device_type, dtype=torch.bfloat16)
scaler = torch.amp.GradScaler(enabled=False)        # fp16 时设为 True

with ctx:                                           # 前向在低精度下算（矩阵乘等）
    loss = model(torch.randn(8, 4)).pow(2).mean()
scaler.scale(loss).backward()                       # fp16：把 loss 放大，避免梯度下溢
scaler.unscale_(opt)                                # 裁剪前先把梯度缩放回来
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(opt); scaler.update()                   # 若出现 inf/nan 会跳过这一步
opt.zero_grad(set_to_none=True)
```

另外 📍 `RMSNorm.forward` 中 `self.norm(x.float())` 再 `.type_as(x)`：平方求均值这种规约操作在低精度下误差大，所以临时升到 fp32 计算。

## 2.11 保存与加载

```python
import torch, os, tempfile
from torch import nn

model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
path = os.path.join(tempfile.mkdtemp(), "m.pth")

sd = {k: v.half().cpu() for k, v in model.state_dict().items()}   # 📍 以 fp16 保存，体积减半
torch.save(sd, path + ".tmp")
os.replace(path + ".tmp", path)          # 📍 lm_checkpoint：先写临时文件再原子替换，防止写一半断电损坏

new = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
missing, unexpected = new.load_state_dict(torch.load(path), strict=False)   # 加载时自动转回 fp32
print(missing, unexpected)               # → [] []
```

- `strict=False`：允许缺少或多出 key（📍 `init_model` 这样加载，方便在不同阶段增删模块）。
- `torch.compile(model)` 后真实模型在 `model._orig_mod`；DDP 包装后在 `model.module`。保存前都要「剥壳」，否则 key 会带 `_orig_mod.` / `module.` 前缀。

## 2.12 分布式 DDP 速览

📍 `init_distributed_mode()`、`DistributedSampler`、`DistributedDataParallel`、`dist.all_reduce`

```bash
# 单机 2 卡启动：torchrun 会给每个进程设置环境变量 RANK / LOCAL_RANK / WORLD_SIZE
torchrun --nproc_per_node 2 train_pretrain.py
```

核心概念：
- **数据并行**：每张卡一份完整模型，各吃不同的数据；反向时 DDP 自动对所有卡的梯度做 **all-reduce 求平均**，因此各卡参数始终一致。
- `DistributedSampler`：把数据集切成不重叠的 `WORLD_SIZE` 份；每个 epoch 要 `set_epoch(epoch)` 才能换一种打乱方式。
- 只在 `rank 0` 打印日志、保存模型（📍 `is_main_process()`、`Logger`）。
- **等效 batch = batch_size × accumulation_steps × WORLD_SIZE**。
- DDP 要求每次反向所有参数都有梯度，否则报 unused parameters —— 📍 这就是 `MOEFeedForward` 里给未被选中专家加 `0 * sum(p.sum())` 的原因。
- 📍 PPO 中 `dist.all_reduce(approx_kl_val, op=AVG)`：让所有卡基于同一个 KL 值决定是否提前停止，避免一张卡停了其他卡还在等梯度同步而死锁。

### 自测 2
1. `nn.Linear(768, 384).weight.shape` 是什么？
2. `x.transpose(1,2).view(...)` 为什么可能报错？怎么修？
3. 不调用 `optimizer.zero_grad()` 会发生什么？
4. 为什么 bf16 训练通常不需要 GradScaler？
5. `torch.gather(log_probs, 2, labels.unsqueeze(2))` 中 `unsqueeze(2)` 的作用是什么？

---

# 第三部分 数学

## 3.1 矩阵乘法与形状

- `(m, k) @ (k, n) → (m, n)`：内侧维度必须相等。
- 批量矩阵乘：前面的维度视为 batch 并逐个相乘（支持广播），**只有最后两维做矩阵乘**。

```python
import torch
q = torch.randn(2, 8, 5, 96)                 # (bsz, heads, seq_q, dim)
k = torch.randn(2, 8, 7, 96)                 # (bsz, heads, seq_k, dim)
scores = q @ k.transpose(-2, -1)             # (2, 8, 5, 96) @ (2, 8, 96, 7)
print(scores.shape)                          # → (2, 8, 5, 7)：每个 query 对每个 key 的分数
```

计算量：`(m,k)@(k,n)` 约 `2mkn` 次浮点运算。粗略地说，Transformer 训练每个 token 的计算量 ≈ `6 × 参数量`（前向 2 + 反向 4）。

## 3.2 softmax 与温度

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

```python
import torch
import torch.nn.functional as F

z = torch.tensor([2.0, 1.0, 0.1])
print(F.softmax(z, dim=-1))                   # → [0.659, 0.242, 0.099]，和为 1

# 数值稳定：先减去最大值（结果不变，但避免 e^1000 溢出）
big = torch.tensor([1000., 1001.])
print(torch.exp(big) / torch.exp(big).sum())  # → [nan, nan]
print(F.softmax(big, dim=-1))                 # → [0.269, 0.731]，PyTorch 内部已处理

# 温度 T：softmax(z / T)。T<1 更尖锐（更确定），T>1 更平缓（更随机）
for T in (0.5, 1.0, 2.0):
    print(T, F.softmax(z / T, dim=-1))
# 📍 generate 中 logits / temperature；蒸馏中 softmax(teacher_logits / T)

# -inf 经过 softmax 变成 0 概率：这就是 mask 的实现方式
print(F.softmax(torch.tensor([1., float("-inf"), 1.]), dim=-1))   # → [0.5, 0., 0.5]
```

## 3.3 交叉熵、对数概率与困惑度

- 对「真实下一个 token」为 y 的一个位置：$\text{CE} = -\log p_\theta(y)$。
- 语言模型 loss = 所有有效位置 CE 的平均 = **平均负对数似然（NLL）**。
- 困惑度 $\text{PPL} = e^{\text{loss}}$：可以理解为「模型平均在几个候选词之间犹豫」。
- 随机初始化的模型：$p \approx 1/V$，loss ≈ $\ln V$。MiniMind 词表 6400 → 初始 loss ≈ **8.76**。看到训练开始时 loss 在 8~9 左右就说明一切正常。

```python
import math, torch
import torch.nn.functional as F
V = 6400
print(math.log(V))                                        # → 8.764
logits = torch.zeros(1, V)                                # 均匀分布
print(F.cross_entropy(logits, torch.tensor([42])).item()) # → 8.764
print(math.exp(2.0))                                      # loss=2.0 时 PPL≈7.39
```

`log_softmax` 比 `log(softmax(x))` 数值更稳定，所以代码里一律用 `F.log_softmax`。

## 3.4 KL 散度及其估计

$$\text{KL}(P\|Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)} \ge 0$$

- 衡量用 Q 近似 P 的「信息损失」；**不对称**：$\text{KL}(P\|Q) \ne \text{KL}(Q\|P)$。
- $\text{KL} = \text{CE}(P, Q) - H(P)$：当 P 固定（如 teacher），最小化 KL 等价于最小化交叉熵。

```python
import torch
import torch.nn.functional as F

p = torch.tensor([0.7, 0.2, 0.1])
q = torch.tensor([0.5, 0.3, 0.2])
kl_pq = (p * (p / q).log()).sum()
kl_qp = (q * (q / p).log()).sum()
print(kl_pq.item(), kl_qp.item())                 # → 0.0851 0.0920（不对称）

# ⚠️ F.kl_div(input, target) 的约定：input 是「对数概率」log q，target 是概率 p，算的是 KL(p || q)
print(F.kl_div(q.log(), p, reduction="sum").item())   # → 0.0851 = KL(p||q)
# 📍 train_distillation.py：F.kl_div(student_log_probs, teacher_probs) → KL(teacher || student)
```

**RL 中的 KL 估计（k1 / k2 / k3）**：在 LLM 里词表太大，无法对所有 token 求和，只能用「采样到的那个 token」来估计。设样本来自 $\pi_\theta$，$r = \pi_{ref}(x)/\pi_\theta(x)$：

| 估计器 | 公式 | 无偏？ | 非负？ |
|---|---|---|---|
| k1 | $-\log r$ | ✅ | ❌（单样本可能为负） |
| k2 | $\frac12(\log r)^2$ | ❌（有偏但方差小） | ✅ |
| k3 | $(r-1) - \log r$ | ✅ | ✅ |

📍 `train_grpo.py`：`kl_div = ref_logp - logp; per_token_kl = exp(kl_div) - kl_div - 1` 就是 k3。
📍 `train_ppo.py`：`approx_kl = 0.5 * log_ratio**2` 是 k2。

```python
import torch
torch.manual_seed(0)
p = torch.tensor([0.6, 0.3, 0.1]); q = torch.tensor([0.4, 0.4, 0.2])   # p=policy, q=ref
true_kl = (p * (p / q).log()).sum()
x = torch.multinomial(p, 200000, replacement=True)                    # 从 policy 采样
log_r = (q[x] / p[x]).log()
print(f"true {true_kl:.4f} | k1 {(-log_r).mean():.4f} | k2 {(0.5*log_r**2).mean():.4f} | k3 {(log_r.exp()-1-log_r).mean():.4f}")
# → true 0.0877 | k1 ≈0.088 | k2 ≈0.08x（有偏） | k3 ≈0.088
```

## 3.5 旋转矩阵与复数：RoPE 的数学

二维旋转 θ 角：
$$R(\theta)=\begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix}$$

关键性质：$R(a)^\top R(b) = R(b-a)$。所以如果把位置 m 的 q 旋转 $m\theta$、位置 n 的 k 旋转 $n\theta$：

$$(R(m\theta)q)^\top (R(n\theta)k) = q^\top R((n-m)\theta)\, k$$

点积**只依赖相对位置 n−m** —— 这就是 RoPE 的全部核心思想。等价地，把二维向量看成复数 $z=x+iy$，旋转就是乘以 $e^{i\theta}$。

高维时把 d 维向量两两配对成 d/2 个二维平面，第 i 个平面用频率 $\theta_i = \text{base}^{-2i/d}$：低 i 转得快（高频，捕捉近距离），高 i 转得慢（低频，捕捉远距离）。MiniMind 用 `base = rope_theta = 1e6`。

📍 MiniMind 的 `rotate_half` 写法把第 i 维和第 i+d/2 维配对（而不是相邻的 2i、2i+1），两者数学上等价。

```python
import torch

def precompute(dim, end, base=1e6):                 # 精简自 precompute_freqs_cis
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))    # (dim/2,)
    t = torch.arange(end).float()
    ang = torch.outer(t, freqs)                     # (end, dim/2)：位置 × 频率
    return torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)

def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

def rope(x, cos, sin):                               # x: (..., dim)
    return x * cos + rotate_half(x) * sin

dim = 8
cos, sin = precompute(dim, 100)
q, k = torch.randn(dim), torch.randn(dim)

def score(m, n):
    return (rope(q, cos[m], sin[m]) * rope(k, cos[n], sin[n])).sum()

print(score(3, 7).item(), score(53, 57).item())     # 相对距离都是 4 → 两个值相等
print(score(3, 7).item(), score(3, 8).item())       # 相对距离不同 → 值不同
print(rope(q, cos[5], sin[5]).norm().item(), q.norm().item())   # 旋转不改变向量长度
```

## 3.6 sigmoid / logsigmoid

$$\sigma(x) = \frac{1}{1+e^{-x}},\qquad \log\sigma(x) = -\log(1+e^{-x})$$

📍 DPO：`loss = -F.logsigmoid(beta * logits)`。它是 Bradley-Terry 偏好模型的负对数似然：「chosen 比 rejected 好」的概率为 $\sigma(\text{奖励差})$。

```python
import torch, math
import torch.nn.functional as F
print(-F.logsigmoid(torch.tensor(0.)).item(), math.log(2))   # → 0.6931 0.6931：DPO 起始 loss = ln2
print(-F.logsigmoid(torch.tensor(5.)).item())                # 差距越大 loss 越接近 0
print(F.logsigmoid(torch.tensor(-1000.)).item())             # → -1000，数值稳定（手写 log(sigmoid) 会得到 -inf）
```

## 3.7 均值、方差与标准化

$$\hat{x} = \frac{x-\mu}{\sigma+\epsilon}$$

📍 GRPO 的组内优势：同一个 prompt 采样的 G 个回答，`A = (r - mean) / (std + 1e-4)`。
📍 PPO：对优势做带 mask 的全局标准化。

```python
import torch
rewards = torch.tensor([1., 0., 0.5, 1.,   0.2, 0.2, 0.2, 0.2])   # 2 个 prompt × 4 个回答
G = 4
g = rewards.view(-1, G)
mean = g.mean(1).repeat_interleave(G)
std = g.std(1, unbiased=False).repeat_interleave(G)
adv = (rewards - mean) / (std + 1e-4)
print(adv)       # 第二组奖励完全相同 → 优势全为 0 → 这组样本没有学习信号
```

`unbiased=False` 用 N 做分母（总体方差），`True` 用 N−1（样本方差）。

## 3.8 低秩分解

- 矩阵 $W\in\mathbb{R}^{d\times d}$ 有 $d^2$ 个参数；若 $\Delta W = BA$，$B\in\mathbb{R}^{d\times r}, A\in\mathbb{R}^{r\times d}$，只需 $2dr$ 个参数，秩 ≤ r。
- LoRA 的假设：**微调时的权重更新 ΔW 是低秩的**。

```python
import torch
d, r = 768, 16
print(d * d, 2 * d * r, f"{2*d*r/(d*d):.1%}")    # → 589824 24576 4.2%

A = torch.randn(r, d) * 0.02                      # 📍 A 高斯初始化
B = torch.zeros(d, r)                             # 📍 B 零初始化 → 初始 ΔW = 0，不破坏原模型
print(torch.linalg.matrix_rank(torch.randn(d, r) @ A).item())   # → 16
```

### 自测 3
1. 词表 V=6400，随机初始化模型的初始 loss 大约是多少？
2. `F.kl_div(a, b)` 中 a、b 分别应该是什么？计算的是哪个方向的 KL？
3. 用一句话说明 RoPE 为什么能编码相对位置。
4. DPO 训练开始时 loss 为什么约等于 0.693？

---

# 第四部分 深度学习与 Transformer

## 4.1 语言建模与 shift

**自回归语言模型**：$p(x_1,\dots,x_T)=\prod_t p(x_t\mid x_{<t})$。训练目标就是在每个位置预测下一个 token。

```
input_ids : [BOS]  今    天    天    气
预测目标  :  今    天    天    气   [EOS]
```

📍 MiniMind 的做法：Dataset 返回 `input_ids` 和**未错位**的 `labels`，由模型内部做 shift：
```py
# 源码摘录（不可单独运行）：MiniMindForCausalLM.forward
x, y = logits[..., :-1, :], labels[..., 1:]      # 第 t 个位置的输出去预测第 t+1 个 token
```
📍 `DPODataset` 则是在 Dataset 里就错位好了（`x = ids[:-1]`、`y = ids[1:]`）。**读代码时一定要先搞清楚 shift 发生在哪里**。

一次前向就能同时得到所有位置的预测（Teacher Forcing），前提是用**因果 mask** 保证位置 t 看不到 t 之后的内容。

## 4.2 分词：BPE

- **BPE**（Byte Pair Encoding）：从字符（或字节）开始，反复把语料中出现频率最高的相邻对合并成新 token，直到达到目标词表大小。
- **Byte-level BPE**：以 UTF-8 字节为最小单位，任何字符串都能编码，不会出现 OOV（未登录词）。📍 MiniMind 的 `train_tokenizer.py` 用 `tokenizers` 库训练，词表 6400。
- **特殊 token**：📍 `bos_token = <|im_start|>`、`eos_token = <|im_end|>`、`pad_token = <|endoftext|>`，以及 `<think>`、`<tool_call>` 等。
- **Chat Template**：把多轮对话 `[{role, content}]` 拼成带特殊 token 的字符串。大致形如：
  ```
  <|im_start|>system
  你是一个助手<|im_end|>
  <|im_start|>user
  你好<|im_end|>
  <|im_start|>assistant
  你好！<|im_end|>
  ```
  📍 `SFTDataset.generate_labels` 正是以 `<|im_start|>assistant\n` 和 `<|im_end|>\n` 作为边界，只对 assistant 回复计算 loss。

词表大小的权衡：词表越大，一个 token 能承载的字符越多（序列越短），但 embedding 和 lm_head 的参数量 = V × d 也越大。对 64M 的小模型，V=6400 时两者（共享后）只占约 4.9M。

## 4.3 缩放点积注意力（Scaled Dot-Product Attention）

$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

- Q（query，我在找什么）、K（key，我有什么）、V（value，我的内容）都由输入经过线性层得到。
- **为什么除以 $\sqrt{d_k}$**：若 q、k 各分量独立、均值 0、方差 1，则点积的方差为 $d_k$。不缩放时 softmax 输入过大，会趋于 one-hot，梯度接近 0。
- **因果 mask M**：上三角（未来位置）为 $-\infty$，softmax 之后权重为 0。

```python
import math, torch
import torch.nn.functional as F

torch.manual_seed(0)
B, H, T, D = 1, 2, 4, 8
q, k, v = torch.randn(B, H, T, D), torch.randn(B, H, T, D), torch.randn(B, H, T, D)

# 手写版本（📍 Attention.forward 中的非 flash 分支）
scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)             # (B, H, T, T)
mask = torch.full((T, T), float("-inf")).triu(1)              # 对角线以上为 -inf
print(mask)
attn = F.softmax(scores + mask, dim=-1)
out_manual = attn @ v                                         # (B, H, T, D)
print(attn[0, 0])                                             # 下三角矩阵，每行和为 1

# PyTorch 内置的融合实现（Flash Attention 等）—— 📍 flash 分支
out_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(torch.allclose(out_manual, out_sdpa, atol=1e-6))       # → True

# 为什么要缩放：
x = (torch.randn(10000, 96) * torch.randn(10000, 96)).sum(-1)
print(x.var().item())                                         # ≈ 96 = d_k
```

## 4.4 多头：MHA / MQA / GQA

- **多头（Multi-Head）**：把 d_model 切成 h 个头，每个头在 $d_k = d_{model}/h$ 的子空间里独立做注意力，最后拼接再过 `o_proj`。不同的头可以关注不同类型的关系。
- **MHA**：Q、K、V 都有 h 个头。
- **MQA**：只有 1 组 K/V，所有 Q 头共享。
- **GQA**：K/V 有 g 组（1 < g < h），每 h/g 个 Q 头共享一组。📍 MiniMind：`num_attention_heads=8, num_key_value_heads=4`。

为什么要减少 KV 头？推理时需要缓存所有历史 token 的 K 和 V（KV Cache，见 4.10），它的大小正比于 KV 头数。GQA 在几乎不掉效果的前提下让 KV Cache 减半。

```python
import torch
import torch.nn.functional as F
from torch import nn

class GQA(nn.Module):
    def __init__(self, d=64, n_heads=8, n_kv=4):
        super().__init__()
        self.h, self.kv, self.hd = n_heads, n_kv, d // n_heads
        self.q = nn.Linear(d, n_heads * self.hd, bias=False)
        self.k = nn.Linear(d, n_kv * self.hd, bias=False)     # K、V 投影更小
        self.v = nn.Linear(d, n_kv * self.hd, bias=False)
        self.o = nn.Linear(n_heads * self.hd, d, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)     # (B, 8, T, hd)
        k = self.k(x).view(B, T, self.kv, self.hd).transpose(1, 2)    # (B, 4, T, hd)
        v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        rep = self.h // self.kv
        k = k.repeat_interleave(rep, dim=1)                           # (B, 8, T, hd)，等价于 📍 repeat_kv
        v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).reshape(B, T, -1))

m = GQA()
print(m(torch.randn(2, 5, 64)).shape)       # → (2, 5, 64)
print(m.k.weight.shape, m.q.weight.shape)   # → (32, 64) (64, 64)：K 投影参数只有 Q 的一半
```

📍 MiniMind 还额外在每个头上对 q、k 做 RMSNorm（**QK-Norm**，来自 Qwen3），防止注意力分数过大、使训练更稳定。

## 4.5 位置编码

注意力本身是**置换不变**的（打乱 token 顺序，每个 token 的输出只是跟着换位置），所以必须显式注入位置信息。

| 方法 | 做法 | 代表模型 |
|---|---|---|
| 正弦绝对位置编码 | 在 embedding 上加 sin/cos 向量 | 原始 Transformer |
| 可学习绝对位置编码 | `nn.Embedding(max_len, d)` 加到输入上 | GPT-2、BERT |
| **RoPE** | 旋转 q、k（见 3.5），点积只依赖相对位置 | LLaMA、Qwen、📍 MiniMind |
| ALiBi | 在注意力分数上加与距离成正比的偏置 | BLOOM |

**长度外推**：训练时只见过长度 L，推理时超过 L，低频维度会转到从未见过的角度。**YaRN** 的思路是对低频维度按比例缩放频率（相当于插值），高频维度保持不变，中间平滑过渡。📍 `precompute_freqs_cis` 中的 `ramp` 就是在做这件事，推理时通过 `--inference_rope_scaling` 开启。

## 4.6 FFN 与激活函数

- 经典 FFN：`W2 · act(W1 · x)`，中间维度通常为 4d。
- **GLU 家族**：`W_down · (act(W_gate · x) ⊙ W_up · x)`，多一个门控分支。act 为 SiLU 时就是 **SwiGLU**（📍 `FeedForward`）。
- 因为有 3 个矩阵，为保持参数量与 4d 的经典 FFN 相当，中间维度一般取 ≈ 8d/3。📍 MiniMind 取 `ceil(d·π/64)·64 = 2432`。

```python
import torch
import torch.nn.functional as F
from torch import nn

x = torch.linspace(-3, 3, 7)
print("relu", F.relu(x))
print("gelu", F.gelu(x))
print("silu", F.silu(x))          # silu(x) = x·sigmoid(x)，又叫 Swish；平滑、负半轴有小的负值

class SwiGLU(nn.Module):          # 📍 与 FeedForward 一致
    def __init__(self, d=768, hidden=2432):
        super().__init__()
        self.gate_proj = nn.Linear(d, hidden, bias=False)
        self.up_proj = nn.Linear(d, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

f = SwiGLU()
print(sum(p.numel() for p in f.parameters()) / 1e6, "M")   # → 5.6M（经典 4d FFN 为 2×768×3072 = 4.7M）
```

## 4.7 归一化：LayerNorm vs RMSNorm，Pre-Norm vs Post-Norm

$$\text{LayerNorm}(x)=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\cdot\gamma+\beta\qquad\text{RMSNorm}(x)=\frac{x}{\sqrt{\text{mean}(x^2)+\epsilon}}\cdot\gamma$$

RMSNorm 去掉了减均值和 β，计算更少，效果相当。

```python
import torch
from torch import nn

class RMSNorm(nn.Module):                       # 📍 与 model_minimind.py 一致
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        x32 = x.float()
        return (self.weight * x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)

x = torch.randn(2, 3, 16) * 5 + 3
y = RMSNorm(16)(x)
print(y.pow(2).mean(-1))                        # 每个向量的均方值都 ≈ 1
ln = nn.LayerNorm(16)(x)
print(ln.mean(-1).abs().max().item(), ln.std(-1, unbiased=False).mean().item())   # 均值 0、标准差 1
```

**Pre-Norm vs Post-Norm**：
```
Post-Norm（原始 Transformer）：x = Norm(x + Sublayer(x))
Pre-Norm （现代 LLM，📍 MiniMind）：x = x + Sublayer(Norm(x))
```
Pre-Norm 让残差路径成为一条「干净的高速公路」，梯度可以直接从最后一层流回第一层，深层网络更易训练、对学习率和 warmup 不那么敏感。代价是最后要多加一个 Norm（📍 `MiniMindModel.norm`）。

## 4.8 残差连接

`y = x + f(x)`，反向时 $\frac{\partial y}{\partial x} = I + \frac{\partial f}{\partial x}$，恒等项保证梯度不会在深层网络中消失。

📍 `MiniMindBlock.forward`：
```py
# 源码摘录（不可单独运行）
hidden_states = hidden_states + attn(input_layernorm(hidden_states))
hidden_states = hidden_states + mlp(post_attention_layernorm(hidden_states))
```

## 4.9 100 行从零实现一个迷你 GPT 并训练 ⭐

把上面所有零件组装起来。结构与 MiniMind 完全同构（Pre-Norm + RMSNorm + RoPE + GQA + SwiGLU + 权重共享 + 模型内 shift），只是更小。**建议在读 `model_minimind.py` 之前先亲手敲一遍**，之后再读源码会非常轻松。

```python
import math, torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(0)

# ---------------- 零件 ----------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def precompute_rope(hd, max_len, base=1e6):
    f = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    ang = torch.outer(torch.arange(max_len).float(), f)
    return torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)   # (max_len, hd)

def apply_rope(x, cos, sin):                      # x: (B, T, H, hd)
    half = x.shape[-1] // 2
    rot = torch.cat([-x[..., half:], x[..., :half]], -1)
    return x * cos[:, None, :] + rot * sin[:, None, :]

class Attention(nn.Module):
    def __init__(self, d, n_heads, n_kv):
        super().__init__()
        self.h, self.kv, self.hd = n_heads, n_kv, d // n_heads
        self.q_proj = nn.Linear(d, n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(d, n_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(d, n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(n_heads * self.hd, d, bias=False)
    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.hd)
        k = self.k_proj(x).view(B, T, self.kv, self.hd)
        v = self.v_proj(x).view(B, T, self.kv, self.hd)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.h // self.kv
        q, k, v = q.transpose(1, 2), k.repeat_interleave(rep, 2).transpose(1, 2), v.repeat_interleave(rep, 2).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))

class FFN(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate_proj, self.up_proj = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Block(nn.Module):
    def __init__(self, d, n_heads, n_kv, hidden):
        super().__init__()
        self.attn, self.mlp = Attention(d, n_heads, n_kv), FFN(d, hidden)
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)     # Pre-Norm + 残差
        x = x + self.mlp(self.norm2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab, d=64, n_layers=2, n_heads=4, n_kv=2, max_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([Block(d, n_heads, n_kv, math.ceil(d * math.pi / 64) * 64) for _ in range(n_layers)])
        self.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.embed.weight = self.lm_head.weight        # 权重共享
        cos, sin = precompute_rope(d // n_heads, max_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, ids, labels=None):
        T = ids.shape[1]
        h = self.embed(ids)
        for layer in self.layers:
            h = layer(h, self.cos[:T], self.sin[:T])
        logits = self.lm_head(self.norm(h))
        loss = None
        if labels is not None:                         # 模型内部 shift
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            logits, _ = self(ids)                      # 无 KV Cache 的朴素版本：每步都重算整个序列
            nxt = torch.multinomial(F.softmax(logits[:, -1] / temperature, -1), 1)
            ids = torch.cat([ids, nxt], 1)
        return ids

# ---------------- 数据：字符级 ----------------
text = "大道至简。从零开始训练一个语言模型，理解每一行代码。" * 50
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
def batch(bs=16, T=32):
    ix = torch.randint(0, len(data) - T, (bs,))
    x = torch.stack([data[i:i + T] for i in ix])
    return x, x.clone()                                # labels = input_ids（shift 在模型内做）

# ---------------- 训练 ----------------
model = MiniGPT(len(chars))
print(f"params: {sum(p.numel() for p in model.parameters())/1e3:.1f}K, 初始 loss 理论值 ln(V)={math.log(len(chars)):.2f}")
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
for step in range(301):
    x, y = batch()
    _, loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    if step % 100 == 0:
        print(step, round(loss.item(), 3))

start = torch.tensor([[stoi["大"]]])
print("".join(chars[i] for i in model.generate(start, 25, temperature=0.5)[0].tolist()))
# → loss 从 ≈3.4 降到 0.01 以下，并能背出「大道至简。从零开始训练一个语言模型……」
```

## 4.10 KV Cache

**问题**：自回归生成第 t 个 token 时，朴素做法要把整个前缀重新算一遍，总计算量 O(T²)。
**观察**：因果注意力下，历史 token 的 K、V 不会随新 token 改变。
**做法**：把每层历史的 K、V 缓存起来，新 token 只需算自己的 q、k、v，再把 k、v 拼到缓存后面。

📍 `Attention.forward`：`xk = torch.cat([past_key_value[0], xk], dim=1)`；`MiniMindModel.forward` 通过缓存长度得到 `start_pos`，从而取出新 token 对应位置的 RoPE。

```python
import torch
import torch.nn.functional as F
torch.manual_seed(0)

d = 16
Wq, Wk, Wv = (torch.randn(d, d) for _ in range(3))
x = torch.randn(1, 6, d)                                  # 6 个 token

def attn_full(x):                                         # 一次性算整个序列
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

def attn_step(x_t, cache):                                # 只输入 1 个新 token
    q, k, v = x_t @ Wq, x_t @ Wk, x_t @ Wv
    if cache is not None:
        k = torch.cat([cache[0], k], dim=1)
        v = torch.cat([cache[1], v], dim=1)
    # 单个 query 可以看到所有历史 key，因此不需要 causal mask
    out = F.scaled_dot_product_attention(q, k, v)
    return out, (k, v)

full = attn_full(x)
cache, outs = None, []
for t in range(x.shape[1]):
    o, cache = attn_step(x[:, t:t + 1], cache)
    outs.append(o)
print(torch.allclose(full, torch.cat(outs, 1), atol=1e-5))    # → True：结果与完整计算一致
print("cache shape:", cache[0].shape)                         # → (1, 6, 16)
```

KV Cache 的大小 = `2（K和V）× 层数 × KV头数 × head_dim × 序列长度 × batch × 每元素字节数`。这正是 GQA 要减少 KV 头数的原因。

## 4.11 解码采样策略

📍 `MiniMindForCausalLM.generate` 的处理顺序：**temperature → repetition penalty → top-k → top-p → 采样**。

```python
import torch
import torch.nn.functional as F
torch.manual_seed(0)

logits = torch.tensor([[3.0, 2.5, 1.0, 0.5, -1.0, -2.0]])

# 1) 贪心：永远取最大，确定但容易重复
print(logits.argmax(-1))                                          # → tensor([0])

# 2) 温度
t = logits / 0.85

# 3) 重复惩罚：已出现过的 token，正 logit 除以 penalty、负 logit 乘以 penalty（都会降低概率）
seen, penalty = torch.tensor([1]), 1.2
s = t[0, seen]; t[0, seen] = torch.where(s > 0, s / penalty, s * penalty)

# 4) top-k：只保留最大的 k 个
k = 4
t[t < torch.topk(t, k)[0][..., -1, None]] = float("-inf")

# 5) top-p（nucleus）：按概率从大到小累加，保留累计概率刚好超过 p 的最小集合
p = 0.85
sorted_logits, sorted_idx = torch.sort(t, descending=True)
cum = torch.cumsum(F.softmax(sorted_logits, -1), -1)
mask = cum > p
mask[..., 1:] = mask[..., :-1].clone()   # 右移一位：让「刚好越过 p 的那个 token」也保留
mask[..., 0] = False                     # 概率最大的 token 永远保留
t[mask.scatter(1, sorted_idx, mask)] = float("-inf")   # 把 mask 映射回原词表顺序
print(F.softmax(t, -1))

# 6) 按概率采样
print(torch.multinomial(F.softmax(t, -1), 1))
```

## 4.12 训练技巧速记

| 技巧 | 作用 | MiniMind 中的位置 |
|---|---|---|
| 学习率调度（warmup + cosine） | 前期小步稳定、后期小步收敛 | `get_lr`（只有 cosine，无 warmup） |
| 梯度累积 | 显存不够时用多个小 batch 模拟大 batch | `loss / accumulation_steps`，每 N 步 `step()` |
| 梯度裁剪 | 防止梯度爆炸 | `clip_grad_norm_(..., 1.0)` |
| 混合精度 | 省显存、加速 | `autocast` + `GradScaler` |
| 权重衰减 | 正则化 | AdamW 默认 0.01 |
| Dropout | 正则化（大模型预训练通常设为 0） | `config.dropout = 0.0` |
| 断点续训 | 保存 model + optimizer + scaler + step | `lm_checkpoint`、`SkipBatchSampler` |
| `torch.compile` | 图编译加速 | `--use_compile 1` |

梯度累积为什么要除以 N：梯度是累加的，累加 N 次 `loss/N` 的梯度 = 大 batch 平均 loss 的梯度。

```python
import torch
torch.manual_seed(0)
w = torch.nn.Parameter(torch.randn(3)); X = torch.randn(8, 3)
(X @ w).pow(2).mean().backward(); g_big = w.grad.clone(); w.grad = None
for chunk in X.chunk(4):                                   # 4 个 micro-batch
    ((chunk @ w).pow(2).mean() / 4).backward()
print(torch.allclose(g_big, w.grad))                       # → True
```

## 4.13 MoE（混合专家）入门

- 把一个 FFN 换成 E 个「专家」FFN + 一个**路由器（gate）**。每个 token 只激活 top-k 个专家。
- **总参数量大，但每个 token 的计算量只相当于 k 个专家**。📍 minimind-3-moe：4 个专家、top-1，总参 198M、激活参数 64M（记作 198M-A64M）。
- **负载均衡问题**：路由器容易「偏科」，总把 token 发给少数专家，导致其余专家学不到东西。解决方法是加辅助损失 `aux_loss = E · Σ_i f_i · P_i`，其中 f_i 是分给专家 i 的 token 比例、P_i 是路由到专家 i 的平均概率。完全均匀时该值最小。

```python
import torch
import torch.nn.functional as F
from torch import nn
torch.manual_seed(0)

class MoE(nn.Module):                           # 精简自 📍 MOEFeedForward
    def __init__(self, d=16, E=4, k=1):
        super().__init__()
        self.E, self.k = E, k
        self.gate = nn.Linear(d, E, bias=False)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d, 32), nn.SiLU(), nn.Linear(32, d)) for _ in range(E)])
    def forward(self, x):                        # x: (N, d)，N 个 token
        probs = F.softmax(self.gate(x), -1)                      # (N, E)
        w, idx = torch.topk(probs, self.k, -1)                   # 每个 token 选 k 个专家
        y = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            m = (idx == e)                                       # (N, k)
            if m.any():
                tok = m.any(-1).nonzero().flatten()              # 路由到专家 e 的 token
                y.index_add_(0, tok, expert(x[tok]) * w[m].view(-1, 1))
        load = F.one_hot(idx, self.E).float().mean(0).sum(0) / self.k   # f_i
        aux = self.E * (load * probs.mean(0)).sum()              # 负载均衡损失
        return y, aux, load

moe = MoE()
y, aux, load = moe(torch.randn(100, 16))
print(y.shape, round(aux.item(), 3), load)       # 各专家分到的 token 比例
```

### 自测 4
1. 为什么注意力分数要除以 $\sqrt{d_k}$？
2. GQA 相比 MHA 节省了什么？MiniMind 的 KV Cache 相比同配置 MHA 省了多少？
3. Pre-Norm 相比 Post-Norm 的主要优势是什么？
4. KV Cache 为什么在单 token 解码时不需要 causal mask？
5. MoE 中 198M-A64M 是什么意思？

---

# 附录 自测题答案

**自测 1**
1. `type=bool` 会对字符串调用 `bool()`，任何非空字符串（包括 `"0"`、`"False"`）都为 `True`。
2. `[2, 2, 2]`，闭包晚绑定。
3. 返回 `model` 本身（默认值）。

**自测 2**
1. `torch.Size([384, 768])`，即 `[out_features, in_features]`。
2. `transpose` 后内存不连续，`view` 要求连续内存。改用 `reshape`，或者先 `.contiguous()` 再 `view`。
3. 梯度会在多个 step 之间持续累加，更新方向错误（除非是有意的梯度累积）。
4. bf16 的指数位与 fp32 相同（8 位），表示范围一样大，小梯度不会下溢。
5. `gather` 要求 `index` 与 `input` 的维数相同；把 `(B, T)` 变成 `(B, T, 1)`，表示在词表维上每个位置取 1 个元素，之后再 `squeeze(-1)` 去掉。

**自测 3**
1. ln 6400 ≈ 8.76。
2. a 是对数概率（例如 student 的 `log_softmax`），b 是概率（例如 teacher 的 `softmax`），计算的是 KL(b ‖ exp(a))。
3. 位置 m 的 q 旋转 mθ，位置 n 的 k 旋转 nθ，由于 $R(m\theta)^\top R(n\theta)=R((n-m)\theta)$，点积只依赖相对位置 n−m。
4. 策略模型初始时等于参考模型，两个 log ratio 相减为 0，$-\log\sigma(0)=\ln 2\approx0.693$。

**自测 4**
1. 点积方差随 $d_k$ 线性增长，不缩放时 softmax 饱和成近似 one-hot，梯度消失。
2. 节省 K/V 投影参数，更重要的是推理时的 KV Cache。MiniMind KV 头数 4、Q 头数 8，KV Cache 为 MHA 的一半。
3. 残差路径上没有 Norm，梯度可以直接回传，深层网络训练更稳定，对 warmup 和学习率不那么敏感。
4. 新 token 位于序列最末尾，本来就应该看到所有历史 token，没有「未来」需要屏蔽。
5. 总参数 198M，每个 token 实际激活（参与计算）的参数为 64M。

---

📚 **想深入某一块时推荐的资料**
- Python：《流畅的 Python》第 7 章（闭包与装饰器）、第 15 章（上下文管理器）
- PyTorch：官方 *Learn the Basics* 教程；`torch.gather` / `scatter` 文档中的示意图
- 数学：3Blue1Brown《线性代数的本质》；John Schulman 博客 *Approximating KL Divergence*（k1/k2/k3）
- Transformer：Jay Alammar *The Illustrated Transformer*；Andrej Karpathy *Let's build GPT: from scratch*（视频，与 4.9 节思路一致）；苏剑林《Transformer 升级之路》（RoPE / YaRN）
