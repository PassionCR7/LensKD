# LensKD

**白盒知识蒸馏的最优形态：双投影器跨词表对齐 + 熵门控自适应散度 + 在线策略迭代**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Cite](https://img.shields.io/badge/cite-CITATION.cff-lightgrey)](CITATION.cff)

针对小模型（25.83M）从大模型（596M，23 倍容量鸿沟）蒸馏的真实工程实现。
不是论文复述，是踩过每个坑之后的可用版本。

> 姊妹项目 **[ChorusKD](https://github.com/PassionCR7/ChorusKD)** — 黑盒（闭源 API）蒸馏，
> 只有文本通道时的最优形态。

> **使用与引用**：本仓库采用 Apache-2.0，**要求署名**。改进、再分发或用于
> 学术工作时，须在文档或致谢中注明项目名、作者与仓库地址，并保留 `NOTICE`
> 文件。详见 [License 与署名要求](#license-与署名要求)。

---

## 为什么需要它

标准白盒蒸馏的做法是：教师有 151936 维词表，学生有 6400 维，取交集 5870 维，
在子集上重新归一化后算 KL。这条路有三个无法回避的缺陷：

| 缺陷 | 实测值 |
|---|---|
| 教师概率质量丢失 | **13.80%**（86.20% 才落在公共子集） |
| 教师 top-1 被重归一化抹掉的位置 | **11.28%** |
| 教师意图被系统性扭曲 | 子集外的高概率 token 直接消失 |

LensKD 换了思路：**不对齐词表，而是对齐"谁在用哪个 head"**。

```
教师空间：h_t @ W_t^T
学生空间：h_s @ W_s^T

投影后：
  学生空间里看教师 = P_t2s(h_t) @ W_s^T      ← 与 W_s @ h_s 同空间
  教师空间里看学生 = P_s2t(h_s) @ W_t^T      ← 与 W_t @ h_t 同空间
```

两边都在**完整词表**上算 KL，零过滤、零丢质量。

---

## 核心发现：投影会抹平分布

这是实现过程中最有价值的发现，也是本仓库与"直接照抄 DSKD"的分水岭。

伪逆初始化后做诊断，得到如下数据：

```
教师原生熵              mean = 1.328   （上界 log(151936) = 11.93，很尖锐）
学生原生熵              mean = 2.492   （上界 log(6400)   =  8.76）
教师投影到学生空间后熵  mean = 7.334   ← 接近上界，几乎均匀！
```

熵从 **1.33 涨到 7.33**，分布被彻底抹平。直接后果：

1. 熵门控 `gate = sigmoid((H-τ)/β)` 恒等于 1 → **全部走前向 KL**，门控机制完全失效
2. KD 损失虚高到 7.84，但没有信息量（教的是均匀分布）
3. 训练早期投影器在输出噪声，反而污染学生参数

**这不是实现 bug，是数学必然。** 验证方法：

```
lstsq 实际残差        = 0.88621
投影理论下界          = 0.88621    ← 完全相同
```

原因：教师 lm_head 是 `[151936, 1024]`，秩 1024（满秩）；学生是 `[6400, 512]`，
秩仅 512。用一半维度的空间复现 1024 维的判别信息，残差等于 `W_t` 在 `W_s`
行空间外的正交分量占比。最小二乘解已经是**理论最优**，没有改进空间。

### 解决方案：双路径融合

LensKD 同时保留两条 KD 路径：

| 路径 | 机制 | 优点 | 缺点 |
|---|---|---|---|
| **proj** | 投影 + 共享学生 head，6400 维 | 零概率质量丢失 | 分布被抹平（熵 7.33） |
| **native** | 教师原生 logits 限制在公共子集 | 分布真实尖锐（熵 1.28） | 13.8% 质量丢失 |

```python
kd = (1 - r) * kl_native + r * kl_proj      # r = kd_proj_ratio，默认 0.3
```

**默认偏向 native（0.7）**，因为尖锐的教师信号远比"无丢失但均匀"的信号有用。

实测效果：

| 指标 | 只用 proj 路径 | 双路径融合 |
|---|---|---|
| `kd` | 7.84 | **3.26** |
| native 路径熵 | — | **1.28**（真实尖锐） |
| 熵门控均值 | 1.00（失效） | **native 0.42 / proj 0.98** |
| 门控是否分流 | 否 | **是** |

`proj=0.98` 印证抹平问题存在，`native=0.42` 说明这条路径真正在做逐位置区分。

---

## 七个阶段

| 阶段 | 模块 | 做什么 |
|---|---|---|
| **A** 特征提取 | `hidden_capture.py` | forward hook 抓全层 hidden，不改模型定义 |
| **B** 双投影器 | `projector.py` | P_t2s / P_s2t + 伪逆初始化 |
| **C** ETA 对齐 | `eta_align.py` | token 文本严格相等的单调双射，支持 1→k / k→1 |
| **D** 熵门控散度 | `divergence.py` | 低熵走反向 KL，高熵走前向 KL |
| **E** 表征对齐 | `divergence.py` | 按相对深度映射层 + cosine（非 MSE） |
| **F** 联合损失 | `loss.py` | CE + 双路径 KD + 表征 |
| **G** 在线策略 | `train.py` + `data.py` | 学生自采样回灌，λ 从 0.2 渐进到 0.7 |

### 各阶段的关键决策

**C · ETA 而非字符边界**

原实现用「字符区间末尾相等」判断对齐，命中率 76~85%。ETA 直接比较 token
解码文本，能捕获 `教师"蓝色"` ↔ `学生"蓝"+"色"` 这类情形，且支持一对多
（教师一个位置的分布按学生 token 数均分复制）。

**D · 为什么必须自适应**

- 前向 KL `KL(p_t||p_s)` 是 mode-covering：学生必须在教师有概率处也有概率，
  结果被拉向概率质量**重心**。多峰分布的重心落在峰间低密度区——这是学生
  "说话骑墙、什么都沾一点"的几何成因。
- 反向 KL `KL(p_s||p_t)` 是 mode-seeking：塌到教师最有信心的模式，适合
  数学/代码，但开放式对话会死板。

逐位置自适应：教师确定（低熵）时收敛，教师开放（高熵）时保多样。
用 sigmoid 而非硬阈值，避免 τ 附近梯度不连续导致小 batch 震荡。

**E · cosine 而非 MSE**

两侧 hidden 模长分布完全不同（1024 维 vs 512 维，RMSNorm 后统计特性也不同），
MSE 会被尺度差异主导。cosine 只约束方向。

表征项权重必须**衰减**（`rep_decay_to=0.2`）：早期帮学生快速靠近教师表示空间，
后期让分布 KL 主导，因为最终评估看生成质量而非 hidden 相似度。

**G · 在线策略的三条纪律**

1. **λ 渐进**，不是固定 0.5。学生只有 25.83M，早期自采样质量极差，教师在这些
   OOD prefix 上的分布本身不可靠（DAgger 的 interactive-expert 假设失效）。
2. **纯 on-policy（λ=1）绝对不能用**，教师反馈噪声会压过信号。
3. **自采样降频**（每 4 步一次）。生成比前向慢一个量级，每步都做会让训练时间
   翻 3 倍以上。

---

## 快速开始

### 环境

```bash
pip install -r requirements.txt
```

需要本地有：
- 学生模型定义：`D:\workspace\xllm\model\XLLMModel.py`
- 学生基线权重：`D:\workspace\xllm\out\full_sft_512.pth`
- 教师模型：Qwen3-0.6B（路径见 `config.py` 的 `TeacherCfg.model_path`）
- SFT 语料：`D:\workspace\xllm\dataset\sft_t2t_mini.jsonl`

> **关于权重文件**：本仓库**不包含**任何 `.pth` 权重（单个 116MB，超过 GitHub
> 100MB 硬限制，且训练产物不应进版本库）。`out/` 已在 `.gitignore` 中排除。
> 蒸馏产物需按下方命令自行训练生成。

### 训练

```bash
# 小规模验证（8 条样本，2 epoch）
python train.py --limit 8 --epochs 2 --batch_size 2 --grad_accum 1 --max_seq_len 256

# 正式训练
python train.py --epochs 2 --batch_size 2 --grad_accum 4 --max_seq_len 512
```

显存紧张时（8GB 卡）：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py ...
```

预期日志：

```
[词表] 公共 token 5870 | 教师词表 151669 | 学生词表 6400 | 覆盖率 91.7%（相对学生）
[投影器] 伪逆初始化 | 公共 token 5870 | 相对残差 t2s=0.8863 s2t=0.7422 | 理论下界 0.8863
[层映射] 学生 8 层 -> 教师 28 层: [0, 4, 8, 12, 15, 19, 23, 27]
[预热] warmup 50/50 rep_loss 0.2192
  iter 4/4 step 8/8 | loss 4.4954 ce 1.5305 kd 3.1173 rep 0.2315 (w=0.030) lam 0.11 |
  gate[nat 0.42 proj 0.98] H[nat 1.28] tok 105 |
```

`t2s=0.8863` 等于理论下界——说明初始化已达最优，不要再指望调参降低它。
`gate[nat 0.42 proj 0.98]` 的差异正是双路径存在的理由。

### 推理

```bash
python infer.py --weight lens_kd --prompt "用三句话解释什么是知识蒸馏。"
python infer.py --interactive
```

投影器是训练期旁路，**推理只需学生自己的 `.pth`**，部署零额外开销。
（`out/*_proj.pt` 只在做特征分析时需要。）

---

## 目录结构

```
LensKD/
├── config.py           # 全部超参，含双路径融合与各阶段理由
├── hidden_capture.py   # A：hook 抓全层 hidden
├── projector.py        # B：双投影器 + 伪逆初始化 + 下界诊断
├── eta_align.py        # C：ETA 精确 token 对齐
├── divergence.py       # D+E：熵门控 KL + 表征 cosine
├── loss.py             # F：CE + 双路径 KD + 表征
├── data.py             # 双 tokenizer 编码 + 截断 + 在线策略序列
├── train.py            # G：主流程 + 投影器预热 + 在线策略
├── infer.py            # 推理
└── out/                # 权重输出
```

---

## 实现中的坑（都已修，记录在此避免重蹈）

| 问题 | 现象 | 根因与修复 |
|---|---|---|
| **数据全被丢弃** | `4 条样本（跳过 991 条）` | 本语料学生 token 中位 676，`max_seq_len=256` 下 99% 丢弃。改为「保留 prompt 前缀 + 从 assistant 开头截取剩余预算」，保证答案区间非空 |
| **教师侧位置错乱** | KL 无意义 | 截断后必须用 `s_tok.decode(s_ids)` 反解文本再喂教师，否则两侧内容不一致 |
| **skipped 统计错乱** | 只取 4 条却报跳过 991 | `itertools.islice` 预取了 2000 行，改为 `for ln in f` 边读边凑 |
| **CUDA OOM** | `Tried to allocate 138 MiB` | `[B,Lt,151936]` fp32 张量。改为只用 hidden 现算对齐行的 logits + 分块（chunk=64） |
| **dtype 不匹配** | `mat1 and mat2 must have same dtype` | bf16/fp32 混用。`compute_losses` 开头统一转 `.float()` |
| **熵门控失效** | `gate=1.00` 恒定 | 投影抹平教师分布（见上文）。用双路径融合解决 |
| **日志不打印** | 训练跑完无输出 | 日志挂在 `global_step % log_interval`，`grad_accum>1` 时增长慢。改为按 iteration 触发 |
| **`layernotm` 拼写** | 静默漏加载 8 个 norm 权重 | `XLLMModel` 输出键名拼错，`load_state_dict(strict=False)` 不报错。用 `fix_keys()` 修正 |

---

## 创新点与出处

本仓库不是对任何单篇论文的实现，而是**多个上游方法在真实工程约束下的融合与改进**。
下表区分两件事：哪些技术来自上游（附论文链接），哪些是本仓库自己的贡献。

### 一、上游技术来源（附论文链接）

| 技术 | 出处 | 论文 / 代码 |
|---|---|---|
| 双空间投影 + 伪逆初始化 | DSKD | [EMNLP 2024](https://aclanthology.org/2024.emnlp-main.1010/) · [arXiv:2406.17328](https://arxiv.org/abs/2406.17328) · [代码](https://github.com/songmzhang/DSKD) |
| 扩展版（投影器形式化 + ETA 算法） | DSKDv2 | [arXiv:2504.11426](https://arxiv.org/abs/2504.11426) · [代码](https://github.com/songmzhang/DSKDv2) |
| 在线策略蒸馏（GKD） | Agarwal et al. | [ICLR 2024 · arXiv:2306.13649](https://arxiv.org/abs/2306.13649) |
| 反向 KL（mode-seeking）目标 | MiniLLM | [ICLR 2024 · arXiv:2306.08543](https://arxiv.org/abs/2306.08543) |
| 逐 token 前向/反向 KL 混合 | ToDi | [EMNLP 2025 · arXiv:2505.16297](https://aclanthology.org/2025.emnlp-main.409/) |
| 教师熵驱动的门控信号 | EOPD | [ICML 2026 · arXiv:2603.07079](https://arxiv.org/abs/2603.07079) |
| 中间层表征对齐 | TinyBERT | [Findings of EMNLP 2020 · arXiv:1909.10351](https://arxiv.org/abs/1909.10351) |

> **修正说明**：早期版本将 DSKD 标注为 "ACL 2024"，实为 **EMNLP 2024 main**
> （Anthology ID `2024.emnlp-main.1010`）。DSKDv2 是其 2025 年期刊扩展版，
> 首次给出投影器的完整形式化与 ETA 算法，本仓库的 ETA 实现参照的是 DSKDv2。

### 二、本仓库的原创贡献

| # | 贡献 | 对应模块 | 与上游的差异 |
|---|---|---|---|
| **1** | **发现并量化投影抹平效应**：伪逆投影后教师熵从 1.33 升至 7.33（接近均匀分布上界），据此解释熵门控为何恒为 1 而失效 | `projector.py` | DSKD 只报告投影有效，未讨论分布形状退化。这是本仓库的实测发现 |
| **2** | **双路径 KL 融合**：同时保留 native（尖锐但有 13.8% 质量丢失）与 proj（零丢失但被抹平）两条路径，按 `r=0.3` 加权 | `loss.py` | 上游方法只选一条路径。融合后 `kd` 从 7.84 降至 3.26，门控恢复分流能力（native 0.42 / proj 0.98） |
| **3** | **投影残差理论下界的解析计算**：证明 `lstsq` 残差 0.88621 恰等于维度决定的不可约下界，排除"调参可降低残差"的误判 | `projector.py` | DSKD 未给出该下界。工程价值在于：避免把数学必然当成实现 bug 反复调试 |
| **4** | **熵门控在双路径上分别计算**：暴露 proj 路径的门控失真（恒 1.0），使 native 路径的逐位置区分能力可观测 | `divergence.py` | ToDi / EOPD 单路径计算，无法观测这种路径间的门控差异 |
| **5** | **在线策略的三条工程纪律**：λ 从 0.2 渐进到 0.7（非固定值）、自采样降频至每 4 步、实证 λ=1 在 23× 容量鸿沟下不可用 | `train.py`, `data.py` | GKD 假设 interactive-expert 成立；在 596M→25.83M 场景下该假设失效，需上述约束 |
| **6** | **截断策略与教师侧位置一致性**：保留 prompt 前缀 + 从 assistant 起始截取剩余预算；截断后反解文本重编码喂教师 | `data.py` | 解决 99% 样本被丢弃与两侧位置错位的耦合问题，上游未涉及 |

**核心结论一句话**：DSKD 说"统一输出空间就能跨词表蒸馏"，本仓库补充"统一之后分布会被
抹平，所以必须同时保留原生路径"。这是本仓库唯一可被称为新发现的点，其余为工程改进。

---

## 已知局限

1. **容量鸿沟 23 倍**。ACL 2025 证明最优教师规模与学生规模线性相关，596M 对
   25.83M 可能不是最优组合。要验证这点需扫多档教师规模。
2. **投影残差下界 0.886 不可突破**。这是维度决定的，本仓库的做法是"绕开它"
   （native 路径补信号）而不是"解决它"。
3. **在线策略只做 batch 级混合**，未做 token 级的重要性加权。
4. 样本量小时（<100）输出质量差是正常的，趋势正确即可。
5. **双路径权重 `r=0.3` 未做系统扫描**。当前取值基于单点实验，缺乏消融曲线支撑。

---

## License 与署名要求

本项目采用 **[Apache License 2.0](LICENSE)**（自 MIT 迁移而来）。

**为什么不是 MIT**：MIT 以 "without restriction" 授予权利，在其之上附加"必须署名"
会造成条款冲突，法律上不可执行。Apache-2.0 第 4(b)/(c)/(d) 条提供了有约束力的义务，
是能在保持开源属性同时强制署名的标准方案。

### 你必须做的事

若你使用、修改、扩展、再分发或基于 LensKD 构建工作（包括学术研究、商业产品、
衍生开源项目、或作为更大系统的组件），**必须**：

| 要求 | 法律依据 | 具体做法 |
|---|---|---|
| **注明出处** | Apache-2.0 §4(c)、NOTICE | 在文档、论文或产品致谢中写明：项目名 **LensKD**、作者 **PassionCR7**、地址 **https://github.com/PassionCR7/LensKD** |
| **传递 NOTICE** | Apache-2.0 §4(d) | 再分发时附带 [`NOTICE`](NOTICE) 文件的可读副本 |
| **标注修改** | Apache-2.0 §4(b) | 任何被改动的文件须带有醒目的"已修改"声明 |
| **不得冒认** | 著作权法 + §4(c) | 不得将本项目的设计、实现或实验结论声称为自己的原创，不得移除署名 |

### 学术引用

```bibtex
@software{lenskd2026,
  title  = {LensKD: Dual-Projector Cross-Vocabulary Alignment with
            Entropy-Gated Adaptive Divergence for White-Box Distillation},
  author = {PassionCR7},
  year   = {2026},
  url    = {https://github.com/PassionCR7/LensKD}
}
```

机器可读元数据见 [`CITATION.cff`](CITATION.cff)，GitHub 会据此生成 "Cite this repository" 按钮。

### 上游致谢

本项目的技术基础来自 DSKD、GKD、MiniLLM、ToDi、EOPD、TinyBERT 等工作，
完整清单与论文链接见上文 [上游技术来源](#一上游技术来源附论文链接) 及 [`NOTICE`](NOTICE)。
这些是各自作者的独立研究成果，本项目仅在其基础上做增量改进。