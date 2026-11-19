"""D + E 阶段：熵门控自适应散度，以及表征层对齐。

D. 为什么不能只用前向 KL
------------------------
前向 KL  KL(p_t || p_s) 是 mode-covering：学生在教师有概率的地方都必须有概率，
否则 log(p_t/p_s) 里 p_s→0 时惩罚爆炸。后果是学生分布被拉向教师概率质量的重心，
而多峰分布的重心落在峰与峰之间的低密度区——这正是学生「说话骑墙、什么都沾一点」
的几何成因。

反向 KL  KL(p_s || p_t) 是 mode-seeking：学生塌到教师最有信心的那个模式上。
适合「答案唯一」的场景（数学、代码），但会牺牲多样性，开放式对话会变得死板。

最优方案是**逐位置自适应**（ToDi 2025 / Entropy-Aware OPD 2026）：
教师在该位置确定（低熵）时用反向 KL 收敛，教师开放（高熵）时用前向 KL 保多样。

    gate = sigmoid((H_t - tau) / beta)
    kl   = gate * KL_forward + (1 - gate) * KL_reverse

门控用 sigmoid 而非硬阈值，理由是硬阈值在 tau 附近梯度不连续，
在小 batch 下容易震荡。

E. 为什么还要对齐表征
--------------------
logits 是 hidden 经过 lm_head 压缩后的产物，151936 个标量看似丰富，
但语法结构、指代关系、推理路径这些信息在「映射到词表分布」的过程中已被平均掉。
TinyBERT 时代就证明中间层对齐能带来额外收益。

用 cosine 而非 MSE：两侧 hidden 的模长分布完全不同（教师 1024 维、学生 512 维，
RMSNorm 后统计特性也不同），MSE 会被尺度差异主导，cosine 只约束方向。

表征项权重必须**衰减**。早期它帮学生快速靠近教师表示空间，后期应让分布 KL 主导
（最终评估看的是生成质量，不是 hidden 相似度）。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 自适应散度
# --------------------------------------------------------------------------- #
def entropy_gate(
    logits_teacher: torch.Tensor,
    tau: float = 1.5,
    beta: float = 0.5,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按教师每位置熵算门控权重。

    返回
    ----
    gate : [N]      1 → 前向 KL（保多样），0 → 反向 KL（收敛）
    H    : [N]      教师熵（未归一化到 [0,1]，直接用 nats）
    """
    log_p = F.log_softmax(logits_teacher.float() / temperature, dim=-1)
    H = -(log_p.exp() * log_p).sum(dim=-1)          # [N]
    gate = torch.sigmoid((H - tau) / max(beta, 1e-6))
    return gate, H


def adaptive_kl(
    logits_teacher: torch.Tensor,
    logits_student: torch.Tensor,
    tau: float = 1.5,
    beta: float = 0.5,
    temperature: float = 1.0,
    weights: torch.Tensor | None = None,
    return_parts: bool = False,
):
    """熵门控的逐位置自适应 KL。

    参数
    ----
    logits_teacher, logits_student : [N, V] 同一空间、同一词表
    weights : [N] 额外权重（对齐权重、token mask 等）

    返回
    ----
    loss, (可选) dict 含 gate/kl_fwd/kl_rev 均值，用于监控
    """
    T = temperature
    log_p_t = F.log_softmax(logits_teacher.float() / T, dim=-1)
    log_p_s = F.log_softmax(logits_student.float() / T, dim=-1)

    p_t = log_p_t.exp()
    p_s = log_p_s.exp()

    # 前向 KL(p_t || p_s)：mode-covering
    kl_fwd = (p_t * (log_p_t - log_p_s)).sum(dim=-1)      # [N]
    # 反向 KL(p_s || p_t)：mode-seeking
    kl_rev = (p_s * (log_p_s - log_p_t)).sum(dim=-1)      # [N]

    gate, H = entropy_gate(logits_teacher, tau, beta, temperature)
    kl = gate * kl_fwd + (1.0 - gate) * kl_rev

    if weights is not None:
        w = weights.float()
        kl = kl * w
        denom = w.sum().clamp(min=1e-6)
    else:
        denom = torch.tensor(float(kl.numel()), device=kl.device, dtype=kl.dtype)

    loss = kl.sum() / denom * (T ** 2)

    if return_parts:
        return loss, {
            "gate_mean": float(gate.mean()),
            "kl_fwd_mean": float(kl_fwd.mean()),
            "kl_rev_mean": float(kl_rev.mean()),
            "entropy_mean": float(H.mean()),
        }
    return loss


# --------------------------------------------------------------------------- #
# 表征对齐
# --------------------------------------------------------------------------- #
def layer_mapping(num_teacher: int, num_student: int, mode: str = "relative") -> list[int]:
    """建立 学生层 -> 教师层 的映射。

    relative：按相对深度采样。学生第 i 层对教师第 round(i * L_t / L_s) 层。
              比「固定层号」合理，因为层数不同时，语义阶段（浅层语法、深层语义）
              是按比例对应的，不是按绝对下标。
    uniform ：等间隔，与 relative 基本等价，保留作可读性更好的版本。
    """
    if mode == "uniform":
        step = max(1, num_teacher // num_student)
        return [min(i * step, num_teacher - 1) for i in range(num_student)]

    # relative
    idx = []
    for i in range(num_student):
        j = round(i * (num_teacher - 1) / max(num_student - 1, 1))
        idx.append(int(min(max(j, 0), num_teacher - 1)))
    return idx


def representation_loss(
    h_teacher_proj: torch.Tensor,
    h_student: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """投影后 hidden 的 cosine 距离。

    参数
    ----
    h_teacher_proj : [B, L, d_s]  教师 hidden 已投到学生空间
    h_student      : [B, L, d_s]  学生 hidden
    mask           : [B, L] 只对齐有效位置（通常只看 assistant 区间）

    返回标量：mean(1 - cos_sim)
    """
    h_t = h_teacher_proj.float()
    h_s = h_student.float()

    # 两侧都先 L2 归一化，cosine 等于点积
    h_t = F.normalize(h_t, dim=-1, eps=1e-6)
    h_s = F.normalize(h_s, dim=-1, eps=1e-6)
    cos = (h_t * h_s).sum(dim=-1)              # [B, L]

    if mask is not None:
        m = mask.float()
        return ((1.0 - cos) * m).sum() / m.sum().clamp(min=1e-6)
    return (1.0 - cos).mean()


def rep_weight_schedule(step: int, total_steps: int, w0: float, decay_to: float) -> float:
    """表征项权重线性衰减：w0 → w0 * decay_to。"""
    if total_steps <= 0:
        return w0
    t = min(max(step / total_steps, 0.0), 1.0)
    return w0 * (1.0 - t * (1.0 - decay_to))