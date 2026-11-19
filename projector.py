"""B 阶段：双投影器统一输出空间。

这是 LensKD 最关键的一步。它把「跨词表」这个约束整个消掉。

问题回顾
--------
教师分布定义在 151936 维，学生定义在 6400 维。原实现从教师里挑出
5870 个能映射的维度，再在子集上重新归一化——实测丢失 13.8% 概率质量，
11.28% 的位置上教师 top-1 被抹掉。

DSKD 的洞察
-----------
分布维度不同，根因是两个模型的 **prediction head 不同**。那就不对齐词表，
而是对齐「谁在用哪个 head」：

    教师空间：W_t @ P_t2s 作用于学生 hidden   → 与 W_t @ h_t 同空间
    学生空间：W_s @ h_s                        → 与 W_s @ P_s2t @ h_t 同空间

于是两边都能在**完整**词表维度上算 KL，零过滤、零丢质量。

理想初始化（关键，不能随机）
----------------------------
投影器的初值来自一个最小二乘问题。设公共 token 集合为 S：
    想让学生 head 复现教师在这些 token 上的输出：
        W_s[S] @ P_t2s ≈ W_t[S]
    ⇒ P_t2s = pinv(W_s[S]) @ W_t[S]        [d_s, d_t]

同理：
        W_t[S] @ P_s2t ≈ W_s[S]
    ⇒ P_s2t = pinv(W_t[S]) @ W_s[S]        [d_t, d_s]

随机初始化会让投影器一开始输出噪声，训练前若干步纯粹在学「投影本身」而不是
学蒸馏信号。伪逆初始化让投影器开局就「近似正确」，梯度可以直接优化蒸馏目标。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# byte-level BPE 词表 → 原始字节
# --------------------------------------------------------------------------- #
def _bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_U2B = {v: k for k, v in _bytes_to_unicode().items()}


def token_to_bytes(token: str) -> bytes:
    try:
        return bytes(_U2B[ch] for ch in token)
    except KeyError:
        return token.encode("utf-8", "ignore")


def common_token_indices(
    teacher_vocab: dict[str, int],
    student_vocab: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回公共 token 的 (教师 id, 学生 id) 张量对。

    注意：这里的对齐**只用于构造投影器初值**，不参与训练损失。
    所以即使某些 token 没对上也无所谓——投影器是可学的，初值不完美会被后续训练修正。
    这与「在子集上算 KL」有本质区别。
    """
    t_by_bytes: dict[bytes, int] = {}
    for tok, tid in sorted(teacher_vocab.items(), key=lambda kv: kv[1]):
        b = token_to_bytes(tok)
        if b and b not in t_by_bytes:
            t_by_bytes[b] = tid

    t_ids, s_ids = [], []
    for tok, sid in student_vocab.items():
        b = token_to_bytes(tok)
        tid = t_by_bytes.get(b)
        if tid is not None:
            t_ids.append(tid)
            s_ids.append(sid)

    return torch.tensor(t_ids, dtype=torch.long), torch.tensor(s_ids, dtype=torch.long)


# --------------------------------------------------------------------------- #
# 投影器
# --------------------------------------------------------------------------- #
class DualProjector(nn.Module):
    """教师 <-> 学生 表示空间的双向线性投影。

    参数
    ----
    d_t : 教师 hidden 维度
    d_s : 学生 hidden 维度
    init : "pseudo_inverse" | "identity_like" | "random"
    norm_after : 投影后是否 RMSNorm（两侧尺度差异大时必须开）
    """

    def __init__(
        self,
        d_t: int,
        d_s: int,
        init: str = "pseudo_inverse",
        norm_after: bool = True,
    ):
        super().__init__()
        self.d_t, self.d_s = d_t, d_s
        self.norm_after = norm_after

        # P_t2s: [d_s, d_t]  教师 → 学生空间
        self.p_t2s = nn.Linear(d_t, d_s, bias=False)
        # P_s2t: [d_t, d_s]  学生 → 教师空间
        self.p_s2t = nn.Linear(d_s, d_t, bias=False)

        if init == "identity_like":
            self._init_identity_like()
        elif init == "random":
            self._init_random()
        # pseudo_inverse 需要外部权重，构造后再调 init_from_heads()

    # ---- 初始化方式 ----
    def _init_random(self):
        nn.init.normal_(self.p_t2s.weight, std=0.02)
        nn.init.normal_(self.p_s2t.weight, std=0.02)

    def _init_identity_like(self):
        """截断/补零恒等：min(d_t,d_s) 维直接对应，其余置零。"""
        with torch.no_grad():
            d = min(self.d_t, self.d_s)
            self.p_t2s.weight.zero_()
            self.p_s2t.weight.zero_()
            self.p_t2s.weight[:d, :d] = torch.eye(d)
            self.p_s2t.weight[:d, :d] = torch.eye(d)
            # 缩放补偿：投影后维度不同，用 sqrt 比例稳住范数
            scale = (self.d_s / self.d_t) ** 0.5
            self.p_t2s.weight.mul_(scale)
            self.p_s2t.weight.mul_(1.0 / scale)

    @torch.no_grad()
    def init_from_heads(
        self,
        w_teacher: torch.Tensor,
        w_student: torch.Tensor,
        t_ids: torch.Tensor,
        s_ids: torch.Tensor,
    ) -> dict:
        """用两侧 lm_head 在公共 token 上做最小二乘，求伪逆初值。

        参数
        ----
        w_teacher : [V_t, d_t]   教师 lm_head
        w_student : [V_s, d_s]   学生 lm_head
        t_ids, s_ids : 公共 token 的两侧 id

        返回诊断信息（残差可以拿来判断初始化质量）。
        """
        device = self.p_t2s.weight.device
        w_t = w_teacher[t_ids.to(w_teacher.device)].float()      # [S, d_t]
        w_s = w_student[s_ids.to(w_student.device)].float()      # [S, d_s]

        # P_t2s: 解 W_s @ P = W_t  →  P = pinv(W_s) @ W_t   [d_s, d_t]
        p_t2s = torch.linalg.lstsq(w_s, w_t).solution            # [d_s, d_t]
        # P_s2t: 解 W_t @ P = W_s  →  P = pinv(W_t) @ W_s   [d_t, d_s]
        p_s2t = torch.linalg.lstsq(w_t, w_s).solution            # [d_t, d_s]

        # nn.Linear.weight 形状是 [out, in]，正好对应 [d_s, d_t] 与 [d_t, d_s]
        self.p_t2s.weight.copy_(p_t2s.to(device))
        self.p_s2t.weight.copy_(p_s2t.to(device))

        # 残差诊断
        res_t2s = (w_s @ p_t2s - w_t).norm() / w_t.norm().clamp(min=1e-8)
        res_s2t = (w_t @ p_s2t - w_s).norm() / w_s.norm().clamp(min=1e-8)

        # 重要认识：t2s 的残差存在**数学下界**，不可能降到 0。
        # 原因：W_t 是 1024 维满秩（奇异值 14.5 到 0.49，秩 1024），
        # 而 W_s 只有 512 维（秩 512）。用 512 维空间去复现 1024 维的内容，
        # 最小二乘解已经是最优，其残差等于「W_t 在 W_s 张成子空间外的正交分量」
        # 占 W_t 的比例。实测该下界 = 0.886，与 lstsq 结果完全一致。
        #
        # 结论：投影器初值无法"完美"，只能"理论最优"。
        # 因此训练初期需要给投影器单独预热（见 train.py 的 proj_warmup），
        # 让它在真实数据分布上继续收敛，而不是指望初始化一次到位。
        return {
            "common_tokens": int(t_ids.numel()),
            "rel_residual_t2s": float(res_t2s),
            "rel_residual_s2t": float(res_s2t),
            # 后者是理论下界，用于判断初始化是否已达到最优
            "theoretical_lb_t2s": self._projection_lower_bound(w_s, w_t),
        }

    @staticmethod
    def _projection_lower_bound(w_s: torch.Tensor, w_t: torch.Tensor) -> float:
        """W_t 投影到 W_s 行空间后的相对残差下界。

        若实际 lstsq 残差与此值相等，说明初始化已经最优，没有改进空间。
        """
        try:
            q, _ = torch.linalg.qr(w_s)
            proj = q @ (q.t() @ w_t)
            return float((proj - w_t).norm() / w_t.norm().clamp(min=1e-8))
        except Exception:
            return float("nan")

    # ---- 前向 ----
    def t2s(self, h_t: torch.Tensor) -> torch.Tensor:
        """教师 hidden [B,L,d_t] -> 学生空间 [B,L,d_s]。"""
        h = self.p_t2s(h_t)
        if self.norm_after:
            h = F.rms_norm(h, (h.shape[-1],))
        return h

    def s2t(self, h_s: torch.Tensor) -> torch.Tensor:
        """学生 hidden [B,L,d_s] -> 教师空间 [B,L,d_t]。"""
        h = self.p_s2t(h_s)
        if self.norm_after:
            h = F.rms_norm(h, (h.shape[-1],))
        return h


# --------------------------------------------------------------------------- #
# 双空间 logits
# --------------------------------------------------------------------------- #
def dual_space_logits(
    proj: DualProjector,
    h_t: torch.Tensor,
    h_s: torch.Tensor,
    w_teacher: torch.Tensor,
    w_student: torch.Tensor,
) -> dict:
    """产出两个空间上的 (教师分布, 学生分布) 对。

    学生空间（6400 维）：
        logits_t = h_t_proj @ W_s^T ,  logits_s = h_s @ W_s^T
    教师空间（151936 维）：
        logits_t = h_t @ W_t^T      ,  logits_s = h_s_proj @ W_t^T

    两个空间都是**完整词表**，没有任何维度过滤。
    """
    # ---- 学生空间 ----
    h_t_in_s = proj.t2s(h_t)
    lg_t_stu_space = h_t_in_s @ w_student.t()
    lg_s_stu_space = h_s @ w_student.t()

    # ---- 教师空间 ----
    h_s_in_t = proj.s2t(h_s)
    lg_t_tea_space = h_t @ w_teacher.t()
    lg_s_tea_space = h_s_in_t @ w_teacher.t()

    return {
        "student_space": (lg_t_stu_space, lg_s_stu_space),
        "teacher_space": (lg_t_tea_space, lg_s_tea_space),
    }