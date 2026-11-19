"""F 阶段：联合损失。

    L = ce_weight  · CE(student_logits, hard_labels)          # 保基础语言能力
      + kd_weight  · [KL_dual_space × T²]                     # 分布对齐（双向）
      + rep_weight · L_representation                         # 表征对齐

三个容易踩的点
--------------
1. **CE 权重不能压太低**。学生只有 25.83M，纯 KD 会让它遗忘基础语言能力
   （catastrophic forgetting）。建议 ce:kd 不高于 1:2。

2. **两个空间的温度必须分开**。学生空间 6400 维、教师空间 151936 维，
   分布尖锐程度差一个量级。用同一个 T 会让其中一个过平滑或过尖锐。

3. **对齐后必须做移位**。位置 i 的 logits 预测 i+1，所以 logits 用 [:-1]，
   标签用 [1:]。对齐索引里的位置也必须同步位移，否则整体错开一位
   ——这个坑在原 whitebox 实现里真实发生过（t_mask 用了 [:, 1:] 而
   logits 用了 [:, :-1]，导致 KL 虚高 27%）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from divergence import adaptive_kl, representation_loss


def gather_aligned_logits(
    logits: torch.Tensor,
    b_idx: torch.Tensor,
    pos_idx: torch.Tensor,
) -> torch.Tensor:
    """从 [B, L, V] 按 (batch, position) 取出对应行，得到 [M, V]。"""
    if b_idx.numel() == 0:
        return logits.new_zeros((0, logits.shape[-1]))
    return logits[b_idx, pos_idx]


def compute_losses(
    s_out,
    t_out,
    proj,
    w_teacher: torch.Tensor,
    w_student: torch.Tensor,
    batch: dict,
    cfg,
    rep_w: float,
) -> dict:
    """计算全部损失项。

    参数
    ----
    s_out : 学生输出，需含 logits 与最后一层 hidden
    t_out : dict，含 teacher_hiddens [L_t,B,Lt,d_t] 与 teacher_logits [B,Lt,V_t]
    proj  : DualProjector
    rep_w : 当前步的表征项权重（由 schedule 给出，已考虑衰减）
    """
    s_ids = batch["s_ids"]
    t_ids = batch["t_ids"]
    ans_mask = batch["ans_mask"]
    s_mask = batch["s_mask"]
    t_mask = batch["t_mask"]
    align = batch["align"]

    s_logits = s_out.logits              # [B, Ls, V_s]
    s_hidden = s_out.hidden_states       # [B, Ls, d_s]（最后一层，已过 norm）

    # 统一精度：学生侧统一转 float32 计算，教师权重也按需转。
    # 混用 bf16/fp32 会直接报 dtype 不匹配；而全用 bf16 会让 KL 累积误差
    # （原 whitebox 实现也是统一 float32 算 KL，理由相同）。
    w_teacher = w_teacher.float()
    w_student = w_student.float()
    s_logits = s_logits.float()
    s_hidden = s_hidden.float()

    # ------------------------------------------------------------------ #
    # 1. CE（硬标签）
    # ------------------------------------------------------------------ #
    shift_labels = s_ids[:, 1:].contiguous()
    shift_s_logits = s_logits[:, :-1, :].contiguous()
    # 只在 assistant 区间算 CE：位置 i 的 logits 预测 i+1，所以 mask 也要移位
    shift_ans = ans_mask[:, 1:].contiguous()
    ce_flat = F.cross_entropy(
        shift_s_logits.view(-1, shift_s_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.shape)
    ce = (ce_flat * shift_ans).sum() / shift_ans.sum().clamp(min=1)

    # ------------------------------------------------------------------ #
    # 2. 双空间自适应 KL
    # ------------------------------------------------------------------ #
    t_hidden_full = t_out["teacher_hiddens"]     # [L_t, B, Lt, d_t]
    # 注意：不使用 t_out["teacher_logits"]。教师空间 logits 由 hidden 现算，
    # 避免一次性 materialize [B, Lt, 151936]。

    # 用最后一层教师 hidden（即 lm_head 的输入）做投影
    h_t_last = t_hidden_full[-1]                 # [B, Lt, d_t]

    kd_total = s_logits.new_zeros(())
    kd_parts = {}
    T_s = cfg.train.temp_student_space
    T_t = cfg.train.temp_teacher_space

    b_idx = align["b_idx"]
    s_pos = align["s_idx"]
    t_pos = align["t_idx"]
    w_align = align["w"]

    if b_idx.numel() > 0:
        # 位置对齐 + 移位：位置 i 预测 i+1
        # 对齐里登记的是"token 位置"，我们用它来预测下一个 token，
        # 因此取该位置本身的 logits（不需要额外移位），但必须保证 i+1 有效。
        valid_s = s_pos + 1 < s_ids.shape[1]
        valid_t = t_pos + 1 < t_ids.shape[1]
        keep = valid_s & valid_t
        sb, sp, tp, ww = b_idx[keep], s_pos[keep], t_pos[keep], w_align[keep]

        if sb.numel() > 0:
            # ================= native 路径（词表对齐子集）=================
            # 这条路用教师**原生** logits 在可映射子集上算 KL。
            # 它有 13.8% 质量丢失（在子集上重归一化），但分布是真实尖锐的。
            # 与 proj 路径互补：proj 零丢失但被抹平，native 有丢失但信号强。
            native_map = t_out.get("native_map")     # (t_idx_full, s_idx_full)
            kl_native = None
            if native_map is not None and cfg.train.kd_proj_ratio < 1.0:
                map_t, map_s = native_map
                chunk = 64
                kl_n_sum, cnt_n = 0.0, 0
                g_n, f_n, r_n, h_n = 0.0, 0.0, 0.0, 0.0
                for st_i in range(0, sb.numel(), chunk):
                    sl = slice(st_i, st_i + chunk)
                    sb_c, sp_c, tp_c, ww_c = sb[sl], sp[sl], tp[sl], ww[sl]

                    # 教师原生 logits（只算需要的行）
                    lg_t_nat = h_t_last[sb_c, tp_c] @ w_teacher.t()   # [m, V_t]
                    # 学生 logits 在教师词表的对齐子集上取值
                    lg_s_nat = s_logits[sb_c, sp_c][:, map_s]         # [m, S_common]
                    lg_t_sub = lg_t_nat[:, map_t]                     # [m, S_common]

                    kl_c, pc = adaptive_kl(
                        lg_t_sub, lg_s_nat,
                        tau=cfg.train.entropy_tau,
                        beta=cfg.train.entropy_beta,
                        temperature=cfg.train.temp_native,
                        weights=ww_c,
                        return_parts=True,
                    )
                    m = sb_c.numel()
                    kl_n_sum += float(kl_c.detach()) * m
                    cnt_n += m
                    g_n += pc["gate_mean"] * m
                    f_n += pc["kl_fwd_mean"] * m
                    r_n += pc["kl_rev_mean"] * m
                    h_n += pc["entropy_mean"] * m
                    del lg_t_nat, lg_s_nat, lg_t_sub, kl_c
                    torch.cuda.empty_cache()

                if cnt_n > 0:
                    kl_native = kl_n_sum / cnt_n
                    kd_parts.update({
                        "n_gate_mean": g_n / cnt_n,
                        "n_entropy_mean": h_n / cnt_n,
                        "n_kl_fwd_mean": f_n / cnt_n,
                        "n_kl_rev_mean": r_n / cnt_n,
                    })

            # ================= proj 路径（双投影，零丢失）=================
            # 学生空间只需 [m, 6400]，显存压力小，但仍按块处理保持一致
            chunk = 128
            kl_s_sum, cnt_s = 0.0, 0
            g_s, f_s, r_s, h_s = 0.0, 0.0, 0.0, 0.0
            for st_i in range(0, sb.numel(), chunk):
                sl = slice(st_i, st_i + chunk)
                sb_c, sp_c, tp_c, ww_c = sb[sl], sp[sl], tp[sl], ww[sl]

                h_t_rows = h_t_last[sb_c, tp_c]                  # [m, d_t]
                h_s_rows = s_hidden[sb_c, sp_c]                  # [m, d_s]
                lg_t_rows = proj.t2s(h_t_rows) @ w_student.t()   # [m, 6400]
                lg_s_rows = h_s_rows @ w_student.t()             # [m, 6400]

                kl_c, pc = adaptive_kl(
                    lg_t_rows, lg_s_rows,
                    tau=cfg.train.entropy_tau,
                    beta=cfg.train.entropy_beta,
                    temperature=T_s,
                    weights=ww_c,
                    return_parts=True,
                )
                m = sb_c.numel()
                kl_s_sum += float(kl_c.detach()) * m
                cnt_s += m
                g_s += pc["gate_mean"] * m
                f_s += pc["kl_fwd_mean"] * m
                r_s += pc["kl_rev_mean"] * m
                h_s += pc["entropy_mean"] * m
                del lg_t_rows, lg_s_rows, kl_c

            if cnt_s > 0:
                kd_total = kd_total + kd_total.new_tensor(kl_s_sum / cnt_s)
                kd_parts.update({
                    "s_gate_mean": g_s / cnt_s,
                    "s_kl_fwd_mean": f_s / cnt_s,
                    "s_kl_rev_mean": r_s / cnt_s,
                    "s_entropy_mean": h_s / cnt_s,
                })

        # ---- 教师空间（151936 维）----
        # 注意：这条路在 151936 维上对**教师侧也**做一次 log_softmax，
        # 代价高（每 token ~600KB 中间量）且其信息已被 native 路径覆盖
        # （native 就是教师原生分布 + 学生限制在子集）。
        # 因此默认关闭，仅在需要「完整双空间」时开启（cfg.train.use_teacher_space）。
        if sb.numel() > 0 and getattr(cfg.train, "use_teacher_space", False):
            chunk = 64
            kl_t_sum = 0.0
            kl_t_cnt = 0
            gate_acc, fwd_acc, rev_acc, H_acc = 0.0, 0.0, 0.0, 0.0
            for st_i in range(0, sb.numel(), chunk):
                sl = slice(st_i, st_i + chunk)
                sb_c, sp_c, tp_c, ww_c = sb[sl], sp[sl], tp[sl], ww[sl]

                # 只对需要的行做投影与前向，避免 [B,Lt,1024]->[B,Lt,151936]
                h_s_rows = s_hidden[sb_c, sp_c]              # [m, d_s]
                h_t_rows = h_t_last[sb_c, tp_c]              # [m, d_t]
                lg_s_rows = proj.s2t(h_s_rows) @ w_teacher.t()   # [m, V_t]
                lg_t_rows = h_t_rows @ w_teacher.t()             # [m, V_t]

                kl_c, pc = adaptive_kl(
                    lg_t_rows, lg_s_rows,
                    tau=cfg.train.entropy_tau,
                    beta=cfg.train.entropy_beta,
                    temperature=T_t,
                    weights=ww_c,
                    return_parts=True,
                )
                m = sb_c.numel()
                kl_t_sum += float(kl_c.detach()) * m
                kl_t_cnt += m
                gate_acc += pc["gate_mean"] * m
                fwd_acc += pc["kl_fwd_mean"] * m
                rev_acc += pc["kl_rev_mean"] * m
                H_acc += pc["entropy_mean"] * m

                # 立即释放，避免下一块叠加
                del lg_s_rows, lg_t_rows, kl_c
                torch.cuda.empty_cache()

            if kl_t_cnt > 0:
                kl_t_avg = kl_t_sum / kl_t_cnt
                kd_total = kd_total + kd_total.new_tensor(kl_t_avg)
                kd_parts.update({
                    "t_gate_mean": gate_acc / kl_t_cnt,
                    "t_kl_fwd_mean": fwd_acc / kl_t_cnt,
                    "t_kl_rev_mean": rev_acc / kl_t_cnt,
                    "t_entropy_mean": H_acc / kl_t_cnt,
                })

        kd_parts["aligned_tokens"] = int(sb.numel())

        # ---- 两条 KD 路径按比例融合 ----
        # proj 路径：零质量丢失，但 1024->512 投影把分布抹平（实测熵 1.33→7.33）
        # native 路径：有 13.8% 质量丢失，但教师分布保持真实尖锐
        # 融合权重默认 0.3 : 0.7（偏向 native），理由见 config.py 的注释。
        r = float(getattr(cfg.train, "kd_proj_ratio", 0.5))
        kl_proj = kd_total if kd_total.numel() > 0 else None
        if kl_native is not None:
            kd = (1.0 - r) * kd_total.new_tensor(kl_native) + r * kd_total
        else:
            kd = kd_total
        kd_parts["kd_proj_ratio"] = r
    else:
        kd = s_logits.new_zeros(())
        kd_parts["aligned_tokens"] = 0

    # ------------------------------------------------------------------ #
    # 3. 表征对齐（按相对深度选层）
    # ------------------------------------------------------------------ #
    rep = s_logits.new_zeros(())
    if rep_w > 0 and t_out.get("layer_map") is not None:
        layer_map = t_out["layer_map"]              # 学生层 -> 教师层
        h_t_sel = torch.stack(
            [t_hidden_full[j] for j in layer_map], dim=0
        )                                           # [L_s, B, Lt, d_t]
        # 教师侧投影到学生空间后与学生逐层比较
        B_, Lt_, dt_ = h_t_sel.shape[1], h_t_sel.shape[2], h_t_sel.shape[3]
        h_t_flat = h_t_sel.reshape(-1, Lt_, dt_)
        h_t_proj = proj.t2s(h_t_flat).reshape(len(layer_map), B_, Lt_, -1)

        # 学生侧全部层（含最后 norm 之后那层）
        s_layers = s_out.all_hidden                  # [L_s, B, Ls, d_s]

        # 教师与学生序列长度不同，需按 ETA 对齐位置后再比
        if sb.numel() > 0:
            reps = []
            for li in range(min(h_t_proj.shape[0], s_layers.shape[0])):
                h_ti = h_t_proj[li]                                  # [B,Lt,d_s]
                h_si = s_layers[li]                                  # [B,Ls,d_s]
                a = gather_aligned_logits(h_ti, sb, tp)
                b = gather_aligned_logits(h_si, sb, sp)
                reps.append(representation_loss(a, b))
            if reps:
                rep = torch.stack(reps).mean()

    # ------------------------------------------------------------------ #
    # 4. 组装
    # ------------------------------------------------------------------ #
    tr = cfg.train
    total = tr.ce_weight * ce + tr.kd_weight * kd + rep_w * rep

    return {
        "loss": total,
        "ce": float(ce.detach()),
        "kd": float(kd.detach()),
        "rep": float(rep.detach()) if torch.is_tensor(rep) else 0.0,
        "rep_w": rep_w,
        **kd_parts,
    }