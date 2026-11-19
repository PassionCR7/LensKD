"""G 阶段 + 训练主流程。

在线策略迭代（G）
-----------------
off-policy 的根本问题：训练时喂给教师的是数据集里的标准回答，推理时学生
基于**自己写的**上下文继续生成。两个分布不同，误差沿序列累积——一步错，
后面全错，因为学生从没在「自己写错的上下文」里被教过怎么办。

GKD 的解法（DAgger for LLM）：让学生自己生成，教师在这些学生生成的 prefix
上给分布，学生从中学习「犯错了怎么回来」。

本实现的三个关键决策
--------------------
1. **λ 渐进**，不是固定 0.5。学生只有 25.83M，早期自采样质量极差，
   教师在这些 OOD prefix 上的分布本身也不可靠（DAgger 的 interactive-expert
   假设失效）。所以 λ 从 0.2 线性升到 0.7，让模型先学会基本格式再学纠错。

2. **纯 on-policy（λ=1）绝对不能用**。教师反馈的噪声会压过信号。

3. **自采样降频**（每 on_policy_every 步做一次）。生成比前向慢一个量级，
   每步都做会让训练时间翻 3 倍以上，而收益边际递减。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = r"D:\workspace\xllm"
sys.path.insert(0, PROJECT_ROOT)     # 为了 model.XLLMModel
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from config import LensKDConfig                      # noqa: E402
from data import (                                   # noqa: E402
    LensDataset,
    collate_pairs,
    encode_onpolicy_sequence,
)
from divergence import (                              # noqa: E402
    layer_mapping,
    representation_loss,
    rep_weight_schedule,
)
from hidden_capture import HiddenCapture, collect_teacher_hiddens  # noqa: E402
from loss import compute_losses                      # noqa: E402
from projector import (                              # noqa: E402
    DualProjector,
    common_token_indices,
)

from model.XLLMModel import XLLMConfig, XLLMForCausalLM   # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


# --------------------------------------------------------------------------- #
# 学生加载
# --------------------------------------------------------------------------- #
def fix_keys(sd: dict) -> tuple[dict, int]:
    """修正项目 checkpoint 里的 layernotm 拼写错误。

    out/pretrain_512.pth 有 8 个键拼成 post_attention_layernotm.weight。
    直接 load_state_dict(strict=False) 会静默漏加载这些归一化权重，
    模型"加载成功"但行为异常。
    """
    n = 0
    out = {}
    for k, v in sd.items():
        if "layernotm" in k:
            k = k.replace("layernotm", "layernorm")
            n += 1
        out[k] = v
    return out, n


def load_student(cfg: LensKDConfig, device: str):
    s = cfg.student
    lm_cfg = XLLMConfig(
        hidden_size=s.hidden_size,
        num_hidden_layers=s.num_hidden_layers,
        use_moe=s.use_moe,
        vocab_size=s.vocab_size,
    )
    model = XLLMForCausalLM(lm_cfg)

    if s.from_weight and s.from_weight.lower() != "none":
        moe = "_moe" if s.use_moe else ""
        suffix = f"_{s.hidden_size}{moe}"
        stem = s.from_weight if s.from_weight.endswith(suffix) else f"{s.from_weight}{suffix}"
        path = os.path.join(s.weight_dir, f"{stem}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到学生权重: {path}")
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sd, nfix = fix_keys(sd)
        if nfix:
            print(f"[学生] 修正 {nfix} 个键名拼写错误 (layernotm -> layernorm)")
        r = model.load_state_dict(sd, strict=False)
        if r.missing_keys:
            print(f"[学生][警告] 缺失 {len(r.missing_keys)} 个权重")
        else:
            print("[学生] 权重完整加载")
    else:
        print("[学生] 随机初始化")

    model = model.to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"[学生] 参数量 {n/1e6:.2f} M | 词表 {s.vocab_size} | "
          f"hidden {s.hidden_size} | layers {s.num_hidden_layers}")
    return model


# --------------------------------------------------------------------------- #
# 教师加载
# --------------------------------------------------------------------------- #
def load_teacher(cfg: LensKDConfig, device: str):
    dt = torch.bfloat16 if cfg.teacher.dtype == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.teacher.model_path,
        dtype=dt,
        trust_remote_code=True,
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in model.parameters())
    print(f"[教师] {cfg.teacher.model_name if hasattr(cfg.teacher, 'model_name') else 'Qwen3'} "
          f"参数量 {n/1e6:.1f} M | dtype {cfg.teacher.dtype}")
    return model


# --------------------------------------------------------------------------- #
# 自采样（G 阶段）
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_student(model, s_tok, prompt_ids: torch.Tensor, device, max_new: int, T: float):
    """让当前学生自由生成，返回生成后的完整 token 序列（含 prompt）。"""
    model.eval()
    out = model.generate(
        input_ids=prompt_ids,
        attention_mask=torch.ones_like(prompt_ids),
        max_new_tokens=max_new,
        do_sample=T > 0,
        temperature=max(T, 1e-6),
        top_p=0.9,
        pad_token_id=s_tok.pad_token_id,
        eos_token_id=s_tok.eos_token_id,
    )
    model.train()
    return out


def make_onpolicy_batch(
    student,
    s_tok,
    t_tok,
    ds: LensDataset,
    device: str,
    cfg,
    n: int,
    rng: random.Random,
) -> dict | None:
    """抽 n 条样本，让学生自己续写 answer 部分，再组成一个训练 batch。

    流程（GKD / DAgger for LLM 的核心）：
      1. 取离线样本的 prompt 部分（assistant 之前）
      2. 学生自由生成 → 得到「学生自己会写的」answer
      3. 教师在这段**学生生成的**文本上给分布
      4. 用它算 KD → 学生学会「走错之后怎么回来」

    第 2 步用采样而非贪心：贪心只能看到一条路径，采样的多样性才能
    覆盖多种犯错方式。
    """
    if len(ds) == 0:
        return None

    picked = []
    tries = 0
    while len(picked) < n and tries < n * 4:
        tries += 1
        x = ds[rng.randrange(len(ds))]
        if x.s_ans_start < 4 or x.s_ans_start >= len(x.s_ids):
            continue

        prompt_ids = torch.tensor([x.s_ids[:x.s_ans_start]], dtype=torch.long, device=device)
        # 生成长度 = 离线答案长度，但不设过大上限，控制耗时
        max_new = min(max(x.s_ans_end - x.s_ans_start, 8), cfg.train.on_policy_max_new)

        gen = sample_student(
            student, s_tok, prompt_ids, device,
            max_new=max_new, T=cfg.train.sample_temperature,
        )
        gen_ids = gen[0, prompt_ids.shape[1]:].tolist()
        if not gen_ids:
            continue

        pair = encode_onpolicy_sequence(
            s_tok, t_tok, prompt_ids[0].tolist(), gen_ids, cfg.data.max_seq_len
        )
        if pair is not None:
            picked.append(pair)

    if not picked:
        return None
    return collate_pairs(picked, s_tok.pad_token_id, t_tok.pad_token_id,
                         cfg.data.max_seq_len)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="LensKD 白盒蒸馏（双投影器 + 熵门控 + 在线策略）")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--learning_rate", type=float, default=None)
    ap.add_argument("--max_seq_len", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--kd_weight", type=float, default=None)
    ap.add_argument("--ce_weight", type=float, default=None)
    ap.add_argument("--rep_weight", type=float, default=None)
    ap.add_argument("--entropy_tau", type=float, default=None)
    ap.add_argument("--no_on_policy", action="store_true", help="关闭学生自采样")
    ap.add_argument("--proj_init", type=str, default=None,
                    choices=["pseudo_inverse", "identity_like", "random"])
    ap.add_argument("--save_weight", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    cfg = LensKDConfig()
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.grad_accum is not None:
        cfg.train.grad_accum = args.grad_accum
    if args.learning_rate is not None:
        cfg.train.learning_rate = args.learning_rate
    if args.max_seq_len is not None:
        cfg.data.max_seq_len = args.max_seq_len
    if args.num_samples is not None:
        cfg.data.num_samples = args.num_samples
    if args.kd_weight is not None:
        cfg.train.kd_weight = args.kd_weight
    if args.ce_weight is not None:
        cfg.train.ce_weight = args.ce_weight
    if args.rep_weight is not None:
        cfg.train.rep_weight = args.rep_weight
    if args.entropy_tau is not None:
        cfg.train.entropy_tau = args.entropy_tau
    if args.no_on_policy:
        cfg.train.on_policy = False
    if args.proj_init is not None:
        cfg.projector.init = args.proj_init
    if args.save_weight is not None:
        cfg.student.save_weight = args.save_weight

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    print("=" * 72)
    print("LensKD | 双投影器 + ETA 对齐 + 熵门控自适应散度 + 在线策略")
    print(f"  ce_w={cfg.train.ce_weight} kd_w={cfg.train.kd_weight} "
          f"rep_w={cfg.train.rep_weight}")
    print(f"  T_student_space={cfg.train.temp_student_space} "
          f"T_teacher_space={cfg.train.temp_teacher_space}")
    print(f"  entropy_tau={cfg.train.entropy_tau}  on_policy={cfg.train.on_policy}")
    print("=" * 72)

    # ---- tokenizer ----
    s_tok = AutoTokenizer.from_pretrained(cfg.student.model_dir, trust_remote_code=True)
    t_tok = AutoTokenizer.from_pretrained(cfg.teacher.model_path, trust_remote_code=True)

    # ---- 模型 ----
    student = load_student(cfg, device)
    teacher = load_teacher(cfg, device)

    # ---- 双投影器 ----
    proj = DualProjector(
        d_t=cfg.teacher.hidden_size,
        d_s=cfg.student.hidden_size,
        init=cfg.projector.init,
        norm_after=cfg.projector.norm_after,
    ).to(device)

    # 公共 token 索引：两个用途
    #   1) 伪逆初始化投影器
    #   2) native 路径的 KL（教师原生分布限制在可映射子集上）
    tv = dict(t_tok.get_vocab())
    sv = dict(s_tok.get_vocab())
    t_ids, s_ids = common_token_indices(tv, sv)
    native_map = (
        t_ids.to(device, non_blocking=True),
        s_ids.to(device, non_blocking=True),
    )
    print(f"[词表] 公共 token {t_ids.numel()} | 教师词表 {len(tv)} "
          f"| 学生词表 {len(sv)} | 覆盖率 "
          f"{t_ids.numel()/len(sv)*100:.1f}%（相对学生）")

    if cfg.projector.init == "pseudo_inverse":
        diag = proj.init_from_heads(
            teacher.lm_head.weight.detach(),
            student.lm_head.weight.detach(),
            t_ids.cpu(), s_ids.cpu(),
        )
        print(f"[投影器] 伪逆初始化 | 公共 token {diag['common_tokens']} | "
              f"相对残差 t2s={diag['rel_residual_t2s']:.4f} "
              f"s2t={diag['rel_residual_s2t']:.4f} "
              f"| 理论下界 {diag['theoretical_lb_t2s']:.4f}")
    else:
        print(f"[投影器] {cfg.projector.init} 初始化")

    if not cfg.projector.trainable:
        for p in proj.parameters():
            p.requires_grad_(False)

    # ---- 数据 ----
    t0 = time.time()
    ds = LensDataset(cfg, s_tok, t_tok, limit=args.limit)
    print(f"[数据] {len(ds)} 条样本（扫描 {ds.scanned} 行，截断 {ds.truncated} 条），"
          f"加载耗时 {time.time()-t0:.1f}s")

    def _collate(batch):
        return collate_pairs(batch, s_tok.pad_token_id, t_tok.pad_token_id,
                             cfg.data.max_seq_len)

    loader = DataLoader(
        ds, batch_size=cfg.train.batch_size, shuffle=True,
        collate_fn=_collate, num_workers=0, drop_last=False,
    )

    # ---- 优化器 ----
    student_params = [p for p in student.parameters() if p.requires_grad]
    proj_params = [p for p in proj.parameters() if p.requires_grad]
    param_groups = [{"params": student_params, "lr": cfg.train.learning_rate}]
    if proj_params:
        param_groups.append({"params": proj_params, "lr": cfg.train.projector_lr})
    opt = torch.optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)

    # ---- 教师权重与层映射（固定）----
    w_teacher = teacher.lm_head.weight.detach()
    w_student = student.lm_head.weight.detach()
    layer_map = layer_mapping(
        num_teacher=len(teacher.model.layers),
        num_student=cfg.student.num_hidden_layers,
        mode=cfg.train.rep_schedule,
    )
    print(f"[层映射] 学生 {cfg.student.num_hidden_layers} 层 -> 教师 "
          f"{len(teacher.model.layers)} 层: {layer_map}")

    # ---- 抓取学生全层 hidden ----
    cap = HiddenCapture(student.model)

    # ------------------------------------------------------------------ #
    # 投影器预热：只训 projector，不算 CE/KD，把表征先对齐
    # ------------------------------------------------------------------ #
    if cfg.projector.trainable and cfg.train.proj_warmup_steps > 0:
        print(f"\n[预热] 投影器预热 {cfg.train.proj_warmup_steps} 步"
              f"（只训 projector，不含 CE/KD）")
        warm_opt = torch.optim.AdamW(
            proj.parameters(), lr=cfg.train.proj_warmup_lr, weight_decay=0.0
        )
        # 学生参数冻结，只让投影器学
        for p in student.parameters():
            p.requires_grad_(False)
        student.eval()

        warm_iter = iter(loader)
        for ws in range(cfg.train.proj_warmup_steps):
            try:
                batch = next(warm_iter)
            except StopIteration:
                warm_iter = iter(loader)
                batch = next(warm_iter)

            s_ids_w = batch["s_ids"].to(device)
            t_ids_w = batch["t_ids"].to(device)
            align_w = {k: v.to(device) if torch.is_tensor(v) else v
                       for k, v in batch["align"].items()}

            with torch.no_grad():
                t_hid, _ = collect_teacher_hiddens(teacher, t_ids_w)
            t_hid = t_hid.float()

            with torch.no_grad():
                with cap.capture() as store:
                    s_out_w = student(input_ids=s_ids_w)
                s_all = HiddenCapture.finalize(store, s_out_w.hidden_states).float()

            # 逐层 cosine 对齐（只优化 projector）
            sb = align_w["b_idx"]
            sp = align_w["s_idx"]
            tp = align_w["t_idx"]
            if sb.numel() == 0:
                continue

            losses = []
            for li, tj in enumerate(layer_map):
                if li >= s_all.shape[0]:
                    break
                h_t = t_hid[tj]                          # [B,Lt,d_t]
                h_t_proj = proj.t2s(h_t)                 # [B,Lt,d_s]
                a = h_t_proj[sb, tp]
                b = s_all[li][sb, sp]
                losses.append(representation_loss(a, b))
            if not losses:
                continue
            wl = torch.stack(losses).mean()
            warm_opt.zero_grad(set_to_none=True)
            wl.backward()
            torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
            warm_opt.step()

            if (ws + 1) % max(1, cfg.train.proj_warmup_steps // 5) == 0:
                print(f"  warmup {ws+1}/{cfg.train.proj_warmup_steps} "
                      f"rep_loss {wl.item():.4f}", flush=True)

        for p in student.parameters():
            p.requires_grad_(True)
        student.train()
        print("[预热] 完成，开始蒸馏\n")

    # ---- 训练循环 ----
    steps_per_epoch = (len(loader) + cfg.train.grad_accum - 1) // cfg.train.grad_accum
    total_steps = steps_per_epoch * cfg.train.epochs
    global_step = 0
    t_start = time.time()
    history = []
    rng = random.Random(cfg.train.seed)
    onpolicy_batches = 0

    student.train()
    for epoch in range(cfg.train.epochs):
        ep_loss = ep_ce = ep_kd = ep_rep = ep_lam = 0.0
        nb = 0
        opt.zero_grad(set_to_none=True)

        for it, batch in enumerate(loader):
            s_ids = batch["s_ids"].to(device)
            t_ids_b = batch["t_ids"].to(device)
            batch_d = {
                k: (v.to(device) if torch.is_tensor(v) else
                    {kk: vv.to(device) for kk, vv in v.items()})
                for k, v in batch.items()
            }

            # ---- 教师前向（no_grad，全层 hidden）----
            # 注意：默认不保留完整 logits —— [B,Lt,151936] 的 fp32 张量
            # 在 batch=2/L=256 时就有 300MB+，8GB 卡上很容易 OOM。
            # loss 里改为「按需用 hidden 现算对齐行的 logits」，
            # 数值上完全等价（同一个 lm_head、同一个 hidden），但显存降一个量级。
            with torch.no_grad():
                t_hiddens, _ = collect_teacher_hiddens(teacher, t_ids_b)
            t_out = {
                "teacher_hiddens": t_hiddens.float(),
                "teacher_logits": None,     # 由 loss 内部按需计算
                "layer_map": layer_map,
                "native_map": native_map,   # (教师 id, 学生 id)，native 路径用
            }

            # ---- 学生前向（抓全层）----
            with cap.capture() as store:
                s_out = student(input_ids=s_ids)
            all_hidden = HiddenCapture.finalize(store, s_out.hidden_states)
            s_out.all_hidden = all_hidden.float()

            # ---- 表征权重（在线策略与主损失共用同一个值）----
            rep_w = rep_weight_schedule(
                global_step, total_steps,
                cfg.train.rep_weight, cfg.train.rep_decay_to,
            )

            # ---- 在线策略：用学生自采样序列替换部分样本 ----
            lam = 0.0
            outs_op = None
            if (cfg.train.on_policy
                    and global_step > 0
                    and global_step % cfg.train.on_policy_every == 0):
                prog = global_step / max(total_steps, 1)
                lam = cfg.train.lam_start + (
                    cfg.train.lam_end - cfg.train.lam_start
                ) * prog

                op = make_onpolicy_batch(
                    student, s_tok, t_tok, ds, device, cfg,
                    n=max(1, cfg.train.batch_size // 2),
                    rng=rng,
                )
                if op is not None:
                    onpolicy_batches += 1
                    # λ 混合：离线 batch 与自采样 batch 各算一次损失后加权。
                    # 不做样本级拼接，因为两侧长度分布差异大，拼一起会让
                    # padding 比例失衡（自采样序列通常短得多）。
                    op_device = {
                        k: (v.to(device) if torch.is_tensor(v) else
                            {kk: vv.to(device) for kk, vv in v.items()})
                        for k, v in op.items()
                    }
                    with torch.no_grad():
                        t_hid_op, _ = collect_teacher_hiddens(
                            teacher, op_device["t_ids"]
                        )
                    t_out_op = {
                        "teacher_hiddens": t_hid_op.float(),
                        "teacher_logits": None,
                        "layer_map": layer_map,
                        "native_map": native_map,
                    }
                    with cap.capture() as store_op:
                        s_out_op = student(input_ids=op_device["s_ids"])
                    s_out_op.all_hidden = HiddenCapture.finalize(
                        store_op, s_out_op.hidden_states
                    ).float()

                    outs_op = compute_losses(
                        s_out_op, t_out_op, proj, w_teacher, w_student,
                        op_device, cfg, rep_w,
                    )
                else:
                    lam = 0.0

            # ---- 损失（离线 + λ·在线）----
            outs = compute_losses(
                s_out, t_out, proj, w_teacher, w_student,
                batch_d, cfg, rep_w,
            )
            loss = outs["loss"]
            if outs_op is not None and lam > 0:
                # 混合 loss 与 λ 线性插值等价：
                #   (1-λ)·L_off + λ·L_on = L_off + λ·(L_on - L_off)
                # 这里用直接加权形式，保持梯度尺度稳定
                loss = (1.0 - lam) * outs["loss"] + lam * outs_op["loss"]
            (loss / cfg.train.grad_accum).backward()

            ep_loss += float(loss.detach())
            ep_ce += outs["ce"]
            ep_kd += outs["kd"]
            ep_rep += outs["rep"]
            ep_lam += lam
            nb += 1

            if (it + 1) % cfg.train.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    student_params + proj_params, cfg.train.grad_clip
                )
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
            elif (it + 1) == len(loader):
                # 尾批不足 grad_accum 时必须补一次 step，否则样本量小
                # （如 4 条样本 + grad_accum=4 之外的余数）时末尾梯度被丢弃，
                # 严重时整个 epoch 一次更新都不发生。
                torch.nn.utils.clip_grad_norm_(
                    student_params + proj_params, cfg.train.grad_clip
                )
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

            # 日志按 iteration 触发（grad_accum>1 时 global_step 增长慢）
            if (it + 1) % cfg.train.log_interval == 0 or (it + 1) == len(loader):
                el = time.time() - t_start
                print(
                    f"  iter {it+1}/{len(loader)} step {global_step}/{total_steps} | "
                    f"loss {ep_loss/nb:.4f} ce {ep_ce/nb:.4f} "
                    f"kd {ep_kd/nb:.4f} rep {ep_rep/nb:.4f} "
                    f"(w={rep_w:.3f}) lam {ep_lam/nb:.2f} | "
                    f"gate[nat {outs.get('n_gate_mean', float('nan')):.2f} "
                    f"proj {outs.get('s_gate_mean', float('nan')):.2f}] "
                    f"H[nat {outs.get('n_entropy_mean', float('nan')):.2f}] "
                    f"tok {outs.get('aligned_tokens', 0)} | "
                    f"{el:.0f}s",
                    flush=True,
                )
                history.append({
                    "step": global_step,
                    "loss": ep_loss / nb,
                    "ce": ep_ce / nb,
                    "kd": ep_kd / nb,
                    "rep": ep_rep / nb,
                    "lam": ep_lam / nb,
                })
                ep_loss = ep_ce = ep_kd = ep_rep = ep_lam = 0.0
                nb = 0

        print(f"Epoch {epoch+1}/{cfg.train.epochs} 完成 | "
              f"耗时 {time.time()-t_start:.0f}s "
              f"| 在线策略 batch {onpolicy_batches} 个")

    # ---- 保存 ----
    os.makedirs(cfg.student.save_dir, exist_ok=True)
    save_path = os.path.join(
        cfg.student.save_dir, f"{cfg.student.save_weight}_{cfg.student.hidden_size}.pth"
    )
    state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
    torch.save(state, save_path)
    print(f"[保存] {save_path}")

    # 投影器也存一份（infer 需要）
    proj_path = os.path.join(
        cfg.student.save_dir, f"{cfg.student.save_weight}_proj.pt"
    )
    torch.save(
        {
            "p_t2s": proj.p_t2s.weight.detach().cpu(),
            "p_s2t": proj.p_s2t.weight.detach().cpu(),
            "d_t": proj.d_t, "d_s": proj.d_s,
            "norm_after": proj.norm_after,
            "layer_map": layer_map,
        },
        proj_path,
    )
    print(f"[保存] {proj_path}")

    with open(os.path.join(cfg.student.save_dir, "train_history.json"), "w",
              encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"[总耗时] {(time.time()-t_start)/60:.1f} 分钟")


if __name__ == "__main__":
    main()