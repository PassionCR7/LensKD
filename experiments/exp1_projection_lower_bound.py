"""实验一：验证投影残差理论下界（正文 README 中"0.88621 恰等于理论下界"这一结论）。

结论待验证
----------
README 声称：
  * lstsq 实际残差 = 0.88621
  * 该值恰等于「W_t 在 W_s 行空间外的正交分量」占比所决定的理论下界
  * 所以初始化已达最优，不存在降低残差的调参空间

本脚本要回答三个问题
--------------------
Q1  实际最小二乘残差是多少？是否等于 0.88621（容差 1e-3）？
Q2  它是否等于三种独立算法给出的理论下界？
       (a) QR 正交投影残差
       (b) SVD 截断重构残差（取前 d_s 个奇异值）
       (c) 秩亏修正后的下界 sqrt(1 - sum_{i<=d_s} s_i^2 / sum_i s_i^2)
Q3  残差随数据规模 S（公共 token 数）如何变化？是否随 S 增大而收敛？

同时跑消融：换随机初始化后残差是多少，验证"伪逆初始化已最优"的论断。

用法
----
    python experiments/exp1_projection_lower_bound.py
    python experiments/exp1_projection_lower_bound.py --device cuda --out experiments/results

输出
----
    experiments/results/exp1_projection_lower_bound.json   结构化结果
    stdout                                                 人类可读报告
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TeacherCfg, StudentCfg                      # noqa: E402
from projector import DualProjector, common_token_indices      # noqa: E402


# --------------------------------------------------------------------------- #
# 数据准备
# --------------------------------------------------------------------------- #
def load_student_lm_head(weight_path: str) -> torch.Tensor:
    """从学生 .pth 中取出 lm_head.weight，形状 [V_s, d_s]。"""
    sd = torch.load(weight_path, map_location="cpu", weights_only=True)
    key = "lm_head.weight"
    if key not in sd:
        raise KeyError(f"{weight_path} 中没有 {key}，实际键：{list(sd)[:10]}")
    return sd[key].float()


def load_teacher_lm_head(model_dir: str) -> torch.Tensor:
    """从 safetensors 中流式取出 lm_head.weight，形状 [V_t, d_t]。"""
    from safetensors import safe_open

    p = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(p):
        # 分片存储时按索引找
        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"找不到权重：{model_dir}")
        import json as _json
        wmap = _json.load(open(idx_path))["weight_map"]
        p = os.path.join(model_dir, wmap["lm_head.weight"])

    with safe_open(p, framework="pt") as f:
        return f.get_tensor("lm_head.weight").float()


# --------------------------------------------------------------------------- #
# 三种独立算法求理论下界
# --------------------------------------------------------------------------- #
def lower_bound_qr(w_s: torch.Tensor, w_t: torch.Tensor) -> float:
    """(a) QR 分解：把 W_t 正交投影到 W_s 的列空间，算剩余范数占比。

    W_s 形状 [S, d_s]。QR 得 Q [S, k]。投影 = Q Qᵀ W_t。
    残差 = ||(I - Q Qᵀ) W_t|| / ||W_t||。
    """
    q, _ = torch.linalg.qr(w_s)              # [S, k], k = min(S, d_s)
    proj = q @ (q.t() @ w_t)
    return float((proj - w_t).norm() / w_t.norm().clamp(min=1e-12))


def lower_bound_svd(w_s: torch.Tensor, w_t: torch.Tensor) -> float:
    """(b) SVD 截断重构：用 W_s 的主子空间近似 W_t。

    取 W_s @ pinv(W_s) 作为投影算子，等价于保留前 rank 个奇异向量。
    数值上与 QR 应一致，用作交叉验证。
    """
    proj = w_s @ (torch.linalg.pinv(w_s) @ w_t)
    return float((proj - w_t).norm() / w_t.norm().clamp(min=1e-12))


def lower_bound_spectrum(w_s: torch.Tensor, w_t: torch.Tensor) -> float:
    """(c) 谱分解闭式：残差² = 1 - Σ_{i≤d_s} σ_i² / Σ_all σ_i²。

    这里 σ_i 是 W_t 的奇异值。前提是 W_s 满秩且其列空间能覆盖 W_t 的
    前 d_s 个主方向——这个前提在随机高维下只是近似，所以此式给出的是
    *谱意义上的期望值*，与 (a)(b) 可能有小幅差异。差异本身值得报告。
    """
    sv = torch.linalg.svdvals(w_t.float())   # 降序
    d = min(w_s.shape[1], w_t.shape[1])
    total = (sv ** 2).sum()
    head = (sv[:d] ** 2).sum()
    return float(torch.sqrt((1 - head / total).clamp(min=0)))


# --------------------------------------------------------------------------- #
# 主实验
# --------------------------------------------------------------------------- #
def run(args) -> dict:
    t0 = time.time()
    device = torch.device(args.device)

    print("=" * 78)
    print("实验一：投影残差理论下界验证")
    print("=" * 78)

    # ---- 1. 载入两侧 lm_head ----
    tcfg, scfg = TeacherCfg(), StudentCfg()
    print(f"\n[载入] 教师 lm_head: {tcfg.model_path}")
    w_t_full = load_teacher_lm_head(tcfg.model_path)
    print(f"       形状 {tuple(w_t_full.shape)}  (V_t={w_t_full.shape[0]}, d_t={w_t_full.shape[1]})")

    sp = os.path.join(scfg.weight_dir, f"{scfg.from_weight}_512.pth")
    print(f"[载入] 学生 lm_head: {sp}")
    w_s_full = load_student_lm_head(sp)
    print(f"       形状 {tuple(w_s_full.shape)}  (V_s={w_s_full.shape[0]}, d_s={w_s_full.shape[1]})")

    # ---- 2. 求公共 token ----
    import json as _json
    t_vocab = _json.load(open(os.path.join(tcfg.model_path, "vocab.json"), encoding="utf-8"))
    s_vocab = {}
    s_tok_path = os.path.join(scfg.model_dir, "tokenizer.json")
    s_tok = _json.load(open(s_tok_path, encoding="utf-8"))
    s_vocab = s_tok.get("model", {}).get("vocab", s_vocab)
    if not s_vocab:
        raise RuntimeError(f"{s_tok_path} 中找不到 model.vocab")

    t_ids, s_ids = common_token_indices(t_vocab, s_vocab)
    S = int(t_ids.numel())
    print(f"\n[词表] 公共 token S = {S} | 教师 {len(t_vocab)} | 学生 {len(s_vocab)}"
          f" | 覆盖率 {S/len(s_vocab)*100:.1f}%（相对学生）")

    # ---- 3. 秩与条件数诊断 ----
    print("\n[诊断] lm_head 的秩与谱")
    for name, w in [("教师 W_t", w_t_full), ("学生 W_s", w_s_full)]:
        sv = torch.linalg.svdvals(w)
        r = int((sv > sv[0] * 1e-6).sum())
        print(f"       {name}: 形状 {tuple(w.shape)}, 数值秩 {r}, "
              f"σ_max={sv[0]:.4f}, σ_min={sv[-1]:.4f}, cond={float(sv[0]/sv[-1]):.2f}")

    # ---- 4. 取公共子集，转 device ----
    w_t = w_t_full[t_ids].to(device)          # [S, d_t]
    w_s = w_s_full[s_ids].to(device)          # [S, d_s]
    print(f"\n[子集] W_t[S] {tuple(w_t.shape)} | W_s[S] {tuple(w_s.shape)}")

    # ---- 5. Q1: 实际最小二乘残差 ----
    print("\n" + "-" * 78)
    print("Q1  实际最小二乘残差")
    print("-" * 78)

    p_t2s = torch.linalg.lstsq(w_s, w_t).solution          # [d_s, d_t]
    res_t2s = (w_s @ p_t2s - w_t).norm() / w_t.norm()

    p_s2t = torch.linalg.lstsq(w_t, w_s).solution          # [d_t, d_s]
    res_s2t = (w_t @ p_s2t - w_s).norm() / w_s.norm()

    print(f"  t2s (教师->学生, 1024->512) 残差 = {float(res_t2s):.6f}")
    print(f"  s2t (学生->教师, 512->1024) 残差 = {float(res_s2t):.6f}")
    print(f"  README 声称的 t2s 值 = 0.88621")
    delta = abs(float(res_t2s) - 0.88621)
    print(f"  与声称值之差 = {delta:.6f}  "
          f"({'✅ 一致' if delta < 1e-3 else '❌ 不一致'})")

    # ---- 6. Q2: 三种算法求下界 ----
    print("\n" + "-" * 78)
    print("Q2  理论下界（三种独立算法）")
    print("-" * 78)

    lb_qr = lower_bound_qr(w_s, w_t)
    lb_svd = lower_bound_svd(w_s, w_t)
    lb_spec = lower_bound_spectrum(w_s, w_t)

    print(f"  (a) QR 正交投影残差      = {lb_qr:.6f}")
    print(f"  (b) SVD 伪逆投影残差     = {lb_svd:.6f}")
    print(f"  (c) 谱闭式下界           = {lb_spec:.6f}")
    print(f"  实际 lstsq 残差          = {float(res_t2s):.6f}")
    print()
    for tag, lb in [("(a) QR", lb_qr), ("(b) SVD", lb_svd), ("(c) 谱", lb_spec)]:
        d = abs(float(res_t2s) - lb)
        print(f"  |lstsq - {tag}| = {d:.8f}  "
              f"({'✅ 相等（差<1e-5）' if d < 1e-5 else '⚠ 有差异'})")

    # ---- 7. Q3: 残差随 S 变化 ----
    print("\n" + "-" * 78)
    print("Q3  残差随公共 token 数 S 的变化")
    print("-" * 78)
    print(f"  {'S':>7}  {'lstsq_res':>12}  {'QR_lb':>12}  {'gap':>10}")
    sweep = []
    for frac in args.sweep:
        n = max(int(S * frac), w_s.shape[1] + 10)
        if n > S:
            continue
        idx = torch.randperm(S, device=device)[:n]
        wt_sub, ws_sub = w_t[idx], w_s[idx]
        r = float((ws_sub @ torch.linalg.lstsq(ws_sub, wt_sub).solution - wt_sub).norm()
                  / wt_sub.norm())
        lb = lower_bound_qr(ws_sub, wt_sub)
        sweep.append({"S": n, "lstsq_res": r, "qr_lb": lb, "gap": abs(r - lb)})
        print(f"  {n:>7}  {r:>12.6f}  {lb:>12.6f}  {abs(r-lb):>10.2e}")

    # ---- 8. 消融：三种初始化对比 ----
    print("\n" + "-" * 78)
    print("消融  三种初始化的残差（验证伪逆初始化已最优）")
    print("-" * 78)

    d_t, d_s = w_t.shape[1], w_s.shape[1]
    ablations = {}
    for init in ["pseudo_inverse", "identity_like", "random"]:
        proj = DualProjector(d_t=d_t, d_s=d_s, init=init, norm_after=False).to(device)
        if init == "pseudo_inverse":
            proj.init_from_heads(w_t_full, w_s_full, t_ids, s_ids)
        # 统一用同一个定义度量：||W_s[S] P - W_t[S]|| / ||W_t[S]||
        # 这样三种初始化完全可比，不受各自内部实现差异影响
        P = proj.p_t2s.weight.detach()                       # [d_s, d_t]
        r = float((w_s @ P - w_t).norm() / w_t.norm().clamp(min=1e-12))
        ablations[init] = r
        print(f"  {init:<16} 相对残差 = {r:.6f}")

    # ---- 9. 汇总 ----
    elapsed = time.time() - t0
    print("\n" + "=" * 78)
    print(f"完成，用时 {elapsed:.1f}s")
    print("=" * 78)

    result = {
        "experiment": "exp1_projection_lower_bound",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(elapsed, 2),
        "device": str(device),
        "config": {
            "teacher_vocab": int(w_t_full.shape[0]),
            "teacher_dim": int(w_t_full.shape[1]),
            "student_vocab": int(w_s_full.shape[0]),
            "student_dim": int(w_s_full.shape[1]),
            "common_tokens": S,
        },
        "q1_actual_residual": {
            "t2s": float(res_t2s),
            "s2t": float(res_s2t),
            "readme_claimed": 0.88621,
            "abs_diff": delta,
            "matches_claim": delta < 1e-3,
        },
        "q2_lower_bounds": {
            "qr": lb_qr,
            "svd": lb_svd,
            "spectrum": lb_spec,
            "lstsq": float(res_t2s),
            "gaps": {
                "qr": abs(float(res_t2s) - lb_qr),
                "svd": abs(float(res_t2s) - lb_svd),
                "spectrum": abs(float(res_t2s) - lb_spec),
            },
        },
        "q3_sweep": sweep,
        "ablation_init": ablations,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="experiments/results")
    ap.add_argument("--sweep", nargs="*", type=float,
                    default=[0.1, 0.25, 0.5, 0.75, 1.0])
    args = ap.parse_args()

    res = run(args)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "exp1_projection_lower_bound.json"
    p.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果已写入 {p}")


if __name__ == "__main__":
    main()