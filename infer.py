"""LensKD 推理：加载蒸馏后的学生权重做对话生成。

用法
----
    python infer.py
    python infer.py --weight lens_kd --prompt "介绍一下你自己"
    python infer.py --interactive

注意：LensKD 训练时**没有改动学生结构**（投影器是旁路，只在训练期存在），
所以推理只需要学生自己的 .pth，不需要 _proj.pt。这是双投影器方案的一个
额外好处：部署端零额外开销。
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = r"D:\workspace\xllm"
sys.path.insert(0, PROJECT_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from config import LensKDConfig                      # noqa: E402
from model.XLLMModel import XLLMConfig, XLLMForCausalLM   # noqa: E402
from transformers import AutoTokenizer               # noqa: E402


def fix_keys(sd: dict) -> tuple[dict, int]:
    """修正 checkpoint 里的 layernotm 拼写错误（与 train.py 一致）。"""
    n = 0
    out = {}
    for k, v in sd.items():
        if "layernotm" in k:
            k = k.replace("layernotm", "layernorm")
            n += 1
        out[k] = v
    return out, n


def load_model(weight: str, device: str, cfg: LensKDConfig):
    s = cfg.student
    lm_cfg = XLLMConfig(
        hidden_size=s.hidden_size,
        num_hidden_layers=s.num_hidden_layers,
        use_moe=s.use_moe,
        vocab_size=s.vocab_size,
    )
    model = XLLMForCausalLM(lm_cfg)

    # weight 支持 "lens_kd"（本项目产物）或 "full_sft"（基线）等
    path = os.path.join(s.save_dir, f"{weight}_{s.hidden_size}.pth")
    if not os.path.exists(path):
        path = os.path.join(s.weight_dir, f"{weight}_{s.hidden_size}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到权重: {weight}")

    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd, nfix = fix_keys(sd)
    if nfix:
        print(f"[权重] 修正 {nfix} 个键名 (layernotm -> layernorm)")
    r = model.load_state_dict(sd, strict=False)
    if r.missing_keys:
        print(f"[警告] 缺失 {len(r.missing_keys)} 个权重: {r.missing_keys[:3]}...")

    model = model.to(device).eval()
    print(f"[权重] {path}")
    return model


def build_prompt(tok, text: str, system: str | None) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": text})
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"<|user|>{text}<|assistant|>"


@torch.no_grad()
def generate(
    model, tok, prompt: str, device: str,
    max_new: int = 256, temperature: float = 0.7,
    top_p: float = 0.9, repetition_penalty: float = 1.3,
) -> str:
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-6),
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    gen = out[0, ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser(description="LensKD 推理")
    ap.add_argument("--weight", type=str, default="lens_kd",
                    help="权重名：lens_kd（蒸馏产物）或 full_sft（基线）")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--repetition_penalty", type=float, default=1.3)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = LensKDConfig()
    tok = AutoTokenizer.from_pretrained(cfg.student.model_dir, trust_remote_code=True)
    model = load_model(args.weight, device, cfg)

    demos = [
        "用三句话解释什么是知识蒸馏。",
        "帮我把这句话改得更简洁：我觉得这个事情可能大概应该是这样做的。",
        "写一个 Python 函数，判断一个数是不是质数。",
    ]

    if args.interactive:
        print("\n输入 quit 退出\n")
        while True:
            try:
                q = input("你: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if not q:
                continue
            p = build_prompt(tok, q, None)
            a = generate(model, tok, p, device, args.max_new,
                         args.temperature, repetition_penalty=args.repetition_penalty)
            print(f"模型: {a}\n")
        return

    prompts = [args.prompt] if args.prompt else demos
    for q in prompts:
        p = build_prompt(tok, q, None)
        a = generate(model, tok, p, device, args.max_new,
                     args.temperature, repetition_penalty=args.repetition_penalty)
        print("=" * 72)
        print(f"[问] {q}")
        print(f"[答] {a}")
    print("=" * 72)


if __name__ == "__main__":
    main()