"""数据管线：SFT 语料 → 双 tokenizer 编码 + ETA 对齐索引。

设计要点
--------
1. 教师与学生的 tokenizer 都要编码同一段**纯文本**（chat template 渲染后的字符串），
   而不是各自 apply_chat_template——否则 special token 数量不同，文本就不同了。
2. 对齐只在 assistant 区间有意义（system/user 部分是条件，不是预测目标）。
3. 索引张量在 collate 时按 batch 内最大长度重建，避免在样本层面做 padding。
"""

from __future__ import annotations


import json
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from eta_align import _decode_pieces, build_eta_alignment


@dataclass
class EncodedPair:
    """一条样本的双侧编码结果。"""

    s_ids: list[int]
    t_ids: list[int]
    # assistant 区间（学生侧下标，闭开区间 [start, end)）
    s_ans_start: int
    s_ans_end: int
    # 学生位置 -> (教师位置列表, 权重)
    align: list[tuple[int, list[int]]]


def _render_text(student_tok, conv: list[dict]) -> str:
    """用模板渲染成纯文本。教师侧也编码这同一个字符串。"""
    msgs = [m for m in conv if m.get("role") in ("system", "user", "assistant")]
    try:
        return student_tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # 兜底：手工拼
        parts = []
        for m in msgs:
            parts.append(f"<|{m['role']}|>{m['content']}")
        return "".join(parts)


def _find_answer_span(student_tok, conv: list[dict], s_ids: list[int]) -> tuple[int, int]:
    """定位 assistant 回答在学生 token 序列中的区间。

    做法：分别编码「到最后一个 user 为止」和「完整对话」，前者的长度就是
    assistant 内容的起始位置。比逐 token 匹配 special token 稳健。
    """
    if not any(m.get("role") == "assistant" for m in conv):
        return 0, len(s_ids)

    cut = len(conv)
    for i in range(len(conv) - 1, -1, -1):
        if conv[i].get("role") == "assistant":
            cut = i
            break

    prefix_conv = conv[:cut]
    try:
        prefix_txt = student_tok.apply_chat_template(
            prefix_conv, tokenize=False, add_generation_prompt=True
        )
        prefix_ids = student_tok(prefix_txt, add_special_tokens=False).input_ids
        start = len(prefix_ids)
    except Exception:
        start = 0

    start = min(max(start, 0), len(s_ids))
    return start, len(s_ids)


class LensDataset(Dataset):
    """读取 SFT jsonl，双侧编码，并预先算好 ETA 对齐。"""

    def __init__(self, cfg, student_tok, teacher_tok, limit: int | None = None):
        self.cfg = cfg
        self.s_tok = student_tok
        self.t_tok = teacher_tok
        self.samples: list[EncodedPair] = []

        path = cfg.data.sft_file
        max_len = cfg.data.max_seq_len
        n_target = limit or cfg.data.num_samples

        skipped_long = 0
        truncated = 0
        scanned = 0
        # 注意：这里不能用 islice(f, num_samples) —— num_samples 是「采集目标数」，
        # 而跳过超长样本后实际需要读取更多行。用 while 持续读，直到凑够目标数。
        with open(path, encoding="utf-8") as f:
            for ln in f:
                if len(self.samples) >= n_target:
                    break
                scanned += 1
                try:
                    obj = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                conv = obj.get("conversations") or obj.get("messages")
                if not conv:
                    continue

                text = _render_text(self.s_tok, conv)
                s_full = self.s_tok(text, add_special_tokens=False).input_ids
                if len(s_full) < 8:
                    continue

                # ---- 长度处理 ----
                # 实测本语料学生侧 token 中位数 676，几乎全部超过 max_seq_len。
                # 直接丢弃会让数据集空掉（128 长度下丢弃率 99%+）。
                # 正确做法：保留 prompt 前缀 + 从 assistant 开头截取剩余预算，
                # 这样 assistant 区间始终非空，且语义连续。
                a0_full, a1_full = _find_answer_span(self.s_tok, conv, s_full)
                if len(s_full) > max_len:
                    truncated += 1
                    prompt_len = a0_full
                    budget = max_len - prompt_len
                    if budget < 16:
                        # prompt 本身就吃掉全部预算，退化为「只保留末尾 max_len」
                        s_ids = s_full[-max_len:]
                        a_start = max(0, a0_full - (len(s_full) - max_len))
                        a_end = max_len
                    else:
                        s_ids = s_full[:prompt_len] + s_full[a0_full:a0_full + budget]
                        a_start = prompt_len
                        a_end = len(s_ids)
                else:
                    s_ids = s_full
                    a_start, a_end = a0_full, a1_full

                if a_end <= a_start:
                    continue

                # 教师侧编码同一段**已截断的文本**，保证两侧内容一致。
                # 注意：不能直接用完整 text，否则教师序列比学生长，
                # ETA 对齐会把「学生看不到的尾部」也纳入，位置语义就不对了。
                t_ids_full = self.t_tok(text, add_special_tokens=False).input_ids
                if len(s_full) > max_len:
                    # 用学生截断后的实际文本反解，确保两侧文本完全相同
                    kept_text = self.s_tok.decode(
                        s_ids, skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    t_ids = self.t_tok(kept_text, add_special_tokens=False).input_ids
                else:
                    t_ids = t_ids_full

                # ETA 对齐（训练时只取 assistant 区间）
                s_pieces = _decode_pieces(self.s_tok, s_ids)
                t_pieces = _decode_pieces(self.t_tok, t_ids)
                eta = build_eta_alignment(t_pieces, s_pieces)

                self.samples.append(
                    EncodedPair(
                        s_ids=s_ids,
                        t_ids=t_ids,
                        s_ans_start=a_start,
                        s_ans_end=a_end,
                        align=eta.pairs,
                    )
                )

        self.skipped_long = skipped_long
        self.truncated = truncated
        self.scanned = scanned

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# --------------------------------------------------------------------------- #
# G 阶段：在线策略序列的编码
# --------------------------------------------------------------------------- #
def encode_onpolicy_sequence(
    s_tok,
    t_tok,
    prompt_ids: list[int],
    gen_ids: list[int],
    max_len: int,
) -> EncodedPair | None:
    """把「学生自己生成的序列」编码成 EncodedPair。

    与学生自采样配套使用。关键约束和离线数据完全一致：
    教师必须编码**同一段文本**，否则 ETA 对齐无意义。

    与离线样本的唯一区别是 assistant 区间由我们直接知道：
    它就从 prompt 结束处开始（`len(prompt_ids)`），不用再猜 chat template。
    """
    if not gen_ids:
        return None

    s_ids = list(prompt_ids) + list(gen_ids)
    a_start = len(prompt_ids)

    # 超长时保留 prompt 前缀 + 从 assistant 开头截断（与离线策略同口径）
    if len(s_ids) > max_len:
        budget = max_len - a_start
        if budget < 16:
            s_ids = s_ids[-max_len:]
            a_start = max(0, a_start - (len(prompt_ids) + len(gen_ids) - max_len))
        else:
            s_ids = s_ids[:a_start] + s_ids[a_start:a_start + budget]
    a_end = len(s_ids)

    if a_end - a_start < 2:
        return None

    # 教师侧编码学生实际产出的这段文本（解码回来，保证字节级一致）
    kept_text = s_tok.decode(
        s_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    t_ids = t_tok(kept_text, add_special_tokens=False).input_ids
    if len(t_ids) < 4:
        return None

    s_pieces = _decode_pieces(s_tok, s_ids)
    t_pieces = _decode_pieces(t_tok, t_ids)
    eta = build_eta_alignment(t_pieces, s_pieces)

    return EncodedPair(
        s_ids=s_ids,
        t_ids=t_ids,
        s_ans_start=a_start,
        s_ans_end=a_end,
        align=eta.pairs,
    )


def collate_pairs(batch: list[EncodedPair], s_pad: int, t_pad: int, max_len: int):
    """把一个 batch 拼成张量。

    教师侧长度与学生侧不同，需要各自 padding。
    对齐索引按 token 绝对位置构造，随后在损失里映射到「移位后」的下标。
    """
    B = len(batch)
    Ls = min(max(len(x.s_ids) for x in batch), max_len)
    Lt = min(max(len(x.t_ids) for x in batch), max_len)

    s_ids = torch.full((B, Ls), s_pad, dtype=torch.long)
    t_ids = torch.full((B, Lt), t_pad, dtype=torch.long)
    s_mask = torch.zeros((B, Ls), dtype=torch.bool)      # 有效 token
    t_mask = torch.zeros((B, Lt), dtype=torch.bool)
    ans_mask = torch.zeros((B, Ls), dtype=torch.bool)    # assistant 区间

    for b, x in enumerate(batch):
        ls = min(len(x.s_ids), Ls)
        lt = min(len(x.t_ids), Lt)
        s_ids[b, :ls] = torch.tensor(x.s_ids[:ls], dtype=torch.long)
        t_ids[b, :lt] = torch.tensor(x.t_ids[:lt], dtype=torch.long)
        s_mask[b, :ls] = True
        t_mask[b, :lt] = True
        a0 = min(x.s_ans_start, Ls)
        a1 = min(x.s_ans_end, Ls)
        if a1 > a0:
            ans_mask[b, a0:a1] = True
        else:
            ans_mask[b, 1:] = s_mask[b, 1:]

    # ---- 对齐索引（batch 内扁平化）----
    # 只保留落在有效长度内、且属于 assistant 区间的学生位置
    flat_t, flat_s, flat_w, flat_b = [], [], [], []
    for b, x in enumerate(batch):
        ls = min(len(x.s_ids), Ls)
        lt = min(len(x.t_ids), Lt)
        a0, a1 = x.s_ans_start, min(x.s_ans_end, ls)
        for tp, s_list in x.align:
            if tp >= lt:
                continue
            keep = [sp for sp in s_list if a0 <= sp < a1]
            if not keep:
                continue
            k = len(keep)
            for sp in keep:
                flat_t.append(tp)
                flat_s.append(sp)
                flat_w.append(1.0 / k)
                flat_b.append(b)

    align = {
        "t_idx": torch.tensor(flat_t, dtype=torch.long),
        "s_idx": torch.tensor(flat_s, dtype=torch.long),
        "w": torch.tensor(flat_w, dtype=torch.float32),
        "b_idx": torch.tensor(flat_b, dtype=torch.long),
    }

    return {
        "s_ids": s_ids,
        "t_ids": t_ids,
        "s_mask": s_mask,
        "t_mask": t_mask,
        "ans_mask": ans_mask,
        "align": align,
    }