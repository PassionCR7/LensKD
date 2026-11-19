"""C 阶段：ETA 精确 token 对齐。

投影器解决了「词表维度不同」，但「序列位置不同」还在：同一段文字被两套
tokenizer 切成不同粒度的 token，同一个下标指的不是同一段内容。

两种对齐策略的差别
------------------
原 whitebox 用**字符区间边界相等**（s_end == t_end），实测命中率 76%~85%。

LensKD 用 **ETA（Exact Token Alignment）**：直接比较 token 解码后的文本，
文本完全相同才算对齐。命中率更高，原因：

    文本 "天空是蓝色的"
    教师切：  ["天空", "是", "蓝色", "的"]
    学生切：  ["天", "空", "是", "蓝", "色", "的"]

    字符边界法：只有 "是"、"的" 命中（"天空" vs "天"+"空" 边界不齐）
    ETA 法    ：同样命中 "是"、"的"，但还能捕获 "蓝色" vs "蓝"+"色" 这类
                两边都有多字符 token 的情形

更重要的是 ETA 支持**一对多**：教师一个 "天空" 与学生两个 "天"+"空" 之间，
可以把教师该位置的分布复制给学生的两个位置（各按字数加权），而不是简单丢弃。
这样覆盖率显著提升。

为什么只在对齐位算 KL
---------------------
`logits[j]` 表示「看完 [0:j] 后预测下一个 token」。只有两边恰好看完同一段
文本时，预测才可比。错位位置硬算 KL 等于用「多看几个字」的分布去教
「少看几个字」的预测，是系统性偏差。

实现约定
--------
* 用 token 的**解码文本**做键，但解码时关闭 clean_up_tokenization，
  否则空格/标点会被规范化，造成假匹配。
* 一对多时，教师的概率按学生侧 token 数均分（因为学生需要连续预测多个 token，
  每个位置都该看到「教师在这个前缀处的意图」）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ETAAlignment:
    """一次对齐的结果。

    Attributes
    ----------
    pairs:
        list[(t_pos, [s_pos, ...])]，教师位置 → 学生位置（可一对多）
    """

    pairs: list[tuple[int, list[int]]]

    @property
    def num_matched_teacher(self) -> int:
        return len(self.pairs)

    @property
    def num_matched_student(self) -> int:
        return sum(len(s) for _, s in self.pairs)


def _decode_pieces(tokenizer, ids: torch.Tensor | list[int]) -> list[str]:
    """逐 token 解码，关闭所有规范化。

    必须逐个解而不是 decode(list)，否则 BPE 会在列表层面再做一次合并，
    拿到的不是单个 token 的文本。
    """
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    out = []
    for i in ids:
        out.append(
            tokenizer.decode(
                [i],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    return out


def build_eta_alignment(
    teacher_pieces: list[str],
    student_pieces: list[str],
) -> ETAAlignment:
    """在两侧 token 文本序列上做单调对齐（保持原顺序，不回退）。

    用双指针 + 贪心匹配，原因：
    * 两侧都是同一段文本的不同切分，顺序天然单调
    * 贪心在单调序列上的结果与最长公共子序列等价，但 O(n) 而非 O(n²)

    一对多的处理：若教师一个 token 的文本等于学生连续 k 个 token 的拼接，
    则建立 1→k 的映射。
    """
    pairs: list[tuple[int, list[int]]] = []
    i = j = 0
    nt, ns = len(teacher_pieces), len(student_pieces)

    while i < nt and j < ns:
        t_txt = teacher_pieces[i]
        s_txt = student_pieces[j]

        # 情形 1：文本完全相同，1→1
        if t_txt == s_txt:
            pairs.append((i, [j]))
            i += 1
            j += 1
            continue

        # 情形 2：教师一个 token == 学生连续多个 token 的拼接（1→k）
        acc = ""
        k = 0
        matched = False
        while j + k < ns and len(acc) < len(t_txt) + 8:   # 长度上限防跑飞
            acc += student_pieces[j + k]
            k += 1
            if acc == t_txt:
                pairs.append((i, list(range(j, j + k))))
                i += 1
                j += k
                matched = True
                break
            if len(acc) > len(t_txt):
                break
        if matched:
            continue

        # 情形 3：学生一个 token == 教师连续多个 token 的拼接（k→1）
        acc = ""
        k = 0
        matched = False
        while i + k < nt and len(acc) < len(s_txt) + 8:
            acc += teacher_pieces[i + k]
            k += 1
            if acc == s_txt:
                pairs.append((i, [j]))     # 只登记教师首位置，避免重复计损失
                i += k
                j += 1
                matched = True
                break
            if len(acc) > len(s_txt):
                break
        if matched:
            continue

        # 情形 4：都不匹配，跳过较短的一侧（贪心推进）
        if len(t_txt) <= len(s_txt):
            i += 1
        else:
            j += 1

    return ETAAlignment(pairs=pairs)


def alignment_index_tensors(
    align: ETAAlignment,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把对齐结果转成可索引张量，便于批量 gather。

    返回
    ----
    t_idx : [M]        教师位置
    s_idx : [M]        对应学生位置（一对多时展开成多行）
    w     : [M]        权重（一对多时 1/k，保证总质量不放大）
    """
    t_idx, s_idx, w = [], [], []
    for tp, s_list in align.pairs:
        k = len(s_list)
        for sp in s_list:
            t_idx.append(tp)
            s_idx.append(sp)
            w.append(1.0 / k)
    return (
        torch.tensor(t_idx, dtype=torch.long, device=device),
        torch.tensor(s_idx, dtype=torch.long, device=device),
        torch.tensor(w, dtype=torch.float32, device=device),
    )