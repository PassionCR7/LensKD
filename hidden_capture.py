"""A 阶段：编码与特征提取。

xllm 的 XLLMModel.forward 只返回最后一层 hidden_states，中间层直接丢弃。
LensKD 需要**所有层**（表征对齐要用），所以这里用 forward hook 抓取，
不改动 model/XLLMModel.py 一个字符——这样两个目录可以并存、互不污染。

为什么用 hook 而不是改模型
--------------------------
1. 原模型是项目公共资产（distillation_blackbox/whitebox 都在用），改它风险高
2. hook 只读不写，前向行为完全不变，得到的 logits 与原模型一致
3. 需要时可随时摘掉，零副作用

显存注意
--------
抓全层 hidden 会比只留最后一层多吃显存：
    8 层 × B × L × 512 × 2byte (bf16)
    = 8 × 2 × 256 × 512 × 2 ≈ 4.2 MB / 样本
batch=2 时约 8.4MB，可忽略。但如果以后调大 max_seq_len，要留意。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


class HiddenCapture:
    """给 XLLMModel.layers 挂前向 hook，收集每层输出。

    用法：
        cap = HiddenCapture(student.model)
        with cap.capture() as store:
            out = student(input_ids=ids)
        layers = store["hiddens"]     # list of [B, L, H]，长度 = num_layers
    """

    def __init__(self, base_model):
        # base_model 是 XLLMModel（不是 XLLMForCausalLM）
        self.base_model = base_model
        self._store: dict | None = None
        self._handles: list = []

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if self._store is None:
                return
            # XLLMBlock.forward 返回 (hidden_states, present_kv)
            h = output[0] if isinstance(output, tuple) else output
            self._store["hiddens"][layer_idx] = h

        return hook

    @contextmanager
    def capture(self) -> Iterator[dict]:
        layers = self.base_model.layers
        store = {"hiddens": [None] * len(layers)}
        self._store = store
        self._handles = [
            layer.register_forward_hook(self._make_hook(i))
            for i, layer in enumerate(layers)
        ]
        try:
            yield store
        finally:
            for h in self._handles:
                h.remove()
            self._handles = []
            self._store = None

    @staticmethod
    def finalize(store: dict, last_hidden: torch.Tensor) -> torch.Tensor:
        """把 list 拼成 [num_layers, B, L, H]，并补上最终归一化后的那层。

        注意：hook 抓到的是每层 Block 的输出（未过最终 RMSNorm）。
        最后再 append 一层 model.norm 之后的 hidden，它就是 lm_head 的输入，
        也是语义上最有意义的一层，务必纳入。
        """
        lifted = [h for h in store["hiddens"] if h is not None]
        lifted.append(last_hidden)
        return torch.stack(lifted, dim=0)


def collect_teacher_hiddens(model, input_ids, attention_mask=None):
    """教师前向，返回 (hiddens, logits)。

    Qwen3 的 output_hidden_states=True 会返回
    hidden_states[0] = embedding 输出，hidden_states[1:] = 各层输出（最后一层已过 norm）。
    这里去掉 embedding 那层，保证与学生的口径一致（都是「每层 Block 之后」）。
    """
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hs = out.hidden_states
    # hs[0] 是 embedding，丢掉
    layers = torch.stack(list(hs[1:]), dim=0)
    return layers, out.logits