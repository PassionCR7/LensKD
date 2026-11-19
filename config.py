"""LensKD：白盒蒸馏的理论最优形态。

设计目标
--------
不复用「在教师 151936 维里挑 5870 维算 KL」这条路线——那条路实测丢失
13.8% 的教师概率质量、11.28% 的位置上教师 top-1 被重归一化抹掉。

LensKD 的做法是**绕过词表问题**：用双投影器把两侧 hidden state 投到对方
表示空间，共享同一个 lm_head，让分布天然在同一维度上。于是：

* 没有 token 过滤 → 没有概率质量丢失
* 没有子集重归一化 → 不存在「教师意图被抹掉」
* 侧向还能做表征对齐（logits 已将中间层信息平均掉，hidden 没有）

七个阶段对应模块
----------------
A 编码与特征提取   → hidden_capture.py （hook 抓全层 hidden）
B 双投影器         → projector.py     （P_t2s / P_s2t + 伪逆初始化）
C ETA 精确对齐     → eta_align.py     （token 文本严格相等的双射）
D 熵门控散度       → divergence.py    （低熵反向KL / 高熵前向KL）
E 表征层对齐       → divergence.py    （投影后 cosine）
F 联合损失         → loss.py
G 在线策略迭代     → train.py         （学生自采样回灌，λ 渐进）
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
PROJECT_ROOT = r"D:\workspace\xllm"
LENS_ROOT = r"D:\workspace\LensKD"


# --------------------------------------------------------------------------- #
# 教师
# --------------------------------------------------------------------------- #
@dataclass
class TeacherCfg:
    """教师：Qwen3-0.6B，全精度前向（量化会系统偏移 logits）。"""

    model_path: str = os.environ.get(
        "XLLM_TEACHER_PATH",
        r"D:\models\models\Qwen--Qwen3-0.6B\snapshots\master",
    )
    dtype: str = "bfloat16"
    # 教师 hidden 维度，Qwen3-0.6B 是 1024
    hidden_size: int = 1024
    num_layers: int = 28
    vocab_size: int = 151936


# --------------------------------------------------------------------------- #
# 学生
# --------------------------------------------------------------------------- #
@dataclass
class StudentCfg:
    """学生：xllm Dense 512，25.83M。"""

    model_dir: str = os.path.join(PROJECT_ROOT, "model")
    weight_dir: str = os.path.join(PROJECT_ROOT, "out")
    from_weight: str = "full_sft"
    hidden_size: int = 512
    num_hidden_layers: int = 8
    vocab_size: int = 6400
    use_moe: bool = False
    save_dir: str = os.path.join(LENS_ROOT, "out")
    save_weight: str = "lens_kd"


# --------------------------------------------------------------------------- #
# 投影器
# --------------------------------------------------------------------------- #
@dataclass
class ProjectorCfg:
    """双投影器配置。

    init 三种模式：
      * "pseudo_inverse" —— 从两侧 lm_head 的最小二乘解求初值（论文做法，推荐）
      * "identity_like"  —— 截断/补零恒等，退化为「直接对齐可比维度」
      * "random"         —— 随机初始化（仅作消融基线）
    """

    init: str = "pseudo_inverse"
    # 投影后是否做 RMSNorm，避免两侧 hidden 尺度差异主导 cosine
    norm_after: bool = True
    # 投影器是否参与训练（True=可学，False=冻结）
    trainable: bool = True


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
@dataclass
class DataCfg:
    # 与 whitebox 共用同一份 SFT 语料，保证可比
    sft_file: str = os.path.join(PROJECT_ROOT, "dataset", "sft_t2t_mini.jsonl")
    num_samples: int = 2000
    max_seq_len: int = 256
    # 只统计 assistant 区间的损失（与标准 SFT 口径一致）
    answer_only: bool = True


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
@dataclass
class TrainCfg:
    epochs: int = 2
    batch_size: int = 2
    grad_accum: int = 4
    learning_rate: float = 5e-5
    projector_lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    dtype: str = "bfloat16"
    log_interval: int = 10
    seed: int = 42

    # ---- 损失权重 ----
    ce_weight: float = 1.0
    kd_weight: float = 1.0
    rep_weight: float = 0.1

    # ---- 双路径 KD 融合 ----
    # 实测发现：教师原生熵 1.33（尖锐），经 1024->512 投影后再过学生 head，
    # 熵升到 7.33（上界 8.76），即分布被抹平成接近均匀。
    # 直接后果是熵门控恒为 1（全走前向 KL），且 KD 值虚高无信息。
    #
    # 根因：教师 lm_head 是 1024 维满秩，学生只有 512 维，用一半容量复现
    # 1024 维的判别信息在数学上不可能（t2s 残差下界 0.886 已证明）。
    #
    # 因此 LensKD 保留**两条 KD 路径**并融合：
    #   proj 路径：投影 + 共享 head，零质量丢失，但分布被抹平
    #   native 路径：原始词表对齐（仅用于取对齐子集），分布真实尖锐
    # 融合权重 kd_proj_ratio 给两者加权。默认偏向 native（0.3/0.7），
    # 因为尖锐的教师信号远比「无丢失但均匀」的信号有用。
    kd_proj_ratio: float = 0.3
    # native 路径的温度（与学生空间一致，因为它作用在学生词表上）
    temp_native: float = 1.2

    # 教师空间（151936 维）那条 KL 支路。
    # 默认关闭：它的信息已被 native 路径覆盖（native 就是「教师原生分布 +
    # 学生限制在可映射子集」），但在 151936 维上算两次 log_softmax 显存代价高
    # （每 token 约 600KB 中间量），8GB 卡上是主要 OOM 来源。
    # 只有在显存充裕、想跑「完整双空间」消融时再打开。
    use_teacher_space: bool = False

    # ---- 温度 ----
    # 学生空间（6400 维）与教师空间（151936 维）分布尖锐程度不同，分开调
    temp_student_space: float = 1.2
    temp_teacher_space: float = 1.5

    # ---- 熵门控 ----
    # 教师每位置熵高于 tau 时切前向 KL（保留多样性），低于时切反向 KL（收敛）
    entropy_tau: float = 1.5
    # 门控温度：越大过渡越平滑
    entropy_beta: float = 0.5

    # ---- 投影器预热 ----
    # 伪逆初始化只能达到「理论最优」（t2s 残差下界 0.886，见 projector.py 说明），
    # 教师分布经投影后仍严重失真（实测导出的熵接近均匀分布上限）。
    # 因此先用纯表征损失单独训练投影器若干步，让它在真实数据上收敛，
    # 再开始蒸馏。否则早期 KD 项全是噪声，反而污染学生参数。
    proj_warmup_steps: int = 50
    # 预热阶段的学习率（可比正式训练大，因为只训投影器）
    proj_warmup_lr: float = 1e-3

    # ---- 表征对齐 ----
    # 按相对深度做层映射：学生第 i 层 <- 教师第 round(i * L_t / L_s) 层
    rep_schedule: str = "relative"
    # 表征项权重线性衰减到该比例（早期拉近空间，后期交给分布 KL）
    rep_decay_to: float = 0.2

    # ---- 在线策略（G 阶段）----
    # 是否开启学生自采样
    on_policy: bool = True
    # 混合比例 λ：用学生自采样序列的比例，从 lam_start 线性升到 lam_end
    lam_start: float = 0.2
    lam_end: float = 0.7
    # 自采样温度
    sample_temperature: float = 0.9
    # 自采样最大新 token 数上限（防止个别长样本让生成阶段卡死）
    on_policy_max_new: int = 128
    # 每 k 步做一次自采样（自采样比普通前向慢一个量级，降低频率控制耗时）
    on_policy_every: int = 4


@dataclass
class LensKDConfig:
    teacher: TeacherCfg = field(default_factory=TeacherCfg)
    student: StudentCfg = field(default_factory=StudentCfg)
    projector: ProjectorCfg = field(default_factory=ProjectorCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)