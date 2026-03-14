"""
Configuration for Full Fine-Tuning.

This module contains all hyperparameters and settings needed for full fine-tuning.
Each parameter is documented with its purpose, typical range, and impact.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Configuration for the base model."""

    # ── Model Selection ──────────────────────────────────────────────
    model_name_or_path: str = "gpt2"
    """
    The pretrained model to fine-tune.
    Options for learning (small models):
      - "gpt2"           (~124M params, good for learning)
      - "gpt2-medium"    (~355M params)
      - "distilgpt2"     (~82M params, fastest)
    Production models (require significant GPU):
      - "meta-llama/Llama-2-7b-hf"
      - "mistralai/Mistral-7B-v0.1"
    """

    tokenizer_name: Optional[str] = None
    """If None, uses the same as model_name_or_path."""

    torch_dtype: str = "auto"
    """
    Model precision: 'auto', 'float32', 'float16', 'bfloat16'.
    - float32: Full precision, most memory, best numerical stability
    - float16: Half precision, saves ~50% memory, slight numerical risk
    - bfloat16: Better dynamic range than fp16, preferred on Ampere+ GPUs
    """

    trust_remote_code: bool = False
    """Whether to trust remote code from HuggingFace Hub."""


@dataclass
class DataConfig:
    """Configuration for dataset loading and preprocessing."""

    # ── Dataset ──────────────────────────────────────────────────────
    dataset_name: str = "wikitext"
    """HuggingFace dataset name. Examples: 'wikitext', 'imdb', 'ag_news'."""

    dataset_config: str = "wikitext-2-raw-v1"
    """Dataset configuration/subset name."""

    max_seq_length: int = 512
    """
    Maximum sequence length for tokenization.
    - Shorter = faster training, less memory
    - Longer = more context, more memory
    - GPT-2 max: 1024 tokens
    - LLaMA max: 4096 tokens
    Typical: 256-1024 for fine-tuning
    """

    preprocessing_num_workers: int = 4
    """Number of processes for data preprocessing."""

    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"

    max_train_samples: Optional[int] = None
    """Limit training samples (useful for debugging). None = use all."""

    max_eval_samples: Optional[int] = None
    """Limit evaluation samples. None = use all."""


@dataclass
class TrainingConfig:
    """
    Core training hyperparameters.

    These are the most important settings that affect training quality.
    """

    # ── Output ───────────────────────────────────────────────────────
    output_dir: str = "./results/full_finetune"
    """Directory for checkpoints, logs, and final model."""

    overwrite_output_dir: bool = True

    # ── Training Duration ────────────────────────────────────────────
    num_train_epochs: int = 3
    """
    Number of passes through the entire training dataset.
    - 1 epoch: minimal training, useful for very large datasets
    - 3 epochs: standard for most fine-tuning tasks
    - 5-10 epochs: for small datasets, watch for overfitting
    """

    max_steps: int = -1
    """If > 0, overrides num_train_epochs. Useful for quick experiments."""

    # ── Batch Size ───────────────────────────────────────────────────
    per_device_train_batch_size: int = 4
    """
    Batch size per GPU during training.
    - Limited by GPU memory
    - Smaller = less memory, noisier gradients
    - Larger = more stable, more memory
    Typical: 2-32 depending on model size and GPU memory
    """

    per_device_eval_batch_size: int = 8
    """Batch size per GPU during evaluation (can be larger since no gradients)."""

    gradient_accumulation_steps: int = 4
    """
    CRITICAL for full fine-tuning on limited hardware.
    Effective batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus

    Example: batch_size=4, accumulation=4, 1 GPU → effective batch = 16

    This simulates a larger batch size without extra memory.
    """

    # ── Learning Rate ────────────────────────────────────────────────
    learning_rate: float = 2e-5
    """
    THE most important hyperparameter for fine-tuning.

    Guidelines:
    - Full fine-tuning: 1e-5 to 5e-5 (conservative to avoid catastrophic forgetting)
    - Pretrain from scratch: 1e-4 to 3e-4
    - Too high: training is unstable, model diverges
    - Too low: training is very slow, may get stuck

    The learning rate for fine-tuning should be MUCH lower than pretraining
    because we want to gently adapt the model, not overwrite what it learned.
    """

    lr_scheduler_type: str = "cosine"
    """
    Learning rate schedule:
    - "linear": linearly decay to 0 (simple, reliable)
    - "cosine": cosine annealing (smooth, popular, often better)
    - "cosine_with_restarts": cosine with warm restarts
    - "constant": no decay (rarely used for fine-tuning)
    - "constant_with_warmup": constant after warmup
    """

    warmup_ratio: float = 0.06
    """
    Fraction of total steps for learning rate warmup.
    During warmup, LR increases linearly from 0 to the target.

    Purpose: prevents early training instability when the model
    hasn't seen the new data distribution yet.

    Typical: 0.03-0.1 (3-10% of training)
    """

    warmup_steps: int = 0
    """If > 0, overrides warmup_ratio. Set explicit warmup steps."""

    # ── Regularization ───────────────────────────────────────────────
    weight_decay: float = 0.01
    """
    L2 regularization penalty.
    Applied to all parameters EXCEPT bias and LayerNorm weights.

    Purpose: prevents overfitting by penalizing large weights.
    - 0.0: no regularization
    - 0.01: standard (used in BERT, GPT-2)
    - 0.1: aggressive (use for very small datasets)
    """

    max_grad_norm: float = 1.0
    """
    Gradient clipping threshold.
    Prevents exploding gradients during training.

    Clips gradients whose L2 norm exceeds this value.
    - 1.0: standard, used in most transformer training
    - 0.5: more aggressive clipping
    """

    # ── Memory Optimization ──────────────────────────────────────────
    gradient_checkpointing: bool = True
    """
    ESSENTIAL for full fine-tuning on limited hardware.

    Instead of storing all activations for backward pass,
    recompute them during backprop. Trades ~30% more compute
    for ~60-70% less memory.

    Rule of thumb: Always enable for full fine-tuning unless you
    have abundant GPU memory.
    """

    fp16: bool = False
    """
    Mixed precision training with FP16.
    Reduces memory by ~50% and speeds up training on compatible GPUs.
    Use on NVIDIA GPUs with Tensor Cores (V100, T4, A100, etc.)
    """

    bf16: bool = False
    """
    Mixed precision with BF16 (Brain Floating Point).
    Better numerical stability than FP16.
    Requires Ampere+ GPUs (A100, RTX 3090, etc.)
    Preferred over FP16 when available.
    """

    # ── Logging & Evaluation ─────────────────────────────────────────
    logging_dir: str = "./results/full_finetune/logs"
    logging_steps: int = 50
    """Log training metrics every N steps."""

    eval_strategy: str = "steps"
    """When to evaluate: 'no', 'steps', 'epoch'."""

    eval_steps: int = 200
    """Evaluate every N steps (if eval_strategy='steps')."""

    # ── Checkpointing ────────────────────────────────────────────────
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    """Keep only the last N checkpoints to save disk space."""

    load_best_model_at_end: bool = True
    """Load the best checkpoint at the end of training (requires eval)."""

    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # ── Early Stopping ───────────────────────────────────────────────
    early_stopping_patience: int = 3
    """Stop training if eval metric doesn't improve for N evaluations."""

    # ── Reproducibility ──────────────────────────────────────────────
    seed: int = 42
    """Random seed for reproducibility."""

    # ── Distributed Training ─────────────────────────────────────────
    dataloader_num_workers: int = 4
    ddp_find_unused_parameters: bool = False
    """Set True if some parameters might not receive gradients."""


@dataclass
class GenerationConfig:
    """Configuration for text generation during evaluation."""
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    num_beams: int = 1
    repetition_penalty: float = 1.2


def get_default_config():
    """Return all default configurations."""
    return ModelConfig(), DataConfig(), TrainingConfig(), GenerationConfig()


def print_config_summary(model_cfg, data_cfg, train_cfg):
    """Print a readable summary of the configuration."""
    print("=" * 60)
    print("FULL FINE-TUNING CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"\n📦 Model: {model_cfg.model_name_or_path}")
    print(f"📊 Dataset: {data_cfg.dataset_name} ({data_cfg.dataset_config})")
    print(f"📏 Max Sequence Length: {data_cfg.max_seq_length}")
    print(f"\n🎯 Epochs: {train_cfg.num_train_epochs}")
    print(f"📦 Batch Size (per device): {train_cfg.per_device_train_batch_size}")
    print(f"📦 Gradient Accumulation: {train_cfg.gradient_accumulation_steps}")
    eff_batch = train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps
    print(f"📦 Effective Batch Size: {eff_batch}")
    print(f"\n📈 Learning Rate: {train_cfg.learning_rate}")
    print(f"📈 LR Scheduler: {train_cfg.lr_scheduler_type}")
    print(f"📈 Warmup Ratio: {train_cfg.warmup_ratio}")
    print(f"\n🛡️ Weight Decay: {train_cfg.weight_decay}")
    print(f"🛡️ Max Grad Norm: {train_cfg.max_grad_norm}")
    print(f"\n💾 Gradient Checkpointing: {train_cfg.gradient_checkpointing}")
    print(f"💾 FP16: {train_cfg.fp16}")
    print(f"💾 BF16: {train_cfg.bf16}")
    print(f"\n📁 Output Dir: {train_cfg.output_dir}")
    print("=" * 60)
