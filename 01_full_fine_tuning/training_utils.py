"""
Training Utilities for Full Fine-Tuning.

Covers critical techniques for efficient and stable training:
  - Gradient checkpointing (memory optimization)
  - Mixed precision training (FP16/BF16)
  - Learning rate scheduling
  - Gradient accumulation
  - Distributed training setup
  - Model parameter analysis
  - Custom callbacks (early stopping, logging)
"""

import os
import math
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
from transformers import (
    PreTrainedModel,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. MODEL PARAMETER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def analyze_model_parameters(model: PreTrainedModel) -> Dict:
    """
    Analyze model parameters — essential before full fine-tuning.

    In FULL fine-tuning, ALL parameters are trainable.
    This function helps you understand:
    - Total parameter count
    - Memory requirements
    - Parameter distribution across layers

    This is the fundamental difference from PEFT methods:
    Full FT trains 100% of parameters, PEFT trains <1%.
    """
    total_params = 0
    trainable_params = 0
    param_details = {}

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

        # Group by module type
        module_type = name.split('.')[0]
        if module_type not in param_details:
            param_details[module_type] = {"total": 0, "trainable": 0}
        param_details[module_type]["total"] += param.numel()
        if param.requires_grad:
            param_details[module_type]["trainable"] += param.numel()

    # Memory estimation (approximate)
    # Each param in FP32 = 4 bytes
    # During training: param (4B) + gradient (4B) + optimizer states (8B for AdamW) = 16B per param
    memory_inference_fp32 = total_params * 4 / (1024 ** 3)  # GB
    memory_training_fp32 = total_params * 16 / (1024 ** 3)  # GB
    memory_training_fp16 = total_params * 10 / (1024 ** 3)  # GB (mixed precision)

    result = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percentage": 100.0 * trainable_params / total_params,
        "frozen_params": total_params - trainable_params,
        "memory_inference_fp32_gb": memory_inference_fp32,
        "memory_training_fp32_gb": memory_training_fp32,
        "memory_training_fp16_gb": memory_training_fp16,
        "param_details": param_details,
    }

    print("\n" + "=" * 60)
    print("MODEL PARAMETER ANALYSIS")
    print("=" * 60)
    print(f"Total Parameters:      {total_params:>15,}")
    print(f"Trainable Parameters:  {trainable_params:>15,}  ({result['trainable_percentage']:.1f}%)")
    print(f"Frozen Parameters:     {result['frozen_params']:>15,}")
    print(f"\nEstimated Memory Requirements:")
    print(f"  Inference (FP32):    {memory_inference_fp32:>8.2f} GB")
    print(f"  Training  (FP32):    {memory_training_fp32:>8.2f} GB")
    print(f"  Training  (FP16):    {memory_training_fp16:>8.2f} GB")
    print(f"\nParameter Distribution:")
    for module, details in param_details.items():
        pct = 100.0 * details["total"] / total_params
        print(f"  {module:30s}: {details['total']:>12,} ({pct:5.1f}%)")
    print("=" * 60)

    return result


# ═══════════════════════════════════════════════════════════════════════
# 2. GRADIENT CHECKPOINTING
# ═══════════════════════════════════════════════════════════════════════

def setup_gradient_checkpointing(model: PreTrainedModel, enable: bool = True):
    """
    Enable gradient checkpointing for memory-efficient training.

    HOW IT WORKS:
    ─────────────
    Normal backpropagation stores ALL intermediate activations (forward pass outputs)
    to compute gradients during the backward pass. This is VERY memory-hungry.

    With gradient checkpointing:
    1. Only save activations at "checkpoint" boundaries
    2. During backward pass, recompute missing activations on-the-fly
    3. Trade ~30% more compute time for ~60-70% less activation memory

    VISUAL:

    Without checkpointing:
        Layer 1 → [save act1] → Layer 2 → [save act2] → ... → Layer N → [save actN]
        Memory: O(N)

    With checkpointing:
        Layer 1 → [save act1] → Layer 2 → Layer 3 → [save act3] → ...
        Memory: O(√N)  (roughly)

    WHY IT MATTERS FOR FULL FINE-TUNING:
    Full fine-tuning computes gradients for ALL layers, so activation memory
    is at its maximum. Checkpointing is almost always worth the compute tradeoff.
    """
    if enable:
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing ENABLED")
        print("   → ~60-70% less activation memory")
        print("   → ~30% slower training (recomputation cost)")
    else:
        model.gradient_checkpointing_disable()
        print("❌ Gradient checkpointing DISABLED")

    return model


# ═══════════════════════════════════════════════════════════════════════
# 3. MIXED PRECISION SETUP
# ═══════════════════════════════════════════════════════════════════════

def check_mixed_precision_support() -> Dict[str, bool]:
    """
    Check GPU support for mixed precision training.

    MIXED PRECISION EXPLAINED:
    ──────────────────────────
    Instead of using FP32 (32-bit) for everything, mixed precision:
    - Keeps a master copy of weights in FP32 (for numerical stability)
    - Computes forward/backward passes in FP16 or BF16 (faster, less memory)
    - Uses loss scaling to prevent underflow in FP16

    FP16 vs BF16:
    - FP16: Supported on most modern NVIDIA GPUs (V100+, T4, etc.)
             Needs loss scaling to handle small gradients
    - BF16: Supported on Ampere+ (A100, RTX 3090+)
             Better dynamic range, no loss scaling needed
             Generally preferred when available

    Memory savings: ~50% reduction in activation memory
    Speed improvement: 1.5-3x on supported hardware
    """
    support = {
        "cuda_available": torch.cuda.is_available(),
        "fp16_supported": False,
        "bf16_supported": False,
        "gpu_name": "N/A",
    }

    if torch.cuda.is_available():
        support["gpu_name"] = torch.cuda.get_device_name(0)
        # FP16 support (compute capability >= 7.0)
        major, minor = torch.cuda.get_device_capability()
        support["fp16_supported"] = major >= 7
        # BF16 support (compute capability >= 8.0, i.e., Ampere+)
        support["bf16_supported"] = major >= 8

    print("\n" + "=" * 60)
    print("MIXED PRECISION SUPPORT")
    print("=" * 60)
    print(f"CUDA Available: {support['cuda_available']}")
    print(f"GPU: {support['gpu_name']}")
    print(f"FP16 Supported: {support['fp16_supported']}")
    print(f"BF16 Supported: {support['bf16_supported']}")
    print("=" * 60)

    return support


# ═══════════════════════════════════════════════════════════════════════
# 4. CUSTOM TRAINING CALLBACKS
# ═══════════════════════════════════════════════════════════════════════

class DetailedLoggingCallback(TrainerCallback):
    """
    Custom callback that logs detailed training information.

    Callbacks are hooks into the training loop that let you add
    custom behavior without modifying the Trainer class itself.

    Available hooks:
    - on_train_begin, on_train_end
    - on_epoch_begin, on_epoch_end
    - on_step_begin, on_step_end
    - on_evaluate
    - on_save
    - on_log
    """

    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []

    def on_train_begin(self, args, state, control, **kwargs):
        print("\n🚀 Training started!")
        print(f"   Total steps: {state.max_steps}")
        print(f"   Epochs: {args.num_train_epochs}")

    def on_log(self, args, state: TrainerState, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
                self.learning_rates.append(logs.get("learning_rate", 0))
            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            eval_loss = metrics.get("eval_loss", "N/A")
            perplexity = math.exp(eval_loss) if isinstance(eval_loss, float) else "N/A"
            print(f"\n📊 Evaluation @ step {state.global_step}:")
            print(f"   Loss: {eval_loss}")
            print(f"   Perplexity: {perplexity}")

    def on_train_end(self, args, state, control, **kwargs):
        print("\n✅ Training completed!")
        print(f"   Total steps: {state.global_step}")
        if self.train_losses:
            print(f"   Final train loss: {self.train_losses[-1]:.4f}")
        if self.eval_losses:
            print(f"   Best eval loss: {min(self.eval_losses):.4f}")


class GradientMonitorCallback(TrainerCallback):
    """
    Monitors gradient norms during training.

    WHY MONITOR GRADIENTS:
    - Exploding gradients → loss becomes NaN, training crashes
    - Vanishing gradients → model stops learning
    - Healthy gradients → smooth training progress

    In full fine-tuning, gradient issues are more common because
    ALL parameters are being updated simultaneously.
    """

    def __init__(self, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.log_every_n_steps == 0 and model is not None:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5

            print(f"   [Step {state.global_step}] Gradient norm: {total_norm:.4f}")


# ═══════════════════════════════════════════════════════════════════════
# 5. TRAINING ARGUMENTS BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_training_arguments(train_cfg) -> TrainingArguments:
    """
    Build HuggingFace TrainingArguments from our config.

    This centralizes all training hyperparameters and applies
    best practices for full fine-tuning.
    """
    # Determine precision based on hardware
    precision = check_mixed_precision_support()

    use_fp16 = train_cfg.fp16
    use_bf16 = train_cfg.bf16

    # Auto-detect best precision if not specified
    if not use_fp16 and not use_bf16 and precision["cuda_available"]:
        if precision["bf16_supported"]:
            use_bf16 = True
            print("Auto-selected BF16 mixed precision")
        elif precision["fp16_supported"]:
            use_fp16 = True
            print("Auto-selected FP16 mixed precision")

    return TrainingArguments(
        output_dir=train_cfg.output_dir,
        overwrite_output_dir=train_cfg.overwrite_output_dir,

        # Duration
        num_train_epochs=train_cfg.num_train_epochs,
        max_steps=train_cfg.max_steps,

        # Batch size & accumulation
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,

        # Optimizer & LR
        learning_rate=train_cfg.learning_rate,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_ratio=train_cfg.warmup_ratio,
        warmup_steps=train_cfg.warmup_steps,
        weight_decay=train_cfg.weight_decay,
        max_grad_norm=train_cfg.max_grad_norm,

        # Memory optimization
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        fp16=use_fp16,
        bf16=use_bf16,

        # Logging
        logging_dir=train_cfg.logging_dir,
        logging_steps=train_cfg.logging_steps,
        report_to="none",  # Set to "tensorboard" or "wandb" in production

        # Evaluation
        eval_strategy=train_cfg.eval_strategy,
        eval_steps=train_cfg.eval_steps,

        # Checkpointing
        save_strategy=train_cfg.save_strategy,
        save_steps=train_cfg.save_steps,
        save_total_limit=train_cfg.save_total_limit,
        load_best_model_at_end=train_cfg.load_best_model_at_end,
        metric_for_best_model=train_cfg.metric_for_best_model,
        greater_is_better=train_cfg.greater_is_better,

        # Reproducibility
        seed=train_cfg.seed,

        # Performance
        dataloader_num_workers=train_cfg.dataloader_num_workers,
    )


# ═══════════════════════════════════════════════════════════════════════
# 6. DISTRIBUTED TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def setup_distributed_training():
    """
    Information and setup for distributed training.

    DISTRIBUTED TRAINING STRATEGIES:
    ─────────────────────────────────
    1. DataParallel (DP):
       - Simplest approach
       - Replicates model on all GPUs
       - Splits batches across GPUs
       - Gathers gradients on GPU 0 (bottleneck)
       - NOT recommended for large models

    2. DistributedDataParallel (DDP):
       - HuggingFace Trainer default
       - Each GPU gets its own process
       - All-reduce for gradient synchronization
       - Near-linear scaling with # GPUs
       - Command: torchrun --nproc_per_node=4 train.py

    3. Fully Sharded Data Parallel (FSDP):
       - Shards model parameters across GPUs
       - Each GPU only holds a fraction of the model
       - Enables training models larger than a single GPU's memory
       - Command: accelerate launch --fsdp train.py

    4. DeepSpeed ZeRO:
       - Progressive memory optimization (Stage 1, 2, 3)
       - Stage 1: Shard optimizer states
       - Stage 2: Shard gradients + optimizer states
       - Stage 3: Shard everything (params + gradients + optimizer)
       - Integrated with HuggingFace Trainer

    For FULL fine-tuning, DDP or FSDP is almost always needed
    for models > 1B parameters.
    """
    info = {
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "commands": {
            "ddp": "torchrun --nproc_per_node=NUM_GPUS full_finetune.py",
            "fsdp": "accelerate launch --fsdp full_finetune.py",
            "deepspeed": "deepspeed full_finetune.py --deepspeed ds_config.json",
        }
    }

    print("\n" + "=" * 60)
    print("DISTRIBUTED TRAINING SETUP")
    print("=" * 60)
    print(f"Available GPUs: {info['num_gpus']}")
    if info['num_gpus'] > 1:
        print("\nRecommended launch commands:")
        for strategy, cmd in info['commands'].items():
            print(f"  {strategy.upper():>10}: {cmd}")
    elif info['num_gpus'] == 1:
        print("Single GPU detected. Standard training will be used.")
        print("Consider gradient checkpointing + mixed precision to maximize efficiency.")
    else:
        print("No GPU detected. Training will run on CPU (very slow).")
    print("=" * 60)

    return info


# ═══════════════════════════════════════════════════════════════════════
# 7. WEIGHT INITIALIZATION & FREEZING UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def freeze_layers(model: PreTrainedModel, layers_to_freeze: List[str]):
    """
    Freeze specific layers (set requires_grad=False).

    While FULL fine-tuning updates everything, sometimes you might want
    to freeze the embedding layer or early transformer layers as a
    compromise between full FT and PEFT.

    Common strategies:
    - Freeze embeddings only
    - Freeze first N transformer layers
    - Freeze everything except the LM head
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_freeze):
            param.requires_grad = False
            frozen_count += 1

    print(f"Froze {frozen_count} parameter tensors matching: {layers_to_freeze}")
    return model


def unfreeze_all(model: PreTrainedModel):
    """
    Unfreeze ALL parameters — this is what makes it FULL fine-tuning.
    Every parameter will receive gradients and be updated.
    """
    for param in model.parameters():
        param.requires_grad = True
    print("✅ All parameters unfrozen (FULL fine-tuning mode)")
    return model
