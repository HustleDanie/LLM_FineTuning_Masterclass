"""
Adapter Training Pipeline
===========================

Complete training pipeline for adapter-based fine-tuning:

1. Training with HuggingFace PEFT
   - Configuring adapters via PEFT's BottleneckConfig
   - Preparing models for adapter training
   - SFTTrainer integration

2. Training with adapter-transformers Library
   - AdapterConfig setup
   - Training heads and adapter stacking
   - Multi-task adapter training

3. Custom Training Loop
   - Manual adapter training with PyTorch
   - Gradient management for frozen vs adapter params
   - Learning rate scheduling strategies

4. AdapterDrop — Efficient Training & Inference
   - Dropping adapters from lower layers during training
   - Structured dropout for regularization
   - Inference speedup by removing adapters

5. Best Practices
   - Hyperparameter recommendations
   - Adapter-specific optimization tips
   - Common pitfalls and solutions

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import time
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


# ============================================================================
# SECTION 1: TRAINING WITH HUGGINGFACE PEFT
# ============================================================================

@dataclass
class AdapterTrainingConfig:
    """
    Complete configuration for adapter training.
    
    Adapter training is similar to LoRA training but with key differences:
    - Adapters ADD new parameters (vs LoRA which decomposes existing ones)
    - Adapters have non-zero inference overhead (vs LoRA which can merge)
    - Adapters are more naturally composable (AdapterFusion, stacking)
    """
    # ── Model ────────────────────────────────────────────────────
    base_model_name: str = "distilgpt2"
    
    # ── Adapter Config ───────────────────────────────────────────
    bottleneck_size: int = 64          # Bottleneck dimension
    non_linearity: str = "relu"        # Activation: relu, gelu, silu
    adapter_dropout: float = 0.1       # Dropout in adapter
    adapter_config: str = "pfeiffer"   # "pfeiffer" or "houlsby"
    use_layer_norm: bool = True        # LayerNorm before down-proj
    scaling: float = 1.0               # Output scaling factor
    
    # ── Training ─────────────────────────────────────────────────
    output_dir: str = "./adapter_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-3        # Adapters use higher LR than LoRA!
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    lr_scheduler_type: str = "linear"
    max_seq_length: int = 512
    
    # ── Logging ──────────────────────────────────────────────────
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_total_limit: int = 2


def train_with_peft_adapters(config: Optional[AdapterTrainingConfig] = None):
    """
    Train adapters using HuggingFace PEFT library.
    
    PEFT supports bottleneck adapters through various config classes.
    This demonstrates the standard PEFT workflow adapted for adapter training.
    """
    if config is None:
        config = AdapterTrainingConfig()
    
    print("=" * 65)
    print("  ADAPTER TRAINING WITH HUGGINGFACE PEFT")
    print("=" * 65)
    
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from peft import get_peft_model, TaskType
    from datasets import load_dataset
    from trl import SFTTrainer
    
    # ── Step 1: Load Model ───────────────────────────────────────
    print(f"\n[1/5] Loading {config.base_model_name}...")
    
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    
    # ── Step 2: Configure Adapter ────────────────────────────────
    print(f"\n[2/5] Configuring {config.adapter_config} adapter "
          f"(r={config.bottleneck_size})...")
    
    # PEFT supports adapters via LoraConfig with specific settings,
    # or through custom adapter configs. Here we use a LoRA-style
    # approach that creates adapter-like behavior, since PEFT's
    # primary adapter support is through LoRA variants.
    #
    # For true bottleneck adapters, the adapter-transformers library
    # (shown in Section 2) provides native support.
    
    from peft import LoraConfig
    
    # We use LoRA as the PEFT adapter mechanism
    # For true bottleneck adapters, see Section 2 with adapter-transformers
    peft_config = LoraConfig(
        r=config.bottleneck_size,
        lora_alpha=config.bottleneck_size * 2,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=config.adapter_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # ── Step 3: Prepare Dataset ──────────────────────────────────
    print(f"\n[3/5] Preparing dataset...")
    
    dataset = load_dataset("Abirate/english_quotes", split="train")
    
    def format_example(example):
        quote = example.get("quote", "")
        author = example.get("author", "Unknown") 
        return {"text": f"Quote: \"{quote}\" — {author}"}
    
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    
    print(f"  Train: {len(split['train'])} | Eval: {len(split['test'])}")
    
    # ── Step 4: Train ────────────────────────────────────────────
    print(f"\n[4/5] Training...")
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_total_limit=config.save_total_limit,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
    )
    
    start_time = time.time()
    result = trainer.train()
    train_time = time.time() - start_time
    
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Final loss: {result.training_loss:.4f}")
    
    # ── Step 5: Save ─────────────────────────────────────────────
    print(f"\n[5/5] Saving adapter to {config.output_dir}...")
    
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    adapter_size = sum(
        os.path.getsize(os.path.join(config.output_dir, f))
        for f in os.listdir(config.output_dir)
        if os.path.isfile(os.path.join(config.output_dir, f))
    ) / (1024 * 1024)
    
    print(f"  Adapter size: {adapter_size:.2f} MB")
    print(f"\n  ✓ Training complete!")
    
    return model, tokenizer, result


# ============================================================================
# SECTION 2: TRAINING WITH ADAPTER-TRANSFORMERS
# ============================================================================

def train_with_adapter_transformers():
    """
    Training with the adapter-transformers library.
    
    adapter-transformers is a fork of HuggingFace transformers that
    provides native support for bottleneck adapters, AdapterFusion,
    and the AdapterHub ecosystem.
    
    Note: This section shows the API patterns. Install with:
        pip install adapter-transformers
    """
    print("\n" + "=" * 65)
    print("  ADAPTER TRAINING WITH adapter-transformers")
    print("=" * 65)
    
    code = '''
# pip install adapter-transformers
# Note: adapter-transformers replaces the standard transformers package

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AdapterConfig, AdapterTrainer

# ── Load Model ───────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# ── Add Bottleneck Adapter ──────────────────────────────────────
# Pfeiffer configuration (1 adapter after FFN per layer)
adapter_config = AdapterConfig(
    mh_adapter=False,           # No adapter after multi-head attention
    output_adapter=True,        # Adapter after FFN output
    reduction_factor=16,        # d/r = 16 → bottleneck = d/16
    non_linearity="relu",       # Activation function
    original_ln_before=True,    # LayerNorm before adapter
    original_ln_after=True,     # LayerNorm after adapter
    residual_before_ln=True,    # Residual before LayerNorm
)

# Add adapter with a name
model.add_adapter("sentiment_task", config=adapter_config)

# Activate adapter for training
model.train_adapter("sentiment_task")
# This automatically:
# 1. Freezes all base model parameters
# 2. Makes adapter parameters trainable
# 3. Sets the active adapter

# ── Houlsby configuration (2 adapters per layer) ────────────────
houlsby_config = AdapterConfig(
    mh_adapter=True,            # Adapter after multi-head attention
    output_adapter=True,        # Adapter after FFN output  
    reduction_factor=16,
    non_linearity="relu",
)

model.add_adapter("houlsby_task", config=houlsby_config)

# ── Training ─────────────────────────────────────────────────────
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./adapter_output",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    learning_rate=1e-3,          # Higher LR for adapters
    warmup_steps=100,
    logging_steps=50,
)

# Use AdapterTrainer (adapter-aware version of Trainer)
trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

# ── Save Adapter ─────────────────────────────────────────────────
model.save_adapter("./saved_adapters/sentiment", "sentiment_task")
# This saves ONLY the adapter weights (~2 MB for a GPT-2 model)

# ── Load Adapter Later ──────────────────────────────────────────
model.load_adapter("./saved_adapters/sentiment", load_as="sentiment_task")
model.set_active_adapters("sentiment_task")
'''
    print(code)
    
    # Multi-task training
    print("\n  ── Multi-Task Adapter Training ──")
    multi_task_code = '''
# Train separate adapters for each task, sharing the same base model

tasks = {
    "sentiment": {"dataset": "sst2", "adapter_r": 64},
    "nli":       {"dataset": "mnli", "adapter_r": 64},
    "qa":        {"dataset": "squad", "adapter_r": 128},
}

for task_name, task_config in tasks.items():
    # Add task-specific adapter
    adapter_config = AdapterConfig(reduction_factor=768 // task_config["adapter_r"])
    model.add_adapter(task_name, config=adapter_config)
    
    # Train ONLY this adapter
    model.train_adapter(task_name)
    
    # Load task dataset
    dataset = load_dataset(task_config["dataset"])
    
    # Train
    trainer = AdapterTrainer(model=model, train_dataset=dataset["train"], ...)
    trainer.train()
    
    # Save this adapter
    model.save_adapter(f"./adapters/{task_name}", task_name)

# Now you have 3 task-specific adapters sharing the same base model!
# Total storage: base_model + 3 × adapter_size
# vs. 3 × full_model if fine-tuning separately
'''
    print(multi_task_code)
    return code


# ============================================================================
# SECTION 3: CUSTOM TRAINING LOOP
# ============================================================================

class CustomAdapterTrainer:
    """
    Manual adapter training with PyTorch training loop.
    
    Demonstrates the low-level mechanics of adapter training,
    including proper gradient management and optimization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        adapter_lr: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_steps: int = 1000,
    ):
        self.model = model
        self.adapter_lr = adapter_lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        # Separate adapter parameters from frozen parameters
        self.adapter_params = []
        self.frozen_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.adapter_params.append(param)
            else:
                self.frozen_params.append(param)
        
        n_adapter = sum(p.numel() for p in self.adapter_params)
        n_frozen = sum(p.numel() for p in self.frozen_params)
        print(f"  Adapter params: {n_adapter:,}")
        print(f"  Frozen params:  {n_frozen:,}")
        
        # Optimizer — only for adapter parameters
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
    
    def _create_optimizer(self):
        """Create optimizer with proper parameter groups."""
        # Separate params that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for param in self.adapter_params:
            if param.dim() >= 2:  # Weight matrices
                decay_params.append(param)
            else:  # Biases, LayerNorm
                no_decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        return torch.optim.AdamW(
            param_groups,
            lr=self.adapter_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    
    def _create_scheduler(self):
        """Linear warmup + cosine decay scheduler."""
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps
            )
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        self.model.train()
        
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (only adapter params have gradients)
        torch.nn.utils.clip_grad_norm_(self.adapter_params, max_norm=1.0)
        
        # Update
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, eval_dataloader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        for batch in eval_dataloader:
            outputs = self.model(**batch)
            total_loss += outputs.loss.item() * batch["input_ids"].numel()
            total_tokens += batch["input_ids"].numel()
        
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 100))
        
        return {"loss": avg_loss, "perplexity": perplexity}
    
    @staticmethod
    def demonstrate_custom_training():
        """Show the full custom training loop."""
        print("\n" + "=" * 65)
        print("  CUSTOM ADAPTER TRAINING LOOP")
        print("=" * 65)
        
        code = '''
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset

# Load model and inject adapters
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

# Apply PEFT adapter (LoRA or bottleneck)
from peft import get_peft_model, LoraConfig, TaskType
config = LoraConfig(r=64, lora_alpha=128, target_modules=["c_attn", "c_proj"],
                    task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, config)

# Prepare dataset
dataset = load_dataset("Abirate/english_quotes", split="train[:500]")
dataset = dataset.map(lambda x: tokenizer(
    f"Quote: \\"{x['quote']}\\"",
    truncation=True, max_length=256, padding="max_length",
    return_tensors="pt"
))
dataset.set_format("torch", columns=["input_ids", "attention_mask"])

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Create trainer
trainer = CustomAdapterTrainer(
    model=model,
    adapter_lr=1e-3,
    warmup_steps=50,
    max_steps=500,
)

# Training loop
for epoch in range(3):
    epoch_loss = 0
    steps = 0
    
    for batch in dataloader:
        # Add labels for causal LM
        batch["labels"] = batch["input_ids"].clone()
        batch = {k: v.to(model.device) for k, v in batch.items()}
        
        loss = trainer.train_step(batch)
        epoch_loss += loss
        steps += 1
        
        if steps % 10 == 0:
            lr = trainer.scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1} | Step {steps} | "
                  f"Loss: {loss:.4f} | LR: {lr:.2e}")
    
    avg_loss = epoch_loss / max(steps, 1)
    print(f"  Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")

# Save
model.save_pretrained("./custom_adapter_output")
'''
        print(code)
        return code


# ============================================================================
# SECTION 4: ADAPTERDROP
# ============================================================================

class AdapterDrop:
    """
    AdapterDrop: Efficient Adapters through Dropping
    (Rücklé et al., 2021)
    
    Key ideas:
    1. Not all adapter layers contribute equally
    2. Lower layers can be dropped with minimal quality loss
    3. This speeds up both training and inference
    
    The insight: Lower transformer layers capture general features,
    while upper layers capture task-specific features. Adapters in
    lower layers often have minimal impact.
    """
    
    @staticmethod
    def analyze_layer_importance():
        """
        Analyze which adapter layers contribute most to task performance.
        """
        print("\n" + "=" * 65)
        print("  ADAPTERDROP: LAYER IMPORTANCE ANALYSIS")
        print("=" * 65)
        
        # Simulated importance scores (based on gradient norms)
        n_layers = 12
        
        # Typical importance pattern: lower layers matter less
        importance_scores = [
            0.12, 0.15, 0.18, 0.22, 0.28,  # Layers 0-4: low importance
            0.35, 0.42, 0.55, 0.68, 0.78,  # Layers 5-9: increasing
            0.91, 0.95,                     # Layers 10-11: highest
        ]
        
        print(f"\n  Layer importance scores (gradient norm analysis):")
        print(f"  {'Layer':>7} {'Importance':>12} {'Bar':>30}")
        print("  " + "─" * 52)
        
        for i, score in enumerate(importance_scores):
            bar = "█" * int(score * 30)
            label = ""
            if i < 3:
                label = " ← Can drop"
            elif i >= 10:
                label = " ← Critical"
            print(f"  Layer {i:>2}  {score:>10.3f}  {bar}{label}")
        
        print(f"\n  Key finding:")
        print(f"    Layers 0-4:  Low importance → Safe to drop")
        print(f"    Layers 5-8:  Medium → Drop with some quality loss")
        print(f"    Layers 9-11: High → Must keep for task performance")
    
    @staticmethod
    def demonstrate_adapterdrop():
        """Demonstrate AdapterDrop configurations."""
        print("\n" + "=" * 65)
        print("  ADAPTERDROP CONFIGURATIONS")
        print("=" * 65)
        
        configs = """
  ── Configuration 1: Drop First N Layers ─────────────────────────
  
  Layer  0:  [Frozen Transformer]  ← No adapter (dropped)
  Layer  1:  [Frozen Transformer]  ← No adapter (dropped)
  Layer  2:  [Frozen Transformer]  ← No adapter (dropped)
  Layer  3:  [Frozen Transformer]  ← No adapter (dropped)
  Layer  4:  [Frozen Transformer]  ← No adapter (dropped)
  Layer  5:  [Frozen Transformer] → [Adapter] ← Active
  Layer  6:  [Frozen Transformer] → [Adapter] ← Active
  Layer  7:  [Frozen Transformer] → [Adapter] ← Active
  Layer  8:  [Frozen Transformer] → [Adapter] ← Active
  Layer  9:  [Frozen Transformer] → [Adapter] ← Active
  Layer 10:  [Frozen Transformer] → [Adapter] ← Active
  Layer 11:  [Frozen Transformer] → [Adapter] ← Active
  
  Parameters: 7/12 adapters = 58% of full adapter params
  Speed:      ~25% faster inference (skip 5 adapter forward passes)
  Quality:    ~98% of full adapter performance
  
  ── Configuration 2: Structured Dropout During Training ──────────
  
  During each training step, randomly drop adapters with probability p:
  
  Step 1: [✓] [✓] [✗] [✓] [✗] [✓] [✓] [✓] [✗] [✓] [✓] [✓]
  Step 2: [✗] [✓] [✓] [✓] [✓] [✗] [✓] [✓] [✓] [✓] [✗] [✓]
  Step 3: [✓] [✗] [✓] [✗] [✓] [✓] [✗] [✓] [✓] [✓] [✓] [✓]
  ...
  
  This acts as a regularizer (like Dropout) and makes the model
  robust to having adapters removed at inference time.
  
  ── Configuration 3: Progressive Dropping ────────────────────────
  
  Start with all adapters, gradually drop lower layers:
  
  Epoch 1: All 12 adapters active
  Epoch 2: Drop layers 0-1 (10 adapters)
  Epoch 3: Drop layers 0-3 (8 adapters)
  Epoch 4: Drop layers 0-5 (6 adapters)
  Final:   Only layers 6-11 have adapters
  
  This allows lower layers to "warm up" before being removed.
"""
        print(configs)
        
        # Speedup analysis
        print("  ── Speedup Analysis ──")
        print(f"\n  {'Dropped Layers':>16} {'Adapter Params':>16} {'Speedup':>10} "
              f"{'Quality':>10}")
        print("  " + "─" * 55)
        
        total_layers = 12
        data = [
            (0, 100, 1.00, 100.0),
            (2, 83, 1.08, 99.5),
            (4, 67, 1.18, 99.0),
            (5, 58, 1.25, 98.5),
            (6, 50, 1.33, 97.8),
            (8, 33, 1.50, 95.2),
            (10, 17, 1.72, 88.5),
        ]
        
        for dropped, pct, speedup, quality in data:
            active = total_layers - dropped
            print(f"  {dropped:>2}/{total_layers:>2} dropped  "
                  f"{pct:>12}%     {speedup:>8.2f}x  {quality:>8.1f}%")
        
        print(f"\n  → Dropping 4-5 layers gives best speed/quality trade-off")
    
    @staticmethod
    def implement_adapterdrop():
        """Code implementation of AdapterDrop."""
        print("\n" + "=" * 65)
        print("  ADAPTERDROP IMPLEMENTATION")
        print("=" * 65)
        
        code = '''
import torch
import torch.nn as nn
import random

class AdapterDropWrapper(nn.Module):
    """Wraps an adapter with dropout/dropping capability."""
    
    def __init__(self, adapter: nn.Module, drop_prob: float = 0.0):
        super().__init__()
        self.adapter = adapter
        self.drop_prob = drop_prob
        self.is_dropped = False  # For permanent dropping
    
    def forward(self, x):
        # If permanently dropped, pass through
        if self.is_dropped:
            return x
        
        # During training, randomly drop
        if self.training and self.drop_prob > 0:
            if random.random() < self.drop_prob:
                return x  # Skip adapter
        
        return self.adapter(x)
    
    def set_dropped(self, dropped: bool):
        """Permanently enable/disable this adapter."""
        self.is_dropped = dropped


class AdapterDropManager:
    """Manages AdapterDrop across all layers."""
    
    def __init__(self, model, adapters: list):
        self.model = model
        self.adapters = adapters  # List of AdapterDropWrapper
    
    def set_training_drop_rate(self, drop_prob: float):
        """Set random drop rate for training."""
        for adapter in self.adapters:
            adapter.drop_prob = drop_prob
    
    def drop_first_n_layers(self, n: int):
        """Permanently drop adapters from first n layers."""
        for i, adapter in enumerate(self.adapters):
            adapter.set_dropped(i < n)
        
        active = sum(1 for a in self.adapters if not a.is_dropped)
        print(f"Dropped {n} layers, {active} adapters remain active")
    
    def progressive_drop(self, epoch: int, total_epochs: int):
        """Progressively drop more layers as training proceeds."""
        # Linear schedule: drop more layers in later epochs
        max_drop = len(self.adapters) // 2  # Drop at most half
        n_drop = int(max_drop * epoch / total_epochs)
        self.drop_first_n_layers(n_drop)
        return n_drop
    
    def find_optimal_drop_count(self, eval_fn, min_quality: float = 0.95):
        """Find maximum layers to drop while maintaining quality."""
        baseline = eval_fn()
        
        for n in range(len(self.adapters)):
            self.drop_first_n_layers(n)
            quality = eval_fn()
            
            relative = quality / baseline
            print(f"  Drop {n}: quality = {quality:.4f} "
                  f"({relative:.1%} of baseline)")
            
            if relative < min_quality:
                optimal = n - 1
                self.drop_first_n_layers(optimal)
                print(f"  Optimal: drop {optimal} layers")
                return optimal
        
        return len(self.adapters) - 1
'''
        print(code)
        return code


# ============================================================================
# SECTION 5: BEST PRACTICES & HYPERPARAMETERS
# ============================================================================

class AdapterBestPractices:
    """
    Best practices and hyperparameter recommendations for adapter training.
    """
    
    @staticmethod
    def print_hyperparameter_guide():
        """Comprehensive hyperparameter guide for adapters."""
        print("\n" + "=" * 65)
        print("  ADAPTER TRAINING: HYPERPARAMETER GUIDE")
        print("=" * 65)
        
        guide = """
┌─────────────────────────────────────────────────────────────────┐
│ HYPERPARAMETER RECOMMENDATIONS                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Learning Rate:                                                  │
│   Adapters: 1e-3 to 5e-3  (MUCH higher than LoRA!)             │
│   LoRA:     1e-4 to 3e-4                                       │
│   Full FT:  1e-5 to 5e-5                                       │
│                                                                 │
│   Why higher? Adapters are randomly initialized (not            │
│   decomposed from existing weights), so they need larger        │
│   updates to learn meaningful transformations.                  │
│                                                                 │
│ Bottleneck Size:                                                │
│   Small models (< 1B):     r = 32-64                           │
│   Medium models (1-7B):    r = 64-128                          │
│   Large models (> 7B):     r = 128-256                         │
│   Rule of thumb: r ≈ d/16 to d/8                               │
│                                                                 │
│ Configuration:                                                  │
│   Start with Pfeiffer (1 adapter/layer) for efficiency          │
│   Use Houlsby (2 adapters/layer) if quality matters more        │
│   Pfeiffer is ~50% fewer params with ~95% of Houlsby quality   │
│                                                                 │
│ Activation Function:                                            │
│   ReLU:  Default, works well for most tasks                     │
│   GELU:  Slightly better for NLU tasks                         │
│   SiLU:  Good for generative tasks                             │
│                                                                 │
│ Dropout:                                                        │
│   Small datasets: 0.1-0.2                                      │
│   Large datasets: 0.0-0.05                                     │
│                                                                 │
│ Weight Decay: 0.01 (standard)                                   │
│                                                                 │
│ Warmup: 3-6% of total steps                                    │
│                                                                 │
│ Batch Size:                                                     │
│   Larger is generally better for adapters                       │
│   16-32 effective batch size recommended                        │
│   Use gradient accumulation if GPU memory is limited            │
│                                                                 │
│ Training Epochs:                                                │
│   3-10 epochs (adapters converge faster than full FT)           │
│   Monitor validation loss for early stopping                    │
│                                                                 │
│ Scheduler:                                                      │
│   Linear warmup + linear/cosine decay                          │
│   Cosine tends to work slightly better                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

COMPARISON WITH LoRA TRAINING:
─────────────────────────────────────────────────────────────────
  Setting          Adapters           LoRA
─────────────────────────────────────────────────────────────────
  LR               1e-3               2e-4
  Epochs           3-5                3-5
  Batch size       16-32              8-16
  Weight decay     0.01               0.01
  Warmup           3-6%               3%
  Dropout          0.1                0.05
  Parameters       0.5-2% of model    0.1-0.5% of model
  Speed            Slower (overhead)  Faster (no overhead)
  Quality          Excellent          Excellent
  Composability    Excellent          Limited
  Merge possible   No                 Yes
─────────────────────────────────────────────────────────────────
"""
        print(guide)
    
    @staticmethod
    def print_common_issues():
        """Common pitfalls and solutions."""
        print("\n" + "=" * 65)
        print("  COMMON ISSUES & SOLUTIONS")
        print("=" * 65)
        
        issues = """
1. TRAINING LOSS DOESN'T DECREASE
   Cause: Learning rate too low
   Fix:   Increase LR to 1e-3 or even 5e-3
          (Adapters need higher LR than LoRA or full FT)

2. TRAINING IS UNSTABLE (LOSS SPIKES)
   Cause: Learning rate too high or no warmup
   Fix:   Add warmup (5-10% of steps)
          Reduce LR to 5e-4
          Add gradient clipping (max_norm=1.0)

3. OVERFITTING (TRAIN LOSS LOW, EVAL LOSS HIGH)
   Cause: Too many adapter parameters or too few data
   Fix:   Reduce bottleneck size
          Switch from Houlsby to Pfeiffer
          Increase dropout to 0.2
          Use AdapterDrop for regularization

4. SLOW CONVERGENCE
   Cause: Poor initialization or small bottleneck
   Fix:   Verify up-projection is zero-initialized
          Try larger bottleneck size
          Check that base model params are actually frozen

5. INFERENCE IS SLOW
   Cause: Adapter overhead on every forward pass
   Fix:   Use AdapterDrop to remove lower-layer adapters
          Switch to Pfeiffer config (fewer adapters)
          Consider LoRA if inference speed is critical
          (LoRA can be merged for zero overhead)

6. ADAPTER DOESN'T INTEGRATE WITH VLLM/TGI
   Cause: Serving engines don't support adapter architecture
   Fix:   These engines support LoRA natively, not adapters
          For adapter models, use custom serving (FastAPI)
          Or convert to knowledge-distilled student model

7. MULTIPLE TASKS INTERFERE
   Cause: Training adapter on task B forgets task A
   Fix:   Train SEPARATE adapters per task (share base model)
          Use AdapterFusion to combine task adapters
          Never train one adapter on multiple tasks sequentially
"""
        print(issues)
    
    @staticmethod
    def adapter_vs_lora_decision():
        """When to use adapters vs LoRA."""
        print("\n" + "=" * 65)
        print("  WHEN TO USE ADAPTERS vs LoRA")
        print("=" * 65)
        
        decision = """
USE ADAPTERS WHEN:
  ✓ You need to serve many tasks from one base model
  ✓ AdapterFusion is needed for cross-task knowledge transfer
  ✓ You want a modular, plug-and-play system
  ✓ Inference latency is not the primary concern
  ✓ You want the AdapterHub ecosystem

USE LoRA WHEN:
  ✓ Inference speed matters (LoRA merges for zero overhead)
  ✓ Memory is very tight (LoRA typically has fewer params)
  ✓ You need vLLM/TGI serving support
  ✓ You want the simplest possible setup
  ✓ Single-task fine-tuning

USE BOTH (hybrid):
  ✓ Adapters for task modules + LoRA for cross-cutting concerns
  ✓ Some frameworks support combining PEFT methods

BOTTOM LINE:
  → LoRA is more popular due to merge capability and simplicity
  → Adapters excel at multi-task and compositional scenarios
  → Quality is comparable between the two methods
  → Choose based on your deployment requirements
"""
        print(decision)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all adapter training demonstrations."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║              ADAPTER TRAINING PIPELINE                       ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: PEFT training
    print("\n  Running PEFT adapter training (distilgpt2 demo)...")
    config = AdapterTrainingConfig(
        num_train_epochs=1,
        per_device_train_batch_size=4,
        bottleneck_size=32,
        save_strategy="no",
        logging_steps=5,
    )
    train_with_peft_adapters(config)
    
    # Section 2: adapter-transformers
    train_with_adapter_transformers()
    
    # Section 3: Custom training loop
    CustomAdapterTrainer.demonstrate_custom_training()
    
    # Section 4: AdapterDrop
    adapter_drop = AdapterDrop()
    adapter_drop.analyze_layer_importance()
    adapter_drop.demonstrate_adapterdrop()
    adapter_drop.implement_adapterdrop()
    
    # Section 5: Best practices
    practices = AdapterBestPractices()
    practices.print_hyperparameter_guide()
    practices.print_common_issues()
    practices.adapter_vs_lora_decision()
    
    print("\n" + "=" * 65)
    print("  MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered in this module:
    ✓ PEFT adapter training pipeline
    ✓ adapter-transformers library usage
    ✓ Custom PyTorch training loop for adapters
    ✓ AdapterDrop for efficient training and inference
    ✓ Hyperparameter guide and best practices
    ✓ Adapters vs LoRA decision framework
    """)


if __name__ == "__main__":
    main()
