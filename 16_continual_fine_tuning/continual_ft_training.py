"""
Continual Fine-Tuning in Practice — LoRA Adapters, Data Mixing, HuggingFace
=============================================================================

Production-ready continual fine-tuning with:

1. Task-Specific LoRA Adapters — separate adapter per task
2. Data Mixing Strategies — curriculum for continual learning
3. Continual Pretraining with HuggingFace — domain adaptation pipeline
4. Progressive Training Schedule — learning rate & data scheduling
5. Full Pipeline: Multi-Domain Continual Fine-Tuning

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ============================================================================
# SECTION 1: TASK-SPECIFIC LoRA ADAPTERS
# ============================================================================

def task_specific_lora_adapters():
    """
    Architecture-based continual learning with separate LoRA per task.
    The MOST PRACTICAL approach for real LLM continual fine-tuning.
    """
    print("=" * 70)
    print("  SECTION 1: TASK-SPECIFIC LoRA ADAPTERS")
    print("=" * 70)
    
    print(f"""
  ═══ Why LoRA Adapters for Continual Learning ═══
  
  The simplest and most effective approach:
  
  1. Keep base model FROZEN (shared knowledge)
  2. Train separate LoRA adapter for each new task/domain
  3. Each adapter is small (0.1-1% of base model)
  4. NO interference between tasks (zero forgetting!)
  5. Switch adapters at inference time
  
  ┌──────────────────────────────────────────────┐
  │              BASE MODEL (frozen)              │
  │  ┌─────────┐ ┌─────────┐ ┌─────────┐        │
  │  │ LoRA    │ │ LoRA    │ │ LoRA    │  ...   │
  │  │ Task A  │ │ Task B  │ │ Task C  │        │
  │  │ (2MB)   │ │ (2MB)   │ │ (2MB)   │        │
  │  └─────────┘ └─────────┘ └─────────┘        │
  └──────────────────────────────────────────────┘
""")
    
    # === Simulate LoRA per task ===
    
    class BaseModel(nn.Module):
        """Simulated frozen base model."""
        def __init__(self, d_model=64, n_layers=4):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(d_model, d_model, bias=False) 
                for _ in range(n_layers)
            ])
            self.head = nn.Linear(d_model, 4)
            
        def forward(self, x, lora_deltas=None):
            for i, layer in enumerate(self.layers):
                h = layer(x)
                if lora_deltas and i in lora_deltas:
                    h = h + lora_deltas[i](x)  # LoRA residual
                x = F.relu(h)
            return self.head(x)
    
    class LoRAAdapter(nn.Module):
        """Single LoRA adapter for one task."""
        def __init__(self, d_model=64, rank=4, n_layers=4):
            super().__init__()
            self.adapters = nn.ModuleDict()
            for i in range(n_layers):
                self.adapters[str(i)] = nn.Sequential(
                    nn.Linear(d_model, rank, bias=False),
                    nn.Linear(rank, d_model, bias=False)
                )
                # Initialize B to zero (LoRA init)
                nn.init.zeros_(self.adapters[str(i)][1].weight)
        
        def get_deltas(self):
            return {int(k): v for k, v in self.adapters.items()}
        
        def param_count(self):
            return sum(p.numel() for p in self.parameters())
    
    # Create synthetic tasks
    torch.manual_seed(42)
    n_tasks = 4
    tasks = []
    for t in range(n_tasks):
        W = torch.randn(64, 1)
        x_train = torch.randn(200, 64)
        scores = x_train @ W
        y_train = (scores > scores.median()).long().squeeze() * (t % 4)
        y_train = y_train % 4
        
        x_test = torch.randn(50, 64)
        y_test = (x_test @ W > (x_test @ W).median()).long().squeeze() * (t % 4)
        y_test = y_test % 4
        tasks.append({'train': (x_train, y_train), 'test': (x_test, y_test)})
    
    # Shared frozen base
    base = BaseModel(d_model=64, n_layers=4)
    for p in base.parameters():
        p.requires_grad = False  # Freeze base
    
    adapters = {}
    
    print(f"  Base model params: {sum(p.numel() for p in base.parameters()):,}")
    
    for task_idx in range(n_tasks):
        # Create new adapter for this task
        adapter = LoRAAdapter(d_model=64, rank=4, n_layers=4)
        optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
        
        tx, ty = tasks[task_idx]['train']
        
        for epoch in range(100):
            logits = base(tx, lora_deltas=adapter.get_deltas())
            loss = F.cross_entropy(logits, ty)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        adapters[f'task_{task_idx}'] = adapter
        
        print(f"\n  Task {task_idx+1} adapter: {adapter.param_count():,} params "
              f"({adapter.param_count()/sum(p.numel() for p in base.parameters())*100:.1f}% of base)")
        
        # Evaluate ALL tasks with their respective adapters
        for j in range(task_idx + 1):
            ex, ey = tasks[j]['test']
            with torch.no_grad():
                preds = base(ex, lora_deltas=adapters[f'task_{j}'].get_deltas()).argmax(1)
                acc = (preds == ey).float().mean().item()
            print(f"    Task {j+1} accuracy (using task_{j+1} adapter): {acc:.1%}")
    
    print(f"""
  ═══ Key Insight ═══
  
  With per-task adapters, there is ZERO forgetting by design!
  Each adapter only modifies the base model for its specific task.
  
  Practical considerations:
  • Storage: ~2-8 MB per LoRA adapter (for a 7B model with r=16)
  • 100 tasks = 200-800MB total adapter storage (very manageable)
  • Inference: Load base model once, swap adapters per request
  • Can merge popular adapters into base model periodically
""")
    
    del base, adapters


# ============================================================================
# SECTION 2: DATA MIXING STRATEGIES
# ============================================================================

def data_mixing_strategies():
    """Optimal data mixing for continual pretraining/fine-tuning."""
    print("\n\n" + "=" * 70)
    print("  SECTION 2: DATA MIXING STRATEGIES")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ Data Mixing for Continual Learning ═══
  
  When you must use a SINGLE model (not per-task adapters),
  data mixing is critical to prevent forgetting.
  
  Strategies:
  ┌────────────────────────────────────────────────────────┐
  │ 1. Fixed Ratio   — constant new:old ratio             │
  │ 2. Temperature   — sample domains by T-scaled weights  │
  │ 3. Curriculum    — gradually shift from old to new     │
  │ 4. DRO-based     — upweight worst-performing domain    │
  └────────────────────────────────────────────────────────┘
""")
    
    @dataclass
    class DataMixer:
        """Multi-domain data mixer with various strategies."""
        domain_sizes: List[int]  # Number of samples per domain
        
        def fixed_ratio(self, new_domain_weight: float = 0.7) -> List[float]:
            """Fixed sampling ratio: new domain gets fixed weight."""
            n = len(self.domain_sizes)
            if n == 1:
                return [1.0]
            old_weight = (1.0 - new_domain_weight) / (n - 1)
            return [old_weight] * (n - 1) + [new_domain_weight]
        
        def temperature_scaled(self, temperature: float = 2.0) -> List[float]:
            """Sample proportional to domain_size^(1/T)."""
            sizes = torch.tensor(self.domain_sizes, dtype=torch.float)
            scaled = sizes ** (1.0 / temperature)
            probs = scaled / scaled.sum()
            return probs.tolist()
        
        def curriculum(self, step: int, total_steps: int,
                       warmup_ratio: float = 0.1) -> List[float]:
            """Start with more old data, gradually shift to new."""
            progress = step / total_steps
            n = len(self.domain_sizes)
            if n == 1:
                return [1.0]
            
            # New domain weight increases from 0.3 to 0.9
            new_weight = 0.3 + 0.6 * min(1.0, progress / (1.0 - warmup_ratio))
            old_weight = (1.0 - new_weight) / (n - 1)
            return [old_weight] * (n - 1) + [new_weight]
        
        def dro_weighted(self, domain_losses: List[float],
                         eta: float = 1.0) -> List[float]:
            """Distributionally Robust Optimization: upweight high-loss domains."""
            losses = torch.tensor(domain_losses)
            weights = torch.exp(eta * losses)
            weights = weights / weights.sum()
            return weights.tolist()
    
    # Demo: mixing across 4 domains
    mixer = DataMixer(domain_sizes=[1000, 800, 600, 400])
    
    print(f"  Domain sizes: {mixer.domain_sizes}")
    print(f"\n  ── Fixed Ratio (70% new) ──")
    weights = mixer.fixed_ratio(0.7)
    for i, w in enumerate(weights):
        bar = "█" * int(w * 40)
        label = "NEW" if i == len(weights) - 1 else f"Old-{i+1}"
        print(f"  {label:>6}: {w:.2f} {bar}")
    
    print(f"\n  ── Temperature Scaling (T=2.0) ──")
    weights = mixer.temperature_scaled(temperature=2.0)
    for i, w in enumerate(weights):
        bar = "█" * int(w * 40)
        label = "NEW" if i == len(weights) - 1 else f"Old-{i+1}"
        print(f"  {label:>6}: {w:.2f} {bar}")
    
    print(f"\n  ── Curriculum (progress 0% → 50% → 100%) ──")
    for pct in [0.0, 0.5, 1.0]:
        step = int(pct * 1000)
        weights = mixer.curriculum(step, 1000)
        print(f"  Step {step:>4}:", end="")
        for i, w in enumerate(weights):
            label = "new" if i == len(weights) - 1 else f"old{i+1}"
            print(f"  {label}={w:.2f}", end="")
        print()
    
    print(f"\n  ── DRO-Weighted (upweight struggling domains) ──")
    domain_losses = [0.5, 1.2, 0.3, 0.8]  # Domain 2 is struggling
    weights = mixer.dro_weighted(domain_losses, eta=1.0)
    for i, (w, l) in enumerate(zip(weights, domain_losses)):
        bar = "█" * int(w * 40)
        label = "NEW" if i == len(weights) - 1 else f"Old-{i+1}"
        print(f"  {label:>6} (loss={l:.1f}): {w:.2f} {bar}")
    
    # === Run training with different mixing strategies ===
    print(f"\n  ── Training Comparison: Mixing Strategies ──")
    
    # Create simple benchmark
    n_tasks = 3
    in_dim = 8
    torch.manual_seed(42)
    
    task_data = []
    for t in range(n_tasks):
        W = torch.randn(in_dim, 1) * (t + 1)
        x = torch.randn(200, in_dim)
        y = ((x @ W).squeeze() > 0).long()
        
        x_test = torch.randn(50, in_dim)
        y_test = ((x_test @ W).squeeze() > 0).long()
        task_data.append({'train': (x, y), 'test': (x_test, y_test)})
    
    def train_with_mixing(strategy: str) -> List[float]:
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        replay_data = []
        
        for task_idx in range(n_tasks):
            tx, ty = task_data[task_idx]['train']
            
            for epoch in range(80):
                # New task loss
                loss = F.cross_entropy(model(tx), ty)
                
                # Mix in old data
                if replay_data:
                    if strategy == "uniform":
                        replay_weight = 1.0 / (task_idx + 1)
                    elif strategy == "heavy_replay":
                        replay_weight = 0.5
                    elif strategy == "curriculum":
                        replay_weight = max(0.1, 0.5 - 0.4 * epoch / 80)
                    else:
                        replay_weight = 0.0
                    
                    if replay_weight > 0:
                        for rx, ry in replay_data:
                            loss += replay_weight * F.cross_entropy(model(rx), ry)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            # Store subset for replay
            idx = torch.randperm(len(tx))[:30]
            replay_data.append((tx[idx], ty[idx]))
        
        # Final evaluation
        model.eval()
        accs = []
        for t in task_data:
            ex, ey = t['test']
            with torch.no_grad():
                acc = (model(ex).argmax(1) == ey).float().mean().item()
            accs.append(acc)
        return accs
    
    strategies = ["none", "uniform", "heavy_replay", "curriculum"]
    print(f"\n  {'Strategy':>15} │ {'T1':>6} {'T2':>6} {'T3':>6} │ {'Avg':>6}")
    print(f"  {'─'*15}─┼─{'─'*20}─┼─{'─'*6}")
    
    for strat in strategies:
        accs = train_with_mixing(strat)
        avg = sum(accs) / len(accs)
        print(f"  {strat:>15} │ {accs[0]:>5.1%} {accs[1]:>5.1%} {accs[2]:>5.1%} │ {avg:>5.1%}")
    
    print(f"""
  ═══ Data Mixing Best Practices ═══
  
  1. Always include SOME old domain data (even 5% helps)
  2. Use balanced sampling for earliest tasks (most vulnerable)
  3. Curriculum mixing works well: start with more replay, taper off
  4. Monitor per-domain loss during training
  5. For LLMs: mix at the document level, not sentence level
""")


# ============================================================================
# SECTION 3: CONTINUAL PRETRAINING WITH HUGGINGFACE
# ============================================================================

def continual_pretraining_pipeline():
    """Practical continual pretraining/fine-tuning with HuggingFace."""
    print("\n\n" + "=" * 70)
    print("  SECTION 3: CONTINUAL PRETRAINING WITH HUGGINGFACE")
    print("=" * 70)
    
    print(f"""
  ═══ Continual Pretraining Pipeline ═══
  
  Goal: Adapt a general LLM to a new domain while preserving capabilities.
  
  Example: GPT → Medical GPT → Cardiology GPT → Patient Interaction GPT
  
  ┌─────────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐
  │ Base LLM    │ →  │ Domain   │ →  │ Subdomain │ →  │ Task     │
  │ (general)   │    │ pretrain │    │ finetune  │    │ finetune │
  └─────────────┘    └──────────┘    └───────────┘    └──────────┘
  
  Key: Each stage should PRESERVE previous capabilities!
""")
    
    hf_pipeline_code = '''
# ═══════════════════════════════════════════════════════════════
# PRODUCTION CODE: Continual Pretraining with HuggingFace
# ═══════════════════════════════════════════════════════════════

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, concatenate_datasets
import torch

# ─── Step 1: Load base model ───
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# ─── Step 2: Apply LoRA for parameter-efficient adaptation ───
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                        # Rank
    lora_alpha=32,               # Scaling
    lora_dropout=0.05,
    target_modules=["c_attn"],   # GPT-2 attention
    # For LLaMA: target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
model = get_peft_model(model, lora_config)

# ─── Step 3: Prepare domain data with replay mixing ───
# New domain data
new_domain = load_dataset("text", data_files="domain_data.txt", split="train")

# Replay data from previous domain (5-10% of new domain size)
replay_data = load_dataset("text", data_files="replay_data.txt", split="train")

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

new_tokenized = new_domain.map(tokenize, batched=True, remove_columns=["text"])
replay_tokenized = replay_data.map(tokenize, batched=True, remove_columns=["text"])

# Mix: 90% new domain, 10% replay
# Subsample replay to achieve desired ratio
replay_size = len(new_tokenized) // 9  # ~10% of total
replay_subset = replay_tokenized.select(range(min(replay_size, len(replay_tokenized))))
mixed_dataset = concatenate_datasets([new_tokenized, replay_subset]).shuffle(seed=42)

# ─── Step 4: Training with continual learning best practices ───
training_args = TrainingArguments(
    output_dir="./continual_ft_output",
    
    # LOW learning rate (critical for preserving knowledge)
    learning_rate=2e-5,          # Lower than regular fine-tuning
    
    # Warmup prevents sudden parameter shifts
    warmup_ratio=0.1,            # 10% warmup
    
    # Moderate training (not too many epochs)
    num_train_epochs=2,          # 1-3 epochs typical
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    
    # Gradient clipping (prevents catastrophic updates)
    max_grad_norm=0.5,           # Lower than default 1.0
    
    # Learning rate schedule
    lr_scheduler_type="cosine",  # Smooth decay
    
    # Weight decay (mild regularization)
    weight_decay=0.01,
    
    # Logging
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=500,
    
    # FP16 for efficiency
    fp16=torch.cuda.is_available(),
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM (not masked LM)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=mixed_dataset,
    data_collator=data_collator,
)

# ─── Step 5: Train ───
trainer.train()

# ─── Step 6: Save adapter (not full model) ───
model.save_pretrained("./domain_adapter_v1")
# Total storage: ~10-20MB instead of full model

# ─── Step 7: For next domain, create new adapter ───
# model = AutoModelForCausalLM.from_pretrained(model_name)
# new_lora_config = LoraConfig(...)  # Can use different rank
# model = get_peft_model(model, new_lora_config)
# ... train on domain 2 ...
# model.save_pretrained("./domain_adapter_v2")
'''
    
    print(hf_pipeline_code)
    
    print(f"""
  ═══ Continual Pretraining Checklist ═══
  
  □ Use LoRA (limits parameter change, acts as implicit regularization)
  □ Lower learning rate than standard fine-tuning (1e-5 to 5e-5)
  □ Include 5-10% replay data from previous domains
  □ Use warmup (10-15% of training steps)
  □ Lower gradient clipping (0.3-0.5 vs normal 1.0)
  □ Evaluate on ALL previous domains periodically
  □ Save adapter checkpoints for rollback
  □ Use cosine LR schedule (smoother than linear)
""")


# ============================================================================
# SECTION 4: PROGRESSIVE TRAINING SCHEDULE
# ============================================================================

def progressive_training_schedule():
    """Training schedule design for multi-domain continual learning."""
    print("\n\n" + "=" * 70)
    print("  SECTION 4: PROGRESSIVE TRAINING SCHEDULE")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ Progressive Training Schedule ═══
  
  Key principle: Later domains need LESS training to adapt
  (the model has learned how to learn from domain adaptation).
  
  Schedule design:
  ┌────────────────────────────────────────────────┐
  │  Domain 1: 3 epochs, lr=3e-5, warmup=15%      │
  │  Domain 2: 2 epochs, lr=2e-5, warmup=10%      │
  │  Domain 3: 2 epochs, lr=1e-5, warmup=10%      │
  │  Domain 4: 1 epoch,  lr=5e-6, warmup=5%       │
  └────────────────────────────────────────────────┘
""")
    
    @dataclass
    class DomainSchedule:
        """Training schedule for one domain."""
        domain_name: str
        epochs: int
        learning_rate: float
        warmup_ratio: float
        replay_ratio: float    # Fraction of training data from replay
        lora_rank: int
        max_grad_norm: float
    
    class ProgressiveScheduler:
        """Generates progressively conservative training schedules."""
        
        def __init__(self, base_lr: float = 3e-5, base_epochs: int = 3,
                     decay_factor: float = 0.7):
            self.base_lr = base_lr
            self.base_epochs = base_epochs
            self.decay_factor = decay_factor
            self.schedules: List[DomainSchedule] = []
        
        def add_domain(self, name: str) -> DomainSchedule:
            n = len(self.schedules)
            decay = self.decay_factor ** n
            
            schedule = DomainSchedule(
                domain_name=name,
                epochs=max(1, int(self.base_epochs * decay)),
                learning_rate=self.base_lr * decay,
                warmup_ratio=min(0.15, 0.05 + 0.05 * (n > 0)),
                replay_ratio=min(0.3, 0.05 * n),  # More replay for later domains
                lora_rank=min(32, 16 + 4 * n),     # Higher rank for later domains
                max_grad_norm=max(0.3, 1.0 - 0.2 * n)  # Tighter clipping later
            )
            self.schedules.append(schedule)
            return schedule
        
        def display(self):
            print(f"\n  {'Domain':>12} │ {'Epochs':>6} │ {'LR':>10} │ {'Warmup':>6} │ "
                  f"{'Replay':>6} │ {'LoRA r':>6} │ {'Grad Clip':>9}")
            print(f"  {'─'*12}─┼─{'─'*6}─┼─{'─'*10}─┼─{'─'*6}─┼─"
                  f"{'─'*6}─┼─{'─'*6}─┼─{'─'*9}")
            
            for s in self.schedules:
                print(f"  {s.domain_name:>12} │ {s.epochs:>6} │ {s.learning_rate:>10.1e} │ "
                      f"{s.warmup_ratio:>5.0%} │ {s.replay_ratio:>5.0%} │ "
                      f"{s.lora_rank:>6} │ {s.max_grad_norm:>9.1f}")
    
    # Demo: 5-domain progressive schedule
    scheduler = ProgressiveScheduler(base_lr=3e-5, base_epochs=3, decay_factor=0.75)
    
    domains = ["General", "Medical", "Cardiology", "ECG Reports", "Patient Chat"]
    for d in domains:
        scheduler.add_domain(d)
    
    scheduler.display()
    
    print(f"""
  ═══ Key Principles ═══
  
  1. REDUCE learning rate for later domains
     • Early domains: 3e-5 (standard fine-tuning)
     • Later domains: 5e-6 to 1e-5 (gentler updates)
  
  2. INCREASE replay ratio for later domains
     • More accumulated knowledge to protect
     • Domain 2: 5% replay, Domain 5: 20% replay
  
  3. INCREASE LoRA rank for task complexity
     • Simple domain shift: r=8-16
     • Complex new capabilities: r=32-64
  
  4. DECREASE gradient clipping for stability
     • Early: max_grad_norm=1.0
     • Later: max_grad_norm=0.3-0.5
""")
    
    # === Simulate progressive training ===
    in_dim = 8
    n_tasks = 4
    
    task_data = []
    for t in range(n_tasks):
        torch.manual_seed(t + 100)
        W = torch.randn(in_dim, 1) * (t + 1)
        x = torch.randn(200, in_dim)
        y = ((x @ W).squeeze() > 0).long()
        x_test = torch.randn(50, in_dim)
        y_test = ((x_test @ W).squeeze() > 0).long()
        task_data.append({'train': (x, y), 'test': (x_test, y_test)})
    
    def train_progressive(use_schedule: bool) -> List[float]:
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        replay_data = []
        
        for task_idx in range(n_tasks):
            # Progressive schedule
            if use_schedule:
                decay = 0.75 ** task_idx
                lr = 1e-3 * decay
                n_epochs = max(40, int(100 * decay))
                replay_weight = min(0.5, 0.1 * task_idx)
            else:
                lr = 1e-3
                n_epochs = 80
                replay_weight = 0.2 if task_idx > 0 else 0
            
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            tx, ty = task_data[task_idx]['train']
            
            for epoch in range(n_epochs):
                loss = F.cross_entropy(model(tx), ty)
                
                if replay_data and replay_weight > 0:
                    for rx, ry in replay_data:
                        loss += replay_weight * F.cross_entropy(model(rx), ry)
                
                opt.zero_grad()
                loss.backward()
                
                # Gradient clipping (tighter for later tasks)
                if use_schedule:
                    max_norm = max(0.5, 2.0 - 0.5 * task_idx)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                
                opt.step()
            
            idx = torch.randperm(len(tx))[:30]
            replay_data.append((tx[idx], ty[idx]))
        
        model.eval()
        accs = []
        for t in task_data:
            ex, ey = t['test']
            with torch.no_grad():
                acc = (model(ex).argmax(1) == ey).float().mean().item()
            accs.append(acc)
        return accs
    
    print(f"\n  ── Progressive vs Fixed Schedule ──\n")
    
    fixed_accs = train_progressive(use_schedule=False)
    prog_accs = train_progressive(use_schedule=True)
    
    print(f"  {'Schedule':>12} │ {'T1':>6} {'T2':>6} {'T3':>6} {'T4':>6} │ {'Avg':>6}")
    print(f"  {'─'*12}─┼─{'─'*26}─┼─{'─'*6}")
    
    fixed_avg = sum(fixed_accs) / len(fixed_accs)
    prog_avg = sum(prog_accs) / len(prog_accs)
    
    print(f"  {'Fixed':>12} │ ", end="")
    for a in fixed_accs:
        print(f"{a:>5.1%} ", end="")
    print(f"│ {fixed_avg:>5.1%}")
    
    print(f"  {'Progressive':>12} │ ", end="")
    for a in prog_accs:
        print(f"{a:>5.1%} ", end="")
    print(f"│ {prog_avg:>5.1%}")


# ============================================================================
# SECTION 5: FULL PIPELINE — MULTI-DOMAIN CONTINUAL FINE-TUNING
# ============================================================================

def full_pipeline():
    """Complete multi-domain continual fine-tuning pipeline."""
    print("\n\n" + "=" * 70)
    print("  SECTION 5: FULL PIPELINE — MULTI-DOMAIN CONTINUAL FT")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ End-to-End Continual Fine-Tuning Pipeline ═══
  
  Scenario: Adapt a model across 4 sequential domains.
  
  Combining ALL techniques:
  1. Per-task LoRA adapters (architecture-based)
  2. Data mixing with replay (replay-based)
  3. EWC regularization (regularization-based)
  4. Progressive training schedule
  5. Comprehensive evaluation
""")
    
    # === Unified Continual Learner ===
    
    class ContinualLearner:
        """
        Production-grade continual learning system combining
        multiple anti-forgetting strategies.
        """
        
        def __init__(self, model: nn.Module, config: dict):
            self.base_model = model
            self.config = config
            
            # Regularization (EWC)
            self.fisher_dicts = []
            self.optimal_params = []
            
            # Replay buffer
            self.replay_buffer_x = []
            self.replay_buffer_y = []
            
            # Tracking
            self.task_count = 0
            self.accuracy_matrix = []
            self.training_losses = []
        
        def compute_fisher(self, data_x, data_y, n_samples=100):
            fisher = {n: torch.zeros_like(p) 
                      for n, p in self.base_model.named_parameters() 
                      if p.requires_grad}
            
            self.base_model.eval()
            n = min(n_samples, len(data_x))
            
            for i in range(n):
                self.base_model.zero_grad()
                loss = F.cross_entropy(
                    self.base_model(data_x[i:i+1]), data_y[i:i+1])
                loss.backward()
                
                for name, p in self.base_model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[name] += p.grad.data ** 2
            
            for name in fisher:
                fisher[name] /= n
            
            self.base_model.train()
            return fisher
        
        def ewc_penalty(self):
            if not self.fisher_dicts:
                return torch.tensor(0.0)
            
            loss = 0.0
            for fisher, params in zip(self.fisher_dicts, self.optimal_params):
                for name, p in self.base_model.named_parameters():
                    if p.requires_grad and name in fisher:
                        loss += (fisher[name] * (p - params[name]) ** 2).sum()
            
            return self.config.get('ewc_lambda', 100.0) / 2 * loss
        
        def get_replay_batch(self, batch_size=32):
            if not self.replay_buffer_x:
                return None
            
            all_x = torch.cat(self.replay_buffer_x)
            all_y = torch.cat(self.replay_buffer_y)
            n = min(batch_size, len(all_x))
            idx = torch.randperm(len(all_x))[:n]
            return all_x[idx], all_y[idx]
        
        def train_on_task(self, task_name: str, train_x, train_y, 
                          eval_fn=None):
            """Train on a new task with all protections."""
            self.task_count += 1
            
            # Progressive schedule
            decay = 0.8 ** (self.task_count - 1)
            lr = self.config.get('base_lr', 1e-3) * decay
            n_epochs = max(40, int(self.config.get('base_epochs', 80) * decay))
            replay_weight = min(0.5, self.config.get('replay_base', 0.1) * (self.task_count - 1))
            
            optimizer = torch.optim.Adam(
                [p for p in self.base_model.parameters() if p.requires_grad], 
                lr=lr)
            
            losses = []
            
            for epoch in range(n_epochs):
                # Task loss
                logits = self.base_model(train_x)
                task_loss = F.cross_entropy(logits, train_y)
                
                # EWC penalty
                ewc_loss = self.ewc_penalty()
                
                # Replay loss
                replay_loss = torch.tensor(0.0)
                replay_batch = self.get_replay_batch(32)
                if replay_batch is not None:
                    rx, ry = replay_batch
                    replay_loss = F.cross_entropy(self.base_model(rx), ry)
                
                total = task_loss + ewc_loss + replay_weight * replay_loss
                
                optimizer.zero_grad()
                total.backward()
                
                # Progressive gradient clipping
                max_norm = max(0.3, 1.0 - 0.2 * (self.task_count - 1))
                torch.nn.utils.clip_grad_norm_(
                    self.base_model.parameters(), max_norm)
                
                optimizer.step()
                losses.append(total.item())
            
            self.training_losses.append(losses)
            
            # Register task for future protection
            fisher = self.compute_fisher(train_x, train_y)
            self.fisher_dicts.append(fisher)
            self.optimal_params.append(
                {n: p.data.clone() for n, p in self.base_model.named_parameters()})
            
            # Add to replay buffer
            buf_size = self.config.get('replay_buffer_size', 50)
            idx = torch.randperm(len(train_x))[:buf_size]
            self.replay_buffer_x.append(train_x[idx])
            self.replay_buffer_y.append(train_y[idx])
            
            # Evaluate
            if eval_fn:
                accs = eval_fn()
                self.accuracy_matrix.append(accs)
            
            return {
                'task': task_name,
                'epochs': n_epochs,
                'lr': lr,
                'replay_weight': replay_weight,
                'final_loss': losses[-1]
            }
        
        def summary(self):
            """Print comprehensive summary."""
            if not self.accuracy_matrix:
                return
            
            T = len(self.accuracy_matrix)
            final_accs = self.accuracy_matrix[-1]
            
            # Metrics
            aa = sum(final_accs[:T]) / T
            bwt = sum(final_accs[j] - self.accuracy_matrix[j][j] 
                      for j in range(T-1)) / max(1, T-1)
            
            fm_vals = []
            for j in range(T-1):
                max_after = max(self.accuracy_matrix[k][j] 
                               for k in range(j, T))
                fm_vals.append(max_after - final_accs[j])
            fm = sum(fm_vals) / len(fm_vals) if fm_vals else 0
            
            print(f"\n  ── Pipeline Summary ──")
            print(f"  Average Accuracy:    {aa:.1%}")
            print(f"  Backward Transfer:   {bwt:+.1%}")
            print(f"  Forgetting Measure:  {fm:.1%}")
            
            return {'AA': aa, 'BWT': bwt, 'FM': fm}
    
    # === Run full pipeline ===
    
    in_dim = 8
    n_tasks = 4
    
    task_data = []
    for t in range(n_tasks):
        torch.manual_seed(t * 42 + 7)
        W = torch.randn(in_dim, 1) * 2
        dims = list(range(t * 2, t * 2 + 4))
        dims = [d % in_dim for d in dims]
        
        x = torch.randn(300, in_dim)
        features = x[:, dims]
        y = ((features @ W[:4]).squeeze() > 0).long()
        
        x_test = torch.randn(80, in_dim)
        feat_test = x_test[:, dims]
        y_test = ((feat_test @ W[:4]).squeeze() > 0).long()
        
        task_data.append({
            'name': f'Domain {t+1}',
            'train': (x, y), 
            'test': (x_test, y_test)
        })
    
    # Method 1: Full pipeline
    print(f"\n  Running Full Continual Learning Pipeline...\n")
    
    model = nn.Sequential(
        nn.Linear(in_dim, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 2)
    )
    
    learner = ContinualLearner(model, {
        'base_lr': 1e-3,
        'base_epochs': 80,
        'ewc_lambda': 1000.0,
        'replay_buffer_size': 50,
        'replay_base': 0.15,
    })
    
    def evaluate_all():
        model.eval()
        accs = []
        for t in task_data:
            ex, ey = t['test']
            with torch.no_grad():
                acc = (model(ex).argmax(1) == ey).float().mean().item()
            accs.append(acc)
        model.train()
        return accs
    
    for task in task_data:
        info = learner.train_on_task(
            task['name'], *task['train'], eval_fn=evaluate_all)
        
        accs = learner.accuracy_matrix[-1]
        print(f"  {info['task']:>10}: epochs={info['epochs']:>2}, "
              f"lr={info['lr']:.1e}, replay_w={info['replay_weight']:.2f} │ ", end="")
        for i, a in enumerate(accs):
            print(f"D{i+1}={a:.0%} ", end="")
        print()
    
    full_metrics = learner.summary()
    
    # Method 2: Naive baseline
    print(f"\n  Running Naive Baseline (no protection)...\n")
    
    torch.manual_seed(42)
    naive_model = nn.Sequential(
        nn.Linear(in_dim, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 2)
    )
    opt = torch.optim.Adam(naive_model.parameters(), lr=1e-3)
    
    naive_matrix = []
    for task in task_data:
        tx, ty = task['train']
        for _ in range(80):
            loss = F.cross_entropy(naive_model(tx), ty)
            opt.zero_grad(); loss.backward(); opt.step()
        
        naive_model.eval()
        accs = []
        for t in task_data:
            ex, ey = t['test']
            with torch.no_grad():
                acc = (naive_model(ex).argmax(1) == ey).float().mean().item()
            accs.append(acc)
        naive_model.train()
        naive_matrix.append(accs)
    
    naive_final = naive_matrix[-1]
    naive_aa = sum(naive_final[:n_tasks]) / n_tasks
    naive_bwt = sum(naive_final[j] - naive_matrix[j][j] for j in range(n_tasks-1)) / (n_tasks-1)
    
    print(f"  Naive — AA: {naive_aa:.1%}, BWT: {naive_bwt:+.1%}")
    
    # Comparison
    print(f"""
  ╔══════════════════════════════════════════════════════╗
  ║          FULL PIPELINE vs NAIVE BASELINE             ║
  ╠══════════════════════════════════════════════════════╣
  ║  Metric          │  Full Pipeline  │  Naive         ║
  ║  ─────────────── │ ─────────────── │ ──────────     ║
  ║  Avg Accuracy    │  {full_metrics['AA']:>12.1%}   │  {naive_aa:>8.1%}     ║
  ║  Back. Transfer  │  {full_metrics['BWT']:>+12.1%}   │  {naive_bwt:>+8.1%}     ║
  ║  Forgetting      │  {full_metrics['FM']:>12.1%}   │  N/A          ║
  ╚══════════════════════════════════════════════════════╝
""")
    
    print(f"""
  ═══ Production Deployment Summary ═══
  
  For production continual fine-tuning of LLMs:
  
  1. ALWAYS use LoRA/QLoRA (limits catastrophic change)
  2. ALWAYS include replay data (5-20% of training data)
  3. Use progressive schedule (lower LR for later domains)
  4. Monitor per-domain performance continuously
  5. Keep adapter checkpoints for rollback
  6. Consider per-task adapters if domains are very different
  7. Merge stable adapters into base model periodically
  
  The combination of LoRA + Replay + Progressive Schedule
  handles most practical continual fine-tuning scenarios.
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  CONTINUAL FT TRAINING — LoRA ADAPTERS, DATA MIXING, PIPELINE   ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    task_specific_lora_adapters()
    data_mixing_strategies()
    continual_pretraining_pipeline()
    progressive_training_schedule()
    full_pipeline()
    
    print("\n" + "=" * 70)
    print("  TRAINING MODULE COMPLETE")
    print("=" * 70)
    print("""
    Covered:
    ✓ Task-specific LoRA adapters (zero forgetting by design)
    ✓ Data mixing strategies (fixed, temperature, curriculum, DRO)
    ✓ Continual pretraining with HuggingFace (production code)
    ✓ Progressive training schedule (decay LR, increase replay)
    ✓ Full pipeline combining all techniques
    """)


if __name__ == "__main__":
    main()
