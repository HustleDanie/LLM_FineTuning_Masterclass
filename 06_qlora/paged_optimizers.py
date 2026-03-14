"""
Paged Optimizers — QLoRA's Third Innovation
=============================================

Paged optimizers prevent out-of-memory (OOM) errors by using
NVIDIA's unified memory to automatically page optimizer states
between GPU and CPU memory.

The Problem:
  Even with QLoRA's 4-bit base model, optimizer states for LoRA
  parameters can cause GPU OOM during gradient checkpointing or
  when processing long sequences that create activation spikes.

The Solution:
  Paged optimizers (from bitsandbytes) use CUDA unified memory:
  - Optimizer states are allocated in unified memory
  - NVIDIA's memory manager automatically pages data between
    GPU RAM and CPU RAM as needed
  - During forward pass: optimizer states can be paged to CPU
  - During optimizer step: states are paged back to GPU
  - This is AUTOMATIC — no manual management needed!

This module covers:
1. Why optimizer states consume so much memory
2. CUDA Unified Memory explained
3. Paged optimizers from bitsandbytes
4. When to use paged optimizers
5. Performance implications

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional
from dataclasses import dataclass


# ============================================================================
# SECTION 1: OPTIMIZER STATE MEMORY
# ============================================================================

class OptimizerMemoryAnalysis:
    """
    Analyze how much memory different optimizers consume.
    
    The key insight: optimizers store STATES for each parameter.
    For AdamW (the standard optimizer), this means:
    - First moment (m): same size as parameter (FP32)
    - Second moment (v): same size as parameter (FP32)
    - Total: 2x the parameter memory in FP32
    
    For a model with P trainable parameters:
    - AdamW states = P × 4 × 2 = 8P bytes
    - SGD states = P × 4 × 1 = 4P bytes (momentum only)
    """
    
    @staticmethod
    def analyze_optimizer_memory():
        """Show optimizer memory for different scenarios."""
        print("=" * 70)
        print("OPTIMIZER STATE MEMORY ANALYSIS")
        print("=" * 70)
        
        print("""
  Optimizer State Sizes (per trainable parameter):
  ═══════════════════════════════════════════════════════
  
  SGD:
    - No states (vanilla)         → 0 bytes/param
    - With momentum              → 4 bytes/param (1× FP32)
  
  AdamW:
    - First moment (m)           → 4 bytes/param (FP32)
    - Second moment (v)          → 4 bytes/param (FP32)
    - Total                      → 8 bytes/param
  
  8-bit AdamW (bitsandbytes):
    - First moment (m)           → 1 byte/param (INT8)
    - Second moment (v)          → 1 byte/param (INT8)
    - Total                      → 2 bytes/param
  
  Paged AdamW 32-bit:
    - Same as AdamW              → 8 bytes/param
    - BUT: can be paged to CPU when not needed
  
  Paged AdamW 8-bit:
    - Same as 8-bit AdamW        → 2 bytes/param
    - BUT: can be paged to CPU when not needed
""")
        
        # Scenarios
        scenarios = {
            "Full FT (7B model, all params)": {
                "trainable_params": 7e9,
                "note": "Full fine-tuning",
            },
            "LoRA r=16 (7B model)": {
                "trainable_params": 40e6,
                "note": "LoRA adapters only",
            },
            "LoRA r=64 (7B model)": {
                "trainable_params": 160e6,
                "note": "Higher rank LoRA",
            },
            "QLoRA r=16 (65B model)": {
                "trainable_params": 300e6,
                "note": "QLoRA on large model",
            },
        }
        
        optimizers = {
            "AdamW FP32": 8,       # bytes per param
            "AdamW 8-bit": 2,
            "SGD+momentum": 4,
        }
        
        print(f"\n  Optimizer State Memory (GB):")
        print(f"  {'Scenario':<35}", end="")
        for opt in optimizers:
            print(f" {opt:>15}", end="")
        print()
        print("  " + "-" * 80)
        
        for name, info in scenarios.items():
            params = info["trainable_params"]
            print(f"  {name:<35}", end="")
            for opt_name, bytes_per_param in optimizers.items():
                mem_gb = params * bytes_per_param / (1024**3)
                print(f" {mem_gb:>13.2f}GB", end="")
            print()
        
        print("\n  Key insight:")
        print("    For QLoRA with LoRA r=16, optimizer states are ~0.3 GB — manageable.")
        print("    But with higher ranks or more trainable params, this grows quickly.")
        print("    Paged optimizers provide a safety net against OOM during spikes.")
    
    @staticmethod
    def show_memory_spike_problem():
        """
        Demonstrate WHY paged optimizers are needed even when
        average memory usage is below GPU capacity.
        """
        print("\n" + "=" * 70)
        print("THE MEMORY SPIKE PROBLEM")
        print("=" * 70)
        
        diagram = """
  GPU Memory Usage During Training:
  ═══════════════════════════════════════════════════════════
  
  GPU VRAM (24 GB)
  ─────────────────────────────────────── Capacity
  
                    ┌──┐
                    │  │ ← SPIKE! (OOM!)
               ┌────┘  └──┐
          ┌────┘          └────┐
     ─────┘                    └─────── Average  
  ─────────────────────────────────────
  
  Time →   Forward    Backward    Optimizer
           pass       pass        step
  
  Memory spikes happen because:
  
  1. ACTIVATION MEMORY: Long sequences create large activation
     tensors that must be stored for the backward pass.
     
  2. GRADIENT ACCUMULATION: When accumulating gradients over
     multiple steps, memory temporarily increases.
     
  3. OPTIMIZER STEP: The optimizer needs to load all states
     (moments) into GPU memory simultaneously.
     
  4. MIXED COMPONENTS: At the optimizer step, you need:
     - Model weights (4-bit NF4)
     - LoRA weights (BF16)
     - Gradients (BF16)
     - Optimizer states (FP32)  ← These can be paged!
     
  Solution: Paged optimizers move states to CPU during
  forward/backward and page them back during optimizer step.
  If GPU runs low, states are automatically evicted to CPU.
"""
        print(diagram)


# ============================================================================
# SECTION 2: CUDA UNIFIED MEMORY
# ============================================================================

class UnifiedMemoryExplained:
    """
    Explains CUDA Unified Memory, the technology that enables
    paged optimizers.
    """
    
    @staticmethod
    def explain():
        """
        CUDA Unified Memory allows CPU and GPU to share a single
        address space with automatic page migration.
        """
        print("\n" + "=" * 70)
        print("CUDA UNIFIED MEMORY — THE FOUNDATION OF PAGED OPTIMIZERS")
        print("=" * 70)
        
        explanation = """
  Traditional Memory Model:
  ═══════════════════════════════════════════════════════════
  
  ┌──────────────────┐         ┌──────────────────┐
  │   CPU Memory     │         │   GPU Memory     │
  │   (System RAM)   │ ──────> │   (VRAM)         │
  │                  │ <────── │                  │
  │   Manual         │ cudaMemcpy │   Manual      │
  │   allocation     │         │   allocation     │
  └──────────────────┘         └──────────────────┘
  
  Problem: You must manually manage transfers.
  If GPU runs out of memory → OOM error!
  
  
  Unified Memory Model:
  ═══════════════════════════════════════════════════════════
  
  ┌────────────────────────────────────────┐
  │         Unified Virtual Address Space   │
  │                                        │
  │   ┌────────┐    ┌────────┐            │
  │   │ Page 1 │    │ Page 2 │ ...        │
  │   └───┬────┘    └───┬────┘            │
  │       │             │                  │
  │   ┌───▼────┐    ┌───▼────┐            │
  │   │  CPU   │    │  GPU   │ ← Pages    │
  │   │ (RAM)  │    │ (VRAM) │   migrate  │
  │   └────────┘    └────────┘   AUTO!    │
  └────────────────────────────────────────┘
  
  How it works:
  1. Data is allocated in a unified address space
  2. When GPU accesses a page that's on CPU → automatic page fault
  3. The NVIDIA driver migrates the page to GPU
  4. When GPU memory is full → least-used pages evicted to CPU
  5. ALL of this is AUTOMATIC — no manual management!
  
  For Paged Optimizers:
  - Optimizer states are allocated in unified memory
  - During forward/backward: states may be on CPU (not needed)
  - During optimizer step: states are paged to GPU (needed)
  - If GPU is full: some states stay on CPU, paged on-demand
  
  Performance:
  - PCIe bandwidth: ~32 GB/s (PCIe 4.0 x16)
  - GPU memory bandwidth: ~2 TB/s (A100)
  - So: paging adds ~60x latency per page access
  - BUT: optimizer step is NOT the bottleneck (forward/backward is)
  - In practice: <5% slowdown for most workloads
"""
        print(explanation)


# ============================================================================
# SECTION 3: PAGED OPTIMIZERS FROM BITSANDBYTES
# ============================================================================

class PagedOptimizers:
    """
    Using paged optimizers from bitsandbytes.
    """
    
    @staticmethod
    def show_available_optimizers():
        """
        List all paged optimizer variants available in bitsandbytes.
        """
        print("\n" + "=" * 70)
        print("PAGED OPTIMIZERS IN BITSANDBYTES")
        print("=" * 70)
        
        optimizers = """
  Available Paged Optimizers:
  ═══════════════════════════════════════════════════════════
  
  From bitsandbytes:
  ┌──────────────────────────────┬─────────┬───────────────┐
  │ Optimizer                    │ States  │ Memory/Param  │
  ├──────────────────────────────┼─────────┼───────────────┤
  │ PagedAdamW                   │ FP32    │ 8 bytes       │
  │ PagedAdamW8bit               │ INT8    │ 2 bytes       │
  │ PagedAdamW32bit              │ FP32    │ 8 bytes       │
  │ PagedLion                    │ FP32    │ 4 bytes       │
  │ PagedLion8bit                │ INT8    │ 1 byte        │
  └──────────────────────────────┴─────────┴───────────────┘
  
  How to import:
    import bitsandbytes as bnb
    optimizer = bnb.optim.PagedAdamW(model.parameters(), lr=2e-4)
    optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=2e-4)
    optimizer = bnb.optim.PagedAdamW32bit(model.parameters(), lr=2e-4)
    optimizer = bnb.optim.PagedLion(model.parameters(), lr=1e-5)
    optimizer = bnb.optim.PagedLion8bit(model.parameters(), lr=1e-5)
  
  Via HuggingFace TrainingArguments:
    TrainingArguments(
        optim="paged_adamw_32bit",   # FP32 paged AdamW
        # or
        optim="paged_adamw_8bit",    # 8-bit paged AdamW
    )
  
  Non-paged 8-bit optimizers (also from bitsandbytes):
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2e-4)
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=2e-4)
    optimizer = bnb.optim.Lion8bit(model.parameters(), lr=1e-5)
"""
        print(optimizers)
    
    @staticmethod
    def show_usage_with_trainer():
        """
        How to use paged optimizers with HuggingFace Trainer.
        """
        print("\n" + "=" * 70)
        print("USING PAGED OPTIMIZERS WITH HF TRAINER")
        print("=" * 70)
        
        code = '''
from transformers import TrainingArguments, Trainer

# ── Method 1: Via optim string (easiest) ─────────────────────────
training_args = TrainingArguments(
    output_dir="./output",
    optim="paged_adamw_32bit",    # ← Paged optimizer!
    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    fp16=True,
)

# Other optim string options:
#   "adamw_torch"           → Standard PyTorch AdamW
#   "adamw_bnb_8bit"        → 8-bit AdamW (not paged)
#   "paged_adamw_8bit"      → 8-bit Paged AdamW
#   "paged_adamw_32bit"     → 32-bit Paged AdamW
#   "paged_lion_8bit"       → 8-bit Paged Lion
#   "paged_lion_32bit"      → 32-bit Paged Lion

# ── Method 2: Custom optimizer object ────────────────────────────
import bitsandbytes as bnb

optimizer = bnb.optim.PagedAdamW8bit(
    model.parameters(),
    lr=2e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    optimizers=(optimizer, None),  # (optimizer, scheduler)
)

# ── Method 3: Complete QLoRA setup with paged optimizer ──────────
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import torch

# 4-bit quantized model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# Training with paged optimizer
training_args = TrainingArguments(
    output_dir="./qlora_output",
    optim="paged_adamw_8bit",     # ← Paged + 8-bit!
    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,   # Save activation memory
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
'''
        print(code)
        return code
    
    @staticmethod
    def show_when_to_use():
        """
        Decision guide for when to use paged optimizers.
        """
        print("\n" + "=" * 70)
        print("WHEN TO USE PAGED OPTIMIZERS — DECISION GUIDE")
        print("=" * 70)
        
        guide = """
  ┌─────────────────────────────────────────────────────────────┐
  │              PAGED OPTIMIZER DECISION TREE                   │
  │                                                             │
  │  Are you getting OOM errors during training?                │
  │  ├── YES → Use paged_adamw_8bit (most memory efficient)    │
  │  │                                                          │
  │  └── NO → Is your GPU memory >90% utilized?                │
  │       ├── YES → Use paged_adamw_32bit (safety net)         │
  │       │         or paged_adamw_8bit (more savings)          │
  │       │                                                     │
  │       └── NO → Standard optimizer is fine                   │
  │                adamw_torch or adamw_bnb_8bit               │
  └─────────────────────────────────────────────────────────────┘
  
  RECOMMENDATIONS BY SCENARIO:
  ═══════════════════════════════════════════════════════════════
  
  QLoRA on small GPU (≤16 GB VRAM):
    → paged_adamw_8bit (minimum memory)
    → Combine with gradient_checkpointing=True
    
  QLoRA on medium GPU (24 GB VRAM):
    → paged_adamw_32bit (safety + full precision optimizer)
    → Can use higher batch sizes
    
  QLoRA on large GPU (≥40 GB VRAM):
    → adamw_torch is usually fine
    → Paged optimizer provides safety net against spikes
    
  LoRA (no quantization) on any GPU:
    → paged_adamw_8bit if memory is tight
    → adamw_torch if memory is comfortable
  
  PERFORMANCE IMPACT:
  ═══════════════════════════════════════════════════════════════
  
  ┌──────────────────────┬────────────┬─────────────────────┐
  │ Optimizer            │ Overhead   │ When it matters     │
  ├──────────────────────┼────────────┼─────────────────────┤
  │ paged_adamw_32bit    │ 0-3%       │ Only during paging  │
  │ paged_adamw_8bit     │ 1-5%       │ Quantization + page │
  │ adamw_bnb_8bit       │ 1-3%       │ Quantization only   │
  │ adamw_torch          │ 0% (base)  │ Baseline            │
  └──────────────────────┴────────────┴─────────────────────┘
  
  → The overhead is negligible in almost all cases.
  → The memory savings can be the difference between
    training and OOM failure!
"""
        print(guide)


# ============================================================================
# SECTION 4: 8-BIT OPTIMIZER INTERNALS
# ============================================================================

class EightBitOptimizerInternals:
    """
    How 8-bit optimizers work internally.
    
    Key idea: Store optimizer states (moments) in INT8 instead of FP32.
    This saves 75% of optimizer memory.
    
    But INT8 has limited range — how to maintain quality?
    Answer: Dynamic quantization with per-tensor scaling.
    """
    
    @staticmethod
    def demonstrate_8bit_adam():
        """
        Show how 8-bit Adam works step by step.
        """
        print("\n" + "=" * 70)
        print("8-BIT ADAM INTERNALS")
        print("=" * 70)
        
        explanation = """
  Standard Adam Update:
  ═══════════════════════════════════════════════
    m_t = β₁ · m_{t-1} + (1 - β₁) · g_t        # First moment (FP32)
    v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²       # Second moment (FP32)
    m̂_t = m_t / (1 - β₁^t)                      # Bias correction
    v̂_t = v_t / (1 - β₂^t)                      # Bias correction
    θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε)     # Update

  8-bit Adam (bitsandbytes):
  ═══════════════════════════════════════════════
  Same math, but m and v stored in INT8!
  
  For each optimizer step:
    1. DEQUANTIZE: m_fp32 = dequantize(m_int8)
    2. UPDATE: m_fp32 = β₁ · m_fp32 + (1-β₁) · g
    3. QUANTIZE: m_int8 = quantize(m_fp32)
    4. Same for v
    5. Apply standard Adam update rule
  
  Dynamic Quantization:
    - Each tensor gets its own scale factor
    - The scale changes each step (dynamic)
    - Uses block-based quantization for better accuracy
    - Typical error: < 0.1% vs FP32 Adam
"""
        print(explanation)
        
        # Simulate 8-bit Adam
        print("  Simulated comparison: FP32 Adam vs 8-bit Adam")
        print("  " + "-" * 50)
        
        torch.manual_seed(42)
        
        # Simple optimization problem
        d = 100
        W_fp32 = torch.randn(d, requires_grad=False).clone()
        W_8bit = W_fp32.clone()
        
        # Simulated optimizer states
        m_fp32 = torch.zeros(d)
        v_fp32 = torch.zeros(d)
        m_8bit_stored = torch.zeros(d, dtype=torch.int8)
        v_8bit_stored = torch.zeros(d, dtype=torch.int8)
        m_scale = torch.tensor(1.0)  # Dynamic scale
        v_scale = torch.tensor(1.0)
        
        lr = 0.001
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        n_steps = 100
        diffs = []
        
        for t in range(1, n_steps + 1):
            # Simulated gradient
            g = torch.randn(d) * 0.01
            
            # FP32 Adam
            m_fp32 = beta1 * m_fp32 + (1 - beta1) * g
            v_fp32 = beta2 * v_fp32 + (1 - beta2) * g**2
            m_hat = m_fp32 / (1 - beta1**t)
            v_hat = v_fp32 / (1 - beta2**t)
            W_fp32 = W_fp32 - lr * m_hat / (v_hat.sqrt() + eps)
            
            # 8-bit Adam (simulated)
            # Dequantize
            m_deq = m_8bit_stored.float() * m_scale / 127
            v_deq = v_8bit_stored.float() * v_scale / 127
            
            # Update
            m_deq = beta1 * m_deq + (1 - beta1) * g
            v_deq = beta2 * v_deq + (1 - beta2) * g**2
            
            # Quantize back to INT8
            m_scale = m_deq.abs().max().clamp(min=1e-10)
            v_scale = v_deq.abs().max().clamp(min=1e-10)
            m_8bit_stored = (m_deq / m_scale * 127).round().clamp(-128, 127).to(torch.int8)
            v_8bit_stored = (v_deq / v_scale * 127).round().clamp(-128, 127).to(torch.int8)
            
            # Apply update (in FP32)
            m_hat_8 = m_deq / (1 - beta1**t)
            v_hat_8 = v_deq / (1 - beta2**t)
            W_8bit = W_8bit - lr * m_hat_8 / (v_hat_8.sqrt() + eps)
            
            diffs.append((W_fp32 - W_8bit).abs().max().item())
        
        print(f"\n  After {n_steps} steps:")
        print(f"    Max weight difference:  {diffs[-1]:.6f}")
        print(f"    Mean weight difference: {sum(diffs)/len(diffs):.6f}")
        print(f"    Relative difference:    {diffs[-1] / W_fp32.abs().mean().item():.4%}")
        print(f"\n  → 8-bit Adam closely tracks FP32 Adam")
        print(f"  → The quantization noise in optimizer states is negligible")


# ============================================================================
# SECTION 5: GRADIENT CHECKPOINTING SYNERGY
# ============================================================================

class GradientCheckpointingSynergy:
    """
    Gradient checkpointing works synergistically with paged optimizers
    in QLoRA training.
    """
    
    @staticmethod
    def explain_gradient_checkpointing():
        """
        Gradient checkpointing trades compute for memory:
        - Instead of storing ALL activations for the backward pass,
          only store checkpoints every N layers
        - Recompute intermediate activations during backward pass
        - Saves ~60-70% activation memory at ~20-30% compute cost
        """
        print("\n" + "=" * 70)
        print("GRADIENT CHECKPOINTING + PAGED OPTIMIZERS")
        print("=" * 70)
        
        explanation = """
  Without Gradient Checkpointing:
  ═══════════════════════════════════════════════════════════
  
  Forward pass stores ALL activations:
  Layer 1 → [save act1] → Layer 2 → [save act2] → ... → Layer N → [save actN]
  
  Memory: O(N) activations stored
  
  
  With Gradient Checkpointing:
  ═══════════════════════════════════════════════════════════
  
  Forward pass stores ONLY checkpoints:
  Layer 1 → [SAVE] → Layer 2 → [skip] → Layer 3 → [SAVE] → ...
  
  Backward pass RECOMPUTES from checkpoints:
  Layer 3 backward needs act2 → recompute from checkpoint at Layer 1
  
  Memory: O(√N) activations stored
  Compute: ~1.3x forward pass (recomputation overhead)
  
  
  COMBINED with QLoRA + Paged Optimizers:
  ═══════════════════════════════════════════════════════════
  
  ┌────────────────────────────────────────────────────┐
  │  Memory Component         │ Technique    │ Savings │
  ├────────────────────────────┼──────────────┼─────────┤
  │  Base model weights       │ NF4 (4-bit)  │ 75%     │
  │  Quantization scales      │ Double quant │ 75%*    │
  │  Optimizer states         │ Paged 8-bit  │ 75%**   │
  │  Activations              │ Grad ckpt    │ 60-70%  │
  │  LoRA weights + gradients │ (small)      │ —       │
  └────────────────────────────┴──────────────┴─────────┘
  * = of the scale overhead
  ** = of the optimizer state memory
  
  This combination is what allows training 65B models on 48GB GPUs!
"""
        print(explanation)
    
    @staticmethod
    def show_combined_config():
        """
        Complete QLoRA config with all memory optimizations.
        """
        print("\n" + "=" * 70)
        print("COMPLETE MEMORY-OPTIMIZED QLoRA CONFIG")
        print("=" * 70)
        
        code = '''
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer

# ══════════════════════════════════════════════════════════════════
# MAXIMUM MEMORY EFFICIENCY CONFIG
# ══════════════════════════════════════════════════════════════════

# 1. NF4 + Double Quantization (base model in 4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. Prepare for k-bit training (handles gradient checkpointing compatibility)
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,   # ← Activation memory savings
)

# 3. LoRA (small trainable adapter)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# 4. Paged 8-bit optimizer + gradient checkpointing
training_args = TrainingArguments(
    output_dir="./qlora_output",
    
    # Optimizer: Paged 8-bit AdamW
    optim="paged_adamw_8bit",          # ← Paged optimizer!
    learning_rate=2e-4,
    weight_decay=0.01,
    
    # Gradient checkpointing (activation memory)
    gradient_checkpointing=True,        # ← Activation savings!
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    # Gradient accumulation (effective batch size)
    per_device_train_batch_size=2,      # Small batch per step
    gradient_accumulation_steps=8,       # Effective batch = 16
    
    # Training
    num_train_epochs=3,
    max_steps=-1,
    warmup_ratio=0.03,
    
    # Precision
    bf16=True,                          # BF16 compute
    
    # Logging
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    report_to="none",
)

# 5. Train!
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=512,
)

trainer.train()

# Memory usage with all optimizations for LLaMA-7B:
#   Base model (NF4 + double quant):  ~3.5 GB
#   LoRA weights (BF16):              ~0.04 GB  
#   Optimizer states (8-bit paged):   ~0.08 GB
#   Activations (grad checkpoint):    ~1.5 GB
#   Gradients + overhead:             ~0.5 GB
#   ───────────────────────────────────────────
#   TOTAL:                            ~5.6 GB
#   
#   → Fits on a 8GB GPU! (with small batch size)
'''
        print(code)
        return code


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all paged optimizer demonstrations."""
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║        PAGED OPTIMIZERS — QLoRA's THIRD INNOVATION             ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    # Section 1: Optimizer memory analysis
    analysis = OptimizerMemoryAnalysis()
    analysis.analyze_optimizer_memory()
    analysis.show_memory_spike_problem()
    
    # Section 2: Unified memory
    unified = UnifiedMemoryExplained()
    unified.explain()
    
    # Section 3: Paged optimizer usage
    paged = PagedOptimizers()
    paged.show_available_optimizers()
    paged.show_usage_with_trainer()
    paged.show_when_to_use()
    
    # Section 4: 8-bit optimizer internals
    internals = EightBitOptimizerInternals()
    internals.demonstrate_8bit_adam()
    
    # Section 5: Gradient checkpointing synergy
    synergy = GradientCheckpointingSynergy()
    synergy.explain_gradient_checkpointing()
    synergy.show_combined_config()
    
    print("\n" + "=" * 70)
    print("  MODULE COMPLETE")
    print("=" * 70)
    print("""
    Key takeaways:
    ✓ Optimizer states (AdamW moments) consume 8 bytes/param in FP32
    ✓ Paged optimizers use CUDA unified memory for auto CPU↔GPU paging
    ✓ Memory spikes during training are handled automatically
    ✓ 8-bit paged optimizer: 2 bytes/param with CPU overflow
    ✓ Combined with NF4 + double quant + gradient checkpointing:
      → 7B model trainable on ~6 GB GPU memory!
    ✓ Performance overhead is negligible (<5%)
    """)


if __name__ == "__main__":
    main()
