"""
LoRA Merge & Deployment Strategies
===================================

This module covers everything needed to go from a trained LoRA adapter
to a production-ready model:

1. Weight Merging Fundamentals
   - How LoRA weights merge back into the base model
   - The math: W_merged = W_base + (alpha/r) * B @ A
   - Precision considerations during merging

2. Merging Methods
   - Standard merge (merge_and_unload)
   - Weighted merging for multi-adapter scenarios
   - Task arithmetic with LoRA adapters
   - TIES merging for conflict resolution

3. Multi-Adapter Management
   - Loading multiple adapters simultaneously
   - Adapter switching at inference time
   - Adapter composition and stacking

4. Export Formats
   - SafeTensors export
   - GGUF conversion for llama.cpp
   - ONNX export for cross-platform deployment

5. Deployment Patterns
   - Serving merged models
   - Serving with dynamic adapter loading
   - Batched inference with different adapters

6. Optimization for Inference
   - Quantization after merging
   - KV-cache optimization
   - Continuous batching considerations

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import json
import os
import shutil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================================
# SECTION 1: WEIGHT MERGING FUNDAMENTALS
# ============================================================================

class LoRAMergingEngine:
    """
    Demonstrates the mathematics and mechanics of merging LoRA weights
    back into a base model.
    
    Key equation:
        W_merged = W_base + (alpha / r) * B @ A
    
    Where:
        - W_base: Original pretrained weight matrix [d_out x d_in]
        - A: LoRA down-projection [r x d_in], initialized from N(0, σ²)
        - B: LoRA up-projection [d_out x r], initialized to zeros
        - alpha: Scaling factor (hyperparameter)
        - r: Rank (hyperparameter)
    """
    
    def __init__(self):
        self.merge_history = []
    
    def demonstrate_merge_math(
        self,
        d_in: int = 768,
        d_out: int = 768,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        """
        Step-by-step demonstration of the LoRA merge operation.
        
        Shows exactly what happens when we merge LoRA weights back
        into the base model.
        """
        print("=" * 70)
        print("LoRA WEIGHT MERGING — STEP BY STEP")
        print("=" * 70)
        
        # Step 1: Create base weight matrix
        print(f"\n1. Base weight matrix W_base: [{d_out} x {d_in}]")
        W_base = torch.randn(d_out, d_in) * 0.02  # Typical initialization scale
        print(f"   Shape: {W_base.shape}")
        print(f"   Norm:  {W_base.norm():.4f}")
        print(f"   Parameters: {W_base.numel():,}")
        
        # Step 2: Create LoRA matrices
        print(f"\n2. LoRA matrices (rank={rank}):")
        A = torch.randn(rank, d_in) * (1.0 / rank**0.5)  # Kaiming-like init
        B = torch.zeros(d_out, rank)  # Zero init → ΔW starts at 0
        print(f"   A (down-proj): [{rank} x {d_in}] = {A.numel():,} params")
        print(f"   B (up-proj):   [{d_out} x {rank}] = {B.numel():,} params")
        print(f"   Total LoRA params: {A.numel() + B.numel():,}")
        print(f"   Compression ratio: {W_base.numel() / (A.numel() + B.numel()):.1f}x")
        
        # Step 3: Simulate training (B changes from zeros)
        print("\n3. After training (B is no longer zero):")
        B_trained = torch.randn(d_out, rank) * 0.01  # Simulated trained values
        A_trained = A + torch.randn_like(A) * 0.005  # A also changes during training
        
        # Step 4: Compute the weight delta
        print(f"\n4. Computing weight delta ΔW = (α/r) × B @ A")
        scaling = alpha / rank
        print(f"   Scaling factor α/r = {alpha}/{rank} = {scaling:.2f}")
        
        delta_W = scaling * (B_trained @ A_trained)
        print(f"   ΔW shape: {delta_W.shape}")
        print(f"   ΔW norm:  {delta_W.norm():.4f}")
        print(f"   ΔW/W_base ratio: {delta_W.norm() / W_base.norm():.4f}")
        
        # Step 5: Merge
        print("\n5. Merging: W_merged = W_base + ΔW")
        W_merged = W_base + delta_W
        print(f"   W_merged shape: {W_merged.shape}")
        print(f"   W_merged norm:  {W_merged.norm():.4f}")
        
        # Step 6: Verify equivalence
        print("\n6. Verification — Forward pass equivalence:")
        x = torch.randn(1, d_in)
        
        # Unmerged path: y = x @ W_base.T + scaling * x @ A.T @ B.T
        y_unmerged = (x @ W_base.T) + scaling * (x @ A_trained.T @ B_trained.T)
        
        # Merged path: y = x @ W_merged.T
        y_merged = x @ W_merged.T
        
        diff = (y_unmerged - y_merged).abs().max().item()
        print(f"   Max absolute difference: {diff:.2e}")
        print(f"   Equivalent: {'✓ YES' if diff < 1e-5 else '✗ NO'}")
        
        return {
            "base_params": W_base.numel(),
            "lora_params": A.numel() + B.numel(),
            "compression_ratio": W_base.numel() / (A.numel() + B.numel()),
            "delta_norm": delta_W.norm().item(),
            "merge_error": diff,
        }
    
    def demonstrate_precision_effects(self):
        """
        Shows how numerical precision affects merging quality.
        
        Important: Merging in fp16 can introduce more error than fp32.
        Best practice: Merge in fp32, then quantize.
        """
        print("\n" + "=" * 70)
        print("PRECISION EFFECTS ON MERGING")
        print("=" * 70)
        
        d = 1024
        rank = 16
        alpha = 32.0
        scaling = alpha / rank
        
        # Create reference in fp64
        W_base = torch.randn(d, d, dtype=torch.float64) * 0.02
        A = torch.randn(rank, d, dtype=torch.float64) * 0.01
        B = torch.randn(d, rank, dtype=torch.float64) * 0.01
        
        W_merged_fp64 = W_base + scaling * (B @ A)
        
        results = {}
        for dtype, name in [
            (torch.float32, "FP32"),
            (torch.float16, "FP16"),
            (torch.bfloat16, "BF16"),
        ]:
            W_b = W_base.to(dtype)
            A_d = A.to(dtype)
            B_d = B.to(dtype)
            
            W_m = W_b + scaling * (B_d @ A_d)
            
            # Compare against fp64 reference
            error = (W_m.to(torch.float64) - W_merged_fp64).abs()
            max_err = error.max().item()
            mean_err = error.mean().item()
            
            results[name] = {"max_error": max_err, "mean_error": mean_err}
            print(f"\n  {name}:")
            print(f"    Max absolute error:  {max_err:.2e}")
            print(f"    Mean absolute error: {mean_err:.2e}")
        
        print("\n  → Recommendation: Always merge in FP32 for maximum accuracy.")
        print("    Convert to FP16/BF16/quantized AFTER merging.")
        
        return results


# ============================================================================
# SECTION 2: MERGING METHODS
# ============================================================================

class AdvancedMergingStrategies:
    """
    Advanced strategies for merging LoRA adapters, including:
    - Standard merge
    - Weighted merge (multiple adapters with weights)
    - Task arithmetic
    - TIES merging
    """
    
    @staticmethod
    def standard_merge_with_peft():
        """
        Standard merging using HuggingFace PEFT library.
        This is the most common approach.
        """
        print("\n" + "=" * 70)
        print("STANDARD MERGE WITH PEFT")
        print("=" * 70)
        
        code = '''
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Step 1: Load base model ──────────────────────────────────────
base_model_name = "distilgpt2"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32,   # Merge in fp32 for accuracy
    device_map="cpu",            # CPU is fine for merging
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# ── Step 2: Load LoRA adapter on top of base model ──────────────
adapter_path = "./my_lora_adapter"
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    torch_dtype=torch.float32,
)

# ── Step 3: Merge and unload ────────────────────────────────────
# This permanently merges LoRA weights into the base model
# and removes the LoRA layers, returning a standard model
merged_model = model.merge_and_unload(
    progressbar=True,       # Show merge progress
    safe_merge=True,        # Check for NaN after merging
)

# ── Step 4: Verify the merge ────────────────────────────────────
# The merged model should have NO LoRA parameters
total_params = sum(p.numel() for p in merged_model.parameters())
print(f"Total parameters after merge: {total_params:,}")

# Test generation
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = merged_model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# ── Step 5: Save the merged model ───────────────────────────────
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")

# The saved model is a standard HF model — no PEFT dependency needed
# to load it later!
'''
        print(code)
        return code
    
    @staticmethod
    def weighted_adapter_merge():
        """
        Merge multiple LoRA adapters with different weights.
        Useful for combining task-specific adapters.
        
        Example: 70% coding adapter + 30% writing adapter
        """
        print("\n" + "=" * 70)
        print("WEIGHTED MULTI-ADAPTER MERGE")
        print("=" * 70)
        
        code = '''
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Load first adapter
model = PeftModel.from_pretrained(base_model, "./adapter_coding")

# Load additional adapters with names
model.load_adapter("./adapter_writing", adapter_name="writing")
model.load_adapter("./adapter_math", adapter_name="math")

# ── Method 1: Weighted merge via add_weighted_adapter ────────────
# Combine adapters with specific weights
model.add_weighted_adapter(
    adapters=["default", "writing", "math"],   # Adapter names
    weights=[0.5, 0.3, 0.2],                   # Weights for each
    adapter_name="combined",                     # Name for merged adapter
    combination_type="linear",                   # Linear combination
)

# Activate the combined adapter
model.set_adapter("combined")

# Now merge into base model
merged = model.merge_and_unload()
merged.save_pretrained("./merged_multi_adapter")

# ── Method 2: Task arithmetic combination types ──────────────────
# PEFT supports several combination strategies:

# 1. "linear" — Simple weighted sum: W = Σ(w_i * ΔW_i)
model.add_weighted_adapter(
    adapters=["default", "writing"],
    weights=[0.7, 0.3],
    adapter_name="linear_combo",
    combination_type="linear",
)

# 2. "ties" — TIES merging (trim, elect sign, disjoint merge)
model.add_weighted_adapter(
    adapters=["default", "writing", "math"],
    weights=[1.0, 1.0, 1.0],
    adapter_name="ties_combo",
    combination_type="ties",
    density=0.5,  # Keep top 50% of params by magnitude
)

# 3. "dare_ties" — DARE + TIES (drop and rescale + TIES)
model.add_weighted_adapter(
    adapters=["default", "writing", "math"],
    weights=[1.0, 1.0, 1.0],
    adapter_name="dare_ties_combo",
    combination_type="dare_ties",
    density=0.5,
)

# 4. "dare_linear" — DARE + linear combination
model.add_weighted_adapter(
    adapters=["default", "writing"],
    weights=[0.6, 0.4],
    adapter_name="dare_linear_combo",
    combination_type="dare_linear",
    density=0.7,
)
'''
        print(code)
        return code
    
    def demonstrate_task_arithmetic(self):
        """
        Task arithmetic with LoRA adapters.
        
        Concept: LoRA deltas can be added, subtracted, and scaled
        like vectors to create new behaviors.
        
        - Addition: Combine capabilities
        - Subtraction: Remove capabilities
        - Scaling: Control strength
        """
        print("\n" + "=" * 70)
        print("TASK ARITHMETIC WITH LoRA ADAPTERS")
        print("=" * 70)
        
        d_in, d_out, rank = 64, 64, 4
        alpha = 8.0
        scaling = alpha / rank
        
        # Base model weight
        W_base = torch.randn(d_out, d_in) * 0.02
        
        # Task A adapter (e.g., "coding")
        A_a = torch.randn(rank, d_in) * 0.01
        B_a = torch.randn(d_out, rank) * 0.01
        delta_A = scaling * (B_a @ A_a)
        
        # Task B adapter (e.g., "writing")
        A_b = torch.randn(rank, d_in) * 0.01
        B_b = torch.randn(d_out, rank) * 0.01
        delta_B = scaling * (B_b @ A_b)
        
        # Task C adapter (e.g., "toxicity" - unwanted behavior)
        A_c = torch.randn(rank, d_in) * 0.01
        B_c = torch.randn(d_out, rank) * 0.01
        delta_C = scaling * (B_c @ A_c)
        
        print("\n1. ADDITION — Combine coding + writing:")
        W_combined = W_base + delta_A + delta_B
        print(f"   W_combined = W_base + ΔW_coding + ΔW_writing")
        print(f"   Norm change: {W_base.norm():.4f} → {W_combined.norm():.4f}")
        
        print("\n2. SUBTRACTION — Remove toxicity:")
        W_detoxified = W_base + delta_A - 0.5 * delta_C
        print(f"   W_detoxified = W_base + ΔW_coding - 0.5 × ΔW_toxicity")
        print(f"   Norm change: {W_base.norm():.4f} → {W_detoxified.norm():.4f}")
        
        print("\n3. SCALING — Control adapter strength:")
        for strength in [0.25, 0.5, 0.75, 1.0, 1.5]:
            W_scaled = W_base + strength * delta_A
            print(f"   strength={strength:.2f}: norm={W_scaled.norm():.4f}, "
                  f"Δnorm={(W_scaled - W_base).norm():.4f}")
        
        print("\n4. INTERPOLATION — Smooth transition between tasks:")
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            W_interp = W_base + (1 - t) * delta_A + t * delta_B
            print(f"   t={t:.2f}: coding={(1-t):.0%}, writing={t:.0%}, "
                  f"norm={W_interp.norm():.4f}")
        
        print("\n  → Task arithmetic enables flexible model composition")
        print("    without retraining!")
    
    def demonstrate_ties_merging(self):
        """
        TIES (Trim, Elect Sign, Disjoint merge) — a method for
        resolving conflicts when merging multiple adapters.
        
        Steps:
        1. TRIM: Zero out small-magnitude changes (keep top-k%)
        2. ELECT SIGN: For each parameter, vote on the sign
        3. DISJOINT MERGE: Average only same-sign values
        """
        print("\n" + "=" * 70)
        print("TIES MERGING (Trim, Elect Sign, Disjoint Merge)")
        print("=" * 70)
        
        # Simulated task vectors (LoRA deltas)
        d = 16
        n_adapters = 3
        density = 0.5  # Keep top 50%
        
        # Create task vectors with some conflicts (opposing signs)
        task_vectors = []
        for i in range(n_adapters):
            tv = torch.randn(d) * 0.1
            task_vectors.append(tv)
            print(f"\n  Task vector {i+1}: {tv[:8].tolist()}")
        
        # Step 1: TRIM — zero out small values
        print(f"\n  Step 1: TRIM (density={density}, keep top {density*100:.0f}%)")
        trimmed = []
        for i, tv in enumerate(task_vectors):
            threshold = tv.abs().quantile(1 - density)
            mask = tv.abs() >= threshold
            trimmed_tv = tv * mask.float()
            trimmed.append(trimmed_tv)
            n_kept = mask.sum().item()
            print(f"    Adapter {i+1}: kept {n_kept}/{d} values "
                  f"(threshold={threshold:.4f})")
        
        # Step 2: ELECT SIGN — majority vote
        print("\n  Step 2: ELECT SIGN (majority vote per parameter)")
        sign_votes = torch.stack([torch.sign(t) for t in trimmed])
        # For non-zero entries, count positive vs negative
        elected_sign = torch.sign(sign_votes.sum(dim=0))
        # Handle ties (sum=0) by taking first adapter's sign
        ties_mask = elected_sign == 0
        if ties_mask.any():
            elected_sign[ties_mask] = torch.sign(trimmed[0][ties_mask])
        print(f"    Elected signs: {elected_sign[:8].tolist()}")
        
        # Step 3: DISJOINT MERGE — average same-sign values only
        print("\n  Step 3: DISJOINT MERGE (average same-sign values)")
        merged = torch.zeros(d)
        counts = torch.zeros(d)
        
        for trimmed_tv in trimmed:
            # Only include values that agree with elected sign
            agree_mask = (torch.sign(trimmed_tv) == elected_sign) & (trimmed_tv != 0)
            merged += trimmed_tv * agree_mask.float()
            counts += agree_mask.float()
        
        # Average where we have contributions
        counts = counts.clamp(min=1)
        merged = merged / counts
        
        print(f"    Merged result: {merged[:8].tolist()}")
        
        # Compare with simple average
        simple_avg = torch.stack(task_vectors).mean(dim=0)
        print(f"\n  Comparison:")
        print(f"    Simple average norm: {simple_avg.norm():.4f}")
        print(f"    TIES merged norm:    {merged.norm():.4f}")
        print(f"    → TIES reduces interference between conflicting adapters")


# ============================================================================
# SECTION 3: MULTI-ADAPTER MANAGEMENT
# ============================================================================

class MultiAdapterManager:
    """
    Manages multiple LoRA adapters for dynamic loading and switching.
    
    Use cases:
    - Serving different tasks with the same base model
    - A/B testing adapter variants
    - Personalized adapters per user/tenant
    """
    
    @staticmethod
    def dynamic_adapter_loading():
        """
        Load and switch between adapters at inference time.
        """
        print("\n" + "=" * 70)
        print("DYNAMIC ADAPTER LOADING & SWITCHING")
        print("=" * 70)
        
        code = '''
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model once
base_model = AutoModelForCausalLM.from_pretrained(
    "distilgpt2",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Load the first adapter (becomes "default")
model = PeftModel.from_pretrained(
    base_model,
    "./adapters/coding_adapter",
    adapter_name="coding",
)

# Load additional adapters
model.load_adapter("./adapters/writing_adapter", adapter_name="writing")
model.load_adapter("./adapters/math_adapter", adapter_name="math")

# ── Switch between adapters ──────────────────────────────────────
def generate_with_adapter(model, tokenizer, prompt, adapter_name):
    """Generate text using a specific adapter."""
    model.set_adapter(adapter_name)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Use coding adapter
result = generate_with_adapter(model, tokenizer,
    "Write a Python function to sort a list:", "coding")

# Switch to writing adapter (no model reload needed!)
result = generate_with_adapter(model, tokenizer,
    "Write a poem about autumn:", "writing")

# Switch to math adapter
result = generate_with_adapter(model, tokenizer,
    "Solve: What is the derivative of x²?", "math")

# ── Disable all adapters (use base model) ────────────────────────
with model.disable_adapter():
    # This context manager temporarily disables LoRA
    inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)
    base_outputs = model.generate(**inputs, max_new_tokens=50)
    print("Base model output:", tokenizer.decode(base_outputs[0]))

# ── Check active adapter ─────────────────────────────────────────
print(f"Active adapter: {model.active_adapter}")
print(f"Available adapters: {list(model.peft_config.keys())}")

# ── Delete an adapter to free memory ─────────────────────────────
model.delete_adapter("math")
'''
        print(code)
        return code
    
    @staticmethod
    def adapter_serving_architecture():
        """
        Architecture for serving multiple adapters efficiently.
        """
        print("\n" + "=" * 70)
        print("ADAPTER SERVING ARCHITECTURE")
        print("=" * 70)
        
        architecture = '''
┌─────────────────────────────────────────────────────────┐
│                    Request Router                        │
│   Routes requests to appropriate adapter based on       │
│   task type, user ID, or A/B test assignment            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Base Model (Shared, in GPU)                 │
│                                                         │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│   │ Adapter A │  │ Adapter B │  │ Adapter C │  ← Hot   │
│   │ (coding)  │  │ (writing) │  │ (math)    │  adapters │
│   └──────────┘  └──────────┘  └──────────┘            │
│                                                         │
│   ┌──────────────────────────────────────────┐         │
│   │         Adapter Cache (LRU)               │ ← Cold  │
│   │  Evicts least-used adapters from GPU      │ adapters │
│   └──────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────┘

Memory Budget Analysis:
─────────────────────────────────────────────────────
  Base model (7B, fp16):           ~14 GB
  Each LoRA adapter (r=16):        ~20-40 MB
  100 adapters in memory:          ~2-4 GB
  Total:                           ~16-18 GB
  
  → One GPU can serve 100+ different "models"!
─────────────────────────────────────────────────────

Serving Strategies:
  1. Pre-merge → Fastest inference, but N copies of model
  2. Dynamic switching → One base model, swap adapters per request
  3. Batched adapters → Group requests by adapter, batch process
'''
        print(architecture)
        
        # Implementation example
        code = '''
import torch
from collections import OrderedDict
from typing import Optional

class AdapterCache:
    """LRU cache for LoRA adapters on GPU."""
    
    def __init__(self, model, max_adapters: int = 10):
        self.model = model
        self.max_adapters = max_adapters
        self.loaded_adapters = OrderedDict()  # name -> adapter_path
        self.adapter_dir = "./adapters"
    
    def get_adapter(self, adapter_name: str) -> None:
        """Load adapter if not cached, evict LRU if needed."""
        if adapter_name in self.loaded_adapters:
            # Move to end (most recently used)
            self.loaded_adapters.move_to_end(adapter_name)
        else:
            # Evict LRU if cache is full
            if len(self.loaded_adapters) >= self.max_adapters:
                evicted_name, _ = self.loaded_adapters.popitem(last=False)
                self.model.delete_adapter(evicted_name)
                print(f"Evicted adapter: {evicted_name}")
            
            # Load new adapter
            adapter_path = f"{self.adapter_dir}/{adapter_name}"
            self.model.load_adapter(adapter_path, adapter_name=adapter_name)
            self.loaded_adapters[adapter_name] = adapter_path
            print(f"Loaded adapter: {adapter_name}")
        
        # Activate the adapter
        self.model.set_adapter(adapter_name)
    
    def generate(self, adapter_name: str, prompt: str, **kwargs):
        """Generate with a specific adapter."""
        self.get_adapter(adapter_name)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        return self.model.generate(**inputs, **kwargs)
'''
        print(code)
        return code


# ============================================================================
# SECTION 4: EXPORT FORMATS
# ============================================================================

class ModelExporter:
    """
    Export merged LoRA models to various formats for deployment.
    """
    
    @staticmethod
    def export_safetensors():
        """
        Export to SafeTensors format (recommended for HuggingFace ecosystem).
        SafeTensors is memory-mapped, fast to load, and secure.
        """
        print("\n" + "=" * 70)
        print("EXPORT TO SAFETENSORS")
        print("=" * 70)
        
        code = '''
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load and merge
base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model = PeftModel.from_pretrained(base_model, "./my_lora_adapter")
merged = model.merge_and_unload()

# Save as SafeTensors (default in modern transformers)
output_dir = "./exported/safetensors_model"
merged.save_pretrained(
    output_dir,
    safe_serialization=True,    # Use safetensors format
    max_shard_size="2GB",       # Shard large models
)

# Also save tokenizer and config
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.save_pretrained(output_dir)

# The output directory contains:
#   - model.safetensors (or model-00001-of-00002.safetensors for sharded)
#   - config.json
#   - tokenizer.json
#   - tokenizer_config.json
#   - special_tokens_map.json

print(f"Model exported to {output_dir}")
print("Files:", os.listdir(output_dir))
'''
        print(code)
        return code
    
    @staticmethod
    def export_gguf():
        """
        Export to GGUF format for llama.cpp inference.
        GGUF supports various quantization levels (Q4_0, Q4_K_M, Q5_K_M, Q8_0).
        """
        print("\n" + "=" * 70)
        print("EXPORT TO GGUF (for llama.cpp)")
        print("=" * 70)
        
        code = '''
# ── Method 1: Using llama.cpp's convert script ──────────────────
# First, merge and save as HF format:

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "./my_lora_adapter")
merged = model.merge_and_unload()
merged.save_pretrained("./merged_hf_model")

# Then convert using llama.cpp (run in terminal):
# 
#   # Clone llama.cpp if needed
#   git clone https://github.com/ggerganov/llama.cpp
#   cd llama.cpp
#   
#   # Convert HF model to GGUF (FP16)
#   python convert_hf_to_gguf.py ../merged_hf_model --outfile model-f16.gguf --outtype f16
#   
#   # Quantize to various levels
#   ./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
#   ./llama-quantize model-f16.gguf model-q5_k_m.gguf Q5_K_M
#   ./llama-quantize model-f16.gguf model-q8_0.gguf Q8_0

# ── Method 2: Using LoRA adapter directly with llama.cpp ─────────
# llama.cpp can apply LoRA at runtime (no merging needed):
#
#   ./llama-cli -m base_model.gguf --lora my_adapter.gguf -p "prompt"
#
# This loads the base model + applies LoRA dynamically.
# Useful for testing different adapters without re-exporting.

# ── Quantization Level Reference ─────────────────────────────────
quantization_levels = {
    "Q2_K":   {"bits": 2.5, "quality": "Very low",  "size_7b": "2.7 GB"},
    "Q3_K_M": {"bits": 3.3, "quality": "Low",       "size_7b": "3.3 GB"},
    "Q4_0":   {"bits": 4.0, "quality": "Medium-low", "size_7b": "3.8 GB"},
    "Q4_K_M": {"bits": 4.5, "quality": "Medium",    "size_7b": "4.1 GB"},
    "Q5_0":   {"bits": 5.0, "quality": "Medium",    "size_7b": "4.7 GB"},
    "Q5_K_M": {"bits": 5.5, "quality": "Medium-high","size_7b": "4.8 GB"},
    "Q6_K":   {"bits": 6.6, "quality": "High",      "size_7b": "5.5 GB"},
    "Q8_0":   {"bits": 8.0, "quality": "Very high",  "size_7b": "7.2 GB"},
    "F16":    {"bits": 16,  "quality": "Original",   "size_7b": "13.5 GB"},
}

print("\\nGGUF Quantization Levels:")
print(f"{'Level':<10} {'Bits':>5} {'Quality':<15} {'7B Size':>10}")
print("-" * 45)
for level, info in quantization_levels.items():
    print(f"{level:<10} {info['bits']:>5.1f} {info['quality']:<15} "
          f"{info['size_7b']:>10}")
'''
        print(code)
        return code
    
    @staticmethod
    def export_onnx():
        """
        Export to ONNX format for cross-platform deployment.
        """
        print("\n" + "=" * 70)
        print("EXPORT TO ONNX")
        print("=" * 70)
        
        code = '''
# ── Using Optimum for ONNX export ────────────────────────────────
# pip install optimum[onnxruntime]

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

# Option 1: Export merged HF model to ONNX
model = ORTModelForCausalLM.from_pretrained(
    "./merged_hf_model",
    export=True,                 # Trigger ONNX export
    provider="CPUExecutionProvider",
)
tokenizer = AutoTokenizer.from_pretrained("./merged_hf_model")

# Save ONNX model
model.save_pretrained("./exported/onnx_model")
tokenizer.save_pretrained("./exported/onnx_model")

# Option 2: Export with optimization
from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig

optimizer = ORTOptimizer.from_pretrained(model)
optimization_config = OptimizationConfig(
    optimization_level=99,       # Maximum optimization
    optimize_for_gpu=False,      # Set True for GPU deployment
    fp16=False,                  # Set True for FP16 inference
)
optimizer.optimize(
    save_dir="./exported/onnx_optimized",
    optimization_config=optimization_config,
)

# Option 3: Export with quantization
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

quantizer = ORTQuantizer.from_pretrained(model)
quant_config = AutoQuantizationConfig.avx512_vnni(
    is_static=False,             # Dynamic quantization
    per_channel=True,
)
quantizer.quantize(
    save_dir="./exported/onnx_quantized",
    quantization_config=quant_config,
)

# Inference with ONNX model
from optimum.onnxruntime import ORTModelForCausalLM

ort_model = ORTModelForCausalLM.from_pretrained("./exported/onnx_optimized")
inputs = tokenizer("Hello", return_tensors="pt")
outputs = ort_model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
'''
        print(code)
        return code


# ============================================================================
# SECTION 5: DEPLOYMENT PATTERNS
# ============================================================================

class DeploymentPatterns:
    """
    Production deployment patterns for LoRA fine-tuned models.
    """
    
    @staticmethod
    def vllm_deployment():
        """
        Deploy with vLLM — fastest serving engine for LLMs.
        Supports dynamic LoRA adapter loading.
        """
        print("\n" + "=" * 70)
        print("DEPLOYMENT WITH vLLM")
        print("=" * 70)
        
        code = '''
# ── Option 1: Serve merged model ─────────────────────────────────
# Terminal:
#   python -m vllm.entrypoints.openai.api_server \\
#       --model ./merged_model \\
#       --host 0.0.0.0 \\
#       --port 8000

# ── Option 2: Serve with dynamic LoRA loading ────────────────────
# Terminal:
#   python -m vllm.entrypoints.openai.api_server \\
#       --model meta-llama/Llama-2-7b-hf \\
#       --enable-lora \\
#       --lora-modules coding=./adapters/coding \\
#                      writing=./adapters/writing \\
#       --max-lora-rank 64 \\
#       --host 0.0.0.0 \\
#       --port 8000

# Client usage (OpenAI-compatible API):
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
)

# Use specific adapter via model name
response = client.chat.completions.create(
    model="coding",  # ← This selects the LoRA adapter
    messages=[
        {"role": "user", "content": "Write a quicksort in Python"}
    ],
    max_tokens=200,
)
print(response.choices[0].message.content)

# ── Option 3: Programmatic vLLM with LoRA ────────────────────────
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_lora=True,
    max_lora_rank=64,
    max_loras=4,           # Max adapters loaded simultaneously
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=200)

# Make a request with a specific LoRA adapter
outputs = llm.generate(
    ["Write a poem about AI:"],
    sampling_params,
    lora_request=LoRARequest(
        lora_name="writing",
        lora_int_id=1,
        lora_path="./adapters/writing",
    ),
)
'''
        print(code)
        return code
    
    @staticmethod
    def text_generation_inference():
        """
        Deploy with HuggingFace Text Generation Inference (TGI).
        """
        print("\n" + "=" * 70)
        print("DEPLOYMENT WITH TGI (Text Generation Inference)")
        print("=" * 70)
        
        code = '''
# ── Docker deployment with TGI ───────────────────────────────────
# Serve merged model:
#   docker run --gpus all -p 8080:80 \\
#       -v ./merged_model:/model \\
#       ghcr.io/huggingface/text-generation-inference:latest \\
#       --model-id /model \\
#       --max-batch-total-tokens 4096 \\
#       --max-input-tokens 1024 \\
#       --max-total-tokens 2048

# Serve with LoRA adapter:
#   docker run --gpus all -p 8080:80 \\
#       -v ./adapters:/adapters \\
#       ghcr.io/huggingface/text-generation-inference:latest \\
#       --model-id meta-llama/Llama-2-7b-hf \\
#       --lora-adapters coding=/adapters/coding \\
#       --max-batch-total-tokens 4096

# Client usage:
from huggingface_hub import InferenceClient

client = InferenceClient("http://localhost:8080")

# Simple generation
output = client.text_generation(
    "Write a function to reverse a string:",
    max_new_tokens=200,
    temperature=0.7,
)
print(output)

# Chat-style generation
output = client.chat.completions.create(
    model="tgi",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Explain decorators in Python."},
    ],
    max_tokens=500,
)
print(output.choices[0].message.content)
'''
        print(code)
        return code
    
    @staticmethod
    def fastapi_deployment():
        """
        Custom FastAPI deployment with adapter management.
        """
        print("\n" + "=" * 70)
        print("CUSTOM FASTAPI DEPLOYMENT")
        print("=" * 70)
        
        code = '''
# server.py — Custom FastAPI server with multi-adapter support
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from collections import OrderedDict
import asyncio

app = FastAPI(title="LoRA Model Server")

# ── Global State ─────────────────────────────────────────────────
class ModelServer:
    def __init__(self):
        self.base_model = None
        self.model = None
        self.tokenizer = None
        self.loaded_adapters = OrderedDict()
        self.max_cached_adapters = 10
        self.lock = asyncio.Lock()
    
    def initialize(self, base_model_name: str):
        """Load base model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.base_model.eval()
    
    async def generate(
        self,
        prompt: str,
        adapter_name: Optional[str] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
    ) -> str:
        async with self.lock:
            if adapter_name and self.model:
                # Load/activate adapter
                if adapter_name not in self.loaded_adapters:
                    adapter_path = f"./adapters/{adapter_name}"
                    if len(self.loaded_adapters) >= self.max_cached_adapters:
                        evicted, _ = self.loaded_adapters.popitem(last=False)
                        self.model.delete_adapter(evicted)
                    self.model.load_adapter(adapter_path, adapter_name)
                    self.loaded_adapters[adapter_name] = adapter_path
                
                self.loaded_adapters.move_to_end(adapter_name)
                self.model.set_adapter(adapter_name)
                active_model = self.model
            else:
                active_model = self.base_model
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(active_model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = active_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                )
            
            return self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

server = ModelServer()

# ── API Models ───────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    adapter: Optional[str] = None
    max_new_tokens: int = 100
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    text: str
    adapter_used: Optional[str]

# ── Endpoints ────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    server.initialize("distilgpt2")

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    try:
        text = await server.generate(
            request.prompt,
            request.adapter,
            request.max_new_tokens,
            request.temperature,
        )
        return GenerateResponse(text=text, adapter_used=request.adapter)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/adapters")
async def list_adapters():
    return {
        "loaded": list(server.loaded_adapters.keys()),
        "max_cached": server.max_cached_adapters,
    }

# Run with: uvicorn server:app --host 0.0.0.0 --port 8000
'''
        print(code)
        return code


# ============================================================================
# SECTION 6: POST-MERGE OPTIMIZATION
# ============================================================================

class PostMergeOptimization:
    """
    Optimization techniques to apply after merging LoRA weights.
    """
    
    @staticmethod
    def quantize_after_merge():
        """
        Quantize the merged model for efficient deployment.
        
        Best practice: Merge in FP32 → Quantize → Deploy
        """
        print("\n" + "=" * 70)
        print("POST-MERGE QUANTIZATION")
        print("=" * 70)
        
        code = '''
# ── Method 1: GPTQ Quantization ─────────────────────────────────
# pip install auto-gptq optimum
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer

# Load merged FP32 model
model = AutoModelForCausalLM.from_pretrained(
    "./merged_model",
    torch_dtype=torch.float32,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("./merged_model")

# Quantize to 4-bit with GPTQ
quantizer = GPTQQuantizer(
    bits=4,
    dataset="c4",           # Calibration dataset
    block_name_to_quantize="transformer.h",  # Model-specific
    model_seqlen=2048,
)

quantized_model = quantizer.quantize_model(model, tokenizer)
quantized_model.save_pretrained("./merged_model_gptq_4bit")

# ── Method 2: AWQ Quantization ───────────────────────────────────
# pip install autoawq
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(
    "./merged_model",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("./merged_model")

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}

model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized("./merged_model_awq_4bit")

# ── Method 3: BitsAndBytes (dynamic, no calibration needed) ──────
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load merged model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "./merged_model",
    quantization_config=bnb_config,
    device_map="auto",
)
# Note: BnB quantization is dynamic (applied during load)
# The model on disk remains in its original precision
'''
        print(code)
        return code
    
    @staticmethod
    def benchmark_merged_model():
        """
        Benchmark the merged model for latency and throughput.
        """
        print("\n" + "=" * 70)
        print("BENCHMARK MERGED MODEL")
        print("=" * 70)
        
        code = '''
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def benchmark_model(model_path, num_runs=10, max_new_tokens=50):
    """Benchmark a model's inference performance."""
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Warmup
    prompt = "The quick brown fox"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    for _ in range(3):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=10)
    
    # Benchmark
    latencies = []
    total_tokens = 0
    
    for i in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        latency = end - start
        n_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        latencies.append(latency)
        total_tokens += n_tokens
    
    avg_latency = sum(latencies) / len(latencies)
    tokens_per_sec = total_tokens / sum(latencies)
    
    # Memory usage
    if torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
    else:
        import psutil
        mem_gb = psutil.Process().memory_info().rss / 1e9
    
    results = {
        "model": model_path,
        "avg_latency_ms": avg_latency * 1000,
        "tokens_per_second": tokens_per_sec,
        "memory_gb": mem_gb,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
    
    print(f"\\nBenchmark Results for {model_path}:")
    print(f"  Avg latency:      {results['avg_latency_ms']:.1f} ms")
    print(f"  Throughput:        {results['tokens_per_second']:.1f} tokens/sec")
    print(f"  Memory:            {results['memory_gb']:.2f} GB")
    print(f"  Parameters:        {results['num_parameters']:,}")
    
    return results

# Compare base vs merged vs quantized
results = {}
for model_path in [
    "./base_model",
    "./merged_model",
    "./merged_model_gptq_4bit",
]:
    results[model_path] = benchmark_model(model_path)

# Print comparison table
print("\\n" + "=" * 60)
print(f"{'Model':<30} {'Latency':>10} {'Tok/s':>10} {'Memory':>10}")
print("-" * 60)
for path, r in results.items():
    name = path.split("/")[-1][:28]
    print(f"{name:<30} {r['avg_latency_ms']:>8.1f}ms "
          f"{r['tokens_per_second']:>8.1f} {r['memory_gb']:>8.2f}GB")
'''
        print(code)
        return code


# ============================================================================
# SECTION 7: COMPLETE MERGE & DEPLOY PIPELINE
# ============================================================================

def complete_merge_deploy_pipeline():
    """
    End-to-end pipeline: Train → Merge → Optimize → Deploy
    
    This ties everything together into a single workflow.
    """
    print("=" * 70)
    print("COMPLETE LoRA MERGE & DEPLOYMENT PIPELINE")
    print("=" * 70)
    
    code = '''
import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)
from datasets import load_dataset
from trl import SFTTrainer

# ═══════════════════════════════════════════════════════════════════
# STAGE 1: TRAINING
# ═══════════════════════════════════════════════════════════════════

def stage_1_train(
    base_model_name: str = "distilgpt2",
    output_dir: str = "./pipeline/adapter",
):
    """Train a LoRA adapter."""
    print("\\n[Stage 1] Training LoRA adapter...")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and prepare dataset
    dataset = load_dataset("Abirate/english_quotes", split="train[:500]")
    
    def format_example(example):
        return {"text": f"Quote: {example['quote']}\\nAuthor: {example['author']}"}
    
    dataset = dataset.map(format_example)
    
    # Training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    
    # Save adapter (NOT full model)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"[Stage 1] Adapter saved to {output_dir}")
    return output_dir


# ═══════════════════════════════════════════════════════════════════
# STAGE 2: MERGING
# ═══════════════════════════════════════════════════════════════════

def stage_2_merge(
    base_model_name: str = "distilgpt2",
    adapter_dir: str = "./pipeline/adapter",
    output_dir: str = "./pipeline/merged",
):
    """Merge LoRA weights into base model."""
    print("\\n[Stage 2] Merging LoRA adapter into base model...")
    
    # Load in FP32 for accurate merging
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    
    # Load adapter
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        torch_dtype=torch.float32,
    )
    
    # Merge and verify
    merged_model = model.merge_and_unload(
        progressbar=True,
        safe_merge=True,
    )
    
    # Save merged model
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Verify: merged model has no PEFT params
    has_lora = any("lora" in n for n, _ in merged_model.named_parameters())
    print(f"  Has LoRA params after merge: {has_lora} (should be False)")
    print(f"[Stage 2] Merged model saved to {output_dir}")
    
    return output_dir


# ═══════════════════════════════════════════════════════════════════
# STAGE 3: VALIDATION
# ═══════════════════════════════════════════════════════════════════

def stage_3_validate(
    base_model_name: str = "distilgpt2",
    adapter_dir: str = "./pipeline/adapter",
    merged_dir: str = "./pipeline/merged",
):
    """Validate that merged model matches adapter model output."""
    print("\\n[Stage 3] Validating merge correctness...")
    
    tokenizer = AutoTokenizer.from_pretrained(merged_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load adapter model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    adapter_model = PeftModel.from_pretrained(base_model, adapter_dir)
    adapter_model.eval()
    
    # Load merged model
    merged_model = AutoModelForCausalLM.from_pretrained(merged_dir)
    merged_model.eval()
    
    # Compare outputs on test prompts
    test_prompts = [
        "Quote: The only way to",
        "Quote: In the beginning",
        "Quote: Life is what happens",
    ]
    
    all_match = True
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            adapter_logits = adapter_model(**inputs).logits
            merged_logits = merged_model(**inputs).logits
        
        max_diff = (adapter_logits - merged_logits).abs().max().item()
        match = max_diff < 1e-3
        all_match &= match
        
        print(f"  Prompt: '{prompt[:40]}...'")
        print(f"    Max logit difference: {max_diff:.2e} "
              f"{'✓' if match else '✗'}")
    
    print(f"\\n[Stage 3] Validation {'PASSED ✓' if all_match else 'FAILED ✗'}")
    return all_match


# ═══════════════════════════════════════════════════════════════════
# STAGE 4: OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════

def stage_4_optimize(
    merged_dir: str = "./pipeline/merged",
    output_dir: str = "./pipeline/optimized",
):
    """Apply post-merge optimizations."""
    print("\\n[Stage 4] Optimizing merged model...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to FP16 for deployment
    model = AutoModelForCausalLM.from_pretrained(
        merged_dir,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(merged_dir)
    
    # Save in FP16
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(output_dir)
    
    # Report size reduction
    def dir_size(path):
        total = 0
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
        return total / (1024 ** 2)  # MB
    
    fp32_size = dir_size(merged_dir)
    fp16_size = dir_size(output_dir)
    
    print(f"  FP32 size: {fp32_size:.1f} MB")
    print(f"  FP16 size: {fp16_size:.1f} MB")
    print(f"  Reduction: {(1 - fp16_size/fp32_size)*100:.1f}%")
    print(f"[Stage 4] Optimized model saved to {output_dir}")
    
    return output_dir


# ═══════════════════════════════════════════════════════════════════
# STAGE 5: TEST
# ═══════════════════════════════════════════════════════════════════

def stage_5_test(
    model_dir: str = "./pipeline/optimized",
):
    """Final test of the deployed model."""
    print("\\n[Stage 5] Testing final model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    test_prompts = [
        "Quote: The best way to predict the future",
        "Quote: Life is",
        "Quote: Everyone thinks of changing the world",
    ]
    
    print("\\n  ── Generation Samples ──")
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\\n  Input:  {prompt}")
        print(f"  Output: {text}")
    
    print("\\n[Stage 5] Testing complete!")


# ═══════════════════════════════════════════════════════════════════
# RUN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def run_full_pipeline():
    """Execute the complete merge & deploy pipeline."""
    print("=" * 70)
    print("  FULL LoRA MERGE & DEPLOYMENT PIPELINE")
    print("=" * 70)
    
    base_model = "distilgpt2"
    
    # Stage 1: Train
    adapter_dir = stage_1_train(base_model)
    
    # Stage 2: Merge
    merged_dir = stage_2_merge(base_model, adapter_dir)
    
    # Stage 3: Validate
    is_valid = stage_3_validate(base_model, adapter_dir, merged_dir)
    if not is_valid:
        print("\\n⚠ Validation failed! Check merge precision.")
    
    # Stage 4: Optimize
    optimized_dir = stage_4_optimize(merged_dir)
    
    # Stage 5: Test
    stage_5_test(optimized_dir)
    
    print("\\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\\n  Adapter:   {adapter_dir}")
    print(f"  Merged:    {merged_dir}")
    print(f"  Optimized: {optimized_dir}")
    print(f"  \\nThe model in '{optimized_dir}' is ready for deployment!")


if __name__ == "__main__":
    run_full_pipeline()
'''
    print(code)
    return code


# ============================================================================
# SECTION 8: MERGE SAFETY & TROUBLESHOOTING
# ============================================================================

class MergeTroubleshooting:
    """
    Common issues and solutions when merging and deploying LoRA models.
    """
    
    @staticmethod
    def print_troubleshooting_guide():
        """Print a comprehensive troubleshooting guide."""
        print("\n" + "=" * 70)
        print("LoRA MERGE & DEPLOY TROUBLESHOOTING GUIDE")
        print("=" * 70)
        
        guide = """
┌─────────────────────────────────────────────────────────────────┐
│ COMMON ISSUES & SOLUTIONS                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 1. NaN values after merging                                     │
│    Cause: FP16 overflow during merge computation                │
│    Fix:   Merge in FP32, then convert to FP16                   │
│           Use safe_merge=True to detect NaN                     │
│                                                                 │
│ 2. Output quality degrades after merging                        │
│    Cause: Precision loss during merge                           │
│    Fix:   Compare logits before/after merge                     │
│           If max diff > 1e-3, check dtype                       │
│           Ensure base model version matches training            │
│                                                                 │
│ 3. "Size mismatch" error when loading adapter                   │
│    Cause: Base model version changed since training             │
│    Fix:   Pin exact base model revision during training:        │
│           model = AutoModel.from_pretrained(name, revision="v1")│
│                                                                 │
│ 4. Memory error during merging                                  │
│    Cause: Model too large for available RAM                     │
│    Fix:   Use device_map="cpu" for merging                      │
│           Merge on machine with more RAM                        │
│           Use streaming/sharded loading                         │
│                                                                 │
│ 5. GGUF conversion fails                                        │
│    Cause: Model architecture not supported by llama.cpp         │
│    Fix:   Check llama.cpp supported architectures               │
│           Use latest version of llama.cpp                       │
│           Try the HF-to-GGUF converter script                   │
│                                                                 │
│ 6. Merged model much larger than expected                       │
│    Cause: Saved in FP32 instead of FP16                         │
│    Fix:   model.half() before saving, or                        │
│           save_pretrained with torch_dtype=torch.float16        │
│                                                                 │
│ 7. Different outputs between adapter and merged model           │
│    Cause: Scaling factor mismatch or partial merge              │
│    Fix:   Verify alpha/r ratio matches training config          │
│           Ensure ALL LoRA layers were merged                    │
│           Check model.merge_and_unload(safe_merge=True)         │
│                                                                 │
│ 8. vLLM/TGI fail to load merged model                          │
│    Cause: Missing config files or incompatible format           │
│    Fix:   Ensure config.json is saved with the model            │
│           Check tokenizer files are present                     │
│           Verify model architecture is supported                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

BEST PRACTICES CHECKLIST:
  □ Train and save adapter checkpoints
  □ Load base model in FP32 for merging
  □ Use safe_merge=True
  □ Validate merged model against adapter model
  □ Convert to FP16/BF16 for deployment
  □ Run benchmark before deploying
  □ Save tokenizer with merged model
  □ Pin base model revision
  □ Test with real prompts, not just perplexity
  □ Keep adapter weights for rollback
"""
        print(guide)


# ============================================================================
# MAIN — Run all demonstrations
# ============================================================================

def main():
    """Run all merge & deployment demonstrations."""
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║            LoRA MERGE & DEPLOYMENT STRATEGIES                  ║")
    print("║                                                                ║")
    print("║  From trained adapter to production-ready model                ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    # Section 1: Merging fundamentals
    engine = LoRAMergingEngine()
    engine.demonstrate_merge_math()
    engine.demonstrate_precision_effects()
    
    # Section 2: Advanced merging strategies
    strategies = AdvancedMergingStrategies()
    strategies.standard_merge_with_peft()
    strategies.weighted_adapter_merge()
    strategies.demonstrate_task_arithmetic()
    strategies.demonstrate_ties_merging()
    
    # Section 3: Multi-adapter management
    manager = MultiAdapterManager()
    manager.dynamic_adapter_loading()
    manager.adapter_serving_architecture()
    
    # Section 4: Export formats
    exporter = ModelExporter()
    exporter.export_safetensors()
    exporter.export_gguf()
    exporter.export_onnx()
    
    # Section 5: Deployment patterns
    deploy = DeploymentPatterns()
    deploy.vllm_deployment()
    deploy.text_generation_inference()
    deploy.fastapi_deployment()
    
    # Section 6: Post-merge optimization
    optimizer = PostMergeOptimization()
    optimizer.quantize_after_merge()
    optimizer.benchmark_merged_model()
    
    # Section 7: Complete pipeline
    complete_merge_deploy_pipeline()
    
    # Section 8: Troubleshooting
    troubleshooter = MergeTroubleshooting()
    troubleshooter.print_troubleshooting_guide()
    
    print("\n" + "=" * 70)
    print("  MODULE COMPLETE")
    print("=" * 70)
    print("""
    Covered in this module:
    ✓ Weight merging math and precision effects
    ✓ Standard, weighted, task arithmetic, and TIES merging
    ✓ Multi-adapter management and caching
    ✓ Export: SafeTensors, GGUF, ONNX
    ✓ Deployment: vLLM, TGI, FastAPI
    ✓ Post-merge quantization and benchmarking
    ✓ Complete end-to-end pipeline
    ✓ Troubleshooting guide
    """)


if __name__ == "__main__":
    main()
