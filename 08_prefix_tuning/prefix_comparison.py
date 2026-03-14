"""
Prefix Tuning — Comparison with Other Methods
================================================

Systematic comparison of prefix tuning against other PEFT methods:

1. Method Overview Comparison
   - Architecture differences
   - Parameter placement
   - Modification strategy

2. Quantitative Comparison
   - Parameters, performance, speed
   - Memory footprint
   - Convergence characteristics

3. Qualitative Comparison  
   - Composability, modularity
   - Deployment patterns
   - Serving infrastructure

4. When to Use What
   - Decision framework
   - Task-specific recommendations

5. Hands-On Comparison
   - Train same task with multiple methods
   - Fair comparison setup

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import time
import math
from typing import Dict, Optional


# ============================================================================
# SECTION 1: METHOD OVERVIEW COMPARISON
# ============================================================================

def method_overview():
    """Visual overview of different PEFT methods."""
    print("=" * 65)
    print("  SECTION 1: PEFT METHOD OVERVIEW")
    print("=" * 65)
    
    overview = """
  ═══════════════════════════════════════════════════════════════
  Where each method adds/modifies parameters:
  ═══════════════════════════════════════════════════════════════
  
  TRANSFORMER BLOCK
  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │  Input → [PREFIX adds virtual tokens to K, V]            │
  │             ↓                                            │
  │  ┌─────────────────────────────────────────────────────┐ │
  │  │           Multi-Head Attention                       │ │
  │  │                                                     │ │
  │  │   Q = X·Wq  (+LoRA: X·A·B)                         │ │
  │  │   K = X·Wk  (+LoRA: X·A·B)  [+PREFIX: [Pk; K]]    │ │
  │  │   V = X·Wv  (+LoRA: X·A·B)  [+PREFIX: [Pv; V]]    │ │
  │  │                                                     │ │
  │  └──────────────────────┬──────────────────────────────┘ │
  │                         ↓                                │
  │              [+ADAPTER (Houlsby): bottleneck module]      │
  │                         ↓                                │
  │  ┌─────────────────────────────────────────────────────┐ │
  │  │           Feed-Forward Network                       │ │
  │  │                                                     │ │
  │  │   (+LoRA: modify W_up and/or W_down via A·B)        │ │
  │  │   (+IA3: scale activations via learned vectors)     │ │
  │  │   (+BitFit: only train biases)                      │ │
  │  │                                                     │ │
  │  └──────────────────────┬──────────────────────────────┘ │
  │                         ↓                                │
  │              [+ADAPTER (both): bottleneck module]         │
  │                         ↓                                │
  │                      Output                              │
  └──────────────────────────────────────────────────────────┘
  
  
  Modification Strategy:
  ═══════════════════════════════════════════════════════════════
  
  ┌─────────────────┬────────────────────────────────────────────┐
  │ Method          │ What it modifies                           │
  ├─────────────────┼────────────────────────────────────────────┤
  │ Full FT         │ All model weights                          │
  │ LoRA            │ Adds low-rank matrices to existing weights │
  │ QLoRA           │ LoRA + 4-bit quantized base model          │
  │ Adapters        │ Inserts new bottleneck modules             │
  │ PREFIX TUNING   │ Prepends learned vectors to attention K, V │
  │ Prompt Tuning   │ Prepends learned vectors to input only     │
  │ P-Tuning (v2)   │ Like prefix tuning with different training │
  │ IA³             │ Scales activations via learned vectors     │
  │ BitFit          │ Only trains bias terms                     │
  └─────────────────┴────────────────────────────────────────────┘
"""
    print(overview)


# ============================================================================
# SECTION 2: QUANTITATIVE COMPARISON
# ============================================================================

def quantitative_comparison():
    """Detailed quantitative comparison of methods."""
    print("\n" + "=" * 65)
    print("  SECTION 2: QUANTITATIVE COMPARISON")
    print("=" * 65)
    
    # Model: GPT-2 (124M params), d=768, 12 layers, 12 heads
    base_params = 124_000_000
    d_model = 768
    n_layers = 12
    
    methods = {
        "Full Fine-Tuning": {
            "trainable": base_params,
            "total": base_params,
            "accuracy": 94.2,
            "inference_overhead": "0%",
            "training_speed": "1.0x (baseline)",
            "memory": "~500 MB",
            "merge": "N/A (is base)",
        },
        "LoRA (r=8)": {
            "trainable": n_layers * 2 * 8 * d_model * 2,   # Q, V projections
            "total": base_params + n_layers * 2 * 8 * d_model * 2,
            "accuracy": 93.8,
            "inference_overhead": "0% (after merge)",
            "training_speed": "1.8x faster",
            "memory": "~550 MB",
            "merge": "✓ Yes",
        },
        "LoRA (r=64)": {
            "trainable": n_layers * 2 * 64 * d_model * 2,
            "total": base_params + n_layers * 2 * 64 * d_model * 2,
            "accuracy": 94.0,
            "inference_overhead": "0% (after merge)",
            "training_speed": "1.5x faster",
            "memory": "~600 MB",
            "merge": "✓ Yes",
        },
        "Adapter (Pfeiffer)": {
            "trainable": n_layers * 2 * 64 * d_model,     # Down + up projections
            "total": base_params + n_layers * 2 * 64 * d_model,
            "accuracy": 93.5,
            "inference_overhead": "~5-10%",
            "training_speed": "1.6x faster",
            "memory": "~560 MB",
            "merge": "✗ No",
        },
        "Prefix (L=20)": {
            "trainable": n_layers * 2 * 20 * d_model,
            "total": base_params + n_layers * 2 * 20 * d_model,
            "accuracy": 92.5,
            "inference_overhead": "~2-5%",
            "training_speed": "2.0x faster",
            "memory": "~510 MB",
            "merge": "✗ No",
        },
        "Prefix (L=50)": {
            "trainable": n_layers * 2 * 50 * d_model,
            "total": base_params + n_layers * 2 * 50 * d_model,
            "accuracy": 93.2,
            "inference_overhead": "~5-10%",
            "training_speed": "1.8x faster",
            "memory": "~530 MB",
            "merge": "✗ No",
        },
        "Prompt Tuning (L=20)": {
            "trainable": 20 * d_model,
            "total": base_params + 20 * d_model,
            "accuracy": 89.5,
            "inference_overhead": "~1%",
            "training_speed": "2.5x faster",
            "memory": "~505 MB",
            "merge": "✗ No",
        },
        "IA³": {
            "trainable": n_layers * 3 * d_model,  # K, V, FFN scaling vectors
            "total": base_params + n_layers * 3 * d_model,
            "accuracy": 91.0,
            "inference_overhead": "0% (after merge)",
            "training_speed": "2.2x faster",
            "memory": "~505 MB",
            "merge": "✓ Yes",
        },
        "BitFit": {
            "trainable": base_params * 0.001,  # ~0.1% bias terms
            "total": base_params,
            "accuracy": 89.0,
            "inference_overhead": "0%",
            "training_speed": "2.0x faster",
            "memory": "~500 MB",
            "merge": "N/A (modifies biases)",
        },
    }
    
    # Parameter comparison table
    print(f"\n  {'Method':>22} {'Trainable':>12} {'% Model':>9} {'Accuracy':>10}")
    print(f"  {'─'*22}─{'─'*12}─{'─'*9}─{'─'*10}")
    
    for name, m in methods.items():
        trainable = int(m["trainable"])
        pct = trainable / base_params * 100
        acc = m["accuracy"]
        print(f"  {name:>22}  {trainable:>10,}  {pct:>7.3f}%  {acc:>8.1f}%")
    
    # Efficiency comparison
    print(f"\n  ─── Efficiency Comparison ───")
    print(f"  {'Method':>22} {'Inference OH':>14} {'Merge':>8} {'Speed':>16}")
    print(f"  {'─'*22}─{'─'*14}─{'─'*8}─{'─'*16}")
    
    for name, m in methods.items():
        print(f"  {name:>22}  {m['inference_overhead']:>12}  "
              f"{m['merge']:>6}  {m['training_speed']:>14}")
    
    # Highlight prefix tuning position
    print(f"""
  KEY TAKEAWAYS FOR PREFIX TUNING:
  ─────────────────────────────────────────────────────────────
  • Parameters: Between prompt tuning and adapters/LoRA
  • Quality: Good but slightly below LoRA/adapters
  • Inference: Small overhead (extra K, V positions)  
  • Cannot merge: Unlike LoRA, prefix is always separate
  • Best at: Modularity and task switching
  • LR requirement: Much higher than LoRA (1e-2 vs 1e-4)
""")


# ============================================================================
# SECTION 3: QUALITATIVE COMPARISON
# ============================================================================

def qualitative_comparison():
    """Qualitative comparison of method properties."""
    print("\n" + "=" * 65)
    print("  SECTION 3: QUALITATIVE COMPARISON")
    print("=" * 65)
    
    comparison = """
  ═══════════════════════════════════════════════════════════════
  KEY PROPERTIES COMPARISON
  ═══════════════════════════════════════════════════════════════
  
  ┌──────────────────┬──────┬──────┬────────┬────────┬─────────┐
  │ Property         │ LoRA │Adapt.│ Prefix │Prompt T│  IA³    │
  ├──────────────────┼──────┼──────┼────────┼────────┼─────────┤
  │ Composability    │ Poor │ Good │ Good   │ Good   │ Fair    │
  │ Task switching   │ Fair │ Good │ Excel. │ Excel. │ Fair    │
  │ Merge to base    │ ✓    │ ✗    │ ✗      │ ✗      │ ✓      │
  │ Inference speed  │ ★★★★ │ ★★★  │ ★★★    │ ★★★★   │ ★★★★   │
  │ Training speed   │ ★★★  │ ★★★  │ ★★★★   │ ★★★★★  │ ★★★★   │
  │ Quality          │ ★★★★ │ ★★★★ │ ★★★    │ ★★     │ ★★★    │
  │ Param efficiency │ ★★★  │ ★★   │ ★★★★   │ ★★★★★  │ ★★★★★  │
  │ Stability        │ ★★★★ │ ★★★★ │ ★★★    │ ★★     │ ★★★★   │
  │ Framework support│ ★★★★★│ ★★★  │ ★★★★   │ ★★★★   │ ★★★    │
  │ vLLM/TGI support │ ✓    │ ✗    │ ✓*     │ ✓*     │ ✓      │
  └──────────────────┴──────┴──────┴────────┴────────┴─────────┘
  
  * Prefix/Prompt tuning supported via prompt prepending in some
    serving frameworks, but not as natively as LoRA adapters.
  
  
  ═══ COMPOSABILITY ═══
  ─────────────────────────────────────────────────────────────
  LoRA:     Can stack but must retrain; merging is destructive
  Adapters: AdapterFusion provides elegant composition
  Prefix:   Simply concatenate or interpolate prefixes!
  
  Example: To combine Task A + Task B with prefix tuning:
    P_combined = concat(P_A, P_B)  or
    P_combined = 0.5 * P_A + 0.5 * P_B
  
  No retraining needed! This is prefix tuning's superpower.
  
  
  ═══ TASK SWITCHING ═══
  ─────────────────────────────────────────────────────────────
  LoRA:     Load adapter weights ~50ms (swap modules)
  Adapters: Load adapter weights ~50ms
  Prefix:   Swap prefix vectors ~1ms (just tensor replacement!)
  
  Prefix tuning allows near-instant task switching because
  prefix vectors are tiny tensors, not model modules.
  
  
  ═══ SERVING PATTERNS ═══
  ─────────────────────────────────────────────────────────────
  LoRA in production:
    - vLLM natively supports multiple LoRA adapters
    - Can serve 100s of LoRA adapters sharing one base model
    - No overhead after merge
  
  Prefix in production:
    - Treat prefix as a special prompt (past_key_values)
    - Very easy to implement in custom serving code
    - Slightly increases KV cache size
    - Can batch requests with different prefixes!
  
  Adapter in production:
    - Requires custom serving infrastructure
    - Not supported by vLLM/TGI natively
    - Best for in-house deployments
"""
    print(comparison)


# ============================================================================
# SECTION 4: WHEN TO USE WHAT
# ============================================================================

def decision_framework():
    """Decision framework for choosing PEFT methods."""
    print("\n" + "=" * 65)
    print("  SECTION 4: DECISION FRAMEWORK")
    print("=" * 65)
    
    framework = """
  ═══════════════════════════════════════════════════════════════
                    PEFT METHOD DECISION TREE
  ═══════════════════════════════════════════════════════════════
  
  START: What's your primary concern?
  │
  ├─→ Maximum quality
  │   │
  │   ├─→ Budget allows full FT? → Full Fine-Tuning
  │   └─→ Need efficiency? → LoRA (r=64-128) or Adapters
  │
  ├─→ Minimum parameters
  │   │
  │   ├─→ Need decent quality? → Prefix Tuning (L=20-50)
  │   └─→ Extreme efficiency? → Prompt Tuning or IA³
  │
  ├─→ Fast inference (no overhead)
  │   │
  │   ├─→ Good quality needed? → LoRA (merge after training)
  │   └─→ Very few params needed? → IA³ (merge after training)
  │
  ├─→ Multi-task / task switching
  │   │
  │   ├─→ Many tasks, fast switching? → ★ Prefix Tuning
  │   ├─→ Task composition needed? → Adapters + AdapterFusion
  │   └─→ Each task independently? → LoRA per task
  │
  ├─→ vLLM/TGI serving
  │   │
  │   └─→ LoRA (best supported in all serving frameworks)
  │
  └─→ Minimal code changes
      │
      ├─→ Quick setup? → LoRA via PEFT (3 lines of code)
      └─→ Even simpler? → Prompt Tuning (no model modification)
  
  
  ═══════════════════════════════════════════════════════════════
              USE PREFIX TUNING WHEN:
  ═══════════════════════════════════════════════════════════════
  
  ✓ You need to switch between many tasks rapidly
  ✓ You want to interpolate between task behaviors
  ✓ Parameter count must be very small (< 1% of model)
  ✓ You're working with encoder-decoder models (T5, BART)
    — prefix tuning is particularly effective here!
  ✓ You want a simple, interpretable adaptation mechanism
  ✓ You need to batch requests with different tasks efficiently
  
  ═══════════════════════════════════════════════════════════════
              AVOID PREFIX TUNING WHEN:
  ═══════════════════════════════════════════════════════════════
  
  ✗ Maximum quality is critical (use LoRA or full FT)
  ✗ Context window is very limited (prefix consumes tokens)
  ✗ You need zero inference overhead (use LoRA + merge)
  ✗ Training stability is a concern (prefix needs careful tuning)
  ✗ You want vLLM/TGI native support (use LoRA)
"""
    print(framework)


# ============================================================================
# SECTION 5: HANDS-ON COMPARISON
# ============================================================================

def hands_on_comparison():
    """
    Fair comparison: train the same task with multiple PEFT methods.
    """
    print("\n" + "=" * 65)
    print("  SECTION 5: HANDS-ON COMPARISON")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import (
        get_peft_model,
        LoraConfig,
        PrefixTuningConfig,
        PromptTuningConfig,
        PromptTuningInit,
        TaskType,
    )
    from datasets import load_dataset
    from trl import SFTTrainer
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset (same for all methods)
    print("\n  Preparing dataset...")
    dataset = load_dataset("Abirate/english_quotes", split="train")
    dataset = dataset.map(
        lambda x: {"text": f"Quote: \"{x.get('quote', '')}\" — {x.get('author', 'Unknown')}"},
        remove_columns=dataset.column_names,
    )
    split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"  Train: {len(split['train'])} | Eval: {len(split['test'])}")
    
    # Define methods to compare
    configs = {
        "LoRA (r=8)": LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
        ),
        "Prefix (L=20)": PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,
            prefix_projection=True,
            encoder_hidden_size=256,
        ),
        "Prompt (L=20)": PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,
            prompt_tuning_init=PromptTuningInit.RANDOM,
        ),
    }
    
    # Method-specific learning rates
    learning_rates = {
        "LoRA (r=8)": 2e-4,
        "Prefix (L=20)": 3e-2,
        "Prompt (L=20)": 3e-1,
    }
    
    results = {}
    
    for method_name, peft_config in configs.items():
        print(f"\n  {'─'*55}")
        print(f"  Training: {method_name}")
        print(f"  {'─'*55}")
        
        # Fresh model for each method
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.config.pad_token_id = tokenizer.pad_token_id
        model = get_peft_model(model, peft_config)
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        pct = trainable / total * 100
        
        print(f"  Trainable: {trainable:,} ({pct:.4f}%)")
        
        lr = learning_rates[method_name]
        training_args = TrainingArguments(
            output_dir=f"./comparison_{method_name.replace(' ', '_')}",
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=lr,
            warmup_ratio=0.06,
            lr_scheduler_type="cosine",
            logging_steps=50,
            save_strategy="no",
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
        train_result = trainer.train()
        train_time = time.time() - start_time
        
        # Evaluate
        eval_result = trainer.evaluate()
        
        results[method_name] = {
            "trainable_params": trainable,
            "trainable_pct": pct,
            "train_loss": train_result.training_loss,
            "eval_loss": eval_result["eval_loss"],
            "train_time": train_time,
            "learning_rate": lr,
        }
        
        print(f"  Train loss: {train_result.training_loss:.4f}")
        print(f"  Eval loss:  {eval_result['eval_loss']:.4f}")
        print(f"  Time: {train_time:.1f}s")
        
        # Clean up
        del model, trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Summary table
    print(f"\n\n  {'='*65}")
    print(f"  COMPARISON RESULTS")
    print(f"  {'='*65}")
    
    print(f"\n  {'Method':>16} {'Params':>10} {'%':>7} {'Train':>8} "
          f"{'Eval':>8} {'Time':>8} {'LR':>10}")
    print(f"  {'─'*16}─{'─'*10}─{'─'*7}─{'─'*8}─{'─'*8}─{'─'*8}─{'─'*10}")
    
    for name, r in results.items():
        print(f"  {name:>16}  {r['trainable_params']:>8,}  "
              f"{r['trainable_pct']:>5.3f}%  "
              f"{r['train_loss']:>6.3f}  {r['eval_loss']:>6.3f}  "
              f"{r['train_time']:>5.1f}s  {r['learning_rate']:>8.0e}")
    
    print(f"""
  ANALYSIS:
  ─────────────────────────────────────────────────────────────
  • LoRA typically achieves the lowest loss (most expressive)
  • Prefix tuning uses fewer parameters than LoRA
  • Prompt tuning uses the FEWEST parameters but highest loss
  • Prefix needs 100× higher LR than LoRA
  • All methods train much faster than full fine-tuning
  
  Each method excels in different scenarios — there is no
  single "best" method. Choose based on your requirements!
""")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all comparison demonstrations."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║       PREFIX TUNING vs OTHER PEFT METHODS                    ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: Overview
    method_overview()
    
    # Section 2: Quantitative
    quantitative_comparison()
    
    # Section 3: Qualitative
    qualitative_comparison()
    
    # Section 4: Decision framework
    decision_framework()
    
    # Section 5: Hands-on comparison
    hands_on_comparison()
    
    print("\n" + "=" * 65)
    print("  MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered:
    ✓ Method overview (where each adds parameters)
    ✓ Quantitative comparison (params, accuracy, speed)
    ✓ Qualitative comparison (composability, serving, merging)
    ✓ Decision framework for choosing PEFT methods
    ✓ Hands-on comparison (LoRA vs Prefix vs Prompt)
    """)


if __name__ == "__main__":
    main()
