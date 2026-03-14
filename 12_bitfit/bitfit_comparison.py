"""
BitFit Comparison — vs LoRA, IA³, and Other Methods
====================================================

Comprehensive comparison of BitFit with other PEFT approaches:

1. BitFit vs LoRA vs IA³ Head-to-Head
   - Parameter counts, speed, convergence
   - Same data, same model

2. All PEFT Methods Comparison
   - BitFit in the full PEFT landscape
   - Multi-dimensional analysis

3. Practical Training Experiment
   - Train all methods on identical setup
   - Measure loss convergence and efficiency

4. Decision Framework
   - When to use BitFit
   - Complete method selection guide

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict


# ============================================================================
# SECTION 1: BITFIT vs LORA vs IA³ HEAD-TO-HEAD
# ============================================================================

def head_to_head():
    """Direct comparison of BitFit, LoRA, and IA³."""
    print("=" * 65)
    print("  SECTION 1: BitFit vs LoRA vs IA³ HEAD-TO-HEAD")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, IA3Config, TaskType
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    text = "Comparing parameter-efficient fine-tuning methods for language"
    inputs = tokenizer(text, return_tensors="pt")
    
    configs = {}
    
    # ─── BitFit (manual) ───
    bitfit_model = AutoModelForCausalLM.from_pretrained(model_name)
    bitfit_model.config.pad_token_id = tokenizer.pad_token_id
    for param in bitfit_model.parameters():
        param.requires_grad = False
    for name, param in bitfit_model.named_parameters():
        if "bias" in name:
            param.requires_grad = True
    configs["BitFit"] = (bitfit_model, 1e-2)
    
    # ─── IA³ ───
    ia3_base = AutoModelForCausalLM.from_pretrained(model_name)
    ia3_base.config.pad_token_id = tokenizer.pad_token_id
    ia3_model = get_peft_model(ia3_base, IA3Config(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn", "mlp.c_fc"],
        feedforward_modules=["mlp.c_fc"],
    ))
    configs["IA³"] = (ia3_model, 1e-2)
    
    # ─── LoRA (r=4) ───
    lora_s_base = AutoModelForCausalLM.from_pretrained(model_name)
    lora_s_base.config.pad_token_id = tokenizer.pad_token_id
    lora_s_model = get_peft_model(lora_s_base, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4, lora_alpha=8,
        target_modules=["c_attn"],
    ))
    configs["LoRA (r=4)"] = (lora_s_model, 3e-4)
    
    # ─── LoRA (r=8) ───
    lora_base = AutoModelForCausalLM.from_pretrained(model_name)
    lora_base.config.pad_token_id = tokenizer.pad_token_id
    lora_model = get_peft_model(lora_base, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16,
        target_modules=["c_attn"],
    ))
    configs["LoRA (r=8)"] = (lora_model, 3e-4)
    
    results = {}
    
    for name, (model, lr) in configs.items():
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Forward speed
        with torch.no_grad():
            model(**inputs, labels=inputs["input_ids"])  # warmup
        
        start = time.time()
        for _ in range(50):
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
        fwd_time = (time.time() - start) / 50
        
        # Training speed
        model.train()
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
        )
        
        start = time.time()
        for _ in range(10):
            out = model(**inputs, labels=inputs["input_ids"])
            out.loss.backward()
            opt.step()
            opt.zero_grad()
        train_time = (time.time() - start) / 10
        
        results[name] = {
            "trainable": trainable,
            "pct": trainable / total * 100,
            "fwd_ms": fwd_time * 1000,
            "train_ms": train_time * 1000,
        }
        
        del opt
    
    print(f"\n  {'Method':<16} {'Params':>10} {'%':>8} {'Fwd':>8} {'Train':>8}")
    print(f"  {'─'*16}─{'─'*10}─{'─'*8}─{'─'*8}─{'─'*8}")
    
    for name, r in results.items():
        print(f"  {name:<16} {r['trainable']:>10,} {r['pct']:>7.4f}% "
              f"{r['fwd_ms']:>6.1f}ms {r['train_ms']:>6.1f}ms")
    
    detailed = """
  ═══ Head-to-Head Analysis ═══
  
  ┌──────────────────┬────────────┬────────────┬────────────┐
  │ Dimension        │ BitFit     │ IA³        │ LoRA       │
  ├──────────────────┼────────────┼────────────┼────────────┤
  │ What trains      │ Bias terms │ Rescaling  │ Low-rank   │
  │                  │ (b in Wx+b)│ vectors    │ matrices   │
  ├──────────────────┼────────────┼────────────┼────────────┤
  │ Operation        │ Threshold  │ Multipli-  │ Additive   │
  │                  │ shift      │ cative     │ projection │
  ├──────────────────┼────────────┼────────────┼────────────┤
  │ New modules?     │ None       │ None       │ None       │
  ├──────────────────┼────────────┼────────────┼────────────┤
  │ Inference cost   │ Zero       │ Zero*      │ Zero*      │
  │                  │ (inherent) │ (after     │ (after     │
  │                  │            │  merge)    │  merge)    │
  ├──────────────────┼────────────┼────────────┼────────────┤
  │ Implementation   │ 3 lines!   │ ~10 lines  │ ~20 lines  │
  │ complexity       │            │            │            │
  ├──────────────────┼────────────┼────────────┼────────────┤
  │ Requires model   │ Must have  │ Works with │ Works with │
  │ compatibility    │ bias terms!│ any model  │ any model  │
  └──────────────────┴────────────┴────────────┴────────────┘
  
  *After merging into base weights
"""
    print(detailed)
    
    for name, (model, _) in configs.items():
        del model


# ============================================================================
# SECTION 2: ALL PEFT METHODS COMPARISON
# ============================================================================

def all_methods_comparison():
    """Compare BitFit against all major PEFT methods."""
    print("\n\n" + "=" * 65)
    print("  SECTION 2: ALL PEFT METHODS COMPARISON")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import (
        get_peft_model,
        LoraConfig, IA3Config,
        PromptTuningConfig, PromptTuningInit,
        PrefixTuningConfig,
        TaskType,
    )
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    text = "Comparing all fine-tuning methods for large language models"
    
    results = {}
    
    # ─── BitFit ───
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "bias" in name:
            param.requires_grad = True
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    
    results["BitFit"] = {"trainable": trainable, "pct": trainable/total*100, "merges": "N/A"}
    del model
    
    # ─── PEFT methods ───
    peft_configs = {
        "IA³": IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["c_attn", "mlp.c_fc"],
            feedforward_modules=["mlp.c_fc"],
        ),
        "LoRA (r=8)": LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8, lora_alpha=16, target_modules=["c_attn"],
        ),
        "Prompt Tuning": PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,
            prompt_tuning_init=PromptTuningInit.RANDOM,
        ),
        "Prefix Tuning": PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,
            prefix_projection=False,
        ),
    }
    
    merge_info = {
        "IA³": "Yes",
        "LoRA (r=8)": "Yes",
        "Prompt Tuning": "No",
        "Prefix Tuning": "No",
    }
    
    for name, config in peft_configs.items():
        base = AutoModelForCausalLM.from_pretrained(model_name)
        base.config.pad_token_id = tokenizer.pad_token_id
        model = get_peft_model(base, config)
        
        t = sum(p.numel() for p in model.parameters())
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        results[name] = {"trainable": tr, "pct": tr/t*100, "merges": merge_info[name]}
        del model, base
    
    print(f"\n  {'Method':<18} {'Trainable':>12} {'%':>8} {'Merges?':>8} {'Arch Δ':>8}")
    print(f"  {'─'*18}─{'─'*12}─{'─'*8}─{'─'*8}─{'─'*8}")
    
    arch_changes = {
        "BitFit": "None",
        "IA³": "None",
        "LoRA (r=8)": "None",
        "Prompt Tuning": "Tokens",
        "Prefix Tuning": "Tokens",
    }
    
    for name, r in results.items():
        print(f"  {name:<18} {r['trainable']:>12,} {r['pct']:>7.3f}% "
              f"{r['merges']:>8} {arch_changes[name]:>8}")
    
    print(f"""
  ═══ Comprehensive Ranking ═══
  
  Parameter Efficiency (fewest params):
    IA³ > BitFit > Prompt Tuning > Prefix Tuning > LoRA
    
  Implementation Simplicity:
    BitFit >>>>>> IA³ > Prompt Tuning > LoRA > Prefix Tuning
    BitFit wins MASSIVELY — it's just freeze/unfreeze!
    
  NLU Performance (GLUE/SuperGLUE):
    LoRA > IA³ ≈ BitFit > Prefix Tuning > Prompt Tuning
    
  NLG Performance (generation):
    LoRA >> Prefix > IA³ > BitFit > Prompt Tuning
    
  Few-Shot (< 64 examples):
    IA³ > BitFit > LoRA > Prompt Tuning
    
  Model Compatibility:
    LoRA = IA³ = Prompt Tuning > Prefix >>>>>>> BitFit
    (BitFit requires models WITH bias terms!)
""")


# ============================================================================
# SECTION 3: PRACTICAL TRAINING EXPERIMENT
# ============================================================================

def practical_experiment():
    """Train BitFit, LoRA, and IA³ on the same data."""
    print("\n\n" + "=" * 65)
    print("  SECTION 3: PRACTICAL TRAINING EXPERIMENT")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, IA3Config, TaskType
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Training data
    texts = [
        "The neural network processes input data through multiple layers.",
        "Attention mechanisms allow models to focus on relevant tokens.",
        "Fine-tuning adapts pretrained weights to downstream tasks.",
        "Gradient descent iteratively minimizes the loss function.",
        "Regularization techniques prevent models from overfitting.",
    ] * 10
    
    encodings = tokenizer(
        texts, padding="max_length", truncation=True,
        max_length=32, return_tensors="pt",
    )
    labels = encodings["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    method_configs = {
        "BitFit": {"type": "bitfit", "lr": 1e-2},
        "IA³": {
            "type": "peft",
            "config": IA3Config(
                task_type=TaskType.CAUSAL_LM,
                target_modules=["c_attn", "mlp.c_fc"],
                feedforward_modules=["mlp.c_fc"],
            ),
            "lr": 1e-2,
        },
        "LoRA (r=8)": {
            "type": "peft",
            "config": LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8, lora_alpha=16,
                target_modules=["c_attn"],
            ),
            "lr": 3e-4,
        },
    }
    
    all_losses = {}
    
    for name, cfg in method_configs.items():
        print(f"\n  Training: {name} (lr={cfg['lr']})")
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.config.pad_token_id = tokenizer.pad_token_id
        
        if cfg["type"] == "bitfit":
            for param in model.parameters():
                param.requires_grad = False
            for pname, param in model.named_parameters():
                if "bias" in pname:
                    param.requires_grad = True
        else:
            model = get_peft_model(model, cfg["config"])
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Trainable: {trainable:,}")
        
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["lr"],
            weight_decay=0.0 if name in ["BitFit", "IA³"] else 0.01,
        )
        
        losses = []
        model.train()
        batch_size = 10
        
        for epoch in range(8):
            total_loss = 0
            batches = 0
            
            for i in range(0, len(texts), batch_size):
                batch_ids = encodings["input_ids"][i:i+batch_size]
                batch_mask = encodings["attention_mask"][i:i+batch_size]
                batch_labels = labels[i:i+batch_size]
                
                out = model(
                    input_ids=batch_ids,
                    attention_mask=batch_mask,
                    labels=batch_labels,
                )
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += out.loss.item()
                batches += 1
            
            avg = total_loss / batches
            losses.append(avg)
            if (epoch + 1) % 2 == 0:
                print(f"    Epoch {epoch+1}: loss = {avg:.4f}")
        
        all_losses[name] = losses
        del model, optimizer
    
    # Summary
    print(f"\n  ── Convergence Summary ──")
    print(f"  {'Method':<16} {'Start':>8} {'End':>8} {'Reduction':>10}")
    print(f"  {'─'*16}─{'─'*8}─{'─'*8}─{'─'*10}")
    
    for name, losses in all_losses.items():
        print(f"  {name:<16} {losses[0]:>8.4f} {losses[-1]:>8.4f} "
              f"{losses[0]-losses[-1]:>10.4f}")
    
    print(f"\n  Visualization (loss per epoch):")
    for name, losses in all_losses.items():
        bars = " ".join(f"{l:.2f}" for l in losses)
        print(f"    {name:<16}: {bars}")


# ============================================================================
# SECTION 4: DECISION FRAMEWORK
# ============================================================================

def decision_framework():
    """When to use BitFit vs other methods."""
    print("\n\n" + "=" * 65)
    print("  SECTION 4: DECISION FRAMEWORK")
    print("=" * 65)
    
    print(f"""
  ═══════════════════════════════════════════════════════════════
   WHEN TO USE BITFIT — DECISION TREE
  ═══════════════════════════════════════════════════════════════
  
  Start: What's your situation?
  │
  ├─► Model has bias terms?
  │   ├─► NO → Cannot use BitFit! Use LoRA or IA³
  │   │       (LLaMA, Falcon, etc. often have bias=False)
  │   │
  │   └─► YES → Continue ↓
  │
  ├─► Need absolute simplest implementation?
  │   └─► BitFit! (3 lines of code)
  │
  ├─► Few-shot (4-64 examples)?
  │   └─► BitFit or IA³ (both excellent)
  │
  ├─► NLU task (classification, NER, QA)?
  │   └─► BitFit is competitive (try it first!)
  │
  ├─► NLG task (generation, chat, summarization)?
  │   └─► LoRA (BitFit is weaker for generation)
  │
  ├─► Need to try quickly and iterate?
  │   └─► BitFit → IA³ → LoRA (increasing complexity)
  │
  ├─► Maximum performance needed?
  │   └─► LoRA > Full FT >> BitFit
  │
  └─► Want to combine methods?
      └─► BitFit + LoRA (use PEFT's bias="all")
  
  
  ═══════════════════════════════════════════════════════════════
   BITFIT SWEET SPOTS
  ═══════════════════════════════════════════════════════════════
  
  ★★★★★ (Perfect fit):
    - Quick baseline / prototype
    - Few-shot NLU tasks
    - Extreme simplicity required
    - Many tasks, minimal storage budget
  
  ★★★★ (Great fit):
    - Standard classification (BERT-based)
    - Sentiment analysis, topic detection
    - As a complement to LoRA (bias="all")
  
  ★★★ (Decent):
    - Medium-data NLU with bias-enabled models
    - Knowledge probing experiments
  
  ★★ (Consider alternatives):
    - Text generation tasks
    - Models without bias terms
  
  ★ (Use something else):
    - Maximum performance needed
    - Complex generation (chat, creative)
    - Models without bias terms
  
  
  ═══════════════════════════════════════════════════════════════
   COMPLETE PEFT METHOD SELECTION GUIDE (UPDATED)
  ═══════════════════════════════════════════════════════════════
  
  Scenario                 │ 1st Choice  │ 2nd Choice  │ Avoid
  ─────────────────────────┼─────────────┼─────────────┼──────────
  Quick prototype          │ BitFit      │ IA³         │ Full FT
  Few-shot (< 64 ex.)      │ IA³         │ BitFit      │ Full FT
  Classification            │ LoRA        │ BitFit      │ -
  NER / Extraction          │ LoRA        │ P-Tuning v2 │ BitFit
  Generation (chat)         │ LoRA        │ Prefix      │ BitFit
  Multi-task (100+ tasks)   │ IA³/BitFit  │ Prompt Tune │ Full FT
  Max performance           │ Full FT     │ LoRA        │ BitFit
  Memory constrained        │ BitFit/IA³  │ Prompt Tune │ Full FT
  Bias-free models          │ LoRA        │ IA³         │ BitFit!
  ─────────────────────────┴─────────────┴─────────────┴──────────
  
  
  ═══════════════════════════════════════════════════════════════
   THE BITFIT PHILOSOPHY
  ═══════════════════════════════════════════════════════════════
  
  "A pretrained model already knows all the right features.
   Task adaptation is primarily about adjusting WHICH features
   should activate — and that's controlled by the bias terms."
  
  BitFit's radical simplicity carries a profound insight:
  • The weight matrices W encode WHAT features exist
  • The bias terms b encode WHEN they should fire
  • For a new task, often only the WHEN needs to change
  
  This makes BitFit:
  1. The simplest PEFT method (no code changes!)
  2. A strong baseline (surprisingly competitive)
  3. A perfect starting point (iterate toward complexity)
  
  The progression of methods:
    BitFit → IA³ → LoRA → Full Fine-Tuning
    (simplest)              (most expressive)
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all comparisons."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     BitFit COMPARISON — vs LoRA, IA³, AND OTHER METHODS      ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: Head-to-head
    head_to_head()
    
    # Section 2: All methods
    all_methods_comparison()
    
    # Section 3: Practical experiment
    practical_experiment()
    
    # Section 4: Decision framework
    decision_framework()
    
    print("\n" + "=" * 65)
    print("  MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered:
    ✓ BitFit vs LoRA vs IA³ head-to-head
    ✓ All PEFT methods compared
    ✓ Practical training experiment on same data
    ✓ Complete decision framework and selection guide
    ✓ The BitFit philosophy and progression of methods
    """)


if __name__ == "__main__":
    main()
