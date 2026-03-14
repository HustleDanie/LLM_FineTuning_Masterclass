"""
IA³ Comparison — vs LoRA, Adapters, Prompt Methods
====================================================

Comprehensive comparison of IA³ with other PEFT methods:

1. IA³ vs LoRA — Detailed Head-to-Head
   - Parameters, expressiveness, inference cost
   - When each method wins

2. IA³ vs All PEFT Methods
   - Soft prompts, adapters, LoRA, IA³
   - Multi-dimensional comparison

3. Practical Training Comparison
   - Train multiple methods on same data
   - Compare convergence and performance

4. Decision Framework
   - When to use IA³
   - Complete selection guide

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict


# ============================================================================
# SECTION 1: IA³ vs LORA HEAD-TO-HEAD
# ============================================================================

def compare_ia3_vs_lora():
    """Detailed comparison of IA³ and LoRA."""
    print("=" * 65)
    print("  SECTION 1: IA³ vs LoRA HEAD-TO-HEAD")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, IA3Config, LoraConfig, TaskType
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # ─── IA³ ───
    ia3_base = AutoModelForCausalLM.from_pretrained(model_name)
    ia3_base.config.pad_token_id = tokenizer.pad_token_id
    ia3_model = get_peft_model(ia3_base, IA3Config(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn", "mlp.c_fc"],
        feedforward_modules=["mlp.c_fc"],
    ))
    
    # ─── LoRA (small) ───
    lora_s_base = AutoModelForCausalLM.from_pretrained(model_name)
    lora_s_base.config.pad_token_id = tokenizer.pad_token_id
    lora_s_model = get_peft_model(lora_s_base, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4, lora_alpha=8, lora_dropout=0.05,
        target_modules=["c_attn"],
    ))
    
    # ─── LoRA (standard) ───
    lora_base = AutoModelForCausalLM.from_pretrained(model_name)
    lora_base.config.pad_token_id = tokenizer.pad_token_id
    lora_model = get_peft_model(lora_base, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["c_attn"],
    ))
    
    models = {
        "IA³": ia3_model,
        "LoRA (r=4)": lora_s_model,
        "LoRA (r=8)": lora_model,
    }
    
    text = "The key differences between parameter-efficient methods are"
    inputs = tokenizer(text, return_tensors="pt")
    
    results = {}
    for name, model in models.items():
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Speed benchmark
        with torch.no_grad():
            model(**inputs, labels=inputs["input_ids"])  # warmup
        
        start = time.time()
        for _ in range(50):
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
        fwd_time = (time.time() - start) / 50
        
        # Training step speed
        model.train()
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3,
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
            "loss": out.loss.item(),
        }
        
        del opt
    
    # Display
    print(f"\n  {'Method':<16} {'Params':>10} {'%':>8} {'Fwd':>7} {'Train':>7}")
    print(f"  {'─'*16}─{'─'*10}─{'─'*8}─{'─'*7}─{'─'*7}")
    
    for name, r in results.items():
        print(f"  {name:<16} {r['trainable']:>10,} {r['pct']:>7.4f}% "
              f"{r['fwd_ms']:>5.1f}ms {r['train_ms']:>5.1f}ms")
    
    detailed = """
  ═══ IA³ vs LoRA: Detailed Analysis ═══
  
  ┌──────────────────┬────────────────────┬────────────────────┐
  │ Dimension        │ IA³                │ LoRA               │
  ├──────────────────┼────────────────────┼────────────────────┤
  │ Operation        │ Multiplicative     │ Additive           │
  │                  │ y = l ⊙ Wx         │ y = Wx + BAx       │
  ├──────────────────┼────────────────────┼────────────────────┤
  │ Params per layer │ O(d)               │ O(r × d)           │
  │ (typical)        │ ~5K                │ ~50K-150K          │
  ├──────────────────┼────────────────────┼────────────────────┤
  │ Expressiveness   │ Diagonal only      │ Full low-rank      │
  │                  │ (can't mix dims)   │ (can mix dims)     │
  ├──────────────────┼────────────────────┼────────────────────┤
  │ Merging          │ ✓ Row/col scaling  │ ✓ Add ΔW to W      │
  ├──────────────────┼────────────────────┼────────────────────┤
  │ Init             │ Ones (identity)    │ Gaussian + Zero    │
  ├──────────────────┼────────────────────┼────────────────────┤
  │ Few-shot (4-64)  │ ★★★★ Champion     │ ★★★ Good           │
  │ Medium data      │ ★★★ Good          │ ★★★★ Excellent     │
  │ Large data       │ ★★ Fair           │ ★★★★ Excellent     │
  ├──────────────────┼────────────────────┼────────────────────┤
  │ Learning rate    │ 1e-2 to 5e-2       │ 1e-4 to 3e-4       │
  │ Weight decay     │ 0.0                │ 0.01               │
  ├──────────────────┼────────────────────┼────────────────────┤
  │ Multi-task cost  │ ~50KB per task      │ ~500KB per task     │
  │ Storage          │ Extremely cheap    │ Cheap              │
  └──────────────────┴────────────────────┴────────────────────┘
"""
    print(detailed)
    
    for model in models.values():
        del model


# ============================================================================
# SECTION 2: ALL PEFT METHODS COMPARISON
# ============================================================================

def compare_all_methods():
    """Compare IA³ against all major PEFT methods."""
    print("\n\n" + "=" * 65)
    print("  SECTION 2: IA³ vs ALL PEFT METHODS")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import (
        get_peft_model,
        IA3Config, LoraConfig,
        PromptTuningConfig, PromptTuningInit,
        PrefixTuningConfig,
        PromptEncoderConfig, PromptEncoderReparameterizationType,
        TaskType,
    )
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    configs = {
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
        "P-Tuning v1": PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,
            encoder_reparameterization_type=PromptEncoderReparameterizationType.LSTM,
            encoder_hidden_size=256,
        ),
        "Prefix Tuning": PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,
            prefix_projection=False,
        ),
    }
    
    text = "Comparing all parameter-efficient fine-tuning methods"
    results = {}
    
    for name, config in configs.items():
        base = AutoModelForCausalLM.from_pretrained(model_name)
        base.config.pad_token_id = tokenizer.pad_token_id
        model = get_peft_model(base, config)
        
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            model(**inputs, labels=inputs["input_ids"])
        
        start = time.time()
        for _ in range(30):
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
        elapsed = (time.time() - start) / 30
        
        results[name] = {
            "trainable": trainable,
            "pct": trainable / total * 100,
            "time_ms": elapsed * 1000,
            "loss": out.loss.item(),
        }
        
        del model, base
    
    print(f"\n  {'Method':<18} {'Trainable':>12} {'%':>8} {'Time':>8} {'Merges?':>8}")
    print(f"  {'─'*18}─{'─'*12}─{'─'*8}─{'─'*8}─{'─'*8}")
    
    merge_info = {
        "IA³": "Yes",
        "LoRA (r=8)": "Yes",
        "Prompt Tuning": "No",
        "P-Tuning v1": "No",
        "Prefix Tuning": "No",
    }
    
    for name, r in results.items():
        print(f"  {name:<18} {r['trainable']:>12,} {r['pct']:>7.3f}% "
              f"{r['time_ms']:>6.1f}ms {merge_info[name]:>8}")
    
    print(f"""
  ═══ Multi-Dimensional Comparison ═══
  
  Parameter Efficiency (fewer = better):
    IA³ >>>>>> Prompt Tuning > LoRA > Prefix > P-Tuning v1
    
  Expressiveness (what can it learn):
    Full FT >> LoRA > Prefix/P-Tuning v2 > P-Tuning v1 > IA³ > Prompt Tuning
    
  Few-Shot Performance:
    IA³ > LoRA > P-Tuning v1 > Prefix > Prompt Tuning
    
  Large Dataset Performance:
    LoRA > Full FT > Prefix/P-Tuning v2 > IA³ > Prompt Tuning
    
  Inference Overhead:
    IA³ = LoRA = 0 (after merging)
    Prompt methods: extra tokens in context
    Adapters: extra layers in forward pass
    
  Storage per Task:
    IA³: ~50-200KB
    LoRA: ~500KB-5MB
    Prompt Tuning: ~10-100KB
    Prefix: ~200KB-2MB
    Full FT: model-sized (GBs!)
""")


# ============================================================================
# SECTION 3: PRACTICAL TRAINING COMPARISON
# ============================================================================

def practical_comparison():
    """Train IA³, LoRA, and Prefix Tuning on the same data."""
    print("\n\n" + "=" * 65)
    print("  SECTION 3: PRACTICAL TRAINING COMPARISON")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import (
        get_peft_model,
        IA3Config, LoraConfig, PrefixTuningConfig,
        TaskType,
    )
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create training data
    import random
    random.seed(42)
    
    texts = [
        "The weather is beautiful today.",
        "Machine learning is transforming industries.",
        "Natural language processing enables AI communication.",
        "Deep neural networks learn hierarchical features.",
        "Parameter-efficient methods save computational resources.",
    ] * 10
    
    encodings = tokenizer(
        texts, padding="max_length", truncation=True,
        max_length=32, return_tensors="pt",
    )
    labels = encodings["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    configs = {
        "IA³": (
            IA3Config(
                task_type=TaskType.CAUSAL_LM,
                target_modules=["c_attn", "mlp.c_fc"],
                feedforward_modules=["mlp.c_fc"],
            ),
            1e-2,   # LR
        ),
        "LoRA (r=8)": (
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8, lora_alpha=16, lora_dropout=0.05,
                target_modules=["c_attn"],
            ),
            3e-4,
        ),
        "Prefix Tuning": (
            PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=20,
                prefix_projection=False,
            ),
            3e-3,
        ),
    }
    
    all_losses = {}
    
    for name, (config, lr) in configs.items():
        print(f"\n  Training: {name} (lr={lr})")
        
        base = AutoModelForCausalLM.from_pretrained(model_name)
        base.config.pad_token_id = tokenizer.pad_token_id
        model = get_peft_model(base, config)
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Trainable: {trainable:,}")
        
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=0.0 if "IA³" in name else 0.01,
        )
        
        losses = []
        model.train()
        
        for epoch in range(8):
            total_loss = 0
            for i in range(0, len(texts), 10):
                batch_ids = encodings["input_ids"][i:i+10]
                batch_mask = encodings["attention_mask"][i:i+10]
                batch_labels = labels[i:i+10]
                
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
            
            avg = total_loss / (len(texts) // 10)
            losses.append(avg)
            if (epoch + 1) % 2 == 0:
                print(f"    Epoch {epoch+1}: loss = {avg:.4f}")
        
        all_losses[name] = losses
        del model, base, optimizer
    
    # Summary
    print(f"\n  ── Convergence Summary ──")
    print(f"  {'Method':<18} {'Start':>8} {'End':>8} {'Reduction':>10} {'Epochs':>8}")
    print(f"  {'─'*18}─{'─'*8}─{'─'*8}─{'─'*10}─{'─'*8}")
    
    for name, losses in all_losses.items():
        print(f"  {name:<18} {losses[0]:>8.4f} {losses[-1]:>8.4f} "
              f"{losses[0]-losses[-1]:>10.4f} {len(losses):>8}")
    
    print(f"\n  Note: IA³ uses 10-100× fewer parameters than LoRA")
    print(f"  but achieves comparable loss reduction on this small dataset.")


# ============================================================================
# SECTION 4: DECISION FRAMEWORK
# ============================================================================

def decision_framework():
    """When to use IA³ vs other methods."""
    print("\n\n" + "=" * 65)
    print("  SECTION 4: DECISION FRAMEWORK")
    print("=" * 65)
    
    print("""
  ═══════════════════════════════════════════════════════════════
   WHEN TO USE IA³ — DECISION TREE
  ═══════════════════════════════════════════════════════════════
  
  Start: What's your scenario?
  │
  ├─► Few-shot learning (4-64 examples)?
  │   └─► IA³ (champion for few-shot!)
  │
  ├─► Need absolute minimal parameters?
  │   └─► IA³ (10-100× fewer than LoRA)
  │
  ├─► Many tasks, one base model?
  │   └─► IA³ (tiny vectors per task, ~50KB each)
  │
  ├─► Need zero inference overhead?
  │   ├─► IA³ (merge vectors into weights)
  │   └─► LoRA (merge ΔW into weights)
  │
  ├─► Large dataset, complex task?
  │   └─► LoRA (more expressive, better at scale)
  │
  ├─► Text generation (chat, creative)?
  │   └─► LoRA (better generation quality)
  │
  ├─► Classification / NLU?
  │   └─► IA³ or LoRA (both work well)
  │
  └─► Unsure?
      └─► Start with IA³ (fastest to try), switch to LoRA if needed
  
  
  ═══════════════════════════════════════════════════════════════
   IA³ SWEET SPOTS
  ═══════════════════════════════════════════════════════════════
  
  ★★★★★ (Perfect fit):
    - Few-shot classification (4-64 examples)
    - Multi-task deployment (100+ tasks, one model)
    - Quick prototyping and experimentation
    - Resource-constrained training
    
  ★★★★ (Great fit):
    - Standard classification with medium data
    - NER and extraction tasks
    - Knowledge adaptation across domains
    
  ★★★ (Decent fit):
    - Full dataset fine-tuning
    - Simple generation tasks
    
  ★★ (Consider alternatives):
    - Complex generation (chat, creative writing)
    - Tasks requiring new feature combinations
    - Very long sequence tasks
    
  ★ (Use something else):
    - When you need maximum performance at any cost
    - Drastic model behavior changes
    
    
  ═══════════════════════════════════════════════════════════════
   COMPLETE PEFT RANKING BY SCENARIO
  ═══════════════════════════════════════════════════════════════
  
  Scenario                 │ Best → Worst
  ─────────────────────────┼──────────────────────────────
  Few-shot (< 64 ex.)      │ IA³ > LoRA > P-Tuning > PT
  Classification            │ LoRA ≈ IA³ > P-Tuning v2
  NER / Extraction          │ P-Tuning v2 > LoRA > IA³
  Generation (chat)         │ LoRA >> IA³ > Prefix
  Multi-task (100+ tasks)   │ IA³ > PT > Prefix > LoRA
  Fastest training          │ IA³ > PT > LoRA > Full FT
  Max performance           │ Full FT > LoRA > P-Tuning v2
  Memory constrained        │ IA³ > PT > LoRA
  ─────────────────────────┴──────────────────────────────
  
  PT = Prompt Tuning
  
  
  ═══════════════════════════════════════════════════════════════
   THE IA³ PHILOSOPHY
  ═══════════════════════════════════════════════════════════════
  
  "The pretrained model already knows everything it needs.
   Task adaptation is just about knowing which features 
   to amplify and which to suppress."
  
  This is IA³'s core assumption. When it holds:
    → IA³ is incredibly efficient
    → Fewer parameters = faster, cheaper, less overfitting
    
  When it doesn't hold (truly new knowledge needed):
    → LoRA or full fine-tuning is better
    → More parameters = more expressiveness
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all comparisons."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     IA³ COMPARISON — vs LoRA, ADAPTERS, PROMPT METHODS       ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: IA³ vs LoRA
    compare_ia3_vs_lora()
    
    # Section 2: All methods
    compare_all_methods()
    
    # Section 3: Practical training
    practical_comparison()
    
    # Section 4: Decision framework
    decision_framework()
    
    print("\n" + "=" * 65)
    print("  MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered:
    ✓ IA³ vs LoRA detailed head-to-head
    ✓ All PEFT methods compared (params, speed, merging)
    ✓ Practical training comparison on same data
    ✓ Decision framework with scenario rankings
    ✓ IA³ philosophy and sweet spots
    """)


if __name__ == "__main__":
    main()
