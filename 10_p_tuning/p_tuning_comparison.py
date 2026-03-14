"""
P-Tuning Comparison — v1 vs v2 vs Other Methods
=================================================

Comprehensive comparison of P-Tuning variants:

1. Head-to-Head: v1 vs v2
   - Architecture, signal propagation, parameter efficiency
   - When to use each

2. P-Tuning vs Prompt Tuning vs Prefix Tuning
   - All soft-prompt methods compared
   - Theoretical and practical differences

3. P-Tuning vs LoRA / Adapters
   - Different PEFT paradigms compared
   - Performance, efficiency, flexibility

4. Decision Framework
   - Which method for which scenario
   - Complete decision tree

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, Any


# ============================================================================
# SECTION 1: V1 vs V2 HEAD-TO-HEAD
# ============================================================================

def compare_v1_v2():
    """Detailed comparison of P-Tuning v1 and v2."""
    print("=" * 65)
    print("  SECTION 1: P-TUNING V1 vs V2 HEAD-TO-HEAD")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import (
        get_peft_model,
        PromptEncoderConfig,
        PromptEncoderReparameterizationType,
        PrefixTuningConfig,
        TaskType,
    )
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # ─── V1 Setup ───
    v1_base = AutoModelForCausalLM.from_pretrained(model_name)
    v1_base.config.pad_token_id = tokenizer.pad_token_id
    
    v1_config = PromptEncoderConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
        encoder_reparameterization_type=PromptEncoderReparameterizationType.LSTM,
        encoder_hidden_size=256,
        encoder_num_layers=2,
    )
    v1_model = get_peft_model(v1_base, v1_config)
    
    # ─── V2 Setup ───
    v2_base = AutoModelForCausalLM.from_pretrained(model_name)
    v2_base.config.pad_token_id = tokenizer.pad_token_id
    
    v2_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
        prefix_projection=False,
    )
    v2_model = get_peft_model(v2_base, v2_config)
    
    # ─── Parameter Comparison ───
    v1_total = sum(p.numel() for p in v1_model.parameters())
    v1_trainable = sum(p.numel() for p in v1_model.parameters() if p.requires_grad)
    v2_total = sum(p.numel() for p in v2_model.parameters())
    v2_trainable = sum(p.numel() for p in v2_model.parameters() if p.requires_grad)
    
    print(f"\n  {'Metric':<35} {'V1 (LSTM)':>15} {'V2 (Deep)':>15}")
    print(f"  {'─'*35}─{'─'*15}─{'─'*15}")
    print(f"  {'Total parameters':<35} {v1_total:>15,} {v2_total:>15,}")
    print(f"  {'Trainable parameters':<35} {v1_trainable:>15,} {v2_trainable:>15,}")
    print(f"  {'Trainable %':<35} {v1_trainable/v1_total*100:>14.3f}% {v2_trainable/v2_total*100:>14.3f}%")
    print(f"  {'Prompt injection depth':<35} {'Input only':>15} {'All layers':>15}")
    print(f"  {'Encoder architecture':<35} {'BiLSTM+MLP':>15} {'None':>15}")
    print(f"  {'Virtual tokens':<35} {'20':>15} {'20':>15}")
    
    # ─── Speed Comparison ───
    text = "The capital of France is Paris and the language spoken there is"
    inputs = tokenizer(text, return_tensors="pt")
    
    # Warmup
    with torch.no_grad():
        v1_model(**inputs, labels=inputs["input_ids"])
        v2_model(**inputs, labels=inputs["input_ids"])
    
    # V1 forward time
    start = time.time()
    for _ in range(50):
        with torch.no_grad():
            v1_model(**inputs, labels=inputs["input_ids"])
    v1_time = (time.time() - start) / 50
    
    # V2 forward time
    start = time.time()
    for _ in range(50):
        with torch.no_grad():
            v2_model(**inputs, labels=inputs["input_ids"])
    v2_time = (time.time() - start) / 50
    
    print(f"\n  {'Forward pass time (avg)':<35} {v1_time*1000:>13.1f}ms {v2_time*1000:>13.1f}ms")
    
    # ─── Gradient Flow ───
    v1_model.train()
    v2_model.train()
    
    v1_out = v1_model(**inputs, labels=inputs["input_ids"])
    v1_out.loss.backward()
    v1_grad_norm = sum(
        p.grad.norm().item() for p in v1_model.parameters()
        if p.requires_grad and p.grad is not None
    )
    
    v2_out = v2_model(**inputs, labels=inputs["input_ids"])
    v2_out.loss.backward()
    v2_grad_norm = sum(
        p.grad.norm().item() for p in v2_model.parameters()
        if p.requires_grad and p.grad is not None
    )
    
    print(f"  {'Total gradient norm':<35} {v1_grad_norm:>15.4f} {v2_grad_norm:>15.4f}")
    
    v1_v2_comparison = """
  ═══ V1 vs V2 Summary ═══
  
  P-Tuning V1:
    ✓ LSTM encoder creates coherent prompts
    ✓ Works well for knowledge probing (cloze tasks)
    ✓ Fewer trainable params (encoder reuses weights)
    ✗ Input-only injection → signal decays in deep models
    ✗ Struggles with hard tasks (NER, QA) at small scales
    ✗ Extra LSTM forward pass adds latency
    
  P-Tuning V2:
    ✓ Deep prompts at every layer → no signal decay
    ✓ Matches full FT across all model sizes
    ✓ Works on ALL tasks including NER, QA
    ✓ Simpler (no encoder needed)
    ✗ More trainable parameters (scales with num_layers)
    ✗ Storage grows with model depth
    
  When to choose V1:
    - Small models + simple classification
    - Knowledge probing (LAMA-style)
    - Need minimal parameter count
    
  When to choose V2:
    - Any model size + any task type
    - NER, QA, or hard sequence labeling
    - Need to match full fine-tuning
"""
    print(v1_v2_comparison)
    
    del v1_model, v2_model


# ============================================================================
# SECTION 2: SOFT PROMPT METHOD COMPARISON
# ============================================================================

def compare_soft_prompt_methods():
    """Compare all soft-prompt-based methods."""
    print("\n\n" + "=" * 65)
    print("  SECTION 2: ALL SOFT PROMPT METHODS COMPARED")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import (
        get_peft_model,
        PromptTuningConfig, PromptTuningInit,
        PromptEncoderConfig, PromptEncoderReparameterizationType,
        PrefixTuningConfig,
        TaskType,
    )
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    configs = {}
    results = {}
    
    # ─── 1. Prompt Tuning ───
    configs["Prompt Tuning"] = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
        prompt_tuning_init=PromptTuningInit.RANDOM,
    )
    
    # ─── 2. P-Tuning v1 (LSTM) ───
    configs["P-Tuning v1"] = PromptEncoderConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
        encoder_reparameterization_type=PromptEncoderReparameterizationType.LSTM,
        encoder_hidden_size=256,
        encoder_num_layers=2,
    )
    
    # ─── 3. Prefix Tuning ───
    configs["Prefix Tuning"] = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
        prefix_projection=True,
        encoder_hidden_size=256,
    )
    
    # ─── 4. P-Tuning v2 (Deep, no projection) ───
    configs["P-Tuning v2"] = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
        prefix_projection=False,
    )
    
    text = "Understanding the differences between soft prompting methods"
    
    for name, config in configs.items():
        base = AutoModelForCausalLM.from_pretrained(model_name)
        base.config.pad_token_id = tokenizer.pad_token_id
        model = get_peft_model(base, config)
        
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        inputs = tokenizer(text, return_tensors="pt")
        
        # Speed test
        with torch.no_grad():
            model(**inputs, labels=inputs["input_ids"])  # warmup
        
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
    
    # Display comparison table
    print(f"\n  {'Method':<20} {'Trainable':>12} {'%':>8} {'Time':>8} {'Init Loss':>10}")
    print(f"  {'─'*20}─{'─'*12}─{'─'*8}─{'─'*8}─{'─'*10}")
    
    for name, r in results.items():
        print(f"  {name:<20} {r['trainable']:>12,} {r['pct']:>7.3f}% "
              f"{r['time_ms']:>6.1f}ms {r['loss']:>10.4f}")
    
    comparison = """
  ═══ Soft Prompt Method Taxonomy ═══
  
  ┌──────────────────────────────────────────────────────────┐
  │               All Soft Prompt Methods                    │
  ├──────────────┬───────────────────────────────────────────┤
  │ Shallow      │ Deep (all layers)                        │
  │ (input only) │                                          │
  ├──────────────┼───────────────────────────────────────────┤
  │ Prompt       │ Prefix Tuning    P-Tuning v2             │
  │ Tuning       │ (with MLP        (direct                 │
  │              │  reparametrization) optimization)         │
  │ P-Tuning v1  │                                          │
  │ (with LSTM   │ Both inject prompts as key-value pairs   │
  │  encoder)    │ at every transformer layer                │
  └──────────────┴───────────────────────────────────────────┘
  
  Key Distinctions:
  
  Prompt Tuning vs P-Tuning v1:
    Both are shallow (input-only), but P-Tuning v1 uses an
    LSTM encoder to generate inter-dependent prompts.
    
  Prefix Tuning vs P-Tuning v2:
    Both are deep (all layers). Prefix uses MLP reparametrization;
    P-Tuning v2 uses direct optimization. P-Tuning v2's 
    contribution is universality (works at all scales/tasks).
    
  Scaling behavior:
    - Prompt Tuning: only matches FT at >10B parameters
    - P-Tuning v1:   good at 300M–3B, struggles on hard tasks
    - Prefix Tuning:  good, but mainly tested on NLG
    - P-Tuning v2:   matches FT at ALL scales (300M–10B+)
"""
    print(comparison)


# ============================================================================
# SECTION 3: P-TUNING vs LORA / ADAPTERS
# ============================================================================

def compare_with_lora():
    """Compare P-Tuning methods with LoRA."""
    print("\n\n" + "=" * 65)
    print("  SECTION 3: P-TUNING vs LoRA")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import (
        get_peft_model,
        PromptEncoderConfig, PromptEncoderReparameterizationType,
        PrefixTuningConfig,
        LoraConfig,
        TaskType,
    )
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    methods = {
        "P-Tuning v1": PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,
            encoder_reparameterization_type=PromptEncoderReparameterizationType.LSTM,
            encoder_hidden_size=256,
        ),
        "P-Tuning v2": PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,
            prefix_projection=False,
        ),
        "LoRA (r=8)": LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["c_attn"],
        ),
        "LoRA (r=16)": LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn"],
        ),
    }
    
    text = "Comparing different parameter-efficient fine-tuning approaches"
    results = {}
    
    for name, config in methods.items():
        base = AutoModelForCausalLM.from_pretrained(model_name)
        base.config.pad_token_id = tokenizer.pad_token_id
        model = get_peft_model(base, config)
        
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        inputs = tokenizer(text, return_tensors="pt")
        
        # Speed
        with torch.no_grad():
            model(**inputs, labels=inputs["input_ids"])
        start = time.time()
        for _ in range(30):
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
        elapsed = (time.time() - start) / 30
        
        # Training step
        model.train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        
        train_out = model(**inputs, labels=inputs["input_ids"])
        initial_loss = train_out.loss.item()
        train_out.loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            post_out = model(**inputs, labels=inputs["input_ids"])
            post_loss = post_out.loss.item()
        
        results[name] = {
            "trainable": trainable,
            "pct": trainable / total * 100,
            "time_ms": elapsed * 1000,
            "initial_loss": initial_loss,
            "after_1_step": post_loss,
            "loss_drop": initial_loss - post_loss,
        }
        
        del model, base, optimizer
    
    print(f"\n  {'Method':<16} {'Params':>10} {'%':>7} {'Time':>7}"
          f" {'Init Loss':>10} {'After 1':>10} {'Drop':>8}")
    print(f"  {'─'*16}─{'─'*10}─{'─'*7}─{'─'*7}─{'─'*10}─{'─'*10}─{'─'*8}")
    
    for name, r in results.items():
        print(f"  {name:<16} {r['trainable']:>10,} {r['pct']:>6.3f}% "
              f"{r['time_ms']:>5.1f}ms {r['initial_loss']:>10.4f} "
              f"{r['after_1_step']:>10.4f} {r['loss_drop']:>8.4f}")
    
    comparison = """
  ═══ P-Tuning vs LoRA Comparison ═══
  
  ┌─────────────────┬───────────────────┬────────────────────┐
  │ Aspect          │ P-Tuning (v1/v2)  │ LoRA               │
  ├─────────────────┼───────────────────┼────────────────────┤
  │ Where changes   │ Input/KV space    │ Weight matrices    │
  │ How it works    │ Adds soft tokens  │ Low-rank updates   │
  │ Merging         │ Not mergeable     │ Fully mergeable    │
  │ Inference cost  │ Slight overhead   │ Zero overhead      │
  │ Multi-task      │ Swap prompt only  │ Swap adapter       │
  │ Memory          │ Very low          │ Low                │
  │ Sequence length │ Reduced by P      │ Unchanged          │
  │ Generation      │ Good              │ Excellent          │
  │ NLU tasks       │ Excellent (v2)    │ Excellent          │
  └─────────────────┴───────────────────┴────────────────────┘
  
  When to prefer P-Tuning:
    - Need absolute minimal parameters
    - Multi-task serving (one model, swap prompts)
    - NLU-focused tasks (classification, NER)
    - Research into prompt-based learning
    
  When to prefer LoRA:
    - Need to merge weights (zero inference overhead)
    - Generation tasks (chat, creative)
    - Maximum compatibility with serving frameworks
    - Wider community support and tooling
    
  When to use P-Tuning v2 specifically:
    - Small models (< 10B) where prompt tuning fails
    - Hard tasks (NER, extractive QA)
    - Need full-FT-level performance with PEFT efficiency
"""
    print(comparison)


# ============================================================================
# SECTION 4: DECISION FRAMEWORK
# ============================================================================

def decision_framework():
    """Complete decision framework for choosing P-Tuning variants."""
    print("\n\n" + "=" * 65)
    print("  SECTION 4: DECISION FRAMEWORK")
    print("=" * 65)
    
    framework = """
  ═══════════════════════════════════════════════════════════════
   WHEN TO USE P-TUNING (DECISION TREE)
  ═══════════════════════════════════════════════════════════════
  
  Start: What's your task?
  │
  ├─► Simple classification, model > 10B
  │   └─► Prompt Tuning (simplest, sufficient at scale)
  │
  ├─► Knowledge probing / cloze tasks
  │   └─► P-Tuning v1 (LSTM + cloze templates)
  │
  ├─► NER / Extractive QA / Hard NLU
  │   └─► P-Tuning v2 (deep prompts, matches full FT)
  │
  ├─► Text generation (chat, creative, summarization)
  │   └─► LoRA (better for generation, mergeable)
  │
  ├─► Multi-task serving (many tasks, one model)
  │   └─► P-Tuning v2 or Prompt Tuning
  │       (swap prompts per task, tiny storage)
  │
  └─► Maximum performance, any cost
      └─► Full Fine-Tuning or LoRA
  
  
  ═══════════════════════════════════════════════════════════════
   MODEL SIZE GUIDE
  ═══════════════════════════════════════════════════════════════
  
  Model < 1B:
  ├─► Simple NLU → P-Tuning v2
  ├─► Hard NLU → P-Tuning v2 or LoRA
  └─► Generation → LoRA
  
  Model 1B–10B:
  ├─► Any NLU → P-Tuning v2 ≈ Full FT
  ├─► Generation → LoRA
  └─► Multi-task → P-Tuning v2
  
  Model > 10B:
  ├─► Simple NLU → Prompt Tuning (sufficient!)
  ├─► Hard NLU → P-Tuning v2 or LoRA
  ├─► Generation → LoRA or QLoRA
  └─► Everything → Full FT overkill, use PEFT
  
  
  ═══════════════════════════════════════════════════════════════
   COMPLETE METHOD COMPARISON
  ═══════════════════════════════════════════════════════════════
  
  Method         │ Params  │ Merges│ NLU  │ NLG  │ Scale│ Hard │
  ───────────────┼─────────┼───────┼──────┼──────┼──────┼──────│
  Full FT        │ 100%    │  N/A  │ ★★★★ │ ★★★★ │ All  │ ★★★★ │
  LoRA           │ 0.1-1%  │  Yes  │ ★★★★ │ ★★★★ │ All  │ ★★★  │
  Adapters       │ 1-5%    │  No   │ ★★★  │ ★★★  │ All  │ ★★★  │
  Prefix Tuning  │ 0.1%    │  No   │ ★★★  │ ★★★  │ >1B  │ ★★   │
  Prompt Tuning  │ 0.01%   │  No   │ ★★   │ ★★   │ >10B │ ★    │
  P-Tuning v1    │ 0.1%    │  No   │ ★★★  │ ★★   │ >300M│ ★★   │
  P-Tuning v2    │ 0.1-1%  │  No   │ ★★★★ │ ★★★  │ All  │ ★★★★ │
  ───────────────┴─────────┴───────┴──────┴──────┴──────┴──────┘
  
  Legend: ★ = basic, ★★ = good, ★★★ = very good, ★★★★ = excellent
  
  
  ═══════════════════════════════════════════════════════════════
   P-TUNING FAMILY EVOLUTION
  ═══════════════════════════════════════════════════════════════
  
  2020: Prompt Tuning
    "Just optimize a few continuous tokens at the input"
    → Works, but only for >10B models
    
  2021: P-Tuning v1
    "Use LSTM to generate better continuous prompts"  
    → Unlocks GPT for NLU, great for knowledge probing
    
  2021: Prefix Tuning
    "Add soft prompts at every layer, not just input"
    → Deep prompts solve signal decay
    
  2022: P-Tuning v2
    "Deep prompts work universally, for all sizes & tasks"
    → Matches full FT, validated comprehensively
    
  The key insight progression:
    Discrete → Continuous → Encoded → Deep → Universal
"""
    print(framework)


# ============================================================================
# SECTION 5: PRACTICAL EXPERIMENTS
# ============================================================================

def run_practical_comparison():
    """Run a mini training comparison on actual data."""
    print("\n\n" + "=" * 65)
    print("  SECTION 5: PRACTICAL TRAINING COMPARISON")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import (
        get_peft_model,
        PromptEncoderConfig, PromptEncoderReparameterizationType,
        PrefixTuningConfig,
        LoraConfig,
        TaskType,
    )
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create simple training data
    import random
    random.seed(42)
    
    texts = [
        "The weather is sunny today",
        "Machine learning is fascinating",
        "Python is a great language",
        "Neural networks learn patterns",
        "Data science requires statistics",
    ] * 10
    
    encodings = tokenizer(
        texts, padding="max_length", truncation=True,
        max_length=32, return_tensors="pt",
    )
    labels = encodings["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    dataset = torch.utils.data.TensorDataset(
        encodings["input_ids"], encodings["attention_mask"], labels,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    
    configs = {
        "P-Tuning v1": PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,
            encoder_reparameterization_type=PromptEncoderReparameterizationType.LSTM,
            encoder_hidden_size=256,
        ),
        "P-Tuning v2": PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,
            prefix_projection=False,
        ),
        "LoRA (r=8)": LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["c_attn"],
        ),
    }
    
    all_losses = {}
    
    for name, config in configs.items():
        print(f"\n  Training: {name}")
        
        base = AutoModelForCausalLM.from_pretrained(model_name)
        base.config.pad_token_id = tokenizer.pad_token_id
        model = get_peft_model(base, config)
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Trainable: {trainable:,}")
        
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=3e-3 if "P-Tuning" in name else 3e-4,
        )
        
        losses = []
        model.train()
        
        for epoch in range(5):
            epoch_loss = 0
            for batch in loader:
                ids, mask, labs = batch
                out = model(input_ids=ids, attention_mask=mask, labels=labs)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += out.loss.item()
            
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            print(f"    Epoch {epoch+1}: loss = {avg_loss:.4f}")
        
        all_losses[name] = losses
        del model, base, optimizer
    
    # Summary
    print(f"\n  ── Training Summary ──")
    print(f"  {'Method':<16} {'Start':>8} {'End':>8} {'Reduction':>10}")
    print(f"  {'─'*16}─{'─'*8}─{'─'*8}─{'─'*10}")
    
    for name, losses in all_losses.items():
        print(f"  {name:<16} {losses[0]:>8.4f} {losses[-1]:>8.4f} "
              f"{losses[0]-losses[-1]:>10.4f}")
    
    print(f"\n  Note: This is a tiny experiment. Real-world results may differ.")
    print(f"  P-Tuning methods generally need more epochs to converge.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all comparisons."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     P-TUNING COMPARISON — V1 VS V2 VS OTHER METHODS         ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: V1 vs V2
    compare_v1_v2()
    
    # Section 2: All soft prompt methods
    compare_soft_prompt_methods()
    
    # Section 3: P-Tuning vs LoRA
    compare_with_lora()
    
    # Section 4: Decision framework
    decision_framework()
    
    # Section 5: Practical comparison
    run_practical_comparison()
    
    print("\n" + "=" * 65)
    print("  MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered:
    ✓ V1 vs V2 detailed comparison (params, speed, gradients)
    ✓ All soft prompt methods compared
    ✓ P-Tuning vs LoRA / Adapters
    ✓ Complete decision framework with decision tree
    ✓ Practical training comparison experiment
    """)


if __name__ == "__main__":
    main()
