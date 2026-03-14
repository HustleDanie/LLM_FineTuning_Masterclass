# Concept 10: P-Tuning / P-Tuning v2

## Overview

**P-Tuning** (Liu et al., 2021 — "GPT Understands, Too") and **P-Tuning v2** (Liu et al., 2022 — 
"P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks")
are two related but distinct methods for parameter-efficient fine-tuning.

**P-Tuning v1** introduced the idea of using a **trainable prompt encoder (LSTM/MLP)** to generate 
continuous prompts, rather than directly optimizing prompt embeddings. This dramatically improved 
training stability and performance on knowledge-probing and NLU tasks.

**P-Tuning v2** extended this to **deep continuous prompts at every layer** (similar to prefix tuning)
but with a focus on matching full fine-tuning across ALL model scales and tasks — not just large models.

## The P-Tuning Family Tree

```
Discrete Prompts (manual)
    │
    ├── Prompt Tuning (Lester, 2021)
    │     └── Soft tokens at input only, no encoder
    │
    ├── P-Tuning v1 (Liu, 2021)           ◄── THIS MODULE
    │     └── Soft tokens at input, WITH LSTM/MLP encoder
    │
    ├── Prefix Tuning (Li & Liang, 2021)
    │     └── Soft K,V at every layer, with MLP reparameterization
    │
    └── P-Tuning v2 (Liu, 2022)           ◄── THIS MODULE
          └── Deep prompts at every layer, optimized for all scales
```

## P-Tuning v1: Architecture

```
Traditional Prompt Tuning:
  [p₁, p₂, ..., pₙ] → directly optimized embeddings
  Problem: Unstable optimization, prompts are independent

P-Tuning v1:
  [h₁, h₂, ..., hₙ] → LSTM Encoder → [p₁, p₂, ..., pₙ]
  
  The LSTM creates dependencies between prompt tokens!
  
  ┌──────────────────────────────────────────────────────┐
  │ Prompt Encoder (LSTM + MLP)                          │
  │                                                      │
  │  h₁ → [LSTM] → [MLP] → p₁                          │
  │  h₂ → [LSTM] → [MLP] → p₂   (sequential dependency)│
  │  h₃ → [LSTM] → [MLP] → p₃                          │
  │  ...                                                 │
  │  hₙ → [LSTM] → [MLP] → pₙ                          │
  │                                                      │
  │  The LSTM ensures prompt tokens are "coherent"       │
  │  rather than independently optimized.                │
  └──────────────────────────────────────────────────────┘
  
  Then: [p₁...pₙ, x₁...xₘ] → Frozen Transformer → Output
```

### Why an LSTM Encoder?

1. **Inter-token dependency**: Each prompt token "knows about" previous tokens
2. **Smoother optimization**: The encoder provides a better gradient landscape  
3. **Reparameterization**: Maps from a learned hidden space to embedding space
4. **Bidirectional context**: Bi-LSTM captures both left and right context

## P-Tuning v2: Architecture

```
P-Tuning v2 = Deep Prompt Tuning (prompts at EVERY layer)

Layer 1:  [P¹₁...P¹ₙ | x₁...xₘ]  → Attention + FFN → output₁
Layer 2:  [P²₁...P²ₙ | output₁]   → Attention + FFN → output₂  
Layer 3:  [P³₁...P³ₙ | output₂]   → Attention + FFN → output₃
  ...
Layer L:  [Pᴸ₁...Pᴸₙ | outputₗ₋₁] → Attention + FFN → final

Each layer gets its OWN set of learnable prompt tokens.
This is essentially prefix tuning without the MLP reparameterization.
```

### P-Tuning v2 vs Prefix Tuning

| Aspect | Prefix Tuning | P-Tuning v2 |
|--------|---------------|-------------|
| Where | K, V in attention | Input to each layer |
| Reparameterization | MLP required | Optional (not needed) |
| Training | Train MLP, discard at inference | Direct optimization |
| Focus | Generation tasks | NLU tasks (classification, NER, QA) |
| Scale | Works at all scales | **Specifically optimized for small models** |

## Key Innovation of P-Tuning v2

> "Prompt tuning methods can be comparable to fine-tuning universally
> across scales (from 300M to 10B) and tasks (including hard sequence
> labeling tasks like NER)."

The paper showed that by:
1. Adding prompts at every layer (not just input)
2. Using task-specific prediction heads
3. Optimizing prompt length per task
4. Applying to both NLU and NLG

...you can match full fine-tuning even on **small models** and **hard tasks**.

## Parameter Count

```
P-Tuning v1:
  Prompt embeddings:  num_tokens × d_model
  LSTM encoder:       ~4 × (d_model × d_hidden + d_hidden²)  (bidirectional)
  MLP head:           d_hidden × d_model
  Total:              Typically 0.1-1% of model

P-Tuning v2:
  Per-layer prompts:  num_layers × num_tokens × d_model
  Total:              Same as prefix tuning (0.1-3% of model)
  
  Example (BERT-base, 20 tokens):
  = 12 layers × 20 tokens × 768 dim
  = 184,320 parameters (0.17% of 110M)
```

## Advantages

- ✅ **P-Tuning v1**: LSTM encoder dramatically improves prompt optimization stability
- ✅ **P-Tuning v2**: Matches full FT across ALL model scales (not just large)
- ✅ **P-Tuning v2**: Works on hard tasks (NER, extractive QA) where prompt tuning fails
- ✅ Both: Parameter-efficient with frozen base model
- ✅ Both: Modular — swap prompts for different tasks

## Limitations

- ❌ **P-Tuning v1**: LSTM encoder adds training complexity
- ❌ **P-Tuning v1**: Still input-only (less expressive than v2)
- ❌ **P-Tuning v2**: More params than simple prompt tuning
- ❌ **P-Tuning v2**: Cannot merge into base model (like prefix tuning)

## Files in This Module

| File | Description |
|------|-------------|
| `p_tuning_theory.py` | Theory: prompt encoders, deep prompts, why v2 works at all scales |
| `p_tuning_v1.py` | P-Tuning v1 from scratch: LSTM prompt encoder, template patterns |
| `p_tuning_v2.py` | P-Tuning v2 from scratch: deep layer-wise prompts |
| `p_tuning_training.py` | Training with PEFT library, NLU tasks, save/load |
| `p_tuning_comparison.py` | v1 vs v2 vs prompt tuning vs prefix tuning comparison |

## Key References

- Liu et al. (2021): "GPT Understands, Too" (P-Tuning v1)
- Liu et al. (2022): "P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally"
- Lester et al. (2021): "The Power of Scale for Parameter-Efficient Prompt Tuning"
- Li & Liang (2021): "Prefix-Tuning: Optimizing Continuous Prompts for Generation"
