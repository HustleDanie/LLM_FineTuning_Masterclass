# Concept 9: Prompt Tuning

## Overview

**Prompt Tuning** (Lester et al., 2021 — "The Power of Scale for Parameter-Efficient Prompt Tuning") is the
simplest parameter-efficient fine-tuning method. It prepends a small number of **learnable continuous vectors
(soft prompts)** to the input embedding layer — and nothing else. The entire base model stays frozen.

Unlike **prefix tuning** (which injects learned key-value pairs at every transformer layer), prompt tuning
operates **only at the input**. This makes it extraordinarily lightweight, yet remarkably effective —
especially at scale.

## Key Insight

> As models grow larger, the gap between prompt tuning and full fine-tuning **closes to zero**.
> At 10B+ parameters, prompt tuning matches full fine-tuning while training only ~0.001% of parameters.

```
Full FT accuracy at different scales:
  Small  (100M):  ████████████████████  90%
  Medium (1B):    █████████████████████ 93%
  Large  (10B):   █████████████████████ 95%

Prompt Tuning accuracy:
  Small  (100M):  ██████████████        70%   ← Big gap
  Medium (1B):    ███████████████████   88%   ← Gap closing
  Large  (10B):   █████████████████████ 94.5% ← Nearly matches!
```

## Architecture

```
Standard Fine-Tuning:
  Input tokens → [Embedding] → [Transformer Layers] → Output
                  (all trainable)

Prompt Tuning:
  Soft Prompt (trainable)
       ↓
  [P₁ P₂ ... Pₙ | x₁ x₂ ... xₘ] → [Frozen Transformer] → Output
       ↑                    ↑
  Learned vectors     Input embeddings
  (N soft tokens)     (from tokenizer)

  Only the soft prompt vectors are trained!
```

### How It Works

1. Initialize N learnable vectors of dimension `d_model`
2. Prepend them to the input embeddings (before the first transformer layer)
3. The combined sequence `[soft_prompt; input]` flows through the frozen model
4. Backpropagate through the frozen model to update ONLY the soft prompt vectors
5. At inference, prepend the same soft prompt to steer model behavior

### Parameter Count

```
Trainable parameters = num_virtual_tokens × d_model

Example (GPT-2, 20 soft tokens):
  = 20 × 768
  = 15,360 parameters
  = 0.012% of model (vs 124M total)
```

Compare with Prefix Tuning:
```
Prefix: num_layers × 2 × num_tokens × d_model = 12 × 2 × 20 × 768 = 368,640
Prompt: num_tokens × d_model                   = 20 × 768            = 15,360

Prompt tuning uses 24× fewer parameters than prefix tuning!
```

## Prompt Tuning vs Prefix Tuning

| Aspect | Prompt Tuning | Prefix Tuning |
|--------|---------------|---------------|
| Where applied | Input embedding only | Every attention layer (K, V) |
| Parameters | `L × d` | `layers × 2 × L × d` |
| Reparameterization | Not needed | MLP for stability |
| Training stability | Can be unstable (small models) | More stable (more params) |
| Scaling behavior | Matches full FT at scale | Good at all scales |
| Simplicity | ★★★★★ | ★★★ |
| Expressiveness | Lower (input only) | Higher (every layer) |

## Initialization Strategies (Critical!)

The way soft prompts are initialized has a **huge** impact on performance:

| Strategy | Description | Quality |
|----------|-------------|---------|
| Random | Random vectors from normal distribution | Poor |
| Vocab Sample | Sample from existing token embeddings | Good |
| Text Init | Initialize from actual text tokens (e.g., class labels) | Best |

Text initialization with semantically relevant tokens (like "Classify this text as positive or negative:")
provides a strong starting point that dramatically improves convergence.

## Advantages

- ✅ **Extreme parameter efficiency** — fewest trainable parameters of any PEFT method
- ✅ **Dead simple** — just prepend vectors to input
- ✅ **Scales beautifully** — matches full FT on large models
- ✅ **Instant task switching** — swap tiny prompt vectors
- ✅ **Batch different tasks** — different prompts in same batch
- ✅ **No model modification** — frozen model architecture unchanged
- ✅ **Prompt ensembling** — average multiple prompts for robustness

## Limitations

- ❌ **Struggles on small models** — big quality gap below 1B params
- ❌ **Initialization sensitive** — random init can fail completely
- ❌ **Context window cost** — soft tokens consume input positions
- ❌ **Less expressive** — only modifies input, not internal representations
- ❌ **Training instability** — LR sensitivity, longer convergence

## Files in This Module

| File | Description |
|------|-------------|
| `prompt_tuning_theory.py` | Soft prompts, gradient flow, why scale matters |
| `prompt_from_scratch.py` | Build prompt tuning from scratch in PyTorch |
| `prompt_training.py` | Train with HuggingFace PEFT library |
| `prompt_advanced.py` | Initialization strategies, ensembling, transfer |
| `prompt_at_scale.py` | Scaling laws, multi-task serving, production |

## Key References

- Lester et al. (2021): "The Power of Scale for Parameter-Efficient Prompt Tuning"
- Liu et al. (2023): "GPT Understands, Too" (P-Tuning)
- Su et al. (2022): "On Transferability of Prompt Tuning for NLP"
- Vu et al. (2022): "SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer"
