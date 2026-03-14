# Concept 12: BitFit (Bias-Term Fine-Tuning)

## Overview

**BitFit** (Bias-terms Fine-Tuning), introduced by Ben Zaken et al. (2022), is a remarkably simple
parameter-efficient fine-tuning method: **freeze all model weights and only train the bias terms**.

Despite updating only ~0.08-0.1% of all parameters, BitFit achieves surprisingly competitive
results on NLU benchmarks, revealing that bias terms encode far more task-relevant information
than their tiny size suggests.

## Core Idea

```
Standard Fine-Tuning:    Update ALL parameters (W and b)
BitFit:                  Freeze W, update ONLY b

Forward Pass:  y = Wx + b
                │       │
                frozen   ← trainable (BitFit)
```

## Where Are Bias Terms?

In a Transformer, bias terms appear in:

```
┌─────────────────────────────────────────────────┐
│                  Transformer Layer               │
│                                                  │
│  ┌─────────────────────────────────────────┐     │
│  │     Multi-Head Attention                │     │
│  │  Q = Wx + b_q  ← bias trainable        │     │
│  │  K = Wx + b_k  ← bias trainable        │     │
│  │  V = Wx + b_v  ← bias trainable        │     │
│  │  Out = Wx + b_o ← bias trainable       │     │
│  └─────────────────────────────────────────┘     │
│                     │                            │
│  ┌─────────────────────────────────────────┐     │
│  │     LayerNorm                           │     │
│  │  γ (scale) + β (shift) ← β trainable   │     │
│  └─────────────────────────────────────────┘     │
│                     │                            │
│  ┌─────────────────────────────────────────┐     │
│  │     Feed-Forward Network                │     │
│  │  FFN1: Wx + b_ff1 ← bias trainable     │     │
│  │  FFN2: Wx + b_ff2 ← bias trainable     │     │
│  └─────────────────────────────────────────┘     │
│                     │                            │
│  ┌─────────────────────────────────────────┐     │
│  │     LayerNorm                           │     │
│  │  γ (scale) + β (shift) ← β trainable   │     │
│  └─────────────────────────────────────────┘     │
└─────────────────────────────────────────────────┘
```

## Parameter Count

| Component        | Weight Params | Bias Params | Ratio     |
|------------------|---------------|-------------|-----------|
| Q/K/V/O proj     | 4 × d² each  | 4 × d       | d : 1     |
| FFN (up + down)  | 2 × d × 4d   | d + 4d      | ~1.6d : 1 |
| LayerNorm        | 2 × d (γ)    | 2 × d (β)   | 1 : 1     |

For GPT-2 (d=768, 12 layers):
- Total parameters: ~124M
- Bias parameters: ~100K (~0.08%)

## BitFit Variants

| Variant       | What's Trainable              | % Params (GPT-2) |
|---------------|-------------------------------|-------------------|
| Full BitFit   | All bias terms + LN shifts    | ~0.08-0.10%       |
| Attn-only     | Only attention biases         | ~0.03%            |
| FF-only       | Only feed-forward biases      | ~0.04%            |
| Query-only    | Only query bias (b_q)         | ~0.007%           |
| LN-only       | Only LayerNorm β parameters   | ~0.015%           |

## Why Does It Work?

1. **Bias as activation threshold**: Each bias shifts the activation threshold,
   controlling which features "fire" for a given input
2. **Task-specific feature selection**: By adjusting biases, the model changes
   which pretrained features are activated for the new task
3. **Implicit regularization**: Extreme parameter reduction prevents overfitting
4. **Preserved knowledge**: Frozen weights retain pretrained representations

## Key Results (from the Paper)

- On GLUE benchmark: BitFit achieves 88-91% of full fine-tuning performance
- On some tasks (SST-2, RTE): Nearly matches full fine-tuning
- On complex tasks (MNLI, QQP): Gap increases but remains competitive
- Few-shot: Excellent due to strong regularization

## BitFit vs Other PEFT Methods

| Aspect            | BitFit   | IA³      | LoRA     | Prompt Tuning |
|-------------------|----------|----------|----------|---------------|
| Trainable %       | ~0.08%   | ~0.04%   | ~0.5-2%  | ~0.01-0.1%    |
| Complexity        | Trivial  | Simple   | Moderate | Moderate      |
| Architectural Δ   | None     | None     | None     | Extra tokens  |
| Inference cost    | Zero     | Zero*    | Zero*    | Extra compute |
| NLU performance   | Good     | Good     | Very good| Moderate      |
| NLG performance   | Limited  | Moderate | Very good| Moderate      |

*After merging

## Advantages

- **Extreme simplicity**: No new modules, no code changes — just freeze/unfreeze
- **Zero inference overhead**: Same architecture, same speed
- **Tiny storage**: ~100-400KB per task checkpoint
- **Strong baseline**: Punches well above its weight on NLU tasks
- **Composable**: Can stack with other methods

## Limitations

- **Limited expressiveness**: Can only shift activations, not transform them
- **Weaker on generation**: Text generation benefits from weight updates
- **Model-dependent**: Works better on models with many bias terms
- **Ceiling effect**: Can't match LoRA on complex tasks

## Files

| File | Description |
|------|-------------|
| `bitfit_theory.py` | Why bias-only training works, gradient analysis, feature selection theory |
| `bitfit_from_scratch.py` | Manual BitFit implementation, selective freezing, bias extraction |
| `bitfit_training.py` | Training pipelines, variants, hyperparameter guide |
| `bitfit_comparison.py` | BitFit vs LoRA vs IA³, decision framework |

## References

- Ben Zaken et al. (2022): "BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models"
- Subsequent analyses in PEFT literature comparing bias-term importance
