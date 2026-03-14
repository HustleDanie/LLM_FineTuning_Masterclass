# Concept 8: Prefix Tuning

## Overview

**Prefix Tuning** (Li & Liang, 2021) prepends learnable continuous vectors — called **prefixes** — to the keys and values of the attention mechanism at every transformer layer. Only the prefix parameters are trained; all original model weights remain frozen.

This is a **soft prompt** method: instead of modifying model weights or inserting new modules (like adapters), we modify the *input* to the attention mechanism by prepending virtual tokens that the model attends to.

## Key Insight

> Instead of engineering a discrete text prompt, we can learn a continuous "prefix" in the activation space that steers the model's behavior — achieving task-specific adaptation with orders of magnitude fewer parameters.

## Architecture

```
Standard Transformer Attention:
  Q = X · W_q    K = X · W_k    V = X · W_v
  Attention(Q, K, V) = softmax(Q · K^T / √d) · V

Prefix Tuning:
  P_k, P_v ∈ ℝ^(l × d)              ← Learnable prefix matrices (l = prefix length)
  K' = [P_k ; X · W_k]              ← Prepend prefix to keys
  V' = [P_v ; X · W_v]              ← Prepend prefix to values
  Attention(Q, K', V') = softmax(Q · K'^T / √d) · V'

  The model "attends to" virtual prefix tokens alongside real tokens
```

## How It Differs from Discrete Prompts

```
Discrete Prompt:
  Input: "Translate English to French: The cat sat on the mat"
  - Uses existing vocabulary tokens
  - Constrained to token embedding space
  - Found via search/engineering

Prefix Tuning:
  Input: [P₁ P₂ ... Pₗ] + "The cat sat on the mat"
  - P_i are continuous vectors, NOT real tokens
  - Exist in the activation space of EVERY layer
  - Found via backpropagation
  - Much more expressive than discrete tokens
```

## The Reparameterization Trick

Direct optimization of prefix parameters is unstable. The solution: **reparameterize** through a small MLP:

```
P[i,:] = MLP(E[i,:])    where E is a smaller embedding matrix

E ∈ ℝ^(l × d')    →    MLP: d' → d    →    P ∈ ℝ^(l × d)

- Train E and MLP jointly
- After training, discard MLP and keep only the resulting P values
- This acts as a regularizer and stabilizes training
```

## Parameter Count

```
Prefix parameters per layer:
  Keys:   P_k ∈ ℝ^(l × d)    →  l × d parameters
  Values: P_v ∈ ℝ^(l × d)    →  l × d parameters
  Total per layer: 2 × l × d

Total prefix parameters:
  L layers × 2 × l × d

Example (GPT-2 Medium):
  L = 24, l = 20, d = 1024
  Total = 24 × 2 × 20 × 1024 = 983,040 (~1M params)
  vs. GPT-2 Medium: 355M params → 0.28% trainable
```

## Advantages

| Advantage | Details |
|-----------|---------|
| Extremely parameter-efficient | < 0.1-1% of model parameters |
| No architectural changes | Just prepend to K, V at each layer |
| Modular | Swap prefix = swap task |
| No inference overhead | Once prefix is computed, it's just extra tokens |
| Composable | Can concatenate prefixes from different tasks |

## Limitations

| Limitation | Details |
|------------|---------|
| Reduces effective context | Prefix tokens consume part of the context window |
| Initialization-sensitive | Poor initialization → poor convergence |
| Limited capacity | Fewer params → less adaptation power than adapters/LoRA |
| Training instability | Requires reparameterization trick to stabilize |

## Files in This Module

| File | Description |
|------|-------------|
| `prefix_tuning_theory.py` | Mathematical foundations, attention mechanics with prefixes |
| `prefix_from_scratch.py` | Build prefix tuning from scratch in PyTorch |
| `prefix_training.py` | Training with HuggingFace PEFT `PrefixTuningConfig` |
| `prefix_advanced.py` | Reparameterization MLP, multi-task prefix, prefix transfer |
| `prefix_comparison.py` | Prefix tuning vs prompt tuning vs adapters vs LoRA |

## Key Papers

- **Prefix-Tuning** — Li & Liang (2021): "Prefix-Tuning: Optimizing Continuous Prompts for Generation"
- **P-Tuning v2** — Liu et al. (2022): Deep prompt tuning (prefix at every layer)
- **Prompt Tuning** — Lester et al. (2021): Soft prompts at input layer only (simpler variant)
