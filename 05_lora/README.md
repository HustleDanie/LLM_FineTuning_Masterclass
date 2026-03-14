# Concept 5: LoRA (Low-Rank Adaptation) — Deep Dive

## Overview

LoRA (Low-Rank Adaptation of Large Language Models) is the most widely adopted
parameter-efficient fine-tuning method. This module goes **far beyond** the brief
overview in Concept 4, providing a rigorous treatment of the mathematics, design
decisions, implementation details, and practical recipes that make LoRA work.

**Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
(Hu et al., 2021)

---

## Core Insight

Pre-trained weight matrices are full-rank (~thousands), but the *update* needed
for a downstream task lives in a much lower-dimensional subspace. LoRA exploits
this by decomposing the weight update into two small matrices:

```
W' = W + ΔW = W + B·A

W  ∈ ℝ^{d×k}        (frozen original weights)
A  ∈ ℝ^{r×k}        (low-rank down-projection, trained)
B  ∈ ℝ^{d×r}        (low-rank up-projection, trained)
r  << min(d, k)     (rank, typically 4-64)
```

### Why It Works — Intrinsic Dimensionality

Aghajanyan et al. (2020) showed that fine-tuning objectives have a low
**intrinsic dimensionality** — you can project gradient updates into a random
low-dimensional subspace and still recover ~90% of full fine-tuning performance.
LoRA makes this subspace *learnable* rather than random.

---

## Module Files

| File | Description |
|------|-------------|
| `lora_math.py` | Mathematical foundations: SVD, low-rank approximation, intrinsic dimensionality experiments |
| `lora_from_scratch.py` | Complete LoRA implementation from scratch with detailed commentary |
| `lora_rank_analysis.py` | How rank affects quality, per-layer rank importance, rank search strategies |
| `lora_hyperparams.py` | Target module selection, alpha/rank ratio, dropout, learning rate, and full tuning guide |
| `lora_training.py` | Production training pipeline using HuggingFace PEFT |
| `lora_variants.py` | LoRA+, rsLoRA, DoRA, AdaLoRA, DyLoRA, QA-LoRA, and other extensions |
| `lora_merge_deploy.py` | Weight merging, multi-adapter serving, quantized deployment, GGUF export |

---

## Key Concepts Covered

### Mathematics
- Singular Value Decomposition (SVD) connection
- Low-rank matrix approximation (Eckart–Young theorem)
- Intrinsic dimensionality of fine-tuning
- Why ΔW is approximately low-rank in practice

### Implementation
- Forward pass: `h = Wx + (BAx) · (α/r)`
- Initialization: A ~ N(0, σ²), B = 0 (so ΔW = 0 at init)
- Scaling factor α/r and its effect on learning dynamics
- Gradient flow through the low-rank path

### Design Decisions
- Which modules to adapt (attention vs. MLP vs. all linear)
- Rank selection strategies (fixed, per-layer, adaptive)
- Alpha-to-rank ratio and its relationship to learning rate
- Dropout on the low-rank path

### Practical Recipes
- Chat/instruction tuning: r=16-64, all linear layers
- Classification: r=8-16, attention only
- Domain adaptation: r=32-128, all linear + embeddings
- Memory-constrained: r=4-8, query/value only

---

## Quick Comparison: LoRA vs Full Fine-Tuning

| Aspect | Full FT | LoRA (r=16) |
|--------|---------|-------------|
| Trainable params (7B model) | 7B (100%) | ~16M (0.23%) |
| GPU memory (7B model) | ~56 GB | ~18 GB |
| Training speed | 1x | ~1.3x faster |
| Checkpoint size | ~14 GB | ~33 MB |
| Quality (instruction following) | Baseline | ~95-100% |
| Multi-task serving | N copies | 1 base + N adapters |

---

## Prerequisites

- Complete understanding of Concepts 1-4
- `pip install torch transformers peft datasets accelerate bitsandbytes`
