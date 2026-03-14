# Concept 6: QLoRA — Quantized Low-Rank Adaptation

## Overview

**QLoRA** (Quantized LoRA) enables fine-tuning of massive language models on a single consumer GPU by combining three key innovations:

1. **4-bit NormalFloat (NF4) quantization** — A new data type optimized for normally-distributed neural network weights
2. **Double quantization** — Quantize the quantization constants to save additional memory
3. **Paged optimizers** — Use CPU memory as overflow for optimizer states via unified memory

With QLoRA, you can fine-tune a **65B parameter model on a single 48GB GPU** or a **33B model on a 24GB GPU** — tasks that would otherwise require multiple A100 GPUs.

## Key Insight

> The base model weights are frozen and quantized to 4-bit. Only the LoRA adapter weights (in FP16/BF16) are trained. During the forward pass, 4-bit weights are dequantized on-the-fly to BF16 for computation, and gradients flow only through the LoRA parameters.

## Memory Breakdown (7B Model)

| Component | LoRA (FP16) | QLoRA (NF4) |
|-----------|-------------|-------------|
| Base model weights | 14 GB | **3.5 GB** |
| LoRA adapter (r=16) | 40 MB | 40 MB |
| Optimizer states | 80 MB | 80 MB |
| Activations/gradients | ~2 GB | ~2 GB |
| **Total** | **~16 GB** | **~5.6 GB** |

## Files in This Module

| File | Description |
|------|-------------|
| `quantization_fundamentals.py` | Quantization theory: INT8, FP16, BF16, NF4 — how they work and compare |
| `nf4_deep_dive.py` | NF4 data type from scratch: information-theoretic optimality, implementation |
| `double_quantization.py` | Double quantization: quantizing the quantization constants |
| `paged_optimizers.py` | Paged optimizers: CPU offloading via CUDA unified memory |
| `qlora_training.py` | Complete QLoRA training pipeline with best practices |
| `qlora_vs_lora.py` | Head-to-head comparison: memory, speed, and quality analysis |

## The QLoRA Paper

- **Title**: QLoRA: Efficient Finetuning of Quantized Language Models
- **Authors**: Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer
- **Published**: May 2023
- **Key finding**: QLoRA matches 16-bit full fine-tuning performance while reducing memory by 4x

## Prerequisites

```bash
pip install torch transformers datasets peft trl bitsandbytes accelerate
```

> **Note**: `bitsandbytes` is the library that provides NF4 quantization and paged optimizers. On Windows, use `bitsandbytes-windows` or the latest version which has Windows support.
