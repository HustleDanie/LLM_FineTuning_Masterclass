# Concept 4: Parameter-Efficient Fine-Tuning (PEFT) — Overview

## Why PEFT?

Full fine-tuning updates **ALL** model parameters. For modern LLMs, this is:

| Model | Parameters | Full FT Memory (fp16) | Full FT Memory (fp32) |
|-------|------------|----------------------|----------------------|
| GPT-2 | 124M | ~0.5 GB | ~1.0 GB |
| LLaMA-7B | 7B | ~14 GB | ~28 GB |
| LLaMA-13B | 13B | ~26 GB | ~52 GB |
| LLaMA-70B | 70B | ~140 GB | ~280 GB |
| GPT-3 | 175B | ~350 GB | ~700 GB |

**Note:** Training memory is 3-4x model memory (optimizer states + gradients + activations).

**PEFT solves this** by training only a TINY fraction of parameters (0.1-5%), achieving comparable results at a fraction of the cost.

## PEFT Method Taxonomy

```
PEFT Methods
├── Additive Methods (add new parameters)
│   ├── Adapters         — Small bottleneck layers inserted into transformer blocks
│   ├── Prefix Tuning    — Learnable prefix tokens prepended to keys/values
│   ├── Prompt Tuning    — Learnable soft tokens prepended to input
│   └── P-Tuning v2      — Learnable prompts at every layer
│
├── Reparameterization Methods (decompose weight updates)
│   ├── LoRA             — Low-rank decomposition of weight updates (ΔW = BA)
│   ├── QLoRA            — LoRA + 4-bit quantized base model
│   └── IA³              — Learned vectors that rescale activations
│
├── Selective Methods (train subset of existing parameters)
│   ├── BitFit           — Train only bias terms
│   └── Layer Freezing   — Train only specific layers
│
└── Hybrid Methods (combine approaches)
    ├── MAM Adapter      — Parallel adapter + prefix tuning
    └── UniPELT          — Combines LoRA + prefix + adapter with gating
```

## Comparison at a Glance

| Method | Trainable % | Memory Savings | Inference Overhead | Quality vs Full FT |
|--------|------------|----------------|-------------------|-------------------|
| Full FT | 100% | None | None | Baseline |
| LoRA | 0.1-1% | ~60-70% | None (merged) | ~98-100% |
| QLoRA | 0.1-1% | ~80-90% | Slight | ~95-100% |
| Adapters | 1-5% | ~50-60% | Slight latency | ~95-98% |
| Prefix Tuning | 0.1% | ~60% | Prefix overhead | ~90-95% |
| Prompt Tuning | <0.1% | ~70% | Prefix overhead | ~85-95% |
| P-Tuning v2 | 0.1-1% | ~60% | Prefix overhead | ~93-97% |
| IA³ | <0.1% | ~70% | Negligible | ~90-95% |
| BitFit | ~0.1% | ~60% | None | ~85-92% |

## Key Insight: The PEFT Library

HuggingFace's `peft` library provides a unified API for all these methods:

```python
from peft import get_peft_model, LoraConfig, TaskType

# 1. Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# 2. Define PEFT config (e.g., LoRA)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
)

# 3. Wrap model — only PEFT parameters are trainable
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622%
```

## Files in This Module

| File | Description |
|------|-------------|
| `peft_methods.py` | Detailed explanation of every PEFT method with architecture diagrams |
| `memory_analysis.py` | Memory/compute comparison: Full FT vs PEFT methods |
| `peft_config.py` | Configuration patterns for all PEFT methods using the `peft` library |
| `peft_training.py` | Complete training pipeline with PEFT (LoRA demo) |
| `peft_comparison.py` | Head-to-head comparison: train same model with different PEFT methods |
| `peft_advanced.py` | Advanced topics: merging, multi-adapter, serving strategies |
