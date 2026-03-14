# Concept 7: Adapter Layers

## Overview

**Adapter layers** are small bottleneck modules inserted between the layers of a pretrained model. During fine-tuning, only the adapter parameters are trained while the original model weights remain frozen. This was one of the earliest PEFT methods, introduced by Houlsby et al. (2019).

## Key Idea

Instead of modifying existing weights (like LoRA), adapters **add new parameters** to the model architecture:

```
Input
  │
  ▼
[Frozen Layer]  ──→  h_frozen
  │
  ▼
[Adapter Module]     h_adapter = Up(Act(Down(h_frozen)))
  │                  where Down: d → r, Up: r → d
  ▼
h_frozen + h_adapter  (residual connection)
  │
  ▼
Output
```

## Adapter Architecture

```
         ┌──────────────┐
    x ──→│  Down-project │──→ (d → r)  bottleneck
         │   (d × r)     │
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │  Activation   │──→ ReLU / GELU / SiLU
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │  Up-project   │──→ (r → d)  restore dim
         │   (r × d)     │
         └──────┬───────┘
                │
    x ─────────+──────────→  residual connection
                │
              output
```

## LoRA vs Adapters

| Feature | LoRA | Adapters |
|---------|------|----------|
| Where | Modifies existing weights | Adds new modules |
| Inference overhead | Zero (after merge) | Non-zero (extra layers) |
| Parameter count | Very low | Low |
| Architecture change | None | Yes (new layers) |
| Mergeability | Yes | No (separate modules) |
| Composability | Limited | Excellent (AdapterFusion) |

## Files in This Module

| File | Description |
|------|-------------|
| `adapter_architecture.py` | Bottleneck adapter math and from-scratch implementation |
| `adapter_variants.py` | Houlsby, Pfeiffer, parallel adapters, and more |
| `adapter_fusion.py` | AdapterFusion for combining task-specific adapters |
| `adapter_training.py` | Complete adapter training pipeline with HuggingFace |
| `adapter_hub.py` | Adapter sharing, composition, and hub patterns |

## Key Papers

1. **Houlsby et al. (2019)** — "Parameter-Efficient Transfer Learning for NLP" (original adapters)
2. **Pfeiffer et al. (2021)** — "AdapterFusion: Non-Destructive Task Composition"  
3. **He et al. (2022)** — "Towards a Unified View of Parameter-Efficient Transfer Learning"
4. **Rücklé et al. (2021)** — "AdapterDrop: On the Efficiency of Adapters in Transformers"

## Quick Start

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model, BottleneckConfig

model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Configure bottleneck adapter
config = BottleneckConfig(
    bottleneck_size=64,         # Bottleneck dimension
    non_linearity="relu",      # Activation function
    adapter_dropout=0.1,
    target_modules=["c_attn", "c_proj"],
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# → trainable params: ~0.5M (0.6% of total)
```
