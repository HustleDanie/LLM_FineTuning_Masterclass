# 01 — Full Fine-Tuning

## What Is Full Fine-Tuning?

Full fine-tuning updates **every single parameter** in the pretrained model.
This is the most straightforward approach — you take a pretrained LLM and continue
training it on your domain-specific dataset with a standard supervised learning loop.

```
┌─────────────────────────┐
│   Pretrained LLM        │  (e.g., GPT-2, LLaMA, Mistral)
│   All parameters frozen │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Task-Specific Dataset │  (e.g., medical texts, legal docs, code)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Train ENTIRE Network  │  ← All weights are updated
│   (all layers, all      │
│    attention heads,     │
│    embeddings, etc.)    │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Domain-Specific Model │
└─────────────────────────┘
```

## Key Characteristics

| Property | Detail |
|----------|--------|
| **Parameters Updated** | ALL (100%) |
| **Compute Cost** | Highest among all methods |
| **GPU Requirement** | Large (16GB+ for small models, 80GB+ for 7B+ models) |
| **Flexibility** | Maximum — model can fully adapt to new domain |
| **Risk** | Catastrophic forgetting if not managed |
| **Training Speed** | Slowest (all gradients computed) |

## When to Use Full Fine-Tuning

1. **Domain-specific models** — You want the model to deeply learn a new domain (medical, legal, finance)
2. **Enough data** — You have a large, high-quality dataset (>10K examples minimum)
3. **Sufficient compute** — You have access to multiple GPUs or cloud instances
4. **Maximum performance** — PEFT methods aren't achieving the quality you need

## When NOT to Use

- Limited GPU memory (use LoRA/QLoRA instead)
- Small dataset (<1K examples) — risk of overfitting
- Quick experimentation (use prompt tuning or adapters)

## Files in This Module

| File | Description |
|------|-------------|
| `full_finetune.py` | Complete training script with all best practices |
| `config.py` | Hyperparameter configuration with explanations |
| `data_utils.py` | Dataset loading, preprocessing, tokenization |
| `evaluation.py` | Evaluation metrics and generation testing |
| `train_causal_lm.py` | Causal language model fine-tuning (text generation) |
| `train_seq_classification.py` | Sequence classification fine-tuning (sentiment, etc.) |
| `training_utils.py` | Utilities: gradient checkpointing, mixed precision, logging |

## Key Techniques Demonstrated

- [x] Full parameter update with AdamW optimizer
- [x] Learning rate scheduling (linear warmup + cosine decay)
- [x] Gradient accumulation (simulate larger batch sizes)
- [x] Gradient checkpointing (trade compute for memory)
- [x] Mixed precision training (FP16/BF16)
- [x] Weight decay and regularization
- [x] Early stopping
- [x] Model checkpointing and resumption
- [x] Evaluation during training
- [x] Distributed training setup (multi-GPU)

## Run

```bash
# Basic training
python full_finetune.py

# With custom config
python full_finetune.py --model_name gpt2 --dataset_name wikitext --epochs 3

# Sequence classification
python train_seq_classification.py

# Causal LM fine-tuning
python train_causal_lm.py
```
