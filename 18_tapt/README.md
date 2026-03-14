# Concept 18: Task-Adaptive Pretraining (TAPT)

## Overview

**Task-Adaptive Pretraining (TAPT)** is a lightweight continued pretraining technique that adapts a language model to the *specific distribution of a downstream task's data* — using only the **unlabeled text** from the task itself. Unlike DAPT, which uses a large domain corpus, TAPT operates on a much smaller and more targeted dataset: the exact texts the model will encounter during fine-tuning.

The key insight from Gururangan et al. (2020): even a small amount of continued pretraining on task-relevant text yields consistent improvements, and **TAPT is complementary to DAPT** — combining both gives the best results.

## Why TAPT Works

```
Pretraining Distribution (Web text):     ████████████████████████████████
                                                                        
Domain Distribution (e.g., Biomedical):        ██████████████████        
                                                                        
Task Distribution (e.g., Drug Reviews):              ██████              
                                                                        
TAPT narrows the model's focus to the exact task distribution:
                                                                        
Before TAPT: Model "sees" broadly     → wastes capacity on irrelevant text
After TAPT:  Model "sees" task text   → concentrates capacity on what matters
```

### Distribution Narrowing

TAPT performs continued language modeling on the **unlabeled task data** (ignoring labels), forcing the model to better represent exactly the text it will encounter during supervised fine-tuning.

## TAPT vs DAPT

| Aspect | DAPT | TAPT |
|--------|------|------|
| **Data source** | Large domain corpus (papers, code, etc.) | Task dataset itself (unlabeled) |
| **Data size** | 10M–1B+ tokens | 500–50K examples (small!) |
| **Vocabulary** | Broad domain terms | Task-specific terms |
| **Training time** | Hours to days | Minutes to hours |
| **Compute cost** | Moderate to high | Very low |
| **Best for** | Domain shift | Task-specific adaptation |
| **Combined?** | ✓ Best with TAPT | ✓ Best with DAPT |

## How TAPT Works

### Step-by-Step

1. **Take your task dataset** (e.g., sentiment reviews, NER documents)
2. **Strip the labels** — keep only the input text
3. **Continue pretraining** the base model on this unlabeled text using standard language modeling objective (CLM or MLM)
4. **Fine-tune** the now-adapted model on the labeled task as usual

### Training Considerations

| Parameter | Recommended Value | Rationale |
|-----------|-------------------|-----------|
| Learning rate | 1e-5 to 5e-5 | Low to preserve knowledge |
| Epochs | 10–100 (data is small!) | More passes needed on tiny data |
| Warmup | 5–10% of steps | Smooth start |
| LR schedule | Cosine or linear decay | Standard practice |
| Batch size | 8–32 | Small data, moderate batch |
| Max seq length | Match task input length | Align distributions |

### Key Difference: Epoch Count

TAPT uses **many more epochs** than DAPT because the dataset is tiny. While you'd never run DAPT for 100 epochs on a billion-token corpus, running TAPT for 50–100 epochs on 5,000 examples is perfectly reasonable — and beneficial.

## Data Augmentation for TAPT

Since TAPT data is small, data augmentation helps:

### Curated TAPT (Gururangan et al. 2020)

1. Embed all task examples using the model
2. For each task example, retrieve **k nearest neighbors** from a large unlabeled corpus
3. Add these neighbors to the TAPT data
4. Run TAPT on this expanded dataset

This effectively expands the TAPT corpus with **task-similar** examples from a larger pool.

### Other Augmentation Strategies

- **Backtranslation**: Translate task text to another language and back
- **Paraphrase generation**: Use a seq2seq model to generate paraphrases
- **Sentence reordering**: Shuffle sentences within documents
- **Task-relevant retrieval**: BM25/semantic search from general corpus

## When to Use TAPT

### ✓ Always Beneficial When:
- You have unlabeled task data available (you usually do — it's your training set!)
- Task distribution differs from pretraining data
- Compute budget is limited (TAPT is very cheap)
- You want to squeeze extra performance cheaply

### ✗ May Not Help When:
- Task data is extremely small (< 100 examples)
- Task text is indistinguishable from pretraining data
- Model is already domain-adapted (diminishing returns)

## Combined Pipeline: DAPT + TAPT

```
Base Model → [DAPT] → [TAPT] → Task Fine-Tuning
              broad     narrow    supervised
              domain    task      labels
              
Gururangan et al. (2020) results:
  Base + FT:           82.3%
  DAPT + FT:           86.7%  (+4.4%)
  TAPT + FT:           84.1%  (+1.8%)
  DAPT + TAPT + FT:    88.2%  (+5.9%)  ★ Best!
```

## Files in This Module

| File | Description |
|------|-------------|
| `README.md` | This overview document |
| `tapt_theory.py` | Theory: task distribution analysis, overlap metrics, when TAPT helps |
| `tapt_from_scratch.py` | From-scratch: CLM-TAPT, MLM-TAPT, Curated TAPT, multi-epoch scheduling |
| `tapt_training.py` | Production: HuggingFace pipeline, LoRA-TAPT, Curated TAPT with retrieval |
| `tapt_comparison.py` | Analysis: TAPT vs DAPT vs combined, epoch scaling, augmentation effects |

## Key References

1. **Gururangan et al. (2020)** — "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks" (ACL 2020)
2. **Howard & Ruder (2018)** — ULMFiT: Universal Language Model Fine-tuning
3. **Sun et al. (2019)** — "How to Fine-Tune BERT for Text Classification"
4. **Chronopoulou et al. (2019)** — "An Embarrassingly Simple Approach for Transfer Learning from Pretrained Language Models"
