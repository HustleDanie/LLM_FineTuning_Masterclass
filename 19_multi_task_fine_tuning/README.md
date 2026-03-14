# Concept 19: Multi-Task Fine-Tuning

## Overview

**Multi-Task Fine-Tuning (MTL)** trains a single model on multiple tasks simultaneously, leveraging shared representations to improve generalization. Instead of fine-tuning separate models for each task, MTL produces one model that handles classification, NER, QA, summarization, and more — all at once.

The key insight: tasks that share underlying linguistic knowledge benefit from joint training. A model learning sentiment analysis also improves at emotion detection, sarcasm recognition, and topic classification through **positive transfer**.

## Why Multi-Task Fine-Tuning?

```
Traditional Fine-Tuning:              Multi-Task Fine-Tuning:
                                       
Base → FT → Sentiment Model           Base → MTL → One Multi-Task Model
Base → FT → NER Model                         │      ├── Sentiment ✓
Base → FT → QA Model                          │      ├── NER ✓
Base → FT → Summary Model                     │      ├── QA ✓
                                               │      └── Summary ✓
4 separate models                      1 shared model
4× storage, 4× serving cost           1× storage, 1× serving cost
No cross-task knowledge sharing        Rich cross-task transfer
```

## Core Concepts

### 1. Hard Parameter Sharing

The most common MTL architecture: shared encoder + task-specific heads.

```
                ┌── Sentiment Head (Linear → 3 classes)
                │
Input → Shared  ├── NER Head (Linear → BIO tags)
        Encoder │
                ├── QA Head (Linear → start/end span)
                │
                └── Summary Head (LM head → tokens)
```

### 2. Soft Parameter Sharing

Each task has its own encoder, but parameters are regularized to stay similar.

### 3. Task Balancing

Critical challenge: different tasks have different dataset sizes, loss scales, and learning dynamics. Strategies include:
- **Proportional sampling**: Sample batches proportional to dataset size
- **Temperature sampling**: Apply temperature to sampling probabilities
- **Equal sampling**: Round-robin across tasks
- **Dynamic weighting**: Adjust task weights during training

## Task Mixing Strategies

| Strategy | Formula | Pros | Cons |
|----------|---------|------|------|
| Proportional | p_i = N_i / ΣN | Large tasks dominate | Small task underfit |
| Square root | p_i = √N_i / Σ√N | Balanced | Good default |
| Temperature | p_i = N_i^(1/T) / ΣN^(1/T) | Tunable | Extra hyperparameter |
| Equal | p_i = 1/K | Fair to all | Large tasks underfit |

## Instruction-Based MTL

Modern approach: frame ALL tasks as text generation with instructions.

```
Task: Sentiment
Input:  "Classify sentiment: This movie was amazing!"
Output: "positive"

Task: NER  
Input:  "Extract entities: John visited Paris last Monday."
Output: "John [PERSON], Paris [LOCATION], last Monday [DATE]"

Task: QA
Input:  "Answer: What is the capital of France? Context: France is..."
Output: "Paris"
```

This unifies all tasks into a single sequence-to-sequence format, eliminating the need for task-specific heads.

## When to Use Multi-Task Fine-Tuning

### ✓ Recommended When:
- You have multiple related tasks in the same domain
- You want a single model for deployment efficiency
- Small task datasets benefit from shared representations
- Tasks share underlying linguistic features

### ✗ Less Effective When:
- Tasks are unrelated or conflicting (e.g., translation + code generation)
- One task dominates and others become afterthoughts
- Task-specific optimization is critical for each task

## Key Challenges

1. **Negative Transfer**: Unrelated tasks can hurt each other
2. **Task Balancing**: Large vs small tasks, easy vs hard tasks
3. **Gradient Conflict**: Tasks pulling parameters in opposite directions
4. **Catastrophic Forgetting**: Later tasks overwriting earlier task knowledge
5. **Evaluation Complexity**: How to measure overall multi-task performance

## Files in This Module

| File | Description |
|------|-------------|
| `README.md` | This overview document |
| `mtl_theory.py` | Theory: task relatedness, transfer, gradient conflicts, balancing |
| `mtl_from_scratch.py` | From-scratch: hard/soft sharing, task heads, gradient surgery |
| `mtl_training.py` | Production: HuggingFace multi-task pipeline, instruction-based MTL |
| `mtl_comparison.py` | Analysis: single-task vs MTL, task balancing strategies, scaling |

## Key References

1. **Caruana (1997)** — "Multitask Learning" (original MTL paper)
2. **Ruder (2017)** — "An Overview of Multi-Task Learning in Deep Neural Networks"
3. **T5 (Raffel et al., 2020)** — "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
4. **FLAN (Wei et al., 2022)** — "Finetuned Language Models Are Zero-Shot Learners"
5. **ExT5 (Aribandi et al., 2022)** — "ExT5: Towards Extreme Multi-Task Scaling for Transfer Learning"
6. **Aghajanyan et al. (2021)** — "Muppet: Massive Multi-task Representations with Pre-Finetuning"
7. **Yu et al. (2020)** — "Gradient Surgery for Multi-Task Learning" (PCGrad)
