# Concept 20: Retrieval-Augmented Fine-Tuning (RAFT)

## Overview

**Retrieval-Augmented Fine-Tuning** combines retrieval-augmented generation (RAG) with fine-tuning to create models that can effectively use external knowledge during both training and inference. Instead of memorizing all knowledge in parameters, the model learns *how to retrieve and reason over* external documents.

The key insight: rather than stuffing all knowledge into weights (parametric memory), teach the model to leverage a retrieval system (non-parametric memory) — then fine-tune the model to be excellent at using retrieved context.

## Why Retrieval-Augmented Fine-Tuning?

```
Standard Fine-Tuning:                  Retrieval-Augmented Fine-Tuning:

All knowledge → Model Weights          Core reasoning → Model Weights
                                        +
                                        Domain knowledge → Retrieval Index
                                        
Limitations:                           Advantages:
• Knowledge frozen at training time     • Knowledge updated by updating index
• Hallucination on rare facts           • Grounded in retrieved evidence
• Large model needed for recall         • Smaller model + large knowledge base
• Can't cite sources                    • Can cite exact sources
```

## Core Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Query     │───→│   Retriever  │───→│ Retrieved Docs   │
│             │    │  (Dense/BM25)│    │ [doc1, doc2, ...] │
└─────────────┘    └──────────────┘    └────────┬──────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │   Fine-Tuned LLM      │
                                    │  (Trained to use       │
                                    │   retrieved context)   │
                                    └───────────┬───────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │   Grounded Answer      │
                                    │  (with citations)      │
                                    └───────────────────────┘
```

## Key Approaches

### 1. RAFT (Retrieval Augmented Fine-Tuning)
Train the model to distinguish relevant ("oracle") documents from irrelevant ("distractor") documents in the retrieved context. The model learns to extract answers from the right document and ignore noise.

### 2. RA-DIT (Retrieval-Augmented Dual Instruction Tuning)
Jointly fine-tune the retriever AND the language model. Both components learn to work together optimally.

### 3. REPLUG (Retrieve and Plug)
Treat the retriever as a plug-in module. The LM is fine-tuned to handle prepended retrieved documents, and the retriever is tuned using the LM's signal.

### 4. Self-RAG (Self-Reflective RAG)
Fine-tune the model to decide WHEN to retrieve, WHAT to retrieve, and whether the retrieved content is useful — all through special reflection tokens.

## RAFT Training Strategy

The signature RAFT approach (Zhang et al., 2024):

| Training Example Type | Oracle Doc | Distractor Docs | Proportion |
|----------------------|------------|------------------|------------|
| Type 1: Oracle + Distractors | ✅ Present | ✅ Present | ~60% |
| Type 2: Oracle Only | ✅ Present | ❌ Absent | ~20% |
| Type 3: Distractors Only | ❌ Absent | ✅ Present | ~20% |

This teaches the model to:
- Find relevant info among noise (Type 1)
- Use good context when available (Type 2)  
- Rely on parametric knowledge when retrieval fails (Type 3)

## When to Use Retrieval-Augmented Fine-Tuning

### ✓ Recommended When:
- Domain has frequently changing knowledge (news, legal, medical)
- Factual accuracy is critical (must cite sources)
- Knowledge base is too large to memorize in parameters
- Need to ground answers in specific documents
- Want updatable knowledge without retraining

### ✗ Less Effective When:
- Task is purely reasoning/creative (no external knowledge needed)
- Latency requirements prohibit retrieval step
- Knowledge base is small enough to fit in model parameters
- Real-time streaming tasks with no retrieval opportunity

## Files in This Module

| File | Description |
|------|-------------|
| `README.md` | This overview document |
| `raft_theory.py` | Theory: RAG fundamentals, retrieval-generation coupling, knowledge grounding |
| `raft_from_scratch.py` | From-scratch: retriever, reader, RAFT training loop, Self-RAG |
| `raft_training.py` | Production: HuggingFace RAFT pipeline, dense retrieval, evaluation |
| `raft_comparison.py` | Analysis: RAG vs RAFT vs standard FT, retrieval strategies, scaling |

## Key References

1. **RAFT (Zhang et al., 2024)** — "RAFT: Adapting Language Model to Domain-Specific RAG"
2. **RAG (Lewis et al., 2020)** — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
3. **REPLUG (Shi et al., 2023)** — "REPLUG: Retrieval-Augmented Black-Box Language Models"
4. **Self-RAG (Asai et al., 2023)** — "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
5. **RA-DIT (Lin et al., 2023)** — "RA-DIT: Retrieval-Augmented Dual Instruction Tuning"
6. **REALM (Guu et al., 2020)** — "REALM: Retrieval-Augmented Language Model Pre-Training"
7. **Atlas (Izacard et al., 2022)** — "Few-shot Learning with Retrieval Augmented Language Models"
