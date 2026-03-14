# Concept 17: Domain-Adaptive Pretraining (DAPT)

## Overview

**Domain-Adaptive Pretraining (DAPT)** is the practice of continuing to pretrain a general-purpose language model on a large corpus of **unlabeled, domain-specific text** before fine-tuning it on a downstream task. The seminal work by Gururangan et al. (2020) — *"Don't Stop Pretraining"* — demonstrated that even models pretrained on massive general corpora benefit significantly from additional pretraining on in-domain data.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  General LM  │ →  │  DAPT on     │ →  │  Task Fine-  │
│  (e.g. BERT, │    │  Domain Text │    │  Tuning      │
│   LLaMA)     │    │  (unlabeled) │    │  (labeled)   │
└──────────────┘    └──────────────┘    └──────────────┘
     Stage 0             Stage 1             Stage 2
```

## Why DAPT Works

### Domain Shift Problem
General LMs are pretrained on broad web text (Wikipedia, books, web crawls), but real-world tasks often involve **specialized domains** where:
- **Vocabulary differs**: Medical text has terms like "tachycardia," legal text uses "estoppel"
- **Statistics shift**: Word co-occurrence patterns change across domains
- **Syntax/style varies**: Scientific prose vs. social media vs. legal contracts
- **World knowledge differs**: Domain-specific facts not well-represented in general corpora

### What DAPT Does
1. **Adapts representations**: Internal model representations shift toward domain-specific patterns
2. **Learns domain vocabulary**: Subword tokenization becomes more efficient for domain terms
3. **Captures domain knowledge**: Absorbs factual and procedural knowledge from domain text
4. **Improves downstream performance**: Typically +1-10% on domain-specific tasks

## Key Concepts

### Domain Distance
The benefit of DAPT scales with the **distance** between the general pretrain domain and the target domain:
- **Large domain shift** (general → biomedical): Large DAPT benefit (+5-10%)
- **Small domain shift** (general → news): Smaller DAPT benefit (+1-3%)
- **No domain shift** (general → general): Minimal or no benefit

### Vocabulary Overlap Analysis
Before DAPT, analyze how well the existing tokenizer handles domain text:
```
General LM vocabulary overlap with domain:
  News:      85-90% (high overlap → small DAPT benefit)
  Biomedical: 60-70% (moderate → medium benefit)
  Legal:     65-75% (moderate → medium benefit)  
  Code:      40-60% (low overlap → large DAPT benefit)
```

### Data Requirements
| Domain Distance | Recommended DAPT Data | Typical Improvement |
|----------------|----------------------|-------------------|
| Close (news) | 10-50M tokens | +1-3% |
| Medium (legal, scientific) | 50-500M tokens | +3-7% |
| Far (biomedical, code) | 500M-5B tokens | +5-10% |

## DAPT vs Related Techniques

| Technique | Data Used | Goal |
|-----------|-----------|------|
| **DAPT** | Unlabeled domain text | Adapt representations to domain |
| **TAPT** (Concept 18) | Unlabeled task-specific text | Adapt to specific task distribution |
| **Continual Pretraining** (Concept 16) | Sequential domains | Adapt across multiple domains |
| **Fine-Tuning** | Labeled task data | Learn specific task mapping |

DAPT and TAPT are complementary — applying DAPT then TAPT then fine-tuning often yields the best results.

## Training Considerations

### Learning Rate
- Use **lower LR** than original pretraining (1e-5 to 5e-5)
- The model is already well-initialized; large updates destroy learned features

### Duration
- Typically 1-5 epochs over domain corpus
- Monitor perplexity on held-out domain text
- Diminishing returns after ~2-3 epochs

### Data Quality
- Remove duplicates (near-deduplication crucial at scale)
- Filter low-quality documents
- Domain relevance filtering (not all "medical" text is equally useful)

### Catastrophic Forgetting
- DAPT risks forgetting general capabilities
- Mitigations: low LR, short training, data mixing (5-10% general data)
- LoRA/QLoRA for DAPT limits forgetting significantly

## Files in This Module

| File | Description |
|------|-------------|
| `dapt_theory.py` | Domain shift analysis, vocabulary overlap, perplexity measurement, domain distance metrics |
| `dapt_from_scratch.py` | Implementing DAPT from scratch: masked LM pretraining, causal LM continuation, curriculum strategies |
| `dapt_training.py` | Production DAPT with HuggingFace: full pipeline, LoRA-based DAPT, data preparation |
| `dapt_comparison.py` | DAPT vs no-DAPT, domain distance effects, data scaling, decision framework |

## Key References

1. **Gururangan et al. (2020)** — *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks* (ACL 2020)
2. **Lee et al. (2020)** — *BioBERT: a pre-trained biomedical language representation model*
3. **Beltagy et al. (2019)** — *SciBERT: A Pretrained Language Model for Scientific Text*
4. **Chalkidis et al. (2020)** — *LEGAL-BERT: The Muppets straight out of Law School*
5. **Ke et al. (2023)** — *Continual Pre-training of Language Models*
6. **Gupta et al. (2023)** — *Continual Pre-Training of Large Language Models: How to (re)warm your model?*
