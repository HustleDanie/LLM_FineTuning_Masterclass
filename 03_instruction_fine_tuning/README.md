# 03 — Instruction Fine-Tuning

## What Is Instruction Fine-Tuning?

Instruction fine-tuning is a **specialized form of SFT** where the model learns
**instruction-following behavior** using structured datasets with explicit
`instruction`, `input`, and `output` fields.

While SFT broadly covers prompt→response training, instruction fine-tuning
specifically focuses on making models that can:
- Follow diverse instructions accurately
- Handle tasks with optional context/input
- Generalize to unseen instruction types
- Maintain consistent output formatting

```
┌─────────────────────────────────────────────────┐
│   Pretrained LLM (knows language)               │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│   Instruction Dataset                           │
│                                                 │
│   ┌─────────────────────────────────────────┐   │
│   │ instruction: "Summarize the following"  │   │
│   │ input:       "Long article text..."     │   │
│   │ output:      "Concise summary..."       │   │
│   └─────────────────────────────────────────┘   │
│                                                 │
│   ┌─────────────────────────────────────────┐   │
│   │ instruction: "Translate to French"      │   │
│   │ input:       "Hello world"              │   │
│   │ output:      "Bonjour le monde"         │   │
│   └─────────────────────────────────────────┘   │
│                                                 │
│   ┌─────────────────────────────────────────┐   │
│   │ instruction: "Write a haiku about rain" │   │
│   │ input:       "" (optional)              │   │
│   │ output:      "Soft drops on the roof.." │   │
│   └─────────────────────────────────────────┘   │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│   Instruction-Following Model                   │
│   (assistants, chat models, reasoning tasks)    │
└─────────────────────────────────────────────────┘
```

## Instruction FT vs General SFT

| Aspect | General SFT | Instruction Fine-Tuning |
|--------|------------|------------------------|
| **Data format** | prompt → response | instruction + input → output |
| **Focus** | Conversation style | Task completion |
| **Dataset diversity** | Can be narrow | Must cover many task types |
| **Key metric** | Response quality | Instruction adherence |
| **Examples** | ChatGPT conversations | FLAN, Alpaca, Dolly |
| **Generalization** | To similar conversations | To unseen instructions |

## Why Instruction Diversity Matters

The magic of instruction-tuned models is **zero-shot generalization**:
after training on diverse instructions, the model can follow NEW instructions
it has never seen. This requires:

1. **Task diversity** — Summarization, translation, QA, classification, creative writing, etc.
2. **Instruction diversity** — Same task phrased in many different ways
3. **Format diversity** — Lists, paragraphs, code, JSON, etc.
4. **Difficulty diversity** — Simple to complex reasoning

## Key Research Papers

- **FLAN** (Google, 2022): Showed instruction tuning dramatically improves zero-shot
- **Self-Instruct** (2023): Use LLMs to generate instruction data
- **Alpaca** (Stanford, 2023): 52K instructions from GPT-3.5
- **Dolly** (Databricks, 2023): 15K human-written instructions

## Files in This Module

| File | Description |
|------|-------------|
| `instruction_tuning.py` | Complete instruction fine-tuning pipeline |
| `instruction_datasets.py` | Dataset creation, loading, and formatting |
| `instruction_templates.py` | Instruction formatting strategies & prompt engineering |
| `task_categories.py` | Task taxonomy and category-balanced sampling |
| `self_instruct.py` | Self-Instruct: generating instruction data with LLMs |
| `evaluation_instruction.py` | Instruction-following evaluation suite |

## Key Techniques

- [x] Structured instruction/input/output format
- [x] Task-diverse dataset construction
- [x] Category-balanced training
- [x] Instruction augmentation (rephrasing)
- [x] Self-Instruct pipeline
- [x] Zero-shot generalization testing
- [x] Cross-task evaluation
- [x] Instruction complexity scaling

## Run

```bash
python instruction_tuning.py
python self_instruct.py
```
