# 02 — Supervised Fine-Tuning (SFT)

## What Is Supervised Fine-Tuning?

Supervised Fine-Tuning (SFT) is the **most common method** for creating instruction-following
and chat models. Unlike basic full fine-tuning on raw text, SFT trains the model on
**prompt → ideal response** pairs, teaching it to generate helpful, structured outputs.

```
┌─────────────────────────────────┐
│   Pretrained Language Model     │  (knows language, lacks task behavior)
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│   Prompt-Response Pair Dataset  │
│                                 │
│   Prompt: "Explain transformers"│
│   Response: "Transformers are..."│
│                                 │
│   Prompt: "Write a poem about X"│
│   Response: "Roses are red..."  │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│   SFT Training Loop             │
│   - Model sees prompt           │
│   - Generates response tokens   │
│   - Loss computed ONLY on       │
│     response tokens (masked)    │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│   Instruction-Following Model   │  (ChatGPT-like behavior)
└─────────────────────────────────┘
```

## SFT vs Full Fine-Tuning

| Aspect | Full Fine-Tuning | Supervised Fine-Tuning |
|--------|-----------------|----------------------|
| **Data Format** | Raw text | Prompt → Response pairs |
| **Objective** | Next token prediction on all tokens | Next token prediction on **response tokens only** |
| **Goal** | Domain adaptation | Behavior/instruction following |
| **Loss Masking** | No masking | Prompt tokens masked from loss |
| **Used For** | Domain-specific LLMs | Chat models, assistants |
| **Example** | Train on medical papers | Train on Q&A conversations |

## Why SFT Is So Important

SFT is the **critical bridge** between a raw pretrained model and a useful assistant:

1. **Pretrained model** → Predicts next token (can't follow instructions)
2. **After SFT** → Follows instructions, answers questions, generates structured output
3. **After RLHF/DPO** → Aligned with human preferences (helpful, harmless, honest)

Almost every chat model (ChatGPT, Claude, LLaMA-Chat, Mistral-Instruct) goes through SFT.

## Key Concept: Response-Only Loss Masking

The most important technique in SFT is **masking the loss on prompt tokens**:

```
Tokens:    [USER] What is AI? [ASSISTANT] AI is...
Labels:    [ -100  -100 -100    -100        AI  is... ]
                                              ↑
                                    Loss computed here only!
```

Why? We don't want to teach the model to generate the user's prompt.
We only want it to learn how to generate good responses.

## Files in This Module

| File | Description |
|------|-------------|
| `sft_training.py` | Complete SFT pipeline using TRL's SFTTrainer |
| `sft_from_scratch.py` | SFT implemented from scratch (for deep understanding) |
| `data_formatting.py` | Dataset formatting: chat templates, prompt-response pairs |
| `loss_masking.py` | Deep dive into response-only loss masking |
| `conversation_templates.py` | Chat templates (ChatML, Alpaca, Vicuna, LLaMA, etc.) |
| `evaluation_sft.py` | SFT-specific evaluation: response quality, formatting, diversity |

## Key Techniques Demonstrated

- [x] Prompt-response pair formatting
- [x] Response-only loss masking (the core technique)
- [x] Chat template application (ChatML, Alpaca, etc.)
- [x] Using TRL's SFTTrainer (industry standard)
- [x] Manual SFT implementation (educational)
- [x] Multi-turn conversation handling
- [x] System prompts and role-based formatting
- [x] Packing: fitting multiple short examples into one sequence
- [x] Dataset deduplication and quality filtering
- [x] Response quality evaluation

## Run

```bash
# Using TRL SFTTrainer (recommended)
python sft_training.py

# From-scratch implementation (educational)
python sft_from_scratch.py
```
