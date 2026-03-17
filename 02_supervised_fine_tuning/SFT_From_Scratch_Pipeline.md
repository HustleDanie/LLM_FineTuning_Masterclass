# End-to-End SFT From-Scratch Pipeline

## What is this file doing?

It's teaching a language model (like GPT-2) to **answer questions properly**. Out of the box, GPT-2 just predicts the next word — it doesn't know how to have a conversation. This script trains it to see a question and produce a good answer.

---

## Initialization (main)

```
╔══════════════════════════════════════════════════════════════════════╗
║                        INITIALIZATION (main)                        ║
╚══════════════════════════════════════════════════════════════════════╝
                                  │
           ┌──────────────────────┼──────────────────────┐
           ▼                      ▼                      ▼
   Load Tokenizer          Load Model (GPT-2)    Set pad_token = eos_token
  (AutoTokenizer)        (AutoModelForCausalLM)   (if missing)
           │                      │                      │
           └──────────────────────┼──────────────────────┘
                                  │
                                  ▼
              ┌───────────────────────────────────┐
              │   Raw Data: prompts[] + responses[]│
              │   8 QA pairs (6 train / 2 eval)   │
              └───────────────────┬───────────────┘
                                  │
                                  ▼
```

### Plain English

Three things happen before any training starts:

1. **Load the tokenizer** — This is the tool that converts human-readable text into numbers (tokens) that the model understands. Think of it like a dictionary that maps words/subwords to unique IDs.

2. **Load the model (DistilGPT-2)** — This is the pretrained language model. It already knows English grammar and general knowledge from being trained on internet text, but it doesn't know how to follow a question-answer format yet. That's what we're about to teach it.

3. **Fix the padding token** — GPT-2 was never designed for batched training with padding, so it doesn't have a padding token. We reuse the end-of-sentence token (`eos_token`) as the pad token. Without this, the tokenizer would crash when trying to pad sequences to the same length.

4. **Prepare the data** — 8 simple ML question-answer pairs are hardcoded. 6 go to training (the model learns from these), 2 go to evaluation (used to check if the model is improving, but never trained on).

---

## Phase 1 — Data Pipeline (`SFTDataset.__getitem__`)

**The problem:** You have questions and answers as raw text. The model needs numbers (tokens), not text. And crucially, you need to tell the model *"learn the answer part, but don't try to learn the question part."*

```
For EACH (prompt, response) pair:

 ┌─────────────────────────────────────────────────────────────────┐
 │ STEP 1: Template Formatting                                     │
 │                                                                  │
 │  Raw prompt: "What is machine learning?"                        │
 │  Raw response: "Machine learning is a branch of AI..."         │
 │                         │                                        │
 │                         ▼                                        │
 │  prompt_text  = "### Human: What is machine learning?\n         │
 │                  ### Assistant:"                                 │
 │  response_text = " Machine learning is a branch of AI..."      │
 │  full_text     = prompt_text + response_text                    │
 └─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ STEP 2: Separate Tokenization (to find the boundary)            │
 │                                                                  │
 │  prompt_tokens = tokenizer(prompt_text)  ──► [tok₁ tok₂ ... tokₚ]│
 │                                                 ▲                │
 │                                      prompt_length = P           │
 │                                                                  │
 │  full_tokens = tokenizer(full_text,                             │
 │                  truncation=True,                                │
 │                  max_length=256,                                 │
 │                  padding="max_length")                           │
 │                                                                  │
 │  ──► [tok₁ tok₂ ... tokₚ tokₚ₊₁ ... tokₙ PAD PAD ... PAD]     │
 │       ◄── prompt ──►◄── response ──►◄──── padding ────►        │
 │       (P tokens)     (R tokens)      (256 - P - R tokens)       │
 └─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ STEP 3: Loss Masking (THE KEY SFT OPERATION)                    │
 │                                                                  │
 │  input_ids:      [tok₁  tok₂ ... tokₚ  tokₚ₊₁ ... tokₙ  PAD ]│
 │                                                                  │
 │  labels BEFORE:  [tok₁  tok₂ ... tokₚ  tokₚ₊₁ ... tokₙ  PAD ]│
 │                   │                     │                  │    │
 │         mask prompt (=-100)    KEEP response tokens   mask pad  │
 │                   │                     │              (=-100)   │
 │                   ▼                     ▼                  ▼    │
 │  labels AFTER:   [-100  -100 ... -100  tokₚ₊₁ ... tokₙ  -100 ]│
 │                   ◄── IGNORED ──►◄── LOSS HERE ──►◄─IGNORED─►  │
 │                                                                  │
 │  WHY: CrossEntropyLoss(ignore_index=-100) skips -100 positions  │
 │       Model ONLY learns to predict response tokens              │
 └─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ STEP 4: Return Tensors                                          │
 │                                                                  │
 │  {                                                               │
 │    "input_ids":      tensor[256]   (all tokens)                 │
 │    "attention_mask":  tensor[256]  (1 for real, 0 for padding)  │
 │    "labels":         tensor[256]   (-100 for prompt/pad)        │
 │  }                                                               │
 └─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
                    DataLoader(batch_size=2, shuffle=True)
                               │
                               ▼
```

### Plain English

1. **Wraps the question in a template** — gives the model a consistent format so it knows where questions end and answers begin.
2. **Glues question + answer together** into one string.
3. **Converts text → numbers (tokenization)** — tokenizes the question alone first to count how many tokens it takes, then tokenizes the full string and pads to 256 tokens.
4. **Creates labels with masking** — question tokens get -100 (ignore), answer tokens keep real values (learn from these), padding tokens get -100 (ignore).

**Why mask the question?** You don't want the model to learn to *generate* questions. You want it to learn: *"when I see this question, I should produce this answer."*

---

## Phase 2 — Optimizer & Scheduler Setup

```
 ┌─────────────────────────────────────────────────────────────────┐
 │ Parameter Grouping for Weight Decay                              │
 │                                                                  │
 │  All named parameters                                            │
 │         │                                                        │
 │         ├──► Group 1: Regular params ──► weight_decay = 0.01    │
 │         │    (linear weights, embeddings, etc.)                  │
 │         │                                                        │
 │         └──► Group 2: Bias + LayerNorm ──► weight_decay = 0.0   │
 │              (these should NOT be decayed)                       │
 │                                                                  │
 └───────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
           AdamW(lr=3e-5, weight_decay applied per group)
                         │
                         ▼
     Linear Warmup + Linear Decay Scheduler
     │
     │  LR
     │  ▲   /\
     │  │  /  \
     │  │ / warmup \  linear decay
     │  │/    (5)    \_______________
     │  └──────────────────────────── ► steps
     │  0    5                  total
```

### Plain English

- **Optimizer (AdamW):** The algorithm that adjusts model weights. It applies "weight decay" (a small penalty to keep weights small) to most parameters, but **not** to bias terms and layer normalization — decaying those hurts performance.
- **Learning rate scheduler:** LR starts small (warmup — first 5 steps), then gradually decreases. Like: start gently, pick up speed, then slow down near the finish line.

---

## Phase 3 — Training Loop (`train_sft_from_scratch`)

```
 FOR each epoch (1 → 3):
 │
 │  FOR each batch (step) in DataLoader:
 │  │
 │  │  ┌──────────────────────────────────────────────────────────┐
 │  │  │ STEP 1: Move to Device                                   │
 │  │  │  input_ids      ──► GPU/CPU                              │
 │  │  │  attention_mask  ──► GPU/CPU                              │
 │  │  │  labels          ──► GPU/CPU                              │
 │  │  └──────────────────────────┬───────────────────────────────┘
 │  │                             │
 │  │                             ▼
 │  │  ┌──────────────────────────────────────────────────────────┐
 │  │  │ STEP 2: FORWARD PASS                                     │
 │  │  │                                                           │
 │  │  │  input_ids[batch, 256]                                    │
 │  │  │        │                                                  │
 │  │  │        ▼                                                  │
 │  │  │  ┌──────────────┐                                        │
 │  │  │  │  Embedding    │  token ids → dense vectors             │
 │  │  │  └──────┬───────┘                                        │
 │  │  │         ▼                                                 │
 │  │  │  ┌──────────────┐                                        │
 │  │  │  │  Transformer  │  N layers of self-attention + FFN      │
 │  │  │  │  Blocks       │  (causal mask: can only see past)      │
 │  │  │  └──────┬───────┘                                        │
 │  │  │         ▼                                                 │
 │  │  │  ┌──────────────┐                                        │
 │  │  │  │  LM Head      │  hidden_state → vocab logits           │
 │  │  │  └──────┬───────┘                                        │
 │  │  │         ▼                                                 │
 │  │  │  logits[batch, 256, vocab_size]                           │
 │  │  │                                                           │
 │  │  │  HuggingFace internally computes:                        │
 │  │  │    shift_logits = logits[..., :-1, :]  (predict next)    │
 │  │  │    shift_labels = labels[..., 1:]      (shifted target)  │
 │  │  │                                                           │
 │  │  │    loss = CrossEntropyLoss(                               │
 │  │  │             shift_logits,                                 │
 │  │  │             shift_labels,                                 │
 │  │  │             ignore_index=-100  ◄── skips prompt+pad       │
 │  │  │           )                                               │
 │  │  │                                                           │
 │  │  │  Position view:                                           │
 │  │  │  logits:  [p₁  p₂ ... pₚ  pₚ₊₁ ... pₙ₋₁]  predictions │
 │  │  │  labels:  [-100 ... -100   rₚ₊₁ ... rₙ  ]  targets      │
 │  │  │                            ◄── loss only here ──►         │
 │  │  └──────────────────────────┬───────────────────────────────┘
 │  │                             │
 │  │                             ▼
 │  │  ┌──────────────────────────────────────────────────────────┐
 │  │  │ STEP 3: BACKWARD PASS                                    │
 │  │  │                                                           │
 │  │  │  loss = loss / gradient_accumulation_steps (=2)           │
 │  │  │       │                                                   │
 │  │  │       ▼                                                   │
 │  │  │  loss.backward()                                          │
 │  │  │       │                                                   │
 │  │  │       ▼                                                   │
 │  │  │  Gradients flow: loss → LM Head → Transformer → Embedding│
 │  │  │                                                           │
 │  │  │  Key insight: Loss comes from response positions only,    │
 │  │  │  but gradients update ALL model parameters because        │
 │  │  │  the computation graph connects everything.               │
 │  │  │                                                           │
 │  │  │  param.grad += ∂loss/∂param  (accumulated, not replaced)  │
 │  │  └──────────────────────────┬───────────────────────────────┘
 │  │                             │
 │  │                             ▼
 │  │  ┌──────────────────────────────────────────────────────────┐
 │  │  │ STEP 4: OPTIMIZER STEP (every 2 accumulation steps)       │
 │  │  │                                                           │
 │  │  │  if (step + 1) % 2 == 0:                                  │
 │  │  │       │                                                   │
 │  │  │       ├──► clip_grad_norm_(max=1.0)                       │
 │  │  │       │    Rescale gradients if ‖g‖ > 1.0                 │
 │  │  │       │    Prevents catastrophic updates                  │
 │  │  │       │                                                   │
 │  │  │       ├──► optimizer.step()                               │
 │  │  │       │    θ = θ - lr × (grad + weight_decay × θ)        │
 │  │  │       │                                                   │
 │  │  │       ├──► scheduler.step()                               │
 │  │  │       │    Adjust LR (warmup → decay)                    │
 │  │  │       │                                                   │
 │  │  │       └──► optimizer.zero_grad()                          │
 │  │  │            Reset accumulated gradients to 0               │
 │  │  └──────────────────────────┬───────────────────────────────┘
 │  │                             │
 │  │                             ▼
 │  │  ┌──────────────────────────────────────────────────────────┐
 │  │  │ STEP 5: PERIODIC EVALUATION (every 5 optimizer steps)     │
 │  │  │                                                           │
 │  │  │  model.eval()                                             │
 │  │  │       │                                                   │
 │  │  │       ▼                                                   │
 │  │  │  with torch.no_grad():                                    │
 │  │  │    for eval_batch in eval_loader:                         │
 │  │  │      same forward pass, accumulate loss                   │
 │  │  │       │                                                   │
 │  │  │       ▼                                                   │
 │  │  │  eval_loss = total_loss / num_steps                       │
 │  │  │  perplexity = e^(eval_loss)                               │
 │  │  │       │                                                   │
 │  │  │       ├──► Track best_eval_loss                           │
 │  │  │       │                                                   │
 │  │  │       └──► model.train()  (resume training mode)          │
 │  │  └──────────────────────────────────────────────────────────┘
 │  │
 │  └──► next batch
 │
 └──► next epoch
```

### Plain English

For each batch of 2 examples:

1. **Forward pass** — Feed tokens into the model. It predicts the next token at each position and computes the **loss** (how wrong it was), but **only on answer positions** (question/padding labels are -100, so they're skipped).

2. **Backward pass** — PyTorch traces back through every calculation and computes gradients: *"how should I adjust each weight to make the loss smaller?"* Even though loss only came from answer tokens, gradients flow through the entire model because all layers were involved.

3. **Gradient accumulation** — Instead of updating after every batch of 2, it waits for 2 batches (effectively 4 examples) before updating. Simulates a larger batch size without needing more memory.

4. **Optimizer step (every 2 batches):**
   - **Gradient clipping:** If gradients are too large, scale them down (max magnitude = 1.0).
   - **Optimizer update:** Adjust all weights to reduce the loss.
   - **Scheduler update:** Adjust learning rate per the warmup/decay schedule.
   - **Zero gradients:** Clear accumulated gradients for the next round.

5. **Evaluation (every 5 optimizer steps):** Switch to eval mode, run eval examples without gradients, measure loss and **perplexity** (= e^loss — lower = better, model is less "confused").

---

## Phase 4 — Post-Training Inference

```
 ┌─────────────────────────────────────────────────────────────────┐
 │  model.eval()                                                    │
 │         │                                                        │
 │         ▼                                                        │
 │  For each test prompt:                                           │
 │                                                                  │
 │  "### Human: What is ML?\n### Assistant:"                       │
 │         │                                                        │
 │         ▼  tokenizer(prompt)                                     │
 │  input_ids ──► model.generate(                                   │
 │                  max_new_tokens=100,                              │
 │                  temperature=0.7,        ◄── sampling randomness │
 │                  do_sample=True,         ◄── stochastic decode   │
 │                  repetition_penalty=1.2  ◄── reduce repetition   │
 │                )                                                  │
 │         │                                                        │
 │         ▼  Autoregressive loop:                                  │
 │                                                                  │
 │    [prompt_toks] ──model──► logit ──sample──► tok₁               │
 │    [prompt_toks, tok₁] ──model──► logit ──sample──► tok₂        │
 │    [prompt_toks, tok₁, tok₂] ──model──► ...  (up to 100 tokens) │
 │         │                                                        │
 │         ▼  tokenizer.decode(output)                              │
 │  Final answer string                                             │
 └─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
```

### Plain English

After training, the model generates answers to new questions. It predicts one token at a time — each new token gets appended and fed back in to predict the next, up to 100 tokens. Temperature (0.7) controls randomness and repetition_penalty (1.2) stops the model from repeating itself.

---

## Phase 5 — Save

```
 ┌─────────────────────────────────────────────────────────────────┐
 │  model.save_pretrained("./results/sft_from_scratch/final")      │
 │    └──► Saves: config.json, pytorch_model.bin (all weights)     │
 │                                                                  │
 │  tokenizer.save_pretrained("./results/sft_from_scratch/final")  │
 │    └──► Saves: tokenizer.json, vocab, merges, special_tokens    │
 └─────────────────────────────────────────────────────────────────┘
```

### Plain English

Once training is done, you need to save everything to disk so you can use the model later without retraining:

- **`model.save_pretrained()`** — Saves two files: `config.json` (the model's architecture settings like number of layers, hidden size, etc.) and `pytorch_model.bin` (all the updated weight values — this is the actual "knowledge" the model learned).

- **`tokenizer.save_pretrained()`** — Saves the tokenizer files (vocabulary, merge rules, special token mappings). You need these to convert text to tokens the same way during inference as you did during training. If you save the model but not the tokenizer, you won't be able to use the model properly later.

The saved files are a complete package — anyone can load them with `AutoModelForCausalLM.from_pretrained("./results/sft_from_scratch/final")` and start generating answers immediately.

---

## Evaluation Function (`evaluate_sft`)

Simple function:
- Turn off gradient computation (we're only measuring, not learning).
- Run every eval example through the model.
- Average the loss across all examples.
- Return that average loss.

---

## Summary: The Core SFT Idea in One Flow

```
Raw (prompt, response)
        │
        ▼
  Template format ──► "### Human: {prompt}\n### Assistant: {response}"
        │
        ▼
  Tokenize ──► [tok₁ ... tokₚ | tokₚ₊₁ ... tokₙ | PAD ... PAD]
        │           prompt         response           padding
        ▼
  Mask labels ──► [-100 ... -100 | tokₚ₊₁ ... tokₙ | -100 ... -100]
        │          NO gradient       GRADIENT HERE      NO gradient
        ▼
  Forward ──► logits ──► CrossEntropy(only on response positions)
        │
        ▼
  Backward ──► gradients through ALL params but sourced ONLY from response loss
        │
        ▼
  Optimizer ──► update weights ──► model learns to generate good responses
```

> **One-sentence summary:** You take a pretrained language model, show it question-answer pairs where **only the answer part counts as a learning signal**, and after enough repetitions, the model learns to generate good answers when given new questions.
