# End-to-End Instruction Fine-Tuning Pipeline

## Overview

```
╔══════════════════════════════════════════════════════════════════════════╗
║              INSTRUCTION FINE-TUNING PIPELINE (main entry)              ║
╚══════════════════════════════════════════════════════════════════════════╝
                                   │
                                   ▼
                    InstructionTuningConfig (dataclass)
                    ┌──────────────────────────────────┐
                    │ model_name = "distilgpt2"        │
                    │ template   = "alpaca"             │
                    │ mask_instruction = True           │
                    │ max_examples = 200                │
                    │ lr = 2e-5, epochs = 3             │
                    │ batch_size = 4, grad_accum = 4    │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
                    train_instruction_model(config)
                                   │
                                   ▼
```

---

## STEP 1 — Load Model & Tokenizer

```
 ┌─────────────────────────────────────────────────────────────────┐
 │  AutoTokenizer.from_pretrained("distilgpt2")                    │
 │  AutoModelForCausalLM.from_pretrained("distilgpt2")             │
 │         │                                                        │
 │         ▼                                                        │
 │  if no pad_token → set pad_token = eos_token                    │
 │         │                                                        │
 │         ▼                                                        │
 │  Print param count (total vs trainable)                         │
 └─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
```

### Plain English

Same as SFT — load a small pretrained model. It knows English but doesn't know how to follow instructions yet. Fix the missing pad token.

---

## STEP 2 — Data Preparation (`prepare_instruction_data`)

This is a 6-sub-step pipeline all on its own:

```
 ┌─────────────────────────────────────────────────────────────────┐
 │ [1/6] LOAD RAW DATA                                             │
 │                                                                  │
 │  if source == "local":                                           │
 │    create_instruction_dataset()  ──► list of dicts              │
 │  else:                                                           │
 │    load_instruction_dataset("dolly" or "alpaca")                │
 │                                                                  │
 │  Each example looks like:                                        │
 │  {                                                               │
 │    "instruction": "Explain what a neural network is",           │
 │    "input":       "",           ◄── optional extra context       │
 │    "output":      "A neural network is...",                     │
 │    "category":    "open_qa"     ◄── task type label              │
 │  }                                                               │
 │                                                                  │
 │  Shuffle + cap at max_examples (200)                            │
 └─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ [2/6] QUALITY FILTER                                            │
 │                                                                  │
 │  filter_instruction_quality(dataset)                            │
 │         │                                                        │
 │         ▼                                                        │
 │  Removes bad examples:                                           │
 │    • Instructions that are too short/empty                      │
 │    • Outputs that are too short/empty                           │
 │    • Duplicates or near-duplicates                              │
 │                                                                  │
 │  e.g., 200 examples → 185 examples                             │
 └─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ [3/6] BALANCE CATEGORIES                                        │
 │                                                                  │
 │  balance_dataset_by_category(dataset)                           │
 │         │                                                        │
 │         ▼                                                        │
 │  BEFORE:  80 open_qa, 5 summarization, 60 creative, 40 classify │
 │           (model would mostly learn Q&A, neglect other tasks)   │
 │                                                                  │
 │  AFTER:   40 open_qa, 40 summarization, 40 creative, 40 classify│
 │           (even exposure to all task types)                      │
 │                                                                  │
 │  WHY: If one category dominates, the model only gets good at    │
 │       that task. Balancing = better generalization to ANY task.  │
 └─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ [4/6] FORMAT WITH TEMPLATE (Alpaca style)                       │
 │                                                                  │
 │  Raw example:                                                    │
 │    instruction: "Summarize the following"                       │
 │    input:       "Long article text..."                          │
 │    output:      "Concise summary..."                            │
 │         │                                                        │
 │         ▼  format_alpaca(example)                                │
 │                                                                  │
 │  Formatted text:                                                 │
 │  ┌──────────────────────────────────────────────┐               │
 │  │ Below is an instruction that describes a task│               │
 │  │ along with an input that provides context.   │               │
 │  │ Write a response that completes the request. │               │
 │  │                                              │               │
 │  │ ### Instruction:                             │  ◄── PROMPT   │
 │  │ Summarize the following                      │      (masked) │
 │  │                                              │               │
 │  │ ### Input:                                   │               │
 │  │ Long article text...                         │               │
 │  │                                              │               │
 │  │ ### Response:                                │               │
 │  │ Concise summary...                           │  ◄── OUTPUT   │
 │  └──────────────────────────────────────────────┘    (trained)  │
 └─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ [5/6] TOKENIZE + LOSS MASKING                                   │
 │                                                                  │
 │  For each example:                                               │
 │                                                                  │
 │  1. Find where "### Response:" appears in the text              │
 │     (this is the response_marker boundary)                      │
 │                                                                  │
 │  2. Tokenize the full text                                       │
 │                                                                  │
 │  3. Tokenize just the instruction part (up to and including     │
 │     "### Response:") to count its token length                  │
 │                                                                  │
 │  4. Mask labels:                                                 │
 │                                                                  │
 │  tokens: [Below is... Instruction: ... Input: ... Response: | answer tokens]
 │  labels: [-100  -100  -100  -100  -100  -100  -100  -100   | real token ids]
 │           ◄────────── INSTRUCTION (ignored) ──────────────►  ◄─ OUTPUT ─►
 │                                                                  │
 │  Same idea as SFT — only the OUTPUT part produces loss          │
 └─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ [6/6] TRAIN / EVAL SPLIT                                        │
 │                                                                  │
 │  90% → train_dataset                                             │
 │  10% → eval_dataset                                              │
 └─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
```

### Plain English

This is the big difference from basic SFT. Instead of simple prompt→response pairs, you have **structured instruction data** with an explicit `instruction`, optional `input`, and `output`. Two extra steps happen that SFT doesn't do:
- **Quality filtering** — removes garbage examples before training.
- **Category balancing** — ensures the model sees equal amounts of summarization, QA, creative writing, classification, etc. Without this, the model would overfit to whatever task appears most in the data.

The template (Alpaca format) and loss masking work the same as SFT — instruction tokens get -100, output tokens get trained.

---

## STEP 3 — Data Collator (`InstructionDataCollator`)

```
 ┌─────────────────────────────────────────────────────────────────┐
 │  DataLoader pulls a batch of examples                           │
 │  (each has DIFFERENT lengths after tokenization)                │
 │                                                                  │
 │  Example A: [tok tok tok tok tok]          (5 tokens)           │
 │  Example B: [tok tok tok tok tok tok tok]  (7 tokens)           │
 │         │                                                        │
 │         ▼  InstructionDataCollator pads to batch max length      │
 │                                                                  │
 │  Example A: [tok tok tok tok tok PAD PAD]                       │
 │  Example B: [tok tok tok tok tok tok tok]                       │
 │                                                                  │
 │  input_ids pad with:      pad_token_id                          │
 │  attention_mask pad with: 0                                      │
 │  labels pad with:         -100 (ignored in loss)                │
 └─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
```

### Plain English

Unlike `sft_from_scratch.py` which pre-padded everything to a fixed 256 tokens, this script uses **dynamic padding** — each batch is only padded to the length of the longest example in that batch. This wastes less compute on padding tokens. The collator handles this on-the-fly per batch.

---

## STEP 4 — Training (HuggingFace `Trainer`)

```
 ┌─────────────────────────────────────────────────────────────────┐
 │  TrainingArguments(                                              │
 │    epochs=3, batch=4, grad_accum=4,    effective batch = 16     │
 │    lr=2e-5, scheduler=cosine,                                   │
 │    warmup_ratio=0.05, weight_decay=0.01,                        │
 │    load_best_model_at_end=True         ◄── early stopping-like  │
 │  )                                                               │
 │         │                                                        │
 │         ▼                                                        │
 │  Trainer(                                                        │
 │    model, training_args,                                         │
 │    train_dataset, eval_dataset,                                  │
 │    data_collator=InstructionDataCollator                        │
 │  )                                                               │
 │         │                                                        │
 │         ▼                                                        │
 │  trainer.train()                                                 │
 │         │                                                        │
 │         ▼    (inside Trainer, this happens automatically)        │
 │                                                                  │
 │  FOR each epoch:                                                 │
 │  │  FOR each batch:                                              │
 │  │  │                                                            │
 │  │  │   forward pass ──► logits ──► CrossEntropyLoss             │
 │  │  │        (loss only on output tokens, -100 ignored)         │
 │  │  │              │                                             │
 │  │  │              ▼                                             │
 │  │  │   backward pass ──► gradients                              │
 │  │  │              │                                             │
 │  │  │              ▼  (every 4 batches)                          │
 │  │  │   clip gradients + optimizer.step() + scheduler.step()    │
 │  │  │                                                            │
 │  │  END                                                          │
 │  │                                                               │
 │  │  eval: compute eval_loss + perplexity on eval_dataset        │
 │  │  save checkpoint if best eval_loss so far                    │
 │  END                                                             │
 │         │                                                        │
 │         ▼                                                        │
 │  Load best checkpoint (lowest eval_loss across all epochs)      │
 └─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
```

### Plain English

This is the same forward→backward→optimize loop as `sft_from_scratch.py`, but you don't write it manually. The HuggingFace `Trainer` does it all for you. The key differences:
- **Cosine LR scheduler** (smooth curve) instead of linear decay.
- **`load_best_model_at_end=True`** — after training, it automatically reverts to the checkpoint with the lowest eval loss, so you don't accidentally use an overfitted version.
- Evaluates at the **end of each epoch** (not every N steps).

---

## STEP 5 — Evaluation & Test Generation

```
 ┌─────────────────────────────────────────────────────────────────┐
 │  [5a] trainer.evaluate()                                        │
 │         │                                                        │
 │         ▼                                                        │
 │  eval_loss ──► perplexity = e^(eval_loss)                       │
 │                                                                  │
 │  [5b] Generate responses to 3 test instructions:                │
 │                                                                  │
 │  "Explain what a neural network is in simple terms."            │
 │  "Write a haiku about programming."                             │
 │  "List three benefits of open-source software."                 │
 │         │                                                        │
 │         ▼  format_alpaca(instruction) → prompt text              │
 │         ▼  tokenize(prompt) → input_ids                          │
 │         ▼  model.generate(                                       │
 │               max_new_tokens=256,                                │
 │               temperature=0.7,                                   │
 │               top_p=0.9           ◄── nucleus sampling           │
 │            )                                                     │
 │         │                                                        │
 │         ▼  Decode ONLY the new tokens (strip the prompt)         │
 │         ▼  Print instruction + response                          │
 └─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
```

### Plain English

Two kinds of evaluation happen. First, the numeric eval — run the held-out examples through the model and compute loss/perplexity (how "confused" the model is). Second, a **qualitative check** — give it 3 diverse test instructions and see what it actually generates. This lets you visually inspect if it's following instructions properly.

Note: it decodes **only the generated tokens** (strips the prompt), unlike the SFT script which decoded everything.

---

## STEP 6 — Save

```
 ┌─────────────────────────────────────────────────────────────────┐
 │  trainer.save_model()                                            │
 │    └──► config.json + pytorch_model.bin                         │
 │                                                                  │
 │  tokenizer.save_pretrained()                                    │
 │    └──► tokenizer files                                          │
 │                                                                  │
 │  Save instruction_tuning_config.json                            │
 │    └──► All hyperparameters as JSON (for reproducibility)       │
 │                                                                  │
 │  Output: ./instruction_tuned_distilgpt2/                        │
 └─────────────────────────────────────────────────────────────────┘
```

### Plain English

Saves everything: model weights, tokenizer, AND the training config as JSON. The config file is a nice touch — if someone loads your model later, they can see exactly what settings you used to train it.

---

## BONUS — Inference Utility (`run_instruction`)

```
 ┌─────────────────────────────────────────────────────────────────┐
 │  run_instruction("What is machine learning?", model, tokenizer) │
 │         │                                                        │
 │         ▼                                                        │
 │  Wrap in Alpaca template (without output section)               │
 │  "Below is an instruction...                                    │
 │   ### Instruction: What is machine learning?                    │
 │   ### Response:"                                                 │
 │         │                                                        │
 │         ▼  tokenize → generate → decode new tokens only         │
 │         │                                                        │
 │         ▼                                                        │
 │  return "Machine learning is..."                                │
 └─────────────────────────────────────────────────────────────────┘
```

### Plain English

A simple helper function for production use. Give it an instruction string, it wraps it in the Alpaca template (without the output part), runs generation, strips the prompt from the output, and returns just the answer. This is what you'd call in your app.

---

## How This Differs From `sft_from_scratch.py`

```
sft_from_scratch.py                    instruction_tuning.py
─────────────────────                  ─────────────────────────
Manual training loop                   HuggingFace Trainer (automatic)
Simple prompt→response pairs           Structured instruction/input/output
No quality filtering                   Quality filter removes bad examples
No category balancing                  Balances across task types
Fixed-length padding (256)             Dynamic padding per batch
Linear LR schedule                     Cosine LR schedule
No config saving                       Saves config as JSON
Tests 2 prompts after                  Tests 3 diverse instructions after
Educational (learn internals)          Production-ready approach
```

The core idea is identical — mask the input, train on the output. But instruction tuning adds **data quality and diversity** on top of the SFT technique, because the goal isn't just "answer questions" — it's "follow ANY instruction well."

---

## Where Instruction Fine-Tuning Is the ONLY Choice

### The ONE thing instruction fine-tuning does that nothing else can

**Teaching a model to follow arbitrary, never-before-seen instructions.** That's the unique superpower.

### Scenario: You need a model that handles ANY task a user throws at it

```
User might ask:
  "Summarize this article"
  "Write Python code for a linked list"
  "Translate this to Spanish"
  "Classify this email as spam or not"
  "Rewrite this paragraph in a formal tone"
  "Extract all dates from this text"

You DON'T know in advance what the user will ask.
```

### Why other methods fail here

```
Method                    Why it fails
──────────────────────────────────────────────────────────────────
Regular SFT               You trained on Q&A pairs. The model 
                          learned to answer questions. But when 
                          someone says "Translate this" or 
                          "Rewrite in formal tone" — it doesn't 
                          understand those are instructions to 
                          follow. It just tries to "answer" them 
                          like a question.

RAG                       Retrieval gives the model INFORMATION, 
                          not ABILITY. The model can retrieve a 
                          document about translation, but it still 
                          doesn't know HOW to translate on command.

Continued Pretraining     Gives the model knowledge of a domain, 
                          but doesn't teach it to DO things on 
                          command. A model pretrained on code 
                          repositories knows code, but won't 
                          reliably write code WHEN ASKED TO.

Prompt Engineering        Works for 1-2 specific tasks if you 
                          craft the prompt carefully. But you'd 
                          need a different prompt for every 
                          possible task — impossible when you 
                          don't know what users will ask.

RLHF/DPO                 These REFINE an already instruction-
                          following model. They can't teach 
                          instruction-following from scratch.
                          You need instruction FT FIRST, then RLHF.
```

### The concrete business scenario

```
Client: "We're building an internal AI assistant for our 500 employees. 
         Engineers will ask it to write code. 
         Marketing will ask it to draft emails. 
         Legal will ask it to summarize contracts. 
         HR will ask it to rewrite job descriptions.
         Finance will ask it to extract numbers from reports.
         
         We can't predict every possible request."

                      │
                      ▼

The ONLY training method: Instruction Fine-Tuning
                      │
                      ▼

WHY: You train on DIVERSE instruction types:
  - Summarization instructions
  - Code generation instructions  
  - Translation instructions
  - Classification instructions
  - Extraction instructions
  - Rewriting instructions
  - Creative writing instructions
  - ...hundreds of task categories
                      │
                      ▼

RESULT: The model learns the META-SKILL of 
        "read an instruction → do what it says"
        
        Now it can follow NEW instructions it 
        has never seen during training.
        
        This is called ZERO-SHOT GENERALIZATION.
```

### Why regular SFT can't do this

```
Regular SFT training data:
  Q: "What is machine learning?"  →  A: "Machine learning is..."
  Q: "Explain neural networks"    →  A: "Neural networks are..."
  Q: "How does attention work?"   →  A: "Attention computes..."

What the model learns: "When I see a question, give a factual answer."

What it CAN'T do after SFT:
  "Translate 'hello' to French"         → ??? (not a question)
  "Write a poem about the ocean"        → ??? (not a factual answer)
  "Classify this as positive/negative"  → ??? (never saw this format)
```

```
Instruction FT training data:
  Instruction: "Summarize this"      → Output: (summary)
  Instruction: "Translate to French" → Output: (translation)  
  Instruction: "Write a poem about X" → Output: (poem)
  Instruction: "Classify sentiment"  → Output: (classification)
  Instruction: "Extract all names"   → Output: (list of names)
  ... hundreds of different task TYPES

What the model learns: "Read the instruction. Do whatever it says."

What it CAN do after instruction FT:
  "Convert this CSV to JSON"  → (does it, even though never trained on this exact task)
  
  It GENERALIZES because it learned the CONCEPT of following instructions,
  not just the answers to specific questions.
```

### The one-liner

```
Regular SFT    = teaches a model to do ONE type of thing well
Instruction FT = teaches a model to do ANYTHING you tell it to

               The difference is GENERALIZATION.
```

Instruction fine-tuning is the **only** method that produces a general-purpose assistant from a base model. Every chatbot, every AI assistant, every "do anything" model (ChatGPT, Claude, LLaMA-Chat) went through instruction fine-tuning. There is no alternative path to get there.

---

## Where Instruction Tuning Is Mostly Applied

### 1. Building General-Purpose AI Assistants (the #1 use case)

This is where 90% of instruction tuning happens. Every major chatbot went through it:

```
Base model (LLaMA, Mistral, GPT, etc.)
        │
        ▼
   Instruction Tuning
        │
        ▼
   ChatGPT, Claude, Gemini, LLaMA-Chat, Mistral-Instruct
```

The model needs to handle *anything* — code, writing, math, translation, analysis — so it gets trained on thousands of diverse instruction types.

### 2. Open-Source "Instruct" Model Releases

Almost every open-weight model has an instruction-tuned variant:

```
LLaMA-3-8B         → LLaMA-3-8B-Instruct
Mistral-7B          → Mistral-7B-Instruct
Phi-3-mini          → Phi-3-mini-Instruct
Qwen-2-7B           → Qwen-2-7B-Instruct
Gemma-2-9B          → Gemma-2-9B-it (instruction-tuned)
```

Companies release the base model, then the community (or the company itself) applies instruction tuning to make it usable as an assistant.

### 3. Enterprise Internal Assistants

```
Company downloads open model
        │
        ▼
   Instruction tune on:
   • Internal documentation tasks
   • Company-specific workflows  
   • Multi-department use cases (legal, HR, engineering, finance)
        │
        ▼
   Internal "do anything" assistant deployed on-premise
```

### 4. Code Assistants

```
Base code model (CodeLLaMA, StarCoder, DeepSeek-Coder)
        │
        ▼
   Instruction tuned on:
   • "Write a function that..."
   • "Debug this code..."
   • "Explain this code..."
   • "Refactor this to..."
   • "Write tests for..."
        │
        ▼
   GitHub Copilot-like tools, coding chatbots
```

### 5. Multilingual / Cross-lingual Models

```
Instruction tune on the SAME task types but in multiple languages:
   • "Summarize this" (English)
   • "Résume ceci" (French)
   • "これを要約して" (Japanese)
   • "Fasse dies zusammen" (German)
        │
        ▼
   Model that follows instructions in ANY language
```

### Where you'll see it LEAST

| Area | Why instruction tuning isn't the go-to |
|---|---|
| **Single-task apps** (e.g., only sentiment analysis) | Regular SFT or even a classifier is simpler and cheaper |
| **Knowledge-heavy retrieval** (e.g., company FAQ bot) | RAG handles this without any training |
| **Domain adaptation** (e.g., learn medical vocabulary) | Continued pretraining (DAPT) is better for this |
| **Style/tone matching** (e.g., "sound like our brand") | Regular SFT on tone-specific examples is more targeted |

### Bottom line

Instruction tuning lives at the **foundation layer** of modern AI products. If you're building anything that needs to understand and execute varied user requests — that's instruction tuning. It's the reason we went from "GPT-3 that just autocompletes" to "ChatGPT that actually does what you ask."
