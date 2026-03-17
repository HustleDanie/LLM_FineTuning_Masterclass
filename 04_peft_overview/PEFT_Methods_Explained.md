# PEFT Methods — Plain-Language Explained

> Based on `peft_methods.py` — All 8 PEFT techniques broken down simply.

---

## The Core Idea

Big AI models have **billions of settings** (called parameters). But when you want the model to do a specific job — say, answer customer questions or summarize legal docs — you **don't actually need to change all those billions of settings**. The task you want lives in a much smaller "space" inside the model.

**PEFT (Parameter-Efficient Fine-Tuning)** is about finding clever shortcuts to adjust the model using only a tiny fraction of its settings, instead of retraining the whole thing.

Think of it like this: you have a **massive factory** with 10,000 machines. You don't need to rebuild the whole factory to make a new product — you just tweak a handful of machines. PEFT is the art of figuring out **which handful to tweak**.

---

## The 8 PEFT Methods

---

### 1. LoRA (Low-Rank Adaptation)

**The Analogy:**
Imagine you have a giant mixing board in a recording studio with thousands of knobs. Instead of touching every knob, someone gives you a **small remote control with just 16 sliders** that maps to combinations of the big knobs. You move those 16 sliders, and they adjust the big board for you.

**What it does:**
- Freezes (locks) all the original model weights
- Adds two small "side matrices" (A and B) next to certain layers
- Only trains those small matrices
- At inference time, these side matrices **merge back** into the original weights — so there's **zero slowdown** when the model is running

**Key details:**
- Trains **0.1–1%** of parameters
- No inference overhead (weights merge)
- The "rank" (r) controls how many sliders you get — typically 4 to 64

```
Original model:  y = W · x
With LoRA:       y = W · x + (B · A) · x · (α/r)
                     ─────   ─────────────────────
                     frozen       trained (tiny)
```

**Best for:** General-purpose fine-tuning when you want efficiency without sacrificing much quality.

---

### 2. QLoRA (Quantized LoRA)

**The Analogy:**
Take the LoRA approach above, but first **compress the entire factory down** so it takes up way less space. The factory still works, it's just stored more efficiently. Then apply the same small remote control (LoRA) on top.

**What it does:**
- Compresses (quantizes) the base model to **4-bit** precision (instead of 16-bit)
- Applies regular LoRA adapters on top (kept at 16-bit)
- Uses clever tricks: NF4 data format, double quantization, paged optimizers

**Why it matters:**
- A 65-billion-parameter model that normally needs **multiple GPUs** can now fine-tune on **a single 48GB GPU**
- This is what made fine-tuning large models accessible to regular people and small companies

**Key details:**
- Trains **0.1–1%** of parameters
- Slight inference overhead (due to quantized base)
- Slightly slower training (compression/decompression steps)

**Best for:** When you want LoRA but **don't have enough GPU memory** for the full model.

---

### 3. IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)

**The Analogy:**
Imagine the model is a band playing music. IA³ doesn't add new instruments or change the sheet music — it just gives the sound engineer a **set of volume knobs**, one per instrument. Turn some up, turn some down. That's it.

**What it does:**
- Learns a simple **scaling vector** (a list of numbers) for three spots in each transformer layer:
  - Keys (in attention)
  - Values (in attention)
  - Feed-forward output
- Each number in the vector just **multiplies** the existing activation — making it louder or softer

**Key details:**
- Trains **less than 0.01%** of parameters (the fewest of ANY method)
- Negligible inference overhead
- Great for few-shot learning (learning from very few examples)

```
Normal:    k = W_k · x
With IA³:  k = l_k ⊙ (W_k · x)
               ───
               learned scaling vector (just multiplies)
```

**Best for:** When you have **very little data** or want the absolute lightest-weight fine-tuning possible.

---

### 4. Adapters (Bottleneck Adapters)

**The Analogy:**
Think of the model as an **assembly line** in a factory. Adapters are like inserting **small custom workstations** between the existing machines. Each workstation takes the product, makes a small modification, and passes it along. The original machines stay untouched.

**What it does:**
- Inserts small "bottleneck" layers **inside** each transformer block
- Each adapter: takes input → shrinks it down → applies a function → expands it back → adds it to the original
- Can be placed after self-attention, after the feed-forward layer, or both

**Key details:**
- Trains **1–5%** of parameters
- **Does add inference latency** (extra layers = extra computation at runtime)
- Very modular — you can mix and match adapters for different tasks

```
Original flow:   Attention → Output
With Adapters:   Attention → [Adapter: shrink → ReLU → expand → + residual] → Output
```

**Best for:** When you need **high modularity** — e.g., one base model serving multiple tasks, each with its own adapter that can be swapped in/out.

---

### 5. Prefix Tuning

**The Analogy:**
Imagine you're giving a presentation, and at **every slide**, there's an invisible cue card that only you can see, telling you how to bias your delivery. These cue cards aren't part of the actual content — they just steer how you process everything on that slide.

**What it does:**
- Prepends learnable "virtual tokens" to the **keys and values** at **every attention layer**
- The model "sees" these extra tokens and they influence how it processes the real input
- Uses a small MLP to generate the prefix vectors (for training stability)

**Key details:**
- Trains **0.1–1%** of parameters
- Slightly reduces effective context length (prefix tokens take up space)
- Can mix multiple prefixes for multi-task scenarios

```
Normal attention:     K = [real keys]
With Prefix Tuning:   K = [PREFIX keys | real keys]   ← prefix is learned
                      V = [PREFIX values | real values]
(at EVERY layer)
```

**Best for:** **Generation tasks** (summarization, translation) where you want to steer the model's behavior across all layers without modifying any weights.

---

### 6. Prompt Tuning

**The Analogy:**
Same as prefix tuning, but simpler — you only get **one set of cue cards at the very beginning**, not at every layer. You put a few invisible "hint words" in front of the input, and the model figures out what to do from there.

**What it does:**
- Prepends learnable "soft tokens" to the **input embedding only** (first layer)
- The rest of the model is completely frozen
- Typically uses 10–100 virtual tokens

**Key finding from the research:**
As models get bigger, prompt tuning gets more effective. At **10 billion+ parameters**, prompt tuning nearly matches full fine-tuning.

**Key details:**
- Trains **less than 0.1%** of parameters
- Minimal inference overhead (just a few extra tokens)
- The simplest PEFT method to implement

```
Normal input:         [token₁, token₂, ..., tokenₙ]
With Prompt Tuning:   [SOFT₁, SOFT₂, ..., SOFTₖ, token₁, token₂, ..., tokenₙ]
                       ─────────────────────────
                       learned (these aren't real words)
```

**Best for:** **Very large models** where you want the absolute simplest fine-tuning approach with minimal parameters.

---

### 7. P-Tuning v2

**The Analogy:**
Think of it as **Prefix Tuning's bigger sibling** designed for understanding tasks (not just generation). It puts learnable cue cards at every layer (like prefix tuning) but drops some of the complexity and adds task-specific output heads.

**What it does:**
- Adds learnable continuous prompts at **every layer** (like prefix tuning)
- Removes the MLP reparameterization trick (simpler)
- Can place prompts at different positions (not just the front)
- Designed to work well even on **smaller models**

**Key finding:**
Deep prompt tuning (prompts at all layers) is critical for matching full fine-tuning performance on smaller models.

**Key details:**
- Trains **0.1–1%** of parameters
- Prompt tokens at every layer
- Works across all model scales (unlike basic prompt tuning)

**Best for:** **NLU tasks** (classification, entity recognition, question answering) where you need prefix tuning's power but with better stability and small-model support.

---

### 8. BitFit (Bias-terms Fine-Tuning)

**The Analogy:**
Imagine every machine on the assembly line has a **small dial** on it. The machine itself is locked — you can't rebuild it. But that little dial? You can turn it. BitFit says: **only turn the dials, don't touch anything else.**

**What it does:**
- Freezes **every weight matrix** in the model
- Only trains the **bias terms** (the small additive constants in each layer)
- Also trains LayerNorm parameters (γ and β)

**Why it's surprising:**
Bias terms are less than 0.1% of a model's parameters, yet training only these can produce **surprisingly good results** — especially for classification-type tasks.

**Key details:**
- Trains **~0.1%** of parameters
- Zero inference overhead (no architecture changes)
- The simplest method to implement (just freeze everything except biases)

```
Normal layer:   y = W · x + b     ← W and b both trained
BitFit:         y = W · x + b     ← W frozen, only b trained
```

**Best for:** Quick experiments and **understanding (NLU) tasks** when you want the simplest possible approach. Less effective for generation tasks.

---

## Comparison Table

| Method | What Gets Trained | % of Params | Inference Speed | Best Use Case |
|--------|-------------------|-------------|-----------------|---------------|
| **LoRA** | Side matrices (A, B) | 0.1–1% | Same as original (merge) | General fine-tuning |
| **QLoRA** | Side matrices + 4-bit base | 0.1–1% | Slightly slower | Large models, limited GPU |
| **IA³** | Scaling vectors | <0.01% | Nearly same | Few-shot, minimal changes |
| **Adapters** | Bottleneck layers | 1–5% | Slower (extra layers) | Multi-task modularity |
| **Prefix Tuning** | Virtual K/V at every layer | 0.1–1% | Slightly slower | Generation tasks |
| **Prompt Tuning** | Soft tokens at input | <0.1% | Nearly same | Very large models |
| **P-Tuning v2** | Deep prompts at every layer | 0.1–1% | Slightly slower | NLU tasks, smaller models |
| **BitFit** | Only bias terms | ~0.1% | Same as original | Quick NLU experiments |

---

## The Three Categories

| Category | Methods | Strategy |
|----------|---------|----------|
| **Reparameterization** | LoRA, QLoRA, IA³ | Rewrite the weight update in a cheaper form |
| **Additive** | Adapters, Prefix Tuning, Prompt Tuning, P-Tuning v2 | Add new small trainable components to the model |
| **Selective** | BitFit | Pick a subset of existing parameters to train |

---

## Quick Decision Guide

```
Need to fine-tune a model?
│
├── How big is your model?
│   ├── Very large (10B+) and limited GPU → QLoRA
│   ├── Large but have decent GPU → LoRA
│   └── Small/medium → LoRA or Adapters
│
├── How much data do you have?
│   ├── Very little (few-shot) → IA³ or Prompt Tuning
│   └── Enough data → LoRA
│
├── What's your task?
│   ├── Generation (writing, translation) → LoRA, Prefix Tuning
│   ├── Understanding (classification, NER) → LoRA, P-Tuning v2, BitFit
│   └── Multi-task serving → Adapters
│
└── Want simplest possible approach?
    ├── Yes → BitFit (NLU) or Prompt Tuning (generation)
    └── No, want best results → LoRA or QLoRA
```

---

## Architecture Diagrams

### Full Fine-Tuning (Baseline)
```
┌─────────────────────────────────────────────┐
│         FULL FINE-TUNING                     │
│         (All parameters updated)             │
│                                              │
│  Input ──→ [Self-Attention] ──→ [Add+Norm]  │
│             ████ ALL UPDATED ████            │
│                     │                        │
│             [Feed-Forward]   ──→ [Add+Norm]  │
│             ████ ALL UPDATED ████            │
│                     │                        │
│                  Output                      │
└─────────────────────────────────────────────┘
Memory: 100% | Trainable params: 100%
```

### LoRA
```
┌─────────────────────────────────────────────┐
│         LoRA                                 │
│         (Low-rank updates to Q, V)           │
│                                              │
│  Input ──→ [Self-Attention]─→ [Add+Norm]    │
│             W_q ── frozen                    │
│              └─ B_q·A_q ── TRAINED (rank r)  │
│             W_v ── frozen                    │
│              └─ B_v·A_v ── TRAINED (rank r)  │
│                     │                        │
│             [Feed-Forward] ──→ [Add+Norm]    │
│             ████ FROZEN ████                 │
│                     │                        │
│                  Output                      │
│                                              │
│  h = W·x + (B·A)·x · (α/r)                 │
└─────────────────────────────────────────────┘
Memory: ~30% of Full FT | Trainable: 0.1-1%
```

### Adapters
```
┌─────────────────────────────────────────────┐
│         ADAPTERS                             │
│         (Bottleneck layers inserted)         │
│                                              │
│  Input ──→ [Self-Attention] ──→ [Add+Norm]  │
│             ████ FROZEN ████                 │
│                     │                        │
│             ┌───────────────┐                │
│             │ ADAPTER (NEW) │                │
│             │ Down: d → r   │  ◄── TRAINED  │
│             │ ReLU          │                │
│             │ Up:   r → d   │                │
│             │ + Residual    │                │
│             └───────────────┘                │
│                     │                        │
│             [Feed-Forward] ──→ [Add+Norm]    │
│             ████ FROZEN ████                 │
│                     │                        │
│             ┌───────────────┐                │
│             │ ADAPTER (NEW) │  ◄── TRAINED  │
│             └───────────────┘                │
│                  Output                      │
└─────────────────────────────────────────────┘
Memory: ~40% of Full FT | Trainable: 1-5%
```

### Prefix Tuning
```
┌─────────────────────────────────────────────┐
│         PREFIX TUNING                        │
│         (Learnable prefixes at every layer)  │
│                                              │
│  Layer 1:                                    │
│    K = [P_k¹ | K₁]     ◄── P_k¹ TRAINED    │
│    V = [P_v¹ | V₁]     ◄── P_v¹ TRAINED    │
│    [Self-Attention with extended K,V]        │
│    ████ FROZEN ████                          │
│                                              │
│  Layer 2:                                    │
│    K = [P_k² | K₂]     ◄── P_k² TRAINED    │
│    V = [P_v² | V₂]     ◄── P_v² TRAINED    │
│    ...                                       │
│                                              │
│  Each layer gets its own prefix vectors      │
└─────────────────────────────────────────────┘
Memory: ~35% of Full FT | Trainable: 0.1%
```

### Prompt Tuning
```
┌─────────────────────────────────────────────┐
│         PROMPT TUNING                        │
│         (Soft tokens at input ONLY)          │
│                                              │
│  [P₁ P₂ ... Pₙ | x₁ x₂ ... xₘ]           │
│   ^^^^^^^^^^^    ^^^^^^^^^^^^^^^             │
│   TRAINABLE      REAL INPUT                  │
│   (soft tokens)  (frozen embeddings)         │
│          │                                   │
│          ▼                                   │
│   [Transformer Layers — ALL FROZEN]          │
│          │                                   │
│          ▼                                   │
│       Output                                 │
└─────────────────────────────────────────────┘
Memory: ~30% of Full FT | Trainable: <0.01%
```

### BitFit
```
┌─────────────────────────────────────────────┐
│         BITFIT                                │
│         (Only bias terms trained)            │
│                                              │
│  [Self-Attention]                            │
│    W_q · x + b_q    W_q FROZEN, b_q TRAINED │
│    W_k · x + b_k    W_k FROZEN, b_k TRAINED │
│    W_v · x + b_v    W_v FROZEN, b_v TRAINED │
│                                              │
│  [LayerNorm]         γ, β — TRAINED          │
│                                              │
│  [Feed-Forward]                              │
│    W₁ · x + b₁     W₁ FROZEN, b₁ TRAINED   │
│    W₂ · x + b₂     W₂ FROZEN, b₂ TRAINED   │
│                                              │
│  [LayerNorm]         γ, β — TRAINED          │
└─────────────────────────────────────────────┘
Memory: ~30% of Full FT | Trainable: ~0.1%
```

---

## Parameter Comparison (GPT-2 Scale: d=768, 12 layers)

| Method | Trainable Params | % of Total |
|--------|----------------:|----------:|
| Full Fine-Tuning | 84,934,656 | 100.0% |
| LoRA (r=16, all linear) | 1,769,472 | 2.08% |
| Adapters (r=64) | 1,218,816 | 1.43% |
| LoRA (r=16, Q+V only) | 589,824 | 0.69% |
| Prefix Tuning (len=20) | 368,640 | 0.43% |
| BitFit | 92,160 | 0.11% |
| IA³ | 27,648 | 0.03% |
| Prompt Tuning (20 tokens) | 15,360 | 0.02% |
