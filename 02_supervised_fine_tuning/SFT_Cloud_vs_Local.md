# Where is SFT Used — Cloud LLMs vs Local LLMs

SFT is used in **both**, but in very different ways.

---

## Cloud LLMs (GPT-4, Claude, Gemini, etc.)

SFT is a **core part of how these models are built**. The pipeline is:

```
Pretrained base model (trillions of tokens of internet text)
        │
        ▼
   ┌─────────┐
   │   SFT   │  ◄── Trained on millions of human-written (instruction, response) pairs
   └────┬────┘      by large annotation teams. This is what makes the model
        │           "helpful" and able to follow instructions.
        ▼
   ┌─────────┐
   │  RLHF   │  ◄── Further aligned using human preference feedback
   └────┬────┘
        ▼
   Deployed as API (ChatGPT, Claude, etc.)
```

You **don't** do SFT yourself here. OpenAI/Anthropic/Google already did it. You just call their API. Some providers (OpenAI, Google) offer a "fine-tuning API" where you can do an additional round of SFT on top of their already-SFT'd model for your specific use case — but that runs on **their** cloud infrastructure.

---

## Local LLMs (LLaMA, Mistral, Phi, Qwen, etc.)

This is where **you** actually run SFT yourself. The typical scenario:

```
Download open-weight base model (e.g., LLaMA-3-8B)
        │
        ▼
   ┌─────────────────────┐
   │   SFT (you do this) │  ◄── On YOUR data, on YOUR hardware (or rented GPU)
   └──────────┬──────────┘      Using scripts like sft_from_scratch.py
              │                 or TRL's SFTTrainer
              ▼
   Model that follows YOUR specific instructions
   (runs locally, no API costs, full data privacy)
```

**Common local SFT use cases:**
- Company trains a model on internal docs/customer service data
- Researcher fine-tunes for a specific domain (medical, legal, code)
- Developer makes a small model (7B) behave like a specialized assistant
- Hobbyist fine-tunes for a particular style or persona

---

## The Key Difference

| | Cloud LLM | Local LLM |
|---|---|---|
| **Who does SFT?** | The provider (OpenAI, etc.) | **You** |
| **Data** | Their proprietary data | Your custom data |
| **Hardware** | Their massive GPU clusters | Your GPU / rented cloud GPU |
| **Cost** | Per-token API fees forever | One-time training cost, free inference |
| **Privacy** | Data sent to third party | Data stays on your machine |
| **Model size** | 100B+ params (hidden) | Usually 1B–70B (open weights) |

---

## What Does SFT Actually Do?

SFT does two distinct things — domain specialization is one, but not the full picture:

**1. Behavior alignment (the primary use)** — Teaching the model *how* to respond. A base model like LLaMA just predicts the next word — it doesn't know it should answer questions, follow instructions, or be helpful. SFT teaches it the format: "when you see a question, produce a helpful answer." This is why ChatGPT feels like an assistant while raw GPT-3 just rambles.

**2. Domain specialization (the secondary use)** — Teaching the model *what* to respond about. Fine-tuning on medical QA pairs, legal documents, or coding examples so it becomes better at that specific domain.

```
Base model (knows language, grammar, general knowledge)
     │
     ├──► SFT for behavior ──► "Be a helpful assistant"
     │                          (general-purpose, like ChatGPT)
     │
     ├──► SFT for domain ──► "Be a medical expert"
     │                        (specialized knowledge)
     │
     └──► SFT for both ──► "Be a helpful medical assistant"
                            (most common in practice)
```

In practice, most SFT does **both at the same time** — you're training on domain-specific instruction-response pairs, so the model learns the response format *and* the domain knowledge simultaneously.

But if you had to pick the defining purpose of SFT: it's **teaching the model to follow instructions and respond in a useful way**, not just autocomplete text. Domain specialization is a bonus that comes naturally when your training data is domain-specific.

---

## Limitations of SFT on Local LLMs

There's no hard rule that says "you can't SFT this model," but there are **practical limitations** that determine what's realistic.

### GPU Memory is the Main Bottleneck

SFT needs to hold in memory: the model weights + gradients + optimizer states + activations. That's roughly **4× the model size** in VRAM for full SFT.

```
Model Size    Full SFT VRAM Needed     Realistic GPU
─────────────────────────────────────────────────────
  1B params   ~  8 GB                  Consumer GPU (RTX 3060)
  3B params   ~ 24 GB                  RTX 3090 / 4090
  7B params   ~ 56 GB                  A100 (80GB) or 2× A6000
 13B params   ~104 GB                  2× A100 80GB
 70B params   ~560 GB                  8× A100 cluster
```

**So full SFT on a 70B model on a single consumer GPU? Impossible.**

### Workarounds That Extend Your Reach

This is exactly why all those PEFT methods in this repo exist:

```
Can't afford full SFT?
        │
        ├──► LoRA / QLoRA ──► Only trains ~1-2% of params
        │    (folders 05, 06)   7B model fits on a single 24GB GPU
        │
        ├──► Adapters ──► Adds small trainable modules
        │    (folder 07)
        │
        ├──► Prefix/Prompt Tuning ──► Trains only a few hundred params
        │    (folders 08, 09)          Works on very limited hardware
        │
        └──► BitFit ──► Only trains bias terms
             (folder 12)
```

With **QLoRA** (4-bit quantization + LoRA), you can fine-tune a **70B model on a single 48GB GPU**. That's the game-changer.

### Other Practical Limits

| Limitation | Why it matters |
|---|---|
| **License** | Some models restrict commercial use (e.g., older LLaMA 1). Check the license before training. |
| **Architecture** | SFT works on any autoregressive (decoder-only) model: GPT, LLaMA, Mistral, Phi, Qwen, etc. It also works on encoder-decoder models (T5, BART) with slight adjustments. NOT designed for encoder-only models (BERT) — those use different fine-tuning. |
| **Data quantity** | You need enough quality instruction-response pairs. A few hundred can work, but a few thousand is better. Too few = overfitting. |
| **Base model quality** | SFT can't fix a bad base model. If the pretrained model doesn't understand the language or domain at all, SFT won't magically add that knowledge — it can only reshape what's already there. |
| **Catastrophic forgetting** | Too much SFT on narrow data can make the model forget its general abilities. It becomes great at your task but terrible at everything else. |

### Can You SFT Any Model?

```
Can you SFT any model?  YES — technically.

Can you SFT any model    NO — you need:
on any hardware?           • Enough VRAM (or use LoRA/QLoRA to reduce it)
                           • A compatible license
                           • Sufficient training data
                           • A base model that already has the right
                             language/knowledge foundation
```

**In practice:** thanks to QLoRA, almost any open-weight model up to 70B is fine-tunable on a single rented cloud GPU ($1-3/hr). The real limitation today isn't the technique — it's the **quality of your training data**.

---

## Real-World Business Scenarios Where SFT Is Essential

Here are the scenarios where SFT is the right (and sometimes only) tool:

### 1. Custom Chatbot With Proprietary Knowledge

```
Client: "We have 10,000 internal support tickets. Build us an AI 
         that answers customer questions the way our best agents do."

Why SFT:  RAG alone won't work well here — it's not about retrieving 
          documents, it's about mimicking a specific STYLE and TONE 
          of responding. SFT trains the model to respond like their 
          best agents naturally.
```

### 2. Data Privacy / Compliance — Data Cannot Leave the Building

```
Client: "We're a bank/hospital/government agency. Our data CANNOT 
         be sent to OpenAI's API. Ever. Regulatory requirement."

Why SFT:  You MUST use a local model. You download an open-weight 
          model, SFT it on-premise with their private data, and 
          deploy it internally. No data leaves their network.

Industries: Healthcare (HIPAA), Finance (SOC2), Legal, Defense, EU (GDPR)
```

### 3. The Task Is Too Specialized for General Models

```
Client: "We need a model that reads radiology reports and generates 
         structured findings in our specific template format."

Why SFT:  GPT-4 doesn't know your client's exact template. Prompt 
          engineering gets you 70% there. SFT gets you 95%+. The 
          model learns the EXACT output format, terminology, and 
          edge cases from hundreds of real examples.

Other examples:
  • Legal contract clause extraction in a specific jurisdiction
  • Code generation for a proprietary internal framework
  • Translating between niche domain languages (e.g., insurance codes)
```

### 4. Cost at Scale — API Bills Are Killing the Budget

```
Client: "We're making 500,000 API calls/day to GPT-4. 
         Our monthly bill is $50,000+."

Why SFT:  Fine-tune a 7B or 13B model with LoRA/QLoRA on their 
          specific task. Deploy on a $2,000 GPU. 
          Monthly cost drops from $50K to ~$200 (electricity + hosting).
          
          The smaller SFT'd model often MATCHES GPT-4 on the 
          specific task it was trained for.
```

### 5. Latency Requirements — Speed Matters

```
Client: "We need responses in <100ms for our real-time trading 
         platform / autocomplete / in-game NPC dialogue."

Why SFT:  Cloud API = network round trip + queue time = 500ms-2s.
          Local SFT'd small model (1B-3B) on a GPU = 20-50ms.
          No comparison.
```

### 6. The Client Wants to OWN the Model

```
Client: "We don't want to depend on OpenAI. If they change pricing, 
         shut down, or alter their model, our product breaks."

Why SFT:  SFT on an open-weight model = they own the weights forever.
          No vendor lock-in. No surprise API changes. No dependency.
```

### 7. Consistent, Deterministic Outputs

```
Client: "Every time we call GPT-4, we get slightly different answers. 
         We need the EXACT same output format every time for our 
         automated pipeline."

Why SFT:  SFT'd models with temperature=0 are highly consistent.
          You control the model version — it never changes unless 
          YOU retrain it. Cloud models get updated without notice.
```

### Quick Decision Framework

```
Does the client need any of these?
        │
        ├── Data must stay private?          ──► SFT (local model)
        ├── Very specific output format?     ──► SFT
        ├── High volume (>10K calls/day)?    ──► SFT saves money
        ├── Low latency (<200ms)?            ──► SFT (local)
        ├── No vendor dependency?            ──► SFT (local)
        ├── Domain too niche for GPT-4?      ──► SFT
        │
        └── None of the above?               ──► Just use the API
            General tasks, low volume,           (prompt engineering
            no privacy concerns                   is enough)
```

### The Honest Truth

For **most** simple business tasks (summarization, basic Q&A, email drafting), prompt engineering with a cloud API is faster and cheaper to set up. You'd recommend SFT to a client when there's a **clear reason** the API won't work — privacy, cost at scale, latency, format precision, or ownership. Those are the jobs where knowing SFT makes you the person they need to hire.

---

## SFT vs Other Techniques — Where SFT Is Uniquely the Best

### Where SFT is NOT the best option (other techniques win)

```
"I want the model to know about our company's documents"
    └──► RAG is better. Retrieval + generation. No retraining needed.

"I want the model to understand a new domain deeply (e.g., medicine)"
    └──► Continued Pretraining (DAPT/TAPT — folders 17, 18) is better.
         You feed it raw domain text so it absorbs the vocabulary 
         and knowledge at a deeper level than SFT can.

"I want the model to produce JSON / follow a schema"
    └──► Structured output / constrained decoding is better.
         Tools like Outlines, LMQL, or function calling handle 
         this without any training at all.

"I want better answers for my specific task"
    └──► Prompt engineering + few-shot examples often gets you 
         90% of the way with zero training.
```

### Where SFT is genuinely THE best option

**SFT wins when you need to change HOW the model behaves — its personality, tone, style, conversational pattern, and response habits.**

**1. Mimicking a specific voice/tone/persona**
```
"Respond like a friendly but professional insurance agent who 
 always acknowledges the customer's frustration first, then 
 gives the answer in 2-3 bullet points, and ends with a 
 reassuring closing."

Why only SFT:  This is a BEHAVIORAL pattern. Prompt engineering 
               can approximate it, but it drifts. After 5-10 turns 
               the model forgets the persona. SFT bakes it into 
               the weights — the model BECOMES that persona. 
               It never drifts because it's not following 
               instructions anymore, it's following its training.
```

**2. Teaching a new conversational format the model has never seen**
```
"Our system uses a custom multi-turn format:
 [AGENT_QUERY] ... [TOOL_CALL] ... [TOOL_RESULT] ... [AGENT_REPLY]
 The model needs to know when to call tools vs reply directly."

Why only SFT:  The model has never seen your custom format. 
               No amount of prompting reliably teaches a new 
               interaction protocol. SFT trains the model to 
               natively understand your format as if it was 
               always part of its training.
```

**3. Consistent refusal/safety behavior**
```
"The model must ALWAYS refuse to discuss competitors, never 
 speculate about pricing, and always redirect medical questions 
 to a professional."

Why only SFT:  Prompts can be jailbroken. System prompts leak. 
               SFT'd behavior is much harder to override because 
               it's in the weights, not in context that can be 
               manipulated. This is literally why OpenAI uses SFT 
               + RLHF for safety — prompting alone isn't robust.
```

**4. Turning a base model into an instruction-following model**
```
"We downloaded LLaMA-3-8B base. It just autocompletes text. 
 We need it to actually answer questions."

Why only SFT:  This is THE original purpose of SFT. A base model 
               doesn't know what a "question" or "instruction" is. 
               RAG can't help — the model doesn't know how to USE 
               the retrieved context. Prompting a base model is 
               unreliable. SFT is the ONLY way to teach it the 
               concept of "I receive instructions, I produce answers."
```

### The One-Liner

```
RAG         = teaches the model WHAT to know    (knowledge)
Pretraining = teaches the model WHAT words mean (understanding)  
Prompting   = tells the model what to do NOW    (temporary)
SFT         = teaches the model HOW to behave   (permanent)
                                    ▲
                            This is its unique strength
```

SFT changes the model's **default behavior patterns**. Everything else either adds knowledge or gives temporary instructions. When the client says "I want the model to *always* act this way, not just when prompted" — that's SFT territory, and nothing else does it as well.

---

## Bottom Line

SFT is the **same technique** in both cases — mask the prompt, train on the response. The difference is just **who runs it and where**. The script in this folder (`sft_from_scratch.py`) teaches you the mechanics so you can do it yourself on local/open models.
