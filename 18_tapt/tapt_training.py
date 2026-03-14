"""
TAPT Training — Production Pipeline with HuggingFace
======================================================

Production-ready TAPT implementations:

1. Full TAPT Pipeline — end-to-end with HuggingFace Trainer
2. LoRA-TAPT with PEFT — parameter-efficient task adaptation
3. Curated TAPT with Retrieval — semantic retrieval for data expansion
4. Combined DAPT + TAPT Pipeline — multi-stage adaptation
5. Task-Specific Data Preparation — preparing task text for TAPT

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# ============================================================================
# SECTION 1: FULL TAPT PIPELINE WITH HUGGINGFACE
# ============================================================================

def full_tapt_pipeline():
    """Production-grade TAPT with HuggingFace Trainer."""
    print("=" * 70)
    print("  SECTION 1: FULL TAPT PIPELINE WITH HUGGINGFACE")
    print("=" * 70)
    
    pipeline_code = '''
# ═══════════════════════════════════════════════════════════════
# PRODUCTION CODE: Full TAPT Pipeline
# ═══════════════════════════════════════════════════════════════

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
import torch
import math

# ─── Step 1: Load pretrained model ───
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# ─── Step 2: Prepare task data (strip labels, keep text only) ───
# Example: IMDB sentiment → extract just the review text

# Option A: Load a classification dataset and extract text
task_dataset = load_dataset("imdb")

# TAPT key insight: we only need the TEXT, not the labels
def extract_text(examples):
    """Strip labels, keep only input text for language modeling."""
    return {"text": examples["text"]}

tapt_data = task_dataset.map(
    extract_text,
    remove_columns=["label"],  # Drop labels!
    desc="Extracting text for TAPT",
)

# Option B: From a custom dataset
# texts = [example["input_text"] for example in my_task_data]
# tapt_data = Dataset.from_dict({"text": texts})

# ─── Step 3: Tokenize for language modeling ───
def tokenize_for_lm(examples, block_size=256):
    """Tokenize and chunk into fixed-length sequences."""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=block_size,
        padding=False,
    )
    return tokenized

tokenized = tapt_data.map(
    tokenize_for_lm,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing for TAPT",
)

# Group into blocks (more efficient than variable-length)
def group_texts(examples, block_size=256):
    """Concatenate all texts and split into chunks."""
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // block_size) * block_size
    
    result = {
        k: [t[i:i+block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized.map(
    group_texts,
    batched=True,
    desc="Grouping into blocks",
)

# ─── Step 4: TAPT Training Arguments ───
# CRITICAL: TAPT uses MANY epochs on SMALL data

n_examples = len(lm_dataset["train"])
print(f"TAPT dataset size: {n_examples} blocks")

# Epoch calculation based on dataset size
if n_examples < 1000:
    n_epochs = 100      # Very small → many epochs
elif n_examples < 5000:
    n_epochs = 50       # Small → moderate epochs
elif n_examples < 20000:
    n_epochs = 20       # Medium
else:
    n_epochs = 5        # Large → few epochs (approaching DAPT)

training_args = TrainingArguments(
    output_dir="./tapt_output",
    
    # === CRITICAL TAPT SETTINGS ===
    
    # Learning rate: same as DAPT (low!)
    learning_rate=2e-5,
    
    # MANY epochs on small data
    num_train_epochs=n_epochs,
    
    # Warmup: 5-10% of total steps
    warmup_ratio=0.06,
    
    # Cosine schedule (smooth decay over many epochs)
    lr_scheduler_type="cosine",
    
    # Gradient clipping
    max_grad_norm=1.0,
    
    # Mild weight decay
    weight_decay=0.01,
    
    # === COMPUTE SETTINGS ===
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    
    # FP16
    fp16=torch.cuda.is_available(),
    
    # === MONITORING ===
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    seed=42,
)

# ─── Step 5: Data Collator ───
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM for GPT-style models
    # mlm=True,  # Use for BERT-style models
)

# ─── Step 6: Trainer ───
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"] if "test" in lm_dataset else None,
    data_collator=data_collator,
)

# ─── Step 7: Evaluate baseline ───
if "test" in lm_dataset:
    baseline = trainer.evaluate()
    print(f"Baseline PPL: {math.exp(baseline['eval_loss']):.2f}")

# ─── Step 8: Run TAPT ───
trainer.train()

# ─── Step 9: Save ───
model.save_pretrained("./tapt_model")
tokenizer.save_pretrained("./tapt_model")

# ─── Step 10: Now fine-tune on labeled task ───
# from transformers import AutoModelForSequenceClassification
# tapt_model = AutoModelForSequenceClassification.from_pretrained(
#     "./tapt_model", num_labels=2)
# # ... train on labeled data as usual ...
'''
    
    print(pipeline_code)
    
    print(f"""
  ═══ TAPT vs DAPT Training Arguments ═══
  
  ┌─────────────────────┬──────────────┬──────────────┐
  │ Parameter           │ DAPT         │ TAPT         │
  ├─────────────────────┼──────────────┼──────────────┤
  │ learning_rate       │ 2e-5         │ 2e-5         │
  │ num_train_epochs    │ 1-3          │ 5-100 ★      │
  │ warmup_ratio        │ 0.05-0.10    │ 0.05-0.10    │
  │ lr_scheduler_type   │ cosine       │ cosine       │
  │ data_size           │ 10M-1B tok   │ 500-50K ex ★ │
  │ training_time       │ Hours-Days   │ Min-Hours ★  │
  └─────────────────────┴──────────────┴──────────────┘
  
  ★ = key differences. TAPT: more epochs, less data, faster!
""")


# ============================================================================
# SECTION 2: LoRA-TAPT WITH PEFT
# ============================================================================

def lora_tapt_with_peft():
    """Parameter-efficient TAPT using HuggingFace PEFT library."""
    print("\n\n" + "=" * 70)
    print("  SECTION 2: LoRA-TAPT WITH PEFT LIBRARY")
    print("=" * 70)
    
    lora_code = '''
# ═══════════════════════════════════════════════════════════════
# PRODUCTION CODE: LoRA-TAPT with PEFT
# ═══════════════════════════════════════════════════════════════

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, get_peft_model, TaskType, PeftModel
)
import torch

# ─── Step 1: Load model ───
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# ─── Step 2: Configure LoRA for TAPT ───
# TAPT uses LOWER rank than DAPT (task data is smaller/simpler)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    
    # Lower rank for TAPT (vs r=32 for DAPT)
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    
    # Target attention layers (sufficient for TAPT)
    target_modules=[
        "c_attn",       # GPT-2 attention
        "c_proj",       # GPT-2 attention output
        # For LLaMA: "q_proj", "k_proj", "v_proj", "o_proj"
    ],
    
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ─── Step 3: Prepare task data for LM ───
# [Same as Section 1: extract text, tokenize, chunk]

# ─── Step 4: Training ───
training_args = TrainingArguments(
    output_dir="./lora_tapt_output",
    
    # Higher LR for LoRA
    learning_rate=3e-4,
    
    # Many epochs (TAPT on small data)
    num_train_epochs=50,
    
    warmup_ratio=0.06,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    weight_decay=0.01,
    
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    fp16=torch.cuda.is_available(),
    
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

trainer.train()

# ─── Step 5: Save TAPT adapter ───
model.save_pretrained("./lora_tapt_adapter")

# ─── Step 6: Downstream task fine-tuning ───
# Option A: Merge adapter, then fine-tune
# merged = model.merge_and_unload()
# merged.save_pretrained("./tapt_merged")
# task_model = AutoModelForSequenceClassification.from_pretrained(
#     "./tapt_merged", num_labels=2)

# Option B: Stack another LoRA for task FT
# base = AutoModelForCausalLM.from_pretrained(model_name)
# tapt_model = PeftModel.from_pretrained(base, "./lora_tapt_adapter")
# tapt_model = tapt_model.merge_and_unload()
#
# task_lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, ...)
# task_model = get_peft_model(tapt_model, task_lora)
'''
    
    print(lora_code)
    
    print(f"""
  ═══ LoRA Configuration: TAPT vs DAPT vs Task FT ═══
  
  ┌─────────────┬──────────┬──────────┬──────────┐
  │ Parameter   │ DAPT     │ TAPT     │ Task FT  │
  ├─────────────┼──────────┼──────────┼──────────┤
  │ rank        │ 32       │ 8-16     │ 4-8      │
  │ alpha       │ 64       │ 16-32    │ 8-16     │
  │ LR          │ 5e-4     │ 3e-4     │ 2e-4     │
  │ dropout     │ 0.05     │ 0.05     │ 0.1      │
  │ target      │ All attn │ Attn     │ Q, V     │
  │ epochs      │ 1-3      │ 20-100   │ 3-10     │
  │ adapter size│ ~20 MB   │ ~5 MB    │ ~2 MB    │
  └─────────────┴──────────┴──────────┴──────────┘
  
  Pattern: rank and adapter size DECREASE as data gets smaller
  but epochs INCREASE to compensate.
""")


# ============================================================================
# SECTION 3: CURATED TAPT WITH RETRIEVAL
# ============================================================================

def curated_tapt_with_retrieval():
    """Expand TAPT data using semantic retrieval from a large corpus."""
    print("\n\n" + "=" * 70)
    print("  SECTION 3: CURATED TAPT WITH RETRIEVAL")
    print("=" * 70)
    
    retrieval_code = '''
# ═══════════════════════════════════════════════════════════════
# PRODUCTION CODE: Curated TAPT with Retrieval
# ═══════════════════════════════════════════════════════════════

from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets
import torch
import torch.nn.functional as F
import numpy as np

# ─── Step 1: Embed task examples ───
# Use a pretrained sentence encoder

encoder_name = "sentence-transformers/all-MiniLM-L6-v2"
encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_name)
encoder_model = AutoModel.from_pretrained(encoder_name)

def encode_texts(texts, batch_size=32):
    """Encode texts using a sentence transformer."""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = encoder_tokenizer(
            batch, padding=True, truncation=True, 
            max_length=256, return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = encoder_model(**inputs)
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            token_embs = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).float()
            embeddings = (token_embs * mask_expanded).sum(1) / mask_expanded.sum(1)
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        all_embeddings.append(embeddings.cpu())
    
    return torch.cat(all_embeddings, dim=0)

# ─── Step 2: Embed task and pool data ───
# Task data
task_dataset = load_dataset("imdb", split="train[:1000]")
task_texts = task_dataset["text"]
task_embeddings = encode_texts(task_texts)

# Large unlabeled pool (e.g., C4, or domain-specific corpus)
# pool_dataset = load_dataset("c4", "en", split="train[:100000]")
# pool_texts = pool_dataset["text"]
# pool_embeddings = encode_texts(pool_texts)

# ─── Step 3: Retrieve k nearest neighbors ───
def retrieve_neighbors(task_embs, pool_embs, pool_texts, k=5):
    """Retrieve k nearest neighbors from pool for each task example."""
    # Cosine similarity
    similarities = torch.mm(task_embs, pool_embs.t())
    
    # Get top-k per task example
    topk_sims, topk_idx = similarities.topk(k, dim=1)
    
    # Collect unique retrieved texts
    unique_indices = set()
    for row in topk_idx:
        for idx in row.tolist():
            unique_indices.add(idx)
    
    retrieved_texts = [pool_texts[i] for i in sorted(unique_indices)]
    
    return retrieved_texts, topk_sims.mean().item()

# retrieved, avg_sim = retrieve_neighbors(task_embeddings, pool_embeddings, pool_texts)
# print(f"Retrieved {len(retrieved)} unique examples (avg sim: {avg_sim:.3f})")

# ─── Step 4: Quality filtering of retrieved data ───
def filter_by_similarity(retrieved_texts, task_embs, pool_embs, 
                         pool_texts, min_similarity=0.3):
    """Keep only high-similarity retrieved examples."""
    retrieved_embs = encode_texts(retrieved_texts)
    
    # Mean similarity to task centroid
    task_centroid = task_embs.mean(dim=0, keepdim=True)
    similarities = F.cosine_similarity(retrieved_embs, task_centroid)
    
    filtered = [text for text, sim in zip(retrieved_texts, similarities) 
                if sim >= min_similarity]
    
    return filtered

# ─── Step 5: Build Curated TAPT dataset ───
def build_curated_dataset(task_texts, retrieved_texts, 
                          task_weight=2.0):
    """Combine task and retrieved data with upsampling of task data."""
    # Upsample task examples (they're the most relevant!)
    n_task_repeats = int(task_weight)
    expanded_task = task_texts * n_task_repeats
    
    all_texts = expanded_task + retrieved_texts
    return Dataset.from_dict({"text": all_texts})

# curated_dataset = build_curated_dataset(task_texts, filtered_retrieved)
# print(f"Curated TAPT dataset: {len(curated_dataset)} examples")
# Then tokenize and train as in Section 1
'''
    
    print(retrieval_code)
    
    print(f"""
  ═══ Curated TAPT Architecture ═══
  
  ┌────────────┐         ┌──────────────┐
  │ Task Data  │────────→│ Encode with  │──→ Task embeddings
  │ (1K docs)  │         │ sentence     │
  └────────────┘         │ transformer  │
                         └──────────────┘
                                │
  ┌────────────┐         ┌──────▼───────┐
  │ Pool Data  │────────→│ k-NN search  │──→ Retrieved docs
  │ (100K docs)│         │ (cosine sim) │    (5K docs)
  └────────────┘         └──────────────┘
                                │
                         ┌──────▼───────┐
                         │ Quality      │──→ Filtered docs
                         │ threshold    │    (3K docs)
                         └──────────────┘
                                │
                         ┌──────▼───────┐
                         │ Combine:     │──→ Curated Dataset
                         │ task (2x) +  │    (5K docs)
                         │ retrieved    │
                         └──────────────┘
  
  ═══ Curated TAPT Scaling Recommendations ═══
  
  ┌───────────────┬─────────────┬─────────────┬──────────────┐
  │ Task Size     │ k neighbors │ Pool Size   │ Expansion    │
  ├───────────────┼─────────────┼─────────────┼──────────────┤
  │ < 500 docs    │ 10-20       │ 100K+       │ 5-10x        │
  │ 500-5K docs   │ 5-10        │ 50K+        │ 2-5x         │
  │ 5K-20K docs   │ 3-5         │ 50K+        │ 1.5-3x       │
  │ > 20K docs    │ Skip        │ —           │ Not needed   │
  └───────────────┴─────────────┴─────────────┴──────────────┘
""")


# ============================================================================
# SECTION 4: COMBINED DAPT + TAPT PIPELINE
# ============================================================================

def combined_dapt_tapt_pipeline():
    """Full multi-stage pipeline: DAPT → TAPT → Task Fine-Tuning."""
    print("\n\n" + "=" * 70)
    print("  SECTION 4: COMBINED DAPT + TAPT PIPELINE")
    print("=" * 70)
    
    combined_code = '''
# ═══════════════════════════════════════════════════════════════
# PRODUCTION CODE: DAPT → TAPT → Task Fine-Tuning Pipeline
# ═══════════════════════════════════════════════════════════════

from transformers import (
    AutoModelForCausalLM, AutoModelForSequenceClassification,
    AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset
import torch
import math

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ═══════════════════════════════════════════════════
# STAGE 1: DAPT (Domain-Adaptive Pretraining)
# ═══════════════════════════════════════════════════
print("=" * 40)
print("STAGE 1: DAPT on domain corpus")
print("=" * 40)

model = AutoModelForCausalLM.from_pretrained(model_name)

# LoRA for DAPT
dapt_lora = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    target_modules=["c_attn", "c_proj", "c_fc"],
    lora_dropout=0.05,
)
dapt_model = get_peft_model(model, dapt_lora)

# Train on domain corpus (e.g., biomedical papers)
# ... [training code from Concept 17] ...

# Merge DAPT adapter into base
dapt_model = dapt_model.merge_and_unload()
dapt_model.save_pretrained("./stage1_dapt")
print("DAPT complete, saved to ./stage1_dapt")

# ═══════════════════════════════════════════════════
# STAGE 2: TAPT (Task-Adaptive Pretraining)
# ═══════════════════════════════════════════════════
print("=" * 40)
print("STAGE 2: TAPT on task text")
print("=" * 40)

# Load DAPT-adapted model
model = AutoModelForCausalLM.from_pretrained("./stage1_dapt")

# LoRA for TAPT (lower rank)
tapt_lora = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
)
tapt_model = get_peft_model(model, tapt_lora)

# Extract unlabeled text from task dataset
task_dataset = load_dataset("imdb")
# Strip labels for TAPT
tapt_texts = task_dataset["train"]["text"]
# ... tokenize and train for many epochs ...

# Merge TAPT adapter
tapt_model = tapt_model.merge_and_unload()
tapt_model.save_pretrained("./stage2_tapt")
print("TAPT complete, saved to ./stage2_tapt")

# ═══════════════════════════════════════════════════
# STAGE 3: Task Fine-Tuning (Supervised)
# ═══════════════════════════════════════════════════
print("=" * 40)
print("STAGE 3: Task Fine-Tuning")
print("=" * 40)

# Load adapted model for classification
model = AutoModelForSequenceClassification.from_pretrained(
    "./stage2_tapt",
    num_labels=2,
)

# LoRA for task fine-tuning (lowest rank)
task_lora = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=4,
    lora_alpha=8,
    target_modules=["c_attn"],
    lora_dropout=0.1,
)
task_model = get_peft_model(model, task_lora)

# Train on labeled data
training_args = TrainingArguments(
    output_dir="./stage3_task_ft",
    learning_rate=2e-4,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# trainer = Trainer(
#     model=task_model,
#     args=training_args,
#     train_dataset=task_dataset["train"],
#     eval_dataset=task_dataset["test"],
# )
# trainer.train()

print("Pipeline complete: DAPT → TAPT → Task FT!")
'''
    
    print(combined_code)
    
    print(f"""
  ═══ Combined Pipeline Summary ═══
  
  ┌──────────┬───────────┬──────────┬─────────┬──────────────────┐
  │ Stage    │ LoRA Rank │ LR       │ Epochs  │ Data             │
  ├──────────┼───────────┼──────────┼─────────┼──────────────────┤
  │ 1. DAPT  │ 32        │ 5e-4     │ 1-3     │ Domain corpus    │
  │ 2. TAPT  │ 8         │ 3e-4     │ 20-100  │ Task text (no labels) │
  │ 3. FT    │ 4         │ 2e-4     │ 3-10    │ Labeled task data│
  └──────────┴───────────┴──────────┴─────────┴──────────────────┘
  
  Each stage:
  1. Creates a new LoRA adapter on the result of the previous stage
  2. Merges the adapter before moving to the next stage
  3. Uses decreasing rank (broader → narrower adaptation)
  
  Expected Results (Gururangan et al. style):
  
  ┌──────────────────┬──────────┬─────────────────────┐
  │ Configuration    │ F1 Score │ Cost                 │
  ├──────────────────┼──────────┼─────────────────────┤
  │ Base + FT        │ 82.3%    │ $ (just FT)          │
  │ TAPT + FT        │ 84.1%    │ $ (very cheap)       │
  │ DAPT + FT        │ 86.7%    │ $$$ (domain PT)      │
  │ DAPT + TAPT + FT │ 88.2%    │ $$$$ (both)     ★    │
  │ Curated TAPT + FT│ 85.6%    │ $$ (retrieval + PT)  │
  └──────────────────┴──────────┴─────────────────────┘
  
  ★ = Best quality, $$$$ = Most expensive but highest impact
""")


# ============================================================================
# SECTION 5: TASK-SPECIFIC DATA PREPARATION
# ============================================================================

def task_data_preparation():
    """Preparing different task formats for TAPT."""
    print("\n\n" + "=" * 70)
    print("  SECTION 5: TASK-SPECIFIC DATA PREPARATION")
    print("=" * 70)
    
    data_prep_code = '''
# ═══════════════════════════════════════════════════════════════
# TASK DATA PREPARATION FOR TAPT
# ═══════════════════════════════════════════════════════════════

from datasets import Dataset

# ─── Task Type 1: Text Classification ───
# Just extract the input text (drop labels)

def prepare_classification_for_tapt(dataset, text_column="text"):
    """Classification data → TAPT data."""
    texts = dataset[text_column]
    return Dataset.from_dict({"text": texts})

# Example:
# imdb_tapt = prepare_classification_for_tapt(load_dataset("imdb")["train"])
# sst2_tapt = prepare_classification_for_tapt(load_dataset("glue", "sst2")["train"],
#                                              text_column="sentence")

# ─── Task Type 2: Named Entity Recognition (NER) ───
# Reconstruct sentences from token-level data

def prepare_ner_for_tapt(dataset, token_col="tokens"):
    """NER data → TAPT data."""
    texts = [" ".join(example[token_col]) for example in dataset]
    return Dataset.from_dict({"text": texts})

# Example:
# conll_tapt = prepare_ner_for_tapt(load_dataset("conll2003")["train"])

# ─── Task Type 3: Question Answering ───
# Combine context + question

def prepare_qa_for_tapt(dataset, context_col="context", question_col="question"):
    """QA data → TAPT data (concatenate context + question)."""
    texts = []
    seen_contexts = set()
    
    for example in dataset:
        # Add unique contexts (avoid duplicates)
        ctx = example[context_col]
        if ctx not in seen_contexts:
            texts.append(ctx)
            seen_contexts.add(ctx)
        
        # Add question-context pairs
        qa_text = f"{example[question_col]} {ctx}"
        texts.append(qa_text)
    
    return Dataset.from_dict({"text": texts})

# ─── Task Type 4: Natural Language Inference (NLI) ───
# Combine premise + hypothesis

def prepare_nli_for_tapt(dataset, premise_col="premise",
                          hypothesis_col="hypothesis"):
    """NLI data → TAPT data."""
    texts = []
    for example in dataset:
        # Add each sentence separately AND combined
        texts.append(example[premise_col])
        texts.append(example[hypothesis_col])
        texts.append(f"{example[premise_col]} {example[hypothesis_col]}")
    
    return Dataset.from_dict({"text": texts})

# ─── Task Type 5: Summarization ───
# Use both source documents and summaries

def prepare_summarization_for_tapt(dataset, doc_col="article",
                                    summary_col="highlights"):
    """Summarization data → TAPT data."""
    texts = []
    for example in dataset:
        texts.append(example[doc_col])
        texts.append(example[summary_col])
    
    return Dataset.from_dict({"text": texts})

# ─── Task Type 6: Relation Extraction ───
# Extract sentences with entity mentions

def prepare_re_for_tapt(dataset, text_col="text"):
    """Relation extraction data → TAPT data."""
    texts = [example[text_col] for example in dataset]
    return Dataset.from_dict({"text": texts})
'''
    
    print(data_prep_code)
    
    # Show task-specific TAPT tips
    print(f"""
  ═══ Task-Specific TAPT Tips ═══
  
  ┌──────────────────┬──────────────────────────────────────────────┐
  │ Task             │ TAPT Preparation Tips                        │
  ├──────────────────┼──────────────────────────────────────────────┤
  │ Classification   │ Use full input text, drop labels             │
  │ NER              │ Reconstruct full sentences from tokens       │
  │ QA               │ Deduplicate contexts, include questions      │
  │ NLI              │ Include premises, hypotheses, and pairs      │
  │ Summarization    │ Include both documents AND summaries         │
  │ Relation Extract │ Focus on sentences with entity mentions      │
  │ Translation      │ Use source and target language texts         │
  │ Code             │ Include function bodies and docstrings       │
  └──────────────────┴──────────────────────────────────────────────┘
  
  ═══ General TAPT Data Preparation Checklist ═══
  
  ☐ Extract all TEXT from your task dataset (drop labels)
  ☐ Include BOTH sides for pair tasks (NLI, QA)
  ☐ Deduplicate passages that appear multiple times
  ☐ Keep text in its original form (don't over-clean for LM)
  ☐ If short texts, consider concatenating for efficiency
  ☐ Split 95/5 for train/validation
  ☐ Consider Curated TAPT if dataset < 1000 examples
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  TAPT TRAINING — PRODUCTION PIPELINE WITH HUGGINGFACE           ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    full_tapt_pipeline()
    lora_tapt_with_peft()
    curated_tapt_with_retrieval()
    combined_dapt_tapt_pipeline()
    task_data_preparation()
    
    print("\n" + "=" * 70)
    print("  TRAINING MODULE COMPLETE")
    print("=" * 70)
    print("""
    Covered:
    ✓ Full TAPT pipeline with HuggingFace Trainer
    ✓ LoRA-TAPT with PEFT (parameter-efficient)
    ✓ Curated TAPT with semantic retrieval
    ✓ Combined DAPT + TAPT + Task FT pipeline
    ✓ Task-specific data preparation for 6 task types
    """)


if __name__ == "__main__":
    main()
