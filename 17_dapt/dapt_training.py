"""
DAPT Training — Production Pipeline with HuggingFace
=====================================================

Production-ready DAPT implementations:

1. Full DAPT Pipeline — end-to-end with HuggingFace Trainer
2. LoRA-DAPT with PEFT — parameter-efficient domain adaptation
3. Data Preparation — domain corpus cleaning and tokenization
4. Monitoring & Evaluation — tracking domain adaptation progress
5. Multi-Stage Pipeline — DAPT → TAPT → Task Fine-Tuning

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# ============================================================================
# SECTION 1: FULL DAPT PIPELINE WITH HUGGINGFACE
# ============================================================================

def full_dapt_pipeline():
    """Production-grade DAPT with HuggingFace Trainer."""
    print("=" * 70)
    print("  SECTION 1: FULL DAPT PIPELINE WITH HUGGINGFACE")
    print("=" * 70)
    
    pipeline_code = '''
# ═══════════════════════════════════════════════════════════════
# PRODUCTION CODE: Full DAPT Pipeline
# ═══════════════════════════════════════════════════════════════

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch

# ─── Step 1: Load pretrained model ───
model_name = "distilgpt2"  # Replace with your base model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# ─── Step 2: Load and tokenize domain corpus ───
# Option A: Load from local files
# dataset = load_dataset("text", data_files={
#     "train": "domain_corpus_train.txt",
#     "validation": "domain_corpus_val.txt"
# })

# Option B: Load from HuggingFace Hub
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def tokenize_function(examples):
    """Tokenize and chunk text into fixed-length sequences."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        return_overflowing_tokens=True,
        return_length=True,
    )

tokenized = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing",
)

# Group texts into chunks for efficient training
def group_texts(examples):
    """Concatenate texts and split into chunks of block_size."""
    block_size = 512
    
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // block_size) * block_size
    
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized.map(
    group_texts,
    batched=True,
    desc="Grouping texts",
)

# ─── Step 3: DAPT Training Arguments ───
training_args = TrainingArguments(
    output_dir="./dapt_output",
    
    # === CRITICAL DAPT SETTINGS ===
    
    # Learning rate: 5-10x LOWER than original pretraining
    learning_rate=2e-5,
    
    # Short training: 1-3 epochs
    num_train_epochs=2,
    
    # Warmup: 10% of steps (prevents sudden distribution shift)
    warmup_ratio=0.1,
    
    # Cosine schedule: smooth LR decay
    lr_scheduler_type="cosine",
    
    # Gradient clipping: tighter than default
    max_grad_norm=0.5,
    
    # Weight decay: mild regularization
    weight_decay=0.01,
    
    # === COMPUTE SETTINGS ===
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,  # Effective batch: 32
    
    # FP16/BF16
    fp16=torch.cuda.is_available(),
    # bf16=True,  # Use on Ampere+ GPUs
    
    # === MONITORING ===
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=3,  # Keep last 3 checkpoints
    
    # Load best model at end
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # Reproducibility
    seed=42,
    data_seed=42,
)

# ─── Step 4: Data Collator ───
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM (not masked LM)
)

# ─── Step 5: Create Trainer ───
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["validation"],
    data_collator=data_collator,
)

# ─── Step 6: Evaluate BEFORE DAPT (baseline) ───
print("Baseline evaluation (before DAPT):")
baseline_metrics = trainer.evaluate()
print(f"  Perplexity: {math.exp(baseline_metrics['eval_loss']):.2f}")

# ─── Step 7: Run DAPT ───
trainer.train()

# ─── Step 8: Evaluate AFTER DAPT ───
print("Post-DAPT evaluation:")
post_metrics = trainer.evaluate()
print(f"  Perplexity: {math.exp(post_metrics['eval_loss']):.2f}")

# ─── Step 9: Save domain-adapted model ───
model.save_pretrained("./domain_adapted_model")
tokenizer.save_pretrained("./domain_adapted_model")

print("DAPT complete! Model saved to ./domain_adapted_model")
'''
    
    print(pipeline_code)
    
    print(f"""
  ═══ Key DAPT Training Parameters ═══
  
  ┌─────────────────────┬──────────────┬──────────────────────────────┐
  │ Parameter           │ Value        │ Why                          │
  ├─────────────────────┼──────────────┼──────────────────────────────┤
  │ learning_rate       │ 1e-5 to 5e-5 │ Low to preserve knowledge   │
  │ num_train_epochs    │ 1-3          │ Short to avoid forgetting    │
  │ warmup_ratio        │ 0.05-0.15    │ Smooth start                 │
  │ lr_scheduler_type   │ "cosine"     │ Smooth decay                 │
  │ max_grad_norm       │ 0.3-0.5      │ Prevent large updates        │
  │ weight_decay        │ 0.01-0.1     │ Mild regularization          │
  │ effective_batch     │ 32-128       │ Stable gradients             │
  └─────────────────────┴──────────────┴──────────────────────────────┘
""")


# ============================================================================
# SECTION 2: LoRA-DAPT WITH PEFT
# ============================================================================

def lora_dapt_with_peft():
    """Parameter-efficient DAPT using HuggingFace PEFT library."""
    print("\n\n" + "=" * 70)
    print("  SECTION 2: LoRA-DAPT WITH PEFT LIBRARY")
    print("=" * 70)
    
    lora_code = '''
# ═══════════════════════════════════════════════════════════════
# PRODUCTION CODE: LoRA-DAPT with PEFT
# ═══════════════════════════════════════════════════════════════

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, get_peft_model, TaskType,
    PeftModel, prepare_model_for_kbit_training
)
from datasets import load_dataset
import torch
import math

# ─── Configuration ───
model_name = "distilgpt2"
domain = "biomedical"  # Your target domain

# ─── Step 1: Load model ───
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# ─── Step 2: Configure LoRA for DAPT ───
# For DAPT, we typically use HIGHER rank than task fine-tuning
# because domain adaptation needs more capacity

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    
    # Higher rank for DAPT (vs r=8-16 for task FT)
    r=32,
    lora_alpha=64,          # alpha = 2 * r (standard)
    lora_dropout=0.05,
    
    # Target MORE modules for comprehensive domain adaptation
    target_modules=[
        "c_attn",           # GPT-2: Q, K, V attention (combined)
        "c_proj",           # GPT-2: attention output projection
        "c_fc",             # GPT-2: FFN up projection
        # For LLaMA/Mistral:
        # "q_proj", "k_proj", "v_proj", "o_proj",
        # "gate_proj", "up_proj", "down_proj",
    ],
    
    # Don't use bias for DAPT
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Typical output: "trainable params: 0.4M || all params: 82M || trainable%: 0.49%"

# ─── Step 3: Prepare domain data ───
# [Same tokenization as Section 1]
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def tokenize_and_chunk(examples, block_size=512):
    tokenized = tokenizer(examples["text"], truncation=False)
    concatenated = sum(tokenized["input_ids"], [])
    total = (len(concatenated) // block_size) * block_size
    chunks = [concatenated[i:i+block_size] for i in range(0, total, block_size)]
    return {"input_ids": chunks, "labels": [c.copy() for c in chunks]}

lm_dataset = dataset.map(
    tokenize_and_chunk,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# ─── Step 4: Training arguments for LoRA-DAPT ───
training_args = TrainingArguments(
    output_dir=f"./lora_dapt_{domain}",
    
    # LoRA can use HIGHER learning rate than full DAPT
    learning_rate=5e-4,          # 10x higher than full DAPT
    
    num_train_epochs=3,           # Can train slightly longer
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,            # Standard clipping OK
    weight_decay=0.01,
    
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    
    fp16=torch.cuda.is_available(),
    
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
    
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["validation"],
    data_collator=data_collator,
)

# ─── Step 5: Train ───
trainer.train()

# ─── Step 6: Save LoRA adapter ───
model.save_pretrained(f"./lora_dapt_{domain}")
# Size: ~5-20 MB (vs 500MB-14GB for full model)

# ─── Step 7: Load for downstream use ───
# base_model = AutoModelForCausalLM.from_pretrained(model_name)
# adapted_model = PeftModel.from_pretrained(base_model, f"./lora_dapt_{domain}")
#
# Now you can:
# A) Fine-tune the adapted model on downstream task
# B) Add another LoRA on top for task-specific adaptation
# C) Merge the adapter into base: adapted_model.merge_and_unload()
'''
    
    print(lora_code)
    
    print(f"""
  ═══ LoRA-DAPT vs Full DAPT Comparison ═══
  
  ┌────────────────────┬──────────────┬──────────────┐
  │ Aspect             │ Full DAPT    │ LoRA-DAPT    │
  ├────────────────────┼──────────────┼──────────────┤
  │ Trainable params   │ 100%         │ 0.3-1%       │
  │ GPU memory         │ Full model   │ ~60% less    │
  │ Training speed     │ Baseline     │ 2-5x faster  │
  │ Storage per domain │ Full model   │ 5-20 MB      │
  │ Domain adaptation  │ Best         │ ~90% of full │
  │ General preserv.   │ Risk of loss │ Very safe    │
  │ Learning rate      │ 1e-5 to 5e-5 │ 1e-4 to 1e-3│
  │ Recommended for    │ Single domain│ Multi-domain │
  └────────────────────┴──────────────┴──────────────┘
  
  LoRA-DAPT is recommended for most practical scenarios!
""")


# ============================================================================
# SECTION 3: DATA PREPARATION FOR DAPT
# ============================================================================

def data_preparation():
    """Domain corpus preparation and quality filtering."""
    print("\n\n" + "=" * 70)
    print("  SECTION 3: DATA PREPARATION FOR DAPT")
    print("=" * 70)
    
    print(f"""
  ═══ Domain Corpus Preparation Pipeline ═══
  
  Raw domain text → Clean, deduplicated, tokenized training data
  
  ┌─────────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Raw Domain  │ →  │ Clean &  │ →  │ Dedup &  │ →  │ Tokenize │
  │ Corpus      │    │ Filter   │    │ Quality  │    │ & Chunk  │
  └─────────────┘    └──────────┘    └──────────┘    └──────────┘
""")
    
    data_prep_code = '''
# ═══════════════════════════════════════════════════════════════
# DATA PREPARATION PIPELINE FOR DAPT
# ═══════════════════════════════════════════════════════════════

import re
import hashlib
from collections import Counter
from typing import List, Dict, Set

# ─── Step 1: Basic Text Cleaning ───
def clean_text(text: str) -> str:
    """Clean a single document for language model training."""
    # Remove URLs
    text = re.sub(r'https?://\\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\\S+@\\S+', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\\s+', ' ', text).strip()
    
    # Remove very short lines (likely noise)
    lines = text.split('\\n')
    lines = [l for l in lines if len(l.split()) >= 5]
    text = '\\n'.join(lines)
    
    return text

# ─── Step 2: Quality Filtering ───
def quality_filter(text: str, min_words: int = 50,
                   max_words: int = 100000,
                   min_avg_word_len: float = 3.0,
                   max_special_ratio: float = 0.3) -> bool:
    """Filter out low-quality documents."""
    words = text.split()
    
    # Length filter
    if len(words) < min_words or len(words) > max_words:
        return False
    
    # Average word length (too short = likely noise)
    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len < min_avg_word_len:
        return False
    
    # Special character ratio
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special_chars / len(text) > max_special_ratio:
        return False
    
    # Repetition filter (detect boilerplate)
    unique_words = set(words)
    if len(unique_words) / len(words) < 0.1:  # >90% repeated
        return False
    
    return True

# ─── Step 3: Near-Deduplication ───
def compute_minhash(text: str, n_gram: int = 5, n_hashes: int = 128) -> List[int]:
    """Compute MinHash signature for near-duplicate detection."""
    # Create n-gram shingles
    words = text.lower().split()
    shingles = set()
    for i in range(len(words) - n_gram + 1):
        shingle = ' '.join(words[i:i+n_gram])
        shingles.add(shingle)
    
    if not shingles:
        return [0] * n_hashes
    
    # Compute MinHash
    minhash = []
    for seed in range(n_hashes):
        min_hash = float('inf')
        for shingle in shingles:
            h = int(hashlib.md5(
                f"{seed}:{shingle}".encode()).hexdigest(), 16) % (2**32)
            min_hash = min(min_hash, h)
        minhash.append(min_hash)
    
    return minhash

def estimate_jaccard(mh1: List[int], mh2: List[int]) -> float:
    """Estimate Jaccard similarity from MinHash signatures."""
    return sum(1 for a, b in zip(mh1, mh2) if a == b) / len(mh1)

def deduplicate(documents: List[str], threshold: float = 0.8) -> List[str]:
    """Remove near-duplicate documents using MinHash LSH."""
    signatures = [compute_minhash(doc) for doc in documents]
    
    keep = []
    seen_sigs: List[List[int]] = []
    
    for i, (doc, sig) in enumerate(zip(documents, signatures)):
        is_dup = False
        for seen_sig in seen_sigs:
            if estimate_jaccard(sig, seen_sig) > threshold:
                is_dup = True
                break
        
        if not is_dup:
            keep.append(doc)
            seen_sigs.append(sig)
    
    return keep

# ─── Step 4: Domain Relevance Filtering ───
def domain_relevance_score(text: str, domain_keywords: Set[str],
                           window_size: int = 100) -> float:
    """Score document relevance to target domain using keyword density."""
    words = text.lower().split()
    if len(words) < window_size:
        domain_words = sum(1 for w in words if w in domain_keywords)
        return domain_words / len(words) if words else 0
    
    # Sliding window: find densest region
    max_density = 0
    for i in range(0, len(words) - window_size, window_size // 2):
        window = words[i:i+window_size]
        density = sum(1 for w in window if w in domain_keywords) / window_size
        max_density = max(max_density, density)
    
    return max_density

# ─── Step 5: Full Pipeline ───
def prepare_domain_corpus(
    raw_documents: List[str],
    domain_keywords: Set[str],
    min_relevance: float = 0.01,
    dedup_threshold: float = 0.8,
) -> List[str]:
    """Full data preparation pipeline."""
    print(f"Starting with {len(raw_documents)} documents")
    
    # Clean
    cleaned = [clean_text(doc) for doc in raw_documents]
    cleaned = [doc for doc in cleaned if doc.strip()]
    print(f"After cleaning: {len(cleaned)}")
    
    # Quality filter
    quality = [doc for doc in cleaned if quality_filter(doc)]
    print(f"After quality filter: {len(quality)}")
    
    # Domain relevance
    relevant = [doc for doc in quality 
                if domain_relevance_score(doc, domain_keywords) >= min_relevance]
    print(f"After domain filter: {len(relevant)}")
    
    # Deduplicate
    deduped = deduplicate(relevant, threshold=dedup_threshold)
    print(f"After deduplication: {len(deduped)}")
    
    return deduped

# ─── Usage Example ───
# medical_keywords = {
#     "patient", "treatment", "diagnosis", "clinical", "disease",
#     "symptom", "therapy", "medication", "hospital", "physician",
#     "surgery", "chronic", "acute", "prognosis", "pathology",
# }
# 
# clean_corpus = prepare_domain_corpus(
#     raw_documents=raw_docs,
#     domain_keywords=medical_keywords,
#     min_relevance=0.01,
# )
'''
    
    print(data_prep_code)
    
    # Demonstrate with synthetic data
    print(f"\n  ── Simulated Data Preparation ──\n")
    
    torch.manual_seed(42)
    
    # Simulate a corpus with varying quality
    quality_dist = {
        "High quality, domain-relevant": 45,
        "Medium quality, domain-relevant": 20,
        "High quality, off-domain": 15,
        "Low quality (duplicates, noise)": 12,
        "Near-duplicates": 8,
    }
    
    total = sum(quality_dist.values())
    print(f"  Raw corpus: {total} documents")
    print()
    
    remaining = total
    for category, count in quality_dist.items():
        pct = count / total * 100
        if "Low quality" in category or "duplicate" in category:
            removed = count
            remaining -= removed
            status = f"→ REMOVED ({removed})"
        elif "off-domain" in category:
            removed = count
            remaining -= removed
            status = f"→ FILTERED ({removed})"
        else:
            status = f"→ KEPT ({count})"
        
        bar = "█" * int(pct / 2)
        print(f"  {category:>40}: {count:>3} ({pct:>4.0f}%) {bar} {status}")
    
    print(f"\n  Final clean corpus: {remaining} documents ({remaining/total:.0%} of original)")
    
    print(f"""
  ═══ Data Preparation Checklist ═══
  
  □ Remove HTML/markup tags
  □ Remove URLs and email addresses
  □ Normalize unicode and whitespace
  □ Filter by minimum length (50+ words)
  □ Filter by quality (word length, repetition, special chars)
  □ Filter by domain relevance (keyword density)
  □ Near-deduplicate using MinHash LSH
  □ Split into train/validation (95/5)
  □ Tokenize and chunk into fixed-length sequences
  □ Save in efficient format (Arrow, Parquet)
""")


# ============================================================================
# SECTION 4: MONITORING & EVALUATION
# ============================================================================

def monitoring_evaluation():
    """Tracking DAPT progress and evaluating adaptation quality."""
    print("\n\n" + "=" * 70)
    print("  SECTION 4: MONITORING & EVALUATION")
    print("=" * 70)
    
    print(f"""
  ═══ What to Monitor During DAPT ═══
  
  1. Domain Perplexity (PRIMARY): Should decrease
  2. General Perplexity (SECONDARY): Should not increase much
  3. Training Loss: Should decrease smoothly
  4. Learning Rate: Should follow schedule
  5. Gradient Norm: Should be stable
""")
    
    monitoring_code = '''
# ═══════════════════════════════════════════════════════════════
# DAPT MONITORING WITH CUSTOM CALLBACKS
# ═══════════════════════════════════════════════════════════════

from transformers import TrainerCallback, TrainerState, TrainerControl
import math
import json
from pathlib import Path

class DAPTMonitorCallback(TrainerCallback):
    """Monitor domain adaptation progress during DAPT."""
    
    def __init__(self, general_eval_dataset=None, tokenizer=None,
                 log_file="dapt_metrics.jsonl"):
        self.general_eval_dataset = general_eval_dataset
        self.tokenizer = tokenizer
        self.log_file = Path(log_file)
        self.metrics_history = []
        self.baseline_general_ppl = None
        self.baseline_domain_ppl = None
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log comprehensive metrics after each evaluation."""
        if metrics is None:
            return
        
        step = state.global_step
        domain_ppl = math.exp(metrics.get("eval_loss", 0))
        
        entry = {
            "step": step,
            "domain_ppl": domain_ppl,
            "domain_loss": metrics.get("eval_loss", 0),
            "train_loss": state.log_history[-1].get("loss", 0) if state.log_history else 0,
        }
        
        # Track baseline
        if self.baseline_domain_ppl is None:
            self.baseline_domain_ppl = domain_ppl
        
        entry["domain_ppl_reduction"] = (
            1 - domain_ppl / self.baseline_domain_ppl) * 100
        
        # Evaluate on general data if available
        if self.general_eval_dataset is not None:
            model = kwargs.get("model")
            if model is not None:
                model.eval()
                # Compute general perplexity
                # (simplified — in practice use Trainer.evaluate())
                entry["general_ppl"] = None  # Would compute here
        
        self.metrics_history.append(entry)
        
        # Log
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\\n")
        
        # Print summary
        print(f"\\n  Step {step}: Domain PPL={domain_ppl:.2f} "
              f"({entry['domain_ppl_reduction']:+.1f}% from baseline)")
        
        # Early stopping signal
        if entry["domain_ppl_reduction"] < -10:
            print("  ⚠ WARNING: Domain PPL increased >10% from baseline!")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Print final DAPT summary."""
        if not self.metrics_history:
            return
        
        first = self.metrics_history[0]
        last = self.metrics_history[-1]
        
        print(f"\\n{'='*50}")
        print(f"  DAPT COMPLETE")
        print(f"  Domain PPL: {first['domain_ppl']:.1f} → {last['domain_ppl']:.1f}")
        print(f"  Improvement: {last['domain_ppl_reduction']:+.1f}%")
        print(f"{'='*50}")


class EarlyStoppingOnForgetting(TrainerCallback):
    """Stop training if general capabilities degrade too much."""
    
    def __init__(self, max_general_ppl_increase: float = 0.20,
                 patience: int = 3):
        self.max_increase = max_general_ppl_increase
        self.patience = patience
        self.baseline_loss = None
        self.violations = 0
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        
        current_loss = metrics.get("eval_loss", 0)
        
        if self.baseline_loss is None:
            self.baseline_loss = current_loss
            return
        
        increase = (current_loss - self.baseline_loss) / self.baseline_loss
        
        if increase > self.max_increase:
            self.violations += 1
            print(f"  ⚠ Loss increased {increase:.1%} from baseline "
                  f"(violation {self.violations}/{self.patience})")
            
            if self.violations >= self.patience:
                print(f"  🛑 Stopping DAPT: too much forgetting!")
                control.should_training_stop = True
        else:
            self.violations = 0

# ─── Usage ───
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=domain_dataset,
#     eval_dataset=domain_val_dataset,
#     data_collator=data_collator,
#     callbacks=[
#         DAPTMonitorCallback(log_file="dapt_log.jsonl"),
#         EarlyStoppingOnForgetting(max_general_ppl_increase=0.20),
#     ],
# )
'''
    
    print(monitoring_code)
    
    # Simulate monitoring output
    print(f"\n  ── Simulated DAPT Monitoring Output ──\n")
    
    torch.manual_seed(42)
    
    steps = [0, 500, 1000, 1500, 2000, 2500, 3000]
    domain_ppls = [85.2, 62.1, 48.3, 41.7, 38.2, 36.1, 35.3]
    general_ppls = [22.1, 22.3, 22.8, 23.1, 23.5, 24.2, 24.8]
    
    print(f"  {'Step':>6} │ {'Domain PPL':>10} │ {'General PPL':>11} │ {'Δ Domain':>8} │ {'Δ General':>9} │ {'Status':>8}")
    print(f"  {'─'*6}─┼─{'─'*10}─┼─{'─'*11}─┼─{'─'*8}─┼─{'─'*9}─┼─{'─'*8}")
    
    for step, d_ppl, g_ppl in zip(steps, domain_ppls, general_ppls):
        d_delta = (1 - d_ppl / domain_ppls[0]) * 100
        g_delta = (g_ppl / general_ppls[0] - 1) * 100
        
        if g_delta > 15:
            status = "⚠ WARN"
        elif g_delta > 20:
            status = "🛑 STOP"
        else:
            status = "✓ OK"
        
        print(f"  {step:>6} │ {d_ppl:>10.1f} │ {g_ppl:>11.1f} │ {d_delta:>+7.1f}% │ {g_delta:>+8.1f}% │ {status:>8}")
    
    print(f"""
  ═══ Evaluation after DAPT ═══
  
  After DAPT, evaluate on these benchmarks:
  
  1. Domain perplexity (held-out domain text)
  2. General perplexity (WikiText, C4 sample)
  3. Downstream task performance (before fine-tuning)
  4. Domain-specific probing tasks:
     - Fill-in-the-blank with domain terms
     - Domain QA (if available)
     - Domain text generation quality
""")


# ============================================================================
# SECTION 5: MULTI-STAGE PIPELINE (DAPT → TAPT → TASK FT)
# ============================================================================

def multi_stage_pipeline():
    """Complete multi-stage adaptation: DAPT → TAPT → Task Fine-Tuning."""
    print("\n\n" + "=" * 70)
    print("  SECTION 5: MULTI-STAGE PIPELINE (DAPT → TAPT → TASK FT)")
    print("=" * 70)
    
    print(f"""
  ═══ The Full Adaptation Pipeline ═══
  
  Gururangan et al. (2020) showed that combining DAPT and TAPT
  gives the BEST results. The full pipeline:
  
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Base LLM │ →  │  DAPT    │ →  │  TAPT    │ →  │  Task    │
  │          │    │ (domain) │    │ (task)   │    │  FT      │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
       │               │               │               │
   General          Domain-         Task-          Task-
   knowledge        adapted         adapted        specialized
  
  Each stage narrows the distribution:
  General → Domain → Task-specific → Labeled task
""")
    
    pipeline_code = '''
# ═══════════════════════════════════════════════════════════════
# MULTI-STAGE PIPELINE: DAPT → TAPT → Task Fine-Tuning
# ═══════════════════════════════════════════════════════════════

from transformers import (
    AutoModelForCausalLM, AutoModelForSequenceClassification,
    AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset
import torch

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ═══════════════════════════════════════════════════
# STAGE 1: DAPT (Domain-Adaptive Pretraining)
# ═══════════════════════════════════════════════════

print("=== Stage 1: DAPT ===")

# Load base model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Apply LoRA for DAPT
dapt_lora = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,                   # Higher rank for domain adaptation
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"],
)
model = get_peft_model(model, dapt_lora)

# Load domain corpus (e.g., biomedical papers)
# domain_data = load_dataset("text", data_files="biomedical_corpus.txt")
# ... tokenize and train ...

# Save DAPT adapter
model.save_pretrained("./stage1_dapt_adapter")

# Merge DAPT adapter into base for next stage
model = model.merge_and_unload()
model.save_pretrained("./stage1_dapt_merged")

# ═══════════════════════════════════════════════════
# STAGE 2: TAPT (Task-Adaptive Pretraining)
# ═══════════════════════════════════════════════════

print("=== Stage 2: TAPT ===")

# Load DAPT-adapted model
model = AutoModelForCausalLM.from_pretrained("./stage1_dapt_merged")

# Apply NEW LoRA for TAPT (can be lower rank)
tapt_lora = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                   # Lower rank: task data is smaller
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["c_attn"],
)
model = get_peft_model(model, tapt_lora)

# Load task-specific UNLABELED data
# task_data = load_dataset("text", data_files="task_unlabeled.txt")
# ... tokenize and train ...

# Merge TAPT adapter
model = model.merge_and_unload()
model.save_pretrained("./stage2_tapt_merged")

# ═══════════════════════════════════════════════════
# STAGE 3: Task Fine-Tuning (Supervised)
# ═══════════════════════════════════════════════════

print("=== Stage 3: Task Fine-Tuning ===")

# Load from classification model (add classification head)
model = AutoModelForSequenceClassification.from_pretrained(
    "./stage2_tapt_merged",
    num_labels=2,  # Binary classification example
)

# Apply LoRA for task fine-tuning
task_lora = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                    # Lowest rank: labeled data is smallest
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["c_attn"],
)
model = get_peft_model(model, task_lora)

# Load labeled task data
# task_dataset = load_dataset("csv", data_files="task_labeled.csv")
# ... tokenize and train with Trainer ...

# Save final model
model.save_pretrained("./stage3_task_ft")
print("Multi-stage pipeline complete!")
'''
    
    print(pipeline_code)
    
    # Show expected improvements at each stage
    print(f"""
  ═══ Expected Improvements at Each Stage ═══
  
  Example: Biomedical Sentiment Classification
  
  ┌──────────────────┬────────────┬──────────────────────────────┐
  │ Stage            │ Accuracy   │ Improvement                  │
  ├──────────────────┼────────────┼──────────────────────────────┤
  │ Base + Task FT   │ 82.3%      │ Baseline                     │
  │ DAPT + Task FT   │ 86.7%      │ +4.4% (domain knowledge)     │
  │ TAPT + Task FT   │ 84.1%      │ +1.8% (task distribution)    │
  │ DAPT+TAPT+FT     │ 88.2%      │ +5.9% (both combined!) ★    │
  └──────────────────┴────────────┴──────────────────────────────┘
  
  Key: DAPT and TAPT are COMPLEMENTARY, not redundant!
  
  ═══ LoRA Configuration per Stage ═══
  
  ┌─────────┬──────┬──────────┬──────────┬───────────────────────┐
  │ Stage   │ Rank │ LR       │ Epochs   │ Data Size             │
  ├─────────┼──────┼──────────┼──────────┼───────────────────────┤
  │ DAPT    │ 32   │ 5e-4     │ 1-3      │ 100M-1B tokens        │
  │ TAPT    │ 16   │ 3e-4     │ 3-5      │ 1M-50M tokens         │
  │ Task FT │ 8    │ 2e-4     │ 3-10     │ 1K-100K examples      │
  └─────────┴──────┴──────────┴──────────┴───────────────────────┘
  
  Pattern: Rank DECREASES, epochs INCREASE, data size DECREASES
  as we go from broad adaptation to specific task learning.
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  DAPT TRAINING — PRODUCTION PIPELINE WITH HUGGINGFACE           ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    full_dapt_pipeline()
    lora_dapt_with_peft()
    data_preparation()
    monitoring_evaluation()
    multi_stage_pipeline()
    
    print("\n" + "=" * 70)
    print("  TRAINING MODULE COMPLETE")
    print("=" * 70)
    print("""
    Covered:
    ✓ Full DAPT pipeline with HuggingFace Trainer
    ✓ LoRA-DAPT with PEFT library (parameter-efficient)
    ✓ Domain corpus preparation (cleaning, dedup, quality filter)
    ✓ DAPT monitoring with custom callbacks
    ✓ Multi-stage pipeline: DAPT → TAPT → Task Fine-Tuning
    """)


if __name__ == "__main__":
    main()
