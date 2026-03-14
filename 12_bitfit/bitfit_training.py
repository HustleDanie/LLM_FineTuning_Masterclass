"""
BitFit Training — Practical Training Pipelines
===============================================

Complete training implementations for BitFit:

1. Custom Training Loop
   - Manual bias-only training on GPT-2
   - Learning rate scheduling, gradient monitoring

2. HuggingFace Trainer Pipeline
   - BitFit with Trainer API
   - Callbacks for bias monitoring

3. Few-Shot BitFit
   - BitFit with minimal data (4-32 examples)
   - Why bias-only excels with few examples

4. BitFit for Classification
   - Sequence classification with frozen backbone
   - Adding a classification head

5. Hyperparameter Guide
   - Optimal LR, batch size, epochs
   - Per-model-family recommendations

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional


# ============================================================================
# SECTION 1: CUSTOM TRAINING LOOP
# ============================================================================

def custom_training_loop():
    """Full custom training loop for BitFit."""
    print("=" * 65)
    print("  SECTION 1: CUSTOM TRAINING LOOP")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # ─── Step 1: Apply BitFit ───
    for param in model.parameters():
        param.requires_grad = False
    
    trainable_params = []
    for name, param in model.named_parameters():
        if "bias" in name:
            param.requires_grad = True
            trainable_params.append(param)
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in trainable_params)
    print(f"\n  BitFit applied: {trainable:,} / {total:,} params trainable "
          f"({trainable/total*100:.4f}%)")
    
    # ─── Step 2: Dataset ───
    texts = [
        "Machine learning models learn from data patterns.",
        "Natural language processing is a subfield of AI.",
        "Transformers use attention mechanisms for sequence modeling.",
        "Fine-tuning adapts pretrained models to specific tasks.",
        "Parameter efficiency reduces computational requirements.",
        "Bias terms control activation thresholds in networks.",
        "Deep learning has revolutionized computer vision and NLP.",
        "Transfer learning leverages pretrained representations.",
    ] * 5  # Repeat for more training data
    
    inputs = tokenizer(
        texts, padding="max_length", truncation=True,
        max_length=32, return_tensors="pt",
    )
    labels = inputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    dataset_size = len(texts)
    batch_size = 8
    
    # ─── Step 3: Optimizer & Scheduler ───
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=1e-2,            # Higher LR for bias terms
        weight_decay=0.0,   # No weight decay for biases
        betas=(0.9, 0.999),
    )
    
    num_epochs = 6
    total_steps = (dataset_size // batch_size) * num_epochs
    
    # Cosine LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-4,
    )
    
    # ─── Step 4: Training ───
    print(f"\n  Training: {num_epochs} epochs, batch_size={batch_size}, "
          f"lr=1e-2 → 1e-4")
    print(f"  {'Epoch':>6} {'Loss':>10} {'LR':>12} {'Grad Norm':>12} {'Bias Δ':>10}")
    print(f"  {'─'*6}─{'─'*10}─{'─'*12}─{'─'*12}─{'─'*10}")
    
    model.train()
    
    # Record initial biases
    initial_biases = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_biases[name] = param.data.clone()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_grad_norm = 0
        num_batches = 0
        
        # Shuffle indices
        indices = torch.randperm(dataset_size)
        
        for i in range(0, dataset_size, batch_size):
            batch_idx = indices[i:i+batch_size]
            if len(batch_idx) < 2:
                continue
            
            batch_ids = inputs["input_ids"][batch_idx]
            batch_mask = inputs["attention_mask"][batch_idx]
            batch_labels = labels[batch_idx]
            
            out = model(
                input_ids=batch_ids,
                attention_mask=batch_mask,
                labels=batch_labels,
            )
            
            out.loss.backward()
            
            # Compute gradient norm for bias params
            grad_norm = torch.sqrt(sum(
                p.grad.norm() ** 2
                for p in trainable_params
                if p.grad is not None
            )).item()
            
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += out.loss.item()
            epoch_grad_norm += grad_norm
            num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_grad = epoch_grad_norm / max(num_batches, 1)
        
        # Measure total bias change from init
        total_delta = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_delta += (param.data - initial_biases[name]).abs().mean().item()
        num_bias_sets = len(initial_biases)
        avg_delta = total_delta / max(num_bias_sets, 1)
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"  {epoch+1:>6} {avg_loss:>10.4f} {current_lr:>12.6f} "
              f"{avg_grad:>12.4f} {avg_delta:>10.6f}")
    
    # ─── Step 5: Inference ───
    print(f"\n  ── Inference ──")
    model.eval()
    
    prompts = [
        "Machine learning",
        "The transformer model",
        "Fine-tuning is",
    ]
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"    \"{prompt}\" → {text}")
    
    del model


# ============================================================================
# SECTION 2: HUGGINGFACE TRAINER PIPELINE
# ============================================================================

def trainer_pipeline():
    """BitFit with HuggingFace Trainer API."""
    print("\n\n" + "=" * 65)
    print("  SECTION 2: HUGGINGFACE TRAINER PIPELINE")
    print("=" * 65)
    
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
        TrainerCallback,
    )
    from datasets import Dataset
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Apply BitFit
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "bias" in name:
            param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  BitFit: {trainable:,} / {total:,} trainable ({trainable/total*100:.4f}%)")
    
    # Dataset
    texts = [
        "The study of artificial intelligence has made great progress.",
        "Deep learning uses neural networks with many layers.",
        "Attention mechanisms help models focus on relevant information.",
        "Transfer learning leverages knowledge from pretrained models.",
        "Language models predict the next word in a sequence.",
        "Fine-tuning adapts a general model to a specific domain.",
        "Gradient descent optimizes model parameters iteratively.",
        "Regularization helps prevent overfitting to training data.",
    ] * 8
    
    dataset = Dataset.from_dict({"text": texts})
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=48,
            padding="max_length",
        )
    
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    # Custom callback to monitor bias changes
    class BitFitMonitorCallback(TrainerCallback):
        def __init__(self, model):
            self.initial_biases = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.initial_biases[name] = param.data.clone()
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                model = kwargs.get("model")
                if model:
                    total_delta = 0
                    for name, param in model.named_parameters():
                        if name in self.initial_biases:
                            delta = (param.data - self.initial_biases[name]).abs().mean()
                            total_delta += delta.item()
                    avg = total_delta / max(len(self.initial_biases), 1)
                    print(f"    [BitFit] Avg bias Δ from init: {avg:.6f}")
    
    # Training arguments
    import tempfile, os
    output_dir = os.path.join(tempfile.gettempdir(), "bitfit_trainer")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=1e-2,           # High LR for BitFit
        weight_decay=0.0,             # No weight decay
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        callbacks=[BitFitMonitorCallback(model)],
    )
    
    print(f"\n  Training with Trainer API...")
    trainer.train()
    
    print(f"\n  ✓ Trainer pipeline complete!")
    
    # Cleanup
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    
    del model, trainer


# ============================================================================
# SECTION 3: FEW-SHOT BITFIT
# ============================================================================

def few_shot_bitfit():
    """BitFit with very few training examples."""
    print("\n\n" + "=" * 65)
    print("  SECTION 3: FEW-SHOT BITFIT")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Few-shot dataset (only 8 examples!)
    few_shot_data = [
        "Q: What is machine learning? A: A method where computers learn from data.",
        "Q: What is a neural network? A: A computing system inspired by the brain.",
        "Q: What is NLP? A: Processing and understanding human language with AI.",
        "Q: What is fine-tuning? A: Adapting a pretrained model to a specific task.",
        "Q: What is a transformer? A: A model architecture using attention mechanisms.",
        "Q: What is transfer learning? A: Reusing knowledge from one task for another.",
        "Q: What is backpropagation? A: Computing gradients to update model weights.",
        "Q: What is overfitting? A: When a model memorizes training data too closely.",
    ]
    
    print(f"\n  Training data: {len(few_shot_data)} examples (few-shot!)")
    
    inputs = tokenizer(
        few_shot_data,
        padding="max_length",
        truncation=True,
        max_length=48,
        return_tensors="pt",
    )
    labels = inputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    # Compare: BitFit vs LoRA on few-shot
    results = {}
    
    # ─── BitFit ───
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "bias" in name:
            param.requires_grad = True
    
    trainable_bf = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-2,
    )
    
    model.train()
    bf_losses = []
    for epoch in range(20):  # More epochs for few-shot
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        bf_losses.append(out.loss.item())
    
    results["BitFit"] = {
        "params": trainable_bf,
        "losses": bf_losses,
        "final_loss": bf_losses[-1],
    }
    del model, opt
    
    # ─── LoRA ───
    from peft import get_peft_model, LoraConfig, TaskType
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16,
        target_modules=["c_attn"],
    ))
    
    trainable_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4,
    )
    
    model.train()
    lora_losses = []
    for epoch in range(20):
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        lora_losses.append(out.loss.item())
    
    results["LoRA (r=8)"] = {
        "params": trainable_lora,
        "losses": lora_losses,
        "final_loss": lora_losses[-1],
    }
    del model, opt
    
    # Results
    print(f"\n  Few-Shot Results (8 examples, 20 epochs):")
    print(f"\n  {'Method':<16} {'Params':>10} {'Start Loss':>12} {'Final Loss':>12} {'Δ':>10}")
    print(f"  {'─'*16}─{'─'*10}─{'─'*12}─{'─'*12}─{'─'*10}")
    
    for name, r in results.items():
        start = r["losses"][0]
        final = r["final_loss"]
        delta = start - final
        print(f"  {name:<16} {r['params']:>10,} {start:>12.4f} {final:>12.4f} {delta:>10.4f}")
    
    print(f"""
  ═══ Few-Shot Analysis ═══
  
  With only 8 examples:
  • BitFit ({trainable_bf:,} params) — Strong regularization prevents overfitting
  • LoRA ({trainable_lora:,} params) — More params, risk of memorization
  
  Why BitFit excels in few-shot:
  1. Extreme parameter efficiency → very little to overfit
  2. Bias gradients are input-independent → cleaner signal
  3. Implicit feature selection → leverages pretrained knowledge
  4. Higher learning rate → faster convergence on few samples
""")


# ============================================================================
# SECTION 4: BITFIT FOR CLASSIFICATION
# ============================================================================

def bitfit_classification():
    """BitFit for sequence classification task."""
    print("\n\n" + "=" * 65)
    print("  SECTION 4: BITFIT FOR CLASSIFICATION")
    print("=" * 65)
    
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Apply BitFit: freeze all, unfreeze biases + classification head
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze bias terms in transformer
    for name, param in model.named_parameters():
        if "bias" in name:
            param.requires_grad = True
    
    # Also unfreeze classification head (newly initialized)
    for param in model.score.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"\n  Classification model (2 classes):")
    print(f"    Total: {total:,}")
    print(f"    Trainable: {trainable:,} ({trainable/total*100:.4f}%)")
    print(f"    (includes bias terms + classification head)")
    
    # Synthetic binary classification data
    positive_texts = [
        "This movie was fantastic and enjoyable.",
        "I loved every moment of this experience.",
        "An excellent performance by the entire cast.",
        "Highly recommended, truly wonderful work.",
        "Beautiful and inspiring, a masterpiece.",
    ] * 4
    
    negative_texts = [
        "This was terrible and a waste of time.",
        "I really disliked the poor quality.",
        "An awful experience from start to finish.",
        "Completely disappointing and frustrating.",
        "Boring and unimaginative, avoid this.",
    ] * 4
    
    all_texts = positive_texts + negative_texts
    all_labels = [1] * len(positive_texts) + [0] * len(negative_texts)
    
    # Shuffle
    import random
    random.seed(42)
    combined = list(zip(all_texts, all_labels))
    random.shuffle(combined)
    all_texts, all_labels = zip(*combined)
    
    inputs = tokenizer(
        list(all_texts),
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    labels = torch.tensor(all_labels)
    
    # Train
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-3,
    )
    
    model.train()
    batch_size = 10
    
    print(f"\n  Training on {len(all_texts)} examples ({len(positive_texts)} pos, "
          f"{len(negative_texts)} neg)")
    print(f"\n  {'Epoch':>6} {'Loss':>10} {'Accuracy':>10}")
    print(f"  {'─'*6}─{'─'*10}─{'─'*10}")
    
    for epoch in range(10):
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for i in range(0, len(all_texts), batch_size):
            batch_ids = inputs["input_ids"][i:i+batch_size]
            batch_mask = inputs["attention_mask"][i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            out = model(
                input_ids=batch_ids,
                attention_mask=batch_mask,
                labels=batch_labels,
            )
            
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += out.loss.item()
            preds = out.logits.argmax(dim=-1)
            correct += (preds == batch_labels).sum().item()
            total_samples += len(batch_labels)
        
        avg_loss = total_loss / (len(all_texts) // batch_size)
        accuracy = correct / total_samples
        
        if (epoch + 1) % 2 == 0:
            print(f"  {epoch+1:>6} {avg_loss:>10.4f} {accuracy:>9.1%}")
    
    # Test inference
    print(f"\n  ── Classification Inference ──")
    model.eval()
    
    test_texts = [
        "This is wonderful and amazing!",
        "Terrible quality, very disappointing.",
        "A great and delightful experience.",
        "I hated every minute of it.",
    ]
    
    test_inputs = tokenizer(
        test_texts, padding=True, truncation=True,
        max_length=32, return_tensors="pt",
    )
    
    with torch.no_grad():
        out = model(**test_inputs)
        probs = F.softmax(out.logits, dim=-1)
        preds = out.logits.argmax(dim=-1)
    
    label_map = {0: "Negative", 1: "Positive"}
    for text, pred, prob in zip(test_texts, preds, probs):
        print(f"    \"{text[:40]}...\"")
        print(f"      → {label_map[pred.item()]} "
              f"(confidence: {prob[pred].item():.1%})")
    
    del model


# ============================================================================
# SECTION 5: HYPERPARAMETER GUIDE
# ============================================================================

def hyperparameter_guide():
    """Optimal hyperparameters for BitFit across model families."""
    print("\n\n" + "=" * 65)
    print("  SECTION 5: HYPERPARAMETER GUIDE")
    print("=" * 65)
    
    print(f"""
  ═══════════════════════════════════════════════════════════════
   BITFIT HYPERPARAMETER GUIDE
  ═══════════════════════════════════════════════════════════════
  
  ┌─────────────────┬────────────────────────────────────────────┐
  │ Parameter       │ Recommendation                             │
  ├─────────────────┼────────────────────────────────────────────┤
  │ Learning Rate   │ 1e-3 to 5e-2 (much higher than full FT)   │
  │                 │ Start with 1e-2 for most models            │
  ├─────────────────┼────────────────────────────────────────────┤
  │ Weight Decay    │ 0.0 (biases should NOT be decayed)         │
  │                 │ This is critical — decay hurts BitFit      │
  ├─────────────────┼────────────────────────────────────────────┤
  │ Batch Size      │ 8-32 (smaller often better for few-shot)   │
  ├─────────────────┼────────────────────────────────────────────┤
  │ Epochs          │ 3-20 depending on dataset size             │
  │                 │ Few-shot: 10-20 epochs                     │
  │                 │ Large data: 3-5 epochs                     │
  ├─────────────────┼────────────────────────────────────────────┤
  │ Warmup          │ 5-10% of total steps                       │
  ├─────────────────┼────────────────────────────────────────────┤
  │ Scheduler       │ Cosine or linear decay                     │
  ├─────────────────┼────────────────────────────────────────────┤
  │ Grad Clipping   │ 1.0 (important with high LR)              │
  ├─────────────────┼────────────────────────────────────────────┤
  │ Optimizer       │ AdamW (β₁=0.9, β₂=0.999)                  │
  └─────────────────┴────────────────────────────────────────────┘
  
  
  ═══ Learning Rate Comparison Across Methods ═══
  
  ┌─────────────────┬──────────────┬─────────────────────────────┐
  │ Method          │ Typical LR   │ Why?                        │
  ├─────────────────┼──────────────┼─────────────────────────────┤
  │ Full Fine-Tune  │ 1e-5 — 5e-5  │ Many params, easy to break  │
  │ LoRA            │ 1e-4 — 3e-4  │ Low-rank, moderate params   │
  │ IA³             │ 1e-2 — 5e-2  │ Few params, rescaling       │
  │ BitFit          │ 1e-3 — 5e-2  │ Very few params, shifts     │
  │ Prompt Tuning   │ 1e-3 — 1e-1  │ Soft tokens, continuous     │
  │ Prefix Tuning   │ 1e-3 — 5e-3  │ Virtual prefix tokens       │
  └─────────────────┴──────────────┴─────────────────────────────┘
  
  Rule of thumb: Fewer trainable params → Higher learning rate needed
  
  
  ═══ Per-Model-Family Settings ═══
  
  GPT-2 / DistilGPT-2:
    • target: All bias terms
    • lr: 1e-2
    • Note: GPT-2 Conv1D has biases in c_attn, c_proj, c_fc
    
  LLaMA / Mistral:
    • Note: LLaMA does NOT have bias terms by default!
    • Need to check model config for bias=True/False
    • If no biases → BitFit is NOT applicable
    • Alternative: Train LayerNorm/RMSNorm parameters only
    
  BERT / RoBERTa:
    • target: All bias terms + LayerNorm β
    • lr: 5e-3 to 2e-2
    • Excellent results on GLUE/SuperGLUE
    
  T5 / FLAN-T5:
    • Note: T5 uses relative bias in attention (special case)
    • Train layer_norm.bias (T5 uses simplified LN)
    • lr: 1e-3 to 1e-2
    
  
  ═══ Important Caveats ═══
  
  1. NOT ALL MODELS HAVE BIASES!
     - LLaMA, Falcon: bias=False by default
     - Check: model.config.use_bias or inspect layers
     - For bias-free models: adapt by training LN params
  
  2. WEIGHT DECAY = 0 IS CRITICAL
     - Bias terms are small; decay pushes them toward zero
     - Zero is the pretrained value → decay undoes training!
     - Always set weight_decay=0.0 for BitFit
  
  3. GRADIENT CLIPPING MATTERS
     - High LR + bias gradients can cause instability
     - Always clip to 1.0 or 0.5
  
  4. BITFIT COMPLEMENTS OTHER METHODS
     - LoRA + BitFit: PEFT `bias="all"` parameter
     - IA³ + BitFit: Manually unfreeze biases after applying IA³
""")
    
    # Demonstrate checking for bias availability
    print("  ── Checking Model for Bias Terms ──")
    
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    has_bias = False
    bias_count = 0
    no_bias_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                has_bias = True
                bias_count += 1
            else:
                no_bias_layers.append(name)
    
    print(f"\n  Model: distilgpt2")
    print(f"  Has bias terms: {'Yes ✓' if has_bias else 'No ✗'}")
    print(f"  Linear layers with bias: {bias_count}")
    if no_bias_layers:
        print(f"  Linear layers WITHOUT bias: {len(no_bias_layers)}")
    
    print(f"\n  → BitFit is {'applicable ✓' if has_bias else 'NOT applicable ✗'}"
          f" for this model")
    
    del model


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all training pipelines."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║        BitFit TRAINING — PRACTICAL TRAINING PIPELINES        ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: Custom loop
    custom_training_loop()
    
    # Section 2: Trainer API
    trainer_pipeline()
    
    # Section 3: Few-shot
    few_shot_bitfit()
    
    # Section 4: Classification
    bitfit_classification()
    
    # Section 5: Hyperparameters
    hyperparameter_guide()
    
    print("\n" + "=" * 65)
    print("  TRAINING MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered:
    ✓ Custom training loop with gradient monitoring
    ✓ HuggingFace Trainer pipeline with callbacks
    ✓ Few-shot BitFit (8 examples)
    ✓ Classification with BitFit backbone
    ✓ Hyperparameter guide (LR, WD, scheduling, caveats)
    """)


if __name__ == "__main__":
    main()
