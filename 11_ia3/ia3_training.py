"""
IA³ — Training with HuggingFace PEFT Library
===============================================

Production-ready training pipelines:

1. IA³ Configuration with PEFT
   - IA3Config setup and options
   - Target module selection

2. Complete Training Pipeline
   - Data preparation
   - Custom training loop
   - Save/load workflow

3. Training with Trainer API
   - HuggingFace Trainer integration
   - Evaluation and metrics

4. Few-Shot Training
   - IA³'s sweet spot: 4-64 examples
   - Comparison with in-context learning

5. Hyperparameter Guide
   - Optimal settings for IA³
   - Task-specific recommendations

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import os
from typing import Optional, Dict


# ============================================================================
# SECTION 1: IA³ CONFIGURATION WITH PEFT
# ============================================================================

def setup_ia3_basic():
    """
    Configure IA³ using HuggingFace PEFT.
    """
    print("=" * 65)
    print("  SECTION 1: IA³ CONFIGURATION WITH PEFT")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, IA3Config, TaskType
    
    model_name = "distilgpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # ─── IA³ Config ───
    ia3_config = IA3Config(
        task_type=TaskType.CAUSAL_LM,
        
        # Target modules for IA³ rescaling
        # For GPT-2: c_attn (K, V) and mlp.c_fc (FF intermediate)
        target_modules=["c_attn", "mlp.c_fc"],
        
        # Which modules should have their inputs fed-forward
        # (controls where the rescaling vector is applied)
        feedforward_modules=["mlp.c_fc"],
        
        # Initialize IA³ vectors
        init_ia3_weights=True,  # True = initialize to ones (identity)
    )
    
    print(f"\n  IA³ Config:")
    print(f"  ─────────────────────────────────")
    print(f"  Target modules:      {ia3_config.target_modules}")
    print(f"  FF modules:          {ia3_config.feedforward_modules}")
    print(f"  Init to ones:        {ia3_config.init_ia3_weights}")
    print(f"  Task type:           {ia3_config.task_type}")
    
    # Apply PEFT
    model = get_peft_model(model, ia3_config)
    model.print_trainable_parameters()
    
    # Show trainable parameters
    print(f"\n  Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"    {name}: {param.shape} (init={param.data[:3].tolist()}...)")
    
    # Test forward pass
    text = "IA³ uses learned rescaling vectors to adapt"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    
    print(f"\n  Test forward pass:")
    print(f"  Input: '{text}'")
    print(f"  Loss: {outputs.loss.item():.4f}")
    
    del model
    return ia3_config


def setup_ia3_variants():
    """
    Show different IA³ configurations.
    """
    print("\n\n  ─── IA³ Variants ───")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, IA3Config, TaskType
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    configs = {
        "Attention only (K,V)": IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["c_attn"],
            feedforward_modules=[],
        ),
        "FF only": IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["mlp.c_fc"],
            feedforward_modules=["mlp.c_fc"],
        ),
        "Full (K,V + FF)": IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["c_attn", "mlp.c_fc"],
            feedforward_modules=["mlp.c_fc"],
        ),
    }
    
    print(f"\n  {'Config':<25} {'Trainable':>12} {'%':>8}")
    print(f"  {'─'*25}─{'─'*12}─{'─'*8}")
    
    for name, config in configs.items():
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.config.pad_token_id = tokenizer.pad_token_id
        model = get_peft_model(model, config)
        
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  {name:<25} {trainable:>12,} {trainable/total*100:>7.4f}%")
        del model


# ============================================================================
# SECTION 2: COMPLETE TRAINING PIPELINE
# ============================================================================

def create_dataset(tokenizer, num_samples=100, max_length=48):
    """Create a synthetic dataset."""
    import random
    random.seed(42)
    
    positive = [
        "This product is amazing and works perfectly.",
        "I absolutely love this, best purchase ever.",
        "Outstanding quality and fast delivery.",
        "Exceeds all my expectations, highly recommend.",
        "Incredible value for money, very satisfied.",
    ]
    
    negative = [
        "This product is terrible, complete waste of money.",
        "I regret buying this, awful quality.",
        "Horrible experience, would not recommend.",
        "Worst purchase I have ever made.",
        "Completely disappointed, does not work at all.",
    ]
    
    texts = []
    for _ in range(num_samples):
        if random.random() > 0.5:
            text = random.choice(positive)
            label = "positive"
        else:
            text = random.choice(negative)
            label = "negative"
        texts.append(f"Review: {text} Sentiment: {label}")
    
    encodings = tokenizer(
        texts, padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    
    labels = encodings["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    return torch.utils.data.TensorDataset(
        encodings["input_ids"], encodings["attention_mask"], labels,
    )


def train_ia3_custom_loop():
    """
    Train IA³ with a custom training loop.
    """
    print("\n\n" + "=" * 65)
    print("  SECTION 2: CUSTOM TRAINING LOOP")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, IA3Config, TaskType
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    ia3_config = IA3Config(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn", "mlp.c_fc"],
        feedforward_modules=["mlp.c_fc"],
    )
    
    model = get_peft_model(model, ia3_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"\n  Device: {device}")
    model.print_trainable_parameters()
    
    # Dataset
    dataset = create_dataset(tokenizer, num_samples=80, max_length=48)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=8)
    
    # Optimizer — IA³ uses higher LR (fewer params)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-2,          # High LR for IA³
        weight_decay=0.0,  # No weight decay (vectors should stay near 1)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    # Training
    print(f"\n  Training IA³...")
    print(f"  {'Epoch':>7} {'Train Loss':>12} {'Val Loss':>12} {'LR':>12}")
    print(f"  {'─'*7}─{'─'*12}─{'─'*12}─{'─'*12}")
    
    for epoch in range(5):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            ids, mask, labs = [b.to(device) for b in batch]
            outputs = model(input_ids=ids, attention_mask=mask, labels=labs)
            
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += outputs.loss.item()
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                ids, mask, labs = [b.to(device) for b in batch]
                outputs = model(input_ids=ids, attention_mask=mask, labels=labs)
                val_loss += outputs.loss.item()
        val_loss /= len(val_loader)
        
        lr = scheduler.get_last_lr()[0]
        print(f"  {epoch+1:>7} {train_loss:>12.4f} {val_loss:>12.4f} {lr:>12.6f}")
        scheduler.step()
    
    # Save
    save_dir = "./ia3_checkpoint"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    if os.path.exists(save_dir):
        files = os.listdir(save_dir)
        total_size = sum(
            os.path.getsize(os.path.join(save_dir, f))
            for f in files if os.path.isfile(os.path.join(save_dir, f))
        )
        print(f"\n  Saved to {save_dir}/")
        print(f"  Files: {len(files)} | Size: {total_size / 1024:.1f} KB")
    
    # Load and inference
    print(f"\n  Loading for inference...")
    from peft import PeftModel
    
    base = AutoModelForCausalLM.from_pretrained(model_name)
    base.config.pad_token_id = tokenizer.pad_token_id
    loaded = PeftModel.from_pretrained(base, save_dir)
    loaded = loaded.to(device)
    loaded.eval()
    
    prompt = "Review: This product is absolutely wonderful"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = loaded.generate(
            **inputs, max_new_tokens=15, do_sample=False,
        )
    
    print(f"  Prompt: '{prompt}'")
    print(f"  Output: '{tokenizer.decode(output[0], skip_special_tokens=True)}'")
    
    # Cleanup
    import shutil
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    
    del model, loaded
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================================
# SECTION 3: TRAINER API PIPELINE
# ============================================================================

def train_with_trainer():
    """
    Training IA³ with HuggingFace Trainer.
    """
    print("\n\n" + "=" * 65)
    print("  SECTION 3: TRAINING WITH TRAINER API")
    print("=" * 65)
    
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import get_peft_model, IA3Config, TaskType
    from datasets import Dataset
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    ia3_config = IA3Config(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn", "mlp.c_fc"],
        feedforward_modules=["mlp.c_fc"],
    )
    
    model = get_peft_model(model, ia3_config)
    model.print_trainable_parameters()
    
    # Dataset
    texts = [
        "Machine learning models learn from data.",
        "Deep learning uses neural network architectures.",
        "Fine-tuning adapts pretrained models to tasks.",
        "IA3 learns rescaling vectors for adaptation.",
        "Parameter efficiency reduces computational costs.",
    ] * 20
    
    def tokenize(examples):
        return tokenizer(
            examples["text"], padding="max_length",
            truncation=True, max_length=48,
        )
    
    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    split = tokenized.train_test_split(test_size=0.2, seed=42)
    
    training_args = TrainingArguments(
        output_dir="./ia3_trainer",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=1e-2,         # High LR for IA³
        weight_decay=0.0,           # No weight decay
        warmup_steps=5,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    print(f"\n  Starting Trainer...")
    result = trainer.train()
    
    print(f"\n  Training complete!")
    print(f"  Final loss: {result.training_loss:.4f}")
    
    eval_result = trainer.evaluate()
    print(f"  Eval loss: {eval_result['eval_loss']:.4f}")
    
    # Cleanup
    import shutil
    for d in ["./ia3_trainer"]:
        if os.path.exists(d):
            shutil.rmtree(d)
    
    del model, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================================
# SECTION 4: FEW-SHOT TRAINING
# ============================================================================

def few_shot_training():
    """
    IA³'s sweet spot: few-shot learning.
    Train with just 4-64 examples.
    """
    print("\n\n" + "=" * 65)
    print("  SECTION 4: FEW-SHOT TRAINING")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, IA3Config, TaskType
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Few-shot examples (just 8!)
    few_shot_texts = [
        "Question: What is the capital of France? Answer: Paris",
        "Question: What is the capital of Japan? Answer: Tokyo",
        "Question: What is the capital of Germany? Answer: Berlin",
        "Question: What is the capital of Italy? Answer: Rome",
        "Question: What is the capital of Spain? Answer: Madrid",
        "Question: What is the capital of Brazil? Answer: Brasilia",
        "Question: What is the capital of Canada? Answer: Ottawa",
        "Question: What is the capital of Australia? Answer: Canberra",
    ]
    
    encodings = tokenizer(
        few_shot_texts, padding="max_length", truncation=True,
        max_length=32, return_tensors="pt",
    )
    labels = encodings["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    print(f"\n  Few-shot setup:")
    print(f"  Training examples: {len(few_shot_texts)}")
    print(f"  Max length: 32 tokens")
    
    # Train IA³
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    ia3_config = IA3Config(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn", "mlp.c_fc"],
        feedforward_modules=["mlp.c_fc"],
    )
    
    model = get_peft_model(model, ia3_config)
    model = model.to(device)
    model.print_trainable_parameters()
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-3,
    )
    
    ids = encodings["input_ids"].to(device)
    mask = encodings["attention_mask"].to(device)
    labs = labels.to(device)
    
    print(f"\n  Training on {len(few_shot_texts)} examples...")
    model.train()
    
    for epoch in range(20):
        outputs = model(input_ids=ids, attention_mask=mask, labels=labs)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:>3}: loss = {outputs.loss.item():.4f}")
    
    # Test
    model.eval()
    test_prompt = "Question: What is the capital of France? Answer:"
    test_inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **test_inputs, max_new_tokens=5, do_sample=False,
        )
    
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n  Test:")
    print(f"  Prompt: '{test_prompt}'")
    print(f"  Output: '{generated}'")
    
    # IA³ vector analysis
    print(f"\n  IA³ vector analysis after few-shot training:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            deviation = (param - 1.0).abs().mean().item()
            max_dev = (param - 1.0).abs().max().item()
            active = ((param - 1.0).abs() > 0.05).sum().item()
            short_name = name.split(".")[-2] + "." + name.split(".")[-1]
            print(f"    {short_name:>30}: mean_dev={deviation:.4f}, "
                  f"max_dev={max_dev:.4f}, active={active}/{param.numel()}")
    
    print(f"""
  ═══ Few-Shot IA³ Insights ═══
  
  With just 8 examples, IA³ can:
    - Learn task format (Q&A pattern)
    - Adjust feature importance per task
    - Not overfit (too few params to memorize)
    
  Comparison with alternatives:
  
  Method              │ Works with 8 examples?
  ────────────────────┼───────────────────────
  Full fine-tuning    │ ✗ Severe overfitting
  LoRA (r=8)          │ ⚠ Mild overfitting risk
  Prompt Tuning       │ ✗ Not enough signal
  IA³                 │ ✓ Sweet spot!
  In-context learning │ ✓ But wastes context length
""")
    
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================================
# SECTION 5: HYPERPARAMETER GUIDE
# ============================================================================

def hyperparameter_guide():
    """Comprehensive hyperparameter recommendations."""
    print("\n\n" + "=" * 65)
    print("  SECTION 5: HYPERPARAMETER GUIDE")
    print("=" * 65)
    
    print("""
  ═══════════════════════════════════════════════════════════════
   IA³ RECOMMENDED SETTINGS
  ═══════════════════════════════════════════════════════════════
  
  Target Modules:
  ┌──────────────────────┬─────────────────────────────────────┐
  │ Model Family         │ Recommended Targets                 │
  ├──────────────────────┼─────────────────────────────────────┤
  │ GPT-2                │ c_attn, mlp.c_fc                    │
  │ LLaMA/Mistral        │ k_proj, v_proj, down_proj           │
  │ BERT                 │ key, value, intermediate.dense       │
  │ T5                   │ k, v, wi (encoder + decoder)        │
  └──────────────────────┴─────────────────────────────────────┘
  
  Feed-Forward Modules (must be subset of target_modules):
  ┌──────────────────────┬─────────────────────────────────────┐
  │ Model Family         │ feedforward_modules                 │
  ├──────────────────────┼─────────────────────────────────────┤
  │ GPT-2                │ mlp.c_fc                            │
  │ LLaMA/Mistral        │ down_proj                           │
  │ BERT                 │ intermediate.dense                   │
  │ T5                   │ wi                                  │
  └──────────────────────┴─────────────────────────────────────┘
  
  Training Parameters:
  ┌─────────────────────┬──────────────────────────────────────┐
  │ Parameter           │ Recommendation                       │
  ├─────────────────────┼──────────────────────────────────────┤
  │ Learning rate       │ 1e-2 to 5e-2 (much higher than LoRA)│
  │ Optimizer           │ AdamW or Adam                        │
  │ Weight decay        │ 0.0 (vectors should stay near 1)     │
  │ Warmup              │ 0-5% of total steps                  │
  │ Scheduler           │ Cosine or constant                   │
  │ Gradient clipping   │ 1.0                                  │
  │ Batch size          │ 4-32 (smaller okay due to few params)│
  │ Epochs              │ 5-50 (depends on dataset size)       │
  │ Init IA³ weights    │ True (always initialize to ones)     │
  └─────────────────────┴──────────────────────────────────────┘
  
  Few-Shot Specific:
  ┌─────────────────────┬──────────────────────────────────────┐
  │ Parameter           │ Recommendation                       │
  ├─────────────────────┼──────────────────────────────────────┤
  │ Learning rate       │ 1e-3 to 1e-2                         │
  │ Epochs              │ 20-100 (more epochs, few examples)   │
  │ Batch size          │ = dataset size (all examples)        │
  │ Gradient accum      │ 1 (no need to accumulate)            │
  │ Early stopping      │ Yes (monitor validation loss)        │
  └─────────────────────┴──────────────────────────────────────┘
  
  ═══════════════════════════════════════════════════════════════
   KEY DIFFERENCES FROM OTHER METHODS
  ═══════════════════════════════════════════════════════════════
  
  Learning Rate Comparison:
  ┌──────────────────────┬─────────────────────────────────────┐
  │ Method               │ Typical LR Range                    │
  ├──────────────────────┼─────────────────────────────────────┤
  │ Full Fine-Tuning     │ 2e-5 to 5e-5                        │
  │ LoRA                 │ 1e-4 to 3e-4                        │
  │ P-Tuning v1/v2       │ 1e-3 to 5e-3                        │
  │ Prompt Tuning        │ 1e-3 to 1e-2                        │
  │ IA³                  │ 1e-2 to 5e-2     ← highest!        │
  └──────────────────────┴─────────────────────────────────────┘
  
  Why IA³ needs high LR:
    - Fewest trainable parameters of any method
    - Vectors initialized near 1 → need strong signal to move
    - Each parameter has outsized effect → bigger steps okay
    - No risk of catastrophic changes (just rescaling)
  
  Weight Decay:
    - Full FT: 0.01 (standard)
    - LoRA: 0.01 (standard)
    - IA³: 0.0 (NO weight decay!)
    
  Why no weight decay for IA³:
    - Weight decay pushes params toward 0
    - IA³ vectors should stay near 1 (not 0!)
    - Decay would fight the identity initialization
    - The vectors' proximity to 1 IS the regularization
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all training demonstrations."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     IA³ — TRAINING WITH PEFT LIBRARY                        ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: Configuration
    setup_ia3_basic()
    setup_ia3_variants()
    
    # Section 2: Custom training loop
    train_ia3_custom_loop()
    
    # Section 3: Trainer API
    train_with_trainer()
    
    # Section 4: Few-shot
    few_shot_training()
    
    # Section 5: Hyperparameter guide
    hyperparameter_guide()
    
    print("\n" + "=" * 65)
    print("  MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered:
    ✓ IA³ configuration with PEFT (IA3Config)
    ✓ Target module selection per model family
    ✓ Custom training loop with save/load
    ✓ HuggingFace Trainer integration
    ✓ Few-shot training (8 examples!)
    ✓ Comprehensive hyperparameter guide
    """)


if __name__ == "__main__":
    main()
