"""
P-Tuning — Training with HuggingFace PEFT Library
===================================================

Production-ready training pipelines using PEFT:

1. P-Tuning v1 with PEFT
   - PromptEncoderConfig (LSTM/MLP encoder)
   - Automatic template handling

2. P-Tuning v2 with PEFT
   - PrefixTuningConfig (deep prompts via past_key_values)
   - PEFT treats v2 as a form of prefix tuning

3. Complete Training Pipeline
   - Data preparation for NLU tasks
   - Training loop with evaluation
   - Save/load/merge workflow

4. Hyperparameter Guide
   - Optimal settings for v1 and v2
   - Task-specific recommendations

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import os


# ============================================================================
# SECTION 1: P-TUNING V1 WITH PEFT
# ============================================================================

def setup_p_tuning_v1():
    """
    Configure P-Tuning v1 using HuggingFace PEFT.
    
    PEFT uses PromptEncoderConfig for P-Tuning v1:
    - Builds LSTM/MLP prompt encoder automatically
    - Manages freezing and gradient flow
    - Supports save/load out of the box
    """
    print("=" * 65)
    print("  SECTION 1: P-TUNING V1 WITH PEFT")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import (
        get_peft_model,
        PromptEncoderConfig,
        PromptEncoderReparameterizationType,
        TaskType,
    )
    
    # Load base model
    model_name = "distilgpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # ─── P-Tuning v1 Config ───
    # PromptEncoderConfig = P-Tuning v1 in PEFT
    peft_config = PromptEncoderConfig(
        task_type=TaskType.CAUSAL_LM,
        
        # Number of soft prompt tokens
        num_virtual_tokens=20,
        
        # Encoder type: LSTM or MLP
        # LSTM = original P-Tuning v1 (recommended)
        # MLP = simpler alternative
        encoder_reparameterization_type=PromptEncoderReparameterizationType.LSTM,
        
        # Encoder architecture
        encoder_hidden_size=256,   # LSTM hidden dimension
        encoder_num_layers=2,      # LSTM layers
        encoder_dropout=0.0,       # Dropout in encoder
    )
    
    print(f"\n  P-Tuning v1 Config:")
    print(f"  ─────────────────────────────────")
    print(f"  Virtual tokens:   {peft_config.num_virtual_tokens}")
    print(f"  Encoder type:     {peft_config.encoder_reparameterization_type}")
    print(f"  Encoder hidden:   {peft_config.encoder_hidden_size}")
    print(f"  Encoder layers:   {peft_config.encoder_num_layers}")
    
    # Apply PEFT
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Verify structure
    print(f"\n  PEFT model structure (prompt encoder):")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"    {name}: {param.shape}")
    
    # Test forward pass
    text = "P-Tuning v1 uses an LSTM encoder to generate"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    
    print(f"\n  Test forward pass:")
    print(f"  Input: '{text}'")
    print(f"  Loss: {outputs.loss.item():.4f}")
    
    del model
    return peft_config


def setup_p_tuning_v1_mlp():
    """
    P-Tuning v1 with MLP encoder instead of LSTM.
    """
    print("\n\n  ─── MLP Variant ───")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import (
        get_peft_model,
        PromptEncoderConfig,
        PromptEncoderReparameterizationType,
        TaskType,
    )
    
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    mlp_config = PromptEncoderConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
        encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
        encoder_hidden_size=256,
        encoder_num_layers=2,
        encoder_dropout=0.1,
    )
    
    model = get_peft_model(model, mlp_config)
    model.print_trainable_parameters()
    
    print(f"\n  MLP encoder trainable params:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"    {name}: {param.shape}")
    
    del model


# ============================================================================
# SECTION 2: P-TUNING V2 WITH PEFT
# ============================================================================

def setup_p_tuning_v2():
    """
    Configure P-Tuning v2 using HuggingFace PEFT.
    
    PEFT implements P-Tuning v2 as PrefixTuning:
    - Both use deep prompts at every layer
    - Both inject via past_key_values
    - P-Tuning v2 = Prefix Tuning applied universally
    """
    print("\n\n" + "=" * 65)
    print("  SECTION 2: P-TUNING V2 WITH PEFT (via PrefixTuning)")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, PrefixTuningConfig, TaskType
    
    model_name = "distilgpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # ─── P-Tuning v2 Config (= Prefix Tuning in PEFT) ───
    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        
        # Number of prefix tokens per layer
        num_virtual_tokens=20,
        
        # Whether to use an MLP reparameterization
        # True = trains faster (like prefix tuning original)
        # False = direct optimization (like P-Tuning v2 paper)
        prefix_projection=False,
        
        # If prefix_projection=True, set hidden dim
        # encoder_hidden_size=256,
    )
    
    print(f"\n  P-Tuning v2 Config (as PrefixTuning):")
    print(f"  ─────────────────────────────────")
    print(f"  Virtual tokens:     {peft_config.num_virtual_tokens}")
    print(f"  Prefix projection:  {peft_config.prefix_projection}")
    print(f"  Task type:          {peft_config.task_type}")
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Show deep prompt structure
    print(f"\n  PEFT model structure (deep prompts):")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"    {name}: {param.shape}")
    
    # Test
    text = "P-Tuning v2 injects deep prompts at every layer"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    
    print(f"\n  Test forward pass:")
    print(f"  Input: '{text}'")
    print(f"  Loss: {outputs.loss.item():.4f}")
    
    del model
    return peft_config


# ============================================================================
# SECTION 3: COMPLETE TRAINING PIPELINE
# ============================================================================

def create_training_dataset(tokenizer, num_samples=200, max_length=64):
    """Create a synthetic classification-style dataset."""
    
    templates = {
        "positive": [
            "This movie was absolutely wonderful and entertaining.",
            "I loved every minute of this brilliant film.",
            "An outstanding performance by the entire cast.",
            "Truly a masterpiece of modern cinema.",
            "The best movie I have seen this year.",
        ],
        "negative": [
            "This movie was terrible and a waste of time.",
            "I hated this boring and predictable film.",
            "A complete disaster from start to finish.",
            "The worst movie I have ever watched.",
            "Absolutely dreadful acting and terrible script.",
        ],
    }
    
    import random
    random.seed(42)
    
    texts = []
    for _ in range(num_samples):
        label = random.choice(["positive", "negative"])
        text = random.choice(templates[label])
        # Format as: "Review: {text} Sentiment: {label}"
        texts.append(f"Review: {text} Sentiment: {label}")
    
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    
    # For causal LM, labels = input_ids (with padding masked)
    labels = encodings["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    dataset = torch.utils.data.TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        labels,
    )
    
    return dataset


def train_p_tuning_v1_pipeline():
    """
    Full training pipeline for P-Tuning v1.
    """
    print("\n\n" + "=" * 65)
    print("  SECTION 3: COMPLETE TRAINING PIPELINE")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import (
        get_peft_model,
        PromptEncoderConfig,
        PromptEncoderReparameterizationType,
        TaskType,
    )
    
    # ─── Setup ───
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    peft_config = PromptEncoderConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
        encoder_reparameterization_type=PromptEncoderReparameterizationType.LSTM,
        encoder_hidden_size=256,
        encoder_num_layers=2,
    )
    
    model = get_peft_model(model, peft_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"\n  Device: {device}")
    model.print_trainable_parameters()
    
    # ─── Dataset ───
    dataset = create_training_dataset(tokenizer, num_samples=100, max_length=48)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=8,
    )
    
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # ─── Optimizer ───
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-3,        # Higher LR for prompt encoder
        weight_decay=0.01,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=3,  # 3 epochs
    )
    
    # ─── Training ───
    print(f"\n  Training P-Tuning v1...")
    print(f"  {'Epoch':>7} {'Train Loss':>12} {'Val Loss':>12} {'LR':>12}")
    print(f"  {'─'*7}─{'─'*12}─{'─'*12}─{'─'*12}")
    
    num_epochs = 3
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_loss += outputs.loss.item()
        
        val_loss /= len(val_loader)
        lr = scheduler.get_last_lr()[0]
        
        print(f"  {epoch+1:>7} {train_loss:>12.4f} {val_loss:>12.4f} {lr:>12.6f}")
        
        scheduler.step()
    
    # ─── Save ───
    save_dir = "./p_tuning_v1_checkpoint"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    print(f"\n  Model saved to {save_dir}/")
    
    # Show saved files
    if os.path.exists(save_dir):
        files = os.listdir(save_dir)
        total_size = sum(
            os.path.getsize(os.path.join(save_dir, f))
            for f in files if os.path.isfile(os.path.join(save_dir, f))
        )
        print(f"  Saved files: {len(files)}")
        print(f"  Total size: {total_size / 1024:.1f} KB")
    
    # ─── Load and Inference ───
    print(f"\n  Loading model for inference...")
    
    from peft import PeftModel
    
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    loaded_model = PeftModel.from_pretrained(base_model, save_dir)
    loaded_model = loaded_model.to(device)
    loaded_model.eval()
    
    test_prompt = "Review: This movie was absolutely wonderful"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = loaded_model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            temperature=1.0,
        )
    
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n  Prompt: '{test_prompt}'")
    print(f"  Output: '{generated}'")
    
    # Cleanup
    import shutil
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    
    del model, loaded_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================================
# SECTION 4: TRAINER API PIPELINE
# ============================================================================

def train_with_trainer_api():
    """
    Training with HuggingFace Trainer (recommended for production).
    """
    print("\n\n" + "=" * 65)
    print("  SECTION 4: TRAINING WITH TRAINER API")
    print("=" * 65)
    
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import (
        get_peft_model,
        PromptEncoderConfig,
        PromptEncoderReparameterizationType,
        PrefixTuningConfig,
        TaskType,
    )
    from datasets import Dataset
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Choose config: v1 or v2
    # ─── V1 (LSTM encoder) ───
    v1_config = PromptEncoderConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
        encoder_reparameterization_type=PromptEncoderReparameterizationType.LSTM,
        encoder_hidden_size=256,
        encoder_num_layers=2,
    )
    
    # ─── V2 (Deep prompts) ───
    v2_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
        prefix_projection=False,
    )
    
    # Use v2 for this demo
    peft_config = v2_config
    model = get_peft_model(model, peft_config)
    
    print(f"\n  Using P-Tuning v2 (PrefixTuning):")
    model.print_trainable_parameters()
    
    # Create dataset
    texts = [
        "The weather today is sunny and warm.",
        "Machine learning models require large datasets.",
        "Python is a popular programming language.",
        "Neural networks can learn complex patterns.",
        "Fine-tuning adapts pretrained models to new tasks.",
    ] * 20  # 100 examples
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=48,
        )
    
    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.train_test_split(test_size=0.2, seed=42)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./p_tuning_v2_trainer",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=3e-3,
        weight_decay=0.01,
        warmup_steps=10,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
    )
    
    print(f"\n  Starting Trainer...")
    result = trainer.train()
    
    print(f"\n  Training complete!")
    print(f"  Final loss: {result.training_loss:.4f}")
    print(f"  Steps: {result.global_step}")
    
    # Evaluate
    eval_result = trainer.evaluate()
    print(f"  Eval loss: {eval_result['eval_loss']:.4f}")
    
    # Save
    model.save_pretrained("./p_tuning_v2_final")
    print(f"  Model saved to ./p_tuning_v2_final/")
    
    # Cleanup
    import shutil
    for d in ["./p_tuning_v2_trainer", "./p_tuning_v2_final"]:
        if os.path.exists(d):
            shutil.rmtree(d)
    
    del model, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================================
# SECTION 5: HYPERPARAMETER GUIDE
# ============================================================================

def hyperparameter_guide():
    """
    Comprehensive hyperparameter recommendations for P-Tuning.
    """
    print("\n\n" + "=" * 65)
    print("  SECTION 5: HYPERPARAMETER GUIDE")
    print("=" * 65)
    
    guide = """
  ═══════════════════════════════════════════════════════════════
   P-TUNING V1 RECOMMENDED SETTINGS
  ═══════════════════════════════════════════════════════════════
  
  Virtual Tokens:
  ┌─────────────────────┬──────────────────────────────────────┐
  │ Task Type           │ Recommended num_virtual_tokens       │
  ├─────────────────────┼──────────────────────────────────────┤
  │ Classification      │ 10–20                                │
  │ Knowledge Probing   │ 20–40 (cloze format)                 │
  │ Question Answering  │ 20–30                                │
  │ NER                 │ 15–25                                │
  └─────────────────────┴──────────────────────────────────────┘
  
  Encoder Settings:
  ┌─────────────────────┬──────────────────────────────────────┐
  │ Parameter           │ Recommendation                       │
  ├─────────────────────┼──────────────────────────────────────┤
  │ Encoder type        │ LSTM (default, better than MLP)      │
  │ encoder_hidden_size │ 128–512 (256 is safe default)        │
  │ encoder_num_layers  │ 2 (more adds cost, little benefit)   │
  │ encoder_dropout     │ 0.0–0.1 (v1 is already regularized) │
  └─────────────────────┴──────────────────────────────────────┘
  
  Training:
  ┌─────────────────────┬──────────────────────────────────────┐
  │ Parameter           │ Recommendation                       │
  ├─────────────────────┼──────────────────────────────────────┤
  │ Learning rate       │ 1e-3 to 5e-3 (higher than LoRA!)    │
  │ Optimizer           │ AdamW                                │
  │ Weight decay        │ 0.01                                 │
  │ Warmup              │ 5–10% of total steps                 │
  │ Scheduler           │ Cosine or linear decay               │
  │ Gradient clipping   │ 1.0                                  │
  │ Batch size          │ 8–32                                 │
  │ Epochs              │ 10–30 (more than LoRA/full FT)       │
  └─────────────────────┴──────────────────────────────────────┘
  
  ═══════════════════════════════════════════════════════════════
   P-TUNING V2 RECOMMENDED SETTINGS
  ═══════════════════════════════════════════════════════════════
  
  Virtual Tokens:
  ┌──────────────────────┬─────────────────────────────────────┐
  │ Task / Model Size    │ Recommended num_virtual_tokens      │
  ├──────────────────────┼─────────────────────────────────────┤
  │ Small (< 1B)         │ 10–20                               │
  │ Medium (1B–10B)      │ 20–40                               │
  │ Large (> 10B)        │ 10–20 (less needed for big models)  │
  │                      │                                     │
  │ Simple classification│ 10                                  │
  │ NER / Extraction     │ 20–30                               │
  │ QA / Complex NLU     │ 20–40                               │
  └──────────────────────┴─────────────────────────────────────┘
  
  Architecture:
  ┌─────────────────────┬──────────────────────────────────────┐
  │ Parameter           │ Recommendation                       │
  ├─────────────────────┼──────────────────────────────────────┤
  │ prefix_projection   │ False (direct, paper default)        │
  │                     │ True if unstable training            │
  │ encoder_hidden_size │ 256–512 (only if projection=True)    │
  └─────────────────────┴──────────────────────────────────────┘
  
  Training:
  ┌─────────────────────┬──────────────────────────────────────┐
  │ Parameter           │ Recommendation                       │
  ├─────────────────────┼──────────────────────────────────────┤
  │ Learning rate       │ 1e-3 to 1e-2 (v2 can go higher)     │
  │ Optimizer           │ AdamW                                │
  │ Weight decay        │ 0.01                                 │
  │ Warmup              │ 3–5% of steps                        │
  │ Scheduler           │ Cosine                               │
  │ Gradient clipping   │ 1.0                                  │
  │ Batch size          │ 16–64                                │
  │ Epochs              │ 10–50                                │
  └─────────────────────┴──────────────────────────────────────┘
  
  ═══════════════════════════════════════════════════════════════
   KEY DIFFERENCES FROM OTHER METHODS
  ═══════════════════════════════════════════════════════════════
  
  Learning Rate:
    Full FT:       2e-5 to 5e-5
    LoRA:          1e-4 to 3e-4
    P-Tuning v1:   1e-3 to 5e-3     ← 10x higher!
    P-Tuning v2:   1e-3 to 1e-2     ← even higher!
  
  Why higher LR?
    - Fewer parameters → need bigger updates per step
    - Prompt parameters are unconstrained (no pretrained init)
    - LSTM encoder adds smoothing (v1)
    
  Epochs:
    Full FT:       1–5
    LoRA:          3–10
    P-Tuning:      10–50            ← more epochs needed!
  
  Why more epochs?
    - Small parameter space → needs more iterations
    - Prompts start from random → longer convergence
"""
    print(guide)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all training demonstrations."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     P-TUNING — TRAINING WITH PEFT LIBRARY                   ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: V1 with PEFT
    setup_p_tuning_v1()
    setup_p_tuning_v1_mlp()
    
    # Section 2: V2 with PEFT
    setup_p_tuning_v2()
    
    # Section 3: Full training pipeline
    train_p_tuning_v1_pipeline()
    
    # Section 4: Trainer API
    train_with_trainer_api()
    
    # Section 5: Hyperparameter guide
    hyperparameter_guide()
    
    print("\n" + "=" * 65)
    print("  MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered:
    ✓ P-Tuning v1 setup with PEFT (LSTM & MLP encoders)
    ✓ P-Tuning v2 setup with PEFT (PrefixTuningConfig)
    ✓ Complete training pipeline with custom loop
    ✓ Training with HuggingFace Trainer API
    ✓ Save/load/inference workflow
    ✓ Comprehensive hyperparameter guide
    """)


if __name__ == "__main__":
    main()
