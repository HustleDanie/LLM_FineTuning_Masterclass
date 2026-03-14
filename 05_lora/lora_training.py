"""
LoRA Production Training Pipeline
===================================

A complete, production-ready LoRA training pipeline using HuggingFace PEFT.
This is what you should use for real projects (as opposed to the from-scratch
implementation which is for learning).

Covers:
1. Setting up LoRA with PEFT library
2. Data preparation for different tasks
3. Training with Trainer and custom loops
4. Evaluation during training
5. Checkpointing and resuming
6. Post-training: save, load, merge, and push to Hub
"""

import torch
import os
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field


# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================

@dataclass
class LoRATrainingConfig:
    """Complete configuration for LoRA training."""
    
    # Model
    model_name: str = "distilgpt2"
    task_type: str = "CAUSAL_LM"  # CAUSAL_LM, SEQ_CLS, SEQ_2_SEQ_LM, TOKEN_CLS
    
    # LoRA
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None  # None = auto-detect
    bias: str = "none"  # "none", "all", "lora_only"
    modules_to_save: Optional[List[str]] = None
    
    # Training
    output_dir: str = "./lora_output"
    num_epochs: int = 3
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # Data
    max_seq_length: int = 512
    dataset_name: Optional[str] = None
    dataset_text_field: str = "text"
    
    # Logging & saving
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    
    # Other
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False


# ===========================================================================
# 2. SETUP FUNCTIONS
# ===========================================================================

def setup_model_and_tokenizer(config: LoRATrainingConfig):
    """
    Load the base model and tokenizer, then apply LoRA.
    
    This function demonstrates the standard PEFT workflow:
    1. Load base model
    2. Create LoRA config
    3. Apply LoRA with get_peft_model()
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    
    print(f"Loading model: {config.model_name}")
    
    # -------------------------------------------------------------------
    # Load tokenizer
    # -------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # -------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,  # Use float16/bfloat16 for larger models
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # Enable gradient checkpointing if requested (saves memory)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Required for gradient checkpointing with LoRA
        model.enable_input_require_grads()
    
    # -------------------------------------------------------------------
    # Create LoRA configuration
    # -------------------------------------------------------------------
    # Map task type string to PEFT TaskType enum
    task_type_map = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_CLS": TaskType.SEQ_CLS,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        "TOKEN_CLS": TaskType.TOKEN_CLS,
    }
    
    # Auto-detect target modules if not specified
    target_modules = config.target_modules
    if target_modules is None:
        # PEFT will auto-detect with target_modules="all-linear"
        target_modules = "all-linear"  # PEFT >= 0.6.0 feature
        print("  Using auto-detected target modules (all linear layers)")
    
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias=config.bias,
        task_type=task_type_map.get(config.task_type, TaskType.CAUSAL_LM),
        modules_to_save=config.modules_to_save,
    )
    
    print(f"\n  LoRA Configuration:")
    print(f"    rank = {lora_config.r}")
    print(f"    alpha = {lora_config.lora_alpha}")
    print(f"    dropout = {lora_config.lora_dropout}")
    print(f"    target_modules = {lora_config.target_modules}")
    print(f"    bias = {lora_config.bias}")
    
    # -------------------------------------------------------------------
    # Apply LoRA to model
    # -------------------------------------------------------------------
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    # Output: "trainable params: X || all params: Y || trainable%: Z%"
    
    return model, tokenizer, lora_config


def prepare_dataset(
    config: LoRATrainingConfig,
    tokenizer,
    split: str = "train",
):
    """
    Prepare a dataset for LoRA training.
    
    Handles both:
    - HuggingFace dataset loading
    - Custom text formatting and tokenization
    """
    from datasets import load_dataset
    
    if config.dataset_name:
        dataset = load_dataset(config.dataset_name, split=split)
    else:
        # Demo dataset for testing
        demo_data = [
            "LoRA enables efficient fine-tuning of large language models by decomposing weight updates into low-rank matrices.",
            "The key insight is that fine-tuning updates have low intrinsic dimensionality.",
            "By freezing pre-trained weights and training only small adapter matrices, LoRA reduces memory and compute costs.",
            "The rank hyperparameter controls the expressiveness of the adaptation.",
            "Alpha scaling ensures consistent behavior regardless of the rank chosen.",
            "Target module selection determines which layers receive LoRA adapters.",
            "Modern practice adapts all linear layers, not just attention projections.",
            "LoRA weights can be merged into base weights for zero-overhead inference.",
        ] * 20  # Repeat for a larger dataset
        
        from datasets import Dataset
        dataset = Dataset.from_dict({"text": demo_data})
    
    # Tokenize
    def tokenize_function(examples):
        result = tokenizer(
            examples[config.dataset_text_field],
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
        )
        # For causal LM, labels = input_ids
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    
    print(f"\n  Dataset prepared:")
    print(f"    Split: {split}")
    print(f"    Samples: {len(tokenized)}")
    print(f"    Max length: {config.max_seq_length}")
    
    return tokenized


# ===========================================================================
# 3. TRAINING WITH HUGGINGFACE TRAINER
# ===========================================================================

def train_with_trainer(config: LoRATrainingConfig):
    """
    Train LoRA using HuggingFace Trainer.
    
    This is the RECOMMENDED approach for production training.
    It handles:
    - Mixed precision
    - Gradient accumulation
    - Checkpointing
    - Evaluation
    - Logging (to WandB, TensorBoard, etc.)
    """
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    
    # -------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------
    model, tokenizer, lora_config = setup_model_and_tokenizer(config)
    train_dataset = prepare_dataset(config, tokenizer, split="train")
    
    # Data collator handles dynamic padding and label shifting
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # -------------------------------------------------------------------
    # Training arguments
    # -------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        fp16=config.fp16,
        bf16=config.bf16,
        seed=config.seed,
        report_to="none",  # Change to "wandb" or "tensorboard" for logging
        remove_unused_columns=False,
        # LoRA-specific: don't need to save optimizer states for base model
        optim="adamw_torch",
    )
    
    # -------------------------------------------------------------------
    # Create Trainer
    # -------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # -------------------------------------------------------------------
    # Train!
    # -------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STARTING LoRA TRAINING")
    print("=" * 50)
    
    train_result = trainer.train()
    
    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    print(f"\n  Training complete!")
    print(f"  Metrics: {train_result.metrics}")
    
    # Save the LoRA adapter (NOT the full model)
    adapter_save_path = os.path.join(config.output_dir, "final_adapter")
    model.save_pretrained(adapter_save_path)
    tokenizer.save_pretrained(adapter_save_path)
    
    print(f"\n  LoRA adapter saved to: {adapter_save_path}")
    
    # Check saved size
    adapter_size = sum(
        os.path.getsize(os.path.join(adapter_save_path, f))
        for f in os.listdir(adapter_save_path)
        if os.path.isfile(os.path.join(adapter_save_path, f))
    )
    print(f"  Adapter size: {adapter_size / 1024 / 1024:.2f} MB")
    
    return model, tokenizer


# ===========================================================================
# 4. CUSTOM TRAINING LOOP (for advanced control)
# ===========================================================================

def train_custom_loop(config: LoRATrainingConfig):
    """
    Custom training loop for when you need fine-grained control.
    
    Use this when:
    - Custom loss functions
    - Complex data pipelines
    - Non-standard optimization
    - Research experiments
    """
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForLanguageModeling, get_scheduler
    
    # Setup
    model, tokenizer, lora_config = setup_model_and_tokenizer(config)
    train_dataset = prepare_dataset(config, tokenizer)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    
    # -------------------------------------------------------------------
    # Optimizer: only trainable parameters!
    # -------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Learning rate scheduler
    num_training_steps = (
        len(dataloader) * config.num_epochs // config.gradient_accumulation_steps
    )
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # -------------------------------------------------------------------
    # Device setup
    # -------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("CUSTOM TRAINING LOOP")
    print("=" * 50)
    
    model.train()
    global_step = 0
    accumulated_loss = 0.0
    
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    max_norm=config.max_grad_norm,
                )
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Logging
                if global_step % config.logging_steps == 0:
                    avg_loss = accumulated_loss / config.logging_steps
                    lr = scheduler.get_last_lr()[0]
                    print(f"  Step {global_step}: "
                          f"loss={avg_loss:.4f}, lr={lr:.2e}")
                    accumulated_loss = 0.0
                
                # Save checkpoint
                if global_step % config.save_steps == 0:
                    ckpt_dir = os.path.join(
                        config.output_dir, f"checkpoint-{global_step}"
                    )
                    model.save_pretrained(ckpt_dir)
                    print(f"  Saved checkpoint: {ckpt_dir}")
            
            epoch_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"\n  Epoch {epoch+1}/{config.num_epochs}: avg_loss={avg_epoch_loss:.4f}")
    
    # Save final
    final_dir = os.path.join(config.output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n  Training complete! Saved to {final_dir}")
    
    return model, tokenizer


# ===========================================================================
# 5. POST-TRAINING: LOAD, MERGE, AND INFERENCE
# ===========================================================================

def load_and_merge_lora(
    base_model_name: str,
    adapter_path: str,
    merge: bool = True,
    device: str = "auto",
):
    """
    Load a trained LoRA adapter and optionally merge it into the base model.
    
    This is the standard workflow for deploying a LoRA-tuned model:
    1. Load the base model
    2. Load the LoRA adapter on top
    3. Merge weights (optional, for inference efficiency)
    4. Save the merged model (optional)
    
    Parameters:
    -----------
    base_model_name : str
        HuggingFace model name or path to base model
    adapter_path : str
        Path to the saved LoRA adapter
    merge : bool
        Whether to merge LoRA weights into the base model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    print(f"Loading base model: {base_model_name}")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device if torch.cuda.is_available() else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    print(f"Loading LoRA adapter: {adapter_path}")
    
    # Load adapter on top of base model
    model = PeftModel.from_pretrained(model, adapter_path)
    
    if merge:
        print("Merging LoRA weights into base model...")
        # merge_and_unload() does three things:
        # 1. Computes W' = W + (α/r) · B · A for each LoRA layer
        # 2. Replaces the LoRA layer with a standard nn.Linear(W')
        # 3. Removes all LoRA parameters from the model
        # 
        # After this, the model is a standard HuggingFace model
        # with ZERO inference overhead from LoRA
        model = model.merge_and_unload()
        print("  Merged! Model is now a standard model with no LoRA overhead.")
    
    return model, tokenizer


def generate_with_lora(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """
    Generate text using a LoRA-tuned model.
    """
    device = next(model.parameters()).device
    model.eval()
    
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            "prompt": prompt,
            "generated": generated,
            "new_tokens": len(outputs[0]) - len(inputs["input_ids"][0]),
        })
    
    return results


# ===========================================================================
# 6. MULTI-ADAPTER TRAINING AND SERVING
# ===========================================================================

def demonstrate_multi_adapter():
    """
    Show how to train and serve multiple LoRA adapters on one base model.
    
    Production pattern:
    - Train separate adapters for different tasks/customers
    - Load them dynamically based on the request
    - Share the base model in GPU memory
    """
    print("\n" + "=" * 70)
    print("MULTI-ADAPTER TRAINING AND SERVING")
    print("=" * 70)
    
    print("""
    ARCHITECTURE:
    
    ┌─────────────────────────────────────────────────────────────┐
    │                    Shared Base Model (7B)                    │
    │                  [Frozen, loaded once in GPU]                │
    ├────────────┬────────────┬────────────┬──────────────────────┤
    │ Adapter A  │ Adapter B  │ Adapter C  │    Adapter N...      │
    │ (Chat)     │ (Code)     │ (Medical)  │    (Custom)          │
    │ ~30MB      │ ~30MB      │ ~30MB      │    ~30MB             │
    └────────────┴────────────┴────────────┴──────────────────────┘
    
    MEMORY: 14GB (base) + 30MB per adapter << 14GB × N models
    
    WORKFLOW:
    
    1. Train adapters independently:
       adapter_a = train(base_model, data_a, task="chat")
       adapter_b = train(base_model, data_b, task="code")
    
    2. At serving time:
       model = load_base_model()
       model.load_adapter("adapter_a", adapter_name="chat")
       model.load_adapter("adapter_b", adapter_name="code")
       
       # Switch per request:
       model.set_adapter("chat")    # For chat requests
       output = model.generate(...)
       
       model.set_adapter("code")    # For code requests
       output = model.generate(...)
    
    PEFT CODE EXAMPLE:
    """)
    
    print("""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    
    # Load base model once
    base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # Load first adapter
    model = PeftModel.from_pretrained(base_model, "path/to/adapter_chat")
    
    # Load additional adapters
    model.load_adapter("path/to/adapter_code", adapter_name="code")
    model.load_adapter("path/to/adapter_medical", adapter_name="medical")
    
    # Switch between adapters (zero cost!)
    model.set_adapter("default")   # Chat adapter (first loaded)
    chat_output = model.generate(...)
    
    model.set_adapter("code")      # Code adapter
    code_output = model.generate(...)
    
    model.set_adapter("medical")   # Medical adapter
    medical_output = model.generate(...)
    
    # Disable all adapters (use base model only)
    with model.disable_adapter():
        base_output = model.generate(...)
    """)


# ===========================================================================
# 7. TRAINING TIPS AND TROUBLESHOOTING
# ===========================================================================

def print_training_tips():
    """Print practical tips for LoRA training."""
    print("\n" + "=" * 70)
    print("LoRA TRAINING TIPS & TROUBLESHOOTING")
    print("=" * 70)
    
    print("""
    ═══ COMMON ISSUES AND FIXES ═══
    
    1. LOSS NOT DECREASING
       • Check: Is learning rate too small? Try 5x-10x larger
       • Check: Are LoRA layers actually trainable? print_trainable_parameters()
       • Check: Is the data formatted correctly? Print a few samples
       • Check: Gradient norm — if near 0, there's a computation issue
    
    2. LOSS EXPLODES (NaN or Inf)
       • Fix: Reduce learning rate (halve it)
       • Fix: Enable gradient clipping (max_grad_norm=1.0)
       • Fix: Use bf16 instead of fp16 (better numerical range)
       • Fix: Check for data issues (empty samples, wrong tokenization)
    
    3. VALIDATION LOSS FLAT BUT TRAIN LOSS DECREASES (overfitting)
       • Fix: Increase dropout (0.05 → 0.1)
       • Fix: Decrease rank (overfitting = too much capacity)
       • Fix: Add more training data
       • Fix: Early stopping
    
    4. OUT OF MEMORY (OOM)
       • Fix: Reduce batch_size, increase gradient_accumulation
       • Fix: Enable gradient checkpointing
       • Fix: Reduce max_seq_length
       • Fix: Use QLoRA (4-bit quantization, see Concept 06)
       • Fix: Reduce rank
    
    5. TRAINING IS SLOW
       • Fix: Enable bf16/fp16 mixed precision
       • Fix: Increase batch size (if memory allows)
       • Fix: Use Flash Attention (if supported)
       • Fix: Reduce max_seq_length (if data allows)
    
    ═══ TRAINING BEST PRACTICES ═══
    
    1. ALWAYS validate before training:
       • Run one forward pass to check shapes
       • Print a few tokenized examples to verify formatting
       • Verify trainable parameter count makes sense
    
    2. Monitor during training:
       • Loss curve (should decrease and plateau)
       • Learning rate (should follow schedule)
       • Gradient norm (should be stable, not growing)
       • Eval metrics (to catch overfitting early)
    
    3. After training:
       • Compare outputs: base model vs LoRA model
       • Check generation quality on diverse prompts
       • Measure task-specific metrics
       • Save adapter and config for reproducibility
    
    4. Reproducibility:
       • Set seed in training args AND data processing
       • Save the exact LoRA config used
       • Log all hyperparameters
       • Save tokenizer alongside adapter
    
    ═══ MEMORY ESTIMATION ═══
    
    Base model memory (fp16):
      Model_Size_GB ≈ Num_Params × 2 bytes / (1024³)
      7B model ≈ 14 GB (fp16) or 7 GB (4-bit)
    
    LoRA training overhead:
      LoRA_Params ≈ 2 × r × d_model × num_target_layers
      Optimizer states ≈ LoRA_Params × 8 bytes (AdamW)
      Gradients ≈ LoRA_Params × 2 bytes
      Activations ≈ batch_size × seq_len × d_model × num_layers × 2 bytes
    
    Total ≈ Base + LoRA_overhead + Activations
    Rule of thumb: ~1.3x base model memory for LoRA training
    """)


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    # Quick example with Trainer
    config = LoRATrainingConfig(
        model_name="distilgpt2",
        lora_rank=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        output_dir="./lora_demo_output",
        num_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=5,
        save_steps=50,
        max_seq_length=128,
    )
    
    print("LoRA Production Training Pipeline")
    print("=" * 70)
    
    # Train with Trainer (recommended)
    try:
        model, tokenizer = train_with_trainer(config)
        
        # Generate
        results = generate_with_lora(
            model, tokenizer,
            prompts=[
                "LoRA fine-tuning allows",
                "The key advantage of low-rank adaptation is",
            ],
            max_new_tokens=50,
        )
        
        print("\n  Generation results:")
        for r in results:
            print(f"    Prompt: {r['prompt']}")
            print(f"    Output: {r['generated'][:100]}...")
            print()
        
    except Exception as e:
        print(f"  Error: {e}")
        print("  Make sure to install: pip install transformers peft datasets")
    
    # Print tips
    print_training_tips()
    demonstrate_multi_adapter()
