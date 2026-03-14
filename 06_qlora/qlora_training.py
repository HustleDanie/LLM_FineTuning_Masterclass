"""
QLoRA Training Pipeline
========================

Complete end-to-end QLoRA training implementation covering:

1. QLoRA Configuration
   - BitsAndBytes 4-bit quantization config
   - LoRA config for quantized models
   - Training arguments optimized for QLoRA

2. Model Loading
   - Loading models in 4-bit with NF4
   - Preparing models for k-bit training
   - Memory-efficient model setup

3. Dataset Preparation
   - Instruction formatting for chat models
   - Tokenization with proper padding/truncation
   - Completion-only loss masking

4. Training Loop
   - SFTTrainer with QLoRA
   - Gradient checkpointing for memory savings
   - Gradient accumulation strategies
   - Mixed-precision training with quantized models

5. Saving & Loading
   - Saving QLoRA adapters
   - Loading adapters back onto quantized base
   - Merging QLoRA into full-precision model

6. Practical Recipes
   - 7B model on 24GB GPU
   - 13B model on 24GB GPU
   - 70B model on 48GB GPU
   - Multi-GPU QLoRA with FSDP

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


# ============================================================================
# SECTION 1: QLoRA CONFIGURATION
# ============================================================================

@dataclass
class QLoRAConfig:
    """
    Complete configuration for QLoRA training.
    
    QLoRA = 4-bit Quantized Base Model + LoRA Adapters + Paged Optimizers
    
    The key insight: We freeze the base model in 4-bit precision and
    only train the LoRA adapter weights in FP16/BF16. This reduces
    memory by ~4x compared to standard LoRA on a FP16 model.
    """
    
    # ── Base Model ────────────────────────────────────────────────
    base_model_name: str = "distilgpt2"
    
    # ── Quantization Config ──────────────────────────────────────
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"           # "nf4" or "fp4"
    bnb_4bit_compute_dtype: str = "bfloat16"   # Compute in BF16
    bnb_4bit_use_double_quant: bool = True     # Nested quantization
    
    # ── LoRA Config ──────────────────────────────────────────────
    lora_r: int = 16                           # LoRA rank
    lora_alpha: int = 32                       # Scaling factor
    lora_dropout: float = 0.05                 # Dropout on LoRA layers
    lora_target_modules: Optional[List[str]] = None  # Auto-detect if None
    lora_bias: str = "none"                    # "none", "all", "lora_only"
    
    # ── Training Config ──────────────────────────────────────────
    output_dir: str = "./qlora_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4       # Effective batch = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 512
    
    # ── Memory Optimization ──────────────────────────────────────
    gradient_checkpointing: bool = True        # Trade compute for memory
    optim: str = "paged_adamw_8bit"            # Paged 8-bit optimizer
    
    # ── Logging & Saving ─────────────────────────────────────────
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    
    def get_compute_dtype(self):
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.bnb_4bit_compute_dtype, torch.bfloat16)
    
    def estimate_memory(self, model_params_billions: float) -> Dict[str, float]:
        """
        Estimate memory requirements for QLoRA training.
        
        Components:
        1. Base model (4-bit): ~0.5 GB per billion params
        2. LoRA adapters (FP16): ~0.01-0.05 GB depending on rank
        3. Optimizer states (8-bit paged): ~0.02-0.1 GB for LoRA params
        4. Gradients (FP16): Same as LoRA adapter size
        5. Activations: Depends on batch size and seq length
        """
        # Base model in 4-bit
        base_model_gb = model_params_billions * 0.5  # ~0.5 GB/B in 4-bit
        
        # LoRA parameters (rough estimate)
        # Assuming ~30% of base params are targeted, with rank r
        targeted_params = model_params_billions * 1e9 * 0.3
        lora_params = targeted_params * self.lora_r * 2 / targeted_params**0.5
        lora_params = min(lora_params, model_params_billions * 1e9 * 0.02)
        lora_gb = lora_params * 2 / 1e9  # FP16 = 2 bytes
        
        # Optimizer states (8-bit Adam: 2 states, 1 byte each)
        optimizer_gb = lora_params * 2 / 1e9  # 8-bit: 1 byte per state × 2 states
        
        # Gradients
        gradient_gb = lora_gb  # Same size as LoRA params in FP16
        
        # Activations (rough estimate)
        batch_tokens = (self.per_device_train_batch_size * 
                       self.max_seq_length)
        hidden_size_est = model_params_billions ** 0.5 * 2048  # Rough
        activation_gb = batch_tokens * hidden_size_est * 2 / 1e9
        if self.gradient_checkpointing:
            activation_gb *= 0.3  # ~70% savings
        
        total_gb = (base_model_gb + lora_gb + optimizer_gb + 
                   gradient_gb + activation_gb)
        
        return {
            "base_model_4bit_gb": round(base_model_gb, 2),
            "lora_adapters_gb": round(lora_gb, 3),
            "optimizer_states_gb": round(optimizer_gb, 3),
            "gradients_gb": round(gradient_gb, 3),
            "activations_gb": round(activation_gb, 2),
            "total_estimated_gb": round(total_gb, 2),
            "recommended_gpu": (
                "8 GB" if total_gb < 7 else
                "16 GB" if total_gb < 14 else
                "24 GB" if total_gb < 22 else
                "48 GB" if total_gb < 44 else
                "80 GB" if total_gb < 76 else
                "Multi-GPU"
            ),
        }
    
    def print_config(self):
        """Print the full configuration in a readable format."""
        print("=" * 65)
        print("  QLoRA TRAINING CONFIGURATION")
        print("=" * 65)
        print(f"\n  Base Model:        {self.base_model_name}")
        print(f"\n  ── Quantization ──")
        print(f"  4-bit:             {self.load_in_4bit}")
        print(f"  Quant type:        {self.bnb_4bit_quant_type}")
        print(f"  Compute dtype:     {self.bnb_4bit_compute_dtype}")
        print(f"  Double quant:      {self.bnb_4bit_use_double_quant}")
        print(f"\n  ── LoRA ──")
        print(f"  Rank:              {self.lora_r}")
        print(f"  Alpha:             {self.lora_alpha}")
        print(f"  Dropout:           {self.lora_dropout}")
        print(f"  Target modules:    {self.lora_target_modules or 'auto'}")
        print(f"  Bias:              {self.lora_bias}")
        print(f"\n  ── Training ──")
        print(f"  Epochs:            {self.num_train_epochs}")
        print(f"  Batch size:        {self.per_device_train_batch_size}")
        print(f"  Grad accumulation: {self.gradient_accumulation_steps}")
        eff_batch = (self.per_device_train_batch_size * 
                    self.gradient_accumulation_steps)
        print(f"  Effective batch:   {eff_batch}")
        print(f"  Learning rate:     {self.learning_rate}")
        print(f"  Optimizer:         {self.optim}")
        print(f"  Grad checkpoint:   {self.gradient_checkpointing}")
        print(f"  Max seq length:    {self.max_seq_length}")


def demonstrate_qlora_config():
    """Show QLoRA configurations for different model sizes."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║          QLoRA MEMORY ESTIMATION BY MODEL SIZE               ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    configs = {
        "7B (e.g., LLaMA-2-7B)": QLoRAConfig(
            base_model_name="meta-llama/Llama-2-7b-hf",
            lora_r=16,
            per_device_train_batch_size=4,
            max_seq_length=512,
        ),
        "13B (e.g., LLaMA-2-13B)": QLoRAConfig(
            base_model_name="meta-llama/Llama-2-13b-hf",
            lora_r=16,
            per_device_train_batch_size=2,
            max_seq_length=512,
        ),
        "70B (e.g., LLaMA-2-70B)": QLoRAConfig(
            base_model_name="meta-llama/Llama-2-70b-hf",
            lora_r=16,
            per_device_train_batch_size=1,
            max_seq_length=256,
        ),
    }
    
    model_sizes = {"7B": 7, "13B": 13, "70B": 70}
    
    for name, config in configs.items():
        size_key = name.split(" ")[0]
        size_b = model_sizes[size_key]
        mem = config.estimate_memory(size_b)
        
        print(f"\n  {name}:")
        print(f"    Base model (4-bit):  {mem['base_model_4bit_gb']:>6.2f} GB")
        print(f"    LoRA adapters:       {mem['lora_adapters_gb']:>6.3f} GB")
        print(f"    Optimizer states:    {mem['optimizer_states_gb']:>6.3f} GB")
        print(f"    Gradients:           {mem['gradients_gb']:>6.3f} GB")
        print(f"    Activations:         {mem['activations_gb']:>6.2f} GB")
        print(f"    ─────────────────────────────────")
        print(f"    TOTAL:               {mem['total_estimated_gb']:>6.2f} GB")
        print(f"    Recommended GPU:     {mem['recommended_gpu']}")


# ============================================================================
# SECTION 2: COMPLETE QLoRA TRAINING PIPELINE
# ============================================================================

def qlora_training_pipeline(config: Optional[QLoRAConfig] = None):
    """
    Complete QLoRA training pipeline.
    
    Steps:
    1. Configure quantization
    2. Load model in 4-bit
    3. Prepare for k-bit training
    4. Apply LoRA adapters
    5. Prepare dataset
    6. Train with SFTTrainer
    7. Save adapter
    
    This is a fully runnable pipeline using distilgpt2 for demonstration.
    """
    if config is None:
        config = QLoRAConfig()
    
    config.print_config()
    
    print("\n" + "=" * 65)
    print("  STARTING QLoRA TRAINING PIPELINE")
    print("=" * 65)
    
    # ── Step 1: BitsAndBytes Quantization Config ─────────────────
    print("\n[Step 1/7] Configuring 4-bit quantization...")
    
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=config.get_compute_dtype(),
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )
    
    print(f"  Quant type: {config.bnb_4bit_quant_type}")
    print(f"  Compute dtype: {config.bnb_4bit_compute_dtype}")
    print(f"  Double quantization: {config.bnb_4bit_use_double_quant}")
    
    # ── Step 2: Load Model in 4-bit ─────────────────────────────
    print(f"\n[Step 2/7] Loading {config.base_model_name} in 4-bit...")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        trust_remote_code=True,
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Print model memory footprint
    total_params = sum(p.numel() for p in model.parameters())
    memory_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    print(f"  Parameters: {total_params:,}")
    print(f"  Memory footprint: {memory_bytes / 1e6:.1f} MB")
    
    # Check which layers are quantized
    quantized_layers = 0
    non_quantized_layers = 0
    for name, param in model.named_parameters():
        if hasattr(param, 'quant_state'):
            quantized_layers += 1
        else:
            non_quantized_layers += 1
    print(f"  Quantized layers: {quantized_layers}")
    print(f"  Non-quantized layers: {non_quantized_layers}")
    
    # ── Step 3: Prepare for K-bit Training ───────────────────────
    print("\n[Step 3/7] Preparing model for k-bit training...")
    
    from peft import prepare_model_for_kbit_training
    
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.gradient_checkpointing,
    )
    
    # What prepare_model_for_kbit_training does:
    # 1. Freezes all base model parameters
    # 2. Casts layer norms to FP32 for training stability
    # 3. Enables gradient checkpointing if requested
    # 4. Sets model to training mode
    # 5. Enables input gradients for LoRA backward pass
    
    print("  ✓ All base parameters frozen")
    print("  ✓ Layer norms cast to FP32")
    if config.gradient_checkpointing:
        print("  ✓ Gradient checkpointing enabled")
    print("  ✓ Input gradients enabled")
    
    # ── Step 4: Apply LoRA Adapters ──────────────────────────────
    print("\n[Step 4/7] Applying LoRA adapters...")
    
    from peft import LoraConfig, get_peft_model, TaskType
    
    # Auto-detect target modules if not specified
    target_modules = config.lora_target_modules
    if target_modules is None:
        # Common target modules for popular architectures
        architecture_targets = {
            "gpt2": ["c_attn", "c_proj"],
            "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "phi": ["q_proj", "k_proj", "v_proj", "dense"],
            "gemma": ["q_proj", "k_proj", "v_proj", "o_proj"],
        }
        
        model_type = getattr(model.config, "model_type", "").lower()
        target_modules = architecture_targets.get(model_type, ["c_attn"])
        print(f"  Auto-detected target modules for '{model_type}': {target_modules}")
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameter summary
    model.print_trainable_parameters()
    
    # Detailed parameter breakdown
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()
    
    print(f"\n  Trainable:   {trainable:>12,} ({trainable/1e6:.2f}M)")
    print(f"  Frozen:      {frozen:>12,} ({frozen/1e6:.2f}M)")
    print(f"  % Trainable: {100 * trainable / (trainable + frozen):.4f}%")
    
    # ── Step 5: Prepare Dataset ──────────────────────────────────
    print("\n[Step 5/7] Preparing dataset...")
    
    from datasets import load_dataset
    
    dataset = load_dataset("Abirate/english_quotes", split="train")
    print(f"  Dataset size: {len(dataset)} examples")
    
    # Format as instruction-following data
    def format_instruction(example):
        """Format each example as an instruction-response pair."""
        quote = example.get("quote", "")
        author = example.get("author", "Unknown")
        tags = example.get("tags", [])
        
        # Create instruction format
        if tags:
            tag_str = ", ".join(tags[:3]) if isinstance(tags, list) else str(tags)
            instruction = f"Generate a {tag_str} quote."
        else:
            instruction = "Generate an inspiring quote."
        
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n\"{quote}\" — {author}"
        )
        return {"text": text}
    
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    
    # Show sample
    print(f"\n  Sample formatted example:")
    sample = dataset[0]["text"]
    for line in sample.split("\n")[:6]:
        print(f"    {line}")
    if len(sample.split("\n")) > 6:
        print(f"    ...")
    
    # Train/eval split
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"\n  Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
    
    # ── Step 6: Train ────────────────────────────────────────────
    print("\n[Step 6/7] Starting training...")
    
    from transformers import TrainingArguments
    from trl import SFTTrainer
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        optim=config.optim,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.gradient_checkpointing else None,
        max_grad_norm=0.3,                  # Gradient clipping
        group_by_length=True,               # Group similar lengths (faster)
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )
    
    # Use SFTTrainer for supervised fine-tuning
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=None,  # Already applied PEFT
    )
    
    # Print training info
    total_steps = (
        len(train_dataset) // 
        (config.per_device_train_batch_size * config.gradient_accumulation_steps)
        * config.num_train_epochs
    )
    print(f"  Total training steps: ~{total_steps}")
    print(f"  Effective batch size: "
          f"{config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    
    # Memory before training
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    # Train!
    train_start = time.time()
    train_result = trainer.train()
    train_time = time.time() - train_start
    
    print(f"\n  Training completed in {train_time:.1f}s")
    print(f"  Final loss: {train_result.training_loss:.4f}")
    
    # Memory after training
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak GPU memory: {peak:.2f} GB")
    
    # ── Step 7: Save ─────────────────────────────────────────────
    print(f"\n[Step 7/7] Saving adapter to {config.output_dir}...")
    
    # Save only the LoRA adapter (not the full model!)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # Also save the training config for reproducibility
    config_dict = {
        "base_model_name": config.base_model_name,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_target_modules": target_modules,
        "lora_dropout": config.lora_dropout,
        "bnb_4bit_quant_type": config.bnb_4bit_quant_type,
        "bnb_4bit_compute_dtype": config.bnb_4bit_compute_dtype,
        "bnb_4bit_use_double_quant": config.bnb_4bit_use_double_quant,
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_train_epochs,
        "batch_size": config.per_device_train_batch_size,
        "grad_accum": config.gradient_accumulation_steps,
        "optimizer": config.optim,
        "training_loss": train_result.training_loss,
        "training_time_seconds": train_time,
    }
    
    with open(os.path.join(config.output_dir, "qlora_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # List saved files
    saved_files = os.listdir(config.output_dir)
    print(f"  Saved files: {saved_files}")
    
    # Adapter size
    adapter_size = sum(
        os.path.getsize(os.path.join(config.output_dir, f))
        for f in saved_files
        if os.path.isfile(os.path.join(config.output_dir, f))
    ) / (1024 * 1024)
    print(f"  Adapter size: {adapter_size:.2f} MB")
    
    print("\n" + "=" * 65)
    print("  QLoRA TRAINING COMPLETE")
    print("=" * 65)
    
    return model, tokenizer, train_result


# ============================================================================
# SECTION 3: INFERENCE WITH QLoRA
# ============================================================================

def qlora_inference(
    base_model_name: str = "distilgpt2",
    adapter_path: str = "./qlora_output",
    prompts: Optional[List[str]] = None,
):
    """
    Load a QLoRA adapter and run inference.
    
    Two modes:
    1. Quantized inference: Load base in 4-bit + adapter (Memory efficient)
    2. Merged inference: Merge into FP16 model (Faster, larger)
    """
    print("=" * 65)
    print("  QLoRA INFERENCE")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    
    if prompts is None:
        prompts = [
            "### Instruction:\nGenerate an inspiring quote.\n\n### Response:\n",
            "### Instruction:\nGenerate a love quote.\n\n### Response:\n",
            "### Instruction:\nGenerate a wisdom quote.\n\n### Response:\n",
        ]
    
    # ── Mode 1: Quantized Inference ──────────────────────────────
    print("\n[Mode 1] Quantized inference (4-bit base + adapter)...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    print("\n  ── Quantized Inference Results ──")
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
            )
        
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        # Show just the first line of the instruction
        instruction = prompt.split("\n")[1] if "\n" in prompt else prompt[:50]
        print(f"\n  Prompt: {instruction}")
        print(f"  Response: {response[:200]}")
    
    if torch.cuda.is_available():
        mem_quantized = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n  Memory used (quantized): {mem_quantized:.2f} GB")
    
    # ── Mode 2: Merged Inference ─────────────────────────────────
    print("\n\n[Mode 2] Merged inference (full precision)...")
    
    # For merging, we need the base model in FP16/FP32
    del model, base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.float16,
    )
    
    # Merge adapter into base model
    merged_model = model.merge_and_unload()
    merged_model.eval()
    
    print("  ✓ Adapter merged into base model")
    
    print("\n  ── Merged Inference Results ──")
    for prompt in prompts[:1]:  # Just one example
        inputs = tokenizer(prompt, return_tensors="pt").to(merged_model.device)
        
        with torch.no_grad():
            outputs = merged_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
        
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        instruction = prompt.split("\n")[1] if "\n" in prompt else prompt[:50]
        print(f"\n  Prompt: {instruction}")
        print(f"  Response: {response[:200]}")
    
    if torch.cuda.is_available():
        mem_merged = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n  Memory used (merged): {mem_merged:.2f} GB")
    
    return merged_model, tokenizer


# ============================================================================
# SECTION 4: PRACTICAL RECIPES
# ============================================================================

class QLoRARecipes:
    """
    Practical QLoRA recipes for different hardware configurations.
    These are battle-tested configurations for common scenarios.
    """
    
    @staticmethod
    def recipe_7b_24gb():
        """
        Recipe: Fine-tune a 7B model on a 24GB GPU (e.g., RTX 3090/4090).
        
        Memory budget:
        - Base model (4-bit):     ~3.5 GB
        - LoRA (r=64):            ~0.1 GB
        - Optimizer (8-bit):      ~0.1 GB
        - Activations (bs=4):     ~4-6 GB
        - Overhead:               ~2 GB
        ─────────────────────────────────
        Total:                    ~10-12 GB (fits in 24 GB!)
        """
        print("\n" + "=" * 65)
        print("  RECIPE: 7B MODEL ON 24GB GPU")
        print("=" * 65)
        
        config = '''
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch

# ── Quantization ─────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ── Load Model ───────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",   # If available
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# ── Prepare & Apply LoRA ─────────────────────────────────────────
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=64,                                      # Higher rank for 7B
    lora_alpha=16,                             # alpha/r = 0.25
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",   # MLP layers too
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected: ~1.5% trainable (about 100M params)

# ── Training Arguments ───────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./llama2-7b-qlora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,             # Effective batch: 16
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",                  # Paged optimizer
    fp16=False,
    bf16=True,                                 # BF16 compute
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=0.3,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    group_by_length=True,
    report_to="wandb",                         # Or "none"
)

# ── Train ────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    max_seq_length=2048,
)

trainer.train()
model.save_pretrained("./llama2-7b-qlora/final")
'''
        print(config)
        return config
    
    @staticmethod
    def recipe_13b_24gb():
        """
        Recipe: Fine-tune a 13B model on a 24GB GPU.
        
        Tighter memory budget — need smaller batch size and shorter sequences.
        """
        print("\n" + "=" * 65)
        print("  RECIPE: 13B MODEL ON 24GB GPU")
        print("=" * 65)
        
        config = '''
# Key differences from 7B recipe:
# - Smaller batch size (2 instead of 4)
# - More gradient accumulation (8 instead of 4)  
# - Shorter max sequence length (1024 instead of 2048)
# - Lower LoRA rank (32 instead of 64)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,            # Critical for 13B!
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

lora_config = LoraConfig(
    r=32,                                       # Lower rank to save memory
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",    
    ],                                          # Skip MLP to save memory
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir="./llama2-13b-qlora",
    num_train_epochs=2,
    per_device_train_batch_size=2,              # Smaller batch
    gradient_accumulation_steps=8,              # More accumulation
    learning_rate=1e-4,                         # Slightly lower LR
    optim="paged_adamw_8bit",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=0.3,
    save_strategy="steps",
    save_steps=100,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    max_seq_length=1024,                        # Shorter sequences
)

# Monitor memory during training:
#   Peak expected: ~18-20 GB
#   If OOM: reduce batch_size to 1 and increase grad_accum to 16
'''
        print(config)
        return config
    
    @staticmethod
    def recipe_70b_48gb():
        """
        Recipe: Fine-tune a 70B model on a 48GB GPU (e.g., A6000).
        
        This pushes the limits — every memory optimization matters.
        """
        print("\n" + "=" * 65)
        print("  RECIPE: 70B MODEL ON 48GB GPU")
        print("=" * 65)
        
        config = '''
# 70B in 4-bit ≈ 35 GB. Leaves ~13 GB for training.
# Every optimization is critical!

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,            # Saves ~3 GB on 70B
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto",                          # Spread across GPU
    attn_implementation="flash_attention_2",    # Critical for memory
)

lora_config = LoraConfig(
    r=16,                                       # Low rank to save memory
    lora_alpha=16,                              # alpha/r = 1.0
    target_modules=["q_proj", "v_proj"],        # Minimal targets
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir="./llama2-70b-qlora",
    num_train_epochs=1,                         # Single epoch
    per_device_train_batch_size=1,              # Minimum batch size
    gradient_accumulation_steps=16,             # Large accumulation
    learning_rate=5e-5,                         # Conservative LR
    optim="paged_adamw_8bit",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=0.3,
    save_strategy="steps",
    save_steps=50,
    dataloader_num_workers=0,                   # Minimize CPU memory
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    max_seq_length=512,                         # Short sequences
)

# If STILL OOM:
# 1. Reduce max_seq_length to 256
# 2. Use only ["q_proj"] as target
# 3. Set r=8 (minimum practical rank)
# 4. Use "paged_adamw_32bit" and try gradient offloading
'''
        print(config)
        return config
    
    @staticmethod
    def recipe_multi_gpu_fsdp():
        """
        Recipe: Multi-GPU QLoRA with FSDP (Fully Sharded Data Parallel).
        
        For when a single GPU isn't enough.
        """
        print("\n" + "=" * 65)
        print("  RECIPE: MULTI-GPU QLoRA WITH FSDP")
        print("=" * 65)
        
        config = '''
# ── fsdp_config.yaml ─────────────────────────────────────────────
# Save this as fsdp_config.yaml

fsdp_config = """
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4                 # Number of GPUs
"""

# ── Launch command ────────────────────────────────────────────────
# accelerate launch --config_file fsdp_config.yaml train_qlora.py

# ── Training script (train_qlora.py) ─────────────────────────────
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

accelerator = Accelerator()

# Each process loads the model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map={"": accelerator.process_index},  # Each GPU gets full model
)

lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# With FSDP, the LoRA params are sharded across GPUs
# So 4 GPUs with a 70B model:
#   Base model: ~35 GB / GPU (4-bit, each GPU has full copy)
#   LoRA params: ~200MB total, ~50MB/GPU (sharded)
#   Optimizer: ~400MB total, ~100MB/GPU (sharded)

training_args = TrainingArguments(
    output_dir="./llama2-70b-qlora-fsdp",
    per_device_train_batch_size=4,              # Per GPU
    gradient_accumulation_steps=2,              # Effective: 4*4*2 = 32
    learning_rate=2e-4,
    bf16=True,
    gradient_checkpointing=True,
    fsdp="full_shard auto_wrap",
    fsdp_transformer_layer_cls_to_wrap="LlamaDecoderLayer",
)
'''
        print(config)
        return config
    
    @staticmethod
    def print_recipe_comparison():
        """Print a comparison table of all recipes."""
        print("\n" + "=" * 65)
        print("  QLoRA RECIPE COMPARISON")
        print("=" * 65)
        
        table = """
┌────────────┬──────────┬──────────┬──────────┬──────────────┐
│ Setting    │ 7B/24GB  │ 13B/24GB │ 70B/48GB │ 70B/4×24GB   │
├────────────┼──────────┼──────────┼──────────┼──────────────┤
│ LoRA r     │ 64       │ 32       │ 16       │ 32           │
│ Targets    │ QKV+MLP  │ QKVO     │ Q,V only │ QKVO         │
│ Batch size │ 4        │ 2        │ 1        │ 4/GPU        │
│ Grad accum │ 4        │ 8        │ 16       │ 2            │
│ Eff. batch │ 16       │ 16       │ 16       │ 32           │
│ Seq length │ 2048     │ 1024     │ 512      │ 1024         │
│ LR         │ 2e-4     │ 1e-4     │ 5e-5     │ 2e-4         │
│ GPU mem    │ ~12 GB   │ ~20 GB   │ ~42 GB   │ ~38 GB/GPU   │
│ Speed      │ Fast     │ Medium   │ Slow     │ Fast         │
└────────────┴──────────┴──────────┴──────────┴──────────────┘

Key Principles:
  → As model grows: reduce rank, targets, batch, seq length
  → As GPUs increase: can increase rank and batch size
  → Always use: NF4 + double quant + paged_adamw_8bit
  → Always use: gradient checkpointing + flash attention
"""
        print(table)


# ============================================================================
# SECTION 5: ADAPTER MANAGEMENT FOR QLoRA
# ============================================================================

def demonstrate_qlora_adapter_management():
    """
    Show how to save, load, merge, and share QLoRA adapters.
    """
    print("=" * 65)
    print("  QLoRA ADAPTER MANAGEMENT")
    print("=" * 65)
    
    code = '''
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# ═══════════════════════════════════════════════════════════════════
# 1. SAVING ADAPTERS
# ═══════════════════════════════════════════════════════════════════

# After training, save ONLY the adapter (tiny!)
model.save_pretrained("./my_qlora_adapter")
# This saves:
#   adapter_config.json  (~1 KB) — LoRA hyperparameters
#   adapter_model.safetensors  (~50-200 MB) — LoRA weights

# Compare sizes:
#   Full 7B model:     ~13 GB (FP16)
#   QLoRA adapter:     ~50-200 MB (0.5-1.5% of full model)
#   Savings:           ~65-260x smaller!

# ═══════════════════════════════════════════════════════════════════
# 2. LOADING ADAPTERS (QUANTIZED)
# ═══════════════════════════════════════════════════════════════════

# Load base model in 4-bit (same config as training!)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Load adapter on top
model = PeftModel.from_pretrained(
    base_model,
    "./my_qlora_adapter",
)
model.eval()

# ═══════════════════════════════════════════════════════════════════
# 3. MERGING QLORA INTO FULL MODEL
# ═══════════════════════════════════════════════════════════════════

# IMPORTANT: Cannot merge directly from 4-bit model!
# Must load base model in FP16/FP32 first.

# Step 1: Load base model in higher precision
base_model_fp16 = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Step 2: Load QLoRA adapter onto FP16 model
model_fp16 = PeftModel.from_pretrained(
    base_model_fp16,
    "./my_qlora_adapter",
    torch_dtype=torch.float16,
)

# Step 3: Merge
merged_model = model_fp16.merge_and_unload()

# Step 4: Save merged model
merged_model.save_pretrained("./merged_7b_model", safe_serialization=True)

# The merged model is a standard HF model — no PEFT or BnB needed!

# ═══════════════════════════════════════════════════════════════════
# 4. SHARING ON HUGGING FACE HUB
# ═══════════════════════════════════════════════════════════════════

# Push adapter only (recommended — tiny upload)
model.push_to_hub(
    "your-username/llama2-7b-my-task-qlora",
    commit_message="QLoRA adapter for my task",
)
tokenizer.push_to_hub("your-username/llama2-7b-my-task-qlora")

# Or push merged model (large upload, but easier for users)
merged_model.push_to_hub(
    "your-username/llama2-7b-my-task-merged",
    commit_message="Merged QLoRA model",
)

# Users can then load adapter with:
#   model = PeftModel.from_pretrained(base, "your-username/llama2-7b-my-task-qlora")
# Or load merged model directly:
#   model = AutoModelForCausalLM.from_pretrained("your-username/llama2-7b-my-task-merged")

# ═══════════════════════════════════════════════════════════════════
# 5. RESUME TRAINING FROM CHECKPOINT
# ═══════════════════════════════════════════════════════════════════

# QLoRA checkpoints can be resumed:
from peft import PeftModel, LoraConfig

# Load quantized base
base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Load the checkpoint adapter
model = PeftModel.from_pretrained(
    base,
    "./qlora_output/checkpoint-500",           # Resume from step 500
    is_trainable=True,                         # Important! Enable training
)

# Continue training from checkpoint
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    resume_from_checkpoint="./qlora_output/checkpoint-500",
)
trainer.train(resume_from_checkpoint=True)
'''
    print(code)
    return code


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the complete QLoRA training demonstration."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║              QLoRA TRAINING PIPELINE                         ║")
    print("║                                                              ║")
    print("║  4-bit Quantized Base + LoRA Adapters + Paged Optimizers     ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: Configuration & memory estimation
    demonstrate_qlora_config()
    
    # Section 2: Run the training pipeline (with distilgpt2 demo)
    print("\n\n" + "═" * 65)
    print("  RUNNING TRAINING PIPELINE (distilgpt2 demo)")
    print("═" * 65)
    
    config = QLoRAConfig(
        base_model_name="distilgpt2",
        lora_r=8,
        lora_alpha=16,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        max_seq_length=256,
        output_dir="./qlora_demo_output",
        save_strategy="no",
        logging_steps=5,
    )
    
    model, tokenizer, result = qlora_training_pipeline(config)
    
    # Section 3: Practical recipes
    print("\n\n" + "═" * 65)
    print("  PRACTICAL RECIPES")
    print("═" * 65)
    
    recipes = QLoRARecipes()
    recipes.recipe_7b_24gb()
    recipes.recipe_13b_24gb()
    recipes.recipe_70b_48gb()
    recipes.recipe_multi_gpu_fsdp()
    recipes.print_recipe_comparison()
    
    # Section 4: Adapter management
    demonstrate_qlora_adapter_management()
    
    print("\n" + "=" * 65)
    print("  MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered in this module:
    ✓ QLoRA configuration and memory estimation
    ✓ Complete training pipeline (load → prepare → train → save)
    ✓ Quantized and merged inference
    ✓ Practical recipes: 7B, 13B, 70B, multi-GPU
    ✓ Adapter management: save, load, merge, share, resume
    """)


if __name__ == "__main__":
    main()
