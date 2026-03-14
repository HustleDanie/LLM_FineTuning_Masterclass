"""
═══════════════════════════════════════════════════════════════════════════
PEFT TRAINING PIPELINE — Complete Training with PEFT (LoRA Demo)
═══════════════════════════════════════════════════════════════════════════

This module demonstrates the full PEFT training workflow using the
HuggingFace `peft` library. We use LoRA as the primary example since
it's the most widely used PEFT method.

WORKFLOW:
─────────
1. Load base model
2. Configure PEFT (LoRA)
3. Apply PEFT — automatically freezes base model
4. Train with standard Trainer (only PEFT params update)
5. Save adapter weights (tiny checkpoint: ~10-50MB vs multi-GB)
6. Load and merge for inference
"""

import os
import json
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

# PEFT imports (with fallback)
try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        PeftModel,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠ peft not installed. Install: pip install peft")


# ═══════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PEFTTrainingConfig:
    """Configuration for PEFT training pipeline."""
    # Model
    model_name: str = "distilgpt2"
    max_seq_length: int = 256

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    lora_bias: str = "none"

    # Training
    learning_rate: float = 2e-4  # Higher LR for PEFT than full FT!
    num_epochs: int = 3
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    lr_scheduler: str = "cosine"
    fp16: bool = False

    # Output
    output_dir: str = "./peft_lora_model"
    logging_steps: int = 10

    def __post_init__(self):
        if self.lora_target_modules is None:
            # GPT-2 uses combined attention matrix 'c_attn' and output 'c_proj'
            self.lora_target_modules = ["c_attn"]


# ═══════════════════════════════════════════════════════════════════════
# 2. DEMO DATASET
# ═══════════════════════════════════════════════════════════════════════

def create_demo_dataset(n_samples: int = 200) -> Dataset:
    """
    Create a small demo instruction dataset for PEFT training.
    """
    examples = [
        # Knowledge
        {"text": "### Instruction: What is Python?\n### Response: Python is a high-level, interpreted programming language known for its readability and versatility. It supports multiple paradigms including procedural, object-oriented, and functional programming."},
        {"text": "### Instruction: Explain what a database is.\n### Response: A database is an organized collection of structured information stored electronically. It allows for efficient storage, retrieval, and manipulation of data using a database management system (DBMS)."},
        {"text": "### Instruction: What is machine learning?\n### Response: Machine learning is a subset of artificial intelligence where computer systems learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data."},
        {"text": "### Instruction: Describe the solar system.\n### Response: The solar system consists of the Sun and everything that orbits around it, including eight planets, dwarf planets, moons, asteroids, and comets. The planets in order are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."},
        {"text": "### Instruction: What is photosynthesis?\n### Response: Photosynthesis is the process by which green plants and some organisms convert light energy into chemical energy. Using sunlight, water, and carbon dioxide, plants produce glucose and oxygen."},
        # Creative
        {"text": "### Instruction: Write a haiku about rain.\n### Response: Drops fall from gray skies\nPuddles form on tired streets\nEarth drinks and grows green"},
        {"text": "### Instruction: Write a short story opening.\n### Response: The clock struck midnight as Sarah opened the dusty journal she'd found hidden behind the wallpaper. The first entry was dated exactly one hundred years ago, written in her own handwriting."},
        # Analysis
        {"text": "### Instruction: Compare cats and dogs as pets.\n### Response: Cats are independent, low-maintenance pets that groom themselves and are suited for smaller spaces. Dogs are more social, require regular walks and training, but offer strong companionship and loyalty. Both make wonderful pets depending on lifestyle."},
        {"text": "### Instruction: What are the pros of remote work?\n### Response: Remote work offers flexible schedules, eliminates commute time, reduces office costs, enables access to global talent, improves work-life balance, and often increases productivity due to fewer office distractions."},
        # Coding
        {"text": "### Instruction: Write a Python function to reverse a string.\n### Response: def reverse_string(s):\n    return s[::-1]\n\n# Example usage:\n# reverse_string('hello')  # Returns 'olleh'"},
        {"text": "### Instruction: Explain what a for loop is.\n### Response: A for loop is a control flow statement that iterates over a sequence of elements. It executes a block of code repeatedly for each item in the sequence, such as items in a list, characters in a string, or a range of numbers."},
        {"text": "### Instruction: What is an API?\n### Response: An API (Application Programming Interface) is a set of protocols and tools that allows different software applications to communicate with each other. It defines the methods and data formats for requesting and exchanging information between systems."},
        # Summarization
        {"text": "### Instruction: Summarize the concept of supply and demand.\n### Response: Supply and demand is an economic model where price is determined by the relationship between product availability (supply) and consumer desire (demand). When demand exceeds supply, prices rise; when supply exceeds demand, prices fall."},
        # Reasoning
        {"text": "### Instruction: Why is the sky blue?\n### Response: The sky appears blue because of Rayleigh scattering. Sunlight contains all colors, but shorter blue wavelengths are scattered more by atmospheric molecules than longer red wavelengths. This scattered blue light reaches our eyes from all directions, making the sky appear blue."},
        {"text": "### Instruction: How does gravity work?\n### Response: Gravity is a fundamental force of attraction between objects with mass. According to Einstein's general relativity, massive objects curve spacetime around them, and other objects follow these curves. The strength of gravity depends on mass and distance between objects."},
    ]

    # Repeat and slightly vary to reach n_samples
    dataset_list = []
    for i in range(n_samples):
        dataset_list.append(examples[i % len(examples)])

    return Dataset.from_list(dataset_list)


# ═══════════════════════════════════════════════════════════════════════
# 3. PEFT MODEL SETUP
# ═══════════════════════════════════════════════════════════════════════

def setup_peft_model(config: PEFTTrainingConfig):
    """
    Load base model and apply PEFT (LoRA).

    WHAT HAPPENS INSIDE get_peft_model():
    ──────────────────────────────────────
    1. Freezes ALL base model parameters (requires_grad = False)
    2. Identifies target modules matching target_modules patterns
    3. Replaces target Linear layers with LoRA equivalents
    4. LoRA layers have both frozen 'weight' and trainable 'lora_A', 'lora_B'
    5. Returns a PeftModel wrapper

    The resulting model:
    - Base weights: FROZEN (loaded but not updated)
    - LoRA A matrices: TRAINABLE (initialized with random Gaussian)
    - LoRA B matrices: TRAINABLE (initialized to zero)
    - Other layers: FROZEN
    """
    print(f"\n{'═' * 60}")
    print("SETTING UP PEFT MODEL")
    print(f"{'═' * 60}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print(f"\n[1] Loading base model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(config.model_name)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Base model parameters: {total_params:,}")

    if not PEFT_AVAILABLE:
        print("    ⚠ PEFT not available. Returning base model for demonstration.")
        return model, tokenizer

    # Configure LoRA
    print(f"\n[2] Configuring LoRA:")
    print(f"    Rank: {config.lora_rank}")
    print(f"    Alpha: {config.lora_alpha}")
    print(f"    Target modules: {config.lora_target_modules}")
    print(f"    Dropout: {config.lora_dropout}")

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply PEFT
    print(f"\n[3] Applying LoRA to model...")
    model = get_peft_model(model, lora_config)

    # Print trainable parameter summary
    model.print_trainable_parameters()

    # Detailed breakdown
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\n    Trainable: {trainable:,} ({trainable/total_params*100:.4f}%)")
    print(f"    Frozen:    {frozen:,} ({frozen/total_params*100:.4f}%)")

    # Show which modules have LoRA
    print(f"\n[4] LoRA modules:")
    for name, module in model.named_modules():
        if "lora_" in name and hasattr(module, "weight"):
            print(f"    {name}: {tuple(module.weight.shape)}")

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════
# 4. TOKENIZATION
# ═══════════════════════════════════════════════════════════════════════

def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int = 256) -> Dataset:
    """Tokenize dataset for causal language modeling."""
    def tokenize(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(
        tokenize,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    return tokenized


# ═══════════════════════════════════════════════════════════════════════
# 5. TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_peft_model(config: PEFTTrainingConfig) -> Dict:
    """
    Complete PEFT training pipeline.

    KEY DIFFERENCES FROM FULL FINE-TUNING:
    ──────────────────────────────────────
    1. Learning rate is HIGHER (2e-4 vs 2e-5)
       - LoRA params need larger updates since they start from zero
       - Base model is frozen so no risk of catastrophic forgetting

    2. Fewer epochs needed
       - PEFT has fewer params so converges faster
       - But also more prone to overfitting (monitor eval loss!)

    3. Memory usage is MUCH lower
       - Only LoRA params stored in optimizer states
       - Gradients only computed for LoRA params
       - GPU memory dominated by model weights + activations

    4. Checkpoint size is tiny
       - Only save LoRA adapter weights (~10-50 MB)
       - Base model weights are NOT saved (just referenced)
    """
    # Setup model
    model, tokenizer = setup_peft_model(config)

    # Prepare data
    print(f"\n{'═' * 60}")
    print("PREPARING DATA")
    print(f"{'═' * 60}")
    dataset = create_demo_dataset(200)
    tokenized = tokenize_dataset(dataset, tokenizer, config.max_seq_length)

    split = tokenized.train_test_split(test_size=0.1, seed=42)
    print(f"  Train: {len(split['train'])}, Eval: {len(split['test'])}")

    # Training arguments
    print(f"\n{'═' * 60}")
    print("TRAINING")
    print(f"{'═' * 60}")

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type=config.lr_scheduler,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        seed=42,
    )

    effective_batch = config.per_device_batch_size * config.gradient_accumulation_steps
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Total epochs: {config.num_epochs}")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=data_collator,
    )

    # Train!
    train_result = trainer.train()

    print(f"\n  ✓ Training complete!")
    print(f"  Train loss: {train_result.training_loss:.4f}")
    print(f"  Runtime: {train_result.metrics.get('train_runtime', 0):.1f}s")

    # Evaluate
    eval_metrics = trainer.evaluate()
    import math
    print(f"  Eval loss: {eval_metrics['eval_loss']:.4f}")
    print(f"  Eval perplexity: {math.exp(eval_metrics['eval_loss']):.2f}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "trainer": trainer,
        "train_result": train_result,
        "eval_metrics": eval_metrics,
    }


# ═══════════════════════════════════════════════════════════════════════
# 6. SAVING & LOADING PEFT MODELS
# ═══════════════════════════════════════════════════════════════════════

def save_peft_model(model, tokenizer, output_dir: str):
    """
    Save PEFT adapter weights.

    WHAT GETS SAVED:
    ────────────────
    - adapter_config.json: PEFT configuration (method, rank, target_modules, etc.)
    - adapter_model.safetensors: Only the adapter weights (~10-50 MB!)

    WHAT IS NOT SAVED:
    ──────────────────
    - Base model weights (they're just referenced by model name)
    - Optimizer states (not needed for inference)

    This is why PEFT checkpoints are so small!
    """
    print(f"\n{'═' * 60}")
    print("SAVING PEFT MODEL")
    print(f"{'═' * 60}")

    os.makedirs(output_dir, exist_ok=True)

    if PEFT_AVAILABLE and hasattr(model, 'save_pretrained'):
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Show saved files
        for f in os.listdir(output_dir):
            size = os.path.getsize(os.path.join(output_dir, f))
            print(f"  {f}: {size / 1024:.1f} KB")
    else:
        print("  (PEFT not available — would save adapter_config.json + adapter_model.safetensors)")


def load_peft_model(base_model_name: str, adapter_dir: str):
    """
    Load a PEFT model for inference.

    LOADING WORKFLOW:
    ─────────────────
    1. Load the BASE model (full size)
    2. Load adapter weights on top (tiny)
    3. Optionally MERGE for faster inference

    The adapter weights are loaded and applied to the base model,
    adding the LoRA low-rank updates to the appropriate layers.
    """
    print(f"\n{'═' * 60}")
    print("LOADING PEFT MODEL")
    print(f"{'═' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if PEFT_AVAILABLE:
        # Load base model
        print(f"  [1] Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

        # Load adapter
        print(f"  [2] Loading adapter from: {adapter_dir}")
        model = PeftModel.from_pretrained(base_model, adapter_dir)

        print(f"  ✓ Model loaded with adapter")
        return model, tokenizer
    else:
        print("  (PEFT not available — returning base model)")
        model = AutoModelForCausalLM.from_pretrained(base_model_name)
        return model, tokenizer


def merge_and_unload(model):
    """
    Merge LoRA weights into base model and remove adapter.

    WHEN TO MERGE:
    ──────────────
    - For deployment: Merged model runs at full speed with no overhead
    - When you don't need to swap adapters
    - For converting to other formats (GGUF, ONNX, etc.)

    WHEN NOT TO MERGE:
    ──────────────────
    - When you want to swap adapters at runtime
    - When running multiple adapters (multi-task)
    - When you need to continue training
    """
    if PEFT_AVAILABLE and hasattr(model, 'merge_and_unload'):
        print("\n  Merging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
        print("  ✓ Merged! Model is now a standard model with no adapter overhead.")
        return merged_model
    else:
        print("  (Merge not available)")
        return model


# ═══════════════════════════════════════════════════════════════════════
# 7. GENERATION/INFERENCE
# ═══════════════════════════════════════════════════════════════════════

def generate_with_peft(
    model,
    tokenizer,
    instruction: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
) -> str:
    """Generate a response using the PEFT-trained model."""
    prompt = f"### Instruction: {instruction}\n### Response:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


# ═══════════════════════════════════════════════════════════════════════
# MAIN — Run the full PEFT pipeline
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 70)
    print("  PEFT TRAINING PIPELINE — LoRA DEMO")
    print("═" * 70)
    print()
    print("This demonstrates the complete PEFT lifecycle:")
    print("  1. Load base model")
    print("  2. Apply LoRA (freeze base, add trainable adapters)")
    print("  3. Train (only adapter params update)")
    print("  4. Save (tiny checkpoint)")
    print("  5. Load & merge (for deployment)")
    print("  6. Generate (inference)")
    print()

    # Configure
    config = PEFTTrainingConfig(
        model_name="distilgpt2",
        max_seq_length=256,
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules=["c_attn"],
        learning_rate=2e-4,
        num_epochs=3,
        per_device_batch_size=8,
        output_dir="./peft_demo_lora",
    )

    # Train
    results = train_peft_model(config)

    model = results["model"]
    tokenizer = results["tokenizer"]

    # Save
    save_peft_model(model, tokenizer, config.output_dir)

    # Generate examples
    print(f"\n{'═' * 60}")
    print("GENERATION EXAMPLES")
    print(f"{'═' * 60}")

    test_prompts = [
        "What is Python?",
        "Write a haiku about coding.",
        "Explain what a neural network is.",
        "What are the benefits of exercise?",
    ]

    for prompt in test_prompts:
        response = generate_with_peft(model, tokenizer, prompt, max_new_tokens=100)
        print(f"\n  Q: {prompt}")
        print(f"  A: {response[:200]}")

    print(f"\n{'═' * 60}")
    print("  PEFT PIPELINE COMPLETE!")
    print(f"{'═' * 60}")
