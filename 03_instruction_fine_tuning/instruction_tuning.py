"""
═══════════════════════════════════════════════════════════════════════════
INSTRUCTION FINE-TUNING — Complete Training Pipeline
═══════════════════════════════════════════════════════════════════════════

This is the MAIN training script that ties everything together:
1. Load and prepare instruction dataset
2. Format with templates
3. Apply loss masking (train on outputs only)
4. Train with proper hyperparameters
5. Evaluate instruction-following ability

INSTRUCTION TUNING vs SFT:
──────────────────────────
While SFT is the technique, instruction tuning is the APPLICATION:
- SFT = training method (supervised, on demonstrations)
- Instruction Tuning = training GOAL (teach model to follow instructions)

Instruction tuning typically:
- Uses diverse task categories (not just one task)
- Emphasizes zero-shot generalization (follow any instruction)
- Focuses on instruction-output format (not free-form chat)
- Benefits from scaled task diversity (more tasks = better generalization)
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

# Local imports
from instruction_datasets import (
    create_instruction_dataset,
    load_instruction_dataset,
    balance_dataset_by_category,
    filter_instruction_quality,
)
from instruction_templates import (
    format_alpaca,
    format_chat_instruction,
    get_response_marker,
)
from self_instruct import (
    SelfInstructPipeline,
    SEED_INSTRUCTIONS,
    score_instruction_quality,
)
from evaluation_instruction import (
    InstructionFollowingEvaluator,
    run_evaluation_suite,
    compute_response_metrics,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class InstructionTuningConfig:
    """
    Configuration for instruction fine-tuning.

    KEY HYPERPARAMETERS FOR INSTRUCTION TUNING:
    ────────────────────────────────────────────
    - learning_rate: 1e-5 to 5e-5 (lower than pretraining)
    - epochs: 2-5 (watch for overfitting, especially with small datasets)
    - batch_size: 4-16 (effective, with gradient accumulation)
    - max_seq_length: 512-2048 (depends on instruction length distribution)
    - warmup_ratio: 0.03-0.1
    - weight_decay: 0.01-0.1
    - template: "alpaca" or "chat" (match your deployment format)

    CRITICAL CHOICES:
    - Loss masking: Train on OUTPUTS ONLY (mask instructions in loss)
    - Dataset balance: Ensure diverse task categories
    - Quality filtering: Remove low-quality examples before training
    """
    # Model
    model_name: str = "distilgpt2"  # Use small model for demo
    max_seq_length: int = 512

    # Template
    template_name: str = "alpaca"  # "alpaca", "chat", "dolly"
    mask_instruction: bool = True  # Critical: mask instruction tokens in loss

    # Dataset
    dataset_source: str = "local"  # "local", "dolly", "alpaca"
    max_examples: int = 1000
    balance_categories: bool = True
    quality_filter: bool = True
    train_split: float = 0.9

    # Training
    learning_rate: float = 2e-5
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # effective batch = 16
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"
    fp16: bool = False
    bf16: bool = False

    # Output
    output_dir: str = "./instruction_tuned_model"
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"

    # Generation (for evaluation)
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


# ═══════════════════════════════════════════════════════════════════════
# 2. DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════

def prepare_instruction_data(
    config: InstructionTuningConfig,
    tokenizer,
) -> Dict:
    """
    Complete data preparation pipeline for instruction tuning.

    STEPS:
    ──────
    1. Load raw instruction data
    2. (Optional) Quality filter
    3. (Optional) Balance across categories
    4. Format with template
    5. Tokenize with loss masking
    6. Split into train/eval
    """
    print("\n" + "=" * 60)
    print("PREPARING INSTRUCTION DATA")
    print("=" * 60)

    # ─── Step 1: Load data ────────────────────────────────────────
    print("\n[1/6] Loading instruction dataset...")
    if config.dataset_source == "local":
        raw_data = create_instruction_dataset()
        dataset = Dataset.from_list(raw_data)
    else:
        dataset = load_instruction_dataset(config.dataset_source)

    if config.max_examples and len(dataset) > config.max_examples:
        dataset = dataset.shuffle(seed=42).select(range(config.max_examples))

    print(f"  Loaded {len(dataset)} examples")

    # ─── Step 2: Quality filter ───────────────────────────────────
    if config.quality_filter:
        print("\n[2/6] Filtering low-quality examples...")
        before = len(dataset)
        dataset = filter_instruction_quality(dataset)
        print(f"  {before} → {len(dataset)} examples (removed {before - len(dataset)})")
    else:
        print("\n[2/6] Skipping quality filter")

    # ─── Step 3: Balance categories ───────────────────────────────
    if config.balance_categories and "category" in dataset.column_names:
        print("\n[3/6] Balancing categories...")
        dataset = balance_dataset_by_category(dataset)
        print(f"  Balanced to {len(dataset)} examples")
    else:
        print("\n[3/6] Skipping category balancing")

    # ─── Step 4: Format with template ─────────────────────────────
    print(f"\n[4/6] Formatting with '{config.template_name}' template...")

    def format_example(example):
        if config.template_name == "chat":
            text = format_chat_instruction(example, tokenizer, include_output=True)
        else:
            text = format_alpaca(example, include_output=True)
        return {"text": text}

    dataset = dataset.map(format_example, desc="Formatting")
    print(f"  Example formatted text length: {len(dataset[0]['text'])} chars")

    # ─── Step 5: Tokenize ─────────────────────────────────────────
    print(f"\n[5/6] Tokenizing (max_length={config.max_seq_length})...")

    response_marker = get_response_marker(config.template_name)

    def tokenize_and_mask(example):
        text = example["text"]

        # Tokenize full text
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=config.max_seq_length,
            padding=False,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        if config.mask_instruction:
            # Find where the response starts
            labels = list(input_ids)  # Copy

            # Tokenize just the instruction part (up to response marker)
            marker_pos = text.find(response_marker)
            if marker_pos != -1:
                instruction_part = text[:marker_pos + len(response_marker)]
                instruction_tokens = tokenizer(
                    instruction_part,
                    truncation=True,
                    max_length=config.max_seq_length,
                    padding=False,
                )
                instruction_len = len(instruction_tokens["input_ids"])

                # Mask instruction tokens (set to -100)
                for i in range(min(instruction_len, len(labels))):
                    labels[i] = -100
            else:
                # If marker not found, mask first half as safeguard
                half = len(labels) // 2
                for i in range(half):
                    labels[i] = -100
        else:
            labels = list(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    tokenized = dataset.map(
        tokenize_and_mask,
        remove_columns=dataset.column_names,
        desc="Tokenizing & masking",
    )

    # Show masking statistics
    sample_labels = tokenized[0]["labels"]
    masked = sum(1 for l in sample_labels if l == -100)
    total = len(sample_labels)
    print(f"  Sample: {masked}/{total} tokens masked ({masked/total:.1%} instruction, {1-masked/total:.1%} output)")

    # ─── Step 6: Train/eval split ─────────────────────────────────
    print(f"\n[6/6] Splitting train/eval ({config.train_split:.0%}/{1-config.train_split:.0%})...")
    split = tokenized.train_test_split(test_size=1 - config.train_split, seed=42)
    print(f"  Train: {len(split['train'])}, Eval: {len(split['test'])}")

    return {
        "train_dataset": split["train"],
        "eval_dataset": split["test"],
        "raw_dataset": dataset,
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. CUSTOM DATA COLLATOR WITH PADDING
# ═══════════════════════════════════════════════════════════════════════

class InstructionDataCollator:
    """
    Data collator that handles variable-length instruction sequences.

    Pads input_ids, attention_mask, and labels correctly:
    - input_ids padded with pad_token_id
    - attention_mask padded with 0
    - labels padded with -100 (ignored in loss)
    """

    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: List[Dict]) -> Dict:
        # Find max length in batch
        max_len = min(
            max(len(f["input_ids"]) for f in features),
            self.max_length,
        )

        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for f in features:
            # Truncate if needed
            input_ids = f["input_ids"][:max_len]
            attention_mask = f["attention_mask"][:max_len]
            labels = f["labels"][:max_len]

            # Pad
            padding_length = max_len - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)

        # Convert to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        return batch


# ═══════════════════════════════════════════════════════════════════════
# 4. TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_instruction_model(config: InstructionTuningConfig) -> Dict:
    """
    Complete instruction fine-tuning pipeline.

    PIPELINE:
    ─────────
    1. Load model & tokenizer
    2. Prepare data (load, filter, format, tokenize, mask)
    3. Configure training arguments
    4. Train with Trainer
    5. Evaluate instruction following
    6. Save model

    Returns dict with model, tokenizer, and evaluation results.
    """
    print("\n" + "═" * 70)
    print("  INSTRUCTION FINE-TUNING PIPELINE")
    print("═" * 70)

    # ─── Step 1: Load model ───────────────────────────────────────
    print(f"\n[STEP 1] Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name)

    # Ensure special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {param_count:,} total, {trainable:,} trainable")

    # ─── Step 2: Prepare data ─────────────────────────────────────
    print(f"\n[STEP 2] Preparing instruction data...")
    data = prepare_instruction_data(config, tokenizer)

    # ─── Step 3: Training arguments ───────────────────────────────
    print(f"\n[STEP 3] Configuring training...")

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
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        eval_strategy=config.eval_strategy,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        remove_unused_columns=False,
        seed=42,
    )

    effective_batch = (
        config.per_device_batch_size * config.gradient_accumulation_steps
    )
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  LR scheduler: {config.lr_scheduler}")
    print(f"  Loss masking: {'ON' if config.mask_instruction else 'OFF'}")

    # ─── Step 4: Train ────────────────────────────────────────────
    print(f"\n[STEP 4] Training...")

    data_collator = InstructionDataCollator(tokenizer, config.max_seq_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train_dataset"],
        eval_dataset=data["eval_dataset"],
        data_collator=data_collator,
    )

    train_result = trainer.train()

    print(f"\n  Training complete!")
    print(f"  Train loss: {train_result.training_loss:.4f}")
    print(f"  Train time: {train_result.metrics.get('train_runtime', 0):.1f}s")

    # ─── Step 5: Evaluate ─────────────────────────────────────────
    print(f"\n[STEP 5] Evaluating...")

    eval_metrics = trainer.evaluate()
    print(f"  Eval loss: {eval_metrics['eval_loss']:.4f}")
    import math
    print(f"  Eval perplexity: {math.exp(eval_metrics['eval_loss']):.2f}")

    # Generate some example responses
    print(f"\n[STEP 5b] Generating example responses...")

    test_instructions = [
        {"instruction": "Explain what a neural network is in simple terms.", "input": ""},
        {"instruction": "Write a haiku about programming.", "input": ""},
        {"instruction": "List three benefits of open-source software.", "input": ""},
    ]

    model.eval()
    for example in test_instructions:
        prompt = format_alpaca(example, include_output=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        print(f"\n  Instruction: {example['instruction']}")
        print(f"  Response: {response[:200]}...")

    # ─── Step 6: Save ─────────────────────────────────────────────
    print(f"\n[STEP 6] Saving model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    # Save config
    import json
    config_path = os.path.join(config.output_dir, "instruction_tuning_config.json")
    os.makedirs(config.output_dir, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2)

    print(f"\n{'═' * 70}")
    print("  INSTRUCTION FINE-TUNING COMPLETE!")
    print(f"{'═' * 70}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "train_result": train_result,
        "eval_metrics": eval_metrics,
    }


# ═══════════════════════════════════════════════════════════════════════
# 5. INFERENCE UTILITY
# ═══════════════════════════════════════════════════════════════════════

def run_instruction(
    instruction: str,
    model,
    tokenizer,
    input_text: str = "",
    template: str = "alpaca",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Run a single instruction through the fine-tuned model.

    This is the inference function you'd use in production.
    """
    example = {"instruction": instruction, "input": input_text}

    if template == "chat":
        prompt = format_chat_instruction(example, tokenizer, include_output=False)
    else:
        prompt = format_alpaca(example, include_output=False)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return response.strip()


# ═══════════════════════════════════════════════════════════════════════
# MAIN — Run the full pipeline
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("INSTRUCTION FINE-TUNING MASTERCLASS")
    print("=" * 70)
    print()
    print("This script demonstrates the complete instruction tuning pipeline:")
    print("  1. Dataset creation and preparation")
    print("  2. Template formatting (Alpaca style)")
    print("  3. Tokenization with loss masking")
    print("  4. Training with HuggingFace Trainer")
    print("  5. Evaluation (instruction following + generation)")
    print("  6. Model saving and inference")
    print()

    # Configure
    config = InstructionTuningConfig(
        model_name="distilgpt2",  # Small model for demo
        max_seq_length=512,
        template_name="alpaca",
        mask_instruction=True,
        dataset_source="local",
        max_examples=200,  # Small for demo speed
        balance_categories=True,
        quality_filter=True,
        learning_rate=2e-5,
        num_epochs=3,
        per_device_batch_size=4,
        gradient_accumulation_steps=4,
        output_dir="./instruction_tuned_distilgpt2",
    )

    # Train
    results = train_instruction_model(config)

    # Interactive demo
    print("\n" + "=" * 70)
    print("INTERACTIVE DEMO")
    print("=" * 70)
    print("Type an instruction (or 'quit' to exit):\n")

    model = results["model"]
    tokenizer = results["tokenizer"]

    demo_instructions = [
        "What is machine learning?",
        "Write a short poem about the moon.",
        "List 3 tips for better sleep.",
    ]

    for instruction in demo_instructions:
        print(f"\n> {instruction}")
        response = run_instruction(instruction, model, tokenizer)
        print(f"  {response[:300]}")
