"""
═══════════════════════════════════════════════════════════════════════════
FULL FINE-TUNING — Causal Language Model (Text Generation)
═══════════════════════════════════════════════════════════════════════════

This script demonstrates full fine-tuning specifically for TEXT GENERATION tasks.
The model learns to generate text in the style of your training data.

USE CASES:
  - Domain-specific text generation (medical reports, legal documents)
  - Creative writing in a specific style
  - Code generation for specific languages/frameworks
  - Chatbot personality adaptation

This is a simplified, focused version of full_finetune.py for quick experiments.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from datasets import load_dataset


def main():
    # ═══════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ═══════════════════════════════════════════════════════════════
    MODEL_NAME = "distilgpt2"      # Small model for demonstration
    DATASET = "wikitext"
    DATASET_CONFIG = "wikitext-2-raw-v1"
    MAX_LENGTH = 256               # Shorter for faster training
    EPOCHS = 2
    BATCH_SIZE = 4
    LEARNING_RATE = 3e-5
    OUTPUT_DIR = "./results/causal_lm"

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ═══════════════════════════════════════════════════════════════
    # LOAD MODEL & TOKENIZER
    # ═══════════════════════════════════════════════════════════════
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Set pad token (GPT-2 doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # KEY POINT: All parameters are trainable (FULL fine-tuning)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    model.to(device)

    # ═══════════════════════════════════════════════════════════════
    # LOAD & TOKENIZE DATASET
    # ═══════════════════════════════════════════════════════════════
    print(f"\nLoading dataset: {DATASET}...")
    dataset = load_dataset(DATASET, DATASET_CONFIG)

    def tokenize_and_chunk(examples):
        """Tokenize and concatenate texts into fixed-length chunks."""
        tokenized = tokenizer(
            examples["text"],
            truncation=False,
            padding=False,
        )

        # Concatenate all tokens
        all_input_ids = sum(tokenized["input_ids"], [])
        all_attention_mask = sum(tokenized["attention_mask"], [])

        # Chunk into blocks of MAX_LENGTH
        total_length = (len(all_input_ids) // MAX_LENGTH) * MAX_LENGTH
        input_id_chunks = [
            all_input_ids[i:i + MAX_LENGTH]
            for i in range(0, total_length, MAX_LENGTH)
        ]
        attention_mask_chunks = [
            all_attention_mask[i:i + MAX_LENGTH]
            for i in range(0, total_length, MAX_LENGTH)
        ]

        return {
            "input_ids": input_id_chunks,
            "attention_mask": attention_mask_chunks,
            "labels": input_id_chunks,  # For causal LM, labels = input_ids
        }

    tokenized_dataset = dataset.map(
        tokenize_and_chunk,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    print(f"Training examples: {len(tokenized_dataset['train'])}")
    print(f"Validation examples: {len(tokenized_dataset['validation'])}")

    # ═══════════════════════════════════════════════════════════════
    # GENERATE BEFORE TRAINING (baseline)
    # ═══════════════════════════════════════════════════════════════
    print("\n── Before Fine-Tuning ──")
    test_prompt = "The history of science shows that"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50, do_sample=True,
                                temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    # ═══════════════════════════════════════════════════════════════
    # TRAINING
    # ═══════════════════════════════════════════════════════════════
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        weight_decay=0.01,
        max_grad_norm=1.0,
        gradient_checkpointing=True,  # Save memory
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        seed=42,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )

    print("\n🔥 Starting full fine-tuning (Causal LM)...")
    trainer.train()

    # ═══════════════════════════════════════════════════════════════
    # EVALUATE
    # ═══════════════════════════════════════════════════════════════
    eval_results = trainer.evaluate()
    perplexity = 2.71828 ** eval_results["eval_loss"]
    print(f"\n📊 Eval Loss: {eval_results['eval_loss']:.4f}")
    print(f"📊 Perplexity: {perplexity:.2f}")

    # ═══════════════════════════════════════════════════════════════
    # GENERATE AFTER TRAINING
    # ═══════════════════════════════════════════════════════════════
    print("\n── After Fine-Tuning ──")
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50, do_sample=True,
                                temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    # ═══════════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════════
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"\n✅ Model saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
