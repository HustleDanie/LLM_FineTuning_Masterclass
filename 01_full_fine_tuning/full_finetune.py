"""
═══════════════════════════════════════════════════════════════════════════
FULL FINE-TUNING — Complete Training Script
═══════════════════════════════════════════════════════════════════════════

This is the main training script for FULL fine-tuning of a causal language model.
It demonstrates the complete pipeline:

1. Load pretrained model (all parameters)
2. Prepare dataset with proper tokenization
3. Configure training with all optimizations
4. Train with gradient checkpointing, mixed precision, etc.
5. Evaluate and save the fine-tuned model

WHAT MAKES THIS "FULL" FINE-TUNING:
────────────────────────────────────
Every single parameter in the model is updated during training.
No freezing, no adapters, no low-rank approximations.
The model's entire neural network adapts to your data.

Usage:
    python full_finetune.py
    python full_finetune.py --model_name gpt2-medium --epochs 5
    torchrun --nproc_per_node=4 full_finetune.py  # Multi-GPU
"""

import os
import sys
import argparse
import math
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

# Local imports
from config import ModelConfig, DataConfig, TrainingConfig, print_config_summary
from data_utils import (
    load_text_dataset,
    tokenize_for_causal_lm,
    get_data_collator,
    inspect_dataset,
    compute_dataset_statistics,
)
from training_utils import (
    analyze_model_parameters,
    setup_gradient_checkpointing,
    check_mixed_precision_support,
    build_training_arguments,
    unfreeze_all,
    DetailedLoggingCallback,
    GradientMonitorCallback,
)
from evaluation import (
    compute_perplexity,
    generate_text,
    analyze_weight_changes,
    print_training_summary,
)


def parse_args():
    """Parse command-line arguments (override config defaults)."""
    parser = argparse.ArgumentParser(description="Full Fine-Tuning of LLMs")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Pretrained model name or path")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="HuggingFace dataset name")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Per-device training batch size")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=None,
                        help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit training samples (for debugging)")
    return parser.parse_args()


def main():
    # ── Step 1: Configuration ────────────────────────────────────────
    print("\n" + "█" * 60)
    print("█  FULL FINE-TUNING PIPELINE")
    print("█" * 60)

    args = parse_args()
    model_cfg = ModelConfig()
    data_cfg = DataConfig()
    train_cfg = TrainingConfig()

    # Override from CLI
    if args.model_name:
        model_cfg.model_name_or_path = args.model_name
    if args.dataset_name:
        data_cfg.dataset_name = args.dataset_name
    if args.epochs:
        train_cfg.num_train_epochs = args.epochs
    if args.batch_size:
        train_cfg.per_device_train_batch_size = args.batch_size
    if args.learning_rate:
        train_cfg.learning_rate = args.learning_rate
    if args.max_seq_length:
        data_cfg.max_seq_length = args.max_seq_length
    if args.output_dir:
        train_cfg.output_dir = args.output_dir
    if args.max_train_samples:
        data_cfg.max_train_samples = args.max_train_samples

    print_config_summary(model_cfg, data_cfg, train_cfg)

    # Set seed for reproducibility
    set_seed(train_cfg.seed)

    # ── Step 2: Load Tokenizer ───────────────────────────────────────
    print("\n📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.tokenizer_name or model_cfg.model_name_or_path,
        trust_remote_code=model_cfg.trust_remote_code,
    )

    # GPT-2 doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"   Set pad_token = eos_token ({tokenizer.eos_token})")

    print(f"   Vocab size: {len(tokenizer):,}")

    # ── Step 3: Load Model ───────────────────────────────────────────
    print("\n📦 Loading pretrained model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name_or_path,
        torch_dtype=model_cfg.torch_dtype,
        trust_remote_code=model_cfg.trust_remote_code,
    )

    # CRITICAL: Ensure ALL parameters are trainable (FULL fine-tuning)
    model = unfreeze_all(model)

    # Analyze parameters — see what we're training
    param_analysis = analyze_model_parameters(model)

    # Save original weights for comparison after training
    print("   Saving original weights for post-training comparison...")
    original_state_dict = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    # ── Step 4: Setup Gradient Checkpointing ─────────────────────────
    if train_cfg.gradient_checkpointing:
        model = setup_gradient_checkpointing(model, enable=True)
        # Required for gradient checkpointing with HF Trainer
        model.config.use_cache = False

    # ── Step 5: Load and Prepare Dataset ─────────────────────────────
    print("\n📊 Preparing dataset...")
    raw_datasets = load_text_dataset(
        dataset_name=data_cfg.dataset_name,
        dataset_config=data_cfg.dataset_config,
        max_train_samples=data_cfg.max_train_samples,
        max_eval_samples=data_cfg.max_eval_samples,
    )

    # Tokenize and chunk
    tokenized_datasets = tokenize_for_causal_lm(
        raw_datasets=raw_datasets,
        tokenizer=tokenizer,
        max_seq_length=data_cfg.max_seq_length,
        num_proc=data_cfg.preprocessing_num_workers,
    )

    # Inspect the data
    train_dataset = tokenized_datasets[data_cfg.train_split]
    eval_dataset = tokenized_datasets[data_cfg.validation_split]

    inspect_dataset(train_dataset, tokenizer, n_samples=2)
    stats = compute_dataset_statistics(train_dataset)
    print(f"\n📈 Dataset Statistics:")
    print(f"   Training examples: {stats['num_examples']:,}")
    print(f"   Total tokens: {stats['total_tokens']:,}")
    print(f"   Avg sequence length: {stats['avg_length']:.0f}")

    # ── Step 6: Data Collator ────────────────────────────────────────
    data_collator = get_data_collator("causal_lm", tokenizer)

    # ── Step 7: Build Training Arguments ─────────────────────────────
    print("\n⚙️ Building training arguments...")
    training_args = build_training_arguments(train_cfg)

    # ── Step 8: Setup Callbacks ──────────────────────────────────────
    callbacks = [
        DetailedLoggingCallback(),
        GradientMonitorCallback(log_every_n_steps=100),
    ]

    # Early stopping
    if train_cfg.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=train_cfg.early_stopping_patience
            )
        )
        print(f"   Early stopping enabled (patience={train_cfg.early_stopping_patience})")

    # ── Step 9: Pre-training Generation Test ─────────────────────────
    test_prompts = [
        "The meaning of life is",
        "Machine learning can be used to",
        "In the year 2050,",
    ]

    print("\n📝 Pre-training generation sample:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pre_training_outputs = generate_text(
        model, tokenizer, test_prompts, max_new_tokens=50
    )
    for prompt, output in zip(test_prompts, pre_training_outputs):
        print(f"   Prompt: {prompt}")
        print(f"   Output: {output[:150]}...")
        print()

    # ── Step 10: Initialize Trainer ──────────────────────────────────
    print("\n🏋️ Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # ── Step 11: TRAIN! ──────────────────────────────────────────────
    print("\n" + "🔥" * 30)
    print("   STARTING FULL FINE-TUNING")
    print("🔥" * 30)

    # Check for resuming from checkpoint
    last_checkpoint = None
    if os.path.isdir(train_cfg.output_dir):
        from transformers.trainer_utils import get_last_checkpoint
        last_checkpoint = get_last_checkpoint(train_cfg.output_dir)
        if last_checkpoint:
            print(f"   Resuming from checkpoint: {last_checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # ── Step 12: Save Final Model ────────────────────────────────────
    print("\n💾 Saving fine-tuned model...")
    trainer.save_model(os.path.join(train_cfg.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(train_cfg.output_dir, "final_model"))

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # ── Step 13: Evaluate ────────────────────────────────────────────
    print("\n📊 Final Evaluation...")
    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics["eval_loss"]
    perplexity = compute_perplexity(eval_loss)

    print(f"\n📊 Final Results:")
    print(f"   Eval Loss: {eval_loss:.4f}")
    print(f"   Perplexity: {perplexity:.2f}")

    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # ── Step 14: Post-training Generation Test ───────────────────────
    print("\n📝 Post-training generation sample:")
    post_training_outputs = generate_text(
        model, tokenizer, test_prompts, max_new_tokens=50
    )
    for prompt, output in zip(test_prompts, post_training_outputs):
        print(f"   Prompt: {prompt}")
        print(f"   Output: {output[:150]}...")
        print()

    # ── Step 15: Weight Change Analysis ──────────────────────────────
    print("\n🔍 Analyzing weight changes...")
    finetuned_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    analyze_weight_changes(original_state_dict, finetuned_state_dict)

    # ── Step 16: Training Summary ────────────────────────────────────
    logging_callback = [c for c in callbacks if isinstance(c, DetailedLoggingCallback)][0]
    print_training_summary(
        logging_callback.train_losses,
        logging_callback.eval_losses,
        logging_callback.learning_rates,
    )

    print("\n" + "█" * 60)
    print("█  FULL FINE-TUNING COMPLETE!")
    print(f"█  Model saved to: {train_cfg.output_dir}/final_model")
    print("█" * 60)


if __name__ == "__main__":
    main()
