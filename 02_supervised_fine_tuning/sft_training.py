"""
═══════════════════════════════════════════════════════════════════════════
SUPERVISED FINE-TUNING — Using TRL's SFTTrainer (Industry Standard)
═══════════════════════════════════════════════════════════════════════════

This script demonstrates the RECOMMENDED way to do SFT in production
using HuggingFace TRL's SFTTrainer.

TRL (Transformer Reinforcement Learning) is the industry-standard library
for SFT, RLHF, and DPO. SFTTrainer handles:
  - Automatic chat template application
  - Response-only loss masking (via DataCollatorForCompletionOnlyLM)
  - Sequence packing for efficiency
  - PEFT/LoRA integration
  - All standard HuggingFace Trainer features

Pipeline:
  1. Load pretrained model
  2. Load prompt-response dataset
  3. Apply conversation template
  4. Train with response-only loss
  5. Evaluate and save

Usage:
    pip install trl
    python sft_training.py
"""

import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

# TRL — the industry standard for SFT
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM


def create_demo_dataset() -> Dataset:
    """
    Create a small demonstration SFT dataset.

    In production, you would use:
    - databricks/databricks-dolly-15k (15K, open license)
    - tatsu-lab/alpaca (52K, Stanford)
    - HuggingFaceH4/ultrachat_200k (200K, multi-turn)
    - Your own curated prompt-response pairs
    """
    examples = [
        {
            "prompt": "What is machine learning?",
            "response": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data, learn from it, and make predictions or decisions."
        },
        {
            "prompt": "Explain the difference between supervised and unsupervised learning.",
            "response": "Supervised learning uses labeled data where both inputs and desired outputs are provided, allowing the model to learn the mapping between them. Examples include classification and regression. Unsupervised learning works with unlabeled data, finding hidden patterns or structures. Examples include clustering and dimensionality reduction."
        },
        {
            "prompt": "What is a neural network?",
            "response": "A neural network is a computing system inspired by biological neural networks in the brain. It consists of layers of interconnected nodes (neurons) that process information. Each connection has a weight that adjusts during training. Neural networks can learn complex patterns and are the foundation of deep learning."
        },
        {
            "prompt": "How does gradient descent work?",
            "response": "Gradient descent is an optimization algorithm used to minimize a loss function. It works by: 1) Computing the gradient (direction of steepest increase) of the loss function with respect to model parameters, 2) Updating parameters in the opposite direction of the gradient, 3) Repeating until convergence. The learning rate controls the step size of each update."
        },
        {
            "prompt": "What is transfer learning?",
            "response": "Transfer learning is a technique where a model trained on one task is reused as the starting point for a model on a different but related task. Instead of training from scratch, you leverage knowledge from a pretrained model. This is especially useful when you have limited data for your target task. Fine-tuning LLMs is a form of transfer learning."
        },
        {
            "prompt": "Explain what overfitting is and how to prevent it.",
            "response": "Overfitting occurs when a model learns the training data too well, including noise and outliers, resulting in poor performance on new unseen data. Prevention strategies include: 1) Using more training data, 2) Applying regularization (L1, L2, dropout), 3) Early stopping, 4) Data augmentation, 5) Using simpler models, 6) Cross-validation for hyperparameter tuning."
        },
        {
            "prompt": "What are transformers in deep learning?",
            "response": "Transformers are a neural network architecture introduced in 2017 that relies on self-attention mechanisms to process sequences. Unlike RNNs, transformers process all positions in parallel, making them much faster to train. Key components include multi-head attention, positional encoding, and feed-forward layers. They are the foundation of models like GPT, BERT, and LLaMA."
        },
        {
            "prompt": "Write a Python function to calculate the factorial of a number.",
            "response": "Here's a Python function to calculate factorial:\n\n```python\ndef factorial(n):\n    \"\"\"Calculate factorial of n recursively.\"\"\"\n    if n < 0:\n        raise ValueError(\"Factorial is not defined for negative numbers\")\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\n# Iterative version (more efficient)\ndef factorial_iterative(n):\n    \"\"\"Calculate factorial of n iteratively.\"\"\"\n    if n < 0:\n        raise ValueError(\"Factorial is not defined for negative numbers\")\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n```"
        },
        {
            "prompt": "Summarize the concept of attention mechanism in 2-3 sentences.",
            "response": "The attention mechanism allows neural networks to focus on the most relevant parts of the input when producing each output element. It computes a weighted sum of all input representations, where the weights represent the relevance of each input to the current output. Self-attention, used in transformers, applies this mechanism within a single sequence to capture dependencies regardless of distance."
        },
        {
            "prompt": "What is the difference between GPT and BERT?",
            "response": "GPT (Generative Pretrained Transformer) is an autoregressive model that predicts the next token, making it ideal for text generation. It only looks at previous tokens (left-to-right). BERT (Bidirectional Encoder Representations from Transformers) is a masked language model that looks at both left and right context simultaneously, making it better for understanding tasks like classification and question answering. GPT is a decoder, BERT is an encoder."
        },
    ]

    return Dataset.from_list(examples)


def format_for_sft(example, tokenizer):
    """
    Format a prompt-response pair using the model's chat template.

    This is the KEY step: converting raw data into the format the model expects.
    """
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]

    # Use the tokenizer's built-in chat template
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    else:
        # Fallback for models without chat templates (e.g., base GPT-2)
        text = f"### Human: {example['prompt']}\n### Assistant: {example['response']}"

    return {"text": text}


def main():
    print("\n" + "█" * 60)
    print("█  SUPERVISED FINE-TUNING (SFT) — TRL SFTTrainer")
    print("█" * 60)

    # ═══════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ═══════════════════════════════════════════════════════════════
    MODEL_NAME = "distilgpt2"       # Small model for demo
    MAX_SEQ_LENGTH = 512
    OUTPUT_DIR = "./results/sft"
    LEARNING_RATE = 2e-5
    EPOCHS = 3
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION = 4

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: LOAD MODEL & TOKENIZER
    # ═══════════════════════════════════════════════════════════════
    print("\n📦 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Set pad token (required for SFT)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model: {MODEL_NAME}")
    print(f"   Parameters: {total_params:,}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: PREPARE DATASET
    # ═══════════════════════════════════════════════════════════════
    print("\n📊 Preparing dataset...")

    # Option A: Use our demo dataset
    dataset = create_demo_dataset()

    # Option B: Load from HuggingFace (uncomment to use)
    # dataset = load_dataset("databricks/databricks-dolly-15k", split="train[:1000]")
    # dataset = dataset.map(lambda x: {
    #     "prompt": x["instruction"] + (f"\n\nContext: {x['context']}" if x["context"] else ""),
    #     "response": x["response"]
    # })

    # Format for SFT
    dataset = dataset.map(
        lambda x: format_for_sft(x, tokenizer),
        desc="Formatting for SFT",
    )

    # Split into train/eval
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"   Training examples: {len(train_dataset)}")
    print(f"   Eval examples: {len(eval_dataset)}")
    print(f"\n   Sample formatted text:")
    print(f"   {train_dataset[0]['text'][:200]}...")

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: CONFIGURE RESPONSE-ONLY LOSS MASKING
    # ═══════════════════════════════════════════════════════════════
    """
    DataCollatorForCompletionOnlyLM is the KEY component for SFT.

    It automatically masks the prompt tokens in the loss computation,
    so the model only learns to generate the RESPONSE portion.

    How it works:
    1. It looks for the `response_template` in the tokenized text
    2. Everything BEFORE the template is masked (label = -100)
    3. Everything AFTER the template computes loss

    Without this: model wastes compute learning to predict user prompts
    With this: model focuses entirely on generating good responses
    """

    # The response template must match exactly what appears in your formatted text
    response_template = "### Assistant:"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )
    print(f"\n🎯 Response-only loss masking enabled")
    print(f"   Response template: '{response_template}'")

    # ═══════════════════════════════════════════════════════════════
    # STEP 4: CONFIGURE TRAINING
    # ═══════════════════════════════════════════════════════════════
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        max_seq_length=MAX_SEQ_LENGTH,
        fp16=torch.cuda.is_available(),
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        seed=42,
        # SFT-specific settings
        packing=False,          # Set True for efficiency with short examples
        dataset_text_field="text",  # Column containing formatted text
    )

    # ═══════════════════════════════════════════════════════════════
    # STEP 5: PRE-TRAINING GENERATION TEST
    # ═══════════════════════════════════════════════════════════════
    print("\n── Before SFT ──")
    model.to(device)
    test_prompt = "### Human: What is deep learning?\n### Assistant:"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=100, temperature=0.7,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"   Prompt: What is deep learning?")
    print(f"   Response: {response[len(test_prompt):][:200]}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 6: TRAIN WITH SFTTrainer
    # ═══════════════════════════════════════════════════════════════
    print("\n🔥 Starting SFT Training...")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    # Show what the loss masking looks like on one example
    print("\n📋 Loss masking demo on first example:")
    sample_encoding = tokenizer(train_dataset[0]["text"], return_tensors="pt")
    sample_batch = collator([{
        "input_ids": sample_encoding["input_ids"][0].tolist(),
        "attention_mask": sample_encoding["attention_mask"][0].tolist(),
    }])
    masked_count = (sample_batch["labels"][0] == -100).sum().item()
    total_count = sample_batch["labels"][0].shape[0]
    print(f"   Total tokens: {total_count}")
    print(f"   Masked (prompt): {masked_count} ({100*masked_count/total_count:.1f}%)")
    print(f"   Trained (response): {total_count - masked_count} ({100*(total_count-masked_count)/total_count:.1f}%)")

    # TRAIN!
    train_result = trainer.train()

    # ═══════════════════════════════════════════════════════════════
    # STEP 7: EVALUATE
    # ═══════════════════════════════════════════════════════════════
    print("\n📊 Evaluation:")
    eval_results = trainer.evaluate()
    print(f"   Eval Loss: {eval_results['eval_loss']:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 8: POST-TRAINING GENERATION TEST
    # ═══════════════════════════════════════════════════════════════
    print("\n── After SFT ──")
    test_prompts = [
        "### Human: What is deep learning?\n### Assistant:",
        "### Human: Explain what a neural network is.\n### Assistant:",
        "### Human: What is the purpose of fine-tuning an LLM?\n### Assistant:",
    ]

    model.eval()
    for test_prompt in test_prompts:
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=150, temperature=0.7,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        prompt_text = test_prompt.split("### Human: ")[1].split("\n")[0]
        print(f"\n   Q: {prompt_text}")
        print(f"   A: {response[len(test_prompt):][:200]}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 9: SAVE
    # ═══════════════════════════════════════════════════════════════
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"\n✅ SFT model saved to {OUTPUT_DIR}/final")

    print("\n" + "█" * 60)
    print("█  SFT TRAINING COMPLETE!")
    print("█" * 60)


if __name__ == "__main__":
    main()
