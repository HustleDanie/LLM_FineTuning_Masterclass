"""
═══════════════════════════════════════════════════════════════════════════
SUPERVISED FINE-TUNING — From Scratch Implementation
═══════════════════════════════════════════════════════════════════════════

This script implements SFT WITHOUT using TRL, building every component
manually. This is purely EDUCATIONAL — to deeply understand what
SFTTrainer does under the hood.

You will learn:
  1. How to manually create labels with loss masking
  2. How the training loop works step by step
  3. How gradients flow only through response tokens
  4. How to handle conversation formatting manually
  5. The complete forward/backward pass for SFT

WHEN TO USE THIS: Never in production. Use TRL's SFTTrainer instead.
This exists purely so you understand the internals.
"""

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from datasets import Dataset
from typing import List, Dict, Optional
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════
# 1. CUSTOM SFT DATASET
# ═══════════════════════════════════════════════════════════════════════

class SFTDataset(TorchDataset):
    """
    Custom PyTorch dataset for SFT with manual loss masking.

    This implements the core SFT data pipeline:
    1. Format prompt + response into a single text
    2. Tokenize the full text
    3. Create labels where prompt tokens are masked (-100)
    4. Pad/truncate to uniform length

    Understanding this class is understanding SFT.
    """

    def __init__(
        self,
        prompts: List[str],
        responses: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        prompt_template: str = "### Human: {prompt}\n### Assistant:",
        response_template: str = " {response}",
    ):
        self.prompts = prompts
        self.responses = responses
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.response_template = response_template

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]

        # ── Step 1: Format the full text ─────────────────────────
        prompt_text = self.prompt_template.format(prompt=prompt)
        response_text = self.response_template.format(response=response)
        full_text = prompt_text + response_text

        # ── Step 2: Tokenize ─────────────────────────────────────
        # Tokenize prompt and response SEPARATELY to know the boundary
        prompt_tokens = self.tokenizer(
            prompt_text,
            add_special_tokens=True,
            truncation=False,
        )
        full_tokens = self.tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]

        # ── Step 3: Create masked labels ─────────────────────────
        # THIS IS THE KEY SFT OPERATION
        prompt_length = len(prompt_tokens["input_ids"])
        labels = list(input_ids)  # Copy

        # Mask prompt tokens (set to -100 = ignore in loss)
        for i in range(min(prompt_length, len(labels))):
            labels[i] = -100

        # Also mask padding tokens
        for i in range(len(labels)):
            if attention_mask[i] == 0:
                labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# Need this for type hints
from transformers import PreTrainedTokenizer


# ═══════════════════════════════════════════════════════════════════════
# 2. CUSTOM SFT TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════

def train_sft_from_scratch(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    train_dataset: SFTDataset,
    eval_dataset: Optional[SFTDataset] = None,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-5,
    warmup_steps: int = 10,
    max_grad_norm: float = 1.0,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 4,
    eval_every_n_steps: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Complete SFT training loop implemented from scratch.

    This function shows EXACTLY what happens during SFT:

    For each batch:
    ─────────────────
    1. FORWARD PASS:
       - Input: [prompt_tokens] [response_tokens] [pad_tokens]
       - Model predicts next token at each position
       - Output: logits of shape (batch, seq_len, vocab_size)

    2. LOSS COMPUTATION:
       - Labels: [-100, -100, ..., response_tokens, -100, ...]
       - CrossEntropyLoss(ignore_index=-100)
       - Loss ONLY computed on response token positions
       - Prompt and padding positions contribute ZERO to the loss

    3. BACKWARD PASS:
       - Gradients flow from response token positions
       - But they flow through ALL model parameters
       - This is how the model learns to generate responses

    4. OPTIMIZER STEP:
       - Update ALL parameters (full SFT, not PEFT)
       - Apply weight decay, gradient clipping
       - Update learning rate schedule
    """

    model.to(device)
    model.train()

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    eval_loader = None
    if eval_dataset:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    # ── Optimizer Setup ──────────────────────────────────────────
    # Standard AdamW with weight decay
    # We exclude bias and LayerNorm from weight decay
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training Loop ────────────────────────────────────────────
    print("\n" + "🔥" * 30)
    print("   STARTING SFT FROM-SCRATCH TRAINING")
    print("🔥" * 30)
    print(f"\n   Device: {device}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation: {gradient_accumulation_steps}")
    print(f"   Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"   Total steps: {total_steps}")
    print(f"   Learning rate: {learning_rate}")

    global_step = 0
    total_loss = 0.0
    best_eval_loss = float("inf")
    training_log = []

    for epoch in range(epochs):
        print(f"\n{'═' * 50}")
        print(f"EPOCH {epoch + 1}/{epochs}")
        print(f"{'═' * 50}")

        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(progress_bar):
            # ── Step 1: Move batch to device ─────────────────
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # ── Step 2: Forward pass ─────────────────────────
            # The model computes:
            #   logits[i] = prediction for position i+1
            #   loss = CrossEntropy(logits[:-1], labels[1:])
            #   Only positions where labels != -100 contribute
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

            # ── Step 3: Backward pass ────────────────────────
            # Gradients flow from response token loss through entire model
            loss.backward()

            epoch_loss += loss.item() * gradient_accumulation_steps

            # Count response tokens (non-masked labels)
            response_tokens = (labels != -100).sum().item()
            epoch_tokens += response_tokens

            # ── Step 4: Optimizer step (every N accumulation steps)
            if (step + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping — prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "resp_tokens": response_tokens,
                })

                training_log.append({
                    "step": global_step,
                    "loss": loss.item() * gradient_accumulation_steps,
                    "lr": current_lr,
                })

            # ── Step 5: Evaluation ───────────────────────────
            if eval_loader and global_step > 0 and global_step % eval_every_n_steps == 0:
                eval_loss = evaluate_sft(model, eval_loader, device)
                print(f"\n   📊 Step {global_step} | Eval Loss: {eval_loss:.4f} | "
                      f"Perplexity: {math.exp(eval_loss):.2f}")

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    print(f"   🏆 New best eval loss!")

                model.train()  # Back to training mode

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\n   Epoch {epoch+1} Summary:")
        print(f"   Avg Loss: {avg_epoch_loss:.4f}")
        print(f"   Total response tokens trained: {epoch_tokens:,}")

    # ── Final Evaluation ─────────────────────────────────────────
    if eval_loader:
        final_eval_loss = evaluate_sft(model, eval_loader, device)
        print(f"\n📊 Final Eval Loss: {final_eval_loss:.4f}")
        print(f"📊 Final Perplexity: {math.exp(final_eval_loss):.2f}")
        print(f"📊 Best Eval Loss: {best_eval_loss:.4f}")

    return model, training_log


# ═══════════════════════════════════════════════════════════════════════
# 3. EVALUATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def evaluate_sft(model, eval_loader, device):
    """
    Evaluate the SFT model.
    Computes average loss over response tokens only.
    """
    model.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            total_steps += 1

    return total_loss / max(total_steps, 1)


# ═══════════════════════════════════════════════════════════════════════
# 4. MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 60)
    print("█  SFT FROM SCRATCH — Understanding the Internals")
    print("█" * 60)

    # Config
    MODEL_NAME = "distilgpt2"
    MAX_LENGTH = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(42)

    # Load model and tokenizer
    print("\n📦 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Create datasets
    prompts = [
        "What is machine learning?",
        "Explain neural networks.",
        "What is deep learning?",
        "How does backpropagation work?",
        "What is a transformer model?",
        "Explain gradient descent.",
        "What is overfitting?",
        "How does attention work in transformers?",
    ]

    responses = [
        "Machine learning is a branch of AI that enables computers to learn from data without being explicitly programmed.",
        "Neural networks are computing systems inspired by biological neurons, consisting of layers of interconnected nodes that process information.",
        "Deep learning is a subset of machine learning that uses neural networks with many layers to learn complex patterns from large amounts of data.",
        "Backpropagation is an algorithm for training neural networks that computes gradients of the loss function with respect to each weight by propagating errors backwards through the network.",
        "A transformer is a neural network architecture that uses self-attention mechanisms to process input sequences in parallel, enabling it to capture long-range dependencies efficiently.",
        "Gradient descent is an optimization algorithm that iteratively adjusts model parameters by moving in the direction that minimizes the loss function.",
        "Overfitting occurs when a model learns the training data too well, including noise, and fails to generalize to new unseen data.",
        "Attention in transformers computes weighted relationships between all positions in a sequence, allowing the model to focus on the most relevant parts of the input.",
    ]

    # Split into train/eval
    train_prompts = prompts[:6]
    train_responses = responses[:6]
    eval_prompts = prompts[6:]
    eval_responses = responses[6:]

    train_dataset = SFTDataset(
        prompts=train_prompts,
        responses=train_responses,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    eval_dataset = SFTDataset(
        prompts=eval_prompts,
        responses=eval_responses,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    # Inspect one example
    print("\n🔍 Inspecting one training example:")
    sample = train_dataset[0]
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    mask_count = (labels == -100).sum().item()
    total_count = labels.shape[0]
    non_pad = (sample["attention_mask"] == 1).sum().item()

    print(f"   Total tokens: {total_count}")
    print(f"   Non-padding tokens: {non_pad}")
    print(f"   Masked (prompt): {mask_count}")
    print(f"   Trained (response): {non_pad - mask_count}")
    print(f"   Decoded prompt+response:")
    decoded = tokenizer.decode(input_ids[input_ids != tokenizer.pad_token_id])
    print(f"   {decoded[:200]}...")

    # Train!
    model, log = train_sft_from_scratch(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        epochs=3,
        batch_size=2,
        learning_rate=3e-5,
        warmup_steps=5,
        gradient_accumulation_steps=2,
        eval_every_n_steps=5,
        device=device,
    )

    # Test generation
    print("\n── Post-SFT Generation ──")
    model.eval()
    test_prompts = [
        "### Human: What is machine learning?\n### Assistant:",
        "### Human: Explain what a GPU is used for.\n### Assistant:",
    ]

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=100, temperature=0.7,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\n   Q: {prompt.split('### Human: ')[1].split(chr(10))[0]}")
        print(f"   A: {response[len(prompt):][:200]}")

    # Save
    output_dir = "./results/sft_from_scratch/final"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ Model saved to {output_dir}")

    print("\n" + "█" * 60)
    print("█  SFT FROM-SCRATCH TRAINING COMPLETE!")
    print("█" * 60)


if __name__ == "__main__":
    main()
