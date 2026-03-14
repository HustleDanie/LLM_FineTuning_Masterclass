"""
Evaluation Utilities for Full Fine-Tuning.

Covers:
  - Perplexity calculation (primary metric for language models)
  - Text generation evaluation
  - Classification metrics (accuracy, F1, etc.)
  - Before/after comparison
  - Loss curves visualization
"""

import math
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset


# ═══════════════════════════════════════════════════════════════════════
# 1. PERPLEXITY — THE KEY METRIC FOR LANGUAGE MODELS
# ═══════════════════════════════════════════════════════════════════════

def compute_perplexity(eval_loss: float) -> float:
    """
    Compute perplexity from evaluation loss.

    PERPLEXITY EXPLAINED:
    ─────────────────────
    Perplexity = e^(cross_entropy_loss)

    Intuition: If a model has perplexity of 20, it's as uncertain as
    choosing uniformly among 20 options at each step.

    Lower perplexity = better model:
    - ~1: Perfect prediction (memorized data)
    - ~20-30: Good language model
    - ~50-100: Okay model
    - ~1000+: Poor model

    For fine-tuning, we typically see perplexity DROP compared to
    the pretrained model (on the target domain).
    """
    try:
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity


def evaluate_perplexity(
    model: PreTrainedModel,
    eval_dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute perplexity on a dataset.

    This performs a full evaluation pass:
    1. Feed each batch through the model
    2. Collect cross-entropy losses
    3. Average losses and compute perplexity
    """
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForLanguageModeling

    model.eval()
    device = next(model.parameters()).device

    if max_samples:
        eval_dataset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collator)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            # Count tokens (excluding padding)
            num_tokens = (batch["labels"] != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = compute_perplexity(avg_loss)

    results = {
        "eval_loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens_evaluated": total_tokens,
    }

    print(f"\n📊 Perplexity Evaluation:")
    print(f"   Loss: {avg_loss:.4f}")
    print(f"   Perplexity: {perplexity:.2f}")
    print(f"   Tokens evaluated: {total_tokens:,}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# 2. TEXT GENERATION EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    repetition_penalty: float = 1.2,
) -> List[str]:
    """
    Generate text from prompts using the fine-tuned model.

    GENERATION PARAMETERS EXPLAINED:
    ─────────────────────────────────
    - temperature: Controls randomness (0=deterministic, 1=creative, >1=chaotic)
    - top_p (nucleus sampling): Only sample from tokens whose cumulative prob >= top_p
    - top_k: Only sample from top K most probable tokens
    - repetition_penalty: Penalize tokens that already appeared (>1 = less repetition)
    - do_sample: If False, use greedy decoding (always pick most probable token)

    After fine-tuning, the model should generate text that reflects
    the style and content of your training data.
    """
    model.eval()
    device = next(model.parameters()).device
    generated_texts = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(generated)

    return generated_texts


def compare_before_after(
    original_model: PreTrainedModel,
    finetuned_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_new_tokens: int = 100,
):
    """
    Compare text generation before and after fine-tuning.

    This is the most intuitive way to see the impact of fine-tuning:
    same prompt, different model → different outputs.
    """
    print("\n" + "=" * 70)
    print("BEFORE vs AFTER FINE-TUNING COMPARISON")
    print("=" * 70)

    for prompt in prompts:
        print(f"\n📝 Prompt: {prompt}")
        print("-" * 70)

        # Generate from original model
        original_output = generate_text(
            original_model, tokenizer, [prompt],
            max_new_tokens=max_new_tokens
        )[0]
        print(f"\n🔵 Original model:")
        print(f"   {original_output}")

        # Generate from fine-tuned model
        finetuned_output = generate_text(
            finetuned_model, tokenizer, [prompt],
            max_new_tokens=max_new_tokens
        )[0]
        print(f"\n🟢 Fine-tuned model:")
        print(f"   {finetuned_output}")

        print("-" * 70)


# ═══════════════════════════════════════════════════════════════════════
# 3. CLASSIFICATION METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_classification_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute metrics for sequence classification tasks.

    Used as the `compute_metrics` function in HuggingFace Trainer.

    Returns accuracy, F1, precision, and recall.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # For classification, take argmax of logits
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=-1)

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
    }

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# 4. WEIGHT CHANGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def analyze_weight_changes(
    original_state_dict: Dict,
    finetuned_state_dict: Dict,
) -> Dict[str, float]:
    """
    Analyze how much weights changed during full fine-tuning.

    INSIGHT: In full FT, ALL weights change. This function quantifies
    the magnitude of changes per layer, which helps understand:
    - Which layers adapted most (usually later layers)
    - Whether training was too aggressive (very large changes)
    - Whether learning rate was appropriate

    Typical findings in full fine-tuning:
    - Embedding layers: moderate change
    - Earlier transformer layers: smaller changes
    - Later transformer layers: larger changes
    - LM head: largest changes (closest to output)
    """
    changes = {}

    for key in original_state_dict:
        if key in finetuned_state_dict:
            original = original_state_dict[key].float()
            finetuned = finetuned_state_dict[key].float()

            # L2 norm of the change
            diff_norm = torch.norm(finetuned - original).item()
            original_norm = torch.norm(original).item()

            # Relative change (percentage)
            relative_change = (diff_norm / (original_norm + 1e-8)) * 100

            changes[key] = {
                "absolute_change": diff_norm,
                "original_norm": original_norm,
                "relative_change_pct": relative_change,
            }

    # Print summary
    print("\n" + "=" * 60)
    print("WEIGHT CHANGE ANALYSIS (Full Fine-Tuning)")
    print("=" * 60)

    # Group by layer type and show top changes
    sorted_changes = sorted(
        changes.items(),
        key=lambda x: x[1]["relative_change_pct"],
        reverse=True
    )

    print("\nTop 10 most-changed parameter groups:")
    for name, stats in sorted_changes[:10]:
        print(f"  {name[:50]:50s}: {stats['relative_change_pct']:8.2f}% change")

    print("\nBottom 5 least-changed parameter groups:")
    for name, stats in sorted_changes[-5:]:
        print(f"  {name[:50]:50s}: {stats['relative_change_pct']:8.2f}% change")

    avg_change = np.mean([v["relative_change_pct"] for v in changes.values()])
    print(f"\nAverage relative change: {avg_change:.2f}%")
    print("=" * 60)

    return changes


# ═══════════════════════════════════════════════════════════════════════
# 5. LOSS CURVE UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def print_training_summary(
    train_losses: List[float],
    eval_losses: List[float],
    learning_rates: List[float],
):
    """
    Print a text-based summary of training progress.

    In production, you'd use TensorBoard or Weights & Biases
    for visualization. This provides a quick text overview.
    """
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    if train_losses:
        print(f"\nTraining Loss:")
        print(f"  Start:  {train_losses[0]:.4f}")
        print(f"  End:    {train_losses[-1]:.4f}")
        print(f"  Min:    {min(train_losses):.4f}")
        print(f"  Change: {train_losses[-1] - train_losses[0]:+.4f}")

    if eval_losses:
        print(f"\nEvaluation Loss:")
        print(f"  Start:  {eval_losses[0]:.4f}")
        print(f"  End:    {eval_losses[-1]:.4f}")
        print(f"  Min:    {min(eval_losses):.4f}")
        print(f"  Best Perplexity: {compute_perplexity(min(eval_losses)):.2f}")

    if learning_rates:
        print(f"\nLearning Rate:")
        print(f"  Start:  {learning_rates[0]:.2e}")
        print(f"  Peak:   {max(learning_rates):.2e}")
        print(f"  End:    {learning_rates[-1]:.2e}")

    print("=" * 60)
