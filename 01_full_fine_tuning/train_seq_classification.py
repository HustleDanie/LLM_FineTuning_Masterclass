"""
═══════════════════════════════════════════════════════════════════════════
FULL FINE-TUNING — Sequence Classification
═══════════════════════════════════════════════════════════════════════════

This script demonstrates full fine-tuning for CLASSIFICATION tasks.
The model learns to predict a class label for a given text input.

USE CASES:
  - Sentiment analysis (positive/negative/neutral)
  - Topic classification (news categories)
  - Spam detection
  - Intent classification

KEY DIFFERENCE FROM CAUSAL LM:
  - Uses AutoModelForSequenceClassification (adds classification head)
  - Labels are class indices (0, 1, 2, ...) instead of token IDs
  - Uses cross-entropy loss over class logits
  - Dynamic padding instead of fixed-length chunking

This demonstrates that full fine-tuning works for ANY downstream task,
not just text generation.
"""

import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report


def compute_metrics(eval_pred):
    """
    Compute classification metrics.

    This function is called by the Trainer during evaluation.
    It receives raw model predictions and ground truth labels.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }


def main():
    # ═══════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ═══════════════════════════════════════════════════════════════
    MODEL_NAME = "distilbert-base-uncased"  # Good for classification
    DATASET = "imdb"                         # Sentiment analysis
    MAX_LENGTH = 256
    EPOCHS = 3
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    OUTPUT_DIR = "./results/seq_classification"
    NUM_LABELS = 2  # positive, negative

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ═══════════════════════════════════════════════════════════════
    # LOAD MODEL & TOKENIZER
    # ═══════════════════════════════════════════════════════════════
    print(f"\nLoading {MODEL_NAME} for classification...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # AutoModelForSequenceClassification adds a classification head
    # on top of the pretrained transformer
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        # Map label IDs to human-readable names
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    # Verify ALL parameters are trainable (FULL fine-tuning)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"Classification head parameters: ~{NUM_LABELS * 768:,}")

    # ═══════════════════════════════════════════════════════════════
    # LOAD & TOKENIZE DATASET
    # ═══════════════════════════════════════════════════════════════
    print(f"\nLoading dataset: {DATASET}...")
    dataset = load_dataset(DATASET)

    # Limit for faster demo (remove for full training)
    dataset["train"] = dataset["train"].select(range(2000))
    dataset["test"] = dataset["test"].select(range(500))

    def tokenize_function(examples):
        """
        Tokenize for classification:
        - Truncate to max_length
        - Don't pad here (dynamic padding in collator is more efficient)
        """
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    print(f"Training examples: {len(tokenized_dataset['train'])}")
    print(f"Test examples: {len(tokenized_dataset['test'])}")

    # Show label distribution
    from collections import Counter
    label_counts = Counter(tokenized_dataset["train"]["label"])
    print(f"Label distribution: {dict(label_counts)}")

    # ═══════════════════════════════════════════════════════════════
    # TRAINING
    # ═══════════════════════════════════════════════════════════════
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=1,  # Less needed for classification
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="linear",     # Linear decay is standard for BERT-style
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=42,
    )

    # Dynamic padding: pad each batch to the longest sequence in that batch
    # More efficient than padding everything to MAX_LENGTH
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n🔥 Starting full fine-tuning (Classification)...")
    trainer.train()

    # ═══════════════════════════════════════════════════════════════
    # EVALUATE
    # ═══════════════════════════════════════════════════════════════
    print("\n📊 Final Evaluation:")
    eval_results = trainer.evaluate()
    print(f"   Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"   F1 Score: {eval_results['eval_f1']:.4f}")
    print(f"   Loss: {eval_results['eval_loss']:.4f}")

    # Detailed classification report
    predictions = trainer.predict(tokenized_dataset["test"])
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    print("\n📋 Classification Report:")
    print(classification_report(
        labels, preds,
        target_names=["negative", "positive"]
    ))

    # ═══════════════════════════════════════════════════════════════
    # INFERENCE DEMO
    # ═══════════════════════════════════════════════════════════════
    print("\n── Inference Demo ──")
    model.to(device)
    model.eval()

    test_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible film. Waste of time and money. Would not recommend.",
        "The plot was okay but the acting was phenomenal.",
    ]

    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_label = torch.argmax(probs, dim=-1).item()

        sentiment = "positive" if pred_label == 1 else "negative"
        confidence = probs[0][pred_label].item()

        print(f"\n   Text: {text[:80]}...")
        print(f"   Prediction: {sentiment} (confidence: {confidence:.2%})")

    # ═══════════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════════
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"\n✅ Model saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
