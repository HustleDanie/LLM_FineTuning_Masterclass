"""
Data Utilities for Full Fine-Tuning.

Covers:
  - Loading datasets from HuggingFace Hub
  - Tokenization strategies (causal LM vs classification)
  - Text chunking for efficient training
  - Data collation with dynamic padding
  - Train/validation/test splits
"""

from typing import Optional, Dict, Any
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer
import torch


# ═══════════════════════════════════════════════════════════════════════
# 1. DATASET LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_text_dataset(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    train_split: str = "train",
    validation_split: str = "validation",
    test_split: str = "test",
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> DatasetDict:
    """
    Load a text dataset from HuggingFace Hub.

    For full fine-tuning of causal language models, we typically use
    plain text datasets where the model learns to predict the next token.

    Args:
        dataset_name: HuggingFace dataset identifier
        dataset_config: Dataset configuration/subset
        max_train_samples: Limit training data (for debugging)
        max_eval_samples: Limit eval data

    Returns:
        DatasetDict with train, validation, and test splits
    """
    print(f"Loading dataset: {dataset_name} ({dataset_config})")

    raw_datasets = load_dataset(dataset_name, dataset_config)

    # Optionally limit samples for faster experimentation
    if max_train_samples is not None and train_split in raw_datasets:
        raw_datasets[train_split] = raw_datasets[train_split].select(
            range(min(max_train_samples, len(raw_datasets[train_split])))
        )
    if max_eval_samples is not None and validation_split in raw_datasets:
        raw_datasets[validation_split] = raw_datasets[validation_split].select(
            range(min(max_eval_samples, len(raw_datasets[validation_split])))
        )

    print(f"  Train: {len(raw_datasets.get(train_split, []))} samples")
    print(f"  Validation: {len(raw_datasets.get(validation_split, []))} samples")
    print(f"  Test: {len(raw_datasets.get(test_split, []))} samples")

    return raw_datasets


def load_classification_dataset(
    dataset_name: str = "imdb",
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> DatasetDict:
    """
    Load a classification dataset (e.g., sentiment analysis).

    For sequence classification fine-tuning, we need labeled data
    with text and corresponding class labels.
    """
    print(f"Loading classification dataset: {dataset_name}")
    raw_datasets = load_dataset(dataset_name)

    if max_train_samples:
        raw_datasets["train"] = raw_datasets["train"].select(
            range(min(max_train_samples, len(raw_datasets["train"])))
        )
    if max_eval_samples and "test" in raw_datasets:
        raw_datasets["test"] = raw_datasets["test"].select(
            range(min(max_eval_samples, len(raw_datasets["test"])))
        )

    return raw_datasets


# ═══════════════════════════════════════════════════════════════════════
# 2. TOKENIZATION FOR CAUSAL LANGUAGE MODELING
# ═══════════════════════════════════════════════════════════════════════

def tokenize_for_causal_lm(
    raw_datasets: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 512,
    text_column: str = "text",
    num_proc: int = 4,
) -> DatasetDict:
    """
    Tokenize text data for causal language modeling.

    For causal LM, the labels are the same as the input_ids
    (shifted by one position internally by the model).

    IMPORTANT CONCEPT: Text Chunking
    ─────────────────────────────────
    Instead of truncating each text to max_seq_length (wasteful),
    we concatenate all texts and then split into fixed-length chunks.
    This ensures maximum utilization of each training example.

    Before chunking:
        ["Short text.", "Another short.", "A very long text that goes on..."]

    After concatenation + chunking (chunk_size=10 tokens):
        [tokens[0:10], tokens[10:20], tokens[20:30], ...]

    This is the standard approach used in GPT-2, LLaMA, etc.
    """

    def tokenize_function(examples):
        """Tokenize all texts in batch."""
        return tokenizer(
            examples[text_column],
            truncation=False,  # Don't truncate — we'll chunk later
            padding=False,     # Don't pad — we'll chunk to equal lengths
        )

    print("Tokenizing dataset...")
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=raw_datasets[list(raw_datasets.keys())[0]].column_names,
        desc="Tokenizing",
    )

    # ── Chunking: Concatenate all tokens and split into blocks ────
    block_size = max_seq_length

    def group_texts(examples):
        """
        Concatenate all texts and create chunks of block_size.

        This is the KEY preprocessing step for causal LM fine-tuning:
        1. Flatten all token IDs into one long sequence
        2. Split into equal-length chunks
        3. Each chunk becomes one training example

        The labels for causal LM are identical to input_ids.
        The model will internally shift them to predict the next token.
        """
        # Concatenate all texts
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])

        # Drop the small remainder (we could pad instead but this is simpler)
        total_length = (total_length // block_size) * block_size

        # Split into chunks of block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }

        # For causal LM, labels = input_ids
        result["labels"] = result["input_ids"].copy()
        return result

    print(f"Grouping texts into chunks of {block_size} tokens...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc="Chunking",
    )

    return lm_datasets


# ═══════════════════════════════════════════════════════════════════════
# 3. TOKENIZATION FOR SEQUENCE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════

def tokenize_for_classification(
    raw_datasets: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 512,
    text_column: str = "text",
    label_column: str = "label",
    num_proc: int = 4,
) -> DatasetDict:
    """
    Tokenize text for sequence classification.

    Unlike causal LM, here we:
    - Truncate/pad each example independently
    - Preserve the label column
    - Use dynamic padding during training (via data collator)
    """

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_seq_length,
            padding=False,  # We'll use dynamic padding in the collator
        )

    print("Tokenizing for classification...")
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=[c for c in raw_datasets["train"].column_names
                        if c not in [label_column]],
        desc="Tokenizing",
    )

    return tokenized_datasets


# ═══════════════════════════════════════════════════════════════════════
# 4. DATA COLLATORS
# ═══════════════════════════════════════════════════════════════════════

class CausalLMDataCollator:
    """
    Data collator for causal language modeling.

    Since we already chunked texts to equal length during preprocessing,
    we mainly need to convert lists to tensors. This collator also
    demonstrates how to handle padding if chunks have variable lengths.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, mlm: bool = False):
        self.tokenizer = tokenizer
        self.mlm = mlm  # Masked LM flag (False for causal/autoregressive)

    def __call__(self, examples):
        # Stack all examples into a batch
        batch = {
            key: torch.tensor([example[key] for example in examples])
            for key in examples[0].keys()
        }
        return batch


def get_data_collator(task: str, tokenizer: PreTrainedTokenizer):
    """
    Factory function for data collators.

    Args:
        task: 'causal_lm' or 'classification'
        tokenizer: The tokenizer (needed for padding)

    Returns:
        Appropriate data collator
    """
    from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding

    if task == "causal_lm":
        # Standard collator for language modeling
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # False = causal LM (next token prediction)
        )
    elif task == "classification":
        # Dynamic padding: pads each batch to the longest sequence
        # More efficient than padding everything to max_length
        return DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
        )
    else:
        raise ValueError(f"Unknown task: {task}")


# ═══════════════════════════════════════════════════════════════════════
# 5. DATASET INSPECTION UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def inspect_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizer, n_samples: int = 3):
    """
    Print sample data for inspection.
    Always inspect your data before training!
    """
    print("\n" + "=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    print(f"Number of examples: {len(dataset)}")
    print(f"Features: {dataset.features}")

    for i in range(min(n_samples, len(dataset))):
        print(f"\n--- Sample {i + 1} ---")
        example = dataset[i]

        if "input_ids" in example:
            tokens = example["input_ids"][:50]  # Show first 50 tokens
            decoded = tokenizer.decode(tokens)
            print(f"  Tokens (first 50): {tokens}")
            print(f"  Decoded: {decoded[:200]}...")
            print(f"  Sequence length: {len(example['input_ids'])}")

        if "label" in example:
            print(f"  Label: {example['label']}")

        if "labels" in example:
            print(f"  Labels length: {len(example['labels'])}")

    print("=" * 60)


def compute_dataset_statistics(dataset: Dataset) -> Dict[str, Any]:
    """Compute useful statistics about the tokenized dataset."""
    lengths = [len(example["input_ids"]) for example in dataset]
    return {
        "num_examples": len(dataset),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "avg_length": sum(lengths) / len(lengths),
        "total_tokens": sum(lengths),
    }
