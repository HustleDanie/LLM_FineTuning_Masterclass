"""
═══════════════════════════════════════════════════════════════════════════
LOSS MASKING — The Core Technique of SFT
═══════════════════════════════════════════════════════════════════════════

Loss masking is THE fundamental concept that distinguishes SFT from
regular language model training.

In standard LM training:
    Loss is computed on ALL tokens → model learns to predict everything

In SFT:
    Loss is computed ONLY on response tokens → model learns to generate
    good responses, not to reproduce the user's prompt

This module provides:
  1. Manual implementation of loss masking (educational)
  2. Visualization of what gets masked
  3. Different masking strategies
  4. How TRL implements this internally
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from transformers import PreTrainedTokenizer, PreTrainedModel


# ═══════════════════════════════════════════════════════════════════════
# 1. UNDERSTANDING LOSS MASKING
# ═══════════════════════════════════════════════════════════════════════

def explain_loss_masking():
    """
    Visual explanation of loss masking in SFT.

    This is the MOST IMPORTANT concept in SFT.
    """
    explanation = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║          LOSS MASKING IN SUPERVISED FINE-TUNING              ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║                                                               ║
    ║  Example conversation:                                        ║
    ║  User: What is AI?                                           ║
    ║  Assistant: AI is artificial intelligence.                    ║
    ║                                                               ║
    ║  ─────────────────────────────────────────────────────────── ║
    ║                                                               ║
    ║  Standard LM Training (NO masking):                          ║
    ║                                                               ║
    ║  Tokens:  [User] [What] [is] [AI] [?] [Asst] [AI] [is] ... ║
    ║  Labels:  [User] [What] [is] [AI] [?] [Asst] [AI] [is] ... ║
    ║  Loss:     ✓      ✓     ✓    ✓    ✓    ✓      ✓    ✓       ║
    ║                                                               ║
    ║  Problem: Model learns to generate "What is AI?" — we don't ║
    ║  want the model to mimic user prompts!                       ║
    ║                                                               ║
    ║  ─────────────────────────────────────────────────────────── ║
    ║                                                               ║
    ║  SFT Training (WITH masking):                                ║
    ║                                                               ║
    ║  Tokens:  [User] [What] [is] [AI] [?] [Asst] [AI] [is] ... ║
    ║  Labels:  [-100] [-100] [-100][-100][-100][-100] [AI] [is]..║
    ║  Loss:     ✗      ✗     ✗    ✗    ✗    ✗      ✓    ✓       ║
    ║                                                               ║
    ║  -100 is PyTorch's ignore_index for CrossEntropyLoss.        ║
    ║  The model only learns to generate the RESPONSE portion.     ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(explanation)


# ═══════════════════════════════════════════════════════════════════════
# 2. MANUAL LOSS MASKING IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════

def create_labels_with_masking(
    input_ids: List[int],
    tokenizer: PreTrainedTokenizer,
    response_start_token: str = "### Assistant:",
    mask_value: int = -100,
) -> List[int]:
    """
    Create labels with prompt tokens masked.

    HOW IT WORKS:
    ─────────────
    1. Tokenize the response start marker (e.g., "### Assistant:")
    2. Find where it appears in the input_ids
    3. Set all labels BEFORE that point to -100 (ignore)
    4. Keep labels AFTER that point as the actual token IDs

    Args:
        input_ids: Tokenized input sequence
        tokenizer: The tokenizer (for encoding the marker)
        response_start_token: String that marks where the response begins
        mask_value: Value to use for ignored tokens (-100 for PyTorch CE loss)

    Returns:
        Labels with prompt tokens masked to -100
    """
    # Tokenize the response marker
    marker_ids = tokenizer.encode(response_start_token, add_special_tokens=False)

    # Find the marker in input_ids
    response_start_idx = find_subsequence(input_ids, marker_ids)

    if response_start_idx == -1:
        # Marker not found — mask everything (this example produces no loss)
        print(f"   ⚠️ Response marker '{response_start_token}' not found!")
        return [mask_value] * len(input_ids)

    # Mask everything up to and including the response marker
    mask_until = response_start_idx + len(marker_ids)

    labels = [mask_value] * mask_until + input_ids[mask_until:]

    return labels


def find_subsequence(sequence: List[int], subsequence: List[int]) -> int:
    """Find the starting index of a subsequence within a sequence."""
    for i in range(len(sequence) - len(subsequence) + 1):
        if sequence[i:i + len(subsequence)] == subsequence:
            return i
    return -1


# ═══════════════════════════════════════════════════════════════════════
# 3. MULTI-TURN CONVERSATION MASKING
# ═══════════════════════════════════════════════════════════════════════

def create_multiturn_labels(
    input_ids: List[int],
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    mask_value: int = -100,
) -> List[int]:
    """
    Create labels for multi-turn conversations.

    In multi-turn SFT, we mask ALL user turns and only compute loss
    on ALL assistant turns:

    [SYSTEM] [USER_1] [ASSISTANT_1] [USER_2] [ASSISTANT_2]
    [-100]   [-100]   [loss here]   [-100]   [loss here]

    This teaches the model to generate good responses at EVERY turn,
    not just the last one.
    """
    labels = [mask_value] * len(input_ids)

    # Tokenize each message and find its position
    current_pos = 0
    full_text = ""

    for msg in messages:
        # Approximate position tracking
        role_text = f"{msg['role']}: {msg['content']}"
        role_tokens = tokenizer.encode(role_text, add_special_tokens=False)

        if msg["role"] == "assistant":
            # Don't mask assistant tokens
            content_tokens = tokenizer.encode(msg["content"], add_special_tokens=False)

            # Find content tokens in input_ids starting from current_pos
            start_idx = find_subsequence(
                input_ids[current_pos:],
                content_tokens[:min(5, len(content_tokens))]  # Match first 5 tokens
            )

            if start_idx != -1:
                abs_start = current_pos + start_idx
                abs_end = min(abs_start + len(content_tokens), len(input_ids))
                labels[abs_start:abs_end] = input_ids[abs_start:abs_end]

        current_pos += len(role_tokens)

    return labels


# ═══════════════════════════════════════════════════════════════════════
# 4. CUSTOM SFT DATA COLLATOR WITH MASKING
# ═══════════════════════════════════════════════════════════════════════

class SFTDataCollator:
    """
    Custom data collator that applies response-only loss masking.

    This is what TRL's SFTTrainer does internally.
    Here we implement it manually for educational purposes.

    The collator:
    1. Receives a batch of examples
    2. Pads them to equal length
    3. Creates labels with prompt tokens masked to -100
    4. Returns the batch ready for training
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        response_marker: str = "### Assistant:",
        max_length: int = 2048,
        padding: str = "max_length",
    ):
        self.tokenizer = tokenizer
        self.response_marker = response_marker
        self.max_length = max_length
        self.padding = padding
        self.mask_value = -100

        # Pre-tokenize the response marker
        self.marker_ids = tokenizer.encode(
            response_marker, add_special_tokens=False
        )

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Process a batch of examples."""
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for example in examples:
            input_ids = example["input_ids"]

            # Truncate if needed
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]

            # Create masked labels
            labels = self._create_masked_labels(input_ids)

            # Pad
            padding_length = self.max_length - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            labels = labels + [self.mask_value] * padding_length

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

    def _create_masked_labels(self, input_ids: List[int]) -> List[int]:
        """Mask prompt tokens in the labels."""
        response_start = find_subsequence(input_ids, self.marker_ids)

        if response_start == -1:
            # No response marker found — mask everything
            return [self.mask_value] * len(input_ids)

        # Mask everything before and including the marker
        mask_until = response_start + len(self.marker_ids)
        labels = [self.mask_value] * mask_until + input_ids[mask_until:]

        return labels


# ═══════════════════════════════════════════════════════════════════════
# 5. VISUALIZATION OF MASKING
# ═══════════════════════════════════════════════════════════════════════

def visualize_masking(
    text: str,
    tokenizer: PreTrainedTokenizer,
    response_marker: str = "### Assistant:",
):
    """
    Visualize which tokens are masked vs trained.

    This is incredibly useful for debugging SFT:
    - Verify that ONLY response tokens get gradient
    - Check that the response marker is found correctly
    - Ensure multi-turn masking works as expected
    """
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    labels = create_labels_with_masking(input_ids, tokenizer, response_marker)

    print("\n" + "=" * 70)
    print("LOSS MASKING VISUALIZATION")
    print("=" * 70)

    # Decode token by token
    print(f"\n{'Token':<20} {'ID':>6} {'Label':>6} {'In Loss?':>10}")
    print("-" * 50)

    for i, (token_id, label) in enumerate(zip(input_ids, labels)):
        token_str = tokenizer.decode([token_id])
        in_loss = "✗ MASKED" if label == -100 else "✓ TRAIN"
        label_str = str(label) if label != -100 else "-100"
        print(f"{repr(token_str):<20} {token_id:>6} {label_str:>6} {in_loss:>10}")

    # Summary
    masked_count = sum(1 for l in labels if l == -100)
    trained_count = len(labels) - masked_count
    print(f"\nTotal tokens: {len(labels)}")
    print(f"Masked (prompt): {masked_count} ({100*masked_count/len(labels):.1f}%)")
    print(f"Trained (response): {trained_count} ({100*trained_count/len(labels):.1f}%)")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
# 6. CUSTOM LOSS FUNCTION FOR SFT
# ═══════════════════════════════════════════════════════════════════════

def compute_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """
    Compute the SFT loss manually.

    This is what happens inside the model when you pass labels:

    1. Shift logits and labels (the model predicts the NEXT token)
    2. Compute cross-entropy loss
    3. Tokens with label=-100 are automatically ignored

    Understanding this is crucial:
    - logits shape: (batch_size, seq_len, vocab_size)
    - labels shape: (batch_size, seq_len)
    - Loss is averaged ONLY over non-masked tokens
    """
    # Shift so that tokens < n predict token n
    # (Standard causal LM: at position i, predict position i+1)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # -100 tokens are ignored!
    loss = loss_fct(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
    )

    return loss


# ═══════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from transformers import AutoTokenizer

    explain_loss_masking()

    # Load a tokenizer for demo
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Demo: Visualize masking on a simple example
    example_text = "### Human: What is artificial intelligence?\n### Assistant: Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence."

    visualize_masking(example_text, tokenizer, "### Assistant:")

    # Demo: Multi-turn conversation
    multi_turn_text = (
        "### Human: What is AI?\n"
        "### Assistant: AI is artificial intelligence.\n"
        "### Human: How does it work?\n"
        "### Assistant: AI works by learning patterns from data using mathematical models."
    )

    print("\n\n── Multi-turn Example ──")
    visualize_masking(multi_turn_text, tokenizer, "### Assistant:")
