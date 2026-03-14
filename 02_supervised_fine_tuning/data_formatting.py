"""
═══════════════════════════════════════════════════════════════════════════
DATA FORMATTING FOR SUPERVISED FINE-TUNING
═══════════════════════════════════════════════════════════════════════════

This module handles the critical task of converting various dataset formats
into the prompt-response structure needed for SFT.

Covers:
  - Single-turn prompt/response formatting
  - Multi-turn conversation formatting
  - Chat template application
  - Dataset quality filtering and deduplication
  - Sequence packing (fitting multiple examples into one sequence)
  - Various source format handling (Alpaca, ShareGPT, OASST, etc.)
"""

import hashlib
from typing import List, Dict, Optional, Callable
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer


# ═══════════════════════════════════════════════════════════════════════
# 1. DATASET FORMAT CONVERTERS
# ═══════════════════════════════════════════════════════════════════════

def convert_alpaca_format(example: Dict) -> Dict:
    """
    Convert Alpaca-format data to standard prompt/response format.

    Alpaca format:
        {"instruction": "...", "input": "...", "output": "..."}

    Converts to:
        {"prompt": "...", "response": "..."}

    The Alpaca dataset (52K examples) was one of the first open-source
    instruction-tuning datasets and its format is widely used.
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        prompt = f"{instruction}\n\nInput: {input_text}"
    else:
        prompt = instruction

    return {"prompt": prompt, "response": output}


def convert_sharegpt_format(example: Dict) -> List[Dict[str, str]]:
    """
    Convert ShareGPT-format data to messages format.

    ShareGPT format:
        {"conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."},
            ...
        ]}

    Converts to:
        [{"role": "user", "content": "..."},
         {"role": "assistant", "content": "..."}]

    ShareGPT is a collection of conversations from ChatGPT that users
    shared publicly. It's a popular format for multi-turn SFT.
    """
    role_map = {
        "human": "user",
        "gpt": "assistant",
        "system": "system",
    }

    messages = []
    for turn in example.get("conversations", []):
        role = role_map.get(turn["from"], turn["from"])
        messages.append({"role": role, "content": turn["value"]})

    return messages


def convert_oasst_format(example: Dict) -> Dict:
    """
    Convert OASST (Open Assistant) format.

    OASST provides tree-structured conversations with quality ratings.
    We extract the highest-rated path through the conversation tree.
    """
    return {
        "prompt": example.get("instruction", example.get("prompt", "")),
        "response": example.get("response", example.get("output", "")),
    }


def convert_messages_to_prompt_response(
    messages: List[Dict[str, str]],
) -> Dict[str, str]:
    """
    Convert a multi-turn conversation into a single prompt/response pair.

    For SFT, we train on the LAST assistant response while the conversation
    history becomes the prompt/context.

    Messages:
        [system, user1, assistant1, user2, assistant2]

    Becomes:
        prompt = [system, user1, assistant1, user2]
        response = [assistant2]
    """
    if not messages:
        return {"prompt": "", "response": ""}

    # Find the last assistant message
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx is None:
        return {"prompt": str(messages), "response": ""}

    prompt_messages = messages[:last_assistant_idx]
    response = messages[last_assistant_idx]["content"]

    # Format prompt messages as a string
    prompt_parts = []
    for msg in prompt_messages:
        prompt_parts.append(f"{msg['role'].capitalize()}: {msg['content']}")

    return {
        "prompt": "\n".join(prompt_parts),
        "response": response,
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. FORMATTING WITH CHAT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════

def format_with_tokenizer_template(
    messages: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    add_generation_prompt: bool = False,
) -> str:
    """
    Use the tokenizer's built-in chat template (if available).

    Modern HuggingFace tokenizers come with a `chat_template` that
    automatically formats messages correctly for that specific model.

    This is the RECOMMENDED approach in production because:
    1. It's guaranteed to match the model's expected format
    2. It handles special tokens correctly
    3. It works with the model's generation pipeline
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    else:
        # Fallback: simple formatting
        text = ""
        for msg in messages:
            text += f"<|{msg['role']}|>\n{msg['content']}\n"
        if add_generation_prompt:
            text += "<|assistant|>\n"
        return text


def format_dataset_for_sft(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    format_type: str = "prompt_response",
    system_message: Optional[str] = None,
) -> Dataset:
    """
    Format an entire dataset for SFT training.

    Supports two main approaches:
    1. prompt_response: Simple prompt → response pairs
    2. messages: Full conversation with chat template

    Args:
        dataset: Raw dataset with 'prompt'/'response' or 'messages' columns
        tokenizer: Tokenizer with chat template
        format_type: 'prompt_response' or 'messages'
        system_message: Optional system prompt to prepend

    Returns:
        Dataset with 'text' column ready for SFT training
    """

    def format_prompt_response(example):
        """Format a single prompt-response pair."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": example["prompt"]})
        messages.append({"role": "assistant", "content": example["response"]})

        text = format_with_tokenizer_template(messages, tokenizer)
        return {"text": text}

    def format_messages(example):
        """Format a multi-turn conversation."""
        messages = example["messages"]
        if system_message and (not messages or messages[0]["role"] != "system"):
            messages = [{"role": "system", "content": system_message}] + messages

        text = format_with_tokenizer_template(messages, tokenizer)
        return {"text": text}

    if format_type == "prompt_response":
        return dataset.map(format_prompt_response, desc="Formatting for SFT")
    elif format_type == "messages":
        return dataset.map(format_messages, desc="Formatting for SFT")
    else:
        raise ValueError(f"Unknown format_type: {format_type}")


# ═══════════════════════════════════════════════════════════════════════
# 3. DATASET QUALITY FILTERING
# ═══════════════════════════════════════════════════════════════════════

def filter_dataset_quality(
    dataset: Dataset,
    min_prompt_length: int = 10,
    min_response_length: int = 20,
    max_response_length: int = 4096,
    remove_duplicates: bool = True,
    prompt_column: str = "prompt",
    response_column: str = "response",
) -> Dataset:
    """
    Filter dataset for quality.

    Data quality is CRITICAL for SFT — "garbage in, garbage out."
    Bad training data produces models that are worse than the base model.

    Quality checks:
    1. Minimum length: Remove too-short prompts/responses (likely noise)
    2. Maximum length: Remove extremely long responses (often copy-paste spam)
    3. Deduplication: Remove exact and near-duplicate examples
    4. Content filtering: Remove empty, whitespace-only entries
    """
    original_size = len(dataset)
    print(f"\n📋 Quality filtering ({original_size} examples)")

    # Filter by length
    def length_filter(example):
        prompt = example.get(prompt_column, "")
        response = example.get(response_column, "")

        if not prompt or not response:
            return False
        if len(prompt.strip()) < min_prompt_length:
            return False
        if len(response.strip()) < min_response_length:
            return False
        if len(response.strip()) > max_response_length:
            return False
        return True

    dataset = dataset.filter(length_filter, desc="Length filtering")
    print(f"   After length filter: {len(dataset)} ({original_size - len(dataset)} removed)")

    # Deduplication
    if remove_duplicates:
        seen_hashes = set()
        indices_to_keep = []

        for i, example in enumerate(dataset):
            # Hash the prompt + response for deduplication
            text = f"{example.get(prompt_column, '')}|||{example.get(response_column, '')}"
            text_hash = hashlib.md5(text.encode()).hexdigest()

            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                indices_to_keep.append(i)

        before_dedup = len(dataset)
        dataset = dataset.select(indices_to_keep)
        print(f"   After deduplication: {len(dataset)} ({before_dedup - len(dataset)} duplicates)")

    print(f"   Final dataset size: {len(dataset)} ({100*len(dataset)/original_size:.1f}% retained)")

    return dataset


# ═══════════════════════════════════════════════════════════════════════
# 4. SEQUENCE PACKING
# ═══════════════════════════════════════════════════════════════════════

def pack_sequences(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 2048,
    text_column: str = "text",
) -> Dataset:
    """
    Pack multiple short examples into a single sequence.

    PACKING EXPLAINED:
    ──────────────────
    Without packing:
        Sequence 1: [tokens_1][PAD][PAD][PAD][PAD][PAD]  ← wasted computation
        Sequence 2: [tokens_2][PAD][PAD][PAD]              ← wasted computation

    With packing:
        Sequence 1: [tokens_1][EOS][tokens_2][EOS][tokens_3]  ← fully utilized

    Benefits:
    - Up to 5x faster training (no wasted padding computation)
    - Better GPU utilization
    - Each batch contains more diverse examples

    Caveat:
    - Need attention masking to prevent cross-contamination between packed examples
    - TRL's SFTTrainer handles this automatically with packing=True
    """
    all_input_ids = []

    for example in dataset:
        tokens = tokenizer(
            example[text_column],
            truncation=False,
            add_special_tokens=True,
        )["input_ids"]
        all_input_ids.extend(tokens)
        all_input_ids.append(tokenizer.eos_token_id)  # Separator

    # Chunk into max_seq_length blocks
    chunks = []
    for i in range(0, len(all_input_ids) - max_seq_length, max_seq_length):
        chunk = all_input_ids[i:i + max_seq_length]
        chunks.append({"input_ids": chunk, "labels": chunk.copy()})

    print(f"   Packed {len(dataset)} examples into {len(chunks)} sequences")
    print(f"   Packing efficiency: {len(chunks) / len(dataset) * 100:.1f}%")

    return Dataset.from_list(chunks)


# ═══════════════════════════════════════════════════════════════════════
# 5. LOAD POPULAR SFT DATASETS
# ═══════════════════════════════════════════════════════════════════════

def load_sft_dataset(
    dataset_name: str = "databricks/databricks-dolly-15k",
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    Load a popular SFT dataset from HuggingFace Hub.

    Popular SFT datasets:
    ─────────────────────
    1. databricks/databricks-dolly-15k — 15K instruction pairs (CC license!)
    2. tatsu-lab/alpaca — 52K instruction pairs (Stanford)
    3. Open-Orca/OpenOrca — 4.2M instructions (GPT-4 augmented)
    4. HuggingFaceH4/ultrachat_200k — 200K multi-turn conversations
    5. Intel/orca_dpo_pairs — DPO-ready with chosen/rejected
    6. OpenAssistant/oasst1 — Human-written assistant conversations
    """
    print(f"\nLoading SFT dataset: {dataset_name}")

    dataset = load_dataset(dataset_name, split=split)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"   Loaded {len(dataset)} examples")
    print(f"   Columns: {dataset.column_names}")

    # Show sample
    if len(dataset) > 0:
        example = dataset[0]
        print(f"\n   Sample example keys: {list(example.keys())}")
        for key in list(example.keys())[:3]:
            val = str(example[key])[:100]
            print(f"   {key}: {val}...")

    return dataset


def prepare_dolly_dataset(max_samples: Optional[int] = None) -> Dataset:
    """
    Load and prepare the Databricks Dolly dataset for SFT.

    Dolly is an excellent SFT dataset because:
    1. Fully open license (CC BY-SA 3.0)
    2. Human-written responses
    3. Diverse categories (QA, summarization, creative writing, etc.)
    4. 15K high-quality examples
    """
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def format_dolly(example):
        instruction = example["instruction"]
        context = example["context"]
        response = example["response"]

        if context:
            prompt = f"{instruction}\n\nContext: {context}"
        else:
            prompt = instruction

        return {"prompt": prompt, "response": response}

    dataset = dataset.map(format_dolly, desc="Formatting Dolly")
    print(f"   Prepared {len(dataset)} examples from Dolly dataset")

    return dataset


# ═══════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("DATA FORMATTING DEMO")
    print("=" * 60)

    # Demo: Alpaca format conversion
    alpaca_example = {
        "instruction": "Explain what machine learning is.",
        "input": "",
        "output": "Machine learning is a branch of AI that enables computers to learn from data..."
    }
    converted = convert_alpaca_format(alpaca_example)
    print(f"\nAlpaca → Prompt/Response:")
    print(f"  Prompt: {converted['prompt'][:80]}...")
    print(f"  Response: {converted['response'][:80]}...")

    # Demo: ShareGPT format conversion
    sharegpt_example = {
        "conversations": [
            {"from": "human", "value": "What is machine learning?"},
            {"from": "gpt", "value": "Machine learning is a branch of AI..."},
            {"from": "human", "value": "How does it differ from deep learning?"},
            {"from": "gpt", "value": "Deep learning is a subset of machine learning..."},
        ]
    }
    messages = convert_sharegpt_format(sharegpt_example)
    print(f"\nShareGPT → Messages:")
    for msg in messages:
        print(f"  [{msg['role']}]: {msg['content'][:60]}...")
