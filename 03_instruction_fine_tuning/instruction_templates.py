"""
═══════════════════════════════════════════════════════════════════════════
INSTRUCTION TEMPLATES — Formatting Strategies for Instruction Tuning
═══════════════════════════════════════════════════════════════════════════

How you FORMAT the instruction data critically affects learning.
This module covers different templating strategies and their trade-offs.

Key insight: The template determines:
  1. How the model distinguishes instruction from input from output
  2. How well the model generalizes to new instructions
  3. What conversation format the model learns
"""

from typing import Dict, Optional, List
from transformers import PreTrainedTokenizer


# ═══════════════════════════════════════════════════════════════════════
# 1. ALPACA-STYLE TEMPLATE (Most Popular for Instruction Tuning)
# ═══════════════════════════════════════════════════════════════════════

ALPACA_TEMPLATE_WITH_INPUT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_TEMPLATE_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""

ALPACA_PROMPT_WITH_INPUT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

ALPACA_PROMPT_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def format_alpaca(example: Dict, include_output: bool = True) -> str:
    """
    Format an instruction example using the Alpaca template.

    WHY ALPACA FORMAT IS POPULAR:
    ─────────────────────────────
    1. Clear section markers (###) help the model learn boundaries
    2. System prompt ("Below is an instruction...") sets expectations
    3. Separate Input field handles context when needed
    4. "### Response:" marks where generation should begin

    The model learns:
    - Everything before "### Response:" is context (masked in loss)
    - Everything after "### Response:" is what to generate (trained on)
    """
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text.strip():
        if include_output:
            return ALPACA_TEMPLATE_WITH_INPUT.format(
                instruction=instruction, input=input_text, output=output
            )
        else:
            return ALPACA_PROMPT_WITH_INPUT.format(
                instruction=instruction, input=input_text
            )
    else:
        if include_output:
            return ALPACA_TEMPLATE_NO_INPUT.format(
                instruction=instruction, output=output
            )
        else:
            return ALPACA_PROMPT_NO_INPUT.format(instruction=instruction)


# ═══════════════════════════════════════════════════════════════════════
# 2. CHAT-STYLE TEMPLATE (For Modern Chat Models)
# ═══════════════════════════════════════════════════════════════════════

def format_chat_instruction(
    example: Dict,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    system_message: str = "You are a helpful AI assistant that follows instructions carefully.",
    include_output: bool = True,
) -> str:
    """
    Format instruction data as a chat conversation.

    WHEN TO USE CHAT FORMAT:
    ────────────────────────
    - When your base model was pretrained with chat format
    - When you want the model to work as a chatbot
    - When using models with built-in chat templates (Mistral, LLaMA-3)

    The instruction becomes the user message,
    and the output becomes the assistant message.
    """
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example.get("output", "")

    # Combine instruction and input for the user message
    if input_text.strip():
        user_message = f"{instruction}\n\n{input_text}"
    else:
        user_message = instruction

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    if include_output:
        messages.append({"role": "assistant", "content": output})

    # Use tokenizer's chat template if available
    if tokenizer and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=not include_output
        )

    # Fallback: simple chat format
    text = ""
    for msg in messages:
        text += f"<|{msg['role']}|>\n{msg['content']}\n"
    if not include_output:
        text += "<|assistant|>\n"
    return text


# ═══════════════════════════════════════════════════════════════════════
# 3. DOLLY/DATABRICKS TEMPLATE
# ═══════════════════════════════════════════════════════════════════════

DOLLY_TEMPLATE_WITH_CONTEXT = """### Instruction:
{instruction}

### Context:
{context}

### Response:
{output}"""

DOLLY_TEMPLATE_NO_CONTEXT = """### Instruction:
{instruction}

### Response:
{output}"""


def format_dolly(example: Dict, include_output: bool = True) -> str:
    """
    Format using Dolly's template.

    Dolly uses "Context" instead of "Input", and has no system prompt.
    Simpler than Alpaca but equally effective.
    """
    instruction = example["instruction"]
    context = example.get("input", example.get("context", ""))
    output = example.get("output", example.get("response", ""))

    if context.strip():
        template = DOLLY_TEMPLATE_WITH_CONTEXT if include_output else \
            "### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n"
    else:
        template = DOLLY_TEMPLATE_NO_CONTEXT if include_output else \
            "### Instruction:\n{instruction}\n\n### Response:\n"

    return template.format(instruction=instruction, context=context, output=output)


# ═══════════════════════════════════════════════════════════════════════
# 4. FLAN-STYLE TEMPLATE (Google's Approach)
# ═══════════════════════════════════════════════════════════════════════

FLAN_TEMPLATES = {
    "standard": "{instruction}\n{input}\n\n{output}",
    "cot": "{instruction}\n{input}\n\nLet me think step by step.\n{output}",
    "direct": "Q: {instruction} {input}\nA: {output}",
    "options": "{instruction}\n{input}\nOptions:\n{options}\nAnswer: {output}",
}


def format_flan(
    example: Dict,
    template_type: str = "standard",
    include_output: bool = True,
) -> str:
    """
    FLAN-style formatting (Google's instruction tuning approach).

    KEY FLAN INNOVATION:
    ────────────────────
    FLAN (Fine-tuned LAnguage Net) showed that using MANY DIFFERENT
    templates for the same task dramatically improves generalization.

    Instead of one fixed template, FLAN uses 10+ templates per task:
    - "Summarize the following article."
    - "What is a short summary of the above?"
    - "TL;DR"
    - "Write a brief summary."

    This teaches the model that different wordings mean the same thing.
    """
    template = FLAN_TEMPLATES.get(template_type, FLAN_TEMPLATES["standard"])

    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example.get("output", "")

    if include_output:
        return template.format(
            instruction=instruction,
            input=input_text,
            output=output,
            options=example.get("options", ""),
        )
    else:
        # Return up to the output
        parts = template.split("{output}")
        return parts[0].format(
            instruction=instruction,
            input=input_text,
            options=example.get("options", ""),
        )


# ═══════════════════════════════════════════════════════════════════════
# 5. INSTRUCTION AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════

def augment_instruction(instruction: str, n_variants: int = 3) -> List[str]:
    """
    Generate paraphrased variants of an instruction.

    AUGMENTATION TECHNIQUES:
    ────────────────────────
    1. Prefix variation: Add/change leading phrases
    2. Tone variation: Formal vs casual
    3. Specificity variation: Detailed vs concise
    4. Format variation: Question vs imperative

    In production, you'd use an LLM for this (Self-Instruct approach).
    Here we show simple rule-based augmentation.
    """
    variants = [instruction]  # Original

    # Prefix augmentations
    prefixes = [
        "Please ", "Could you ", "I need you to ",
        "Your task is to ", "Help me ",
    ]

    # Suffix augmentations
    suffixes = [
        " Be concise.", " Provide a detailed answer.",
        " Think step by step.", " Be specific.",
    ]

    for prefix in prefixes[:n_variants]:
        # Don't add prefix if instruction already starts similarly
        if not instruction.lower().startswith(prefix.lower().strip()):
            variants.append(prefix + instruction[0].lower() + instruction[1:])

    for suffix in suffixes[:n_variants]:
        if not instruction.endswith('.'):
            variants.append(instruction + '.' + suffix)
        else:
            variants.append(instruction + suffix)

    return variants[:n_variants + 1]


# ═══════════════════════════════════════════════════════════════════════
# 6. TEMPLATE REGISTRY & FORMATTER
# ═══════════════════════════════════════════════════════════════════════

TEMPLATE_REGISTRY = {
    "alpaca": format_alpaca,
    "dolly": format_dolly,
}


def format_instruction_dataset(
    dataset,
    template_name: str = "alpaca",
    tokenizer: Optional[PreTrainedTokenizer] = None,
    include_output: bool = True,
):
    """
    Apply a formatting template to an entire dataset.

    This is the main function you call to prepare your instruction
    dataset for training.
    """
    if template_name == "chat":
        formatter = lambda ex: {"text": format_chat_instruction(
            ex, tokenizer, include_output=include_output
        )}
    elif template_name in TEMPLATE_REGISTRY:
        formatter = lambda ex: {"text": TEMPLATE_REGISTRY[template_name](
            ex, include_output=include_output
        )}
    else:
        raise ValueError(f"Unknown template: {template_name}")

    formatted = dataset.map(formatter, desc=f"Formatting with '{template_name}' template")
    return formatted


# ═══════════════════════════════════════════════════════════════════════
# 7. RESPONSE MARKER EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def get_response_marker(template_name: str) -> str:
    """
    Get the response marker string for loss masking.

    The response marker tells the loss masking where the OUTPUT begins.
    Everything before this marker is masked (label=-100).
    """
    markers = {
        "alpaca": "### Response:\n",
        "dolly": "### Response:\n",
        "chat": "<|assistant|>\n",
        "flan_standard": "\n\n",
        "flan_direct": "A: ",
    }
    return markers.get(template_name, "### Response:\n")


# ═══════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("INSTRUCTION TEMPLATE COMPARISON")
    print("=" * 70)

    example = {
        "instruction": "Summarize the following text in 2 sentences.",
        "input": "Machine learning is a rapidly growing field that has transformed many industries. From healthcare diagnostics to autonomous vehicles, ML algorithms are being deployed at scale.",
        "output": "Machine learning is a fast-growing technology transforming multiple industries. It powers applications from medical diagnostics to self-driving cars.",
        "category": "summarization",
    }

    for template_name in ["alpaca", "dolly"]:
        print(f"\n{'─' * 70}")
        print(f"TEMPLATE: {template_name.upper()}")
        print(f"{'─' * 70}")
        if template_name in TEMPLATE_REGISTRY:
            print(TEMPLATE_REGISTRY[template_name](example))
        print(f"\nResponse marker: '{get_response_marker(template_name)}'")

    # FLAN variants
    for flan_type in ["standard", "cot", "direct"]:
        print(f"\n{'─' * 70}")
        print(f"TEMPLATE: FLAN ({flan_type})")
        print(f"{'─' * 70}")
        print(format_flan(example, template_type=flan_type))

    # Augmentation demo
    print(f"\n{'─' * 70}")
    print("INSTRUCTION AUGMENTATION")
    print(f"{'─' * 70}")
    variants = augment_instruction("Summarize the following text in 2 sentences.")
    for i, v in enumerate(variants):
        print(f"  [{i}] {v}")
