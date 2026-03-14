"""
═══════════════════════════════════════════════════════════════════════════
CONVERSATION TEMPLATES FOR SFT
═══════════════════════════════════════════════════════════════════════════

Different models use different chat/conversation templates.
When doing SFT, you MUST use the correct template for your base model,
or the model won't learn the expected conversation format.

This module implements all major conversation templates used in the industry.

WHY TEMPLATES MATTER:
─────────────────────
The template defines:
1. How the model distinguishes between user input and its own output
2. Where the model should start and stop generating
3. How multi-turn conversations are structured

Using the wrong template = the model learns garbage formatting.
"""

from typing import List, Dict, Optional


# ═══════════════════════════════════════════════════════════════════════
# 1. CHATML FORMAT (Used by OpenAI, Mistral-Instruct, many others)
# ═══════════════════════════════════════════════════════════════════════

def format_chatml(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
) -> str:
    """
    ChatML (Chat Markup Language) format.

    Used by: OpenAI models, Mistral-Instruct, many fine-tuned models

    Format:
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        What is AI?<|im_end|>
        <|im_start|>assistant
        AI is...<|im_end|>

    This is the most widely adopted format in the industry.
    """
    formatted = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    if add_generation_prompt:
        formatted += "<|im_start|>assistant\n"

    return formatted


# ═══════════════════════════════════════════════════════════════════════
# 2. ALPACA FORMAT (Stanford Alpaca, widely used for instruction tuning)
# ═══════════════════════════════════════════════════════════════════════

def format_alpaca(
    instruction: str,
    input_text: str = "",
    output: str = "",
    include_output: bool = True,
) -> str:
    """
    Alpaca format — one of the earliest and most popular instruction formats.

    Used by: Stanford Alpaca, many LoRA fine-tunes

    Format (with input):
        Below is an instruction that describes a task, paired with an input
        that provides further context. Write a response that appropriately
        completes the request.

        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
        {output}

    Format (without input):
        Below is an instruction that describes a task.
        Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Response:
        {output}
    """
    if input_text:
        prompt = (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )

    if include_output and output:
        prompt += output

    return prompt


# ═══════════════════════════════════════════════════════════════════════
# 3. VICUNA FORMAT (Used in Vicuna, ShareGPT-based models)
# ═══════════════════════════════════════════════════════════════════════

def format_vicuna(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
) -> str:
    """
    Vicuna conversation format.

    Used by: Vicuna, ShareGPT-based models

    Format:
        A chat between a curious user and an artificial intelligence assistant.
        The assistant gives helpful, detailed, and polite answers.

        USER: What is AI?
        ASSISTANT: AI is...

        USER: Tell me more.
        ASSISTANT: Sure! ...
    """
    formatted = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
    )

    for message in messages:
        role = message["role"].upper()
        if role == "SYSTEM":
            continue  # System message is the preamble above
        elif role == "USER":
            formatted += f"USER: {message['content']}\n"
        elif role in ("ASSISTANT", "BOT"):
            formatted += f"ASSISTANT: {message['content']}\n"

    if add_generation_prompt:
        formatted += "ASSISTANT: "

    return formatted


# ═══════════════════════════════════════════════════════════════════════
# 4. LLAMA-2 CHAT FORMAT
# ═══════════════════════════════════════════════════════════════════════

def format_llama2_chat(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
) -> str:
    """
    LLaMA-2 Chat format.

    Used by: Meta's LLaMA-2-Chat models

    Format:
        <s>[INST] <<SYS>>
        You are a helpful assistant.
        <</SYS>>

        What is AI? [/INST] AI is... </s>
        <s>[INST] Tell me more [/INST] Sure! ... </s>

    Note the special tokens: [INST], [/INST], <<SYS>>, <</SYS>>
    """
    formatted = ""
    system_message = None

    for i, message in enumerate(messages):
        if message["role"] == "system":
            system_message = message["content"]
            continue

        if message["role"] == "user":
            if i == 0 or (i == 1 and system_message):
                # First user message — include system prompt
                if system_message:
                    formatted += f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{message['content']} [/INST] "
                else:
                    formatted += f"<s>[INST] {message['content']} [/INST] "
            else:
                formatted += f"<s>[INST] {message['content']} [/INST] "

        elif message["role"] == "assistant":
            formatted += f"{message['content']} </s>\n"

    if add_generation_prompt and not formatted.endswith("[/INST] "):
        pass  # Generation prompt already added

    return formatted


# ═══════════════════════════════════════════════════════════════════════
# 5. LLAMA-3 / MISTRAL-INSTRUCT FORMAT
# ═══════════════════════════════════════════════════════════════════════

def format_llama3(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
) -> str:
    """
    LLaMA-3 / Mistral format using special header tokens.

    Format:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful assistant.<|eot_id|>
        <|start_header_id|>user<|end_header_id|>

        What is AI?<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>

        AI is...<|eot_id|>
    """
    formatted = "<|begin_of_text|>"

    for message in messages:
        role = message["role"]
        content = message["content"]
        formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>\n"

    if add_generation_prompt:
        formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    return formatted


# ═══════════════════════════════════════════════════════════════════════
# 6. SIMPLE PROMPT-RESPONSE FORMAT (For basic SFT)
# ═══════════════════════════════════════════════════════════════════════

def format_simple(
    prompt: str,
    response: str = "",
    include_response: bool = True,
) -> str:
    """
    Simplest possible format for basic SFT.

    Format:
        ### Human: What is AI?
        ### Assistant: AI is...

    Good for: Learning, simple Q&A, when you don't need a specific template.
    """
    text = f"### Human: {prompt}\n### Assistant:"
    if include_response and response:
        text += f" {response}"
    return text


# ═══════════════════════════════════════════════════════════════════════
# 7. UNIVERSAL TEMPLATE APPLIER
# ═══════════════════════════════════════════════════════════════════════

TEMPLATE_MAP = {
    "chatml": format_chatml,
    "vicuna": format_vicuna,
    "llama2": format_llama2_chat,
    "llama3": format_llama3,
}


def apply_chat_template(
    messages: List[Dict[str, str]],
    template_name: str = "chatml",
    add_generation_prompt: bool = False,
) -> str:
    """
    Apply a named chat template to a list of messages.

    Args:
        messages: List of {"role": "user/assistant/system", "content": "..."}
        template_name: One of 'chatml', 'vicuna', 'llama2', 'llama3'
        add_generation_prompt: Add the assistant prompt for generation

    Returns:
        Formatted string ready for tokenization
    """
    if template_name not in TEMPLATE_MAP:
        raise ValueError(
            f"Unknown template: {template_name}. "
            f"Available: {list(TEMPLATE_MAP.keys())}"
        )

    return TEMPLATE_MAP[template_name](messages, add_generation_prompt)


def get_response_start_token(template_name: str) -> str:
    """
    Get the token/string that marks the START of the assistant's response.

    This is crucial for loss masking: we only compute loss on tokens
    AFTER this marker.
    """
    markers = {
        "chatml": "<|im_start|>assistant\n",
        "alpaca": "### Response:\n",
        "vicuna": "ASSISTANT: ",
        "llama2": "[/INST] ",
        "llama3": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "simple": "### Assistant:",
    }
    return markers.get(template_name, "### Assistant:")


# ═══════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example conversation
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What are transformers in deep learning?"},
        {"role": "assistant", "content": "Transformers are a type of neural network architecture..."},
        {"role": "user", "content": "How do they handle long sequences?"},
        {"role": "assistant", "content": "Transformers use self-attention mechanisms..."},
    ]

    print("=" * 70)
    for template in ["chatml", "vicuna", "llama2", "llama3"]:
        print(f"\n{'─' * 70}")
        print(f"TEMPLATE: {template.upper()}")
        print(f"{'─' * 70}")
        print(apply_chat_template(messages, template))

    # Alpaca format (different API since it's not multi-turn)
    print(f"\n{'─' * 70}")
    print("TEMPLATE: ALPACA")
    print(f"{'─' * 70}")
    print(format_alpaca(
        instruction="Explain transformers in deep learning",
        output="Transformers are a type of neural network architecture..."
    ))

    print(f"\n{'─' * 70}")
    print("TEMPLATE: SIMPLE")
    print(f"{'─' * 70}")
    print(format_simple(
        prompt="What are transformers?",
        response="Transformers are a type of neural network architecture..."
    ))
