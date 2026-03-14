"""
═══════════════════════════════════════════════════════════════════════════
PEFT CONFIGURATION — HuggingFace PEFT Library Configuration Patterns
═══════════════════════════════════════════════════════════════════════════

The HuggingFace `peft` library provides a UNIFIED API for all PEFT methods.
This module shows how to configure each method correctly.

PEFT LIBRARY ARCHITECTURE:
──────────────────────────
1. Define a PeftConfig (LoraConfig, PrefixTuningConfig, etc.)
2. Wrap model with get_peft_model(model, config)
3. Only PEFT parameters are trainable — base model is frozen
4. Train normally with HuggingFace Trainer
5. Save only PEFT weights (tiny checkpoint)
6. Load with PeftModel.from_pretrained(base_model, adapter_path)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# NOTE: These imports require `pip install peft`
# We wrap them to allow the file to be imported even without peft installed
try:
    from peft import (
        LoraConfig,
        PrefixTuningConfig,
        PromptTuningConfig,
        PromptTuningInit,
        PromptEncoderConfig,   # P-Tuning v2
        IA3Config,
        AdaLoraConfig,
        LoHaConfig,
        LoKrConfig,
        TaskType,
        get_peft_model,
        PeftModel,
        prepare_model_for_kbit_training,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠ peft library not installed. Install with: pip install peft")
    print("  Code structure is still viewable for learning purposes.")


# ═══════════════════════════════════════════════════════════════════════
# 1. LoRA CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

def get_lora_config(
    task_type: str = "CAUSAL_LM",
    rank: int = 16,
    alpha: int = 32,
    target_modules: Optional[List[str]] = None,
    dropout: float = 0.05,
    bias: str = "none",
    modules_to_save: Optional[List[str]] = None,
) -> Dict:
    """
    Build LoRA configuration with detailed parameter explanations.

    KEY PARAMETERS:
    ───────────────
    r (rank): Rank of the low-rank decomposition.
        - Higher r = more parameters, more capacity, but diminishing returns
        - Typical values: 4, 8, 16, 32, 64
        - Rule of thumb: Start with 8-16, increase if underfitting

    lora_alpha: Scaling factor = alpha / r
        - Controls the magnitude of the LoRA update
        - Higher alpha = larger updates = faster adaptation
        - Convention: alpha = 2*r (so scaling = 2.0)
        - Some prefer alpha = r (scaling = 1.0) for stability

    target_modules: Which weight matrices to apply LoRA to
        - ["q_proj", "v_proj"]: Original LoRA paper (conservative)
        - ["q_proj", "k_proj", "v_proj", "o_proj"]: All attention
        - ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
           "up_proj", "down_proj"]: ALL linear layers (aggressive)
        - More targets = more capacity but more parameters

    lora_dropout: Dropout on LoRA layers
        - 0.0-0.1 typical
        - Helps prevent overfitting, especially with small datasets

    bias: How to handle bias terms
        - "none": Don't train any biases (most common)
        - "all": Train all biases
        - "lora_only": Train biases in LoRA layers only

    modules_to_save: Additional modules to always train (not LoRA)
        - e.g., ["embed_tokens", "lm_head"] for vocabulary adaptation
    """
    # Map string to TaskType
    task_map = {
        "CAUSAL_LM": "CAUSAL_LM",
        "SEQ_2_SEQ_LM": "SEQ_2_SEQ_LM",
        "SEQ_CLS": "SEQ_CLS",
        "TOKEN_CLS": "TOKEN_CLS",
        "QUESTION_ANS": "QUESTION_ANS",
        "FEATURE_EXTRACTION": "FEATURE_EXTRACTION",
    }

    # Default target modules for common architectures
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    config_dict = {
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": target_modules,
        "lora_dropout": dropout,
        "bias": bias,
        "task_type": task_map.get(task_type, task_type),
    }

    if modules_to_save:
        config_dict["modules_to_save"] = modules_to_save

    if PEFT_AVAILABLE:
        task = getattr(TaskType, task_type, TaskType.CAUSAL_LM)
        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias=bias,
            task_type=task,
            modules_to_save=modules_to_save,
        )
        return {"config": config, "config_dict": config_dict}

    return {"config": None, "config_dict": config_dict}


# MODEL-SPECIFIC TARGET MODULES REFERENCE
TARGET_MODULES_BY_ARCHITECTURE = {
    "gpt2": {
        "attention_only": ["c_attn"],  # GPT-2 uses combined QKV matrix
        "all_linear": ["c_attn", "c_proj", "c_fc"],
    },
    "gpt_neox": {
        "attention_only": ["query_key_value"],
        "all_linear": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    },
    "llama": {
        "attention_qv": ["q_proj", "v_proj"],
        "attention_all": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "mistral": {
        "attention_qv": ["q_proj", "v_proj"],
        "attention_all": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "falcon": {
        "attention_only": ["query_key_value"],
        "all_linear": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    },
    "phi": {
        "attention_qv": ["q_proj", "v_proj"],
        "all_linear": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    },
}


# ═══════════════════════════════════════════════════════════════════════
# 2. QLoRA CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

def get_qlora_config(
    rank: int = 16,
    alpha: int = 32,
    target_modules: Optional[List[str]] = None,
) -> Dict:
    """
    QLoRA configuration — LoRA + 4-bit quantization.

    ADDITIONAL REQUIREMENTS:
    ────────────────────────
    1. Load model with BitsAndBytesConfig for 4-bit quantization
    2. Call prepare_model_for_kbit_training(model) before applying LoRA
    3. LoRA config is the same as regular LoRA

    QUANTIZATION CONFIG:
    ────────────────────
    - load_in_4bit=True:  Use 4-bit quantization
    - bnb_4bit_quant_type="nf4":  NormalFloat4 (optimal for neural net weights)
    - bnb_4bit_compute_dtype=torch.bfloat16:  Compute in bf16 during forward pass
    - bnb_4bit_use_double_quant=True:  Quantize the quantization constants
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]

    # LoRA config (same as regular LoRA)
    lora_config_dict = {
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": target_modules,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    # Quantization config (for model loading)
    quant_config_dict = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
    }

    return {
        "lora_config": lora_config_dict,
        "quantization_config": quant_config_dict,
        "usage_example": """
# QLoRA Complete Setup:
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import torch

# 1. Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 2. Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# 3. Prepare for k-bit training (handles gradient checkpointing, etc.)
model = prepare_model_for_kbit_training(model)

# 4. Apply LoRA
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=[...])
model = get_peft_model(model, lora_config)
""",
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. PREFIX TUNING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

def get_prefix_tuning_config(
    num_virtual_tokens: int = 20,
    encoder_hidden_size: int = 128,
) -> Dict:
    """
    Prefix Tuning configuration.

    KEY PARAMETERS:
    ───────────────
    num_virtual_tokens: Number of virtual prefix tokens
        - More tokens = more capacity but reduces effective context
        - Typical: 10-50
        - Start with 20, tune based on performance

    encoder_hidden_size: Size of the reparameterization MLP
        - The prefix embeddings are learned through an MLP for stability
        - Typical: 128-256 for small models, 512+ for large models
    """
    config_dict = {
        "num_virtual_tokens": num_virtual_tokens,
        "encoder_hidden_size": encoder_hidden_size,
        "task_type": "CAUSAL_LM",
    }

    if PEFT_AVAILABLE:
        config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=num_virtual_tokens,
            encoder_hidden_size=encoder_hidden_size,
        )
        return {"config": config, "config_dict": config_dict}

    return {"config": None, "config_dict": config_dict}


# ═══════════════════════════════════════════════════════════════════════
# 4. PROMPT TUNING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

def get_prompt_tuning_config(
    num_virtual_tokens: int = 20,
    prompt_tuning_init: str = "RANDOM",
    prompt_tuning_init_text: Optional[str] = None,
) -> Dict:
    """
    Prompt Tuning configuration.

    KEY PARAMETERS:
    ───────────────
    num_virtual_tokens: Number of soft prompt tokens
        - Typical: 8-100
        - More tokens = more parameters but diminishing returns

    prompt_tuning_init: How to initialize soft prompts
        - "RANDOM": Random initialization
        - "TEXT": Initialize from the embeddings of actual text
        Text initialization often converges faster!

    prompt_tuning_init_text: Text to use for initialization
        - e.g., "Classify the following text: " for classification
        - Gives the model a "head start" with meaningful embeddings
    """
    config_dict = {
        "num_virtual_tokens": num_virtual_tokens,
        "prompt_tuning_init": prompt_tuning_init,
        "task_type": "CAUSAL_LM",
    }

    if prompt_tuning_init_text:
        config_dict["prompt_tuning_init_text"] = prompt_tuning_init_text

    if PEFT_AVAILABLE:
        init_enum = (
            PromptTuningInit.TEXT if prompt_tuning_init == "TEXT"
            else PromptTuningInit.RANDOM
        )
        kwargs = {
            "task_type": TaskType.CAUSAL_LM,
            "num_virtual_tokens": num_virtual_tokens,
            "prompt_tuning_init": init_enum,
        }
        if prompt_tuning_init == "TEXT" and prompt_tuning_init_text:
            kwargs["prompt_tuning_init_text"] = prompt_tuning_init_text
            kwargs["tokenizer_name_or_path"] = "gpt2"  # Needs tokenizer for text init

        config = PromptTuningConfig(**kwargs)
        return {"config": config, "config_dict": config_dict}

    return {"config": None, "config_dict": config_dict}


# ═══════════════════════════════════════════════════════════════════════
# 5. IA³ CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

def get_ia3_config(
    target_modules: Optional[List[str]] = None,
    feedforward_modules: Optional[List[str]] = None,
) -> Dict:
    """
    IA³ configuration.

    KEY PARAMETERS:
    ───────────────
    target_modules: Which modules to apply IA³ to
        - Default: key, value, and feedforward projections
        - For LLaMA: ["k_proj", "v_proj", "down_proj"]

    feedforward_modules: Which of target_modules are feedforward layers
        - IA³ applies rescaling differently for attention vs FFN
        - Attention: rescale after projection (l_k ⊙ K, l_v ⊙ V)
        - FFN: rescale before down-projection
    """
    if target_modules is None:
        target_modules = ["k_proj", "v_proj", "down_proj"]
    if feedforward_modules is None:
        feedforward_modules = ["down_proj"]

    config_dict = {
        "target_modules": target_modules,
        "feedforward_modules": feedforward_modules,
        "task_type": "CAUSAL_LM",
    }

    if PEFT_AVAILABLE:
        config = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            feedforward_modules=feedforward_modules,
        )
        return {"config": config, "config_dict": config_dict}

    return {"config": None, "config_dict": config_dict}


# ═══════════════════════════════════════════════════════════════════════
# 6. ADVANCED LoRA VARIANTS
# ═══════════════════════════════════════════════════════════════════════

def get_adalora_config(
    init_r: int = 12,
    target_r: int = 4,
    beta1: float = 0.85,
    beta2: float = 0.85,
    tinit: int = 200,
    tfinal: int = 1000,
    deltaT: int = 10,
) -> Dict:
    """
    AdaLoRA configuration — Adaptive LoRA rank per layer.

    AdaLoRA starts with a higher rank and PRUNES it during training,
    allocating more parameters to important weight matrices and fewer
    to less important ones.

    KEY PARAMETERS:
    ───────────────
    init_r: Initial rank for all layers
    target_r: Target average rank after pruning
    tinit: Warmup steps before starting to prune
    tfinal: Step to reach target rank
    deltaT: Interval between importance scoring
    """
    config_dict = {
        "init_r": init_r,
        "target_r": target_r,
        "beta1": beta1,
        "beta2": beta2,
        "tinit": tinit,
        "tfinal": tfinal,
        "deltaT": deltaT,
        "task_type": "CAUSAL_LM",
    }

    if PEFT_AVAILABLE:
        config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            init_r=init_r,
            target_r=target_r,
            beta1=beta1,
            beta2=beta2,
            tinit=tinit,
            tfinal=tfinal,
            deltaT=deltaT,
        )
        return {"config": config, "config_dict": config_dict}

    return {"config": None, "config_dict": config_dict}


# ═══════════════════════════════════════════════════════════════════════
# 7. CONFIGURATION RECIPES (Best Practices)
# ═══════════════════════════════════════════════════════════════════════

RECIPES = {
    "chat_lora_conservative": {
        "description": "Conservative LoRA for chat/instruction tuning (recommended starting point)",
        "method": "lora",
        "config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
        },
        "training": {
            "learning_rate": 2e-4,
            "num_epochs": 3,
            "warmup_ratio": 0.03,
            "weight_decay": 0.001,
            "lr_scheduler": "cosine",
            "max_grad_norm": 0.3,
        },
    },
    "chat_lora_aggressive": {
        "description": "Aggressive LoRA with all linear layers for maximum quality",
        "method": "lora",
        "config": {
            "r": 64,
            "lora_alpha": 128,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
        },
        "training": {
            "learning_rate": 1e-4,
            "num_epochs": 3,
            "warmup_ratio": 0.05,
            "weight_decay": 0.01,
        },
    },
    "qlora_7b": {
        "description": "QLoRA for 7B model on consumer GPU (24GB VRAM)",
        "method": "qlora",
        "config": {
            "r": 64,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
        },
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
        },
        "training": {
            "learning_rate": 2e-4,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "num_epochs": 1,
            "warmup_ratio": 0.03,
            "max_grad_norm": 0.3,
            "gradient_checkpointing": True,
        },
    },
    "classification_lora": {
        "description": "LoRA for sequence classification (e.g., sentiment analysis)",
        "method": "lora",
        "config": {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "modules_to_save": ["classifier"],  # Always train the classification head
        },
        "training": {
            "learning_rate": 3e-4,
            "num_epochs": 5,
            "warmup_ratio": 0.1,
        },
    },
    "prompt_tuning_minimal": {
        "description": "Minimal prompt tuning — fewest possible parameters",
        "method": "prompt_tuning",
        "config": {
            "num_virtual_tokens": 20,
            "prompt_tuning_init": "TEXT",
            "prompt_tuning_init_text": "Classify the following text: ",
        },
        "training": {
            "learning_rate": 3e-2,  # Higher LR for prompt tuning!
            "num_epochs": 10,
        },
    },
}


def print_recipes():
    """Print all configuration recipes."""
    print(f"\n{'═' * 70}")
    print("PEFT CONFIGURATION RECIPES")
    print(f"{'═' * 70}")

    for name, recipe in RECIPES.items():
        print(f"\n{'─' * 70}")
        print(f"Recipe: {name}")
        print(f"Description: {recipe['description']}")
        print(f"Method: {recipe['method']}")
        print(f"\n  PEFT Config:")
        for k, v in recipe["config"].items():
            print(f"    {k}: {v}")
        print(f"\n  Training Config:")
        for k, v in recipe["training"].items():
            print(f"    {k}: {v}")
        if "quantization" in recipe:
            print(f"\n  Quantization Config:")
            for k, v in recipe["quantization"].items():
                print(f"    {k}: {v}")


# ═══════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("PEFT CONFIGURATION GUIDE")
    print("=" * 70)

    # Show all configs
    print("\n1. LoRA Configuration:")
    lora_cfg = get_lora_config(rank=16, alpha=32, target_modules=["q_proj", "v_proj"])
    for k, v in lora_cfg["config_dict"].items():
        print(f"   {k}: {v}")

    print("\n2. QLoRA Configuration:")
    qlora_cfg = get_qlora_config(rank=16, alpha=32)
    print(f"   LoRA config: {qlora_cfg['lora_config']}")
    print(f"   Quant config: {qlora_cfg['quantization_config']}")

    print("\n3. Prefix Tuning Configuration:")
    prefix_cfg = get_prefix_tuning_config(num_virtual_tokens=20)
    for k, v in prefix_cfg["config_dict"].items():
        print(f"   {k}: {v}")

    print("\n4. Prompt Tuning Configuration:")
    prompt_cfg = get_prompt_tuning_config(
        num_virtual_tokens=20,
        prompt_tuning_init="TEXT",
        prompt_tuning_init_text="Classify: ",
    )
    for k, v in prompt_cfg["config_dict"].items():
        print(f"   {k}: {v}")

    print("\n5. IA³ Configuration:")
    ia3_cfg = get_ia3_config()
    for k, v in ia3_cfg["config_dict"].items():
        print(f"   {k}: {v}")

    print("\n6. AdaLoRA Configuration:")
    adalora_cfg = get_adalora_config()
    for k, v in adalora_cfg["config_dict"].items():
        print(f"   {k}: {v}")

    # Target modules reference
    print(f"\n{'─' * 70}")
    print("TARGET MODULES BY ARCHITECTURE:")
    for arch, targets in TARGET_MODULES_BY_ARCHITECTURE.items():
        print(f"\n  {arch}:")
        for scope, modules in targets.items():
            print(f"    {scope}: {modules}")

    # Recipes
    print_recipes()
