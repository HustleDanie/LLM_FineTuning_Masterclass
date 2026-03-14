"""
═══════════════════════════════════════════════════════════════════════════
PEFT COMPARISON — Head-to-Head: Train Same Model with Different Methods
═══════════════════════════════════════════════════════════════════════════

The best way to understand PEFT methods is to compare them directly
on the same model and dataset. This module runs each method and
compares training speed, memory, parameter count, and model quality.

WHAT WE COMPARE:
────────────────
1. Full Fine-Tuning (baseline)
2. LoRA (rank=16)
3. Prefix Tuning (20 tokens)
4. Prompt Tuning (20 tokens)
5. IA³
6. BitFit (bias only)
"""

import time
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import OrderedDict

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

try:
    from peft import (
        get_peft_model,
        LoraConfig,
        PrefixTuningConfig,
        PromptTuningConfig,
        PromptTuningInit,
        IA3Config,
        TaskType,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠ peft not installed. Only full FT and BitFit will be compared.")


# ═══════════════════════════════════════════════════════════════════════
# 1. SHARED DATASET
# ═══════════════════════════════════════════════════════════════════════

def create_comparison_dataset(n_samples: int = 100) -> Dataset:
    """Small dataset for fair comparison."""
    templates = [
        "### Instruction: {instr}\n### Response: {resp}",
    ]
    tasks = [
        ("What is artificial intelligence?",
         "AI is the simulation of human intelligence by computer systems, including learning, reasoning, and self-correction."),
        ("Explain gravity in simple terms.",
         "Gravity is a force that pulls objects toward each other. The more mass an object has, the stronger its gravitational pull."),
        ("Write a function to add two numbers in Python.",
         "def add(a, b):\n    return a + b"),
        ("What causes rain?",
         "Rain occurs when water vapor in the atmosphere condenses into droplets that become heavy enough to fall."),
        ("List three benefits of reading.",
         "1. Improves vocabulary and language skills\n2. Enhances critical thinking\n3. Reduces stress"),
        ("What is the speed of light?",
         "The speed of light in vacuum is approximately 299,792,458 meters per second, or about 186,000 miles per second."),
        ("Describe the water cycle.",
         "The water cycle involves evaporation of surface water, condensation into clouds, and precipitation back to Earth's surface."),
        ("What is a variable in programming?",
         "A variable is a named storage location in memory that holds a value which can be changed during program execution."),
        ("How do solar panels work?",
         "Solar panels convert sunlight into electricity using photovoltaic cells made of semiconductor materials like silicon."),
        ("What is DNA?",
         "DNA is a molecule that carries genetic instructions for the development, functioning, and reproduction of all living organisms."),
    ]

    examples = []
    for i in range(n_samples):
        instr, resp = tasks[i % len(tasks)]
        text = templates[0].format(instr=instr, resp=resp)
        examples.append({"text": text})

    return Dataset.from_list(examples)


# ═══════════════════════════════════════════════════════════════════════
# 2. MODEL PREPARATION STRATEGIES
# ═══════════════════════════════════════════════════════════════════════

def prepare_model_full_ft(model_name: str):
    """Full fine-tuning — all parameters trainable."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # All params already trainable by default
    return model, "Full Fine-Tuning"


def prepare_model_lora(model_name: str, rank: int = 16, alpha: int = 32):
    """LoRA — low-rank adapter on attention."""
    if not PEFT_AVAILABLE:
        return None, "LoRA (PEFT not available)"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["c_attn"],  # GPT-2 specific
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    return model, f"LoRA (r={rank})"


def prepare_model_prefix(model_name: str, num_virtual_tokens: int = 20):
    """Prefix Tuning — learnable prefix at each layer."""
    if not PEFT_AVAILABLE:
        return None, "Prefix Tuning (PEFT not available)"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=num_virtual_tokens,
    )
    model = get_peft_model(model, config)
    return model, f"Prefix Tuning ({num_virtual_tokens} tokens)"


def prepare_model_prompt(model_name: str, num_virtual_tokens: int = 20):
    """Prompt Tuning — soft tokens at input only."""
    if not PEFT_AVAILABLE:
        return None, "Prompt Tuning (PEFT not available)"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=num_virtual_tokens,
        prompt_tuning_init=PromptTuningInit.RANDOM,
    )
    model = get_peft_model(model, config)
    return model, f"Prompt Tuning ({num_virtual_tokens} tokens)"


def prepare_model_ia3(model_name: str):
    """IA³ — learned rescaling vectors."""
    if not PEFT_AVAILABLE:
        return None, "IA³ (PEFT not available)"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    config = IA3Config(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn", "c_proj"],
        feedforward_modules=["c_proj"],
    )
    model = get_peft_model(model, config)
    return model, "IA³"


def prepare_model_bitfit(model_name: str):
    """BitFit — train only bias terms + LayerNorm."""
    model = AutoModelForCausalLM.from_pretrained(model_name)

    for name, param in model.named_parameters():
        if "bias" in name or "ln_" in name or "LayerNorm" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model, "BitFit"


# ═══════════════════════════════════════════════════════════════════════
# 3. COMPARISON ENGINE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ComparisonResult:
    """Results from one method's training run."""
    method_name: str
    total_params: int
    trainable_params: int
    trainable_pct: float
    train_loss: float
    eval_loss: float
    eval_perplexity: float
    train_time_seconds: float
    checkpoint_size_mb: float = 0.0
    peak_memory_gb: float = 0.0


def run_single_method(
    model,
    method_name: str,
    tokenizer,
    train_dataset,
    eval_dataset,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    output_dir: str = "./comparison_temp",
) -> ComparisonResult:
    """Train a single model configuration and return metrics."""
    import math

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = trainable / total_params * 100

    print(f"\n  Method: {method_name}")
    print(f"  Trainable: {trainable:,} / {total_params:,} ({trainable_pct:.4f}%)")

    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{method_name.replace(' ', '_')}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="no",  # Don't save to speed up comparison
        report_to="none",
        seed=42,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    start_time = time.time()
    train_result = trainer.train()
    train_time = time.time() - start_time

    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics["eval_loss"]
    perplexity = math.exp(eval_loss)

    # Peak memory (if CUDA)
    peak_mem = 0.0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.reset_peak_memory_stats()

    return ComparisonResult(
        method_name=method_name,
        total_params=total_params,
        trainable_params=trainable,
        trainable_pct=trainable_pct,
        train_loss=train_result.training_loss,
        eval_loss=eval_loss,
        eval_perplexity=perplexity,
        train_time_seconds=train_time,
        peak_memory_gb=peak_mem,
    )


def run_comparison(
    model_name: str = "distilgpt2",
    n_samples: int = 100,
    num_epochs: int = 2,
    batch_size: int = 8,
    methods: Optional[List[str]] = None,
) -> List[ComparisonResult]:
    """
    Run head-to-head comparison of PEFT methods.

    COMPARISON METHODOLOGY:
    ───────────────────────
    - Same base model for all methods
    - Same dataset, same split
    - Same number of epochs
    - Same batch size
    - Only training method differs
    """
    print(f"\n{'═' * 70}")
    print("PEFT METHOD COMPARISON")
    print(f"{'═' * 70}")
    print(f"  Model: {model_name}")
    print(f"  Dataset size: {n_samples}")
    print(f"  Epochs: {num_epochs}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create and tokenize dataset
    dataset = create_comparison_dataset(n_samples)

    def tokenize(example):
        tokens = tokenizer(example["text"], truncation=True, max_length=256, padding="max_length")
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(tokenize, remove_columns=["text"])
    split = tokenized.train_test_split(test_size=0.1, seed=42)

    # Define methods to compare
    if methods is None:
        methods = ["full_ft", "lora", "prefix", "prompt", "ia3", "bitfit"]

    method_builders = {
        "full_ft": lambda: prepare_model_full_ft(model_name),
        "lora": lambda: prepare_model_lora(model_name, rank=16),
        "prefix": lambda: prepare_model_prefix(model_name, num_virtual_tokens=20),
        "prompt": lambda: prepare_model_prompt(model_name, num_virtual_tokens=20),
        "ia3": lambda: prepare_model_ia3(model_name),
        "bitfit": lambda: prepare_model_bitfit(model_name),
    }

    results = []
    for method_key in methods:
        if method_key not in method_builders:
            print(f"\n  Skipping unknown method: {method_key}")
            continue

        model, name = method_builders[method_key]()
        if model is None:
            print(f"\n  Skipping {name} (not available)")
            continue

        result = run_single_method(
            model=model,
            method_name=name,
            tokenizer=tokenizer,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
        results.append(result)

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print comparison table
    print_comparison_table(results)

    return results


# ═══════════════════════════════════════════════════════════════════════
# 4. RESULTS DISPLAY
# ═══════════════════════════════════════════════════════════════════════

def print_comparison_table(results: List[ComparisonResult]):
    """Print formatted comparison table."""
    print(f"\n\n{'═' * 95}")
    print("COMPARISON RESULTS")
    print(f"{'═' * 95}")

    header = (f"{'Method':<25} {'Trainable':>12} {'%':>8} "
              f"{'Train Loss':>11} {'Eval Loss':>10} {'PPL':>8} {'Time(s)':>8}")
    print(f"\n{header}")
    print("─" * 95)

    for r in results:
        print(f"{r.method_name:<25} "
              f"{r.trainable_params:>12,} "
              f"{r.trainable_pct:>7.3f}% "
              f"{r.train_loss:>11.4f} "
              f"{r.eval_loss:>10.4f} "
              f"{r.eval_perplexity:>8.1f} "
              f"{r.train_time_seconds:>8.1f}")

    # Relative to full FT
    full_ft = next((r for r in results if "Full" in r.method_name), None)
    if full_ft:
        print(f"\n{'─' * 95}")
        print("RELATIVE TO FULL FINE-TUNING:")
        print(f"{'─' * 95}")
        for r in results:
            if r == full_ft:
                continue
            param_savings = (1 - r.trainable_params / full_ft.trainable_params) * 100
            time_savings = (1 - r.train_time_seconds / full_ft.train_time_seconds) * 100
            ppl_diff = r.eval_perplexity - full_ft.eval_perplexity
            print(f"  {r.method_name:<25} "
                  f"Param savings: {param_savings:>6.2f}% | "
                  f"Time savings: {time_savings:>6.1f}% | "
                  f"PPL diff: {ppl_diff:>+7.1f}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 70)
    print("  PEFT METHOD HEAD-TO-HEAD COMPARISON")
    print("═" * 70)
    print()
    print("Running all methods on the same model and dataset...")
    print("Model: distilgpt2 (82M params)")
    print()

    # Run comparison with a small dataset for demo
    results = run_comparison(
        model_name="distilgpt2",
        n_samples=100,
        num_epochs=2,
        batch_size=8,
        methods=["full_ft", "lora", "bitfit"],  # Start with these (no peft needed for bitfit)
    )

    print(f"\n{'═' * 70}")
    print("  COMPARISON COMPLETE!")
    print(f"{'═' * 70}")
    print()
    print("KEY TAKEAWAYS:")
    print("  • LoRA achieves near-identical quality with <1% of parameters")
    print("  • BitFit is the simplest but least expressive method")
    print("  • Training time savings come from reduced gradient computation")
    print("  • Memory savings come from reduced optimizer states")
    print("  • All PEFT methods preserve the base model's general knowledge")
