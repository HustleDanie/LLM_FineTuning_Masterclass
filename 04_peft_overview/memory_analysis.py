"""
═══════════════════════════════════════════════════════════════════════════
MEMORY ANALYSIS — Comparing Memory Requirements Across Methods
═══════════════════════════════════════════════════════════════════════════

Understanding memory usage is CRITICAL for choosing the right PEFT method.
This module provides tools to estimate and compare memory requirements.

GPU MEMORY DURING TRAINING:
───────────────────────────
Total Memory = Model Weights + Optimizer States + Gradients + Activations

For AdamW optimizer (most common):
- Model weights:     N parameters × bytes_per_param
- Optimizer states:  2 × N × bytes_per_param (momentum + variance)
- Gradients:         N × bytes_per_param
- Activations:       Depends on batch size, seq length, model architecture

So for fp32: Training memory ≈ 4N × 4 bytes = 16N bytes
For fp16 (mixed precision):     ≈ 2N + 4N + 2N ≈ 8N (approximate)
For bf16:                       ≈ similar to fp16
"""

import torch
from dataclasses import dataclass
from typing import Dict, Optional, List


# ═══════════════════════════════════════════════════════════════════════
# 1. MODEL SIZE REFERENCE
# ═══════════════════════════════════════════════════════════════════════

MODEL_SPECS = {
    "distilgpt2": {
        "params_M": 82,        # Millions of parameters
        "d_model": 768,
        "n_layers": 6,
        "n_heads": 12,
        "d_ff": 3072,
    },
    "gpt2": {
        "params_M": 124,
        "d_model": 768,
        "n_layers": 12,
        "n_heads": 12,
        "d_ff": 3072,
    },
    "gpt2-medium": {
        "params_M": 355,
        "d_model": 1024,
        "n_layers": 24,
        "n_heads": 16,
        "d_ff": 4096,
    },
    "gpt2-large": {
        "params_M": 774,
        "d_model": 1280,
        "n_layers": 36,
        "n_heads": 20,
        "d_ff": 5120,
    },
    "gpt2-xl": {
        "params_M": 1558,
        "d_model": 1600,
        "n_layers": 48,
        "n_heads": 25,
        "d_ff": 6400,
    },
    "llama-7b": {
        "params_M": 6738,
        "d_model": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "d_ff": 11008,
    },
    "llama-13b": {
        "params_M": 13015,
        "d_model": 5120,
        "n_layers": 40,
        "n_heads": 40,
        "d_ff": 13824,
    },
    "llama-70b": {
        "params_M": 64862,
        "d_model": 8192,
        "n_layers": 80,
        "n_heads": 64,
        "d_ff": 28672,
    },
}


# ═══════════════════════════════════════════════════════════════════════
# 2. MEMORY ESTIMATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MemoryEstimate:
    """Estimated GPU memory for a training configuration."""
    model_weights_gb: float
    optimizer_states_gb: float
    gradients_gb: float
    activations_gb: float
    total_gb: float
    trainable_params_M: float
    total_params_M: float
    trainable_pct: float


def estimate_memory(
    total_params_M: float,
    trainable_params_M: float,
    d_model: int = 768,
    n_layers: int = 12,
    batch_size: int = 4,
    seq_length: int = 512,
    precision: str = "fp16",
    optimizer: str = "adamw",
    gradient_checkpointing: bool = False,
) -> MemoryEstimate:
    """
    Estimate GPU memory requirements for training.

    MEMORY BREAKDOWN:
    ─────────────────
    1. Model Weights: All parameters (including frozen) must be in GPU memory
       - fp32: 4 bytes/param
       - fp16/bf16: 2 bytes/param
       - int8: 1 byte/param
       - nf4 (QLoRA): 0.5 bytes/param

    2. Optimizer States: Only for TRAINABLE parameters
       - AdamW: 2 states × 4 bytes/param = 8 bytes/trainable_param
       - SGD: 4 bytes/trainable_param (just momentum)
       - Adafactor: ~4 bytes/trainable_param (factored)

    3. Gradients: Only for TRAINABLE parameters
       - Same precision as model: 2-4 bytes/trainable_param

    4. Activations: Depends on batch_size × seq_len × model_size
       - Without checkpointing: O(n_layers × batch × seq × d_model)
       - With checkpointing: O(sqrt(n_layers) × batch × seq × d_model)
    """
    total_params = total_params_M * 1e6
    trainable_params = trainable_params_M * 1e6

    # Bytes per parameter based on precision
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "nf4": 0.5,
    }
    bpp = bytes_per_param.get(precision, 2)

    # 1. Model weights (ALL params need to be loaded)
    model_bytes = total_params * bpp

    # 2. Optimizer states (only trainable params, always fp32)
    optimizer_multiplier = {
        "adamw": 8,       # 2 states × 4 bytes
        "adam": 8,
        "sgd": 4,         # 1 state × 4 bytes
        "adafactor": 4,   # Approximately
    }
    opt_mult = optimizer_multiplier.get(optimizer, 8)
    optimizer_bytes = trainable_params * opt_mult

    # 3. Gradients (trainable params only)
    gradient_bytes = trainable_params * bpp

    # If mixed precision, gradients stored in fp16 but master weights in fp32
    if precision == "fp16":
        # Need fp32 copy of trainable params for optimizer
        optimizer_bytes += trainable_params * 4  # fp32 master weights

    # 4. Activations (rough estimate)
    # Each layer stores: batch × seq × d_model × bytes (attention + FFN)
    # Factor of ~10-12 for typical transformer layer activations
    activation_factor = 10
    if gradient_checkpointing:
        # Checkpointing reduces activation memory by ~sqrt(n_layers)/n_layers
        effective_layers = max(1, int(n_layers ** 0.5))
    else:
        effective_layers = n_layers

    activation_bytes = (
        effective_layers * batch_size * seq_length * d_model * bpp * activation_factor
    )

    # Convert to GB
    to_gb = lambda b: b / (1024 ** 3)

    total_bytes = model_bytes + optimizer_bytes + gradient_bytes + activation_bytes

    return MemoryEstimate(
        model_weights_gb=to_gb(model_bytes),
        optimizer_states_gb=to_gb(optimizer_bytes),
        gradients_gb=to_gb(gradient_bytes),
        activations_gb=to_gb(activation_bytes),
        total_gb=to_gb(total_bytes),
        trainable_params_M=trainable_params_M,
        total_params_M=total_params_M,
        trainable_pct=trainable_params_M / total_params_M * 100,
    )


# ═══════════════════════════════════════════════════════════════════════
# 3. PEFT METHOD MEMORY COMPARISON
# ═══════════════════════════════════════════════════════════════════════

def compute_peft_params(
    model_name: str,
    method: str = "lora",
    **kwargs,
) -> float:
    """
    Compute trainable parameters (in millions) for a PEFT method.

    Returns trainable_params_M for the given model and method.
    """
    spec = MODEL_SPECS.get(model_name, MODEL_SPECS["gpt2"])
    d = spec["d_model"]
    n = spec["n_layers"]
    d_ff = spec["d_ff"]

    if method == "full_ft":
        return spec["params_M"]

    elif method == "lora":
        rank = kwargs.get("rank", 16)
        # Default targets: Q, V projections
        n_targets = kwargs.get("n_targets", 2)  # Q, V
        # Each target: d×r + r×d parameters for A and B
        per_layer = n_targets * 2 * d * rank
        return n * per_layer / 1e6

    elif method == "lora_all":
        rank = kwargs.get("rank", 16)
        # All linear layers: Q, K, V, O, FFN_up, FFN_down = 6
        n_targets = 6
        per_layer = n_targets * 2 * d * rank
        return n * per_layer / 1e6

    elif method == "qlora":
        # Same trainable params as LoRA, but base model is quantized
        return compute_peft_params(model_name, "lora", **kwargs)

    elif method == "adapters":
        bottleneck = kwargs.get("bottleneck", 64)
        # 2 adapters per layer (after attn, after FFN)
        per_adapter = d * bottleneck + bottleneck * d + bottleneck + d
        per_layer = 2 * per_adapter
        return n * per_layer / 1e6

    elif method == "prefix_tuning":
        prefix_len = kwargs.get("prefix_len", 20)
        # prefix K and V at every layer
        per_layer = 2 * prefix_len * d
        return n * per_layer / 1e6

    elif method == "prompt_tuning":
        n_tokens = kwargs.get("n_tokens", 20)
        # Only input embeddings
        return n_tokens * d / 1e6

    elif method == "ia3":
        # 3 vectors per layer (k, v, ff)
        per_layer = 3 * d
        return n * per_layer / 1e6

    elif method == "bitfit":
        # Biases only (approximate)
        per_layer = 6 * d + 4 * d  # attn biases + FFN biases + LayerNorm
        return n * per_layer / 1e6

    else:
        raise ValueError(f"Unknown method: {method}")


def compare_methods_memory(
    model_name: str = "llama-7b",
    batch_size: int = 4,
    seq_length: int = 512,
    gradient_checkpointing: bool = True,
) -> Dict:
    """
    Compare memory requirements across all PEFT methods for a given model.
    """
    spec = MODEL_SPECS.get(model_name, MODEL_SPECS["gpt2"])
    total_params_M = spec["params_M"]

    methods = {
        "Full FT (fp32)": {"method": "full_ft", "precision": "fp32"},
        "Full FT (fp16)": {"method": "full_ft", "precision": "fp16"},
        "LoRA (r=16, Q+V)": {"method": "lora", "precision": "fp16", "rank": 16},
        "LoRA (r=16, all)": {"method": "lora_all", "precision": "fp16", "rank": 16},
        "QLoRA (r=16)": {"method": "qlora", "precision": "nf4", "rank": 16},
        "Adapters (r=64)": {"method": "adapters", "precision": "fp16", "bottleneck": 64},
        "Prefix Tuning (20)": {"method": "prefix_tuning", "precision": "fp16", "prefix_len": 20},
        "Prompt Tuning (20)": {"method": "prompt_tuning", "precision": "fp16", "n_tokens": 20},
        "IA³": {"method": "ia3", "precision": "fp16"},
        "BitFit": {"method": "bitfit", "precision": "fp16"},
    }

    print(f"\n{'═' * 80}")
    print(f"MEMORY COMPARISON: {model_name} ({total_params_M}M params)")
    print(f"Batch size: {batch_size}, Seq length: {seq_length}, "
          f"Gradient checkpointing: {gradient_checkpointing}")
    print(f"{'═' * 80}")

    header = f"{'Method':<25} {'Train Params':>12} {'%':>8} {'Weights':>8} {'Optim':>8} {'Grad':>6} {'Act':>6} {'TOTAL':>8}"
    print(f"\n{header}")
    print("─" * len(header))

    results = {}
    for name, cfg in methods.items():
        method = cfg["method"]
        precision = cfg["precision"]

        # Compute trainable params
        if method == "full_ft":
            trainable_M = total_params_M
        else:
            kwargs = {k: v for k, v in cfg.items() if k not in ("method", "precision")}
            trainable_M = compute_peft_params(model_name, method, **kwargs)

        # Estimate memory
        est = estimate_memory(
            total_params_M=total_params_M,
            trainable_params_M=trainable_M,
            d_model=spec["d_model"],
            n_layers=spec["n_layers"],
            batch_size=batch_size,
            seq_length=seq_length,
            precision=precision,
            gradient_checkpointing=gradient_checkpointing,
        )

        results[name] = est

        print(f"{name:<25} {trainable_M:>10.2f}M {est.trainable_pct:>7.3f}% "
              f"{est.model_weights_gb:>7.1f}G {est.optimizer_states_gb:>7.1f}G "
              f"{est.gradients_gb:>5.1f}G {est.activations_gb:>5.1f}G "
              f"{est.total_gb:>7.1f}G")

    return results


# ═══════════════════════════════════════════════════════════════════════
# 4. GPU COMPATIBILITY CHECK
# ═══════════════════════════════════════════════════════════════════════

COMMON_GPUS = {
    "RTX 3060": 12,
    "RTX 3070": 8,
    "RTX 3080": 10,
    "RTX 3090": 24,
    "RTX 4060": 8,
    "RTX 4070": 12,
    "RTX 4080": 16,
    "RTX 4090": 24,
    "A10G": 24,
    "A100-40GB": 40,
    "A100-80GB": 80,
    "H100-80GB": 80,
    "V100-16GB": 16,
    "V100-32GB": 32,
    "T4": 16,
}


def check_gpu_compatibility(
    model_name: str = "llama-7b",
    batch_size: int = 4,
    seq_length: int = 512,
) -> None:
    """
    Check which GPU + PEFT method combinations are feasible.

    Prints a matrix of model × method × GPU combinations showing
    which configurations fit in memory.
    """
    spec = MODEL_SPECS.get(model_name)
    if not spec:
        print(f"Model '{model_name}' not in database")
        return

    total_params_M = spec["params_M"]

    methods_to_check = [
        ("Full FT (fp16)", "full_ft", "fp16", {}),
        ("LoRA (r=16)", "lora", "fp16", {"rank": 16}),
        ("QLoRA (r=16)", "qlora", "nf4", {"rank": 16}),
        ("Adapters", "adapters", "fp16", {"bottleneck": 64}),
        ("Prompt Tuning", "prompt_tuning", "fp16", {}),
        ("IA³", "ia3", "fp16", {}),
    ]

    print(f"\n{'═' * 90}")
    print(f"GPU COMPATIBILITY: {model_name} ({total_params_M}M params)")
    print(f"Batch: {batch_size}, Seq: {seq_length}, Gradient Checkpointing: ON")
    print(f"{'═' * 90}")

    # Header
    gpu_names = ["T4", "RTX 3090", "RTX 4090", "A100-40GB", "A100-80GB", "H100-80GB"]
    header = f"{'Method':<20}" + "".join(f"{g:>12}" for g in gpu_names)
    print(f"\n{header}")
    print("─" * len(header))

    for name, method, precision, kwargs in methods_to_check:
        if method == "full_ft":
            trainable_M = total_params_M
        else:
            trainable_M = compute_peft_params(model_name, method, **kwargs)

        est = estimate_memory(
            total_params_M=total_params_M,
            trainable_params_M=trainable_M,
            d_model=spec["d_model"],
            n_layers=spec["n_layers"],
            batch_size=batch_size,
            seq_length=seq_length,
            precision=precision,
            gradient_checkpointing=True,
        )

        row = f"{name:<20}"
        for gpu in gpu_names:
            vram = COMMON_GPUS[gpu]
            if est.total_gb <= vram * 0.9:  # 90% threshold for safety
                row += f"{'✓ ' + f'{est.total_gb:.0f}G':>12}"
            else:
                row += f"{'✗ ' + f'{est.total_gb:.0f}G':>12}"
        print(row)


# ═══════════════════════════════════════════════════════════════════════
# 5. LIVE MEMORY MEASUREMENT
# ═══════════════════════════════════════════════════════════════════════

def measure_actual_memory(model, device: str = "cuda") -> Dict:
    """
    Measure actual GPU memory usage of a loaded model.
    Requires CUDA GPU.
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available. Run on GPU for actual measurements."}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model = model.to(device)
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "model_params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "trainable_params_M": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ) / 1e6,
    }


def measure_training_memory(
    model,
    tokenizer,
    sample_text: str = "Hello world, this is a test of memory usage.",
    batch_size: int = 4,
    seq_length: int = 128,
    device: str = "cuda",
) -> Dict:
    """
    Measure actual memory during a forward + backward pass.
    Requires CUDA GPU.
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    model = model.to(device)
    model.train()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    mem_after_model = torch.cuda.memory_allocated() / 1024**3

    # Create dummy batch
    inputs = tokenizer(
        [sample_text] * batch_size,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_length,
    ).to(device)
    inputs["labels"] = inputs["input_ids"].clone()

    mem_after_data = torch.cuda.memory_allocated() / 1024**3

    # Forward pass
    outputs = model(**inputs)
    loss = outputs.loss
    mem_after_forward = torch.cuda.memory_allocated() / 1024**3

    # Backward pass
    loss.backward()
    mem_after_backward = torch.cuda.memory_allocated() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3

    model.zero_grad()
    torch.cuda.empty_cache()

    return {
        "model_loaded_gb": mem_after_model,
        "with_data_gb": mem_after_data,
        "after_forward_gb": mem_after_forward,
        "after_backward_gb": mem_after_backward,
        "peak_gb": peak,
        "activations_gb": mem_after_forward - mem_after_data,
        "gradients_gb": mem_after_backward - mem_after_forward,
    }


# ═══════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("PEFT MEMORY ANALYSIS")
    print("=" * 80)

    # Compare methods for different model sizes
    for model in ["gpt2", "llama-7b", "llama-13b", "llama-70b"]:
        compare_methods_memory(model, batch_size=4, seq_length=512)

    # GPU compatibility
    print("\n\n")
    for model in ["llama-7b", "llama-13b", "llama-70b"]:
        check_gpu_compatibility(model, batch_size=4, seq_length=512)

    # Single estimation example
    print(f"\n\n{'═' * 70}")
    print("DETAILED ESTIMATE: LLaMA-7B with LoRA (rank=16)")
    print(f"{'═' * 70}")

    est = estimate_memory(
        total_params_M=6738,
        trainable_params_M=compute_peft_params("llama-7b", "lora", rank=16),
        d_model=4096,
        n_layers=32,
        batch_size=4,
        seq_length=512,
        precision="fp16",
        gradient_checkpointing=True,
    )

    print(f"\n  Total params:     {est.total_params_M:,.0f}M")
    print(f"  Trainable params: {est.trainable_params_M:,.2f}M ({est.trainable_pct:.3f}%)")
    print(f"  Model weights:    {est.model_weights_gb:.1f} GB")
    print(f"  Optimizer states: {est.optimizer_states_gb:.1f} GB")
    print(f"  Gradients:        {est.gradients_gb:.2f} GB")
    print(f"  Activations:      {est.activations_gb:.1f} GB")
    print(f"  ─────────────────────────────")
    print(f"  TOTAL:            {est.total_gb:.1f} GB")
