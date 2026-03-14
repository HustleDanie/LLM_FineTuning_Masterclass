"""
═══════════════════════════════════════════════════════════════════════════
PEFT METHODS — Detailed Explanation of Every PEFT Technique
═══════════════════════════════════════════════════════════════════════════

This module provides a comprehensive overview of all major PEFT methods,
including their architecture, mathematical formulation, and intuition.

CORE PRINCIPLE:
───────────────
LLM weight matrices are OVER-PARAMETERIZED for downstream tasks.
The "intrinsic dimensionality" of the task-specific adaptation is
much lower than the full parameter space. PEFT exploits this by
constraining the update to a low-dimensional subspace.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════
# 1. METHOD CATALOG — All PEFT Methods Explained
# ═══════════════════════════════════════════════════════════════════════

PEFT_METHOD_CATALOG = {
    # ─── REPARAMETERIZATION METHODS ───────────────────────────────
    "lora": {
        "full_name": "Low-Rank Adaptation (LoRA)",
        "paper": "Hu et al., 2021 — 'LoRA: Low-Rank Adaptation of Large Language Models'",
        "category": "reparameterization",
        "description": """
        LoRA freezes all pretrained weights and injects trainable low-rank
        decomposition matrices into transformer layers.

        Instead of updating W directly:
            W_new = W + ΔW

        LoRA decomposes the update:
            ΔW = B × A    where B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)

        Forward pass:
            h = W·x + (B·A)·x · (α/r)

        Key hyperparameters:
            r (rank): Typically 4-64. Lower = fewer params but less capacity
            α (alpha): Scaling factor, usually 2×r. Controls update magnitude
            target_modules: Which weight matrices to adapt (q, k, v, o, mlp)
        """,
        "trainable_percent": "0.1-1%",
        "inference_overhead": "None (weights can be merged: W_merged = W + B·A)",
        "pros": [
            "No inference latency (merge weights)",
            "Very memory efficient",
            "Can swap adapters at runtime",
            "Works across all model sizes",
        ],
        "cons": [
            "Rank limits expressiveness",
            "Not optimal for very different target domains",
            "Choosing target modules requires knowledge",
        ],
    },

    "qlora": {
        "full_name": "Quantized LoRA (QLoRA)",
        "paper": "Dettmers et al., 2023 — 'QLoRA: Efficient Finetuning of Quantized LLMs'",
        "category": "reparameterization",
        "description": """
        QLoRA combines LoRA with 4-bit quantization of the base model:

        1. Base model weights quantized to 4-bit (NF4 format)
        2. LoRA adapters remain in fp16/bf16
        3. Computation uses double quantization for memory savings

        Key innovations:
        - NormalFloat4 (NF4): Optimal 4-bit data type for normally distributed weights
        - Double Quantization: Quantize the quantization constants too
        - Paged Optimizers: Use CPU memory for optimizer states that don't fit in GPU

        This enables fine-tuning a 65B model on a single 48GB GPU!
        """,
        "trainable_percent": "0.1-1%",
        "inference_overhead": "Slight (quantized base model)",
        "pros": [
            "Massive memory savings (fit 65B on one GPU)",
            "Minimal quality loss vs full LoRA",
            "Democratizes large model fine-tuning",
        ],
        "cons": [
            "Slightly slower training (quantize/dequantize)",
            "Quantization can lose some information",
            "Requires bitsandbytes library",
        ],
    },

    "ia3": {
        "full_name": "Infused Adapter by Inhibiting and Amplifying Inner Activations (IA³)",
        "paper": "Liu et al., 2022 — 'Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning'",
        "category": "reparameterization",
        "description": """
        IA³ learns RESCALING VECTORS that multiply (element-wise) the
        keys, values, and feedforward activations.

        For attention:
            k = l_k ⊙ (W_k · x)    — rescale keys
            v = l_v ⊙ (W_v · x)    — rescale values

        For feedforward:
            h = l_ff ⊙ (W_ff · x)  — rescale FFN output

        Where l_k, l_v, l_ff are learned vectors (not matrices!).
        This means IA³ adds negligible parameters.
        """,
        "trainable_percent": "<0.01%",
        "inference_overhead": "Negligible (element-wise multiply)",
        "pros": [
            "Fewest trainable parameters of any method",
            "Very fast training",
            "Good few-shot performance",
        ],
        "cons": [
            "Limited capacity (just scaling)",
            "May not work well for complex task shifts",
        ],
    },

    # ─── ADDITIVE METHODS ─────────────────────────────────────────

    "adapters": {
        "full_name": "Bottleneck Adapters",
        "paper": "Houlsby et al., 2019 — 'Parameter-Efficient Transfer Learning for NLP'",
        "category": "additive",
        "description": """
        Adapters insert small bottleneck layers INSIDE each transformer block.

        Architecture of one adapter:
            x → LayerNorm → Down-project (d→r) → ReLU → Up-project (r→d) → + x (residual)

        Placement options:
            - Serial: After self-attention AND after FFN (Houlsby et al.)
            - Parallel: In parallel with self-attention (He et al.)

        The bottleneck dimension r controls the parameter budget.
        Typical r values: 8, 16, 32, 64.
        """,
        "trainable_percent": "1-5%",
        "inference_overhead": "Added latency from extra layers",
        "pros": [
            "Well-studied, robust",
            "Modular (can compose adapters for different tasks)",
            "Good at task-specific adaptation",
        ],
        "cons": [
            "Adds inference latency (cannot merge like LoRA)",
            "More parameters than LoRA",
            "Sequential bottleneck may limit throughput",
        ],
    },

    "prefix_tuning": {
        "full_name": "Prefix Tuning",
        "paper": "Li & Liang, 2021 — 'Prefix-Tuning: Optimizing Continuous Prompts for Generation'",
        "category": "additive",
        "description": """
        Prefix Tuning prepends learnable continuous vectors to the keys and
        values at EVERY attention layer.

        For layer l:
            K_l = [P_k^l ; K_l]    — prepend prefix keys
            V_l = [P_v^l ; V_l]    — prepend prefix values

        where P_k^l, P_v^l ∈ R^(prefix_len × d_head)

        Since training directly is unstable, uses reparameterization:
            P = MLP(P')    — P' is a smaller embedding, MLP projects up

        The prefix acts as "virtual tokens" that steer the model's attention.
        """,
        "trainable_percent": "0.1-1%",
        "inference_overhead": "Prefix tokens increase context length",
        "pros": [
            "Very few parameters",
            "Doesn't modify model architecture",
            "Can mix multiple prefixes for multi-task",
        ],
        "cons": [
            "Reduces effective context length",
            "Training can be unstable",
            "Performance sensitive to prefix length",
        ],
    },

    "prompt_tuning": {
        "full_name": "Prompt Tuning",
        "paper": "Lester et al., 2021 — 'The Power of Scale for Parameter-Efficient Prompt Tuning'",
        "category": "additive",
        "description": """
        Prompt Tuning is a SIMPLIFIED version of Prefix Tuning:
        Learnable soft tokens are prepended to the INPUT EMBEDDING only
        (not at every layer).

        x_new = [P ; x]    where P ∈ R^(n_tokens × d_model)

        Key finding: As model size increases, prompt tuning approaches
        full fine-tuning performance. At 10B+ parameters, the gap is minimal.

        Typically uses 10-100 virtual tokens.
        """,
        "trainable_percent": "<0.1%",
        "inference_overhead": "Minimal (a few extra tokens)",
        "pros": [
            "Simplest PEFT method",
            "Extremely few parameters",
            "Scales well with model size",
            "Easy to implement",
        ],
        "cons": [
            "Less effective for small models",
            "Limited expressiveness (only input layer)",
            "Performance can lag behind LoRA/adapters",
        ],
    },

    "p_tuning_v2": {
        "full_name": "P-Tuning v2",
        "paper": "Liu et al., 2021 — 'P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks'",
        "category": "additive",
        "description": """
        P-Tuning v2 is essentially Prefix Tuning adapted for NLU tasks.
        It adds learnable continuous prompts at EVERY layer (like prefix tuning)
        but also:
        1. Removes the reparameterization (MLP) trick
        2. Adds prompts to different positions (not just prefix)
        3. Uses task-specific heads

        Key finding: Deep prompt tuning (all layers) is critical for
        matching fine-tuning on smaller models.
        """,
        "trainable_percent": "0.1-1%",
        "inference_overhead": "Prompt tokens at every layer",
        "pros": [
            "Works across all model scales (unlike prompt tuning)",
            "Comparable to fine-tuning on NLU benchmarks",
            "More stable than prefix tuning",
        ],
        "cons": [
            "More parameters than prompt tuning",
            "Implementation more complex",
        ],
    },

    # ─── SELECTIVE METHODS ────────────────────────────────────────

    "bitfit": {
        "full_name": "BitFit (Bias-terms Fine-Tuning)",
        "paper": "Ben Zaken et al., 2022 — 'BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models'",
        "category": "selective",
        "description": """
        BitFit trains ONLY the bias terms in the model, freezing all
        weight matrices.

        For each layer:
            Freeze: W_q, W_k, W_v, W_o, W_ff1, W_ff2
            Train:  b_q, b_k, b_v, b_o, b_ff1, b_ff2, LayerNorm params

        Surprisingly effective! Bias terms are <0.1% of parameters but
        carry significant task-specific information.

        Works best for NLU tasks; less effective for generation tasks.
        """,
        "trainable_percent": "~0.1%",
        "inference_overhead": "None",
        "pros": [
            "Simplest to implement",
            "No architecture changes",
            "No inference overhead",
            "Good theoretical grounding",
        ],
        "cons": [
            "Limited capacity",
            "Weaker for generation tasks",
            "Not competitive with LoRA for most tasks",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════
# 2. ARCHITECTURE VISUALIZATIONS (ASCII)
# ═══════════════════════════════════════════════════════════════════════

def print_transformer_block(method: str = "full_ft"):
    """
    Print ASCII architecture diagram showing where each PEFT method
    modifies the transformer block.
    """
    diagrams = {
        "full_ft": """
    ┌─────────────────────────────────────────────┐
    │         FULL FINE-TUNING                     │
    │         (All parameters updated)             │
    │                                              │
    │  Input ──→ [Self-Attention] ──→ [Add+Norm]  │
    │             ████ ALL UPDATED ████            │
    │                     │                        │
    │             [Feed-Forward]   ──→ [Add+Norm]  │
    │             ████ ALL UPDATED ████            │
    │                     │                        │
    │                  Output                      │
    └─────────────────────────────────────────────┘
    Memory: 100% | Trainable params: 100%
        """,

        "lora": """
    ┌─────────────────────────────────────────────┐
    │         LoRA                                 │
    │         (Low-rank updates to Q, V projections│
    │                                              │
    │  Input ──→ [Self-Attention]─→ [Add+Norm]    │
    │             W_q ── frozen                    │
    │              └─ B_q·A_q ── TRAINED (rank r)  │
    │             W_v ── frozen                    │
    │              └─ B_v·A_v ── TRAINED (rank r)  │
    │                     │                        │
    │             [Feed-Forward] ──→ [Add+Norm]    │
    │             ████ FROZEN ████                 │
    │                     │                        │
    │                  Output                      │
    │                                              │
    │  h = W·x + (B·A)·x · (α/r)                 │
    └─────────────────────────────────────────────┘
    Memory: ~30% of Full FT | Trainable: 0.1-1%
        """,

        "adapters": """
    ┌─────────────────────────────────────────────┐
    │         ADAPTERS                             │
    │         (Bottleneck layers inserted)         │
    │                                              │
    │  Input ──→ [Self-Attention] ──→ [Add+Norm]  │
    │             ████ FROZEN ████                 │
    │                     │                        │
    │             ┌───────────────┐                │
    │             │ ADAPTER (NEW) │                │
    │             │ Down: d → r   │  ◄── TRAINED  │
    │             │ ReLU          │                │
    │             │ Up:   r → d   │                │
    │             │ + Residual    │                │
    │             └───────────────┘                │
    │                     │                        │
    │             [Feed-Forward] ──→ [Add+Norm]    │
    │             ████ FROZEN ████                 │
    │                     │                        │
    │             ┌───────────────┐                │
    │             │ ADAPTER (NEW) │  ◄── TRAINED  │
    │             └───────────────┘                │
    │                  Output                      │
    └─────────────────────────────────────────────┘
    Memory: ~40% of Full FT | Trainable: 1-5%
        """,

        "prefix_tuning": """
    ┌─────────────────────────────────────────────┐
    │         PREFIX TUNING                        │
    │         (Learnable prefixes at every layer)  │
    │                                              │
    │  Layer 1:                                    │
    │    K = [P_k¹ | K₁]     ◄── P_k¹ TRAINED    │
    │    V = [P_v¹ | V₁]     ◄── P_v¹ TRAINED    │
    │    [Self-Attention with extended K,V]        │
    │    ████ FROZEN ████                          │
    │                                              │
    │  Layer 2:                                    │
    │    K = [P_k² | K₂]     ◄── P_k² TRAINED    │
    │    V = [P_v² | V₂]     ◄── P_v² TRAINED    │
    │    ...                                       │
    │                                              │
    │  Each layer gets its own prefix vectors      │
    └─────────────────────────────────────────────┘
    Memory: ~35% of Full FT | Trainable: 0.1%
        """,

        "prompt_tuning": """
    ┌─────────────────────────────────────────────┐
    │         PROMPT TUNING                        │
    │         (Soft tokens at input ONLY)          │
    │                                              │
    │  [P₁ P₂ ... Pₙ | x₁ x₂ ... xₘ]           │
    │   ^^^^^^^^^^^    ^^^^^^^^^^^^^^^             │
    │   TRAINABLE      REAL INPUT                  │
    │   (soft tokens)  (frozen embeddings)         │
    │          │                                   │
    │          ▼                                   │
    │   [Transformer Layers — ALL FROZEN]          │
    │          │                                   │
    │          ▼                                   │
    │       Output                                 │
    └─────────────────────────────────────────────┘
    Memory: ~30% of Full FT | Trainable: <0.01%
        """,

        "bitfit": """
    ┌─────────────────────────────────────────────┐
    │         BITFIT                                │
    │         (Only bias terms trained)            │
    │                                              │
    │  [Self-Attention]                            │
    │    W_q · x + b_q    W_q FROZEN, b_q TRAINED │
    │    W_k · x + b_k    W_k FROZEN, b_k TRAINED │
    │    W_v · x + b_v    W_v FROZEN, b_v TRAINED │
    │                                              │
    │  [LayerNorm]         γ, β — TRAINED          │
    │                                              │
    │  [Feed-Forward]                              │
    │    W₁ · x + b₁     W₁ FROZEN, b₁ TRAINED   │
    │    W₂ · x + b₂     W₂ FROZEN, b₂ TRAINED   │
    │                                              │
    │  [LayerNorm]         γ, β — TRAINED          │
    └─────────────────────────────────────────────┘
    Memory: ~30% of Full FT | Trainable: ~0.1%
        """,
    }

    if method in diagrams:
        print(diagrams[method])
    else:
        print(f"No diagram available for '{method}'")


# ═══════════════════════════════════════════════════════════════════════
# 3. MINIMAL IMPLEMENTATIONS (For Understanding)
# ═══════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """
    Minimal LoRA implementation to understand the math.

    Standard Linear: y = Wx + b
    LoRA Linear:     y = Wx + (B·A)x · (α/r) + b

    W is FROZEN. Only A and B are trained.
    A is initialized with random Gaussian, B initialized to zero.
    This means ΔW = B·A starts as zero → no change at initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 8.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen pretrained weight
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features), requires_grad=False
        )
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)

        # LoRA matrices — THESE are trained
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))  # Zero init!

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original path (frozen)
        base_output = nn.functional.linear(x, self.weight, self.bias)

        # LoRA path (trained)
        lora_output = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling

        return base_output + lora_output

    def merge_weights(self) -> nn.Linear:
        """Merge LoRA weights into base weight for inference."""
        merged = nn.Linear(self.weight.shape[1], self.weight.shape[0])
        merged.weight.data = self.weight.data + (self.lora_B @ self.lora_A) * self.scaling
        merged.bias.data = self.bias.data
        return merged

    @property
    def trainable_params(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()

    @property
    def total_params(self) -> int:
        return self.weight.numel() + self.bias.numel() + self.trainable_params


class AdapterLayer(nn.Module):
    """
    Minimal Adapter implementation.

    Architecture: x → LayerNorm → Down-project → ReLU → Up-project → + x
    """

    def __init__(self, d_model: int, bottleneck: int = 64):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.down_project = nn.Linear(d_model, bottleneck)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(bottleneck, d_model)

        # Initialize up_project to near-zero so adapter starts as identity
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual  # Residual connection

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class PromptTuningEmbedding(nn.Module):
    """
    Minimal Prompt Tuning implementation.

    Prepends n_tokens learnable embeddings to the input.
    """

    def __init__(self, n_tokens: int = 20, d_model: int = 768):
        super().__init__()
        # Learnable soft prompt tokens
        self.soft_prompt = nn.Parameter(torch.randn(n_tokens, d_model) * 0.01)

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        input_embeddings: (batch, seq_len, d_model)
        returns: (batch, n_tokens + seq_len, d_model)
        """
        batch_size = input_embeddings.shape[0]
        # Expand soft prompt for batch
        prompt = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        # Prepend to input
        return torch.cat([prompt, input_embeddings], dim=1)

    @property
    def trainable_params(self) -> int:
        return self.soft_prompt.numel()


class IA3Layer(nn.Module):
    """
    Minimal IA³ implementation.

    Learns a simple SCALING VECTOR for activations.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Learned rescaling vector (initialized to ones = identity)
        self.scaling_vector = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scaling_vector  # Element-wise multiply

    @property
    def trainable_params(self) -> int:
        return self.scaling_vector.numel()


class BitFitWrapper:
    """
    BitFit: Freeze all weight matrices, train only biases.

    This is not a layer but a STRATEGY applied to existing model.
    """

    @staticmethod
    def apply(model: nn.Module) -> nn.Module:
        """Freeze everything except bias terms and LayerNorm."""
        for name, param in model.named_parameters():
            if "bias" in name or "LayerNorm" in name or "layernorm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        return model

    @staticmethod
    def count_trainable(model: nn.Module) -> Dict:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return {
            "trainable": trainable,
            "total": total,
            "percentage": trainable / total * 100,
        }


# ═══════════════════════════════════════════════════════════════════════
# 4. PARAMETER COMPARISON
# ═══════════════════════════════════════════════════════════════════════

def compare_peft_params(d_model: int = 768, n_layers: int = 12):
    """
    Compare trainable parameters across PEFT methods for a given model.
    """
    print(f"\n{'═' * 70}")
    print(f"PEFT PARAMETER COMPARISON")
    print(f"Model: d_model={d_model}, n_layers={n_layers}")
    print(f"{'═' * 70}")

    # Approximate total params for transformer
    # Each layer: 4*d² (attention) + 8*d² (FFN) + biases ≈ 12*d²
    total_params = n_layers * 12 * d_model * d_model
    print(f"\nApprox total model params: {total_params:,}")

    methods = {}

    # Full FT
    methods["Full Fine-Tuning"] = total_params

    # LoRA (rank=16, on Q and V)
    lora_r = 16
    lora_per_layer = 2 * (d_model * lora_r + lora_r * d_model)  # A and B for Q, V
    methods["LoRA (r=16, Q+V)"] = n_layers * lora_per_layer

    # LoRA (rank=16, on all linear)
    lora_all_per_layer = 6 * (d_model * lora_r + lora_r * d_model)
    methods["LoRA (r=16, all)"] = n_layers * lora_all_per_layer

    # Adapters (bottleneck=64)
    adapter_r = 64
    adapter_per_layer = 2 * (d_model * adapter_r + adapter_r * d_model + adapter_r + d_model)
    methods["Adapters (r=64)"] = n_layers * adapter_per_layer

    # Prefix Tuning (prefix_len=20)
    prefix_len = 20
    prefix_per_layer = 2 * prefix_len * d_model  # prefix keys and values
    methods["Prefix Tuning (len=20)"] = n_layers * prefix_per_layer

    # Prompt Tuning (20 tokens)
    methods["Prompt Tuning (20 tokens)"] = 20 * d_model

    # IA³
    ia3_per_layer = 3 * d_model  # k, v, ff vectors
    methods["IA³"] = n_layers * ia3_per_layer

    # BitFit
    bitfit_per_layer = 6 * d_model + 2 * 2 * d_model  # 6 bias terms + 2 LayerNorm (γ, β)
    methods["BitFit"] = n_layers * bitfit_per_layer

    # Print comparison
    print(f"\n{'Method':<30} {'Trainable Params':>18} {'% of Full':>10}")
    print("─" * 60)
    for name, count in sorted(methods.items(), key=lambda x: x[1], reverse=True):
        pct = count / total_params * 100
        print(f"{name:<30} {count:>18,} {pct:>9.4f}%")

    return methods


# ═══════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("PEFT METHODS — COMPREHENSIVE OVERVIEW")
    print("=" * 70)

    # Show method catalog
    for key, info in PEFT_METHOD_CATALOG.items():
        print(f"\n{'─' * 70}")
        print(f"  {info['full_name']}")
        print(f"  Category: {info['category']}")
        print(f"  Paper: {info['paper']}")
        print(f"  Trainable params: {info['trainable_percent']}")
        print(f"  Inference overhead: {info['inference_overhead']}")
        print(f"  Pros: {', '.join(info['pros'][:2])}")
        print(f"  Cons: {', '.join(info['cons'][:2])}")

    # Architecture diagrams
    print(f"\n\n{'═' * 70}")
    print("ARCHITECTURE DIAGRAMS")
    print(f"{'═' * 70}")
    for method in ["lora", "adapters", "prefix_tuning", "prompt_tuning", "bitfit"]:
        print(f"\n{'─' * 70}")
        print_transformer_block(method)

    # Minimal implementation demos
    print(f"\n\n{'═' * 70}")
    print("MINIMAL IMPLEMENTATION DEMOS")
    print(f"{'═' * 70}")

    d_model = 768
    batch_size = 2
    seq_len = 32
    x = torch.randn(batch_size, seq_len, d_model)

    # LoRA
    lora = LoRALinear(d_model, d_model, rank=16, alpha=32)
    lora_out = lora(x)
    print(f"\nLoRA: {lora.trainable_params:,} trainable / {lora.total_params:,} total "
          f"({lora.trainable_params/lora.total_params*100:.2f}%)")

    # Adapter
    adapter = AdapterLayer(d_model, bottleneck=64)
    adapter_out = adapter(x)
    print(f"Adapter: {adapter.trainable_params:,} trainable params")

    # Prompt Tuning
    prompt = PromptTuningEmbedding(n_tokens=20, d_model=d_model)
    prompt_out = prompt(x)
    print(f"Prompt Tuning: {prompt.trainable_params:,} params, "
          f"input shape {x.shape} → {prompt_out.shape}")

    # IA³
    ia3 = IA3Layer(d_model)
    ia3_out = ia3(x)
    print(f"IA³: {ia3.trainable_params:,} params")

    # Parameter comparison
    compare_peft_params(d_model=768, n_layers=12)   # GPT-2 sized
    compare_peft_params(d_model=4096, n_layers=32)   # LLaMA-7B sized
