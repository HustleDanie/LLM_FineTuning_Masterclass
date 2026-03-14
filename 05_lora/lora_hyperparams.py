"""
LoRA Hyperparameter Tuning Guide
==================================

Complete guide to ALL LoRA hyperparameters, not just rank.

This module covers:
1. Target module selection (which layers to adapt)
2. Alpha (scaling factor) tuning
3. Dropout on the LoRA path
4. Learning rate selection for LoRA
5. Combined hyperparameter search strategy
6. Architecture-specific target module maps

These decisions interact with each other — this module shows
how to think about them holistically.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import math


# ===========================================================================
# 1. TARGET MODULE SELECTION
# ===========================================================================

# Architecture → module name mapping
# This is the most architecture-specific part of LoRA configuration
TARGET_MODULE_MAP = {
    # -----------------------------------------------------------------------
    # GPT-2 / DistilGPT-2
    # -----------------------------------------------------------------------
    "gpt2": {
        "attention_qkv": ["c_attn"],         # Combined Q/K/V projection
        "attention_output": ["c_proj"],        # Attention output projection  
        "mlp_up": ["c_fc"],                    # MLP up-projection (h → 4h)
        "mlp_down": ["c_proj"],                # Note: same name as attn output!
        # Recommended targets:
        "minimal": ["c_attn"],                 # Q/K/V only
        "attention": ["c_attn", "c_proj"],     # All attention
        "all_linear": ["c_attn", "c_proj", "c_fc"],  # All linear layers
    },
    
    # -----------------------------------------------------------------------
    # LLaMA / LLaMA-2 / LLaMA-3
    # -----------------------------------------------------------------------
    "llama": {
        "query": ["q_proj"],
        "key": ["k_proj"],
        "value": ["v_proj"],
        "attention_output": ["o_proj"],
        "mlp_gate": ["gate_proj"],             # Gate projection in SwiGLU
        "mlp_up": ["up_proj"],                 # Up projection
        "mlp_down": ["down_proj"],             # Down projection
        # Recommended targets:
        "minimal": ["q_proj", "v_proj"],       # Original LoRA paper style
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    },
    
    # -----------------------------------------------------------------------
    # Mistral / Mixtral
    # -----------------------------------------------------------------------
    "mistral": {
        "query": ["q_proj"],
        "key": ["k_proj"],
        "value": ["v_proj"],
        "attention_output": ["o_proj"],
        "mlp_gate": ["gate_proj"],
        "mlp_up": ["up_proj"],
        "mlp_down": ["down_proj"],
        # For Mixtral (MoE): each expert has gate/up/down
        # ["w1", "w2", "w3"] in some implementations
        "minimal": ["q_proj", "v_proj"],
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    },
    
    # -----------------------------------------------------------------------
    # Falcon
    # -----------------------------------------------------------------------
    "falcon": {
        "query_key_value": ["query_key_value"],  # Combined QKV
        "dense": ["dense"],                        # Attention output
        "mlp": ["dense_h_to_4h", "dense_4h_to_h"],
        "minimal": ["query_key_value"],
        "attention": ["query_key_value", "dense"],
        "all_linear": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    },
    
    # -----------------------------------------------------------------------
    # Phi-2 / Phi-3
    # -----------------------------------------------------------------------
    "phi": {
        "qkv": ["Wqkv"],                          # Combined QKV
        "output": ["out_proj"],
        "mlp": ["fc1", "fc2"],
        "minimal": ["Wqkv"],
        "attention": ["Wqkv", "out_proj"],
        "all_linear": ["Wqkv", "out_proj", "fc1", "fc2"],
    },
    
    # -----------------------------------------------------------------------
    # BERT / RoBERTa / DeBERTa
    # -----------------------------------------------------------------------
    "bert": {
        "query": ["query"],
        "key": ["key"],
        "value": ["value"],
        "attention_output": ["output.dense"],
        "intermediate": ["intermediate.dense"],
        "classifier": ["output.dense"],
        "minimal": ["query", "value"],
        "attention": ["query", "key", "value", "output.dense"],
        "all_linear": ["query", "key", "value", "output.dense",
                       "intermediate.dense"],
    },
}


def get_target_modules(
    architecture: str,
    strategy: str = "attention",
) -> List[str]:
    """
    Get recommended target modules for a given architecture and strategy.
    
    Parameters:
    -----------
    architecture : str
        Model architecture (gpt2, llama, mistral, falcon, phi, bert)
    strategy : str
        How many modules to target:
        - "minimal": Only Q/V or combined QKV (least params, fastest)
        - "attention": All attention projections (good balance)
        - "all_linear": All linear layers (most params, best quality)
    
    Returns:
    --------
    List of module name patterns to match
    """
    if architecture not in TARGET_MODULE_MAP:
        raise ValueError(f"Unknown architecture: {architecture}. "
                        f"Available: {list(TARGET_MODULE_MAP.keys())}")
    
    arch_map = TARGET_MODULE_MAP[architecture]
    if strategy not in arch_map:
        raise ValueError(f"Unknown strategy: {strategy}. "
                        f"Available: {[k for k in arch_map if k in ['minimal','attention','all_linear']]}")
    
    return arch_map[strategy]


def print_target_module_guide():
    """
    Print comprehensive guide for target module selection.
    """
    print("=" * 70)
    print("TARGET MODULE SELECTION GUIDE")
    print("=" * 70)
    
    print("""
    WHICH LAYERS TO ADAPT?
    
    The original LoRA paper adapted only Q and V attention projections.
    Modern practice often adapts more layers. Here's the decision guide:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │  STRATEGY          │ MODULES       │ WHEN TO USE               │
    ├─────────────────────────────────────────────────────────────────┤
    │  Minimal (Q/V)      │ 2 per layer   │ • Tight memory budget     │
    │                     │               │ • Simple task             │
    │                     │               │ • Quick experiments       │
    │                     │               │ • ~0.1% of params         │
    ├─────────────────────────────────────────────────────────────────┤
    │  Attention-only     │ 4 per layer   │ • Good default            │
    │  (Q/K/V/O)         │               │ • Most tasks              │
    │                     │               │ • ~0.2% of params         │
    ├─────────────────────────────────────────────────────────────────┤
    │  All linear layers  │ 7 per layer   │ • Best quality            │
    │  (attn + MLP)      │               │ • Complex tasks           │
    │                     │               │ • Domain adaptation       │
    │                     │               │ • ~0.5% of params         │
    ├─────────────────────────────────────────────────────────────────┤
    │  All + embeddings   │ 7+ per layer  │ • New vocabulary needed   │
    │                     │               │ • Cross-lingual transfer  │
    │                     │               │ • ~1% of params           │
    └─────────────────────────────────────────────────────────────────┘
    
    KEY FINDINGS FROM LITERATURE:
    
    1. QLoRA paper (Dettmers et al., 2023):
       "Adapting all linear layers is consistently better than QV-only"
       
    2. Practical observation:
       Going from QV-only → all attention → all linear:
       • QV-only:     85% of full FT quality
       • All attention: 92% of full FT quality  
       • All linear:  97% of full FT quality
    
    3. MLP layers matter more for:
       • Factual knowledge acquisition
       • Domain-specific terminology
       • Code generation
    
    4. Attention layers matter more for:
       • Instruction following
       • Format adherence
       • Chat behavior
    
    EMBEDDINGS:
    
    Most LoRA configs don't touch embeddings because:
    • They're already efficient (vocabulary × hidden_dim)
    • LoRA on embeddings requires special handling
    • Only useful when vocabulary needs updating
    
    But for cross-lingual or specialized domain work, adding LoRA
    to embeddings can help.
    """)


# ===========================================================================
# 2. ALPHA / SCALING FACTOR
# ===========================================================================

def explain_alpha_scaling():
    """
    Deep dive into the alpha hyperparameter and its interaction with rank.
    """
    print("\n" + "=" * 70)
    print("ALPHA (SCALING FACTOR) DEEP DIVE")
    print("=" * 70)
    
    print("""
    α (alpha) controls the magnitude of LoRA updates: scaling = α/r
    
    THERE ARE TWO PHILOSOPHIES:
    
    1. FIXED ALPHA (Original paper, HuggingFace default):
       α is set to a fixed value (e.g., 16) regardless of rank.
       
       When you change rank:
       r=8,  α=16 → scaling = 2.0
       r=16, α=16 → scaling = 1.0
       r=32, α=16 → scaling = 0.5
       
       EFFECT: Larger rank → smaller per-step updates
       CONSEQUENCE: May need to adjust learning rate when changing rank
       
    2. PROPORTIONAL ALPHA (More common in practice):
       α = constant × r (typically α = 2r)
       
       When you change rank:
       r=8,  α=16 → scaling = 2.0
       r=16, α=32 → scaling = 2.0
       r=32, α=64 → scaling = 2.0
       
       EFFECT: Consistent update magnitude across ranks
       CONSEQUENCE: Same learning rate works for any rank!
    
    RECOMMENDATION: Use proportional alpha (α = 2r) as default.
    """)
    
    # Demonstrate the effect numerically
    torch.manual_seed(42)
    d = 768
    
    print(f"  Numerical demonstration (d={d}):")
    print(f"  After 1 gradient step with lr=3e-4:")
    print(f"  {'Rank':>6} {'Alpha':>6} {'α/r':>6} {'Update Norm':>12} {'Relative':>10}")
    print("  " + "-" * 44)
    
    # Simulate one gradient step
    norms = []
    configs = [
        # Fixed alpha = 16
        (4, 16, "fixed α=16"),
        (8, 16, "fixed α=16"),
        (16, 16, "fixed α=16"),
        (32, 16, "fixed α=16"),
        (64, 16, "fixed α=16"),
    ]
    
    first_norm = None
    for r, alpha, label in configs:
        scaling = alpha / r
        
        # Simulate: A and B after one gradient step
        A = torch.randn(r, d) / math.sqrt(r)
        B = torch.randn(d, r) * 0.001  # Small B after 1 step (starts at 0)
        
        delta_W = scaling * (B @ A)
        norm = delta_W.norm().item()
        norms.append(norm)
        
        if first_norm is None:
            first_norm = norm
        
        print(f"  {r:>6} {alpha:>6} {scaling:>6.2f} {norm:>12.6f} {norm/first_norm:>10.2f}x")
    
    print(f"\n  With fixed α=16: update magnitude DECREASES with rank")
    print(f"  (α/r gets smaller, so each step does less)")
    
    print(f"\n  Now with proportional α = 2r:")
    first_norm = None
    for r in [4, 8, 16, 32, 64]:
        alpha = 2 * r
        scaling = alpha / r  # Always = 2
        
        A = torch.randn(r, d) / math.sqrt(r)
        B = torch.randn(d, r) * 0.001
        
        delta_W = scaling * (B @ A)
        norm = delta_W.norm().item()
        
        if first_norm is None:
            first_norm = norm
        
        print(f"  r={r:>3}, α={alpha:>4}: scaling={scaling:.1f}, "
              f"||ΔW||={norm:.6f}, relative={norm/first_norm:.2f}x")
    
    print(f"\n  With proportional α: update magnitude is CONSISTENT across ranks!")


# ===========================================================================
# 3. DROPOUT
# ===========================================================================

def explain_lora_dropout():
    """
    LoRA dropout: what it is, when to use it, and how it works.
    """
    print("\n" + "=" * 70)
    print("LoRA DROPOUT")
    print("=" * 70)
    
    print("""
    LoRA applies dropout to the INPUT of the LoRA path, NOT the output.
    
    Forward pass:
        h = Wx + (α/r) · B · A · Dropout(x) + b
                                    ↑
                                  HERE
    
    WHY DROPOUT ON INPUT (not activations)?
    - Dropping input features forces A to be robust to missing inputs
    - More stable than dropping low-rank activations (which would be r-dimensional)
    - Consistent with standard dropout in neural networks
    
    WHEN TO USE:
    ┌──────────────────────────────────────────────────────────────┐
    │ Dataset size relative to model │ Recommended dropout        │
    ├──────────────────────────────────────────────────────────────┤
    │ Very small (<1K examples)       │ 0.1 - 0.2                 │
    │ Small (1K - 10K examples)       │ 0.05 - 0.1               │
    │ Medium (10K - 100K examples)    │ 0.0 - 0.05               │
    │ Large (>100K examples)          │ 0.0 (no dropout needed)   │
    └──────────────────────────────────────────────────────────────┘
    
    IMPORTANT: LoRA dropout is DIFFERENT from model dropout!
    - Model dropout: Applied in attention and MLP layers of the base model
    - LoRA dropout: Applied only in the LoRA side paths
    - Both can be active simultaneously
    
    PRACTICAL NOTE:
    Most successful LoRA configurations use dropout=0.05 or dropout=0.0.
    LoRA already acts as a regularizer (limited rank = limited capacity),
    so additional dropout is often unnecessary.
    """)
    
    # Demonstrate dropout effect
    torch.manual_seed(42)
    d = 768
    r = 16
    alpha = 32
    
    x = torch.randn(1, 10, d)
    
    # Without dropout
    A = torch.randn(r, d) / math.sqrt(r)
    B = torch.randn(d, r) * 0.01
    
    no_drop = (alpha / r) * (x @ A.T @ B.T)
    
    # With dropout (evaluate variance across runs)
    dropout = nn.Dropout(0.1)
    drop_outputs = []
    for _ in range(100):
        x_drop = dropout(x)
        out = (alpha / r) * (x_drop @ A.T @ B.T)
        drop_outputs.append(out)
    
    drops = torch.stack(drop_outputs)
    
    print(f"\n  Dropout effect on LoRA output (100 forward passes):")
    print(f"    Without dropout - output norm: {no_drop.norm().item():.4f}")
    print(f"    With dropout 0.1:")
    print(f"      Mean output norm:     {drops.norm(dim=-1).mean().item():.4f}")
    print(f"      Std of output norm:   {drops.norm(dim=-1).std().item():.4f}")
    print(f"      Coefficient of var:   {drops.norm(dim=-1).std().item() / drops.norm(dim=-1).mean().item():.4f}")


# ===========================================================================
# 4. LEARNING RATE FOR LoRA
# ===========================================================================

def explain_learning_rate():
    """
    Learning rate selection is critical for LoRA and differs from full fine-tuning.
    """
    print("\n" + "=" * 70)
    print("LEARNING RATE FOR LoRA")
    print("=" * 70)
    
    print("""
    LoRA typically uses 2-10x HIGHER learning rate than full fine-tuning.
    
    WHY?
    - Full FT: updates spread across ALL parameters → small LR is fine
    - LoRA: updates concentrated in few parameters → need larger LR to
      have the same effective update magnitude on the output
    
    TYPICAL RANGES:
    ┌────────────────────────────────────────────────────────────────┐
    │ Method              │ Learning Rate Range │ Typical Default    │
    ├────────────────────────────────────────────────────────────────┤
    │ Full Fine-Tuning    │ 1e-5 to 5e-5        │ 2e-5              │
    │ LoRA (attention)    │ 1e-4 to 5e-4        │ 2e-4              │
    │ LoRA (all linear)   │ 5e-5 to 3e-4        │ 1e-4              │
    │ QLoRA               │ 1e-4 to 5e-4        │ 2e-4              │
    └────────────────────────────────────────────────────────────────┘
    
    INTERACTION WITH ALPHA:
    
    The effective learning rate for LoRA is: lr_effective = lr × (α/r)
    
    If α/r = 2 (common): lr=1e-4 gives effective lr=2e-4
    If α/r = 1 (HF default): lr=1e-4 gives effective lr=1e-4
    If α/r = 0.5 (large r, fixed α): lr=1e-4 gives effective lr=5e-5
    
    This is why PROPORTIONAL α is preferred — it decouples LR from rank.
    
    LEARNING RATE SCHEDULE:
    
    • Warmup: 3-10% of total steps (LoRA benefits from warmup)
    • Decay: Cosine decay to 0 or 10% of peak LR
    • Linear decay also works well
    • Constant LR is okay for short training runs
    
    DIFFERENT LR FOR DIFFERENT PARTS:
    
    Some practitioners use different LRs for A vs B:
    - Higher LR for A (it's initialized non-zero, has immediate gradient)
    - Lower LR for B (it starts at zero, gradients are initially small)
    
    But this is rarely necessary — same LR works fine in practice.
    """)
    
    # Demonstrate LR × Alpha interaction
    print(f"\n  LR × Alpha interaction (showing effective step sizes):")
    print(f"  {'LR':>10} {'α':>6} {'r':>4} {'α/r':>6} {'Effective':>12}")
    print("  " + "-" * 42)
    
    for lr in [1e-5, 5e-5, 1e-4, 3e-4, 1e-3]:
        for alpha, r in [(16, 16), (32, 16), (64, 32)]:
            scaling = alpha / r
            effective = lr * scaling
            print(f"  {lr:>10.0e} {alpha:>6} {r:>4} {scaling:>6.1f} {effective:>12.0e}")
        print()


# ===========================================================================
# 5. COMPLETE HYPERPARAMETER CONFIGURATION
# ===========================================================================

@dataclass
class LoRAHyperparamConfig:
    """Complete LoRA hyperparameter configuration with documentation."""
    
    # Core LoRA parameters
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Training parameters
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"       # cosine, linear, constant
    num_epochs: int = 3
    max_steps: int = -1                 # Override epochs if > 0
    
    # Batch / gradient
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Regularization
    neftune_noise_alpha: Optional[float] = None  # NEFTune: Add noise to embeddings
    label_smoothing: float = 0.0
    
    # LoRA-specific
    bias: str = "none"                  # "none", "all", "lora_only"
    modules_to_save: Optional[List[str]] = None  # e.g., ["lm_head", "embed_tokens"]
    
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps
    
    def effective_scaling(self) -> float:
        return self.alpha / self.rank
    
    def summary(self) -> str:
        return (
            f"LoRA Config: r={self.rank}, α={self.alpha}, "
            f"scaling={self.effective_scaling():.2f}, "
            f"dropout={self.dropout}, "
            f"targets={self.target_modules}, "
            f"lr={self.learning_rate}, "
            f"eff_batch={self.effective_batch_size()}"
        )


# Pre-built configurations for common use cases
LORA_RECIPES = {
    "chat_7b_basic": LoRAHyperparamConfig(
        rank=16,
        alpha=32,
        dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        learning_rate=2e-4,
        num_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=4,
    ),
    
    "chat_7b_quality": LoRAHyperparamConfig(
        rank=64,
        alpha=128,
        dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        learning_rate=1e-4,
        num_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=8,
    ),
    
    "qlora_7b_consumer_gpu": LoRAHyperparamConfig(
        rank=16,
        alpha=32,
        dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        learning_rate=2e-4,
        num_epochs=1,
        batch_size=1,
        gradient_accumulation_steps=16,
    ),
    
    "code_7b": LoRAHyperparamConfig(
        rank=64,
        alpha=128,
        dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        learning_rate=1e-4,
        num_epochs=3,
        batch_size=2,
        gradient_accumulation_steps=8,
    ),
    
    "classification_bert": LoRAHyperparamConfig(
        rank=8,
        alpha=16,
        dropout=0.1,
        target_modules=["query", "value"],
        learning_rate=3e-4,
        num_epochs=5,
        batch_size=16,
        gradient_accumulation_steps=1,
        modules_to_save=["classifier"],
    ),
    
    "tiny_experiment": LoRAHyperparamConfig(
        rank=4,
        alpha=8,
        dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        learning_rate=5e-4,
        num_epochs=1,
        batch_size=8,
        gradient_accumulation_steps=1,
    ),
}


def print_all_recipes():
    """Print all pre-built LoRA recipes."""
    print("\n" + "=" * 70)
    print("PRE-BUILT LoRA RECIPES")
    print("=" * 70)
    
    for name, config in LORA_RECIPES.items():
        print(f"\n  📋 {name}:")
        print(f"     {config.summary()}")
        if name == "chat_7b_basic":
            print(f"     Use case: General chat/instruction tuning on a 7B model")
            print(f"     GPU: 24GB+ (A100, 4090, etc.)")
        elif name == "qlora_7b_consumer_gpu":
            print(f"     Use case: QLoRA on consumer hardware")
            print(f"     GPU: 12-16GB (3090, 4070 Ti, etc.)")
        elif name == "code_7b":
            print(f"     Use case: Code generation fine-tuning")
            print(f"     GPU: 24GB+ (code tasks need more rank)")
        elif name == "classification_bert":
            print(f"     Use case: Text classification with BERT")
            print(f"     GPU: 8GB (small model, small rank)")


# ===========================================================================
# 6. HYPERPARAMETER SEARCH STRATEGY
# ===========================================================================

def explain_search_strategy():
    """
    Recommended strategy for finding optimal LoRA hyperparameters.
    """
    print("\n" + "=" * 70)
    print("HYPERPARAMETER SEARCH STRATEGY")
    print("=" * 70)
    
    print("""
    DON'T search all hyperparameters simultaneously! Use this staged approach:
    
    ═══════════════════════════════════════════════════════════════════
    STAGE 1: Quick validation (1 run, 10 minutes)
    ═══════════════════════════════════════════════════════════════════
    
    Use defaults: r=16, α=32, lr=2e-4, target=all_attention
    
    Goal: Verify the pipeline works and model can learn.
    If loss doesn't decrease → data/code issue, not hyperparams.
    
    ═══════════════════════════════════════════════════════════════════
    STAGE 2: Target module search (3 runs)
    ═══════════════════════════════════════════════════════════════════
    
    Compare with FIXED r=16, α=32, lr=2e-4:
      Run 1: target = Q/V only (minimal)
      Run 2: target = all attention (Q/K/V/O)
      Run 3: target = all linear layers
    
    Pick the best target strategy, then move to Stage 3.
    
    TYPICAL FINDING: all_linear > all_attention > Q/V
    But Q/V may be preferred if memory is tight.
    
    ═══════════════════════════════════════════════════════════════════
    STAGE 3: Rank search (5 runs)
    ═══════════════════════════════════════════════════════════════════
    
    With best target from Stage 2, compare:
      r ∈ {4, 8, 16, 32, 64}, α = 2r
    
    Pick the rank with best val_loss / training_time tradeoff.
    
    ═══════════════════════════════════════════════════════════════════
    STAGE 4: Learning rate search (5 runs)
    ═══════════════════════════════════════════════════════════════════
    
    With best target and rank, compare:
      lr ∈ {5e-5, 1e-4, 2e-4, 5e-4, 1e-3}
    
    ═══════════════════════════════════════════════════════════════════
    STAGE 5: Fine-tune remaining (2-3 runs)
    ═══════════════════════════════════════════════════════════════════
    
    With best configs from above:
      - Try dropout 0.0 vs 0.05 vs 0.1
      - Try different number of epochs
      - Try different warmup ratios
    
    TOTAL: ~16 runs to find near-optimal configuration.
    With 30min per run on a single GPU: ~8 hours total.
    Most experiments can be done with 10-20% of full data.
    
    ═══════════════════════════════════════════════════════════════════
    SHORTCUT: If you don't have time for search
    ═══════════════════════════════════════════════════════════════════
    
    Use this configuration — it works well 80% of the time:
    
      r = 32
      α = 64
      dropout = 0.05
      target_modules = all linear layers
      lr = 2e-4
      warmup_ratio = 0.03
      weight_decay = 0.01
      lr_scheduler = cosine
      num_epochs = 3
      gradient_accumulation = enough for effective batch size 32-64
    """)


# ===========================================================================
# 7. BIAS TRAINING IN LoRA
# ===========================================================================

def explain_bias_options():
    """
    LoRA offers three options for handling bias parameters.
    """
    print("\n" + "=" * 70)
    print("BIAS TRAINING OPTIONS IN LoRA")
    print("=" * 70)
    
    print("""
    OPTION 1: bias="none" (DEFAULT, most common)
    
      • Freeze ALL biases along with weights
      • Simplest configuration
      • Works well for most tasks
      • Smallest checkpoint size
    
    OPTION 2: bias="all"
    
      • Train ALL bias parameters in the model
      • Adds very few extra parameters (bias is tiny: d_out per layer)
      • Can help for classification tasks
      • Extra params: ~0.01% of model (negligible)
    
    OPTION 3: bias="lora_only"
    
      • Only train biases in layers that have LoRA adapters
      • Compromise between "none" and "all"
      • Slightly more capacity without much overhead
    
    ┌─────────────────────────────────────────────────────────────┐
    │ Use Case              │ Recommended Bias Setting            │
    ├─────────────────────────────────────────────────────────────┤
    │ General fine-tuning   │ "none" (default)                    │
    │ Classification        │ "all" (biases help with class logits│
    │ Small dataset         │ "none" (avoid overfitting)          │
    │ Maximum quality       │ "all" (tiny overhead, may help)     │
    └─────────────────────────────────────────────────────────────┘
    """)
    
    # Demonstrate the parameter count difference
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        
        total_params = sum(p.numel() for p in model.parameters())
        bias_params = sum(p.numel() for n, p in model.named_parameters() if "bias" in n)
        
        print(f"  Example: distilgpt2")
        print(f"    Total parameters:  {total_params:>12,}")
        print(f"    Bias parameters:   {bias_params:>12,}")
        print(f"    Bias percentage:   {bias_params/total_params*100:>12.4f}%")
        print(f"    → Training biases adds negligible overhead!")
        
    except ImportError:
        print(f"  Typical bias parameter count: <0.1% of total model parameters")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print_target_module_guide()
    
    print(f"\n\n  Available architectures: {list(TARGET_MODULE_MAP.keys())}")
    for arch in TARGET_MODULE_MAP:
        for strategy in ["minimal", "attention", "all_linear"]:
            if strategy in TARGET_MODULE_MAP[arch]:
                modules = get_target_modules(arch, strategy)
                print(f"    {arch:>10} / {strategy:<12}: {modules}")
    
    explain_alpha_scaling()
    explain_lora_dropout()
    explain_learning_rate()
    print_all_recipes()
    explain_search_strategy()
    explain_bias_options()
    
    print("\n" + "=" * 70)
    print("HYPERPARAMETER GUIDE SUMMARY")  
    print("=" * 70)
    print("""
    QUICK REFERENCE — LoRA Hyperparameters:
    
    • rank (r): 16 (default) | 4-8 (minimal) | 32-64 (quality)
    • alpha (α): 2×r (proportional) | 16 (fixed, original paper)
    • dropout: 0.05 (default) | 0.0 (large data) | 0.1 (small data)
    • target_modules: all_linear (best) > attention (good) > q/v (minimal)
    • learning_rate: 2e-4 (default) | 1e-4 (conservative) | 5e-4 (aggressive)
    • bias: "none" (default) | "all" (classification)
    • lr_scheduler: cosine (recommended) | linear | constant
    • warmup_ratio: 0.03 (recommended)
    """)
