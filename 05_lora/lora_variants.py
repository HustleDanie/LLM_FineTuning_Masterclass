"""
LoRA Variants and Extensions
==============================

LoRA's success has inspired many variants that improve upon the original
in different ways. This module covers the most important ones:

1. LoRA+  — Different learning rates for A and B matrices
2. rsLoRA — Rank-stabilized scaling (replaces α/r with α/√r)
3. DoRA   — Weight-Decomposed Low-Rank Adaptation (direction + magnitude)
4. AdaLoRA — Adaptive rank allocation across layers
5. DyLoRA — Dynamic rank training (train all ranks simultaneously)
6. QA-LoRA — Quantization-aware LoRA
7. LoRA-FA — Frozen-A LoRA (freeze A, only train B)
8. VeRA   — Very-parameter-efficient LoRA (shared random A/B)
9. GLoRA  — Generalized LoRA (adds prompts + scaling)
10. LongLoRA — LoRA for long-context fine-tuning

Each variant is explained with its motivation, math, and when to use it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List


# ===========================================================================
# 1. LoRA+ (Hayou et al., 2024)
# ===========================================================================

class LoRAPlusLinear(nn.Module):
    """
    LoRA+ uses different learning rates for A and B matrices.
    
    Key Insight:
    A and B have fundamentally different roles:
    - A: projects input to low-rank space (feature extraction)
    - B: projects from low-rank space to output (feature mapping)
    
    The paper shows that B should have a HIGHER learning rate than A
    for optimal convergence. Specifically:
    
    lr_B = lr_A × ratio  (where ratio ≈ 16 works well)
    
    This simple change can improve convergence speed significantly.
    
    Paper: "LoRA+: Efficient Low Rank Adaptation of Large Models"
    
    IMPLEMENTATION NOTE: In practice, you implement this by passing
    separate parameter groups to the optimizer, not by modifying the layer.
    """
    
    def __init__(self, in_features, out_features, rank=16, alpha=32.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.weight.requires_grad_(False)
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
    
    def forward(self, x):
        base = F.linear(x, self.weight)
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base + lora
    
    @staticmethod
    def get_optimizer_groups(model, lr_A=1e-4, lr_ratio=16.0, weight_decay=0.01):
        """
        Create optimizer parameter groups with different LRs for A and B.
        
        Usage:
            groups = LoRAPlusLinear.get_optimizer_groups(model, lr_A=1e-4, lr_ratio=16)
            optimizer = torch.optim.AdamW(groups)
        """
        group_A = {"params": [], "lr": lr_A, "weight_decay": weight_decay}
        group_B = {"params": [], "lr": lr_A * lr_ratio, "weight_decay": weight_decay}
        group_other = {"params": [], "lr": lr_A, "weight_decay": weight_decay}
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_A" in name:
                group_A["params"].append(param)
            elif "lora_B" in name:
                group_B["params"].append(param)
            else:
                group_other["params"].append(param)
        
        groups = []
        if group_A["params"]:
            groups.append(group_A)
        if group_B["params"]:
            groups.append(group_B)
        if group_other["params"]:
            groups.append(group_other)
        
        print(f"  LoRA+ optimizer groups:")
        print(f"    A matrices: {len(group_A['params'])} params, lr={lr_A}")
        print(f"    B matrices: {len(group_B['params'])} params, lr={lr_A * lr_ratio}")
        
        return groups


# ===========================================================================
# 2. rsLoRA — Rank-Stabilized LoRA (Kalajdzievski, 2023)
# ===========================================================================

class RsLoRALinear(nn.Module):
    """
    rsLoRA (Rank-Stabilized LoRA) fixes a scaling issue in standard LoRA.
    
    Problem with standard LoRA:
    The scaling factor α/r means that as r increases, each element of B·A
    contributes LESS to the output. This can make training unstable when
    comparing results across different ranks.
    
    rsLoRA solution:
    Replace the scaling factor α/r with α/√r.
    
    Mathematical justification:
    - With random init, ||B·A·x||² scales as r (not r²) due to zero-init of B
    - So the correct normalization is 1/√r, not 1/r
    - This gives more consistent training dynamics across ranks
    
    Standard LoRA: h = Wx + (α/r) · B·A·x
    rsLoRA:        h = Wx + (α/√r) · B·A·x
    
    Paper: "Scaling Data-Constrained Language Models"
    
    When to use: When comparing results across different ranks,
    or when using very high ranks (r > 64).
    """
    
    def __init__(self, in_features, out_features, rank=16, alpha=32.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.weight.requires_grad_(False)
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # KEY DIFFERENCE: √r instead of r
        self.scaling = alpha / math.sqrt(rank)
    
    def forward(self, x):
        base = F.linear(x, self.weight)
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base + lora


# ===========================================================================
# 3. DoRA — Weight-Decomposed LoRA (Liu et al., 2024)
# ===========================================================================

class DoRALinear(nn.Module):
    """
    DoRA (Weight-Decomposed Low-Rank Adaptation) decomposes the weight
    update into MAGNITUDE and DIRECTION components:
    
    Standard LoRA:
        W' = W + (α/r) · B·A
        
    DoRA:
        W' = m · (W + (α/r) · B·A) / ||W + (α/r) · B·A||_col
        
    where:
        m ∈ ℝ^{d_out} — trainable magnitude vector (one per output dim)
        The denominator normalizes each COLUMN of the updated weight
    
    Motivation:
    - Full fine-tuning changes both the direction and magnitude of weight columns
    - Standard LoRA conflates these two updates
    - DoRA separates them: LoRA handles direction, m handles magnitude
    - This is inspired by Weight Normalization (Salimans & Kingma, 2016)
    
    Benefits:
    - Matches full fine-tuning quality more closely
    - More stable training (magnitude and direction don't interfere)
    - Works especially well for small ranks
    
    Paper: "DoRA: Weight-Decomposed Low-Rank Adaptation"
    
    When to use: When you need maximum quality with small rank (r=4-16),
    or when LoRA quality doesn't match full fine-tuning.
    """
    
    def __init__(self, in_features, out_features, rank=16, alpha=32.0):
        super().__init__()
        
        # Frozen pre-trained weight
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.weight.requires_grad_(False)
        
        # LoRA matrices (for direction update)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        
        # DoRA magnitude vector — initialized from pre-trained weight norms
        # Each element corresponds to one output neuron
        with torch.no_grad():
            weight_norm = self.weight.norm(dim=1, keepdim=True)  # (d_out, 1)
        self.magnitude = nn.Parameter(weight_norm.squeeze())  # (d_out,)
    
    def forward(self, x):
        # Step 1: Compute updated weight (direction only)
        delta_W = self.scaling * (self.lora_B @ self.lora_A)  # (d_out, d_in)
        W_updated = self.weight + delta_W
        
        # Step 2: Normalize columns to get direction
        W_norm = W_updated.norm(dim=1, keepdim=True)  # (d_out, 1)
        W_direction = W_updated / (W_norm + 1e-8)
        
        # Step 3: Apply learned magnitude
        # m · (direction)
        W_final = self.magnitude.unsqueeze(1) * W_direction  # (d_out, d_in)
        
        # Step 4: Standard linear forward
        return F.linear(x, W_final)


# ===========================================================================
# 4. AdaLoRA — Adaptive Budget Allocation (Zhang et al., 2023)
# ===========================================================================

class AdaLoRALinear(nn.Module):
    """
    AdaLoRA dynamically adjusts the rank of each LoRA layer during training.
    
    Key Idea:
    Instead of fixed rank, AdaLoRA uses SVD-parameterization:
    
        ΔW = P · Λ · Q
        
    where:
        P ∈ ℝ^{d_out × r}   (left singular vectors)
        Λ ∈ ℝ^{r × r}       (diagonal: singular values)
        Q ∈ ℝ^{r × d_in}    (right singular vectors)
    
    During training, AdaLoRA prunes singular values with small importance
    scores, effectively reducing the rank of less important layers.
    
    Importance Score:
        s_i = |λ_i| × (||p_i||₂ + ||q_i||₂) / 2
    
    Process:
    1. Start with rank r_init (e.g., 64) for all layers
    2. Periodically compute importance scores
    3. Prune the least important singular values
    4. Final ranks vary per layer based on importance
    
    Benefits:
    - Automatically allocates rank where it's needed
    - Budget-efficient: total params are fixed, distributed optimally
    - Often outperforms fixed-rank LoRA with same total params
    
    Paper: "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning"
    
    When to use: When you want optimal rank allocation without manual tuning.
    Available in HuggingFace PEFT as `AdaLoraConfig`.
    """
    
    def __init__(self, in_features, out_features, rank=64, alpha=128.0):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.weight.requires_grad_(False)
        
        # SVD parameterization instead of B·A
        self.P = nn.Parameter(torch.randn(out_features, rank) * 0.01)       # Left singular vectors
        self.Lambda = nn.Parameter(torch.ones(rank) * 0.01)                  # Singular values
        self.Q = nn.Parameter(torch.randn(rank, in_features) * 0.01)        # Right singular vectors
        
        self.scaling = alpha / rank
        self.rank = rank
        
        # Mask for pruned singular values
        self.register_buffer("rank_mask", torch.ones(rank, dtype=torch.bool))
    
    def forward(self, x):
        # Apply mask to prune singular values
        active_Lambda = self.Lambda * self.rank_mask.float()
        
        # ΔW = P · diag(Λ) · Q
        delta_W = self.P * active_Lambda.unsqueeze(0) @ self.Q
        
        return F.linear(x, self.weight + self.scaling * delta_W)
    
    def compute_importance(self) -> torch.Tensor:
        """Compute importance score for each singular value."""
        with torch.no_grad():
            p_norms = self.P.norm(dim=0)       # (rank,)
            q_norms = self.Q.norm(dim=1)       # (rank,)
            importance = self.Lambda.abs() * (p_norms + q_norms) / 2
        return importance
    
    def prune_rank(self, n_prune: int = 1):
        """Prune the n_prune least important singular values."""
        importance = self.compute_importance()
        # Only consider active singular values
        importance[~self.rank_mask] = float('inf')
        
        # Find least important
        _, indices = importance.topk(n_prune, largest=False)
        self.rank_mask[indices] = False
        
        active = self.rank_mask.sum().item()
        print(f"    Pruned {n_prune} SVs, active rank: {active}/{self.rank}")


# ===========================================================================
# 5. DyLoRA — Dynamic Rank LoRA (Valipour et al., 2023)
# ===========================================================================

class DyLoRALinear(nn.Module):
    """
    DyLoRA trains LoRA to work at ANY rank ≤ r_max.
    
    Problem with standard LoRA:
    If you train with rank 16, you MUST use rank 16 at inference.
    Want rank 8? You need to retrain from scratch.
    
    DyLoRA solution:
    During training, randomly sample a rank b ∈ {1, ..., r_max}
    and only use the first b rows/columns of A and B.
    
    Training:
    For each forward pass:
    1. Sample b ~ Uniform{1, ..., r_max}
    2. Use A[:b, :] and B[:, :b]
    3. Compute loss and update only those rows/columns
    
    At inference:
    Use ANY rank from 1 to r_max without retraining!
    This is like a "rank elastic" LoRA.
    
    Paper: "DyLoRA: Parameter-Efficient Tuning of Pre-trained Models 
            using Dynamic Search-Free Low-Rank Adaptation"
    
    When to use: When you want flexibility to choose rank at deployment time,
    or when you want to find optimal rank without grid search.
    """
    
    def __init__(self, in_features, out_features, max_rank=32, alpha=64.0):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.weight.requires_grad_(False)
        
        self.max_rank = max_rank
        self.alpha = alpha
        
        self.lora_A = nn.Parameter(torch.randn(max_rank, in_features) / math.sqrt(max_rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, max_rank))
        
        self._inference_rank = max_rank  # Can be changed at inference time
    
    def forward(self, x):
        if self.training:
            # Sample a random rank for this forward pass
            b = torch.randint(1, self.max_rank + 1, (1,)).item()
        else:
            # Use the configured inference rank
            b = self._inference_rank
        
        scaling = self.alpha / b
        
        # Use only first b dimensions
        A_b = self.lora_A[:b, :]      # (b, d_in)
        B_b = self.lora_B[:, :b]      # (d_out, b)
        
        base = F.linear(x, self.weight)
        lora = F.linear(F.linear(x, A_b), B_b) * scaling
        return base + lora
    
    def set_inference_rank(self, rank: int):
        """Set the rank to use at inference time."""
        assert 1 <= rank <= self.max_rank
        self._inference_rank = rank
        print(f"  Inference rank set to {rank}/{self.max_rank}")


# ===========================================================================
# 6. LoRA-FA — Frozen-A LoRA (Zhang et al., 2023)
# ===========================================================================

class LoRAFALinear(nn.Module):
    """
    LoRA-FA freezes the A matrix after initialization, training only B.
    
    Motivation:
    - A is initialized randomly and stays close to init even after training
    - The meaningful adaptation happens mostly in B
    - Freezing A halves the trainable parameters AND halves the memory
    
    Math:
        Standard LoRA: ΔW = B · A       (both A and B trained)
        LoRA-FA:       ΔW = B · A_frozen (only B trained)
    
    Since A is frozen (acts like a random projection), this is similar to
    the Random Projection baseline but with B being Learnable — and it
    works surprisingly well!
    
    Benefits:
    - 50% fewer trainable parameters than standard LoRA
    - 50% less optimizer memory
    - Comparable quality for many tasks
    
    When to use: Very tight memory budget, or as a quick experiment.
    
    Limitation: May underperform when the optimal A differs significantly
    from random initialization (complex domain shifts).
    """
    
    def __init__(self, in_features, out_features, rank=16, alpha=32.0):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.weight.requires_grad_(False)
        
        # A is FROZEN (random projection)
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features) / math.sqrt(rank),
            requires_grad=False  # FROZEN!
        )
        
        # Only B is trainable
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.scaling = alpha / rank
    
    def forward(self, x):
        base = F.linear(x, self.weight)
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base + lora


# ===========================================================================
# 7. VeRA — Vector-based Random Matrix Adaptation (Kopiczko et al., 2024)
# ===========================================================================

class VeRALinear(nn.Module):
    """
    VeRA (Very-parameter-efficient Random matrix Adaptation) pushes parameter
    efficiency to the extreme by SHARING random A and B matrices across all layers.
    
    Standard LoRA per layer:  2 × r × d parameters
    VeRA per layer:          d_out + d_in parameters (!)
    
    How it works:
    1. Generate random A and B matrices ONCE (shared across all layers)
    2. Each layer learns only two tiny vectors: d and b
    3. ΔW = B · diag(b) · diag(d) · A
    
    Where:
        A ∈ ℝ^{r × d_in}    — shared, frozen, random
        B ∈ ℝ^{d_out × r}   — shared, frozen, random
        d ∈ ℝ^{r}           — per-layer trainable scaling
        b ∈ ℝ^{r}           — per-layer trainable scaling
    
    The diagonal matrices diag(d) and diag(b) allow each layer to "select"
    which dimensions of the shared projection are important.
    
    Parameter savings (LLaMA-7B example):
    - LoRA (r=16, all linear): ~16M params
    - VeRA: ~1.5M params (10x fewer!)
    
    Paper: "VeRA: Vector-based Random Matrix Adaptation"
    
    When to use: Extreme parameter efficiency, serving many adapters,
    or when adapter size must be minimized.
    """
    
    # Class-level shared random matrices
    _shared_A: Optional[torch.Tensor] = None
    _shared_B: Optional[torch.Tensor] = None
    
    def __init__(self, in_features, out_features, rank=256, alpha=512.0, seed=42):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.weight.requires_grad_(False)
        
        self.rank = rank
        self.scaling = alpha / rank
        
        # Shared random projections (frozen, generated from seed)
        gen = torch.Generator().manual_seed(seed)
        
        # In a real implementation, these would be shared across all VeRA layers
        # Here we create per-instance for simplicity
        self.register_buffer("shared_A", torch.randn(rank, in_features, generator=gen) / math.sqrt(rank))
        self.register_buffer("shared_B", torch.randn(out_features, rank, generator=gen) / math.sqrt(out_features))
        
        # Per-layer trainable vectors (MUCH smaller than full A, B matrices)
        self.d_vec = nn.Parameter(torch.ones(rank))        # Scaling for A
        self.b_vec = nn.Parameter(torch.zeros(rank))       # Scaling for B
    
    def forward(self, x):
        base = F.linear(x, self.weight)
        
        # Apply per-layer scaling: diag(b) and diag(d)
        # ΔW = shared_B · diag(b_vec) · diag(d_vec) · shared_A
        # Efficient computation: scale rows/columns instead of matrix multiply
        
        # Step 1: Scale A rows by d_vec
        scaled_A = self.shared_A * self.d_vec.unsqueeze(1)   # (r, d_in)
        
        # Step 2: Scale B columns by b_vec
        scaled_B = self.shared_B * self.b_vec.unsqueeze(0)   # (d_out, r)
        
        # Step 3: Standard LoRA forward
        lora = F.linear(F.linear(x, scaled_A), scaled_B) * self.scaling
        
        return base + lora


# ===========================================================================
# 8. COMPARISON TABLE
# ===========================================================================

def print_variant_comparison():
    """Print a comprehensive comparison of all LoRA variants."""
    print("=" * 90)
    print("LoRA VARIANTS COMPARISON")
    print("=" * 90)
    
    print("""
    ┌──────────────┬───────────────────────────────────────────────────────────────────┐
    │ Variant      │ Key Change             │ Params   │ Quality │ When to Use        │
    ├──────────────┼───────────────────────────────────────────────────────────────────┤
    │ LoRA         │ Baseline               │ 2·r·d    │ ★★★★   │ General default     │
    │ LoRA+        │ Different lr for A/B    │ 2·r·d    │ ★★★★+  │ Faster convergence  │
    │ rsLoRA       │ α/√r scaling           │ 2·r·d    │ ★★★★   │ Comparing ranks     │
    │ DoRA         │ Direction + magnitude   │ 2·r·d+d  │ ★★★★★  │ Max quality         │
    │ AdaLoRA      │ Adaptive rank per layer │ varies   │ ★★★★+  │ Optimal allocation  │
    │ DyLoRA       │ Train all ranks at once │ 2·r·d    │ ★★★★   │ Rank flexibility    │
    │ LoRA-FA      │ Freeze A matrix        │ r·d      │ ★★★    │ Memory constrained  │
    │ VeRA         │ Shared random A/B      │ 2·r      │ ★★★    │ Extreme efficiency  │
    └──────────────┴───────────────────────────────────────────────────────────────────┘

    PARAMETER COUNT COMPARISON (for a 768×768 weight with rank 16):
    
    ┌──────────────┬─────────────────┬──────────────┐
    │ Method       │ Trainable Params│ vs LoRA      │
    ├──────────────┼─────────────────┼──────────────┤
    │ Full FT      │     589,824     │  24.0x more  │
    │ LoRA         │      24,576     │  1.0x (base) │
    │ LoRA+        │      24,576     │  1.0x (same) │
    │ rsLoRA       │      24,576     │  1.0x (same) │
    │ DoRA         │      25,344     │  1.03x       │
    │ AdaLoRA      │      ~24,576    │  ~1.0x       │
    │ DyLoRA       │      24,576     │  1.0x        │
    │ LoRA-FA      │      12,288     │  0.5x        │
    │ VeRA         │       1,536     │  0.06x       │
    └──────────────┴─────────────────┴──────────────┘
    """)


def print_selection_guide():
    """When to use which variant."""
    print("\n" + "=" * 70)
    print("WHICH VARIANT SHOULD I USE?")
    print("=" * 70)
    
    print("""
    START HERE → Standard LoRA (r=16, α=32)
    
    Then consider:
    
    ❓ "I want better quality without changing rank"
       → DoRA (adds magnitude learning, ~3% more params)
       → LoRA+ (just change optimizer groups, zero overhead)
    
    ❓ "I want to compare results across different ranks"
       → rsLoRA (consistent scaling across ranks)
    
    ❓ "I don't know what rank to use for each layer"
       → AdaLoRA (automatic rank allocation)
       → DyLoRA (train once, deploy at any rank)
    
    ❓ "I need to minimize adapter size at all costs"
       → VeRA (10-50x fewer params than LoRA)
       → LoRA-FA (50% fewer params than LoRA)
    
    ❓ "I need to serve hundreds of adapters"
       → VeRA (tiny adapters, shared base projections)
    
    ❓ "I'm doing QLoRA on a consumer GPU"
       → Standard LoRA is fine (quantization is the bottleneck, not LoRA variant)
    
    PRACTICAL ADVICE:
    
    1. For MOST users: Standard LoRA is sufficient. The gains from
       variants are typically 1-3%, not transformative.
    
    2. If you have time to experiment: Try DoRA first (easy to add,
       consistently improves quality).
    
    3. LoRA+ is FREE — just modify optimizer groups. Always worth trying.
    
    4. AdaLoRA makes sense when you have many heterogeneous layers
       (e.g., models with both attention and MLP adapters).
    
    5. VeRA is a research technique best suited for very specific
       deployment scenarios (many adapters, tiny storage).
    """)


# ===========================================================================
# 9. PRACTICAL IMPLEMENTATION WITH PEFT
# ===========================================================================

def peft_variant_configs():
    """
    Show how to use LoRA variants with HuggingFace PEFT.
    """
    print("\n" + "=" * 70)
    print("USING VARIANTS WITH HuggingFace PEFT")
    print("=" * 70)
    
    print("""
    # Standard LoRA
    from peft import LoraConfig
    config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear")
    
    # rsLoRA (available in PEFT >= 0.7.0)
    config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules="all-linear",
        use_rslora=True,           # ← KEY: enable rsLoRA scaling
    )
    
    # DoRA (available in PEFT >= 0.8.0)
    config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules="all-linear",
        use_dora=True,             # ← KEY: enable DoRA
    )
    
    # AdaLoRA
    from peft import AdaLoraConfig
    config = AdaLoraConfig(
        init_r=64,                   # Initial rank (will be pruned)
        target_r=16,                 # Target average rank after pruning
        lora_alpha=64,
        target_modules="all-linear",
        total_step=1000,             # Total training steps
        warmup_step=100,             # Steps before pruning starts
    )
    
    # LoRA+ (optimizer-level, works with any LoRA config)
    model = get_peft_model(base_model, lora_config)
    
    # Separate parameter groups
    params_A = [p for n, p in model.named_parameters() 
                if "lora_A" in n and p.requires_grad]
    params_B = [p for n, p in model.named_parameters() 
                if "lora_B" in n and p.requires_grad]
    params_other = [p for n, p in model.named_parameters() 
                    if "lora_" not in n and p.requires_grad]
    
    optimizer = torch.optim.AdamW([
        {"params": params_A, "lr": 1e-4},
        {"params": params_B, "lr": 1e-4 * 16},  # 16x for LoRA+
        {"params": params_other, "lr": 1e-4},
    ])
    """)


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("LoRA VARIANTS AND EXTENSIONS")
    print("=" * 70)
    
    # Demo each variant
    d_in, d_out, rank = 768, 768, 16
    x = torch.randn(2, 10, d_in)
    
    variants = {
        "Standard LoRA": lambda: LoRAPlusLinear(d_in, d_out, rank, alpha=32),
        "rsLoRA": lambda: RsLoRALinear(d_in, d_out, rank, alpha=32),
        "DoRA": lambda: DoRALinear(d_in, d_out, rank, alpha=32),
        "AdaLoRA": lambda: AdaLoRALinear(d_in, d_out, rank=64, alpha=128),
        "DyLoRA": lambda: DyLoRALinear(d_in, d_out, max_rank=32, alpha=64),
        "LoRA-FA": lambda: LoRAFALinear(d_in, d_out, rank, alpha=32),
        "VeRA": lambda: VeRALinear(d_in, d_out, rank=256, alpha=512),
    }
    
    print(f"\nForward pass test (d_in={d_in}, d_out={d_out}, input={x.shape}):\n")
    
    for name, create_fn in variants.items():
        layer = create_fn()
        output = layer(x)
        
        trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        total = sum(p.numel() for p in layer.parameters())
        
        print(f"  {name:<15}  output={output.shape}  "
              f"trainable={trainable:>8,}  total={total:>10,}")
    
    print_variant_comparison()
    print_selection_guide()
    peft_variant_configs()
