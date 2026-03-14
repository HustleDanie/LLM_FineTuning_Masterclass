"""
LoRA Rank Analysis
==================

Deep dive into the most important LoRA hyperparameter: the rank r.

This module covers:
1. How rank affects model quality and efficiency
2. Per-layer rank importance analysis
3. Rank search strategies (grid search, adaptive)
4. Practical rank selection guidelines
5. Relationship between rank and task difficulty

The rank r determines the expressiveness of the LoRA update:
- Too low: ΔW can't capture the needed adaptation → underfitting
- Too high: Wastes parameters, slower training, potential overfitting
- Sweet spot: Depends on task, model, and data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


# ===========================================================================
# 1. RANK AND APPROXIMATION QUALITY
# ===========================================================================

def demonstrate_rank_quality_tradeoff():
    """
    Show how rank affects the quality of weight update approximation.
    
    For a given target update ΔW, we decompose it at different ranks 
    and measure the reconstruction error.
    """
    print("=" * 70)
    print("RANK vs QUALITY TRADEOFF")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Simulate weight updates of different types
    d = 768  # Typical hidden dim
    
    scenarios = {
        "Easy task (strong low-rank structure)": _create_easy_update(d),
        "Medium task (moderate low-rank)": _create_medium_update(d),
        "Hard task (weak low-rank)": _create_hard_update(d),
    }
    
    ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    for scenario_name, delta_W in scenarios.items():
        print(f"\n  {scenario_name}:")
        print(f"  {'Rank':>6} {'Params':>10} {'Rel Error':>12} {'Recovered':>10}")
        print(f"  " + "-" * 42)
        
        U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
        total_energy = (S ** 2).sum().item()
        total_params = d * d
        
        for r in ranks:
            # Best rank-r approximation via SVD
            delta_W_r = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
            
            # Reconstruction error
            error = torch.norm(delta_W - delta_W_r, p='fro').item()
            rel_error = error / torch.norm(delta_W, p='fro').item()
            
            # Energy captured
            energy = (S[:r] ** 2).sum().item() / total_energy * 100
            
            # LoRA params
            lora_params = 2 * d * r
            
            print(f"  {r:>6} {lora_params:>10,} {rel_error:>12.6f} {energy:>9.2f}%")
    
    print(f"\n KEY INSIGHTS:")
    print(f"  • Easy tasks: r=4-8 recovers >95% → use low rank!")
    print(f"  • Medium tasks: r=16-32 needed for >95% → typical choice")
    print(f"  • Hard tasks: r=64-128 needed → consider full fine-tuning")
    print(f"  • Diminishing returns: going from r=64 to r=128 helps little")


def _create_easy_update(d):
    """Strongly low-rank: e.g., similar domain fine-tuning."""
    signal = torch.randn(d, 4) @ torch.randn(4, d) * 0.01
    noise = torch.randn(d, d) * 0.0001
    return signal + noise


def _create_medium_update(d):
    """Moderately low-rank: e.g., instruction tuning."""
    signal = torch.randn(d, 32) @ torch.randn(32, d) * 0.005
    noise = torch.randn(d, d) * 0.001
    return signal + noise


def _create_hard_update(d):
    """Weakly low-rank: e.g., cross-lingual or code adaptation."""
    signal = torch.randn(d, 128) @ torch.randn(128, d) * 0.003
    noise = torch.randn(d, d) * 0.002
    return signal + noise


# ===========================================================================
# 2. PER-LAYER RANK IMPORTANCE
# ===========================================================================

def analyze_per_layer_importance():
    """
    Not all layers need the same rank!
    
    Analysis shows that:
    - Attention layers typically need higher rank than MLP layers
    - Lower layers (closer to input) often need less rank
    - Upper layers (closer to output) may need more rank
    - The output projection often benefits from higher rank
    
    This motivates adaptive rank methods like AdaLoRA.
    """
    print("\n" + "=" * 70)
    print("PER-LAYER RANK IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        
        print(f"\nModel: distilgpt2")
        print(f"\nSVD analysis of each weight matrix — rank needed for 90% energy:")
        print(f"(Lower rank needed → easier for LoRA to approximate)\n")
        
        layer_info = []
        
        for name, param in model.named_parameters():
            if param.dim() == 2 and min(param.shape) >= 32:
                W = param.data.float()
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                total = (S ** 2).sum().item()
                cum = torch.cumsum(S ** 2, dim=0) / total
                
                r90 = (cum < 0.90).sum().item() + 1
                r95 = (cum < 0.95).sum().item() + 1
                
                # Spectral gap: ratio of σ_r to σ_1 at r=16
                r_test = min(16, len(S) - 1)
                gap = (S[r_test] / S[0]).item() if S[0] > 0 else 0
                
                layer_info.append({
                    "name": name,
                    "shape": tuple(param.shape),
                    "r90": r90,
                    "r95": r95,
                    "spectral_gap_r16": gap,
                    "top_sv": S[0].item(),
                })
        
        # Group by layer type
        print(f"{'Layer Name':<45} {'Shape':>14} {'r(90%)':>7} {'r(95%)':>7} {'Gap@16':>8}")
        print("-" * 85)
        
        for info in layer_info:
            print(f"{info['name']:<45} {str(info['shape']):>14} "
                  f"{info['r90']:>7} {info['r95']:>7} "
                  f"{info['spectral_gap_r16']:>8.4f}")
        
        # Summary by layer type
        print(f"\n  Summary by layer type:")
        
        attn_r90 = [i["r90"] for i in layer_info if "attn" in i["name"]]
        mlp_r90 = [i["r90"] for i in layer_info if "mlp" in i["name"] or "c_fc" in i["name"]]
        
        if attn_r90:
            print(f"    Attention layers: avg r(90%) = {sum(attn_r90)/len(attn_r90):.0f}")
        if mlp_r90:
            print(f"    MLP layers:       avg r(90%) = {sum(mlp_r90)/len(mlp_r90):.0f}")
        
        print(f"\n  RECOMMENDATION:")
        print(f"    • Use the same rank for simplicity, OR")
        print(f"    • Use AdaLoRA to automatically distribute rank budget")
        
    except ImportError:
        print("  [Install transformers to run per-layer analysis]")
        _simulate_per_layer_analysis()


def _simulate_per_layer_analysis():
    """Simulated version of per-layer analysis."""
    print("\n  Simulated per-layer rank importance:")
    
    layers = [
        ("Layer 0 - Q/K/V projection", (768, 2304), 45, 89),
        ("Layer 0 - Output projection", (768, 768), 32, 67),
        ("Layer 0 - MLP up", (3072, 768), 52, 98),
        ("Layer 0 - MLP down", (768, 3072), 48, 91),
        ("Layer 3 - Q/K/V projection", (768, 2304), 41, 82),
        ("Layer 3 - Output projection", (768, 768), 28, 58),
        ("Layer 3 - MLP up", (3072, 768), 55, 103),
        ("Layer 3 - MLP down", (768, 3072), 50, 95),
        ("Layer 5 - Q/K/V projection", (768, 2304), 38, 76),
        ("Layer 5 - Output projection", (768, 768), 25, 51),
    ]
    
    print(f"  {'Layer':<35} {'Shape':>14} {'r(90%)':>8} {'r(95%)':>8}")
    print("  " + "-" * 68)
    for name, shape, r90, r95 in layers:
        print(f"  {name:<35} {str(shape):>14} {r90:>8} {r95:>8}")


# ===========================================================================
# 3. RANK SEARCH STRATEGIES
# ===========================================================================

@dataclass
class RankSearchResult:
    """Result of a rank search experiment."""
    rank: int
    trainable_params: int
    loss: float
    perplexity: float
    training_time_relative: float  # Relative to r=1


def grid_search_rank(
    model_name: str = "distilgpt2",
    ranks: List[int] = [2, 4, 8, 16, 32, 64],
    alpha_ratio: float = 2.0,
    num_train_steps: int = 50,
):
    """
    Grid search over LoRA ranks to find the optimal value.
    
    Strategy: Train with each rank for a fixed number of steps,
    then compare validation loss.
    
    Parameters:
    -----------
    ranks : list of int
        Rank values to try
    alpha_ratio : float
        Set alpha = alpha_ratio * rank (consistent scaling)
    num_train_steps : int
        Training steps per rank (keep small for quick search)
    """
    print("\n" + "=" * 70)
    print("GRID SEARCH OVER LoRA RANKS")
    print("=" * 70)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  [Install transformers to run grid search]")
        _simulate_grid_search(ranks)
        return
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare evaluation data
    eval_text = (
        "The transformer architecture uses self-attention mechanisms "
        "to process sequences in parallel. This enables efficient "
        "training on large datasets. Fine-tuning adapts pre-trained "
        "models to specific downstream tasks."
    )
    eval_tokens = tokenizer(eval_text, return_tensors="pt", truncation=True, max_length=64)
    eval_ids = eval_tokens["input_ids"]
    eval_labels = eval_ids.clone()
    
    # Training data
    train_texts = [
        "LoRA enables efficient fine-tuning by decomposing weight updates.",
        "The rank determines the expressiveness of the adaptation.",
        "Lower ranks use fewer parameters but may underfit complex tasks.",
        "Higher ranks capture more information but risk overfitting.",
    ]
    train_tokens = tokenizer(train_texts, return_tensors="pt", padding=True, 
                             truncation=True, max_length=64)
    train_ids = train_tokens["input_ids"]
    train_mask = train_tokens["attention_mask"]
    train_labels = train_ids.clone()
    train_labels[train_mask == 0] = -100
    
    results: List[RankSearchResult] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n  Model: {model_name}")
    print(f"  Ranks to search: {ranks}")
    print(f"  Steps per rank: {num_train_steps}")
    print(f"  Alpha ratio: {alpha_ratio} (α = {alpha_ratio} × r)")
    print()
    
    for rank in ranks:
        # Fresh model for each rank
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        alpha = alpha_ratio * rank
        
        # Count trainable params
        total_lora_params = 0
        for name, param in model.named_parameters():
            param.requires_grad = False
        
        # Note: For a proper implementation, we'd use LoRAModel here
        # but for speed we'll use a simplified approach
        from lora_from_scratch import LoRAModel
        lora_model = LoRAModel(
            model=model,
            target_modules=["c_attn", "c_proj"],
            rank=rank,
            alpha=alpha,
        )
        
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in trainable)
        
        optimizer = torch.optim.AdamW(trainable, lr=3e-4)
        
        # Train
        model.train()
        train_ids_dev = train_ids.to(device)
        train_mask_dev = train_mask.to(device)
        train_labels_dev = train_labels.to(device)
        
        for step in range(num_train_steps):
            optimizer.zero_grad()
            outputs = model(
                input_ids=train_ids_dev,
                attention_mask=train_mask_dev,
                labels=train_labels_dev,
            )
            outputs.loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            eval_out = model(input_ids=eval_ids.to(device), labels=eval_labels.to(device))
            eval_loss = eval_out.loss.item()
            eval_ppl = math.exp(min(eval_loss, 20))  # Cap to avoid overflow
        
        result = RankSearchResult(
            rank=rank,
            trainable_params=n_params,
            loss=eval_loss,
            perplexity=eval_ppl,
            training_time_relative=rank / ranks[0],  # Approx
        )
        results.append(result)
        
        print(f"  r={rank:>3}, α={alpha:>4.0f}: "
              f"params={n_params:>8,}  loss={eval_loss:.4f}  ppl={eval_ppl:.2f}")
        
        del model, optimizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Find optimal rank
    best = min(results, key=lambda x: x.loss)
    print(f"\n  Best rank: r={best.rank} (loss={best.loss:.4f}, ppl={best.perplexity:.2f})")
    
    # Efficiency analysis
    print(f"\n  Efficiency analysis (quality per parameter):")
    baseline_loss = results[-1].loss  # Highest rank as baseline
    for r in results:
        efficiency = (1 / max(r.loss, 0.01)) / r.trainable_params * 1e6
        print(f"    r={r.rank:>3}: efficiency score = {efficiency:.4f}")
    
    return results


def _simulate_grid_search(ranks):
    """Simulated grid search results."""
    print(f"\n  Simulated grid search results (distilgpt2):")
    print(f"  {'Rank':>6} {'Params':>10} {'Loss':>8} {'PPL':>8} {'Status':>10}")
    print("  " + "-" * 46)
    
    # Typical pattern: diminishing returns
    simulated = {
        2:  (12288,  4.82, 123.9, "underfit"),
        4:  (24576,  4.41,  82.1, ""),
        8:  (49152,  4.15,  63.4, ""),
        16: (98304,  3.98,  53.5, "good"),
        32: (196608, 3.92,  50.4, "best"),
        64: (393216, 3.91,  49.9, "diminishing"),
    }
    
    for r in ranks:
        if r in simulated:
            params, loss, ppl, status = simulated[r]
            print(f"  {r:>6} {params:>10,} {loss:>8.2f} {ppl:>8.1f} {status:>10}")


# ===========================================================================
# 4. ADAPTIVE RANK ALLOCATION
# ===========================================================================

class AdaptiveRankAllocator:
    """
    Allocate LoRA rank budget across layers based on importance.
    
    Instead of using the same rank for every layer, distribute a
    total rank budget based on which layers benefit most.
    
    Methods:
    1. Gradient-based: Layers with larger gradients get more rank
    2. SVD-based: Layers where ΔW has slower spectral decay get more rank
    3. Sensitivity-based: Layers where rank changes affect loss most
    
    This is conceptually similar to AdaLoRA but implemented as a
    pre-training analysis step.
    """
    
    def __init__(self, model: nn.Module, target_modules: List[str]):
        self.model = model
        self.target_modules = target_modules
        self._target_layers: Dict[str, nn.Linear] = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if any(t in name for t in target_modules):
                    self._target_layers[name] = module
    
    def allocate_by_parameter_count(
        self,
        total_rank_budget: int,
        min_rank: int = 2,
        max_rank: int = 128,
    ) -> Dict[str, int]:
        """
        Allocate rank proportional to layer size.
        
        Larger layers get proportionally more rank to maintain
        a similar ratio of LoRA params to original params.
        
        Parameters:
        -----------
        total_rank_budget : int
            Sum of all ranks across layers
        min_rank : int
            Minimum rank for any layer
        max_rank : int
            Maximum rank for any layer
        """
        # Calculate total weight parameters
        layer_sizes = {}
        total_size = 0
        for name, layer in self._target_layers.items():
            size = layer.weight.numel()
            layer_sizes[name] = size
            total_size += size
        
        # Distribute rank proportionally
        allocations = {}
        for name, size in layer_sizes.items():
            raw_rank = total_rank_budget * (size / total_size)
            rank = int(max(min_rank, min(max_rank, round(raw_rank))))
            allocations[name] = rank
        
        # Adjust to meet budget
        allocated = sum(allocations.values())
        if allocated != total_rank_budget:
            # Simple adjustment: scale largest allocations
            diff = total_rank_budget - allocated
            sorted_layers = sorted(allocations.items(), key=lambda x: -x[1])
            for i, (name, rank) in enumerate(sorted_layers):
                if diff == 0:
                    break
                adjustment = 1 if diff > 0 else -1
                new_rank = max(min_rank, min(max_rank, rank + adjustment))
                if new_rank != rank:
                    allocations[name] = new_rank
                    diff -= adjustment
        
        return allocations
    
    def allocate_by_gradient_importance(
        self,
        total_rank_budget: int,
        dataloader,
        num_batches: int = 10,
        min_rank: int = 2,
        max_rank: int = 128,
    ) -> Dict[str, int]:
        """
        Allocate rank based on gradient magnitudes.
        
        Layers with larger gradients during initial training are more
        important and should receive more rank.
        
        This requires a forward-backward pass on some training data.
        """
        # Temporarily enable gradients on all target layers
        original_requires_grad = {}
        for name, layer in self._target_layers.items():
            original_requires_grad[name] = layer.weight.requires_grad
            layer.weight.requires_grad_(True)
        
        # Accumulate gradient norms
        grad_norms = {name: 0.0 for name in self._target_layers}
        
        self.model.train()
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            for name, layer in self._target_layers.items():
                if layer.weight.grad is not None:
                    grad_norms[name] += layer.weight.grad.norm().item()
            
            self.model.zero_grad()
        
        # Restore original requires_grad
        for name, layer in self._target_layers.items():
            layer.weight.requires_grad_(original_requires_grad[name])
        
        # Normalize gradient norms
        total_norm = sum(grad_norms.values())
        if total_norm == 0:
            # Fallback to uniform
            rank_per_layer = total_rank_budget // len(self._target_layers)
            return {name: rank_per_layer for name in self._target_layers}
        
        importance = {name: norm / total_norm for name, norm in grad_norms.items()}
        
        # Allocate proportionally
        allocations = {}
        for name, imp in importance.items():
            raw_rank = total_rank_budget * imp
            rank = int(max(min_rank, min(max_rank, round(raw_rank))))
            allocations[name] = rank
        
        return allocations
    
    @staticmethod
    def print_allocation(allocations: Dict[str, int]):
        """Pretty-print rank allocations."""
        total_rank = sum(allocations.values())
        print(f"\n  Rank Allocation (total budget: {total_rank}):")
        print(f"  {'Layer':<45} {'Rank':>6}")
        print("  " + "-" * 53)
        for name, rank in allocations.items():
            bar = "█" * (rank // 2) + "▌" * (rank % 2)
            print(f"  {name:<45} {rank:>6}  {bar}")


# ===========================================================================
# 5. RANK SELECTION GUIDELINES
# ===========================================================================

def print_rank_selection_guide():
    """
    Comprehensive guide for selecting LoRA rank based on your use case.
    """
    print("\n" + "=" * 70)
    print("LoRA RANK SELECTION GUIDE")
    print("=" * 70)
    
    guide = """
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    RANK SELECTION DECISION TREE                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  What is your task type?                                           │
    │                                                                     │
    │  ├── Similar to pre-training (e.g., chat in same language)         │
    │  │   └── r = 4-16  (low rank sufficient)                          │
    │  │                                                                  │
    │  ├── Moderate shift (e.g., general → instruction following)        │
    │  │   └── r = 16-64  (most common choice)                          │
    │  │                                                                  │
    │  ├── Large shift (e.g., English → code, or new domain)            │
    │  │   └── r = 64-128  (needs more expressiveness)                  │
    │  │                                                                  │
    │  └── Extreme shift (e.g., language → math reasoning)              │
    │      └── r = 128-256 or consider full fine-tuning                 │
    │                                                                     │
    │  What is your model size?                                          │
    │                                                                     │
    │  ├── < 1B parameters                                               │
    │  │   └── r = 4-16  (small models need less rank)                  │
    │  │                                                                  │
    │  ├── 1B - 7B parameters                                           │
    │  │   └── r = 8-32  (most common range)                            │
    │  │                                                                  │
    │  ├── 7B - 13B parameters                                          │
    │  │   └── r = 16-64  (larger models can use more rank)             │
    │  │                                                                  │
    │  └── > 13B parameters                                              │
    │      └── r = 32-128  (but often r=16-32 is sufficient!)           │
    │                                                                     │
    │  What are your constraints?                                        │
    │                                                                     │
    │  ├── Memory limited (consumer GPU)                                 │
    │  │   └── r = 4-8  (minimize overhead)                             │
    │  │                                                                  │
    │  ├── Balanced (A100/H100)                                          │
    │  │   └── r = 16-64  (quality/efficiency sweet spot)               │
    │  │                                                                  │
    │  └── Quality is paramount (compute available)                      │
    │      └── r = 64-128  (maximize quality)                           │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

    COMMON RECIPES:
    
    ┌──────────────────────────────┬──────┬───────┬────────────────────────┐
    │ Use Case                     │ Rank │ Alpha │ Notes                  │
    ├──────────────────────────────┼──────┼───────┼────────────────────────┤
    │ Chat fine-tuning (7B)        │   16 │    32 │ Most popular config    │
    │ Instruction tuning (7B)      │   32 │    64 │ Slightly more capacity │
    │ Code fine-tuning (7B)        │   64 │   128 │ Code needs more rank   │
    │ Classification (BERT-base)   │    8 │    16 │ Simple task, low rank  │
    │ Translation (7B)             │   32 │    64 │ Moderate domain shift  │
    │ Math reasoning (7B)          │  128 │   256 │ Complex task           │
    │ QLoRA on consumer GPU        │    8 │    16 │ Memory-optimized       │
    │ Multi-task (shared adapter)  │   64 │   128 │ Needs more capacity    │
    └──────────────────────────────┴──────┴───────┴────────────────────────┘

    RANK SELECTION RULES OF THUMB:
    
    1. START with r=16 — it works well for most tasks
    2. If loss plateaus early → increase rank (underfitting LoRA capacity)
    3. If validation loss rises → decrease rank (overfitting)
    4. Double the rank → ~doubles LoRA params but <5% more training time
    5. When in doubt, r=32 with α=64 is a safe choice for 7B+ models
    6. For QLoRA (4-bit), use r=16-64 (higher rank compensates for quantization)
    
    ALPHA SELECTION:
    
    • Default: α = 2 × r  (most common, recommended starting point)
    • Conservative: α = r  (smaller updates, more stable)
    • Aggressive: α = 4 × r  (larger updates, faster but riskier)
    • Fixed α (Hu et al.): α = 16 for all ranks (original paper approach)
      Warning: this means effective scaling changes with r!
    """
    print(guide)


# ===========================================================================
# 6. RANK AND EFFECTIVE UPDATE MAGNITUDE
# ===========================================================================

def analyze_rank_and_magnitude():
    """
    Analyze how rank and alpha interact to determine the effective
    magnitude of weight updates.
    
    This is important because:
    1. The update magnitude affects training stability
    2. Larger updates → need smaller learning rate
    3. The α/r scaling is designed to decouple rank from update magnitude
    """
    print("\n" + "=" * 70)
    print("RANK, ALPHA, AND UPDATE MAGNITUDE")
    print("=" * 70)
    
    torch.manual_seed(42)
    d = 768
    
    print(f"\n  Matrix size: {d}×{d}")
    print(f"\n  After 1 training step (simulated):")
    print(f"  {'Rank':>6} {'Alpha':>6} {'α/r':>6} {'||ΔW||_F':>12} {'||ΔW||/||W||':>14}")
    print("  " + "-" * 48)
    
    W = torch.randn(d, d) * 0.02  # Pre-trained weight
    W_norm = W.norm().item()
    
    configs = [
        (4, 8), (4, 16), (4, 32),     # Fixed low rank, varying alpha
        (16, 16), (16, 32), (16, 64),  # Fixed medium rank, varying alpha
        (64, 64), (64, 128), (64, 256),# Fixed high rank, varying alpha
        # Constant alpha/r = 2:
        (4, 8), (8, 16), (16, 32), (32, 64), (64, 128),
    ]
    
    seen = set()
    for r, alpha in configs:
        if (r, alpha) in seen:
            continue
        seen.add((r, alpha))
        
        scaling = alpha / r
        
        # Simulate trained LoRA weights
        A = torch.randn(r, d) * (1.0 / r**0.5)
        B = torch.randn(d, r) * 0.01  # After some training
        
        delta_W = scaling * (B @ A)
        delta_norm = delta_W.norm().item()
        ratio = delta_norm / W_norm
        
        print(f"  {r:>6} {alpha:>6} {scaling:>6.2f} {delta_norm:>12.6f} {ratio:>14.6f}")
    
    print(f"\n  OBSERVATIONS:")
    print(f"  • When α/r is constant, update magnitude is roughly constant across ranks")
    print(f"  • This means ONE learning rate works across different rank choices!")
    print(f"  • Higher α/r → larger updates → may need lower learning rate")
    print(f"  • The HuggingFace PEFT default: α = r (scaling = 1.0)")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    demonstrate_rank_quality_tradeoff()
    analyze_per_layer_importance()
    print_rank_selection_guide()
    analyze_rank_and_magnitude()
    
    # Uncomment to run grid search (requires transformers):
    # grid_search_rank()
    
    print("\n" + "=" * 70)
    print("RANK ANALYSIS SUMMARY")
    print("=" * 70)
    print("""
    1. Rank r is the most important LoRA hyperparameter
    2. r=16 is a strong default for most tasks and model sizes
    3. Not all layers need the same rank — adaptive allocation helps
    4. The α/r scaling decouples rank from update magnitude
    5. Grid search with 5 rank values takes ~5x a single run
    6. Diminishing returns above r=64 for most tasks
    7. When quality is critical: increase rank and target more modules
    """)
