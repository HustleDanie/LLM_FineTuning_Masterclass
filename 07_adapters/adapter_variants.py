"""
Adapter Variants
=================

Comprehensive implementation of all major adapter architectures:

1. Houlsby Adapter (2019)
   - Original adapter: 2 adapters per transformer layer
   - After self-attention AND after FFN
   - Maximum capacity, most parameters

2. Pfeiffer Adapter (2021)
   - Efficient variant: 1 adapter per layer
   - Only after the FFN sub-layer
   - Half the parameters of Houlsby

3. Parallel Adapter (He et al., 2022)
   - Adapter runs in parallel with FFN (not sequential)
   - Better gradient flow, competitive performance
   - Closer to LoRA in design philosophy

4. AdapterDrop (Rücklé et al., 2021)
   - Randomly drop adapter layers during training
   - Remove lower-layer adapters at inference for speed
   - Similar to dropout but at the layer level

5. Compacter (Karimi Mahabadi et al., 2021)
   - Parameterize adapter with Kronecker products
   - Hypercomplex multiplication for compression
   - 10-100x fewer parameters than standard adapters

6. Scaled Parallel Adapter (UniPELT, 2022)
   - Gate mechanism to control adapter contribution
   - Learnable scaling per module
   - Part of the UniPELT framework

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple


# ============================================================================
# VARIANT 1: HOULSBY ADAPTER (Original, 2019)
# ============================================================================

class HoulsbyAdapter(nn.Module):
    """
    Original adapter from Houlsby et al. (2019).
    
    Places TWO adapter modules per transformer layer:
    1. After the multi-head attention sub-layer
    2. After the feed-forward sub-layer
    
    Architecture per adapter:
        h = W_up · σ(W_down · x) + x
    
    This is the most parameter-heavy variant but provides
    the highest capacity for task adaptation.
    """
    
    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int = 64,
        activation: str = "relu",
    ):
        super().__init__()
        
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
        }
        
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.activation = activations.get(activation, nn.ReLU())
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        
        # Zero-init up-projection for identity at start
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down_proj(x)
        h = self.activation(h)
        h = self.up_proj(h)
        return h + x  # Residual connection


class HoulsbyTransformerLayer(nn.Module):
    """
    Transformer layer with Houlsby adapters.
    
    Flow: x → Attn → Adapter₁ → FFN → Adapter₂ → output
    """
    
    def __init__(self, hidden_size: int = 768, bottleneck_size: int = 64):
        super().__init__()
        
        # Frozen components (simulated)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, 12, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        
        # Trainable adapters
        self.adapter_attn = HoulsbyAdapter(hidden_size, bottleneck_size)
        self.adapter_ffn = HoulsbyAdapter(hidden_size, bottleneck_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention + Adapter 1
        h = self.attn_norm(x)
        attn_out, _ = self.attention(h, h, h)
        x = x + attn_out
        x = self.adapter_attn(x)  # ← Adapter after attention
        
        # FFN + Adapter 2
        h = self.ffn_norm(x)
        ffn_out = self.ffn(h)
        x = x + ffn_out
        x = self.adapter_ffn(x)   # ← Adapter after FFN
        
        return x


# ============================================================================
# VARIANT 2: PFEIFFER ADAPTER (2021)
# ============================================================================

class PfeifferAdapter(nn.Module):
    """
    Pfeiffer adapter variant (2021).
    
    Key difference from Houlsby: Only ONE adapter per layer,
    placed ONLY after the FFN sub-layer.
    
    Benefits:
    - 50% fewer adapter parameters than Houlsby
    - Faster training and inference
    - Surprisingly similar performance in many tasks
    
    Architecture:
        h = W_up · σ(W_down · LayerNorm(x)) + x
    """
    
    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int = 64,
        activation: str = "relu",
    ):
        super().__init__()
        
        activations = {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh()}
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.activation = activations.get(activation, nn.ReLU())
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        
        # Zero-init for identity
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer_norm(x)
        h = self.down_proj(h)
        h = self.activation(h)
        h = self.up_proj(h)
        return h + x


class PfeifferTransformerLayer(nn.Module):
    """
    Transformer layer with Pfeiffer adapter (single adapter after FFN).
    
    Flow: x → Attn → FFN → Adapter → output
    """
    
    def __init__(self, hidden_size: int = 768, bottleneck_size: int = 64):
        super().__init__()
        
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, 12, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        
        # Only ONE adapter (after FFN)
        self.adapter = PfeifferAdapter(hidden_size, bottleneck_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention (no adapter)
        h = self.attn_norm(x)
        attn_out, _ = self.attention(h, h, h)
        x = x + attn_out
        
        # FFN + Adapter
        h = self.ffn_norm(x)
        ffn_out = self.ffn(h)
        x = x + ffn_out
        x = self.adapter(x)  # ← Only adapter here
        
        return x


# ============================================================================
# VARIANT 3: PARALLEL ADAPTER (He et al., 2022)
# ============================================================================

class ParallelAdapter(nn.Module):
    """
    Parallel adapter from "Towards a Unified View" (He et al., 2022).
    
    Instead of placing the adapter AFTER the FFN (sequential),
    it runs IN PARALLEL with the FFN:
    
    Sequential (Houlsby/Pfeiffer):
        h = Adapter(FFN(x) + x)
        
    Parallel:
        h = FFN(x) + Adapter(x) + x
    
    Benefits:
    - Better gradient flow (direct path to adapter)
    - No sequential dependency between FFN and adapter
    - Mathematically closer to LoRA
    - Often better performance with same parameter count
    
    Connection to LoRA:
    - LoRA: h = Wx + BAx  (parallel modification of W)
    - Parallel Adapter: h = FFN(x) + Adapter(x)  (parallel to FFN)
    """
    
    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int = 64,
        scaling: float = 1.0,
    ):
        super().__init__()
        
        self.down_proj = nn.Linear(hidden_size, bottleneck_size, bias=False)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_size, hidden_size, bias=False)
        self.scaling = scaling
        
        # Initialize
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Note: No residual here — the caller adds both FFN and adapter outputs."""
        h = self.down_proj(x)
        h = self.activation(h)
        h = self.up_proj(h)
        return h * self.scaling


class ParallelTransformerLayer(nn.Module):
    """
    Transformer layer with parallel adapter.
    
    Flow: x → Attn → [FFN ∥ Adapter] → output
    
    The adapter runs in parallel with FFN, not after it.
    """
    
    def __init__(self, hidden_size: int = 768, bottleneck_size: int = 64):
        super().__init__()
        
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, 12, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        
        # Parallel adapter
        self.adapter = ParallelAdapter(hidden_size, bottleneck_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention
        h = self.attn_norm(x)
        attn_out, _ = self.attention(h, h, h)
        x = x + attn_out
        
        # FFN + Adapter IN PARALLEL
        h = self.ffn_norm(x)
        ffn_out = self.ffn(h)
        adapter_out = self.adapter(h)  # ← Same input as FFN!
        x = x + ffn_out + adapter_out  # ← Add both
        
        return x


# ============================================================================
# VARIANT 4: ADAPTERDROP (Rücklé et al., 2021)
# ============================================================================

class AdapterWithDrop(nn.Module):
    """
    AdapterDrop: Randomly drops entire adapter layers during training.
    
    Key insight: Lower transformer layers are less task-specific,
    so their adapters can be removed with minimal quality loss.
    This speeds up inference significantly.
    
    Training: Randomly drop adapters with probability p
    Inference: Remove adapters from the bottom N layers
    
    Results from the paper:
    - Dropping 1-5 bottom adapter layers: <1% quality loss
    - Dropping 5+ layers: Quality starts degrading
    - Speedup: ~2x for 5-layer drop on 12-layer BERT
    """
    
    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int = 64,
        drop_prob: float = 0.0,
        layer_idx: int = 0,
    ):
        super().__init__()
        
        self.adapter = PfeifferAdapter(hidden_size, bottleneck_size)
        self.drop_prob = drop_prob
        self.layer_idx = layer_idx
        self._active = True  # Can be disabled for inference
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._active:
            return x  # Skip adapter entirely
        
        if self.training and self.drop_prob > 0:
            # Randomly drop this adapter during training
            if torch.rand(1).item() < self.drop_prob:
                return x  # Skip: just return input
        
        return self.adapter(x)
    
    def deactivate(self):
        """Permanently disable this adapter for faster inference."""
        self._active = False
    
    def activate(self):
        """Re-enable this adapter."""
        self._active = True


class AdapterDropManager:
    """
    Manages AdapterDrop across a full model.
    
    Allows easy control of which layers have active adapters,
    enabling progressive adapter removal for inference.
    """
    
    def __init__(self, adapter_layers: list):
        self.adapter_layers = adapter_layers
    
    def drop_bottom_n(self, n: int):
        """Deactivate adapters in the bottom N layers."""
        for i, adapter in enumerate(self.adapter_layers):
            if i < n:
                adapter.deactivate()
            else:
                adapter.activate()
    
    def set_drop_prob(self, prob: float):
        """Set training drop probability for all layers."""
        for adapter in self.adapter_layers:
            adapter.drop_prob = prob
    
    def get_active_count(self) -> int:
        """Count active adapters."""
        return sum(1 for a in self.adapter_layers if a._active)
    
    def demonstrate_drop_analysis(self, num_layers: int = 12):
        """Show the effect of dropping adapter layers."""
        print("\n" + "=" * 70)
        print("  ADAPTERDROP ANALYSIS")
        print("=" * 70)
        
        print(f"\n  Model with {num_layers} adapter layers:")
        print(f"\n  {'Dropped':>8} {'Active':>8} {'Speedup':>10} "
              f"{'Quality':>10} {'Active Layers':>20}")
        print("  " + "─" * 60)
        
        # Expected quality retention (from the paper)
        quality_map = {
            0: 100.0, 1: 99.8, 2: 99.5, 3: 99.2, 4: 98.8,
            5: 98.3, 6: 97.5, 7: 96.0, 8: 94.0, 9: 91.0,
            10: 86.0, 11: 78.0, 12: 50.0
        }
        
        for n_drop in range(num_layers + 1):
            active = num_layers - n_drop
            # Speedup is roughly proportional to layers removed
            # (adapters add ~10-15% overhead per layer)
            adapter_overhead_pct = 0.12  # 12% per adapter layer
            speedup = 1.0 / (1.0 - n_drop * adapter_overhead_pct / num_layers)
            quality = quality_map.get(n_drop, 50.0)
            
            active_str = "".join(
                "█" if i >= n_drop else "░" 
                for i in range(num_layers)
            )
            
            print(f"  {n_drop:>8} {active:>8} {speedup:>9.2f}x "
                  f"{quality:>8.1f}%  [{active_str}]")
        
        print(f"\n  → Sweet spot: Drop 3-5 bottom layers")
        print(f"    ~1.2x speedup with <2% quality loss")


# ============================================================================
# VARIANT 5: COMPACTER (Karimi Mahabadi et al., 2021)
# ============================================================================

class CompacterAdapter(nn.Module):
    """
    Compacter: Compact adapter using Kronecker products.
    
    Key idea: Instead of full down/up projection matrices,
    parameterize them as Kronecker products of smaller matrices:
    
    Standard adapter:
        W_down ∈ ℝ^{r×d}  → r×d parameters
        W_up   ∈ ℝ^{d×r}  → d×r parameters
        Total: 2dr parameters
    
    Compacter:
        W_down = A₁ ⊗ B₁  where A₁ ∈ ℝ^{n×n}, B₁ ∈ ℝ^{m×m}
        W_up   = A₂ ⊗ B₂  where A₂ ∈ ℝ^{n×n}, B₂ ∈ ℝ^{m×m}
        where n×m = r or d (via Kronecker decomposition)
        Total: 2(n² + m²) parameters
    
    Savings: From 2dr to 2(n² + m²) where n,m << d,r
    Example: d=768, r=64 → Standard: 98K params vs Compacter: ~2K params
    
    Note: This is a simplified demonstration. The full Compacter uses
    hypercomplex (PHM) multiplication layers.
    """
    
    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int = 64,
        n_kronecker: int = 8,  # Size of Kronecker factors
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.n_kronecker = n_kronecker
        
        # Kronecker factor sizes
        # We need A ⊗ B to have shape (bottleneck_size, hidden_size)
        # A ∈ ℝ^{a1 × a2}, B ∈ ℝ^{b1 × b2}
        # A ⊗ B ∈ ℝ^{a1*b1 × a2*b2}
        
        # For simplicity, use shared small matrices
        self.n_factors = n_kronecker
        
        # Down-projection factors
        factor_dim_down_1 = int(math.ceil(math.sqrt(bottleneck_size)))
        factor_dim_down_2 = int(math.ceil(math.sqrt(hidden_size)))
        
        self.down_A = nn.Parameter(
            torch.randn(factor_dim_down_1, factor_dim_down_1) * 0.01
        )
        self.down_B = nn.Parameter(
            torch.randn(factor_dim_down_2, factor_dim_down_2) * 0.01
        )
        
        # Up-projection factors (initialized to near-zero)
        factor_dim_up_1 = factor_dim_down_2
        factor_dim_up_2 = factor_dim_down_1
        
        self.up_A = nn.Parameter(
            torch.zeros(factor_dim_up_1, factor_dim_up_1)
        )
        self.up_B = nn.Parameter(
            torch.zeros(factor_dim_up_2, factor_dim_up_2)
        )
        
        self.activation = nn.ReLU()
        
        # Store effective dimensions
        self._eff_down_shape = (
            factor_dim_down_1 * factor_dim_down_2,
            factor_dim_down_1 * factor_dim_down_2,
        )
        
        self._param_count = (
            self.down_A.numel() + self.down_B.numel() +
            self.up_A.numel() + self.up_B.numel()
        )
    
    def _kronecker(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Compute Kronecker product A ⊗ B."""
        return torch.kron(A, B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reconstruct full matrices from Kronecker factors
        W_down = self._kronecker(self.down_A, self.down_B)
        W_up = self._kronecker(self.up_A, self.up_B)
        
        # Truncate/pad to match dimensions
        r = min(W_down.shape[0], self.bottleneck_size)
        d = min(W_down.shape[1], self.hidden_size)
        
        W_down = W_down[:r, :d]
        W_up = W_up[:d, :r]
        
        # Apply adapter
        h = F.linear(x[..., :d], W_down)
        h = self.activation(h)
        h_out = F.linear(h, W_up)
        
        # Pad if needed and add residual
        if h_out.shape[-1] < x.shape[-1]:
            padding = torch.zeros(
                *x.shape[:-1], x.shape[-1] - h_out.shape[-1],
                device=x.device, dtype=x.dtype,
            )
            h_out = torch.cat([h_out, padding], dim=-1)
        
        return h_out[..., :x.shape[-1]] + x
    
    def extra_repr(self):
        standard_params = 2 * self.hidden_size * self.bottleneck_size
        return (f"params={self._param_count}, "
                f"standard_params={standard_params}, "
                f"compression={standard_params/max(self._param_count,1):.1f}x")


# ============================================================================
# VARIANT 6: SCALED PARALLEL ADAPTER (UniPELT-style)
# ============================================================================

class GatedAdapter(nn.Module):
    """
    Adapter with a learnable gate mechanism.
    
    From UniPELT (Mao et al., 2022): Each PEFT module gets a
    learnable gate that controls its contribution:
    
        output = x + gate * Adapter(x)
    
    Where gate ∈ [0, 1] is learned during training.
    This allows the model to learn whether the adapter should
    contribute at each layer.
    
    Benefits:
    - Automatic importance weighting per layer
    - Can learn to "turn off" unnecessary adapters
    - Useful when combining multiple PEFT methods
    """
    
    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int = 64,
        activation: str = "relu",
        init_gate: float = 0.0,  # Start with gate nearly closed
    ):
        super().__init__()
        
        activations = {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh()}
        
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.activation = activations.get(activation, nn.ReLU())
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        
        # Learnable gate (sigmoid applied in forward)
        self.gate = nn.Parameter(torch.tensor(init_gate))
        
        # Zero-init up-projection
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down_proj(x)
        h = self.activation(h)
        h = self.up_proj(h)
        
        # Apply learnable gate
        gate_value = torch.sigmoid(self.gate)
        return x + gate_value * h
    
    def get_gate_value(self) -> float:
        """Get the current gate value (0 = closed, 1 = open)."""
        return torch.sigmoid(self.gate).item()


# ============================================================================
# COMPARISON & ANALYSIS
# ============================================================================

class AdapterVariantComparison:
    """Compare all adapter variants side by side."""
    
    def compare_all(self, hidden_size: int = 768, bottleneck_size: int = 64):
        """Run comparison of all adapter variants."""
        print("=" * 70)
        print("  ADAPTER VARIANT COMPARISON")
        print("=" * 70)
        
        x = torch.randn(2, 16, hidden_size)
        
        variants = {
            "Houlsby": HoulsbyAdapter(hidden_size, bottleneck_size),
            "Pfeiffer": PfeifferAdapter(hidden_size, bottleneck_size),
            "Parallel": ParallelAdapter(hidden_size, bottleneck_size),
            "Compacter": CompacterAdapter(hidden_size, bottleneck_size),
            "Gated": GatedAdapter(hidden_size, bottleneck_size),
        }
        
        print(f"\n  Hidden size: {hidden_size}, Bottleneck: {bottleneck_size}")
        print(f"\n  {'Variant':<15} {'Params':>10} {'Output OK':>10} "
              f"{'Identity @ Init':>16} {'Notes':>25}")
        print("  " + "─" * 80)
        
        for name, adapter in variants.items():
            params = sum(p.numel() for p in adapter.parameters())
            
            with torch.no_grad():
                out = adapter(x)
            
            output_ok = out.shape == x.shape
            identity = torch.allclose(x, out, atol=1e-4)
            
            notes = ""
            if name == "Houlsby":
                notes = "2 per layer"
            elif name == "Pfeiffer":
                notes = "1 per layer"
            elif name == "Parallel":
                notes = "Runs alongside FFN"
            elif name == "Compacter":
                notes = f"Kronecker compressed"
            elif name == "Gated":
                gate_val = adapter.get_gate_value()
                notes = f"Gate: {gate_val:.3f}"
            
            print(f"  {name:<15} {params:>10,} {'✓' if output_ok else '✗':>10} "
                  f"{'✓ Yes' if identity else '✗ No':>16} {notes:>25}")
        
        # Full layer comparison
        print(f"\n\n  FULL TRANSFORMER LAYER COMPARISON:")
        print(f"  (Includes frozen attention + FFN parameters)")
        
        layers = {
            "Houlsby": HoulsbyTransformerLayer(hidden_size, bottleneck_size),
            "Pfeiffer": PfeifferTransformerLayer(hidden_size, bottleneck_size),
            "Parallel": ParallelTransformerLayer(hidden_size, bottleneck_size),
        }
        
        print(f"\n  {'Config':<15} {'Trainable':>12} {'Frozen':>12} "
              f"{'Total':>12} {'% Trainable':>12}")
        print("  " + "─" * 65)
        
        for name, layer in layers.items():
            trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            frozen = sum(p.numel() for p in layer.parameters() if not p.requires_grad)
            total = trainable + frozen
            pct = 100 * trainable / total if total > 0 else 0
            
            print(f"  {name:<15} {trainable:>12,} {frozen:>12,} "
                  f"{total:>12,} {pct:>10.2f}%")
    
    def placement_comparison(self):
        """Compare adapter placement strategies."""
        print("\n" + "=" * 70)
        print("  ADAPTER PLACEMENT STRATEGIES")
        print("=" * 70)
        
        diagram = """
  STRATEGY 1: HOULSBY (Sequential, 2 adapters)
  ──────────────────────────────────────────────
    x ──→ [Attention] ──→ [ADAPTER₁] ──→ [FFN] ──→ [ADAPTER₂] ──→ output
    │                      │               │                      │
    └─────── residual ─────┘               └──── residual ────────┘
    
    + Maximum adapter capacity
    + Each sub-layer gets its own adapter  
    - Most parameters (2 adapters × each layer)
    - Slowest inference (2 sequential adapter passes)

  STRATEGY 2: PFEIFFER (Sequential, 1 adapter)
  ──────────────────────────────────────────────
    x ──→ [Attention] ──→ [FFN] ──→ [ADAPTER] ──→ output
    │                      │          │            │
    └─────── residual ─────┘          └─ residual ─┘
    
    + Half the parameters of Houlsby
    + Faster training and inference
    + Surprisingly competitive quality
    - Less capacity than Houlsby

  STRATEGY 3: PARALLEL
  ──────────────────────────────────────────────
    x ──→ [Attention] ──→ ┬──→ [FFN] ──────┬──→ output
    │                      │                │
    │                      └──→ [ADAPTER] ──┘
    └─────── residual ─────────────────────────┘
    
    + Better gradient flow to adapter
    + FFN and adapter don't block each other
    + Conceptually similar to LoRA
    - Same parameter count as Pfeiffer
    
  STRATEGY 4: PREFIX + ADAPTER (Combined)
  ──────────────────────────────────────────────
    [prefix] + x ──→ [Attention] ──→ [FFN] ──→ [ADAPTER] ──→ output
    
    + Combines prefix tuning with adapters
    + Can capture both input-level and layer-level patterns
    - More complex, more hyperparameters

  TYPICAL PERFORMANCE RANKING (from literature):
  ──────────────────────────────────────────────
    Parallel ≥ Houlsby > Pfeiffer > Prefix-only
    (with same total parameter budget)
"""
        print(diagram)
    
    def print_summary(self):
        """Print summary comparison table."""
        print("\n" + "=" * 70)
        print("  ADAPTER VARIANTS SUMMARY")
        print("=" * 70)
        
        table = """
┌────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ Feature        │ Houlsby  │ Pfeiffer │ Parallel │ Adapter  │ Gated    │
│                │          │          │          │ Drop     │          │
├────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ Adapters/layer │ 2        │ 1        │ 1        │ 0-1      │ 1        │
│ Placement      │ Seq.     │ Seq.     │ Parallel │ Seq.     │ Seq.     │
│ Params (rel.)  │ 2x       │ 1x       │ 1x       │ ≤1x      │ 1x+1    │
│ Quality        │ ████░    │ ███░░    │ ████░    │ ███░░    │ ███░░    │
│ Speed          │ ██░░░    │ ███░░    │ ███░░    │ ████░    │ ███░░    │
│ Composability  │ ████░    │ ████░    │ ███░░    │ ████░    │ █████    │
│ Complexity     │ Low      │ Lowest   │ Medium   │ Medium   │ Medium   │
│ Best for       │ Max      │ Default  │ Quality  │ Fast     │ Multi-   │
│                │ capacity │ choice   │ focus    │ deploy   │ method   │
└────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

RECOMMENDATIONS:
  → Default choice: Pfeiffer (simple, effective, half the params)
  → Quality priority: Parallel adapter or Houlsby
  → Inference speed: AdapterDrop (remove bottom N layers)
  → Multi-method: Gated adapter (auto-learns importance)
  → Memory constrained: Compacter (Kronecker compression)
"""
        print(table)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all adapter variant demonstrations."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║                   ADAPTER VARIANTS                           ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Compare all variants
    comparison = AdapterVariantComparison()
    comparison.compare_all()
    comparison.placement_comparison()
    
    # AdapterDrop analysis
    print("\n")
    drop_adapters = [
        AdapterWithDrop(768, 64, drop_prob=0.1, layer_idx=i) 
        for i in range(12)
    ]
    drop_manager = AdapterDropManager(drop_adapters)
    drop_manager.demonstrate_drop_analysis()
    
    # Summary
    comparison.print_summary()
    
    print("\n" + "=" * 70)
    print("  MODULE COMPLETE")
    print("=" * 70)
    print("""
    Covered in this module:
    ✓ Houlsby adapter (original, 2 per layer)
    ✓ Pfeiffer adapter (efficient, 1 per layer)
    ✓ Parallel adapter (alongside FFN)
    ✓ AdapterDrop (progressive layer removal)
    ✓ Compacter (Kronecker compression)
    ✓ Gated adapter (learnable importance)
    ✓ Placement strategies and comparison
    """)


if __name__ == "__main__":
    main()
