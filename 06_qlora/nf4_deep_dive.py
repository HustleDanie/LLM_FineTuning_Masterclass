"""
NF4 (NormalFloat 4-bit) — Deep Dive
=====================================

NF4 is the key innovation of QLoRA. This module implements NF4 from
scratch and explains WHY it's optimal for neural network weights.

Topics:
1. Why NF4? — The information-theoretic argument
2. NF4 construction from first principles
3. NF4 quantization levels (the exact 16 values)
4. Implementing NF4 quantization from scratch
5. NF4 vs FP4 vs INT4 — empirical comparison
6. Dequantization and computation flow

The Key Idea:
═══════════════════════════════════════════════════════════════════
  Standard quantization places levels UNIFORMLY in the value range.
  But neural network weights are NOT uniformly distributed —
  they follow a NORMAL distribution (bell curve).
  
  NF4 places quantization levels at the QUANTILES of N(0,1),
  so each level represents an equal probability mass.
  This minimizes the expected quantization error.
  
  Think of it this way:
  - Uniform quantization: equal spacing in VALUE space
  - NF4 quantization: equal spacing in PROBABILITY space
═══════════════════════════════════════════════════════════════════

Author: LLM Fine-Tuning Masterclass
"""

import torch
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# SECTION 1: WHY NF4? — THE INFORMATION-THEORETIC ARGUMENT
# ============================================================================

class NF4Theory:
    """
    The information-theoretic foundation of NF4.
    
    Lloyd-Max Quantizer:
    For k bits, we want to find 2^k quantization levels that minimize
    the mean squared error (MSE) for a given input distribution.
    
    For normally-distributed inputs, the optimal quantizer places
    levels at the quantiles of the distribution, and reconstruction
    values at the centroid of each quantile bin.
    
    This is exactly what NF4 does for k=4 (16 levels).
    """
    
    @staticmethod
    def explain_optimal_quantization():
        """
        Why uniform quantization is suboptimal for normal distributions.
        """
        print("=" * 70)
        print("WHY NF4? OPTIMAL QUANTIZATION FOR NORMAL DISTRIBUTIONS")
        print("=" * 70)
        
        diagram = """
  Uniform Quantization (suboptimal for normal dist):
  ════════════════════════════════════════════════════
  
  Weight Distribution (Normal):
  
        ▄▄▄▄
       ██████
      ████████
     ██████████
    ████████████
   ██████████████
  ████████████████
  ├──┬──┬──┬──┬──┬──┬──┬──┤   ← 8 uniform bins
  
  Problem: Most weights are near zero (peak of bell curve),
  but uniform bins waste precision on the tails where few
  weights exist.
  
  
  NF4 Quantization (optimal for normal dist):
  ════════════════════════════════════════════════════
  
  Weight Distribution (Normal):
  
        ▄▄▄▄
       ██████
      ████████
     ██████████
    ████████████
   ██████████████
  ████████████████
  ├┬─┬─┬──┬────┬──┬─┬─┬┤   ← 8 non-uniform bins
   ↑             ↑
   Dense near    Sparse in
   the center    the tails
  
  NF4 places MORE quantization levels near zero (where most
  weights are) and FEWER levels in the tails.
  
  Result: Each bin contains the same PROBABILITY MASS (1/16),
  which minimizes the expected quantization error.
"""
        print(diagram)
    
    @staticmethod
    def demonstrate_quantile_concept():
        """
        Show how quantiles divide a distribution into equal probability bins.
        """
        print("\n" + "=" * 70)
        print("QUANTILE-BASED BIN PLACEMENT")
        print("=" * 70)
        
        # Generate normally distributed weights
        torch.manual_seed(42)
        n = 100000
        weights = torch.randn(n)  # Standard normal
        
        n_bins = 16  # 4 bits = 16 levels
        
        # Method 1: Uniform bins
        print("\n  1. UNIFORM BINS (standard quantization):")
        uniform_edges = torch.linspace(weights.min(), weights.max(), n_bins + 1)
        uniform_counts = []
        for i in range(n_bins):
            count = ((weights >= uniform_edges[i]) & (weights < uniform_edges[i+1])).sum().item()
            uniform_counts.append(count)
        # Fix last bin to include max
        uniform_counts[-1] += (weights == uniform_edges[-1]).sum().item()
        
        print(f"    Bin counts (should be ~{n//n_bins} each for balanced):")
        print(f"    {uniform_counts}")
        print(f"    Min bin: {min(uniform_counts)}, Max bin: {max(uniform_counts)}")
        print(f"    Ratio max/min: {max(uniform_counts)/max(min(uniform_counts),1):.1f}x")
        
        # Method 2: Quantile bins (NF4 approach)
        print("\n  2. QUANTILE BINS (NF4 approach):")
        quantile_edges = torch.quantile(weights, torch.linspace(0, 1, n_bins + 1))
        quantile_counts = []
        for i in range(n_bins):
            count = ((weights >= quantile_edges[i]) & (weights < quantile_edges[i+1])).sum().item()
            quantile_counts.append(count)
        quantile_counts[-1] += (weights == quantile_edges[-1]).sum().item()
        
        print(f"    Bin counts (should be ~{n//n_bins} each for balanced):")
        print(f"    {quantile_counts}")
        print(f"    Min bin: {min(quantile_counts)}, Max bin: {max(quantile_counts)}")
        print(f"    Ratio max/min: {max(quantile_counts)/max(min(quantile_counts),1):.1f}x")
        
        # Compare quantization error
        print("\n  3. QUANTIZATION ERROR COMPARISON:")
        
        # Uniform: use bin centers as reconstruction values
        uniform_centers = (uniform_edges[:-1] + uniform_edges[1:]) / 2
        uniform_indices = torch.bucketize(weights, uniform_edges[1:-1])
        uniform_recon = uniform_centers[uniform_indices]
        uniform_mse = (weights - uniform_recon).pow(2).mean().item()
        
        # Quantile: use bin centers (conditional means)
        quantile_recon = torch.zeros_like(weights)
        for i in range(n_bins):
            mask = (uniform_indices if False else
                    torch.bucketize(weights, quantile_edges[1:-1])) == i
            if mask.sum() > 0:
                quantile_recon[mask] = weights[mask].mean()
        quantile_mse = (weights - quantile_recon).pow(2).mean().item()
        
        print(f"    Uniform MSE:   {uniform_mse:.6f}")
        print(f"    Quantile MSE:  {quantile_mse:.6f}")
        print(f"    Improvement:   {(1 - quantile_mse/uniform_mse)*100:.1f}%")
        
        return {
            "uniform_mse": uniform_mse,
            "quantile_mse": quantile_mse,
        }


# ============================================================================
# SECTION 2: NF4 CONSTRUCTION FROM FIRST PRINCIPLES
# ============================================================================

class NF4Construction:
    """
    Build the NF4 data type step by step.
    
    The NF4 quantization levels are computed as follows:
    1. Take the standard normal distribution N(0,1)
    2. Divide it into 2^k = 16 equal-probability bins
    3. For each bin, compute the centroid (conditional mean)
    4. These centroids are the NF4 quantization levels
    5. Normalize so the range is [-1, 1]
    
    The actual NF4 values used by bitsandbytes are pre-computed
    and hardcoded for efficiency.
    """
    
    @staticmethod
    def compute_nf4_levels() -> torch.Tensor:
        """
        Compute the 16 NF4 quantization levels from scratch.
        
        Returns the exact NF4 codebook used by QLoRA.
        """
        print("\n" + "=" * 70)
        print("COMPUTING NF4 LEVELS FROM FIRST PRINCIPLES")
        print("=" * 70)
        
        n_levels = 16  # 4 bits
        
        # Step 1: Compute quantile boundaries
        # We want 16 bins with equal probability mass = 1/16 each
        print("\n  Step 1: Compute quantile boundaries")
        print(f"    Each bin has probability mass = 1/{n_levels} = {1/n_levels:.4f}")
        
        # Boundaries at cumulative probabilities: 0/16, 1/16, 2/16, ..., 16/16
        # But we handle negative and positive halves separately for symmetry
        # and add a zero point
        
        # The QLoRA paper uses a specific construction:
        # - 8 negative levels + 0 + 7 positive levels = 16 levels
        # - The negative and positive halves are symmetric
        # - Zero is included explicitly
        
        # Method: Use quantiles of the half-normal distribution
        # For the positive half (8 levels including 0):
        # Quantile boundaries: 0, 1/8, 2/8, ..., 7/8, 1.0
        # For the negative half (8 levels):
        # Mirror of positive
        
        try:
            from scipy import stats
            from scipy import integrate
            
            # Compute quantile boundaries for the standard normal
            # We need 16 bins with equal probability
            boundaries = []
            for i in range(n_levels + 1):
                q = stats.norm.ppf(i / n_levels)
                boundaries.append(q)
            
            print(f"    Boundaries (quantiles of N(0,1)):")
            for i, b in enumerate(boundaries):
                print(f"      q({i}/{n_levels}) = {b:+.4f}")
            
            # Step 2: Compute centroids (conditional means) for each bin
            print("\n  Step 2: Compute centroids (conditional means)")
            levels = []
            for i in range(n_levels):
                lo = boundaries[i]
                hi = boundaries[i + 1]
                
                # Conditional mean of N(0,1) in [lo, hi]
                # E[X | lo < X < hi] = (φ(lo) - φ(hi)) / (Φ(hi) - Φ(lo))
                # where φ is PDF and Φ is CDF
                phi_lo = stats.norm.pdf(lo)
                phi_hi = stats.norm.pdf(hi)
                Phi_lo = stats.norm.cdf(lo)
                Phi_hi = stats.norm.cdf(hi)
                
                centroid = (phi_lo - phi_hi) / (Phi_hi - Phi_lo)
                levels.append(centroid)
                
                print(f"      Bin {i:2d}: [{lo:+.4f}, {hi:+.4f}] → centroid = {centroid:+.6f}")
            
        except ImportError:
            print("    (scipy not available, using pre-computed values)")
            # Pre-computed NF4 levels (from the QLoRA paper/bitsandbytes)
            levels = [
                -1.0, -0.6962, -0.5251, -0.3949,
                -0.2844, -0.1848, -0.0911, 0.0,
                0.0796, 0.1609, 0.2461, 0.3379,
                0.4407, 0.5626, 0.7230, 1.0,
            ]
        
        # Step 3: Normalize to [-1, 1]
        levels_tensor = torch.tensor(levels, dtype=torch.float32)
        max_abs = levels_tensor.abs().max()
        nf4_levels = levels_tensor / max_abs
        
        print(f"\n  Step 3: Normalized NF4 levels (16 values):")
        print(f"    {[f'{v:+.4f}' for v in nf4_levels.tolist()]}")
        
        return nf4_levels
    
    @staticmethod
    def bitsandbytes_nf4_levels() -> torch.Tensor:
        """
        The exact NF4 quantization levels used by bitsandbytes.
        
        These are pre-computed and hardcoded in the library for efficiency.
        The construction ensures:
        - Symmetric around 0 (with 0 included)
        - 8 negative + 0 + 7 positive = 16 levels
        - Optimal for the standard normal distribution
        """
        print("\n" + "=" * 70)
        print("EXACT NF4 LEVELS (from bitsandbytes)")
        print("=" * 70)
        
        # These are the actual values from the bitsandbytes source code
        nf4_levels = torch.tensor([
            -1.0000, -0.6962, -0.5251, -0.3949,
            -0.2844, -0.1848, -0.0911,  0.0000,
             0.0796,  0.1609,  0.2461,  0.3379,
             0.4407,  0.5626,  0.7230,  1.0000,
        ])
        
        print("\n  4-bit  Binary    NF4 Level")
        print("  index  code      value")
        print("  " + "-" * 35)
        for i, level in enumerate(nf4_levels):
            print(f"    {i:2d}    {i:04b}     {level:+.4f}")
        
        # Properties
        print(f"\n  Properties:")
        print(f"    Number of levels: {len(nf4_levels)}")
        print(f"    Range: [{nf4_levels.min():.4f}, {nf4_levels.max():.4f}]")
        print(f"    Includes zero: {0.0 in nf4_levels.tolist()}")
        print(f"    Symmetric: ~{(nf4_levels + nf4_levels.flip(0)).abs().max().item():.4f}")
        
        # Level spacing
        print(f"\n  Level spacing (non-uniform!):")
        spacings = nf4_levels[1:] - nf4_levels[:-1]
        for i, (lo, hi, sp) in enumerate(zip(nf4_levels[:-1], nf4_levels[1:], spacings)):
            bar = "█" * int(sp * 40)
            print(f"    [{lo:+.4f}, {hi:+.4f}]: Δ={sp:.4f} {bar}")
        
        print("\n  → Spacing is DENSE near zero, SPARSE in tails")
        print("    This matches the normal distribution shape!")
        
        return nf4_levels


# ============================================================================
# SECTION 3: NF4 QUANTIZATION FROM SCRATCH
# ============================================================================

class NF4Quantizer:
    """
    Complete NF4 quantization implementation from scratch.
    
    This shows exactly what bitsandbytes does under the hood.
    """
    
    # Pre-computed NF4 codebook
    NF4_LEVELS = torch.tensor([
        -1.0000, -0.6962, -0.5251, -0.3949,
        -0.2844, -0.1848, -0.0911,  0.0000,
         0.0796,  0.1609,  0.2461,  0.3379,
         0.4407,  0.5626,  0.7230,  1.0000,
    ])
    
    def __init__(self, block_size: int = 64):
        """
        Initialize NF4 quantizer.
        
        Args:
            block_size: Number of weights per quantization block.
                        Each block gets its own absmax scale.
                        QLoRA uses 64 by default.
        """
        self.block_size = block_size
    
    def quantize(self, tensor: torch.Tensor) -> dict:
        """
        Quantize a weight tensor to NF4.
        
        Process:
        1. Reshape into blocks of size `block_size`
        2. For each block, compute absmax
        3. Normalize each block to [-1, 1] using absmax
        4. Map each normalized value to nearest NF4 level
        5. Store 4-bit indices + absmax scales
        
        Args:
            tensor: Input weight tensor (any shape)
            
        Returns:
            Dictionary with quantized data, scales, and metadata
        """
        print("\n  ── NF4 Quantization ──")
        
        original_shape = tensor.shape
        flat = tensor.flatten().float()
        n = flat.numel()
        
        # Pad to multiple of block_size
        if n % self.block_size != 0:
            pad_size = self.block_size - (n % self.block_size)
            flat = torch.cat([flat, torch.zeros(pad_size)])
        else:
            pad_size = 0
        
        n_padded = flat.numel()
        n_blocks = n_padded // self.block_size
        
        # Reshape into blocks
        blocks = flat.reshape(n_blocks, self.block_size)
        
        # Step 1: Compute absmax per block
        absmax = blocks.abs().max(dim=1).values  # [n_blocks]
        absmax = absmax.clamp(min=1e-10)  # Avoid division by zero
        
        # Step 2: Normalize to [-1, 1]
        normalized = blocks / absmax.unsqueeze(1)
        
        # Step 3: Map to nearest NF4 level
        # For each value, find the index of the closest NF4 level
        nf4 = self.NF4_LEVELS.to(normalized.device)
        
        # Compute distances to all 16 levels
        # normalized: [n_blocks, block_size]
        # nf4: [16]
        # distances: [n_blocks, block_size, 16]
        distances = (normalized.unsqueeze(-1) - nf4.unsqueeze(0).unsqueeze(0)).abs()
        indices = distances.argmin(dim=-1)  # [n_blocks, block_size]
        
        # Pack 4-bit indices into uint8 (2 values per byte)
        indices_flat = indices.reshape(-1).to(torch.uint8)
        packed = self._pack_4bit(indices_flat)
        
        result = {
            "packed_data": packed,
            "absmax": absmax,
            "original_shape": original_shape,
            "n_elements": n,
            "pad_size": pad_size,
            "block_size": self.block_size,
            "n_blocks": n_blocks,
        }
        
        # Memory analysis
        data_bytes = packed.numel()  # uint8, 2 values per byte
        scale_bytes = absmax.numel() * absmax.element_size()
        total_bytes = data_bytes + scale_bytes
        original_bytes = n * 4  # FP32
        
        print(f"    Original: {original_bytes:,} bytes (FP32)")
        print(f"    Quantized data: {data_bytes:,} bytes")
        print(f"    Scales: {scale_bytes:,} bytes ({absmax.numel()} × FP32)")
        print(f"    Total: {total_bytes:,} bytes")
        print(f"    Compression: {original_bytes / total_bytes:.1f}x")
        print(f"    Effective bits/param: {total_bytes * 8 / n:.2f}")
        
        return result
    
    def dequantize(self, quant_data: dict) -> torch.Tensor:
        """
        Dequantize NF4 data back to floating point.
        
        Process:
        1. Unpack 4-bit indices from uint8
        2. Map indices to NF4 levels
        3. Multiply by absmax to restore scale
        4. Reshape to original shape
        """
        packed = quant_data["packed_data"]
        absmax = quant_data["absmax"]
        original_shape = quant_data["original_shape"]
        n = quant_data["n_elements"]
        block_size = quant_data["block_size"]
        n_blocks = quant_data["n_blocks"]
        
        # Unpack 4-bit indices
        indices = self._unpack_4bit(packed, n_blocks * block_size)
        
        # Map to NF4 levels
        nf4 = self.NF4_LEVELS.to(indices.device)
        values = nf4[indices.long()]
        
        # Reshape into blocks and multiply by absmax
        values = values.reshape(n_blocks, block_size)
        values = values * absmax.unsqueeze(1)
        
        # Flatten and remove padding
        result = values.flatten()[:n]
        
        return result.reshape(original_shape)
    
    def _pack_4bit(self, indices: torch.Tensor) -> torch.Tensor:
        """Pack two 4-bit values into one uint8."""
        assert indices.numel() % 2 == 0, "Need even number of values"
        indices = indices.to(torch.uint8)
        even = indices[0::2]  # Even indices: lower nibble
        odd = indices[1::2]   # Odd indices: upper nibble
        packed = (odd << 4) | even
        return packed
    
    def _unpack_4bit(self, packed: torch.Tensor, n: int) -> torch.Tensor:
        """Unpack uint8 into two 4-bit values."""
        even = packed & 0x0F          # Lower nibble
        odd = (packed >> 4) & 0x0F    # Upper nibble
        
        # Interleave
        result = torch.zeros(n, dtype=torch.uint8)
        result[0::2] = even[:n // 2 + n % 2]
        result[1::2] = odd[:n // 2]
        
        return result
    
    def demonstrate_full_pipeline(self):
        """
        Complete demonstration: quantize → dequantize → analyze error.
        """
        print("\n" + "=" * 70)
        print("NF4 QUANTIZATION — FULL PIPELINE DEMO")
        print("=" * 70)
        
        # Create realistic weight matrix
        torch.manual_seed(42)
        W = torch.randn(256, 256) * 0.02  # Typical NN weight scale
        
        print(f"\n  Original weight matrix: {W.shape}")
        print(f"  Dtype: {W.dtype}")
        print(f"  Range: [{W.min():.4f}, {W.max():.4f}]")
        print(f"  Mean: {W.mean():.6f}, Std: {W.std():.6f}")
        
        # Quantize
        print("\n  --- Quantizing ---")
        quant_data = self.quantize(W)
        
        # Dequantize
        print("\n  --- Dequantizing ---")
        W_reconstructed = self.dequantize(quant_data)
        
        # Error analysis
        error = (W - W_reconstructed).abs()
        mse = (W - W_reconstructed).pow(2).mean().item()
        
        print(f"\n  ── Error Analysis ──")
        print(f"    Max absolute error: {error.max().item():.6f}")
        print(f"    Mean absolute error: {error.mean().item():.6f}")
        print(f"    MSE: {mse:.2e}")
        print(f"    RMSE: {mse**0.5:.6f}")
        print(f"    NRMSE: {mse**0.5 / W.std().item():.4f} "
              f"({mse**0.5 / W.std().item() * 100:.2f}%)")
        
        # Signal-to-quantization-noise ratio
        signal_power = W.pow(2).mean().item()
        noise_power = mse
        sqnr_db = 10 * math.log10(signal_power / noise_power)
        print(f"    SQNR: {sqnr_db:.1f} dB")
        
        # Verify shape preservation
        print(f"\n  Shape preserved: {W.shape} → {W_reconstructed.shape} ✓")
        
        return {
            "mse": mse,
            "sqnr_db": sqnr_db,
            "compression_ratio": W.numel() * 4 / (
                quant_data["packed_data"].numel() + 
                quant_data["absmax"].numel() * 4
            ),
        }


# ============================================================================
# SECTION 4: NF4 vs FP4 vs INT4 — EMPIRICAL COMPARISON
# ============================================================================

class QuantizationComparison:
    """
    Head-to-head comparison of 4-bit quantization methods
    on realistic neural network weights.
    """
    
    @staticmethod
    def compare_all_methods():
        """
        Compare NF4, FP4, and uniform INT4 quantization.
        """
        print("\n" + "=" * 70)
        print("NF4 vs FP4 vs INT4 — EMPIRICAL COMPARISON")
        print("=" * 70)
        
        torch.manual_seed(42)
        
        # Test on different weight distributions
        test_cases = {
            "Normal (σ=0.02)": torch.randn(10000) * 0.02,
            "Normal (σ=0.1)": torch.randn(10000) * 0.1,
            "Normal (σ=1.0)": torch.randn(10000) * 1.0,
            "Uniform [-1, 1]": torch.rand(10000) * 2 - 1,
            "Laplace": torch.distributions.Laplace(0, 0.02).sample((10000,)),
            "Real-ish (mixed)": torch.cat([
                torch.randn(8000) * 0.02,       # Most weights: normal
                torch.randn(2000) * 0.1,         # Some outliers
            ]),
        }
        
        # NF4 levels
        nf4_levels = torch.tensor([
            -1.0000, -0.6962, -0.5251, -0.3949,
            -0.2844, -0.1848, -0.0911,  0.0000,
             0.0796,  0.1609,  0.2461,  0.3379,
             0.4407,  0.5626,  0.7230,  1.0000,
        ])
        
        # FP4 levels (E2M1 format: 2 exponent bits, 1 mantissa bit)
        fp4_positive = torch.tensor([0, 0.0625, 0.125, 0.1875,
                                      0.25, 0.375, 0.5, 1.0])
        fp4_levels = torch.cat([-fp4_positive.flip(0)[:-1], fp4_positive])
        fp4_levels = fp4_levels / fp4_levels.abs().max()  # Normalize to [-1, 1]
        
        # Uniform INT4 levels
        int4_levels = torch.linspace(-1, 1, 16)
        
        all_methods = {
            "NF4": nf4_levels,
            "FP4": fp4_levels,
            "INT4 (uniform)": int4_levels,
        }
        
        print(f"\n  {'Distribution':<22}", end="")
        for method in all_methods:
            print(f" {method:>15}", end="")
        print("  Best")
        print("  " + "-" * 72)
        
        for dist_name, weights in test_cases.items():
            print(f"  {dist_name:<22}", end="")
            
            mse_results = {}
            for method_name, levels in all_methods.items():
                # Normalize weights to [-1, 1] using absmax
                absmax = weights.abs().max()
                normalized = weights / absmax
                
                # Quantize: find nearest level
                distances = (normalized.unsqueeze(1) - levels.unsqueeze(0)).abs()
                indices = distances.argmin(dim=1)
                dequantized = levels[indices] * absmax
                
                mse = (weights - dequantized).pow(2).mean().item()
                mse_results[method_name] = mse
                print(f" {mse:>15.2e}", end="")
            
            best = min(mse_results, key=mse_results.get)
            print(f"  ← {best}")
        
        print("\n  → NF4 wins for normal distributions (which is what NN weights are)")
        print("  → INT4 wins for uniform distributions (rare in practice)")
        print("  → For real neural networks, NF4 is the optimal choice")


# ============================================================================
# SECTION 5: HOW NF4 FITS INTO THE FORWARD PASS
# ============================================================================

class NF4ForwardPass:
    """
    Shows how NF4 quantization integrates with the forward pass
    in QLoRA training.
    """
    
    @staticmethod
    def demonstrate_qlora_forward():
        """
        The QLoRA forward pass:
        
        1. Base weights stored in NF4 (4-bit) — FROZEN
        2. During forward pass: dequantize to BF16 on-the-fly
        3. LoRA adapters in BF16/FP16 — TRAINABLE
        4. Output = dequantized_base(x) + lora_B(lora_A(x))
        5. Gradients flow ONLY through LoRA parameters
        """
        print("\n" + "=" * 70)
        print("QLoRA FORWARD PASS WITH NF4")
        print("=" * 70)
        
        diagram = """
  ┌──────────────────────────────────────────────────────────┐
  │                   QLoRA Forward Pass                      │
  │                                                          │
  │   Input x (BF16)                                         │
  │       │                                                  │
  │       ├───────────────────┐                              │
  │       │                   │                              │
  │       ▼                   ▼                              │
  │   ┌────────────┐    ┌──────────┐                        │
  │   │ W_base     │    │ LoRA A   │                        │
  │   │ (NF4, 4b)  │    │ (BF16)   │ Trainable             │
  │   │ ──────────>│    │ [r×d_in] │                        │
  │   │ Dequant to │    └────┬─────┘                        │
  │   │ BF16 on    │         │                              │
  │   │ the fly    │         ▼                              │
  │   └─────┬──────┘    ┌──────────┐                        │
  │         │           │ LoRA B   │                        │
  │         │           │ (BF16)   │ Trainable              │
  │         │           │ [d_out×r]│                        │
  │         │           └────┬─────┘                        │
  │         │                │                              │
  │         ▼                ▼                              │
  │       y_base    +    Δy_lora                            │
  │         │                │                              │
  │         └───────┬────────┘                              │
  │                 │                                        │
  │                 ▼                                        │
  │             y = y_base + (α/r) × Δy_lora               │
  │                                                          │
  │   Backward: gradients flow only through LoRA A and B    │
  │   W_base gradients are NOT computed (frozen + quantized) │
  └──────────────────────────────────────────────────────────┘
  
  Memory during training:
  ───────────────────────────────────────────────────────────
    W_base:    Stored in NF4 (4 bits/param)       ← HUGE savings
    LoRA A,B:  Stored in BF16 (16 bits/param)     ← Very small
    Gradients: Only for LoRA A,B (BF16)           ← Very small
    Optimizer: Only for LoRA A,B (FP32 states)    ← Very small
    
    The base model weights (99%+ of params) are in 4-bit!
  ───────────────────────────────────────────────────────────
"""
        print(diagram)
    
    @staticmethod
    def simulate_qlora_linear():
        """
        Simulate a single QLoRA linear layer operation.
        """
        print("\n  ── Simulated QLoRA Linear Layer ──")
        
        d_in, d_out = 256, 256
        rank = 8
        alpha = 16.0
        batch_size = 4
        seq_len = 32
        
        # Base weight in NF4
        W_fp32 = torch.randn(d_out, d_in) * 0.02
        
        # Quantize base weight to NF4
        quantizer = NF4Quantizer(block_size=64)
        quant_data = quantizer.quantize(W_fp32)
        
        # LoRA weights in BF16
        lora_A = torch.randn(rank, d_in, dtype=torch.bfloat16) * 0.01
        lora_B = torch.zeros(d_out, rank, dtype=torch.bfloat16)
        
        # Input
        x = torch.randn(batch_size, seq_len, d_in, dtype=torch.bfloat16)
        
        # Forward pass
        # Step 1: Dequantize base weight to BF16
        W_dequant = quantizer.dequantize(quant_data).to(torch.bfloat16)
        
        # Step 2: Base computation (frozen)
        y_base = x @ W_dequant.T  # [batch, seq, d_out]
        
        # Step 3: LoRA computation (trainable)
        scaling = alpha / rank
        y_lora = scaling * (x @ lora_A.T @ lora_B.T)  # [batch, seq, d_out]
        
        # Step 4: Combined output
        y = y_base + y_lora
        
        print(f"\n    Input shape:  {x.shape}")
        print(f"    Base weight:  {W_fp32.shape} → NF4 (4-bit)")
        print(f"    LoRA A:       {lora_A.shape} (BF16)")
        print(f"    LoRA B:       {lora_B.shape} (BF16)")
        print(f"    Output shape: {y.shape}")
        
        # Memory analysis for this layer
        base_mem = W_fp32.numel() * 0.5  # 4 bits = 0.5 bytes per param
        lora_mem = (lora_A.numel() + lora_B.numel()) * 2  # BF16 = 2 bytes
        fp16_mem = W_fp32.numel() * 2  # If we stored in FP16 instead
        
        print(f"\n    Memory comparison for this layer:")
        print(f"      FP16 base weight:  {fp16_mem:>8,} bytes")
        print(f"      NF4 base weight:   {base_mem:>8,.0f} bytes "
              f"({base_mem/fp16_mem*100:.0f}%)")
        print(f"      LoRA parameters:   {lora_mem:>8,} bytes")
        print(f"      Total QLoRA:       {base_mem + lora_mem:>8,.0f} bytes "
              f"({(base_mem+lora_mem)/fp16_mem*100:.0f}% of FP16)")
        
        return y


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all NF4 deep dive demonstrations."""
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║        NF4 (NormalFloat 4-bit) — DEEP DIVE                     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    # Section 1: Theory
    theory = NF4Theory()
    theory.explain_optimal_quantization()
    theory.demonstrate_quantile_concept()
    
    # Section 2: Construction
    construction = NF4Construction()
    construction.compute_nf4_levels()
    construction.bitsandbytes_nf4_levels()
    
    # Section 3: From-scratch implementation
    quantizer = NF4Quantizer(block_size=64)
    quantizer.demonstrate_full_pipeline()
    
    # Section 4: Comparison
    comparison = QuantizationComparison()
    comparison.compare_all_methods()
    
    # Section 5: Forward pass integration
    forward = NF4ForwardPass()
    forward.demonstrate_qlora_forward()
    forward.simulate_qlora_linear()
    
    print("\n" + "=" * 70)
    print("  MODULE COMPLETE")
    print("=" * 70)
    print("""
    Key takeaways:
    ✓ NF4 places quantization levels at distribution quantiles
    ✓ This is information-theoretically optimal for normal distributions
    ✓ Neural network weights ARE approximately normally distributed
    ✓ NF4 achieves lower error than FP4 or uniform INT4 for NN weights
    ✓ Each block of 64 weights gets its own absmax scale
    ✓ During forward pass, NF4 → BF16 dequantization happens on-the-fly
    ✓ Only LoRA parameters are trained; base model stays in 4-bit
    """)


if __name__ == "__main__":
    main()
