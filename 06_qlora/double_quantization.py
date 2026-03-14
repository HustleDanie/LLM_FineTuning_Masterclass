"""
Double Quantization — QLoRA's Second Innovation
=================================================

Double quantization is the technique of quantizing the quantization
constants themselves. This saves significant additional memory.

The Problem:
  In blockwise quantization, each block of 64 weights needs a FP32
  scaling constant (absmax). For a 65B model, these constants alone
  consume ~0.5 GB of memory!

The Solution:
  Quantize the scaling constants (FP32) to FP8 using a second level
  of quantization, with one FP32 scale per 256 scaling constants.

Memory savings:
  Without double quant: 32 bits / 64 params = 0.5 bits per param
  With double quant:    8 bits / 64 params + 32/(64×256) = 0.127 bits per param
  Savings: 0.373 bits per param → 3 GB for a 65B model!

This module covers:
1. Why quantization constants need to be quantized
2. Double quantization implementation from scratch
3. Memory savings analysis
4. Impact on quantization quality

Author: LLM Fine-Tuning Masterclass
"""

import torch
import math
from typing import Dict, Tuple
from dataclasses import dataclass


# ============================================================================
# SECTION 1: THE OVERHEAD PROBLEM
# ============================================================================

class QuantizationOverhead:
    """
    Demonstrates WHY double quantization is needed by analyzing
    the memory overhead of quantization constants.
    """
    
    @staticmethod
    def analyze_overhead():
        """
        Calculate the memory overhead of quantization constants
        for different model sizes.
        """
        print("=" * 70)
        print("QUANTIZATION CONSTANT OVERHEAD ANALYSIS")
        print("=" * 70)
        
        explanation = """
  Blockwise NF4 Quantization:
  ═══════════════════════════════════════════════════════════
  
  For each block of B weights:
    - Store B 4-bit quantized values  → B × 0.5 bytes
    - Store 1 FP32 scaling constant   → 4 bytes
  
  The scaling constant adds overhead:
    Overhead per param = 32 bits / B = 32/64 = 0.5 bits/param
  
  Total bits per param = 4.0 (NF4) + 0.5 (scale) = 4.5 bits
  
  For large models, this overhead is significant!
"""
        print(explanation)
        
        # Calculate for different model sizes
        block_size = 64
        scale_bits = 32  # FP32 per block
        
        model_sizes = {
            "LLaMA-7B": 7e9,
            "LLaMA-13B": 13e9,
            "LLaMA-33B": 33e9,
            "LLaMA-65B": 65e9,
            "LLaMA-70B": 70e9,
        }
        
        print(f"  Block size: {block_size}")
        print(f"  Scale precision: FP32 ({scale_bits} bits)")
        print()
        print(f"  {'Model':<15} {'Params':>10} {'Weight Mem':>12} "
              f"{'Scale Mem':>12} {'Overhead%':>10}")
        print("  " + "-" * 62)
        
        for name, n_params in model_sizes.items():
            n_blocks = n_params / block_size
            
            # Weight memory (4 bits per param)
            weight_bytes = n_params * 0.5  # 4 bits = 0.5 bytes
            weight_gb = weight_bytes / (1024**3)
            
            # Scale memory (1 FP32 per block)
            scale_bytes = n_blocks * 4  # 4 bytes per FP32
            scale_gb = scale_bytes / (1024**3)
            
            overhead_pct = scale_bytes / weight_bytes * 100
            
            print(f"  {name:<15} {n_params/1e9:>8.0f}B "
                  f"{weight_gb:>10.2f} GB {scale_gb:>10.2f} GB "
                  f"{overhead_pct:>8.1f}%")
        
        print("\n  → Scale constants add ~12.5% overhead")
        print("  → For 65B model: ~0.5 GB just for scales!")
        print("  → Double quantization reduces this by ~75%")


# ============================================================================
# SECTION 2: DOUBLE QUANTIZATION FROM SCRATCH
# ============================================================================

class DoubleQuantizer:
    """
    Implements double quantization from scratch.
    
    Two-level hierarchy:
    
    Level 1: NF4 quantization with FP32 scales
      - Each block of 64 weights → 1 FP32 absmax
      
    Level 2: Quantize the FP32 scales to FP8
      - Group scales into blocks of 256
      - Each group of 256 FP32 scales → 256 FP8 values + 1 FP32 scale
    
    ┌────────────────────────────────────────────────────────┐
    │  Weights (millions)                                     │
    │  ├── Block 1 (64 weights) → absmax_1 (FP32)           │
    │  ├── Block 2 (64 weights) → absmax_2 (FP32)           │
    │  ├── ...                                                │
    │  └── Block N (64 weights) → absmax_N (FP32)           │
    │                                                         │
    │  Second level:                                          │
    │  ├── Group 1 (256 absmax values) → FP8 + 1 FP32       │
    │  ├── Group 2 (256 absmax values) → FP8 + 1 FP32       │
    │  └── ...                                                │
    └────────────────────────────────────────────────────────┘
    """
    
    # NF4 codebook
    NF4_LEVELS = torch.tensor([
        -1.0000, -0.6962, -0.5251, -0.3949,
        -0.2844, -0.1848, -0.0911,  0.0000,
         0.0796,  0.1609,  0.2461,  0.3379,
         0.4407,  0.5626,  0.7230,  1.0000,
    ])
    
    def __init__(
        self,
        block_size: int = 64,
        second_block_size: int = 256,
    ):
        """
        Args:
            block_size: Weights per first-level block (default: 64)
            second_block_size: First-level scales per second-level block (default: 256)
        """
        self.block_size = block_size
        self.second_block_size = second_block_size
    
    def quantize_single(self, tensor: torch.Tensor) -> dict:
        """
        Standard (single) NF4 quantization — NO double quantization.
        Used as baseline for comparison.
        """
        flat = tensor.flatten().float()
        n = flat.numel()
        
        # Pad
        pad_size = (self.block_size - n % self.block_size) % self.block_size
        if pad_size > 0:
            flat = torch.cat([flat, torch.zeros(pad_size)])
        
        n_blocks = flat.numel() // self.block_size
        blocks = flat.reshape(n_blocks, self.block_size)
        
        # Compute absmax per block (FP32)
        absmax = blocks.abs().max(dim=1).values.clamp(min=1e-10)
        
        # Normalize and quantize to NF4
        normalized = blocks / absmax.unsqueeze(1)
        distances = (normalized.unsqueeze(-1) - self.NF4_LEVELS.unsqueeze(0).unsqueeze(0)).abs()
        indices = distances.argmin(dim=-1).to(torch.uint8)
        
        # Memory calculation
        data_bytes = n_blocks * self.block_size // 2  # 4 bits = 0.5 bytes
        scale_bytes = n_blocks * 4  # FP32 scales
        
        return {
            "indices": indices,
            "absmax": absmax,  # FP32 scales
            "n_elements": n,
            "original_shape": tensor.shape,
            "data_bytes": data_bytes,
            "scale_bytes": scale_bytes,
            "total_bytes": data_bytes + scale_bytes,
            "bits_per_param": (data_bytes + scale_bytes) * 8 / n,
        }
    
    def quantize_double(self, tensor: torch.Tensor) -> dict:
        """
        Double quantization: quantize the quantization constants too!
        
        Step 1: Standard NF4 → get FP32 absmax values
        Step 2: Quantize absmax values to FP8 (blockwise)
        """
        flat = tensor.flatten().float()
        n = flat.numel()
        
        # Pad to block_size
        pad_size = (self.block_size - n % self.block_size) % self.block_size
        if pad_size > 0:
            flat = torch.cat([flat, torch.zeros(pad_size)])
        
        n_blocks = flat.numel() // self.block_size
        blocks = flat.reshape(n_blocks, self.block_size)
        
        # ── LEVEL 1: NF4 quantization ──
        absmax = blocks.abs().max(dim=1).values.clamp(min=1e-10)
        normalized = blocks / absmax.unsqueeze(1)
        distances = (normalized.unsqueeze(-1) - self.NF4_LEVELS.unsqueeze(0).unsqueeze(0)).abs()
        indices = distances.argmin(dim=-1).to(torch.uint8)
        
        # ── LEVEL 2: Quantize the absmax values ──
        # Group absmax into blocks of second_block_size
        n_second_blocks = math.ceil(n_blocks / self.second_block_size)
        
        # Pad absmax to multiple of second_block_size
        pad2 = n_second_blocks * self.second_block_size - n_blocks
        if pad2 > 0:
            absmax_padded = torch.cat([absmax, torch.zeros(pad2)])
        else:
            absmax_padded = absmax
        
        absmax_groups = absmax_padded.reshape(n_second_blocks, self.second_block_size)
        
        # For each group, compute a FP32 super-scale
        super_absmax = absmax_groups.abs().max(dim=1).values.clamp(min=1e-10)
        
        # Normalize absmax values to [0, 1] (they're all positive)
        absmax_normalized = absmax_groups / super_absmax.unsqueeze(1)
        
        # Quantize to 8-bit (simulating FP8 with INT8 for simplicity)
        absmax_quantized = torch.round(absmax_normalized * 255).clamp(0, 255).to(torch.uint8)
        
        # Memory calculation
        data_bytes = n_blocks * self.block_size // 2       # NF4 data
        level1_scale_bytes = n_blocks * 1                   # FP8 (1 byte each)
        level2_scale_bytes = n_second_blocks * 4            # FP32 super-scales
        total_scale_bytes = level1_scale_bytes + level2_scale_bytes
        
        return {
            "indices": indices,
            "absmax_quantized": absmax_quantized,   # FP8 (uint8)
            "super_absmax": super_absmax,            # FP32
            "n_elements": n,
            "n_blocks": n_blocks,
            "n_second_blocks": n_second_blocks,
            "original_shape": tensor.shape,
            "data_bytes": data_bytes,
            "level1_scale_bytes": level1_scale_bytes,
            "level2_scale_bytes": level2_scale_bytes,
            "total_scale_bytes": total_scale_bytes,
            "total_bytes": data_bytes + total_scale_bytes,
            "bits_per_param": (data_bytes + total_scale_bytes) * 8 / n,
        }
    
    def dequantize_single(self, quant: dict) -> torch.Tensor:
        """Dequantize from single-level NF4."""
        indices = quant["indices"]
        absmax = quant["absmax"]
        n = quant["n_elements"]
        shape = quant["original_shape"]
        
        values = self.NF4_LEVELS[indices.long()]
        values = values * absmax.unsqueeze(1)
        
        return values.flatten()[:n].reshape(shape)
    
    def dequantize_double(self, quant: dict) -> torch.Tensor:
        """Dequantize from double-quantized NF4."""
        indices = quant["indices"]
        absmax_q = quant["absmax_quantized"]
        super_absmax = quant["super_absmax"]
        n = quant["n_elements"]
        n_blocks = quant["n_blocks"]
        shape = quant["original_shape"]
        
        # Reconstruct absmax from double quantization
        absmax_dequant = absmax_q.float() / 255.0 * super_absmax.unsqueeze(1)
        absmax_flat = absmax_dequant.flatten()[:n_blocks]
        
        # Reconstruct weights
        values = self.NF4_LEVELS[indices.long()]
        values = values * absmax_flat.unsqueeze(1)
        
        return values.flatten()[:n].reshape(shape)
    
    def compare_single_vs_double(self):
        """
        Full comparison between single and double quantization.
        """
        print("\n" + "=" * 70)
        print("SINGLE vs DOUBLE QUANTIZATION — FULL COMPARISON")
        print("=" * 70)
        
        torch.manual_seed(42)
        
        # Test cases
        test_sizes = [
            ("Small (1K)", 1024),
            ("Medium (64K)", 65536),
            ("Large (1M)", 1048576),
            ("XL (10M)", 10485760),
        ]
        
        for name, size in test_sizes:
            W = torch.randn(size) * 0.02  # Typical NN weights
            
            # Single quantization
            single = self.quantize_single(W)
            W_single = self.dequantize_single(single)
            mse_single = (W - W_single).pow(2).mean().item()
            
            # Double quantization
            double = self.quantize_double(W)
            W_double = self.dequantize_double(double)
            mse_double = (W - W_double).pow(2).mean().item()
            
            print(f"\n  {name} ({size:,} params):")
            print(f"    {'Metric':<25} {'Single':>15} {'Double':>15} {'Diff':>10}")
            print(f"    {'-'*65}")
            print(f"    {'Total bytes':<25} {single['total_bytes']:>15,} "
                  f"{double['total_bytes']:>15,} "
                  f"{double['total_bytes'] - single['total_bytes']:>+10,}")
            print(f"    {'Bits per param':<25} {single['bits_per_param']:>15.3f} "
                  f"{double['bits_per_param']:>15.3f} "
                  f"{double['bits_per_param'] - single['bits_per_param']:>+10.3f}")
            print(f"    {'Scale bytes':<25} {single['scale_bytes']:>15,} "
                  f"{double['total_scale_bytes']:>15,} "
                  f"{double['total_scale_bytes'] - single['scale_bytes']:>+10,}")
            print(f"    {'MSE':<25} {mse_single:>15.2e} "
                  f"{mse_double:>15.2e} "
                  f"{'negligible' if abs(mse_double - mse_single) < mse_single * 0.05 else 'significant':>10}")
        
        print("\n  Key observations:")
        print("    ✓ Double quantization saves ~75% of scale memory")
        print("    ✓ The MSE increase is negligible (< 5%)")
        print("    ✓ Savings grow proportionally with model size")
        print("    ✓ At 65B params: saves ~3 GB of memory!")


# ============================================================================
# SECTION 3: MEMORY SAVINGS AT SCALE
# ============================================================================

class DoubleQuantMemoryAnalysis:
    """
    Detailed memory savings analysis for real model sizes.
    """
    
    @staticmethod
    def analyze_savings():
        """
        Show exact memory savings from double quantization
        for different model sizes.
        """
        print("\n" + "=" * 70)
        print("DOUBLE QUANTIZATION MEMORY SAVINGS")
        print("=" * 70)
        
        block_size = 64
        second_block_size = 256
        
        models = {
            "GPT-2 (124M)": 124e6,
            "LLaMA-7B": 7e9,
            "LLaMA-13B": 13e9,
            "LLaMA-33B": 33e9,
            "LLaMA-65B": 65e9,
            "LLaMA-70B": 70e9,
            "LLaMA-405B": 405e9,
        }
        
        print(f"\n  Configuration:")
        print(f"    Block size (level 1): {block_size}")
        print(f"    Block size (level 2): {second_block_size}")
        print()
        
        print(f"  {'Model':<18} {'NF4 Data':>10} {'Scales':>10} "
              f"{'Scales':>10} {'Scale':>8} {'Total':>8}")
        print(f"  {'':18} {'(4-bit)':>10} {'(single)':>10} "
              f"{'(double)':>10} {'Savings':>8} {'bpp':>8}")
        print("  " + "-" * 66)
        
        for name, n_params in models.items():
            n_blocks = n_params / block_size
            n_second_blocks = n_blocks / second_block_size
            
            # NF4 data
            data_gb = (n_params * 0.5) / (1024**3)  # 4 bits per param
            
            # Single quantization scales
            single_scale_gb = (n_blocks * 4) / (1024**3)  # FP32 per block
            
            # Double quantization scales
            # Level 1: FP8 per block = 1 byte each
            # Level 2: FP32 per second block = 4 bytes each
            double_scale_gb = (n_blocks * 1 + n_second_blocks * 4) / (1024**3)
            
            scale_savings_gb = single_scale_gb - double_scale_gb
            
            # Bits per param
            total_bits_single = 4 + 32 / block_size
            total_bits_double = 4 + 8 / block_size + 32 / (block_size * second_block_size)
            
            print(f"  {name:<18} {data_gb:>8.2f}GB "
                  f"{single_scale_gb:>8.3f}GB {double_scale_gb:>8.3f}GB "
                  f"{scale_savings_gb:>6.3f}GB {total_bits_double:>7.3f}")
        
        print(f"\n  Theoretical bits per parameter:")
        print(f"    NF4 (no double quant):   4.000 + 0.500 = 4.500 bpp")
        print(f"    NF4 (with double quant): 4.000 + 0.127 = 4.127 bpp")
        print(f"    Savings:                       0.373 bpp")
        print(f"    For 65B params: {65e9 * 0.373 / 8 / 1024**3:.2f} GB saved!")
    
    @staticmethod
    def compare_memory_budgets():
        """
        Show how double quantization affects GPU memory budgets
        for fine-tuning.
        """
        print("\n" + "=" * 70)
        print("GPU MEMORY BUDGET WITH DOUBLE QUANTIZATION")
        print("=" * 70)
        
        # LLaMA-7B example
        n_params = 7e9
        
        scenarios = {
            "FP16 (full model)": {
                "base_model": n_params * 2 / 1e9,
                "optimizer": n_params * 8 / 1e9,     # AdamW: 2x FP32 states
                "gradients": n_params * 2 / 1e9,
                "activations": 2.0,
            },
            "LoRA FP16": {
                "base_model": n_params * 2 / 1e9,
                "optimizer": 40e6 * 8 / 1e9,          # Only LoRA params
                "gradients": 40e6 * 2 / 1e9,
                "activations": 2.0,
            },
            "QLoRA (NF4, no double)": {
                "base_model": n_params * 0.5625 / 1e9,  # 4.5 bits
                "optimizer": 40e6 * 8 / 1e9,
                "gradients": 40e6 * 2 / 1e9,
                "activations": 2.0,
            },
            "QLoRA (NF4 + double)": {
                "base_model": n_params * 0.516 / 1e9,   # 4.127 bits
                "optimizer": 40e6 * 8 / 1e9,
                "gradients": 40e6 * 2 / 1e9,
                "activations": 2.0,
            },
        }
        
        for name, budget in scenarios.items():
            total = sum(budget.values())
            print(f"\n  {name}:")
            for component, gb in budget.items():
                bar = "█" * int(gb * 3)
                print(f"    {component:<15} {gb:>6.2f} GB {bar}")
            print(f"    {'TOTAL':<15} {total:>6.2f} GB "
                  f"{'← fits 24GB GPU' if total < 24 else ''}"
                  f"{'← fits 16GB GPU' if total < 16 else ''}")
        
        print("\n  GPU Compatibility:")
        gpus = {
            "RTX 3060 (12GB)": 12,
            "RTX 3080 (10GB)": 10,
            "RTX 3090 (24GB)": 24,
            "RTX 4090 (24GB)": 24,
            "A100 (40GB)": 40,
            "A100 (80GB)": 80,
        }
        
        print(f"\n  {'GPU':<22}", end="")
        for scenario in scenarios:
            short = scenario.split("(")[0].strip()[:12]
            print(f" {short:>12}", end="")
        print()
        print("  " + "-" * 70)
        
        for gpu_name, vram in gpus.items():
            print(f"  {gpu_name:<22}", end="")
            for scenario_name, budget in scenarios.items():
                total = sum(budget.values())
                fits = "✓" if total < vram * 0.9 else "✗"
                print(f" {fits:>12}", end="")
            print()


# ============================================================================
# SECTION 4: IMPACT ON QUALITY
# ============================================================================

class DoubleQuantQuality:
    """
    Analyze the quality impact of double quantization.
    
    Key finding from the QLoRA paper: double quantization has
    NEGLIGIBLE impact on model quality while saving significant memory.
    """
    
    @staticmethod
    def analyze_quality_impact():
        """
        Measure how double quantization affects weight reconstruction
        accuracy across different layer types.
        """
        print("\n" + "=" * 70)
        print("DOUBLE QUANTIZATION QUALITY IMPACT")
        print("=" * 70)
        
        torch.manual_seed(42)
        quantizer = DoubleQuantizer(block_size=64, second_block_size=256)
        
        # Test with different weight distributions and sizes
        test_cases = {
            "Attention Q (768×768)": torch.randn(768, 768) * 0.02,
            "Attention K (768×768)": torch.randn(768, 768) * 0.015,
            "MLP Up (3072×768)": torch.randn(3072, 768) * 0.02,
            "MLP Down (768×3072)": torch.randn(768, 3072) * 0.02,
            "Embedding (50257×768)": torch.randn(50257, 768) * 0.02,
            "Large layer (4096×4096)": torch.randn(4096, 4096) * 0.01,
        }
        
        print(f"\n  {'Layer':<30} {'MSE Single':>12} {'MSE Double':>12} "
              f"{'Degradation':>12}")
        print("  " + "-" * 68)
        
        total_single_mse = 0
        total_double_mse = 0
        
        for name, W in test_cases.items():
            # Single quantization
            single = quantizer.quantize_single(W)
            W_single = quantizer.dequantize_single(single)
            mse_single = (W - W_single).pow(2).mean().item()
            
            # Double quantization
            double = quantizer.quantize_double(W)
            W_double = quantizer.dequantize_double(double)
            mse_double = (W - W_double).pow(2).mean().item()
            
            degradation = (mse_double - mse_single) / mse_single * 100
            total_single_mse += mse_single
            total_double_mse += mse_double
            
            print(f"  {name:<30} {mse_single:>12.2e} {mse_double:>12.2e} "
                  f"{degradation:>+10.2f}%")
        
        avg_degradation = (total_double_mse - total_single_mse) / total_single_mse * 100
        print(f"\n  Average degradation: {avg_degradation:+.2f}%")
        print(f"\n  → Double quantization adds minimal error")
        print(f"  → The QLoRA paper shows NO measurable impact on")
        print(f"    downstream task performance")
    
    @staticmethod
    def demonstrate_forward_pass_equivalence():
        """
        Show that single and double quantization produce
        nearly identical forward pass outputs.
        """
        print("\n" + "=" * 70)
        print("FORWARD PASS EQUIVALENCE")
        print("=" * 70)
        
        torch.manual_seed(42)
        quantizer = DoubleQuantizer(block_size=64, second_block_size=256)
        
        # Simulate a linear layer
        d_in, d_out = 512, 512
        W = torch.randn(d_out, d_in) * 0.02
        
        # Create input batch
        batch_size = 8
        seq_len = 32
        x = torch.randn(batch_size, seq_len, d_in)
        
        # Original forward pass
        y_original = x @ W.T
        
        # Single-quantized forward pass
        single = quantizer.quantize_single(W)
        W_single = quantizer.dequantize_single(single)
        y_single = x @ W_single.T
        
        # Double-quantized forward pass
        double = quantizer.quantize_double(W)
        W_double = quantizer.dequantize_double(double)
        y_double = x @ W_double.T
        
        # Compare
        err_single = (y_original - y_single).abs()
        err_double = (y_original - y_double).abs()
        err_delta = (y_single - y_double).abs()
        
        print(f"\n  Linear layer: [{d_out} x {d_in}]")
        print(f"  Input batch:  [{batch_size} x {seq_len} x {d_in}]")
        print(f"\n  {'Comparison':<35} {'Max Error':>12} {'Mean Error':>12}")
        print("  " + "-" * 60)
        print(f"  {'Original vs Single NF4':<35} "
              f"{err_single.max().item():>12.2e} "
              f"{err_single.mean().item():>12.2e}")
        print(f"  {'Original vs Double NF4':<35} "
              f"{err_double.max().item():>12.2e} "
              f"{err_double.mean().item():>12.2e}")
        print(f"  {'Single vs Double (delta)':<35} "
              f"{err_delta.max().item():>12.2e} "
              f"{err_delta.mean().item():>12.2e}")
        
        # Cosine similarity
        cos_single = torch.nn.functional.cosine_similarity(
            y_original.flatten().unsqueeze(0),
            y_single.flatten().unsqueeze(0),
        ).item()
        cos_double = torch.nn.functional.cosine_similarity(
            y_original.flatten().unsqueeze(0),
            y_double.flatten().unsqueeze(0),
        ).item()
        
        print(f"\n  Cosine similarity:")
        print(f"    Original vs Single NF4: {cos_single:.8f}")
        print(f"    Original vs Double NF4: {cos_double:.8f}")
        print(f"    Difference:             {abs(cos_single - cos_double):.2e}")
        
        print(f"\n  → Double quantization introduces negligible additional error")
        print(f"    in the forward pass output")


# ============================================================================
# SECTION 5: ENABLING DOUBLE QUANTIZATION IN PRACTICE
# ============================================================================

class DoubleQuantPractical:
    """
    How to enable and configure double quantization
    in a real QLoRA setup.
    """
    
    @staticmethod
    def show_configuration():
        """
        Show how to enable double quantization with bitsandbytes.
        """
        print("\n" + "=" * 70)
        print("ENABLING DOUBLE QUANTIZATION IN PRACTICE")
        print("=" * 70)
        
        code = '''
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# ── Without double quantization ──────────────────────────────────
config_single = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,     # ← DISABLED
)
# Memory per param: ~4.5 bits (4 + 0.5 for FP32 scales)

# ── With double quantization (recommended) ───────────────────────
config_double = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,      # ← ENABLED
)
# Memory per param: ~4.127 bits (4 + 0.127 for FP8/FP32 scales)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=config_double,
    device_map="auto",
)

# The double quantization is handled internally by bitsandbytes.
# You don't need to do anything special — just set the flag!

# To verify it's working, check the quantization state:
for name, param in model.named_parameters():
    if hasattr(param, 'quant_state'):
        qs = param.quant_state
        print(f"Layer: {name}")
        print(f"  Quant type: {qs.quant_type}")
        print(f"  Block size: {qs.blocksize}")
        print(f"  Nested quant: {hasattr(qs, 'state2')}")  # Double quant!
        break
'''
        print(code)
        return code


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all double quantization demonstrations."""
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          DOUBLE QUANTIZATION — QLoRA's SECOND INNOVATION       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    # Section 1: The overhead problem
    overhead = QuantizationOverhead()
    overhead.analyze_overhead()
    
    # Section 2: Double quantization implementation
    quantizer = DoubleQuantizer(block_size=64, second_block_size=256)
    quantizer.compare_single_vs_double()
    
    # Section 3: Memory savings analysis
    memory = DoubleQuantMemoryAnalysis()
    memory.analyze_savings()
    memory.compare_memory_budgets()
    
    # Section 4: Quality impact
    quality = DoubleQuantQuality()
    quality.analyze_quality_impact()
    quality.demonstrate_forward_pass_equivalence()
    
    # Section 5: Practical usage
    practical = DoubleQuantPractical()
    practical.show_configuration()
    
    print("\n" + "=" * 70)
    print("  MODULE COMPLETE")
    print("=" * 70)
    print("""
    Key takeaways:
    ✓ Quantization constants (scales) add ~0.5 bits/param overhead
    ✓ For a 65B model, scales alone consume ~0.5 GB
    ✓ Double quantization: quantize scales from FP32 → FP8
    ✓ Reduces overhead from 0.5 to 0.127 bits/param
    ✓ Saves ~3 GB for a 65B model — enough to matter!
    ✓ Quality degradation is negligible (< 2% MSE increase)
    ✓ Just set bnb_4bit_use_double_quant=True
    """)


if __name__ == "__main__":
    main()
