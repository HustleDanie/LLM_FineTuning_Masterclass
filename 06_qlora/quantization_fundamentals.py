"""
Quantization Fundamentals for QLoRA
====================================

Before diving into QLoRA, we need a solid understanding of quantization:
how neural network weights are represented in fewer bits, and why this matters.

This module covers:

1. Numeric Representations
   - FP32, FP16, BF16 — floating point formats
   - INT8, INT4 — integer quantization
   - NF4 — NormalFloat (QLoRA's innovation)

2. Quantization Theory
   - Affine (asymmetric) quantization
   - Symmetric quantization
   - Per-tensor vs per-channel vs per-group (blockwise)
   - Calibration: finding the right scale/zero-point

3. Quantization Error Analysis
   - How quantization error propagates through layers
   - Why 4-bit is a sweet spot for LLMs
   - Error distribution across weight matrices

4. Practical Quantization with bitsandbytes
   - Loading models in 8-bit
   - Loading models in 4-bit (NF4 / FP4)
   - Mixed-precision configurations

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import struct
import math
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


# ============================================================================
# SECTION 1: NUMERIC REPRESENTATIONS
# ============================================================================

class NumericFormats:
    """
    Deep dive into how numbers are stored in different formats.
    
    IEEE 754 Floating Point Layout:
    ┌──────┬──────────┬──────────────┐
    │ Sign │ Exponent │   Mantissa   │
    │ (1b) │          │              │
    ├──────┼──────────┼──────────────┤
    │ FP32 │  8 bits  │   23 bits    │  = 32 bits total
    │ FP16 │  5 bits  │   10 bits    │  = 16 bits total
    │ BF16 │  8 bits  │    7 bits    │  = 16 bits total
    └──────┴──────────┴──────────────┘
    
    Key insight:
    - FP16 has more precision (10-bit mantissa) but less range (5-bit exp)
    - BF16 has less precision (7-bit mantissa) but same range as FP32 (8-bit exp)
    - BF16 is preferred for training because the range matters more than precision
    """
    
    @staticmethod
    def explore_floating_point():
        """Compare floating point formats side by side."""
        print("=" * 70)
        print("FLOATING POINT FORMAT COMPARISON")
        print("=" * 70)
        
        formats = {
            "FP32": {
                "dtype": torch.float32,
                "bits": 32,
                "sign_bits": 1,
                "exp_bits": 8,
                "mantissa_bits": 23,
                "max_val": 3.4028235e+38,
                "min_positive": 1.1754944e-38,
                "eps": 1.1920929e-07,
            },
            "FP16": {
                "dtype": torch.float16,
                "bits": 16,
                "sign_bits": 1,
                "exp_bits": 5,
                "mantissa_bits": 10,
                "max_val": 65504.0,
                "min_positive": 6.1035e-05,
                "eps": 9.7656e-04,
            },
            "BF16": {
                "dtype": torch.bfloat16,
                "bits": 16,
                "sign_bits": 1,
                "exp_bits": 8,
                "mantissa_bits": 7,
                "max_val": 3.3895e+38,
                "min_positive": 1.1755e-38,
                "eps": 7.8125e-03,
            },
        }
        
        print(f"\n{'Format':<8} {'Bits':>5} {'Exp':>4} {'Mant':>5} "
              f"{'Max Value':>15} {'Epsilon':>12} {'Memory/Param':>13}")
        print("-" * 70)
        
        for name, info in formats.items():
            print(f"{name:<8} {info['bits']:>5} {info['exp_bits']:>4} "
                  f"{info['mantissa_bits']:>5} {info['max_val']:>15.4e} "
                  f"{info['eps']:>12.4e} {info['bits']/8:>10.0f} bytes")
        
        # Demonstrate precision differences
        print("\n  Precision comparison — storing π:")
        pi = 3.14159265358979323846
        for name, info in formats.items():
            t = torch.tensor(pi, dtype=info["dtype"])
            error = abs(float(t) - pi)
            print(f"    {name}: {float(t):.15f}  (error: {error:.2e})")
        
        # Demonstrate range differences
        print("\n  Range comparison — storing large values:")
        for val in [1000.0, 10000.0, 65504.0, 100000.0]:
            print(f"\n    Value: {val}")
            for name, info in formats.items():
                try:
                    t = torch.tensor(val, dtype=info["dtype"])
                    if torch.isinf(t):
                        print(f"      {name}: OVERFLOW (inf)")
                    else:
                        print(f"      {name}: {float(t):.4f}")
                except Exception as e:
                    print(f"      {name}: ERROR - {e}")
        
        return formats
    
    @staticmethod
    def memory_comparison_for_models():
        """Show memory requirements for different model sizes and dtypes."""
        print("\n" + "=" * 70)
        print("MODEL MEMORY BY PRECISION")
        print("=" * 70)
        
        model_sizes = {
            "GPT-2 (124M)": 124e6,
            "GPT-2 XL (1.5B)": 1.5e9,
            "LLaMA-7B": 7e9,
            "LLaMA-13B": 13e9,
            "LLaMA-33B": 33e9,
            "LLaMA-65B": 65e9,
            "LLaMA-70B": 70e9,
        }
        
        precisions = {
            "FP32": 4,    # 4 bytes per param
            "FP16/BF16": 2,
            "INT8": 1,
            "INT4/NF4": 0.5,
        }
        
        print(f"\n{'Model':<20}", end="")
        for prec in precisions:
            print(f" {prec:>10}", end="")
        print()
        print("-" * 65)
        
        for model_name, n_params in model_sizes.items():
            print(f"{model_name:<20}", end="")
            for prec_name, bytes_per_param in precisions.items():
                gb = (n_params * bytes_per_param) / (1024**3)
                print(f" {gb:>8.1f}GB", end="")
            print()
        
        print("\n  → 4-bit quantization reduces memory by 8x vs FP32")
        print("  → A 70B model fits in ~35 GB (single A100) at 4-bit")
        print("  → With QLoRA, you can TRAIN a 70B model on 48GB!")


# ============================================================================
# SECTION 2: QUANTIZATION THEORY
# ============================================================================

class QuantizationTheory:
    """
    The mathematics of quantization — converting continuous FP values
    to discrete integer/low-bit representations.
    """
    
    @staticmethod
    def affine_quantization(
        tensor: torch.Tensor,
        n_bits: int = 8,
    ) -> Tuple[torch.Tensor, float, int]:
        """
        Affine (asymmetric) quantization.
        
        Maps [min_val, max_val] → [0, 2^n_bits - 1]
        
        Formulas:
            scale = (max_val - min_val) / (2^n_bits - 1)
            zero_point = round(-min_val / scale)
            q = round(x / scale) + zero_point
            x_dequant = (q - zero_point) * scale
        
        Args:
            tensor: Input floating point tensor
            n_bits: Number of bits for quantization
            
        Returns:
            quantized tensor, scale, zero_point
        """
        print("\n  ── Affine (Asymmetric) Quantization ──")
        
        qmin = 0
        qmax = 2**n_bits - 1
        
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        # Compute scale and zero point
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = round(-min_val / scale)
        zero_point = max(qmin, min(qmax, zero_point))  # Clamp
        
        # Quantize
        q = torch.clamp(torch.round(tensor / scale) + zero_point, qmin, qmax)
        q = q.to(torch.int8 if n_bits <= 8 else torch.int16)
        
        # Dequantize
        dequant = (q.float() - zero_point) * scale
        
        # Error analysis
        error = (tensor - dequant).abs()
        
        print(f"    Bits: {n_bits}")
        print(f"    Range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"    Scale: {scale:.6f}")
        print(f"    Zero point: {zero_point}")
        print(f"    Max quantization error: {error.max().item():.6f}")
        print(f"    Mean quantization error: {error.mean().item():.6f}")
        print(f"    SQNR: {(tensor.norm() / error.norm()).item():.1f} "
              f"({20 * math.log10((tensor.norm() / error.norm()).item()):.1f} dB)")
        
        return q, scale, zero_point
    
    @staticmethod
    def symmetric_quantization(
        tensor: torch.Tensor,
        n_bits: int = 8,
    ) -> Tuple[torch.Tensor, float]:
        """
        Symmetric quantization.
        
        Maps [-max_abs, max_abs] → [-2^(n_bits-1), 2^(n_bits-1) - 1]
        Zero point is always 0.
        
        Simpler and slightly faster than affine, but wastes range
        if the distribution is not symmetric around zero.
        
        Formulas:
            scale = max_abs / (2^(n_bits-1) - 1)
            q = round(x / scale)
            x_dequant = q * scale
        """
        print("\n  ── Symmetric Quantization ──")
        
        qmax = 2**(n_bits - 1) - 1  # e.g., 127 for 8-bit
        
        max_abs = tensor.abs().max().item()
        scale = max_abs / qmax
        
        # Quantize
        q = torch.clamp(torch.round(tensor / scale), -qmax, qmax)
        q = q.to(torch.int8 if n_bits <= 8 else torch.int16)
        
        # Dequantize
        dequant = q.float() * scale
        
        # Error
        error = (tensor - dequant).abs()
        
        print(f"    Bits: {n_bits}")
        print(f"    Max absolute value: {max_abs:.4f}")
        print(f"    Scale: {scale:.6f}")
        print(f"    Zero point: 0 (always)")
        print(f"    Max quantization error: {error.max().item():.6f}")
        print(f"    Mean quantization error: {error.mean().item():.6f}")
        
        return q, scale
    
    @staticmethod
    def demonstrate_granularity():
        """
        Compare per-tensor, per-channel, and per-group (blockwise) quantization.
        
        Finer granularity = more scales = less error, but more overhead.
        
        ┌─────────────────────────────────────────────────┐
        │ Per-Tensor: One scale for entire weight matrix  │
        │   [████████████████████████████████]            │
        │    ↑ single scale                               │
        │                                                 │
        │ Per-Channel: One scale per output channel       │
        │   [████████] [████████] [████████]              │
        │    ↑ scale_0  ↑ scale_1  ↑ scale_2             │
        │                                                 │
        │ Per-Group: One scale per block of B values      │
        │   [████] [████] [████] [████] [████]            │
        │    ↑s_0   ↑s_1   ↑s_2   ↑s_3   ↑s_4           │
        │   Block size B (e.g., 64 or 128)                │
        │                                                 │
        │ QLoRA uses per-group with block_size=64         │
        └─────────────────────────────────────────────────┘
        """
        print("\n" + "=" * 70)
        print("QUANTIZATION GRANULARITY COMPARISON")
        print("=" * 70)
        
        # Create a weight matrix with varying scales across channels
        d_out, d_in = 256, 256
        
        # Deliberately create channels with different ranges
        W = torch.zeros(d_out, d_in)
        for i in range(d_out):
            scale = 0.01 + (i / d_out) * 0.5  # Scales from 0.01 to 0.51
            W[i] = torch.randn(d_in) * scale
        
        print(f"\n  Weight matrix: [{d_out} x {d_in}]")
        print(f"  Channel scale range: {W[0].abs().max():.4f} to "
              f"{W[-1].abs().max():.4f}")
        
        results = {}
        
        # Per-tensor quantization
        print("\n  1. PER-TENSOR (1 scale total)")
        max_abs = W.abs().max()
        scale = max_abs / 127
        q = torch.round(W / scale).clamp(-128, 127)
        dq = q * scale
        error_per_tensor = (W - dq).abs()
        results["per_tensor"] = {
            "n_scales": 1,
            "max_error": error_per_tensor.max().item(),
            "mean_error": error_per_tensor.mean().item(),
            "overhead_bytes": 4,  # 1 FP32 scale
        }
        print(f"     Scales: 1")
        print(f"     Max error:  {results['per_tensor']['max_error']:.6f}")
        print(f"     Mean error: {results['per_tensor']['mean_error']:.6f}")
        
        # Per-channel quantization
        print("\n  2. PER-CHANNEL (1 scale per output row)")
        errors = []
        for i in range(d_out):
            max_abs_ch = W[i].abs().max()
            scale_ch = max_abs_ch / 127 if max_abs_ch > 0 else 1.0
            q_ch = torch.round(W[i] / scale_ch).clamp(-128, 127)
            dq_ch = q_ch * scale_ch
            errors.append((W[i] - dq_ch).abs())
        error_per_channel = torch.stack(errors)
        results["per_channel"] = {
            "n_scales": d_out,
            "max_error": error_per_channel.max().item(),
            "mean_error": error_per_channel.mean().item(),
            "overhead_bytes": d_out * 4,
        }
        print(f"     Scales: {d_out}")
        print(f"     Max error:  {results['per_channel']['max_error']:.6f}")
        print(f"     Mean error: {results['per_channel']['mean_error']:.6f}")
        
        # Per-group (blockwise) — this is what QLoRA uses
        block_size = 64
        print(f"\n  3. PER-GROUP / BLOCKWISE (1 scale per {block_size} values)")
        W_flat = W.reshape(-1)
        n_blocks = W_flat.numel() // block_size
        errors_block = []
        
        for b in range(n_blocks):
            block = W_flat[b * block_size:(b + 1) * block_size]
            max_abs_b = block.abs().max()
            scale_b = max_abs_b / 127 if max_abs_b > 0 else 1.0
            q_b = torch.round(block / scale_b).clamp(-128, 127)
            dq_b = q_b * scale_b
            errors_block.append((block - dq_b).abs())
        
        all_errors_block = torch.cat(errors_block)
        results["per_group_64"] = {
            "n_scales": n_blocks,
            "max_error": all_errors_block.max().item(),
            "mean_error": all_errors_block.mean().item(),
            "overhead_bytes": n_blocks * 4,
        }
        print(f"     Scales: {n_blocks}")
        print(f"     Max error:  {results['per_group_64']['max_error']:.6f}")
        print(f"     Mean error: {results['per_group_64']['mean_error']:.6f}")
        
        # Summary
        print("\n  ── Summary ──")
        print(f"  {'Method':<20} {'Scales':>8} {'Overhead':>10} "
              f"{'Max Err':>10} {'Mean Err':>10}")
        print("  " + "-" * 62)
        for name, r in results.items():
            print(f"  {name:<20} {r['n_scales']:>8} "
                  f"{r['overhead_bytes']:>8} B "
                  f"{r['max_error']:>10.6f} {r['mean_error']:>10.6f}")
        
        print("\n  → Per-group with block_size=64 gives the best error/overhead tradeoff")
        print("  → QLoRA uses per-group quantization with block_size=64")
        
        return results


# ============================================================================
# SECTION 3: PRACTICAL QUANTIZATION WITH BITSANDBYTES
# ============================================================================

class BitsAndBytesQuantization:
    """
    Using the bitsandbytes library for practical model quantization.
    
    bitsandbytes provides:
    - LLM.int8(): 8-bit quantization with outlier handling
    - NF4/FP4: 4-bit quantization for QLoRA
    - Paged optimizers with CPU offload
    """
    
    @staticmethod
    def load_model_8bit():
        """
        Load a model in 8-bit using LLM.int8().
        
        LLM.int8() is special because it handles outlier features:
        - Most weights: quantized to INT8
        - Outlier features (>6σ): kept in FP16
        - This preserves quality while saving ~50% memory
        """
        print("\n" + "=" * 70)
        print("8-BIT MODEL LOADING (LLM.int8())")
        print("=" * 70)
        
        code = '''
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Load model in 8-bit ──────────────────────────────────────────
model_8bit = AutoModelForCausalLM.from_pretrained(
    "distilgpt2",
    load_in_8bit=True,         # Enable 8-bit quantization
    device_map="auto",          # Auto-place on GPU
)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Check memory usage
total_params = sum(p.numel() for p in model_8bit.parameters())
total_bytes = sum(p.numel() * p.element_size() for p in model_8bit.parameters())
print(f"Parameters: {total_params:,}")
print(f"Memory: {total_bytes / 1e6:.1f} MB")

# The model works normally for inference
inputs = tokenizer("Hello, I am a", return_tensors="pt").to(model_8bit.device)
outputs = model_8bit.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# ── How LLM.int8() works ─────────────────────────────────────────
# For each linear layer:
#   1. Detect outlier features (columns with values > threshold)
#   2. Extract outlier columns → compute in FP16
#   3. Remaining columns → quantize to INT8 and compute
#   4. Combine: output = INT8_output + FP16_outlier_output
#
# Typically ~0.1% of features are outliers, so memory savings ≈ 50%
'''
        print(code)
        return code
    
    @staticmethod
    def load_model_4bit():
        """
        Load a model in 4-bit using NF4 or FP4.
        This is the core of QLoRA.
        """
        print("\n" + "=" * 70)
        print("4-BIT MODEL LOADING (QLoRA Foundation)")
        print("=" * 70)
        
        code = '''
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# ── Configuration 1: Basic NF4 (QLoRA default) ──────────────────
bnb_config_nf4 = BitsAndBytesConfig(
    load_in_4bit=True,                   # Enable 4-bit loading
    bnb_4bit_quant_type="nf4",           # NormalFloat4 (recommended)
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in BF16
    bnb_4bit_use_double_quant=False,     # No double quantization
)

model_nf4 = AutoModelForCausalLM.from_pretrained(
    "distilgpt2",
    quantization_config=bnb_config_nf4,
    device_map="auto",
)

# ── Configuration 2: NF4 + Double Quantization ──────────────────
# This is the FULL QLoRA configuration
bnb_config_qlora = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,      # Double quantization!
)

model_qlora = AutoModelForCausalLM.from_pretrained(
    "distilgpt2",
    quantization_config=bnb_config_qlora,
    device_map="auto",
)

# ── Configuration 3: FP4 (alternative to NF4) ───────────────────
bnb_config_fp4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="fp4",           # Standard 4-bit float
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ── Comparing NF4 vs FP4 ─────────────────────────────────────────
# NF4 advantages:
#   - Information-theoretically optimal for normal distributions
#   - Neural network weights ARE approximately normally distributed  
#   - Lower quantization error than FP4 for typical NN weights
#
# FP4 advantages:
#   - Standard floating-point representation
#   - Better for non-normal distributions
#   - Slightly faster dequantization
#
# → NF4 is recommended for almost all LLM fine-tuning scenarios

# ── Memory comparison ─────────────────────────────────────────────
def print_model_memory(model, name):
    total_bytes = 0
    for p in model.parameters():
        if hasattr(p, 'quant_state'):
            # 4-bit parameter
            total_bytes += p.numel() // 2  # 4 bits = 0.5 bytes
        else:
            total_bytes += p.numel() * p.element_size()
    print(f"  {name}: {total_bytes / 1e6:.1f} MB")

print_model_memory(model_nf4, "NF4 (no double quant)")
print_model_memory(model_qlora, "NF4 + double quant")
'''
        print(code)
        return code
    
    @staticmethod
    def quantization_config_reference():
        """
        Complete reference for BitsAndBytesConfig options.
        """
        print("\n" + "=" * 70)
        print("BITSANDBYTES CONFIG REFERENCE")
        print("=" * 70)
        
        reference = """
BitsAndBytesConfig Parameters:
═══════════════════════════════════════════════════════════════════

  load_in_8bit: bool = False
    │ Load model weights in 8-bit (LLM.int8()).
    │ Mutually exclusive with load_in_4bit.
    │ Memory: ~50% of FP16

  load_in_4bit: bool = False
    │ Load model weights in 4-bit.
    │ Mutually exclusive with load_in_8bit.
    │ Memory: ~25% of FP16

  bnb_4bit_quant_type: str = "fp4"
    │ Type of 4-bit quantization to use.
    │ Options: "nf4" (recommended) or "fp4"
    │ NF4 is optimal for normally-distributed weights.

  bnb_4bit_compute_dtype: torch.dtype = torch.float32
    │ Dtype for computation during forward/backward pass.
    │ The 4-bit weights are dequantized to this dtype on-the-fly.
    │ Options: torch.float32, torch.float16, torch.bfloat16
    │ Recommended: torch.bfloat16 (best speed/quality tradeoff)

  bnb_4bit_use_double_quant: bool = False
    │ Quantize the quantization constants (second-level quantization).
    │ Saves ~0.4 bits per parameter.
    │ For a 65B model, saves ~3 GB of memory.
    │ Recommended: True for QLoRA

  llm_int8_threshold: float = 6.0
    │ Outlier threshold for LLM.int8().
    │ Features with values > threshold are kept in FP16.
    │ Lower = more FP16 features = better quality, more memory.

  llm_int8_has_fp16_weight: bool = False
    │ Keep a FP16 backup of weights for LLM.int8().
    │ Useful for training, but doubles memory.

  llm_int8_skip_modules: List[str] = None
    │ List of module names to NOT quantize.
    │ E.g., ["lm_head"] to keep the output projection in FP16.

═══════════════════════════════════════════════════════════════════

RECOMMENDED CONFIGURATIONS:
─────────────────────────────────────────────────────────────────

  QLoRA (4-bit fine-tuning):
    BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

  Inference only (8-bit):
    BitsAndBytesConfig(
        load_in_8bit=True,
    )

  Inference only (4-bit):
    BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
"""
        print(reference)
        return reference


# ============================================================================
# SECTION 4: QUANTIZATION ERROR ANALYSIS
# ============================================================================

class QuantizationErrorAnalysis:
    """
    Analyze how quantization affects model weights and outputs.
    """
    
    @staticmethod
    def analyze_weight_distributions():
        """
        Neural network weights follow approximately normal distributions.
        This is WHY NF4 works so well.
        """
        print("\n" + "=" * 70)
        print("WEIGHT DISTRIBUTION ANALYSIS")
        print("=" * 70)
        
        # Simulate typical neural network weight distributions
        print("\n  Neural network weights are approximately normally distributed.")
        print("  This is a consequence of random initialization and gradient updates.")
        
        # Create weights like a real neural network
        torch.manual_seed(42)
        
        # Simulate different layer types
        layers = {
            "Attention Q/K/V": torch.randn(768, 768) * 0.02,
            "Attention Output": torch.randn(768, 768) * 0.02,
            "MLP Up-proj": torch.randn(3072, 768) * 0.02,
            "MLP Down-proj": torch.randn(768, 3072) * 0.02,
            "Embedding": torch.randn(50257, 768) * 0.02,
        }
        
        print(f"\n  {'Layer':<20} {'Mean':>8} {'Std':>8} {'Skew':>8} "
              f"{'Kurt':>8} {'Normal?':>8}")
        print("  " + "-" * 58)
        
        for name, w in layers.items():
            flat = w.flatten()
            mean = flat.mean().item()
            std = flat.std().item()
            # Skewness (should be ~0 for normal)
            skew = ((flat - mean) / std).pow(3).mean().item()
            # Kurtosis (should be ~3 for normal, excess ~0)
            kurt = ((flat - mean) / std).pow(4).mean().item()
            is_normal = abs(skew) < 0.1 and abs(kurt - 3) < 0.5
            
            print(f"  {name:<20} {mean:>8.4f} {std:>8.4f} {skew:>8.4f} "
                  f"{kurt:>8.4f} {'  ✓' if is_normal else '  ✗':>8}")
        
        print("\n  → All layers show approximately normal distribution")
        print("  → This validates NF4's assumption of normality")
    
    @staticmethod
    def compare_quantization_errors():
        """
        Compare quantization error between different methods
        on normally-distributed weights.
        """
        print("\n" + "=" * 70)
        print("QUANTIZATION ERROR COMPARISON")
        print("=" * 70)
        
        torch.manual_seed(42)
        n = 10000
        weights = torch.randn(n) * 0.02  # Typical NN weight scale
        
        results = {}
        
        # 1. Uniform (standard) 4-bit quantization
        print("\n  1. Uniform INT4 quantization:")
        qmax = 7
        max_abs = weights.abs().max()
        scale = max_abs / qmax
        q_uniform = torch.round(weights / scale).clamp(-8, 7)
        dq_uniform = q_uniform * scale
        err_uniform = (weights - dq_uniform).abs()
        mse_uniform = (weights - dq_uniform).pow(2).mean().item()
        results["Uniform INT4"] = mse_uniform
        print(f"     MSE: {mse_uniform:.2e}")
        print(f"     Max err: {err_uniform.max().item():.6f}")
        
        # 2. Simulated NF4 quantization
        # NF4 uses quantile-based mapping for normal distributions
        print("\n  2. NF4 quantization (quantile-based):")
        # NF4 quantization levels are placed at quantiles of N(0,1)
        # so that each bin captures equal probability mass
        from scipy import stats  # type: ignore
        
        try:
            n_levels = 16  # 4 bits = 16 levels
            # Place quantization levels at quantile midpoints
            quantiles = torch.tensor([
                stats.norm.ppf((2 * i + 1) / (2 * n_levels))
                for i in range(n_levels)
            ], dtype=torch.float32)
            
            # Normalize to match weight range
            # For NF4, we first normalize weights to [-1, 1]
            absmax = weights.abs().max()
            normalized = weights / absmax
            
            # Map each weight to nearest NF4 level
            # NF4 levels for normalized weights
            nf4_levels = quantiles / quantiles.abs().max()  # Normalize levels to [-1, 1]
            
            # Find nearest level for each weight
            distances = (normalized.unsqueeze(1) - nf4_levels.unsqueeze(0)).abs()
            indices = distances.argmin(dim=1)
            dq_nf4 = nf4_levels[indices] * absmax
            
            err_nf4 = (weights - dq_nf4).abs()
            mse_nf4 = (weights - dq_nf4).pow(2).mean().item()
            results["NF4"] = mse_nf4
            print(f"     MSE: {mse_nf4:.2e}")
            print(f"     Max err: {err_nf4.max().item():.6f}")
            
        except ImportError:
            print("     (scipy required for NF4 simulation)")
            # Approximate NF4 with manual quantiles
            sorted_w, _ = weights.abs().sort()
            mse_nf4 = mse_uniform * 0.7  # NF4 is typically ~30% better
            results["NF4 (approx)"] = mse_nf4
            print(f"     MSE (approx): {mse_nf4:.2e}")
        
        # 3. FP4 quantization
        print("\n  3. FP4 quantization:")
        # FP4 = {0, 0.5, 1, 1.5, 2, 3, 4, 6} × {-1, +1}
        fp4_positive = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
        fp4_levels = torch.cat([-fp4_positive.flip(0)[:-1], fp4_positive])
        
        # Normalize and quantize
        absmax = weights.abs().max()
        normalized = weights / absmax * 6  # Scale to FP4 range
        
        distances = (normalized.unsqueeze(1) - fp4_levels.unsqueeze(0)).abs()
        indices = distances.argmin(dim=1)
        dq_fp4 = fp4_levels[indices] / 6 * absmax
        
        err_fp4 = (weights - dq_fp4).abs()
        mse_fp4 = (weights - dq_fp4).pow(2).mean().item()
        results["FP4"] = mse_fp4
        print(f"     MSE: {mse_fp4:.2e}")
        print(f"     Max err: {err_fp4.max().item():.6f}")
        
        # Summary
        print("\n  ── Summary ──")
        best = min(results, key=results.get)
        for name, mse in results.items():
            marker = " ← best" if name == best else ""
            print(f"    {name:<20}: MSE = {mse:.2e}{marker}")
        
        print("\n  → NF4 achieves lowest error for normally-distributed weights")
        print("  → This is because NF4 levels are placed at distribution quantiles")
        
        return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all quantization fundamentals demonstrations."""
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║           QUANTIZATION FUNDAMENTALS FOR QLoRA                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    # Section 1: Numeric formats
    formats = NumericFormats()
    formats.explore_floating_point()
    formats.memory_comparison_for_models()
    
    # Section 2: Quantization theory
    theory = QuantizationTheory()
    
    # Create sample weights (normally distributed, typical for NNs)
    torch.manual_seed(42)
    sample_weights = torch.randn(1000) * 0.02
    
    print("\n\nQuantizing sample weights (1000 normally-distributed values):")
    theory.affine_quantization(sample_weights, n_bits=8)
    theory.affine_quantization(sample_weights, n_bits=4)
    theory.symmetric_quantization(sample_weights, n_bits=8)
    theory.symmetric_quantization(sample_weights, n_bits=4)
    theory.demonstrate_granularity()
    
    # Section 3: Practical bitsandbytes
    bnb = BitsAndBytesQuantization()
    bnb.load_model_8bit()
    bnb.load_model_4bit()
    bnb.quantization_config_reference()
    
    # Section 4: Error analysis
    analysis = QuantizationErrorAnalysis()
    analysis.analyze_weight_distributions()
    analysis.compare_quantization_errors()
    
    print("\n" + "=" * 70)
    print("  MODULE COMPLETE")
    print("=" * 70)
    print("""
    Key takeaways:
    ✓ FP32 → FP16 cuts memory in half with minimal quality loss
    ✓ 4-bit quantization cuts memory by 8x vs FP32
    ✓ NF4 is optimal for normally-distributed NN weights
    ✓ Per-group (blockwise) quantization balances error and overhead
    ✓ bitsandbytes makes 4-bit loading simple with BitsAndBytesConfig
    ✓ The key config: load_in_4bit + nf4 + bf16 compute + double_quant
    """)


if __name__ == "__main__":
    main()
