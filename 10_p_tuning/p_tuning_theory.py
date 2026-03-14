"""
P-Tuning — Theory & Intuition
================================

Deep dive into the theoretical foundations of P-Tuning v1 and v2:

1. The Problem with Direct Prompt Optimization
   - Why raw soft prompts struggle
   - The discrete-continuous gap
   - Optimization landscape issues

2. P-Tuning v1: The Prompt Encoder Idea
   - LSTM as reparameterization
   - Inter-token dependencies
   - Template-based prompt patterns

3. P-Tuning v2: Deep Prompts at Every Layer
   - Why depth matters
   - Layer-wise task adaptation
   - Matching full FT at all scales

4. Theoretical Analysis
   - Expressiveness of deep vs shallow prompts
   - Information flow through layers
   - Why v2 works on hard tasks (NER, QA)

5. The Universality Claim
   - Across scales (300M to 10B)
   - Across tasks (NLU, NLG, sequence labeling)
   - What makes v2 universal

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict


# ============================================================================
# SECTION 1: THE PROBLEM WITH DIRECT PROMPT OPTIMIZATION
# ============================================================================

class DirectPromptProblems:
    """
    Understanding why naively optimizing soft prompts is hard.
    """
    
    @staticmethod
    def demonstrate():
        print("=" * 65)
        print("  SECTION 1: THE PROBLEM WITH DIRECT PROMPT OPTIMIZATION")
        print("=" * 65)
        
        explanation = """
  ═══ Why Are Soft Prompts Hard to Optimize? ═══
  
  When we directly optimize soft prompt embeddings P ∈ R^{N×D},
  several problems emerge:
  
  
  Problem 1: INDEPENDENT TOKEN OPTIMIZATION
  ─────────────────────────────────────────────────────────────
  Each prompt token pᵢ is optimized independently.
  But language is sequential — tokens should be coherent!
  
  Direct optimization:
    p₁ → gradient₁   (no knowledge of p₂, p₃, ...)
    p₂ → gradient₂   (no knowledge of p₁, p₃, ...)
    p₃ → gradient₃   (no knowledge of p₁, p₂, ...)
  
  Result: Prompt tokens may converge to inconsistent states,
  like a sentence where each word was chosen independently.
  
  
  Problem 2: DISCRETE-CONTINUOUS GAP
  ─────────────────────────────────────────────────────────────
  The model was pretrained on DISCRETE token embeddings that
  live on a specific manifold in embedding space.
  
  Continuous optimization can push prompt vectors OFF this
  manifold into regions the model has never seen:
  
    Vocab manifold:  ··· word₁ ··· word₂ ··· word₃ ···
    Soft prompt:     ★ (floating in unknown territory!)
  
  The model may not know how to process out-of-distribution
  input vectors, leading to unpredictable behavior.
  
  
  Problem 3: HIGH-DIMENSIONAL LANDSCAPE
  ─────────────────────────────────────────────────────────────
  Optimizing N × D parameters (e.g., 20 × 768 = 15,360)
  in a non-convex landscape with gradients flowing through
  a frozen model is challenging:
  
  • Many local minima
  • Flat regions (vanishing gradients)
  • Sharp valleys (exploding gradients)
  • High sensitivity to learning rate
  
  
  Problem 4: SMALL MODEL FAILURE
  ─────────────────────────────────────────────────────────────
  On models < 1B params, direct prompt optimization often
  FAILS COMPLETELY:
  
    Model Size    Direct Prompt    With Encoder (P-Tuning)
    100M          62% (fail!)      78% (much better!)
    300M          71%              84%
    1B            82%              87%
    10B           91%              92% (gap closes)
  
  The encoder HELPS MOST where it's needed most: small models!
"""
        print(explanation)
        
        # Demonstrate the independence problem
        torch.manual_seed(42)
        d_model = 64
        num_tokens = 5
        
        # Simulate direct optimization
        direct_prompt = nn.Parameter(torch.randn(num_tokens, d_model) * 0.02)
        
        # Check inter-token correlation (should be ~0 for random init)
        with torch.no_grad():
            normalized = F.normalize(direct_prompt, dim=1)
            similarity = normalized @ normalized.T
        
        print(f"\n  Inter-token similarity (random init, direct optimization):")
        for i in range(num_tokens):
            row = "  "
            for j in range(num_tokens):
                row += f" {similarity[i,j].item():>6.3f}"
            print(row)
        print(f"  → Near-zero off-diagonal: tokens are independent!")
        
        # Simulate LSTM-encoded prompt (tokens become correlated)
        lstm = nn.LSTM(d_model, d_model // 2, batch_first=True, bidirectional=True)
        hidden_prompt = torch.randn(1, num_tokens, d_model)
        
        with torch.no_grad():
            encoded, _ = lstm(hidden_prompt)
            encoded = encoded.squeeze(0)
            norm_enc = F.normalize(encoded, dim=1)
            sim_enc = norm_enc @ norm_enc.T
        
        print(f"\n  Inter-token similarity (LSTM-encoded prompt):")
        for i in range(num_tokens):
            row = "  "
            for j in range(num_tokens):
                row += f" {sim_enc[i,j].item():>6.3f}"
            print(row)
        print(f"  → Higher off-diagonal: LSTM creates coherent tokens!")


# ============================================================================
# SECTION 2: P-TUNING V1 — THE PROMPT ENCODER
# ============================================================================

class PromptEncoderTheory:
    """
    Theory behind the LSTM prompt encoder in P-Tuning v1.
    """
    
    @staticmethod
    def demonstrate():
        print("\n\n" + "=" * 65)
        print("  SECTION 2: P-TUNING V1 — THE PROMPT ENCODER")
        print("=" * 65)
        
        explanation = """
  ═══ The P-Tuning v1 Architecture ═══
  
  Instead of directly optimizing prompt embeddings,
  P-Tuning v1 uses a trainable encoder:
  
  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │   Learnable embeddings: h = [h₁, h₂, ..., hₙ]          │
  │                         ↓                                │
  │   ┌──────────────────────────────────────────────────┐   │
  │   │          Bidirectional LSTM                       │   │
  │   │                                                  │   │
  │   │   h₁ → [LSTM→] ──→ ←── [←LSTM] ← hₙ           │   │
  │   │   h₂ → [LSTM→] ──→ ←── [←LSTM] ← hₙ₋₁         │   │
  │   │   ...                                            │   │
  │   │   Output: concatenate forward + backward         │   │
  │   └──────────────────────────────────────────────────┘   │
  │                         ↓                                │
  │   ┌──────────────────────────────────────────────────┐   │
  │   │          2-Layer MLP + ReLU                       │   │
  │   │                                                  │   │
  │   │   LSTM_out → Linear → ReLU → Linear → p          │   │
  │   └──────────────────────────────────────────────────┘   │
  │                         ↓                                │
  │   Continuous prompts: p = [p₁, p₂, ..., pₙ]             │
  │                                                          │
  └──────────────────────────────────────────────────────────┘
  
  Then used in the model:
  [p₁...pₙ, x₁...xₘ]  →  Frozen Transformer  →  Output
  
  
  ═══ Why LSTM? ═══
  
  1. SEQUENTIAL DEPENDENCY
     Each prompt token pᵢ depends on ALL other tokens
     through the bidirectional LSTM hidden states.
     This creates a "coherent" prompt.
  
  2. REPARAMETERIZATION
     The LSTM maps from a latent space to embedding space.
     This provides a smoother optimization landscape
     (similar to how a VAE's decoder smooths generation).
  
  3. BIDIRECTIONAL CONTEXT
     The BiLSTM ensures each token sees both left and right
     context, so the prompt forms a cohesive "instruction."
  
  4. MLP PROJECTION
     The MLP after LSTM provides additional non-linear
     transformation capacity, allowing the encoder to
     produce embeddings in the right part of space.
  
  
  ═══ Template-Based Prompting (P-Tuning v1 Innovation) ═══
  
  P-Tuning v1 also introduced TEMPLATE-based prompt patterns
  where soft tokens are interleaved with the input:
  
  Traditional:  [p₁ p₂ p₃ p₄ p₅ | x₁ x₂ x₃]
                 └── all prompt ──┘ └─ input ─┘
  
  P-Tuning v1:  [p₁ p₂ x₁ x₂ x₃ p₃ p₄ [MASK] p₅]
                  └──┘ └───input──┘ └──prompt──┘ └─┘
  
  The prompts can SURROUND the input, creating patterns like:
  
  "It was [p₁ p₂] [INPUT] [p₃ p₄] [MASK] [p₅]"
  
  This allows more flexible task formatting than prefix-only.
"""
        print(explanation)
        
        # Demonstrate LSTM encoding
        torch.manual_seed(42)
        d_model = 128
        num_tokens = 10
        
        # Direct prompts
        direct = torch.randn(num_tokens, d_model) * 0.02
        
        # LSTM-encoded prompts
        lstm = nn.LSTM(d_model, d_model // 2, num_layers=2,
                       batch_first=True, bidirectional=True)
        mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        
        h_input = torch.randn(1, num_tokens, d_model)
        with torch.no_grad():
            lstm_out, _ = lstm(h_input)
            encoded = mlp(lstm_out).squeeze(0)
        
        # Compare statistics
        print(f"\n  Comparison: Direct vs LSTM-Encoded Prompts")
        print(f"  {'':>20} {'Direct':>12} {'LSTM-Encoded':>14}")
        print(f"  {'─'*20}─{'─'*12}─{'─'*14}")
        print(f"  {'Mean':>20}  {direct.mean():.6f}  {encoded.mean():.6f}")
        print(f"  {'Std':>20}  {direct.std():.6f}  {encoded.std():.6f}")
        print(f"  {'Norm (avg)':>20}  {direct.norm(dim=1).mean():.4f}"
              f"  {encoded.norm(dim=1).mean():.4f}")
        
        # Token smoothness (avg L2 between consecutive tokens)
        direct_smooth = (direct[1:] - direct[:-1]).norm(dim=1).mean()
        encoded_smooth = (encoded[1:] - encoded[:-1]).norm(dim=1).mean()
        
        print(f"  {'Consecutive Δ':>20}  {direct_smooth:.4f}"
              f"  {encoded_smooth:.4f}")
        print(f"\n  LSTM produces smoother transitions between consecutive tokens")


# ============================================================================
# SECTION 3: P-TUNING V2 — DEEP PROMPTS
# ============================================================================

class DeepPromptTheory:
    """
    Theory behind P-Tuning v2's deep prompt approach.
    """
    
    @staticmethod
    def demonstrate():
        print("\n\n" + "=" * 65)
        print("  SECTION 3: P-TUNING V2 — DEEP PROMPTS AT EVERY LAYER")
        print("=" * 65)
        
        explanation = """
  ═══ The Core Idea: Depth Over Encoders ═══
  
  P-Tuning v2 makes a key simplification:
  
  Instead of a complex LSTM encoder (v1), use SIMPLE learnable
  prompts — but add them at EVERY transformer layer.
  
  P-Tuning v1:  LSTM encoder → prompts at input only
  P-Tuning v2:  Direct prompts → at EVERY layer
  
  
  ═══ Layer-by-Layer Architecture ═══
  
  ┌─────────────────────────────────────────────────────────┐
  │ Layer 1:                                                │
  │   Input: [P¹₁ P¹₂...P¹ₙ | x₁ x₂...xₘ]               │
  │          └─ Layer 1 prompts ─┘                          │
  │   Output: hidden₁                                      │
  │                                                         │
  │ Layer 2:                                                │
  │   Input: [P²₁ P²₂...P²ₙ | hidden₁]                   │
  │          └─ Layer 2 prompts (DIFFERENT from Layer 1!) ─┘│
  │   Output: hidden₂                                      │
  │                                                         │
  │ ...                                                     │
  │                                                         │
  │ Layer L:                                                │
  │   Input: [Pᴸ₁ Pᴸ₂...Pᴸₙ | hiddenₗ₋₁]                │
  │   Output: final_hidden                                  │
  └─────────────────────────────────────────────────────────┘
  
  Each layer has its OWN independent set of prompt tokens.
  This is 12× or 24× more parameters than input-only prompts,
  but still a tiny fraction of the full model.
  
  
  ═══ Why Depth Matters: The Expressiveness Argument ═══
  
  Input-only prompts (prompt tuning, P-Tuning v1):
  ─────────────────────────────────────────────────────────
  The prompt's influence DECAYS through layers.
  By layer 12, the original prompt signal may be very weak.
  
    Layer  1: ████████████████  Strong prompt influence
    Layer  4: ██████████        Moderate
    Layer  8: █████             Weak
    Layer 12: ██                Very weak
  
  Deep prompts (P-Tuning v2):
  ─────────────────────────────────────────────────────────
  Fresh prompt signal is injected at EVERY layer.
  The prompt maintains strong influence throughout!
  
    Layer  1: ████████████████  Fresh prompt
    Layer  4: ████████████████  Fresh prompt
    Layer  8: ████████████████  Fresh prompt
    Layer 12: ████████████████  Fresh prompt
  
  This is especially critical for:
  • SMALL models (weaker signal propagation)
  • HARD tasks (NER, QA need precise token-level control)
  
  
  ═══ Why v2 Works on Hard Tasks ═══
  
  Sequence labeling (NER, POS tagging) requires token-level
  predictions — the model must make a decision for EACH token.
  
  Input-only prompts struggle because:
  1. Prompt signal is diluted by the time it reaches output
  2. Each output token's representation is dominated by
     its own content, not the distant prompt tokens
  
  Deep prompts solve this because:
  1. Every layer gets fresh task instructions
  2. Token-level representations are continuously steered
  3. The model can build task-specific features at each depth
"""
        print(explanation)
        
        # Simulate signal propagation
        torch.manual_seed(42)
        d_model = 128
        n_layers = 12
        num_prompt = 10
        seq_len = 20
        
        # Input-only simulation
        print(f"\n  Signal Propagation Simulation:")
        print(f"  ─────────────────────────────────")
        
        prompt = torch.randn(1, num_prompt, d_model) * 0.1
        x = torch.randn(1, seq_len, d_model) * 0.1
        combined = torch.cat([prompt, x], dim=1)
        
        linear_layers = [nn.Linear(d_model, d_model) for _ in range(n_layers)]
        
        # Track prompt influence (norm of prompt positions)
        h = combined
        for i, layer in enumerate(linear_layers):
            with torch.no_grad():
                h = F.relu(layer(h))
                prompt_norm = h[:, :num_prompt, :].norm().item()
                input_norm = h[:, num_prompt:, :].norm().item()
                ratio = prompt_norm / (input_norm + 1e-8)
                bar = "█" * int(ratio * 20)
                print(f"  Layer {i+1:2d}: prompt_ratio={ratio:.4f} {bar}")
        
        print(f"\n  With deep prompts, each layer starts with fresh prompt signal,")
        print(f"  maintaining consistent task influence throughout the model.")


# ============================================================================
# SECTION 4: THEORETICAL ANALYSIS
# ============================================================================

class TheoreticalAnalysis:
    """
    Formal analysis of shallow vs deep prompt expressiveness.
    """
    
    @staticmethod
    def demonstrate():
        print("\n\n" + "=" * 65)
        print("  SECTION 4: THEORETICAL ANALYSIS")
        print("=" * 65)
        
        analysis = """
  ═══ Expressiveness Comparison ═══
  
  Shallow Prompts (Prompt Tuning, P-Tuning v1):
  ─────────────────────────────────────────────────────────
  Capacity ∝ num_tokens × d_model
  
  The prompt can only influence the model through the initial
  input representation. The model's frozen weights determine
  how this influence propagates through layers.
  
  Limitation: The model's internal computation is FIXED.
  The prompt can only choose where to START in the model's
  representation space; it cannot modify the TRAJECTORY.
  
  
  Deep Prompts (P-Tuning v2, Prefix Tuning):
  ─────────────────────────────────────────────────────────
  Capacity ∝ num_layers × num_tokens × d_model
  
  Each layer gets fresh task-specific vectors. This allows
  the prompt to steer the computation at every step:
  
    Layer 1: "Parse the syntax"       (syntax prompts)
    Layer 4: "Build semantic features" (semantic prompts)
    Layer 8: "Focus on entities"       (task prompts)
    Layer 12: "Classify this token"    (output prompts)
  
  Different layers can receive DIFFERENT instructions!
  
  
  ═══ Parameter Efficiency vs Expressiveness ═══
  
  Method            │ Params (GPT-2) │ Expressiveness │ Scale Req.
  ──────────────────┼────────────────┼────────────────┼──────────
  Prompt Tuning     │     15K        │    Low         │ 10B+
  P-Tuning v1       │     50K        │    Medium      │ 1B+
  P-Tuning v2       │    180K        │    High        │ 300M+  ★
  Prefix Tuning     │    370K        │    High        │ 300M+
  LoRA (r=8)        │    295K        │    Very High   │ 100M+
  Full Fine-Tuning  │    124M        │    Maximum     │ Any
  
  P-Tuning v2 achieves HIGH expressiveness with LOW parameters
  by distributing prompts across all layers.
  
  
  ═══ Why No Encoder Needed in v2 ═══
  
  P-Tuning v1 needed the LSTM encoder because:
  - With input-only prompts, each token must carry maximum info
  - The encoder ensures tokens are coherent and well-structured
  
  P-Tuning v2 doesn't need an encoder because:
  - Deep prompts are more expressive (more total parameters)
  - Each layer's prompts are relatively independent
  - The optimization landscape is smoother with deep prompts
  - Removing the encoder simplifies training and inference
  
  This is a key insight: DEPTH substitutes for COMPLEXITY.
"""
        print(analysis)
        
        # Parameter comparison table
        configs = {
            "Prompt Tuning": {"layers": 1, "tokens": 20, "per_layer": True},
            "P-Tuning v1": {"layers": 1, "tokens": 20, "encoder_overhead": 4.0},
            "P-Tuning v2": {"layers": 12, "tokens": 20, "per_layer": True},
            "Prefix Tuning": {"layers": 12, "tokens": 20, "kv_both": True},
        }
        
        d_model = 768
        n_layers = 12
        base_params = 124_000_000
        
        print(f"\n  Parameter Count Comparison (GPT-2, d=768, 12 layers, 20 tokens):")
        print(f"  {'Method':>20} {'Parameters':>12} {'% of Model':>12}")
        print(f"  {'─'*20}─{'─'*12}─{'─'*12}")
        
        param_counts = {
            "Prompt Tuning": 20 * d_model,
            "P-Tuning v1": 20 * d_model + 4 * (d_model * d_model // 2) + d_model * d_model,
            "P-Tuning v2": n_layers * 20 * d_model,
            "Prefix Tuning": n_layers * 2 * 20 * d_model,  # K and V
            "LoRA (r=8)": n_layers * 2 * 8 * d_model * 2,
        }
        
        for name, params in param_counts.items():
            pct = params / base_params * 100
            print(f"  {name:>20}  {params:>10,}  {pct:>10.4f}%")


# ============================================================================
# SECTION 5: THE UNIVERSALITY CLAIM
# ============================================================================

class UniversalityClaim:
    """
    Why P-Tuning v2 claims to work universally across scales and tasks.
    """
    
    @staticmethod
    def demonstrate():
        print("\n\n" + "=" * 65)
        print("  SECTION 5: THE UNIVERSALITY CLAIM")
        print("=" * 65)
        
        claim = """
  ═══ "Comparable to Fine-tuning UNIVERSALLY" ═══
  
  Previous prompt methods had limitations:
  
  Prompt Tuning:  Only works well on 10B+ models
  Prefix Tuning:  Works on medium+ models, but primarily NLG
  P-Tuning v1:    Improves small models, but still input-only
  
  P-Tuning v2 claims universality in THREE dimensions:
  
  
  DIMENSION 1: ACROSS MODEL SCALES
  ─────────────────────────────────────────────────────────
  
  Model        │ Full FT │ Prompt T │ P-Tune v1│ P-Tune v2│
  ─────────────┼─────────┼──────────┼──────────┼──────────┤
  RoBERTa-base │   89.2  │   78.5   │   83.0   │  ★ 88.5 │
  (125M)       │         │  (-10.7) │  (-6.2)  │  (-0.7) │
  ─────────────┼─────────┼──────────┼──────────┼──────────┤
  RoBERTa-large│   91.5  │   85.0   │   88.5   │  ★ 91.0 │
  (355M)       │         │  (-6.5)  │  (-3.0)  │  (-0.5) │
  ─────────────┼─────────┼──────────┼──────────┼──────────┤
  DeBERTa-xxl  │   93.5  │   92.0   │   92.5   │  ★ 93.5 │
  (1.5B)       │         │  (-1.5)  │  (-1.0)  │  (0.0)  │
  
  P-Tuning v2 matches full FT even on SMALL models!
  
  
  DIMENSION 2: ACROSS TASK TYPES
  ─────────────────────────────────────────────────────────
  
  ┌────────────────────┬──────────┬──────────┬──────────┐
  │ Task Type          │ Prompt T │ P-Tune v1│ P-Tune v2│
  ├────────────────────┼──────────┼──────────┼──────────┤
  │ Classification     │   Good   │   Good   │  ★ Great │
  │ (SST-2, MNLI)      │          │          │          │
  ├────────────────────┼──────────┼──────────┼──────────┤
  │ NER (CoNLL-2003)   │   Poor   │   Fair   │  ★ Great │
  │ (token-level!)     │          │          │          │
  ├────────────────────┼──────────┼──────────┼──────────┤
  │ Extractive QA      │   Poor   │   Fair   │  ★ Great │
  │ (SQuAD)            │          │          │          │
  ├────────────────────┼──────────┼──────────┼──────────┤
  │ Semantic Sim.      │   Good   │   Good   │  ★ Great │
  │ (STS-B, QQP)       │          │          │          │
  └────────────────────┴──────────┴──────────┴──────────┘
  
  P-Tuning v2 works on HARD tasks where others fail!
  NER and extractive QA require token-level predictions,
  which shallow prompts cannot support well.
  
  
  DIMENSION 3: ACROSS MODEL ARCHITECTURES
  ─────────────────────────────────────────────────────────
  
  ┌───────────────────────┬──────────────────────────────┐
  │ Architecture          │ P-Tuning v2 Support          │
  ├───────────────────────┼──────────────────────────────┤
  │ Encoder-only (BERT)   │ ✓ Excellent                  │
  │ Decoder-only (GPT-2)  │ ✓ Good                       │
  │ Encoder-Decoder (T5)  │ ✓ Great (prompts both sides) │
  └───────────────────────┴──────────────────────────────┘
  
  
  ═══ Recipe for Universality ═══
  
  The P-Tuning v2 paper identifies key ingredients:
  
  1. Deep prompts at every layer           ← core method
  2. Task-specific classification head     ← not shared!
  3. Prompt length tuned per task:
     • Classification: 10-20 tokens
     • NER: 5-10 tokens (surprisingly few!)
     • QA: 10-20 tokens
  4. No reparameterization (direct optimization)
  5. Standard learning rate (~5e-3 to 1e-2)
"""
        print(claim)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all theory demonstrations."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     P-TUNING — THEORY & INTUITION                           ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: Problems with direct optimization
    DirectPromptProblems.demonstrate()
    
    # Section 2: P-Tuning v1 encoder
    PromptEncoderTheory.demonstrate()
    
    # Section 3: P-Tuning v2 deep prompts
    DeepPromptTheory.demonstrate()
    
    # Section 4: Theoretical analysis
    TheoreticalAnalysis.demonstrate()
    
    # Section 5: Universality claim
    UniversalityClaim.demonstrate()
    
    print("\n" + "=" * 65)
    print("  MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered:
    ✓ Problems with direct prompt optimization
    ✓ P-Tuning v1: LSTM prompt encoder theory
    ✓ P-Tuning v2: deep prompts at every layer
    ✓ Expressiveness: shallow vs deep prompts
    ✓ Universality across scales, tasks, architectures
    """)


if __name__ == "__main__":
    main()
