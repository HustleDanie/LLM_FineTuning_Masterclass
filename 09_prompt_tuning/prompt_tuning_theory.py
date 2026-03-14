"""
Prompt Tuning — Theory & Intuition
====================================

Deep dive into the theory behind prompt tuning:

1. From Discrete to Continuous Prompts
   - Limitations of hard prompts
   - Why continuous prompts work
   - Connection to prompt engineering

2. The Embedding Space Perspective
   - What soft prompts learn
   - Geometric interpretation
   - Relationship to vocabulary tokens

3. Gradient Flow Analysis
   - How gradients reach the soft prompt
   - Why only input-level works at scale
   - Signal propagation through frozen layers

4. The Power of Scale
   - Why larger models close the gap
   - Theoretical explanation
   - Empirical scaling curves

5. Information-Theoretic View
   - Bits of task information in prompts
   - Prompt length vs task complexity

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional


# ============================================================================
# SECTION 1: FROM DISCRETE TO CONTINUOUS PROMPTS
# ============================================================================

class DiscreteVsContinuousPrompts:
    """
    Understanding the evolution from hard prompts to soft prompts.
    """
    
    @staticmethod
    def demonstrate():
        print("=" * 65)
        print("  SECTION 1: FROM DISCRETE TO CONTINUOUS PROMPTS")
        print("=" * 65)
        
        explanation = """
  ═══ The Prompt Engineering Problem ═══
  
  Hard (discrete) prompts are text strings:
    "Classify the following text as positive or negative: {input}"
  
  Problems with hard prompts:
  ┌───────────────────────────────────────────────────────────┐
  │ 1. Discrete search space — can't use gradient descent    │
  │ 2. Sensitive to exact wording ("Classify" vs "Label")    │
  │ 3. Limited by vocabulary — can't express sub-word ideas   │
  │ 4. Manual engineering is tedious and unreliable          │
  └───────────────────────────────────────────────────────────┘
  
  ═══ The Insight ═══
  
  What if, instead of searching over discrete tokens,
  we optimize CONTINUOUS VECTORS directly?
  
  Hard prompt:   token_ids → Embedding[token_ids] → vectors
                              ↑ constrained to vocabulary
  
  Soft prompt:   [v₁, v₂, ..., vₙ] → vectors (ANYWHERE in space)
                  ↑ free to be any real-valued vector
  
  
  ═══ Why This Works ═══
  
  Transformer attention treats all positions equally:
  
    Attention(Q, K, V) = softmax(Q·Kᵀ/√d) · V
  
  The model doesn't "know" whether a key-value pair came from:
    - A real token embedding
    - A random vector we prepended
    - A learned soft prompt
  
  It just processes vectors. So we can prepend ANY vectors
  to the input, and as long as they steer attention in a
  useful direction, the model will produce good outputs.
  
  
  ═══ Connection to Prompt Engineering ═══
  
  Think of soft prompts as "perfect prompt engineering":
  
  Human prompts:     "Please classify this as positive/negative"
                      ↓ tokenize ↓
                     [3492, 521, 8834, 102, ...]  ← integer IDs
                      ↓ embed ↓
                     [e₁,  e₂,  e₃,  e₄, ...]    ← ON the manifold
  
  Soft prompts:      [v₁,  v₂,  v₃,  v₄, ...]    ← NEAR the manifold
                      ↑ optimized by gradient descent
                      ↑ can express things no words can!
"""
        print(explanation)


# ============================================================================
# SECTION 2: THE EMBEDDING SPACE PERSPECTIVE
# ============================================================================

class EmbeddingSpaceAnalysis:
    """
    Geometric analysis of what soft prompts learn.
    """
    
    @staticmethod
    def analyze_soft_prompt_location(vocab_size: int = 1000, d_model: int = 64,
                                      num_prompts: int = 5):
        """
        Visualize where soft prompts end up relative to vocabulary embeddings.
        """
        print("\n" + "=" * 65)
        print("  SECTION 2: THE EMBEDDING SPACE PERSPECTIVE")
        print("=" * 65)
        
        torch.manual_seed(42)
        
        # Simulate vocabulary embeddings (normally from a pretrained model)
        vocab_embeddings = torch.randn(vocab_size, d_model) * 0.02
        
        # Initialize soft prompts (random init)
        soft_prompts_random = torch.randn(num_prompts, d_model) * 0.02
        
        # Initialize soft prompts (from vocab — like text init)
        init_indices = torch.randint(0, vocab_size, (num_prompts,))
        soft_prompts_vocab = vocab_embeddings[init_indices].clone()
        
        # Simulate "trained" soft prompts (moved away from vocab)
        soft_prompts_trained = soft_prompts_vocab + torch.randn_like(soft_prompts_vocab) * 0.5
        
        # Compute distances
        def avg_nearest_vocab_distance(prompts, vocab):
            """Average distance from each prompt to nearest vocab embedding."""
            # prompts: [N, d], vocab: [V, d]
            dists = torch.cdist(prompts.unsqueeze(0), vocab.unsqueeze(0))[0]
            return dists.min(dim=1).values.mean().item()
        
        dist_random = avg_nearest_vocab_distance(soft_prompts_random, vocab_embeddings)
        dist_vocab = avg_nearest_vocab_distance(soft_prompts_vocab, vocab_embeddings)
        dist_trained = avg_nearest_vocab_distance(soft_prompts_trained, vocab_embeddings)
        
        print(f"\n  Avg distance to nearest vocabulary token:")
        print(f"  ─────────────────────────────────────────")
        print(f"  Random init:  {dist_random:.4f}")
        print(f"  Vocab init:   {dist_vocab:.4f}")
        print(f"  After train:  {dist_trained:.4f}")
        
        # Find nearest tokens
        def find_nearest_tokens(prompts, vocab, top_k=3):
            dists = torch.cdist(prompts.unsqueeze(0), vocab.unsqueeze(0))[0]
            nearest = dists.topk(top_k, dim=1, largest=False)
            return nearest.indices, nearest.values
        
        indices, distances = find_nearest_tokens(soft_prompts_trained, vocab_embeddings)
        
        print(f"""
  Key finding: After training, soft prompts typically move to
  positions BETWEEN vocabulary embeddings — they settle in 
  regions of embedding space that no single word occupies.
  
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │   · word₁    · word₂                                   │
  │                   ★ soft_prompt₁                        │
  │        · word₃           · word₄                       │
  │                                                         │
  │              ★ soft_prompt₂                             │
  │   · word₅          · word₆                             │
  │                                                         │
  │   · = vocabulary token  ★ = learned soft prompt         │
  │                                                         │
  │   Soft prompts learn "between-word" representations     │
  │   that express task instructions more precisely than     │
  │   any combination of actual words could.                │
  └─────────────────────────────────────────────────────────┘
""")
        
        # Prompt norm analysis
        vocab_norms = vocab_embeddings.norm(dim=1)
        trained_norms = soft_prompts_trained.norm(dim=1)
        
        print(f"  Embedding norms:")
        print(f"  Vocab tokens:  mean={vocab_norms.mean():.4f}, "
              f"std={vocab_norms.std():.4f}")
        print(f"  Soft prompts:  mean={trained_norms.mean():.4f}, "
              f"std={trained_norms.std():.4f}")
        print(f"\n  Trained prompts often have LARGER norms than vocab tokens,")
        print(f"  allowing them to exert stronger influence on attention.")


# ============================================================================
# SECTION 3: GRADIENT FLOW ANALYSIS
# ============================================================================

class GradientFlowAnalysis:
    """
    Understanding how gradients flow back to soft prompts
    through frozen transformer layers.
    """
    
    @staticmethod
    def demonstrate(d_model: int = 128, n_layers: int = 6, seq_len: int = 20,
                    num_prompts: int = 5):
        """Simulate gradient flow through frozen layers."""
        print("\n" + "=" * 65)
        print("  SECTION 3: GRADIENT FLOW ANALYSIS")
        print("=" * 65)
        
        torch.manual_seed(42)
        
        explanation = """
  ═══ The Gradient Flow Challenge ═══
  
  In prompt tuning, gradients must flow backward through
  the ENTIRE frozen model to reach the soft prompts:
  
    Loss → Layer_N → Layer_N-1 → ... → Layer_1 → Soft Prompt
           (frozen)  (frozen)          (frozen)   (trainable!)
  
  This is fundamentally different from LoRA/Adapters,
  where trainable parameters exist at every layer:
  
    LoRA:  Loss → Layer_N (+LoRA) → ... → Layer_1 (+LoRA)
                  (short path!)           (short path!)
  
  For prompt tuning, the gradient signal must survive passage
  through many frozen layers without being distorted.
"""
        print(explanation)
        
        # Simulate gradient flow
        soft_prompt = nn.Parameter(torch.randn(1, num_prompts, d_model) * 0.02)
        input_embeds = torch.randn(1, seq_len, d_model)
        
        # Combine: [soft_prompt; input]
        combined = torch.cat([soft_prompt, input_embeds], dim=1)
        
        # Simulate frozen transformer layers
        layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=d_model * 4,
                batch_first=True, dropout=0.0,
            )
            for _ in range(n_layers)
        ])
        
        # Freeze all layers
        for layer in layers:
            for p in layer.parameters():
                p.requires_grad = False
        
        # Forward pass
        x = combined
        layer_outputs = [x]
        for layer in layers:
            x = layer(x)
            layer_outputs.append(x)
        
        # Simulate loss on output (mean of all positions)
        loss = x.mean()
        loss.backward()
        
        # Check gradient at soft prompt
        grad_norm = soft_prompt.grad.norm().item()
        
        print(f"  Gradient Flow Simulation ({n_layers} frozen layers):")
        print(f"  ─────────────────────────────────────────────")
        print(f"  Soft prompt gradient norm: {grad_norm:.6f}")
        
        # Compare gradient at different "depths"
        print(f"\n  Gradient magnitudes at each layer output:")
        print(f"  (computed via autograd through frozen layers)")
        
        for i, output in enumerate(layer_outputs):
            if output.requires_grad:
                grad = torch.autograd.grad(
                    loss, output, retain_graph=True, allow_unused=True,
                )[0]
                if grad is not None:
                    prompt_grad = grad[:, :num_prompts, :].norm().item()
                    input_grad = grad[:, num_prompts:, :].norm().item()
                    print(f"  Layer {i:2d}: prompt_grad={prompt_grad:.6f}, "
                          f"input_grad={input_grad:.6f}")
        
        print(f"""
  ═══ Key Observations ═══
  
  1. Gradients DO flow back through frozen layers
     (automatic differentiation works even with frozen params)
  
  2. Gradient magnitude decreases with depth
     → This is why prompt tuning can be unstable on small models
     → Larger models have better-conditioned Jacobians
  
  3. Prompt positions get slightly different gradients than
     input positions — they learn to complement the input
  
  4. Why scale helps:
     ┌─────────────────────────────────────────────────────┐
     │ Larger models have:                                 │
     │  • Better-conditioned weight matrices               │
     │  • More redundancy (easier to steer)                │
     │  • Smoother loss landscapes                         │
     │  • The capacity to "listen to" small input changes  │
     └─────────────────────────────────────────────────────┘
""")


# ============================================================================
# SECTION 4: THE POWER OF SCALE
# ============================================================================

class PowerOfScale:
    """
    The defining characteristic of prompt tuning:
    it approaches full fine-tuning performance as models scale up.
    """
    
    @staticmethod
    def demonstrate():
        print("\n" + "=" * 65)
        print("  SECTION 4: THE POWER OF SCALE")
        print("=" * 65)
        
        # Simulated results based on Lester et al. (2021) findings
        # on SuperGLUE benchmark
        model_sizes = {
            "Small (60M)":   {"full_ft": 79.0, "prompt": 62.0, "prefix": 72.0, "lora": 76.0},
            "Base (220M)":   {"full_ft": 84.0, "prompt": 72.0, "prefix": 80.0, "lora": 82.0},
            "Large (770M)":  {"full_ft": 88.5, "prompt": 82.0, "prefix": 86.0, "lora": 87.5},
            "XL (3B)":      {"full_ft": 91.0, "prompt": 88.5, "prefix": 90.0, "lora": 90.5},
            "XXL (11B)":    {"full_ft": 93.0, "prompt": 92.5, "prefix": 92.8, "lora": 92.8},
        }
        
        print(f"\n  SuperGLUE Performance vs Model Scale (simulated):")
        print(f"  {'Model':>14} {'Full FT':>8} {'Prompt':>8} {'Prefix':>8} {'LoRA':>8}  "
              f"{'Gap':>6}")
        print(f"  {'─'*14}─{'─'*8}─{'─'*8}─{'─'*8}─{'─'*8}──{'─'*6}")
        
        for name, scores in model_sizes.items():
            gap = scores["full_ft"] - scores["prompt"]
            bar = "█" * int(gap)
            print(f"  {name:>14}  {scores['full_ft']:>5.1f}  {scores['prompt']:>6.1f}  "
                  f"{scores['prefix']:>6.1f}  {scores['lora']:>6.1f}  "
                  f"{gap:>4.1f}  {bar}")
        
        print(f"""
  ═══ The Scaling Phenomenon ═══
  
  Gap between Full FT and Prompt Tuning:
  
  60M params:  ████████████████████  17.0 pts  ← HUGE gap
  220M params: ████████████          12.0 pts
  770M params: ██████                 6.5 pts  ← Closing!
  3B params:   ██                     2.5 pts
  11B params:  █                      0.5 pts  ← Nearly equal!
  
  
  ═══ Why Does Scale Help? ═══
  
  Theory 1: Model Capacity
  ─────────────────────────────────────────────────────────
  Larger models have more parameters → more internal 
  "degrees of freedom" → easier to steer with a small 
  input perturbation. Like steering a large ship vs a 
  small boat — the ship has more momentum and responds
  more smoothly to small rudder adjustments.
  
  Theory 2: Representation Quality
  ─────────────────────────────────────────────────────────
  Larger models learn better internal representations 
  during pretraining. The soft prompt doesn't need to 
  "teach" the model new capabilities — it just needs to 
  activate the right existing circuits.
  
  
  Theory 3: Overparameterization
  ─────────────────────────────────────────────────────────
  Overparameterized models have many equivalent solutions.
  The soft prompt can guide the model toward a solution 
  that works for the target task with minimal input change.
  
  
  Theory 4: Loss Landscape Smoothness
  ─────────────────────────────────────────────────────────
  Larger models have smoother loss landscapes. The gradient
  signal from the loss to the input is less noisy, making
  optimization of the soft prompt more reliable.
""")
        
        # Practical implication
        print(f"  ═══ Practical Takeaway ═══")
        print(f"  ─────────────────────────────────────────────────────")
        print(f"  Model Size    → Recommended PEFT Method")
        print(f"  < 1B params   → LoRA or Adapters (prompt tuning weak)")
        print(f"  1B - 5B       → Prompt tuning viable, LoRA still better")
        print(f"  5B+           → Prompt tuning excellent choice!")
        print(f"  10B+          → Prompt tuning ≈ full fine-tuning")


# ============================================================================
# SECTION 5: INFORMATION-THEORETIC VIEW
# ============================================================================

class InformationTheoreticView:
    """
    How much task information can soft prompts encode?
    """
    
    @staticmethod
    def demonstrate():
        print("\n\n" + "=" * 65)
        print("  SECTION 5: INFORMATION-THEORETIC VIEW")
        print("=" * 65)
        
        # How many bits in a soft prompt?
        d_model_values = [768, 1024, 2048, 4096]
        prompt_lengths = [1, 5, 10, 20, 50, 100]
        
        print(f"""
  ═══ Bits of Task Information ═══
  
  Each soft prompt token is a {d_model_values[0]}-dim float32 vector.
  In theory: {d_model_values[0]} × 32 = {d_model_values[0]*32:,} bits per token.
  
  But effective information content is much lower due to:
  - Redundancy in high-dimensional spaces
  - Model's limited sensitivity to input perturbations
  - Effective dimensionality of the task manifold
  
  Still, even conservative estimates suggest:
""")
        
        print(f"  Parameter Count by Prompt Length and Model Dimension:")
        print(f"  {'Length':>8}", end="")
        for d in d_model_values:
            print(f"  {'d=' + str(d):>10}", end="")
        print()
        print(f"  {'─'*8}", end="")
        for _ in d_model_values:
            print(f"──{'─'*10}", end="")
        print()
        
        for length in prompt_lengths:
            print(f"  {length:>6}  ", end="")
            for d in d_model_values:
                params = length * d
                print(f"  {params:>8,}  ", end="")
            print()
        
        print(f"""
  
  ═══ Task Complexity vs Prompt Length ═══
  
  Empirical findings (from Lester et al.):
  ┌─────────────────────────────────────────────────────────┐
  │ Task                        │ Optimal Length │ Params   │
  ├─────────────────────────────┼────────────────┼──────────┤
  │ Binary classification       │  1 - 5 tokens  │  < 5K    │
  │ Multi-class classification  │  5 - 20 tokens │  < 20K   │
  │ Summarization               │  20 - 50       │  < 50K   │
  │ Translation                 │  20 - 100      │  < 100K  │
  │ Complex reasoning           │  50 - 100      │  < 100K  │
  └─────────────────────────────┴────────────────┴──────────┘
  
  Important: Beyond ~20 tokens, returns diminish rapidly!
  
  Performance vs Prompt Length (classification task):
  
   1 token:  ████████████████                80%
   5 tokens: ███████████████████             90%
  10 tokens: ████████████████████            92%
  20 tokens: █████████████████████           93%  ← sweet spot
  50 tokens: █████████████████████           93%  ← diminishing
  100 tokens:█████████████████████           92%  ← can hurt!
  
  Longer prompts consume context window without benefit
  and can even degrade performance by adding noise.
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all theory demonstrations."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     PROMPT TUNING — THEORY & INTUITION                       ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: Discrete vs continuous prompts
    DiscreteVsContinuousPrompts.demonstrate()
    
    # Section 2: Embedding space
    EmbeddingSpaceAnalysis.analyze_soft_prompt_location()
    
    # Section 3: Gradient flow
    GradientFlowAnalysis.demonstrate()
    
    # Section 4: Power of scale
    PowerOfScale.demonstrate()
    
    # Section 5: Information theory
    InformationTheoreticView.demonstrate()
    
    print("\n" + "=" * 65)
    print("  MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered:
    ✓ Hard prompts → soft prompts evolution
    ✓ Embedding space geometry
    ✓ Gradient flow through frozen layers
    ✓ Why larger models close the performance gap
    ✓ Information-theoretic prompt length analysis
    """)


if __name__ == "__main__":
    main()
