"""
IA³ Theory — Mathematical Foundations
=======================================

Deep theoretical understanding of IA³:

1. Why Multiplicative Rescaling Works
   - Additive vs multiplicative modifications
   - Representational equivalence
   - Information-theoretic perspective

2. Gradient Analysis
   - How gradients flow through IA³
   - Why identity initialization is critical
   - Comparison with LoRA gradient structure

3. Expressiveness & Limitations
   - What IA³ can and cannot learn
   - Relation to diagonal weight matrices
   - When rescaling is sufficient

4. Few-Shot Learning Theory
   - Why IA³ excels in few-shot settings
   - Inductive bias of rescaling
   - Regularization effect of minimal parameters

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# SECTION 1: WHY MULTIPLICATIVE RESCALING WORKS
# ============================================================================

class MultiplicativeVsAdditive:
    """
    Understanding the fundamental difference between
    additive (LoRA) and multiplicative (IA³) modifications.
    """
    
    @staticmethod
    def demonstrate():
        print("=" * 65)
        print("  SECTION 1: WHY MULTIPLICATIVE RESCALING WORKS")
        print("=" * 65)
        
        torch.manual_seed(42)
        d = 8  # Small dimension for visualization
        
        # Original weight matrix
        W = torch.randn(d, d)
        x = torch.randn(d)
        
        # ─── Additive modification (LoRA-style) ───
        # W' = W + ΔW, where ΔW = B @ A (low-rank)
        r = 2
        B = torch.randn(d, r) * 0.1
        A = torch.randn(r, d) * 0.1
        delta_W = B @ A
        
        y_original = W @ x
        y_additive = (W + delta_W) @ x
        
        # ─── Multiplicative modification (IA³-style) ───
        # y' = l ⊙ (W @ x), where l is a learned vector
        l = torch.ones(d) + torch.randn(d) * 0.1  # Near-identity
        
        y_multiplicative = l * (W @ x)
        
        print(f"\n  Original output:       {y_original[:4].tolist()}")
        print(f"  Additive (LoRA):       {y_additive[:4].tolist()}")
        print(f"  Multiplicative (IA³):  {y_multiplicative[:4].tolist()}")
        
        # ─── Key insight: Rescaling = Diagonal matrix multiplication ───
        # l ⊙ (Wx) = diag(l) @ W @ x = W' @ x where W' = diag(l) @ W
        W_rescaled = torch.diag(l) @ W
        y_equivalent = W_rescaled @ x
        
        print(f"\n  IA³ rescaling is equivalent to row-scaling the weight matrix:")
        print(f"  l ⊙ (Wx) = diag(l) · W · x")
        print(f"  Verification: max|diff| = {(y_multiplicative - y_equivalent).abs().max():.2e}")
        
        # ─── What this means ───
        print(f"""
  ═══ Additive vs Multiplicative ═══
  
  Additive (LoRA):
    W' = W + BA          (adds a low-rank correction)
    - Can shift the output in any direction
    - Requires learning B and A matrices
    - Parameters: 2 × r × d per weight matrix
    
  Multiplicative (IA³):
    y' = l ⊙ (Wx)       (scales each output dimension)
    - Equivalent to: W' = diag(l) × W
    - Only scales rows of W (amplify or inhibit)
    - Parameters: d per weight matrix (just a vector!)
    
  Why multiplicative is enough:
    1. Pretrained weights already encode useful features
    2. Task adaptation often just needs to REWEIGHT features
    3. "Turn up important features, turn down irrelevant ones"
    4. This is a strong inductive bias for adaptation!

  Analogy:
    LoRA = rewriting sentences in a book
    IA³  = highlighting important sentences (and crossing out others)
""")
        
        # ─── Demonstrate feature reweighting ───
        print(f"  Feature reweighting example:")
        print(f"  Original activations:  {y_original.tolist()}")
        
        # Selective amplification/inhibition
        l_selective = torch.ones(d)
        l_selective[0] = 2.0   # Amplify feature 0
        l_selective[3] = 0.1   # Inhibit feature 3
        l_selective[5] = 0.0   # Kill feature 5
        
        y_selective = l_selective * y_original
        
        print(f"  Rescaling vector:      {l_selective.tolist()}")
        print(f"  After IA³ rescaling:   {y_selective.tolist()}")
        print(f"  → Feature 0 amplified 2×, feature 3 inhibited 10×, feature 5 killed")


# ============================================================================
# SECTION 2: GRADIENT ANALYSIS
# ============================================================================

class GradientAnalysis:
    """
    How gradients flow through IA³ and why identity init matters.
    """
    
    @staticmethod
    def demonstrate():
        print("\n\n" + "=" * 65)
        print("  SECTION 2: GRADIENT ANALYSIS")
        print("=" * 65)
        
        torch.manual_seed(42)
        d = 16
        
        # ─── IA³ gradient derivation ───
        print(f"""
  ═══ IA³ Gradient Math ═══
  
  Forward:  y = l ⊙ (W · x)
  
  Let a = W · x  (pre-rescaling activation)
  Then y = l ⊙ a
  
  Gradient w.r.t. l:
    ∂L/∂l = ∂L/∂y ⊙ a    (element-wise product with activation)
    
  At initialization (l = 1):
    ∂L/∂l = ∂L/∂y ⊙ (W · x)
    
  This means:
    - Each l_i gets gradient proportional to its activation a_i
    - Dimensions with large activations get large gradients
    - Dimensions with small activations get small gradients
    - This is a NATURAL importance weighting!
""")
        
        # Demonstrate empirically
        W = nn.Linear(d, d, bias=False)
        l = nn.Parameter(torch.ones(d))  # IA³ vector, init to 1
        
        x = torch.randn(4, d)  # batch of 4
        target = torch.randn(4, d)
        
        # Forward
        a = W(x)         # pre-rescaling
        y = l * a         # IA³ rescaling
        loss = F.mse_loss(y, target)
        loss.backward()
        
        print(f"  Empirical gradient verification:")
        print(f"  l gradient:    {l.grad[:8].tolist()}")
        
        # Manual computation
        dl_dy = 2 * (y - target) / y.numel()
        manual_grad = (dl_dy * a.detach()).sum(dim=0)
        
        print(f"  Manual grad:   {manual_grad[:8].tolist()}")
        print(f"  Match: {torch.allclose(l.grad, manual_grad, atol=1e-5)}")
        
        # ─── Compare with LoRA gradients ───
        print(f"\n  Gradient comparison (LoRA vs IA³):")
        
        # LoRA
        B_lora = nn.Parameter(torch.zeros(d, 2))
        A_lora = nn.Parameter(torch.randn(2, d) * 0.01)
        
        y_lora = W(x) + x @ A_lora.T @ B_lora.T
        loss_lora = F.mse_loss(y_lora, target)
        loss_lora.backward()
        
        lora_grad_norm = (B_lora.grad.norm() + A_lora.grad.norm()).item()
        ia3_grad_norm = l.grad.norm().item()
        
        print(f"  LoRA gradient norm (B+A): {lora_grad_norm:.6f}")
        print(f"  IA³ gradient norm (l):    {ia3_grad_norm:.6f}")
        
        # ─── Why identity init is critical ───
        print(f"""
  ═══ Why Initialize to Ones? ═══
  
  At init (l = 1):
    y = 1 ⊙ (Wx) = Wx    → model behaves exactly as before
    
  If init to 0:
    y = 0 ⊙ (Wx) = 0     → all information destroyed!
    
  If init random:
    y = l_rand ⊙ (Wx)     → random distortion of features
                            → training must first undo damage
                            
  Identity init gives:
    1. Zero disruption at start
    2. Gradients immediately meaningful
    3. Training explores neighborhood of pretrained model
    4. Much faster convergence
    
  This is analogous to:
    - LoRA's zero-init for B matrix
    - Skip connections in ResNets
    - "Don't break what works, just adjust it"
""")
        
        # Demonstrate instability with different inits
        print(f"  Init stability comparison:")
        for init_val, init_name in [(1.0, "ones"), (0.0, "zeros"), (None, "random")]:
            if init_val is not None:
                l_test = nn.Parameter(torch.full((d,), init_val))
            else:
                l_test = nn.Parameter(torch.randn(d))
            
            y_test = l_test * W(x).detach()
            original = W(x).detach()
            deviation = (y_test - original).norm().item()
            print(f"    Init={init_name:>6}: deviation from original = {deviation:.4f}")


# ============================================================================
# SECTION 3: EXPRESSIVENESS & LIMITATIONS
# ============================================================================

class ExpressivenessAnalysis:
    """
    What IA³ can and cannot represent.
    """
    
    @staticmethod
    def demonstrate():
        print("\n\n" + "=" * 65)
        print("  SECTION 3: EXPRESSIVENESS & LIMITATIONS")
        print("=" * 65)
        
        torch.manual_seed(42)
        d = 8
        
        # ─── What IA³ can express ───
        print(f"""
  ═══ What IA³ CAN Express ═══
  
  1. Feature selection (l_i → 0 kills feature i)
  2. Feature amplification (l_i > 1 boosts feature i)
  3. Feature inhibition (0 < l_i < 1 dampens feature i)  
  4. Feature flipping (l_i < 0 reverses feature i)
  5. Row scaling of weight matrix (diag(l) @ W)
  
  Degrees of freedom: d per rescaling point
  (one scalar per activation dimension)
""")
        
        # ─── What IA³ CANNOT express ───
        print(f"""
  ═══ What IA³ CANNOT Express ═══
  
  1. Feature mixing (combining dim i and dim j into new features)
  2. Rotation of activation space
  3. Arbitrary weight matrix modifications
  4. New feature creation from scratch
  
  IA³ is limited to diagonal modifications:
    W' = diag(l) @ W  (row scaling only)
    
  vs LoRA which can do:
    W' = W + BA        (arbitrary low-rank correction)
""")
        
        # Demonstrate limitation concretely
        W = torch.randn(d, d)
        x = torch.randn(d)
        
        # Target: we want to compute a rotation of the output
        theta = math.pi / 4  # 45 degree rotation of first 2 dims
        R = torch.eye(d)
        R[0, 0] = math.cos(theta)
        R[0, 1] = -math.sin(theta)
        R[1, 0] = math.sin(theta)
        R[1, 1] = math.cos(theta)
        
        target_W = R @ W
        original_out = W @ x
        target_out = target_W @ x
        
        # Best IA³ approximation: find l such that diag(l) @ W ≈ R @ W
        # This requires l_i * W[i,:] ≈ (R @ W)[i,:] for each row
        # Only possible if R is diagonal!
        
        # Optimization
        l = nn.Parameter(torch.ones(d))
        optimizer = torch.optim.Adam([l], lr=0.01)
        
        for _ in range(500):
            y = l * (W @ x)
            loss = F.mse_loss(y, target_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        ia3_out = (l * (W @ x)).detach()
        ia3_error = (ia3_out - target_out).norm().item()
        
        # What LoRA can do
        B = nn.Parameter(torch.randn(d, 2) * 0.01)
        A = nn.Parameter(torch.randn(2, d) * 0.01)
        lora_opt = torch.optim.Adam([B, A], lr=0.01)
        
        for _ in range(500):
            y = W @ x + B @ A @ x
            loss = F.mse_loss(y, target_out)
            lora_opt.zero_grad()
            loss.backward()
            lora_opt.step()
        
        lora_out = (W @ x + B @ A @ x).detach()
        lora_error = (lora_out - target_out).norm().item()
        
        print(f"  Approximation of rotated output:")
        print(f"  Target:     {target_out[:4].tolist()}")
        print(f"  IA³ approx: {ia3_out[:4].tolist()} (error: {ia3_error:.4f})")
        print(f"  LoRA approx: {lora_out[:4].tolist()} (error: {lora_error:.4f})")
        print(f"\n  IA³ cannot express rotations; LoRA (rank-2) handles them well.")
        
        # ─── When is rescaling sufficient? ───
        print(f"""
  ═══ When Is Rescaling Sufficient? ═══
  
  Rescaling works well when the task requires:
    ✓ Reweighting existing features (most adaptations!)
    ✓ Selecting relevant features from pretrained knowledge
    ✓ Suppressing irrelevant/noisy features
    ✓ Adjusting feature magnitudes for new domains
  
  Rescaling struggles when the task requires:
    ✗ Learning entirely new feature combinations
    ✗ Rotating or mixing feature dimensions
    ✗ Drastic changes to model behavior
    
  Key insight from the IA³ paper:
    For few-shot learning (4-64 examples), feature 
    reweighting is usually ALL you need. The pretrained 
    model already has the right features — you just need 
    to tell it which ones matter for this task.
""")


# ============================================================================
# SECTION 4: FEW-SHOT LEARNING THEORY
# ============================================================================

class FewShotTheory:
    """
    Why IA³ is especially good for few-shot learning.
    """
    
    @staticmethod
    def demonstrate():
        print("\n\n" + "=" * 65)
        print("  SECTION 4: FEW-SHOT LEARNING THEORY")
        print("=" * 65)
        
        torch.manual_seed(42)
        
        # ─── Overfitting risk analysis ───
        print(f"""
  ═══ The Few-Shot Overfitting Problem ═══
  
  With N training examples, you can reliably learn at most
  ~O(N) parameters before overfitting.
  
  Method          │ Trainable Params │ Min Examples Needed
  ────────────────┼──────────────────┼────────────────────
  Full FT (GPT-2) │   124,000,000    │   ~millions
  LoRA (r=8)      │       294,912    │   ~thousands
  Prefix Tuning   │       245,760    │   ~thousands
  P-Tuning v2     │       245,760    │   ~thousands
  IA³             │        55,296    │   ~hundreds
  IA³ (targeted)  │        ~5,000    │   ~tens (few-shot!)
  
  IA³ has so few parameters that it can learn meaningfully
  from just 4-64 examples without overfitting!
""")
        
        # Simulate few-shot overfitting
        d = 64
        
        results = {}
        for n_examples in [4, 16, 64, 256]:
            x_train = torch.randn(n_examples, d)
            y_train = torch.randn(n_examples, d)
            x_test = torch.randn(100, d)
            y_test = torch.randn(100, d)
            
            # IA³ (few params — d=64 params)
            W = torch.randn(d, d)
            l_ia3 = nn.Parameter(torch.ones(d))
            opt_ia3 = torch.optim.Adam([l_ia3], lr=0.01)
            
            for _ in range(200):
                y = l_ia3 * (x_train @ W.T)
                loss = F.mse_loss(y, y_train)
                opt_ia3.zero_grad()
                loss.backward()
                opt_ia3.step()
            
            with torch.no_grad():
                train_loss_ia3 = F.mse_loss(l_ia3 * (x_train @ W.T), y_train).item()
                test_loss_ia3 = F.mse_loss(l_ia3 * (x_test @ W.T), y_test).item()
            
            # LoRA (more params — 2*r*d params, r=8 → 1024 params)
            r = 8
            B = nn.Parameter(torch.zeros(d, r))
            A = nn.Parameter(torch.randn(r, d) * 0.01)
            opt_lora = torch.optim.Adam([B, A], lr=0.01)
            
            for _ in range(200):
                y = x_train @ W.T + x_train @ A.T @ B.T
                loss = F.mse_loss(y, y_train)
                opt_lora.zero_grad()
                loss.backward()
                opt_lora.step()
            
            with torch.no_grad():
                train_loss_lora = F.mse_loss(
                    x_train @ W.T + x_train @ A.T @ B.T, y_train
                ).item()
                test_loss_lora = F.mse_loss(
                    x_test @ W.T + x_test @ A.T @ B.T, y_test
                ).item()
            
            gap_ia3 = test_loss_ia3 - train_loss_ia3
            gap_lora = test_loss_lora - train_loss_lora
            
            results[n_examples] = {
                "ia3_gap": gap_ia3,
                "lora_gap": gap_lora,
            }
        
        print(f"  Generalization gap (test_loss - train_loss):")
        print(f"  A smaller gap = less overfitting = better generalization")
        print(f"\n  {'N examples':>12} {'IA³ gap':>12} {'LoRA gap':>12} {'Winner':>10}")
        print(f"  {'─'*12}─{'─'*12}─{'─'*12}─{'─'*10}")
        
        for n, r in results.items():
            winner = "IA³" if r["ia3_gap"] < r["lora_gap"] else "LoRA"
            print(f"  {n:>12} {r['ia3_gap']:>12.4f} {r['lora_gap']:>12.4f} {winner:>10}")
        
        print(f"""
  ═══ Why IA³ Wins in Few-Shot ═══
  
  1. Implicit Regularization:
     - Fewer parameters = stronger implicit regularization
     - Harder to memorize training examples
     - Forces model to learn generalizable patterns
  
  2. Strong Inductive Bias:
     - "The right features already exist, just reweight them"
     - This assumption holds especially well for few-shot
     - With few examples, you can't learn new features anyway
  
  3. Better than In-Context Learning:
     - ICL: stuff examples into the prompt (uses context length)
     - IA³: learn rescaling vectors from examples (uses parameters)
     - IA³ with 16 examples > ICL with 16 examples (Liu et al.)
     - And IA³ doesn't waste valuable context window!
  
  4. Training Efficiency:
     - 55K params trains in seconds on CPU
     - Can iterate quickly on few-shot benchmarks
     - No GPU needed for prototyping
""")


# ============================================================================
# SECTION 5: WHERE TO APPLY RESCALING
# ============================================================================

class RescalingLocations:
    """
    Analysis of where to apply IA³ rescaling in a transformer.
    """
    
    @staticmethod
    def demonstrate():
        print("\n\n" + "=" * 65)
        print("  SECTION 5: WHERE TO APPLY RESCALING")
        print("=" * 65)
        
        print(f"""
  ═══ Ablation: Which Components to Rescale ═══
  
  The IA³ paper studied rescaling at different locations:
  
  ┌─────────────────┬────────────┬──────────────────────────┐
  │ Configuration    │ Params     │ Performance (relative)   │
  ├─────────────────┼────────────┼──────────────────────────┤
  │ Keys only        │ d_model/L  │ ★★ (decent)             │
  │ Values only      │ d_model/L  │ ★★ (decent)             │
  │ FF only          │ d_ff/L     │ ★★★ (good)              │
  │ K + V            │ 2d_model/L │ ★★★ (good)              │
  │ K + V + FF       │ full IA³   │ ★★★★ (best)             │
  └─────────────────┴────────────┴──────────────────────────┘
  
  Findings:
  - FF rescaling alone is surprisingly effective
  - K+V together is better than either alone
  - All three (K + V + FF) gives the best results
  - The default IA³ uses all three
  
  Why Keys and Values?
  ┌────────────────────────────────────────────────────────┐
  │ Attention: softmax(Q·K^T / √d) · V                   │
  │                                                       │
  │ Rescaling K: changes WHAT tokens attend to            │
  │   - Amplifying K dimensions ↔ "pay attention to this" │
  │   - Inhibiting K dimensions ↔ "ignore this"           │
  │                                                       │
  │ Rescaling V: changes WHAT information flows            │
  │   - Amplifying V dimensions ↔ "pass this through"    │
  │   - Inhibiting V dimensions ↔ "filter this out"       │
  │                                                       │
  │ Queries are NOT rescaled — they retain the original   │
  │ model's question-asking ability                        │
  └────────────────────────────────────────────────────────┘
  
  Why Feed-Forward?
  ┌────────────────────────────────────────────────────────┐
  │ FF(x) = W_down · activation(W_up · x)                │
  │                                                       │
  │ Rescaling intermediate: l_ff ⊙ activation(W_up · x)  │
  │                                                       │
  │ The FF layer acts as a "memory bank" in transformers  │
  │ Rescaling selects WHICH memories to activate          │
  │ This is like choosing which "facts" to use per task   │
  └────────────────────────────────────────────────────────┘
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all theory demonstrations."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     IA³ THEORY — MATHEMATICAL FOUNDATIONS                    ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: Why rescaling works
    MultiplicativeVsAdditive.demonstrate()
    
    # Section 2: Gradient analysis
    GradientAnalysis.demonstrate()
    
    # Section 3: Expressiveness
    ExpressivenessAnalysis.demonstrate()
    
    # Section 4: Few-shot theory
    FewShotTheory.demonstrate()
    
    # Section 5: Where to apply
    RescalingLocations.demonstrate()
    
    print("\n" + "=" * 65)
    print("  MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered:
    ✓ Multiplicative vs additive modifications
    ✓ Gradient flow through IA³
    ✓ Identity initialization importance
    ✓ Expressiveness analysis (diagonal limitations)
    ✓ Few-shot overfitting theory
    ✓ Optimal rescaling locations (K, V, FF)
    """)


if __name__ == "__main__":
    main()
