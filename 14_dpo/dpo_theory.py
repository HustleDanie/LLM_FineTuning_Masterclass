"""
DPO Theory — Mathematical Foundations of Direct Preference Optimization
========================================================================

Deep dive into the theory behind DPO:

1. RLHFToDP0Derivation
   - From RLHF objective to DPO loss
   - Step-by-step mathematical derivation

2. DPOLossLandscape
   - Visualizing the DPO loss surface
   - Gradient behavior for correct/incorrect preferences

3. BetaAnalysis
   - Effect of β on training dynamics
   - Choosing the right β for your task

4. ImplicitReward
   - How the policy defines a reward function
   - Extracting rewards from a DPO-trained model

5. DPOGradients
   - Understanding what DPO actually optimizes
   - Connection to weighted likelihood

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple


# ============================================================================
# SECTION 1: FROM RLHF OBJECTIVE TO DPO LOSS
# ============================================================================

def rlhf_to_dpo_derivation():
    """Step-by-step derivation of DPO from the RLHF objective."""
    print("=" * 70)
    print("  SECTION 1: FROM RLHF OBJECTIVE TO DPO LOSS")
    print("=" * 70)
    
    print(f"""
  ═══ Step-by-Step Derivation ═══
  
  STEP 1: The RLHF Objective
  ───────────────────────────
  RLHF maximizes expected reward with a KL penalty:
  
    max_π  E_{{x~D, y~π(y|x)}} [ r(x,y) ] - β · KL(π || π_ref)
  
  Expanding the KL term:
  
    max_π  E [ r(x,y) - β · log(π(y|x)/π_ref(y|x)) ]
  
  
  STEP 2: Closed-Form Optimal Policy
  ───────────────────────────────────
  The optimal policy π* that maximizes the above is:
  
    π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y)/β)
  
  Where Z(x) = Σ_y π_ref(y|x) · exp(r(x,y)/β) is a normalizer.
  
  Proof: This is a standard result from the calculus of variations /
  KKT conditions applied to the constrained optimization.
  
  
  STEP 3: Rearranging for the Reward
  ───────────────────────────────────
  Taking log of both sides:
  
    log π*(y|x) = log π_ref(y|x) + r(x,y)/β - log Z(x)
  
  Solving for r(x,y):
  
    r(x,y) = β · log(π*(y|x) / π_ref(y|x)) + β · log Z(x)
  
  KEY INSIGHT: The reward is determined by the log-ratio of the
  optimal policy to the reference, plus a constant per prompt.
  
  
  STEP 4: Substituting into Bradley-Terry
  ────────────────────────────────────────
  The Bradley-Terry preference model says:
  
    P(y_w ≻ y_l | x) = σ(r(x,y_w) - r(x,y_l))
  
  Substituting our reward expression:
  
    P(y_w ≻ y_l | x) = σ(β · [log π*(y_w|x)/π_ref(y_w|x)
                               - log π*(y_l|x)/π_ref(y_l|x)])
  
  NOTE: The log Z(x) terms CANCEL because they only depend on x!
  
  
  STEP 5: The DPO Loss
  ─────────────────────
  Replace π* with our trainable π_θ and maximize log-likelihood:
  
    L_DPO = -E [ log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x)
                           - log π_θ(y_l|x)/π_ref(y_l|x))) ]
  
  This is just a BINARY CROSS-ENTROPY loss on the preference margin!
""")
    
    # Demonstrate the derivation numerically
    print(f"  ── Numerical Verification ──\n")
    
    torch.manual_seed(42)
    beta = 0.1
    
    # Simulate log-probabilities
    log_pi_chosen = torch.tensor(-2.5)    # Policy log-prob of chosen
    log_pi_rejected = torch.tensor(-3.0)  # Policy log-prob of rejected
    log_ref_chosen = torch.tensor(-2.8)   # Reference log-prob of chosen
    log_ref_rejected = torch.tensor(-2.9) # Reference log-prob of rejected
    
    # Log-ratios
    chosen_log_ratio = log_pi_chosen - log_ref_chosen
    rejected_log_ratio = log_pi_rejected - log_ref_rejected
    
    # DPO logit
    dpo_logit = beta * (chosen_log_ratio - rejected_log_ratio)
    
    # DPO loss (should be low when chosen is preferred)
    dpo_loss = -F.logsigmoid(dpo_logit)
    
    # Implicit preference probability
    pref_prob = torch.sigmoid(dpo_logit)
    
    print(f"    log π_θ(y_w|x):     {log_pi_chosen.item():.3f}")
    print(f"    log π_θ(y_l|x):     {log_pi_rejected.item():.3f}")
    print(f"    log π_ref(y_w|x):   {log_ref_chosen.item():.3f}")
    print(f"    log π_ref(y_l|x):   {log_ref_rejected.item():.3f}")
    print(f"    ─────────────────────────────")
    print(f"    Chosen log-ratio:    {chosen_log_ratio.item():.3f}")
    print(f"    Rejected log-ratio:  {rejected_log_ratio.item():.3f}")
    print(f"    DPO logit (β·Δ):    {dpo_logit.item():.3f}")
    print(f"    P(y_w ≻ y_l):       {pref_prob.item():.4f}")
    print(f"    DPO loss:            {dpo_loss.item():.4f}")
    
    # Show that implicit reward preserves preference ordering
    print(f"\n  ── Implicit Rewards (from DPO policy) ──\n")
    
    implicit_r_chosen = beta * chosen_log_ratio
    implicit_r_rejected = beta * rejected_log_ratio
    
    print(f"    r(x, y_w) = β · log(π/π_ref) = {implicit_r_chosen.item():.4f}")
    print(f"    r(x, y_l) = β · log(π/π_ref) = {implicit_r_rejected.item():.4f}")
    print(f"    r(y_w) > r(y_l): {implicit_r_chosen > implicit_r_rejected}  ✓")


# ============================================================================
# SECTION 2: DPO LOSS LANDSCAPE
# ============================================================================

def dpo_loss_landscape():
    """Visualize the DPO loss surface and gradient behavior."""
    print("\n\n" + "=" * 70)
    print("  SECTION 2: DPO LOSS LANDSCAPE")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # ─── Loss as a function of the preference margin ───
    print(f"\n  ── DPO Loss vs Preference Margin ──")
    print(f"""
  The margin = β · (log π_w/π_ref_w - log π_l/π_ref_l)
  
  When margin > 0: Policy correctly prefers chosen
  When margin < 0: Policy incorrectly prefers rejected
""")
    
    margins = torch.linspace(-3.0, 3.0, 13)
    
    print(f"  {'Margin':>8} │ {'Loss':>8} │ {'Gradient':>10} │ {'Visual':>30}")
    print(f"  {'─'*8}─┼─{'─'*8}─┼─{'─'*10}─┼─{'─'*30}")
    
    for m in margins:
        m_t = m.clone().requires_grad_(True)
        loss = -F.logsigmoid(m_t)
        loss.backward()
        grad = m_t.grad.item()
        
        # Visual bar
        loss_val = loss.item()
        bar_len = int(min(loss_val, 3.0) * 10)
        bar = "█" * bar_len
        
        status = "✓" if m.item() > 0 else "✗"
        print(f"  {m.item():>8.2f} │ {loss_val:>8.4f} │ {grad:>10.4f} │ {bar} {status}")
    
    print(f"""
  OBSERVATIONS:
  • Loss → 0 as margin → +∞ (strong correct preference)
  • Loss → ∞ as margin → -∞ (strong wrong preference)
  • Gradient is largest when margin ≈ 0 (uncertain region)
  • Gradient → 0 for large positive margins (already learned)
  
  This is just LOGISTIC REGRESSION on the margin!
""")

    # ─── Loss surface over chosen/rejected log-ratios ───
    print(f"  ── 2D Loss Surface: Chosen vs Rejected Log-Ratio ──\n")
    
    beta = 0.1
    chosen_ratios = [-0.5, 0.0, 0.5, 1.0, 1.5]
    rejected_ratios = [-0.5, 0.0, 0.5, 1.0, 1.5]
    
    print(f"  β = {beta}")
    print(f"  Loss values [rows=chosen log-ratio, cols=rejected log-ratio]:\n")
    
    header = "  " + " " * 12 + "".join(f"rej={r:>5.1f} " for r in rejected_ratios)
    print(header)
    print(f"  " + "─" * (12 + len(rejected_ratios) * 10))
    
    for cr in chosen_ratios:
        row = f"  cho={cr:>5.1f} │"
        for rr in rejected_ratios:
            margin = beta * (cr - rr)
            loss = -F.logsigmoid(torch.tensor(margin)).item()
            if loss < 0.3:
                row += f"  {loss:>5.3f}✓"
            elif loss > 1.0:
                row += f"  {loss:>5.3f}✗"
            else:
                row += f"  {loss:>5.3f} "
        print(row)
    
    print(f"""
  READING THE TABLE:
  • Low loss (✓): chosen log-ratio > rejected log-ratio (correct!)
  • High loss (✗): chosen log-ratio < rejected (wrong preference)
  • Diagonal: tied → loss ≈ 0.693 (= log 2, random chance)
""")


# ============================================================================
# SECTION 3: BETA ANALYSIS
# ============================================================================

def beta_analysis():
    """Effect of β on training dynamics."""
    print("\n\n" + "=" * 70)
    print("  SECTION 3: β (BETA) ANALYSIS")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ What Does β Control? ═══
  
  β controls the KL constraint strength:
  
    L_DPO = -log σ( β · (log π_w/π_ref - log π_l/π_ref) )
  
  • Small β (0.01-0.05): Weak constraint → policy can deviate a lot
  • Medium β (0.1):  Standard → good balance
  • Large β (0.5-1.0): Strong constraint → stays close to reference
""")
    
    # Simulate DPO training with different β values
    print(f"  ── Training Dynamics for Different β ──\n")
    
    # Create a simple preference learning scenario
    n_pairs = 200
    
    # True preference: feature[0] > 0 means chosen should score higher
    features = torch.randn(n_pairs, 4)
    true_preferences = (features[:, 0] > 0).float()  # 1 if chosen wins
    
    betas = [0.01, 0.1, 0.5, 2.0]
    
    print(f"  {'β':>6} │ {'Final Loss':>10} │ {'Accuracy':>10} │ "
          f"{'KL (approx)':>12} │ {'Behavior':>20}")
    print(f"  {'─'*6}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*12}─┼─{'─'*20}")
    
    for beta in betas:
        # Simple linear model as proxy for policy log-ratios
        model = nn.Linear(4, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        for epoch in range(100):
            # Model predicts the margin (chosen - rejected log-ratio)
            margin_pred = model(features).squeeze()
            dpo_logits = beta * margin_pred
            
            # DPO loss: want logits positive when chosen wins
            targets = true_preferences  # 1 for chosen wins
            loss = F.binary_cross_entropy_with_logits(dpo_logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluate
        with torch.no_grad():
            final_margins = model(features).squeeze()
            dpo_logits = beta * final_margins
            final_loss = F.binary_cross_entropy_with_logits(
                dpo_logits, true_preferences
            ).item()
            preds = (dpo_logits > 0).float()
            accuracy = (preds == true_preferences).float().mean().item()
            kl_approx = final_margins.abs().mean().item()
        
        if beta < 0.05:
            behavior = "Aggressive learning"
        elif beta < 0.3:
            behavior = "Balanced"
        elif beta < 1.0:
            behavior = "Conservative"
        else:
            behavior = "Very conservative"
        
        print(f"  {beta:>6.2f} │ {final_loss:>10.4f} │ {accuracy:>9.1%} │ "
              f"{kl_approx:>12.4f} │ {behavior:>20}")
    
    print(f"""
  ═══ β Selection Guidelines ═══
  
  ┌─────────────┬──────────────────────────────────────────┐
  │ β Range     │ When to Use                              │
  ├─────────────┼──────────────────────────────────────────┤
  │ 0.01 - 0.05 │ • Strong SFT base model                 │
  │             │ • High confidence in preference data      │
  │             │ • Want maximum alignment effect           │
  ├─────────────┼──────────────────────────────────────────┤
  │ 0.1 (std)   │ • Default starting point                 │
  │             │ • Most papers use this value              │
  │             │ • Works well in most scenarios            │
  ├─────────────┼──────────────────────────────────────────┤
  │ 0.2 - 0.5   │ • Noisy preference data                  │
  │             │ • Want to preserve base model quality     │
  │             │ • Risk of catastrophic forgetting         │
  ├─────────────┼──────────────────────────────────────────┤
  │ 0.5 - 1.0+  │ • Very noisy data or weak SFT model      │
  │             │ • Subtle alignment changes only           │
  │             │ • Safety-critical applications            │
  └─────────────┴──────────────────────────────────────────┘
  
  PRACTICAL TIP: Start with β=0.1, if training is unstable or
  model quality degrades, increase β. If alignment effect is
  too weak, decrease β.
""")


# ============================================================================
# SECTION 4: IMPLICIT REWARD
# ============================================================================

def implicit_reward():
    """How the DPO policy defines an implicit reward function."""
    print("\n\n" + "=" * 70)
    print("  SECTION 4: IMPLICIT REWARD")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ The Policy IS the Reward Model ═══
  
  Key DPO insight: After training, the policy implicitly defines
  a reward function:
  
    r(x, y) = β · log(π_θ(y|x) / π_ref(y|x))
  
  This reward can be extracted and used for:
  • Evaluating response quality
  • Best-of-N sampling (reranking)
  • Training other models
  
  No separate reward model needed!
""")
    
    # Demonstrate implicit reward extraction
    print(f"  ── Extracting Implicit Rewards ──\n")
    
    # Simulate a trained DPO model vs reference
    vocab_size = 20
    seq_len = 5
    beta = 0.1
    
    # Simulated log-probs from trained and reference models
    # (In practice, you'd get these from actual forward passes)
    
    responses = {
        "Helpful answer": {
            "policy_logprob": -8.5,    # Policy assigns high prob
            "ref_logprob": -12.0,      # Reference assigns lower prob
        },
        "Mediocre answer": {
            "policy_logprob": -10.0,
            "ref_logprob": -10.5,
        },
        "Harmful answer": {
            "policy_logprob": -15.0,   # Policy assigns very low prob
            "ref_logprob": -9.0,       # Reference doesn't penalize
        },
        "Verbose rambling": {
            "policy_logprob": -13.0,
            "ref_logprob": -11.0,
        },
        "Concise & clear": {
            "policy_logprob": -7.0,
            "ref_logprob": -10.0,
        },
    }
    
    print(f"  {'Response':>20} │ {'log π_θ':>8} │ {'log π_ref':>9} │ "
          f"{'Implicit r':>10} │ {'Rank':>5}")
    print(f"  {'─'*20}─┼─{'─'*8}─┼─{'─'*9}─┼─{'─'*10}─┼─{'─'*5}")
    
    reward_list = []
    for name, probs in responses.items():
        log_ratio = probs["policy_logprob"] - probs["ref_logprob"]
        implicit_r = beta * log_ratio
        reward_list.append((name, implicit_r, probs["policy_logprob"], 
                           probs["ref_logprob"]))
    
    # Sort by reward for ranking
    reward_list.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (name, r, lp, lr) in enumerate(reward_list, 1):
        print(f"  {name:>20} │ {lp:>8.1f} │ {lr:>9.1f} │ {r:>10.3f} │ {rank:>5}")
    
    print(f"""
  OBSERVATION:
  • "Concise & clear" gets highest implicit reward (policy upweighted it)
  • "Harmful answer" gets lowest reward (policy strongly downweighted it)
  • The ranking captures alignment preferences without any explicit reward model!
  
  
  ═══ Best-of-N Sampling with Implicit Reward ═══
  
  Use the implicit reward for inference-time alignment:
  
  1. Generate N candidate responses from π_θ
  2. Score each with r(x,y) = β · log(π_θ(y|x)/π_ref(y|x))
  3. Return the highest-scoring response
  
  This combines the DPO policy with best-of-N for extra quality:
  
  ```python
  def best_of_n(prompt, policy, ref_model, tokenizer, n=8, beta=0.1):
      candidates = []
      for _ in range(n):
          response = policy.generate(prompt, do_sample=True)
          
          # Implicit reward
          policy_lp = get_log_prob(policy, prompt, response)
          ref_lp = get_log_prob(ref_model, prompt, response)
          reward = beta * (policy_lp - ref_lp)
          
          candidates.append((response, reward))
      
      # Return best
      return max(candidates, key=lambda x: x[1])[0]
  ```
""")


# ============================================================================
# SECTION 5: DPO GRADIENTS
# ============================================================================

def dpo_gradients():
    """Understanding what DPO actually optimizes."""
    print("\n\n" + "=" * 70)
    print("  SECTION 5: DPO GRADIENT ANALYSIS")
    print("=" * 70)
    
    print(f"""
  ═══ What Does DPO Actually Optimize? ═══
  
  The gradient of the DPO loss w.r.t. θ is:
  
    ∇L = -β · E [ σ(-u) · (∇log π_θ(y_w|x) - ∇log π_θ(y_l|x)) ]
  
  Where u = β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x))
  
  This is a WEIGHTED LIKELIHOOD update:
  • ∇log π_θ(y_w|x): Increase probability of chosen response
  • -∇log π_θ(y_l|x): Decrease probability of rejected response
  • σ(-u): Weight — larger when model gets it WRONG
  
  KEY INSIGHT: DPO focuses learning on examples the model
  finds hardest (where it incorrectly prefers the rejected response).
""")
    
    torch.manual_seed(42)
    
    # Demonstrate the weighting mechanism
    print(f"  ── Gradient Weight σ(-u) Analysis ──\n")
    
    margins = torch.linspace(-3, 3, 13)
    
    print(f"  {'Margin (u)':>11} │ {'σ(-u)':>8} │ {'Meaning':>35} │ {'Bar':>15}")
    print(f"  {'─'*11}─┼─{'─'*8}─┼─{'─'*35}─┼─{'─'*15}")
    
    for u in margins:
        weight = torch.sigmoid(-u).item()
        bar = "█" * int(weight * 15)
        
        if u.item() < -1.0:
            meaning = "Model WRONG → HIGH weight"
        elif u.item() < 0:
            meaning = "Model slightly wrong → medium weight"
        elif u.item() < 1.0:
            meaning = "Model slightly right → medium weight"
        else:
            meaning = "Model RIGHT → LOW weight"
        
        print(f"  {u.item():>11.2f} │ {weight:>8.4f} │ {meaning:>35} │ {bar}")
    
    print(f"""
  IMPLICATIONS:
    
  1. SELF-CORRECTING: DPO automatically focuses on its mistakes.
     Already-learned preferences get small gradients → no overfitting.
  
  2. COMPARISON WITH SFT:
     • SFT: Equal weight on all examples (chosen responses only)
     • DPO: Higher weight on harder examples, uses both chosen AND rejected
     
  3. COMPARISON WITH RLHF:
     • RLHF: Weights come from advantage estimation (complex)
     • DPO: Weights come from current preference margin (simple)
  
  
  ═══ DPO as Weighted Likelihood ═══
  
  DPO gradient can be rewritten as:
  
    ∇L ≈ w_chosen · ∇(log-likelihood of chosen)
        - w_rejected · ∇(log-likelihood of rejected)
  
  Where weights are determined by how well the model currently
  distinguishes between chosen and rejected.
  
  This makes DPO essentially a CONTRASTIVE learning objective:
  • Push up probability of chosen responses  ↑
  • Push down probability of rejected responses ↓
  • Focus on the hardest pairs
""")
    
    # Demonstrate gradient flow during training
    print(f"  ── Gradient Flow During Training ──\n")
    
    # Simple simulation
    d = 8
    model = nn.Linear(d, 1)
    ref_model = nn.Linear(d, 1)
    ref_model.load_state_dict(model.state_dict())
    for p in ref_model.parameters():
        p.requires_grad = False
    
    beta = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Create preference pairs
    n_pairs = 50
    chosen_features = torch.randn(n_pairs, d) + 0.3
    rejected_features = torch.randn(n_pairs, d) - 0.3
    
    print(f"  {'Epoch':>6} │ {'Loss':>8} │ {'Accuracy':>8} │ "
          f"{'Avg Weight':>10} │ {'Max Weight':>10}")
    print(f"  {'─'*6}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*10}─┼─{'─'*10}")
    
    for epoch in range(10):
        # Forward pass
        chosen_scores = model(chosen_features).squeeze()
        rejected_scores = model(rejected_features).squeeze()
        ref_chosen = ref_model(chosen_features).squeeze()
        ref_rejected = ref_model(rejected_features).squeeze()
        
        # Log-ratios
        chosen_log_ratio = chosen_scores - ref_chosen
        rejected_log_ratio = rejected_scores - ref_rejected
        
        # DPO loss
        margins = beta * (chosen_log_ratio - rejected_log_ratio)
        loss = -F.logsigmoid(margins).mean()
        
        # Gradient weights
        with torch.no_grad():
            weights = torch.sigmoid(-margins)
            accuracy = (margins > 0).float().mean()
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            print(f"  {epoch+1:>6} │ {loss.item():>8.4f} │ {accuracy.item():>7.1%} │ "
                  f"{weights.mean().item():>10.4f} │ {weights.max().item():>10.4f}")
    
    print(f"""
  NOTICE: As accuracy increases (model learns correct preferences),
  the average gradient weight σ(-u) decreases — the model focuses
  less on already-learned pairs and more on remaining hard cases.
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all DPO theory sections."""
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║    DPO THEORY — MATHEMATICAL FOUNDATIONS                          ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    # Section 1: Derivation
    rlhf_to_dpo_derivation()
    
    # Section 2: Loss landscape
    dpo_loss_landscape()
    
    # Section 3: Beta analysis
    beta_analysis()
    
    # Section 4: Implicit reward
    implicit_reward()
    
    # Section 5: Gradient analysis
    dpo_gradients()
    
    print("\n" + "=" * 70)
    print("  DPO THEORY MODULE COMPLETE")
    print("=" * 70)
    print("""
    Covered:
    ✓ RLHF → DPO derivation (5-step proof)
    ✓ Loss landscape (logistic regression on preference margin)
    ✓ β analysis (constraint strength, selection guidelines)
    ✓ Implicit reward (policy IS the reward model)
    ✓ Gradient analysis (weighted contrastive likelihood)
    """)


if __name__ == "__main__":
    main()
