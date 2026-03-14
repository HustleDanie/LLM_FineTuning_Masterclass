"""
DPO Comparison — DPO vs Alternative Preference Methods
========================================================

Comprehensive comparison of DPO with related approaches:

1. DPOvsIPO
   - DPO vs Identity Preference Optimization
   - Robustness to label noise

2. DPOvsORPO
   - DPO vs Odds Ratio Preference Optimization
   - Eliminating the reference model entirely

3. DPOvsSimPO
   - DPO vs Simple Preference Optimization
   - Length-normalized, reference-free

4. DPOvsRLHF
   - Detailed DPO vs PPO-based RLHF comparison
   - When each approach wins

5. AlignmentMethodsTimeline
   - Evolution of preference learning methods
   - Practical selection guide

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple


# ============================================================================
# SECTION 1: DPO vs IPO
# ============================================================================

def dpo_vs_ipo():
    """DPO vs Identity Preference Optimization."""
    print("=" * 70)
    print("  SECTION 1: DPO vs IPO")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ The Problem with DPO ═══
  
  DPO assumes the Bradley-Terry preference model:
    P(y_w > y_l) = σ(r(y_w) - r(y_l))
  
  If this assumption is WRONG (e.g., noisy labels, non-transitive
  preferences), DPO can overfit to incorrect preferences.
  
  
  ═══ IPO (Identity Preference Optimization) ═══
  
  Azar et al. (2023) proposed IPO which is more robust:
  
  L_IPO = E[ (log(π_θ(y_w)/π_ref(y_w)) - log(π_θ(y_l)/π_ref(y_l)) - 1/(2β))² ]
  
  Key difference:
  • DPO: sigmoid loss (logistic regression)
  • IPO: squared loss (regression toward fixed target)
  
  IPO doesn't assume Bradley-Terry → more robust to noise!
""")
    
    # Implement both losses
    def dpo_loss(chosen_log_ratio, rejected_log_ratio, beta=0.1):
        logits = beta * (chosen_log_ratio - rejected_log_ratio)
        return -F.logsigmoid(logits).mean()
    
    def ipo_loss(chosen_log_ratio, rejected_log_ratio, beta=0.1):
        diff = chosen_log_ratio - rejected_log_ratio
        target = 1.0 / (2 * beta)
        return ((diff - target) ** 2).mean()
    
    # Compare on clean vs noisy data
    n = 200
    
    # Clean data: chosen genuinely preferred
    clean_chosen_lr = torch.randn(n) + 0.5
    clean_rejected_lr = torch.randn(n) - 0.5
    
    # Noisy data: 20% of labels flipped
    noisy_chosen_lr = clean_chosen_lr.clone()
    noisy_rejected_lr = clean_rejected_lr.clone()
    flip_mask = torch.rand(n) < 0.2
    noisy_chosen_lr[flip_mask], noisy_rejected_lr[flip_mask] = \
        noisy_rejected_lr[flip_mask], noisy_chosen_lr[flip_mask]
    
    print(f"\n  ── Loss Comparison: Clean vs Noisy Data ──\n")
    
    print(f"  {'Condition':>20} │ {'DPO Loss':>10} │ {'IPO Loss':>10} │ {'Diff':>8}")
    print(f"  {'─'*20}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*8}")
    
    for label, c_lr, r_lr in [("Clean data", clean_chosen_lr, clean_rejected_lr),
                               ("20% noise", noisy_chosen_lr, noisy_rejected_lr)]:
        d_loss = dpo_loss(c_lr, r_lr).item()
        i_loss = ipo_loss(c_lr, r_lr).item()
        print(f"  {label:>20} │ {d_loss:>10.4f} │ {i_loss:>10.4f} │ {abs(d_loss-i_loss):>8.4f}")
    
    # Train both on noisy data
    print(f"\n  ── Training on 20% Noisy Labels ──\n")
    
    d_model = 8
    
    # DPO model
    dpo_model = nn.Linear(d_model, 1)
    dpo_opt = torch.optim.Adam(dpo_model.parameters(), lr=0.01)
    
    # IPO model
    ipo_model = nn.Linear(d_model, 1)
    ipo_model.load_state_dict(dpo_model.state_dict())
    ipo_opt = torch.optim.Adam(ipo_model.parameters(), lr=0.01)
    
    # Features
    chosen_feats = torch.randn(n, d_model) + 0.2
    rejected_feats = torch.randn(n, d_model) - 0.2
    
    # Flip 20% (noise)
    flip = torch.rand(n) < 0.2
    noisy_chosen = chosen_feats.clone()
    noisy_rejected = rejected_feats.clone()
    noisy_chosen[flip] = rejected_feats[flip]
    noisy_rejected[flip] = chosen_feats[flip]
    
    # True labels (unflipped)
    true_labels = ~flip
    
    print(f"  {'Epoch':>6} │ {'DPO Acc':>8} │ {'IPO Acc':>8} │ {'DPO TrueAcc':>11} │ {'IPO TrueAcc':>11}")
    print(f"  {'─'*6}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*11}─┼─{'─'*11}")
    
    beta = 0.1
    for epoch in range(30):
        # DPO
        dc = dpo_model(noisy_chosen).squeeze()
        dr = dpo_model(noisy_rejected).squeeze()
        d_logits = beta * (dc - dr)
        d_l = -F.logsigmoid(d_logits).mean()
        dpo_opt.zero_grad(); d_l.backward(); dpo_opt.step()
        
        # IPO
        ic = ipo_model(noisy_chosen).squeeze()
        ir = ipo_model(noisy_rejected).squeeze()
        i_diff = ic - ir
        i_l = ((i_diff - 1/(2*beta)) ** 2).mean()
        ipo_opt.zero_grad(); i_l.backward(); ipo_opt.step()
        
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                # Accuracy on noisy labels (training data)
                dpo_acc = (d_logits > 0).float().mean()
                ipo_acc = (i_diff > 0).float().mean()
                
                # True accuracy (on CLEAN labels)
                dc_t = dpo_model(chosen_feats).squeeze()
                dr_t = dpo_model(rejected_feats).squeeze()
                dpo_true = ((dc_t - dr_t) > 0).float().mean()
                
                ic_t = ipo_model(chosen_feats).squeeze()
                ir_t = ipo_model(rejected_feats).squeeze()
                ipo_true = ((ic_t - ir_t) > 0).float().mean()
            
            print(f"  {epoch+1:>6} │ {dpo_acc:>7.1%} │ {ipo_acc:>7.1%} │ "
                  f"{dpo_true:>10.1%} │ {ipo_true:>10.1%}")
    
    print(f"""
  KEY INSIGHT: IPO is more robust to label noise.
  DPO overfits to noisy labels; IPO generalizes better to true prefs.
  
  USE IPO WHEN:
  • Label noise > 10%
  • Low annotator agreement
  • Preference data from diverse sources
""")
    
    del dpo_model, ipo_model


# ============================================================================
# SECTION 2: DPO vs ORPO
# ============================================================================

def dpo_vs_orpo():
    """DPO vs Odds Ratio Preference Optimization."""
    print("\n\n" + "=" * 70)
    print("  SECTION 2: DPO vs ORPO")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ ORPO: No Reference Model Needed! ═══
  
  Hong et al. (2024) proposed ORPO which eliminates the reference
  model entirely by combining SFT loss with an odds-ratio penalty.
  
  L_ORPO = L_SFT(y_w) + λ · L_OR(y_w, y_l)
  
  Where:
  • L_SFT = standard cross-entropy on chosen response
  • L_OR = -log σ(log odds(y_w) - log odds(y_l))
  
  odds(y) = P(y|x) / (1 - P(y|x))
  
  
  ═══ Pipeline Comparison ═══
  
  DPO:
    SFT Model → Load as Policy + Reference → DPO Training
    (2 models needed)
  
  ORPO:
    Pre-trained Model → ORPO Training (SFT + preference in one!)
    (1 model needed, no separate SFT step!)
  
  Memory: DPO needs 2× model size, ORPO needs only 1×
""")
    
    # Implement ORPO loss
    def orpo_loss(policy_chosen_logps, policy_rejected_logps, 
                  sft_loss_chosen, lambda_weight=1.0):
        """
        ORPO loss = SFT loss + λ * odds ratio loss.
        
        Args:
            policy_chosen_logps: Log P(chosen) per sequence
            policy_rejected_logps: Log P(rejected) per sequence
            sft_loss_chosen: Cross-entropy loss on chosen responses
            lambda_weight: Weight for preference component
        """
        # Convert log-probs to probs and compute odds
        chosen_probs = torch.exp(policy_chosen_logps)
        rejected_probs = torch.exp(policy_rejected_logps)
        
        # Odds = p/(1-p), but clamp for stability
        chosen_odds = chosen_probs / (1 - chosen_probs + 1e-8)
        rejected_odds = rejected_probs / (1 - rejected_probs + 1e-8)
        
        # Log odds ratio
        log_odds_ratio = torch.log(chosen_odds + 1e-8) - torch.log(rejected_odds + 1e-8)
        
        # OR loss (same form as DPO but on odds rather than log-ratios)
        or_loss = -F.logsigmoid(log_odds_ratio).mean()
        
        # Total ORPO loss
        total_loss = sft_loss_chosen + lambda_weight * or_loss
        
        return total_loss, {
            "sft_loss": sft_loss_chosen.item(),
            "or_loss": or_loss.item(),
            "total_loss": total_loss.item(),
        }
    
    # Compare DPO vs ORPO memory and complexity
    print(f"""
  ═══ Head-to-Head: DPO vs ORPO ═══
  
  ┌────────────────────┬──────────────────┬──────────────────┐
  │ Aspect             │ DPO              │ ORPO             │
  ├────────────────────┼──────────────────┼──────────────────┤
  │ Models in memory   │ 2 (policy + ref) │ 1 (policy only)  │
  │ Memory             │ ~2× model size   │ ~1× model size   │
  │ Needs SFT first?   │ YES              │ NO (built-in)    │
  │ Reference model    │ Required         │ Not needed        │
  │ Loss components    │ Preference only  │ SFT + preference │
  │ Training stages    │ 2 (SFT → DPO)   │ 1 (ORPO only)    │
  │ Hyperparameters    │ β, lr            │ λ, lr             │
  │ Performance        │ Strong           │ Comparable        │
  │ Best for           │ Post-SFT align.  │ Single-stage      │
  │                    │                  │ alignment         │
  └────────────────────┴──────────────────┴──────────────────┘
  
  USE ORPO WHEN:
  • Memory constrained (can't fit 2 models)
  • Want simpler pipeline (skip SFT stage)
  • Starting from pre-trained (not SFT) model
  
  USE DPO WHEN:
  • Already have a good SFT model
  • Want fine-grained control with β
  • Need the implicit reward function
  • Better studied, more community resources
""")


# ============================================================================
# SECTION 3: DPO vs SimPO
# ============================================================================

def dpo_vs_simpo():
    """DPO vs Simple Preference Optimization."""
    print("\n\n" + "=" * 70)
    print("  SECTION 3: DPO vs SimPO")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ SimPO: Simpler DPO ═══
  
  Meng et al. (2024) proposed SimPO with two key changes:
  
  1. LENGTH NORMALIZATION: Divide log-prob by response length
     → Prevents bias toward shorter/longer responses
  
  2. NO REFERENCE MODEL: Uses only the policy
     → Even simpler than DPO
  
  L_SimPO = -log σ( β/|y_w| · log π(y_w|x) - β/|y_l| · log π(y_l|x) - γ )
  
  Where:
  • |y_w|, |y_l| = response lengths
  • γ = target reward margin (new hyperparameter)
""")
    
    # Implement SimPO loss
    def simpo_loss(policy_chosen_logps, policy_rejected_logps,
                   chosen_lengths, rejected_lengths,
                   beta=2.0, gamma=0.5):
        """
        SimPO loss — length-normalized, reference-free.
        
        Args:
            policy_chosen_logps: Sum of log-probs for chosen
            policy_rejected_logps: Sum of log-probs for rejected
            chosen_lengths: Number of tokens in chosen responses
            rejected_lengths: Number of tokens in rejected responses
            beta: Temperature (typically higher than DPO, ~2.0)
            gamma: Target reward margin
        """
        # Length-normalized log-probs (average per-token)
        chosen_avg = policy_chosen_logps / chosen_lengths
        rejected_avg = policy_rejected_logps / rejected_lengths
        
        # SimPO logit (no reference model!)
        logits = beta * (chosen_avg - rejected_avg) - gamma
        
        loss = -F.logsigmoid(logits).mean()
        
        return loss, {
            "loss": loss.item(),
            "accuracy": (logits > 0).float().mean().item(),
            "chosen_avg_lp": chosen_avg.mean().item(),
            "rejected_avg_lp": rejected_avg.mean().item(),
        }
    
    def dpo_loss_ref(policy_chosen_logps, policy_rejected_logps,
                     ref_chosen_logps, ref_rejected_logps, beta=0.1):
        """Standard DPO loss for comparison."""
        chosen_lr = policy_chosen_logps - ref_chosen_logps
        rejected_lr = policy_rejected_logps - ref_rejected_logps
        logits = beta * (chosen_lr - rejected_lr)
        loss = -F.logsigmoid(logits).mean()
        return loss, {"loss": loss.item(), "accuracy": (logits > 0).float().mean().item()}
    
    # Compare on synthetic data
    n = 100
    
    # Simulated log-probs with length variation
    chosen_lps = torch.randn(n) * 2 - 15      # Sum of log-probs
    rejected_lps = torch.randn(n) * 2 - 18    # Lower (worse)
    ref_chosen = torch.randn(n) * 2 - 16
    ref_rejected = torch.randn(n) * 2 - 17
    
    # Lengths (chosen: 10-30 tokens, rejected: 5-40 tokens)
    chosen_lens = torch.randint(10, 30, (n,)).float()
    rejected_lens = torch.randint(5, 40, (n,)).float()
    
    simpo_l, simpo_m = simpo_loss(chosen_lps, rejected_lps, chosen_lens, rejected_lens)
    dpo_l, dpo_m = dpo_loss_ref(chosen_lps, rejected_lps, ref_chosen, ref_rejected)
    
    print(f"\n  ── Comparison on Same Data ──\n")
    print(f"    {'Method':>10} │ {'Loss':>8} │ {'Accuracy':>8} │ {'Ref Model?':>10}")
    print(f"    {'─'*10}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*10}")
    print(f"    {'DPO':>10} │ {dpo_m['loss']:>8.4f} │ {dpo_m['accuracy']:>7.1%} │ {'Yes':>10}")
    print(f"    {'SimPO':>10} │ {simpo_m['loss']:>8.4f} │ {simpo_m['accuracy']:>7.1%} │ {'No':>10}")
    
    print(f"""
  ═══ DPO vs SimPO ═══
  
  ┌────────────────────┬──────────────────┬──────────────────┐
  │ Aspect             │ DPO              │ SimPO            │
  ├────────────────────┼──────────────────┼──────────────────┤
  │ Reference model    │ Required         │ Not needed        │
  │ Memory             │ ~2× model size   │ ~1× model size   │
  │ Length bias        │ Possible         │ Normalized        │
  │ β range            │ 0.05 - 0.5       │ 1.0 - 5.0        │
  │ Extra hyper.       │ —                │ γ (margin)        │
  │ Implicit reward    │ Log-ratio based  │ Avg log-prob      │
  │ Simplicity         │ Simple           │ Simpler           │
  │ Maturity           │ Well-studied     │ Newer             │
  └────────────────────┴──────────────────┴──────────────────┘
  
  SimPO is ideal for:
  • Quick experiments (no reference model setup)
  • Varied-length responses
  • Memory-constrained settings
""")


# ============================================================================
# SECTION 4: DPO vs RLHF (DETAILED)
# ============================================================================

def dpo_vs_rlhf_detailed():
    """Detailed DPO vs PPO-based RLHF comparison."""
    print("\n\n" + "=" * 70)
    print("  SECTION 4: DPO vs RLHF — DETAILED COMPARISON")
    print("=" * 70)
    
    print(f"""
  ═══ Architecture Complexity ═══
  
  RLHF/PPO:
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ Policy   │ │Reference │ │ Reward   │ │  Value   │
  │ (train)  │ │ (frozen) │ │ (frozen) │ │  Head    │
  └──────────┘ └──────────┘ └──────────┘ └──────────┘
  Memory: 4× model params  |  Complexity: HIGH
  
  DPO:
  ┌──────────┐ ┌──────────┐
  │ Policy   │ │Reference │
  │ (train)  │ │ (frozen) │
  └──────────┘ └──────────┘
  Memory: 2× model params  |  Complexity: LOW
  
  
  ═══ Training Dynamics ═══
  
  RLHF/PPO:
  • Online — generates responses during training
  • Reward model provides signal
  • PPO update with advantages, clipping, value loss
  • Can explore beyond preference dataset
  • Reward hacking is a real risk
  
  DPO:
  • Offline — static preference dataset
  • No separate reward model needed
  • Simple binary cross-entropy update
  • Limited to preference dataset distribution
  • More stable, less prone to hacking
  
  
  ═══ Empirical Performance ═══
  
  ┌────────────────────┬───────────────┬───────────────┐
  │ Benchmark          │ RLHF/PPO      │ DPO           │
  ├────────────────────┼───────────────┼───────────────┤
  │ AlpacaEval 2.0     │ ★★★★★         │ ★★★★          │
  │ MT-Bench           │ ★★★★★         │ ★★★★½         │
  │ Open LLM Leader.   │ ★★★★★         │ ★★★★          │
  │ ChatBot Arena      │ ★★★★★         │ ★★★★          │
  │ Safety (TruthfulQA)│ ★★★★          │ ★★★★          │
  └────────────────────┴───────────────┴───────────────┘
  
  Note: Gap is narrowing. Online/iterative DPO approaches
  RLHF quality in many settings.
  
  
  ═══ When Each Wins ═══
  
  RLHF WINS:
  ✓ Frontier models (GPT-4, Claude, Gemini)
  ✓ Continuous improvement with new data
  ✓ Tasks where exploration is critical
  ✓ When you have a strong reward model
  ✓ Large compute budget available
  
  DPO WINS:
  ✓ Open-source models (LLaMA, Mistral)
  ✓ Limited compute/memory
  ✓ Need for training stability
  ✓ Good static preference data available
  ✓ First alignment experiment
  ✓ Rapid iteration and experimentation
  
  
  ═══ Engineering Complexity ═══
  
  ┌─────────────────────┬──────────────┬──────────────┐
  │ Aspect              │ RLHF/PPO     │ DPO          │
  ├─────────────────────┼──────────────┼──────────────┤
  │ Codebase size       │ ~1000 lines  │ ~200 lines   │
  │ Debug time          │ Days-weeks   │ Hours-days    │
  │ Hyperparameters     │ 10+          │ 2-3          │
  │ Training time       │ 2-10×        │ 1× baseline  │
  │ Failure modes       │ Many         │ Few           │
  │ Team expertise      │ RL needed    │ NLP enough    │
  │ Reproducibility     │ Harder       │ Easier        │
  └─────────────────────┴──────────────┴──────────────┘
""")


# ============================================================================
# SECTION 5: ALIGNMENT METHODS TIMELINE
# ============================================================================

def alignment_methods_timeline():
    """Evolution of preference learning methods."""
    print("\n\n" + "=" * 70)
    print("  SECTION 5: ALIGNMENT METHODS TIMELINE & SELECTION GUIDE")
    print("=" * 70)
    
    print(f"""
  ═══ Evolution of Alignment Methods ═══
  
  2017 ┃ Christiano et al.
       ┃ "Deep RL from Human Preferences"
       ┃ → First RLHF framework
       ┃
  2020 ┃ Stiennon et al.
       ┃ "Learning to Summarize from Human Feedback"
       ┃ → RLHF applied to language
       ┃
  2022 ┃ Ouyang et al. (OpenAI)
       ┃ "InstructGPT / ChatGPT"
       ┃ → RLHF at scale (SFT → RM → PPO)
       ┃
  2023 ┃ Rafailov et al.
  MAY  ┃ "DPO: Your Language Model Is Secretly a Reward Model"
       ┃ → Eliminates RL, simple preference loss
       ┃
  2023 ┃ Azar et al.
  OCT  ┃ "IPO: A General Paradigm for Learning from Feedback"
       ┃ → More robust to label noise
       ┃
  2024 ┃ Ethayarajh et al.
  JAN  ┃ "KTO: Model Alignment as Prospect Theoretic Optimization"
       ┃ → Only needs binary (thumbs up/down) feedback
       ┃
  2024 ┃ Hong et al.
  MAR  ┃ "ORPO: Monolithic Preference Optimization"
       ┃ → No reference model, combines SFT + preference
       ┃
  2024 ┃ Meng et al.
  MAY  ┃ "SimPO: Simple Preference Optimization"
       ┃ → Length-normalized, reference-free
       ┃
  2024 ┃ Various
       ┃ Online DPO, Iterative DPO, Self-Play DPO
       ┃ → Combining online generation with DPO
       ┃
  2025+┃ Emerging: Process Reward Models, RLVR, Constitutional AI
       ┃ → Per-step rewards, verifiable rewards, AI-guided alignment
  
  
  ═══ PRACTICAL SELECTION GUIDE ═══
  
  START: What are your constraints?
  
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  Q1: Do you have paired preference data (A vs B)?          │
  │      YES → Go to Q2                                        │
  │      NO  → Do you have binary data (👍/👎)?                │
  │            YES → Use KTO                                   │
  │            NO  → Collect preference data first             │
  │                                                             │
  │  Q2: How much GPU memory do you have?                      │
  │      < 2× model size → Use ORPO or SimPO (1 model)        │
  │      ≥ 2× model size → Go to Q3                           │
  │                                                             │
  │  Q3: How complex is your alignment task?                   │
  │      Simple → Use DPO (stable, well-studied)               │
  │      Complex & have compute → Use RLHF/PPO                │
  │      Complex & limited compute → Use Online DPO            │
  │                                                             │
  │  Q4: How noisy is your data?                               │
  │      Clean (>80% agree) → Standard DPO                     │
  │      Noisy (<80% agree) → IPO or DPO + label_smoothing    │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
  
  
  ═══ FINAL SUMMARY TABLE ═══
  
  ┌─────────┬───────┬────────┬────────┬──────────┬─────────────────────┐
  │ Method  │Memory │Ref.Mod │Stab.   │Performance│ Best For            │
  ├─────────┼───────┼────────┼────────┼──────────┼─────────────────────┤
  │ RLHF    │ 4×    │ ✓      │ Low    │ Highest  │ Frontier models     │
  │ DPO     │ 2×    │ ✓      │ High   │ High     │ General alignment   │
  │ IPO     │ 2×    │ ✓      │ High   │ High     │ Noisy labels        │
  │ KTO     │ 2×    │ ✓      │ High   │ Good     │ Binary feedback     │
  │ ORPO    │ 1×    │ ✗      │ High   │ Good     │ Memory-limited      │
  │ SimPO   │ 1×    │ ✗      │ High   │ Good     │ Varied-length resp. │
  │ OnlDPO  │ 2×    │ ✓      │ Med    │ V.High   │ Best of both worlds │
  └─────────┴───────┴────────┴────────┴──────────┴─────────────────────┘
  
  PRACTICAL ADVICE:
  • Start with DPO — it's the best risk/reward trade-off
  • If DPO isn't enough, try Online DPO before jumping to RLHF
  • Only use RLHF/PPO if you have the team and compute for it
  • For most open-source projects, DPO + LoRA is the sweet spot
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all DPO comparison sections."""
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║    DPO COMPARISON — DPO vs ALTERNATIVE METHODS                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    # Section 1: DPO vs IPO
    dpo_vs_ipo()
    
    # Section 2: DPO vs ORPO
    dpo_vs_orpo()
    
    # Section 3: DPO vs SimPO
    dpo_vs_simpo()
    
    # Section 4: DPO vs RLHF
    dpo_vs_rlhf_detailed()
    
    # Section 5: Timeline and selection guide
    alignment_methods_timeline()
    
    print("\n" + "=" * 70)
    print("  DPO COMPARISON MODULE COMPLETE")
    print("=" * 70)
    print("""
    Covered:
    ✓ DPO vs IPO — robustness to label noise
    ✓ DPO vs ORPO — eliminating reference model
    ✓ DPO vs SimPO — length normalization, reference-free
    ✓ DPO vs RLHF — detailed architecture and performance comparison
    ✓ Timeline — evolution from RLHF to modern methods
    ✓ Selection guide — practical decision framework
    """)


if __name__ == "__main__":
    main()
