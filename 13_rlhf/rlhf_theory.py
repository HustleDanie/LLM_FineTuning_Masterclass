"""
RLHF Theory — The Mathematics of Learning from Human Preferences
================================================================

Deep theoretical foundations of RLHF:

1. BradleyTerryModel
   - Preference modeling mathematics
   - From pairwise comparisons to reward scores

2. PPOTheory
   - Policy gradient foundations
   - Proximal Policy Optimization for LLMs
   - Clipping, advantage estimation, value functions

3. KLDivergenceTheory
   - Why KL penalty is essential
   - Reward hacking prevention
   - The exploration-exploitation balance

4. RewardModelTheory
   - Architecture choices
   - Calibration, overoptimization, Goodhart's Law

5. RLHFPipelineTheory
   - End-to-end mathematical framework
   - Why 4 models are needed
   - Convergence and stability analysis

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional


# ============================================================================
# SECTION 1: BRADLEY-TERRY MODEL
# ============================================================================

class BradleyTerryModel:
    """
    The Bradley-Terry model for pairwise preference learning.
    
    THEORY:
    ───────
    Given a prompt x and two responses y_w (preferred) and y_l (dispreferred),
    the probability that y_w is preferred over y_l is modeled as:
    
      P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))
    
    where:
      σ = sigmoid function
      r(x, y) = learned reward function
    
    The training loss maximizes the log-likelihood of observed preferences:
    
      L = -E[log σ(r(x, y_w) - r(x, y_l))]
    
    INTUITION:
    ──────────
    - If r(chosen) >> r(rejected): σ(diff) ≈ 1, loss ≈ 0 (correct!)
    - If r(chosen) << r(rejected): σ(diff) ≈ 0, loss → ∞ (wrong!)
    - If r(chosen) ≈ r(rejected):  σ(diff) ≈ 0.5, loss ≈ 0.69 (uncertain)
    
    This is exactly binary cross-entropy where the "label" is always 1
    (chosen is always preferred).
    """
    
    @staticmethod
    def demonstrate():
        print("=" * 65)
        print("  SECTION 1: BRADLEY-TERRY PREFERENCE MODEL")
        print("=" * 65)
        
        torch.manual_seed(42)
        
        # Simulate reward scores
        print("\n  ── How the Bradley-Terry loss works ──")
        
        # Different reward gaps
        gaps = torch.tensor([-3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0])
        probs = torch.sigmoid(gaps)
        losses = -torch.log(probs + 1e-10)
        
        print(f"\n  {'r(chosen)-r(rejected)':>24} {'P(correct)':>12} {'Loss':>10}")
        print(f"  {'─'*24}─{'─'*12}─{'─'*10}")
        
        for gap, prob, loss in zip(gaps, probs, losses):
            bar = "█" * int(prob.item() * 20)
            print(f"  {gap.item():>24.1f} {prob.item():>12.4f} {loss.item():>10.4f} {bar}")
        
        print(f"""
  KEY INSIGHTS:
  • Loss approaches 0 when chosen reward >> rejected reward
  • Loss increases sharply when the model gets preferences wrong
  • At gap = 0 (equal scores), loss = ln(2) ≈ 0.693
  • The sigmoid ensures smooth, differentiable gradients
""")
        
        # Train a tiny reward model
        print("  ── Training a Minimal Reward Model ──")
        
        d_model = 8
        reward_net = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        
        # Synthetic preference data
        # Feature: dim 0 positive → chosen, dim 0 negative → rejected
        n_pairs = 100
        chosen_features = torch.randn(n_pairs, d_model)
        chosen_features[:, 0] = torch.abs(chosen_features[:, 0]) + 0.5
        
        rejected_features = torch.randn(n_pairs, d_model)
        rejected_features[:, 0] = -torch.abs(rejected_features[:, 0]) - 0.5
        
        optimizer = torch.optim.Adam(reward_net.parameters(), lr=1e-3)
        
        for step in range(200):
            r_chosen = reward_net(chosen_features)
            r_rejected = reward_net(rejected_features)
            
            # Bradley-Terry loss
            loss = -torch.log(torch.sigmoid(r_chosen - r_rejected) + 1e-10).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 50 == 0:
                accuracy = (r_chosen > r_rejected).float().mean()
                print(f"    Step {step+1}: loss={loss.item():.4f}, "
                      f"accuracy={accuracy.item():.2%}, "
                      f"avg_gap={( r_chosen - r_rejected).mean().item():.3f}")
        
        print(f"\n  The reward model learned to assign higher scores to preferred responses!")
    
    @staticmethod
    def demonstrate_preference_data():
        """Show how preference data is structured."""
        print(f"""
  ═══ Preference Data Format ═══
  
  Each example is a triple: (prompt, chosen_response, rejected_response)
  
  Example 1:
    Prompt:   "Explain quantum computing simply."
    Chosen:   "Quantum computers use qubits that can be 0 and 1
               simultaneously, allowing parallel computation..."
    Rejected: "Quantum computing is a type of computing that uses
               quantum mechanics, which is very complex..."
    
  Example 2:
    Prompt:   "Write a haiku about AI."
    Chosen:   "Silicon minds think / Patterns in the data flow /
               Knowledge emerges"
    Rejected: "AI is smart tech / Computers learning from data /
               The future is now"
    
  WHY PAIRWISE?
  ─────────────
  • Absolute ratings are noisy (calibration differs across annotators)
  • Comparative judgments are more consistent
  • "Is A better than B?" is easier than "Rate A on 1-10"
  • Bradley-Terry naturally handles pairwise data
""")


# ============================================================================
# SECTION 2: PPO THEORY
# ============================================================================

class PPOTheory:
    """
    Proximal Policy Optimization for Language Models.
    
    THEORY:
    ───────
    In RLHF, we frame text generation as a reinforcement learning problem:
    
    - State:  The prompt + tokens generated so far
    - Action: The next token to generate
    - Policy: The language model π_θ(token | context)
    - Reward: Reward model score for the complete response
    
    POLICY GRADIENT:
    ────────────────
    The basic policy gradient theorem:
    
      ∇J(θ) = E[∇log π_θ(a|s) · Â(s,a)]
    
    where Â is the advantage (how much better this action was vs average).
    
    Problem: Plain policy gradient has high variance and can take
    destructively large steps.
    
    PPO SOLUTION:
    ─────────────
    PPO clips the policy ratio to prevent large updates:
    
      r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
    
      L_CLIP = E[min(r_t · Ât, clip(r_t, 1-ε, 1+ε) · Ât)]
    
    The clip function ensures r_t stays in [1-ε, 1+ε], preventing
    the policy from changing too dramatically in one step.
    """
    
    @staticmethod
    def demonstrate():
        print("\n\n" + "=" * 65)
        print("  SECTION 2: PPO THEORY")
        print("=" * 65)
        
        torch.manual_seed(42)
        
        # ─── Demonstrate the PPO clipping mechanism ───
        print("\n  ── PPO Clipping Mechanism ──")
        
        epsilon = 0.2
        
        # Policy ratio r(θ) = π_new / π_old
        ratios = torch.linspace(0.5, 2.0, 100)
        
        # Case 1: Positive advantage (good action)
        A_pos = 1.0
        unclipped_pos = ratios * A_pos
        clipped_pos = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * A_pos
        ppo_obj_pos = torch.min(unclipped_pos, clipped_pos)
        
        # Case 2: Negative advantage (bad action) 
        A_neg = -1.0
        unclipped_neg = ratios * A_neg
        clipped_neg = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * A_neg
        ppo_obj_neg = torch.min(unclipped_neg, clipped_neg)
        
        print(f"\n  PPO with ε={epsilon}:")
        print(f"\n  {'Ratio':>8} {'Unclip(A>0)':>12} {'PPO(A>0)':>10} "
              f"{'Unclip(A<0)':>12} {'PPO(A<0)':>10}")
        print(f"  {'─'*8}─{'─'*12}─{'─'*10}─{'─'*12}─{'─'*10}")
        
        for r_val in [0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
            idx = int((r_val - 0.5) / 1.5 * 99)
            idx = min(idx, 99)
            print(f"  {r_val:>8.1f} {unclipped_pos[idx].item():>12.2f} "
                  f"{ppo_obj_pos[idx].item():>10.2f} "
                  f"{unclipped_neg[idx].item():>12.2f} "
                  f"{ppo_obj_neg[idx].item():>10.2f}")
        
        print(f"""
  INTERPRETATION:
  • When advantage > 0 (good action):
    - Ratio > 1+ε: Objective is CAPPED (can't increase too much)
    - Prevents over-exploitation of good actions
    
  • When advantage < 0 (bad action):
    - Ratio < 1-ε: Objective is CAPPED (can't decrease too much)
    - Prevents over-correction of bad actions
    
  • This "pessimistic" bound ensures STABLE training!
""")
        
        # ─── Value Function and Advantage Estimation ───
        print("  ── Advantage Estimation (GAE) ──")
        
        # Simulate a trajectory
        T = 8  # sequence length (tokens)
        gamma = 0.99
        lam = 0.95
        
        rewards = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5])
        values = torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5])
        
        # GAE-λ advantage estimation
        advantages = torch.zeros(T)
        last_gae = 0
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = last_gae = delta + gamma * lam * last_gae
        
        returns = advantages + values
        
        print(f"\n  Token trajectory (reward at end only):")
        print(f"  {'t':>4} {'Reward':>8} {'V(s)':>8} {'Advantage':>10} {'Return':>8}")
        print(f"  {'─'*4}─{'─'*8}─{'─'*8}─{'─'*10}─{'─'*8}")
        
        for t in range(T):
            print(f"  {t:>4} {rewards[t].item():>8.2f} {values[t].item():>8.2f} "
                  f"{advantages[t].item():>10.4f} {returns[t].item():>8.4f}")
        
        print(f"""
  KEY POINTS:
  • Reward is given only at the END (complete response)
  • GAE propagates reward signal backward through the sequence
  • Advantage tells each token: "did you help or hurt?"
  • Higher λ → more bias, less variance (and vice versa)
  • In RLHF: the value head predicts expected future reward
""")
    
    @staticmethod
    def demonstrate_ppo_update():
        """Show a simplified PPO update step."""
        print("  ── Simplified PPO Update ──")
        
        torch.manual_seed(42)
        
        vocab_size = 100
        d_model = 32
        seq_len = 5
        
        # Old policy (frozen)
        old_policy = nn.Linear(d_model, vocab_size)
        
        # New policy (being trained)
        new_policy = nn.Linear(d_model, vocab_size)
        new_policy.load_state_dict(old_policy.state_dict())
        
        # Simulated hidden states and actions
        states = torch.randn(1, seq_len, d_model)
        actions = torch.randint(0, vocab_size, (1, seq_len))
        advantages = torch.tensor([[0.1, -0.2, 0.5, 0.3, 1.0]])
        
        epsilon = 0.2
        optimizer = torch.optim.Adam(new_policy.parameters(), lr=1e-3)
        
        print(f"\n  Running 10 PPO update steps (ε={epsilon}):")
        
        for step in range(10):
            # Compute log probabilities
            with torch.no_grad():
                old_logits = old_policy(states)
                old_log_probs = F.log_softmax(old_logits, dim=-1)
                old_log_probs = old_log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
            
            new_logits = new_policy(states)
            new_log_probs = F.log_softmax(new_logits, dim=-1)
            new_log_probs = new_log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
            
            # Policy ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
            ppo_loss = -torch.min(surr1, surr2).mean()
            
            optimizer.zero_grad()
            ppo_loss.backward()
            optimizer.step()
            
            if (step + 1) % 5 == 0:
                avg_ratio = ratio.mean().item()
                print(f"    Step {step+1}: loss={ppo_loss.item():.4f}, "
                      f"avg_ratio={avg_ratio:.4f}, "
                      f"max_ratio={ratio.max().item():.4f}")
        
        print(f"\n  Ratios stay near 1.0 thanks to clipping → stable updates!")


# ============================================================================
# SECTION 3: KL DIVERGENCE THEORY
# ============================================================================

class KLDivergenceTheory:
    """
    Why KL divergence penalty is critical in RLHF.
    
    THEORY:
    ───────
    Without KL penalty, the policy can "reward hack":
    - Find degenerate outputs that score high on the reward model
    - But are gibberish or adversarial to humans
    
    The KL penalty keeps the policy close to the SFT model:
    
      R_total(x, y) = R_reward(x, y) - β × KL(π_θ || π_ref)
    
    Per-token KL:
      KL = Σ_t [log π_θ(y_t|y_{<t}, x) - log π_ref(y_t|y_{<t}, x)]
    
    This is the sum of log-probability differences at each token.
    """
    
    @staticmethod
    def demonstrate():
        print("\n\n" + "=" * 65)
        print("  SECTION 3: KL DIVERGENCE THEORY")
        print("=" * 65)
        
        torch.manual_seed(42)
        
        # ─── Demonstrate KL divergence between policies ───
        print("\n  ── KL Divergence Between Policies ──")
        
        vocab_size = 10
        
        # Reference policy (SFT model)
        ref_logits = torch.randn(vocab_size)
        ref_probs = F.softmax(ref_logits, dim=-1)
        
        # Shifted policies (varying degrees of drift)
        shifts = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        print(f"\n  {'Shift':>8} {'KL(policy||ref)':>16} {'Max prob Δ':>12}")
        print(f"  {'─'*8}─{'─'*16}─{'─'*12}")
        
        for shift in shifts:
            # Shift the logits
            new_logits = ref_logits.clone()
            new_logits[0] += shift
            new_probs = F.softmax(new_logits, dim=-1)
            
            # KL divergence: Σ p(x) log(p(x)/q(x))
            kl = (new_probs * (new_probs.log() - ref_probs.log())).sum()
            max_prob_delta = (new_probs - ref_probs).abs().max()
            
            bar = "█" * min(40, int(kl.item() * 5))
            print(f"  {shift:>8.1f} {kl.item():>16.4f} {max_prob_delta.item():>12.4f} {bar}")
        
        print(f"""
  As the policy drifts further from reference:
  • KL divergence increases (penalizes more)
  • The distribution becomes more concentrated
  • Risk of reward hacking increases
""")
        
        # ─── Demonstrate reward hacking ───
        print("  ── Reward Hacking Without KL Penalty ──")
        
        # Simulate: reward model gives high score to repetitive outputs
        # Without KL penalty, the policy collapses to repetition
        
        vocab_size = 50
        seq_len = 10
        
        # Simple "reward model" — biased toward token 0
        def biased_reward(tokens):
            return (tokens == 0).float().sum(dim=-1)
        
        # Simple policy
        policy_logits = nn.Parameter(torch.zeros(vocab_size))
        optimizer = torch.optim.Adam([policy_logits], lr=0.1)
        
        # Reference distribution (uniform)
        ref_log_probs = torch.log(torch.ones(vocab_size) / vocab_size)
        
        print(f"\n  Training WITHOUT KL penalty (β=0):")
        
        for step in range(50):
            probs = F.softmax(policy_logits, dim=-1)
            tokens = torch.multinomial(probs.expand(32, -1), seq_len, replacement=True)
            reward = biased_reward(tokens).mean()
            
            # REINFORCE without KL penalty
            log_probs = F.log_softmax(policy_logits, dim=-1)
            loss = -(log_probs[0] * reward)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 10 == 0:
                top_prob = F.softmax(policy_logits.detach(), dim=-1).max().item()
                entropy = -(probs * probs.log()).sum().item()
                print(f"    Step {step+1}: reward={reward.item():.2f}, "
                      f"top_prob={top_prob:.4f}, entropy={entropy:.2f}")
        
        collapsed_probs = F.softmax(policy_logits.detach(), dim=-1)
        print(f"    → Policy COLLAPSED! Top token prob = {collapsed_probs.max().item():.4f}")
        print(f"    → Entropy: {-(collapsed_probs * collapsed_probs.log()).sum().item():.2f} "
              f"(should be ~{math.log(vocab_size):.2f} for uniform)")
        
        # Now WITH KL penalty
        print(f"\n  Training WITH KL penalty (β=0.1):")
        
        policy_logits2 = nn.Parameter(torch.zeros(vocab_size))
        optimizer2 = torch.optim.Adam([policy_logits2], lr=0.1)
        beta = 0.1
        
        for step in range(50):
            probs2 = F.softmax(policy_logits2, dim=-1)
            log_probs2 = F.log_softmax(policy_logits2, dim=-1)
            
            tokens2 = torch.multinomial(probs2.expand(32, -1), seq_len, replacement=True)
            reward2 = biased_reward(tokens2).mean()
            
            # KL divergence from reference
            kl = (probs2 * (log_probs2 - ref_log_probs)).sum()
            
            # KL-penalized training
            loss2 = -(reward2 - beta * kl)
            
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            
            if (step + 1) % 10 == 0:
                top_prob2 = F.softmax(policy_logits2.detach(), dim=-1).max().item()
                entropy2 = -(probs2 * probs2.log()).sum().item()
                print(f"    Step {step+1}: reward={reward2.item():.2f}, "
                      f"kl={kl.item():.4f}, top_prob={top_prob2:.4f}, "
                      f"entropy={entropy2:.2f}")
        
        safe_probs = F.softmax(policy_logits2.detach(), dim=-1)
        print(f"    → Policy stays DIVERSE! Top prob = {safe_probs.max().item():.4f}")
        
        print(f"""
  ═══ The β Trade-off ═══
  
  β too low (→ 0):
    • Policy reward-hacks (collapses to degenerate text)
    • High reward but garbage output
    
  β too high (→ ∞):
    • Policy barely changes from SFT model
    • Safe but no improvement from RLHF
    
  β just right (~0.01-0.2 typically):
    • Policy improves quality while staying coherent
    • This balance is critical to RLHF success!
    
  Typical values: β = 0.01 - 0.2
  Some systems use adaptive β (KL controller)
""")


# ============================================================================
# SECTION 4: REWARD MODEL THEORY
# ============================================================================

class RewardModelTheory:
    """
    Reward model architecture and training theory.
    
    ARCHITECTURE:
    ─────────────
    The reward model is typically:
    1. Initialize from the SFT model (same architecture)
    2. Remove the language model head (token prediction)
    3. Add a scalar value head (linear layer → 1 output)
    4. Train on preference data with Bradley-Terry loss
    
    ┌──────────────┐
    │ Prompt +     │
    │ Response     │ → Transformer → [CLS] or last token → Linear → Scalar
    └──────────────┘                                                  ↓
                                                                   Reward
    """
    
    @staticmethod
    def demonstrate():
        print("\n\n" + "=" * 65)
        print("  SECTION 4: REWARD MODEL THEORY")
        print("=" * 65)
        
        # ─── Reward Model Architecture ───
        print(f"""
  ═══ Reward Model Architecture ═══
  
  Standard approach:
  ┌────────────────────────────────────────────────────┐
  │  Input: [prompt] + [response]                      │
  │         ↓                                          │
  │  Transformer Encoder/Decoder (from SFT model)      │
  │         ↓                                          │
  │  Take last token hidden state h_T                  │
  │         ↓                                          │
  │  Value Head: r = W_v · h_T + b_v                   │
  │         ↓                                          │
  │  Scalar Reward: r ∈ ℝ                              │
  └────────────────────────────────────────────────────┘
  
  Why initialize from SFT model?
  • Already understands language
  • Already knows what "good" text looks like
  • Much faster convergence than training from scratch
  • Better generalization to unseen prompts
""")
        
        # ─── Demonstrate reward model training ───
        print("  ── Training Dynamics ──")
        
        torch.manual_seed(42)
        
        # Simulate reward model training
        d_model = 32
        
        # "Transformer" (simplified as linear + ReLU)
        backbone = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        value_head = nn.Linear(32, 1)
        
        # Generate preference data
        n_pairs = 200
        
        # Quality signal: chosen has positive features, rejected has negative
        chosen = torch.randn(n_pairs, d_model)
        chosen[:, :4] = torch.abs(chosen[:, :4]) + 0.3  # Positive quality signal
        
        rejected = torch.randn(n_pairs, d_model)
        rejected[:, :4] = -torch.abs(rejected[:, :4]) - 0.3  # Negative quality
        
        params = list(backbone.parameters()) + list(value_head.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)
        
        print(f"\n  {'Step':>6} {'Loss':>8} {'Acc':>8} {'Avg Gap':>10} {'Chosen R':>10}")
        print(f"  {'─'*6}─{'─'*8}─{'─'*8}─{'─'*10}─{'─'*10}")
        
        for step in range(300):
            # Forward pass
            h_chosen = backbone(chosen)
            h_rejected = backbone(rejected)
            r_chosen = value_head(h_chosen).squeeze(-1)
            r_rejected = value_head(h_rejected).squeeze(-1)
            
            # Bradley-Terry loss
            loss = -torch.log(torch.sigmoid(r_chosen - r_rejected) + 1e-10).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 60 == 0:
                acc = (r_chosen > r_rejected).float().mean()
                gap = (r_chosen - r_rejected).mean()
                avg_r = r_chosen.mean()
                print(f"  {step+1:>6} {loss.item():>8.4f} {acc.item():>7.1%} "
                      f"{gap.item():>10.3f} {avg_r.item():>10.3f}")
        
        # ─── Goodhart's Law / Overoptimization ───
        print(f"""
  ═══ Reward Model Pitfalls ═══
  
  1. GOODHART'S LAW:
     "When a measure becomes a target, it ceases to be a good measure."
     
     The reward model is an IMPERFECT proxy for human preferences.
     Optimizing too hard against it leads to reward hacking:
     
     Reward Model Score:  ↑↑↑ (very high)
     Actual Quality:      ↓↓↓ (terrible)
     
     Solution: KL penalty + reward model ensembles + periodic retraining
  
  2. DISTRIBUTION SHIFT:
     The reward model was trained on SFT outputs.
     During RL, the policy generates different outputs.
     The reward model may be unreliable on these new outputs.
     
     Solution: Constrain KL divergence, retrain reward model
  
  3. REWARD MODEL SIZE:
     - Typically same size or smaller than the policy model
     - Larger reward models → more accurate but more expensive
     - Common: policy = 7B, reward model = 3-7B
  
  4. PREFERENCE DATA QUALITY:
     - Garbage in → garbage out
     - Need diverse, high-quality annotators
     - Inter-annotator agreement matters
     - Typical agreement rate: 60-80% on subjective tasks
""")


# ============================================================================
# SECTION 5: RLHF PIPELINE THEORY
# ============================================================================

class RLHFPipelineTheory:
    """
    End-to-end RLHF pipeline analysis.
    
    THE FOUR MODELS:
    ────────────────
    1. Policy Model (π_θ): The LLM being trained
    2. Reference Model (π_ref): Frozen copy of SFT model (for KL)
    3. Reward Model (r_φ): Scores (prompt, response) pairs
    4. Value Model (V_ψ): Predicts expected future reward
    
    Memory requirement: 4× model size at minimum!
    """
    
    @staticmethod
    def demonstrate():
        print("\n\n" + "=" * 65)
        print("  SECTION 5: RLHF PIPELINE THEORY")
        print("=" * 65)
        
        print(f"""
  ═══ The Four Models in RLHF ═══
  
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  ┌──────────────┐                                      │
  │  │ Policy (π_θ) │  ← Being trained (generates text)    │
  │  │ "The student" │                                      │
  │  └──────────────┘                                      │
  │         │ generates response                            │
  │         ▼                                               │
  │  ┌──────────────┐     ┌───────────────┐                │
  │  │ Response y   │────▶│ Reward Model  │──▶ r(x,y)     │
  │  └──────────────┘     │ (r_φ) "Judge" │                │
  │         │              └───────────────┘                │
  │         │                     │                         │
  │  ┌──────────────┐            │                         │
  │  │ Value Model  │            │                         │
  │  │ (V_ψ)       │──▶ V(s)    │                         │
  │  │ "Critic"     │            │                         │
  │  └──────────────┘            │                         │
  │         │                     │                         │
  │         ▼                     ▼                         │
  │    Advantage = r(x,y) + γV(s') - V(s)                 │
  │         │                                               │
  │  ┌──────────────┐                                      │
  │  │ Reference    │                                      │
  │  │ Model (π_ref)│──▶ KL(π_θ || π_ref) penalty         │
  │  │ "Anchor"     │                                      │
  │  └──────────────┘                                      │
  │                                                         │
  │  Final reward = r(x,y) - β × KL                       │
  │  Update π_θ using PPO with this reward                 │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
""")
        
        # Memory analysis
        print("  ═══ Memory Requirements ═══")
        
        model_sizes = {
            "GPT-2 (124M)": 124e6,
            "LLaMA-7B": 7e9,
            "LLaMA-13B": 13e9,
            "LLaMA-70B": 70e9,
        }
        
        print(f"\n  {'Model':<18} {'Single':>10} {'RLHF (4×)':>12} {'+ Optimizer':>14}")
        print(f"  {'─'*18}─{'─'*10}─{'─'*12}─{'─'*14}")
        
        for name, params in model_sizes.items():
            single_gb = params * 2 / 1e9   # fp16
            rlhf_gb = single_gb * 4
            with_opt_gb = rlhf_gb + single_gb * 4  # optimizer states
            print(f"  {name:<18} {single_gb:>8.1f}GB {rlhf_gb:>10.1f}GB "
                  f"{with_opt_gb:>12.1f}GB")
        
        print(f"""
  RLHF is VERY memory intensive:
  • 4 copies of the model in memory
  • Plus optimizer states for policy + value model
  • Plus KV cache for generation
  
  Solutions:
  • Use PEFT (LoRA) for policy → share base weights
  • Quantize reference + reward model (8-bit or 4-bit)
  • Offload to CPU when not actively used
  • DeepSpeed ZeRO-3 for model parallelism
""")
        
        # Training flow
        print("  ═══ RLHF Training Loop (Pseudo-code) ═══")
        print(f"""
  for batch in training_data:
      prompts = batch["prompt"]
      
      # 1. GENERATE: Policy creates responses
      responses = policy.generate(prompts)
      
      # 2. SCORE: Reward model evaluates responses
      rewards = reward_model(prompts, responses)
      
      # 3. KL PENALTY: Compare with reference model
      policy_logprobs = policy.log_prob(responses | prompts)
      ref_logprobs = reference.log_prob(responses | prompts)
      kl_penalty = policy_logprobs - ref_logprobs
      adjusted_rewards = rewards - beta * kl_penalty
      
      # 4. ADVANTAGE: Value model estimates baselines
      values = value_model(prompts, responses)
      advantages = compute_gae(adjusted_rewards, values)
      
      # 5. PPO UPDATE: Update policy and value model
      for ppo_epoch in range(ppo_epochs):
          policy_loss = ppo_clip_loss(policy, advantages)
          value_loss = mse_loss(value_model, returns)
          
          update(policy, policy_loss)
          update(value_model, value_loss)
      
      # 6. LOG: Monitor metrics
      log(reward=rewards.mean(), kl=kl_penalty.mean())
""")
        
        # Stability considerations
        print("  ═══ Training Stability Tips ═══")
        print(f"""
  1. Reward normalization: Normalize rewards (mean=0, std=1)
  2. Advantage normalization: Normalize advantages per batch
  3. Learning rate: Very low (1e-6 to 5e-6 for policy)
  4. KL target: Use adaptive β to maintain target KL ≈ 6-10
  5. Mini-batch PPO: Multiple PPO epochs (2-4) per batch
  6. Gradient clipping: Max norm = 1.0
  7. Early stopping: Monitor reward AND KL divergence
  8. Reward baseline: Subtract mean reward (reduces variance)
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all theory sections."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║    RLHF THEORY — LEARNING FROM HUMAN PREFERENCES             ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: Bradley-Terry
    BradleyTerryModel.demonstrate()
    BradleyTerryModel.demonstrate_preference_data()
    
    # Section 2: PPO
    PPOTheory.demonstrate()
    PPOTheory.demonstrate_ppo_update()
    
    # Section 3: KL divergence
    KLDivergenceTheory.demonstrate()
    
    # Section 4: Reward model
    RewardModelTheory.demonstrate()
    
    # Section 5: Pipeline
    RLHFPipelineTheory.demonstrate()
    
    print("\n" + "=" * 65)
    print("  THEORY MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered:
    ✓ Bradley-Terry preference model (pairwise learning)
    ✓ PPO theory (clipping, advantage estimation, value functions)
    ✓ KL divergence (reward hacking, β trade-off)
    ✓ Reward model architecture (Goodhart's Law, pitfalls)
    ✓ Complete RLHF pipeline (4 models, memory, stability)
    """)


if __name__ == "__main__":
    main()
