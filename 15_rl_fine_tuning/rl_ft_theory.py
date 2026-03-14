"""
RL Fine-Tuning Theory — RL Fundamentals for Language Models
============================================================

Deep dive into the theory behind RL-based LLM fine-tuning:

1. PolicyGradientForLLMs
   - From RL basics to language model policy gradients
   - The REINFORCE estimator for text generation

2. VarianceReduction
   - Why raw policy gradients are high-variance
   - Baselines, advantages, and normalization techniques

3. GRPOTheory
   - Group Relative Policy Optimization (DeepSeek)
   - Eliminating the value model with group statistics

4. RewardDesign
   - Designing reward functions for language tasks
   - Verifiable vs learned vs heuristic rewards

5. ExplorationVsExploitation
   - Temperature, sampling strategies, entropy bonuses
   - Balancing quality with diversity in generation

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple


# ============================================================================
# SECTION 1: POLICY GRADIENT FOR LLMs
# ============================================================================

def policy_gradient_for_llms():
    """From RL basics to language model policy gradients."""
    print("=" * 70)
    print("  SECTION 1: POLICY GRADIENT FOR LLMs")
    print("=" * 70)
    
    print(f"""
  ═══ LLM as an RL Agent ═══
  
  In RL fine-tuning, the LLM is treated as a policy:
  
    • State (s):   The prompt x + tokens generated so far y_<t
    • Action (a):  Next token y_t ∈ vocabulary V
    • Policy π_θ:  P(y_t | x, y_<t) — the LLM itself
    • Reward R:    Score for the complete response
    • Episode:     One full generation (prompt → response)
  
  
  ═══ The Policy Gradient Theorem ═══
  
  We want to maximize expected reward:
  
    J(θ) = E_{{y~π_θ(·|x)}} [ R(x, y) ]
  
  The gradient (REINFORCE / Williams 1992):
  
    ∇J(θ) = E [ R(x,y) · ∇log π_θ(y|x) ]
  
  For autoregressive LLMs, log π_θ(y|x) decomposes:
  
    log π_θ(y|x) = Σ_t log π_θ(y_t | x, y_<t)
  
  So the gradient becomes:
  
    ∇J(θ) = E [ R(x,y) · Σ_t ∇log π_θ(y_t | x, y_<t) ]
  
  INTERPRETATION:
  • If R(x,y) > 0: Increase probability of ALL tokens in y
  • If R(x,y) < 0: Decrease probability of ALL tokens in y
  • Same reward for ALL tokens (credit assignment problem)
""")
    
    torch.manual_seed(42)
    
    # Demonstrate REINFORCE gradient
    vocab_size = 20
    d_model = 16
    
    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.rnn = nn.GRU(d_model, d_model, batch_first=True)
            self.head = nn.Linear(d_model, vocab_size)
        
        def forward(self, x):
            h, _ = self.rnn(self.embed(x))
            return self.head(h)
        
        def get_log_prob(self, x):
            """Log prob of sequence under the policy."""
            logits = self(x)
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            token_lp = log_probs.gather(2, x[:, 1:].unsqueeze(-1)).squeeze(-1)
            return token_lp.sum(dim=-1)
        
        def generate(self, prompt, n_tokens=6):
            ids = prompt.clone()
            for _ in range(n_tokens):
                logits = self(ids)[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_t = torch.multinomial(probs, 1)
                ids = torch.cat([ids, next_t], dim=-1)
            return ids
    
    policy = TinyPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    # Reward: unique tokens ratio
    def reward_fn(tokens):
        return len(set(tokens.tolist())) / len(tokens)
    
    print(f"\n  ── REINFORCE Training Demo ──")
    print(f"  Reward = unique_tokens / total_tokens (diversity reward)")
    print(f"\n  {'Step':>6} │ {'Avg Reward':>10} │ {'Avg LogP':>10} │ {'Loss':>10}")
    print(f"  {'─'*6}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}")
    
    for step in range(25):
        batch_size = 16
        prompts = torch.randint(0, vocab_size, (batch_size, 2))
        
        # Generate
        policy.eval()
        with torch.no_grad():
            generated = policy.generate(prompts, n_tokens=6)
        policy.train()
        
        # Score
        rewards = torch.tensor([reward_fn(g[2:]) for g in generated])
        
        # REINFORCE: loss = -R * log π(y|x)
        log_probs = policy.get_log_prob(generated)
        loss = -(rewards * log_probs).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        
        if (step + 1) % 5 == 0:
            print(f"  {step+1:>6} │ {rewards.mean():>10.4f} │ "
                  f"{log_probs.mean().item():>10.2f} │ {loss.item():>10.4f}")
    
    del policy
    
    print(f"""
  KEY TAKEAWAY: REINFORCE is simple but noisy.
  The reward is the SAME for every token — no per-token credit.
  Need variance reduction for practical training.
""")


# ============================================================================
# SECTION 2: VARIANCE REDUCTION
# ============================================================================

def variance_reduction():
    """Why raw policy gradients are high-variance and how to fix it."""
    print("\n\n" + "=" * 70)
    print("  SECTION 2: VARIANCE REDUCTION")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ The Variance Problem ═══
  
  Raw REINFORCE gradient: ∇J = E[ R(y) · ∇log π(y) ]
  
  Problem: If R(y) = 5 for all samples, gradient is ~5× ∇log π(y)
  which pushes up ALL actions even if some are better than others.
  
  High variance → noisy gradients → slow/unstable training
  
  
  ═══ Solution 1: Baseline Subtraction ═══
  
  ∇J = E[ (R(y) - b) · ∇log π(y) ]
  
  Any constant b doesn't change the expectation (math proof),
  but choosing b ≈ E[R] dramatically reduces variance!
  
  • b = mean reward in batch (simplest)
  • b = running average of rewards
  • b = learned value function V(s) (actor-critic)
""")
    
    # Demonstrate variance reduction
    n_samples = 1000
    
    # Simulate rewards and gradients
    rewards = torch.randn(n_samples) * 2 + 5  # Mean=5, Std=2
    fake_grads = torch.randn(n_samples)  # Proxy for ∇log π
    
    # No baseline
    raw_signal = rewards * fake_grads
    
    # Mean baseline
    baseline = rewards.mean()
    baselined_signal = (rewards - baseline) * fake_grads
    
    # Whitened (normalize to mean=0, std=1)
    normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    whitened_signal = normalized_rewards * fake_grads
    
    print(f"\n  ── Gradient Signal Variance ──\n")
    print(f"  {'Method':>20} │ {'Mean':>8} │ {'Std':>8} │ {'Variance Ratio':>14}")
    print(f"  {'─'*20}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*14}")
    
    raw_var = raw_signal.var().item()
    for name, signal in [("No baseline", raw_signal),
                          ("Mean baseline", baselined_signal),
                          ("Whitened", whitened_signal)]:
        print(f"  {name:>20} │ {signal.mean():>8.3f} │ {signal.std():>8.3f} │ "
              f"{signal.var().item()/raw_var:>13.2%}")
    
    print(f"""
  ═══ Solution 2: Advantage Estimation ═══
  
  Instead of R(y), use advantages A(y) = R(y) - V(x):
  
  • V(x) = expected reward from state x (value function)
  • A(y) > 0: response y is BETTER than average
  • A(y) < 0: response y is WORSE than average
  
  Used in PPO (Concept 13) with GAE-λ.
  
  
  ═══ Solution 3: Group Normalization (GRPO) ═══
  
  Generate G responses per prompt, normalize WITHIN group:
  
    A_i = (R_i - mean(R_1...R_G)) / std(R_1...R_G)
  
  No learned value function needed!
  This is the key insight behind GRPO (DeepSeek).
  
  
  ═══ Solution 4: Leave-One-Out (RLOO) ═══
  
  For each sample i, use the average of OTHER samples as baseline:
  
    b_i = (1/(G-1)) · Σ_{{j≠i}} R_j
    A_i = R_i - b_i
  
  Unbiased, lower variance than mean baseline.
""")
    
    # Demonstrate RLOO baseline
    print(f"  ── RLOO vs Mean Baseline ──\n")
    
    G = 8  # Group size
    group_rewards = torch.tensor([2.0, 3.0, 1.5, 4.0, 2.5, 3.5, 1.0, 5.0])
    
    # Mean baseline
    mean_baseline = group_rewards.mean()
    mean_advantages = group_rewards - mean_baseline
    
    # RLOO baseline (leave-one-out)
    rloo_baselines = []
    for i in range(G):
        others = torch.cat([group_rewards[:i], group_rewards[i+1:]])
        rloo_baselines.append(others.mean())
    rloo_baselines = torch.tensor(rloo_baselines)
    rloo_advantages = group_rewards - rloo_baselines
    
    print(f"  {'Sample':>8} │ {'Reward':>8} │ {'Mean BL':>8} │ {'Mean Adv':>8} │ "
          f"{'RLOO BL':>8} │ {'RLOO Adv':>8}")
    print(f"  {'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}")
    
    for i in range(G):
        print(f"  {i+1:>8} │ {group_rewards[i]:>8.2f} │ {mean_baseline:>8.2f} │ "
              f"{mean_advantages[i]:>8.2f} │ {rloo_baselines[i]:>8.2f} │ "
              f"{rloo_advantages[i]:>8.2f}")


# ============================================================================
# SECTION 3: GRPO THEORY
# ============================================================================

def grpo_theory():
    """Group Relative Policy Optimization (DeepSeek)."""
    print("\n\n" + "=" * 70)
    print("  SECTION 3: GRPO — GROUP RELATIVE POLICY OPTIMIZATION")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ GRPO: PPO Without a Value Model ═══
  
  Key insight from DeepSeek: You don't need a value model if you
  generate MULTIPLE responses per prompt and normalize rewards
  within each group.
  
  
  PPO (4 models):                 GRPO (3 models):
  ┌────────┐ ┌────────┐          ┌────────┐ ┌────────┐
  │ Policy │ │  Ref   │          │ Policy │ │  Ref   │
  └────────┘ └────────┘          └────────┘ └────────┘
  ┌────────┐ ┌────────┐          ┌────────┐
  │ Reward │ │ Value  │          │ Reward │  ← No value model!
  └────────┘ └────────┘          └────────┘
  
  
  ═══ GRPO Algorithm ═══
  
  For each prompt x:
    1. Generate G responses: y_1, ..., y_G ~ π_old(·|x)
    2. Score each: r_1, ..., r_G = R(x, y_i)
    3. Compute group advantages:
       A_i = (r_i - mean(r)) / (std(r) + ε)
    4. PPO-style clipped update with these advantages
    5. Add KL penalty: - β · KL(π_θ || π_ref)
  
  
  ═══ GRPO Loss ═══
  
  L = -(1/G) Σ_i [ min(ρ_i·A_i, clip(ρ_i, 1±ε)·A_i) ]
      + β · KL(π_θ || π_ref)
  
  Where ρ_i = π_θ(y_i|x) / π_old(y_i|x)
""")
    
    # Demonstrate GRPO advantage computation
    print(f"  ── GRPO Advantage Computation ──\n")
    
    G = 6  # Group size
    
    # Simulate: one prompt, G responses with different rewards
    rewards = torch.tensor([0.2, 0.8, 0.1, 0.5, 0.9, 0.3])
    
    # GRPO advantages
    mean_r = rewards.mean()
    std_r = rewards.std()
    advantages = (rewards - mean_r) / (std_r + 1e-8)
    
    print(f"  Group of {G} responses for one prompt:")
    print(f"  Mean reward: {mean_r:.3f}, Std: {std_r:.3f}")
    print(f"\n  {'Response':>10} │ {'Reward':>8} │ {'Advantage':>10} │ {'Update':>15}")
    print(f"  {'─'*10}─┼─{'─'*8}─┼─{'─'*10}─┼─{'─'*15}")
    
    for i in range(G):
        if advantages[i] > 0.3:
            update = "↑↑ Strong push up"
        elif advantages[i] > 0:
            update = "↑ Push up"
        elif advantages[i] > -0.3:
            update = "↓ Push down"
        else:
            update = "↓↓ Strong push down"
        
        print(f"  {i+1:>10} │ {rewards[i]:>8.3f} │ {advantages[i]:>10.3f} │ {update:>15}")
    
    # Training simulation
    print(f"\n\n  ── GRPO Training Simulation ──\n")
    
    vocab_size = 20
    d_model = 16
    
    class MiniPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.rnn = nn.GRU(d_model, d_model, batch_first=True)
            self.head = nn.Linear(d_model, vocab_size)
        
        def forward(self, x):
            h, _ = self.rnn(self.embed(x))
            return self.head(h)
        
        def get_token_logprobs(self, x):
            logits = self(x)
            lp = F.log_softmax(logits[:, :-1, :], dim=-1)
            return lp.gather(2, x[:, 1:].unsqueeze(-1)).squeeze(-1)
        
        def generate(self, prompt, n_tokens=4):
            ids = prompt.clone()
            for _ in range(n_tokens):
                logits = self(ids)[:, -1, :]
                next_t = torch.multinomial(F.softmax(logits, dim=-1), 1)
                ids = torch.cat([ids, next_t], dim=-1)
            return ids
    
    policy = MiniPolicy()
    ref_policy = MiniPolicy()
    ref_policy.load_state_dict(policy.state_dict())
    for p in ref_policy.parameters():
        p.requires_grad = False
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    epsilon = 0.2  # Clip range
    beta = 0.05    # KL coefficient
    group_size = 6
    
    def diversity_reward(tokens):
        return len(set(tokens.tolist())) / len(tokens)
    
    print(f"  Config: G={group_size}, ε={epsilon}, β={beta}")
    print(f"\n  {'Step':>6} │ {'Avg Reward':>10} │ {'Adv Std':>8} │ {'KL':>8}")
    print(f"  {'─'*6}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*8}")
    
    for step in range(20):
        n_prompts = 8
        prompts = torch.randint(0, vocab_size, (n_prompts, 2))
        
        step_rewards = []
        step_adv_std = []
        step_kl = []
        
        for p_idx in range(n_prompts):
            prompt = prompts[p_idx:p_idx+1].expand(group_size, -1)
            
            # Generate G responses
            policy.eval()
            with torch.no_grad():
                responses = policy.generate(prompt, n_tokens=5)
            policy.train()
            
            # Score
            rewards = torch.tensor([diversity_reward(r[2:]) for r in responses])
            
            # Group normalization
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            # Per-token log-probs
            old_lp = policy.get_token_logprobs(responses).detach()
            new_lp = policy.get_token_logprobs(responses)
            ref_lp = ref_policy.get_token_logprobs(responses)
            
            # Ratios per token
            ratio = torch.exp(new_lp - old_lp)
            
            # Expand advantages to per-token
            adv_expanded = adv.unsqueeze(1).expand_as(ratio)
            
            # Clipped surrogate
            surr1 = ratio * adv_expanded
            surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * adv_expanded
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # KL penalty
            kl = (new_lp - ref_lp).mean()
            
            loss = policy_loss + beta * kl
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            step_rewards.extend(rewards.tolist())
            step_adv_std.append(adv.std().item())
            step_kl.append(kl.item())
        
        if (step + 1) % 4 == 0:
            print(f"  {step+1:>6} │ {sum(step_rewards)/len(step_rewards):>10.4f} │ "
                  f"{sum(step_adv_std)/len(step_adv_std):>8.4f} │ "
                  f"{sum(step_kl)/len(step_kl):>8.4f}")
    
    del policy, ref_policy
    
    print(f"""
  GRPO ADVANTAGES:
  ✓ No value model → saves ~1× model memory
  ✓ Advantages from group statistics → no learned baseline
  ✓ Same clipping as PPO → stable updates
  ✓ Works especially well with verifiable rewards
""")


# ============================================================================
# SECTION 4: REWARD DESIGN
# ============================================================================

def reward_design():
    """Designing reward functions for language tasks."""
    print("\n\n" + "=" * 70)
    print("  SECTION 4: REWARD DESIGN FOR LANGUAGE TASKS")
    print("=" * 70)
    
    print(f"""
  ═══ Types of Reward Functions ═══
  
  ┌────────────────┬──────────────────────────────────────────┐
  │ Type           │ Examples                                 │
  ├────────────────┼──────────────────────────────────────────┤
  │ VERIFIABLE     │ Math: check final answer matches         │
  │                │ Code: run unit tests, pass/fail          │
  │                │ Logic: verify proof steps                │
  │                │ Factual: check against knowledge base    │
  ├────────────────┼──────────────────────────────────────────┤
  │ LEARNED        │ Reward model trained on preferences      │
  │ (Concept 13)   │ Classifier for quality/safety            │
  │                │ LLM-as-judge scoring                     │
  ├────────────────┼──────────────────────────────────────────┤
  │ HEURISTIC      │ Length penalty/reward                    │
  │                │ Format checking (JSON, code blocks)      │
  │                │ Repetition penalty                       │
  │                │ Keyword presence/absence                 │
  ├────────────────┼──────────────────────────────────────────┤
  │ METRIC-BASED   │ BLEU, ROUGE for translation/summary     │
  │                │ Exact match for QA                       │
  │                │ F1 score for extraction                  │
  ├────────────────┼──────────────────────────────────────────┤
  │ COMPOSITE      │ Weighted sum of multiple signals         │
  │                │ R = w₁·correctness + w₂·format + w₃·len │
  └────────────────┴──────────────────────────────────────────┘
""")
    
    # Demonstrate various reward functions
    print(f"  ── Example Reward Functions ──\n")
    
    # 1. Math verification reward
    def math_reward(response: str, correct_answer: float) -> float:
        """Reward for math problems — binary correctness."""
        import re
        # Extract the last number in the response
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if not numbers:
            return 0.0
        try:
            predicted = float(numbers[-1])
            return 1.0 if abs(predicted - correct_answer) < 0.01 else 0.0
        except ValueError:
            return 0.0
    
    math_examples = [
        ("The answer is 42.", 42.0),
        ("Let me calculate... 2+2 = 5.", 4.0),
        ("After careful computation, I get 3.14159.", 3.14159),
        ("I'm not sure about this.", 7.0),
    ]
    
    print(f"  1. Math Verification Reward:")
    for resp, correct in math_examples:
        r = math_reward(resp, correct)
        print(f"     r={r:.0f}  \"{resp}\" (correct={correct})")
    
    # 2. Format reward
    def format_reward(response: str) -> float:
        """Reward for following a specific format."""
        score = 0.0
        # Check for structured format
        if response.strip().startswith("Answer:"):
            score += 0.3
        if "\n" in response:
            score += 0.2  # Multi-line
        if any(c in response for c in "123456789"):
            score += 0.2  # Contains numbers
        if response.strip().endswith("."):
            score += 0.3  # Ends with period
        return score
    
    format_examples = [
        "Answer: The result is 42.\nThis is because 6 × 7 = 42.",
        "idk maybe 42",
        "Answer: 42.",
    ]
    
    print(f"\n  2. Format Reward:")
    for resp in format_examples:
        r = format_reward(resp)
        print(f"     r={r:.1f}  \"{resp[:50]}\"")
    
    # 3. Composite reward
    def composite_reward(response: str, correct_answer: float,
                         weights: Dict = None) -> float:
        """Weighted combination of reward signals."""
        if weights is None:
            weights = {"correctness": 0.6, "format": 0.3, "length": 0.1}
        
        r_correct = math_reward(response, correct_answer)
        r_format = format_reward(response)
        
        # Length reward: prefer 20-100 chars
        resp_len = len(response)
        r_length = 1.0 if 20 <= resp_len <= 100 else 0.5 if resp_len < 20 else 0.3
        
        total = (weights["correctness"] * r_correct +
                 weights["format"] * r_format +
                 weights["length"] * r_length)
        return total
    
    print(f"\n  3. Composite Reward (0.6·correct + 0.3·format + 0.1·length):")
    print(f"     r={composite_reward('Answer: 42.', 42.0):.2f}  \"Answer: 42.\" (correct, formatted)")
    print(f"     r={composite_reward('42', 42.0):.2f}  \"42\" (correct, no format)")
    print(f"     r={composite_reward('Answer: 43.', 42.0):.2f}  \"Answer: 43.\" (wrong, formatted)")
    
    print(f"""
  ═══ Reward Design Best Practices ═══
  
  1. START SIMPLE: Binary correct/incorrect is often enough
  2. AVOID GAMING: Test for unintended shortcuts
     (e.g., model learns to always output "42")
  3. NORMALIZE: Keep rewards in consistent range [0, 1] or [-1, 1]
  4. COMBINE CAREFULLY: Weighted sum can have unexpected interactions
  5. VERIFY: Test your reward function on edge cases BEFORE training
  
  
  ═══ Verifiable Rewards (RLVR) — The Future ═══
  
  DeepSeek-R1 showed that simple binary rewards on verifiable
  tasks can teach sophisticated REASONING:
  
    Math: R = 1 if answer matches, 0 otherwise
    Code: R = 1 if all tests pass, 0 otherwise
  
  No reward model, no human preferences — just correctness!
  
  This works because:
  • Perfect signal (no noise in ground truth)
  • Forces model to develop reasoning chains
  • Group sampling (GRPO) provides contrast
""")


# ============================================================================
# SECTION 5: EXPLORATION VS EXPLOITATION
# ============================================================================

def exploration_vs_exploitation():
    """Balancing quality with diversity in generation."""
    print("\n\n" + "=" * 70)
    print("  SECTION 5: EXPLORATION VS EXPLOITATION")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ The Exploration-Exploitation Dilemma ═══
  
  EXPLOITATION: Generate the most likely response
    → High quality but no diversity
    → Can't discover better strategies
    → Risk: gets stuck in local optima
  
  EXPLORATION: Sample diverse responses
    → Lower average quality
    → May find novel, better approaches
    → Risk: wastes compute on bad responses
  
  In RL fine-tuning, we need BOTH:
  • Explore to find good reasoning strategies
  • Exploit to refine known-good strategies
""")
    
    # Demonstrate temperature effect
    print(f"  ── Temperature and Sampling ──\n")
    
    # Simulate logits for next token
    logits = torch.tensor([3.0, 1.5, 0.5, 0.1, -0.5, -1.0, -2.0])
    vocab = ["the", "a", "in", "on", "my", "but", "if"]
    
    temperatures = [0.3, 0.7, 1.0, 1.5, 2.0]
    
    print(f"  Raw logits: {logits.tolist()}")
    print(f"\n  {'Temp':>6} │ ", end="")
    for w in vocab:
        print(f"{w:>8}", end="")
    print(f" │ {'Entropy':>8}")
    print(f"  {'─'*6}─┼─{'─'*56}─┼─{'─'*8}")
    
    for T in temperatures:
        probs = F.softmax(logits / T, dim=-1)
        entropy = -(probs * probs.log()).sum()
        
        row = f"  {T:>6.1f} │ "
        for p in probs:
            row += f"{p.item():>8.3f}"
        row += f" │ {entropy.item():>8.3f}"
        print(row)
    
    print(f"""
  INTERPRETATION:
  • T=0.3 (low):  Almost deterministic — always picks "the"
  • T=1.0 (std):  Original distribution — moderate diversity
  • T=2.0 (high): Near-uniform — very exploratory
  
  
  ═══ Sampling Strategies for RL Training ═══
  
  1. TEMPERATURE SCALING:
     • Training: T=0.8-1.2 (enough diversity for GRPO groups)
     • Inference: T=0.0-0.6 (exploit learned policy)
  
  2. TOP-K / TOP-P (NUCLEUS):
     • Top-k=50: Only sample from top 50 tokens
     • Top-p=0.9: Sample from tokens covering 90% probability mass
     • Prevents sampling very unlikely tokens
  
  3. ENTROPY BONUS:
     • Add H(π) term to reward: R' = R + α·H(π)
     • Encourages diversity in generation
     • Prevents premature convergence
     • α typically 0.001-0.01
  
  4. GROUP SIZE (for GRPO):
     • Larger G → more diversity in group
     • G=4: Fast but limited exploration
     • G=16: Good exploration, more compute
     • G=64: Very thorough, expensive
  
  
  ═══ Practical Schedule ═══
  
  Phase 1 (Exploration): High temp, large groups
    T=1.0, G=16, entropy bonus, diverse prompts
  
  Phase 2 (Refinement): Medium temp, medium groups
    T=0.8, G=8, small entropy bonus
  
  Phase 3 (Exploitation): Low temp, smaller groups
    T=0.6, G=4, no entropy bonus
""")
    
    # Demonstrate entropy bonus effect
    print(f"  ── Entropy Bonus Demo ──\n")
    
    # Policy with and without entropy bonus
    base_rewards = torch.tensor([0.8, 0.2, 0.5, 0.9, 0.3])
    policies_entropy = torch.tensor([0.5, 1.2, 0.8, 0.3, 1.5])  # H(π) for each
    
    alpha_values = [0.0, 0.01, 0.05, 0.1]
    
    print(f"  {'α':>6} │ {'Modified Rewards':>50} │ {'Best':>5}")
    print(f"  {'─'*6}─┼─{'─'*50}─┼─{'─'*5}")
    
    for alpha in alpha_values:
        modified = base_rewards + alpha * policies_entropy
        best_idx = modified.argmax().item()
        rewards_str = "  ".join(f"{r:.3f}" for r in modified)
        print(f"  {alpha:>6.2f} │ {rewards_str:>50} │ {best_idx+1:>5}")
    
    print(f"""
  With α=0: Best response is #4 (highest raw reward)
  With α>0: High-entropy responses get a bonus, can change ranking
  → Prevents mode collapse where policy always gives same answer
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all RL fine-tuning theory sections."""
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║    RL FINE-TUNING THEORY — RL FUNDAMENTALS FOR LLMs               ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    # Section 1: Policy gradient
    policy_gradient_for_llms()
    
    # Section 2: Variance reduction
    variance_reduction()
    
    # Section 3: GRPO
    grpo_theory()
    
    # Section 4: Reward design
    reward_design()
    
    # Section 5: Exploration vs exploitation
    exploration_vs_exploitation()
    
    print("\n" + "=" * 70)
    print("  RL FINE-TUNING THEORY MODULE COMPLETE")
    print("=" * 70)
    print("""
    Covered:
    ✓ Policy gradient for LLMs (REINFORCE, log-prob decomposition)
    ✓ Variance reduction (baselines, advantages, RLOO, whitening)
    ✓ GRPO theory (group normalization, no value model)
    ✓ Reward design (verifiable, heuristic, composite, RLVR)
    ✓ Exploration vs exploitation (temperature, entropy bonus, scheduling)
    """)


if __name__ == "__main__":
    main()
