"""
RL Fine-Tuning From Scratch — REINFORCE, GRPO, Expert Iteration
================================================================

Implementations from scratch for deep understanding:

1. REINFORCE with baseline for text generation
2. GRPO (Group Relative Policy Optimization) — full implementation
3. Expert Iteration / ReST (generate → filter → SFT)
4. STaR (Self-Taught Reasoner) with rationale generation
5. RLOO (REINFORCE Leave-One-Out) baseline computation

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import re
from typing import Dict, List, Tuple, Optional


# ============================================================================
# SHARED: Mini Language Model
# ============================================================================

class MiniLanguageModel(nn.Module):
    """
    Small transformer-style LM for RL demonstrations.
    Uses GRU backbone for simplicity (RL concepts are model-agnostic).
    """
    
    def __init__(self, vocab_size: int = 50, d_model: int = 32,
                 n_layers: int = 2, pad_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model, num_layers=n_layers,
                          batch_first=True, dropout=0.1)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits [B, T, V]."""
        h, _ = self.rnn(self.embed(x))
        return self.head(h)
    
    def get_token_log_probs(self, sequences: torch.Tensor,
                            prompt_len: int = 2) -> torch.Tensor:
        """
        Compute log probs for response tokens (after prompt).
        Returns: [B, T_response] log probabilities
        """
        logits = self(sequences)
        # Shift: logits for position t predict token at t+1
        log_probs = F.log_softmax(logits[:, prompt_len-1:-1, :], dim=-1)
        targets = sequences[:, prompt_len:]
        token_lp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        return token_lp
    
    def get_sequence_log_prob(self, sequences: torch.Tensor,
                               prompt_len: int = 2) -> torch.Tensor:
        """Sum of token log probs for the response part."""
        return self.get_token_log_probs(sequences, prompt_len).sum(dim=-1)
    
    @torch.no_grad()
    def generate(self, prompts: torch.Tensor, max_new_tokens: int = 8,
                 temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
        """Autoregressive generation with sampling."""
        self.eval()
        ids = prompts.clone()
        
        for _ in range(max_new_tokens):
            logits = self(ids)[:, -1, :] / temperature
            
            if top_k > 0:
                topk_vals, _ = logits.topk(top_k, dim=-1)
                logits[logits < topk_vals[:, -1:]] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_t], dim=-1)
        
        self.train()
        return ids


# ============================================================================
# SECTION 1: REINFORCE WITH BASELINE
# ============================================================================

def reinforce_with_baseline():
    """REINFORCE algorithm with learned baseline for text generation."""
    print("=" * 70)
    print("  SECTION 1: REINFORCE WITH BASELINE")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  REINFORCE with baseline:
    ∇J(θ) = E[ (R(y) - b) · ∇log π_θ(y|x) ]
  
  We use a learned baseline V(x) ≈ E[R(y)|x] to reduce variance.
  The baseline is trained to predict expected reward.
""")
    
    vocab_size = 30
    prompt_len = 2
    gen_len = 6
    
    # Policy
    policy = MiniLanguageModel(vocab_size=vocab_size, d_model=32)
    
    # Learned baseline (value function)
    class ValueBaseline(nn.Module):
        """Predicts expected reward given a prompt."""
        def __init__(self, vocab_size, d_model):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.rnn = nn.GRU(d_model, d_model, batch_first=True)
            self.head = nn.Linear(d_model, 1)
        
        def forward(self, prompt):
            h, _ = self.rnn(self.embed(prompt))
            return self.head(h[:, -1, :]).squeeze(-1)
    
    baseline = ValueBaseline(vocab_size, 32)
    
    policy_opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    baseline_opt = torch.optim.Adam(baseline.parameters(), lr=3e-3)
    
    # Reward: diversity + no-repeat
    def compute_reward(tokens: torch.Tensor) -> float:
        t = tokens.tolist()
        unique_ratio = len(set(t)) / len(t)
        # Penalty for consecutive repeats
        repeats = sum(1 for i in range(1, len(t)) if t[i] == t[i-1])
        return unique_ratio - 0.3 * repeats / len(t)
    
    # Track metrics
    print(f"\n  ── Training REINFORCE + learned baseline ──")
    print(f"  Reward: diversity - repetition penalty")
    print(f"\n  {'Step':>6} │ {'Avg R':>7} │ {'Baseline MSE':>12} │ "
          f"{'Policy Loss':>11} │ {'Var(Adv)':>9}")
    print(f"  {'─'*6}─┼─{'─'*7}─┼─{'─'*12}─┼─{'─'*11}─┼─{'─'*9}")
    
    running_rewards = []
    
    for step in range(50):
        batch_size = 32
        prompts = torch.randint(0, vocab_size, (batch_size, prompt_len))
        
        # Generate responses
        with torch.no_grad():
            sequences = policy.generate(prompts, max_new_tokens=gen_len)
        
        # Score responses
        rewards = torch.tensor([
            compute_reward(seq[prompt_len:]) for seq in sequences
        ])
        
        # Update baseline (value function)
        predicted_values = baseline(prompts)
        baseline_loss = F.mse_loss(predicted_values, rewards)
        
        baseline_opt.zero_grad()
        baseline_loss.backward()
        baseline_opt.step()
        
        # Compute advantages: A = R - V(x)
        with torch.no_grad():
            values = baseline(prompts)
        advantages = rewards - values
        
        # REINFORCE update
        log_probs = policy.get_sequence_log_prob(sequences, prompt_len)
        policy_loss = -(advantages * log_probs).mean()
        
        policy_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        policy_opt.step()
        
        running_rewards.append(rewards.mean().item())
        
        if (step + 1) % 10 == 0:
            print(f"  {step+1:>6} │ {rewards.mean():>7.3f} │ "
                  f"{baseline_loss.item():>12.4f} │ "
                  f"{policy_loss.item():>11.4f} │ "
                  f"{advantages.var().item():>9.4f}")
    
    # Compare variance: with vs without baseline
    prompts = torch.randint(0, vocab_size, (64, prompt_len))
    with torch.no_grad():
        sequences = policy.generate(prompts, max_new_tokens=gen_len)
        rewards = torch.tensor([compute_reward(seq[prompt_len:]) for seq in sequences])
        values = baseline(prompts)
    
    raw_signal = rewards
    baselined_signal = rewards - values
    
    print(f"\n  ── Variance Comparison ──")
    print(f"  Var(R):       {raw_signal.var().item():.4f}")
    print(f"  Var(R - V):   {baselined_signal.var().item():.4f}")
    print(f"  Reduction:    {1 - baselined_signal.var().item()/raw_signal.var().item():.1%}")
    
    del policy, baseline
    print(f"\n  ✓ Learned baseline reduces gradient variance significantly")


# ============================================================================
# SECTION 2: GRPO — GROUP RELATIVE POLICY OPTIMIZATION
# ============================================================================

def grpo_from_scratch():
    """Full GRPO implementation following DeepSeek's approach."""
    print("\n\n" + "=" * 70)
    print("  SECTION 2: GRPO FROM SCRATCH")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  GRPO Algorithm (DeepSeek):
  
  1. For each prompt x, generate G responses from π_old
  2. Score each response: r_1, ..., r_G
  3. Normalize within group: A_i = (r_i - μ) / (σ + ε)
  4. Compute token-level ratio: ρ_t = π_θ(y_t) / π_old(y_t)
  5. Clipped surrogate loss: L = min(ρ·A, clip(ρ)·A)
  6. Add per-token KL penalty against π_ref
  7. Update policy θ
""")
    
    # Configuration
    vocab_size = 40
    prompt_len = 3
    gen_len = 8
    group_size = 8        # G: responses per prompt
    n_prompts = 6         # Prompts per batch
    clip_eps = 0.2        # PPO clipping
    kl_coeff = 0.04       # KL penalty coefficient
    n_steps = 30
    
    # Initialize models
    policy = MiniLanguageModel(vocab_size=vocab_size, d_model=48)
    ref_policy = copy.deepcopy(policy)
    for p in ref_policy.parameters():
        p.requires_grad = False
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)
    
    # Reward function: unique tokens + ascending pattern bonus
    def grpo_reward(tokens: torch.Tensor) -> float:
        t = tokens.tolist()
        # Diversity component
        diversity = len(set(t)) / len(t)
        # Ascending pattern bonus
        ascending = sum(1 for i in range(1, len(t)) if t[i] > t[i-1]) / (len(t) - 1)
        return 0.5 * diversity + 0.5 * ascending
    
    def grpo_step(prompts: torch.Tensor) -> Dict:
        """One GRPO optimization step."""
        B = prompts.shape[0]
        G = group_size
        
        # ── Step 1: Generate G responses per prompt ──
        expanded_prompts = prompts.unsqueeze(1).expand(B, G, -1).reshape(B*G, -1)
        
        with torch.no_grad():
            sequences = policy.generate(expanded_prompts, max_new_tokens=gen_len,
                                        temperature=0.9)
        
        # ── Step 2: Score each response ──
        rewards_flat = torch.tensor([
            grpo_reward(seq[prompt_len:]) for seq in sequences
        ])
        rewards = rewards_flat.view(B, G)  # [B, G]
        
        # ── Step 3: Group normalization ──
        group_mean = rewards.mean(dim=1, keepdim=True)   # [B, 1]
        group_std = rewards.std(dim=1, keepdim=True)      # [B, 1]
        advantages = (rewards - group_mean) / (group_std + 1e-8)
        advantages_flat = advantages.view(B * G)           # [B*G]
        
        # ── Step 4: Token-level ratios ──
        with torch.no_grad():
            old_token_lp = policy.get_token_log_probs(sequences, prompt_len)
        
        new_token_lp = policy.get_token_log_probs(sequences, prompt_len)
        ref_token_lp = ref_policy.get_token_log_probs(sequences, prompt_len)
        
        # Ratio per token: [B*G, T]
        ratio = torch.exp(new_token_lp - old_token_lp)
        
        # ── Step 5: Clipped surrogate loss ──
        # Expand advantages to per-token: [B*G, 1] → [B*G, T]
        adv_expanded = advantages_flat.unsqueeze(1).expand_as(ratio)
        
        surr1 = ratio * adv_expanded
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_expanded
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # ── Step 6: Per-token KL penalty ──
        # KL(π_θ || π_ref) ≈ (log π_θ - log π_ref)
        kl_per_token = new_token_lp - ref_token_lp
        kl_penalty = kl_coeff * kl_per_token.mean()
        
        # ── Step 7: Total loss and update ──
        total_loss = policy_loss + kl_penalty
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        
        # Compute fraction of clipped ratios
        with torch.no_grad():
            clip_fraction = ((ratio - 1).abs() > clip_eps).float().mean().item()
        
        return {
            "avg_reward": rewards.mean().item(),
            "best_reward": rewards.max().item(),
            "adv_std": advantages.std().item(),
            "kl": kl_per_token.mean().item(),
            "clip_frac": clip_fraction,
            "loss": total_loss.item(),
        }
    
    # Training loop
    print(f"\n  Config: G={group_size}, ε={clip_eps}, β_kl={kl_coeff}")
    print(f"  Prompts/batch: {n_prompts}, Total samples: {n_prompts * group_size}/step")
    print(f"\n  {'Step':>6} │ {'Avg R':>7} │ {'Best R':>7} │ {'KL':>7} │ "
          f"{'Clip%':>6} │ {'Loss':>8}")
    print(f"  {'─'*6}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*8}")
    
    for step in range(n_steps):
        prompts = torch.randint(0, vocab_size, (n_prompts, prompt_len))
        metrics = grpo_step(prompts)
        
        if (step + 1) % 5 == 0:
            print(f"  {step+1:>6} │ {metrics['avg_reward']:>7.3f} │ "
                  f"{metrics['best_reward']:>7.3f} │ {metrics['kl']:>7.4f} │ "
                  f"{metrics['clip_frac']:>5.1%} │ {metrics['loss']:>8.4f}")
    
    # Show group statistics for a sample prompt
    print(f"\n  ── Sample GRPO Group ──")
    sample_prompt = torch.randint(0, vocab_size, (1, prompt_len))
    expanded = sample_prompt.expand(group_size, -1)
    
    with torch.no_grad():
        samples = policy.generate(expanded, max_new_tokens=gen_len, temperature=0.9)
        rewards_g = torch.tensor([grpo_reward(s[prompt_len:]) for s in samples])
    
    mean_r, std_r = rewards_g.mean(), rewards_g.std()
    advantages_g = (rewards_g - mean_r) / (std_r + 1e-8)
    
    print(f"  Prompt tokens: {sample_prompt[0].tolist()}")
    print(f"  Group mean: {mean_r:.3f}, std: {std_r:.3f}")
    for i in range(group_size):
        resp = samples[i][prompt_len:].tolist()
        marker = "✓" if advantages_g[i] > 0 else "✗"
        print(f"    {marker} R={rewards_g[i]:.3f}  A={advantages_g[i]:+.3f}  "
              f"tokens={resp}")
    
    del policy, ref_policy
    
    print(f"""
  KEY GRPO PROPERTIES:
  ✓ No value model — advantages from group statistics
  ✓ Per-token KL keeps policy near reference
  ✓ Clipping prevents too-large updates
  ✓ Group normalization: some responses always + , some always −
""")


# ============================================================================
# SECTION 3: EXPERT ITERATION / ReST
# ============================================================================

def expert_iteration():
    """
    Expert Iteration (ExIt) / ReST:
    Generate → Filter → Fine-tune cycle.
    """
    print("\n\n" + "=" * 70)
    print("  SECTION 3: EXPERT ITERATION / ReST")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  Expert Iteration (Anthony et al., 2017) / ReST (Gulcehre et al., 2023):
  
  Loop:
    1. GENERATE: Sample many responses from current policy
    2. FILTER: Keep only high-reward responses
    3. FINE-TUNE: SFT on the filtered (good) responses
    4. Repeat
  
  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Generate │───→│  Filter  │───→│   SFT    │──┐
  │ π_k(y|x) │    │ R(y)>τ   │    │ on good  │  │
  └──────────┘    └──────────┘    │ outputs  │  │
       ↑                          └──────────┘  │
       └────────────────────────────────────────┘
  
  No gradient through reward! Pure SFT on curated data.
  Much simpler than PPO/GRPO but can be very effective.
""")
    
    vocab_size = 30
    prompt_len = 2
    gen_len = 6
    
    # Initialize policy
    policy = MiniLanguageModel(vocab_size=vocab_size, d_model=32)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    # Reward function
    def quality_reward(tokens: torch.Tensor) -> float:
        t = tokens.tolist()
        unique = len(set(t)) / len(t)
        ascending = sum(1 for i in range(1, len(t)) if t[i] > t[i-1]) / (len(t) - 1)
        no_repeat = 1 - sum(1 for i in range(1, len(t)) if t[i] == t[i-1]) / (len(t) - 1)
        return (unique + ascending + no_repeat) / 3
    
    # SFT training step
    def sft_step(sequences: torch.Tensor) -> float:
        """Standard SFT loss (next-token prediction) on filtered data."""
        logits = policy(sequences)
        # Loss only on response tokens
        response_logits = logits[:, prompt_len-1:-1, :]  # [B, gen_len, V]
        response_targets = sequences[:, prompt_len:]       # [B, gen_len]
        
        loss = F.cross_entropy(
            response_logits.reshape(-1, vocab_size),
            response_targets.reshape(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        return loss.item()
    
    n_iterations = 5
    n_generate = 64    # Samples per iteration
    top_fraction = 0.25  # Keep top 25%
    sft_epochs = 10
    
    print(f"\n  Config: {n_iterations} iterations, {n_generate} samples/iter, "
          f"keep top {top_fraction:.0%}")
    print(f"\n  {'Iter':>6} │ {'Gen Avg R':>9} │ {'Filtered R':>10} │ "
          f"{'SFT Loss':>8} │ {'#Kept':>5}")
    print(f"  {'─'*6}─┼─{'─'*9}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*5}")
    
    for iteration in range(n_iterations):
        # ── Step 1: GENERATE ──
        prompts = torch.randint(0, vocab_size, (n_generate, prompt_len))
        
        with torch.no_grad():
            sequences = policy.generate(prompts, max_new_tokens=gen_len)
        
        # ── Step 2: FILTER ──
        rewards = torch.tensor([quality_reward(s[prompt_len:]) for s in sequences])
        
        # Keep top fraction
        n_keep = max(int(n_generate * top_fraction), 4)
        top_indices = rewards.argsort(descending=True)[:n_keep]
        
        filtered_sequences = sequences[top_indices]
        filtered_rewards = rewards[top_indices]
        
        # ── Step 3: FINE-TUNE (SFT on filtered data) ──
        sft_losses = []
        for epoch in range(sft_epochs):
            # Shuffle
            perm = torch.randperm(n_keep)
            batch_seq = filtered_sequences[perm[:min(16, n_keep)]]
            loss = sft_step(batch_seq)
            sft_losses.append(loss)
        
        print(f"  {iteration+1:>6} │ {rewards.mean():>9.3f} │ "
              f"{filtered_rewards.mean():>10.3f} │ "
              f"{sum(sft_losses)/len(sft_losses):>8.4f} │ {n_keep:>5}")
    
    # Show improvement
    print(f"\n  ── Final Generation Quality ──")
    test_prompts = torch.randint(0, vocab_size, (32, prompt_len))
    with torch.no_grad():
        test_sequences = policy.generate(test_prompts, max_new_tokens=gen_len)
    
    test_rewards = [quality_reward(s[prompt_len:]) for s in test_sequences]
    print(f"  Avg reward after ExIt: {sum(test_rewards)/len(test_rewards):.3f}")
    
    del policy
    
    print(f"""
  EXPERT ITERATION ADVANTAGES:
  ✓ Simple: just generate → filter → SFT
  ✓ No RL gradients — fully supervised
  ✓ Stable training (standard SFT loss)
  ✓ Works well when you can sample many solutions
  
  LIMITATIONS:
  ✗ On-policy: need to regenerate each iteration
  ✗ Wasteful: discards most generated data
  ✗ Ceiling effect: can only learn from what it generates
""")


# ============================================================================
# SECTION 4: STaR — SELF-TAUGHT REASONER
# ============================================================================

def star_from_scratch():
    """
    STaR: Self-Taught Reasoner (Zelikman et al., 2022).
    Generate rationales, filter by correctness, fine-tune.
    """
    print("\n\n" + "=" * 70)
    print("  SECTION 4: STaR — SELF-TAUGHT REASONER")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  STaR extends Expert Iteration with RATIONALE GENERATION:
  
  1. Given (question, answer) pairs
  2. Generate: question → rationale → answer
  3. Filter: keep only rationales that lead to CORRECT answers
  4. RATIONALIZATION: For incorrect ones, show the answer and 
     generate a rationale that reaches it (hint mechanism)
  5. Fine-tune on correct + rationalized examples
  
  KEY INSIGHT: Rationalization step prevents the model from only
  learning from easy problems — it can learn reasoning patterns
  from hard problems too.
  
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │   Generate   │───→│    Check     │───→│  Fine-tune   │
  │  rationales  │    │  correctness │    │  on correct  │
  └──────────────┘    └──────────────┘    └──────────────┘
         │                   │
         │            ┌──────────────┐
         │            │ Rationalize  │───→ also add to
         └───────────→│  (with hint) │    training set
                      └──────────────┘
""")
    
    # Simulate STaR with a simple arithmetic task
    # "Problem": Given a sequence pattern, predict the rule
    
    vocab_size = 30
    prompt_len = 4
    gen_len = 6
    
    policy = MiniLanguageModel(vocab_size=vocab_size, d_model=32)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    # Task: Generated sequence should have tokens that sum to a target
    def create_problem():
        """Create a simple problem: target sum encoded in prompt."""
        # Prompt encodes a target sum (modulo vocab_size)
        target_sum = torch.randint(5, 20, (1,)).item()
        prompt = torch.tensor([target_sum % vocab_size, 
                               (target_sum // vocab_size) % vocab_size,
                               1, 1])  # Markers
        return prompt, target_sum
    
    def check_answer(response_tokens: torch.Tensor, target: int) -> bool:
        """Check if response tokens sum to target (±2)."""
        token_sum = response_tokens.sum().item()
        return abs(token_sum - target) <= 2
    
    def create_rationalization(prompt: torch.Tensor, target: int) -> torch.Tensor:
        """Create a rationalization: construct a good response given the answer."""
        # Distribute target across gen_len tokens
        base = target // gen_len
        remainder = target % gen_len
        tokens = [min(base + (1 if i < remainder else 0), vocab_size - 1) 
                  for i in range(gen_len)]
        return torch.tensor(tokens)
    
    n_iterations = 5
    n_problems = 48
    
    print(f"\n  Task: Generate tokens that sum to target value")
    print(f"  {n_iterations} STaR iterations, {n_problems} problems each")
    print(f"\n  {'Iter':>6} │ {'Correct%':>9} │ {'Rational.':>9} │ "
          f"{'Train Size':>10} │ {'SFT Loss':>8}")
    print(f"  {'─'*6}─┼─{'─'*9}─┼─{'─'*9}─┼─{'─'*10}─┼─{'─'*8}")
    
    for iteration in range(n_iterations):
        # Create problems
        problems = [create_problem() for _ in range(n_problems)]
        prompts = torch.stack([p[0] for p in problems])
        targets = [p[1] for p in problems]
        
        # Step 1: Generate rationales (responses)
        with torch.no_grad():
            sequences = policy.generate(prompts, max_new_tokens=gen_len)
        
        # Step 2: Check correctness
        correct_mask = []
        for i, (seq, target) in enumerate(zip(sequences, targets)):
            correct_mask.append(check_answer(seq[prompt_len:], target))
        
        n_correct = sum(correct_mask)
        
        # Step 3: Rationalization for incorrect ones
        training_sequences = []
        
        for i in range(n_problems):
            if correct_mask[i]:
                # Keep correct responses
                training_sequences.append(sequences[i])
            else:
                # Rationalize: construct a correct response
                rational = create_rationalization(prompts[i], targets[i])
                full_seq = torch.cat([prompts[i], rational])
                training_sequences.append(full_seq)
        
        n_rationalized = n_problems - n_correct
        training_data = torch.stack(training_sequences)
        
        # Step 4: Fine-tune
        losses = []
        for epoch in range(8):
            perm = torch.randperm(len(training_data))
            for batch_start in range(0, len(training_data), 16):
                batch_idx = perm[batch_start:batch_start+16]
                batch = training_data[batch_idx]
                
                logits = policy(batch)
                response_logits = logits[:, prompt_len-1:-1, :]
                response_targets = batch[:, prompt_len:]
                
                loss = F.cross_entropy(
                    response_logits.reshape(-1, vocab_size),
                    response_targets.reshape(-1)
                )
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
        
        print(f"  {iteration+1:>6} │ {n_correct/n_problems:>8.1%} │ "
              f"{n_rationalized:>9d} │ {len(training_data):>10d} │ "
              f"{sum(losses)/len(losses):>8.4f}")
    
    del policy
    
    print(f"""
  STaR KEY INSIGHTS:
  ✓ Rationalization step: learn from failures, not just successes
  ✓ Bootstraps reasoning ability from (question, answer) pairs
  ✓ No reward model needed — just correctness verification
  ✓ Each iteration improves reasoning quality
  
  RELATION TO OTHER METHODS:
  • Expert Iteration without rationalization: only learns from easy problems
  • STaR: learns from ALL problems via the rationalization hint
  • ReST*: STaR variant with reward-based filtering instead of exact match
""")


# ============================================================================
# SECTION 5: RLOO — REINFORCE LEAVE-ONE-OUT
# ============================================================================

def rloo_from_scratch():
    """
    RLOO: REINFORCE with Leave-One-Out baseline.
    Unbiased baseline using other samples in the group.
    """
    print("\n\n" + "=" * 70)
    print("  SECTION 5: RLOO — REINFORCE LEAVE-ONE-OUT")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  RLOO uses the leave-one-out mean as baseline:
  
    For sample i with reward r_i, the baseline is:
    
      b_i = (1/(G-1)) · Σ_{{j≠i}} r_j
    
    Advantage: A_i = r_i - b_i
  
  Properties:
  • Unbiased: E[b_i] = E[R] (unlike sample mean which includes r_i)
  • Lower variance than no baseline
  • GRPO goes further by also dividing by std
""")
    
    vocab_size = 30
    prompt_len = 2
    gen_len = 6
    group_size = 8
    
    policy = MiniLanguageModel(vocab_size=vocab_size, d_model=32)
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)
    
    def diversity_reward(tokens):
        t = tokens.tolist()
        return len(set(t)) / len(t)
    
    def compute_rloo_advantages(rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute RLOO advantages for a group of rewards.
        rewards: [G] tensor
        returns: [G] advantages
        """
        G = rewards.shape[0]
        total = rewards.sum()
        
        # Leave-one-out mean for each sample
        # b_i = (total - r_i) / (G - 1)
        baselines = (total - rewards) / (G - 1)
        advantages = rewards - baselines
        
        return advantages
    
    # Compare RLOO vs mean baseline vs GRPO normalization
    print(f"\n  ── Baseline Comparison (one group) ──\n")
    
    sample_rewards = torch.tensor([0.3, 0.8, 0.5, 0.2, 0.9, 0.4, 0.7, 0.6])
    
    # Mean baseline
    mean_bl = sample_rewards.mean()
    mean_adv = sample_rewards - mean_bl
    
    # RLOO baseline
    rloo_adv = compute_rloo_advantages(sample_rewards)
    
    # GRPO normalization
    grpo_adv = (sample_rewards - sample_rewards.mean()) / (sample_rewards.std() + 1e-8)
    
    print(f"  {'i':>3} │ {'Reward':>7} │ {'Mean Adv':>8} │ {'RLOO Adv':>8} │ {'GRPO Adv':>8}")
    print(f"  {'─'*3}─┼─{'─'*7}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}")
    for i in range(group_size):
        print(f"  {i+1:>3} │ {sample_rewards[i]:>7.3f} │ {mean_adv[i]:>8.3f} │ "
              f"{rloo_adv[i]:>8.3f} │ {grpo_adv[i]:>8.3f}")
    
    print(f"\n  Mean adv variance:  {mean_adv.var():.4f}")
    print(f"  RLOO adv variance:  {rloo_adv.var():.4f}")
    print(f"  GRPO adv variance:  {grpo_adv.var():.4f} (normalized to ~1)")
    
    # RLOO Training
    print(f"\n\n  ── RLOO Training ──\n")
    print(f"  {'Step':>6} │ {'Avg R':>7} │ {'Adv μ':>7} │ {'Loss':>8}")
    print(f"  {'─'*6}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*8}")
    
    for step in range(30):
        n_prompts = 8
        all_losses = []
        all_rewards = []
        all_adv_means = []
        
        for p_idx in range(n_prompts):
            prompt = torch.randint(0, vocab_size, (group_size, prompt_len))
            # Same prompt for all in group
            prompt = prompt[0:1].expand(group_size, -1)
            
            # Generate group
            with torch.no_grad():
                sequences = policy.generate(prompt, max_new_tokens=gen_len)
            
            # Score
            rewards = torch.tensor([diversity_reward(s[prompt_len:]) for s in sequences])
            
            # RLOO advantages
            advantages = compute_rloo_advantages(rewards)
            
            # REINFORCE update
            log_probs = policy.get_sequence_log_prob(sequences, prompt_len)
            loss = -(advantages * log_probs).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            all_losses.append(loss.item())
            all_rewards.extend(rewards.tolist())
            all_adv_means.append(advantages.mean().item())
        
        if (step + 1) % 6 == 0:
            print(f"  {step+1:>6} │ {sum(all_rewards)/len(all_rewards):>7.3f} │ "
                  f"{sum(all_adv_means)/len(all_adv_means):>7.4f} │ "
                  f"{sum(all_losses)/len(all_losses):>8.4f}")
    
    del policy
    
    print(f"""
  RLOO vs GRPO COMPARISON:
  
  RLOO:
  • Advantage = r_i - mean(r_{{j≠i}})
  • Unbiased baseline (does not include own sample)
  • Used in vanilla REINFORCE-style updates
  
  GRPO:
  • Advantage = (r_i - mean(r)) / std(r)
  • Normalized advantages (unit variance)
  • PPO-style clipping for stability
  • KL penalty against reference policy
  
  Both avoid learning a value function, relying on group statistics.
""")


# ============================================================================
# SECTION 6: COMPARISON — ALL METHODS SIDE BY SIDE
# ============================================================================

def method_comparison():
    """Compare all implemented RL methods on the same task."""
    print("\n\n" + "=" * 70)
    print("  SECTION 6: METHOD COMPARISON")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    vocab_size = 30
    prompt_len = 2
    gen_len = 6
    
    def evaluation_reward(tokens):
        t = tokens.tolist()
        diversity = len(set(t)) / len(t)
        no_repeat = 1 - sum(1 for i in range(1, len(t)) if t[i] == t[i-1]) / (len(t) - 1)
        return (diversity + no_repeat) / 2
    
    def evaluate_policy(policy, n_eval=64):
        prompts = torch.randint(0, vocab_size, (n_eval, prompt_len))
        with torch.no_grad():
            seqs = policy.generate(prompts, max_new_tokens=gen_len)
        rewards = [evaluation_reward(s[prompt_len:]) for s in seqs]
        return sum(rewards) / len(rewards)
    
    results = {}
    
    # Method 1: REINFORCE (no baseline)
    policy = MiniLanguageModel(vocab_size=vocab_size, d_model=32)
    opt = torch.optim.Adam(policy.parameters(), lr=5e-4)
    
    for step in range(40):
        prompts = torch.randint(0, vocab_size, (16, prompt_len))
        with torch.no_grad():
            seqs = policy.generate(prompts, max_new_tokens=gen_len)
        rewards = torch.tensor([evaluation_reward(s[prompt_len:]) for s in seqs])
        lp = policy.get_sequence_log_prob(seqs, prompt_len)
        loss = -(rewards * lp).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    
    results["REINFORCE"] = evaluate_policy(policy)
    del policy, opt
    
    # Method 2: REINFORCE + baseline
    torch.manual_seed(42)
    policy = MiniLanguageModel(vocab_size=vocab_size, d_model=32)
    opt = torch.optim.Adam(policy.parameters(), lr=5e-4)
    running_baseline = 0.0
    
    for step in range(40):
        prompts = torch.randint(0, vocab_size, (16, prompt_len))
        with torch.no_grad():
            seqs = policy.generate(prompts, max_new_tokens=gen_len)
        rewards = torch.tensor([evaluation_reward(s[prompt_len:]) for s in seqs])
        running_baseline = 0.9 * running_baseline + 0.1 * rewards.mean().item()
        advantages = rewards - running_baseline
        lp = policy.get_sequence_log_prob(seqs, prompt_len)
        loss = -(advantages * lp).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    
    results["REINFORCE+BL"] = evaluate_policy(policy)
    del policy, opt
    
    # Method 3: RLOO
    torch.manual_seed(42)
    policy = MiniLanguageModel(vocab_size=vocab_size, d_model=32)
    opt = torch.optim.Adam(policy.parameters(), lr=5e-4)
    
    for step in range(40):
        for _ in range(4):  # 4 prompts per step
            prompt = torch.randint(0, vocab_size, (1, prompt_len)).expand(8, -1)
            with torch.no_grad():
                seqs = policy.generate(prompt, max_new_tokens=gen_len)
            rewards = torch.tensor([evaluation_reward(s[prompt_len:]) for s in seqs])
            total = rewards.sum()
            baselines = (total - rewards) / 7
            advantages = rewards - baselines
            lp = policy.get_sequence_log_prob(seqs, prompt_len)
            loss = -(advantages * lp).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
    
    results["RLOO"] = evaluate_policy(policy)
    del policy, opt
    
    # Method 4: GRPO
    torch.manual_seed(42)
    policy = MiniLanguageModel(vocab_size=vocab_size, d_model=32)
    ref = copy.deepcopy(policy)
    for p in ref.parameters(): p.requires_grad = False
    opt = torch.optim.Adam(policy.parameters(), lr=5e-4)
    
    for step in range(40):
        for _ in range(4):
            prompt = torch.randint(0, vocab_size, (1, prompt_len)).expand(8, -1)
            with torch.no_grad():
                seqs = policy.generate(prompt, max_new_tokens=gen_len)
            rewards = torch.tensor([evaluation_reward(s[prompt_len:]) for s in seqs])
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            old_lp = policy.get_token_log_probs(seqs, prompt_len).detach()
            new_lp = policy.get_token_log_probs(seqs, prompt_len)
            ref_lp = ref.get_token_log_probs(seqs, prompt_len)
            ratio = torch.exp(new_lp - old_lp)
            adv_exp = adv.unsqueeze(1).expand_as(ratio)
            s1 = ratio * adv_exp
            s2 = torch.clamp(ratio, 0.8, 1.2) * adv_exp
            loss = -torch.min(s1, s2).mean() + 0.04 * (new_lp - ref_lp).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
    
    results["GRPO"] = evaluate_policy(policy)
    del policy, ref, opt
    
    # Method 5: Expert Iteration
    torch.manual_seed(42)
    policy = MiniLanguageModel(vocab_size=vocab_size, d_model=32)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    for iteration in range(5):
        prompts = torch.randint(0, vocab_size, (64, prompt_len))
        with torch.no_grad():
            seqs = policy.generate(prompts, max_new_tokens=gen_len)
        rewards = torch.tensor([evaluation_reward(s[prompt_len:]) for s in seqs])
        top_k_idx = rewards.argsort(descending=True)[:16]
        filtered = seqs[top_k_idx]
        for epoch in range(8):
            logits = policy(filtered)
            loss = F.cross_entropy(
                logits[:, prompt_len-1:-1].reshape(-1, vocab_size),
                filtered[:, prompt_len:].reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
    
    results["ExIt"] = evaluate_policy(policy)
    del policy, opt
    
    # Display results
    print(f"\n  ── Results (40 training steps, same seed) ──\n")
    print(f"  {'Method':>20} │ {'Avg Reward':>10} │ {'Bar':>25}")
    print(f"  {'─'*20}─┼─{'─'*10}─┼─{'─'*25}")
    
    max_r = max(results.values())
    for method, reward in sorted(results.items(), key=lambda x: -x[1]):
        bar_len = int(20 * reward / max_r)
        bar = "█" * bar_len
        print(f"  {method:>20} │ {reward:>10.4f} │ {bar}")
    
    print(f"""
  ┌─────────────────┬──────────┬───────────┬────────────────────┐
  │ Method          │ Value    │ Memory    │ Stability          │
  │                 │ Model?   │ Overhead  │                    │
  ├─────────────────┼──────────┼───────────┼────────────────────┤
  │ REINFORCE       │ No       │ Low       │ High variance      │
  │ REINFORCE+BL    │ Yes      │ Medium    │ Better variance    │
  │ RLOO            │ No       │ Low       │ Unbiased baseline  │
  │ GRPO            │ No       │ Low       │ Stable (clipped)   │
  │ Expert Iter.    │ No       │ Low       │ Very stable (SFT)  │
  └─────────────────┴──────────┴───────────┴────────────────────┘
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  RL FINE-TUNING FROM SCRATCH — REINFORCE, GRPO, ExIt, STaR, RLOO ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    reinforce_with_baseline()
    grpo_from_scratch()
    expert_iteration()
    star_from_scratch()
    rloo_from_scratch()
    method_comparison()
    
    print("\n" + "=" * 70)
    print("  FROM-SCRATCH MODULE COMPLETE")
    print("=" * 70)
    print("""
    Implemented:
    ✓ REINFORCE with learned baseline
    ✓ GRPO with group normalization + clipping + KL
    ✓ Expert Iteration (generate → filter → SFT)
    ✓ STaR with rationalization
    ✓ RLOO baseline computation
    ✓ Side-by-side comparison of all methods
    """)


if __name__ == "__main__":
    main()
