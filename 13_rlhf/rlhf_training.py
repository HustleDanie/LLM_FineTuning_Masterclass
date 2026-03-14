"""
RLHF Training — Full RLHF Pipeline with PPO
=============================================

Complete RLHF training implementations:

1. SimplePPOFromScratch
   - Minimal PPO implementation for text generation
   - Understanding the core training loop

2. RLHFWithTRL
   - Full RLHF pipeline using TRL's PPOTrainer
   - Production-ready implementation

3. KLController
   - Adaptive KL penalty coefficient
   - Maintaining target KL divergence

4. RewardShaping
   - Reward normalization and clipping
   - Per-token vs per-sequence rewards

5. RLHFMonitoring
   - Key metrics to track during training
   - Detecting reward hacking and collapse

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional


# ============================================================================
# SECTION 1: SIMPLE PPO FROM SCRATCH
# ============================================================================

def simple_ppo_from_scratch():
    """Minimal PPO implementation for text generation."""
    print("=" * 65)
    print("  SECTION 1: SIMPLE PPO FROM SCRATCH")
    print("=" * 65)
    
    torch.manual_seed(42)
    
    # ─── Tiny language model for demonstration ───
    vocab_size = 50
    d_model = 32
    max_len = 8
    
    class TinyLM(nn.Module):
        """Minimal language model with value head."""
        
        def __init__(self, vocab_size, d_model):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.rnn = nn.GRU(d_model, d_model, batch_first=True)
            self.lm_head = nn.Linear(d_model, vocab_size)
            self.value_head = nn.Linear(d_model, 1)
        
        def forward(self, input_ids):
            x = self.embed(input_ids)
            h, _ = self.rnn(x)
            logits = self.lm_head(h)
            values = self.value_head(h).squeeze(-1)
            return logits, values
        
        def generate(self, prompt_ids, max_new_tokens=4):
            """Generate tokens autoregressively."""
            ids = prompt_ids.clone()
            for _ in range(max_new_tokens):
                logits, _ = self(ids)
                next_logit = logits[:, -1, :]
                probs = F.softmax(next_logit, dim=-1)
                next_token = torch.multinomial(probs, 1)
                ids = torch.cat([ids, next_token], dim=-1)
            return ids
    
    # ─── Create policy, reference, and reward models ───
    policy = TinyLM(vocab_size, d_model)
    reference = TinyLM(vocab_size, d_model)
    reference.load_state_dict(policy.state_dict())
    for p in reference.parameters():
        p.requires_grad = False
    
    # Simple reward: prefer sequences with more variety (unique tokens)
    def reward_fn(token_ids):
        """Reward function: more unique tokens = higher reward."""
        rewards = []
        for seq in token_ids:
            unique = len(set(seq.tolist()))
            rewards.append(unique / len(seq) * 2.0 - 1.0)
        return torch.tensor(rewards)
    
    # ─── PPO training loop ───
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    beta = 0.1        # KL penalty coefficient
    epsilon = 0.2     # PPO clip range
    gamma = 0.99      # Discount factor
    lam = 0.95        # GAE lambda
    
    print(f"\n  Config: β={beta}, ε={epsilon}, γ={gamma}, λ={lam}")
    print(f"\n  {'Step':>6} {'Reward':>8} {'KL':>8} {'Policy Loss':>12} {'Value Loss':>12}")
    print(f"  {'─'*6}─{'─'*8}─{'─'*8}─{'─'*12}─{'─'*12}")
    
    for step in range(30):
        # ─── 1. GENERATE: Policy produces responses ───
        batch_size = 16
        prompt = torch.randint(0, vocab_size, (batch_size, 2))  # Short prompts
        
        policy.eval()
        with torch.no_grad():
            generated = policy.generate(prompt, max_new_tokens=6)
        policy.train()
        
        response_ids = generated[:, 2:]  # Just the generated part
        
        # ─── 2. SCORE: Reward model evaluates ───
        rewards = reward_fn(response_ids)
        
        # ─── 3. COMPUTE: Log-probs and values ───
        logits, values = policy(generated)
        log_probs = F.log_softmax(logits, dim=-1)
        
        with torch.no_grad():
            ref_logits, _ = reference(generated)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        
        # Gather log-probs for generated tokens
        response_start = 2
        response_logprobs = log_probs[:, response_start-1:-1, :]  # Shifted
        ref_response_logprobs = ref_log_probs[:, response_start-1:-1, :]
        
        # Per-token log probs for generated tokens
        gen_tokens = generated[:, response_start:]
        action_logprobs = response_logprobs.gather(
            2, gen_tokens.unsqueeze(-1)
        ).squeeze(-1)
        ref_action_logprobs = ref_response_logprobs.gather(
            2, gen_tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        # ─── 4. KL PENALTY ───
        kl_per_token = action_logprobs - ref_action_logprobs
        kl_mean = kl_per_token.mean()
        
        # ─── 5. COMPUTE ADVANTAGES (simplified GAE) ───
        response_values = values[:, response_start:]
        
        # Per-token rewards: 0 everywhere except last token gets the reward
        token_rewards = torch.zeros_like(action_logprobs)
        token_rewards[:, -1] = rewards - beta * kl_per_token.sum(dim=-1)
        
        # GAE
        advantages = torch.zeros_like(token_rewards)
        last_gae = torch.zeros(batch_size)
        
        for t in reversed(range(token_rewards.size(1))):
            if t == token_rewards.size(1) - 1:
                next_value = torch.zeros(batch_size)
            else:
                next_value = response_values[:, t + 1].detach()
            
            delta = token_rewards[:, t] + gamma * next_value - response_values[:, t].detach()
            advantages[:, t] = last_gae = delta + gamma * lam * last_gae
        
        returns = advantages + response_values.detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # ─── 6. PPO UPDATE ───
        # Recompute log-probs (they may have changed)
        new_logits, new_values = policy(generated)
        new_log_probs = F.log_softmax(new_logits, dim=-1)
        new_action_logprobs = new_log_probs[:, response_start-1:-1, :].gather(
            2, gen_tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        # Policy ratio
        ratio = torch.exp(new_action_logprobs - action_logprobs.detach())
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        new_response_values = new_values[:, response_start:]
        value_loss = F.mse_loss(new_response_values, returns)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        
        if (step + 1) % 5 == 0:
            print(f"  {step+1:>6} {rewards.mean().item():>8.3f} "
                  f"{kl_mean.item():>8.4f} {policy_loss.item():>12.4f} "
                  f"{value_loss.item():>12.4f}")
    
    print(f"\n  ✓ PPO training complete!")
    print(f"  The policy learned to generate more diverse token sequences.")
    
    del policy, reference


# ============================================================================
# SECTION 2: RLHF WITH TRL
# ============================================================================

def rlhf_with_trl():
    """Full RLHF pipeline using TRL."""
    print("\n\n" + "=" * 65)
    print("  SECTION 2: RLHF WITH TRL (PPOTrainer)")
    print("=" * 65)
    
    print(f"""
  ═══ TRL PPOTrainer Pipeline ═══
  
  TRL (Transformer Reinforcement Learning) provides PPOTrainer
  which handles the entire RLHF pipeline:
  
  1. PPOConfig: Configure PPO hyperparameters
  2. PPOTrainer: Manages generation, scoring, and PPO updates
  3. Automatic KL penalty and reward processing
  4. Built-in logging and metrics
  
  Typical code structure:
  ─────────────────────
""")
    
    # Show the standard TRL PPO code pattern
    print("""
  ```python
  from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
  from transformers import AutoTokenizer
  
  # 1. Load models
  model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
  ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
  tokenizer = AutoTokenizer.from_pretrained("gpt2")
  
  # 2. Configure PPO
  ppo_config = PPOConfig(
      model_name="gpt2",
      learning_rate=1.41e-5,
      batch_size=16,
      mini_batch_size=4,
      gradient_accumulation_steps=1,
      ppo_epochs=4,                 # PPO epochs per batch
      kl_penalty="kl",              # KL penalty type
      init_kl_coef=0.2,             # Initial β
      target_kl=6.0,                # Target KL divergence
      cliprange=0.2,                # PPO clip range ε
      cliprange_value=0.2,          # Value function clip range
      gamma=1.0,                    # Discount factor
      lam=0.95,                     # GAE lambda
  )
  
  # 3. Create trainer
  ppo_trainer = PPOTrainer(
      config=ppo_config,
      model=model,
      ref_model=ref_model,
      tokenizer=tokenizer,
  )
  
  # 4. Training loop
  for batch in dataloader:
      # Generate responses
      query_tensors = [tokenizer.encode(q, return_tensors="pt") for q in batch]
      response_tensors = ppo_trainer.generate(query_tensors)
      
      # Score with reward model
      rewards = [reward_model(q, r) for q, r in zip(query_tensors, response_tensors)]
      
      # PPO update
      stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
      
      # Log metrics
      print(f"reward: {stats['ppo/mean_scores']:.3f}, "
            f"kl: {stats['objective/kl']:.3f}")
  ```
""")
    
    # Demonstrate a simplified version that actually runs
    print("  ── Simplified Running Example ──")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load policy and reference
    policy = AutoModelForCausalLM.from_pretrained(model_name)
    policy.config.pad_token_id = tokenizer.pad_token_id
    
    reference = AutoModelForCausalLM.from_pretrained(model_name)
    reference.config.pad_token_id = tokenizer.pad_token_id
    for p in reference.parameters():
        p.requires_grad = False
    
    # Simple reward: prefer shorter, more coherent responses
    def simple_reward(text):
        """Simple heuristic reward."""
        score = 0.0
        # Prefer texts that end with a period
        if text.strip().endswith('.'):
            score += 0.5
        # Penalize repetition
        words = text.split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            score += unique_ratio
        # Penalize very short or very long
        if 5 < len(words) < 30:
            score += 0.5
        return score
    
    # Simplified training loop (no PPO overhead for demo)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5)
    beta = 0.1
    
    prompts = [
        "The meaning of life is",
        "Artificial intelligence will",
        "The best way to learn is",
        "Science has shown that",
    ]
    
    print(f"\n  Training on {len(prompts)} prompts for 5 steps:")
    
    for step in range(5):
        total_reward = 0
        total_kl = 0
        
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            
            # Generate
            with torch.no_grad():
                gen_ids = policy.generate(
                    input_ids,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            reward = simple_reward(text)
            total_reward += reward
            
            # Compute log-probs for REINFORCE-style update
            policy_out = policy(gen_ids, labels=gen_ids)
            with torch.no_grad():
                ref_out = reference(gen_ids, labels=gen_ids)
            
            # KL approximation using losses
            kl_approx = (policy_out.loss - ref_out.loss).abs()
            total_kl += kl_approx.item()
            
            # REINFORCE-style loss (simplified PPO)
            adjusted_reward = reward - beta * kl_approx.item()
            loss = -adjusted_reward * policy_out.loss  # Negative because we minimize
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        
        avg_reward = total_reward / len(prompts)
        avg_kl = total_kl / len(prompts)
        print(f"    Step {step+1}: reward={avg_reward:.3f}, kl≈{avg_kl:.4f}")
    
    # Show final generations
    print(f"\n  ── Final Generations ──")
    policy.eval()
    for prompt in prompts[:2]:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            gen_ids = policy.generate(
                input_ids, max_new_tokens=25,
                do_sample=True, temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        print(f"    \"{prompt}\" →")
        print(f"      {text}")
    
    del policy, reference


# ============================================================================
# SECTION 3: KL CONTROLLER
# ============================================================================

def kl_controller():
    """Adaptive KL penalty coefficient."""
    print("\n\n" + "=" * 65)
    print("  SECTION 3: ADAPTIVE KL CONTROLLER")
    print("=" * 65)
    
    class AdaptiveKLController:
        """
        Adaptive KL coefficient controller.
        
        Adjusts β to maintain a target KL divergence:
        - If KL > target: Increase β (penalize more)
        - If KL < target: Decrease β (allow more exploration)
        
        From: Ziegler et al. "Fine-Tuning Language Models from Human Preferences"
        """
        
        def __init__(self, init_kl_coef: float = 0.2, target_kl: float = 6.0,
                     horizon: int = 10000):
            self.kl_coef = init_kl_coef
            self.target_kl = target_kl
            self.horizon = horizon
        
        def update(self, current_kl: float) -> float:
            """Update KL coefficient based on current KL."""
            proportional_error = (current_kl - self.target_kl) / self.target_kl
            mult = 1.0 + proportional_error / self.horizon
            self.kl_coef *= mult
            self.kl_coef = max(0.001, min(10.0, self.kl_coef))  # Clamp
            return self.kl_coef
    
    # Simulate KL controller behavior
    controller = AdaptiveKLController(init_kl_coef=0.2, target_kl=6.0, horizon=100)
    
    print(f"\n  Target KL: {controller.target_kl}")
    print(f"  Initial β: {controller.kl_coef}")
    
    print(f"\n  ── Scenario 1: KL keeps increasing (policy drifting) ──")
    kl_values = [4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 12.0, 9.0, 7.0]
    
    print(f"  {'Step':>6} {'Current KL':>12} {'β (KL coef)':>12} {'Action':>20}")
    print(f"  {'─'*6}─{'─'*12}─{'─'*12}─{'─'*20}")
    
    for i, kl in enumerate(kl_values):
        old_beta = controller.kl_coef
        new_beta = controller.update(kl)
        action = "↑ Increase β" if new_beta > old_beta else "↓ Decrease β"
        if abs(new_beta - old_beta) < 0.001:
            action = "≈ Stable"
        print(f"  {i+1:>6} {kl:>12.2f} {new_beta:>12.4f} {action:>20}")
    
    print(f"""
  ═══ KL Controller Strategies ═══
  
  1. Fixed β:
     • Simple: β = constant (e.g., 0.1)
     • Risk: May over/under-penalize
     • Use when: Quick experiments
  
  2. Adaptive β (above):
     • Adjusts to maintain target KL
     • More stable training
     • Use when: Production training
  
  3. KL target scheduling:
     • Start with high target (allow exploration)
     • Gradually decrease (converge)
     • Use when: Complex tasks needing exploration
  
  Typical target KL values:
  • Small models (GPT-2): 1.0 - 6.0
  • Large models (7B+): 0.5 - 3.0
  • Higher target = more policy change allowed
""")


# ============================================================================
# SECTION 4: REWARD SHAPING
# ============================================================================

def reward_shaping():
    """Reward normalization and shaping techniques."""
    print("\n\n" + "=" * 65)
    print("  SECTION 4: REWARD SHAPING")
    print("=" * 65)
    
    torch.manual_seed(42)
    
    # ─── Reward Normalization ───
    print(f"\n  ── Reward Normalization ──")
    
    # Simulate raw rewards from a reward model
    raw_rewards = torch.randn(100) * 3.0 + 2.0  # Mean=2, Std=3
    
    # Method 1: Running mean/std normalization
    class RunningNormalizer:
        def __init__(self):
            self.mean = 0.0
            self.var = 1.0
            self.count = 0
        
        def update(self, x):
            batch_mean = x.mean().item()
            batch_var = x.var().item()
            batch_count = len(x)
            
            total_count = self.count + batch_count
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * batch_count / total_count
            
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
            new_var = M2 / total_count
            
            self.mean = new_mean
            self.var = max(new_var, 1e-8)
            self.count = total_count
        
        def normalize(self, x):
            return (x - self.mean) / (self.var ** 0.5 + 1e-8)
    
    normalizer = RunningNormalizer()
    
    # Process in batches
    for i in range(0, 100, 20):
        batch = raw_rewards[i:i+20]
        normalizer.update(batch)
        normalized = normalizer.normalize(batch)
        
        print(f"    Batch {i//20+1}: raw_mean={batch.mean():.2f}, "
              f"raw_std={batch.std():.2f} → "
              f"norm_mean={normalized.mean():.2f}, "
              f"norm_std={normalized.std():.2f}")
    
    # ─── Reward Clipping ───
    print(f"\n  ── Reward Clipping ──")
    
    extreme_rewards = torch.tensor([-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0, 50.0])
    clipped = torch.clamp(extreme_rewards, -5.0, 5.0)
    
    print(f"    {'Raw':>8} → {'Clipped':>8}")
    print(f"    {'─'*8}───{'─'*8}")
    for r, c in zip(extreme_rewards, clipped):
        print(f"    {r.item():>8.1f} → {c.item():>8.1f}")
    
    print(f"""
  ═══ Reward Shaping Techniques ═══
  
  1. NORMALIZATION (running mean/std):
     • Keeps rewards centered around 0
     • Stable gradients regardless of reward scale
     • Most important technique!
  
  2. CLIPPING:
     • Prevent extreme rewards from destabilizing training
     • Typical range: [-10, 10] or [-5, 5]
     • Apply after normalization
  
  3. BASELINE SUBTRACTION:
     • Subtract mean reward per batch
     • Reduces variance of policy gradient
     • reward_adj = reward - mean(rewards)
  
  4. PER-TOKEN vs PER-SEQUENCE:
     • Per-sequence: Single reward for entire response
       (easier but sparse signal)
     • Per-token: Distribute reward across tokens
       (denser signal but harder to define)
     • Common: per-sequence reward + per-token KL penalty
  
  5. WHITENING:
     • Normalize advantages to mean=0, std=1
     • Standard technique in PPO
     • advantages = (A - mean(A)) / (std(A) + ε)
""")


# ============================================================================
# SECTION 5: RLHF MONITORING
# ============================================================================

def rlhf_monitoring():
    """Key metrics and monitoring for RLHF training."""
    print("\n\n" + "=" * 65)
    print("  SECTION 5: RLHF MONITORING")
    print("=" * 65)
    
    print(f"""
  ═══════════════════════════════════════════════════════════════
   KEY METRICS TO MONITOR DURING RLHF
  ═══════════════════════════════════════════════════════════════
  
  ┌────────────────────┬───────────┬────────────────────────────┐
  │ Metric             │ Healthy   │ Unhealthy                  │
  ├────────────────────┼───────────┼────────────────────────────┤
  │ Mean Reward        │ ↑ Gradual │ ↑↑↑ Too fast = hacking    │
  │                    │           │ ↓ Diverging                │
  ├────────────────────┼───────────┼────────────────────────────┤
  │ KL Divergence      │ 1-10      │ > 15 = too much drift     │
  │                    │ (stable)  │ < 0.1 = not learning       │
  ├────────────────────┼───────────┼────────────────────────────┤
  │ Policy Entropy     │ Slight ↓  │ ↓↓↓ Collapse (mode drop)  │
  │                    │           │ ↑↑ Random (not learning)   │
  ├────────────────────┼───────────┼────────────────────────────┤
  │ Reward Std         │ Moderate  │ → 0 (all same reward)     │
  │                    │           │ ↑↑↑ (high variance)        │
  ├────────────────────┼───────────┼────────────────────────────┤
  │ Value Loss         │ ↓ Gradual │ ↑ Diverging               │
  ├────────────────────┼───────────┼────────────────────────────┤
  │ Policy Clip Frac   │ 0.1-0.3   │ > 0.5 = updates too large │
  │                    │           │ < 0.01 = not learning      │
  ├────────────────────┼───────────┼────────────────────────────┤
  │ Approx KL          │ < 0.1     │ > 0.3 = PPO epochs too    │
  │ (per PPO epoch)    │           │   many or LR too high      │
  ├────────────────────┼───────────┼────────────────────────────┤
  │ Response Length     │ Stable    │ ↑↑↑ Length hacking         │
  │                    │           │ (gaming reward via length)  │
  └────────────────────┴───────────┴────────────────────────────┘
""")
    
    # Simulate training metrics
    print("  ── Simulated Healthy Training Run ──")
    
    torch.manual_seed(42)
    
    n_steps = 20
    
    # Simulate healthy metrics
    print(f"\n  {'Step':>6} {'Reward':>8} {'KL':>8} {'Entropy':>8} "
          f"{'V_Loss':>8} {'Clip%':>8}")
    print(f"  {'─'*6}─{'─'*8}─{'─'*8}─{'─'*8}─{'─'*8}─{'─'*8}")
    
    reward = 0.5
    kl = 2.0
    entropy = 4.0
    v_loss = 1.0
    clip_frac = 0.15
    
    for step in range(1, n_steps + 1):
        # Simulate gradual improvement
        reward += torch.randn(1).item() * 0.1 + 0.05
        kl += torch.randn(1).item() * 0.3
        kl = max(0.5, min(15, kl))
        entropy -= 0.02 + torch.randn(1).item() * 0.05
        entropy = max(2.0, entropy)
        v_loss *= 0.97
        v_loss += torch.randn(1).item() * 0.02
        v_loss = max(0.1, v_loss)
        clip_frac = 0.15 + torch.randn(1).item() * 0.05
        clip_frac = max(0.01, min(0.5, clip_frac))
        
        if step % 4 == 0:
            status = "✓" if (1 < kl < 10 and entropy > 2.5) else "⚠"
            print(f"  {step:>6} {reward:>8.3f} {kl:>8.2f} {entropy:>8.2f} "
                  f"{v_loss:>8.4f} {clip_frac:>7.1%} {status}")
    
    # Common failure modes
    print(f"""
  ═══ Common Failure Modes ═══
  
  1. REWARD HACKING:
     Symptoms: Reward ↑↑↑ very fast, KL ↑↑↑
     Cause: Policy exploits reward model weaknesses
     Fix: Increase β, use ensemble reward models, retrain RM
     
  2. MODE COLLAPSE:
     Symptoms: Entropy ↓↓↓, all responses look similar
     Cause: β too low, or reward too concentrated
     Fix: Increase β, add entropy bonus, lower learning rate
     
  3. LENGTH HACKING:
     Symptoms: Response length keeps growing
     Cause: Reward model biased toward longer responses
     Fix: Add length penalty, normalize by length
     
  4. KL EXPLOSION:
     Symptoms: KL > 20 and growing
     Cause: Learning rate too high, β too low
     Fix: Reduce LR, increase β, use adaptive KL controller
     
  5. NO LEARNING:
     Symptoms: Reward flat, KL ≈ 0
     Cause: LR too low, reward signal too weak
     Fix: Increase LR, check reward model quality
  
  
  ═══ RLHF Hyperparameter Recommendations ═══
  
  ┌─────────────────────┬──────────────────────────────┐
  │ Parameter           │ Recommended Value            │
  ├─────────────────────┼──────────────────────────────┤
  │ Learning rate       │ 1e-6 to 5e-6                 │
  │ KL initial β        │ 0.05 to 0.2                  │
  │ KL target           │ 2.0 to 8.0                   │
  │ PPO clip ε          │ 0.2                          │
  │ PPO epochs          │ 2-4 per batch                │
  │ Batch size          │ 32-128                       │
  │ Mini-batch size     │ 8-32                         │
  │ GAE λ               │ 0.95                         │
  │ Discount γ          │ 1.0 (common for text)        │
  │ Max grad norm       │ 1.0                          │
  │ Warmup ratio        │ 0.05                         │
  │ Total steps         │ 5K-50K (depends on data)     │
  └─────────────────────┴──────────────────────────────┘
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all RLHF training sections."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║        RLHF TRAINING — FULL PIPELINE WITH PPO                ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: PPO from scratch
    simple_ppo_from_scratch()
    
    # Section 2: TRL pipeline
    rlhf_with_trl()
    
    # Section 3: KL controller
    kl_controller()
    
    # Section 4: Reward shaping
    reward_shaping()
    
    # Section 5: Monitoring
    rlhf_monitoring()
    
    print("\n" + "=" * 65)
    print("  TRAINING MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered:
    ✓ PPO from scratch (generate → score → advantage → update)
    ✓ TRL PPOTrainer pipeline (production code pattern)
    ✓ Adaptive KL controller (maintain target KL)
    ✓ Reward shaping (normalization, clipping, whitening)
    ✓ Training monitoring (metrics, failure modes, recommendations)
    """)


if __name__ == "__main__":
    main()
