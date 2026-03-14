"""
RL Fine-Tuning Training — Task-Reward Training, RLVR, Practical Pipelines
===========================================================================

Practical training implementations:

1. Task-specific reward functions (BLEU, format, correctness)
2. RLVR — RL from Verifiable Rewards (math, code)
3. GRPO training with HuggingFace TRL
4. Composite reward training
5. Training monitoring and diagnostics

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import re
from typing import Dict, List, Tuple, Optional, Callable


# ============================================================================
# SECTION 1: TASK-SPECIFIC REWARD FUNCTIONS
# ============================================================================

def task_specific_rewards():
    """Designing and implementing reward functions for specific tasks."""
    print("=" * 70)
    print("  SECTION 1: TASK-SPECIFIC REWARD FUNCTIONS")
    print("=" * 70)
    
    print(f"""
  Different tasks need different reward signals.
  Here we implement a library of reward functions that can be
  composed for RL fine-tuning.
""")
    
    # ── 1. Math Correctness Reward ──
    class MathReward:
        """Reward for math problem solving."""
        
        def __init__(self, strict: bool = True):
            self.strict = strict
        
        def extract_answer(self, text: str) -> Optional[float]:
            """Extract numerical answer from response."""
            # Try "the answer is X" pattern
            patterns = [
                r'(?:the\s+)?answer\s+is\s+(-?\d+\.?\d*)',
                r'=\s*(-?\d+\.?\d*)\s*$',
                r'\\boxed\{(-?\d+\.?\d*)\}',
                r'(-?\d+\.?\d*)\s*$',
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        continue
            return None
        
        def __call__(self, response: str, correct: float) -> float:
            predicted = self.extract_answer(response)
            if predicted is None:
                return 0.0
            if self.strict:
                return 1.0 if abs(predicted - correct) < 1e-6 else 0.0
            else:
                # Partial credit based on closeness
                error = abs(predicted - correct)
                if error < 1e-6:
                    return 1.0
                elif error < 1:
                    return 0.5
                elif error < 10:
                    return 0.1
                return 0.0
    
    # ── 2. Code Execution Reward ──
    class CodeReward:
        """Reward for code generation tasks."""
        
        def __init__(self, test_cases: List[Dict] = None):
            self.test_cases = test_cases or []
        
        def extract_code(self, response: str) -> Optional[str]:
            """Extract code from markdown code blocks."""
            match = re.search(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
            if match:
                return match.group(1).strip()
            return response.strip()
        
        def __call__(self, response: str, test_cases: List[Dict] = None) -> float:
            """Score based on test case pass rate."""
            cases = test_cases or self.test_cases
            if not cases:
                return 0.0
            
            code = self.extract_code(response)
            if not code:
                return 0.0
            
            passed = 0
            for case in cases:
                try:
                    # In production: use sandboxed execution
                    local_ns = {}
                    exec(code, {}, local_ns)
                    func = list(local_ns.values())[-1]
                    result = func(case['input'])
                    if result == case['expected']:
                        passed += 1
                except Exception:
                    pass
            
            return passed / len(cases)
    
    # ── 3. Format Compliance Reward ──
    class FormatReward:
        """Reward for following specified output format."""
        
        def __init__(self, required_sections: List[str] = None,
                     max_length: int = 500, min_length: int = 10):
            self.required_sections = required_sections or []
            self.max_length = max_length
            self.min_length = min_length
        
        def __call__(self, response: str) -> float:
            score = 0.0
            total_checks = 0
            
            # Length check
            total_checks += 1
            if self.min_length <= len(response) <= self.max_length:
                score += 1.0
            elif len(response) > 0:
                score += 0.3
            
            # Section presence
            for section in self.required_sections:
                total_checks += 1
                if section.lower() in response.lower():
                    score += 1.0
            
            # Ends properly
            total_checks += 1
            if response.strip().endswith(('.', '!', '?', '```')):
                score += 1.0
            
            return score / total_checks if total_checks > 0 else 0.0
    
    # ── 4. Composite Reward ──
    class CompositeReward:
        """Weighted combination of multiple reward signals."""
        
        def __init__(self, rewards: List[Tuple[str, Callable, float]]):
            """
            rewards: list of (name, reward_fn, weight) tuples
            """
            self.rewards = rewards
            total_weight = sum(w for _, _, w in rewards)
            self.rewards = [(n, fn, w/total_weight) for n, fn, w in rewards]
        
        def __call__(self, response: str, **kwargs) -> Dict:
            total = 0.0
            breakdown = {}
            
            for name, fn, weight in self.rewards:
                try:
                    r = fn(response, **{k: v for k, v in kwargs.items()
                                        if k in fn.__code__.co_varnames})
                except TypeError:
                    r = fn(response)
                
                breakdown[name] = r
                total += weight * r
            
            breakdown['total'] = total
            return breakdown
    
    # Demonstrate reward functions
    print(f"  ── Math Reward ──\n")
    math_r = MathReward(strict=False)
    
    test_cases = [
        ("Let me solve: 2+3 = 5. The answer is 5.", 5.0),
        ("I think it's about 4.9", 5.0),
        ("The result equals \\boxed{42}", 42.0),
        ("I don't know the answer", 5.0),
        ("After careful calculation, the answer is 100", 5.0),
    ]
    
    for response, correct in test_cases:
        r = math_r(response, correct)
        extracted = math_r.extract_answer(response)
        print(f"    r={r:.1f}  extracted={extracted}  \"{response[:50]}\"")
    
    print(f"\n  ── Format Reward ──\n")
    format_r = FormatReward(
        required_sections=["step 1", "therefore"],
        min_length=30, max_length=200
    )
    
    format_tests = [
        "Step 1: Calculate. Step 2: Verify. Therefore, the answer is 42.",
        "42",
        "Step 1: We know that x=5. Therefore, 2x=10.",
    ]
    
    for response in format_tests:
        r = format_r(response)
        print(f"    r={r:.2f}  \"{response[:55]}\"")
    
    print(f"\n  ── Composite Reward ──\n")
    composite = CompositeReward([
        ("correctness", lambda resp, correct=5.0: math_r(resp, correct), 0.6),
        ("format", format_r, 0.3),
        ("length", lambda resp: min(len(resp) / 50, 1.0), 0.1),
    ])
    
    test_response = "Step 1: Calculate 2+3. Therefore, the answer is 5."
    result = composite(test_response)
    print(f"    Response: \"{test_response}\"")
    for k, v in result.items():
        print(f"    {k:>15}: {v:.3f}")


# ============================================================================
# SECTION 2: RLVR — RL FROM VERIFIABLE REWARDS
# ============================================================================

def rlvr_training():
    """
    RL from Verifiable Rewards — training with binary correctness signals.
    Key technique from DeepSeek-R1.
    """
    print("\n\n" + "=" * 70)
    print("  SECTION 2: RLVR — RL FROM VERIFIABLE REWARDS")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  RLVR: Train with BINARY correctness rewards on verifiable tasks.
  
  No reward model, no human preferences — just:
    R = 1 if answer is correct
    R = 0 if answer is incorrect
  
  Tasks suitable for RLVR:
  • Math: check if final answer matches ground truth
  • Code: run tests, check pass/fail
  • Logic: verify logical consistency  
  • Factual QA: check against knowledge base
  
  DeepSeek-R1 used RLVR + GRPO to develop reasoning without
  any supervised chain-of-thought data!
""")
    
    # Simulate RLVR training for a math-like task
    vocab_size = 40
    d_model = 48
    prompt_len = 4
    gen_len = 8
    group_size = 8
    
    class MiniLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.rnn = nn.GRU(d_model, d_model, num_layers=2, batch_first=True)
            self.head = nn.Linear(d_model, vocab_size)
        
        def forward(self, x):
            h, _ = self.rnn(self.embed(x))
            return self.head(h)
        
        def get_token_log_probs(self, seq, pl):
            logits = self(seq)
            lp = F.log_softmax(logits[:, pl-1:-1, :], dim=-1)
            return lp.gather(2, seq[:, pl:].unsqueeze(-1)).squeeze(-1)
        
        @torch.no_grad()
        def generate(self, prompts, n_tokens, temp=1.0):
            self.eval()
            ids = prompts.clone()
            for _ in range(n_tokens):
                logits = self(ids)[:, -1, :] / temp
                ids = torch.cat([ids, torch.multinomial(F.softmax(logits, -1), 1)], 1)
            self.train()
            return ids
    
    policy = MiniLM()
    ref_policy = copy.deepcopy(policy)
    for p in ref_policy.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    # Verifiable task: sequence must sum to target (encoded in prompt)
    def create_math_batch(batch_size):
        """Create batch of verifiable math-like problems."""
        targets = torch.randint(3, 15, (batch_size, 1))
        # Encode target in prompt: [target, target*2 % V, 0, 0]
        prompts = torch.cat([
            targets % vocab_size,
            (targets * 2) % vocab_size,
            torch.zeros(batch_size, 1, dtype=torch.long),
            torch.ones(batch_size, 1, dtype=torch.long),
        ], dim=1)
        return prompts, targets.squeeze(1)
    
    def verify(response_tokens: torch.Tensor, target: int) -> float:
        """Binary verification: does the response encode the right answer?"""
        # "Correct" if last token equals target (mod vocab_size)
        last_token = response_tokens[-1].item()
        return 1.0 if (last_token % vocab_size) == (target % vocab_size) else 0.0
    
    # GRPO + RLVR training
    clip_eps = 0.2
    kl_coeff = 0.02
    n_steps = 40
    n_prompts_per_step = 6
    
    print(f"\n  Config: GRPO + binary verification reward")
    print(f"  G={group_size}, ε={clip_eps}, β_kl={kl_coeff}")
    print(f"\n  {'Step':>6} │ {'Accuracy':>8} │ {'Avg R':>7} │ {'KL':>7} │ "
          f"{'Pass Rate':>9}")
    print(f"  {'─'*6}─┼─{'─'*8}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*9}")
    
    for step in range(n_steps):
        prompts, targets = create_math_batch(n_prompts_per_step)
        
        step_rewards = []
        step_kl = []
        step_correct = 0
        step_total = 0
        
        for p_idx in range(n_prompts_per_step):
            prompt = prompts[p_idx:p_idx+1].expand(group_size, -1)
            target = targets[p_idx].item()
            
            # Generate group
            sequences = policy.generate(prompt, gen_len, temp=0.9)
            
            # Binary verification reward
            rewards = torch.tensor([
                verify(seq[prompt_len:], target) for seq in sequences
            ])
            
            step_correct += rewards.sum().item()
            step_total += group_size
            step_rewards.extend(rewards.tolist())
            
            # Skip update if all same reward (no contrast)
            if rewards.std() < 1e-8:
                continue
            
            # GRPO advantages
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            # Compute losses
            old_lp = policy.get_token_log_probs(sequences, prompt_len).detach()
            new_lp = policy.get_token_log_probs(sequences, prompt_len)
            ref_lp = ref_policy.get_token_log_probs(sequences, prompt_len)
            
            ratio = torch.exp(new_lp - old_lp)
            adv_exp = adv.unsqueeze(1).expand_as(ratio)
            
            s1 = ratio * adv_exp
            s2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv_exp
            policy_loss = -torch.min(s1, s2).mean()
            
            kl = (new_lp - ref_lp).mean()
            step_kl.append(kl.item())
            
            loss = policy_loss + kl_coeff * kl
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        
        if (step + 1) % 8 == 0:
            accuracy = step_correct / step_total
            avg_r = sum(step_rewards) / len(step_rewards)
            avg_kl = sum(step_kl) / len(step_kl) if step_kl else 0
            print(f"  {step+1:>6} │ {accuracy:>7.1%} │ {avg_r:>7.3f} │ "
                  f"{avg_kl:>7.4f} │ {accuracy:>8.1%}")
    
    del policy, ref_policy
    
    print(f"""
  RLVR KEY PROPERTIES:
  ✓ Perfect reward signal (ground truth verification)
  ✓ No reward model to train or maintain
  ✓ Binary reward is sufficient with GRPO (group contrast)
  ✓ Forces model to develop internal reasoning strategies
  
  WHY IT WORKS:
  • Group sampling provides the contrast signal
  • Some responses in group will be correct, others wrong
  • GRPO pushes correct responses up, incorrect down
  • Over time, model learns which strategies lead to correctness
""")


# ============================================================================
# SECTION 3: GRPO WITH TRL (HuggingFace Pattern)
# ============================================================================

def grpo_trl_pattern():
    """How to use GRPO with HuggingFace TRL library."""
    print("\n\n" + "=" * 70)
    print("  SECTION 3: GRPO WITH TRL (HuggingFace)")
    print("=" * 70)
    
    print(f"""
  TRL (Transformer Reinforcement Learning) now supports GRPO
  natively via the GRPOTrainer class.
  
  This section shows the EXACT code pattern for production use.
""")
    
    # Full code pattern (documented, ready to use)
    grpo_code = '''
# ═══════════════════════════════════════════════════════════════
# GRPO Training with TRL — Production Code Pattern
# ═══════════════════════════════════════════════════════════════

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, Dataset
import torch
import re

# ── Step 1: Load Model & Tokenizer ──
model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # or any causal LM
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# ── Step 2: Prepare Dataset ──
# GRPO expects a dataset with a "prompt" column
# Each prompt will generate G responses

def format_prompt(example):
    """Format math problems as chat messages."""
    return {
        "prompt": [
            {"role": "user", "content": f"Solve: {example['question']}\\n"
             "Put your final answer in \\\\boxed{{}}."}
        ]
    }

# Example: GSM8K math dataset
dataset = load_dataset("openai/gsm8k", "main", split="train[:1000]")
dataset = dataset.map(format_prompt)

# ── Step 3: Define Reward Function ──
def math_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """
    Reward function for math problems.
    
    Args:
        completions: List of model-generated responses
        **kwargs: Contains 'prompts' and any other dataset columns
    
    Returns:
        List of reward scores (one per completion)
    """
    rewards = []
    
    # Get ground truth answers from kwargs
    answers = kwargs.get("answer", [None] * len(completions))
    
    for completion, answer in zip(completions, answers):
        # Extract answer from \\boxed{}
        match = re.search(r'\\\\boxed\\{([^}]+)\\}', completion)
        if match and answer:
            predicted = match.group(1).strip()
            # Extract number from ground truth
            gt_match = re.search(r'####\\s*(.+)', str(answer))
            gt = gt_match.group(1).strip() if gt_match else str(answer)
            
            reward = 1.0 if predicted == gt else 0.0
        else:
            reward = 0.0
        
        rewards.append(reward)
    
    return rewards

# ── Step 4: Optional Format Reward ──
def format_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """Reward for following the \\\\boxed{} format."""
    rewards = []
    for completion in completions:
        if re.search(r'\\\\boxed\\{[^}]+\\}', completion):
            rewards.append(0.5)  # Has boxed answer
        else:
            rewards.append(0.0)
    return rewards

# ── Step 5: Configure GRPO Training ──
training_args = GRPOConfig(
    output_dir="./grpo_math_output",
    
    # GRPO-specific
    num_generations=8,            # G: group size
    
    # Training
    learning_rate=5e-6,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    max_steps=500,
    
    # Generation
    max_completion_length=512,
    temperature=0.9,
    
    # Optimization
    bf16=True,
    gradient_checkpointing=True,
    
    # Logging
    logging_steps=10,
    save_steps=100,
    report_to="wandb",
)

# ── Step 6: Create Trainer ──
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=dataset,
    reward_funcs=[math_reward_fn, format_reward_fn],
    # Reward weights (optional, default equal)
    # reward_weights=[0.8, 0.2],  
)

# ── Step 7: Train! ──
trainer.train()

# ── Step 8: Save ──
trainer.save_model("./grpo_math_final")

# ═══════════════════════════════════════════════════════════════
# GRPO with LoRA (memory-efficient)
# ═══════════════════════════════════════════════════════════════

from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# Just add peft_config to GRPOTrainer
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=dataset,
    reward_funcs=[math_reward_fn],
    peft_config=lora_config,
)

trainer.train()
'''
    
    print(grpo_code)
    
    print(f"""
  ═══ Key GRPOConfig Parameters ═══
  
  ┌─────────────────────────┬──────────┬────────────────────────────┐
  │ Parameter               │ Default  │ Description                │
  ├─────────────────────────┼──────────┼────────────────────────────┤
  │ num_generations         │ 8        │ G: group size per prompt   │
  │ max_completion_length   │ 256      │ Max response tokens        │
  │ temperature             │ 0.9      │ Sampling temperature       │
  │ beta                    │ 0.04     │ KL penalty coefficient     │
  │ epsilon                 │ 0.2      │ PPO clip range             │
  │ num_iterations          │ 1        │ PPO epochs per batch       │
  │ loss_type               │ "grpo"   │ "grpo" or "bnpo"           │
  └─────────────────────────┴──────────┴────────────────────────────┘
  
  REWARD FUNCTION API:
  • Input: completions (list[str]) + **kwargs (dataset columns)
  • Output: list[float] (one reward per completion)
  • Multiple reward functions → weighted sum
""")


# ============================================================================
# SECTION 4: TRAINING MONITORING AND DIAGNOSTICS
# ============================================================================

def training_diagnostics():
    """Monitoring RL fine-tuning: what to track and how to debug."""
    print("\n\n" + "=" * 70)
    print("  SECTION 4: TRAINING MONITORING & DIAGNOSTICS")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  RL training is notoriously unstable. Here's what to monitor
  and how to diagnose common problems.
  
  
  ═══ Key Metrics to Track ═══
  
  ┌──────────────────────┬──────────────────────────────────────┐
  │ Metric               │ What It Tells You                    │
  ├──────────────────────┼──────────────────────────────────────┤
  │ Mean reward          │ Is the model improving?              │
  │ Reward std           │ Response diversity                   │
  │ KL divergence        │ How far from reference policy        │
  │ Clip fraction        │ How often PPO clipping activates     │
  │ Policy entropy       │ Exploration level                    │
  │ Advantage mean/std   │ Training signal quality              │
  │ Log prob ratio       │ Update magnitude                     │
  │ Value loss (if PPO)  │ Baseline accuracy                    │
  │ Grad norm            │ Training stability                   │
  │ Response length      │ Length gaming detection               │
  └──────────────────────┴──────────────────────────────────────┘
""")
    
    # Simulate a training run with diagnostics
    
    class TrainingMonitor:
        """Track RL training diagnostics."""
        
        def __init__(self, window_size: int = 50):
            self.window_size = window_size
            self.metrics: Dict[str, List[float]] = {
                'reward_mean': [], 'reward_std': [],
                'kl': [], 'clip_frac': [],
                'entropy': [], 'grad_norm': [],
                'advantage_mean': [], 'advantage_std': [],
                'response_length': [],
            }
        
        def log(self, **kwargs):
            for k, v in kwargs.items():
                if k in self.metrics:
                    self.metrics[k].append(v)
        
        def diagnose(self) -> List[str]:
            """Auto-diagnose training issues."""
            issues = []
            
            # Check reward stagnation
            if len(self.metrics['reward_mean']) >= 20:
                recent = self.metrics['reward_mean'][-20:]
                if max(recent) - min(recent) < 0.01:
                    issues.append("⚠ REWARD STAGNATION: Mean reward flat for 20 steps")
            
            # Check KL explosion
            if self.metrics['kl'] and self.metrics['kl'][-1] > 15:
                issues.append("🔴 KL EXPLOSION: KL > 15, policy diverging from reference")
            
            # Check clip fraction
            if self.metrics['clip_frac'] and self.metrics['clip_frac'][-1] > 0.3:
                issues.append("⚠ HIGH CLIP FRACTION: >30% clipped, consider smaller LR")
            
            # Check entropy collapse
            if len(self.metrics['entropy']) >= 10:
                if self.metrics['entropy'][-1] < 0.1 * self.metrics['entropy'][0]:
                    issues.append("🔴 ENTROPY COLLAPSE: Model is mode-collapsing")
            
            # Check advantage stats
            if self.metrics['advantage_std'] and self.metrics['advantage_std'][-1] < 0.01:
                issues.append("⚠ LOW ADVANTAGE VARIANCE: Training signal too weak")
            
            # Check gradient explosion
            if self.metrics['grad_norm'] and self.metrics['grad_norm'][-1] > 100:
                issues.append("🔴 GRADIENT EXPLOSION: Grad norm > 100")
            
            # Check length gaming
            if len(self.metrics['response_length']) >= 20:
                early = sum(self.metrics['response_length'][:10]) / 10
                late = sum(self.metrics['response_length'][-10:]) / 10
                if late > 2 * early:
                    issues.append("⚠ LENGTH GAMING: Response length doubled")
            
            if not issues:
                issues.append("✓ Training looks healthy")
            
            return issues
        
        def summary(self):
            """Print training summary."""
            print(f"\n  ── Training Summary ──\n")
            for key, values in self.metrics.items():
                if values:
                    recent = values[-10:]
                    print(f"    {key:>20}: {sum(recent)/len(recent):>8.4f} "
                          f"(last 10 avg)")
            
            print(f"\n  ── Diagnostics ──")
            for issue in self.diagnose():
                print(f"    {issue}")
    
    # Simulate training with the monitor
    monitor = TrainingMonitor()
    
    print(f"\n  ── Simulated Training Run ──\n")
    
    import random
    random.seed(42)
    
    for step in range(60):
        # Simulate metrics
        base_reward = 0.3 + 0.4 * min(step / 40, 1.0)
        entropy = max(2.5 - 0.03 * step, 0.5)
        
        monitor.log(
            reward_mean=base_reward + random.gauss(0, 0.05),
            reward_std=max(0.15 - 0.001 * step, 0.05) + random.gauss(0, 0.01),
            kl=0.5 + 0.15 * step + random.gauss(0, 0.3),
            clip_frac=min(0.05 + 0.003 * step, 0.25) + random.gauss(0, 0.02),
            entropy=entropy + random.gauss(0, 0.1),
            grad_norm=5.0 + random.gauss(0, 2.0),
            advantage_mean=random.gauss(0, 0.05),
            advantage_std=0.8 + random.gauss(0, 0.1),
            response_length=50 + 0.5 * step + random.gauss(0, 5),
        )
    
    monitor.summary()
    
    print(f"""
  ═══ Common Issues & Fixes ═══
  
  ┌─────────────────────┬──────────────────────────────────────┐
  │ Problem             │ Fix                                  │
  ├─────────────────────┼──────────────────────────────────────┤
  │ Reward stagnation   │ • Increase group size (G)            │
  │                     │ • Increase temperature               │
  │                     │ • Check reward function range         │
  ├─────────────────────┼──────────────────────────────────────┤
  │ KL explosion        │ • Increase KL coefficient β          │
  │                     │ • Decrease learning rate              │
  │                     │ • Reset to reference policy           │
  ├─────────────────────┼──────────────────────────────────────┤
  │ Entropy collapse    │ • Add entropy bonus to reward         │
  │                     │ • Increase temperature                │
  │                     │ • Decrease learning rate              │
  ├─────────────────────┼──────────────────────────────────────┤
  │ Length gaming       │ • Add length penalty to reward        │
  │                     │ • Use length-normalized rewards       │
  │                     │ • Cap max response length             │
  ├─────────────────────┼──────────────────────────────────────┤
  │ Reward hacking      │ • Diversify reward sources            │
  │                     │ • Use reward ensembles                │
  │                     │ • Add format constraints              │
  ├─────────────────────┼──────────────────────────────────────┤
  │ High gradient norm  │ • Reduce learning rate                │
  │                     │ • Increase gradient clipping           │
  │                     │ • Check for NaN in rewards            │
  └─────────────────────┴──────────────────────────────────────┘
""")


# ============================================================================
# SECTION 5: END-TO-END TRAINING EXAMPLE
# ============================================================================

def end_to_end_example():
    """Complete end-to-end GRPO training with all components."""
    print("\n\n" + "=" * 70)
    print("  SECTION 5: END-TO-END GRPO TRAINING")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  Complete training pipeline:
  1. Model initialization + reference copy
  2. Dataset with verifiable problems
  3. Composite reward (correctness + format)
  4. GRPO training loop with monitoring
  5. Evaluation on held-out problems
""")
    
    vocab_size = 40
    d_model = 48
    prompt_len = 4
    gen_len = 8
    
    class PolicyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.rnn = nn.GRU(d_model, d_model, num_layers=2, batch_first=True)
            self.head = nn.Linear(d_model, vocab_size)
        
        def forward(self, x):
            h, _ = self.rnn(self.embed(x))
            return self.head(h)
        
        def get_token_lp(self, seq, pl):
            logits = self(seq)
            lp = F.log_softmax(logits[:, pl-1:-1, :], dim=-1)
            return lp.gather(2, seq[:, pl:].unsqueeze(-1)).squeeze(-1)
        
        @torch.no_grad()
        def generate(self, prompts, n, temp=0.9):
            self.eval()
            ids = prompts.clone()
            for _ in range(n):
                logits = self(ids)[:, -1, :] / temp
                ids = torch.cat([ids, torch.multinomial(F.softmax(logits, -1), 1)], 1)
            self.train()
            return ids
    
    # ── Setup ──
    policy = PolicyModel()
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad = False
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # ── Problem Generator ──
    def make_problems(n):
        targets = torch.randint(2, 12, (n,))
        prompts = torch.stack([
            torch.tensor([t.item() % vocab_size, (t.item()*3) % vocab_size, 0, 1])
            for t in targets
        ])
        return prompts, targets
    
    # ── Composite Reward ──
    def compute_reward(tokens, target):
        t = tokens.tolist()
        # Correctness: last token matches target
        correct = 1.0 if t[-1] % vocab_size == target % vocab_size else 0.0
        # Diversity bonus
        diversity = len(set(t)) / len(t) * 0.3
        # No-repeat bonus
        no_repeat = (1 - sum(1 for i in range(1, len(t)) if t[i] == t[i-1]) 
                     / max(len(t)-1, 1)) * 0.2
        return correct * 0.5 + diversity + no_repeat
    
    # ── Training Config ──
    G = 8
    clip_eps = 0.2
    kl_coeff = 0.03
    n_steps = 50
    n_prompts = 6
    
    # ── Train ──
    print(f"\n  {'Step':>6} │ {'Reward':>7} │ {'Correct%':>8} │ {'KL':>7} │ "
          f"{'Clip%':>6} │ {'LR':>9}")
    print(f"  {'─'*6}─┼─{'─'*7}─┼─{'─'*8}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*9}")
    
    for step in range(n_steps):
        prompts, targets = make_problems(n_prompts)
        
        step_rewards = []
        step_correct = []
        step_kl = []
        step_clip = []
        
        for idx in range(n_prompts):
            prompt_g = prompts[idx:idx+1].expand(G, -1)
            target = targets[idx].item()
            
            seqs = policy.generate(prompt_g, gen_len, temp=0.9)
            rewards = torch.tensor([
                compute_reward(s[prompt_len:], target) for s in seqs
            ])
            
            correct = torch.tensor([
                1.0 if s[prompt_len:][-1].item() % vocab_size == target % vocab_size 
                else 0.0 for s in seqs
            ])
            
            step_rewards.extend(rewards.tolist())
            step_correct.extend(correct.tolist())
            
            if rewards.std() < 1e-8:
                continue
            
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            old_lp = policy.get_token_lp(seqs, prompt_len).detach()
            new_lp = policy.get_token_lp(seqs, prompt_len)
            ref_lp = ref.get_token_lp(seqs, prompt_len)
            
            ratio = torch.exp(new_lp - old_lp)
            adv_exp = adv.unsqueeze(1).expand_as(ratio)
            
            s1 = ratio * adv_exp
            s2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv_exp
            p_loss = -torch.min(s1, s2).mean()
            
            kl = (new_lp - ref_lp).mean()
            clip_frac = ((ratio - 1).abs() > clip_eps).float().mean()
            
            step_kl.append(kl.item())
            step_clip.append(clip_frac.item())
            
            loss = p_loss + kl_coeff * kl
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        if (step + 1) % 10 == 0:
            avg_r = sum(step_rewards) / len(step_rewards)
            acc = sum(step_correct) / len(step_correct)
            avg_kl = sum(step_kl) / len(step_kl) if step_kl else 0
            avg_clip = sum(step_clip) / len(step_clip) if step_clip else 0
            lr = scheduler.get_last_lr()[0]
            print(f"  {step+1:>6} │ {avg_r:>7.3f} │ {acc:>7.1%} │ "
                  f"{avg_kl:>7.4f} │ {avg_clip:>5.1%} │ {lr:>9.6f}")
    
    # ── Evaluation ──
    print(f"\n  ── Evaluation on 100 new problems ──")
    eval_prompts, eval_targets = make_problems(100)
    
    total_reward = 0
    total_correct = 0
    
    for i in range(100):
        with torch.no_grad():
            seq = policy.generate(eval_prompts[i:i+1], gen_len, temp=0.3)
        r = compute_reward(seq[0][prompt_len:], eval_targets[i].item())
        c = 1 if seq[0][-1].item() % vocab_size == eval_targets[i].item() % vocab_size else 0
        total_reward += r
        total_correct += c
    
    print(f"  Avg reward:   {total_reward/100:.3f}")
    print(f"  Accuracy:     {total_correct/100:.1%}")
    
    del policy, ref
    
    print(f"""
  ═══ Training Pipeline Summary ═══
  
  1. Initialize policy + frozen reference
  2. For each step:
     a. Sample prompts
     b. Generate G responses per prompt (temp=0.9)
     c. Score with composite reward
     d. Normalize advantages within group (GRPO)
     e. Clipped surrogate loss + KL penalty
     f. Gradient update with clipping
  3. Evaluate with lower temperature (greedy/low temp)
  
  KEY PRACTICAL TIPS:
  • Start with low KL coefficient, increase if needed
  • Monitor clip fraction: 10-20% is healthy
  • Use cosine LR schedule for stability
  • Generate at temp=0.9, evaluate at temp=0.1-0.3
  • Skip updates when all rewards in group are identical
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  RL FINE-TUNING TRAINING — RLVR, GRPO, PRACTICAL PIPELINES       ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    task_specific_rewards()
    rlvr_training()
    grpo_trl_pattern()
    training_diagnostics()
    end_to_end_example()
    
    print("\n" + "=" * 70)
    print("  TRAINING MODULE COMPLETE")
    print("=" * 70)
    print("""
    Covered:
    ✓ Task-specific reward functions (math, code, format, composite)
    ✓ RLVR — binary verification rewards for math/code
    ✓ GRPO with TRL — production code pattern
    ✓ Training monitoring and diagnostic tools
    ✓ End-to-end GRPO training pipeline with evaluation
    """)


if __name__ == "__main__":
    main()
