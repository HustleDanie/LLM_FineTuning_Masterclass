"""
DPO Training — Production DPO with TRL
========================================

Complete DPO training implementations:

1. DPOWithTRL
   - Full DPO training using TRL's DPOTrainer
   - Configuration and best practices

2. OnlineDPO
   - Online DPO: generate new responses during training
   - Iterative DPO for continuous improvement

3. DPODataPreparation
   - Preparing datasets from common formats
   - Using HuggingFace preference datasets

4. DPOHyperparameters
   - Systematic hyperparameter tuning
   - What matters most in DPO training

5. DPOEvaluation
   - Evaluating DPO-trained models
   - Win-rate estimation and reward analysis

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional


# ============================================================================
# SECTION 1: DPO WITH TRL
# ============================================================================

def dpo_with_trl():
    """Full DPO training using TRL's DPOTrainer."""
    print("=" * 70)
    print("  SECTION 1: DPO WITH TRL")
    print("=" * 70)
    
    print(f"""
  ═══ TRL DPOTrainer — The Standard Way ═══
  
  TRL's DPOTrainer handles all the complexity:
  • Log-prob computation for policy and reference
  • Proper loss with label masking
  • Mixed precision, gradient accumulation
  • Logging, checkpointing, evaluation
  
  ── Standard Code Pattern ──
  
  ```python
  from trl import DPOConfig, DPOTrainer
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from datasets import load_dataset
  
  # 1. Load model and tokenizer
  model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
  tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
  tokenizer.pad_token = tokenizer.eos_token
  
  # 2. Optional: Load reference model separately
  #    (if None with PEFT, uses base model as reference)
  ref_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
  
  # 3. Load preference dataset
  dataset = load_dataset("Anthropic/hh-rlhf", split="train")
  # Must have columns: "prompt", "chosen", "rejected"
  # OR: "chosen" and "rejected" as full conversations
  
  # 4. Configure DPO training
  training_args = DPOConfig(
      output_dir="./dpo_output",
      
      # ─── DPO-specific ───
      beta=0.1,                          # KL constraint strength
      loss_type="sigmoid",               # "sigmoid" (standard), "hinge", "ipo"
      label_smoothing=0.0,               # >0 for cDPO (noisy labels)
      
      # ─── Training ───
      learning_rate=5e-7,                # Low LR for stability
      per_device_train_batch_size=4,
      gradient_accumulation_steps=4,     # Effective batch = 16
      num_train_epochs=1,                # Usually 1-3 epochs
      max_steps=-1,
      warmup_ratio=0.1,
      
      # ─── Optimization ───
      bf16=True,                         # Mixed precision
      gradient_checkpointing=True,       # Save memory
      max_grad_norm=1.0,
      
      # ─── Data ───
      max_length=512,                    # Total sequence length
      max_prompt_length=256,             # Max prompt portion
      
      # ─── Logging ───
      logging_steps=10,
      eval_strategy="steps",
      eval_steps=100,
      save_strategy="steps",
      save_steps=500,
  )
  
  # 5. Create trainer
  trainer = DPOTrainer(
      model=model,
      ref_model=ref_model,               # None if using PEFT
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      tokenizer=tokenizer,
  )
  
  # 6. Train!
  trainer.train()
  
  # 7. Save
  trainer.save_model("./dpo_final")
  ```
""")
    
    # Run a minimal version that actually executes
    print(f"  ── Minimal Running Example ──\n")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    policy = AutoModelForCausalLM.from_pretrained(model_name)
    reference = AutoModelForCausalLM.from_pretrained(model_name)
    for p in reference.parameters():
        p.requires_grad = False
    
    # Preference data in TRL format
    preference_data = [
        {"prompt": "What is AI?",
         "chosen": " AI is artificial intelligence - computer systems designed to perform tasks that normally require human intelligence.",
         "rejected": " AI is when computers become alive and take over the world."},
        {"prompt": "How does rain form?",
         "chosen": " Water evaporates, rises as vapor, cools in the atmosphere to form clouds, then falls as precipitation.",
         "rejected": " Rain is water that falls from the sky because gravity pulls it down. Nobody knows where it comes from."},
        {"prompt": "Why learn math?",
         "chosen": " Math develops logical thinking, problem-solving skills, and is essential for science, engineering, and technology careers.",
         "rejected": " Math is useless. You won't need it after school."},
    ]
    
    # Manual DPO training (simulating what DPOTrainer does internally)
    def compute_logprobs(model, text, tokenizer, max_len=60):
        enc = tokenizer(text, return_tensors="pt", max_length=max_len,
                       truncation=True, padding="max_length")
        outputs = model(input_ids=enc["input_ids"],
                       attention_mask=enc["attention_mask"])
        logits = outputs.logits[:, :-1, :]
        labels = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        return (token_lp * mask).sum(dim=-1)
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-6)
    beta = 0.1
    
    print(f"  Model: {model_name}, β={beta}, lr=5e-6")
    print(f"\n  {'Step':>6} │ {'Loss':>8} │ {'Accuracy':>8}")
    print(f"  {'─'*6}─┼─{'─'*8}─┼─{'─'*8}")
    
    for step in range(6):
        total_loss = 0
        correct = 0
        
        for pair in preference_data:
            chosen_text = pair["prompt"] + pair["chosen"]
            rejected_text = pair["prompt"] + pair["rejected"]
            
            pi_c = compute_logprobs(policy, chosen_text, tokenizer)
            pi_r = compute_logprobs(policy, rejected_text, tokenizer)
            with torch.no_grad():
                ref_c = compute_logprobs(reference, chosen_text, tokenizer)
                ref_r = compute_logprobs(reference, rejected_text, tokenizer)
            
            logits = beta * ((pi_c - ref_c) - (pi_r - ref_r))
            loss = -F.logsigmoid(logits).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            correct += (logits > 0).float().sum().item()
        
        n = len(preference_data)
        print(f"  {step+1:>6} │ {total_loss/n:>8.4f} │ {correct/n:>7.1%}")
    
    print(f"\n  ✓ Training complete!")
    del policy, reference


# ============================================================================
# SECTION 2: ONLINE DPO
# ============================================================================

def online_dpo():
    """Online DPO: generate new responses during training."""
    print("\n\n" + "=" * 70)
    print("  SECTION 2: ONLINE & ITERATIVE DPO")
    print("=" * 70)
    
    print(f"""
  ═══ The Problem with Offline DPO ═══
  
  Standard DPO is OFFLINE:
  • Trained on a static preference dataset
  • Preferences were collected from a different model (e.g., SFT model)
  • Distribution shift: policy learns from data it didn't generate
  
  This can lead to:
  • Overfitting to specific response patterns
  • Poor generalization to its own generation style
  • Reward hacking on formatting rather than substance
  
  
  ═══ Online DPO ═══
  
  Generate fresh responses during training:
  
    for each iteration:
      1. Sample prompts from dataset
      2. Generate N responses per prompt using CURRENT policy
      3. Score responses with reward model (or AI judge)
      4. Pair best vs worst as chosen/rejected
      5. Run DPO update on fresh pairs
  
  Benefits:
  ✓ No distribution shift (training on own generations)
  ✓ Continuously improving data quality
  ✓ Better exploration of response space
  
  
  ═══ Iterative DPO ═══
  
  Alternate between data collection and training:
  
    Round 1:
      • Generate with policy_v0 → collect preferences → DPO → policy_v1
    Round 2:
      • Generate with policy_v1 → collect preferences → DPO → policy_v2
    Round 3:
      • Generate with policy_v2 → collect preferences → DPO → policy_v3
    ...
  
  This is similar to RLHF's online nature but simpler!
""")
    
    # Demonstrate online DPO concept
    torch.manual_seed(42)
    
    print(f"  ── Simulated Online DPO ──\n")
    
    vocab_size = 30
    d_model = 32
    seq_len = 6
    
    class SimpleLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.rnn = nn.GRU(d_model, d_model, batch_first=True)
            self.head = nn.Linear(d_model, vocab_size)
        
        def forward(self, x):
            h, _ = self.rnn(self.embed(x))
            return self.head(h)
        
        def get_logprob(self, x):
            logits = self(x)
            lp = F.log_softmax(logits[:, :-1, :], dim=-1)
            return lp.gather(2, x[:, 1:].unsqueeze(-1)).squeeze(-1).sum(-1)
        
        def generate(self, prompt, n_tokens=4):
            ids = prompt.clone()
            for _ in range(n_tokens):
                logits = self(ids)[:, -1, :]
                next_t = torch.multinomial(F.softmax(logits, dim=-1), 1)
                ids = torch.cat([ids, next_t], dim=-1)
            return ids
    
    # Reward: prefer sequences with more variety
    def reward_fn(seq):
        return len(set(seq.tolist())) / len(seq)
    
    policy = SimpleLM()
    reference = SimpleLM()
    reference.load_state_dict(policy.state_dict())
    for p in reference.parameters():
        p.requires_grad = False
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)
    beta = 0.1
    
    print(f"  {'Round':>6} │ {'Avg Reward':>10} │ {'DPO Loss':>10} │ {'Acc':>6}")
    print(f"  {'─'*6}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*6}")
    
    for round_num in range(10):
        # Step 1: Generate responses with current policy
        n_prompts = 16
        n_responses = 4  # Generate N per prompt
        prompts = torch.randint(0, vocab_size, (n_prompts, 2))
        
        round_rewards = []
        round_loss = 0
        round_acc = 0
        n_updates = 0
        
        for i in range(n_prompts):
            prompt = prompts[i:i+1]
            
            # Generate multiple responses
            responses = []
            rewards = []
            for _ in range(n_responses):
                with torch.no_grad():
                    gen = policy.generate(prompt, n_tokens=4)
                resp = gen[:, 2:]  # Response part
                r = reward_fn(resp[0])
                responses.append(gen)
                rewards.append(r)
            
            round_rewards.extend(rewards)
            
            # Pair best vs worst
            best_idx = max(range(len(rewards)), key=lambda j: rewards[j])
            worst_idx = min(range(len(rewards)), key=lambda j: rewards[j])
            
            if best_idx == worst_idx:
                continue  # Skip if all same
            
            chosen = responses[best_idx]
            rejected = responses[worst_idx]
            
            # DPO update
            pi_c = policy.get_logprob(chosen)
            pi_r = policy.get_logprob(rejected)
            with torch.no_grad():
                ref_c = reference.get_logprob(chosen)
                ref_r = reference.get_logprob(rejected)
            
            logits = beta * ((pi_c - ref_c) - (pi_r - ref_r))
            loss = -F.logsigmoid(logits).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            round_loss += loss.item()
            round_acc += (logits > 0).float().mean().item()
            n_updates += 1
        
        if n_updates > 0 and (round_num + 1) % 2 == 0:
            avg_r = sum(round_rewards) / len(round_rewards)
            print(f"  {round_num+1:>6} │ {avg_r:>10.4f} │ "
                  f"{round_loss/n_updates:>10.4f} │ "
                  f"{round_acc/n_updates:>5.1%}")
    
    print(f"\n  ✓ Online DPO helps the policy improve on its OWN generations")
    del policy, reference


# ============================================================================
# SECTION 3: DPO DATA PREPARATION
# ============================================================================

def dpo_data_preparation():
    """Preparing datasets from common formats."""
    print("\n\n" + "=" * 70)
    print("  SECTION 3: DPO DATA PREPARATION")
    print("=" * 70)
    
    print(f"""
  ═══ Common Dataset Formats → DPO Format ═══
  
  TRL DPOTrainer expects one of these formats:
  
  FORMAT 1 — Standard (prompt + chosen + rejected):
  {{
      "prompt": "What is 2+2?",
      "chosen": "4",
      "rejected": "22"
  }}
  
  FORMAT 2 — Conversational (chosen/rejected as messages):
  {{
      "chosen": [
          {{"role": "user", "content": "What is 2+2?"}},
          {{"role": "assistant", "content": "4"}}
      ],
      "rejected": [
          {{"role": "user", "content": "What is 2+2?"}},
          {{"role": "assistant", "content": "22"}}
      ]
  }}
""")
    
    # Converting from common formats
    print(f"  ── Converting From Anthropic HH-RLHF Format ──\n")
    
    # Anthropic format uses "chosen" and "rejected" as full conversations
    anthropic_example = {
        "chosen": "\n\nHuman: What is photosynthesis?\n\nAssistant: Photosynthesis is the process by which plants convert sunlight into energy.",
        "rejected": "\n\nHuman: What is photosynthesis?\n\nAssistant: I don't know, it's complicated.",
    }
    
    def convert_anthropic_to_dpo(example):
        """Convert Anthropic HH-RLHF format to standard DPO format."""
        chosen = example["chosen"]
        rejected = example["rejected"]
        
        # Extract prompt (everything up to last "Assistant: ")
        parts = chosen.split("\n\nAssistant: ")
        prompt = parts[0] + "\n\nAssistant: "
        chosen_response = parts[-1]
        
        rejected_parts = rejected.split("\n\nAssistant: ")
        rejected_response = rejected_parts[-1]
        
        return {
            "prompt": prompt.strip(),
            "chosen": chosen_response.strip(),
            "rejected": rejected_response.strip(),
        }
    
    converted = convert_anthropic_to_dpo(anthropic_example)
    print(f"    Original chosen: \"{anthropic_example['chosen'][:60]}...\"")
    print(f"    Converted prompt: \"{converted['prompt'][:50]}...\"")
    print(f"    Converted chosen: \"{converted['chosen'][:50]}...\"")
    print(f"    Converted rejected: \"{converted['rejected'][:50]}...\"")
    
    # Converting from rating data to preference pairs
    print(f"\n\n  ── Converting Rating Data to Preferences ──\n")
    
    rated_data = [
        {"prompt": "Explain recursion", "response": "A function that calls itself.", "rating": 4},
        {"prompt": "Explain recursion", "response": "To understand recursion, you must first understand recursion. Just kidding — it's when a function calls itself with a simpler input, eventually reaching a base case.", "rating": 8},
        {"prompt": "Explain recursion", "response": "It's complicated.", "rating": 2},
    ]
    
    def ratings_to_preferences(rated_examples):
        """Convert rated responses into preference pairs."""
        # Group by prompt
        from collections import defaultdict
        by_prompt = defaultdict(list)
        for ex in rated_examples:
            by_prompt[ex["prompt"]].append(ex)
        
        pairs = []
        for prompt, responses in by_prompt.items():
            # Sort by rating
            responses.sort(key=lambda x: x["rating"], reverse=True)
            
            # Create pairs: each higher-rated vs each lower-rated
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    if responses[i]["rating"] > responses[j]["rating"]:
                        pairs.append({
                            "prompt": prompt,
                            "chosen": responses[i]["response"],
                            "rejected": responses[j]["response"],
                            "margin": responses[i]["rating"] - responses[j]["rating"],
                        })
        
        return pairs
    
    pairs = ratings_to_preferences(rated_data)
    print(f"    {len(rated_data)} rated responses → {len(pairs)} preference pairs")
    for p in pairs:
        print(f"      margin={p['margin']}: \"{p['chosen'][:35]}...\" > \"{p['rejected'][:35]}...\"")
    
    print(f"""
  ═══ Data Quality Checklist ═══
  
  □ Same prompt for chosen AND rejected? (CRITICAL)
  □ Clear quality difference between chosen/rejected?
  □ No formatting artifacts or truncation?
  □ Balanced dataset (not all same topic/style)?
  □ Filtered out ties or ambiguous preferences?
  □ Checked for label noise (annotator disagreement)?
  □ Removed duplicate prompts with contradictory preferences?
  
  
  ═══ Popular Preference Datasets ═══
  
  ┌────────────────────────┬──────────┬───────────────────────────┐
  │ Dataset                │ Size     │ Source                    │
  ├────────────────────────┼──────────┼───────────────────────────┤
  │ Anthropic HH-RLHF     │ ~170K    │ Human annotators          │
  │ OpenAssistant/oasst1   │ ~90K     │ Community volunteers      │
  │ UltraFeedback         │ ~64K     │ GPT-4 as judge            │
  │ Nectar                │ ~180K    │ Multi-source aggregation  │
  │ HelpSteer             │ ~37K     │ NVIDIA human annotators   │
  │ Chatbot Arena (LMSYS) │ ~33K     │ Human pairwise voting     │
  │ argilla/distilabel-*  │ Varies   │ AI-generated feedback     │
  └────────────────────────┴──────────┴───────────────────────────┘
""")


# ============================================================================
# SECTION 4: DPO HYPERPARAMETERS
# ============================================================================

def dpo_hyperparameters():
    """Systematic hyperparameter tuning for DPO."""
    print("\n\n" + "=" * 70)
    print("  SECTION 4: DPO HYPERPARAMETERS")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ Critical Hyperparameters (in order of importance) ═══
  
  1. β (beta) — THE most important
  2. Learning rate
  3. Number of epochs
  4. Batch size (effective)
  5. Max sequence length
  6. Label smoothing
""")
    
    # Simulate hyperparameter search
    print(f"  ── Hyperparameter Sensitivity Analysis ──\n")
    
    # Simulate training with different configs
    configs = [
        {"beta": 0.05, "lr": 5e-7, "epochs": 1, "label": "β=0.05, lr=5e-7"},
        {"beta": 0.1,  "lr": 5e-7, "epochs": 1, "label": "β=0.1,  lr=5e-7 (default)"},
        {"beta": 0.1,  "lr": 5e-6, "epochs": 1, "label": "β=0.1,  lr=5e-6"},
        {"beta": 0.1,  "lr": 5e-7, "epochs": 3, "label": "β=0.1,  lr=5e-7, 3 epochs"},
        {"beta": 0.5,  "lr": 5e-7, "epochs": 1, "label": "β=0.5,  lr=5e-7"},
        {"beta": 0.1,  "lr": 1e-5, "epochs": 1, "label": "β=0.1,  lr=1e-5 (too high)"},
    ]
    
    print(f"  {'Config':>35} │ {'Final Loss':>10} │ {'Accuracy':>8} │ "
          f"{'KL':>8} │ {'Status':>15}")
    print(f"  {'─'*35}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*15}")
    
    for cfg in configs:
        # Simulate training outcome (based on general DPO behavior)
        base_loss = 0.693  # log(2)
        
        # β effect: lower β → learns faster but more KL
        beta_factor = 0.1 / cfg["beta"]
        
        # LR effect: higher → faster but risk divergence
        lr_factor = cfg["lr"] / 5e-7
        
        # Epochs: more → more learning but risk overfit
        epoch_factor = cfg["epochs"]
        
        improvement = 0.15 * beta_factor * min(lr_factor, 3) * min(epoch_factor, 2)
        noise = torch.randn(1).item() * 0.02
        
        final_loss = max(0.1, base_loss - improvement + noise)
        accuracy = min(0.95, 0.5 + improvement * 0.6 + noise)
        kl = abs(improvement * 2 + torch.randn(1).item() * 0.3)
        
        # Determine status
        if cfg["lr"] >= 1e-5:
            status = "⚠ May diverge"
            final_loss = 0.8 + torch.randn(1).item() * 0.3
            accuracy = 0.45
        elif cfg["beta"] > 0.3 and cfg["lr"] < 1e-6:
            status = "⚠ Underfit"
        elif cfg["epochs"] > 2 and kl > 2:
            status = "⚠ Overfit risk"
        elif 0.05 <= cfg["beta"] <= 0.2 and cfg["lr"] <= 5e-6:
            status = "✓ Good"
        else:
            status = "~ Okay"
        
        print(f"  {cfg['label']:>35} │ {final_loss:>10.4f} │ "
              f"{accuracy:>7.1%} │ {kl:>8.3f} │ {status:>15}")
    
    print(f"""
  ═══ Recommended Configurations ═══
  
  STARTING POINT (works for most cases):
  ┌─────────────────────┬──────────────────────────────────────┐
  │ Parameter           │ Value                                │
  ├─────────────────────┼──────────────────────────────────────┤
  │ β (beta)            │ 0.1                                  │
  │ Learning rate       │ 5e-7 (full FT) / 5e-5 (LoRA)        │
  │ LR scheduler        │ Cosine or linear with warmup         │
  │ Warmup ratio        │ 0.1                                  │
  │ Epochs              │ 1 (large data) / 3 (small data)      │
  │ Effective batch size │ 32-64                                │
  │ Max grad norm       │ 1.0                                  │
  │ Max length          │ 512-1024                             │
  │ Label smoothing     │ 0.0 (clean data) / 0.1 (noisy)      │
  │ Loss type           │ "sigmoid" (standard DPO)             │
  │ Mixed precision     │ bf16 (if available) / fp16           │
  └─────────────────────┴──────────────────────────────────────┘
  
  FULL FINE-TUNING vs LoRA:
  ┌─────────────────────┬──────────────┬──────────────┐
  │ Parameter           │ Full FT      │ LoRA         │
  ├─────────────────────┼──────────────┼──────────────┤
  │ Learning rate       │ 1e-7 to 5e-7 │ 1e-5 to 5e-5 │
  │ β                   │ 0.1 to 0.3   │ 0.1          │
  │ Epochs              │ 1            │ 1-3          │
  │ Gradient accum.     │ 4-8          │ 2-4          │
  └─────────────────────┴──────────────┴──────────────┘
  
  TUNING PRIORITY:
  1. Fix β=0.1 first, tune learning rate
  2. If loss doesn't decrease: increase LR or decrease β
  3. If loss spikes: decrease LR or increase β
  4. If accuracy plateaus: try more epochs or more data
  5. Last resort: tune label_smoothing and loss_type
""")


# ============================================================================
# SECTION 5: DPO EVALUATION
# ============================================================================

def dpo_evaluation():
    """Evaluating DPO-trained models."""
    print("\n\n" + "=" * 70)
    print("  SECTION 5: DPO EVALUATION")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ How to Evaluate a DPO-Trained Model ═══
  
  DPO evaluation should cover three aspects:
  1. Preference accuracy (did it learn the preferences?)
  2. Generation quality (does it generate good responses?)
  3. Implicit reward analysis (reward distribution shift)
""")
    
    # ─── 1. Preference Accuracy ───
    print(f"  ── 1. Preference Accuracy ──\n")
    
    # Simulate eval results
    n_eval = 200
    
    # Before DPO: random chance
    before_logits = torch.randn(n_eval) * 0.1
    before_acc = (before_logits > 0).float().mean()
    
    # After DPO: learned preferences
    after_logits = torch.randn(n_eval) * 0.5 + 0.8
    after_acc = (after_logits > 0).float().mean()
    
    print(f"    Before DPO: accuracy = {before_acc.item():.1%} (≈ random)")
    print(f"    After DPO:  accuracy = {after_acc.item():.1%} ✓")
    
    # ─── 2. Win Rate Estimation ───
    print(f"\n  ── 2. Win Rate (vs Reference) ──\n")
    
    print(f"""    Win rate measures how often the DPO model's generations
    are preferred over the reference model's generations.
    
    Method:
    1. Generate responses from both models on test prompts
    2. Have a judge (human or GPT-4) compare pairs
    3. Compute win/tie/loss percentages
""")
    
    # Simulate win rate
    n_test = 100
    wins = 58
    ties = 22
    losses = 20
    
    print(f"    Results on {n_test} test prompts:")
    print(f"    ┌─────────┬──────┬───────┐")
    print(f"    │ Wins    │ Ties │ Losses│")
    print(f"    ├─────────┼──────┼───────┤")
    print(f"    │ {wins:>4}    │ {ties:>4} │ {losses:>4}  │")
    print(f"    │ {100*wins/n_test:>4.0f}%   │ {100*ties/n_test:>4.0f}%│ {100*losses/n_test:>4.0f}% │")
    print(f"    └─────────┴──────┴───────┘")
    print(f"    Win rate (excl. ties): {100*wins/(wins+losses):.0f}%")
    
    # ─── 3. Implicit Reward Analysis ───
    print(f"\n  ── 3. Implicit Reward Distribution ──\n")
    
    # Simulate reward distributions
    before_rewards = torch.randn(200) * 0.3
    after_rewards = torch.randn(200) * 0.5 + 0.4
    
    print(f"    {'Metric':>25} │ {'Before DPO':>12} │ {'After DPO':>12}")
    print(f"    {'─'*25}─┼─{'─'*12}─┼─{'─'*12}")
    print(f"    {'Mean implicit reward':>25} │ {before_rewards.mean():>12.4f} │ {after_rewards.mean():>12.4f}")
    print(f"    {'Std implicit reward':>25} │ {before_rewards.std():>12.4f} │ {after_rewards.std():>12.4f}")
    print(f"    {'% positive reward':>25} │ {(before_rewards>0).float().mean()*100:>11.1f}% │ {(after_rewards>0).float().mean()*100:>11.1f}%")
    
    # ─── 4. Divergence Metrics ───
    print(f"\n  ── 4. Policy Divergence Analysis ──\n")
    
    # KL divergence between policy and reference
    kl_values = torch.tensor([0.5, 1.2, 2.3, 3.1, 4.5, 5.8])
    steps = [100, 200, 300, 400, 500, 600]
    
    print(f"    {'Step':>6} │ {'KL(π||π_ref)':>12} │ {'Status':>20}")
    print(f"    {'─'*6}─┼─{'─'*12}─┼─{'─'*20}")
    
    for step, kl in zip(steps, kl_values):
        if kl < 2:
            status = "✓ Close to reference"
        elif kl < 5:
            status = "⚠ Moderate divergence"
        else:
            status = "✗ High divergence"
        print(f"    {step:>6} │ {kl:>12.2f} │ {status:>20}")
    
    print(f"""
  ═══ Evaluation Checklist ═══
  
  □ Preference accuracy on held-out test set > 60%
  □ Win rate vs reference > 55% (judge: human or GPT-4)
  □ KL divergence < 5.0 (model hasn't drifted too far)
  □ No repetitive or degenerate generations
  □ Performance on standard benchmarks maintained (MMLU, etc.)
  □ Reward distribution shifted positive compared to reference
  □ Qualitative review of edge cases and safety scenarios
  
  
  ═══ Common Evaluation Pitfalls ═══
  
  1. DON'T evaluate only on preference accuracy
     → Model may memorize without generalizing
  
  2. DON'T skip generation quality checks
     → Numbers look good but outputs are bad
  
  3. DON'T use same judge for training and eval
     → Inflate metrics by gaming the judge
  
  4. DON'T ignore capability regression
     → Alignment may come at cost of knowledge
     → Always check downstream benchmarks
  
  5. DON'T evaluate on in-distribution only
     → Test on out-of-distribution prompts too
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all DPO training sections."""
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║    DPO TRAINING — PRODUCTION DPO WITH TRL                         ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    # Section 1: TRL DPOTrainer
    dpo_with_trl()
    
    # Section 2: Online DPO
    online_dpo()
    
    # Section 3: Data preparation
    dpo_data_preparation()
    
    # Section 4: Hyperparameters
    dpo_hyperparameters()
    
    # Section 5: Evaluation
    dpo_evaluation()
    
    print("\n" + "=" * 70)
    print("  DPO TRAINING MODULE COMPLETE")
    print("=" * 70)
    print("""
    Covered:
    ✓ TRL DPOTrainer — full production code pattern
    ✓ Online DPO — generate + score + train loop
    ✓ Data preparation — format conversion, dataset sources
    ✓ Hyperparameter guide — β, lr, epochs, sensitivity analysis
    ✓ Evaluation — preference accuracy, win rate, reward analysis
    """)


if __name__ == "__main__":
    main()
