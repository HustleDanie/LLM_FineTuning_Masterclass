"""
DPO From Scratch — Implementing DPO Without Libraries
=======================================================

Build DPO from the ground up to understand every component:

1. DPOLossFunction
   - Implementing the DPO loss step by step
   - Numerical stability considerations

2. DPOTrainingLoop
   - Full training loop with a tiny language model
   - Preference data preparation

3. DPOWithGPT2
   - DPO applied to a real transformer (DistilGPT2)
   - Computing per-token log-probabilities

4. PreferenceDataPipeline
   - Building preference datasets
   - Tokenization and batching strategies

5. DPOWithLoRA
   - Memory-efficient DPO using LoRA
   - Reference model via disabled adapters

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional


# ============================================================================
# SECTION 1: DPO LOSS FUNCTION
# ============================================================================

def dpo_loss_function():
    """Implementing the DPO loss step by step."""
    print("=" * 70)
    print("  SECTION 1: DPO LOSS FUNCTION FROM SCRATCH")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # ─── Step 1: The core DPO loss computation ───
    print(f"""
  ═══ DPO Loss — Step by Step ═══
  
  Input:
    • π_θ(y_w|x): Policy log-prob of chosen response
    • π_θ(y_l|x): Policy log-prob of rejected response
    • π_ref(y_w|x): Reference log-prob of chosen response
    • π_ref(y_l|x): Reference log-prob of rejected response
    • β: Temperature parameter
""")
    
    def dpo_loss_basic(
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        beta: float = 0.1,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Basic DPO loss computation.
        
        All log-probs are summed across the sequence length (per-sequence).
        """
        # Step 1: Compute log-ratios
        chosen_log_ratios = policy_chosen_logps - ref_chosen_logps
        rejected_log_ratios = policy_rejected_logps - ref_rejected_logps
        
        # Step 2: Compute the DPO logit (preference margin)
        logits = beta * (chosen_log_ratios - rejected_log_ratios)
        
        # Step 3: Loss = -log σ(logit)
        loss = -F.logsigmoid(logits).mean()
        
        # Useful metrics
        metrics = {
            "loss": loss.item(),
            "chosen_reward": (beta * chosen_log_ratios).mean().item(),
            "rejected_reward": (beta * rejected_log_ratios).mean().item(),
            "reward_margin": (beta * (chosen_log_ratios - rejected_log_ratios)).mean().item(),
            "accuracy": (logits > 0).float().mean().item(),
        }
        
        return loss, metrics
    
    # Demo with synthetic data
    batch_size = 32
    
    # Simulate: policy slightly prefers chosen over rejected
    policy_chosen = torch.randn(batch_size) - 8.0    # ~-8.0 (higher prob)
    policy_rejected = torch.randn(batch_size) - 10.0  # ~-10.0 (lower prob)
    ref_chosen = torch.randn(batch_size) - 9.0
    ref_rejected = torch.randn(batch_size) - 9.5
    
    loss, metrics = dpo_loss_basic(
        policy_chosen, policy_rejected,
        ref_chosen, ref_rejected,
        beta=0.1
    )
    
    print(f"  ── Results with β=0.1 ──\n")
    for k, v in metrics.items():
        print(f"    {k:>20}: {v:.4f}")
    
    # ─── Step 2: Numerically stable version ───
    print(f"\n\n  ── Numerically Stable DPO Loss ──")
    
    def dpo_loss_stable(
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Numerically stable DPO loss with optional label smoothing.
        
        Label smoothing (cDPO):
          L = (1-ε)·L_DPO(chosen>rejected) + ε·L_DPO(rejected>chosen)
        """
        chosen_log_ratios = policy_chosen_logps - ref_chosen_logps
        rejected_log_ratios = policy_rejected_logps - ref_rejected_logps
        
        logits = beta * (chosen_log_ratios - rejected_log_ratios)
        
        if label_smoothing > 0:
            # Conservative DPO (cDPO): handles noisy preferences
            loss = (
                -(1 - label_smoothing) * F.logsigmoid(logits)
                - label_smoothing * F.logsigmoid(-logits)
            ).mean()
        else:
            # Use logsigmoid for numerically stability
            # logsigmoid(x) = -softplus(-x) avoids overflow
            loss = -F.logsigmoid(logits).mean()
        
        metrics = {
            "loss": loss.item(),
            "accuracy": (logits > 0).float().mean().item(),
            "logits_mean": logits.mean().item(),
            "logits_std": logits.std().item(),
        }
        
        return loss, metrics
    
    # Compare standard vs label-smoothed
    print(f"\n    {'Smoothing':>12} │ {'Loss':>8} │ {'Accuracy':>8}")
    print(f"    {'─'*12}─┼─{'─'*8}─┼─{'─'*8}")
    
    for eps in [0.0, 0.05, 0.1, 0.2]:
        loss, m = dpo_loss_stable(
            policy_chosen, policy_rejected,
            ref_chosen, ref_rejected,
            beta=0.1, label_smoothing=eps
        )
        print(f"    {eps:>12.2f} │ {m['loss']:>8.4f} │ {m['accuracy']:>7.1%}")
    
    print(f"""
  NOTE: Label smoothing (cDPO) helps when preference labels are noisy.
  If ~10% of your labels might be wrong, use label_smoothing=0.1.
""")


# ============================================================================
# SECTION 2: DPO TRAINING LOOP
# ============================================================================

def dpo_training_loop():
    """Full DPO training loop with a tiny language model."""
    print("\n\n" + "=" * 70)
    print("  SECTION 2: DPO TRAINING LOOP (TINY MODEL)")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # ─── Tiny language model ───
    vocab_size = 30
    d_model = 32
    
    class TinyLM(nn.Module):
        """Minimal language model for DPO demonstration."""
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.rnn = nn.GRU(d_model, d_model, batch_first=True)
            self.head = nn.Linear(d_model, vocab_size)
        
        def forward(self, input_ids):
            x = self.embed(input_ids)
            h, _ = self.rnn(x)
            return self.head(h)
        
        def get_sequence_logprob(self, input_ids):
            """Compute log P(sequence) = sum of log P(token_t | tokens_<t)."""
            logits = self(input_ids)
            # Shift: predict token t from context <t
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(
                2, shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Sum across sequence for per-sequence log-prob
            return token_log_probs.sum(dim=-1)
    
    # ─── Create preference data ───
    n_pairs = 200
    seq_len = 8
    
    # Chosen: sequences with lower token values (arbitrary preference)
    chosen_ids = torch.randint(0, vocab_size // 2, (n_pairs, seq_len))
    rejected_ids = torch.randint(vocab_size // 2, vocab_size, (n_pairs, seq_len))
    
    print(f"\n  Preference data: {n_pairs} pairs, seq_len={seq_len}")
    print(f"  Chosen tokens: [0, {vocab_size//2}), Rejected: [{vocab_size//2}, {vocab_size})")
    
    # ─── Initialize policy and reference ───
    policy = TinyLM()
    reference = TinyLM()
    reference.load_state_dict(policy.state_dict())
    for p in reference.parameters():
        p.requires_grad = False
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    beta = 0.1
    batch_size = 32
    
    # ─── Training loop ───
    print(f"\n  ── DPO Training ──")
    print(f"  β={beta}, lr=1e-3, batch_size={batch_size}")
    print(f"\n  {'Epoch':>6} │ {'Loss':>8} │ {'Accuracy':>8} │ "
          f"{'Chosen r':>9} │ {'Rejected r':>10}")
    print(f"  {'─'*6}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*9}─┼─{'─'*10}")
    
    for epoch in range(15):
        epoch_loss = 0
        epoch_acc = 0
        epoch_chosen_r = 0
        epoch_rejected_r = 0
        n_batches = 0
        
        # Shuffle
        perm = torch.randperm(n_pairs)
        chosen_shuffled = chosen_ids[perm]
        rejected_shuffled = rejected_ids[perm]
        
        for i in range(0, n_pairs, batch_size):
            chosen_batch = chosen_shuffled[i:i+batch_size]
            rejected_batch = rejected_shuffled[i:i+batch_size]
            
            # Compute log-probs
            policy_chosen_lp = policy.get_sequence_logprob(chosen_batch)
            policy_rejected_lp = policy.get_sequence_logprob(rejected_batch)
            
            with torch.no_grad():
                ref_chosen_lp = reference.get_sequence_logprob(chosen_batch)
                ref_rejected_lp = reference.get_sequence_logprob(rejected_batch)
            
            # DPO loss
            chosen_log_ratio = policy_chosen_lp - ref_chosen_lp
            rejected_log_ratio = policy_rejected_lp - ref_rejected_lp
            logits = beta * (chosen_log_ratio - rejected_log_ratio)
            loss = -F.logsigmoid(logits).mean()
            
            # Update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            # Track metrics
            with torch.no_grad():
                epoch_loss += loss.item()
                epoch_acc += (logits > 0).float().mean().item()
                epoch_chosen_r += (beta * chosen_log_ratio).mean().item()
                epoch_rejected_r += (beta * rejected_log_ratio).mean().item()
                n_batches += 1
        
        if (epoch + 1) % 3 == 0:
            print(f"  {epoch+1:>6} │ {epoch_loss/n_batches:>8.4f} │ "
                  f"{epoch_acc/n_batches:>7.1%} │ "
                  f"{epoch_chosen_r/n_batches:>9.4f} │ "
                  f"{epoch_rejected_r/n_batches:>10.4f}")
    
    # Final evaluation
    print(f"\n  ── Final Evaluation ──")
    with torch.no_grad():
        test_chosen = torch.randint(0, vocab_size // 2, (50, seq_len))
        test_rejected = torch.randint(vocab_size // 2, vocab_size, (50, seq_len))
        
        pc = policy.get_sequence_logprob(test_chosen)
        pr = policy.get_sequence_logprob(test_rejected)
        rc = reference.get_sequence_logprob(test_chosen)
        rr = reference.get_sequence_logprob(test_rejected)
        
        logits = beta * ((pc - rc) - (pr - rr))
        acc = (logits > 0).float().mean()
        print(f"    Test accuracy: {acc.item():.1%}")
        print(f"    Policy prefers chosen: {(pc > pr).float().mean().item():.1%}")
    
    del policy, reference


# ============================================================================
# SECTION 3: DPO WITH GPT-2
# ============================================================================

def dpo_with_gpt2():
    """DPO applied to a real transformer (DistilGPT2)."""
    print("\n\n" + "=" * 70)
    print("  SECTION 3: DPO WITH DISTILGPT2")
    print("=" * 70)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load policy and reference
    policy = AutoModelForCausalLM.from_pretrained(model_name)
    reference = AutoModelForCausalLM.from_pretrained(model_name)
    for p in reference.parameters():
        p.requires_grad = False
    
    print(f"\n  Model: {model_name}")
    print(f"  Parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # ─── Compute per-token log-probs for DPO ───
    def get_log_probs(model, input_ids, attention_mask, labels):
        """
        Compute per-token log-probabilities for a sequence.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels: (batch, seq_len) — same as input_ids for causal LM
            
        Returns:
            Per-sequence log-probabilities (batch,)
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Shift for causal LM: predict token t from context <t
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        
        # Per-token log-probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_lp = log_probs.gather(
            2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding and sum for per-sequence log-prob
        per_token_lp = per_token_lp * shift_mask
        return per_token_lp.sum(dim=-1)
    
    # ─── Preference data ───
    preference_pairs = [
        {
            "prompt": "The key to happiness is",
            "chosen": " finding meaning in everyday moments and building strong relationships.",
            "rejected": " money and power and getting whatever you want no matter what.",
        },
        {
            "prompt": "To learn programming, you should",
            "chosen": " start with fundamentals, practice with small projects, and gradually tackle harder challenges.",
            "rejected": " just copy code from the internet and hope it works somehow.",
        },
        {
            "prompt": "Climate change is",
            "chosen": " a well-documented scientific phenomenon driven by greenhouse gas emissions from human activities.",
            "rejected": " not real because it was cold yesterday and nothing has changed.",
        },
        {
            "prompt": "A good leader should",
            "chosen": " listen to their team, take responsibility, and communicate a clear vision.",
            "rejected": " always be right and never admit mistakes because that shows weakness.",
        },
    ]
    
    print(f"\n  Training on {len(preference_pairs)} preference pairs")
    
    # Tokenize
    def tokenize_pair(pair, max_len=64):
        chosen_text = pair["prompt"] + pair["chosen"]
        rejected_text = pair["prompt"] + pair["rejected"]
        
        chosen_enc = tokenizer(
            chosen_text, return_tensors="pt",
            max_length=max_len, truncation=True, padding="max_length"
        )
        rejected_enc = tokenizer(
            rejected_text, return_tensors="pt",
            max_length=max_len, truncation=True, padding="max_length"
        )
        
        return chosen_enc, rejected_enc
    
    # ─── DPO Training ───
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-6)
    beta = 0.1
    
    print(f"\n  ── Training (β={beta}, lr=5e-6) ──")
    print(f"  {'Step':>6} │ {'Loss':>8} │ {'Chosen r':>9} │ "
          f"{'Rejected r':>10} │ {'Margin':>8}")
    print(f"  {'─'*6}─┼─{'─'*8}─┼─{'─'*9}─┼─{'─'*10}─┼─{'─'*8}")
    
    for step in range(10):
        total_loss = 0
        total_chosen_r = 0
        total_rejected_r = 0
        
        for pair in preference_pairs:
            chosen_enc, rejected_enc = tokenize_pair(pair)
            
            # Policy log-probs
            policy_chosen_lp = get_log_probs(
                policy,
                chosen_enc["input_ids"], chosen_enc["attention_mask"],
                chosen_enc["input_ids"]
            )
            policy_rejected_lp = get_log_probs(
                policy,
                rejected_enc["input_ids"], rejected_enc["attention_mask"],
                rejected_enc["input_ids"]
            )
            
            # Reference log-probs
            with torch.no_grad():
                ref_chosen_lp = get_log_probs(
                    reference,
                    chosen_enc["input_ids"], chosen_enc["attention_mask"],
                    chosen_enc["input_ids"]
                )
                ref_rejected_lp = get_log_probs(
                    reference,
                    rejected_enc["input_ids"], rejected_enc["attention_mask"],
                    rejected_enc["input_ids"]
                )
            
            # DPO loss
            chosen_log_ratio = policy_chosen_lp - ref_chosen_lp
            rejected_log_ratio = policy_rejected_lp - ref_rejected_lp
            logits = beta * (chosen_log_ratio - rejected_log_ratio)
            loss = -F.logsigmoid(logits).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            with torch.no_grad():
                total_loss += loss.item()
                total_chosen_r += (beta * chosen_log_ratio).item()
                total_rejected_r += (beta * rejected_log_ratio).item()
        
        n = len(preference_pairs)
        margin = (total_chosen_r - total_rejected_r) / n
        
        if (step + 1) % 2 == 0:
            print(f"  {step+1:>6} │ {total_loss/n:>8.4f} │ "
                  f"{total_chosen_r/n:>9.4f} │ "
                  f"{total_rejected_r/n:>10.4f} │ {margin:>8.4f}")
    
    # Test generation
    print(f"\n  ── Generation After DPO ──")
    policy.eval()
    for pair in preference_pairs[:2]:
        input_ids = tokenizer.encode(pair["prompt"], return_tensors="pt")
        with torch.no_grad():
            output = policy.generate(
                input_ids, max_new_tokens=30,
                do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"    Prompt: \"{pair['prompt']}\"")
        print(f"    Output: {text}")
        print()
    
    del policy, reference


# ============================================================================
# SECTION 4: PREFERENCE DATA PIPELINE
# ============================================================================

def preference_data_pipeline():
    """Building preference datasets for DPO."""
    print("\n\n" + "=" * 70)
    print("  SECTION 4: PREFERENCE DATA PIPELINE")
    print("=" * 70)
    
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"""
  ═══ DPO Data Format ═══
  
  Standard format (HuggingFace convention):
  
  {{
      "prompt": "What is the capital of France?",
      "chosen": "The capital of France is Paris...",
      "rejected": "France is a country in Europe..."
  }}
  
  For chat models (multi-turn):
  
  {{
      "chosen": [
          {{"role": "user", "content": "What is 2+2?"}},
          {{"role": "assistant", "content": "2+2 equals 4."}}
      ],
      "rejected": [
          {{"role": "user", "content": "What is 2+2?"}},
          {{"role": "assistant", "content": "That's a complex question..."}}
      ]
  }}
""")
    
    # Build a preference dataset
    raw_data = [
        {
            "prompt": "Explain gravity simply.",
            "chosen": "Gravity is a force that pulls objects toward each other. "
                      "The more massive an object, the stronger its pull. "
                      "Earth's gravity keeps us on the ground.",
            "rejected": "Gravity is the curvature of spacetime caused by mass-energy "
                        "as described by Einstein's field equations Gμν + Λgμν = 8πG/c⁴ Tμν "
                        "which is really complicated.",
        },
        {
            "prompt": "How do I make coffee?",
            "chosen": "1. Boil water. 2. Add ground coffee to a filter. "
                      "3. Pour hot water over the grounds. 4. Wait 3-4 minutes. "
                      "5. Enjoy your coffee!",
            "rejected": "Coffee is a brewed drink made from roasted beans. "
                        "The history of coffee dates back to the 15th century...",
        },
        {
            "prompt": "Is exercise good for you?",
            "chosen": "Yes! Regular exercise improves cardiovascular health, "
                      "strengthens muscles, boosts mood, and helps maintain "
                      "a healthy weight. Even 30 minutes of walking daily helps.",
            "rejected": "It depends on many factors and there's no simple answer.",
        },
    ]
    
    # ─── Tokenization for DPO ───
    print(f"  ── Tokenization Strategy ──\n")
    
    def tokenize_for_dpo(examples, tokenizer, max_length=128):
        """
        Tokenize preference pairs for DPO training.
        
        Important: We need to know which tokens are prompt vs response
        to only compute loss on response tokens.
        """
        batch = {
            "chosen_input_ids": [],
            "chosen_attention_mask": [],
            "chosen_labels": [],       # -100 for prompt tokens
            "rejected_input_ids": [],
            "rejected_attention_mask": [],
            "rejected_labels": [],
        }
        
        for ex in examples:
            prompt = ex["prompt"]
            prompt_enc = tokenizer(prompt, add_special_tokens=False)
            prompt_len = len(prompt_enc["input_ids"])
            
            for key in ["chosen", "rejected"]:
                full_text = prompt + " " + ex[key]
                enc = tokenizer(
                    full_text,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                
                # Create labels: -100 for prompt tokens (don't compute loss)
                labels = enc["input_ids"].clone()
                labels[0, :prompt_len] = -100
                # Also mask padding
                labels[labels == tokenizer.pad_token_id] = -100
                
                batch[f"{key}_input_ids"].append(enc["input_ids"])
                batch[f"{key}_attention_mask"].append(enc["attention_mask"])
                batch[f"{key}_labels"].append(labels)
        
        # Stack into tensors
        for k in batch:
            batch[k] = torch.cat(batch[k], dim=0)
        
        return batch
    
    tokenized = tokenize_for_dpo(raw_data, tokenizer, max_length=80)
    
    print(f"    Batch size: {tokenized['chosen_input_ids'].shape[0]}")
    print(f"    Sequence length: {tokenized['chosen_input_ids'].shape[1]}")
    
    # Show tokenization details for first example
    ex = raw_data[0]
    prompt_tokens = tokenizer.encode(ex["prompt"], add_special_tokens=False)
    chosen_tokens = tokenizer.encode(
        ex["prompt"] + " " + ex["chosen"], add_special_tokens=False
    )
    
    print(f"\n    Example tokenization:")
    print(f"      Prompt: \"{ex['prompt'][:40]}...\"")
    print(f"      Prompt tokens: {len(prompt_tokens)}")
    print(f"      Full chosen tokens: {len(chosen_tokens)}")
    print(f"      Response tokens: {len(chosen_tokens) - len(prompt_tokens)}")
    print(f"      Labels masked (prompt): first {len(prompt_tokens)} tokens = -100")
    
    # Data quality guidelines
    print(f"""
  ═══ Preference Data Best Practices ═══
  
  1. DATA QUALITY:
     • Same prompt must be used for both chosen and rejected
     • Differences should reflect the preference, not randomness
     • Clear quality gap between chosen and rejected
  
  2. DATA VOLUME:
     ┌─────────────────┬───────────────────────────────┐
     │ Model Size      │ Recommended Pairs              │
     ├─────────────────┼───────────────────────────────┤
     │ < 1B params     │ 5K - 20K pairs                │
     │ 1-7B params     │ 10K - 50K pairs               │
     │ 7-70B params    │ 20K - 100K pairs              │
     │ > 70B params    │ 50K - 500K pairs              │
     └─────────────────┴───────────────────────────────┘
  
  3. COMMON DATASETS:
     • Anthropic HH-RLHF: ~170K pairs (helpfulness + harmlessness)
     • OpenAssistant: Community-annotated preference data
     • UltraFeedback: 64K prompts with GPT-4 feedback
     • Nectar: 180K pairs from diverse sources
  
  4. LABEL NOISE:
     • Human annotators agree 60-80% of the time
     • Use label_smoothing=0.1 in DPO loss for noisy data
     • Consider filtering pairs with low annotator agreement
""")


# ============================================================================
# SECTION 5: DPO WITH LORA
# ============================================================================

def dpo_with_lora():
    """Memory-efficient DPO using LoRA."""
    print("\n\n" + "=" * 70)
    print("  SECTION 5: DPO WITH LoRA (MEMORY-EFFICIENT)")
    print("=" * 70)
    
    print(f"""
  ═══ Why LoRA for DPO? ═══
  
  Standard DPO needs 2 full models in memory:
    • Policy model: full parameters (trainable)
    • Reference model: full parameters (frozen)
    → 2× model memory!
  
  LoRA trick: Use ONE base model + adapter
    • Base model (frozen) = reference model
    • Base model + LoRA adapter = policy model
    • Disable adapter → reference forward pass
    • Enable adapter → policy forward pass
    → ~1.01× model memory!
""")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    try:
        from peft import get_peft_model, LoraConfig, TaskType
        has_peft = True
    except ImportError:
        has_peft = False
        print("  [PEFT not installed — showing code pattern]")
    
    if has_peft:
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add LoRA adapter
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["c_attn", "c_proj"],
        )
        
        model = get_peft_model(base_model, lora_config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n  Model: {model_name} + LoRA")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable (LoRA): {trainable_params:,} "
              f"({100*trainable_params/total_params:.2f}%)")
        
        # ─── DPO with LoRA: disable/enable adapter for reference ───
        print(f"\n  ── DPO Forward Pass with LoRA ──")
        
        def get_log_probs_lora(model, input_ids, attention_mask):
            """Get per-sequence log-probs."""
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()
            
            log_probs = F.log_softmax(shift_logits, dim=-1)
            per_token = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            return (per_token * shift_mask).sum(dim=-1)
        
        def dpo_step_lora(model, chosen_ids, chosen_mask, rejected_ids, 
                          rejected_mask, beta=0.1):
            """Single DPO step using LoRA adapter enable/disable."""
            
            # Policy forward pass (adapter ENABLED)
            model.enable_adapter_layers()
            policy_chosen_lp = get_log_probs_lora(model, chosen_ids, chosen_mask)
            policy_rejected_lp = get_log_probs_lora(model, rejected_ids, rejected_mask)
            
            # Reference forward pass (adapter DISABLED)
            with torch.no_grad():
                model.disable_adapter_layers()
                ref_chosen_lp = get_log_probs_lora(model, chosen_ids, chosen_mask)
                ref_rejected_lp = get_log_probs_lora(model, rejected_ids, rejected_mask)
                model.enable_adapter_layers()
            
            # DPO loss
            chosen_ratio = policy_chosen_lp - ref_chosen_lp
            rejected_ratio = policy_rejected_lp - ref_rejected_lp
            logits = beta * (chosen_ratio - rejected_ratio)
            loss = -F.logsigmoid(logits).mean()
            
            return loss, {
                "accuracy": (logits > 0).float().mean().item(),
                "margin": logits.mean().item(),
            }
        
        # Demo training
        pairs = [
            ("Science shows that exercise is", 
             " beneficial for both physical and mental health.",
             " bad for you and you should avoid it."),
            ("The best way to learn is",
             " through practice, patience, and building on fundamentals.",
             " to never try and just give up immediately."),
        ]
        
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4
        )
        
        print(f"\n  ── Training with LoRA DPO ──")
        print(f"  {'Step':>6} │ {'Loss':>8} │ {'Accuracy':>8} │ {'Margin':>8}")
        print(f"  {'─'*6}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}")
        
        for step in range(8):
            step_loss = 0
            step_acc = 0
            step_margin = 0
            
            for prompt, chosen, rejected in pairs:
                chosen_enc = tokenizer(
                    prompt + chosen, return_tensors="pt",
                    max_length=50, truncation=True, padding="max_length"
                )
                rejected_enc = tokenizer(
                    prompt + rejected, return_tensors="pt",
                    max_length=50, truncation=True, padding="max_length"
                )
                
                loss, metrics = dpo_step_lora(
                    model,
                    chosen_enc["input_ids"], chosen_enc["attention_mask"],
                    rejected_enc["input_ids"], rejected_enc["attention_mask"],
                    beta=0.1
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                step_loss += loss.item()
                step_acc += metrics["accuracy"]
                step_margin += metrics["margin"]
            
            n = len(pairs)
            if (step + 1) % 2 == 0:
                print(f"  {step+1:>6} │ {step_loss/n:>8.4f} │ "
                      f"{step_acc/n:>7.1%} │ {step_margin/n:>8.4f}")
        
        del model
    
    # Memory comparison
    print(f"""
  ═══ Memory Comparison ═══
  
  ┌──────────────────────┬────────────┬────────────┬────────────┐
  │ Method               │ 7B Model   │ 13B Model  │ 70B Model  │
  ├──────────────────────┼────────────┼────────────┼────────────┤
  │ Full DPO (2 models)  │ ~28 GB     │ ~52 GB     │ ~280 GB    │
  │ DPO + LoRA           │ ~14.1 GB   │ ~26.1 GB   │ ~140.1 GB  │
  │ DPO + QLoRA          │ ~4.1 GB    │ ~7.1 GB    │ ~36.1 GB   │
  └──────────────────────┴────────────┴────────────┴────────────┘
  
  LoRA DPO: ~50% memory reduction (no duplicate model!)
  QLoRA DPO: ~85% memory reduction (4-bit + LoRA, single GPU)
  
  
  ═══ Code Pattern for TRL DPOTrainer + LoRA ═══
  
  ```python
  from trl import DPOConfig, DPOTrainer
  from peft import LoraConfig
  
  lora_config = LoraConfig(r=16, lora_alpha=32, ...)
  
  training_args = DPOConfig(
      output_dir="./dpo_output",
      beta=0.1,
      learning_rate=5e-5,
      per_device_train_batch_size=4,
      gradient_accumulation_steps=4,
      num_train_epochs=1,
  )
  
  trainer = DPOTrainer(
      model=model,
      ref_model=None,       # ← None! Uses LoRA disable trick
      args=training_args,
      train_dataset=dataset,
      tokenizer=tokenizer,
      peft_config=lora_config,
  )
  
  trainer.train()
  ```
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all DPO from scratch sections."""
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║    DPO FROM SCRATCH — BUILDING DPO WITHOUT LIBRARIES              ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    # Section 1: Loss function
    dpo_loss_function()
    
    # Section 2: Training loop
    dpo_training_loop()
    
    # Section 3: GPT-2 DPO
    dpo_with_gpt2()
    
    # Section 4: Data pipeline
    preference_data_pipeline()
    
    # Section 5: DPO with LoRA
    dpo_with_lora()
    
    print("\n" + "=" * 70)
    print("  DPO FROM SCRATCH MODULE COMPLETE")
    print("=" * 70)
    print("""
    Covered:
    ✓ DPO loss — basic + numerically stable + label smoothing (cDPO)
    ✓ Full training loop — tiny model with preference pairs
    ✓ GPT-2 DPO — real transformer with per-token log-prob computation
    ✓ Data pipeline — tokenization, label masking, best practices
    ✓ LoRA DPO — memory-efficient with adapter disable trick
    """)


if __name__ == "__main__":
    main()
