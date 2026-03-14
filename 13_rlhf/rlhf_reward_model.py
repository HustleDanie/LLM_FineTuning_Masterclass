"""
RLHF Reward Model — Training Reward Models from Preferences
============================================================

Building and training reward models for RLHF:

1. RewardModelFromScratch
   - Build a reward model architecture manually
   - Bradley-Terry training loop

2. RewardModelFromLLM
   - Initialize reward model from a pretrained LLM
   - Add a value/scalar head

3. PreferenceDataset
   - Creating and formatting preference data
   - Data quality and annotation considerations

4. RewardModelEvaluation
   - Evaluating reward model quality
   - Calibration and agreement metrics

5. RewardModelWithTRL
   - Using TRL's RewardTrainer for reward modeling
   - HuggingFace integration

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional


# ============================================================================
# SECTION 1: REWARD MODEL FROM SCRATCH
# ============================================================================

class RewardHead(nn.Module):
    """Scalar value head that converts hidden states to a reward score."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, 1)
        
        # Initialize near zero — reward starts neutral
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, d_model] - last layer outputs
        Returns:
            rewards: [batch] - scalar reward per sequence
        """
        # Take the last token's hidden state
        last_hidden = hidden_states[:, -1, :]  # [batch, d_model]
        return self.linear(self.dropout(last_hidden)).squeeze(-1)  # [batch]


class SimpleRewardModel(nn.Module):
    """
    Reward model built from scratch.
    
    Architecture:
        Input embeddings → Transformer layers → Value head → Scalar reward
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Reward head
        self.reward_head = RewardHead(d_model)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        
        tok = self.token_embed(input_ids)
        pos = self.pos_embed(torch.arange(T, device=input_ids.device))
        x = tok + pos
        
        x = self.transformer(x)
        reward = self.reward_head(x)
        
        return reward


def train_reward_model_scratch():
    """Build and train a reward model from scratch."""
    print("=" * 65)
    print("  SECTION 1: REWARD MODEL FROM SCRATCH")
    print("=" * 65)
    
    torch.manual_seed(42)
    
    vocab_size = 500
    model = SimpleRewardModel(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=32,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Reward Model: {total_params:,} parameters")
    
    # Generate synthetic preference data
    n_pairs = 200
    seq_len = 16
    
    # "Chosen" responses: token 1 appears frequently (quality signal)
    chosen = torch.randint(2, vocab_size, (n_pairs, seq_len))
    chosen[:, :4] = 1  # Quality marker at start
    
    # "Rejected" responses: token 2 appears frequently (low quality)
    rejected = torch.randint(2, vocab_size, (n_pairs, seq_len))
    rejected[:, :4] = 2  # Low quality marker
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 32
    
    print(f"\n  Training: {n_pairs} preference pairs, batch_size={batch_size}")
    print(f"\n  {'Epoch':>6} {'Loss':>10} {'Accuracy':>10} {'Avg Gap':>10}")
    print(f"  {'─'*6}─{'─'*10}─{'─'*10}─{'─'*10}")
    
    for epoch in range(15):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        indices = torch.randperm(n_pairs)
        
        for i in range(0, n_pairs, batch_size):
            batch_idx = indices[i:i+batch_size]
            
            r_chosen = model(chosen[batch_idx])
            r_rejected = model(rejected[batch_idx])
            
            # Bradley-Terry loss
            loss = -torch.log(
                torch.sigmoid(r_chosen - r_rejected) + 1e-10
            ).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_correct += (r_chosen > r_rejected).sum().item()
            epoch_total += len(batch_idx)
        
        if (epoch + 1) % 3 == 0:
            with torch.no_grad():
                avg_gap = (model(chosen[:32]) - model(rejected[:32])).mean()
            acc = epoch_correct / epoch_total
            avg_loss = epoch_loss / (n_pairs // batch_size)
            print(f"  {epoch+1:>6} {avg_loss:>10.4f} {acc:>9.1%} {avg_gap.item():>10.4f}")
    
    print(f"\n  ✓ Reward model trained! Correctly ranks chosen > rejected.")
    
    del model


# ============================================================================
# SECTION 2: REWARD MODEL FROM LLM
# ============================================================================

def reward_model_from_llm():
    """Initialize a reward model from a pretrained LLM."""
    print("\n\n" + "=" * 65)
    print("  SECTION 2: REWARD MODEL FROM LLM")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print(f"\n  Base model: {model_name}")
    print(f"  Hidden size: {base_model.config.n_embd}")
    
    # ─── Method 1: Manual reward model ───
    print(f"\n  ── Method 1: Manual Reward Model ──")
    
    class GPT2RewardModel(nn.Module):
        """Reward model using GPT-2 backbone with a scalar head."""
        
        def __init__(self, base_model, hidden_size):
            super().__init__()
            # Use the transformer backbone (remove LM head)
            self.transformer = base_model.transformer
            self.value_head = nn.Linear(hidden_size, 1, bias=False)
            nn.init.normal_(self.value_head.weight, std=1.0 / math.sqrt(hidden_size + 1))
        
        def forward(self, input_ids, attention_mask=None):
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden = outputs.last_hidden_state
            
            # Get reward from last non-padding token
            if attention_mask is not None:
                # Find last real token position
                seq_lengths = attention_mask.sum(dim=1) - 1
                last_hidden = hidden[torch.arange(hidden.size(0)), seq_lengths]
            else:
                last_hidden = hidden[:, -1, :]
            
            reward = self.value_head(last_hidden).squeeze(-1)
            return reward
    
    reward_model = GPT2RewardModel(base_model, base_model.config.n_embd)
    
    total = sum(p.numel() for p in reward_model.parameters())
    new_params = sum(p.numel() for p in reward_model.value_head.parameters())
    print(f"    Total params: {total:,}")
    print(f"    New params (value head): {new_params:,}")
    print(f"    Reused from GPT-2: {total - new_params:,}")
    
    # Test forward pass
    text_good = "This is a helpful and clear response to the question."
    text_bad = "I don't know. Whatever. Stop asking me things."
    
    for text in [text_good, text_bad]:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            reward = reward_model(inputs["input_ids"], inputs["attention_mask"])
        print(f"    \"{text[:50]}...\"")
        print(f"      → Reward: {reward.item():.4f}")
    
    # ─── Train on a few preference pairs ───
    print(f"\n  ── Quick Training Demo ──")
    
    # Freeze base model, only train value head (efficient approach)
    for param in reward_model.transformer.parameters():
        param.requires_grad = False
    reward_model.value_head.weight.requires_grad = True
    
    chosen_texts = [
        "Machine learning is a branch of AI that enables computers to learn from data.",
        "The transformer architecture uses self-attention to process sequences efficiently.",
        "Python is a versatile programming language widely used in data science.",
        "Neural networks consist of interconnected layers of artificial neurons.",
    ]
    
    rejected_texts = [
        "I don't really know much about that topic honestly.",
        "Transformers are those robots from the movies, right? Just kidding.",
        "Coding is hard and nobody understands it properly anyway.",
        "Just Google it if you want to know about neural networks.",
    ]
    
    optimizer = torch.optim.Adam(
        [p for p in reward_model.parameters() if p.requires_grad],
        lr=1e-3,
    )
    
    for step in range(20):
        total_loss = 0
        correct = 0
        
        for chosen_text, rejected_text in zip(chosen_texts, rejected_texts):
            chosen_enc = tokenizer(chosen_text, return_tensors="pt",
                                  padding="max_length", max_length=32, truncation=True)
            rejected_enc = tokenizer(rejected_text, return_tensors="pt",
                                   padding="max_length", max_length=32, truncation=True)
            
            r_c = reward_model(chosen_enc["input_ids"], chosen_enc["attention_mask"])
            r_r = reward_model(rejected_enc["input_ids"], rejected_enc["attention_mask"])
            
            loss = -torch.log(torch.sigmoid(r_c - r_r) + 1e-10)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (r_c > r_r).item()
        
        if (step + 1) % 5 == 0:
            print(f"    Step {step+1}: loss={total_loss/4:.4f}, "
                  f"accuracy={correct/4:.0%}")
    
    print(f"\n  ✓ Reward model trained on preference data!")
    
    del reward_model, base_model


# ============================================================================
# SECTION 3: PREFERENCE DATASET
# ============================================================================

def preference_dataset():
    """Creating and formatting preference datasets."""
    print("\n\n" + "=" * 65)
    print("  SECTION 3: PREFERENCE DATASET")
    print("=" * 65)
    
    from datasets import Dataset
    
    # ─── Create preference dataset ───
    print(f"\n  ── Creating Preference Dataset ──")
    
    preference_data = {
        "prompt": [
            "Explain what machine learning is.",
            "What is the capital of France?",
            "Write a simple Python function.",
            "Describe the water cycle.",
            "What causes seasons on Earth?",
            "Explain how vaccines work.",
            "What is photosynthesis?",
            "How do computers store data?",
        ],
        "chosen": [
            "Machine learning is a subset of artificial intelligence where computers learn patterns from data to make predictions or decisions without being explicitly programmed for each task.",
            "The capital of France is Paris. It's located in the north-central part of the country along the Seine River and is the most populous city in France.",
            "def greet(name):\n    return f'Hello, {name}! Welcome.'",
            "The water cycle involves evaporation of water from surfaces, condensation into clouds, precipitation as rain or snow, and collection in bodies of water, repeating continuously.",
            "Seasons are caused by Earth's 23.5° axial tilt. As Earth orbits the Sun, different hemispheres receive varying amounts of direct sunlight throughout the year.",
            "Vaccines introduce a weakened or inactive form of a pathogen to train the immune system. This prepares the body to recognize and fight the real pathogen if encountered later.",
            "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen, using chlorophyll in their leaves.",
            "Computers store data as binary digits (bits) — 0s and 1s — using electrical charges in memory chips (RAM) for temporary storage, or magnetic/solid-state media for permanent storage.",
        ],
        "rejected": [
            "It's when machines learn stuff. Like robots learning to do things.",
            "France. The capital is Paris I think.",
            "just write print hello",
            "Water goes up and comes back down as rain.",
            "It gets cold in winter because the sun is farther away.",
            "They inject you with stuff to make you not sick.",
            "Plants eat sunlight.",
            "They just remember things.",
        ],
    }
    
    dataset = Dataset.from_dict(preference_data)
    
    print(f"  Dataset size: {len(dataset)} preference pairs")
    print(f"  Columns: {dataset.column_names}")
    
    # Show an example
    print(f"\n  Example:")
    ex = dataset[0]
    print(f"    Prompt:   {ex['prompt']}")
    print(f"    Chosen:   {ex['chosen'][:80]}...")
    print(f"    Rejected: {ex['rejected'][:80]}...")
    
    # ─── Tokenize for reward model training ───
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_preference_pair(example):
        """Tokenize chosen and rejected responses with the prompt."""
        chosen_text = f"Question: {example['prompt']}\nAnswer: {example['chosen']}"
        rejected_text = f"Question: {example['prompt']}\nAnswer: {example['rejected']}"
        
        chosen_enc = tokenizer(
            chosen_text, truncation=True, max_length=128, padding="max_length",
        )
        rejected_enc = tokenizer(
            rejected_text, truncation=True, max_length=128, padding="max_length",
        )
        
        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
        }
    
    tokenized = dataset.map(tokenize_preference_pair, remove_columns=dataset.column_names)
    
    print(f"\n  Tokenized dataset columns: {tokenized.column_names}")
    print(f"  Chosen sequence length: {len(tokenized[0]['chosen_input_ids'])}")
    print(f"  Rejected sequence length: {len(tokenized[0]['rejected_input_ids'])}")
    
    print(f"""
  ═══ Preference Data Best Practices ═══
  
  Data Quality:
  • Use multiple annotators (3-5 per pair)
  • Measure inter-annotator agreement
  • Filter out tie/unclear preferences
  • Balance prompt types (factual, creative, safety, etc.)
  
  Data Quantity:
  • Minimum: ~1,000 pairs for small models
  • Typical: 10K-100K pairs for 7B models
  • InstructGPT: ~33K comparison pairs
  • More data → better reward model → better RLHF
  
  Common Datasets:
  • Anthropic/hh-rlhf: Helpful + harmless preferences
  • OpenAssistant: Community-annotated conversations
  • Stanford Alpaca: Instruction-following comparisons
  • UltraFeedback: Multi-aspect preference annotations
""")
    
    return tokenized


# ============================================================================
# SECTION 4: REWARD MODEL EVALUATION
# ============================================================================

def evaluate_reward_model():
    """Evaluate reward model quality."""
    print("\n\n" + "=" * 65)
    print("  SECTION 4: REWARD MODEL EVALUATION")
    print("=" * 65)
    
    torch.manual_seed(42)
    
    # Simulate a trained reward model's scores
    n_test = 100
    
    # Ground truth: chosen should have higher reward
    true_chosen_better = torch.ones(n_test).bool()
    
    # Simulated model scores
    chosen_rewards = torch.randn(n_test) + 1.0   # Mean 1.0
    rejected_rewards = torch.randn(n_test) - 0.5  # Mean -0.5
    
    # ─── Metric 1: Pairwise Accuracy ───
    correct = (chosen_rewards > rejected_rewards).sum().item()
    accuracy = correct / n_test
    
    print(f"\n  ── Metric 1: Pairwise Accuracy ──")
    print(f"    Accuracy: {accuracy:.1%} ({correct}/{n_test})")
    print(f"    (Random baseline: 50%)")
    
    # ─── Metric 2: Average Reward Gap ───
    gap = (chosen_rewards - rejected_rewards).mean()
    print(f"\n  ── Metric 2: Average Reward Gap ──")
    print(f"    Mean(r_chosen - r_rejected): {gap.item():.4f}")
    print(f"    Std: {(chosen_rewards - rejected_rewards).std().item():.4f}")
    
    # ─── Metric 3: Calibration ───
    print(f"\n  ── Metric 3: Calibration ──")
    
    # Predicted probability that chosen is better
    probs = torch.sigmoid(chosen_rewards - rejected_rewards)
    
    # Bucket by predicted probability
    buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    
    print(f"  {'Predicted P':>12} {'Actual Rate':>12} {'Count':>8} {'Calibration':>12}")
    print(f"  {'─'*12}─{'─'*12}─{'─'*8}─{'─'*12}")
    
    for low, high in buckets:
        mask = (probs >= low) & (probs < high)
        if mask.sum() > 0:
            actual_rate = (chosen_rewards[mask] > rejected_rewards[mask]).float().mean()
            count = mask.sum().item()
            mid = (low + high) / 2
            cal_error = abs(actual_rate.item() - mid)
            print(f"  {low:.1f}-{high:.1f}       {actual_rate.item():>10.1%} {count:>8} "
                  f"{'✓' if cal_error < 0.15 else '✗':>5} ({cal_error:.3f})")
    
    # ─── Metric 4: Reward distribution ───
    print(f"\n  ── Metric 4: Reward Distributions ──")
    print(f"    Chosen:   mean={chosen_rewards.mean():.3f}, "
          f"std={chosen_rewards.std():.3f}, "
          f"range=[{chosen_rewards.min():.3f}, {chosen_rewards.max():.3f}]")
    print(f"    Rejected: mean={rejected_rewards.mean():.3f}, "
          f"std={rejected_rewards.std():.3f}, "
          f"range=[{rejected_rewards.min():.3f}, {rejected_rewards.max():.3f}]")
    
    overlap = (chosen_rewards.min() < rejected_rewards.max()).item()
    print(f"    Distributions overlap: {'Yes' if overlap else 'No'}")
    
    print(f"""
  ═══ What Makes a Good Reward Model? ═══
  
  Essential:
  • Pairwise accuracy > 65% (>75% is good)
  • Consistent reward gap between chosen/rejected
  • Well-calibrated probabilities
  
  Warning Signs:
  • Accuracy near 50% → barely better than random
  • Very large reward gap → may be overconfident
  • All rewards cluster near 0 → hasn't learned meaningful distinctions
  
  Best Practices:
  • Hold out a validation set of preferences
  • Test on out-of-distribution prompts
  • Check for reward model biases (length, verbosity)
  • Verify on different categories (safety, factual, creative)
""")


# ============================================================================
# SECTION 5: REWARD MODEL WITH TRL
# ============================================================================

def reward_model_trl():
    """Using TRL's tools for reward model training."""
    print("\n\n" + "=" * 65)
    print("  SECTION 5: REWARD MODEL WITH TRL")
    print("=" * 65)
    
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from datasets import Dataset
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"""
  ═══ TRL Reward Model Training ═══
  
  TRL (Transformer Reinforcement Learning) provides:
  • RewardTrainer: Handles preference pair training
  • RewardConfig: Training configuration
  • Automatic Bradley-Terry loss computation
  • Built-in metrics (accuracy, reward gap)
  
  Pipeline:
  1. Load base model as AutoModelForSequenceClassification(num_labels=1)
  2. Prepare dataset with 'chosen' and 'rejected' columns
  3. Use RewardTrainer with RewardConfig
""")
    
    # ─── Create reward model the TRL way ───
    print("  ── Creating Reward Model (TRL-style) ──")
    
    # For reward modeling, we use a sequence classification model with 1 label
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    total = sum(p.numel() for p in model.parameters())
    print(f"    Model: {model_name} → SequenceClassification (1 label)")
    print(f"    Total parameters: {total:,}")
    
    # ─── Prepare dataset in TRL format ───
    preference_data = {
        "prompt": [
            "What is deep learning?",
            "How do I sort a list in Python?",
            "Explain gravity simply.",
            "What is a database?",
        ] * 4,
        "chosen": [
            "Deep learning is a subset of machine learning using multi-layered neural networks to learn hierarchical representations of data.",
            "Use sorted(my_list) for a new sorted list, or my_list.sort() to sort in place. Both default to ascending order.",
            "Gravity is the force that pulls objects toward each other. The more massive an object, the stronger its gravitational pull.",
            "A database is an organized collection of structured data stored electronically, designed for efficient retrieval, updating, and management of information.",
        ] * 4,
        "rejected": [
            "It's complicated AI stuff with lots of math.",
            "Just use sort I guess",
            "Things fall down because of gravity.",
            "It stores data.",
        ] * 4,
    }
    
    dataset = Dataset.from_dict(preference_data)
    
    # Tokenize in the format that reward training expects
    def preprocess(examples):
        chosen_texts = [
            f"Question: {p}\nAnswer: {c}"
            for p, c in zip(examples["prompt"], examples["chosen"])
        ]
        rejected_texts = [
            f"Question: {p}\nAnswer: {r}"
            for p, r in zip(examples["prompt"], examples["rejected"])
        ]
        
        chosen_enc = tokenizer(
            chosen_texts, truncation=True, max_length=64, padding="max_length",
        )
        rejected_enc = tokenizer(
            rejected_texts, truncation=True, max_length=64, padding="max_length",
        )
        
        return {
            "input_ids_chosen": chosen_enc["input_ids"],
            "attention_mask_chosen": chosen_enc["attention_mask"],
            "input_ids_rejected": rejected_enc["input_ids"],
            "attention_mask_rejected": rejected_enc["attention_mask"],
        }
    
    tokenized = dataset.map(preprocess, batched=True,
                           remove_columns=dataset.column_names)
    
    print(f"    Dataset ready: {len(tokenized)} pairs")
    print(f"    Columns: {tokenized.column_names}")
    
    # ─── Manual training loop (TRL-compatible approach) ───
    print(f"\n  ── Training (manual loop, TRL-compatible format) ──")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    model.train()
    
    for epoch in range(5):
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for i in range(0, len(tokenized), 4):
            batch = tokenized[i:i+4]
            
            chosen_ids = torch.tensor(batch["input_ids_chosen"])
            chosen_mask = torch.tensor(batch["attention_mask_chosen"])
            rejected_ids = torch.tensor(batch["input_ids_rejected"])
            rejected_mask = torch.tensor(batch["attention_mask_rejected"])
            
            r_chosen = model(chosen_ids, attention_mask=chosen_mask).logits.squeeze(-1)
            r_rejected = model(rejected_ids, attention_mask=rejected_mask).logits.squeeze(-1)
            
            loss = -torch.log(torch.sigmoid(r_chosen - r_rejected) + 1e-10).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            correct += (r_chosen > r_rejected).sum().item()
            total_samples += len(batch["input_ids_chosen"])
        
        avg_loss = total_loss / (len(tokenized) // 4)
        acc = correct / total_samples
        print(f"    Epoch {epoch+1}: loss={avg_loss:.4f}, accuracy={acc:.1%}")
    
    # Test the trained reward model
    print(f"\n  ── Testing Trained Reward Model ──")
    model.eval()
    
    test_pairs = [
        ("What is AI?",
         "AI is the simulation of human intelligence by computer systems.",
         "Idk something about computers."),
        ("How to learn Python?",
         "Start with basics like variables and loops, then practice with small projects.",
         "just watch youtube lol"),
    ]
    
    for prompt, chosen, rejected in test_pairs:
        chosen_text = f"Question: {prompt}\nAnswer: {chosen}"
        rejected_text = f"Question: {prompt}\nAnswer: {rejected}"
        
        c_enc = tokenizer(chosen_text, return_tensors="pt", truncation=True,
                         max_length=64, padding="max_length")
        r_enc = tokenizer(rejected_text, return_tensors="pt", truncation=True,
                         max_length=64, padding="max_length")
        
        with torch.no_grad():
            r_c = model(c_enc["input_ids"], c_enc["attention_mask"]).logits.item()
            r_r = model(r_enc["input_ids"], r_enc["attention_mask"]).logits.item()
        
        print(f"    Prompt: \"{prompt}\"")
        print(f"      Chosen reward:   {r_c:.4f}")
        print(f"      Rejected reward: {r_r:.4f}")
        print(f"      Correct ranking: {'✓' if r_c > r_r else '✗'}")
        print()
    
    del model


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all reward model sections."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║    RLHF REWARD MODEL — TRAINING FROM PREFERENCES             ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: From scratch
    train_reward_model_scratch()
    
    # Section 2: From LLM
    reward_model_from_llm()
    
    # Section 3: Preference dataset
    preference_dataset()
    
    # Section 4: Evaluation
    evaluate_reward_model()
    
    # Section 5: TRL-style
    reward_model_trl()
    
    print("\n" + "=" * 65)
    print("  REWARD MODEL MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered:
    ✓ Reward model from scratch (architecture + training)
    ✓ Reward model from pretrained LLM (GPT-2 backbone)
    ✓ Preference dataset creation and formatting
    ✓ Evaluation metrics (accuracy, calibration, distributions)
    ✓ TRL-compatible training pipeline
    """)


if __name__ == "__main__":
    main()
