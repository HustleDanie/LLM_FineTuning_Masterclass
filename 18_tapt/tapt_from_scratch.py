"""
TAPT From Scratch — Implementing Task-Adaptive Pretraining
============================================================

From-scratch implementations:

1. Causal LM TAPT — continued pretraining with CLM objective
2. Masked LM TAPT — BERT-style TAPT with masking
3. Multi-Epoch Scheduling — learning rate strategies for many epochs
4. Curated TAPT — retrieval-augmented data expansion
5. LoRA-TAPT — parameter-efficient task adaptation

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Dict, List, Tuple, Optional


# ============================================================================
# SHARED COMPONENTS
# ============================================================================

class SimpleTransformerLM(nn.Module):
    """Minimal transformer language model for demonstrations."""
    
    def __init__(self, vocab_size=500, embed_dim=64, n_heads=4, n_layers=2, 
                 max_seq_len=64, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, 
            dim_feedforward=embed_dim * 4, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
        # Weight tying
        self.lm_head.weight = self.token_embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, return_hidden=False):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        
        h = self.dropout(self.token_embed(x) + self.pos_embed(positions))
        
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        h = self.transformer(h, mask=causal_mask, is_causal=True)
        
        logits = self.lm_head(h)
        
        if return_hidden:
            return logits, h
        return logits
    
    def perplexity(self, x):
        logits = self.forward(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), x[:, 1:].reshape(-1))
        return math.exp(loss.item())


class TaskDataGenerator:
    """Generate synthetic data that simulates a specific task distribution."""
    
    def __init__(self, vocab_size=500, task_vocab_start=100, task_vocab_end=250,
                 seed=42):
        random.seed(seed)
        torch.manual_seed(seed)
        
        self.vocab_size = vocab_size
        self.task_tokens = list(range(task_vocab_start, task_vocab_end))
        self.general_tokens = [i for i in range(vocab_size) 
                               if i not in self.task_tokens]
    
    def generate_task_data(self, n_examples, seq_len):
        """Generate data from the task distribution (concentrated vocabulary)."""
        data = []
        for _ in range(n_examples):
            seq = []
            for j in range(seq_len):
                if random.random() < 0.8:  # 80% task tokens
                    tok = random.choice(self.task_tokens)
                    # Add bigram structure
                    if j > 0 and random.random() < 0.4:
                        tok = (seq[-1] + random.choice([1, 2, 3])) % self.vocab_size
                        if tok not in range(self.task_tokens[0], self.task_tokens[-1]+1):
                            tok = random.choice(self.task_tokens)
                else:
                    tok = random.choice(self.general_tokens)
                seq.append(tok)
            data.append(seq)
        return torch.tensor(data)
    
    def generate_general_data(self, n_examples, seq_len):
        """Generate data from general distribution (broad vocabulary)."""
        data = []
        for _ in range(n_examples):
            seq = []
            for j in range(seq_len):
                # Zipf-like distribution over full vocabulary
                tok = int(random.paretovariate(1.2)) % self.vocab_size
                seq.append(tok)
            data.append(seq)
        return torch.tensor(data)
    
    def generate_domain_data(self, n_examples, seq_len):
        """Generate data from a domain distribution (between general and task)."""
        data = []
        domain_tokens = list(range(50, 300))  # Broader than task
        for _ in range(n_examples):
            seq = []
            for j in range(seq_len):
                if random.random() < 0.5:
                    tok = random.choice(domain_tokens)
                else:
                    tok = random.randint(0, self.vocab_size - 1)
                seq.append(tok)
            data.append(seq)
        return torch.tensor(data)


# ============================================================================
# SECTION 1: CAUSAL LM TAPT
# ============================================================================

def causal_lm_tapt():
    """Standard TAPT using causal language modeling objective."""
    print("=" * 70)
    print("  SECTION 1: CAUSAL LM TAPT (CONTINUED PRETRAINING)")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    vocab_size = 500
    embed_dim = 64
    seq_len = 32
    
    # Generate data
    gen = TaskDataGenerator(vocab_size)
    
    # Phase 1: "Pretrained" model on general data
    general_train = gen.generate_general_data(200, seq_len)
    general_eval = gen.generate_general_data(50, seq_len)
    
    # Phase 2: Task data (small!)
    task_train = gen.generate_task_data(100, seq_len)  # Only 100 examples!
    task_eval = gen.generate_task_data(30, seq_len)
    
    model = SimpleTransformerLM(vocab_size, embed_dim)
    criterion = nn.CrossEntropyLoss()
    
    # Stage 1: General pretraining
    print(f"\n  ── Stage 1: General Pretraining ──\n")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(20):
        model.train()
        perm = torch.randperm(general_train.size(0))
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(general_train), 32):
            batch = general_train[perm[i:i+32]]
            if batch.size(0) == 0:
                continue
            
            logits = model(batch[:, :-1])
            loss = criterion(logits.reshape(-1, vocab_size), batch[:, 1:].reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
    
    general_ppl_before = model.perplexity(general_eval)
    task_ppl_before = model.perplexity(task_eval)
    
    print(f"  After pretraining:")
    print(f"    General PPL: {general_ppl_before:.1f}")
    print(f"    Task PPL:    {task_ppl_before:.1f} ← gap shows need for TAPT")
    
    # Stage 2: TAPT
    print(f"\n  ── Stage 2: Task-Adaptive Pretraining (TAPT) ──\n")
    print(f"  Training on {task_train.size(0)} task examples for multiple epochs")
    print(f"  Using LOWER learning rate: 2e-4 (vs 1e-3 for pretraining)\n")
    
    # Key TAPT settings
    tapt_lr = 2e-4  # Much lower than pretraining
    tapt_epochs = 50  # Many epochs because data is tiny
    warmup_steps = 50
    
    optimizer = torch.optim.Adam(model.parameters(), lr=tapt_lr)
    
    # Cosine schedule with warmup
    total_steps = tapt_epochs * (len(task_train) // 16 + 1)
    
    def get_lr(step):
        if step < warmup_steps:
            return tapt_lr * step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return tapt_lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    step = 0
    results = []
    
    for epoch in range(tapt_epochs):
        model.train()
        perm = torch.randperm(task_train.size(0))
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, len(task_train), 16):
            batch = task_train[perm[i:i+16]]
            if batch.size(0) == 0:
                continue
            
            # Update LR
            lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            
            logits = model(batch[:, :-1])
            loss = criterion(logits.reshape(-1, vocab_size), batch[:, 1:].reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            step += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            task_ppl = model.perplexity(task_eval)
            gen_ppl = model.perplexity(general_eval)
            results.append((epoch + 1, task_ppl, gen_ppl, lr))
    
    print(f"  {'Epoch':>6} │ {'Task PPL':>8} │ {'Gen PPL':>8} │ {'LR':>10} │ {'Task Δ':>8}")
    print(f"  {'─'*6}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*10}─┼─{'─'*8}")
    
    for ep, t_ppl, g_ppl, lr in results:
        t_delta = (1 - t_ppl / task_ppl_before) * 100
        print(f"  {ep:>6} │ {t_ppl:>8.1f} │ {g_ppl:>8.1f} │ {lr:>10.6f} │ {t_delta:>+7.1f}%")
    
    print(f"""
  ═══ CLM-TAPT Results ═══
  
  Task PPL:    {task_ppl_before:.1f} → {results[-1][1]:.1f} ({(1 - results[-1][1]/task_ppl_before)*100:+.1f}%)
  General PPL: {general_ppl_before:.1f} → {results[-1][2]:.1f} ({(results[-1][2]/general_ppl_before - 1)*100:+.1f}%)
  
  TAPT significantly reduced task perplexity with minimal
  general capability degradation, using only {task_train.size(0)} examples!
""")


# ============================================================================
# SECTION 2: MASKED LM TAPT
# ============================================================================

def masked_lm_tapt():
    """TAPT using masked language modeling (BERT-style)."""
    print("\n\n" + "=" * 70)
    print("  SECTION 2: MASKED LM TAPT (BERT-STYLE)")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    vocab_size = 500
    embed_dim = 64
    seq_len = 32
    mask_token_id = vocab_size - 1  # Use last token as [MASK]
    mask_prob = 0.15
    
    class SimpleMaskedLM(nn.Module):
        """Bidirectional transformer for Masked LM TAPT."""
        
        def __init__(self, vocab_size, embed_dim, n_heads=4, n_layers=2):
            super().__init__()
            self.vocab_size = vocab_size
            self.token_embed = nn.Embedding(vocab_size, embed_dim)
            self.pos_embed = nn.Embedding(128, embed_dim)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads,
                dim_feedforward=embed_dim * 4, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.mlm_head = nn.Linear(embed_dim, vocab_size)
            self.mlm_head.weight = self.token_embed.weight
        
        def forward(self, x):
            B, T = x.shape
            positions = torch.arange(T, device=x.device).unsqueeze(0)
            h = self.token_embed(x) + self.pos_embed(positions)
            h = self.encoder(h)  # Bidirectional (no causal mask)
            return self.mlm_head(h)
    
    def create_mlm_batch(batch, mask_prob=0.15, mask_token_id=mask_token_id):
        """Apply 80/10/10 masking strategy."""
        labels = batch.clone()
        masked_input = batch.clone()
        
        # Create mask
        mask = torch.bernoulli(torch.full(batch.shape, mask_prob)).bool()
        labels[~mask] = -100  # Only compute loss on masked positions
        
        # 80% replace with [MASK]
        mask_replace = mask & (torch.rand(batch.shape) < 0.8)
        masked_input[mask_replace] = mask_token_id
        
        # 10% replace with random token
        mask_random = mask & ~mask_replace & (torch.rand(batch.shape) < 0.5)
        masked_input[mask_random] = torch.randint(0, vocab_size - 1, (mask_random.sum(),))
        
        # 10% keep original (remaining masked positions)
        
        return masked_input, labels
    
    # Generate data
    gen = TaskDataGenerator(vocab_size)
    task_train = gen.generate_task_data(100, seq_len)
    task_eval = gen.generate_task_data(30, seq_len)
    
    model = SimpleMaskedLM(vocab_size, embed_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    print(f"\n  ── MLM-TAPT Training ──")
    print(f"  Data: {task_train.size(0)} examples, mask_prob={mask_prob}")
    print(f"  Training for 50 epochs with {mask_prob:.0%} masking\n")
    
    results = []
    for epoch in range(50):
        model.train()
        perm = torch.randperm(task_train.size(0))
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, len(task_train), 16):
            batch = task_train[perm[i:i+16]]
            if batch.size(0) == 0:
                continue
            
            masked_input, labels = create_mlm_batch(batch)
            logits = model(masked_input)
            loss = criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            
            # Evaluate: mask accuracy on eval set
            model.eval()
            with torch.no_grad():
                masked_eval, eval_labels = create_mlm_batch(task_eval)
                eval_logits = model(masked_eval)
                
                # Accuracy on masked positions only
                mask_positions = eval_labels != -100
                if mask_positions.sum() > 0:
                    preds = eval_logits[mask_positions].argmax(dim=-1)
                    targets = eval_labels[mask_positions]
                    accuracy = (preds == targets).float().mean().item()
                else:
                    accuracy = 0.0
            
            results.append((epoch + 1, avg_loss, accuracy))
    
    print(f"  {'Epoch':>6} │ {'MLM Loss':>8} │ {'Mask Acc':>8} │ Visualization")
    print(f"  {'─'*6}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*30}")
    
    for ep, loss, acc in results:
        bar = "█" * int(acc * 20)
        print(f"  {ep:>6} │ {loss:>8.3f} │ {acc:>7.1%} │ {bar}")
    
    print(f"""
  ═══ MLM-TAPT vs CLM-TAPT ═══
  
  ┌─────────────┬──────────────────────┬──────────────────────┐
  │ Aspect      │ CLM-TAPT             │ MLM-TAPT             │
  ├─────────────┼──────────────────────┼──────────────────────┤
  │ Direction   │ Left-to-right only   │ Bidirectional        │
  │ Best for    │ GPT-style models     │ BERT-style models    │
  │ Objective   │ Predict next token   │ Predict masked token │
  │ Context     │ Left context only    │ Full context         │
  │ Typical use │ Text generation      │ Classification, NER  │
  └─────────────┴──────────────────────┴──────────────────────┘
  
  Choose based on your BASE model type:
  - GPT/LLaMA/Mistral → CLM-TAPT
  - BERT/RoBERTa/DeBERTa → MLM-TAPT
""")


# ============================================================================
# SECTION 3: MULTI-EPOCH SCHEDULING
# ============================================================================

def multi_epoch_scheduling():
    """Learning rate scheduling strategies for TAPT's many epochs."""
    print("\n\n" + "=" * 70)
    print("  SECTION 3: MULTI-EPOCH SCHEDULING")
    print("=" * 70)
    
    print(f"""
  ═══ LR Scheduling for TAPT ═══
  
  TAPT runs for many epochs on small data, so scheduling matters.
  We compare 4 strategies.
""")
    
    torch.manual_seed(42)
    
    vocab_size = 500
    embed_dim = 64
    seq_len = 32
    n_epochs = 80
    
    gen = TaskDataGenerator(vocab_size)
    task_train = gen.generate_task_data(80, seq_len)
    task_eval = gen.generate_task_data(30, seq_len)
    
    # Define scheduling strategies
    class ScheduleConfig:
        def __init__(self, name, base_lr, warmup_frac=0.05):
            self.name = name
            self.base_lr = base_lr
            self.warmup_frac = warmup_frac
    
    schedules = {
        "constant": ScheduleConfig("Constant LR", 2e-4),
        "cosine": ScheduleConfig("Cosine Decay", 2e-4, 0.05),
        "linear": ScheduleConfig("Linear Decay", 2e-4, 0.05),
        "cosine_restarts": ScheduleConfig("Cosine w/ Restarts", 2e-4, 0.02),
    }
    
    def get_lr_schedule(schedule_name, step, total_steps, config):
        warmup_steps = int(total_steps * config.warmup_frac)
        
        if step < warmup_steps:
            return config.base_lr * step / max(warmup_steps, 1)
        
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        
        if schedule_name == "constant":
            return config.base_lr
        elif schedule_name == "cosine":
            return config.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        elif schedule_name == "linear":
            return config.base_lr * (1 - progress)
        elif schedule_name == "cosine_restarts":
            # 4 restarts
            n_restarts = 4
            cycle_progress = (progress * n_restarts) % 1.0
            return config.base_lr * 0.5 * (1 + math.cos(math.pi * cycle_progress))
        
        return config.base_lr
    
    steps_per_epoch = max(1, len(task_train) // 16)
    total_steps = n_epochs * steps_per_epoch
    
    all_results = {}
    criterion = nn.CrossEntropyLoss()
    
    for sched_name, config in schedules.items():
        torch.manual_seed(42)
        model = SimpleTransformerLM(vocab_size, embed_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.base_lr)
        
        results = []
        step = 0
        
        for epoch in range(n_epochs):
            model.train()
            perm = torch.randperm(task_train.size(0))
            
            for i in range(0, len(task_train), 16):
                batch = task_train[perm[i:i+16]]
                if batch.size(0) == 0:
                    continue
                
                lr = get_lr_schedule(sched_name, step, total_steps, config)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                
                logits = model(batch[:, :-1])
                loss = criterion(logits.reshape(-1, vocab_size), batch[:, 1:].reshape(-1))
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                step += 1
            
            if (epoch + 1) % 20 == 0:
                task_ppl = model.perplexity(task_eval)
                results.append((epoch + 1, task_ppl))
        
        all_results[sched_name] = results
    
    print(f"\n  ── Task Perplexity by Schedule ({n_epochs} epochs, {task_train.size(0)} examples) ──\n")
    
    header = f"  {'Epoch':>6}"
    for name in schedules:
        header += f" │ {schedules[name].name:>18}"
    print(header)
    print(f"  {'─'*6}" + "─┼─" + ("─" * 18 + "─┼─") * (len(schedules) - 1) + "─" * 18)
    
    for i, epoch in enumerate([20, 40, 60, 80]):
        row = f"  {epoch:>6}"
        for sched_name in schedules:
            if i < len(all_results[sched_name]):
                ppl = all_results[sched_name][i][1]
                row += f" │ {ppl:>18.1f}"
            else:
                row += f" │ {'N/A':>18}"
        print(row)
    
    # Find best schedule
    final_ppls = {name: results[-1][1] for name, results in all_results.items()}
    best_sched = min(final_ppls, key=final_ppls.get)
    
    print(f"\n  Best schedule: {schedules[best_sched].name} (PPL = {final_ppls[best_sched]:.1f})")
    
    print(f"""
  ═══ Scheduling Recommendations for TAPT ═══
  
  ┌────────────────────┬──────────────┬───────────────────────────┐
  │ Schedule           │ When to Use  │ Notes                     │
  ├────────────────────┼──────────────┼───────────────────────────┤
  │ Cosine Decay       │ Most cases   │ Best default for TAPT ★   │
  │ Cosine w/ Restarts │ > 50 epochs  │ Handles long training     │
  │ Linear Decay       │ Moderate     │ Simple, reliable          │
  │ Constant           │ Short TAPT   │ < 10 epochs only          │
  └────────────────────┴──────────────┴───────────────────────────┘
""")


# ============================================================================
# SECTION 4: CURATED TAPT (RETRIEVAL-AUGMENTED)
# ============================================================================

def curated_tapt():
    """Expand TAPT data by retrieving task-similar examples from a large pool."""
    print("\n\n" + "=" * 70)
    print("  SECTION 4: CURATED TAPT (RETRIEVAL-AUGMENTED)")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    vocab_size = 500
    embed_dim = 64
    seq_len = 32
    
    gen = TaskDataGenerator(vocab_size)
    task_train = gen.generate_task_data(50, seq_len)   # Very small!
    task_eval = gen.generate_task_data(30, seq_len)
    
    # Large unlabeled pool (domain + general mixed)
    pool_domain = gen.generate_domain_data(200, seq_len)   # Some task-relevant
    pool_general = gen.generate_general_data(300, seq_len)  # Mostly irrelevant
    pool = torch.cat([pool_domain, pool_general], dim=0)
    
    print(f"""
  ═══ Curated TAPT Pipeline ═══
  
  Task data:    {task_train.size(0)} examples (tiny!)
  Pool:         {pool.size(0)} unlabeled examples
  Goal:         Retrieve task-similar examples from pool to expand TAPT data
""")
    
    # Step 1: Compute embeddings (simple bag-of-tokens representation)
    def compute_embedding(data, vocab_size):
        """Simple bag-of-tokens embedding (production would use model embeddings)."""
        embeddings = torch.zeros(data.size(0), vocab_size)
        for i in range(data.size(0)):
            for tok in data[i]:
                embeddings[i, tok] += 1
        # L2 normalize
        norms = embeddings.norm(dim=1, keepdim=True) + 1e-8
        embeddings = embeddings / norms
        return embeddings
    
    task_embs = compute_embedding(task_train, vocab_size)
    pool_embs = compute_embedding(pool, vocab_size)
    
    # Step 2: Retrieve k nearest neighbors
    k = 5
    
    similarity = torch.mm(task_embs, pool_embs.t())  # [n_task, n_pool]
    topk_sims, topk_indices = similarity.topk(k, dim=1)
    
    # Collect unique retrieved indices
    retrieved_indices = set()
    for row in topk_indices:
        for idx in row.tolist():
            retrieved_indices.add(idx)
    
    # Classify quality: pool[:200] is domain (more relevant), pool[200:] is general
    relevant_count = sum(1 for idx in retrieved_indices if idx < 200)
    irrelevant_count = sum(1 for idx in retrieved_indices if idx >= 200)
    
    print(f"  Step 1: Computed embeddings")
    print(f"  Step 2: Retrieved {k} neighbors per example")
    print(f"  Step 3: Got {len(retrieved_indices)} unique pool examples")
    print(f"    - From domain pool: {relevant_count} ({relevant_count/len(retrieved_indices):.0%})")
    print(f"    - From general pool: {irrelevant_count} ({irrelevant_count/len(retrieved_indices):.0%})")
    
    # Build curated TAPT dataset
    curated_indices = list(retrieved_indices)
    curated_data = pool[curated_indices]
    expanded_train = torch.cat([task_train, curated_data], dim=0)
    
    print(f"\n  Expanded dataset: {task_train.size(0)} → {expanded_train.size(0)} examples ({expanded_train.size(0)/task_train.size(0):.1f}x)")
    
    # Step 3: Compare Standard TAPT vs Curated TAPT
    print(f"\n  ── Training Comparison ──\n")
    
    criterion = nn.CrossEntropyLoss()
    n_epochs_standard = 50
    n_epochs_curated = 30  # Fewer epochs needed with more data
    
    def run_tapt(train_data, eval_data, n_epochs, lr=2e-4, label=""):
        torch.manual_seed(42)
        model = SimpleTransformerLM(vocab_size, embed_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        total_steps = n_epochs * max(1, len(train_data) // 16)
        results = []
        step = 0
        
        for epoch in range(n_epochs):
            model.train()
            perm = torch.randperm(train_data.size(0))
            
            for i in range(0, len(train_data), 16):
                batch = train_data[perm[i:i+16]]
                if batch.size(0) == 0:
                    continue
                
                # Cosine schedule
                warmup = int(total_steps * 0.05)
                if step < warmup:
                    cur_lr = lr * step / max(warmup, 1)
                else:
                    progress = (step - warmup) / max(total_steps - warmup, 1)
                    cur_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))
                
                for pg in optimizer.param_groups:
                    pg['lr'] = cur_lr
                
                logits = model(batch[:, :-1])
                loss = criterion(logits.reshape(-1, vocab_size), batch[:, 1:].reshape(-1))
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                step += 1
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                ppl = model.perplexity(eval_data)
                results.append((epoch + 1, ppl))
        
        return results
    
    standard_results = run_tapt(task_train, task_eval, n_epochs_standard, label="Standard")
    curated_results = run_tapt(expanded_train, task_eval, n_epochs_curated, label="Curated")
    
    print(f"  Standard TAPT ({task_train.size(0)} examples, {n_epochs_standard} epochs):")
    for ep, ppl in standard_results:
        print(f"    Epoch {ep:>3}: PPL = {ppl:.1f}")
    
    print(f"\n  Curated TAPT ({expanded_train.size(0)} examples, {n_epochs_curated} epochs):")
    for ep, ppl in curated_results:
        print(f"    Epoch {ep:>3}: PPL = {ppl:.1f}")
    
    std_final = standard_results[-1][1]
    cur_final = curated_results[-1][1]
    
    print(f"""
  ═══ Curated TAPT Results ═══
  
  Standard TAPT final PPL: {std_final:.1f}
  Curated TAPT final PPL:  {cur_final:.1f}
  Improvement:             {(1 - cur_final/std_final)*100:+.1f}%
  
  Curated TAPT benefits from a larger, more diverse dataset
  while staying focused on task-relevant text.
  
  ═══ Curated TAPT Strategies ═══
  
  ┌───────────────────┬─────────────────────┬────────────────────────┐
  │ Retrieval Method  │ Quality             │ Compute Cost           │
  ├───────────────────┼─────────────────────┼────────────────────────┤
  │ Bag-of-words      │ Basic (lexical)     │ Very low               │
  │ TF-IDF + cosine   │ Good (weighted)     │ Low                    │
  │ BM25              │ Good (sparse)       │ Low                    │
  │ Dense (model emb) │ Best (semantic) ★   │ Moderate               │
  │ Hybrid (BM25+emb) │ Best overall ★★     │ Moderate               │
  └───────────────────┴─────────────────────┴────────────────────────┘
""")


# ============================================================================
# SECTION 5: LoRA-TAPT
# ============================================================================

def lora_tapt():
    """Parameter-efficient TAPT using LoRA adapters."""
    print("\n\n" + "=" * 70)
    print("  SECTION 5: LoRA-TAPT (PARAMETER-EFFICIENT)")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    vocab_size = 500
    embed_dim = 64
    seq_len = 32
    
    gen = TaskDataGenerator(vocab_size)
    task_train = gen.generate_task_data(100, seq_len)
    task_eval = gen.generate_task_data(30, seq_len)
    general_eval = gen.generate_general_data(50, seq_len)
    
    class LoRALinear(nn.Module):
        """LoRA adapter for a linear layer."""
        
        def __init__(self, in_features, out_features, rank=4, alpha=8):
            super().__init__()
            self.original = nn.Linear(in_features, out_features)
            self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
            self.scaling = alpha / rank
        
        def forward(self, x):
            original_out = self.original(x)
            lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
            return original_out + lora_out
        
        @property
        def trainable_params(self):
            return self.lora_A.numel() + self.lora_B.numel()
        
        @property
        def total_params(self):
            return sum(p.numel() for p in self.original.parameters()) + self.trainable_params
    
    class LoRATAPTModel(nn.Module):
        """Transformer with LoRA adapters on attention layers for TAPT."""
        
        def __init__(self, vocab_size, embed_dim, n_heads=4, n_layers=2, 
                     lora_rank=4, lora_alpha=8):
            super().__init__()
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            
            self.token_embed = nn.Embedding(vocab_size, embed_dim)
            self.pos_embed = nn.Embedding(128, embed_dim)
            
            # Build transformer layers with LoRA
            self.layers = nn.ModuleList()
            self.lora_layers = nn.ModuleList()
            
            for _ in range(n_layers):
                layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim, nhead=n_heads,
                    dim_feedforward=embed_dim * 4, batch_first=True
                )
                self.layers.append(layer)
                
                # LoRA on attention output projection
                lora = LoRALinear(embed_dim, embed_dim, lora_rank, lora_alpha)
                self.lora_layers.append(lora)
            
            self.lm_head = nn.Linear(embed_dim, vocab_size)
            self.lm_head.weight = self.token_embed.weight
            
            # Freeze everything except LoRA
            self._freeze_non_lora()
        
        def _freeze_non_lora(self):
            for param in self.parameters():
                param.requires_grad = False
            for lora in self.lora_layers:
                lora.lora_A.requires_grad = True
                lora.lora_B.requires_grad = True
        
        def forward(self, x):
            B, T = x.shape
            positions = torch.arange(T, device=x.device).unsqueeze(0)
            h = self.token_embed(x) + self.pos_embed(positions)
            
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
            
            for layer, lora in zip(self.layers, self.lora_layers):
                h = layer(h, src_mask=causal_mask, is_causal=True)
                h = h + lora(h) - lora.original(h)  # Replace with LoRA version
            
            return self.lm_head(h)
        
        def trainable_param_count(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        def total_param_count(self):
            return sum(p.numel() for p in self.parameters())
    
    # Compare Full TAPT vs LoRA-TAPT at different ranks
    criterion = nn.CrossEntropyLoss()
    
    def eval_ppl(model, data):
        model.eval()
        with torch.no_grad():
            logits = model(data[:, :-1])
            loss = criterion(logits.reshape(-1, vocab_size), data[:, 1:].reshape(-1))
            return math.exp(loss.item())
    
    print(f"\n  ── Comparing Full TAPT vs LoRA-TAPT ──\n")
    
    configs = [
        ("Full TAPT", None, 2e-4),
        ("LoRA r=2", 2, 5e-4),
        ("LoRA r=4", 4, 5e-4),
        ("LoRA r=8", 8, 5e-4),
        ("LoRA r=16", 16, 5e-4),
    ]
    
    results = {}
    
    for name, rank, lr in configs:
        torch.manual_seed(42)
        
        if rank is None:
            # Full TAPT (all params trainable)
            model = SimpleTransformerLM(vocab_size, embed_dim)
            trainable = sum(p.numel() for p in model.parameters())
            total = trainable
        else:
            model = LoRATAPTModel(vocab_size, embed_dim, lora_rank=rank, lora_alpha=rank*2)
            trainable = model.trainable_param_count()
            total = model.total_param_count()
        
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )
        
        ppl_before = eval_ppl(model, task_eval)
        
        for epoch in range(50):
            model.train()
            perm = torch.randperm(task_train.size(0))
            
            for i in range(0, len(task_train), 16):
                batch = task_train[perm[i:i+16]]
                if batch.size(0) == 0:
                    continue
                
                logits = model(batch[:, :-1])
                loss = criterion(logits.reshape(-1, vocab_size), batch[:, 1:].reshape(-1))
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        
        ppl_after = eval_ppl(model, task_eval)
        gen_ppl = eval_ppl(model, general_eval)
        
        results[name] = {
            "trainable": trainable,
            "total": total,
            "pct": trainable / total * 100,
            "task_ppl": ppl_after,
            "task_delta": (1 - ppl_after / ppl_before) * 100,
            "gen_ppl": gen_ppl,
        }
    
    print(f"  {'Method':<14} │ {'Trainable':>10} │ {'% Params':>8} │ {'Task PPL':>8} │ {'Task Δ':>8} │ {'Gen PPL':>8}")
    print(f"  {'─'*14}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}")
    
    for name, r in results.items():
        print(f"  {name:<14} │ {r['trainable']:>10,} │ {r['pct']:>7.1f}% │ {r['task_ppl']:>8.1f} │ {r['task_delta']:>+7.1f}% │ {r['gen_ppl']:>8.1f}")
    
    print(f"""
  ═══ LoRA-TAPT Insights ═══
  
  1. LoRA-TAPT achieves close to full TAPT quality with far fewer params
  2. Higher LR works for LoRA (5e-4 vs 2e-4 for full)
  3. LoRA better preserves general capabilities (lower Gen PPL increase)
  4. For TAPT's small data, even r=4 is often sufficient
  
  ═══ LoRA-TAPT is ideal when: ═══
  
  • You want to serve multiple tasks from one base model
  • Each task gets its own tiny LoRA adapter (few MB each)
  • Prevents catastrophic forgetting of general capabilities
  • Adapter stacking: DAPT-LoRA + TAPT-LoRA can be combined
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  TAPT FROM SCRATCH — IMPLEMENTING TASK-ADAPTIVE PRETRAINING     ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    causal_lm_tapt()
    masked_lm_tapt()
    multi_epoch_scheduling()
    curated_tapt()
    lora_tapt()
    
    print("\n" + "=" * 70)
    print("  FROM-SCRATCH MODULE COMPLETE")
    print("=" * 70)
    print("""
    Covered:
    ✓ Causal LM TAPT: continued pretraining on task text
    ✓ Masked LM TAPT: BERT-style masking on task text
    ✓ Multi-epoch scheduling: cosine, linear, restarts
    ✓ Curated TAPT: retrieval-augmented data expansion
    ✓ LoRA-TAPT: parameter-efficient task adaptation
    """)


if __name__ == "__main__":
    main()
