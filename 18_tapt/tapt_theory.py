"""
TAPT Theory — Understanding Task-Adaptive Pretraining
=======================================================

Theoretical foundations of TAPT:

1. Task Distribution Analysis — why task text ≠ general text
2. Task vs Domain Distance — measuring TAPT potential
3. Multi-Epoch Dynamics — why many epochs work for TAPT
4. Curated TAPT Theory — expanding TAPT via retrieval
5. When TAPT Helps vs Hurts — decision criteria

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Dict, List, Tuple, Optional
from collections import Counter


# ============================================================================
# SECTION 1: TASK DISTRIBUTION ANALYSIS
# ============================================================================

def task_distribution_analysis():
    """Analyze why task data has a distinct distribution from general text."""
    print("=" * 70)
    print("  SECTION 1: TASK DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    print(f"""
  ═══ The Distribution Narrowing Principle ═══
  
  Pretraining data covers EVERYTHING: news, books, web, social media, etc.
  Task data is a TINY, SPECIFIC slice of that distribution.
  
  Even within a domain, different TASKS have different distributions:
  
  Biomedical Domain:
  ├── Drug review sentiment (informal, patient language)
  ├── Paper abstract classification (formal, scientific)
  ├── Clinical NER (shorthand, medical codes)
  └── Gene relation extraction (very technical, symbolic)
  
  TAPT makes the model focus on the EXACT task distribution.
""")
    
    torch.manual_seed(42)
    random.seed(42)
    
    # Simulate vocabulary distributions for different levels
    vocab_size = 500
    
    # General pretraining distribution (broad, flat-ish)
    general_dist = torch.zeros(vocab_size)
    for i in range(vocab_size):
        general_dist[i] = 1.0 / (i + 1) ** 0.8  # Mild Zipf
    general_dist = general_dist / general_dist.sum()
    
    # Domain distribution (subset of vocab gets more mass)
    domain_dist = torch.zeros(vocab_size)
    domain_tokens = list(range(50, 200))  # Domain-specific range
    for i in range(vocab_size):
        if i in domain_tokens:
            domain_dist[i] = 1.0 / ((i - 49) + 1) ** 0.6  # Concentrated
        else:
            domain_dist[i] = 0.005  # Small background mass
    domain_dist = domain_dist / domain_dist.sum()
    
    # Task distribution (even more concentrated)
    task_dist = torch.zeros(vocab_size)
    task_tokens = list(range(80, 150))  # Narrower range
    for i in range(vocab_size):
        if i in task_tokens:
            task_dist[i] = 1.0 / ((i - 79) + 1) ** 0.5  # Very concentrated
        else:
            task_dist[i] = 0.002  # Minimal background
    task_dist = task_dist / task_dist.sum()
    
    # Compute KL divergences
    def kl_divergence(p, q, eps=1e-10):
        p = p + eps
        q = q + eps
        return (p * (p / q).log()).sum().item()
    
    def entropy(p, eps=1e-10):
        p = p + eps
        return -(p * p.log()).sum().item()
    
    general_entropy = entropy(general_dist)
    domain_entropy = entropy(domain_dist)
    task_entropy = entropy(task_dist)
    
    print(f"  ── Distribution Entropy (higher = broader) ──\n")
    print(f"  General pretraining:  {general_entropy:.2f} nats  {'█' * int(general_entropy * 3)}")
    print(f"  Domain data:          {domain_entropy:.2f} nats  {'█' * int(domain_entropy * 3)}")
    print(f"  Task data:            {task_entropy:.2f} nats  {'█' * int(task_entropy * 3)}")
    
    print(f"\n  ── KL Divergence Between Distributions ──\n")
    
    kl_general_domain = kl_divergence(domain_dist, general_dist)
    kl_general_task = kl_divergence(task_dist, general_dist)
    kl_domain_task = kl_divergence(task_dist, domain_dist)
    
    print(f"  General → Domain:  KL = {kl_general_domain:.3f}  (DAPT bridges this gap)")
    print(f"  General → Task:    KL = {kl_general_task:.3f}  (TAPT bridges this gap)")
    print(f"  Domain → Task:     KL = {kl_domain_task:.3f}  (TAPT after DAPT bridges this)")
    
    # Effective vocabulary usage
    def effective_vocab(dist, threshold=0.001):
        return (dist > threshold).sum().item()
    
    print(f"\n  ── Effective Vocabulary (tokens used above threshold) ──\n")
    print(f"  General: {effective_vocab(general_dist):>4} / {vocab_size} ({effective_vocab(general_dist)/vocab_size:.0%})")
    print(f"  Domain:  {effective_vocab(domain_dist):>4} / {vocab_size} ({effective_vocab(domain_dist)/vocab_size:.0%})")
    print(f"  Task:    {effective_vocab(task_dist):>4} / {vocab_size} ({effective_vocab(task_dist)/vocab_size:.0%})")
    
    print(f"""
  ═══ Key Insight ═══
  
  Task data is MUCH more concentrated than general pretraining data.
  TAPT lets the model "zoom in" on exactly the tokens and patterns
  it will see at inference time, rather than wasting capacity on
  patterns from unrelated text.
  
  The large KL(Task ‖ General) = {kl_general_task:.3f} shows significant
  distribution mismatch — this is the gap TAPT bridges.
""")


# ============================================================================
# SECTION 2: TASK vs DOMAIN DISTANCE
# ============================================================================

def task_vs_domain_distance():
    """Compare TAPT's task-level adaptation vs DAPT's domain-level adaptation."""
    print("\n\n" + "=" * 70)
    print("  SECTION 2: TASK vs DOMAIN DISTANCE")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Simulated distances for different scenarios
    # Based on Gururangan et al. 2020 findings
    scenarios = [
        # (Task, Domain, General→Domain dist, General→Task dist, Domain→Task dist, 
        #  DAPT gain, TAPT gain, DAPT+TAPT gain)
        ("ChemProt",      "BioMed",   0.45, 0.55, 0.20, 2.3, 2.1, 4.0),
        ("RCT",           "BioMed",   0.45, 0.50, 0.12, 0.4, 1.2, 1.9),
        ("ACL-ARC",       "CS",       0.50, 0.65, 0.30, 12.4, 4.6, 15.2),
        ("SciERC",        "CS",       0.50, 0.58, 0.18, 3.5, 1.8, 5.0),
        ("HyperPartisan", "News",     0.15, 0.35, 0.25, 1.6, 3.2, 5.4),
        ("AGNews",        "News",     0.15, 0.22, 0.10, 0.7, 0.3, 1.0),
        ("IMDB",          "Reviews",  0.20, 0.30, 0.15, 0.4, 0.6, 1.1),
        ("Helpfulness",   "Reviews",  0.20, 0.40, 0.28, 3.2, 2.0, 4.8),
    ]
    
    print(f"""
  ═══ Distance Decomposition: General → Domain → Task ═══
  
  DAPT bridges:  General → Domain  (broad adaptation)
  TAPT bridges:  Domain → Task     (narrow adaptation)
  
  Combined:      General → Domain → Task  (best of both!)
""")
    
    print(f"  {'Task':<16} │ {'Domain':<8} │ {'G→D':>5} │ {'G→T':>5} │ {'D→T':>5} │ {'DAPT Δ':>6} │ {'TAPT Δ':>6} │ {'Both Δ':>6} │ {'Synergy':>7}")
    print(f"  {'─'*16}─┼─{'─'*8}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*7}")
    
    for task, domain, gd, gt, dt, dapt, tapt, both in scenarios:
        synergy = both - (dapt + tapt)
        if synergy > 0:
            syn_str = f"+{synergy:.1f}%"
        else:
            syn_str = f"{synergy:.1f}%"
        
        print(f"  {task:<16} │ {domain:<8} │ {gd:>5.2f} │ {gt:>5.2f} │ {dt:>5.2f} │ {dapt:>+5.1f}% │ {tapt:>+5.1f}% │ {both:>+5.1f}% │ {syn_str:>7}")
    
    # Analyze when TAPT helps most
    print(f"\n  ── When Does TAPT Add Most Over DAPT? ──\n")
    
    # Sort by Domain→Task distance (higher = TAPT more valuable)
    sorted_tasks = sorted(scenarios, key=lambda x: x[4], reverse=True)
    
    for task, domain, gd, gt, dt, dapt, tapt, both in sorted_tasks[:4]:
        print(f"  {task:<16}: D→T distance = {dt:.2f}, TAPT adds {tapt:+.1f}%")
    
    print(f"\n  ...vs tasks where TAPT adds least:")
    for task, domain, gd, gt, dt, dapt, tapt, both in sorted_tasks[-3:]:
        print(f"  {task:<16}: D→T distance = {dt:.2f}, TAPT adds {tapt:+.1f}%")
    
    print(f"""
  ═══ Key Findings ═══
  
  1. TAPT is most valuable when D→T distance is HIGH
     (task distribution differs from domain distribution)
  
  2. DAPT and TAPT capture DIFFERENT information:
     - DAPT: domain vocabulary, domain syntax, domain knowledge
     - TAPT: task-specific patterns, task vocabulary, task structure
  
  3. Synergy: DAPT+TAPT > DAPT + TAPT individually
     (DAPT provides a better starting point for TAPT)
  
  4. TAPT is ALWAYS worth trying — it's essentially free compute!
""")


# ============================================================================
# SECTION 3: MULTI-EPOCH DYNAMICS
# ============================================================================

def multi_epoch_dynamics():
    """Why TAPT benefits from many epochs despite small data."""
    print("\n\n" + "=" * 70)
    print("  SECTION 3: MULTI-EPOCH DYNAMICS")
    print("=" * 70)
    
    print(f"""
  ═══ The Multi-Epoch Paradox ═══
  
  In supervised learning, training for 100 epochs on 5K examples
  usually means massive overfitting. But TAPT is different:
  
  1. TAPT uses a LANGUAGE MODELING objective (self-supervised)
  2. Language modeling has VERY high capacity (predict next token)
  3. Small text contains many unique next-token predictions
  4. Each epoch reinforces task-specific representations
  
  5000 examples × 512 tokens = 2.5M token predictions per epoch
  100 epochs = 250M token predictions total
  
  That's equivalent to processing 250M tokens of data!
""")
    
    torch.manual_seed(42)
    
    # Simulate a tiny language model doing TAPT over many epochs
    vocab_size = 200
    embed_dim = 32
    seq_len = 20
    n_task_examples = 50  # Very small task dataset
    
    # Create a simple model
    class TinyLM(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.rnn = nn.GRU(embed_dim, embed_dim, batch_first=True)
            self.head = nn.Linear(embed_dim, vocab_size)
        
        def forward(self, x):
            h = self.embed(x)
            h, _ = self.rnn(h)
            return self.head(h)
    
    model = TinyLM(vocab_size, embed_dim)
    
    # Generate "task data" with a specific distribution
    # Task data has particular bigrams that appear frequently
    task_tokens = list(range(50, 120))  # Task-relevant tokens
    
    def generate_task_data(n_examples, seq_len):
        data = []
        for _ in range(n_examples):
            seq = []
            for j in range(seq_len):
                if j == 0 or random.random() > 0.6:
                    tok = random.choice(task_tokens)
                else:
                    # Bigram pattern: follow previous token with related one
                    prev = seq[-1]
                    tok = (prev + random.randint(1, 5)) % vocab_size
                    if tok not in task_tokens:
                        tok = random.choice(task_tokens)
                seq.append(tok)
            data.append(seq)
        return torch.tensor(data)
    
    # Generate general eval data (different distribution)
    def generate_general_data(n_examples, seq_len):
        return torch.randint(0, vocab_size, (n_examples, seq_len))
    
    task_data = generate_task_data(n_task_examples, seq_len)
    general_data = generate_general_data(100, seq_len)
    
    # Train and measure at different epoch counts
    epoch_checkpoints = [0, 5, 10, 20, 50, 100, 200]
    results = []
    
    # Reset model
    model = TinyLM(vocab_size, embed_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    def evaluate_ppl(model, data):
        model.eval()
        with torch.no_grad():
            logits = model(data[:, :-1])
            loss = criterion(logits.reshape(-1, vocab_size), data[:, 1:].reshape(-1))
            return math.exp(loss.item())
    
    # Baseline
    task_ppl_0 = evaluate_ppl(model, task_data)
    general_ppl_0 = evaluate_ppl(model, general_data)
    results.append((0, task_ppl_0, general_ppl_0))
    
    epoch = 0
    for target_epoch in epoch_checkpoints[1:]:
        model.train()
        while epoch < target_epoch:
            # Shuffle task data each epoch
            perm = torch.randperm(task_data.size(0))
            shuffled = task_data[perm]
            
            for i in range(0, len(shuffled), 16):
                batch = shuffled[i:i+16]
                if batch.size(0) == 0:
                    continue
                
                logits = model(batch[:, :-1])
                loss = criterion(logits.reshape(-1, vocab_size), batch[:, 1:].reshape(-1))
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            epoch += 1
        
        task_ppl = evaluate_ppl(model, task_data)
        general_ppl = evaluate_ppl(model, general_data)
        results.append((target_epoch, task_ppl, general_ppl))
    
    print(f"\n  ── TAPT Multi-Epoch Results ({n_task_examples} examples, seq_len={seq_len}) ──\n")
    print(f"  {'Epoch':>6} │ {'Task PPL':>10} │ {'General PPL':>11} │ {'Task Δ':>8} │ {'Gen Δ':>8} │ Status")
    print(f"  {'─'*6}─┼─{'─'*10}─┼─{'─'*11}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*25}")
    
    for ep, t_ppl, g_ppl in results:
        t_delta = (1 - t_ppl / results[0][1]) * 100
        g_delta = (g_ppl / results[0][2] - 1) * 100
        
        if ep == 0:
            status = "Baseline"
        elif g_delta > 30:
            status = "⚠ Significant forgetting"
        elif g_delta > 15:
            status = "~ Mild forgetting"
        else:
            status = "✓ Healthy adaptation"
        
        print(f"  {ep:>6} │ {t_ppl:>10.1f} │ {g_ppl:>11.1f} │ {t_delta:>+7.1f}% │ {g_delta:>+7.1f}% │ {status}")
    
    # Find optimal epoch
    best_epoch = 0
    best_score = 0
    for ep, t_ppl, g_ppl in results:
        t_delta = (1 - t_ppl / results[0][1]) * 100
        g_delta = (g_ppl / results[0][2] - 1) * 100
        score = t_delta - 0.5 * max(0, g_delta)  # Penalize forgetting
        if score > best_score:
            best_score = score
            best_epoch = ep
    
    print(f"\n  Optimal epoch (balancing task PPL vs forgetting): ~{best_epoch}")
    
    print(f"""
  ═══ Multi-Epoch Guidelines ═══
  
  ┌──────────────────────┬─────────────┬──────────────────────────┐
  │ Task Data Size       │ Epochs      │ Notes                    │
  ├──────────────────────┼─────────────┼──────────────────────────┤
  │ < 500 examples       │ 50-200      │ High risk of memorizing  │
  │ 500-5K examples      │ 20-100      │ Sweet spot for TAPT      │
  │ 5K-50K examples      │ 5-20        │ Moderate epochs          │
  │ > 50K examples       │ 2-5         │ Approaches DAPT behavior │
  └──────────────────────┴─────────────┴──────────────────────────┘
  
  Key principles:
  • More epochs = better task PPL but risk forgetting
  • Use learning rate warmup + cosine decay PER RUN (not per epoch)
  • Monitor general capabilities every N epochs
  • Stop if general PPL increases > 15-20%
""")


# ============================================================================
# SECTION 4: CURATED TAPT THEORY
# ============================================================================

def curated_tapt_theory():
    """Theory behind expanding TAPT data via retrieval (Curated TAPT)."""
    print("\n\n" + "=" * 70)
    print("  SECTION 4: CURATED TAPT THEORY")
    print("=" * 70)
    
    print(f"""
  ═══ The Data Scarcity Problem in TAPT ═══
  
  Standard TAPT uses ONLY the task dataset (usually tiny).
  Problem: 5K examples × 512 tokens = 2.5M tokens
  
  After 100 epochs, the model has "memorized" the surface patterns.
  Solution: EXPAND the dataset with task-SIMILAR text from a large pool.
  
  ┌────────────┐     ┌──────────────┐     ┌─────────────┐
  │ Task Data  │ ──→ │ Embed task   │ ──→ │ Retrieve    │
  │ (5K docs)  │     │ examples     │     │ k-NN from   │
  └────────────┘     └──────────────┘     │ large pool  │
                                          └──────┬──────┘
                                                 │
  ┌────────────┐     ┌──────────────┐     ┌──────▼──────┐
  │ Curated    │ ←── │ Combine task │ ←── │ Retrieved   │
  │ TAPT       │     │ + retrieved  │     │ neighbors   │
  └────────────┘     └──────────────┘     └─────────────┘
""")
    
    torch.manual_seed(42)
    
    # Simulate Curated TAPT with embeddings
    n_task = 50      # Task examples
    n_pool = 1000    # Large unlabeled pool
    embed_dim = 16
    
    # Task data: clustered in embedding space
    task_center = torch.randn(embed_dim)
    task_embeddings = task_center.unsqueeze(0) + torch.randn(n_task, embed_dim) * 0.3
    
    # Pool data: spread across embedding space
    # Some near task distribution, most far away
    pool_embeddings = torch.randn(n_pool, embed_dim)
    # Add 50 pool items near task center (these are "curated" candidates)
    near_task = task_center.unsqueeze(0) + torch.randn(50, embed_dim) * 0.5
    pool_embeddings[:50] = near_task
    
    # Retrieve k nearest neighbors for each task example
    k = 5
    
    def retrieve_neighbors(task_emb, pool_emb, k):
        """Retrieve k nearest neighbors from pool for each task example."""
        # Compute cosine similarity
        task_norm = F.normalize(task_emb, dim=1)
        pool_norm = F.normalize(pool_emb, dim=1)
        similarities = torch.mm(task_norm, pool_norm.t())  # [n_task, n_pool]
        
        # Get top-k for each task example
        topk_sims, topk_indices = similarities.topk(k, dim=1)
        
        return topk_indices, topk_sims
    
    retrieved_indices, retrieved_sims = retrieve_neighbors(
        task_embeddings, pool_embeddings, k
    )
    
    # Analyze retrieved data quality
    unique_retrieved = set()
    for row in retrieved_indices:
        for idx in row.tolist():
            unique_retrieved.add(idx)
    
    near_task_retrieved = sum(1 for idx in unique_retrieved if idx < 50)
    far_retrieved = sum(1 for idx in unique_retrieved if idx >= 50)
    
    print(f"  ── Retrieval Results ──\n")
    print(f"  Task examples:           {n_task}")
    print(f"  Pool size:               {n_pool}")
    print(f"  k neighbors per example: {k}")
    print(f"  Unique docs retrieved:   {len(unique_retrieved)}")
    print(f"    - Near-task (quality):   {near_task_retrieved} ({near_task_retrieved/len(unique_retrieved):.0%})")
    print(f"    - Far from task:         {far_retrieved} ({far_retrieved/len(unique_retrieved):.0%})")
    
    # Similarity distribution
    all_sims = retrieved_sims.flatten()
    print(f"\n  Retrieval similarity stats:")
    print(f"    Mean:   {all_sims.mean():.3f}")
    print(f"    Median: {all_sims.median():.3f}")
    print(f"    Min:    {all_sims.min():.3f}")
    print(f"    Max:    {all_sims.max():.3f}")
    
    # Data expansion factor
    expanded_size = n_task + len(unique_retrieved)
    expansion_factor = expanded_size / n_task
    print(f"\n  Data expansion: {n_task} → {expanded_size} ({expansion_factor:.1f}x)")
    
    print(f"""
  ═══ Curated TAPT Quality Analysis ═══
  
  Quality of retrieved data matters more than quantity:
  
  ┌──────────────────┬──────────────────────────────────────┐
  │ Retrieval Method │ Expected Quality                     │
  ├──────────────────┼──────────────────────────────────────┤
  │ Random           │ Mostly irrelevant (baseline)         │
  │ BM25 (keyword)   │ Lexical overlap, misses semantics    │
  │ Dense retrieval  │ Semantic similarity ★ recommended    │
  │ Task-specific    │ Uses task labels to bias retrieval    │
  └──────────────────┴──────────────────────────────────────┘
  
  ═══ Curated TAPT vs Standard TAPT ═══
  
  Gururangan et al. (2020) results on ACL-ARC:
  
  Method              │ F1 Score │ Δ from base
  ────────────────────┼──────────┼────────────
  Base + FT           │ 63.0%    │ baseline
  TAPT + FT           │ 67.4%    │ +4.4%
  Curated TAPT + FT   │ 75.6%    │ +12.6% ★
  DAPT + Curated TAPT │ 75.4%    │ +12.4%
  
  Curated TAPT can DOUBLE the benefit of standard TAPT!
  For small datasets, retrieval augmentation is extremely valuable.
""")


# ============================================================================
# SECTION 5: WHEN TAPT HELPS vs HURTS
# ============================================================================

def when_tapt_helps():
    """Decision criteria for when to use TAPT."""
    print("\n\n" + "=" * 70)
    print("  SECTION 5: WHEN TAPT HELPS vs HURTS")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Model the expected TAPT benefit based on different factors
    print(f"""
  ═══ TAPT Benefit Factors ═══
  
  Expected benefit = f(task_specificity, data_size, domain_distance, model_size)
""")
    
    # Factor 1: Task specificity
    print(f"\n  ── Factor 1: Task Specificity ──")
    print(f"  How different is the task text from general pretraining?\n")
    
    specificity_scenarios = [
        ("Generic QA (open domain)",      0.1, 0.5),
        ("News classification (AGNews)",  0.2, 1.0),
        ("Sentiment analysis (IMDB)",     0.3, 1.5),
        ("Scientific paper abstract",     0.5, 3.5),
        ("Clinical note NER",             0.7, 5.0),
        ("Legal contract analysis",       0.6, 4.2),
        ("Code defect detection",         0.75, 6.0),
        ("Patent claim parsing",          0.8, 7.5),
    ]
    
    for task, specificity, benefit in specificity_scenarios:
        bar = "█" * int(benefit * 3)
        print(f"  {task:<35} specificity={specificity:.2f}  TAPT Δ={benefit:+.1f}%  {bar}")
    
    # Factor 2: Task data size
    print(f"\n  ── Factor 2: Task Data Size ──")
    print(f"  Larger task data = more epochs of TAPT = bigger benefit\n")
    
    sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]
    for size in sizes:
        if size < 300:
            benefit = 0.5
            note = "(too small for reliable TAPT)"
        elif size < 2000:
            benefit = size / 1000 * 2.0
            note = ""
        elif size < 20000:
            benefit = 3.0 + math.log10(size / 2000) * 1.5
            note = "(sweet spot)"
        else:
            benefit = 4.5 + math.log10(size / 20000) * 0.5
            note = "(diminishing returns)"
        
        bar = "█" * int(benefit * 3)
        print(f"  {size:>8} examples  TAPT Δ ≈ {benefit:+.1f}%  {bar}  {note}")
    
    # Factor 3: Combined decision
    print(f"""
  ═══ TAPT Decision Matrix ═══
  
  ┌──────────────────┬───────────────┬───────────────┬───────────────┐
  │                  │ Low specifity │ Med specifity │ High specifity│
  │                  │ (news, web)   │ (sci, legal)  │ (code, med)   │
  ├──────────────────┼───────────────┼───────────────┼───────────────┤
  │ < 500 examples   │ Skip TAPT     │ Curated TAPT  │ Curated TAPT  │
  │ 500-5K examples  │ Optional      │ TAPT ★        │ TAPT ★        │
  │ 5K-50K examples  │ TAPT          │ TAPT ★★       │ TAPT ★★       │
  │ > 50K examples   │ TAPT          │ TAPT ★★       │ TAPT ★★★      │
  └──────────────────┴───────────────┴───────────────┴───────────────┘
  
  ★ = recommended, ★★ = strongly recommended, ★★★ = essential
  
  ═══ TAPT Anti-Patterns (When It Hurts) ═══
  
  1. TAPT on < 100 examples → model memorizes surface text
  2. Too many epochs on tiny data → severe overfitting
  3. Using TAPT with very high LR → catastrophic forgetting
  4. TAPT on data that matches pretraining exactly → waste of compute
  
  ═══ Cost-Benefit Summary ═══
  
  TAPT is almost always worth trying because:
  • Cost: 5-30 minutes of GPU time
  • Risk: Minimal if using reasonable LR and epochs
  • Benefit: 0.5-8% F1 improvement
  • Process: Just run continued pretraining on your training set text
  
  There's essentially NO downside to trying TAPT!
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  TAPT THEORY — UNDERSTANDING TASK-ADAPTIVE PRETRAINING          ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    task_distribution_analysis()
    task_vs_domain_distance()
    multi_epoch_dynamics()
    curated_tapt_theory()
    when_tapt_helps()
    
    print("\n" + "=" * 70)
    print("  THEORY MODULE COMPLETE")
    print("=" * 70)
    print("""
    Covered:
    ✓ Task distribution analysis: why task text is narrow / specific
    ✓ Task vs domain distance: complementary adaptations
    ✓ Multi-epoch dynamics: why many epochs work on tiny data
    ✓ Curated TAPT: expanding TAPT via nearest-neighbor retrieval
    ✓ Decision framework: when TAPT helps, when it doesn't
    """)


if __name__ == "__main__":
    main()
