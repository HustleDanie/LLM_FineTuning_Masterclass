"""
DAPT Theory — Domain Shift Analysis, Vocabulary Overlap, Domain Distance
=========================================================================

Deep theoretical understanding of why and when DAPT works:

1. Domain Shift Visualization — how representations change across domains
2. Vocabulary Overlap Analysis — tokenizer efficiency per domain
3. Perplexity as Domain Distance — measuring domain gap
4. Feature Drift Analysis — how DAPT changes internal representations
5. When DAPT Helps vs Hurts — theoretical framework

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
from collections import Counter


# ============================================================================
# SECTION 1: DOMAIN SHIFT VISUALIZATION
# ============================================================================

def domain_shift_visualization():
    """Demonstrate how domain shift affects model performance."""
    print("=" * 70)
    print("  SECTION 1: DOMAIN SHIFT — WHY DAPT IS NEEDED")
    print("=" * 70)
    
    print(f"""
  ═══ The Domain Shift Problem ═══
  
  A language model pretrained on general text (Wikipedia, web) learns:
  
    P_general(token | context) — probability distribution over general text
  
  But domain-specific text has a DIFFERENT distribution:
  
    P_domain(token | context) — probability distribution over domain text
  
  The KL divergence between these distributions measures the "gap":
  
    D_KL(P_domain || P_general) = Σ P_domain(x) · log[P_domain(x) / P_general(x)]
  
  DAPT minimizes this gap by continuing pretraining on domain data:
  
    P_general → P_adapted ≈ P_domain
""")
    
    torch.manual_seed(42)
    
    # Simulate domain-specific token distributions
    vocab_size = 1000
    
    # General domain: broad, relatively uniform
    general_logits = torch.randn(vocab_size)
    general_dist = F.softmax(general_logits, dim=0)
    
    # Domain-specific: peaked on domain-specific tokens
    domain_logits = torch.randn(vocab_size) * 0.5  # Start from general
    # Boost domain-specific tokens (indices 700-800)
    domain_logits[700:800] += 3.0
    # Suppress general-heavy tokens
    domain_logits[0:100] -= 2.0
    domain_dist = F.softmax(domain_logits, dim=0)
    
    # After DAPT: adapted distribution
    adapted_logits = general_logits * 0.3 + domain_logits * 0.7
    adapted_dist = F.softmax(adapted_logits, dim=0)
    
    # Compute KL divergences
    def kl_div(p, q):
        """KL(P || Q) — measures how Q differs from P."""
        eps = 1e-10
        return (p * torch.log((p + eps) / (q + eps))).sum().item()
    
    kl_general_domain = kl_div(domain_dist, general_dist)
    kl_adapted_domain = kl_div(domain_dist, adapted_dist)
    
    print(f"  Domain distances (KL divergence):")
    print(f"  ┌───────────────────────────────────────────┐")
    print(f"  │ General → Domain:     {kl_general_domain:.4f}                │")
    print(f"  │ DAPT-Adapted → Domain: {kl_adapted_domain:.4f}   ({kl_adapted_domain/kl_general_domain:.0%} of original) │")
    print(f"  │ Reduction:             {(1-kl_adapted_domain/kl_general_domain)*100:.0f}% closer to domain    │")
    print(f"  └───────────────────────────────────────────┘")
    
    # Perplexity comparison
    def perplexity_on(model_dist, text_dist, n_tokens=1000):
        """Simulated perplexity of model on text from given distribution."""
        # Sample tokens from text distribution
        tokens = torch.multinomial(text_dist, n_tokens, replacement=True)
        # Compute log probability under model
        log_probs = torch.log(model_dist[tokens] + 1e-10)
        avg_nll = -log_probs.mean().item()
        return math.exp(avg_nll)
    
    ppl_general = perplexity_on(general_dist, domain_dist)
    ppl_adapted = perplexity_on(adapted_dist, domain_dist)
    
    print(f"\n  Perplexity on domain text:")
    print(f"  General model:   {ppl_general:.1f}")
    print(f"  DAPT-adapted:    {ppl_adapted:.1f}  ({(1-ppl_adapted/ppl_general)*100:+.0f}%)")
    
    # Simulate different domain distances
    print(f"\n  ── Domain Distance Spectrum ──\n")
    print(f"  {'Domain':>14} │ {'Distance':>8} │ {'PPL Before':>10} │ {'PPL After':>9} │ {'Benefit':>7}")
    print(f"  {'─'*14}─┼─{'─'*8}─┼─{'─'*10}─┼─{'─'*9}─┼─{'─'*7}")
    
    domains = [
        ("News", 0.1),
        ("Reviews", 0.3),
        ("Legal", 0.6),
        ("Scientific", 0.8),
        ("Biomedical", 1.2),
        ("Code", 1.5),
    ]
    
    for name, shift_magnitude in domains:
        # Create domain distribution with given shift
        d_logits = general_logits.clone()
        # Shift specific token ranges
        shift_start = int(torch.randint(0, 800, (1,)).item())
        d_logits[shift_start:shift_start+100] += shift_magnitude * 3
        d_logits[:50] -= shift_magnitude
        d_dist = F.softmax(d_logits, dim=0)
        
        # DAPT adapted
        a_logits = general_logits * (1 - 0.6) + d_logits * 0.6
        a_dist = F.softmax(a_logits, dim=0)
        
        kl = kl_div(d_dist, general_dist)
        ppl_before = perplexity_on(general_dist, d_dist)
        ppl_after = perplexity_on(a_dist, d_dist)
        benefit = (1 - ppl_after / ppl_before) * 100
        
        bar = "█" * int(benefit / 3)
        print(f"  {name:>14} │ {kl:>8.3f} │ {ppl_before:>10.1f} │ {ppl_after:>9.1f} │ {benefit:>+5.0f}% {bar}")
    
    print(f"""
  ═══ Key Insight ═══
  
  DAPT benefit SCALES with domain distance:
  • Close domains (news): Small perplexity gap → small DAPT benefit
  • Far domains (biomedical, code): Large gap → large DAPT benefit
  
  This is why BioBERT, SciBERT, CodeBERT all show large improvements:
  their target domains are FAR from the general pretrain distribution.
""")


# ============================================================================
# SECTION 2: VOCABULARY OVERLAP ANALYSIS
# ============================================================================

def vocabulary_overlap_analysis():
    """Analyze how tokenizer efficiency varies across domains."""
    print("\n\n" + "=" * 70)
    print("  SECTION 2: VOCABULARY OVERLAP ANALYSIS")
    print("=" * 70)
    
    print(f"""
  ═══ Why Vocabulary Matters for DAPT ═══
  
  BPE/WordPiece tokenizers are trained on GENERAL text.
  Domain-specific terms get FRAGMENTED:
  
  General text:    "The cat sat on the mat"
  Tokens:          [The] [cat] [sat] [on] [the] [mat]  → 6 tokens
  
  Biomedical text: "Electrocardiographic abnormalities"
  Tokens:          [Ele] [ctro] [card] [iog] [raph] [ic] [abn] [ormal] [ities]
                   → 9 tokens for 2 words!
  
  Implications:
  1. Domain text uses more tokens per word (less efficient)
  2. Rare domain terms may have poor embeddings
  3. Longer sequences needed → higher compute cost
  4. DAPT helps the model learn better representations for these fragments
""")
    
    # Simulate vocabulary coverage analysis
    torch.manual_seed(42)
    
    # Simulated corpora (word frequency distributions)
    general_vocab = {
        "the": 7000, "is": 5000, "a": 4500, "and": 4000, "to": 3800,
        "of": 3500, "in": 3200, "that": 2800, "it": 2500, "for": 2300,
        "was": 2100, "on": 2000, "are": 1800, "as": 1700, "with": 1600,
        "his": 1400, "they": 1300, "at": 1200, "be": 1100, "this": 1000,
        "have": 950, "from": 900, "or": 850, "had": 800, "by": 750,
        "not": 700, "but": 650, "what": 600, "all": 550, "were": 500,
        "we": 450, "when": 400, "your": 350, "can": 300, "said": 280,
        "there": 260, "use": 240, "each": 220, "which": 200, "she": 180,
        "do": 160, "how": 140, "their": 130, "if": 120, "will": 110,
        "up": 100, "about": 90, "out": 80, "many": 70, "then": 60,
    }
    
    domain_vocabs = {
        "News": {
            "the": 6500, "said": 3000, "government": 800, "president": 700,
            "official": 650, "report": 600, "election": 500, "policy": 450,
            "minister": 400, "announced": 380, "statement": 350, "is": 4800,
            "a": 4200, "and": 3800, "to": 3600, "of": 3400, "in": 3100,
            "economic": 300, "million": 280, "according": 260, "public": 240,
        },
        "Biomedical": {
            "patient": 2000, "treatment": 1800, "clinical": 1700,
            "disease": 1600, "study": 1500, "cell": 1400, "protein": 1300,
            "gene": 1200, "receptor": 1100, "inhibitor": 1000,
            "pathogenesis": 800, "etiology": 750, "comorbidity": 700,
            "the": 5000, "of": 4500, "and": 3800, "in": 3200, "a": 2800,
            "tachycardia": 600, "hemorrhagic": 550, "thrombocytopenia": 400,
            "electrocardiographic": 200, "immunohistochemical": 150,
        },
        "Legal": {
            "court": 1800, "plaintiff": 1600, "defendant": 1500,
            "judgment": 1400, "statute": 1300, "pursuant": 1200,
            "jurisdiction": 1100, "liability": 1000, "estoppel": 800,
            "adjudication": 700, "indemnification": 600, "the": 5500,
            "of": 4200, "in": 3500, "to": 3300, "and": 3000, "a": 2600,
            "hereinafter": 500, "notwithstanding": 450, "aforementioned": 400,
            "subpoena": 350, "jurisprudence": 300,
        },
        "Code": {
            "def": 2000, "return": 1800, "import": 1600, "class": 1400,
            "self": 1300, "if": 1200, "for": 1100, "in": 1000,
            "None": 900, "True": 800, "False": 750, "async": 600,
            "await": 550, "yield": 500, "lambda": 450, "kwargs": 400,
            "isinstance": 350, "ValueError": 300, "TypeError": 250,
            "RuntimeError": 200, "the": 800, "and": 500, "a": 400,
        }
    }
    
    print(f"\n  ── Vocabulary Statistics ──\n")
    print(f"  {'Domain':>12} │ {'Unique':>6} │ {'Overlap':>7} │ {'Domain-':>7} │ {'Tokens/':>7} │ {'Fertility':>9}")
    print(f"  {'':>12} │ {'terms':>6} │ {'w/ gen.':>7} │ {'only':>7} │ {'word':>7} │ {'ratio':>9}")
    print(f"  {'─'*12}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*9}")
    
    general_words = set(general_vocab.keys())
    
    for domain_name, vocab in domain_vocabs.items():
        domain_words = set(vocab.keys())
        overlap = len(domain_words & general_words)
        domain_only = len(domain_words - general_words)
        overlap_pct = overlap / len(domain_words) * 100
        
        # Simulate fertility (tokens per word)
        # Domain-specific words need more subword tokens
        avg_word_len = sum(len(w) for w in domain_words) / len(domain_words)
        base_fertility = 1.0 + (avg_word_len - 4) * 0.15
        domain_shift_penalty = domain_only / len(domain_words) * 0.5
        fertility = base_fertility + domain_shift_penalty
        
        print(f"  {domain_name:>12} │ {len(domain_words):>6} │ {overlap_pct:>5.0f}% │ "
              f"{domain_only:>7} │ {fertility:>7.2f} │ "
              f"{'█' * int(fertility * 3):>9}")
    
    print(f"""
  ═══ Fertility Ratio ═══
  
  Fertility = average number of subword tokens per word.
  
  Higher fertility means:
  • Domain words are FRAGMENTED by the tokenizer
  • Each word costs MORE tokens
  • Sequences are LONGER → more compute needed
  • Word representations are SPLIT across fragments
  
  DAPT helps because the model learns better representations
  for domain-specific subword fragments, even though the 
  tokenizer doesn't change.
  
  For EXTREME domain shift, consider:
  1. Extending the tokenizer with domain terms (expensive)
  2. Training domain-specific tokenizer (new model needed)
  3. Using LoRA-DAPT to adapt efficiently (recommended)
""")


# ============================================================================
# SECTION 3: PERPLEXITY AS DOMAIN DISTANCE
# ============================================================================

def perplexity_domain_distance():
    """Using perplexity to measure domain gap and monitor DAPT progress."""
    print("\n\n" + "=" * 70)
    print("  SECTION 3: PERPLEXITY AS DOMAIN DISTANCE")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ Perplexity: The Practical Domain Distance Metric ═══
  
  Perplexity measures how "surprised" a model is by new text:
  
    PPL = exp(-1/N · Σ log P(x_i | x_<i))
  
  Lower perplexity = model predicts domain text well
  Higher perplexity = domain text is "surprising" to the model
  
  Uses:
  1. BEFORE DAPT: Measure baseline domain gap (PPL on domain text)
  2. DURING DAPT: Monitor adaptation progress (PPL should decrease)
  3. AFTER DAPT: Validate domain adaptation (PPL should be low)
  4. DIAGNOSTIC: Also check PPL on general text (shouldn't increase much)
""")
    
    # Simulate DAPT perplexity trajectory
    class SimpleLanguageModel(nn.Module):
        def __init__(self, vocab_size=500, d_model=64, n_heads=4):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = nn.Embedding(128, d_model)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, 
                    dim_feedforward=128, batch_first=True),
                num_layers=2)
            self.output = nn.Linear(d_model, vocab_size)
            self.vocab_size = vocab_size
        
        def forward(self, x):
            seq_len = x.shape[1]
            positions = torch.arange(seq_len).unsqueeze(0).expand_as(x)
            h = self.embedding(x) + self.pos_encoding(positions)
            
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
            h = self.transformer(h, mask=mask)
            return self.output(h)
        
        def perplexity(self, x):
            with torch.no_grad():
                logits = self.forward(x[:, :-1])
                targets = x[:, 1:]
                loss = F.cross_entropy(logits.reshape(-1, self.vocab_size),
                                       targets.reshape(-1))
                return math.exp(loss.item())
    
    vocab_size = 500
    seq_len = 32
    
    # General domain data
    general_dist = torch.ones(vocab_size)
    general_dist[:200] = 5.0  # Common words are more frequent
    general_dist = general_dist / general_dist.sum()
    
    # Domain-specific data (biomedical-like: different token frequencies)
    domain_dist = torch.ones(vocab_size) * 0.5
    domain_dist[150:350] = 4.0  # Domain-specific tokens
    domain_dist[:100] = 2.0     # Some general tokens still common
    domain_dist = domain_dist / domain_dist.sum()
    
    def generate_data(dist, n_samples=100, seq_len=32):
        return torch.multinomial(dist.expand(n_samples, -1), 
                                 seq_len, replacement=True)
    
    general_data = generate_data(general_dist, 200, seq_len)
    domain_data = generate_data(domain_dist, 200, seq_len)
    domain_val = generate_data(domain_dist, 50, seq_len)
    general_val = generate_data(general_dist, 50, seq_len)
    
    # Stage 1: Pretrain on general data
    model = SimpleLanguageModel(vocab_size=vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"  ── Stage 1: General Pretraining ──\n")
    
    for epoch in range(30):
        model.train()
        for i in range(0, len(general_data) - 8, 8):
            batch = general_data[i:i+8]
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, vocab_size),
                                    batch[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    ppl_general_on_general = model.perplexity(general_val)
    ppl_general_on_domain = model.perplexity(domain_val)
    
    print(f"  After general pretraining:")
    print(f"    PPL on general text: {ppl_general_on_general:.1f}")
    print(f"    PPL on domain text:  {ppl_general_on_domain:.1f}")
    print(f"    Domain gap:          {ppl_general_on_domain - ppl_general_on_general:+.1f}")
    
    # Stage 2: DAPT on domain data
    print(f"\n  ── Stage 2: DAPT (Domain-Adaptive Pretraining) ──\n")
    
    # Lower learning rate for DAPT
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    
    print(f"  {'Epoch':>5} │ {'PPL Domain':>10} │ {'PPL General':>11} │ {'Domain Gap':>10} │ {'Progress':>8}")
    print(f"  {'─'*5}─┼─{'─'*10}─┼─{'─'*11}─┼─{'─'*10}─┼─{'─'*8}")
    
    dapt_epochs = 20
    ppl_trajectory = {'domain': [], 'general': []}
    
    for epoch in range(dapt_epochs):
        model.train()
        for i in range(0, len(domain_data) - 8, 8):
            batch = domain_data[i:i+8]
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, vocab_size),
                                    batch[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        ppl_d = model.perplexity(domain_val)
        ppl_g = model.perplexity(general_val)
        ppl_trajectory['domain'].append(ppl_d)
        ppl_trajectory['general'].append(ppl_g)
        
        gap = ppl_d - ppl_g
        progress = (1 - ppl_d / ppl_general_on_domain) * 100
        
        if epoch % 4 == 0 or epoch == dapt_epochs - 1:
            bar_d = "█" * max(0, int(ppl_d / 5))
            print(f"  {epoch+1:>5} │ {ppl_d:>10.1f} │ {ppl_g:>11.1f} │ {gap:>+10.1f} │ {progress:>+6.0f}%")
    
    print(f"""
  ═══ Monitoring DAPT Progress ═══
  
  Watch for these patterns:
  
  ✓ GOOD: Domain PPL decreasing steadily
  ✓ GOOD: General PPL stable or slightly increasing
  ✓ GOOD: Domain gap narrowing
  
  ⚠ WARNING: General PPL rising rapidly → catastrophic forgetting
  ⚠ WARNING: Domain PPL plateau early → need more/better data
  ⚠ WARNING: Domain PPL oscillating → learning rate too high
  
  ═══ Stopping Criteria ═══
  
  Stop DAPT when:
  1. Domain PPL stabilizes (< 5% change between epochs)
  2. General PPL increases > 20% from baseline
  3. Downstream task validation metric peaks
  4. Budget exhausted (compute/time)
  
  Rule of thumb: 1-3 epochs of DAPT is usually sufficient
""")
    
    del model, optimizer


# ============================================================================
# SECTION 4: FEATURE DRIFT ANALYSIS
# ============================================================================

def feature_drift_analysis():
    """How DAPT changes internal representations."""
    print("\n\n" + "=" * 70)
    print("  SECTION 4: FEATURE DRIFT — HOW DAPT CHANGES REPRESENTATIONS")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ Representation Shift During DAPT ═══
  
  DAPT changes the model's internal representations to better
  encode domain-specific patterns. We can measure this by:
  
  1. CKA (Centered Kernel Alignment) between layers before/after DAPT
  2. Cosine similarity of representations for same inputs
  3. Layer-wise analysis: which layers change most?
  
  Key finding from the literature:
  • Lower layers (embeddings, early attention): Change LESS
    → These encode general linguistic features
  • Higher layers (later attention, output): Change MORE
    → These encode task/domain-specific features
""")
    
    # Build a model and analyze layer changes during DAPT
    class AnalyzableModel(nn.Module):
        def __init__(self, vocab_size=300, d_model=64):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.layer1 = nn.Linear(d_model, d_model)
            self.layer2 = nn.Linear(d_model, d_model)
            self.layer3 = nn.Linear(d_model, d_model)
            self.output = nn.Linear(d_model, vocab_size)
            self.vocab_size = vocab_size
        
        def forward(self, x, return_features=False):
            h0 = self.embedding(x).mean(dim=1)
            h1 = F.relu(self.layer1(h0))
            h2 = F.relu(self.layer2(h1))
            h3 = F.relu(self.layer3(h2))
            logits = self.output(h3)
            
            if return_features:
                return logits, {'embed': h0, 'layer1': h1, 
                               'layer2': h2, 'layer3': h3}
            return logits
    
    vocab_size = 300
    model = AnalyzableModel(vocab_size=vocab_size)
    
    # General pretraining
    general_data = torch.randint(0, vocab_size, (200, 16))
    general_targets = torch.randint(0, vocab_size, (200,))
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(30):
        logits = model(general_data)
        loss = F.cross_entropy(logits, general_targets)
        opt.zero_grad(); loss.backward(); opt.step()
    
    # Save pre-DAPT representations
    model.eval()
    probe_data = torch.randint(0, vocab_size, (50, 16))
    with torch.no_grad():
        _, pre_features = model(probe_data, return_features=True)
    pre_features = {k: v.clone() for k, v in pre_features.items()}
    
    # Save pre-DAPT parameters
    pre_params = {n: p.data.clone() for n, p in model.named_parameters()}
    
    # DAPT on domain data
    domain_data = torch.randint(100, vocab_size, (200, 16))  # Different token range
    domain_targets = torch.randint(0, vocab_size, (200,))
    
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    for _ in range(20):
        logits = model(domain_data)
        loss = F.cross_entropy(logits, domain_targets)
        opt.zero_grad(); loss.backward(); opt.step()
    
    # Measure representation drift
    model.eval()
    with torch.no_grad():
        _, post_features = model(probe_data, return_features=True)
    
    print(f"\n  ── Layer-wise Representation Drift ──\n")
    print(f"  {'Layer':>10} │ {'Cosine Sim':>10} │ {'L2 Distance':>11} │ {'Drift':>20}")
    print(f"  {'─'*10}─┼─{'─'*10}─┼─{'─'*11}─┼─{'─'*20}")
    
    for layer_name in ['embed', 'layer1', 'layer2', 'layer3']:
        pre = pre_features[layer_name]
        post = post_features[layer_name]
        
        # Cosine similarity (averaged across samples)
        cos_sim = F.cosine_similarity(pre, post, dim=1).mean().item()
        
        # L2 distance
        l2_dist = (pre - post).norm(dim=1).mean().item()
        
        # Visualize drift
        drift_bar = "▰" * max(1, int((1 - cos_sim) * 50))
        
        print(f"  {layer_name:>10} │ {cos_sim:>10.4f} │ {l2_dist:>11.4f} │ {drift_bar}")
    
    # Parameter-level analysis
    print(f"\n  ── Parameter-wise Change Magnitude ──\n")
    print(f"  {'Parameter':>25} │ {'|Δ| mean':>10} │ {'|Δ| max':>10} │ {'% Changed':>9}")
    print(f"  {'─'*25}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*9}")
    
    for name, param in model.named_parameters():
        delta = (param.data - pre_params[name]).abs()
        mean_change = delta.mean().item()
        max_change = delta.max().item()
        
        # What fraction of params changed significantly
        threshold = pre_params[name].abs().mean().item() * 0.01
        pct_changed = (delta > threshold).float().mean().item() * 100
        
        print(f"  {name:>25} │ {mean_change:>10.6f} │ {max_change:>10.4f} │ {pct_changed:>8.1f}%")
    
    print(f"""
  ═══ Key Observations ═══
  
  1. Later layers change MORE than earlier layers
     → DAPT primarily adapts high-level representations
     → Low-level features (syntax, morphology) are preserved
  
  2. Output layer changes most
     → Domain vocabulary distribution shifts significantly
  
  3. Early embeddings change least
     → Basic word representations are relatively domain-agnostic
  
  This explains why LoRA-DAPT works well:
  • LoRA targets attention layers (mid-to-high level)
  • These are exactly the layers that NEED to change for DAPT
  • Embedding and output layers can be frozen or have separate LR
""")
    
    del model, opt


# ============================================================================
# SECTION 5: WHEN DAPT HELPS vs HURTS
# ============================================================================

def when_dapt_helps():
    """Theoretical framework for when DAPT is beneficial."""
    print("\n\n" + "=" * 70)
    print("  SECTION 5: WHEN DAPT HELPS vs HURTS")
    print("=" * 70)
    
    print(f"""
  ═══ DAPT Benefit Framework ═══
  
  DAPT benefit depends on multiple factors:
  
  Benefit = f(domain_distance, data_quality, data_quantity, model_capacity)
  
  Let's analyze each factor:
""")
    
    torch.manual_seed(42)
    
    # Factor 1: Domain Distance
    print(f"""
  ── Factor 1: Domain Distance ──
  
  Benefit ∝ Domain Distance (up to a point)
  
  Distance │ DAPT Benefit
  ─────────┼──────────────────────────────
  Very Low │ ▪  (~0-2% improvement)
  Low      │ ▪▪▪  (~2-4%)
  Medium   │ ▪▪▪▪▪▪  (~4-7%)  ← Sweet spot
  High     │ ▪▪▪▪▪▪▪▪  (~7-10%)
  Very High│ ▪▪▪▪▪▪▪  (~5-8%, plateaus)
  Extreme  │ ▪▪▪▪  (~3-5%, model struggles)
  
  At EXTREME distance, the domain is so different that the model
  can't easily adapt with continued pretraining alone.
  → Solution: Train domain model from scratch
""")
    
    # Factor 2: Data Quantity
    print(f"""
  ── Factor 2: Data Quantity ──
  
  More data helps, but with diminishing returns:
  
  Data (tokens) │ Improvement
  ──────────────┼──────────────────────────────
      1M        │ ▪▪  (noisy, may not help)
     10M        │ ▪▪▪▪▪  (starting to help)
    100M        │ ▪▪▪▪▪▪▪▪  (good)
      1B        │ ▪▪▪▪▪▪▪▪▪▪  (great)
     10B        │ ▪▪▪▪▪▪▪▪▪▪▪  (diminishing returns)
    100B        │ ▪▪▪▪▪▪▪▪▪▪▪  (plateau)
  
  Rule of thumb: 100M-1B tokens is the "efficient" range for most domains
""")
    
    # Factor 3: Data Quality
    print(f"""
  ── Factor 3: Data Quality ──
  
  Quality │ Effect on DAPT
  ────────┼──────────────────────────────────────────────
  High    │ Clean, relevant, diverse domain text
          │ → Best DAPT outcomes
  Medium  │ Some noise, broad domain coverage
          │ → Good outcomes, moderate filtering helps
  Low     │ Noisy, irrelevant, duplicated
          │ → DAPT may HURT (model learns noise)
          │ → Must clean/filter before DAPT
  
  Quality filtering steps:
  1. Remove duplicates (MinHash dedup)
  2. Remove boilerplate (headers, footers, navigation)
  3. Language filter (keep target language only)
  4. Domain relevance filter (keyword/classifier-based)
  5. Quality filter (perplexity, length, formatting)
""")
    
    # Simulate the interaction of factors
    print(f"\n  ── Simulated DAPT Benefit Map ──\n")
    
    distances = ["Close", "Medium", "Far"]
    quantities = ["Small (10M)", "Medium (100M)", "Large (1B)"]
    
    # Simulated benefit values
    benefits = [
        [1.5, 2.5, 3.0],    # Close domain
        [3.0, 5.5, 7.0],    # Medium domain
        [4.0, 7.0, 9.0],    # Far domain
    ]
    
    print(f"  {'':>15} │ ", end="")
    for q in quantities:
        print(f"{q:>15} ", end="")
    print()
    print(f"  {'─'*15}─┼─{'─'*49}")
    
    for i, dist in enumerate(distances):
        print(f"  {dist:>15} │ ", end="")
        for j, q in enumerate(quantities):
            val = benefits[i][j]
            bar = "█" * int(val)
            print(f"    +{val:.0f}% {bar:>6} ", end="")
        print()
    
    # When DAPT hurts
    print(f"""
  ═══ When DAPT Can HURT ═══
  
  1. Very small domain corpus (<1M tokens)
     → Model overfits to domain, forgets general knowledge
     Solution: Use fewer epochs, higher regularization
  
  2. Low-quality domain data
     → Model learns noise and errors
     Solution: Clean data before DAPT
  
  3. Wrong domain data
     → Data doesn't match actual target task
     Solution: Verify domain relevance with manual inspection
  
  4. Too much DAPT (too many epochs)
     → Catastrophic forgetting of general capabilities
     Solution: Monitor general PPL, use early stopping
  
  5. Learning rate too high
     → Destroys pretrained features
     Solution: Use 10-50x lower LR than original pretraining
  
  ═══ DAPT Decision Checklist ═══
  
  □ Is target domain different from pretraining data?
    → If NO: Skip DAPT, just fine-tune
    → If YES: continue...
  
  □ Do you have >10M tokens of domain text?
    → If NO: Consider TAPT instead (Concept 18)
    → If YES: continue...
  
  □ Is the data clean and relevant?
    → If NO: Clean it first
    → If YES: proceed with DAPT
  
  □ Do you have compute budget for continued pretraining?
    → If NO: Use LoRA-DAPT (much cheaper)
    → If YES: Full DAPT or LoRA-DAPT
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  DAPT THEORY — DOMAIN SHIFT, VOCABULARY, PERPLEXITY, FEATURES   ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    domain_shift_visualization()
    vocabulary_overlap_analysis()
    perplexity_domain_distance()
    feature_drift_analysis()
    when_dapt_helps()
    
    print("\n" + "=" * 70)
    print("  THEORY MODULE COMPLETE")
    print("=" * 70)
    print("""
    Covered:
    ✓ Domain shift and KL divergence between distributions
    ✓ Vocabulary overlap and tokenizer fertility analysis
    ✓ Perplexity as domain distance metric (monitoring DAPT)
    ✓ Feature drift: which layers change during DAPT
    ✓ Framework for when DAPT helps vs hurts
    """)


if __name__ == "__main__":
    main()
