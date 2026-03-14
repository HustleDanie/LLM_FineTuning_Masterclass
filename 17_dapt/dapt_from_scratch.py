"""
DAPT From Scratch — Continued Pretraining, Curriculum DAPT, LoRA-DAPT
======================================================================

Build DAPT from first principles:

1. Causal LM Continued Pretraining — standard DAPT implementation
2. Masked LM DAPT — BERT-style domain adaptation
3. Curriculum DAPT — gradual domain shift during adaptation
4. LoRA-DAPT — parameter-efficient domain adaptation
5. Data Mixing for DAPT — mixing general and domain data

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Dict, List, Tuple, Optional


# ============================================================================
# SHARED COMPONENTS
# ============================================================================

class SimpleTransformerLM(nn.Module):
    """Lightweight causal LM for DAPT demonstrations."""
    
    def __init__(self, vocab_size: int = 500, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, max_len: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, batch_first=True,
            dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.output = nn.Linear(d_model, vocab_size)
        self.output.weight = self.token_emb.weight  # Weight tying
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(positions)
        
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        h = self.transformer(h, mask=mask)
        return self.output(h)
    
    def compute_loss(self, x):
        logits = self.forward(x[:, :-1])
        targets = x[:, 1:]
        return F.cross_entropy(logits.reshape(-1, self.vocab_size),
                               targets.reshape(-1))
    
    @torch.no_grad()
    def perplexity(self, data):
        self.eval()
        total_loss = 0
        n_batches = 0
        for i in range(0, len(data), 16):
            batch = data[i:i+16]
            loss = self.compute_loss(batch)
            total_loss += loss.item()
            n_batches += 1
        self.train()
        return math.exp(total_loss / n_batches)


class DomainDataGenerator:
    """Generate synthetic data from different domains."""
    
    def __init__(self, vocab_size: int = 500, seed: int = 42):
        torch.manual_seed(seed)
        self.vocab_size = vocab_size
        
        # General domain: token frequencies follow Zipf's law
        ranks = torch.arange(1, vocab_size + 1, dtype=torch.float)
        self.general_dist = 1.0 / ranks
        self.general_dist = self.general_dist / self.general_dist.sum()
        
        # Domain-specific: different token distribution
        # Tokens 200-400 are "domain-specific" (boosted)
        self.domain_dist = self.general_dist.clone()
        self.domain_dist[200:400] *= 5.0
        self.domain_dist[:50] *= 0.3  # Domain uses fewer common words
        self.domain_dist = self.domain_dist / self.domain_dist.sum()
    
    def generate(self, n_samples: int, seq_len: int, 
                 domain: str = "general") -> torch.Tensor:
        dist = self.general_dist if domain == "general" else self.domain_dist
        
        # Add bigram structure for more realistic data
        data = torch.zeros(n_samples, seq_len, dtype=torch.long)
        data[:, 0] = torch.multinomial(dist, n_samples, replacement=True)
        
        for t in range(1, seq_len):
            # Slight dependency on previous token
            prev = data[:, t-1]
            adjusted = dist.unsqueeze(0).expand(n_samples, -1).clone()
            # Boost tokens near previous token
            for i in range(n_samples):
                p = prev[i].item()
                lo = max(0, p - 10)
                hi = min(self.vocab_size, p + 10)
                adjusted[i, lo:hi] *= 1.5
            adjusted = adjusted / adjusted.sum(dim=1, keepdim=True)
            data[:, t] = torch.multinomial(adjusted, 1).squeeze()
        
        return data


# ============================================================================
# SECTION 1: CAUSAL LM CONTINUED PRETRAINING (Standard DAPT)
# ============================================================================

def causal_lm_dapt():
    """Standard DAPT: continue pretraining a causal LM on domain text."""
    print("=" * 70)
    print("  SECTION 1: CAUSAL LM CONTINUED PRETRAINING (STANDARD DAPT)")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ Standard DAPT Algorithm ═══
  
  1. Start with pretrained causal LM (GPT-2, LLaMA, etc.)
  2. Continue training with next-token prediction on domain text
  3. Use LOWER learning rate than original pretraining
  4. Train for 1-3 epochs (not too long!)
  
  Loss: L = -1/T · Σ_{t=1}^{T} log P(x_t | x_<t)
  
  Same objective as pretraining, but on domain data!
""")
    
    data_gen = DomainDataGenerator(vocab_size=500)
    seq_len = 32
    
    # Generate data
    general_train = data_gen.generate(500, seq_len, "general")
    domain_train = data_gen.generate(500, seq_len, "domain")
    general_val = data_gen.generate(100, seq_len, "general")
    domain_val = data_gen.generate(100, seq_len, "domain")
    
    # Stage 1: General pretraining
    print(f"  ── Stage 1: General Pretraining ──\n")
    model = SimpleTransformerLM(vocab_size=500, d_model=64, n_layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    for epoch in range(15):
        model.train()
        total_loss = 0
        for i in range(0, len(general_train), 16):
            batch = general_train[i:i+16]
            loss = model.compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
    
    ppl_gen = model.perplexity(general_val)
    ppl_dom = model.perplexity(domain_val)
    print(f"  After general pretraining:")
    print(f"    PPL on general: {ppl_gen:.1f}")
    print(f"    PPL on domain:  {ppl_dom:.1f}  (gap: {ppl_dom - ppl_gen:+.1f})")
    
    # Stage 2: DAPT
    print(f"\n  ── Stage 2: DAPT (Continued Pretraining) ──\n")
    
    # Key: LOWER learning rate
    dapt_lr = 2e-4  # 5x lower than pretraining LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=dapt_lr, weight_decay=0.01)
    
    # Warmup scheduler
    total_steps = 3 * (len(domain_train) // 16)
    warmup_steps = int(0.1 * total_steps)
    
    def get_lr(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
    
    print(f"  Config: lr={dapt_lr}, epochs=3, warmup=10%, cosine schedule")
    print(f"  {'Epoch':>5} │ {'Train Loss':>10} │ {'PPL Domain':>10} │ {'PPL General':>11} │ {'Δ Domain':>8}")
    print(f"  {'─'*5}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*11}─┼─{'─'*8}")
    
    initial_domain_ppl = ppl_dom
    step = 0
    
    for epoch in range(3):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        # Shuffle domain data
        perm = torch.randperm(len(domain_train))
        shuffled = domain_train[perm]
        
        for i in range(0, len(shuffled), 16):
            batch = shuffled[i:i+16]
            loss = model.compute_loss(batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            step += 1
        
        ppl_d = model.perplexity(domain_val)
        ppl_g = model.perplexity(general_val)
        delta = (1 - ppl_d / initial_domain_ppl) * 100
        
        print(f"  {epoch+1:>5} │ {epoch_loss/n_batches:>10.4f} │ {ppl_d:>10.1f} │ "
              f"{ppl_g:>11.1f} │ {delta:>+6.1f}%")
    
    final_ppl_d = model.perplexity(domain_val)
    final_ppl_g = model.perplexity(general_val)
    
    print(f"\n  Summary:")
    print(f"    Domain PPL:  {ppl_dom:.1f} → {final_ppl_d:.1f} ({(1-final_ppl_d/ppl_dom)*100:+.1f}%)")
    print(f"    General PPL: {ppl_gen:.1f} → {final_ppl_g:.1f} ({(1-final_ppl_g/ppl_gen)*100:+.1f}%)")
    
    del model, optimizer
    return final_ppl_d, final_ppl_g


# ============================================================================
# SECTION 2: MASKED LM DAPT (BERT-Style)
# ============================================================================

def masked_lm_dapt():
    """DAPT with Masked Language Modeling (BERT-style models)."""
    print("\n\n" + "=" * 70)
    print("  SECTION 2: MASKED LM DAPT (BERT-Style)")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ MLM-based DAPT ═══
  
  For encoder models (BERT, RoBERTa), DAPT uses Masked LM:
  
  1. Randomly mask 15% of tokens
  2. Predict masked tokens from context
  3. This forces the model to learn domain-specific patterns
  
  Key differences from Causal LM DAPT:
  • Bidirectional context (sees both left and right)
  • Only predicts masked tokens (not all tokens)
  • Same MLM objective as original BERT pretraining
""")
    
    class SimpleMaskedLM(nn.Module):
        """Simplified BERT-like model for MLM DAPT demo."""
        
        def __init__(self, vocab_size=500, d_model=64, n_layers=2, n_heads=4):
            super().__init__()
            self.vocab_size = vocab_size
            self.mask_token_id = vocab_size - 1  # Last token is [MASK]
            
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_embedding = nn.Embedding(64, d_model)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
            self.output = nn.Linear(d_model, vocab_size)
        
        def forward(self, x):
            positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            h = self.embedding(x) + self.pos_embedding(positions)
            h = self.encoder(h)
            return self.output(h)
        
        def mask_and_predict(self, x, mask_prob=0.15):
            """Apply random masking and return predictions + targets."""
            mask = torch.rand_like(x.float()) < mask_prob
            # Don't mask padding (if any) — here we mask everything
            
            masked_x = x.clone()
            # 80% replace with [MASK], 10% random, 10% keep
            mask_indices = mask.nonzero(as_tuple=True)
            
            replace_mask = torch.rand(mask.sum()) < 0.8
            random_mask = (torch.rand(mask.sum()) >= 0.8) & (torch.rand(mask.sum()) < 0.9)
            
            masked_x[mask_indices[0][replace_mask], 
                      mask_indices[1][replace_mask]] = self.mask_token_id
            masked_x[mask_indices[0][random_mask],
                      mask_indices[1][random_mask]] = torch.randint(
                          0, self.vocab_size - 1, (random_mask.sum(),))
            
            logits = self.forward(masked_x)
            
            # Only compute loss on masked positions
            masked_logits = logits[mask]
            masked_targets = x[mask]
            
            if len(masked_targets) == 0:
                return torch.tensor(0.0, requires_grad=True)
            
            return F.cross_entropy(masked_logits, masked_targets)
    
    data_gen = DomainDataGenerator(vocab_size=499)  # Reserve 499 for [MASK]
    
    general_train = data_gen.generate(400, 32, "general")
    domain_train = data_gen.generate(400, 32, "domain")
    domain_val = data_gen.generate(80, 32, "domain")
    
    # Train on general data
    model = SimpleMaskedLM(vocab_size=500, d_model=64, n_layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print(f"  ── General MLM Pretraining ──")
    for epoch in range(15):
        model.train()
        for i in range(0, len(general_train), 16):
            batch = general_train[i:i+16]
            loss = model.mask_and_predict(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate MLM accuracy on domain before DAPT
    model.eval()
    pre_losses = []
    for i in range(0, len(domain_val), 16):
        batch = domain_val[i:i+16]
        with torch.no_grad():
            loss = model.mask_and_predict(batch)
            pre_losses.append(loss.item())
    pre_mlm_loss = sum(pre_losses) / len(pre_losses)
    
    print(f"  Before DAPT — MLM loss on domain: {pre_mlm_loss:.4f}")
    
    # MLM DAPT
    print(f"\n  ── MLM DAPT on Domain Data ──\n")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    
    for epoch in range(5):
        model.train()
        epoch_loss = 0
        n = 0
        for i in range(0, len(domain_train), 16):
            batch = domain_train[i:i+16]
            loss = model.mask_and_predict(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        
        model.eval()
        val_losses = []
        for i in range(0, len(domain_val), 16):
            batch = domain_val[i:i+16]
            with torch.no_grad():
                loss = model.mask_and_predict(batch)
                val_losses.append(loss.item())
        val_loss = sum(val_losses) / len(val_losses)
        
        print(f"  Epoch {epoch+1}: train_loss={epoch_loss/n:.4f}, val_loss={val_loss:.4f}")
    
    improvement = (1 - val_loss / pre_mlm_loss) * 100
    print(f"\n  MLM loss improvement: {pre_mlm_loss:.4f} → {val_loss:.4f} ({improvement:+.1f}%)")
    
    del model, optimizer


# ============================================================================
# SECTION 3: CURRICULUM DAPT
# ============================================================================

def curriculum_dapt():
    """Gradual domain shift during DAPT."""
    print("\n\n" + "=" * 70)
    print("  SECTION 3: CURRICULUM DAPT — GRADUAL DOMAIN SHIFT")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ Curriculum DAPT ═══
  
  Instead of directly training on domain data, GRADUALLY shift
  the training distribution from general to domain-specific:
  
  Phase 1 (start):  80% general + 20% domain
  Phase 2 (middle): 50% general + 50% domain  
  Phase 3 (end):    20% general + 80% domain
  
  This curriculum prevents the model from experiencing a sudden
  distribution shift, leading to smoother adaptation.
  
  ┌────────────────────────────────────────────────┐
  │ General data ██████████████████▓▓▓▓▓▓░░░░░░░░ │
  │ Domain data  ░░░░░░░░░░░░░░▓▓▓▓▓▓▓████████████ │
  │              ─────────────────────────────────── │
  │              Start      Middle         End      │
  └────────────────────────────────────────────────┘
""")
    
    data_gen = DomainDataGenerator(vocab_size=500)
    seq_len = 32
    
    general_train = data_gen.generate(500, seq_len, "general")
    domain_train = data_gen.generate(500, seq_len, "domain")
    domain_val = data_gen.generate(100, seq_len, "domain")
    general_val = data_gen.generate(100, seq_len, "general")
    
    # Train: standard DAPT vs curriculum DAPT
    results = {}
    
    for method in ["direct", "curriculum"]:
        torch.manual_seed(42)
        model = SimpleTransformerLM(vocab_size=500, d_model=64, n_layers=2)
        
        # General pretraining
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(15):
            model.train()
            for i in range(0, len(general_train), 16):
                loss = model.compute_loss(general_train[i:i+16])
                opt.zero_grad(); loss.backward(); opt.step()
        
        # DAPT
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
        total_epochs = 6
        ppl_history = []
        
        for epoch in range(total_epochs):
            model.train()
            
            if method == "curriculum":
                # Progress from 0 to 1
                progress = epoch / (total_epochs - 1)
                domain_ratio = 0.2 + 0.6 * progress  # 0.2 → 0.8
            else:
                domain_ratio = 1.0  # Pure domain data
            
            # Mix data
            n_domain = int(len(domain_train) * domain_ratio)
            n_general = int(len(general_train) * (1 - domain_ratio))
            
            if n_general > 0:
                gen_idx = torch.randperm(len(general_train))[:n_general]
                dom_idx = torch.randperm(len(domain_train))[:n_domain]
                mixed = torch.cat([general_train[gen_idx], domain_train[dom_idx]])
            else:
                mixed = domain_train[torch.randperm(len(domain_train))]
            
            # Shuffle
            mixed = mixed[torch.randperm(len(mixed))]
            
            for i in range(0, len(mixed), 16):
                loss = model.compute_loss(mixed[i:i+16])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            
            ppl_d = model.perplexity(domain_val)
            ppl_g = model.perplexity(general_val)
            ppl_history.append((ppl_d, ppl_g))
        
        results[method] = ppl_history
        del model, opt
    
    # Compare
    print(f"\n  ── Direct DAPT vs Curriculum DAPT ──\n")
    print(f"  {'Epoch':>5} │ {'Direct: Domain':>14} {'General':>8} │ "
          f"{'Curriculum: Domain':>18} {'General':>8}")
    print(f"  {'─'*5}─┼─{'─'*23}─┼─{'─'*27}")
    
    for epoch in range(6):
        d_ppl_d, d_ppl_g = results['direct'][epoch]
        c_ppl_d, c_ppl_g = results['curriculum'][epoch]
        print(f"  {epoch+1:>5} │ {d_ppl_d:>14.1f} {d_ppl_g:>8.1f} │ "
              f"{c_ppl_d:>18.1f} {c_ppl_g:>8.1f}")
    
    # Final comparison
    d_final_d = results['direct'][-1][0]
    d_final_g = results['direct'][-1][1]
    c_final_d = results['curriculum'][-1][0]
    c_final_g = results['curriculum'][-1][1]
    
    print(f"\n  Final Results:")
    print(f"  {'Method':>12} │ {'Domain PPL':>10} │ {'General PPL':>11} │ {'Balance':>7}")
    print(f"  {'─'*12}─┼─{'─'*10}─┼─{'─'*11}─┼─{'─'*7}")
    print(f"  {'Direct':>12} │ {d_final_d:>10.1f} │ {d_final_g:>11.1f} │ {d_final_g/d_final_d:>7.2f}")
    print(f"  {'Curriculum':>12} │ {c_final_d:>10.1f} │ {c_final_g:>11.1f} │ {c_final_g/c_final_d:>7.2f}")
    
    print(f"""
  ═══ Curriculum DAPT Benefits ═══
  
  • Smoother training (less loss spikes)
  • Better preservation of general capabilities
  • More stable final model
  
  When to use curriculum DAPT:
  • Domain is VERY different from general
  • Preserving general knowledge is important
  • You have access to both general and domain data
""")


# ============================================================================
# SECTION 4: LoRA-DAPT (Parameter-Efficient)
# ============================================================================

def lora_dapt():
    """Parameter-efficient DAPT using LoRA."""
    print("\n\n" + "=" * 70)
    print("  SECTION 4: LoRA-DAPT — PARAMETER-EFFICIENT DOMAIN ADAPTATION")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ LoRA-DAPT: Best of Both Worlds ═══
  
  Instead of updating ALL parameters during DAPT, use LoRA:
  • Freeze base model weights
  • Add low-rank adapters to key layers
  • Only train adapter parameters (0.1-1% of model)
  
  Advantages:
  1. MUCH less compute (10-100x cheaper)
  2. NO catastrophic forgetting (base weights unchanged)
  3. Can store multiple domain adapters
  4. Easy to combine with task fine-tuning
  
  W_adapted = W_frozen + B·A  (LoRA decomposition)
""")
    
    class LoRALayer(nn.Module):
        """LoRA adapter for a linear layer."""
        def __init__(self, in_features, out_features, rank=4, alpha=1.0):
            super().__init__()
            self.lora_A = nn.Linear(in_features, rank, bias=False)
            self.lora_B = nn.Linear(rank, out_features, bias=False)
            self.scaling = alpha / rank
            nn.init.kaiming_uniform_(self.lora_A.weight)
            nn.init.zeros_(self.lora_B.weight)
        
        def forward(self, x):
            return self.lora_B(self.lora_A(x)) * self.scaling
    
    class LoRATransformerLM(nn.Module):
        """Transformer LM with LoRA adapters."""
        
        def __init__(self, base_model: SimpleTransformerLM, rank=4):
            super().__init__()
            self.base = base_model
            self.vocab_size = base_model.vocab_size
            
            # Freeze base model
            for p in self.base.parameters():
                p.requires_grad = False
            
            # Add LoRA to transformer layers
            self.lora_adapters = nn.ModuleDict()
            for name, module in self.base.transformer.named_modules():
                if isinstance(module, nn.Linear) and module.in_features == 64:
                    safe_name = name.replace('.', '_')
                    self.lora_adapters[safe_name] = LoRALayer(
                        module.in_features, module.out_features, rank=rank)
            
            self._adapter_hooks = []
            self._register_hooks()
        
        def _register_hooks(self):
            """Register forward hooks to add LoRA outputs."""
            for name, module in self.base.transformer.named_modules():
                if isinstance(module, nn.Linear) and module.in_features == 64:
                    safe_name = name.replace('.', '_')
                    if safe_name in self.lora_adapters:
                        adapter = self.lora_adapters[safe_name]
                        hook = module.register_forward_hook(
                            lambda mod, inp, out, a=adapter: out + a(inp[0]))
                        self._adapter_hooks.append(hook)
        
        def forward(self, x):
            return self.base(x)
        
        def compute_loss(self, x):
            logits = self.forward(x[:, :-1])
            targets = x[:, 1:]
            return F.cross_entropy(logits.reshape(-1, self.vocab_size),
                                   targets.reshape(-1))
        
        def trainable_params(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        def total_params(self):
            return sum(p.numel() for p in self.parameters())
        
        @torch.no_grad()
        def perplexity(self, data):
            self.eval()
            total_loss = 0
            n = 0
            for i in range(0, len(data), 16):
                batch = data[i:i+16]
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                n += 1
            self.train()
            return math.exp(total_loss / n)
    
    data_gen = DomainDataGenerator(vocab_size=500)
    seq_len = 32
    
    general_train = data_gen.generate(500, seq_len, "general")
    domain_train = data_gen.generate(500, seq_len, "domain")
    domain_val = data_gen.generate(100, seq_len, "domain")
    general_val = data_gen.generate(100, seq_len, "general")
    
    # Step 1: Pretrain base model
    base_model = SimpleTransformerLM(vocab_size=500, d_model=64, n_layers=2)
    opt = torch.optim.AdamW(base_model.parameters(), lr=1e-3)
    for _ in range(15):
        base_model.train()
        for i in range(0, len(general_train), 16):
            loss = base_model.compute_loss(general_train[i:i+16])
            opt.zero_grad(); loss.backward(); opt.step()
    del opt
    
    base_ppl_dom = base_model.perplexity(domain_val)
    base_ppl_gen = base_model.perplexity(general_val)
    
    # Step 2: Compare Full DAPT vs LoRA-DAPT
    print(f"  ── Baseline (after general pretraining) ──")
    print(f"  Domain PPL: {base_ppl_dom:.1f}, General PPL: {base_ppl_gen:.1f}\n")
    
    # Full DAPT
    print(f"  ── Full DAPT (all parameters) ──")
    full_model = copy.deepcopy(base_model)
    opt = torch.optim.AdamW(full_model.parameters(), lr=2e-4)
    
    for epoch in range(3):
        full_model.train()
        perm = torch.randperm(len(domain_train))
        for i in range(0, len(domain_train), 16):
            batch = domain_train[perm[i:i+16]]
            loss = full_model.compute_loss(batch)
            opt.zero_grad(); loss.backward(); opt.step()
    
    full_ppl_dom = full_model.perplexity(domain_val)
    full_ppl_gen = full_model.perplexity(general_val)
    full_params = sum(p.numel() for p in full_model.parameters())
    
    print(f"  Trainable params: {full_params:,} (100%)")
    print(f"  Domain PPL: {full_ppl_dom:.1f}, General PPL: {full_ppl_gen:.1f}")
    del full_model, opt
    
    # LoRA-DAPT (different ranks)
    for rank in [2, 4, 8]:
        print(f"\n  ── LoRA-DAPT (rank={rank}) ──")
        lora_model = LoRATransformerLM(copy.deepcopy(base_model), rank=rank)
        
        trainable = lora_model.trainable_params()
        total = lora_model.total_params()
        print(f"  Trainable params: {trainable:,} ({trainable/total*100:.1f}%)")
        
        opt = torch.optim.AdamW(
            [p for p in lora_model.parameters() if p.requires_grad], lr=5e-4)
        
        for epoch in range(3):
            lora_model.train()
            perm = torch.randperm(len(domain_train))
            for i in range(0, len(domain_train), 16):
                batch = domain_train[perm[i:i+16]]
                loss = lora_model.compute_loss(batch)
                opt.zero_grad(); loss.backward(); opt.step()
        
        lora_ppl_dom = lora_model.perplexity(domain_val)
        lora_ppl_gen = lora_model.perplexity(general_val)
        
        print(f"  Domain PPL: {lora_ppl_dom:.1f}, General PPL: {lora_ppl_gen:.1f}")
        
        # Remove hooks before deleting
        for h in lora_model._adapter_hooks:
            h.remove()
        del lora_model, opt
    
    print(f"""
  ═══ LoRA-DAPT Summary ═══
  
  LoRA-DAPT achieves most of Full DAPT's domain adaptation
  while training <5% of parameters and better preserving
  general capabilities.
  
  Recommended LoRA-DAPT configuration:
  • rank=16-32 for domain adaptation
  • target: attention Q, K, V projections + FFN
  • lr: 5e-4 to 1e-3 (higher than full DAPT since fewer params)
  • epochs: 1-3
  • alpha: 2× rank (standard)
""")
    
    del base_model


# ============================================================================
# SECTION 5: DATA MIXING FOR DAPT
# ============================================================================

def data_mixing_dapt():
    """Mixing general and domain data during DAPT."""
    print("\n\n" + "=" * 70)
    print("  SECTION 5: DATA MIXING STRATEGIES FOR DAPT")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ Why Mix Data During DAPT? ═══
  
  Pure domain data DAPT risks forgetting general capabilities.
  Mixing in general data acts as a regularizer.
  
  Strategies:
  1. Fixed ratio (e.g., 90% domain + 10% general)
  2. Annealed ratio (start 50/50, end 95/5 domain)
  3. Domain-proportional (weight by domain size)
  4. Loss-based (upweight domain with higher loss)
""")
    
    data_gen = DomainDataGenerator(vocab_size=500)
    seq_len = 32
    
    general_train = data_gen.generate(500, seq_len, "general")
    domain_train = data_gen.generate(500, seq_len, "domain")
    domain_val = data_gen.generate(100, seq_len, "domain")
    general_val = data_gen.generate(100, seq_len, "general")
    
    mixing_strategies = {
        "pure_domain": lambda epoch, total: (0.0, 1.0),
        "fixed_90_10": lambda epoch, total: (0.1, 0.9),
        "fixed_70_30": lambda epoch, total: (0.3, 0.7),
        "annealed": lambda epoch, total: (
            max(0.05, 0.3 - 0.25 * epoch / total),
            min(0.95, 0.7 + 0.25 * epoch / total)
        ),
    }
    
    results = {}
    
    for strategy_name, mix_fn in mixing_strategies.items():
        torch.manual_seed(42)
        model = SimpleTransformerLM(vocab_size=500, d_model=64, n_layers=2)
        
        # Pretrain
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(15):
            model.train()
            for i in range(0, len(general_train), 16):
                loss = model.compute_loss(general_train[i:i+16])
                opt.zero_grad(); loss.backward(); opt.step()
        
        # DAPT with mixing
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
        n_epochs = 4
        
        for epoch in range(n_epochs):
            model.train()
            gen_ratio, dom_ratio = mix_fn(epoch, n_epochs - 1)
            
            n_domain = max(16, int(len(domain_train) * dom_ratio))
            n_general = max(0, int(len(general_train) * gen_ratio))
            
            # Sample and mix
            dom_idx = torch.randperm(len(domain_train))[:n_domain]
            mixed = [domain_train[dom_idx]]
            
            if n_general > 0:
                gen_idx = torch.randperm(len(general_train))[:n_general]
                mixed.append(general_train[gen_idx])
            
            mixed = torch.cat(mixed)
            mixed = mixed[torch.randperm(len(mixed))]
            
            for i in range(0, len(mixed), 16):
                loss = model.compute_loss(mixed[i:i+16])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        
        ppl_d = model.perplexity(domain_val)
        ppl_g = model.perplexity(general_val)
        results[strategy_name] = (ppl_d, ppl_g)
        del model, opt
    
    # Print comparison
    print(f"\n  ── DAPT Data Mixing Results ──\n")
    print(f"  {'Strategy':>15} │ {'Domain PPL':>10} │ {'General PPL':>11} │ {'Balance':>7} │ {'Rating':>6}")
    print(f"  {'─'*15}─┼─{'─'*10}─┼─{'─'*11}─┼─{'─'*7}─┼─{'─'*6}")
    
    for name, (ppl_d, ppl_g) in results.items():
        balance = ppl_g / ppl_d
        # Lower balance = better (general PPL not much worse than domain PPL)
        if balance < 1.3 and ppl_d < 50:
            rating = "★★★"
        elif balance < 1.5 and ppl_d < 55:
            rating = "★★"
        else:
            rating = "★"
        
        print(f"  {name:>15} │ {ppl_d:>10.1f} │ {ppl_g:>11.1f} │ {balance:>7.2f} │ {rating:>6}")
    
    print(f"""
  ═══ Data Mixing Recommendations ═══
  
  • Pure domain: Best domain adaptation, worst general preservation
  • Fixed 90/10: Good balance for most scenarios
  • Fixed 70/30: Conservative, preserves more general knowledge
  • Annealed: Best overall balance (recommended)
  
  For production DAPT:
  1. Start with 70/30 (domain/general) ratio
  2. Anneal to 95/5 over training
  3. Monitor both domain AND general perplexity
  4. Stop if general PPL increases >20%
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  DAPT FROM SCRATCH — CONTINUED PRETRAINING IMPLEMENTATIONS      ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    causal_lm_dapt()
    masked_lm_dapt()
    curriculum_dapt()
    lora_dapt()
    data_mixing_dapt()
    
    print("\n" + "=" * 70)
    print("  FROM-SCRATCH MODULE COMPLETE")
    print("=" * 70)
    print("""
    Implemented:
    ✓ Standard causal LM DAPT with warmup + cosine schedule
    ✓ Masked LM DAPT (BERT-style domain adaptation)
    ✓ Curriculum DAPT (gradual domain shift)
    ✓ LoRA-DAPT (parameter-efficient, multiple ranks compared)
    ✓ Data mixing strategies (pure, fixed, annealed)
    """)


if __name__ == "__main__":
    main()
