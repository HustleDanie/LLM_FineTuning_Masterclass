"""
BitFit From Scratch — Manual Bias-Only Fine-Tuning
===================================================

Implementing BitFit without any library support:

1. BiasExtractor
   - Identify all bias parameters in any model
   - Categorize by component type

2. BitFitFreezer
   - Selective parameter freezing strategies
   - Freeze weights, unfreeze biases

3. BitFitTransformer
   - Complete Transformer with BitFit applied
   - From-scratch implementation

4. BitFitGPT2
   - Apply BitFit to real GPT-2
   - Manual freeze/unfreeze implementation

5. BitFitVariants
   - Query-only, attention-only, FF-only variants
   - Custom component selection

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Set


# ============================================================================
# SECTION 1: BIAS EXTRACTOR
# ============================================================================

class BiasExtractor:
    """
    Utility to find and categorize all bias parameters in a model.
    
    Works with any nn.Module — identifies:
    - Linear layer biases
    - LayerNorm beta (shift) parameters
    - Embedding biases (if present)
    - Convolutional biases (if present)
    """
    
    @staticmethod
    def extract_bias_params(model: nn.Module) -> Dict[str, List]:
        """Find all bias-like parameters in a model."""
        
        categories = {
            "linear_bias": [],      # nn.Linear bias
            "layernorm_beta": [],   # nn.LayerNorm bias
            "conv_bias": [],        # nn.Conv bias
            "other_bias": [],       # Any parameter with 'bias' in name
        }
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                categories["linear_bias"].append({
                    "name": f"{name}.bias",
                    "param": module.bias,
                    "shape": tuple(module.bias.shape),
                    "numel": module.bias.numel(),
                })
            elif isinstance(module, nn.LayerNorm) and module.bias is not None:
                categories["layernorm_beta"].append({
                    "name": f"{name}.bias",
                    "param": module.bias,
                    "shape": tuple(module.bias.shape),
                    "numel": module.bias.numel(),
                })
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)) and module.bias is not None:
                categories["conv_bias"].append({
                    "name": f"{name}.bias",
                    "param": module.bias,
                    "shape": tuple(module.bias.shape),
                    "numel": module.bias.numel(),
                })
        
        # Catch any bias parameter not captured above
        found_names = set()
        for cat in categories.values():
            for item in cat:
                found_names.add(item["name"])
        
        for name, param in model.named_parameters():
            if "bias" in name and name not in found_names:
                categories["other_bias"].append({
                    "name": name,
                    "param": param,
                    "shape": tuple(param.shape),
                    "numel": param.numel(),
                })
        
        return categories
    
    @staticmethod
    def summarize(model: nn.Module):
        """Print a summary of all bias parameters."""
        categories = BiasExtractor.extract_bias_params(model)
        
        total_params = sum(p.numel() for p in model.parameters())
        total_bias = 0
        
        print(f"\n  {'Category':<20} {'Count':>6} {'Parameters':>12} {'% of Model':>12}")
        print(f"  {'─'*20}─{'─'*6}─{'─'*12}─{'─'*12}")
        
        for cat_name, items in categories.items():
            count = len(items)
            params = sum(item["numel"] for item in items)
            total_bias += params
            pct = params / total_params * 100 if params > 0 else 0
            print(f"  {cat_name:<20} {count:>6} {params:>12,} {pct:>10.4f}%")
        
        print(f"  {'─'*20}─{'─'*6}─{'─'*12}─{'─'*12}")
        print(f"  {'TOTAL':<20} {'':>6} {total_bias:>12,} "
              f"{total_bias/total_params*100:>10.4f}%")
        print(f"  {'MODEL TOTAL':<20} {'':>6} {total_params:>12,}")
        
        return categories
    
    @staticmethod
    def demonstrate():
        print("=" * 65)
        print("  SECTION 1: BIAS EXTRACTOR")
        print("=" * 65)
        
        # Test with a simple model
        print("\n  ── Simple Model ──")
        
        simple = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 10),
        )
        
        BiasExtractor.summarize(simple)
        
        # Test with a real model
        print("\n\n  ── DistilGPT-2 ──")
        
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        BiasExtractor.summarize(model)
        
        del model


# ============================================================================
# SECTION 2: BITFIT FREEZER
# ============================================================================

class BitFitFreezer:
    """
    Applies BitFit freezing strategy to any model.
    
    ALGORITHM:
    ──────────
    1. Freeze ALL parameters: param.requires_grad = False
    2. Unfreeze ONLY bias parameters: bias.requires_grad = True
    
    That's it! BitFit's beauty is in its simplicity.
    """
    
    @staticmethod
    def apply_bitfit(
        model: nn.Module,
        train_layernorm_beta: bool = True,
        train_linear_bias: bool = True,
        verbose: bool = True,
    ) -> Dict[str, int]:
        """
        Apply BitFit to a model.
        
        Args:
            model: The model to apply BitFit to
            train_layernorm_beta: Whether to train LayerNorm β
            train_linear_bias: Whether to train Linear layer biases
            verbose: Print summary
            
        Returns:
            Dictionary with parameter counts
        """
        # Step 1: Freeze EVERYTHING
        for param in model.parameters():
            param.requires_grad = False
        
        trainable_names = []
        
        # Step 2: Selectively unfreeze bias terms
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and train_linear_bias:
                if module.bias is not None:
                    module.bias.requires_grad = True
                    trainable_names.append(f"{name}.bias")
            
            if isinstance(module, nn.LayerNorm) and train_layernorm_beta:
                if module.bias is not None:
                    module.bias.requires_grad = True
                    trainable_names.append(f"{name}.bias")
            
            # Handle Conv1D (used in GPT-2 for attention)
            if hasattr(module, 'bias') and module.bias is not None:
                if isinstance(module.bias, nn.Parameter):
                    if ("bias" in f"{name}.bias" and
                        f"{name}.bias" not in trainable_names):
                        # Check if this is a type we should train
                        if train_linear_bias:
                            module.bias.requires_grad = True
                            trainable_names.append(f"{name}.bias")
        
        # Count parameters
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        
        stats = {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "pct_trainable": trainable / total * 100,
            "num_trainable_params": len(trainable_names),
        }
        
        if verbose:
            print(f"\n  BitFit Applied:")
            print(f"    Total parameters:     {total:>12,}")
            print(f"    Frozen parameters:    {frozen:>12,}")
            print(f"    Trainable (bias):     {trainable:>12,}")
            print(f"    Trainable %:          {stats['pct_trainable']:>11.4f}%")
            print(f"    Trainable param sets:  {len(trainable_names)}")
        
        return stats
    
    @staticmethod
    def verify_bitfit(model: nn.Module):
        """Verify that only bias terms are trainable."""
        violations = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "bias" not in name:
                    violations.append(name)
        
        if violations:
            print(f"  ⚠ WARNING: Non-bias parameters are trainable:")
            for v in violations:
                print(f"    - {v}")
        else:
            print(f"  ✓ BitFit verified: Only bias parameters are trainable")
        
        return len(violations) == 0
    
    @staticmethod
    def demonstrate():
        print("\n\n" + "=" * 65)
        print("  SECTION 2: BITFIT FREEZER")
        print("=" * 65)
        
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        
        print("\n  Before BitFit:")
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
        
        print("\n  After BitFit:")
        stats = BitFitFreezer.apply_bitfit(model)
        
        print(f"\n  Verification:")
        BitFitFreezer.verify_bitfit(model)
        
        # Show all trainable parameter names
        print(f"\n  All trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"    ✓ {name:<45} [{param.numel():>6}]")
        
        del model


# ============================================================================
# SECTION 3: BITFIT TRANSFORMER (FROM SCRATCH)
# ============================================================================

class BitFitAttention(nn.Module):
    """Multi-head attention with BitFit-style training (only biases trainable)."""
    
    def __init__(self, d_model: int, n_heads: int, freeze_weights: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Weight matrices (FROZEN in BitFit)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        if freeze_weights:
            # Freeze weight matrices, keep biases trainable
            self.W_q.weight.requires_grad = False
            self.W_k.weight.requires_grad = False
            self.W_v.weight.requires_grad = False
            self.W_o.weight.requires_grad = False
            # Biases remain requires_grad=True (default)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        
        # Project Q, K, V (weights frozen, biases trainable)
        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class BitFitFeedForward(nn.Module):
    """Feed-forward network with BitFit (only biases trainable)."""
    
    def __init__(self, d_model: int, d_ff: int, freeze_weights: bool = True):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
        
        if freeze_weights:
            self.up.weight.requires_grad = False
            self.down.weight.requires_grad = False
    
    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class BitFitTransformerBlock(nn.Module):
    """Transformer block with BitFit applied."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, freeze_weights: bool = True):
        super().__init__()
        self.attn = BitFitAttention(d_model, n_heads, freeze_weights)
        self.ff = BitFitFeedForward(d_model, d_ff, freeze_weights)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        if freeze_weights:
            # LayerNorm: freeze gamma (weight), keep beta (bias) trainable
            self.ln1.weight.requires_grad = False
            self.ln2.weight.requires_grad = False
            # ln.bias remains requires_grad=True
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class BitFitTransformerModel(nn.Module):
    """Complete Transformer with BitFit for language modeling."""
    
    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 128,
        freeze_weights: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings (frozen in BitFit — no bias terms here)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            BitFitTransformerBlock(d_model, n_heads, d_model * 4, freeze_weights)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        if freeze_weights:
            self.token_embed.weight.requires_grad = False
            self.pos_embed.weight.requires_grad = False
            self.ln_f.weight.requires_grad = False
            self.head.weight.requires_grad = False
            # ln_f.bias remains trainable
    
    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        
        tok = self.token_embed(input_ids)
        pos = self.pos_embed(torch.arange(T, device=input_ids.device))
        x = tok + pos
        
        mask = torch.tril(torch.ones(T, T, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        
        return {"logits": logits, "loss": loss}
    
    def get_param_breakdown(self) -> Dict:
        """Show trainable vs frozen parameter breakdown."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        breakdown = {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "pct": trainable / total * 100,
        }
        
        return breakdown


def demonstrate_from_scratch():
    """Build and train BitFit Transformer from scratch."""
    print("\n\n" + "=" * 65)
    print("  SECTION 3: BITFIT TRANSFORMER (FROM SCRATCH)")
    print("=" * 65)
    
    torch.manual_seed(42)
    
    model = BitFitTransformerModel(
        vocab_size=500,
        d_model=128,
        n_heads=4,
        n_layers=3,
        max_seq_len=64,
        freeze_weights=True,
    )
    
    breakdown = model.get_param_breakdown()
    
    print(f"\n  Model Configuration:")
    print(f"    Vocab: 500, d_model: 128, {3} layers, {4} heads")
    print(f"\n  Parameter Breakdown:")
    print(f"    Total:     {breakdown['total']:>10,}")
    print(f"    Frozen:    {breakdown['frozen']:>10,}")
    print(f"    Trainable: {breakdown['trainable']:>10,} ({breakdown['pct']:.4f}%)")
    
    # List trainable parameters
    print(f"\n  Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"    ✓ {name:<40} [{param.numel():>5}]")
    
    # Training demo
    print(f"\n  ── Training Demo ──")
    
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, 500, (batch_size, seq_len))
    labels = input_ids.clone()
    
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-2,
    )
    
    print(f"\n  {'Step':>6} {'Loss':>10} {'Bias Δ':>10}")
    print(f"  {'─'*6}─{'─'*10}─{'─'*10}")
    
    for step in range(10):
        out = model(input_ids, labels=labels)
        out["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Measure how much biases have changed
        total_delta = 0
        count = 0
        for param in model.parameters():
            if param.requires_grad:
                total_delta += param.data.abs().mean().item()
                count += 1
        avg_delta = total_delta / count
        
        if (step + 1) % 2 == 0:
            print(f"  {step+1:>6} {out['loss'].item():>10.4f} {avg_delta:>10.4f}")
    
    print("\n  ✓ BitFit Transformer training complete!")


# ============================================================================
# SECTION 4: BITFIT ON REAL GPT-2
# ============================================================================

def bitfit_gpt2():
    """Apply BitFit to real GPT-2 model."""
    print("\n\n" + "=" * 65)
    print("  SECTION 4: BITFIT ON REAL GPT-2")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # ─── Manual BitFit Implementation ───
    print("\n  Step 1: Freeze all parameters")
    for param in model.parameters():
        param.requires_grad = False
    
    print("  Step 2: Unfreeze bias parameters")
    bias_count = 0
    bias_params = 0
    
    for name, param in model.named_parameters():
        if "bias" in name:
            param.requires_grad = True
            bias_count += 1
            bias_params += param.numel()
    
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n  Results:")
    print(f"    Bias parameter sets: {bias_count}")
    print(f"    Bias parameters:     {bias_params:,}")
    print(f"    Total parameters:    {total_params:,}")
    print(f"    Trainable %:         {bias_params/total_params*100:.4f}%")
    
    print(f"\n  All trainable parameters in BitFit GPT-2:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"    • {name:<45} shape={list(param.shape)}")
    
    # ─── Quick Training Demo ───
    print(f"\n  ── Training Demo ──")
    
    texts = [
        "BitFit is a simple method that only trains bias terms.",
        "Parameter efficiency is achieved by freezing all weights.",
        "Bias terms control activation thresholds in neural networks.",
        "This approach requires modifying less than 0.1% of parameters.",
    ]
    
    inputs = tokenizer(
        texts, padding=True, truncation=True,
        max_length=32, return_tensors="pt",
    )
    labels = inputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-2,  # BitFit uses higher LR (like IA³)
        weight_decay=0.0,
    )
    
    # Record initial bias values for comparison
    initial_biases = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_biases[name] = param.data.clone()
    
    model.train()
    for epoch in range(5):
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (epoch + 1) % 1 == 0:
            print(f"    Epoch {epoch+1}: loss = {out.loss.item():.4f}")
    
    # Show how biases changed
    print(f"\n  Bias Changes After Training:")
    print(f"  {'Parameter':<45} {'Max Δ':>8} {'Mean Δ':>8}")
    print(f"  {'─'*45}─{'─'*8}─{'─'*8}")
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            delta = (param.data - initial_biases[name]).abs()
            print(f"  {name:<45} {delta.max().item():>8.4f} {delta.mean().item():>8.4f}")
    
    # ─── Save and Load ───
    print(f"\n  ── Save/Load BitFit Checkpoint ──")
    
    # Save only bias parameters
    bitfit_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            bitfit_state[name] = param.data.clone()
    
    import tempfile, os
    save_path = os.path.join(tempfile.gettempdir(), "bitfit_checkpoint.pt")
    torch.save(bitfit_state, save_path)
    
    file_size = os.path.getsize(save_path) / 1024
    print(f"    Saved {len(bitfit_state)} bias tensors to checkpoint")
    print(f"    File size: {file_size:.1f} KB")
    print(f"    (vs full model: ~{total_params * 4 / 1024 / 1024:.0f} MB)")
    
    # Load into fresh model
    fresh_model = AutoModelForCausalLM.from_pretrained(model_name)
    loaded_state = torch.load(save_path, weights_only=True)
    
    for name, param in fresh_model.named_parameters():
        if name in loaded_state:
            param.data = loaded_state[name]
    
    print(f"    ✓ Loaded BitFit checkpoint into fresh model")
    
    os.remove(save_path)
    del model, fresh_model


# ============================================================================
# SECTION 5: BITFIT VARIANTS
# ============================================================================

def bitfit_variants():
    """Implement different BitFit configurations."""
    print("\n\n" + "=" * 65)
    print("  SECTION 5: BITFIT VARIANTS")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM
    
    model_name = "distilgpt2"
    
    def apply_variant(model, variant_name, filter_fn):
        """Apply a custom BitFit variant."""
        for param in model.parameters():
            param.requires_grad = False
        
        trainable = 0
        count = 0
        for name, param in model.named_parameters():
            if filter_fn(name):
                param.requires_grad = True
                trainable += param.numel()
                count += 1
        
        total = sum(p.numel() for p in model.parameters())
        return {
            "name": variant_name,
            "trainable": trainable,
            "total": total,
            "pct": trainable / total * 100,
            "count": count,
        }
    
    variants = {
        "Full BitFit": lambda n: "bias" in n,
        "Attention bias only": lambda n: "bias" in n and "c_attn" in n,
        "FF bias only": lambda n: "bias" in n and "mlp" in n,
        "LayerNorm only": lambda n: "bias" in n and "ln_" in n,
        "Query bias approx": lambda n: "bias" in n and "c_attn" in n,
        "Output proj only": lambda n: "bias" in n and ("c_proj" in n and "mlp" not in n),
        "LN + Attn bias": lambda n: "bias" in n and ("ln_" in n or "c_attn" in n),
        "Attn + FF bias": lambda n: "bias" in n and ("c_attn" in n or "mlp" in n),
    }
    
    print(f"\n  {'Variant':<22} {'Trainable':>12} {'%':>10} {'Param Sets':>12}")
    print(f"  {'─'*22}─{'─'*12}─{'─'*10}─{'─'*12}")
    
    for variant_name, filter_fn in variants.items():
        model = AutoModelForCausalLM.from_pretrained(model_name)
        stats = apply_variant(model, variant_name, filter_fn)
        print(f"  {stats['name']:<22} {stats['trainable']:>12,} "
              f"{stats['pct']:>9.4f}% {stats['count']:>12}")
        del model
    
    print(f"""
  ═══ Variant Selection Guide ═══
  
  • Full BitFit: Best overall; use this as default
  • LN + Attn bias: Second best; good balance
  • Attention bias only: Minimal but surprisingly effective
  • FF bias only: Good for tasks needing knowledge recall
  • LayerNorm only: Extreme minimalism (worse performance)
  
  BitFit + other methods (composable):
  • BitFit + LoRA: Train biases AND low-rank matrices
  • BitFit + IA³: Train biases AND rescaling vectors  
  • These combinations often outperform either alone
""")
    
    # Demonstrate composition with IA³
    print("  ── BitFit + LoRA Composition ──")
    
    from peft import get_peft_model, LoraConfig, TaskType
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = model.config.eos_token_id
    
    # Apply LoRA first
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4, lora_alpha=8,
        target_modules=["c_attn"],
        bias="all",  # This tells LoRA to also train biases!
    )
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"  LoRA (r=4) + all biases (bias='all'):")
    print(f"    Trainable: {trainable:,} ({trainable/total*100:.4f}%)")
    
    # Show which params are trainable
    lora_params = 0
    bias_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora" in name:
                lora_params += param.numel()
            elif "bias" in name:
                bias_params += param.numel()
    
    print(f"    LoRA params:  {lora_params:,}")
    print(f"    Bias params:  {bias_params:,}")
    print(f"    Combined:     {lora_params + bias_params:,}")
    
    del model


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all from-scratch implementations."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║    BitFit FROM SCRATCH — MANUAL BIAS-ONLY FINE-TUNING        ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: Bias extractor
    BiasExtractor.demonstrate()
    
    # Section 2: BitFit freezer
    BitFitFreezer.demonstrate()
    
    # Section 3: From-scratch Transformer
    demonstrate_from_scratch()
    
    # Section 4: Real GPT-2
    bitfit_gpt2()
    
    # Section 5: Variants
    bitfit_variants()
    
    print("\n" + "=" * 65)
    print("  FROM-SCRATCH MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered:
    ✓ Bias parameter extraction and categorization
    ✓ BitFit freezing strategy (freeze all, unfreeze bias)
    ✓ Complete Transformer with BitFit from scratch
    ✓ BitFit on real GPT-2 with save/load
    ✓ Variants (attention-only, FF-only, LN-only, compositions)
    """)


if __name__ == "__main__":
    main()
