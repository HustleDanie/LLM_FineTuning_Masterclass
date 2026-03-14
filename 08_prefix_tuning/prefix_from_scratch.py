"""
Prefix Tuning — From Scratch Implementation
=============================================

Build prefix tuning from scratch in PyTorch:

1. PrefixEmbedding — Learnable prefix parameter table
2. PrefixAttention — Modified attention with prefix K, V
3. PrefixTransformerLayer — Full transformer layer with prefix
4. PrefixModel — Complete prefix-tuned model wrapper
5. Applying Prefix Tuning to GPT-2

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


# ============================================================================
# SECTION 1: PREFIX EMBEDDING
# ============================================================================

class PrefixEmbedding(nn.Module):
    """
    Learnable prefix parameter table with optional reparameterization.
    
    Two modes:
    1. Direct: Directly optimize the prefix vectors (simpler, less stable)
    2. Reparameterized: Use a small MLP to generate prefix vectors (more stable)
    
    For each layer, we need:
    - P_k: prefix for keys   (prefix_len, d_model)
    - P_v: prefix for values (prefix_len, d_model)
    
    Architecture:
    
    Direct mode:
        P[layer] = nn.Parameter(prefix_len, 2 * d_model)
    
    Reparameterized mode:
        E[layer] = nn.Embedding(prefix_len, d_reparam)
        MLP: d_reparam → d_model
        P_k[layer] = MLP_k(E[layer])
        P_v[layer] = MLP_v(E[layer])
    """
    
    def __init__(
        self,
        num_layers: int,
        prefix_len: int,
        d_model: int,
        num_heads: int = 1,
        reparameterize: bool = True,
        reparam_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.prefix_len = prefix_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.reparameterize = reparameterize
        
        if reparameterize:
            # Reparameterization: embedding → MLP → prefix
            self.embedding = nn.Embedding(prefix_len, reparam_dim)
            
            # One MLP that produces both k and v for all layers
            # Output: num_layers * 2 (k+v) * d_model
            self.mlp = nn.Sequential(
                nn.Linear(reparam_dim, reparam_dim),
                nn.Tanh(),
                nn.Linear(reparam_dim, num_layers * 2 * d_model),
            )
            
            self.dropout = nn.Dropout(dropout)
        else:
            # Direct parameterization
            # Shape: (num_layers, 2, prefix_len, d_model)
            self.prefix_params = nn.Parameter(
                torch.randn(num_layers, 2, prefix_len, d_model) * 0.01
            )
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Generate prefix key-value pairs for all layers.
        
        Returns:
            prefix: (num_layers, 2, batch_size, prefix_len, d_model)
                    [layer_idx, 0=key/1=value, batch, prefix_pos, dim]
        """
        if self.reparameterize:
            # Input IDs for prefix positions: [0, 1, 2, ..., prefix_len-1]
            prefix_ids = torch.arange(self.prefix_len, device=self.embedding.weight.device)
            
            # Embed: (prefix_len, reparam_dim)
            embeds = self.embedding(prefix_ids)
            
            # MLP: (prefix_len, num_layers * 2 * d_model)
            prefix_flat = self.mlp(embeds)
            prefix_flat = self.dropout(prefix_flat)
            
            # Reshape: (prefix_len, num_layers, 2, d_model)
            prefix = prefix_flat.view(
                self.prefix_len, self.num_layers, 2, self.d_model
            )
            
            # Permute to: (num_layers, 2, prefix_len, d_model)
            prefix = prefix.permute(1, 2, 0, 3)
        else:
            prefix = self.prefix_params
        
        # Expand for batch: (num_layers, 2, batch_size, prefix_len, d_model)
        prefix = prefix.unsqueeze(2).expand(-1, -1, batch_size, -1, -1)
        
        return prefix.contiguous()
    
    def get_prefix_for_layer(
        self, layer_idx: int, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get P_k, P_v for a specific layer.
        
        Returns:
            P_k: (batch_size, prefix_len, d_model)
            P_v: (batch_size, prefix_len, d_model)
        """
        all_prefix = self.forward(batch_size)
        P_k = all_prefix[layer_idx, 0]  # (batch, prefix_len, d_model)
        P_v = all_prefix[layer_idx, 1]  # (batch, prefix_len, d_model)
        return P_k, P_v


def demonstrate_prefix_embedding():
    """Demonstrate the PrefixEmbedding module."""
    print("=" * 65)
    print("  SECTION 1: PREFIX EMBEDDING")
    print("=" * 65)
    
    num_layers = 6
    prefix_len = 10
    d_model = 256
    batch_size = 4
    
    # Direct mode
    print(f"\n  ── Direct Parameterization ──")
    direct = PrefixEmbedding(
        num_layers, prefix_len, d_model, reparameterize=False
    )
    prefix_direct = direct(batch_size)
    n_direct = sum(p.numel() for p in direct.parameters())
    print(f"  Parameters: {n_direct:,}")
    print(f"  Output shape: {prefix_direct.shape}")
    print(f"  Expected: ({num_layers}, 2, {batch_size}, {prefix_len}, {d_model})")
    
    # Reparameterized mode
    print(f"\n  ── Reparameterized (MLP) ──")
    reparam = PrefixEmbedding(
        num_layers, prefix_len, d_model,
        reparameterize=True, reparam_dim=128,
    )
    prefix_reparam = reparam(batch_size)
    n_reparam = sum(p.numel() for p in reparam.parameters())
    print(f"  Parameters: {n_reparam:,}")
    print(f"  Output shape: {prefix_reparam.shape}")
    
    # Get prefix for specific layer
    P_k, P_v = reparam.get_prefix_for_layer(0, batch_size)
    print(f"\n  Layer 0 prefix keys:   {P_k.shape}")
    print(f"  Layer 0 prefix values: {P_v.shape}")
    
    print(f"\n  Parameter comparison:")
    print(f"    Direct:         {n_direct:>10,} params")
    print(f"    Reparameterized: {n_reparam:>10,} params")
    print(f"    Note: Reparameterized has MORE params during training")
    print(f"    but generates better-conditioned prefix vectors.")
    print(f"    After training, the MLP is discarded and only the")
    print(f"    generated prefix vectors ({n_direct:,}) are kept.")


# ============================================================================
# SECTION 2: PREFIX ATTENTION
# ============================================================================

class PrefixMultiHeadAttention(nn.Module):
    """
    Multi-head attention with prefix key-value pairs.
    
    Each head gets its portion of the prefix:
    - P_k is split into num_heads chunks for key prefix
    - P_v is split into num_heads chunks for value prefix
    
    K' = [P_k_head_i ; K_head_i]  for each head i
    V' = [P_v_head_i ; V_head_i]  for each head i
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # Standard QKV projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        X: torch.Tensor,                    # (B, S, d_model)
        P_k: Optional[torch.Tensor] = None, # (B, L, d_model) prefix keys
        P_v: Optional[torch.Tensor] = None, # (B, L, d_model) prefix values
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional prefix.
        
        If P_k and P_v are provided:
            K = [P_k ; X·W_k],  V = [P_v ; X·W_v]
            Attention is computed over (prefix_len + seq_len) positions
        
        If not provided:
            Standard attention (no prefix)
        """
        B, S, _ = X.shape
        
        # Project Q, K, V
        Q = self.W_q(X)  # (B, S, d_model)
        K = self.W_k(X)  # (B, S, d_model)
        V = self.W_v(X)  # (B, S, d_model)
        
        # Prepend prefix to K, V if provided
        prefix_len = 0
        if P_k is not None and P_v is not None:
            prefix_len = P_k.size(1)
            K = torch.cat([P_k, K], dim=1)  # (B, L+S, d_model)
            V = torch.cat([P_v, V], dim=1)  # (B, L+S, d_model)
        
        # Reshape for multi-head: (B, num_heads, seq, d_head)
        Q = Q.view(B, S, self.num_heads, self.d_head).transpose(1, 2)
        total_kv_len = prefix_len + S
        K = K.view(B, total_kv_len, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(B, total_kv_len, self.num_heads, self.d_head).transpose(1, 2)
        
        # Attention scores: (B, H, S, L+S)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Extend mask to account for prefix positions
            if prefix_len > 0:
                prefix_mask = torch.ones(B, 1, 1, prefix_len, device=X.device)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Output: (B, H, S, d_head) → (B, S, d_model)
        output = (attn_weights @ V).transpose(1, 2).contiguous().view(B, S, self.d_model)
        output = self.W_o(output)
        
        return output, attn_weights


def demonstrate_prefix_attention():
    """Demonstrate prefix multi-head attention."""
    print("\n" + "=" * 65)
    print("  SECTION 2: PREFIX MULTI-HEAD ATTENTION")
    print("=" * 65)
    
    B, S, L = 2, 8, 5
    d_model, num_heads = 128, 4
    
    torch.manual_seed(42)
    attn = PrefixMultiHeadAttention(d_model, num_heads)
    X = torch.randn(B, S, d_model)
    
    # Without prefix
    out_no_prefix, w_no_prefix = attn(X)
    print(f"\n  Without prefix:")
    print(f"    Input:    {X.shape}")
    print(f"    Output:   {out_no_prefix.shape}")
    print(f"    Attn:     {w_no_prefix.shape}  (S×S = {S}×{S})")
    
    # With prefix
    P_k = torch.randn(B, L, d_model)
    P_v = torch.randn(B, L, d_model)
    out_prefix, w_prefix = attn(X, P_k, P_v)
    print(f"\n  With prefix (prefix_len={L}):")
    print(f"    Input:    {X.shape}")
    print(f"    P_k, P_v: ({B}, {L}, {d_model})")
    print(f"    Output:   {out_prefix.shape}  (same shape!)")
    print(f"    Attn:     {w_prefix.shape}  (S×(L+S) = {S}×{L+S})")
    
    # Analyze attention to prefix
    prefix_attn = w_prefix[:, :, :, :L].mean().item()
    real_attn = w_prefix[:, :, :, L:].mean().item()
    print(f"\n  Average attention allocation:")
    print(f"    To prefix: {prefix_attn:.4f}")
    print(f"    To real:   {real_attn:.4f}")
    
    # Compare outputs
    delta = (out_prefix - out_no_prefix).norm().item()
    print(f"\n  Output difference (with vs without prefix): {delta:.4f}")
    print(f"  → Prefix significantly changes the attention output!")


# ============================================================================
# SECTION 3: PREFIX TRANSFORMER LAYER
# ============================================================================

class PrefixTransformerLayer(nn.Module):
    """
    Full transformer layer with prefix tuning.
    
    Standard:
        h = LayerNorm(x + Attention(x))
        h = LayerNorm(h + FFN(h))
    
    With prefix:
        h = LayerNorm(x + PrefixAttention(x, P_k, P_v))
        h = LayerNorm(h + FFN(h))
    
    Only the attention layer is modified — FFN is untouched.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Prefix-augmented attention
        self.attention = PrefixMultiHeadAttention(d_model, num_heads, dropout)
        
        # Standard FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        P_k: Optional[torch.Tensor] = None,
        P_v: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward with optional prefix."""
        # Pre-norm attention with prefix
        normed = self.ln1(x)
        attn_out, _ = self.attention(normed, P_k, P_v, attention_mask)
        x = x + self.dropout(attn_out)
        
        # Pre-norm FFN
        normed = self.ln2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        
        return x


class PrefixTransformer(nn.Module):
    """
    Complete transformer with prefix tuning.
    
    Architecture:
    1. Token embedding + positional embedding
    2. Prefix embedding (generates P_k, P_v for all layers)
    3. N transformer layers (each with prefix attention)
    4. Output head
    
    Only the PrefixEmbedding parameters are trainable!
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_len: int,
        prefix_len: int,
        reparameterize: bool = True,
        reparam_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.prefix_len = prefix_len
        self.num_layers = num_layers
        
        # Token and position embeddings (FROZEN)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Transformer layers (FROZEN)
        self.layers = nn.ModuleList([
            PrefixTransformerLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output head (FROZEN)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Prefix embedding (TRAINABLE!)
        self.prefix_embed = PrefixEmbedding(
            num_layers=num_layers,
            prefix_len=prefix_len,
            d_model=d_model,
            num_heads=num_heads,
            reparameterize=reparameterize,
            reparam_dim=reparam_dim,
            dropout=dropout,
        )
        
        # Freeze everything except prefix
        self._freeze_base_model()
    
    def _freeze_base_model(self):
        """Freeze all parameters except the prefix embedding."""
        for name, param in self.named_parameters():
            if "prefix_embed" not in name:
                param.requires_grad = False
    
    def get_trainable_params(self) -> Dict[str, int]:
        """Get trainable vs frozen parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen
        return {
            "trainable": trainable,
            "frozen": frozen,
            "total": total,
            "trainable_pct": trainable / total * 100,
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,           # (B, S)
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass with prefix tuning.
        
        1. Embed tokens
        2. Generate prefix K, V for all layers
        3. Pass through transformer with prefix at each layer
        4. Compute logits and optional loss
        """
        B, S = input_ids.shape
        device = input_ids.device
        
        # Token + position embeddings
        positions = torch.arange(S, device=device).unsqueeze(0)
        h = self.token_embed(input_ids) + self.pos_embed(positions)
        h = self.embed_dropout(h)
        
        # Generate prefix for all layers
        # Shape: (num_layers, 2, B, prefix_len, d_model)
        all_prefix = self.prefix_embed(B)
        
        # Pass through transformer layers with layer-specific prefix
        for i, layer in enumerate(self.layers):
            P_k = all_prefix[i, 0]  # (B, prefix_len, d_model)
            P_v = all_prefix[i, 1]  # (B, prefix_len, d_model)
            h = layer(h, P_k, P_v, attention_mask)
        
        # Output
        h = self.ln_f(h)
        logits = self.lm_head(h)  # (B, S, vocab_size)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return {"logits": logits, "loss": loss}


def demonstrate_prefix_transformer():
    """Demonstrate the complete prefix transformer."""
    print("\n" + "=" * 65)
    print("  SECTION 3: PREFIX TRANSFORMER (FROM SCRATCH)")
    print("=" * 65)
    
    # Config
    vocab_size = 1000
    d_model = 128
    num_heads = 4
    num_layers = 4
    d_ff = 512
    max_seq_len = 256
    prefix_len = 10
    
    # Build model
    model = PrefixTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        prefix_len=prefix_len,
        reparameterize=True,
        reparam_dim=64,
    )
    
    params = model.get_trainable_params()
    print(f"\n  Model configuration:")
    print(f"    vocab_size:  {vocab_size}")
    print(f"    d_model:     {d_model}")
    print(f"    num_heads:   {num_heads}")
    print(f"    num_layers:  {num_layers}")
    print(f"    prefix_len:  {prefix_len}")
    
    print(f"\n  Parameter breakdown:")
    print(f"    Trainable (prefix): {params['trainable']:>10,}")
    print(f"    Frozen (model):     {params['frozen']:>10,}")
    print(f"    Total:              {params['total']:>10,}")
    print(f"    Trainable %:        {params['trainable_pct']:>9.2f}%")
    
    # Forward pass
    B, S = 2, 20
    input_ids = torch.randint(0, vocab_size, (B, S))
    labels = input_ids.clone()
    
    output = model(input_ids, labels=labels)
    print(f"\n  Forward pass:")
    print(f"    Input:  {input_ids.shape}")
    print(f"    Logits: {output['logits'].shape}")
    print(f"    Loss:   {output['loss']:.4f}")
    
    # Backward pass
    output["loss"].backward()
    
    # Verify gradient flow
    has_grad = 0
    no_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad += 1
        else:
            no_grad += 1
    
    print(f"\n  Gradient check:")
    print(f"    Parameters with gradients:    {has_grad} (prefix)")
    print(f"    Parameters without gradients: {no_grad} (frozen base)")
    
    # Show prefix-specific gradients
    print(f"\n  Prefix parameter gradients:")
    for name, param in model.prefix_embed.named_parameters():
        if param.grad is not None:
            print(f"    {name:>30}: grad_norm = {param.grad.norm():.6f}")


# ============================================================================
# SECTION 4: APPLYING PREFIX TUNING TO GPT-2
# ============================================================================

class GPT2PrefixInjector:
    """
    Inject prefix tuning into a pre-trained GPT-2 model.
    
    This demonstrates how to add prefix parameters to a real model
    without modifying its architecture — we hook into the attention
    mechanism to prepend our learned prefix vectors.
    """
    
    @staticmethod
    def inject_prefix(
        prefix_len: int = 20,
        reparam_dim: int = 512,
    ):
        """
        Inject prefix tuning into GPT-2.
        """
        print("\n" + "=" * 65)
        print("  SECTION 4: PREFIX TUNING ON GPT-2")
        print("=" * 65)
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "distilgpt2"
        print(f"\n  Loading {model_name}...")
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Get model dimensions
        config = model.config
        d_model = config.n_embd       # 768
        num_layers = config.n_layer    # 6 for distilgpt2
        num_heads = config.n_head      # 12
        
        print(f"  d_model={d_model}, layers={num_layers}, heads={num_heads}")
        
        # Create prefix embedding
        prefix_embed = PrefixEmbedding(
            num_layers=num_layers,
            prefix_len=prefix_len,
            d_model=d_model,
            num_heads=num_heads,
            reparameterize=True,
            reparam_dim=reparam_dim,
        )
        
        # Freeze model parameters
        for param in model.parameters():
            param.requires_grad = False
        
        total_params = sum(p.numel() for p in model.parameters())
        prefix_params = sum(p.numel() for p in prefix_embed.parameters())
        
        print(f"\n  Model parameters:  {total_params:>12,} (frozen)")
        print(f"  Prefix parameters: {prefix_params:>12,} (trainable)")
        print(f"  Trainable %:       {prefix_params / total_params * 100:>11.4f}%")
        
        # Demonstrate using past_key_values for prefix injection
        print(f"\n  ── Prefix as past_key_values ──")
        print(f"  GPT-2 supports `past_key_values` argument in forward().")
        print(f"  We can inject prefix as if they are cached KV from")
        print(f"  previous tokens — the model treats them identically!")
        
        # Generate prefix KV pairs in the format GPT-2 expects
        batch_size = 1
        all_prefix = prefix_embed(batch_size)
        # Shape: (num_layers, 2, B, prefix_len, d_model)
        
        # Convert to GPT-2 past_key_values format:
        # tuple of (key, value) for each layer
        # key: (batch, num_heads, prefix_len, d_head)
        d_head = d_model // num_heads
        
        past_key_values = []
        for layer_idx in range(num_layers):
            P_k = all_prefix[layer_idx, 0]  # (B, prefix_len, d_model)
            P_v = all_prefix[layer_idx, 1]  # (B, prefix_len, d_model)
            
            # Reshape to multi-head format
            P_k = P_k.view(batch_size, prefix_len, num_heads, d_head).transpose(1, 2)
            P_v = P_v.view(batch_size, prefix_len, num_heads, d_head).transpose(1, 2)
            
            past_key_values.append((P_k, P_v))
        
        past_key_values = tuple(past_key_values)
        
        print(f"\n  past_key_values structure:")
        print(f"    Length: {len(past_key_values)} (one per layer)")
        print(f"    Key shape:   {past_key_values[0][0].shape}")
        print(f"    Value shape: {past_key_values[0][1].shape}")
        
        # Forward pass with prefix
        text = "The meaning of life is"
        inputs = tokenizer(text, return_tensors="pt")
        
        # Need to create attention mask that includes prefix
        prefix_attention_mask = torch.ones(batch_size, prefix_len, dtype=torch.long)
        full_attention_mask = torch.cat(
            [prefix_attention_mask, inputs["attention_mask"]], dim=1
        )
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=full_attention_mask,
                past_key_values=past_key_values,
            )
        
        print(f"\n  Forward pass with prefix:")
        print(f"    Input: \"{text}\"")
        print(f"    Logits shape: {outputs.logits.shape}")
        
        # Get next token prediction
        next_token_id = outputs.logits[0, -1].argmax()
        next_token = tokenizer.decode(next_token_id)
        print(f"    Next token prediction: \"{next_token}\"")
        
        print(f"\n  ✓ Prefix tuning successfully injected into GPT-2!")
        print(f"    The prefix_embed module's parameters are the only")
        print(f"    trainable parameters. Training would optimize these")
        print(f"    to steer GPT-2 toward the target task.")
        
        return model, prefix_embed, tokenizer


# ============================================================================
# SECTION 5: INITIALIZATION STRATEGIES
# ============================================================================

class PrefixInitialization:
    """
    Initialization strategies for prefix parameters.
    
    Good initialization is critical for prefix tuning stability.
    """
    
    @staticmethod
    def compare_initializations():
        """Compare different prefix initialization strategies."""
        print("\n" + "=" * 65)
        print("  SECTION 5: PREFIX INITIALIZATION STRATEGIES")
        print("=" * 65)
        
        prefix_len = 20
        d_model = 256
        
        initializations = {}
        
        # 1. Random normal
        torch.manual_seed(42)
        initializations["Random Normal (σ=0.02)"] = torch.randn(prefix_len, d_model) * 0.02
        
        # 2. Random uniform
        torch.manual_seed(42)
        initializations["Random Uniform [-0.5, 0.5]"] = torch.rand(prefix_len, d_model) - 0.5
        
        # 3. Xavier/Glorot
        torch.manual_seed(42)
        p = torch.empty(prefix_len, d_model)
        nn.init.xavier_normal_(p)
        initializations["Xavier Normal"] = p
        
        # 4. From vocabulary embeddings
        torch.manual_seed(42)
        vocab_embed = nn.Embedding(1000, d_model)
        task_tokens = torch.randint(0, 1000, (prefix_len,))
        initializations["Vocab Embedding Init"] = vocab_embed(task_tokens).detach()
        
        # 5. Zeros (with reparameterization)
        initializations["Zeros (needs reparam)"] = torch.zeros(prefix_len, d_model)
        
        print(f"\n  {'Strategy':>30} {'Mean':>10} {'Std':>10} {'Norm':>10}")
        print(f"  {'─'*30}─{'─'*10}─{'─'*10}─{'─'*10}")
        
        for name, params in initializations.items():
            print(f"  {name:>30}  {params.mean():>8.5f}  "
                  f"{params.std():>8.5f}  {params.norm():>8.3f}")
        
        print(f"""
  Recommendations:
  ─────────────────────────────────────────────────────────────
  1. With reparameterization (MLP): Use default PyTorch init
     The MLP itself provides good conditioning of the prefix.
     
  2. Without reparameterization:
     a. Best: Initialize from actual token embeddings
        (e.g., embeddings of task-descriptive words)
     b. Good: Random normal with σ = 0.01-0.02
     c. Bad:  Large random values → unstable training
     d. Bad:  Zeros → no gradient signal initially
  
  3. The reparameterization MLP is the best solution for
     initialization stability — it was the key insight of
     the original prefix tuning paper.
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all prefix tuning from-scratch demonstrations."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║         PREFIX TUNING — FROM SCRATCH IMPLEMENTATION          ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: Prefix Embedding
    demonstrate_prefix_embedding()
    
    # Section 2: Prefix Multi-Head Attention
    demonstrate_prefix_attention()
    
    # Section 3: Full Prefix Transformer
    demonstrate_prefix_transformer()
    
    # Section 4: Apply to GPT-2
    GPT2PrefixInjector.inject_prefix(prefix_len=20)
    
    # Section 5: Initialization
    PrefixInitialization.compare_initializations()
    
    print("\n" + "=" * 65)
    print("  MODULE COMPLETE")
    print("=" * 65)
    print("""
    Built from scratch:
    ✓ PrefixEmbedding (direct + reparameterized)
    ✓ PrefixMultiHeadAttention (K, V augmentation)
    ✓ PrefixTransformerLayer (attention + FFN)
    ✓ PrefixTransformer (full model with prefix)
    ✓ GPT-2 prefix injection via past_key_values
    ✓ Initialization strategies
    """)


if __name__ == "__main__":
    main()
