"""
P-Tuning v2 — From Scratch Implementation
==========================================

P-Tuning v2 (Liu et al., 2022) - Deep Prompt Tuning:

1. Deep Prompt Injection
   - Learnable prompts at EVERY transformer layer
   - No LSTM encoder needed (direct optimization)

2. Layer-Wise Prompt Architecture
   - Each layer has independent prompt parameters
   - Fresh signal injection at every depth

3. Full P-Tuning v2 Model
   - Deep prompts + frozen transformer
   - Matches full fine-tuning across scales

4. GPT-2 Integration
   - Real model with deep prompt injection
   - Comparison: shallow vs deep prompts

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple


# ============================================================================
# SECTION 1: DEEP PROMPT INJECTION
# ============================================================================

class DeepPromptPool(nn.Module):
    """
    Manages learnable prompts for every layer of a transformer.
    
    Unlike P-Tuning v1 (LSTM encoder → input-only prompts):
    - v2 has INDEPENDENT prompts at each layer
    - No encoder needed — direct optimization works for deep prompts
    - Each layer's prompts are separate parameters
    
    Shape: [n_layers, num_tokens, d_model]
    
    Why no encoder is needed:
    - Deep prompts provide enough expressive power on their own
    - Inter-layer interaction through attention already creates coherence
    - Direct optimization of more parameters is easier than reparameterizing few
    """
    
    def __init__(
        self,
        n_layers: int,
        num_tokens: int,
        d_model: int,
        init_std: float = 0.02,
    ):
        super().__init__()
        
        self.n_layers = n_layers
        self.num_tokens = num_tokens
        self.d_model = d_model
        
        # Independent prompts for each layer
        # Each is a separate Parameter so they're optimized independently
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.randn(num_tokens, d_model) * init_std)
            for _ in range(n_layers)
        ])
        
        total = sum(p.numel() for p in self.parameters())
        print(f"  DeepPromptPool:")
        print(f"    Layers: {n_layers}")
        print(f"    Tokens per layer: {num_tokens}")
        print(f"    Token dim: {d_model}")
        print(f"    Total parameters: {total:,}")
        print(f"    Per layer: {num_tokens * d_model:,}")
    
    def get_layer_prompts(self, layer_idx: int, batch_size: int = 1) -> torch.Tensor:
        """
        Get prompts for a specific layer.
        
        Returns: [batch_size, num_tokens, d_model]
        """
        prompt = self.prompts[layer_idx]  # [num_tokens, d_model]
        return prompt.unsqueeze(0).expand(batch_size, -1, -1)
    
    def get_all_prompts(self, batch_size: int = 1) -> List[torch.Tensor]:
        """
        Get prompts for all layers.
        
        Returns: List of [batch_size, num_tokens, d_model]
        """
        return [self.get_layer_prompts(i, batch_size) for i in range(self.n_layers)]


def demonstrate_deep_prompts():
    """Demonstrate deep prompt injection."""
    print("=" * 65)
    print("  SECTION 1: DEEP PROMPT INJECTION")
    print("=" * 65)
    
    torch.manual_seed(42)
    
    pool = DeepPromptPool(
        n_layers=6,
        num_tokens=8,
        d_model=256,
    )
    
    # Get prompts for each layer
    all_prompts = pool.get_all_prompts(batch_size=2)
    
    print(f"\n  Generated prompts for all layers:")
    for i, p in enumerate(all_prompts):
        print(f"  Layer {i}: shape={p.shape}, norm={p[0].norm():.4f}")
    
    # Show independence of layer prompts
    print(f"\n  Layer prompt independence:")
    cos_sims = []
    for i in range(len(all_prompts)):
        for j in range(i+1, len(all_prompts)):
            vi = all_prompts[i][0].flatten()
            vj = all_prompts[j][0].flatten()
            sim = F.cosine_similarity(vi.unsqueeze(0), vj.unsqueeze(0)).item()
            cos_sims.append((i, j, sim))
    
    for i, j, sim in cos_sims[:5]:
        print(f"    Layer {i} ↔ {j}: cos_sim = {sim:.4f}")
    
    print(f"    Average similarity: {sum(s for _,_,s in cos_sims)/len(cos_sims):.4f}")
    print(f"    (Near 0 = independent initialization ✓)")


# ============================================================================
# SECTION 2: TRANSFORMER WITH DEEP PROMPT INJECTION
# ============================================================================

class DeepPromptTransformerLayer(nn.Module):
    """
    A single transformer layer that accepts injected prompts.
    
    At each layer:
    1. Prepend layer-specific prompts to key/value
    2. Compute attention (queries see both prompts and hidden states)
    3. Remove prompt positions from output
    
    This is the core mechanism of P-Tuning v2 / Deep Prefix Tuning.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Multi-head attention components
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        
        # Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def _multi_head_attention(
        self,
        query: torch.Tensor,   # [B, Lq, D]
        key: torch.Tensor,     # [B, Lk, D]
        value: torch.Tensor,   # [B, Lk, D]
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Lq = query.shape[:2]
        Lk = key.shape[1]
        
        Q = self.q_proj(query).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(key).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(value).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if causal_mask is not None:
            scores = scores + causal_mask
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.o_proj(out)
    
    def forward(
        self,
        hidden: torch.Tensor,          # [B, L, D]
        prompt_kv: Optional[torch.Tensor] = None,  # [B, P, D]
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward with optional prompt injection.
        
        If prompt_kv is provided:
        - Keys and Values = [prompt_kv; hidden]
        - Queries = hidden only
        - This lets hidden states attend to prompts
        """
        residual = hidden
        hidden_norm = self.norm1(hidden)
        
        if prompt_kv is not None:
            # Concatenate prompts as extra key-value context
            kv_input = torch.cat([prompt_kv, hidden_norm], dim=1)
        else:
            kv_input = hidden_norm
        
        attn_out = self._multi_head_attention(
            query=hidden_norm,
            key=kv_input,
            value=kv_input,
            causal_mask=causal_mask,
        )
        
        hidden = residual + self.dropout(attn_out)
        
        # Feed-forward
        residual = hidden
        hidden = residual + self.dropout(self.ff(self.norm2(hidden)))
        
        return hidden


class PTuningV2Model(nn.Module):
    """
    Complete P-Tuning v2 implementation from scratch.
    
    Architecture:
    ┌──────────────────────────────────────────┐
    │ Input: token_ids → embeddings            │
    │                                          │
    │ Layer 0: [prompt_0; hidden] → attention   │
    │ Layer 1: [prompt_1; hidden] → attention   │
    │ Layer 2: [prompt_2; hidden] → attention   │
    │   ...                                    │
    │ Layer N: [prompt_N; hidden] → attention   │
    │                                          │
    │ Output: hidden → logits                  │
    └──────────────────────────────────────────┘
    
    Each layer has its own learnable prompts, injected as
    extra key-value pairs in attention.
    """
    
    def __init__(
        self,
        vocab_size: int = 5000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        num_prompt_tokens: int = 8,
        max_seq_len: int = 512,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_prompt_tokens = num_prompt_tokens
        self.n_layers = n_layers
        
        # Embeddings (frozen)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers (frozen)
        self.layers = nn.ModuleList([
            DeepPromptTransformerLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)
        
        # Deep prompt pool (TRAINABLE)
        self.prompt_pool = DeepPromptPool(
            n_layers=n_layers,
            num_tokens=num_prompt_tokens,
            d_model=d_model,
        )
        
        # Freeze base model
        self._freeze_base()
        self._print_stats()
    
    def _freeze_base(self):
        for name, param in self.named_parameters():
            if "prompt_pool" not in name:
                param.requires_grad = False
    
    def _print_stats(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n  PTuningV2Model:")
        print(f"    Total: {total:,} | Trainable: {trainable:,} "
              f"({trainable/total*100:.2f}%)")
        print(f"    Prompts: {self.n_layers} layers × "
              f"{self.num_prompt_tokens} tokens × {self.d_model}D")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        B, L = input_ids.shape
        device = input_ids.device
        P = self.num_prompt_tokens
        
        # Embeddings
        positions = torch.arange(L, device=device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Get all prompts
        all_prompts = self.prompt_pool.get_all_prompts(B)
        
        # Forward through each layer with prompt injection
        for layer_idx, layer in enumerate(self.layers):
            prompt_kv = all_prompts[layer_idx].to(device)
            
            # Build causal mask for query attending to [prompt; hidden]
            # Query length: L, Key length: P + L
            causal_mask = torch.zeros(L, P + L, device=device)
            # Queries can attend to all prompts (no masking)
            # But causal masking on the hidden part
            for i in range(L):
                for j in range(P, P + L):
                    if j - P > i:
                        causal_mask[i, j] = float('-inf')
            
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, P+L]
            
            hidden = layer(hidden, prompt_kv=prompt_kv, causal_mask=causal_mask)
        
        hidden = self.final_norm(hidden)
        logits = self.output_head(hidden)
        
        result = {"logits": logits}
        
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss
        
        return result


def demonstrate_deep_prompt_model():
    """Show P-Tuning v2 model."""
    print("\n\n" + "=" * 65)
    print("  SECTION 2: P-TUNING V2 MODEL (FROM SCRATCH)")
    print("=" * 65)
    
    torch.manual_seed(42)
    
    model = PTuningV2Model(
        vocab_size=5000,
        d_model=256,
        n_heads=4,
        n_layers=4,
        num_prompt_tokens=8,
    )
    
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 5000, (batch_size, seq_len))
    labels = torch.randint(0, 5000, (batch_size, seq_len))
    
    output = model(input_ids, labels=labels)
    
    print(f"\n  Forward pass:")
    print(f"  Input: {input_ids.shape}")
    print(f"  Logits: {output['logits'].shape}")
    print(f"  Loss: {output['loss'].item():.4f}")
    
    # Verify gradients
    output["loss"].backward()
    
    print(f"\n  Gradient verification per layer:")
    for i, prompt in enumerate(model.prompt_pool.prompts):
        grad_norm = prompt.grad.norm().item() if prompt.grad is not None else 0
        print(f"  Layer {i} prompt grad norm: {grad_norm:.6f}")
    
    frozen_grads = sum(
        1 for n, p in model.named_parameters()
        if "prompt_pool" not in n and p.grad is not None and p.grad.abs().sum() > 0
    )
    print(f"\n  Frozen parameters with gradients: {frozen_grads}"
          f" {'✓' if frozen_grads == 0 else '✗'}")


# ============================================================================
# SECTION 3: SHALLOW VS DEEP COMPARISON
# ============================================================================

class ShallowPromptModel(nn.Module):
    """
    Shallow prompt model (P-Tuning v1 style, without LSTM).
    Only adds prompts at the input layer for comparison.
    """
    
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 num_prompt_tokens, max_seq_len=512):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_prompt_tokens = num_prompt_tokens
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.layers = nn.ModuleList([
            DeepPromptTransformerLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)
        
        # Only input-level prompts (shallow)
        self.input_prompts = nn.Parameter(
            torch.randn(num_prompt_tokens, d_model) * 0.02
        )
        
        # Freeze base
        for name, param in self.named_parameters():
            if "input_prompts" not in name:
                param.requires_grad = False
    
    def forward(self, input_ids, labels=None):
        B, L = input_ids.shape
        device = input_ids.device
        P = self.num_prompt_tokens
        
        positions = torch.arange(L, device=device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Prepend prompts to input
        prompts = self.input_prompts.unsqueeze(0).expand(B, -1, -1)
        hidden = torch.cat([prompts, hidden], dim=1)  # [B, P+L, D]
        
        total_len = P + L
        causal_mask = torch.triu(
            torch.ones(total_len, total_len, device=device), diagonal=1
        ).masked_fill_(
            torch.triu(torch.ones(total_len, total_len, device=device), diagonal=1) == 1,
            float('-inf')
        ).unsqueeze(0).unsqueeze(0)
        
        for layer in self.layers:
            hidden = layer(hidden, prompt_kv=None, causal_mask=causal_mask)
        
        hidden = self.final_norm(hidden)
        
        # Remove prompt positions
        hidden = hidden[:, P:, :]
        logits = self.output_head(hidden)
        
        result = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            result["loss"] = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1), ignore_index=-100,
            )
        return result


def demonstrate_shallow_vs_deep():
    """Compare shallow prompts (v1-like) vs deep prompts (v2)."""
    print("\n\n" + "=" * 65)
    print("  SECTION 3: SHALLOW vs DEEP PROMPT COMPARISON")
    print("=" * 65)
    
    torch.manual_seed(42)
    
    config = dict(
        vocab_size=3000, d_model=128, n_heads=4,
        n_layers=4, d_ff=512, num_prompt_tokens=8,
    )
    
    # Create both models
    print("\n  Creating Shallow Prompt Model:")
    shallow = ShallowPromptModel(**config)
    shallow_params = sum(p.numel() for p in shallow.parameters() if p.requires_grad)
    
    print(f"\n  Creating Deep Prompt Model:")
    deep = PTuningV2Model(
        vocab_size=config["vocab_size"], d_model=config["d_model"],
        n_heads=config["n_heads"], n_layers=config["n_layers"],
        d_ff=config["d_ff"], num_prompt_tokens=config["num_prompt_tokens"],
    )
    deep_params = sum(p.numel() for p in deep.parameters() if p.requires_grad)
    
    # Compare on random data
    B, L = 4, 32
    input_ids = torch.randint(0, config["vocab_size"], (B, L))
    labels = torch.randint(0, config["vocab_size"], (B, L))
    
    # Training simulation
    shallow_optim = torch.optim.Adam(
        [p for p in shallow.parameters() if p.requires_grad], lr=1e-3
    )
    deep_optim = torch.optim.Adam(
        [p for p in deep.parameters() if p.requires_grad], lr=1e-3
    )
    
    print(f"\n  {'Metric':<30} {'Shallow':>12} {'Deep':>12}")
    print(f"  {'─'*30}─{'─'*12}─{'─'*12}")
    print(f"  {'Trainable parameters':<30} {shallow_params:>12,} {deep_params:>12,}")
    
    shallow_losses = []
    deep_losses = []
    
    steps = 30
    for step in range(steps):
        # Shallow step
        shallow_optim.zero_grad()
        s_out = shallow(input_ids, labels)
        s_out["loss"].backward()
        shallow_optim.step()
        shallow_losses.append(s_out["loss"].item())
        
        # Deep step
        deep_optim.zero_grad()
        d_out = deep(input_ids, labels)
        d_out["loss"].backward()
        deep_optim.step()
        deep_losses.append(d_out["loss"].item())
    
    print(f"  {'Initial loss':<30} {shallow_losses[0]:>12.4f} {deep_losses[0]:>12.4f}")
    print(f"  {'Final loss':<30} {shallow_losses[-1]:>12.4f} {deep_losses[-1]:>12.4f}")
    print(f"  {'Loss reduction':<30} "
          f"{shallow_losses[0]-shallow_losses[-1]:>12.4f} "
          f"{deep_losses[0]-deep_losses[-1]:>12.4f}")
    
    print(f"""
  ═══ Deep vs Shallow — Why Deep Wins ═══
  
  Signal Propagation:
  ┌─────────────────────────────────────────────────────┐
  │ Shallow: Prompts at input → signal decays through   │
  │          many layers → weak influence on output      │
  │                                                     │
  │ Deep: Fresh prompts at EVERY layer → strong signal   │
  │       throughout the entire network                  │
  └─────────────────────────────────────────────────────┘
  
  The deeper the model, the bigger the advantage of v2:
  
  Model Size   │ Shallow   │ Deep (v2) │ Full FT
  ─────────────┼───────────┼───────────┼──────────
  ~300M (BERT) │  ~82%     │  ~88%     │  ~89%
  ~1B          │  ~80%     │  ~88%     │  ~89%
  ~10B         │  ~84%     │  ~90%     │  ~90%
  
  (SuperGLUE/NER benchmarks, approximate)
""")


# ============================================================================
# SECTION 4: GPT-2 WITH DEEP PROMPTS
# ============================================================================

class GPT2PTuningV2(nn.Module):
    """
    Apply P-Tuning v2 (deep prompts) to real GPT-2.
    
    Implementation strategy:
    - Use GPT-2's past_key_values mechanism
    - Pre-compute prompt key-value pairs for each layer
    - Inject as "cached" past context
    """
    
    def __init__(
        self,
        model_name: str = "distilgpt2",
        num_prompt_tokens: int = 10,
    ):
        super().__init__()
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        config = self.model.config
        self.n_layers = config.n_layer
        self.n_heads = config.n_head
        self.d_model = config.n_embd
        self.d_head = self.d_model // self.n_heads
        self.num_prompt_tokens = num_prompt_tokens
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Deep prompts: key-value pairs for each layer
        # past_key_values format: tuple of (key, value) for each layer
        # Each: [batch, n_heads, seq_len, d_head]
        self.prompt_keys = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.n_heads, num_prompt_tokens, self.d_head) * 0.01)
            for _ in range(self.n_layers)
        ])
        self.prompt_values = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.n_heads, num_prompt_tokens, self.d_head) * 0.01)
            for _ in range(self.n_layers)
        ])
        
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in self.parameters())
        
        print(f"\n  GPT2PTuningV2 ({model_name}):")
        print(f"    Total: {total:,} | Trainable: {trainable:,} "
              f"({trainable/total*100:.3f}%)")
        print(f"    Deep prompts: {self.n_layers} layers × "
              f"{num_prompt_tokens} tokens × {self.n_heads} heads × {self.d_head}D")
    
    def _make_past_key_values(self, batch_size: int):
        """
        Create past_key_values from learnable prompts.
        This is GPT-2's mechanism for cached context.
        """
        past = []
        for i in range(self.n_layers):
            key = self.prompt_keys[i].expand(batch_size, -1, -1, -1)
            value = self.prompt_values[i].expand(batch_size, -1, -1, -1)
            past.append((key, value))
        return tuple(past)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Create deep prompts as past_key_values
        past_kv = self._make_past_key_values(batch_size)
        
        # Extend attention mask to include prompt positions
        if attention_mask is not None:
            prompt_mask = torch.ones(
                batch_size, self.num_prompt_tokens,
                device=device, dtype=attention_mask.dtype,
            )
            full_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        else:
            full_mask = None
        
        # Forward with deep prompts
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=full_mask,
            past_key_values=past_kv,
            labels=labels,
        )
        
        return {"loss": outputs.loss, "logits": outputs.logits}


def demonstrate_gpt2_v2():
    """Demonstrate P-Tuning v2 with GPT-2."""
    print("\n\n" + "=" * 65)
    print("  SECTION 4: GPT-2 P-TUNING V2 INTEGRATION")
    print("=" * 65)
    
    model = GPT2PTuningV2(
        model_name="distilgpt2",
        num_prompt_tokens=10,
    )
    
    tokenizer = model.tokenizer
    
    text = "The theory of relativity was developed by"
    inputs = tokenizer(text, return_tensors="pt")
    
    output = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=inputs["input_ids"],
    )
    
    print(f"\n  Input: '{text}'")
    print(f"  Loss: {output['loss'].item():.4f}")
    print(f"  Logits: {output['logits'].shape}")
    
    # Verify gradients
    output["loss"].backward()
    
    print(f"\n  Gradient flow per layer (key prompts):")
    for i, pk in enumerate(model.prompt_keys):
        gn = pk.grad.norm().item() if pk.grad is not None else 0
        print(f"    Layer {i}: grad_norm = {gn:.6f}")
    
    explanation = """
  ═══ P-Tuning v2 Design Insight ═══
  
  Using past_key_values is the REAL implementation trick:
  
  1. GPT-2 natively supports past_key_values (for caching)
  2. We repurpose this as deep prompt injection
  3. Learnable (key, value) at each layer = deep prompts
  4. No model surgery needed!
  
  This is functionally equivalent to Prefix Tuning,
  but P-Tuning v2's contribution was showing this works
  UNIVERSALLY (all model sizes, all task types).
"""
    print(explanation)
    
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all P-Tuning v2 demonstrations."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     P-TUNING V2 — FROM SCRATCH IMPLEMENTATION               ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: Deep prompts
    demonstrate_deep_prompts()
    
    # Section 2: Full model
    demonstrate_deep_prompt_model()
    
    # Section 3: Shallow vs deep
    demonstrate_shallow_vs_deep()
    
    # Section 4: GPT-2
    demonstrate_gpt2_v2()
    
    print("\n" + "=" * 65)
    print("  MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered:
    ✓ Deep Prompt Pool (independent prompts per layer)
    ✓ Transformer with prompt injection (custom attention)
    ✓ Full P-Tuning v2 from scratch
    ✓ Shallow vs Deep comparison (signal propagation)
    ✓ GPT-2 integration via past_key_values
    """)


if __name__ == "__main__":
    main()
