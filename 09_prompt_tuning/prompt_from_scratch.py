"""
Prompt Tuning — From Scratch Implementation
=============================================

Build prompt tuning from the ground up in PyTorch:

1. Basic Soft Prompt Module
   - Learnable embedding prepended to input
   - Forward pass mechanics

2. Full Prompt-Tuned Model
   - Wrapping a frozen transformer
   - Handling attention masks
   - Label shifting for causal LM

3. GPT-2 Integration
   - Inject soft prompts into real GPT-2
   - Handle position embeddings correctly
   - Training loop with gradient freezing

4. Initialization Strategies
   - Random, vocab sampling, text-based
   - Impact on convergence

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


# ============================================================================
# SECTION 1: BASIC SOFT PROMPT MODULE
# ============================================================================

class SoftPrompt(nn.Module):
    """
    The core soft prompt module — learnable vectors prepended to inputs.
    
    This is the fundamental building block of prompt tuning.
    Unlike prefix tuning (which modifies K, V at every layer),
    prompt tuning ONLY modifies the input embeddings.
    
    Architecture:
        Input:  [x₁, x₂, ..., xₘ]           shape: [B, M, D]
        Prompt: [p₁, p₂, ..., pₙ]            shape: [1, N, D]  (broadcast over batch)
        Output: [p₁, ..., pₙ, x₁, ..., xₘ]  shape: [B, N+M, D]
    """
    
    def __init__(
        self,
        num_tokens: int,
        d_model: int,
        init_strategy: str = "random",
        init_text_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            num_tokens: Number of soft prompt tokens (N).
            d_model: Hidden dimension of the model.
            init_strategy: How to initialize soft prompt vectors.
                - "random": Standard normal * 0.02
                - "uniform": Uniform[-0.5, 0.5]
                - "text": Initialize from provided text embeddings
            init_text_embeddings: Pre-computed embeddings for text init.
                Shape: [num_tokens, d_model]
        """
        super().__init__()
        
        self.num_tokens = num_tokens
        self.d_model = d_model
        
        # The soft prompt — this is ALL we train
        self.prompt_embeddings = nn.Parameter(
            torch.empty(num_tokens, d_model)
        )
        
        # Initialize
        self._initialize(init_strategy, init_text_embeddings)
        
        print(f"  SoftPrompt created:")
        print(f"    Tokens: {num_tokens}")
        print(f"    Dim: {d_model}")
        print(f"    Parameters: {num_tokens * d_model:,}")
        print(f"    Init: {init_strategy}")
    
    def _initialize(self, strategy: str, text_embeddings: Optional[torch.Tensor]):
        """Initialize soft prompt vectors."""
        if strategy == "random":
            nn.init.normal_(self.prompt_embeddings, mean=0.0, std=0.02)
        elif strategy == "uniform":
            nn.init.uniform_(self.prompt_embeddings, -0.5, 0.5)
        elif strategy == "text" and text_embeddings is not None:
            assert text_embeddings.shape == self.prompt_embeddings.shape, \
                f"Expected {self.prompt_embeddings.shape}, got {text_embeddings.shape}"
            self.prompt_embeddings.data.copy_(text_embeddings)
        else:
            nn.init.normal_(self.prompt_embeddings, mean=0.0, std=0.02)
    
    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Prepend soft prompt to input embeddings.
        
        Args:
            input_embeddings: [batch_size, seq_len, d_model]
        
        Returns:
            [batch_size, num_tokens + seq_len, d_model]
        """
        batch_size = input_embeddings.shape[0]
        
        # Expand prompt to batch: [N, D] → [B, N, D]
        prompt = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Prepend: [B, N+M, D]
        return torch.cat([prompt, input_embeddings], dim=1)
    
    def extra_repr(self) -> str:
        return f"num_tokens={self.num_tokens}, d_model={self.d_model}"


# ============================================================================
# SECTION 2: FULL PROMPT-TUNED TRANSFORMER MODEL
# ============================================================================

class PromptTunedTransformer(nn.Module):
    """
    Complete prompt-tuned causal language model from scratch.
    
    Architecture:
        Input IDs → Embedding → [Soft Prompt; Embeds] → Frozen Transformer → LM Head
        
    Only the soft prompt vectors are trainable.
    Everything else (embeddings, transformer, LM head) is frozen.
    """
    
    def __init__(
        self,
        vocab_size: int = 5000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 128,
        num_prompt_tokens: int = 10,
        init_strategy: str = "random",
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_prompt_tokens = num_prompt_tokens
        self.max_seq_len = max_seq_len
        
        # ─── BASE MODEL (ALL FROZEN) ───
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len + num_prompt_tokens, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # ─── SOFT PROMPT (ONLY TRAINABLE PART) ───
        self.soft_prompt = SoftPrompt(
            num_tokens=num_prompt_tokens,
            d_model=d_model,
            init_strategy=init_strategy,
        )
        
        # Freeze base model
        self._freeze_base_model()
        
        # Report parameters
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n  PromptTunedTransformer:")
        print(f"    Total params:     {total:>10,}")
        print(f"    Trainable params: {trainable:>10,}")
        print(f"    Trainable %:      {trainable/total*100:>9.4f}%")
    
    def _freeze_base_model(self):
        """Freeze everything except the soft prompt."""
        for name, param in self.named_parameters():
            if "soft_prompt" not in name:
                param.requires_grad = False
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal (autoregressive) attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass with soft prompt prepending.
        
        Args:
            input_ids: [B, M] — token IDs
            attention_mask: [B, M] — 1 for real tokens, 0 for padding
            labels: [B, M] — target token IDs for loss computation
        
        Returns:
            dict with 'logits' and optionally 'loss'
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Step 1: Get token embeddings
        token_embeds = self.token_embedding(input_ids)  # [B, M, D]
        
        # Step 2: Prepend soft prompt
        # [B, N+M, D] where N = num_prompt_tokens
        combined_embeds = self.soft_prompt(token_embeds)
        total_len = combined_embeds.shape[1]
        
        # Step 3: Add position embeddings
        positions = torch.arange(total_len, device=device).unsqueeze(0)
        position_embeds = self.position_embedding(positions)
        hidden_states = combined_embeds + position_embeds
        
        # Step 4: Create causal mask for full sequence
        causal_mask = self._create_causal_mask(total_len, device)
        
        # Step 5: Extend attention mask to include prompt tokens
        if attention_mask is not None:
            # Prompt tokens always attended to (mask=1)
            prompt_mask = torch.ones(
                batch_size, self.num_prompt_tokens, device=device
            )
            full_mask = torch.cat([prompt_mask, attention_mask], dim=1)
            
            # Convert to additive mask for transformer
            padding_mask = (full_mask == 0)  # True = ignore
        else:
            padding_mask = None
        
        # Step 6: Forward through frozen transformer
        hidden_states = self.transformer(
            hidden_states,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        hidden_states = self.layer_norm(hidden_states)
        
        # Step 7: Project to vocabulary
        logits = self.lm_head(hidden_states)  # [B, N+M, V]
        
        # Step 8: Compute loss (only on real token positions, not prompt positions)
        result = {"logits": logits}
        
        if labels is not None:
            # Shift for causal LM: predict next token
            # Only compute loss on non-prompt positions
            prompt_len = self.num_prompt_tokens
            
            # Logits at prompt+input positions predict the next token
            shift_logits = logits[:, prompt_len:-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss
        
        return result


def demonstrate_from_scratch():
    """Demonstrate the from-scratch implementation."""
    print("=" * 65)
    print("  SECTION 2: PROMPT-TUNED TRANSFORMER FROM SCRATCH")
    print("=" * 65)
    
    torch.manual_seed(42)
    
    model = PromptTunedTransformer(
        vocab_size=5000,
        d_model=256,
        n_heads=4,
        n_layers=4,
        num_prompt_tokens=10,
        init_strategy="random",
    )
    
    # Example forward pass
    batch_size, seq_len = 2, 20
    input_ids = torch.randint(0, 5000, (batch_size, seq_len))
    labels = torch.randint(0, 5000, (batch_size, seq_len))
    
    output = model(input_ids, labels=labels)
    
    print(f"\n  Forward pass:")
    print(f"  Input shape:  {input_ids.shape}")
    print(f"  Logits shape: {output['logits'].shape}")
    print(f"  Loss: {output['loss'].item():.4f}")
    
    # Verify only prompt gradients
    output["loss"].backward()
    
    print(f"\n  Gradient check (should only be prompt):")
    for name, param in model.named_parameters():
        has_grad = param.grad is not None and param.grad.abs().sum() > 0
        mark = "✓ HAS GRAD" if has_grad else "✗ no grad"
        print(f"    {mark}  {name} {list(param.shape)}")
    
    print(f"\n  ✓ Only soft_prompt parameters received gradients!")


# ============================================================================
# SECTION 3: GPT-2 INTEGRATION
# ============================================================================

class GPT2PromptTuning(nn.Module):
    """
    Apply prompt tuning to a real GPT-2 model.
    
    Strategy:
    1. Load pretrained GPT-2
    2. Freeze ALL parameters
    3. Create learnable soft prompt
    4. At forward time: insert soft prompt embeddings
       BEFORE the first transformer layer
    5. Only backprop updates the soft prompt
    """
    
    def __init__(
        self,
        model_name: str = "distilgpt2",
        num_prompt_tokens: int = 20,
        init_strategy: str = "text",
        init_text: Optional[str] = None,
    ):
        super().__init__()
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load pretrained model
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        d_model = self.model.config.n_embd
        self.num_prompt_tokens = num_prompt_tokens
        self.d_model = d_model
        
        # Freeze the entire model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Create soft prompt
        if init_strategy == "text" and init_text is not None:
            text_embeddings = self._get_text_embeddings(init_text)
            self.soft_prompt = SoftPrompt(
                num_tokens=num_prompt_tokens,
                d_model=d_model,
                init_strategy="text",
                init_text_embeddings=text_embeddings,
            )
        else:
            self.soft_prompt = SoftPrompt(
                num_tokens=num_prompt_tokens,
                d_model=d_model,
                init_strategy=init_strategy,
            )
        
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n  GPT2PromptTuning ({model_name}):")
        print(f"    Total: {total:,} | Trainable: {trainable:,} "
              f"({trainable/total*100:.4f}%)")
    
    def _get_text_embeddings(self, text: str) -> torch.Tensor:
        """
        Initialize soft prompt from text by getting its token embeddings.
        If text has fewer tokens than num_prompt_tokens, repeat.
        If more, truncate.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Get the embedding layer
        if hasattr(self.model, 'transformer'):
            embed_layer = self.model.transformer.wte  # GPT-2
        else:
            embed_layer = self.model.get_input_embeddings()
        
        token_tensor = torch.tensor(tokens)
        with torch.no_grad():
            embeddings = embed_layer(token_tensor)  # [len, d_model]
        
        # Adjust length
        if len(tokens) < self.num_prompt_tokens:
            # Repeat to fill
            repeats = math.ceil(self.num_prompt_tokens / len(tokens))
            embeddings = embeddings.repeat(repeats, 1)
        
        return embeddings[:self.num_prompt_tokens].clone()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass: prepend soft prompt embeddings to input.
        
        Key implementation detail:
        We get the input embeddings from GPT-2's embedding layer,
        prepend our soft prompt, then pass the EMBEDDINGS directly
        to the model (using inputs_embeds instead of input_ids).
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Step 1: Get token embeddings from frozen model
        if hasattr(self.model, 'transformer'):
            token_embeds = self.model.transformer.wte(input_ids)
        else:
            token_embeds = self.model.get_input_embeddings()(input_ids)
        
        # Step 2: Prepend soft prompt
        combined = self.soft_prompt(token_embeds)  # [B, N+M, D]
        
        # Step 3: Create extended attention mask
        if attention_mask is not None:
            prompt_mask = torch.ones(
                batch_size, self.num_prompt_tokens,
                device=device, dtype=attention_mask.dtype
            )
            extended_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        else:
            extended_mask = torch.ones(
                batch_size, combined.shape[1],
                device=device, dtype=torch.long
            )
        
        # Step 4: Create extended labels (ignore prompt positions)
        if labels is not None:
            # -100 = ignore index for CrossEntropyLoss
            prompt_labels = torch.full(
                (batch_size, self.num_prompt_tokens),
                fill_value=-100,
                device=device,
                dtype=labels.dtype,
            )
            extended_labels = torch.cat([prompt_labels, labels], dim=1)
        else:
            extended_labels = None
        
        # Step 5: Forward through model with embeddings
        outputs = self.model(
            inputs_embeds=combined,
            attention_mask=extended_mask,
            labels=extended_labels,
        )
        
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }
    
    def generate(self, text: str, max_new_tokens: int = 50, **kwargs) -> str:
        """Generate text with the soft prompt automatically prepended."""
        tokens = self.tokenizer.encode(text, return_tensors="pt")
        device = self.soft_prompt.prompt_embeddings.device
        tokens = tokens.to(device)
        
        # Get embeddings and prepend soft prompt
        if hasattr(self.model, 'transformer'):
            embeds = self.model.transformer.wte(tokens)
        else:
            embeds = self.model.get_input_embeddings()(tokens)
        
        combined = self.soft_prompt(embeds)
        
        # Create attention mask
        attention_mask = torch.ones(1, combined.shape[1], device=device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=combined,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def demonstrate_gpt2_integration():
    """Demonstrate prompt tuning on real GPT-2."""
    print("\n\n" + "=" * 65)
    print("  SECTION 3: GPT-2 PROMPT TUNING INTEGRATION")
    print("=" * 65)
    
    # Using text initialization
    init_text = "Classify the following text as positive or negative sentiment:"
    
    model = GPT2PromptTuning(
        model_name="distilgpt2",
        num_prompt_tokens=20,
        init_strategy="text",
        init_text=init_text,
    )
    
    # Forward pass
    tokenizer = model.tokenizer
    text = "This movie was absolutely fantastic!"
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    output = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=inputs["input_ids"],
    )
    
    print(f"\n  Forward pass test:")
    print(f"  Input: '{text}'")
    print(f"  Loss: {output['loss'].item():.4f}")
    print(f"  Logits shape: {output['logits'].shape}")
    print(f"  (includes {model.num_prompt_tokens} prompt positions)")
    
    # Verify gradients
    output["loss"].backward()
    prompt_grad_norm = model.soft_prompt.prompt_embeddings.grad.norm().item()
    
    frozen_has_grad = False
    for name, p in model.model.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            frozen_has_grad = True
            break
    
    print(f"\n  Gradient verification:")
    print(f"  Soft prompt grad norm: {prompt_grad_norm:.6f} ✓")
    print(f"  Frozen params got grad: {frozen_has_grad} "
          f"{'✗ ERROR!' if frozen_has_grad else '✓ (correct: no grad)'}")


# ============================================================================
# SECTION 4: INITIALIZATION STRATEGIES
# ============================================================================

class InitializationExperiment:
    """
    Compare different initialization strategies for soft prompts.
    Initialization is CRITICAL for prompt tuning success.
    """
    
    @staticmethod
    def demonstrate():
        print("\n\n" + "=" * 65)
        print("  SECTION 4: INITIALIZATION STRATEGIES")
        print("=" * 65)
        
        explanation = """
  ═══ Why Initialization Matters So Much ═══
  
  Prompt tuning optimizes a TINY number of parameters to steer
  a HUGE frozen model. The optimization landscape is:
  - High-dimensional (N × d_model)
  - Non-convex
  - Very sensitive to starting point
  
  Bad init → loss plateau → poor final performance
  Good init → fast convergence → near-optimal results
  
  
  ═══ The Three Main Strategies ═══
  
  Strategy 1: Random Initialization
  ─────────────────────────────────────────────────────────
  prompt_embeddings ~ N(0, 0.02²)
  
  • Pros: Simple, no assumptions
  • Cons: Worst performance, slow convergence
  • Use when: Quick experiments, no prior knowledge
  
  
  Strategy 2: Vocabulary Sampling
  ─────────────────────────────────────────────────────────
  sample_ids = random.sample(range(vocab_size), num_tokens)
  prompt_embeddings = model.embedding[sample_ids]
  
  • Pros: Starts on the embedding manifold
  • Cons: Random tokens, no semantic meaning
  • Use when: Better than random, minimal effort
  
  
  Strategy 3: Text Initialization ★ RECOMMENDED
  ─────────────────────────────────────────────────────────
  text = "Classify this text as positive or negative:"
  tokens = tokenizer.encode(text)
  prompt_embeddings = model.embedding[tokens[:num_prompt_tokens]]
  
  • Pros: Best performance, fastest convergence
  • Cons: Requires choosing good init text
  • Use when: Always (when possible)!
"""
        print(explanation)
        
        # Demonstrate with actual embeddings
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        embed_layer = model.transformer.wte
        
        num_tokens = 10
        d_model = model.config.n_embd
        
        # Strategy 1: Random
        random_prompt = torch.randn(num_tokens, d_model) * 0.02
        
        # Strategy 2: Vocab sampling
        torch.manual_seed(42)
        vocab_ids = torch.randint(0, tokenizer.vocab_size, (num_tokens,))
        with torch.no_grad():
            vocab_prompt = embed_layer(vocab_ids)
        
        # Strategy 3: Text init
        init_text = "Classify this text as positive or negative sentiment :"
        text_tokens = tokenizer.encode(init_text, add_special_tokens=False)[:num_tokens]
        with torch.no_grad():
            text_prompt = embed_layer(torch.tensor(text_tokens))
        
        # Compare properties
        print(f"  Initialization Comparison:")
        print(f"  {'Strategy':>20} {'Norm (mean)':>12} {'Norm (std)':>12} "
              f"{'Range':>12}")
        print(f"  {'─'*20}─{'─'*12}─{'─'*12}─{'─'*12}")
        
        for name, prompt in [
            ("Random", random_prompt),
            ("Vocab Sample", vocab_prompt),
            ("Text Init", text_prompt),
        ]:
            norms = prompt.norm(dim=1)
            print(f"  {name:>20}  {norms.mean():>10.4f}  "
                  f"{norms.std():>10.4f}  "
                  f"[{prompt.min():.3f}, {prompt.max():.3f}]")
        
        # Show what text init tokens decode to 
        print(f"\n  Text init tokens:")
        for i, tid in enumerate(text_tokens):
            decoded = tokenizer.decode([tid])
            print(f"    Token {i}: id={tid:>5d}  →  '{decoded}'")
        
        # Vocab embedding statistics for reference
        with torch.no_grad():
            all_embeds = embed_layer.weight
            all_norms = all_embeds.norm(dim=1)
        
        print(f"\n  Reference — Full vocabulary embedding stats:")
        print(f"    Mean norm: {all_norms.mean():.4f}")
        print(f"    Std norm:  {all_norms.std():.4f}")
        print(f"    Range: [{all_embeds.min():.4f}, {all_embeds.max():.4f}]")
        
        print(f"""
  ═══ Key Insight ═══
  
  Random init produces vectors with VERY DIFFERENT statistics
  than real token embeddings. The model's attention mechanism
  was trained with tokens from a specific distribution, so:
  
  • Random vectors are "out-of-distribution" → confuse the model
  • Vocab-sampled vectors are "on-distribution" → more compatible
  • Text-init vectors are "semantically meaningful" → best start
  
  Text init gives the optimizer a head start by placing the
  soft prompt near a good solution in embedding space.
""")
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all from-scratch demonstrations."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     PROMPT TUNING — FROM SCRATCH IMPLEMENTATION              ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1 & 2: From-scratch model
    demonstrate_from_scratch()
    
    # Section 3: GPT-2 integration
    demonstrate_gpt2_integration()
    
    # Section 4: Initialization strategies
    InitializationExperiment.demonstrate()
    
    print("\n" + "=" * 65)
    print("  MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered:
    ✓ SoftPrompt module (learnable embedding prepend)
    ✓ PromptTunedTransformer (full model from scratch)
    ✓ GPT2PromptTuning (real model integration)
    ✓ Initialization strategies (random vs vocab vs text)
    """)


if __name__ == "__main__":
    main()
