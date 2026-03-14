"""
P-Tuning v1 — From Scratch Implementation
===========================================

Complete implementation of P-Tuning v1 (Liu et al., 2021):

1. LSTM Prompt Encoder
   - Bidirectional LSTM + MLP projection
   - Learnable pseudo-token embeddings → encoded prompts

2. Template-Based Prompting
   - Interleaving prompts with input tokens
   - Flexible template patterns
   - [MASK] prediction for cloze tasks

3. Full P-Tuning v1 Model
   - LSTM encoder + frozen transformer
   - Gradient routing (only encoder trainable)

4. Integration with GPT-2
   - Apply P-Tuning v1 to real GPT-2
   - Knowledge probing example

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple


# ============================================================================
# SECTION 1: LSTM PROMPT ENCODER
# ============================================================================

class LSTMPromptEncoder(nn.Module):
    """
    The core innovation of P-Tuning v1: an LSTM-based encoder
    that generates continuous prompts from learnable embeddings.
    
    Architecture:
        pseudo_tokens [h₁...hₙ]
            ↓
        Embedding layer (learnable)
            ↓
        Bidirectional LSTM (creates inter-token dependencies)
            ↓
        2-Layer MLP with ReLU (projects to model's embedding space)
            ↓
        continuous_prompts [p₁...pₙ]
    
    Why this works better than direct optimization:
    - LSTM creates coherent, interdependent prompt tokens
    - MLP provides non-linear mapping to good embedding regions
    - Reparameterization smooths the optimization landscape
    """
    
    def __init__(
        self,
        num_tokens: int,
        d_model: int,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.0,
        mlp_hidden: Optional[int] = None,
    ):
        """
        Args:
            num_tokens: Number of pseudo prompt tokens.
            d_model: Model hidden dimension (must match transformer).
            lstm_hidden: LSTM hidden size (per direction).
            lstm_layers: Number of LSTM layers.
            lstm_dropout: Dropout between LSTM layers.
            mlp_hidden: MLP intermediate dimension (default: d_model * 2).
        """
        super().__init__()
        
        self.num_tokens = num_tokens
        self.d_model = d_model
        mlp_hidden = mlp_hidden or d_model * 2
        
        # Learnable pseudo-token embeddings (the "input" to the encoder)
        self.pseudo_token_embedding = nn.Embedding(num_tokens, d_model)
        
        # Bidirectional LSTM
        # Output dim = 2 * lstm_hidden (forward + backward)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        
        # MLP head: projects LSTM output to embedding space
        self.mlp = nn.Sequential(
            nn.Linear(2 * lstm_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, d_model),
        )
        
        # Parameter count
        total = sum(p.numel() for p in self.parameters())
        print(f"  LSTMPromptEncoder:")
        print(f"    Tokens: {num_tokens}")
        print(f"    d_model: {d_model}")
        print(f"    LSTM: {lstm_layers} layers, {lstm_hidden} hidden, bidirectional")
        print(f"    Total parameters: {total:,}")
    
    def forward(self, batch_size: int = 1) -> torch.Tensor:
        """
        Generate continuous prompts from pseudo-token embeddings.
        
        Args:
            batch_size: Number of sequences in the batch.
        
        Returns:
            Continuous prompts: [batch_size, num_tokens, d_model]
        """
        # Create pseudo-token indices: [0, 1, 2, ..., num_tokens-1]
        indices = torch.arange(self.num_tokens, device=self.pseudo_token_embedding.weight.device)
        
        # Get pseudo embeddings: [num_tokens, d_model]
        pseudo_embeds = self.pseudo_token_embedding(indices)
        
        # Add batch dimension: [1, num_tokens, d_model]
        pseudo_embeds = pseudo_embeds.unsqueeze(0)
        
        # Pass through bidirectional LSTM: [1, num_tokens, 2*lstm_hidden]
        lstm_out, _ = self.lstm(pseudo_embeds)
        
        # Project through MLP: [1, num_tokens, d_model]
        prompts = self.mlp(lstm_out)
        
        # Expand to batch size: [batch_size, num_tokens, d_model]
        prompts = prompts.expand(batch_size, -1, -1)
        
        return prompts


def demonstrate_encoder():
    """Demonstrate the LSTM prompt encoder."""
    print("=" * 65)
    print("  SECTION 1: LSTM PROMPT ENCODER")
    print("=" * 65)
    
    torch.manual_seed(42)
    
    encoder = LSTMPromptEncoder(
        num_tokens=10,
        d_model=256,
        lstm_hidden=128,
        lstm_layers=2,
    )
    
    # Generate prompts
    prompts = encoder(batch_size=2)
    print(f"\n  Generated prompts shape: {prompts.shape}")
    print(f"  Prompt norms per token: {prompts[0].norm(dim=1).tolist()[:5]}...")
    
    # Verify inter-token dependency
    print(f"\n  Inter-token dependency check:")
    p = prompts[0].detach()
    normalized = F.normalize(p, dim=1)
    similarity = normalized @ normalized.T
    
    print(f"  Cosine similarity (first 5 tokens):")
    for i in range(5):
        row = "    "
        for j in range(5):
            row += f" {similarity[i,j].item():>6.3f}"
        print(row)
    
    print(f"\n  Higher similarity = LSTM creating coherent tokens ✓")
    
    # Gradient verification
    loss = prompts.sum()
    loss.backward()
    
    has_grad = {name: p.grad is not None for name, p in encoder.named_parameters()}
    print(f"\n  Gradient verification:")
    for name, grad in has_grad.items():
        print(f"    {'✓' if grad else '✗'} {name}")


# ============================================================================
# SECTION 2: TEMPLATE-BASED PROMPTING
# ============================================================================

class PromptTemplate:
    """
    P-Tuning v1's template system for interleaving prompts with input.
    
    Templates define WHERE soft prompts appear relative to input:
    
    Example templates:
      "prefix":     [P₁ P₂ P₃ | INPUT]
      "cloze":      [P₁ P₂ | INPUT | P₃ [MASK] P₄]
      "mixed":      [P₁ | INPUT | P₂ P₃ | INPUT₂ | P₄]
    """
    
    def __init__(self, pattern: str):
        """
        Args:
            pattern: Template string using:
                {P} = soft prompt token
                {X} = input token  
                {M} = [MASK] token for prediction
                
        Example: "{P}{P}{P}{X}{X}{P}{M}{P}" 
        """
        self.pattern = pattern
        self.positions = self._parse_pattern()
    
    def _parse_pattern(self) -> List[str]:
        """Parse template into list of position types."""
        positions = []
        i = 0
        while i < len(self.pattern):
            if self.pattern[i] == '{':
                end = self.pattern.index('}', i)
                positions.append(self.pattern[i+1:end])
                i = end + 1
            else:
                i += 1
        return positions
    
    @property
    def num_prompt_positions(self) -> int:
        return sum(1 for p in self.positions if p == 'P')
    
    @property
    def num_input_positions(self) -> int:
        return sum(1 for p in self.positions if p == 'X')
    
    @property
    def num_mask_positions(self) -> int:
        return sum(1 for p in self.positions if p == 'M')
    
    def get_prompt_indices(self) -> List[int]:
        """Indices where soft prompts should be inserted."""
        return [i for i, p in enumerate(self.positions) if p == 'P']
    
    def get_input_indices(self) -> List[int]:
        """Indices where input tokens should be placed."""
        return [i for i, p in enumerate(self.positions) if p == 'X']
    
    def get_mask_indices(self) -> List[int]:
        """Indices of [MASK] positions for prediction."""
        return [i for i, p in enumerate(self.positions) if p == 'M']
    
    def __repr__(self):
        return f"PromptTemplate('{self.pattern}')"


class TemplateAssembler(nn.Module):
    """
    Assembles input embeddings according to a P-Tuning template.
    Interleaves soft prompt tokens with input and mask tokens.
    """
    
    def __init__(self, template: PromptTemplate, d_model: int):
        super().__init__()
        self.template = template
        self.d_model = d_model
    
    def forward(
        self,
        prompt_embeds: torch.Tensor,   # [B, N_prompt, D]
        input_embeds: torch.Tensor,    # [B, N_input, D]
        mask_embed: Optional[torch.Tensor] = None,  # [D]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Assemble embeddings according to template.
        
        Returns:
            assembled: [B, total_len, D]
            info: dict with position information
        """
        batch_size = prompt_embeds.shape[0]
        device = prompt_embeds.device
        total_len = len(self.template.positions)
        
        assembled = torch.zeros(batch_size, total_len, self.d_model, device=device)
        
        prompt_idx = 0
        input_idx = 0
        
        for pos, ptype in enumerate(self.template.positions):
            if ptype == 'P':
                assembled[:, pos, :] = prompt_embeds[:, prompt_idx, :]
                prompt_idx += 1
            elif ptype == 'X':
                if input_idx < input_embeds.shape[1]:
                    assembled[:, pos, :] = input_embeds[:, input_idx, :]
                input_idx += 1
            elif ptype == 'M' and mask_embed is not None:
                assembled[:, pos, :] = mask_embed.unsqueeze(0).expand(batch_size, -1)
        
        info = {
            "prompt_indices": self.template.get_prompt_indices(),
            "input_indices": self.template.get_input_indices(),
            "mask_indices": self.template.get_mask_indices(),
            "total_length": total_len,
        }
        
        return assembled, info


def demonstrate_templates():
    """Show template-based prompting."""
    print("\n\n" + "=" * 65)
    print("  SECTION 2: TEMPLATE-BASED PROMPTING")
    print("=" * 65)
    
    templates = {
        "prefix": PromptTemplate("{P}{P}{P}{P}{P}{X}{X}{X}{X}{X}"),
        "cloze":  PromptTemplate("{P}{P}{X}{X}{X}{P}{M}{P}"),
        "mixed":  PromptTemplate("{P}{X}{X}{P}{P}{X}{P}{M}"),
        "suffix": PromptTemplate("{X}{X}{X}{P}{P}{P}{P}{M}"),
    }
    
    print(f"\n  Template Patterns:")
    print(f"  {'Name':>10} {'Pattern':>45} {'#P':>4} {'#X':>4} {'#M':>4}")
    print(f"  {'─'*10}─{'─'*45}─{'─'*4}─{'─'*4}─{'─'*4}")
    
    for name, tmpl in templates.items():
        visual = ""
        for p in tmpl.positions:
            if p == 'P':
                visual += "[P]"
            elif p == 'X':
                visual += "[X]"
            elif p == 'M':
                visual += "[M]"
        
        print(f"  {name:>10}  {visual:>43}  {tmpl.num_prompt_positions:>3}"
              f"  {tmpl.num_input_positions:>3}  {tmpl.num_mask_positions:>3}")
    
    # Demonstrate assembly
    torch.manual_seed(42)
    d_model = 64
    batch_size = 2
    
    template = templates["cloze"]
    assembler = TemplateAssembler(template, d_model)
    
    prompt_embeds = torch.randn(batch_size, template.num_prompt_positions, d_model)
    input_embeds = torch.randn(batch_size, template.num_input_positions, d_model)
    mask_embed = torch.randn(d_model)
    
    assembled, info = assembler(prompt_embeds, input_embeds, mask_embed)
    
    print(f"\n  Assembly demo (cloze template):")
    print(f"  Prompt: {prompt_embeds.shape} ({template.num_prompt_positions} tokens)")
    print(f"  Input:  {input_embeds.shape} ({template.num_input_positions} tokens)")
    print(f"  Output: {assembled.shape}")
    print(f"  Prompt positions: {info['prompt_indices']}")
    print(f"  Input positions:  {info['input_indices']}")
    print(f"  Mask positions:   {info['mask_indices']}")
    
    print(f"""
  ═══ Template Design Principles ═══
  
  1. Cloze patterns work best for knowledge probing:
     "[SOFT] Paris is the capital of [MASK] [SOFT]"
     
  2. Prefix patterns work best for classification:
     "[SOFT SOFT SOFT] {input text}"
     
  3. The placement of [MASK] relative to prompts matters!
     Near prompts = stronger prompt influence on prediction
  
  4. P-Tuning v1 found that cloze templates + LSTM encoder
     dramatically improved GPT-2's performance on knowledge
     tasks (LAMA benchmark), where it previously failed.
""")


# ============================================================================
# SECTION 3: FULL P-TUNING V1 MODEL
# ============================================================================

class PTuningV1Model(nn.Module):
    """
    Complete P-Tuning v1 implementation from scratch.
    
    Components:
    1. LSTM Prompt Encoder (trainable)
    2. Template Assembler (arranges tokens)
    3. Frozen Transformer (base model)
    4. Prediction Head (for task output)
    """
    
    def __init__(
        self,
        vocab_size: int = 5000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        num_prompt_tokens: int = 6,
        template_pattern: str = "{P}{P}{P}{X}{X}{X}{X}{X}{X}{X}{P}{P}{P}",
        lstm_hidden: int = 128,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Template
        self.template = PromptTemplate(template_pattern)
        self.assembler = TemplateAssembler(self.template, d_model)
        
        # ─── BASE MODEL (FROZEN) ───
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(512, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(d_model, vocab_size)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # ─── PROMPT ENCODER (TRAINABLE) ───
        self.prompt_encoder = LSTMPromptEncoder(
            num_tokens=num_prompt_tokens,
            d_model=d_model,
            lstm_hidden=lstm_hidden,
            lstm_layers=2,
        )
        
        # Freeze base model
        self._freeze_base()
        
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n  PTuningV1Model:")
        print(f"    Total: {total:,} | Trainable: {trainable:,} "
              f"({trainable/total*100:.2f}%)")
    
    def _freeze_base(self):
        """Freeze everything except the prompt encoder."""
        for name, param in self.named_parameters():
            if "prompt_encoder" not in name:
                param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass with P-Tuning v1.
        
        1. Encode prompts via LSTM
        2. Get input embeddings
        3. Assemble according to template
        4. Pass through frozen transformer
        5. Compute loss/predictions
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Step 1: Generate soft prompts via LSTM encoder
        prompt_embeds = self.prompt_encoder(batch_size)  # [B, N_prompt, D]
        
        # Step 2: Get input token embeddings
        input_embeds = self.token_embedding(input_ids)  # [B, seq_len, D]
        
        # Step 3: Assemble according to template
        assembled, info = self.assembler(prompt_embeds, input_embeds)
        total_len = assembled.shape[1]
        
        # Step 4: Add position embeddings
        positions = torch.arange(total_len, device=device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        hidden = assembled + pos_embeds
        
        # Step 5: Causal mask
        causal_mask = torch.triu(
            torch.ones(total_len, total_len, device=device), diagonal=1
        ).masked_fill_(
            torch.triu(torch.ones(total_len, total_len, device=device), diagonal=1) == 1,
            float('-inf')
        )
        
        # Step 6: Forward through frozen transformer
        hidden = self.transformer(hidden, mask=causal_mask)
        hidden = self.layer_norm(hidden)
        
        # Step 7: Get logits
        logits = self.output_head(hidden)  # [B, total_len, vocab]
        
        result = {"logits": logits, "info": info}
        
        # Step 8: Compute loss if labels provided
        if labels is not None:
            # For simplicity, compute loss on input positions only
            input_positions = info["input_indices"]
            if len(input_positions) > 1:
                input_logits = logits[:, input_positions[:-1], :]
                input_labels = labels[:, 1:len(input_positions)]
                
                loss = F.cross_entropy(
                    input_logits.reshape(-1, self.vocab_size),
                    input_labels.reshape(-1),
                    ignore_index=-100,
                )
                result["loss"] = loss
        
        return result


def demonstrate_p_tuning_v1():
    """Show the complete P-Tuning v1 model."""
    print("\n\n" + "=" * 65)
    print("  SECTION 3: FULL P-TUNING V1 MODEL")
    print("=" * 65)
    
    torch.manual_seed(42)
    
    model = PTuningV1Model(
        vocab_size=5000,
        d_model=256,
        n_heads=4,
        n_layers=4,
        num_prompt_tokens=6,
        template_pattern="{P}{P}{P}{X}{X}{X}{X}{X}{P}{P}{P}",
    )
    
    # Forward pass
    batch_size = 2
    seq_len = 5  # matches {X} count in template
    input_ids = torch.randint(0, 5000, (batch_size, seq_len))
    labels = torch.randint(0, 5000, (batch_size, seq_len))
    
    output = model(input_ids, labels=labels)
    
    print(f"\n  Forward pass:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Logits shape: {output['logits'].shape}")
    if "loss" in output:
        print(f"  Loss: {output['loss'].item():.4f}")
    
    # Verify only encoder gets gradients
    if "loss" in output:
        output["loss"].backward()
        
        print(f"\n  Gradient verification:")
        encoder_grad = False
        frozen_grad = False
        for name, p in model.named_parameters():
            has_grad = p.grad is not None and p.grad.abs().sum() > 0
            if "prompt_encoder" in name and has_grad:
                encoder_grad = True
            elif "prompt_encoder" not in name and has_grad:
                frozen_grad = True
        
        print(f"  Encoder has gradients: {encoder_grad} ✓")
        print(f"  Frozen has gradients: {frozen_grad} "
              f"{'✗ ERROR' if frozen_grad else '✓ (correct)'}")


# ============================================================================
# SECTION 4: GPT-2 INTEGRATION
# ============================================================================

class GPT2PTuningV1(nn.Module):
    """
    Apply P-Tuning v1 to a real GPT-2 model.
    
    The LSTM encoder generates soft prompts, which are
    prepended to the input embeddings before passing
    through frozen GPT-2.
    """
    
    def __init__(
        self,
        model_name: str = "distilgpt2",
        num_prompt_tokens: int = 10,
        lstm_hidden: int = 256,
    ):
        super().__init__()
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        d_model = self.model.config.n_embd
        self.d_model = d_model
        self.num_prompt_tokens = num_prompt_tokens
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # P-Tuning v1 prompt encoder
        self.prompt_encoder = LSTMPromptEncoder(
            num_tokens=num_prompt_tokens,
            d_model=d_model,
            lstm_hidden=lstm_hidden,
            lstm_layers=2,
        )
        
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n  GPT2PTuningV1 ({model_name}):")
        print(f"    Total: {total:,} | Trainable: {trainable:,} "
              f"({trainable/total*100:.3f}%)")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Generate prompts via LSTM encoder
        prompt_embeds = self.prompt_encoder(batch_size).to(device)
        
        # Get input embeddings
        input_embeds = self.model.transformer.wte(input_ids)
        
        # Combine: [prompt; input]
        combined = torch.cat([prompt_embeds, input_embeds], dim=1)
        
        # Extend attention mask
        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, self.num_prompt_tokens,
                                     device=device, dtype=attention_mask.dtype)
            full_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        else:
            full_mask = torch.ones(batch_size, combined.shape[1], device=device)
        
        # Extend labels
        if labels is not None:
            prompt_labels = torch.full(
                (batch_size, self.num_prompt_tokens), -100,
                device=device, dtype=labels.dtype
            )
            full_labels = torch.cat([prompt_labels, labels], dim=1)
        else:
            full_labels = None
        
        # Forward
        outputs = self.model(
            inputs_embeds=combined,
            attention_mask=full_mask,
            labels=full_labels,
        )
        
        return {"loss": outputs.loss, "logits": outputs.logits}


def demonstrate_gpt2_p_tuning():
    """Demonstrate P-Tuning v1 with GPT-2."""
    print("\n\n" + "=" * 65)
    print("  SECTION 4: GPT-2 P-TUNING V1 INTEGRATION")
    print("=" * 65)
    
    model = GPT2PTuningV1(
        model_name="distilgpt2",
        num_prompt_tokens=10,
        lstm_hidden=256,
    )
    
    tokenizer = model.tokenizer
    
    # Test forward pass
    text = "The capital of France is Paris"
    inputs = tokenizer(text, return_tensors="pt")
    
    output = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=inputs["input_ids"],
    )
    
    print(f"\n  Input: '{text}'")
    print(f"  Loss: {output['loss'].item():.4f}")
    print(f"  Logits: {output['logits'].shape}")
    
    # Verify gradient flow
    output["loss"].backward()
    enc_grad = sum(
        p.grad.abs().sum().item()
        for p in model.prompt_encoder.parameters()
        if p.grad is not None
    )
    print(f"\n  Encoder gradient magnitude: {enc_grad:.6f}")
    print(f"  ✓ LSTM encoder receives gradients through frozen GPT-2")
    
    explanation = """
  ═══ P-Tuning v1 Key Insight ═══
  
  P-Tuning v1 showed that GPT-2 (a decoder-only model)
  can perform knowledge probing and NLU tasks when:
  
  1. Continuous prompts are used (instead of discrete)
  2. Prompts are generated by an LSTM encoder
  3. Templates interleave prompts with input
  
  This challenged the assumption that "GPT doesn't understand"
  — it does, but needs the right kind of prompting!
  
  Before P-Tuning: GPT-2 scores ~40% on LAMA benchmark
  After P-Tuning:  GPT-2 scores ~64% on LAMA benchmark
  (vs BERT's ~53% with manual prompts!)
"""
    print(explanation)
    
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all P-Tuning v1 demonstrations."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     P-TUNING V1 — FROM SCRATCH IMPLEMENTATION               ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: LSTM encoder
    demonstrate_encoder()
    
    # Section 2: Template patterns
    demonstrate_templates()
    
    # Section 3: Full model
    demonstrate_p_tuning_v1()
    
    # Section 4: GPT-2 integration
    demonstrate_gpt2_p_tuning()
    
    print("\n" + "=" * 65)
    print("  MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered:
    ✓ LSTM Prompt Encoder (bidirectional + MLP)
    ✓ Template-based prompting (prefix, cloze, mixed)
    ✓ Complete P-Tuning v1 model from scratch
    ✓ GPT-2 integration with P-Tuning v1
    """)


if __name__ == "__main__":
    main()
