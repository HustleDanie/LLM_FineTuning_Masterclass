"""
LoRA Implementation from Scratch
=================================

A complete, heavily-commented implementation of LoRA that you can read
line-by-line to understand every detail. This goes far beyond the minimal
implementation in Concept 4, covering:

1. LoRALinear layer with full forward/backward mechanics
2. Proper initialization (Kaiming for A, zeros for B)
3. Scaling factor (α/r) implementation
4. Dropout on the low-rank path
5. Weight merging and unmerging
6. Injecting LoRA into an existing model
7. Saving and loading LoRA weights
8. Complete training loop with LoRA

This is a TEACHING implementation — production code should use
HuggingFace PEFT (see lora_training.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Set
import math
import copy
import json
import os


# ===========================================================================
# 1. CORE LoRA LINEAR LAYER
# ===========================================================================

class LoRALinear(nn.Module):
    """
    A linear layer augmented with LoRA (Low-Rank Adaptation).
    
    Replaces: h = Wx + b
    With:     h = Wx + (α/r) · B(Ax) + b
    
    where W is frozen and only A, B are trained.
    
    Parameters:
    -----------
    original_layer : nn.Linear
        The pre-trained linear layer to augment
    rank : int
        The rank of the low-rank decomposition (r)
    alpha : float
        The scaling factor (α). Output is scaled by α/r.
    dropout : float
        Dropout probability applied to the LoRA path
    
    Architecture:
    
        x ──────────── W ──────────── + ── h
        │                              │
        └── Dropout ── A ── B ── ×α/r ─┘
            
        W: frozen (d_out × d_in)
        A: trainable (r × d_in)     ← down-projection
        B: trainable (d_out × r)    ← up-projection
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # The key scaling factor
        self.dropout_rate = dropout
        
        # ---------------------------------------------------------------
        # Store the FROZEN original weight
        # We keep it as a regular parameter but will freeze it
        # ---------------------------------------------------------------
        self.weight = original_layer.weight  # (d_out, d_in)
        self.bias = original_layer.bias      # (d_out,) or None
        
        # Freeze original weights
        self.weight.requires_grad_(False)
        if self.bias is not None:
            # Note: bias is typically left trainable in practice
            # but some implementations freeze it too
            self.bias.requires_grad_(False)
        
        # ---------------------------------------------------------------
        # LoRA matrices A and B
        # ---------------------------------------------------------------
        # A: down-projection (d_in → r)
        # Stored as (r, d_in) so we can do Ax directly
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        
        # B: up-projection (r → d_out)
        # Stored as (d_out, r) so we can do B(Ax) directly
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank))
        
        # ---------------------------------------------------------------
        # Dropout (applied before the LoRA path)
        # ---------------------------------------------------------------
        if dropout > 0:
            self.lora_dropout = nn.Dropout(p=dropout)
        else:
            self.lora_dropout = nn.Identity()
        
        # ---------------------------------------------------------------
        # State tracking for merging/unmerging
        # ---------------------------------------------------------------
        self._merged = False
        
        # Initialize LoRA weights
        self.reset_lora_parameters()
    
    def reset_lora_parameters(self):
        """
        Initialize LoRA parameters.
        
        Standard initialization:
        - A: Kaiming uniform (same as nn.Linear default)
        - B: Zeros
        
        This ensures ΔW = B·A = 0 at initialization, so the model
        starts exactly at the pre-trained weights.
        
        Why Kaiming for A?
        - A acts like a linear layer's weight
        - Kaiming initialization preserves variance through the network
        - Using nn.init.kaiming_uniform_ matches PyTorch's default for Linear
        
        Why zeros for B?
        - Ensures output starts at zero: B·A·x = 0 · A·x = 0
        - Any non-zero init would shift activations from pre-trained values
        - The asymmetric init (one zero, one non-zero) is crucial
        """
        # A ~ Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # B = 0
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LoRA-augmented linear layer.
        
        If merged: h = W'x + b  (where W' = W + scaling * B·A)
        If not merged: h = Wx + scaling * B·A·dropout(x) + b
        
        The non-merged version is used during training (need separate
        gradients for A and B). The merged version can be used at
        inference for zero overhead.
        """
        if self._merged:
            # Weights already merged: just a normal linear
            return F.linear(x, self.weight, self.bias)
        
        # -----------------------------------------------------------
        # Standard path: Wx + b
        # -----------------------------------------------------------
        result = F.linear(x, self.weight, self.bias)
        
        # -----------------------------------------------------------
        # LoRA path: scaling * B · A · dropout(x)
        # -----------------------------------------------------------
        # Step 1: Apply dropout to input
        # Dropout is applied to x before the LoRA path, not after
        # This is more stable than dropout on the low-rank activations
        x_dropout = self.lora_dropout(x)
        
        # Step 2: Down-project: A · x → (batch, seq, r)
        # x_dropout: (batch, seq, d_in)
        # lora_A:    (r, d_in)
        # Result:    (batch, seq, r)
        low_rank = F.linear(x_dropout, self.lora_A)
        
        # Step 3: Up-project: B · (Ax) → (batch, seq, d_out)
        # low_rank: (batch, seq, r)
        # lora_B:   (d_out, r)
        # Result:   (batch, seq, d_out)
        lora_output = F.linear(low_rank, self.lora_B)
        
        # Step 4: Scale and add to original output
        result = result + self.scaling * lora_output
        
        return result
    
    def merge_weights(self):
        """
        Merge LoRA weights into the original weight matrix.
        
        W' = W + (α/r) · B · A
        
        After merging:
        - Forward pass is a single matrix multiply (zero overhead)
        - Useful for inference deployment
        - Call unmerge_weights() to restore separation
        """
        if self._merged:
            print("  [Warning] Weights already merged")
            return
        
        # Compute ΔW = scaling * B @ A
        delta_W = self.scaling * (self.lora_B @ self.lora_A)
        
        # Merge into frozen weight
        # We modify .data to avoid autograd issues with frozen params
        self.weight.data += delta_W
        
        self._merged = True
    
    def unmerge_weights(self):
        """
        Remove LoRA contribution from the merged weights.
        
        W = W' - (α/r) · B · A
        
        Useful when you need to:
        - Switch between LoRA adapters
        - Continue training after inference
        """
        if not self._merged:
            print("  [Warning] Weights not merged")
            return
        
        delta_W = self.scaling * (self.lora_B @ self.lora_A)
        self.weight.data -= delta_W
        self._merged = False
    
    def get_delta_weight(self) -> torch.Tensor:
        """Return the current ΔW = (α/r) · B · A"""
        return self.scaling * (self.lora_B @ self.lora_A)
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}, "
            f"dropout={self.dropout_rate}, merged={self._merged}"
        )


# ===========================================================================
# 2. LoRA MODEL WRAPPER — Inject LoRA into Any Model
# ===========================================================================

class LoRAModel(nn.Module):
    """
    Wraps any model and injects LoRA layers into specified linear layers.
    
    This mimics what HuggingFace PEFT's `get_peft_model()` does internally.
    
    Usage:
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        lora_model = LoRAModel(
            model,
            target_modules=["c_attn", "c_proj"],  # Which layers to adapt
            rank=16,
            alpha=32,
        )
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_modules: List[str],
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        modules_to_save: Optional[List[str]] = None,
    ):
        super().__init__()
        
        self.model = model
        self.target_modules = target_modules
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.modules_to_save = modules_to_save or []
        
        # Track which layers got LoRA
        self.lora_layers: Dict[str, LoRALinear] = {}
        
        # Step 1: Freeze ALL parameters
        self._freeze_all_parameters()
        
        # Step 2: Inject LoRA into target modules
        self._inject_lora()
        
        # Step 3: Unfreeze modules_to_save (e.g., classifier head)
        self._unfreeze_modules_to_save()
        
        # Print summary
        self._print_summary()
    
    def _freeze_all_parameters(self):
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _inject_lora(self):
        """
        Walk the model tree and replace matching Linear layers with LoRALinear.
        
        This is the key step — we find all nn.Linear modules whose names
        match the target_modules patterns and wrap them with LoRA.
        """
        # Collect all modules to replace (can't modify during iteration)
        replacements = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if this module matches any target pattern
                if self._should_adapt(name):
                    replacements.append((name, module))
        
        # Apply replacements
        for name, original_linear in replacements:
            lora_layer = LoRALinear(
                original_layer=original_linear,
                rank=self.rank,
                alpha=self.alpha,
                dropout=self.dropout,
            )
            
            # Replace the module in the model's tree
            self._replace_module(name, lora_layer)
            self.lora_layers[name] = lora_layer
        
        print(f"\n  Injected LoRA into {len(self.lora_layers)} layers:")
        for name in self.lora_layers:
            print(f"    • {name}")
    
    def _should_adapt(self, module_name: str) -> bool:
        """Check if a module name matches any target pattern."""
        return any(target in module_name for target in self.target_modules)
    
    def _replace_module(self, name: str, new_module: nn.Module):
        """
        Replace a module in the model tree by name.
        
        Example: name = "transformer.h.0.attn.c_attn"
        We need to navigate to transformer.h.0.attn and replace c_attn.
        """
        parts = name.split(".")
        parent = self.model
        
        # Navigate to parent module
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        
        # Replace the target module
        last_part = parts[-1]
        if last_part.isdigit():
            parent[int(last_part)] = new_module
        else:
            setattr(parent, last_part, new_module)
    
    def _unfreeze_modules_to_save(self):
        """Unfreeze specified modules (e.g., classification head)."""
        for name, param in self.model.named_parameters():
            if any(mod in name for mod in self.modules_to_save):
                param.requires_grad = True
    
    def _print_summary(self):
        """Print parameter summary."""
        total_params = 0
        trainable_params = 0
        
        for param in self.model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"\n  Parameter Summary:")
        print(f"    Total parameters:     {total_params:>12,}")
        print(f"    Trainable parameters: {trainable_params:>12,}")
        print(f"    Trainable %:          {trainable_params/total_params*100:>12.4f}%")
        print(f"    LoRA rank:            {self.rank}")
        print(f"    LoRA alpha:           {self.alpha}")
        print(f"    LoRA scaling (α/r):   {self.alpha/self.rank:.4f}")
    
    def forward(self, **kwargs):
        """Forward through the modified model."""
        return self.model(**kwargs)
    
    def merge_all(self):
        """Merge all LoRA weights for inference."""
        for name, layer in self.lora_layers.items():
            layer.merge_weights()
        print(f"  Merged {len(self.lora_layers)} LoRA layers")
    
    def unmerge_all(self):
        """Unmerge all LoRA weights (restore training state)."""
        for name, layer in self.lora_layers.items():
            layer.unmerge_weights()
        print(f"  Unmerged {len(self.lora_layers)} LoRA layers")
    
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Extract only the LoRA parameters for saving.
        
        Returns a state dict containing only lora_A and lora_B weights.
        This is tiny compared to the full model — the whole point of LoRA!
        """
        lora_state = {}
        for name, param in self.model.named_parameters():
            if "lora_" in name and param.requires_grad:
                lora_state[name] = param.data.clone()
        return lora_state
    
    def save_lora(self, save_dir: str):
        """
        Save LoRA weights and config to a directory.
        
        Saves:
        - lora_weights.pt: The LoRA A and B matrices
        - lora_config.json: Hyperparameters needed to reconstruct
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save weights
        lora_state = self.get_lora_state_dict()
        torch.save(lora_state, os.path.join(save_dir, "lora_weights.pt"))
        
        # Save config
        config = {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "modules_to_save": self.modules_to_save,
            "lora_layers": list(self.lora_layers.keys()),
        }
        with open(os.path.join(save_dir, "lora_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # Report size
        total_bytes = sum(p.nbytes for p in lora_state.values())
        print(f"  Saved {len(lora_state)} LoRA tensors ({total_bytes / 1024:.1f} KB)")
        print(f"  Save directory: {save_dir}")
    
    def load_lora(self, save_dir: str):
        """Load LoRA weights from a directory."""
        lora_state = torch.load(
            os.path.join(save_dir, "lora_weights.pt"),
            map_location="cpu",
            weights_only=True,
        )
        
        # Load the state dict into the model
        model_state = self.model.state_dict()
        for name, tensor in lora_state.items():
            if name in model_state:
                model_state[name].copy_(tensor)
            else:
                print(f"  [Warning] Key {name} not found in model")
        
        print(f"  Loaded {len(lora_state)} LoRA tensors from {save_dir}")
    
    def get_trainable_parameters(self):
        """Get only the trainable parameters (for optimizer)."""
        return [p for p in self.model.parameters() if p.requires_grad]


# ===========================================================================
# 3. GRADIENT ANALYSIS — Understanding What LoRA Learns
# ===========================================================================

def analyze_lora_gradients():
    """
    Analyze gradient flow through LoRA layers to understand training dynamics.
    
    Key questions:
    - How do gradients flow through the low-rank path?
    - How does the rank affect gradient magnitudes?
    - What does the effective gradient on W look like?
    """
    print("=" * 70)
    print("LoRA GRADIENT ANALYSIS")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    d_in, d_out = 256, 256
    rank = 16
    alpha = 32.0
    batch_size = 8
    seq_len = 32
    
    # Create a LoRA layer
    original = nn.Linear(d_in, d_out)
    lora = LoRALinear(original, rank=rank, alpha=alpha)
    
    # Simulate some training steps to get non-zero B
    # (At init, B=0 so gradients on A would be zero via chain rule)
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.01)
    
    # Forward pass
    x = torch.randn(batch_size, seq_len, d_in, requires_grad=True)
    output = lora(x)
    
    # Backward pass (simple loss: sum of outputs)
    loss = output.sum()
    loss.backward()
    
    print(f"\nConfiguration: d_in={d_in}, d_out={d_out}, rank={rank}, α={alpha}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Analyze gradients
    print(f"\nGradient analysis:")
    print(f"  ∂L/∂A  shape: {lora.lora_A.grad.shape}  "
          f"norm: {lora.lora_A.grad.norm():.6f}  "
          f"mean: {lora.lora_A.grad.abs().mean():.6f}")
    print(f"  ∂L/∂B  shape: {lora.lora_B.grad.shape}  "
          f"norm: {lora.lora_B.grad.norm():.6f}  "
          f"mean: {lora.lora_B.grad.abs().mean():.6f}")
    
    # The effective gradient on W (what full fine-tuning would do)
    print(f"\n  ∂L/∂W (frozen, not computed):")
    print(f"    If we WERE computing it, shape would be: ({d_out}, {d_in})")
    print(f"    That's {d_out * d_in:,} gradient values we DON'T need to store!")
    print(f"    LoRA stores only {rank * d_in + d_out * rank:,} gradient values instead")
    
    # Effective update rank
    delta_W = lora.get_delta_weight()
    U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
    
    print(f"\n  Effective ΔW analysis:")
    print(f"    ΔW shape: {delta_W.shape}")
    print(f"    ΔW Frobenius norm: {delta_W.norm():.6f}")
    print(f"    ΔW rank (non-zero SVs): {(S > 1e-6).sum().item()}")
    print(f"    Top-5 singular values: {S[:5].tolist()}")
    print(f"    (Only {rank} can be non-zero due to rank-{rank} constraint)")
    
    # Compare gradient norms across different ranks
    print(f"\n  Gradient norm vs rank (with α scaling):")
    for r in [4, 8, 16, 32, 64]:
        a = alpha  # Fixed alpha
        test_lora = LoRALinear(nn.Linear(d_in, d_out), rank=r, alpha=a)
        with torch.no_grad():
            test_lora.lora_B.normal_(0, 0.01)
        
        test_x = torch.randn(batch_size, seq_len, d_in)
        test_out = test_lora(test_x)
        test_out.sum().backward()
        
        grad_A_norm = test_lora.lora_A.grad.norm().item()
        grad_B_norm = test_lora.lora_B.grad.norm().item()
        scaling = a / r
        
        print(f"    r={r:>3}, α={a:>3}, α/r={scaling:>5.2f}  "
              f"||∂L/∂A||={grad_A_norm:>8.4f}  ||∂L/∂B||={grad_B_norm:>8.4f}")


# ===========================================================================
# 4. COMPLETE TRAINING EXAMPLE FROM SCRATCH
# ===========================================================================

def train_lora_from_scratch():
    """
    Complete example: Train a LoRA-augmented model from scratch.
    
    This demonstrates the full workflow without using HuggingFace PEFT,
    so you can see every step clearly.
    """
    print("\n" + "=" * 70)
    print("COMPLETE LoRA TRAINING FROM SCRATCH")
    print("=" * 70)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  [Install transformers to run this example]")
        return
    
    # -------------------------------------------------------------------
    # Step 1: Load pre-trained model
    # -------------------------------------------------------------------
    print("\n[Step 1] Loading pre-trained model...")
    model_name = "distilgpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"  Model: {model_name}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # -------------------------------------------------------------------
    # Step 2: Inject LoRA
    # -------------------------------------------------------------------
    print("\n[Step 2] Injecting LoRA layers...")
    
    # For GPT-2, the attention layers are named c_attn and c_proj
    lora_model = LoRAModel(
        model=model,
        target_modules=["c_attn", "c_proj"],  # Attention Q, K, V and output
        rank=8,
        alpha=16,
        dropout=0.05,
    )
    
    # -------------------------------------------------------------------
    # Step 3: Prepare optimizer (only LoRA parameters!)
    # -------------------------------------------------------------------
    print("\n[Step 3] Setting up optimizer...")
    
    trainable_params = lora_model.get_trainable_parameters()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=1e-4,          # LoRA typically uses higher LR than full FT
        weight_decay=0.01,
    )
    
    print(f"  Optimizer: AdamW")
    print(f"  Learning rate: 1e-4")
    print(f"  Trainable parameter groups: {len(trainable_params)}")
    
    # -------------------------------------------------------------------
    # Step 4: Create training data
    # -------------------------------------------------------------------
    print("\n[Step 4] Preparing training data...")
    
    texts = [
        "The transformer architecture revolutionized natural language processing.",
        "Low-rank adaptation enables efficient fine-tuning of large models.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Pre-training on large corpora gives models general language understanding.",
        "Fine-tuning adapts a pre-trained model to a specific downstream task.",
        "Parameter-efficient methods reduce the computational cost of fine-tuning.",
        "LoRA decomposes weight updates into low-rank matrices for efficiency.",
        "The scaling factor alpha over r ensures consistent behavior across ranks.",
    ]
    
    # Tokenize
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    
    # For causal LM, labels = input_ids (shifted internally by the model)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100  # Ignore padding in loss
    
    print(f"  Training samples: {len(texts)}")
    print(f"  Token shape: {input_ids.shape}")
    
    # -------------------------------------------------------------------
    # Step 5: Training loop
    # -------------------------------------------------------------------
    print("\n[Step 5] Training...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    
    num_epochs = 5
    model.train()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        loss = outputs.loss
        loss.backward()
        
        # Gradient clipping (important for stable training)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        
        optimizer.step()
        
        # Log training metrics
        grad_norm = sum(p.grad.norm().item() ** 2 for p in trainable_params 
                       if p.grad is not None) ** 0.5
        
        print(f"  Epoch {epoch+1}/{num_epochs}: "
              f"loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")
    
    # -------------------------------------------------------------------
    # Step 6: Analyze what was learned
    # -------------------------------------------------------------------
    print("\n[Step 6] Analyzing learned LoRA weights...")
    
    for name, layer in lora_model.lora_layers.items():
        delta_W = layer.get_delta_weight()
        U, S, Vh = torch.linalg.svd(delta_W.float().cpu(), full_matrices=False)
        
        print(f"\n  {name}:")
        print(f"    ||ΔW||_F = {delta_W.norm().item():.6f}")
        print(f"    Top singular values: {S[:5].tolist()}")
        print(f"    Effective rank (>1% of σ₁): "
              f"{(S > 0.01 * S[0]).sum().item()}")
    
    # -------------------------------------------------------------------
    # Step 7: Save LoRA weights
    # -------------------------------------------------------------------
    print("\n[Step 7] Saving LoRA weights...")
    save_dir = os.path.join(os.path.dirname(__file__), "demo_lora_weights")
    lora_model.save_lora(save_dir)
    
    # -------------------------------------------------------------------
    # Step 8: Test inference with merged weights
    # -------------------------------------------------------------------
    print("\n[Step 8] Testing inference...")
    
    model.eval()
    lora_model.merge_all()
    
    prompt = "LoRA fine-tuning is"
    input_ids_gen = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids_gen,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\n  Prompt: '{prompt}'")
    print(f"  Generated: '{generated}'")
    
    # Unmerge for potential further training
    lora_model.unmerge_all()
    
    print("\n  ✓ Complete LoRA training pipeline executed successfully!")


# ===========================================================================
# 5. MULTI-ADAPTER SUPPORT
# ===========================================================================

class MultiLoRAModel(nn.Module):
    """
    Support multiple LoRA adapters on the same base model.
    
    This is how production systems serve personalized models:
    one base model + multiple tiny LoRA adapters that can be
    swapped in/out per request.
    
    Usage:
        multi = MultiLoRAModel(base_model, target_modules=["c_attn"])
        multi.add_adapter("task_a", rank=8, alpha=16)
        multi.add_adapter("task_b", rank=16, alpha=32)
        multi.set_active("task_a")  # Switch adapters
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_modules: List[str],
    ):
        super().__init__()
        self.model = model
        self.target_modules = target_modules
        self.adapters: Dict[str, Dict[str, LoRALinear]] = {}
        self.active_adapter: Optional[str] = None
        
        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Find and track all linear layers that match targets
        self._target_layers: Dict[str, nn.Linear] = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if any(t in name for t in target_modules):
                    self._target_layers[name] = module
        
        print(f"  MultiLoRA: {len(self._target_layers)} target layers")
    
    def add_adapter(
        self,
        adapter_name: str,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        """Add a new LoRA adapter."""
        if adapter_name in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' already exists")
        
        adapter_layers = {}
        for name, linear in self._target_layers.items():
            lora = LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout)
            adapter_layers[name] = lora
        
        self.adapters[adapter_name] = adapter_layers
        print(f"  Added adapter '{adapter_name}' (rank={rank}, α={alpha})")
        
        # Set as active if no adapter is active
        if self.active_adapter is None:
            self.set_active(adapter_name)
    
    def set_active(self, adapter_name: str):
        """Switch to a different adapter."""
        if adapter_name not in self.adapters:
            raise ValueError(f"Unknown adapter: {adapter_name}")
        
        # Unmerge current if merged
        if self.active_adapter and self.active_adapter in self.adapters:
            for layer in self.adapters[self.active_adapter].values():
                if layer._merged:
                    layer.unmerge_weights()
        
        # Install new adapter's LoRA layers
        for name, lora_layer in self.adapters[adapter_name].items():
            self._replace_module(name, lora_layer)
        
        self.active_adapter = adapter_name
        print(f"  Active adapter: '{adapter_name}'")
    
    def _replace_module(self, name: str, module: nn.Module):
        """Replace module in model tree."""
        parts = name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        if parts[-1].isdigit():
            parent[int(parts[-1])] = module
        else:
            setattr(parent, parts[-1], module)
    
    def list_adapters(self):
        """List all available adapters."""
        for name in self.adapters:
            active = " (active)" if name == self.active_adapter else ""
            layers = len(self.adapters[name])
            print(f"  • {name}: {layers} layers{active}")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("LoRA FROM SCRATCH — Complete Implementation")
    print("=" * 70)
    
    # Demo 1: Basic LoRA layer
    print("\n--- Demo 1: LoRA Linear Layer ---")
    original = nn.Linear(768, 768)
    lora_layer = LoRALinear(original, rank=16, alpha=32, dropout=0.1)
    print(f"  Layer: {lora_layer}")
    
    x = torch.randn(2, 10, 768)
    y = lora_layer(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  At init, output should match original (B=0):")
    y_orig = F.linear(x, original.weight, original.bias)
    print(f"  Max difference: {(y - y_orig).abs().max().item():.2e}")
    
    # Demo 2: Gradient analysis
    print()
    analyze_lora_gradients()
    
    # Demo 3: Full training pipeline
    print()
    train_lora_from_scratch()
