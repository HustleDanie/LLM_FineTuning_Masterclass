"""
AdapterFusion: Combining Multiple Task-Specific Adapters
=========================================================

AdapterFusion (Pfeiffer et al., 2021) allows combining knowledge from
multiple task-specific adapters through a learned attention mechanism.

Key Concepts:
1. Train separate adapters for different tasks
2. Freeze all adapters
3. Learn attention weights that combine adapter outputs
4. The fusion layer learns which adapters are useful for the target task

This enables:
- Non-destructive combination of task knowledge
- Transfer learning from multiple source tasks
- No catastrophic forgetting of individual task knowledge
- Modular composition of capabilities

Architecture:
    Query: hidden state from the transformer
    Key/Value: outputs from each task-specific adapter
    Output: attention-weighted combination of adapter outputs

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Optional, Tuple


# ============================================================================
# TASK-SPECIFIC ADAPTERS (to be fused)
# ============================================================================

class TaskAdapter(nn.Module):
    """
    A simple task-specific adapter to be used in fusion.
    
    Each adapter is trained independently on one task,
    then frozen before fusion training.
    """
    
    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int = 64,
        task_name: str = "generic",
    ):
        super().__init__()
        self.task_name = task_name
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        
        # Initialize for identity
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer_norm(x)
        h = self.down_proj(h)
        h = self.activation(h)
        h = self.up_proj(h)
        return h + x
    
    def get_adapter_output(self, x: torch.Tensor) -> torch.Tensor:
        """Get just the adapter delta (without residual)."""
        h = self.layer_norm(x)
        h = self.down_proj(h)
        h = self.activation(h)
        h = self.up_proj(h)
        return h


# ============================================================================
# ADAPTERFUSION MECHANISM
# ============================================================================

class AdapterFusion(nn.Module):
    """
    AdapterFusion: Attention-based combination of adapter outputs.
    
    Given N task-specific adapters, AdapterFusion learns how to
    combine their outputs using a multi-head attention mechanism:
    
    1. Query = f(hidden_state)
    2. Key_i = g(adapter_i(hidden_state))  for each adapter i
    3. Value_i = adapter_i(hidden_state)   for each adapter i
    4. output = Attention(Q, [K_1...K_N], [V_1...V_N])
    
    The Query, Key projections are learnable.
    The adapter weights are FROZEN during fusion training.
    
    Only the fusion parameters (Q, K, V projections) are trained.
    
    Fusion Parameters = O(d² / h) where d=hidden_size, h=heads
    This is ~590K params for hidden_size=768.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_adapters: int,
        num_heads: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_adapters = num_adapters
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        
        # Fusion attention parameters
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)
    
    def forward(
        self,
        hidden_state: torch.Tensor,
        adapter_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Combine adapter outputs via attention.
        
        Args:
            hidden_state: [batch, seq, hidden] - transformer hidden state
            adapter_outputs: List of [batch, seq, hidden] - one per adapter
            
        Returns:
            fused_output: [batch, seq, hidden]
        """
        batch_size, seq_len, _ = hidden_state.shape
        num_adapters = len(adapter_outputs)
        
        # Query from hidden state: [batch, seq, hidden]
        Q = self.query_proj(hidden_state)
        
        # Stack adapter outputs: [batch, seq, num_adapters, hidden]
        stacked = torch.stack(adapter_outputs, dim=2)
        
        # Keys and Values from adapter outputs
        # Reshape for projection: [batch * seq * num_adapters, hidden]
        stacked_flat = stacked.reshape(-1, self.hidden_size)
        K = self.key_proj(stacked_flat).reshape(batch_size, seq_len, num_adapters, self.hidden_size)
        V = self.value_proj(stacked_flat).reshape(batch_size, seq_len, num_adapters, self.hidden_size)
        
        # Reshape for multi-head attention
        # Q: [batch, seq, heads, head_dim] → [batch, heads, seq, head_dim]
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # K: [batch, seq, adapters, heads, head_dim] → [batch, heads, seq, adapters, head_dim]
        K = K.reshape(batch_size, seq_len, num_adapters, self.num_heads, self.head_dim)
        K = K.permute(0, 3, 1, 2, 4)  # [batch, heads, seq, adapters, head_dim]
        
        V = V.reshape(batch_size, seq_len, num_adapters, self.num_heads, self.head_dim)
        V = V.permute(0, 3, 1, 2, 4)  # [batch, heads, seq, adapters, head_dim]
        
        # Attention: Q attends over adapters
        # Q: [batch, heads, seq, 1, head_dim] @ K^T: [batch, heads, seq, head_dim, adapters]
        # → scores: [batch, heads, seq, 1, adapters]
        Q = Q.unsqueeze(3)  # [batch, heads, seq, 1, head_dim]
        
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scaling
        # attn_scores: [batch, heads, seq, 1, adapters]
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted combination: [batch, heads, seq, 1, adapters] @ [batch, heads, seq, adapters, head_dim]
        # → [batch, heads, seq, 1, head_dim]
        fused = torch.matmul(attn_weights, V)
        fused = fused.squeeze(3)  # [batch, heads, seq, head_dim]
        
        # Reshape back: [batch, seq, hidden]
        fused = fused.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        
        return fused
    
    def get_attention_weights(
        self,
        hidden_state: torch.Tensor,
        adapter_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """Get adapter attention weights for analysis."""
        batch_size, seq_len, _ = hidden_state.shape
        num_adapters = len(adapter_outputs)
        
        Q = self.query_proj(hidden_state)
        stacked = torch.stack(adapter_outputs, dim=2)
        stacked_flat = stacked.reshape(-1, self.hidden_size)
        K = self.key_proj(stacked_flat).reshape(batch_size, seq_len, num_adapters, self.hidden_size)
        
        # Simplified single-head for analysis
        Q_flat = Q.reshape(batch_size, seq_len, 1, self.hidden_size)
        K_flat = K  # [batch, seq, adapters, hidden]
        
        scores = torch.sum(Q_flat * K_flat, dim=-1) / self.scaling
        weights = F.softmax(scores, dim=-1)
        
        return weights  # [batch, seq, num_adapters]


# ============================================================================
# FULL ADAPTERFUSION LAYER
# ============================================================================

class AdapterFusionLayer(nn.Module):
    """
    Complete AdapterFusion layer that manages multiple adapters
    and their fusion for a single transformer layer.
    
    During fusion training:
    - All adapters are FROZEN
    - Only the fusion attention parameters are trained
    - The transformer backbone is also FROZEN
    
    This means: Total trainable params ≈ fusion params only
    """
    
    def __init__(
        self,
        hidden_size: int,
        adapters: Dict[str, TaskAdapter],
        num_heads: int = 1,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.adapter_names = list(adapters.keys())
        
        # Register adapters as submodules (but freeze them)
        self.adapters = nn.ModuleDict(adapters)
        for adapter in self.adapters.values():
            for param in adapter.parameters():
                param.requires_grad = False
        
        # Fusion mechanism (this IS trained)
        self.fusion = AdapterFusion(
            hidden_size=hidden_size,
            num_adapters=len(adapters),
            num_heads=num_heads,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run all adapters, then fuse their outputs.
        
        Flow:
            x → [adapter_1(x), adapter_2(x), ..., adapter_n(x)]
              → Fusion(x, [out_1, ..., out_n])
              → fused_output + x  (residual)
        """
        # Get output from each adapter (no gradients — frozen)
        adapter_outputs = []
        for name in self.adapter_names:
            with torch.no_grad():
                adapter_out = self.adapters[name].get_adapter_output(x)
            adapter_outputs.append(adapter_out)
        
        # Fuse adapter outputs (this part has gradients)
        fused = self.fusion(x, adapter_outputs)
        
        return fused + x  # Residual
    
    def get_trainable_params(self) -> int:
        """Count trainable (fusion) parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_frozen_params(self) -> int:
        """Count frozen (adapter) parameters."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
    
    def analyze_fusion_weights(self, x: torch.Tensor) -> Dict[str, float]:
        """Analyze which adapters the fusion attends to."""
        adapter_outputs = []
        for name in self.adapter_names:
            with torch.no_grad():
                out = self.adapters[name].get_adapter_output(x)
            adapter_outputs.append(out)
        
        weights = self.fusion.get_attention_weights(x, adapter_outputs)
        avg_weights = weights.mean(dim=[0, 1])  # Average over batch and seq
        
        result = {}
        for i, name in enumerate(self.adapter_names):
            result[name] = avg_weights[i].item()
        
        return result


# ============================================================================
# STACKING ADAPTERS (Alternative to Fusion)
# ============================================================================

class AdapterStack(nn.Module):
    """
    Stack adapters sequentially instead of fusing with attention.
    
    AdapterStack vs AdapterFusion:
    
    Stack: x → Adapter₁ → Adapter₂ → Adapter₃ → output
    - Order matters (non-commutative)
    - Each adapter transforms the output of the previous one
    - No additional fusion parameters
    - Can suffer from representational interference
    
    Fusion: x → [A₁(x), A₂(x), A₃(x)] → Attention → output
    - All adapters see the same input
    - Learned weighting determines contribution
    - Additional fusion parameters needed
    - Better handles conflicting adapter knowledge
    """
    
    def __init__(self, adapters: List[TaskAdapter]):
        super().__init__()
        self.adapters = nn.ModuleList(adapters)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adapters sequentially."""
        for adapter in self.adapters:
            x = adapter(x)
        return x


# ============================================================================
# DEMONSTRATION: FULL FUSION PIPELINE
# ============================================================================

class FusionDemonstration:
    """Demonstrate the full AdapterFusion pipeline."""
    
    def demonstrate_full_pipeline(self, hidden_size: int = 256):
        """Walk through the entire fusion process."""
        print("=" * 70)
        print("  ADAPTERFUSION: COMPLETE PIPELINE DEMONSTRATION")
        print("=" * 70)
        
        # ─── Step 1: Create task-specific adapters ───
        print("\n  STEP 1: Create Task-Specific Adapters")
        print("  " + "─" * 50)
        
        task_names = ["sentiment", "nli", "qa", "ner"]
        adapters = {}
        
        for task in task_names:
            adapter = TaskAdapter(hidden_size, bottleneck_size=32, task_name=task)
            
            # Simulate pre-training by randomizing weights
            # (in practice, each would be trained on its task)
            with torch.no_grad():
                for p in adapter.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
                    else:
                        nn.init.normal_(p, std=0.02)
            
            adapters[task] = adapter
            params = sum(p.numel() for p in adapter.parameters())
            print(f"    Adapter '{task}': {params:,} params")
        
        # ─── Step 2: Create fusion layer ───
        print(f"\n  STEP 2: Create AdapterFusion Layer")
        print("  " + "─" * 50)
        
        fusion_layer = AdapterFusionLayer(
            hidden_size=hidden_size,
            adapters=adapters,
            num_heads=1,
        )
        
        trainable = fusion_layer.get_trainable_params()
        frozen = fusion_layer.get_frozen_params()
        print(f"    Trainable params (fusion): {trainable:,}")
        print(f"    Frozen params (adapters):  {frozen:,}")
        print(f"    Ratio: {trainable/max(frozen,1):.2%} fusion vs adapter")
        
        # ─── Step 3: Forward pass ───
        print(f"\n  STEP 3: Forward Pass Through Fusion")
        print("  " + "─" * 50)
        
        x = torch.randn(2, 8, hidden_size)
        print(f"    Input shape:  {list(x.shape)}")
        
        output = fusion_layer(x)
        print(f"    Output shape: {list(output.shape)}")
        print(f"    Output norm:  {output.norm().item():.4f}")
        
        # ─── Step 4: Analyze attention weights ───
        print(f"\n  STEP 4: Analyze Fusion Attention Weights")
        print("  " + "─" * 50)
        
        weights = fusion_layer.analyze_fusion_weights(x)
        print(f"    (Before training — weights should be roughly uniform)")
        
        for task, weight in weights.items():
            bar = "█" * int(weight * 40) + "░" * (40 - int(weight * 40))
            print(f"    {task:>12}: [{bar}] {weight:.3f}")
        
        # ─── Step 5: Simulate fusion training ───
        print(f"\n  STEP 5: Simulate Fusion Training")
        print("  " + "─" * 50)
        
        optimizer = torch.optim.Adam(
            fusion_layer.fusion.parameters(), lr=1e-3
        )
        
        # Create a simple target that aligns more with "sentiment"
        # This simulates the fusion learning that sentiment adapter
        # is most useful for this particular target task
        target = adapters["sentiment"](x).detach()  # Target: sentiment behavior
        
        print(f"    Training fusion to favor 'sentiment' adapter...")
        losses = []
        
        for step in range(100):
            optimizer.zero_grad()
            output = fusion_layer(x)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if step % 20 == 0:
                print(f"      Step {step:>3}: Loss = {loss.item():.6f}")
        
        # ─── Step 6: Post-training analysis ───
        print(f"\n  STEP 6: Post-Training Attention Weights")
        print("  " + "─" * 50)
        
        weights_after = fusion_layer.analyze_fusion_weights(x)
        print(f"    (After training — should favor 'sentiment')")
        
        for task, weight in weights_after.items():
            bar = "█" * int(weight * 40) + "░" * (40 - int(weight * 40))
            marker = " ← target" if task == "sentiment" else ""
            print(f"    {task:>12}: [{bar}] {weight:.3f}{marker}")
        
        print(f"\n    Loss reduction: {losses[0]:.6f} → {losses[-1]:.6f} "
              f"({(1 - losses[-1]/losses[0])*100:.1f}% reduction)")
    
    def compare_fusion_vs_stack(self, hidden_size: int = 256):
        """Compare fusion and stacking approaches."""
        print("\n" + "=" * 70)
        print("  FUSION vs STACKING COMPARISON")
        print("=" * 70)
        
        adapters_dict = {
            name: TaskAdapter(hidden_size, 32, name)
            for name in ["task_a", "task_b", "task_c"]
        }
        adapters_list = list(adapters_dict.values())
        
        # Simulate some training by randomizing
        for a in adapters_list:
            with torch.no_grad():
                for p in a.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        
        # Build fusion and stack
        fusion = AdapterFusionLayer(hidden_size, adapters_dict)
        stack = AdapterStack(adapters_list)
        
        x = torch.randn(2, 8, hidden_size)
        
        # Forward pass
        fusion_out = fusion(x)
        stack_out = stack(x)
        
        print(f"\n  {'Property':<30} {'Fusion':>15} {'Stack':>15}")
        print("  " + "─" * 60)
        
        fusion_params = sum(p.numel() for p in fusion.parameters())
        stack_params = sum(p.numel() for p in stack.parameters())
        
        print(f"  {'Total params':<30} {fusion_params:>15,} {stack_params:>15,}")
        print(f"  {'Output norm':<30} {fusion_out.norm().item():>15.4f} "
              f"{stack_out.norm().item():>15.4f}")
        print(f"  {'Order dependent':<30} {'No':>15} {'Yes':>15}")
        print(f"  {'Composable':<30} {'Yes':>15} {'Limited':>15}")
        print(f"  {'Extra params needed':<30} {'Yes (fusion)':>15} {'No':>15}")
        
        diagram = """
  Fusion:
    x ─┬→ Adapter_A(x) ─┐
       ├→ Adapter_B(x) ─┤→ [ATTENTION] → output
       └→ Adapter_C(x) ─┘   ↑
                             Q from x
  
  Stack:
    x → Adapter_A → Adapter_B → Adapter_C → output
  
  Key differences:
    - Fusion: Each adapter sees the SAME input
    - Stack: Each adapter sees the PREVIOUS adapter's output
    - Fusion: Learned weighting selects useful adapters
    - Stack: Order matters, no selection mechanism
"""
        print(diagram)
    
    def demonstrate_adapter_composition_patterns(self):
        """Show different patterns for composing adapters."""
        print("\n" + "=" * 70)
        print("  ADAPTER COMPOSITION PATTERNS")
        print("=" * 70)
        
        patterns = """
  PATTERN 1: SINGLE ADAPTER (Baseline)
  ─────────────────────────────────────
    Train one adapter per task. Simple but no knowledge sharing.
    
    Task A: model + adapter_A
    Task B: model + adapter_B
    New:    model + adapter_new (train from scratch)

  PATTERN 2: ADAPTER STACKING
  ─────────────────────────────────────
    Apply adapters sequentially. Simple but order-sensitive.
    
    Combined: model + adapter_A → adapter_B
    Issue: adapter_A → adapter_B ≠ adapter_B → adapter_A

  PATTERN 3: ADAPTERFUSION (This module)
  ─────────────────────────────────────
    Learn attention-based combination. Powerful and flexible.
    
    Combined: model + fusion(adapter_A, adapter_B, adapter_C)
    Key: Only fusion layer is trained. Adapters stay frozen.

  PATTERN 4: ADAPTER AVERAGING
  ─────────────────────────────────────
    Average adapter parameters. Simple but can conflict.
    
    Combined: model + (adapter_A + adapter_B) / 2
    Issue: Works only when adapters are in similar regions.

  PATTERN 5: ADAPTER SPLITTING
  ─────────────────────────────────────
    Different adapters for different layers.
    
    Layers 0-3:  adapter_A (general features)
    Layers 4-7:  adapter_B (mid-level features)
    Layers 8-11: adapter_C (task-specific features)

  PATTERN 6: MAD-X (Cross-lingual)
  ─────────────────────────────────────
    Separate language and task adapters:
    
    model + language_adapter(English) + task_adapter(sentiment)
    model + language_adapter(French) + task_adapter(sentiment)
    
    Re-use task adapters across languages!
    
  COMPARISON TABLE:
  ┌───────────────┬──────────┬──────────┬──────────┬──────────┐
  │ Pattern       │ Extra    │ Order    │ Knowledge│ Best     │
  │               │ Params   │ Matters  │ Transfer │ For      │
  ├───────────────┼──────────┼──────────┼──────────┼──────────┤
  │ Single        │ None     │ N/A      │ None     │ Baseline │
  │ Stacking      │ None     │ Yes      │ Some     │ Simple   │
  │ Fusion        │ Fusion   │ No       │ Strong   │ Multi    │
  │ Averaging     │ None     │ No       │ Weak     │ Similar  │
  │ Splitting     │ None     │ N/A      │ Some     │ Diverse  │
  │ MAD-X         │ None     │ Yes      │ Strong   │ Cross-   │
  │               │          │          │          │ lingual  │
  └───────────────┴──────────┴──────────┴──────────┴──────────┘
"""
        print(patterns)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run AdapterFusion demonstrations."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║            ADAPTERFUSION: COMBINING ADAPTERS                 ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    demo = FusionDemonstration()
    
    # Full pipeline demonstration
    demo.demonstrate_full_pipeline(hidden_size=256)
    
    # Compare fusion vs stacking
    demo.compare_fusion_vs_stack(hidden_size=256)
    
    # Composition patterns
    demo.demonstrate_adapter_composition_patterns()
    
    print("\n" + "=" * 70)
    print("  MODULE COMPLETE")
    print("=" * 70)
    print("""
    Covered in this module:
    ✓ AdapterFusion mechanism (attention-based)
    ✓ Query/Key/Value projections for fusion
    ✓ Fusion training pipeline
    ✓ Attention weight analysis
    ✓ Fusion vs Stacking comparison
    ✓ 6 adapter composition patterns
    ✓ MAD-X cross-lingual composition
    """)


if __name__ == "__main__":
    main()
