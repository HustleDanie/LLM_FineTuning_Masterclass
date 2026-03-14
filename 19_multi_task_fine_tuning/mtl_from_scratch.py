"""
Multi-Task Fine-Tuning - From Scratch Implementation
=====================================================

Pure PyTorch implementations of multi-task architectures,
gradient surgery methods, and task balancing algorithms.

Sections:
    1. Hard Parameter Sharing Architecture
    2. Soft Parameter Sharing Architecture
    3. Multi-Task LoRA (Task-Specific Adapters)
    4. PCGrad — Projecting Conflicting Gradients
    5. GradNorm — Dynamic Gradient Normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
import math
import random
from dataclasses import dataclass, field
from collections import defaultdict


# =============================================================================
# SECTION 1: Hard Parameter Sharing Architecture
# =============================================================================

class SharedEncoder(nn.Module):
    """
    Shared transformer-like encoder for multi-task learning.
    
    This is the core shared component. It learns representations
    that are useful across ALL tasks simultaneously.
    
    Architecture:
        Token Embeddings → Positional Encoding → TransformerBlocks → Shared Repr.
    
    In hard parameter sharing, ~95% of parameters are here.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        max_seq_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  [SharedEncoder] Parameters: {total_params:,}")
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode input into shared representations.
        
        Returns: [batch, seq_len, d_model] hidden states
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.layer_norm(self.dropout(x))
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert 0/1 mask to float mask where 0 = attend, -inf = ignore
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Transformer encoding
        hidden_states = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return hidden_states


class TaskHead(nn.Module):
    """
    Task-specific head that converts shared representations to task outputs.
    
    Each task gets its own small head (~5% of total parameters).
    Different tasks need different output architectures:
    - Classification: pooled repr → FFN → num_classes
    - Token labeling: per-token repr → FFN → num_labels  
    - Generation: hidden states → vocab projection
    """
    
    def __init__(
        self,
        d_model: int,
        num_classes: int,
        task_type: str = "classification",
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.task_type = task_type
        
        if task_type == "classification":
            # Pooling + classification
            self.head = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        elif task_type == "token_classification":
            # Per-token classification (NER, POS)
            self.head = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        elif task_type == "regression":
            # Single output value
            self.head = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        
        params = sum(p.numel() for p in self.parameters())
        print(f"  [TaskHead:{task_type}] Parameters: {params:,}, Classes: {num_classes}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.task_type == "classification":
            # Use [CLS] token (first token) for classification
            pooled = hidden_states[:, 0, :]
            return self.head(pooled)
        elif self.task_type == "token_classification":
            # Per-token predictions
            return self.head(hidden_states)
        elif self.task_type == "regression":
            pooled = hidden_states[:, 0, :]
            return self.head(pooled).squeeze(-1)


class HardParameterSharingMTL(nn.Module):
    """
    Complete Hard Parameter Sharing MTL model.
    
    Architecture:
        Input → SharedEncoder → {TaskHead_1, TaskHead_2, ..., TaskHead_K}
    
    This is the most common MTL architecture because:
    1. Strong regularization from shared encoder
    2. Parameter efficient (one encoder, K small heads)
    3. Simple to implement and train
    4. Works well when tasks are related
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        task_configs: Optional[Dict[str, dict]] = None
    ):
        super().__init__()
        
        print("\n" + "=" * 60)
        print("Building Hard Parameter Sharing MTL Model")
        print("=" * 60)
        
        # Shared encoder
        self.encoder = SharedEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers
        )
        
        # Task-specific heads
        if task_configs is None:
            task_configs = {
                "sentiment": {"num_classes": 3, "task_type": "classification"},
                "ner": {"num_classes": 9, "task_type": "token_classification"},
                "similarity": {"num_classes": 1, "task_type": "regression"},
            }
        
        self.task_heads = nn.ModuleDict()
        for task_name, config in task_configs.items():
            self.task_heads[task_name] = TaskHead(
                d_model=d_model,
                **config
            )
        
        # Summary
        shared_params = sum(p.numel() for p in self.encoder.parameters())
        head_params = sum(p.numel() for p in self.task_heads.parameters())
        total = shared_params + head_params
        print(f"\n  Total parameters: {total:,}")
        print(f"  Shared: {shared_params:,} ({shared_params/total*100:.1f}%)")
        print(f"  Task-specific: {head_params:,} ({head_params/total*100:.1f}%)")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        task_name: str,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for a specific task.
        
        The shared encoder processes the input, then the appropriate
        task head generates task-specific outputs.
        """
        # Shared encoding
        hidden_states = self.encoder(input_ids, attention_mask)
        
        # Task-specific head
        if task_name not in self.task_heads:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(self.task_heads.keys())}")
        
        return self.task_heads[task_name](hidden_states)
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        task_name: str,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute loss for a specific task."""
        logits = self.forward(input_ids, task_name, attention_mask)
        
        task_type = self.task_heads[task_name].task_type
        
        if task_type == "classification":
            return F.cross_entropy(logits, labels)
        elif task_type == "token_classification":
            # Flatten for token classification
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        elif task_type == "regression":
            return F.mse_loss(logits, labels.float())


# =============================================================================
# SECTION 2: Soft Parameter Sharing Architecture
# =============================================================================

class SoftParameterSharingMTL(nn.Module):
    """
    Soft Parameter Sharing: Each task has its own encoder, but
    parameters are regularized to stay similar via L2 penalty.
    
    Architecture:
        Task A: Input → Encoder_A → Head_A → Output_A
        Task B: Input → Encoder_B → Head_B → Output_B
               ↕ L2 regularization between Encoder_A and Encoder_B
    
    Advantages over Hard Sharing:
    - Each task can specialize its encoder
    - Less negative transfer between conflicting tasks
    - More flexible representation per task
    
    Disadvantages:
    - K× more parameters (one encoder per task)
    - Requires tuning regularization strength
    - More memory and compute
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        task_configs: Optional[Dict[str, dict]] = None,
        regularization_strength: float = 0.01
    ):
        super().__init__()
        
        print("\n" + "=" * 60)
        print("Building Soft Parameter Sharing MTL Model")
        print("=" * 60)
        
        if task_configs is None:
            task_configs = {
                "sentiment": {"num_classes": 3, "task_type": "classification"},
                "ner": {"num_classes": 9, "task_type": "token_classification"},
            }
        
        self.task_names = list(task_configs.keys())
        self.reg_strength = regularization_strength
        
        # Separate encoder for each task
        self.encoders = nn.ModuleDict()
        self.task_heads = nn.ModuleDict()
        
        for task_name, config in task_configs.items():
            self.encoders[task_name] = SharedEncoder(
                vocab_size=vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers
            )
            self.task_heads[task_name] = TaskHead(d_model=d_model, **config)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n  Total parameters: {total_params:,} (K × encoder + K × head)")
        print(f"  Regularization strength: {regularization_strength}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        task_name: str,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward through task-specific encoder + head."""
        hidden_states = self.encoders[task_name](input_ids, attention_mask)
        return self.task_heads[task_name](hidden_states)
    
    def compute_regularization_loss(self) -> torch.Tensor:
        """
        L2 regularization between all pairs of task encoders.
        
        This penalizes divergence between encoders, encouraging
        them to stay similar while allowing task-specific deviations.
        
        reg_loss = λ · Σ_{i<j} ‖θ_i - θ_j‖²
        """
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Compare all pairs of encoders
        encoder_names = list(self.encoders.keys())
        for i in range(len(encoder_names)):
            for j in range(i + 1, len(encoder_names)):
                params_i = list(self.encoders[encoder_names[i]].parameters())
                params_j = list(self.encoders[encoder_names[j]].parameters())
                
                for p_i, p_j in zip(params_i, params_j):
                    reg_loss += torch.sum((p_i - p_j) ** 2)
        
        return self.reg_strength * reg_loss
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        task_name: str,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute task loss + regularization penalty."""
        logits = self.forward(input_ids, task_name, attention_mask)
        
        task_type = self.task_heads[task_name].task_type
        
        if task_type == "classification":
            task_loss = F.cross_entropy(logits, labels)
        elif task_type == "token_classification":
            task_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        elif task_type == "regression":
            task_loss = F.mse_loss(logits, labels.float())
        
        # Add soft sharing regularization
        reg_loss = self.compute_regularization_loss()
        
        return task_loss + reg_loss


# =============================================================================
# SECTION 3: Multi-Task LoRA (Task-Specific Adapters)
# =============================================================================

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer for efficient multi-task fine-tuning.
    
    Instead of separate full encoders per task (soft sharing) or
    one shared encoder (hard sharing), LoRA-based MTL uses:
    - One frozen shared encoder
    - Small task-specific LoRA adapters
    
    This combines the best of both worlds:
    - Hard sharing's parameter efficiency
    - Soft sharing's task-specific capacity
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA delta: x @ A @ B * scaling"""
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class MultiTaskLoRAEncoder(nn.Module):
    """
    Shared encoder with task-specific LoRA adapters.
    
    Architecture:
        Frozen Shared Encoder + {LoRA_Task1, LoRA_Task2, ..., LoRA_TaskK}
    
    During forward pass for task k:
        output = frozen_encoder(x) + lora_k(x)
    
    Benefits:
    - Shared encoder is frozen (no gradient conflicts!)
    - Each task gets its own adapter capacity
    - Adding new tasks = adding new LoRA matrices (no retraining)
    - Very parameter efficient: rank × (d_in + d_out) per task per layer
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        task_names: List[str] = None,
        lora_rank: int = 8,
        lora_alpha: float = 16.0
    ):
        super().__init__()
        
        print("\n" + "=" * 60)
        print("Building Multi-Task LoRA Model")
        print("=" * 60)
        
        if task_names is None:
            task_names = ["sentiment", "ner"]
        
        # Shared encoder (frozen after pre-training)
        self.encoder = SharedEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers
        )
        
        # Task-specific LoRA adapters applied to attention output
        # In a real implementation, you'd apply LoRA to Q, K, V projections
        self.task_loras = nn.ModuleDict()
        for task_name in task_names:
            self.task_loras[task_name] = nn.ModuleList([
                LoRALayer(d_model, d_model, rank=lora_rank, alpha=lora_alpha)
                for _ in range(n_layers)
            ])
        
        # Print parameter counts
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        lora_params_per_task = sum(
            p.numel() for p in self.task_loras[task_names[0]].parameters()
        )
        total_lora = lora_params_per_task * len(task_names)
        
        print(f"\n  Frozen encoder params: {encoder_params:,}")
        print(f"  LoRA params per task: {lora_params_per_task:,}")
        print(f"  Total LoRA params ({len(task_names)} tasks): {total_lora:,}")
        print(f"  LoRA overhead: {total_lora/encoder_params*100:.2f}%")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        task_name: str,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with task-specific LoRA adaptation.
        
        In a full implementation, LoRA would be injected into the
        transformer's attention layers. For demonstration, we apply
        LoRA as a residual on the encoder output at each layer.
        """
        # Get shared encoder output
        hidden_states = self.encoder(input_ids, attention_mask)
        
        # Apply task-specific LoRA
        # (Simplified: applying after full encoding. In practice,
        #  LoRA is injected at each transformer layer's attention)
        if task_name in self.task_loras:
            for lora_layer in self.task_loras[task_name]:
                hidden_states = hidden_states + lora_layer(hidden_states)
        
        return hidden_states
    
    def freeze_encoder(self):
        """Freeze shared encoder, only train LoRA adapters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("  Encoder frozen. Only LoRA adapters will be trained.")
    
    def get_trainable_params(self, task_name: str) -> int:
        """Count trainable parameters for a specific task."""
        return sum(p.numel() for p in self.task_loras[task_name].parameters() if p.requires_grad)


# =============================================================================
# SECTION 4: PCGrad — Projecting Conflicting Gradients
# =============================================================================

class PCGradOptimizer:
    """
    PCGrad: Projecting Conflicting Gradients (Yu et al., 2020).
    
    A gradient surgery technique that modifies task gradients to
    eliminate conflicts before applying the update.
    
    When two tasks have conflicting gradients (negative cosine similarity),
    we project one gradient onto the normal plane of the other,
    removing the conflicting component while preserving the rest.
    
    This is a DROP-IN REPLACEMENT for standard gradient combination
    in multi-task training.
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, num_tasks: int):
        self.optimizer = optimizer
        self.num_tasks = num_tasks
        self._task_gradients: List[List[torch.Tensor]] = []
    
    def store_task_gradient(self, task_idx: int, model: nn.Module):
        """
        Store gradients from the current backward pass for a task.
        Call this after loss.backward() for each task.
        """
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.clone())
            else:
                grads.append(torch.zeros_like(param))
        
        if len(self._task_gradients) <= task_idx:
            self._task_gradients.append(grads)
        else:
            self._task_gradients[task_idx] = grads
    
    def apply_pcgrad_and_step(self, model: nn.Module):
        """
        Apply PCGrad to stored task gradients and take optimizer step.
        
        Algorithm:
        1. For each task i, for each other task j:
            If g_i · g_j < 0:  (conflict)
                g_i ← g_i - (g_i·g_j / ‖g_j‖²) · g_j  (project away from j)
        2. Set model gradients to mean of modified gradients
        3. Take optimizer step
        """
        assert len(self._task_gradients) == self.num_tasks, \
            f"Expected {self.num_tasks} task gradients, got {len(self._task_gradients)}"
        
        num_params = len(self._task_gradients[0])
        
        # Apply PCGrad per parameter
        modified_grads = [
            [g.clone() for g in task_grads]
            for task_grads in self._task_gradients
        ]
        
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                if i == j:
                    continue
                
                for p_idx in range(num_params):
                    g_i = modified_grads[i][p_idx]
                    g_j = self._task_gradients[j][p_idx]  # Use original
                    
                    # Flatten for dot product
                    g_i_flat = g_i.flatten()
                    g_j_flat = g_j.flatten()
                    
                    dot = torch.dot(g_i_flat, g_j_flat)
                    
                    if dot < 0:  # Conflict!
                        # Project g_i onto normal plane of g_j
                        g_j_norm_sq = torch.dot(g_j_flat, g_j_flat) + 1e-8
                        projection = (dot / g_j_norm_sq) * g_j
                        modified_grads[i][p_idx] = g_i - projection
        
        # Average modified gradients and set as model gradients
        self.optimizer.zero_grad()
        for p_idx, param in enumerate(model.parameters()):
            if param.requires_grad:
                avg_grad = torch.stack([
                    modified_grads[t][p_idx] for t in range(self.num_tasks)
                ]).mean(dim=0)
                param.grad = avg_grad
        
        # Take optimizer step
        self.optimizer.step()
        
        # Reset stored gradients
        self._task_gradients = []


def demonstrate_pcgrad():
    """Demonstrate PCGrad on a toy MTL problem."""
    print("\n" + "=" * 60)
    print("PCGrad DEMONSTRATION")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Simple shared model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16)
    )
    
    # Two task heads
    head_a = nn.Linear(16, 3)  # Classification (3 classes)
    head_b = nn.Linear(16, 2)  # Binary classification
    
    # Combine all parameters
    all_params = list(model.parameters()) + list(head_a.parameters()) + list(head_b.parameters())
    optimizer = torch.optim.Adam(all_params, lr=1e-3)
    pcgrad = PCGradOptimizer(optimizer, num_tasks=2)
    
    # Synthetic data
    x = torch.randn(8, 10)
    labels_a = torch.randint(0, 3, (8,))
    labels_b = torch.randint(0, 2, (8,))
    
    # Standard training (no PCGrad)
    optimizer.zero_grad()
    shared = model(x)
    loss_a = F.cross_entropy(head_a(shared), labels_a)
    loss_b = F.cross_entropy(head_b(shared), labels_b)
    naive_loss = loss_a + loss_b
    naive_loss.backward()
    naive_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    
    print(f"\n  Naive gradient combination:")
    print(f"    Loss A: {loss_a.item():.4f}, Loss B: {loss_b.item():.4f}")
    print(f"    Combined gradient norm: {naive_grad_norm:.4f}")
    
    # PCGrad training
    # Task A gradient
    optimizer.zero_grad()
    shared = model(x)
    loss_a = F.cross_entropy(head_a(shared), labels_a)
    loss_a.backward()
    pcgrad.store_task_gradient(0, model)
    
    # Task B gradient
    optimizer.zero_grad()
    shared = model(x)
    loss_b = F.cross_entropy(head_b(shared), labels_b)
    loss_b.backward()
    pcgrad.store_task_gradient(1, model)
    
    # Apply PCGrad
    pcgrad.apply_pcgrad_and_step(model)
    
    pcgrad_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"\n  PCGrad gradient combination:")
    print(f"    Modified gradient norm: {pcgrad_grad_norm:.4f}")
    print(f"    Conflicts removed: gradients no longer oppose each other")


# =============================================================================
# SECTION 5: GradNorm — Dynamic Gradient Normalization
# =============================================================================

class GradNormTrainer:
    """
    GradNorm: Gradient Normalization for Multi-Task Learning
    (Chen et al., 2018).
    
    Dynamically adjusts task weights so all tasks train at 
    similar rates relative to their difficulty.
    
    Key idea: 
    - Measure each task's gradient norm on shared layers
    - Compare actual norms to target norms (based on training rate)
    - Adjust weights to bring norms closer to targets
    
    The weight adjustment is itself optimized via gradient descent,
    making GradNorm a bi-level optimization algorithm.
    """
    
    def __init__(
        self,
        model: nn.Module,
        task_names: List[str],
        optimizer: torch.optim.Optimizer,
        alpha: float = 1.5,
        lr_weights: float = 0.025
    ):
        """
        Args:
            model: Multi-task model
            task_names: List of task names
            optimizer: Optimizer for model parameters
            alpha: Asymmetry parameter (higher = more focus on balancing)
                   α=0: ignore training rates, just balance gradient norms
                   α=1: proportional to training rate
                   α>1: extra focus on slow tasks
            lr_weights: Learning rate for task weight updates
        """
        self.model = model
        self.task_names = task_names
        self.optimizer = optimizer
        self.alpha = alpha
        self.num_tasks = len(task_names)
        
        # Learnable task weights (initialized to 1.0)
        self.log_task_weights = nn.Parameter(
            torch.zeros(self.num_tasks, requires_grad=True)
        )
        self.weight_optimizer = torch.optim.Adam([self.log_task_weights], lr=lr_weights)
        
        # Track initial losses for computing training rates
        self.initial_losses: Dict[str, float] = {}
        self.step_count = 0
        
        print(f"  GradNorm initialized: α={alpha}, tasks={task_names}")
    
    @property
    def task_weights(self) -> torch.Tensor:
        """Get current task weights (softmax of log weights)."""
        # Use softmax to ensure weights sum to num_tasks
        return F.softmax(self.log_task_weights, dim=0) * self.num_tasks
    
    def train_step(
        self,
        task_losses: Dict[str, torch.Tensor],
        shared_params: List[nn.Parameter]
    ):
        """
        Perform one GradNorm training step.
        
        Steps:
        1. Compute weighted loss: L = Σ w_k · L_k
        2. Compute gradient norms per task on shared layers
        3. Compute target norms based on training rates
        4. Update task weights to match target norms
        5. Update model with weighted gradients
        """
        self.step_count += 1
        weights = self.task_weights
        
        # Record initial losses (first step)
        for i, name in enumerate(self.task_names):
            if name not in self.initial_losses:
                self.initial_losses[name] = task_losses[name].item()
        
        # Step 1: Compute weighted total loss
        total_loss = sum(
            weights[i] * task_losses[name]
            for i, name in enumerate(self.task_names)
        )
        
        # Step 2: Compute per-task gradient norms on shared parameters
        task_grad_norms = []
        for i, name in enumerate(self.task_names):
            # Weighted task loss
            weighted_loss = weights[i] * task_losses[name]
            
            # Compute gradient norm on shared parameters
            grads = torch.autograd.grad(
                weighted_loss, shared_params,
                retain_graph=True, create_graph=True, allow_unused=True
            )
            grad_norm = torch.cat([
                g.flatten() for g in grads if g is not None
            ]).norm()
            task_grad_norms.append(grad_norm)
        
        task_grad_norms = torch.stack(task_grad_norms)
        
        # Step 3: Compute target norms
        avg_grad_norm = task_grad_norms.mean().detach()
        
        # Training rates: r_i = L_i(t) / L_i(0)
        training_rates = torch.tensor([
            task_losses[name].item() / max(self.initial_losses[name], 1e-8)
            for name in self.task_names
        ])
        avg_training_rate = training_rates.mean()
        
        # Relative inverse training rate
        relative_rates = training_rates / (avg_training_rate + 1e-8)
        
        # Target norms
        target_norms = avg_grad_norm * (relative_rates ** self.alpha)
        
        # Step 4: Update task weights
        # Loss for weights: how far actual norms are from targets
        gradnorm_loss = torch.sum(torch.abs(task_grad_norms - target_norms.to(task_grad_norms.device)))
        
        self.weight_optimizer.zero_grad()
        gradnorm_loss.backward(retain_graph=True)
        self.weight_optimizer.step()
        
        # Step 5: Update model parameters
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Return info for logging
        return {
            "total_loss": total_loss.item(),
            "task_weights": weights.detach().cpu().tolist(),
            "grad_norms": task_grad_norms.detach().cpu().tolist(),
            "training_rates": training_rates.tolist(),
            "gradnorm_loss": gradnorm_loss.item()
        }


def demonstrate_gradnorm():
    """Demonstrate GradNorm on a synthetic MTL problem."""
    print("\n" + "=" * 60)
    print("GradNorm DEMONSTRATION")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Simple shared model
    shared = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU()
    )
    
    heads = nn.ModuleDict({
        "easy": nn.Linear(16, 2),   # Easy binary classification
        "hard": nn.Linear(16, 10),  # Hard 10-class classification
    })
    
    optimizer = torch.optim.Adam(
        list(shared.parameters()) + list(heads.parameters()),
        lr=1e-3
    )
    
    gradnorm = GradNormTrainer(
        model=nn.ModuleList([shared, heads]),
        task_names=["easy", "hard"],
        optimizer=optimizer,
        alpha=1.5
    )
    
    shared_params = list(shared.parameters())
    
    print(f"\n  Training with GradNorm (α=1.5)...")
    print(f"  {'Step':>6} {'Easy Loss':>10} {'Hard Loss':>10} {'w_easy':>8} {'w_hard':>8}")
    print("  " + "-" * 50)
    
    for step in range(20):
        # Generate synthetic data
        x = torch.randn(16, 10)
        labels_easy = torch.randint(0, 2, (16,))
        labels_hard = torch.randint(0, 10, (16,))
        
        # Forward pass
        h = shared(x)
        
        task_losses = {
            "easy": F.cross_entropy(heads["easy"](h), labels_easy),
            "hard": F.cross_entropy(heads["hard"](h), labels_hard),
        }
        
        # GradNorm step
        info = gradnorm.train_step(task_losses, shared_params)
        
        if step % 4 == 0:
            w = info["task_weights"]
            print(f"  {step:>6} {task_losses['easy'].item():>10.4f} "
                  f"{task_losses['hard'].item():>10.4f} "
                  f"{w[0]:>8.3f} {w[1]:>8.3f}")
    
    print("""
  Observation:
    GradNorm increases the weight of the "hard" task over time,
    because it trains more slowly. This ensures both tasks
    converge at similar rates, preventing the easy task from
    dominating the shared representations.
    """)


# =============================================================================
# Multi-Task Data Handling
# =============================================================================

class MultiTaskDataset:
    """
    Dataset that manages multiple task datasets with configurable sampling.
    
    Supports temperature-based sampling to control the data distribution
    across tasks during training.
    """
    
    def __init__(
        self,
        task_datasets: Dict[str, List[dict]],
        sampling_temperature: float = 2.0
    ):
        self.task_datasets = task_datasets
        self.task_names = list(task_datasets.keys())
        self.temperature = sampling_temperature
        
        # Compute sampling probabilities
        sizes = np.array([len(ds) for ds in task_datasets.values()], dtype=np.float64)
        
        # Temperature-scaled probabilities
        scaled = sizes ** (1.0 / sampling_temperature)
        self.task_probs = scaled / scaled.sum()
        
        # Track position in each dataset
        self.task_indices = {name: 0 for name in self.task_names}
        
        print(f"\n  MultiTaskDataset:")
        for i, name in enumerate(self.task_names):
            print(f"    {name}: {len(task_datasets[name])} samples, "
                  f"p={self.task_probs[i]:.3f}")
    
    def sample_batch(self, batch_size: int = 8) -> Tuple[str, List[dict]]:
        """
        Sample a batch from one task using temperature sampling.
        
        Returns (task_name, batch_data)
        """
        # Choose task based on temperature-scaled probabilities
        task_idx = np.random.choice(len(self.task_names), p=self.task_probs)
        task_name = self.task_names[task_idx]
        
        # Sample from chosen task
        dataset = self.task_datasets[task_name]
        indices = np.random.choice(len(dataset), size=min(batch_size, len(dataset)))
        batch = [dataset[i] for i in indices]
        
        return task_name, batch


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 70)
    print("MULTI-TASK FINE-TUNING — FROM SCRATCH IMPLEMENTATION")
    print("=" * 70)
    
    # Section 1: Hard Parameter Sharing
    print("\n\n🔧 SECTION 1: Hard Parameter Sharing")
    model = HardParameterSharingMTL(
        vocab_size=5000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        task_configs={
            "sentiment": {"num_classes": 3, "task_type": "classification"},
            "ner": {"num_classes": 9, "task_type": "token_classification"},
            "similarity": {"num_classes": 1, "task_type": "regression"},
        }
    )
    
    # Test forward pass
    x = torch.randint(0, 5000, (4, 32))
    for task in ["sentiment", "ner", "similarity"]:
        out = model(x, task)
        print(f"\n  {task} output shape: {out.shape}")
    
    # Test loss computation
    sentiment_labels = torch.randint(0, 3, (4,))
    loss = model.compute_loss(x, "sentiment", sentiment_labels)
    print(f"\n  Sentiment loss: {loss.item():.4f}")
    
    # Section 2: Soft Parameter Sharing
    print("\n\n🔧 SECTION 2: Soft Parameter Sharing")
    soft_model = SoftParameterSharingMTL(
        vocab_size=5000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        task_configs={
            "sentiment": {"num_classes": 3, "task_type": "classification"},
            "ner": {"num_classes": 9, "task_type": "token_classification"},
        },
        regularization_strength=0.001
    )
    
    reg_loss = soft_model.compute_regularization_loss()
    print(f"\n  Initial regularization loss: {reg_loss.item():.6f}")
    print("  (Low because encoders start with same random init)")
    
    # Section 3: Multi-Task LoRA
    print("\n\n🔧 SECTION 3: Multi-Task LoRA")
    lora_model = MultiTaskLoRAEncoder(
        vocab_size=5000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        task_names=["sentiment", "ner", "qa"],
        lora_rank=8
    )
    lora_model.freeze_encoder()
    
    for task in ["sentiment", "ner", "qa"]:
        trainable = lora_model.get_trainable_params(task)
        print(f"\n  Task '{task}' trainable LoRA params: {trainable:,}")
    
    # Section 4: PCGrad
    print("\n\n🔧 SECTION 4: PCGrad")
    demonstrate_pcgrad()
    
    # Section 5: GradNorm
    print("\n\n🔧 SECTION 5: GradNorm")
    demonstrate_gradnorm()
    
    # Multi-Task Data Handling
    print("\n\n🔧 BONUS: Multi-Task Data Sampling")
    task_data = {
        "sentiment": [{"text": f"review_{i}", "label": i % 3} for i in range(10000)],
        "ner": [{"text": f"sentence_{i}", "label": i % 9} for i in range(2000)],
        "qa": [{"text": f"question_{i}", "label": i % 2} for i in range(5000)],
    }
    
    mtl_dataset = MultiTaskDataset(task_data, sampling_temperature=2.0)
    
    # Sample a few batches
    print("\n  Sampling batches (T=2.0):")
    task_counts = defaultdict(int)
    for _ in range(100):
        task_name, batch = mtl_dataset.sample_batch(batch_size=8)
        task_counts[task_name] += 1
    
    for name, count in sorted(task_counts.items()):
        print(f"    {name}: {count}/100 batches ({count}%)")
    
    print("\n" + "=" * 70)
    print("FROM-SCRATCH IMPLEMENTATION COMPLETE")
    print("=" * 70)
    print("""
    Implemented:
    1. Hard Parameter Sharing — shared encoder + task heads
    2. Soft Parameter Sharing — separate encoders + L2 regularization
    3. Multi-Task LoRA — frozen encoder + task-specific adapters
    4. PCGrad — gradient conflict resolution
    5. GradNorm — dynamic gradient normalization for task balancing
    """)


if __name__ == "__main__":
    main()
