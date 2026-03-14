"""
Knowledge Distillation - From Scratch Implementation
====================================================

PyTorch implementation of teacher-student distillation for LLMs.

Sections:
    1. Teacher & Student Model Setup
    2. Soft/Hard Target Losses & Temperature Scaling
    3. Intermediate Feature Distillation
    4. Self-Distillation
    5. Training Loop Example
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import random
import numpy as np

# =============================================================================
# SECTION 1: Teacher & Student Model Setup
# =============================================================================

class SimpleTransformerLM(nn.Module):
    """
    Minimal transformer language model for distillation demo.
    """
    def __init__(self, vocab_size=1000, d_model=128, n_layers=2, n_heads=4, max_seq_len=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        h = self.encoder(x)
        h = self.ln(h)
        logits = self.lm_head(h)
        return logits, h  # Return hidden for feature distillation

# =============================================================================
# SECTION 2: Soft/Hard Target Losses & Temperature Scaling
# =============================================================================

def distillation_loss(student_logits, teacher_logits, targets, temperature=2.0, alpha=0.5):
    """
    Compute combined distillation loss:
    - Soft target loss: KL(student, teacher) at temperature T
    - Hard target loss: Cross-entropy with ground-truth
    """
    # Soft targets
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_probs = F.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = F.kl_div(s_log_probs, t_probs, reduction='batchmean') * (temperature ** 2)
    # Hard targets
    hard_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), targets.view(-1), ignore_index=-100)
    # Combine
    return alpha * hard_loss + (1 - alpha) * soft_loss

# =============================================================================
# SECTION 3: Intermediate Feature Distillation
# =============================================================================

def feature_distillation_loss(student_hidden, teacher_hidden):
    """
    MSE loss between student and teacher hidden states.
    """
    # Optionally map student to teacher size if needed
    if student_hidden.shape != teacher_hidden.shape:
        min_dim = min(student_hidden.shape[-1], teacher_hidden.shape[-1])
        student_hidden = student_hidden[..., :min_dim]
        teacher_hidden = teacher_hidden[..., :min_dim]
    return F.mse_loss(student_hidden, teacher_hidden)

# =============================================================================
# SECTION 4: Self-Distillation
# =============================================================================

def self_distillation_loss(hidden_states):
    """
    Self-distillation: deeper layers teach shallower layers.
    hidden_states: List of [layer_hidden] from shallowest to deepest
    """
    loss = 0.0
    n = len(hidden_states)
    for i in range(n-1):
        loss += F.mse_loss(hidden_states[i], hidden_states[-1].detach())
    return loss / (n-1)

# =============================================================================
# SECTION 5: Training Loop Example
# =============================================================================

class ToyTextDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=32, vocab_size=1000):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.targets = self.data.clone()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def train_distillation_demo():
    print("="*60)
    print("KNOWLEDGE DISTILLATION FROM SCRATCH DEMO")
    print("="*60)
    torch.manual_seed(42)
    # Teacher: 4 layers, Student: 2 layers
    teacher = SimpleTransformerLM(n_layers=4)
    student = SimpleTransformerLM(n_layers=2)
    teacher.eval()
    # Data
    dataset = ToyTextDataset(num_samples=200, seq_len=16)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    # Training loop
    for epoch in range(2):
        for batch in loader:
            input_ids, targets = batch
            with torch.no_grad():
                t_logits, t_hidden = teacher(input_ids)
            s_logits, s_hidden = student(input_ids)
            # Distillation loss
            loss = distillation_loss(s_logits, t_logits, targets, temperature=2.0, alpha=0.5)
            # Feature distillation
            feat_loss = feature_distillation_loss(s_hidden, t_hidden)
            total_loss = loss + 0.2 * feat_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: Loss={total_loss.item():.4f}")
    print("Distillation training complete.")

if __name__ == "__main__":
    train_distillation_demo()
