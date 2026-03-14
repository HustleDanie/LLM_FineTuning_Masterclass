"""
Knowledge Distillation - HuggingFace Training Pipeline
======================================================

Production pipeline for LLM distillation using HuggingFace Transformers.

Sections:
    1. Data Preparation
    2. Teacher & Student Model Setup
    3. Distillation Trainer (Soft/Hard/Feature Loss)
    4. Evaluation
    5. End-to-End Example
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any
import numpy as np
import random

# =============================================================================
# SECTION 1: Data Preparation
# =============================================================================

class KDDataset(Dataset):
    """Simple dataset for distillation demo."""
    def __init__(self, texts: List[str], tokenizer, max_length=64):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)
    def __len__(self):
        return len(self.encodings['input_ids'])
    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}

# =============================================================================
# SECTION 2: Teacher & Student Model Setup
# =============================================================================

def load_teacher_student(teacher_name="gpt2", student_name="distilgpt2"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    teacher = AutoModelForCausalLM.from_pretrained(teacher_name)
    student = AutoModelForCausalLM.from_pretrained(student_name)
    tokenizer = AutoTokenizer.from_pretrained(student_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return teacher, student, tokenizer

# =============================================================================
# SECTION 3: Distillation Trainer (Soft/Hard/Feature Loss)
# =============================================================================

def kd_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    import torch.nn.functional as F
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_probs = F.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = F.kl_div(s_log_probs, t_probs, reduction='batchmean') * (temperature ** 2)
    hard_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1), ignore_index=-100)
    return alpha * hard_loss + (1 - alpha) * soft_loss

class DistillationTrainer:
    """
    Custom distillation trainer for HuggingFace models.
    """
    def __init__(self, teacher, student, tokenizer, temperature=2.0, alpha=0.5, lr=2e-4):
        self.teacher = teacher.eval()
        self.student = student.train()
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.alpha = alpha
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)
    def train(self, dataset, epochs=1, batch_size=4):
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for batch in loader:
                input_ids = batch['input_ids']
                labels = batch['input_ids']
                with torch.no_grad():
                    t_out = self.teacher(input_ids)[0]
                s_out = self.student(input_ids)[0]
                loss = kd_loss(s_out, t_out, labels, self.temperature, self.alpha)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")

# =============================================================================
# SECTION 4: Evaluation
# =============================================================================

def evaluate_model(model, tokenizer, texts: List[str]):
    model.eval()
    with torch.no_grad():
        for text in texts[:3]:
            inputs = tokenizer(text, return_tensors='pt')
            outputs = model.generate(**inputs, max_new_tokens=20)
            print(f"Prompt: {text}")
            print(f"Output: {tokenizer.decode(outputs[0])}\n")

# =============================================================================
# SECTION 5: End-to-End Example
# =============================================================================

def run_kd_pipeline():
    print("="*60)
    print("KNOWLEDGE DISTILLATION - HUGGINGFACE PIPELINE DEMO")
    print("="*60)
    # Data
    texts = [
        "The capital of France is",
        "The largest planet in the solar system is",
        "The author of Harry Potter is",
        "The boiling point of water is",
    ]
    # Load models
    teacher, student, tokenizer = load_teacher_student()
    # Prepare dataset
    dataset = KDDataset(texts * 10, tokenizer)
    # Train
    trainer = DistillationTrainer(teacher, student, tokenizer)
    trainer.train(dataset, epochs=1, batch_size=2)
    # Evaluate
    print("\nStudent model generations:")
    evaluate_model(student, tokenizer, texts)

if __name__ == "__main__":
    run_kd_pipeline()
