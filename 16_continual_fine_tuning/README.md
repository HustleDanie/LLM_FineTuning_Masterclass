# Concept 16: Continual Fine-Tuning (Lifelong Learning for LLMs)

## Overview

**Continual Fine-Tuning** addresses the challenge of adapting an LLM to a sequence of tasks or domains over time **without catastrophically forgetting** previously learned knowledge. This is essential for production systems that need to evolve, absorb new information, and handle new tasks while retaining their original capabilities.

```
Standard Fine-Tuning (one task):
  Base Model ──→ Task A ──→ Model_A ✓

Continual Fine-Tuning (sequential tasks):
  Base Model ──→ Task A ──→ Task B ──→ Task C ──→ Model_ABC
                    ✓          ✓         ✓
              Still works  Still works  Works!
              on A!        on A & B!
```

## The Core Problem: Catastrophic Forgetting

When a neural network is trained on a new task, it tends to **overwrite** weights important for previous tasks:

```
Without Continual Learning:
  Task A accuracy: 95% → Train on B → Task A accuracy: 40% ✗ (forgotten!)
  Task B accuracy: --  → Train on B → Task B accuracy: 92% ✓

With Continual Learning:
  Task A accuracy: 95% → Train on B → Task A accuracy: 88% ✓ (preserved!)
  Task B accuracy: --  → Train on B → Task B accuracy: 90% ✓
```

## Three Families of Approaches

### 1. Regularization-Based Methods
Constrain weight changes to protect important parameters.

| Method | Mechanism | Key Formula |
|--------|-----------|-------------|
| **EWC** (Elastic Weight Consolidation) | Penalize changes to important weights | L = L_task + λ·Σ F_i·(θ_i - θ*_i)² |
| **SI** (Synaptic Intelligence) | Online importance from path integral | Ω_i = Σ (gradient · Δθ) / Δθ² |
| **MAS** (Memory Aware Synapses) | Importance from output sensitivity | Ω_i = E[|∂L/∂θ_i|] on unlabeled data |
| **L2-SP** | L2 regularization toward pretrained | L = L_task + λ·‖θ - θ_pre‖² |
| **R-EWC** (Rotated EWC) | EWC in reparameterized space | Rotate to diagonalize Fisher |

### 2. Replay-Based Methods
Retain and replay examples from previous tasks.

| Method | Mechanism |
|--------|-----------|
| **Experience Replay** | Store subset of old data, mix into training |
| **Generative Replay** | Use the model itself to generate old-task data |
| **Dark Experience Replay** | Store logits (soft labels) from old model |
| **Gradient Episodic Memory** | Project gradients to not conflict with old data |

### 3. Architecture-Based Methods
Allocate separate parameters for different tasks.

| Method | Mechanism |
|--------|-----------|
| **Progressive Networks** | Add new columns/modules per task |
| **PackNet** | Prune + freeze subsets per task |
| **Task-Specific Adapters** | LoRA/Adapter per task, shared backbone |
| **Mixture of LoRAs** | Route to task-specific LoRA modules |
| **Model Merging** | Merge separately fine-tuned models |

## LLM-Specific Considerations

```
Why LLMs Are Special:
┌─────────────────────────────────────────────────────────────┐
│ 1. MASSIVE PARAMETER SPACE                                   │
│    Billions of parameters → many "free" parameters            │
│    Forgetting less severe than small models (but still real)  │
│                                                               │
│ 2. PRETRAINED KNOWLEDGE IS VALUABLE                           │
│    General language understanding must be preserved            │
│    Domain adaptation should ADD, not REPLACE                  │
│                                                               │
│ 3. PEFT METHODS HELP                                         │
│    LoRA/Adapters naturally limit weight changes               │
│    Task-specific adapters = architecture-based continual      │
│                                                               │
│ 4. INSTRUCTION TUNING CREATES FRAGILITY                      │
│    Instruction-following can be lost with domain fine-tuning  │
│    Need to preserve both knowledge AND capabilities           │
└─────────────────────────────────────────────────────────────┘
```

## Key Metrics for Continual Learning

| Metric | Formula | Measures |
|--------|---------|----------|
| **Average Accuracy** | (1/T)·Σ A_{T,j} | Performance across all tasks after training on T tasks |
| **Backward Transfer** | (1/T-1)·Σ (A_{T,j} - A_{j,j}) | How much old tasks are affected (negative = forgetting) |
| **Forward Transfer** | (1/T-1)·Σ (A_{j-1,j} - baseline_j) | How previous tasks help new ones |
| **Forgetting Measure** | max_k(A_{k,j}) - A_{T,j} | Worst-case forgetting per task |

Where A_{i,j} = accuracy on task j after training up to task i.

## Files in This Module

| File | Description |
|------|-------------|
| `continual_ft_theory.py` | Catastrophic forgetting theory, Fisher Information, importance estimation |
| `continual_ft_from_scratch.py` | EWC, SI, experience replay, generative replay from scratch |
| `continual_ft_training.py` | Practical continual FT with LoRA adapters, data mixing, HuggingFace |
| `continual_ft_comparison.py` | Regularization vs replay vs architecture methods, decision guide |

## References

- Kirkpatrick et al., 2017 — "Overcoming Catastrophic Forgetting in Neural Networks" (EWC)
- Zenke et al., 2017 — "Continual Learning through Synaptic Intelligence" (SI)
- Shin et al., 2017 — "Continual Learning with Deep Generative Replay"
- Rebuffi et al., 2017 — "Learning Multiple Visual Domains with Residual Adapters"
- Ke et al., 2023 — "Continual Pre-training of Language Models"
- Wu et al., 2024 — "LLaMA Pro: Progressive LLaMA with Block Expansion"
- Yadav et al., 2024 — "TIES-Merging: Resolving Interference When Merging Models"
