# Knowledge Distillation for LLMs

This module covers the theory, implementation, and benchmarking of knowledge distillation for large language models (LLMs).

**Knowledge Distillation** is a technique where a smaller (student) model is trained to mimic the behavior of a larger (teacher) model, transferring knowledge for efficiency, compression, or privacy.

## Sections
1. **Theory**: Distillation objectives, loss functions, teacher-student architectures, soft vs hard targets, intermediate feature matching, self-distillation, and recent advances (TinyStories, MiniLLM, LLM-Pruner).
2. **From Scratch**: PyTorch implementation of teacher-student distillation, soft/hard loss, temperature scaling, intermediate layer distillation, and self-distillation.
3. **Training Pipeline**: HuggingFace-based pipeline for distillation (AutoModelForCausalLM, DataCollator, Trainer), dataset preparation, and evaluation.
4. **Comparison**: Benchmarks: full FT vs distillation, ablation (temperature, loss, intermediate), compression/latency/accuracy trade-offs, and decision framework.

## Key References
- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- TinyStories: https://arxiv.org/abs/2305.07759
- MiniLLM: https://arxiv.org/abs/2309.03153
- LLM-Pruner: https://arxiv.org/abs/2310.07243
- DistilBERT: https://arxiv.org/abs/1910.01108
- Self-Distillation: https://arxiv.org/abs/1905.09788

---

**Folder Structure:**
- `kd_theory.py` — Theory and math of distillation
- `kd_from_scratch.py` — PyTorch implementation from scratch
- `kd_training.py` — HuggingFace pipeline for distillation
- `kd_comparison.py` — Benchmarks, ablations, and decision framework
