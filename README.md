# LLM Fine-Tuning Masterclass

A comprehensive, hands-on project covering **21 fine-tuning techniques** for Large Language Models.
Each technique is implemented in its own folder with full code, explanations, and runnable examples.

## Concepts Covered

| # | Technique | Folder | Status |
|---|-----------|--------|--------|
| 1 | Full Fine-Tuning | `01_full_fine_tuning/` | ✅ |
| 2 | Supervised Fine-Tuning (SFT) | `02_supervised_fine_tuning/` | ✅ |
| 3 | Instruction Fine-Tuning | `03_instruction_fine_tuning/` | ✅ |
| 4 | Parameter-Efficient Fine-Tuning (PEFT) | `04_peft_overview/` | ✅ |
| 5 | LoRA | `05_lora/` | ✅ |
| 6 | QLoRA | `06_qlora/` | ✅ |
| 7 | Adapters | `07_adapters/` | ✅ |
| 8 | Prefix Tuning | `08_prefix_tuning/` | ✅ |
| 9 | Prompt Tuning | `09_prompt_tuning/` | ✅ |
| 10 | P-Tuning / P-Tuning v2 | `10_p_tuning/` | ✅ |
| 11 | IA³ | `11_ia3/` | ✅ |
| 12 | BitFit | `12_bitfit/` | ✅ |
| 13 | RLHF | `13_rlhf/` | ✅ |
| 14 | DPO | `14_dpo/` | ✅ |
| 15 | RL Fine-Tuning | `15_rl_fine_tuning/` | ✅ |
| 16 | Continual Fine-Tuning | `16_continual_fine_tuning/` | ✅ |
| 17 | Domain-Adaptive Pretraining | `17_dapt/` | ✅ |
| 18 | Task-Adaptive Pretraining | `18_tapt/` | ✅ |
| 19 | Multi-Task Fine-Tuning | `19_multi_task_fine_tuning/` | ✅ |
| 20 | Retrieval-Augmented Fine-Tuning | `20_retrieval_augmented_fine_tuning/` | ✅ |
| 21 | Knowledge Distillation | `21_knowledge_distillation/` | ✅ |

## Prerequisites

```bash
pip install torch transformers datasets accelerate peft trl bitsandbytes sentencepiece
```

## Hardware Notes

- Concepts 1 (Full FT) require significant GPU memory (16GB+)
- PEFT methods (5-12) can run on consumer GPUs (8GB+)
- QLoRA (6) is specifically designed for single-GPU training
- All examples use small models (GPT-2, DistilGPT-2) for demonstration purposes
