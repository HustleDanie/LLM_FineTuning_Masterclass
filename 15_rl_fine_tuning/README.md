# Concept 15: RL Fine-Tuning (Reward-Based Fine-Tuning)

## Overview

**RL Fine-Tuning** is the broader paradigm of using reinforcement learning to
optimize language models beyond human preference alignment. While RLHF (Concept 13)
and DPO (Concept 14) focus on aligning with human preferences, RL fine-tuning
encompasses any scenario where a **reward signal** drives model improvement —
including task-specific rewards, verifiable outcomes, and programmatic feedback.

```
RL Fine-Tuning Landscape:

                    RL Fine-Tuning
                    /      |       \
                  /        |         \
    Preference-Based   Task-Reward   Verifiable-Reward
    (RLHF, DPO)       Based         Based (RLVR)
    │                  │              │
    │ Human prefs      │ Task metrics │ Correctness checks
    │ Reward models    │ BLEU, ROUGE  │ Math, code, logic
    │                  │ Custom score │ Unit tests
```

## Key Idea

Instead of relying solely on human preference data, RL fine-tuning uses
**programmatic reward functions** that can be:

- **Verifiable**: Math solutions checked for correctness
- **Executable**: Code tested with unit tests
- **Measurable**: BLEU/ROUGE for translation/summarization
- **Rule-based**: Length, format, safety constraints
- **Composite**: Weighted combination of multiple signals

## RL Algorithms for LLM Fine-Tuning

| Algorithm | Type | Key Idea |
|-----------|------|----------|
| **PPO** | On-policy | Clipped surrogate objective, value function |
| **REINFORCE** | On-policy | Simple policy gradient, high variance |
| **GRPO** | On-policy | Group Relative Policy Optimization (DeepSeek) |
| **ReMax** | On-policy | REINFORCE with baseline from max-reward sample |
| **RLOO** | On-policy | REINFORCE Leave-One-Out baseline |
| **Expert Iteration** | Off-policy | Generate → Filter → SFT on best |
| **ReST** | Off-policy | Reinforced Self-Training |
| **STaR** | Off-policy | Self-Taught Reasoner |

## RLVR — RL from Verifiable Rewards

A major trend (2024-2025): using **verifiable** rewards instead of learned
reward models. Pioneered by DeepSeek-R1 and others.

$$R(x, y) = \begin{cases} 1 & \text{if answer is correct (verified)} \\ 0 & \text{otherwise} \end{cases}$$

No reward model needed — just a verification function!

## GRPO (Group Relative Policy Optimization)

DeepSeek's alternative to PPO that eliminates the value model:

$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G} \sum_{i=1}^{G} \min\left(\frac{\pi_\theta}{\pi_{\text{old}}} \hat{A}_i,\; \text{clip}\left(\frac{\pi_\theta}{\pi_{\text{old}}}, 1\pm\epsilon\right) \hat{A}_i\right)$$

Where advantages are computed from **group statistics**:
$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})}$$

- No value model needed (3 models instead of 4)
- Generate G responses per prompt, rank by reward
- Normalize rewards within the group

## Files in This Folder

| File | Description |
|------|-------------|
| `rl_ft_theory.py` | RL fundamentals for LLMs, policy gradient, variance reduction |
| `rl_ft_from_scratch.py` | REINFORCE, GRPO, Expert Iteration from scratch |
| `rl_ft_training.py` | Task-reward training, RLVR, code/math reward functions |
| `rl_ft_comparison.py` | PPO vs GRPO vs REINFORCE vs Expert Iteration comparison |

## References

- Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
- Williams (1992): "Simple Statistical Gradient-Following Algorithms" (REINFORCE)
- Shao et al. (2024): "DeepSeekMath: Pushing the Limits of Mathematical Reasoning" (GRPO)
- DeepSeek-AI (2025): "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL"
- Zelikman et al. (2022): "STaR: Bootstrapping Reasoning With Reasoning"
- Gulcehre et al. (2023): "Reinforced Self-Training (ReST)"
