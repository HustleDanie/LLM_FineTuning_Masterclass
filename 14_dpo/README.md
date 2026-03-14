# Concept 14: Direct Preference Optimization (DPO)

## Overview

**DPO** (Rafailov et al., 2023) eliminates the need for a separate reward model and
reinforcement learning by directly optimizing the policy on preference data. It
recasts the RLHF objective as a simple classification loss over preferred vs
dispreferred response pairs.

```
RLHF Pipeline (complex):
  Preference Data → Train Reward Model → PPO Training (4 models)

DPO Pipeline (simple):
  Preference Data → Direct Policy Optimization (2 models)
```

## Core Insight

The optimal policy under the RLHF objective (KL-constrained reward maximization)
has a closed-form solution:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\!\left(\frac{1}{\beta} r(x,y)\right)$$

Rearranging for the reward:

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

Substituting into the Bradley-Terry preference model, the partition function
$Z(x)$ cancels and we get the **DPO loss**:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\!\left( \beta \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right) \right]$$

Where:
- $y_w$ = preferred (winning) response
- $y_l$ = dispreferred (losing) response
- $\beta$ = temperature controlling deviation from reference
- $\sigma$ = sigmoid function

## Key Properties

| Property | Value |
|----------|-------|
| Models in memory | 2 (policy + reference) |
| GPU memory | ~2× model size |
| Reward model needed | No (implicit) |
| RL algorithm needed | No |
| Training stability | Very high |
| Hyperparameters | Few (β, lr) |
| Data format | Paired preferences (chosen, rejected) |

## β (Beta) Parameter

β controls how far the policy can deviate from the reference:

| β Value | Effect |
|---------|--------|
| β → 0 | Ignores reference, pure preference fitting |
| β = 0.1 | Standard — moderate constraint |
| β = 0.5 | Strong constraint — stays close to reference |
| β → ∞ | No change from reference model |

Typical range: **0.05 – 0.5** (most common: 0.1)

## Implicit Reward

DPO defines an implicit reward function:

$$r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

This means the trained policy IS the reward model — no separate model needed.

## DPO Variants

| Variant | Key Change |
|---------|------------|
| **DPO** | Original — offline, static preference data |
| **IPO** | Identity-PO — more robust to label noise |
| **cDPO** | Conservative — adds label smoothing |
| **RSO** | Rejection Sampling Optimization |
| **ORPO** | Odds Ratio PO — no reference model needed |
| **SimPO** | Simple PO — length-normalized, no reference |
| **Online DPO** | Generates new responses during training |
| **Iterative DPO** | Alternates data generation and DPO |

## Files in This Folder

| File | Description |
|------|-------------|
| `dpo_theory.py` | Mathematical derivation, loss landscape, β analysis |
| `dpo_from_scratch.py` | DPO loss and training implemented from scratch |
| `dpo_training.py` | DPO training with TRL's DPOTrainer |
| `dpo_comparison.py` | DPO vs RLHF, IPO, ORPO, SimPO comparison |

## References

- Rafailov et al. (2023): "Direct Preference Optimization: Your Language Model Is Secretly a Reward Model"
- Azar et al. (2023): "A General Theoretical Paradigm to Understand Learning from Human Feedback" (IPO)
- Hong et al. (2024): "ORPO: Monolithic Preference Optimization without Reference Model"
- Meng et al. (2024): "SimPO: Simple Preference Optimization with a Reference-Free Reward"
