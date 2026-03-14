# Concept 13: RLHF (Reinforcement Learning from Human Feedback)

## Overview

**RLHF** is the technique that transformed language models from text predictors into helpful,
harmless, and honest assistants. It's the key innovation behind ChatGPT, Claude, and other
aligned AI systems.

RLHF uses human preferences to train a reward model, which then guides the language model
to produce outputs that humans prefer — going beyond what supervised fine-tuning alone can achieve.

## The Three Stages of RLHF

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE RLHF PIPELINE                            │
│                                                                 │
│  Stage 1: Supervised Fine-Tuning (SFT)                         │
│  ─────────────────────────────────────                         │
│  Pretrained LLM  ──→  SFT on demonstrations  ──→  SFT Model   │
│                        (human-written)                          │
│                                                                 │
│  Stage 2: Reward Model Training                                │
│  ──────────────────────────────                                │
│  SFT Model generates multiple responses to same prompt          │
│  Humans rank: Response A > Response B                           │
│  Train reward model on these preferences                        │
│                                                                 │
│  ┌──────────┐     ┌──────────┐                                 │
│  │ Prompt + │     │ Reward   │                                 │
│  │ Response │ ──→ │ Model    │ ──→  Scalar Score              │
│  └──────────┘     └──────────┘                                 │
│                                                                 │
│  Stage 3: RL Optimization (PPO)                                │
│  ──────────────────────────────                                │
│  ┌───────┐  generate   ┌────────┐  score   ┌──────┐           │
│  │Policy │ ──────────→ │Response│ ──────→  │Reward│           │
│  │(LLM)  │             └────────┘          │Model │           │
│  └───┬───┘                                 └──┬───┘           │
│      │          ◄── PPO update ───────────────┘               │
│      │                                                         │
│      │  KL penalty: Don't drift too far from SFT model        │
│      └──→ Reference Model (frozen SFT copy)                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Why RLHF? The Alignment Problem

```
SFT alone:    "Predict the next token that looks like training data"
              → Can be helpful but also toxic, hallucinating, sycophantic

RLHF:        "Generate text that humans actually PREFER"
              → Learns subtle qualities: helpfulness, safety, honesty
              → Captures preferences that are hard to specify in rules
```

## Key Components

### 1. Reward Model
- Takes (prompt, response) → scalar reward
- Trained on human comparison data: (prompt, chosen, rejected)
- Loss: Bradley-Terry model: L = -log(σ(r(chosen) - r(rejected)))

### 2. PPO (Proximal Policy Optimization)
- Policy = the LLM being trained
- Action = generating a token
- Reward = reward model score + KL penalty
- PPO clips large policy updates for stability

### 3. KL Divergence Penalty
- Prevents the policy from deviating too far from the SFT model
- reward_total = reward_model(response) - β × KL(policy || reference)
- β controls the trade-off: higher β = more conservative updates

## The Math

### Reward Model Training
```
L_reward = -E[log σ(r_θ(x, y_w) - r_θ(x, y_l))]

where:
  r_θ     = reward model with parameters θ
  x       = prompt
  y_w     = preferred (winning) response
  y_l     = dispreferred (losing) response
  σ       = sigmoid function
```

### PPO Objective
```
L_PPO = E[min(r_t(θ) × Â_t, clip(r_t(θ), 1-ε, 1+ε) × Â_t)]

where:
  r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (probability ratio)
  Â_t    = advantage estimate
  ε      = clip range (typically 0.2)
```

### KL-Penalized Reward
```
R(x, y) = r_φ(x, y) - β × KL(π_θ(y|x) || π_ref(y|x))

where:
  r_φ     = reward model score
  π_θ     = current policy
  π_ref   = reference (SFT) policy (frozen)
  β       = KL penalty coefficient
```

## Practical Considerations

| Aspect | Detail |
|--------|--------|
| Models needed | 4 (!): Policy, Reference, Reward, Value head |
| Memory | 4× model size minimum |
| Data | Human preference pairs (prompt, chosen, rejected) |
| Training time | 10-100× longer than SFT |
| Instability | PPO can be unstable; careful hyperparameter tuning |
| Reward hacking | Model may exploit reward model weaknesses |

## RLHF vs Alternatives

| Method | Reward Model? | RL? | Complexity | Performance |
|--------|--------------|-----|------------|-------------|
| SFT | No | No | Low | Good baseline |
| RLHF (PPO) | Yes | Yes | Very High | Excellent |
| DPO | No | No | Low | Near RLHF |
| RLAIF | AI-generated | Yes | High | Good |
| KTO | No | No | Low | Good |

## Files

| File | Description |
|------|-------------|
| `rlhf_theory.py` | RLHF theory, Bradley-Terry model, PPO math, KL divergence |
| `rlhf_reward_model.py` | Reward model training from scratch and with libraries |
| `rlhf_training.py` | Full RLHF pipeline using TRL's PPOTrainer |
| `rlhf_comparison.py` | RLHF vs SFT vs DPO, decision framework |

## References

- Ouyang et al. (2022): "Training language models to follow instructions with human feedback" (InstructGPT)
- Christiano et al. (2017): "Deep Reinforcement Learning from Human Preferences"
- Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
- Stiennon et al. (2020): "Learning to summarize from human feedback"
- Bai et al. (2022): "Training a Helpful and Harmless Assistant with RLHF" (Anthropic)
