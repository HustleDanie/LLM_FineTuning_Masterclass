"""
RLHF Comparison — RLHF vs Other Alignment Methods
====================================================

Comprehensive comparison of RLHF with alternative approaches:

1. SFTvsRLHF
   - What RLHF adds beyond supervised fine-tuning
   - When SFT is sufficient

2. RLHFvsDPO
   - PPO-based RLHF vs Direct Preference Optimization
   - Complexity, stability, and performance trade-offs

3. RLHFvsRLAIF
   - Human feedback vs AI feedback
   - Cost and scalability considerations

4. RLHFvsKTO
   - Paired preferences vs binary signal
   - When data is limited

5. DecisionFramework
   - When to use RLHF (and when not to)
   - Practical cost-benefit analysis

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple


# ============================================================================
# SECTION 1: SFT vs RLHF
# ============================================================================

def sft_vs_rlhf():
    """What RLHF adds beyond supervised fine-tuning."""
    print("=" * 65)
    print("  SECTION 1: SFT vs RLHF")
    print("=" * 65)
    
    print(f"""
  ═══ The Alignment Pipeline ═══
  
    Pre-trained LLM
         │
         ▼
    ┌─────────────┐
    │     SFT      │  Teach format, follow instructions
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │    RLHF     │  Align quality with human preferences
    └──────┬──────┘
           │
           ▼
    Aligned Model
  
  KEY INSIGHT: SFT teaches WHAT to say.
               RLHF teaches HOW WELL to say it.
""")
    
    # Demonstrate the difference with a concrete example
    print(f"  ── Concrete Example ──")
    print(f"""
  Prompt: "Explain quantum computing in simple terms"
  
  Base model (no training):
    "Quantum computing is a type of computation that harnesses
     quantum mechanical phenomena quantum bits or qubits which
     can exist in superposition blah blah technical jargon..."
  
  After SFT (format + instruction following):
    "Quantum computing uses quantum bits (qubits) that can be
     0 and 1 at the same time, unlike regular bits. This lets
     quantum computers solve certain problems much faster than
     regular computers."
    → Correct format ✓  Follows instruction ✓
    → But: Could be more engaging, clearer, better structured
  
  After RLHF (aligned with human preferences):
    "Think of a regular computer as someone reading a book one
     page at a time. A quantum computer is like reading all pages
     at once! Regular computers use 'bits' (0 or 1), but quantum
     computers use 'qubits' that can be both 0 AND 1 simultaneously.
     This superpower makes them incredibly fast at certain tasks."
    → Correct format ✓  Follows instruction ✓
    → ALSO: Engaging ✓  Clear analogy ✓  Well-structured ✓
""")
    
    # Quantitative comparison via model behavior
    torch.manual_seed(42)
    
    class MockModel(nn.Module):
        """Mock model to demonstrate SFT vs RLHF outputs."""
        def __init__(self, quality_mean, quality_std):
            super().__init__()
            self.quality_mean = quality_mean
            self.quality_std = quality_std
        
        def generate_quality(self, n_samples=100):
            """Simulate quality scores for generated outputs."""
            return torch.normal(
                torch.full((n_samples,), self.quality_mean),
                torch.full((n_samples,), self.quality_std)
            )
    
    base_model = MockModel(quality_mean=3.0, quality_std=2.0)
    sft_model = MockModel(quality_mean=5.5, quality_std=1.5)
    rlhf_model = MockModel(quality_mean=7.5, quality_std=0.8)
    
    n = 200
    base_scores = base_model.generate_quality(n)
    sft_scores = sft_model.generate_quality(n)
    rlhf_scores = rlhf_model.generate_quality(n)
    
    print(f"  ── Quality Score Distribution (simulated, 1-10 scale) ──\n")
    print(f"  {'Model':>12} │ {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6} "
          f"{'% > 7.0':>8}")
    print(f"  {'─'*12}─┼─{'─'*6}─{'─'*6}─{'─'*6}─{'─'*6}─{'─'*8}")
    
    for name, scores in [("Base", base_scores), ("SFT", sft_scores), 
                          ("RLHF", rlhf_scores)]:
        scores = scores.clamp(1, 10)
        pct_above_7 = (scores > 7.0).float().mean() * 100
        print(f"  {name:>12} │ {scores.mean():>6.2f} {scores.std():>6.2f} "
              f"{scores.min():>6.2f} {scores.max():>6.2f} {pct_above_7:>7.1f}%")
    
    print(f"""
  ═══ What RLHF Adds Over SFT ═══
  
  1. QUALITY OVER CORRECTNESS:
     SFT: Learns to produce correct-format outputs
     RLHF: Learns which correct outputs humans prefer
     → RLHF raises the quality ceiling
  
  2. HANDLING AMBIGUITY:
     SFT: One right answer per example
     RLHF: Learns to rank among many valid answers
     → RLHF handles taste/preference
  
  3. SAFETY ALIGNMENT:
     SFT: Can teach safety rules (refusal patterns)
     RLHF: Learns nuanced safety boundaries
     → RLHF better at edge cases
  
  4. REDUCED SYCOPHANCY (with good reward model):
     SFT: May over-agree with user (training data bias)
     RLHF: Can learn to push back appropriately
     → RLHF enables more calibrated responses
  
  WHEN SFT IS SUFFICIENT:
  • Task is well-defined (e.g., translation, extraction)
  • You have high-quality supervised examples
  • Budget/complexity constraints
  • Response quality variation is low
""")


# ============================================================================
# SECTION 2: RLHF vs DPO
# ============================================================================

def rlhf_vs_dpo():
    """PPO-based RLHF vs Direct Preference Optimization."""
    print("\n\n" + "=" * 65)
    print("  SECTION 2: RLHF (PPO) vs DPO")
    print("=" * 65)
    
    print(f"""
  ═══ Architecture Comparison ═══
  
  RLHF (PPO):
  ┌──────────┐   ┌─────────────┐   ┌──────────┐   ┌───────────┐
  │  Policy   │   │  Reference  │   │  Reward  │   │   Value   │
  │  Model    │   │   Model     │   │  Model   │   │   Head    │
  └─────┬────┘   └──────┬──────┘   └────┬─────┘   └─────┬─────┘
        │               │               │               │
        └───────┬───────┴───────┬───────┘               │
                │               │                       │
        ┌───────▼───────┐ ┌────▼────┐           ┌──────▼──────┐
        │  KL Penalty   │ │ Reward  │           │  Advantage  │
        └───────┬───────┘ │ Score   │           │  Estimation │
                │         └────┬────┘           └──────┬──────┘
                └──────┬───────┘───────────────────────┘
                       │
                ┌──────▼──────┐
                │  PPO Update │
                └─────────────┘
  
  Models in memory: 4     Complexity: HIGH
  
  
  DPO (Direct Preference Optimization):
  ┌──────────┐   ┌─────────────┐
  │  Policy   │   │  Reference  │
  │  Model    │   │   Model     │
  └─────┬────┘   └──────┬──────┘
        │               │
        └───────┬───────┘
                │
        ┌───────▼──────────┐
        │  DPO Loss:       │
        │  L = -log σ(β *  │
        │  (log π/π_ref    │
        │    for chosen    │
        │  - log π/π_ref   │
        │    for rejected))│
        └──────────────────┘
  
  Models in memory: 2     Complexity: LOW
""")
    
    # Demonstrate both loss functions
    torch.manual_seed(42)
    
    print(f"  ── Loss Function Comparison ──\n")
    
    # Simulate logits for chosen and rejected responses
    batch_size = 100
    seq_len = 20
    
    # Policy log-probs (simulated)
    policy_chosen_logprobs = torch.randn(batch_size, seq_len) - 1.0
    policy_rejected_logprobs = torch.randn(batch_size, seq_len) - 1.5
    ref_chosen_logprobs = torch.randn(batch_size, seq_len) - 1.2
    ref_rejected_logprobs = torch.randn(batch_size, seq_len) - 1.3
    
    # Sum across sequence
    pi_chosen = policy_chosen_logprobs.sum(dim=1)
    pi_rejected = policy_rejected_logprobs.sum(dim=1)
    ref_chosen = ref_chosen_logprobs.sum(dim=1)
    ref_rejected = ref_rejected_logprobs.sum(dim=1)
    
    # DPO loss
    beta = 0.1
    policy_chosen_ratio = pi_chosen - ref_chosen
    policy_rejected_ratio = pi_rejected - ref_rejected
    dpo_logits = beta * (policy_chosen_ratio - policy_rejected_ratio)
    dpo_loss = -F.logsigmoid(dpo_logits).mean()
    
    # Equivalent RLHF reward (implicit in DPO)
    implicit_reward_chosen = beta * (pi_chosen - ref_chosen)
    implicit_reward_rejected = beta * (pi_rejected - ref_rejected)
    reward_margin = implicit_reward_chosen - implicit_reward_rejected
    
    print(f"    DPO Loss: {dpo_loss.item():.4f}")
    print(f"    Implicit reward margin (chosen - rejected): "
          f"{reward_margin.mean().item():.4f}")
    print(f"    % where chosen > rejected: "
          f"{(reward_margin > 0).float().mean().item()*100:.1f}%")
    
    # Comparison table
    print(f"""
  ═══ Head-to-Head Comparison ═══
  
  ┌────────────────────┬──────────────────┬──────────────────┐
  │ Aspect             │ RLHF (PPO)       │ DPO              │
  ├────────────────────┼──────────────────┼──────────────────┤
  │ Models in memory   │ 4                │ 2                │
  │ GPU memory         │ ~4× model size   │ ~2× model size   │
  │ Implementation     │ Complex          │ Simple (≈ SFT)   │
  │ Stability          │ Tricky           │ Very stable       │
  │ Hyperparameters    │ Many (β,ε,lr,γ..)│ Few (β, lr)      │
  │ Data requirements  │ Prompts only*    │ Preference pairs  │
  │ Online/Offline     │ Online (gen+RL)  │ Offline (static)  │
  │ Reward model       │ Separate model   │ Implicit          │
  │ Quality ceiling    │ Higher**         │ Good              │
  │ Reward hacking     │ Possible         │ Less likely       │
  │ Iteration speed    │ Slow             │ Fast              │
  │ Production use     │ OpenAI, Anthropic│ Most open-source  │
  └────────────────────┴──────────────────┴──────────────────┘
  
  * RLHF generates responses online, scores with reward model
  ** RLHF can explore beyond the preference dataset
  
  WHEN TO CHOOSE RLHF:
  • You have the compute budget (many GPUs)
  • Reward model is well-calibrated
  • Task benefits from online exploration
  • You're building a frontier model
  
  WHEN TO CHOOSE DPO:
  • Limited compute (2x model size vs 4x)
  • Want training stability
  • Have good preference data already
  • First alignment experiment
""")


# ============================================================================
# SECTION 3: RLHF vs RLAIF
# ============================================================================

def rlhf_vs_rlaif():
    """Human feedback vs AI feedback."""
    print("\n\n" + "=" * 65)
    print("  SECTION 3: RLHF vs RLAIF")
    print("=" * 65)
    
    print(f"""
  ═══ Feedback Source Comparison ═══
  
  RLHF: Reinforcement Learning from Human Feedback
  ─────────────────────────────────────────────────
  
    Response A ─┐         ┌─→ Human Annotator 1: A > B
    Response B ─┼─→ Show ─┼─→ Human Annotator 2: A > B
                │  to   ─┤─→ Human Annotator 3: B > A
                │humans  │
                         └─→ Majority vote: A wins
  
  Cost: $1-5 per comparison
  Speed: 500-2000 comparisons per annotator per day
  Quality: Gold standard (but noisy!)
  
  
  RLAIF: Reinforcement Learning from AI Feedback
  ───────────────────────────────────────────────
  
    Response A ─┐         ┌─→ GPT-4 Judge: A is better
    Response B ─┼─→ Ask  ─┤   because...
                │   AI    │
                │ judge   └─→ Score: A=8.5, B=6.2
  
  Cost: $0.01-0.10 per comparison
  Speed: 10,000+ comparisons per hour
  Quality: ~80-90% agreement with humans (for strong judges)
  
  
  ═══ Key Trade-offs ═══
  
  ┌─────────────────┬──────────────────┬──────────────────┐
  │ Aspect          │ RLHF (Human)     │ RLAIF (AI)       │
  ├─────────────────┼──────────────────┼──────────────────┤
  │ Cost per pair   │ $1-5             │ $0.01-0.10       │
  │ Throughput      │ Slow             │ Very fast         │
  │ Scalability     │ Limited          │ Near-unlimited    │
  │ Consistency     │ Low (humans      │ High (determin-   │
  │                 │   disagree)      │   istic)          │
  │ Nuance          │ Excellent        │ Good              │
  │ Safety          │ Better (humans   │ May miss subtle   │
  │                 │   catch harm)    │   harms           │
  │ Bias            │ Human biases     │ AI model biases   │
  │ Coverage        │ Limited by time  │ Can cover more    │
  │ Bootstrap       │ Need humans      │ Need strong judge │
  └─────────────────┴──────────────────┴──────────────────┘
  
  HYBRID APPROACH (most practical):
  1. Use RLAIF for bulk data collection (cheap, fast)
  2. Use RLHF for hard cases and safety (accurate, nuanced)
  3. Use humans to audit AI judgments (quality control)
  
  Example:
  • 50K comparisons from AI feedback ($500-5000)
  • 5K comparisons from human feedback ($5000-25000)
  • 1K human audits of AI decisions ($1000-5000)
  → Total: $6,500-35,000 vs $55K-275K for all-human
""")


# ============================================================================
# SECTION 4: RLHF vs KTO
# ============================================================================

def rlhf_vs_kto():
    """RLHF versus KTO (Kahneman-Tversky Optimization)."""
    print("\n\n" + "=" * 65)
    print("  SECTION 4: RLHF vs KTO")
    print("=" * 65)
    
    print(f"""
  ═══ Data Format Comparison ═══
  
  RLHF/DPO: PAIRED preferences needed
  ────────────────────────────────────
  
  Prompt: "What is photosynthesis?"
  Chosen:   "Photosynthesis is the process by which plants convert..."
  Rejected: "Photosynthesis is like when plants eat sunlight..."
  
  → Need PAIRS of responses rated against each other
  → Expensive to collect (need two responses per prompt)
  → But: captures relative quality well
  
  
  KTO: BINARY signal only (thumbs up/down)
  ─────────────────────────────────────────
  
  Prompt: "What is photosynthesis?"
  Response: "Photosynthesis is the process by which plants convert..."
  Label: 👍 (desirable)
  
  Prompt: "Explain gravity"
  Response: "Gravity is complicated and I don't know..."
  Label: 👎 (undesirable)
  
  → Only need individual responses with thumbs up/down
  → Much cheaper to collect
  → No need to pair responses together
""")
    
    # Demonstrate KTO loss
    torch.manual_seed(42)
    
    print(f"  ── KTO Loss Function ──\n")
    
    print(f"""    KTO Loss (Ethayarajh et al., 2024):
    
    For desirable responses (y_d):
      L_d = (1 - σ(β · (log π(y_d|x)/π_ref(y_d|x) - z_ref)))
    
    For undesirable responses (y_u):
      L_u = (1 - σ(β · (z_ref - log π(y_u|x)/π_ref(y_u|x))))
    
    Where z_ref = E[β · KL(π||π_ref)] is a reference point
    (inspired by Kahneman-Tversky prospect theory)
""")
    
    # Simulate KTO vs DPO
    batch_size = 50
    beta = 0.1
    
    # Simulated log-ratios for chosen/rejected
    chosen_log_ratio = torch.randn(batch_size) * 0.5
    rejected_log_ratio = torch.randn(batch_size) * 0.5 - 0.3
    
    # DPO loss (needs paired data)
    dpo_logits = beta * (chosen_log_ratio - rejected_log_ratio)
    dpo_loss = -F.logsigmoid(dpo_logits).mean()
    
    # KTO loss (unpaired data)
    z_ref = beta * torch.cat([chosen_log_ratio, rejected_log_ratio]).mean()
    kto_desirable = (1 - torch.sigmoid(beta * (chosen_log_ratio - z_ref))).mean()
    kto_undesirable = (1 - torch.sigmoid(beta * (z_ref - rejected_log_ratio))).mean()
    kto_loss = kto_desirable + kto_undesirable
    
    print(f"    DPO loss (paired): {dpo_loss.item():.4f}")
    print(f"    KTO loss (unpaired): {kto_loss.item():.4f}")
    
    print(f"""
  ═══ Comparison Table ═══
  
  ┌─────────────────┬──────────────┬──────────────┬──────────────┐
  │ Aspect          │ RLHF (PPO)   │ DPO          │ KTO          │
  ├─────────────────┼──────────────┼──────────────┼──────────────┤
  │ Data format     │ Prompts +    │ Preference   │ Binary       │
  │                 │ reward model │ pairs        │ (up/down)    │
  ├─────────────────┼──────────────┼──────────────┼──────────────┤
  │ Data cost       │ Medium       │ High         │ LOW          │
  │                 │ (RM training)│ (paired)     │ (unpaired)   │
  ├─────────────────┼──────────────┼──────────────┼──────────────┤
  │ Models in mem   │ 4            │ 2            │ 2            │
  ├─────────────────┼──────────────┼──────────────┼──────────────┤
  │ Complexity      │ High         │ Low          │ Low          │
  ├─────────────────┼──────────────┼──────────────┼──────────────┤
  │ Data efficiency │ Good         │ Good         │ Lower (needs │
  │                 │              │              │ more data)   │
  ├─────────────────┼──────────────┼──────────────┼──────────────┤
  │ Performance     │ Best (at     │ Very Good    │ Good         │
  │                 │ frontier)    │              │              │
  ├─────────────────┼──────────────┼──────────────┼──────────────┤
  │ Best when       │ Frontier     │ Good paired  │ Only have    │
  │                 │ models       │ data exists  │ binary data  │
  └─────────────────┴──────────────┴──────────────┴──────────────┘
""")


# ============================================================================
# SECTION 5: DECISION FRAMEWORK
# ============================================================================

def decision_framework():
    """When to use RLHF, and practical cost-benefit analysis."""
    print("\n\n" + "=" * 65)
    print("  SECTION 5: DECISION FRAMEWORK")
    print("=" * 65)
    
    print(f"""
  ═══ When to Use What ═══
  
  START HERE:  Do you need alignment beyond SFT?
                         │
                    ┌────┴────┐
                    │  Need   │
                  ╔═╧═╗    ╔═╧═════╗
                  ║YES║    ║  NO   ║
                  ╚═╤═╝    ╚═══════╝
                    │      → Use SFT alone
                    │        (extraction, translation,
                    │         well-defined tasks)
                    │
            What data do you have?
                    │
          ┌─────────┼─────────┐
          │         │         │
    ┌─────▼────┐  ┌─▼──────┐ ┌▼────────┐
    │ Paired   │  │ Binary │ │ Neither │
    │ prefs    │  │ signal │ │ (only   │
    │ (A > B)  │  │ (👍/👎) │ │ prompts)│
    └────┬─────┘  └───┬────┘ └────┬────┘
         │            │           │
    ┌────▼────┐  ┌────▼────┐ ┌───▼─────┐
    │ Budget? │  │  Use    │ │ Train   │
    │         │  │  KTO    │ │ reward  │
    │ Low→DPO │  │         │ │ model   │
    │ High→PPO│  └─────────┘ │ then    │
    └─────────┘              │ use PPO │
                             └─────────┘
  
  
  ═══ Cost-Benefit Analysis ═══
  
  ┌─────────────────────────────────────────────────────────────┐
  │ Method     │ Compute  │ Data Cost │ Eng. Time │ Quality    │
  ├────────────┼──────────┼───────────┼───────────┼────────────┤
  │ SFT only   │ $100-1K  │ $1-5K     │ 1-2 days  │ ★★★☆     │
  │ SFT + DPO  │ $200-2K  │ $5-20K    │ 2-5 days  │ ★★★★     │
  │ SFT + KTO  │ $200-2K  │ $1-10K    │ 2-5 days  │ ★★★½     │
  │ SFT + RLHF │ $1-10K   │ $10-50K   │ 1-4 weeks │ ★★★★★    │
  │ Full RLHF  │ $10-100K │ $50-500K  │ 1-3 months│ ★★★★★    │
  │ (frontier) │          │           │           │            │
  └─────────────────────────────────────────────────────────────┘
  
  (Costs are rough estimates for a 7B parameter model)
""")
    
    # Practical recommendations
    print(f"""
  ═══ Practical Recommendations ═══
  
  FOR STARTUPS / SMALL TEAMS:
  1. Start with SFT on high-quality data
  2. If more alignment needed, try DPO first
  3. Only move to RLHF if DPO ceiling is hit
  
  FOR RESEARCH LABS:
  1. SFT baseline → DPO → PPO/RLHF comparison
  2. Invest in reward model quality
  3. Use RLAIF for initial data, RLHF for refinement
  
  FOR PRODUCTION SYSTEMS:
  1. Use RLHF/PPO if you can afford it (frontier quality)
  2. Use online DPO for iterative improvement
  3. Monitor reward hacking continuously
  4. Regular human evaluation alongside automated metrics
  
  
  ═══ RLHF: Pros and Cons Summary ═══
  
  ADVANTAGES:
  ✓ Highest quality alignment (when done right)
  ✓ Can explore beyond static dataset
  ✓ Online learning adapts to distribution shifts
  ✓ Reward model is reusable for evaluation
  ✓ Industry-proven (GPT-4, Claude, Gemini)
  
  DISADVANTAGES:
  ✗ Complex: 4 models, many hyperparameters
  ✗ Expensive: 4× GPU memory, slow training
  ✗ Unstable: reward hacking, mode collapse
  ✗ Reward model quality bottleneck
  ✗ Hard to debug when things go wrong
  ✗ Requires RL expertise on the team
  
  
  ═══ The Future ═══
  
  Trends in alignment research:
  
  1. SIMPLER METHODS winning:
     DPO, KTO, ORPO are all simpler than PPO
     → But PPO still dominates at the frontier
  
  2. ONLINE DPO / ITERATIVE DPO:
     Get benefits of online learning + simplicity of DPO
  
  3. PROCESS REWARD MODELS:
     Reward per reasoning step instead of per response
     → Better for math, coding, reasoning
  
  4. CONSTITUTIONAL AI:
     Use AI principles to generate preference data
     → Scales better than human feedback
  
  5. RLHF + RLCD (Reinforcement Learning from
     Contrastive Distillation):
     → Use contrastive pairs from different model quality
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all RLHF comparison sections."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║      RLHF COMPARISON — RLHF vs OTHER ALIGNMENT METHODS      ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: SFT vs RLHF
    sft_vs_rlhf()
    
    # Section 2: RLHF vs DPO
    rlhf_vs_dpo()
    
    # Section 3: RLHF vs RLAIF
    rlhf_vs_rlaif()
    
    # Section 4: RLHF vs KTO
    rlhf_vs_kto()
    
    # Section 5: Decision framework
    decision_framework()
    
    print("\n" + "=" * 65)
    print("  COMPARISON MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered:
    ✓ SFT vs RLHF — what RLHF adds (quality, nuance, safety)
    ✓ RLHF vs DPO — 4 models vs 2, complexity vs stability
    ✓ RLHF vs RLAIF — human vs AI feedback (cost vs quality)
    ✓ RLHF vs KTO — paired vs binary data requirements
    ✓ Decision framework — when to use each method
    """)


if __name__ == "__main__":
    main()
