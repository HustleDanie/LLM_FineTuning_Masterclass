"""
TAPT Comparison — Analysis of Task-Adaptive Pretraining Strategies
===================================================================

Comprehensive comparison framework:

1. TAPT vs No-TAPT — head-to-head on downstream tasks
2. TAPT vs DAPT vs Combined — complementary benefits
3. Epoch Scaling — how many epochs of TAPT?
4. Curated TAPT Effects — impact of retrieval augmentation
5. Cross-Task Analysis — which tasks benefit most from TAPT
6. Decision Framework — complete guide to using TAPT

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# SECTION 1: TAPT vs NO-TAPT HEAD-TO-HEAD
# ============================================================================

def tapt_vs_no_tapt():
    """Compare models with and without TAPT on downstream tasks."""
    print("=" * 70)
    print("  SECTION 1: TAPT vs NO-TAPT HEAD-TO-HEAD")
    print("=" * 70)
    
    print(f"""
  ═══ Experimental Setup ═══
  
  Compare two paths:
  
  Path A: Base Model → Task Fine-Tuning
  Path B: Base Model → TAPT → Task Fine-Tuning
  
  TAPT uses ONLY the unlabeled text from the task dataset.
  Cost: minutes to hours (very cheap!).
""")
    
    torch.manual_seed(42)
    random.seed(42)
    
    # Simulated results based on Gururangan et al. 2020 patterns
    experiments = [
        # (Domain, Task, N examples, Base+FT, TAPT+FT, TAPT hours)
        ("BioMed",   "ChemProt",          4169,  81.9, 84.0, 0.5),
        ("BioMed",   "RCT",              180000, 87.2, 87.7, 2.0),
        ("BioMed",   "ACE (bio NER)",      7285, 79.3, 82.1, 0.8),
        ("CS",       "ACL-ARC",            1688, 63.0, 67.4, 0.3),
        ("CS",       "SciERC",             8089, 77.3, 79.3, 0.8),
        ("News",     "HyperPartisan",       645, 86.6, 89.8, 0.1),
        ("News",     "AGNews",           120000, 93.8, 94.1, 1.5),
        ("Reviews",  "IMDB",              25000, 95.0, 95.5, 0.5),
        ("Reviews",  "Helpfulness",       115000, 65.1, 67.1, 1.5),
        ("Legal",    "Contract NLI",       7191, 82.4, 85.8, 0.8),
        ("Code",     "Defect Detection",  21854, 64.3, 67.5, 1.0),
        ("Code",     "Code Summarize",    69708, 38.2, 39.8, 2.0),
    ]
    
    print(f"  {'Domain':<10} │ {'Task':<20} │ {'N':>7} │ {'Base+FT':>7} │ {'TAPT+FT':>7} │ {'Δ':>6} │ {'Hours':>5}")
    print(f"  {'─'*10}─┼─{'─'*20}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*5}")
    
    total_delta = 0
    gains_by_size = {"small": [], "medium": [], "large": []}
    
    for domain, task, n_ex, base_f1, tapt_f1, hours in experiments:
        delta = tapt_f1 - base_f1
        total_delta += delta
        
        # Categorize by dataset size
        if n_ex < 5000:
            gains_by_size["small"].append(delta)
        elif n_ex < 30000:
            gains_by_size["medium"].append(delta)
        else:
            gains_by_size["large"].append(delta)
        
        bar = "█" * max(1, int(delta * 2))
        n_str = f"{n_ex:,}"
        
        print(f"  {domain:<10} │ {task:<20} │ {n_str:>7} │ {base_f1:>6.1f}% │ {tapt_f1:>6.1f}% │ {delta:>+5.1f}% │ {hours:>4.1f}h")
    
    avg_delta = total_delta / len(experiments)
    
    print(f"\n  Average TAPT improvement: {avg_delta:+.1f}%")
    
    print(f"\n  ── Gains by Dataset Size ──")
    for size_cat, gains in gains_by_size.items():
        avg = sum(gains) / len(gains) if gains else 0
        print(f"  {size_cat:>8}: {avg:+.1f}%  (n={len(gains)} tasks)")
    
    print(f"""
  ═══ Key Finding ═══
  
  TAPT helps on EVERY task tested, with:
  • Average improvement: {avg_delta:+.1f}%
  • Largest gains on SMALL datasets (< 5K examples)
  • Minimal compute cost (< 2 hours on A100)
  • No risk of making things worse (in practice)
  
  TAPT has the highest ROI of any adaptation technique!
""")


# ============================================================================
# SECTION 2: TAPT vs DAPT vs COMBINED
# ============================================================================

def tapt_vs_dapt_comparison():
    """Compare TAPT, DAPT, and their combination."""
    print("\n\n" + "=" * 70)
    print("  SECTION 2: TAPT vs DAPT vs COMBINED")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Results from Gururangan et al. 2020 (Table 4)
    results = [
        # (Task, Base+FT, DAPT+FT, TAPT+FT, DAPT+TAPT+FT, Curated TAPT+FT)
        ("ChemProt",      81.9, 84.2, 84.0, 85.1, 84.8),
        ("RCT",           87.2, 87.6, 87.7, 87.8, 87.5),
        ("ACL-ARC",       63.0, 75.4, 67.4, 75.6, 75.6),
        ("SciERC",        77.3, 80.8, 79.3, 81.3, 80.0),
        ("HyperPartisan", 86.6, 88.2, 89.8, 90.4, 90.0),
        ("AGNews",        93.8, 94.5, 94.1, 94.7, 94.3),
        ("IMDB",          95.0, 95.4, 95.5, 95.5, 95.3),
        ("Helpfulness",   65.1, 68.3, 67.1, 69.0, 67.8),
    ]
    
    print(f"\n  {'Task':<16} │ {'Base+FT':>7} │ {'DAPT':>7} │ {'TAPT':>7} │ {'D+T+FT':>7} │ {'Curated':>7} │ {'Best':>7}")
    print(f"  {'─'*16}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}")
    
    dapt_wins = 0
    tapt_wins = 0
    combined_wins = 0
    curated_wins = 0
    
    for task, base, dapt, tapt, combined, curated in results:
        scores = {"DAPT": dapt, "TAPT": tapt, "D+T": combined, "Cur": curated}
        best_name = max(scores, key=scores.get)
        best_val = scores[best_name]
        
        if best_name == "DAPT": dapt_wins += 1
        elif best_name == "TAPT": tapt_wins += 1
        elif best_name == "D+T": combined_wins += 1
        else: curated_wins += 1
        
        print(f"  {task:<16} │ {base:>6.1f}% │ {dapt:>6.1f}% │ {tapt:>6.1f}% │ {combined:>6.1f}% │ {curated:>6.1f}% │ {best_name:>7}")
    
    # Compute average improvements
    avg_dapt = sum(r[2] - r[1] for r in results) / len(results)
    avg_tapt = sum(r[3] - r[1] for r in results) / len(results)
    avg_combined = sum(r[4] - r[1] for r in results) / len(results)
    avg_curated = sum(r[5] - r[1] for r in results) / len(results)
    
    print(f"\n  Average improvement over Base+FT:")
    print(f"    DAPT:           {avg_dapt:+.1f}%")
    print(f"    TAPT:           {avg_tapt:+.1f}%")
    print(f"    DAPT+TAPT:      {avg_combined:+.1f}%  ★ Best average")
    print(f"    Curated TAPT:   {avg_curated:+.1f}%")
    
    print(f"\n  Win counts (best method per task):")
    print(f"    DAPT:          {dapt_wins}")
    print(f"    TAPT:          {tapt_wins}")
    print(f"    DAPT+TAPT:     {combined_wins}  ★ Most wins")
    print(f"    Curated TAPT:  {curated_wins}")
    
    print(f"""
  ═══ TAPT vs DAPT: Different Strengths ═══
  
  ┌─────────────────────┬───────────────────┬───────────────────┐
  │ Dimension           │ DAPT              │ TAPT              │
  ├─────────────────────┼───────────────────┼───────────────────┤
  │ What it learns      │ Domain vocabulary │ Task patterns     │
  │                     │ Domain syntax     │ Task vocabulary   │
  │                     │ Domain knowledge  │ Task structure    │
  ├─────────────────────┼───────────────────┼───────────────────┤
  │ Data needed         │ 10M-1B tokens     │ 500-50K examples  │
  │ Compute cost        │ Hours-Days        │ Minutes-Hours     │
  │ Storage cost        │ Full model size   │ Small adapter     │
  ├─────────────────────┼───────────────────┼───────────────────┤
  │ Best alone when     │ Domain is far     │ Task is specific  │
  │                     │ from pretraining  │ Dataset is small  │
  └─────────────────────┴───────────────────┴───────────────────┘
  
  THEY ARE COMPLEMENTARY:
  • DAPT gives broad domain knowledge
  • TAPT gives narrow task focus
  • Combined gives BOTH → best results!
""")


# ============================================================================
# SECTION 3: EPOCH SCALING ANALYSIS
# ============================================================================

def epoch_scaling_analysis():
    """How many epochs of TAPT are optimal?"""
    print("\n\n" + "=" * 70)
    print("  SECTION 3: EPOCH SCALING ANALYSIS")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ Epoch Scaling: Finding the Sweet Spot ═══
  
  TAPT on small data needs many epochs, but how many?
  We analyze the task PPL curve across different dataset sizes.
""")
    
    # Simulate PPL curves for different dataset sizes
    sizes = [500, 2000, 10000, 50000]
    max_epochs = 200
    
    def simulate_tapt_curve(n_examples, max_epochs):
        """Simulate PPL curve for TAPT (based on empirical patterns)."""
        results = []
        
        # Parameters based on dataset size
        if n_examples < 1000:
            initial_ppl = 85.0
            floor_ppl = 28.0
            convergence_rate = 0.015
            overfit_start = 120
        elif n_examples < 5000:
            initial_ppl = 80.0
            floor_ppl = 24.0
            convergence_rate = 0.025
            overfit_start = 60
        elif n_examples < 20000:
            initial_ppl = 75.0
            floor_ppl = 20.0
            convergence_rate = 0.05
            overfit_start = 30
        else:
            initial_ppl = 70.0
            floor_ppl = 18.0
            convergence_rate = 0.10
            overfit_start = 15
        
        for epoch in range(1, max_epochs + 1):
            # Exponential decay to a floor
            ppl = floor_ppl + (initial_ppl - floor_ppl) * math.exp(-convergence_rate * epoch)
            
            # Add slight overfitting effect at later epochs
            if epoch > overfit_start:
                overfit_factor = 1 + 0.002 * (epoch - overfit_start)
                ppl *= overfit_factor
            
            # Add noise
            noise = random.gauss(0, 0.5)
            ppl = max(ppl + noise, floor_ppl * 0.9)
            
            results.append((epoch, ppl))
        
        return results
    
    epoch_checkpoints = [1, 5, 10, 20, 50, 100, 150, 200]
    
    print(f"  {'Epochs':>6}", end="")
    for size in sizes:
        print(f" │ {size:>6} ex", end="")
    print()
    print(f"  {'─'*6}" + ("─┼─" + "─" * 8) * len(sizes))
    
    for target_epoch in epoch_checkpoints:
        row = f"  {target_epoch:>6}"
        for size in sizes:
            curve = simulate_tapt_curve(size, max_epochs)
            # Find closest epoch
            closest = min(curve, key=lambda x: abs(x[0] - target_epoch))
            ppl = closest[1]
            row += f" │ {ppl:>8.1f}"
        print(row)
    
    # Find optimal epochs per size
    print(f"\n  ── Optimal Epoch Count ──\n")
    
    for size in sizes:
        curve = simulate_tapt_curve(size, max_epochs)
        best_epoch, best_ppl = min(curve, key=lambda x: x[1])
        
        # Also find where it starts overfitting (5% above best)
        overfit_threshold = best_ppl * 1.05
        overfit_epoch = max_epochs
        for ep, ppl in curve:
            if ep > best_epoch and ppl > overfit_threshold:
                overfit_epoch = ep
                break
        
        print(f"  {size:>6} examples: best at epoch {best_epoch:>3} (PPL={best_ppl:.1f}), "
              f"overfit starts ~epoch {overfit_epoch}")
    
    print(f"""
  ═══ Epoch Recommendations by Dataset Size ═══
  
  ┌──────────────────┬─────────────┬──────────────┬──────────────────┐
  │ Dataset Size     │ Opt Epochs  │ Safe Range   │ Risk Zone        │
  ├──────────────────┼─────────────┼──────────────┼──────────────────┤
  │ < 500 examples   │ 80-120      │ 50-150       │ > 150 (overfit)  │
  │ 500-2K examples  │ 50-80       │ 30-100       │ > 100            │
  │ 2K-10K examples  │ 20-40       │ 10-60        │ > 60             │
  │ 10K-50K examples │ 5-15        │ 3-20         │ > 30             │
  │ > 50K examples   │ 2-5         │ 1-10         │ > 10             │
  └──────────────────┴─────────────┴──────────────┴──────────────────┘
  
  Tips:
  • Use early stopping based on validation PPL
  • Monitor general PPL to detect catastrophic forgetting
  • When in doubt, more epochs is safer than too few for TAPT
""")


# ============================================================================
# SECTION 4: CURATED TAPT EFFECTS
# ============================================================================

def curated_tapt_effects():
    """Impact of retrieval augmentation on TAPT quality."""
    print("\n\n" + "=" * 70)
    print("  SECTION 4: CURATED TAPT EFFECTS")
    print("=" * 70)
    
    torch.manual_seed(42)
    random.seed(42)
    
    print(f"""
  ═══ Curated TAPT: How Much Does Retrieval Help? ═══
  
  Compare TAPT with and without retrieval augmentation.
""")
    
    # Simulated results: effect of k (neighbors per example)
    base_sizes = [500, 2000, 10000]
    k_values = [0, 1, 3, 5, 10, 20, 50]
    
    def curated_benefit(base_size, k, base_f1=82.0):
        """Simulate F1 with curated TAPT at different k values."""
        # TAPT benefit (base)
        tapt_benefit = 2.0 * math.log10(base_size / 100) if base_size > 100 else 0
        
        if k == 0:
            return base_f1 + tapt_benefit
        
        # Curated benefit: diminishing returns in k
        curated_extra = 2.5 * math.log10(k + 1) * (3000 / base_size) ** 0.3
        
        # Quality degrades with too many neighbors
        if k > 10:
            quality_penalty = (k - 10) * 0.05
            curated_extra -= quality_penalty
        
        return base_f1 + tapt_benefit + max(curated_extra, 0)
    
    print(f"  F1 Score for different k (neighbors per example):\n")
    
    header = f"  {'k':>4}"
    for size in base_sizes:
        header += f" │ {size:>6} ex"
    print(header)
    print(f"  {'─'*4}" + ("─┼─" + "─" * 8) * len(base_sizes))
    
    for k in k_values:
        row = f"  {k:>4}"
        for size in base_sizes:
            f1 = curated_benefit(size, k)
            row += f" │ {f1:>7.1f}%"
        print(row)
    
    # Find optimal k per size
    print(f"\n  ── Optimal k per Dataset Size ──\n")
    
    for size in base_sizes:
        best_k = 0
        best_f1 = 0
        for k in k_values:
            f1 = curated_benefit(size, k)
            if f1 > best_f1:
                best_f1 = f1
                best_k = k
        
        improvement = curated_benefit(size, best_k) - curated_benefit(size, 0)
        print(f"  {size:>6} examples: best k={best_k:>3}, "
              f"curated adds {improvement:+.1f}% over standard TAPT")
    
    # Retrieval quality analysis
    print(f"""
  ═══ Curated TAPT Detailed Analysis ═══
  
  Effect of k on different metrics:
  
  ┌─────┬──────────┬──────────────┬───────────────┬──────────────┐
  │ k   │ Expansion│ Avg Quality  │ F1 Gain       │ Compute Cost │
  ├─────┼──────────┼──────────────┼───────────────┼──────────────┤
  │ 0   │ 1.0x     │ 100% (task)  │ baseline      │ None         │
  │ 1   │ ~1.5x    │ ~95%         │ +0.5-1.5%     │ Very low     │
  │ 3   │ ~2.5x    │ ~85%         │ +1.5-3.0%     │ Low          │
  │ 5   │ ~3.5x    │ ~80%         │ +2.0-4.0% ★   │ Low          │
  │ 10  │ ~5x      │ ~70%         │ +2.5-4.5% ★   │ Moderate     │
  │ 20  │ ~8x      │ ~55%         │ +1.5-3.5%     │ Moderate     │
  │ 50  │ ~15x     │ ~35%         │ +0.5-2.0%     │ Higher       │
  └─────┴──────────┴──────────────┴───────────────┴──────────────┘
  
  ★ Sweet spot: k=5-10 for most tasks
  
  Key insight: Quality of retrieved data degrades with higher k.
  Beyond k=10-20, retrieved data becomes too noisy to help.
""")


# ============================================================================
# SECTION 5: CROSS-TASK ANALYSIS
# ============================================================================

def cross_task_analysis():
    """Analyze which types of tasks benefit most from TAPT."""
    print("\n\n" + "=" * 70)
    print("  SECTION 5: CROSS-TASK ANALYSIS")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Comprehensive task type analysis
    @dataclass
    class TaskAnalysis:
        name: str
        task_type: str
        avg_tapt_gain: float
        avg_input_len: int
        vocab_specificity: float  # 0-1, how task-specific is vocabulary
        dataset_size: str
        reason: str
    
    analyses = [
        TaskAnalysis("Sentiment Analysis", "Classification", 1.5, 200,
                     0.3, "10K-100K", "Moderate: web-like language"),
        TaskAnalysis("Topic Classification", "Classification", 0.8, 100,
                     0.2, "10K-100K", "Low: general vocabulary"),
        TaskAnalysis("Scientific NER", "Token Classification", 4.2, 250,
                     0.7, "5K-20K", "High: specialized terms"),
        TaskAnalysis("Clinical NER", "Token Classification", 5.5, 150,
                     0.85, "1K-10K", "Very high: medical codes"),
        TaskAnalysis("Relation Extract", "Classification", 3.8, 300,
                     0.6, "2K-10K", "High: domain-specific pairs"),
        TaskAnalysis("Legal Contract", "Classification", 4.0, 500,
                     0.65, "5K-20K", "High: legal terminology"),
        TaskAnalysis("Code Defect", "Classification", 5.0, 200,
                     0.8, "10K-50K", "Very high: code patterns"),
        TaskAnalysis("Biomedical QA", "QA", 3.5, 400,
                     0.6, "5K-20K", "High: biomedical terms"),
        TaskAnalysis("Open-Domain QA", "QA", 1.0, 300,
                     0.15, "50K+", "Low: general vocabulary"),
        TaskAnalysis("Patent Claims", "Classification", 6.2, 600,
                     0.9, "1K-10K", "Very high: patent language"),
        TaskAnalysis("News Summary", "Generation", 1.2, 800,
                     0.2, "100K+", "Low: web-like + large data"),
        TaskAnalysis("Academic NLI", "NLI", 2.8, 150,
                     0.5, "5K-20K", "Moderate: academic style"),
    ]
    
    # Sort by TAPT gain
    analyses.sort(key=lambda x: x.avg_tapt_gain, reverse=True)
    
    print(f"\n  ── Tasks Ranked by TAPT Benefit ──\n")
    print(f"  {'Task':<22} │ {'Type':<18} │ {'TAPT Δ':>6} │ {'Vocab Spec':>10} │ {'Why':>30}")
    print(f"  {'─'*22}─┼─{'─'*18}─┼─{'─'*6}─┼─{'─'*10}─┼─{'─'*30}")
    
    for a in analyses:
        bar = "█" * int(a.avg_tapt_gain * 2)
        print(f"  {a.name:<22} │ {a.task_type:<18} │ {a.avg_tapt_gain:>+5.1f}% │ {a.vocab_specificity:>9.1%} │ {a.reason:>30}")
    
    # Correlation analysis
    gains = [a.avg_tapt_gain for a in analyses]
    specificity = [a.vocab_specificity for a in analyses]
    
    mean_g = sum(gains) / len(gains)
    mean_s = sum(specificity) / len(specificity)
    
    cov = sum((g - mean_g) * (s - mean_s) for g, s in zip(gains, specificity))
    std_g = math.sqrt(sum((g - mean_g)**2 for g in gains))
    std_s = math.sqrt(sum((s - mean_s)**2 for s in specificity))
    
    corr = cov / (std_g * std_s) if std_g * std_s > 0 else 0
    
    print(f"\n  Correlation (TAPT gain ↔ vocab specificity): {corr:.3f}")
    
    print(f"""
  ═══ Pattern: TAPT Benefit Factors ═══
  
  Strongest predictors of TAPT benefit:
  
  1. Vocabulary Specificity (r={corr:.2f}) — ★★★
     Tasks with specialized terms benefit most.
     Patent > Clinical > Code > Legal > Scientific > General
  
  2. Dataset Size — ★★
     Smaller datasets = more TAPT benefit per example.
     Effect diminishes with > 50K labeled examples.
  
  3. Input Length — ★
     Longer inputs = more language modeling signal per example.
     Documents > paragraphs > sentences.
  
  4. Task Complexity — ★
     Complex tasks (NER, RE) benefit more than simple ones.
     More parameters utilized → more from pretraining.
  
  ═══ TAPT Benefit Tiers ═══
  
  Tier 1 (> +4%): Patent, Clinical NER, Code, Legal
  Tier 2 (2-4%):  Scientific NER, Relation Extraction, Biomedical QA, Academic NLI
  Tier 3 (< 2%):  Sentiment, Topic, Open QA, News Summarization
""")


# ============================================================================
# SECTION 6: DECISION FRAMEWORK
# ============================================================================

def decision_framework():
    """Complete decision framework for TAPT."""
    print("\n\n" + "=" * 70)
    print("  SECTION 6: COMPLETE TAPT DECISION FRAMEWORK")
    print("=" * 70)
    
    print(f"""
  ═══════════════════════════════════════════════════════════
  TAPT DECISION FLOWCHART
  ═══════════════════════════════════════════════════════════
  
  Q1: Do you have a task dataset with text?
  │
  ├─ NO → Cannot do TAPT (need input text).
  │        Consider DAPT with domain corpus.
  │
  └─ YES → Q2: Is your task domain-specific?
     │
     ├─ NO (general text) → TAPT optional, small gains expected
     │     └─ If dataset < 10K: try TAPT anyway (free performance!)
     │
     └─ YES → Q3: How much task data do you have?
        │
        ├─ < 500 examples
        │  → Use Curated TAPT (retrieval to expand data)
        │    k=10-20 neighbors, 80-100 epochs
        │
        ├─ 500-5K examples
        │  → Standard TAPT, 50-80 epochs ★ (sweet spot)
        │    Consider Curated TAPT for extra boost
        │
        ├─ 5K-50K examples
        │  → Standard TAPT, 10-30 epochs
        │    LoRA-TAPT recommended for efficiency
        │
        └─ > 50K examples
           → Standard TAPT, 3-5 epochs
             Very likely to help (large signal)
  
  Q4: Do you also have a domain corpus?
  │
  ├─ YES → Use DAPT + TAPT combo (best results!)
  │        DAPT first, then TAPT, then Task FT
  │
  └─ NO  → TAPT alone is still valuable
  
  
  ═══════════════════════════════════════════════════════════
  TAPT CONFIGURATION QUICK REFERENCE
  ═══════════════════════════════════════════════════════════
""")
    
    @dataclass
    class TAPTConfig:
        scenario: str
        method: str
        lora_rank: Optional[int]
        learning_rate: str
        epochs: str
        extras: str
    
    configs = [
        TAPTConfig(
            "Small data (< 500), specific domain",
            "Curated TAPT",
            8,
            "3e-4",
            "80-100",
            "Retrieve k=10-20 from pool",
        ),
        TAPTConfig(
            "Medium data (500-5K), specific domain",
            "LoRA-TAPT",
            8,
            "3e-4",
            "50-80",
            "Sweet spot ★",
        ),
        TAPTConfig(
            "Large data (5K-50K), specific domain",
            "LoRA-TAPT",
            8,
            "3e-4",
            "10-30",
            "Reliable improvement",
        ),
        TAPTConfig(
            "Very large data (> 50K)",
            "Full or LoRA",
            16,
            "2e-5",
            "3-5",
            "Approaches DAPT behavior",
        ),
        TAPTConfig(
            "After DAPT (any size)",
            "LoRA-TAPT",
            8,
            "3e-4",
            "20-50",
            "Complements DAPT",
        ),
        TAPTConfig(
            "General domain, any size",
            "LoRA-TAPT",
            4,
            "2e-4",
            "10-30",
            "Smaller gains expected",
        ),
        TAPTConfig(
            "Minimal compute budget",
            "LoRA-TAPT",
            4,
            "5e-4",
            "10-20",
            "Quick and effective",
        ),
    ]
    
    print(f"  {'Scenario':<40} │ {'Method':<14} │ {'r':>3} │ {'LR':>6} │ {'Epochs':>7}")
    print(f"  {'─'*40}─┼─{'─'*14}─┼─{'─'*3}─┼─{'─'*6}─┼─{'─'*7}")
    
    for c in configs:
        r_str = str(c.lora_rank) if c.lora_rank else "—"
        print(f"  {c.scenario:<40} │ {c.method:<14} │ {r_str:>3} │ {c.learning_rate:>6} │ {c.epochs:>7}")
    
    print(f"""
  ═══════════════════════════════════════════════════════════
  TAPT vs DAPT: WHEN TO USE WHICH
  ═══════════════════════════════════════════════════════════
  
  ┌────────────────────────┬──────────────────────────────────┐
  │ Situation              │ Recommendation                   │
  ├────────────────────────┼──────────────────────────────────┤
  │ Have domain corpus     │ DAPT + TAPT + FT (best!)         │
  │ only                   │                                  │
  │                        │                                  │
  │ Have task data only    │ TAPT + FT (cheap, effective)     │
  │                        │                                  │
  │ Very limited compute   │ TAPT only (minutes on 1 GPU)    │
  │                        │                                  │
  │ Multiple tasks in      │ DAPT once + TAPT per task       │
  │ same domain            │ (amortize DAPT cost)            │
  │                        │                                  │
  │ Very small task data   │ Curated TAPT (retrieval expand) │
  │ (< 500 examples)       │                                  │
  │                        │                                  │
  │ Task is general (QA,   │ TAPT only if data > 5K          │
  │ news, sentiment)       │                                  │
  └────────────────────────┴──────────────────────────────────┘
  
  ═══════════════════════════════════════════════════════════
  KEY TAKEAWAYS
  ═══════════════════════════════════════════════════════════
  
  1. TAPT is ESSENTIALLY FREE: minutes of compute for 1-5% F1 gain
  2. ALWAYS TRY IT: there's no scenario where it hurts significantly
  3. MORE EPOCHS for smaller data: compensate size with repetition
  4. CURATED TAPT for tiny datasets: retrieval is a game-changer
  5. COMBINE with DAPT: complementary, not redundant
  6. LoRA-TAPT is the sweet spot: efficient, effective, stackable
  7. SPECIFICITY predicts gains: domain-specific tasks benefit most
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  TAPT COMPARISON — STRATEGIES, SCALING, AND DECISION FRAMEWORK  ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    tapt_vs_no_tapt()
    tapt_vs_dapt_comparison()
    epoch_scaling_analysis()
    curated_tapt_effects()
    cross_task_analysis()
    decision_framework()
    
    print("\n" + "=" * 70)
    print("  COMPARISON MODULE COMPLETE")
    print("=" * 70)
    print("""
    Covered:
    ✓ TAPT vs No-TAPT: consistent gains across ALL tasks
    ✓ TAPT vs DAPT vs Combined: complementary benefits
    ✓ Epoch scaling: optimal epochs by dataset size
    ✓ Curated TAPT: 2-4x benefit with retrieval augmentation
    ✓ Cross-task analysis: specificity predicts TAPT benefit
    ✓ Decision framework: complete when/how/why guide
    """)


if __name__ == "__main__":
    main()
