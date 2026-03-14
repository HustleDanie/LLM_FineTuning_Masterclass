"""
DAPT Comparison — Analysis of Domain-Adaptive Pretraining Strategies
=====================================================================

Comprehensive comparison framework:

1. DAPT vs No-DAPT — head-to-head on downstream tasks
2. Domain Distance Effects — how distance affects DAPT benefit
3. Data Scaling Analysis — tokens needed vs domain distance
4. Strategy Comparison — Full vs LoRA vs Curriculum DAPT
5. Compute Budget Analysis — cost vs benefit framework
6. Decision Framework — when and how to use DAPT

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
# SECTION 1: DAPT vs NO-DAPT HEAD-TO-HEAD
# ============================================================================

def dapt_vs_no_dapt():
    """Compare models with and without DAPT on downstream tasks."""
    print("=" * 70)
    print("  SECTION 1: DAPT vs NO-DAPT HEAD-TO-HEAD")
    print("=" * 70)
    
    print(f"""
  ═══ Experimental Setup ═══
  
  Compare two paths:
  
  Path A: Base Model → Task Fine-Tuning
  Path B: Base Model → DAPT → Task Fine-Tuning
  
  We simulate results across different domains and tasks.
""")
    
    torch.manual_seed(42)
    random.seed(42)
    
    # Simulated results based on Gururangan et al. 2020 patterns
    experiments = [
        # (Domain, Task, Base+FT F1, DAPT+FT F1)
        ("Biomedical", "ChemProt (relation)", 81.9, 84.2),
        ("Biomedical", "RCT (abstract)", 87.2, 87.6),
        ("Computer Sci", "ACL-ARC (citation)", 63.0, 75.4),
        ("Computer Sci", "SciERC (entity)", 77.3, 80.8),
        ("News", "HyperPartisan", 86.6, 88.2),
        ("News", "AGNews (topic)", 93.8, 94.5),
        ("Reviews", "IMDB (sentiment)", 95.0, 95.4),
        ("Reviews", "Helpfulness", 65.1, 68.3),
        ("Legal", "Contract NLI", 82.4, 87.1),
        ("Legal", "Case Outcome", 71.2, 76.8),
        ("Code", "Defect Detection", 64.3, 71.2),
        ("Code", "Clone Detection", 93.1, 95.9),
    ]
    
    print(f"  {'Domain':<14} │ {'Task':<22} │ {'Base+FT':>7} │ {'DAPT+FT':>7} │ {'Δ':>6} │ {'Improvement':>14}")
    print(f"  {'─'*14}─┼─{'─'*22}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*14}")
    
    total_delta = 0
    gains_by_domain = {}
    
    for domain, task, base_f1, dapt_f1 in experiments:
        delta = dapt_f1 - base_f1
        total_delta += delta
        
        if domain not in gains_by_domain:
            gains_by_domain[domain] = []
        gains_by_domain[domain].append(delta)
        
        # Visual indicator
        bars = "█" * max(1, int(delta * 2))
        
        print(f"  {domain:<14} │ {task:<22} │ {base_f1:>6.1f}% │ {dapt_f1:>6.1f}% │ {delta:>+5.1f}% │ {bars}")
    
    avg_delta = total_delta / len(experiments)
    print(f"\n  Average improvement: {avg_delta:+.1f}%")
    
    print(f"\n  ── Gains by Domain ──")
    for domain, gains in gains_by_domain.items():
        avg = sum(gains) / len(gains)
        print(f"  {domain:<14}: {avg:+.1f}%  {'★' if avg > 3 else '  '}")
    
    print(f"""
  ═══ Key Findings ═══
  
  1. DAPT helps across ALL domains tested
  2. Largest gains on domains FURTHEST from pretraining data:
     - CS papers: +8.5% (specialized vocabulary)
     - Legal: +5.2% (domain-specific language)
     - Code: +4.9% (structural differences)
  3. Smallest gains on domains CLOSEST to pretraining:
     - News: +1.2% (similar to Common Crawl)
     - Reviews: +1.8% (web text overlap)
  
  DAPT benefit scales with domain distance!
""")


# ============================================================================
# SECTION 2: DOMAIN DISTANCE EFFECTS
# ============================================================================

def domain_distance_effects():
    """How domain distance affects DAPT benefit."""
    print("\n\n" + "=" * 70)
    print("  SECTION 2: DOMAIN DISTANCE EFFECTS")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Define domains at different distances from general pretraining data
    domains = [
        # (Domain, Distance 0-1, DAPT benefit %, Vocab overlap %)
        ("News/Web",          0.10, 1.2, 92),
        ("Wikipedia",         0.15, 0.8, 95),
        ("Product Reviews",   0.20, 1.8, 88),
        ("Social Media",      0.25, 2.4, 80),
        ("Scientific Papers",  0.40, 4.5, 65),
        ("Legal Documents",   0.50, 5.2, 58),
        ("Biomedical Text",   0.60, 6.8, 48),
        ("Patent Filings",    0.65, 7.2, 45),
        ("Source Code",       0.75, 8.5, 30),
        ("Mathematical Text", 0.80, 9.1, 25),
        ("Clinical Notes",    0.85, 10.2, 20),
        ("Ancient Languages", 0.95, 12.0, 8),
    ]
    
    print(f"""
  ═══ Domain Distance vs DAPT Benefit ═══
  
  Domain distance measured by: KL divergence of token distributions
  DAPT benefit: average F1 improvement on downstream tasks
  
  {'Domain':<20} │ {'Distance':>8} │ {'DAPT Δ':>6} │ {'Vocab Overlap':>13} │ Visualization
  {'─'*20}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*13}─┼─{'─'*30}
""")
    
    for domain, dist, benefit, overlap in domains:
        bar_dist = "░" * int(dist * 20)
        bar_benefit = "█" * int(benefit * 1.5)
        print(f"  {domain:<20} │ {dist:>8.2f} │ {benefit:>+5.1f}% │ {overlap:>12}% │ {bar_benefit}")
    
    # Compute correlation
    distances = [d[1] for d in domains]
    benefits = [d[2] for d in domains]
    
    mean_dist = sum(distances) / len(distances)
    mean_ben = sum(benefits) / len(benefits)
    
    cov = sum((d - mean_dist) * (b - mean_ben) for d, b in zip(distances, benefits))
    std_dist = math.sqrt(sum((d - mean_dist)**2 for d in distances))
    std_ben = math.sqrt(sum((b - mean_ben)**2 for b in benefits))
    
    correlation = cov / (std_dist * std_ben) if std_dist * std_ben > 0 else 0
    
    print(f"\n  Pearson Correlation (distance ↔ DAPT benefit): {correlation:.3f}")
    
    print(f"""
  ═══ The Domain Distance Rule ═══
  
  DAPT benefit ≈ α × domain_distance + ε
  
  Where:
    α ≈ 12.0 (F1 points per unit distance)
    ε ≈ 0.5 (baseline noise)
  
  Practical implications:
  ┌─────────────────┬───────────────┬──────────────────────────┐
  │ Distance Range  │ Expected Gain │ Recommendation           │
  ├─────────────────┼───────────────┼──────────────────────────┤
  │ 0.0 - 0.2       │ 0-2%          │ DAPT optional            │
  │ 0.2 - 0.5       │ 2-5%          │ DAPT recommended         │
  │ 0.5 - 0.8       │ 5-10%         │ DAPT strongly recommended│
  │ 0.8 - 1.0       │ 10%+          │ DAPT essential           │
  └─────────────────┴───────────────┴──────────────────────────┘
""")


# ============================================================================
# SECTION 3: DATA SCALING ANALYSIS
# ============================================================================

def data_scaling_analysis():
    """How much domain data is needed for effective DAPT?"""
    print("\n\n" + "=" * 70)
    print("  SECTION 3: DATA SCALING ANALYSIS")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ Data Scaling: More is Better, But Diminishing Returns ═══
  
  Key question: How many domain tokens do I need?
  Answer: Depends on domain distance and model size.
""")
    
    # Simulate data scaling experiments
    # columns: token count, close domain PPL, medium domain PPL, far domain PPL
    token_counts = ["1M", "5M", "10M", "50M", "100M", "500M", "1B", "5B"]
    token_nums = [1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9, 5e9]
    
    # DAPT benefit (F1 gain) at different data sizes and distances
    # Simulated based on scaling law patterns
    def scaling_benefit(tokens, distance, max_benefit=12.0):
        """Log-linear scaling of DAPT benefit with data size."""
        # Benefit scales as log(tokens) * distance
        log_tokens = math.log10(tokens)
        saturation = 1 - math.exp(-log_tokens / 9)  # Saturates around 1B
        return max_benefit * distance * saturation
    
    print(f"  {'Tokens':>8} │ {'Close (d=0.2)':>14} │ {'Medium (d=0.5)':>14} │ {'Far (d=0.8)':>14}")
    print(f"  {'─'*8}─┼─{'─'*14}─┼─{'─'*14}─┼─{'─'*14}")
    
    for label, n_tokens in zip(token_counts, token_nums):
        close = scaling_benefit(n_tokens, 0.2)
        medium = scaling_benefit(n_tokens, 0.5)
        far = scaling_benefit(n_tokens, 0.8)
        print(f"  {label:>8} │ {close:>+13.1f}% │ {medium:>+13.1f}% │ {far:>+13.1f}%")
    
    print(f"""
  ═══ Data Efficiency Insights ═══
  
  1. For CLOSE domains (News, Web):
     - 1-5M tokens sufficient for marginal gains
     - Diminishing returns after 10M tokens
     - Better to focus on quality over quantity
  
  2. For MEDIUM domains (Scientific, Legal):
     - 10-50M tokens recommended minimum
     - Clear gains up to 100M tokens
     - Quality filtering very important
  
  3. For FAR domains (Code, Medical, Math):
     - 50-500M tokens recommended
     - Gains visible even at 1B+ tokens
     - May need vocabulary expansion
  
  ═══ Tokens-per-Parameter Rule of Thumb ═══
  
  ┌──────────────────┬───────────────────────────┐
  │ Model Size       │ Recommended DAPT Tokens   │
  ├──────────────────┼───────────────────────────┤
  │ 125M params      │ 10M - 100M tokens         │
  │ 350M params      │ 50M - 500M tokens         │
  │ 1.3B params      │ 100M - 2B tokens          │
  │ 7B params        │ 500M - 10B tokens         │
  │ 13B params       │ 1B - 20B tokens           │
  │ 70B params       │ 5B - 100B tokens          │
  └──────────────────┴───────────────────────────┘
  
  Rule: DAPT tokens ≈ 0.1x to 1x model parameters
""")


# ============================================================================
# SECTION 4: STRATEGY COMPARISON
# ============================================================================

def strategy_comparison():
    """Compare Full DAPT vs LoRA-DAPT vs Curriculum DAPT."""
    print("\n\n" + "=" * 70)
    print("  SECTION 4: STRATEGY COMPARISON")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ DAPT Strategy Comparison ═══
  
  Comparing 5 DAPT strategies across key metrics.
""")
    
    @dataclass
    class StrategyResult:
        name: str
        domain_ppl_reduction: float   # % reduction in domain PPL
        general_ppl_increase: float   # % increase in general PPL
        task_f1_gain: float           # F1 improvement on downstream
        trainable_params_pct: float   # % of total params trained
        memory_gb: float              # GPU memory required (7B model)
        hours_a100: float             # Training time on A100
        storage_gb: float             # Output model size
        multi_domain: bool            # Supports multi-domain efficiently
    
    strategies = [
        StrategyResult(
            name="Full DAPT",
            domain_ppl_reduction=42.0,
            general_ppl_increase=12.0,
            task_f1_gain=5.8,
            trainable_params_pct=100.0,
            memory_gb=28.0,
            hours_a100=48.0,
            storage_gb=14.0,
            multi_domain=False,
        ),
        StrategyResult(
            name="LoRA-DAPT (r=16)",
            domain_ppl_reduction=35.0,
            general_ppl_increase=3.0,
            task_f1_gain=5.0,
            trainable_params_pct=0.5,
            memory_gb=18.0,
            hours_a100=24.0,
            storage_gb=0.02,
            multi_domain=True,
        ),
        StrategyResult(
            name="LoRA-DAPT (r=64)",
            domain_ppl_reduction=40.0,
            general_ppl_increase=5.0,
            task_f1_gain=5.5,
            trainable_params_pct=2.0,
            memory_gb=20.0,
            hours_a100=28.0,
            storage_gb=0.08,
            multi_domain=True,
        ),
        StrategyResult(
            name="Curriculum DAPT",
            domain_ppl_reduction=38.0,
            general_ppl_increase=6.0,
            task_f1_gain=5.4,
            trainable_params_pct=100.0,
            memory_gb=28.0,
            hours_a100=56.0,
            storage_gb=14.0,
            multi_domain=False,
        ),
        StrategyResult(
            name="Data Mixed DAPT",
            domain_ppl_reduction=34.0,
            general_ppl_increase=4.0,
            task_f1_gain=5.1,
            trainable_params_pct=100.0,
            memory_gb=28.0,
            hours_a100=52.0,
            storage_gb=14.0,
            multi_domain=False,
        ),
    ]
    
    # Quality comparison
    print(f"\n  ── Quality Metrics ──\n")
    print(f"  {'Strategy':<22} │ {'Domain PPL ↓':>12} │ {'General PPL ↑':>13} │ {'Task F1 Δ':>9}")
    print(f"  {'─'*22}─┼─{'─'*12}─┼─{'─'*13}─┼─{'─'*9}")
    
    for s in strategies:
        print(f"  {s.name:<22} │ {s.domain_ppl_reduction:>+11.1f}% │ {s.general_ppl_increase:>+12.1f}% │ {s.task_f1_gain:>+8.1f}%")
    
    # Efficiency comparison
    print(f"\n  ── Efficiency Metrics (7B model) ──\n")
    print(f"  {'Strategy':<22} │ {'Params':>7} │ {'Memory':>8} │ {'Time':>8} │ {'Storage':>9} │ {'Multi-Domain':>12}")
    print(f"  {'─'*22}─┼─{'─'*7}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*9}─┼─{'─'*12}")
    
    for s in strategies:
        multi = "✓ Yes" if s.multi_domain else "✗ No"
        print(f"  {s.name:<22} │ {s.trainable_params_pct:>6.1f}% │ {s.memory_gb:>6.1f}GB │ {s.hours_a100:>6.0f}h  │ {s.storage_gb:>7.2f}GB │ {multi:>12}")
    
    # Compute cost-effectiveness
    print(f"\n  ── Cost-Effectiveness Score ──\n")
    print(f"  Score = (Task F1 Gain) / (GPU Hours × Memory)\n")
    
    for s in strategies:
        score = s.task_f1_gain / (s.hours_a100 * s.memory_gb) * 1000
        bar = "█" * int(score * 3)
        print(f"  {s.name:<22} │ score: {score:>5.2f}  {bar}")
    
    print(f"""
  ═══ Strategy Recommendations ═══
  
  ┌────────────────────────┬──────────────────────────────────────┐
  │ Scenario               │ Recommended Strategy                 │
  ├────────────────────────┼──────────────────────────────────────┤
  │ Best quality           │ Full DAPT                            │
  │ Best efficiency        │ LoRA-DAPT (r=16-32)            ★    │
  │ Multiple domains       │ LoRA-DAPT (separate adapters)  ★    │
  │ Prevent forgetting     │ Curriculum DAPT or Data Mixing       │
  │ Limited GPU memory     │ LoRA-DAPT (r=16)                     │
  │ Far domain + quality   │ LoRA-DAPT (r=64) or Full DAPT       │
  │ Limited domain data    │ LoRA-DAPT (lower rank prevents       │
  │                        │   overfitting on small corpora)      │
  └────────────────────────┴──────────────────────────────────────┘
  
  ★ = Most scenarios favor LoRA-DAPT for best cost/quality tradeoff
""")


# ============================================================================
# SECTION 5: COMPUTE BUDGET ANALYSIS
# ============================================================================

def compute_budget_analysis():
    """Analyze DAPT cost and ROI across different compute budgets."""
    print("\n\n" + "=" * 70)
    print("  SECTION 5: COMPUTE BUDGET ANALYSIS")
    print("=" * 70)
    
    print(f"""
  ═══ DAPT Cost Estimation Framework ═══
  
  Estimate GPU hours and cost for your DAPT project.
""")
    
    @dataclass
    class ComputeEstimate:
        """Estimate compute cost for a DAPT configuration."""
        model_size_b: float      # Billions of parameters
        tokens_b: float          # Billions of tokens
        method: str              # "full" or "lora"
        gpu_type: str            # "A100_40" or "A100_80" or "H100"
        
        @property
        def gpu_memory_gb(self) -> float:
            """Estimate peak GPU memory."""
            base_memory = self.model_size_b * 2  # FP16
            if self.method == "full":
                optimizer_memory = self.model_size_b * 8  # AdamW states
                gradient_memory = self.model_size_b * 2
                return base_memory + optimizer_memory + gradient_memory
            else:  # LoRA
                return base_memory + self.model_size_b * 0.5  # Much less
        
        @property
        def gpus_needed(self) -> int:
            """Estimate number of GPUs needed."""
            gpu_mem = {"A100_40": 40, "A100_80": 80, "H100": 80}
            avail = gpu_mem.get(self.gpu_type, 40)
            return max(1, math.ceil(self.gpu_memory_gb / avail))
        
        @property
        def tokens_per_second(self) -> float:
            """Estimate throughput (tokens/second/GPU)."""
            # Rough estimates based on benchmarks
            tps = {"A100_40": 10000, "A100_80": 15000, "H100": 25000}
            base_tps = tps.get(self.gpu_type, 10000)
            
            # Scale with model size (larger = slower)
            size_factor = 7.0 / self.model_size_b  # Normalized to 7B
            
            # LoRA is ~1.3x faster
            method_factor = 1.3 if self.method == "lora" else 1.0
            
            return base_tps * size_factor * method_factor
        
        @property
        def total_hours(self) -> float:
            """Total GPU-hours for DAPT."""
            total_tokens = self.tokens_b * 1e9
            seconds = total_tokens / (self.tokens_per_second * self.gpus_needed)
            return seconds / 3600
        
        @property
        def wall_clock_hours(self) -> float:
            """Wall-clock time."""
            return self.total_hours / self.gpus_needed
        
        @property
        def cost_usd(self) -> float:
            """Estimated cost (cloud GPU pricing)."""
            hourly = {"A100_40": 1.50, "A100_80": 2.50, "H100": 3.50}
            rate = hourly.get(self.gpu_type, 2.0)
            return self.total_hours * rate
    
    # Run estimates
    configs = [
        # (Size, Tokens, Method, GPU)
        (1.3,  0.1,   "lora", "A100_40"),
        (1.3,  1.0,   "full", "A100_40"),
        (7.0,  0.5,   "lora", "A100_80"),
        (7.0,  2.0,   "full", "A100_80"),
        (7.0,  2.0,   "lora", "A100_80"),
        (13.0, 1.0,   "lora", "A100_80"),
        (13.0, 5.0,   "full", "H100"),
        (70.0, 5.0,   "lora", "H100"),
        (70.0, 20.0,  "full", "H100"),
    ]
    
    print(f"  {'Model':>6} │ {'Tokens':>6} │ {'Method':>6} │ {'GPU':>8} │ {'#GPUs':>5} │ {'Hours':>7} │ {'Wall':>7} │ {'Cost':>8}")
    print(f"  {'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*8}─┼─{'─'*5}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*8}")
    
    for size, tokens, method, gpu in configs:
        est = ComputeEstimate(size, tokens, method, gpu)
        cost_str = f"${est.cost_usd:,.0f}"
        print(f"  {size:>5.1f}B │ {tokens:>5.1f}B │ {method:>6} │ {gpu:>8} │ {est.gpus_needed:>5} │ {est.total_hours:>6.0f}h │ {est.wall_clock_hours:>6.0f}h │ {cost_str:>8}")
    
    print(f"""
  ═══ Cost Savings with LoRA-DAPT ═══
  
  7B model, 2B tokens, A100_80:
  - Full DAPT: ~{ComputeEstimate(7.0, 2.0, 'full', 'A100_80').cost_usd:,.0f} USD
  - LoRA DAPT: ~{ComputeEstimate(7.0, 2.0, 'lora', 'A100_80').cost_usd:,.0f} USD
  - Savings: ~{(1 - ComputeEstimate(7.0, 2.0, 'lora', 'A100_80').cost_usd / ComputeEstimate(7.0, 2.0, 'full', 'A100_80').cost_usd) * 100:.0f}% with ~90% quality!
  
  13B model, 5B tokens, H100:
  - Full DAPT: ~{ComputeEstimate(13.0, 5.0, 'full', 'H100').cost_usd:,.0f} USD
  - LoRA DAPT: ~{ComputeEstimate(13.0, 5.0, 'lora', 'H100').cost_usd:,.0f} USD
""")


# ============================================================================
# SECTION 6: DECISION FRAMEWORK
# ============================================================================

def decision_framework():
    """Complete decision framework for when and how to use DAPT."""
    print("\n\n" + "=" * 70)
    print("  SECTION 6: DECISION FRAMEWORK")
    print("=" * 70)
    
    print(f"""
  ═══════════════════════════════════════════════════════════
  DAPT DECISION FLOWCHART
  ═══════════════════════════════════════════════════════════
  
  Q1: Is your target domain different from general web text?
  │
  ├─ NO → Skip DAPT. Direct fine-tuning is sufficient.
  │
  └─ YES → Q2: How different is your domain?
     │
     ├─ CLOSE (news, reviews, social media)
     │  → Q3: Do you have >10M domain tokens?
     │     ├─ YES → LoRA-DAPT (r=8-16), 1 epoch
     │     └─ NO  → Skip DAPT, use TAPT instead
     │
     ├─ MEDIUM (scientific, legal, financial)
     │  → Q3: Do you have >50M domain tokens?
     │     ├─ YES → LoRA-DAPT (r=32), 2-3 epochs ★
     │     └─ NO  → LoRA-DAPT (r=16), 1 epoch + data mixing
     │
     └─ FAR (code, medical, math, niche)
        → Q3: Do you have >100M domain tokens?
           ├─ YES → Full DAPT or LoRA-DAPT (r=64), 2-3 epochs
           │        Consider vocabulary expansion
           └─ NO  → LoRA-DAPT (r=32), careful data mixing
                    + TAPT with task-specific data
  
  
  ═══════════════════════════════════════════════════════════
  DAPT CONFIGURATION REFERENCE
  ═══════════════════════════════════════════════════════════
""")
    
    @dataclass 
    class DAPTConfig:
        scenario: str
        method: str
        lora_rank: Optional[int]
        learning_rate: str
        epochs: str
        data_min: str
        extra_notes: str
    
    configs = [
        DAPTConfig(
            "Close domain, abundant data",
            "LoRA-DAPT",
            16,
            "3e-4",
            "1-2",
            "10M tokens",
            "Fast and sufficient",
        ),
        DAPTConfig(
            "Close domain, limited data",
            "Skip DAPT",
            None,
            "—",
            "—",
            "—",
            "Use TAPT instead",
        ),
        DAPTConfig(
            "Medium domain, abundant data",
            "LoRA-DAPT",
            32,
            "3e-4",
            "2-3",
            "50M tokens",
            "Best cost/quality ★",
        ),
        DAPTConfig(
            "Medium domain, limited data",
            "LoRA-DAPT + Mix",
            16,
            "2e-4",
            "2",
            "10M tokens",
            "Mix 80/20 domain/general",
        ),
        DAPTConfig(
            "Far domain, abundant data",
            "Full or LoRA-64",
            64,
            "2e-5 / 5e-4",
            "2-3",
            "100M tokens",
            "Full gives best quality",
        ),
        DAPTConfig(
            "Far domain, limited data",
            "LoRA-DAPT + Mix",
            32,
            "3e-4",
            "2-3",
            "10M tokens",
            "Mix + curriculum",
        ),
        DAPTConfig(
            "Multiple target domains",
            "LoRA-DAPT",
            32,
            "3e-4",
            "2 per domain",
            "10M per domain",
            "Separate adapters!",
        ),
        DAPTConfig(
            "Minimal compute budget",
            "LoRA-DAPT",
            8,
            "5e-4",
            "1",
            "5M tokens",
            "Minimum viable DAPT",
        ),
    ]
    
    print(f"  {'Scenario':<32} │ {'Method':<16} │ {'Rank':>4} │ {'LR':>12} │ {'Epochs':>6} │ {'Min Data':>10}")
    print(f"  {'─'*32}─┼─{'─'*16}─┼─{'─'*4}─┼─{'─'*12}─┼─{'─'*6}─┼─{'─'*10}")
    
    for c in configs:
        rank = str(c.lora_rank) if c.lora_rank else "—"
        print(f"  {c.scenario:<32} │ {c.method:<16} │ {rank:>4} │ {c.learning_rate:>12} │ {c.epochs:>6} │ {c.data_min:>10}")
    
    print(f"\n  ── Notes ──")
    for c in configs:
        if c.extra_notes:
            print(f"  • {c.scenario}: {c.extra_notes}")
    
    print(f"""
  ═══════════════════════════════════════════════════════════
  QUICK REFERENCE: DAPT CHECKLIST
  ═══════════════════════════════════════════════════════════
  
  Before DAPT:
  ☐ Measure domain distance (vocab overlap, tokenizer fertility)
  ☐ Prepare clean domain corpus (dedup, quality filter)
  ☐ Choose method (Full vs LoRA) based on budget
  ☐ Set baseline: evaluate base model PPL on domain
  ☐ Prepare general eval set (for forgetting detection)
  
  During DAPT:
  ☐ Monitor domain PPL (should decrease)
  ☐ Monitor general PPL (should not increase >15-20%)
  ☐ Check gradient norms (should be stable)
  ☐ Save checkpoints regularly
  
  After DAPT:
  ☐ Evaluate domain PPL improvement
  ☐ Evaluate general benchmark preservation
  ☐ Run downstream task (with fine-tuning)
  ☐ Compare against non-DAPT baseline
  ☐ Document configuration for reproducibility
  
  ═══════════════════════════════════════════════════════════
  KEY TAKEAWAYS
  ═══════════════════════════════════════════════════════════
  
  1. DAPT WORKS: Consistent improvements across all domains
  2. DISTANCE MATTERS: Bigger gains for more distant domains
  3. LoRA-DAPT IS THE SWEET SPOT: 90% of full DAPT quality
     at 30% of the cost, with less forgetting
  4. DATA QUALITY > QUANTITY: Clean 10M tokens beats noisy 100M
  5. COMBINE DAPT + TAPT: Complementary, not redundant
  6. MONITOR FORGETTING: Always track general capabilities
  7. USE WARMUP: Prevent sudden distribution shifts
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  DAPT COMPARISON — STRATEGIES, SCALING, AND DECISION FRAMEWORK  ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    dapt_vs_no_dapt()
    domain_distance_effects()
    data_scaling_analysis()
    strategy_comparison()
    compute_budget_analysis()
    decision_framework()
    
    print("\n" + "=" * 70)
    print("  COMPARISON MODULE COMPLETE")
    print("=" * 70)
    print("""
    Covered:
    ✓ DAPT vs No-DAPT: consistent gains across all domains
    ✓ Domain distance: strong correlation with DAPT benefit
    ✓ Data scaling: log-linear with diminishing returns
    ✓ Strategy comparison: LoRA-DAPT best cost/quality
    ✓ Compute budget: cost estimation framework
    ✓ Decision framework: when, how, and why to use DAPT
    """)


if __name__ == "__main__":
    main()
