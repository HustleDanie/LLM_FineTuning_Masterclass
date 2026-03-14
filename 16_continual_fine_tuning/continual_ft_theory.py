"""
Continual Fine-Tuning Theory — Catastrophic Forgetting & Importance Estimation
================================================================================

Deep dive into the theory behind continual learning for LLMs:

1. CatastrophicForgetting
   - Why neural networks forget, loss landscape perspective
   - Empirical demonstration with sequential task training

2. FisherInformation
   - Fisher Information Matrix (FIM) and parameter importance
   - Computing FIM for language models
   - Diagonal approximation and its limitations

3. ImportanceEstimation
   - EWC, SI, MAS importance measures
   - Comparing importance estimation methods

4. StabilityPlasticityDilemma
   - The fundamental tradeoff in continual learning
   - How regularization strength affects the balance

5. ForgetMeasures
   - Quantifying what is forgotten and when
   - Task similarity and its effect on interference

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Dict, List, Tuple, Optional


# ============================================================================
# SECTION 1: CATASTROPHIC FORGETTING
# ============================================================================

def catastrophic_forgetting():
    """Why neural networks forget and empirical demonstration."""
    print("=" * 70)
    print("  SECTION 1: CATASTROPHIC FORGETTING")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ What Is Catastrophic Forgetting? ═══
  
  When a neural network is trained on Task B after Task A, the
  gradient updates for B can OVERWRITE weights critical for A.
  
  Loss Landscape View:
  
  θ* for Task A:     . (minimum in A's loss surface)
  θ* for Task B:          . (minimum in B's loss surface)
  
  Training on B moves θ AWAY from A's minimum:
  
  Before B:  A: ★ ← θ is here (good for A)
  After B:   A: .    B: ★ ← θ moved here (good for B, bad for A)
  
  The more different A and B are, the worse the forgetting.
  
  
  ═══ Why It Happens ═══
  
  1. SHARED REPRESENTATIONS: Features for A get repurposed for B
  2. GRADIENT INTERFERENCE: ∇L_B points away from A's optimum
  3. NO EXPLICIT MEMORY: Network has no mechanism to "remember"
  4. PLASTICITY: The very property that enables learning also
     enables forgetting
""")
    
    # Demonstrate catastrophic forgetting
    class SimpleClassifier(nn.Module):
        def __init__(self, in_dim=8, hidden=32, n_classes=4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_classes)
            )
        
        def forward(self, x):
            return self.net(x)
    
    in_dim = 8
    
    # Create two distinct tasks
    def make_task_data(n, task_id, in_dim=8):
        """Create classification data for task_id (0 or 1)."""
        x = torch.randn(n, in_dim)
        if task_id == 0:
            # Task A: classify based on first 4 dims
            y = (x[:, :4].sum(dim=1) > 0).long() + 2 * (x[:, 1] > 0).long()
        else:
            # Task B: classify based on last 4 dims (different features!)
            y = (x[:, 4:].sum(dim=1) > 0).long() + 2 * (x[:, 5] > 0).long()
        return x, y % 4
    
    def evaluate(model, x, y):
        with torch.no_grad():
            preds = model(x).argmax(dim=1)
            return (preds == y).float().mean().item()
    
    # Test data
    test_a_x, test_a_y = make_task_data(200, task_id=0)
    test_b_x, test_b_y = make_task_data(200, task_id=1)
    
    # Train on Task A
    model = SimpleClassifier(in_dim=in_dim, n_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train_a_x, train_a_y = make_task_data(500, task_id=0)
    
    for epoch in range(50):
        logits = model(train_a_x)
        loss = F.cross_entropy(logits, train_a_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    acc_a_after_a = evaluate(model, test_a_x, test_a_y)
    acc_b_after_a = evaluate(model, test_b_x, test_b_y)
    
    print(f"\n  ── After Training on Task A ──")
    print(f"  Task A accuracy: {acc_a_after_a:.1%} ← trained on this")
    print(f"  Task B accuracy: {acc_b_after_a:.1%} ← random chance")
    
    # Now train on Task B (FORGETTING Task A)
    train_b_x, train_b_y = make_task_data(500, task_id=1)
    
    for epoch in range(50):
        logits = model(train_b_x)
        loss = F.cross_entropy(logits, train_b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    acc_a_after_b = evaluate(model, test_a_x, test_a_y)
    acc_b_after_b = evaluate(model, test_b_x, test_b_y)
    
    print(f"\n  ── After Training on Task B (no protection) ──")
    print(f"  Task A accuracy: {acc_a_after_b:.1%} ← CATASTROPHIC FORGETTING!")
    print(f"  Task B accuracy: {acc_b_after_b:.1%} ← learned this")
    
    print(f"\n  Forgetting: {acc_a_after_a - acc_a_after_b:.1%} accuracy drop on Task A")
    
    del model
    
    print(f"""
  KEY INSIGHT: Without any protection mechanism, training on new
  tasks can DESTROY performance on old tasks — even when the
  network has enough capacity for both.
  
  For LLMs, this manifests as:
  • Loss of general language understanding after domain FT
  • Loss of instruction-following after continued pretraining
  • Loss of safety alignment after task-specific training
  • Loss of earlier domains after sequential domain adaptation
""")


# ============================================================================
# SECTION 2: FISHER INFORMATION MATRIX
# ============================================================================

def fisher_information():
    """Fisher Information Matrix for parameter importance."""
    print("\n\n" + "=" * 70)
    print("  SECTION 2: FISHER INFORMATION MATRIX")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ Fisher Information Matrix (FIM) ═══
  
  The Fisher Information Matrix measures how sensitive the model's
  output distribution is to changes in each parameter.
  
  Definition:
    F = E_{{x~D}} [ ∇log p(y|x,θ) · ∇log p(y|x,θ)ᵀ ]
  
  For parameter θ_i:
    F_ii = E [ (∂log p(y|x,θ) / ∂θ_i)² ]
  
  INTERPRETATION:
  • F_ii large → changing θ_i dramatically changes model output
                 → θ_i is IMPORTANT for current task
  • F_ii small → θ_i can be changed freely
                 → θ_i is NOT critical
  
  
  ═══ Connection to EWC ═══
  
  EWC uses F_ii as importance weights:
  
    L_EWC = L_new_task + (λ/2) · Σ_i F_ii · (θ_i - θ*_i)²
  
  This penalizes moving important parameters away from their 
  values after the previous task (θ*).
  
  
  ═══ Diagonal Approximation ═══
  
  Full FIM is |θ|²-sized matrix (impossible for LLMs!).
  We use the DIAGONAL approximation:
  
    F ≈ diag(F_11, F_22, ..., F_nn)
  
  This assumes parameters are independent (not true, but practical).
""")
    
    # Demonstrate Fisher computation
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 3)
        
        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))
    
    model = TinyModel()
    
    # Train on some data
    x = torch.randn(100, 4)
    y = (x[:, 0] + x[:, 1] > 0).long() + (x[:, 2] > 0).long()
    y = y % 3
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(50):
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Compute diagonal Fisher
    def compute_diagonal_fisher(model, data_x, data_y, n_samples=50):
        """
        Compute diagonal Fisher Information Matrix.
        
        F_ii = E[ (∂ log p(y|x,θ) / ∂θ_i)² ]
        
        Approximated by sampling from the empirical distribution.
        """
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        
        model.eval()
        for i in range(min(n_samples, len(data_x))):
            model.zero_grad()
            
            logits = model(data_x[i:i+1])
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Sample y from model's distribution (empirical Fisher)
            # Or use actual labels (practical Fisher)
            target = data_y[i:i+1]
            loss = F.nll_loss(log_probs, target)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.clone() ** 2
        
        # Average
        for name in fisher:
            fisher[name] /= n_samples
        
        model.train()
        return fisher
    
    fisher = compute_diagonal_fisher(model, x, y)
    
    print(f"\n  ── Diagonal Fisher Information ──\n")
    print(f"  {'Parameter':>15} │ {'Shape':>12} │ {'Mean F_ii':>10} │ "
          f"{'Max F_ii':>10} │ {'Min F_ii':>10}")
    print(f"  {'─'*15}─┼─{'─'*12}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}")
    
    for name, f_vals in fisher.items():
        print(f"  {name:>15} │ {str(list(f_vals.shape)):>12} │ "
              f"{f_vals.mean():>10.6f} │ {f_vals.max():>10.6f} │ "
              f"{f_vals.min():>10.6f}")
    
    # Visualize importance distribution
    all_fisher = torch.cat([f.flatten() for f in fisher.values()])
    
    print(f"\n  ── Fisher Importance Distribution ──")
    print(f"  Total parameters:  {len(all_fisher)}")
    print(f"  Mean importance:   {all_fisher.mean():.6f}")
    print(f"  Median importance: {all_fisher.median():.6f}")
    print(f"  Top 10% threshold: {all_fisher.quantile(0.9):.6f}")
    print(f"  Top 1% threshold:  {all_fisher.quantile(0.99):.6f}")
    
    # Show that few parameters are critical
    pct_important = (all_fisher > all_fisher.mean()).float().mean()
    print(f"\n  Parameters above mean importance: {pct_important:.1%}")
    print(f"  → Most parameters have LOW importance (can be modified)")
    print(f"  → A few parameters are CRITICAL (must be protected)")
    
    del model
    
    print(f"""
  ═══ Fisher for LLMs ═══
  
  For a causal LM with parameters θ:
  
    F_ii = E_{{x~D}} [ (∂ log P(x|θ) / ∂θ_i)² ]
  
  Practical computation:
  1. Sample B batches from the training data
  2. For each batch, compute gradients of log-likelihood
  3. Square each gradient element
  4. Average across batches
  
  In practice:
  • Use ~100-1000 samples (not full dataset)
  • Store only diagonal (one value per parameter)
  • For LLMs: still huge but storable as one tensor per param
  • Can compute per-layer Fisher for efficiency
""")


# ============================================================================
# SECTION 3: IMPORTANCE ESTIMATION METHODS
# ============================================================================

def importance_estimation():
    """Compare EWC, SI, and MAS importance estimation."""
    print("\n\n" + "=" * 70)
    print("  SECTION 3: IMPORTANCE ESTIMATION METHODS")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ Three Ways to Estimate Parameter Importance ═══
  
  1. EWC (Elastic Weight Consolidation):
     Importance = Fisher Information (second-order)
     
     Ω_i = F_ii = E[ (∂log p / ∂θ_i)² ]
     
     Based on: curvature of the loss surface
     Intuition: parameters in steep valleys are important
  
  
  2. SI (Synaptic Intelligence):
     Importance = contribution to loss decrease during training
     
     Ω_i = Σ_t (−∂L/∂θ_i · Δθ_i) / (Δθ_i)²
     
     Based on: how much each parameter "helped" during training
     Intuition: parameters that contributed most are important
     Computed ONLINE during training (no extra pass needed)
  
  
  3. MAS (Memory Aware Synapses):
     Importance = sensitivity of output to parameter changes
     
     Ω_i = E[ |∂f(x)/∂θ_i| ]
     
     Based on: how much output changes if parameter changes
     Intuition: sensitive parameters are important
     Computed on UNLABELED data (no labels needed!)
""")
    
    # Implement all three
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(6, 16)
            self.fc2 = nn.Linear(16, 8)
            self.fc3 = nn.Linear(8, 4)
        
        def forward(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            return self.fc3(h)
    
    model = TinyNet()
    
    # Train on a task
    x = torch.randn(200, 6)
    y = ((x[:, 0] > 0).long() + 2 * (x[:, 3] > 0).long()) % 4
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    # Track SI importance DURING training
    si_omega = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    si_prev_params = {n: p.data.clone() for n, p in model.named_parameters()}
    si_running_sum = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    
    for epoch in range(60):
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        
        # SI: accumulate gradient * parameter_change
        for name, param in model.named_parameters():
            if param.grad is not None:
                si_running_sum[name] += -param.grad.data * (
                    param.data - si_prev_params[name])
        
        optimizer.step()
        
        # Update previous params for SI
        for name, param in model.named_parameters():
            si_prev_params[name] = param.data.clone()
    
    # Finalize SI importance
    for name, param in model.named_parameters():
        delta = param.data - si_prev_params[name]
        si_omega[name] = si_running_sum[name] / (delta ** 2 + 1e-8)
        si_omega[name] = F.relu(si_omega[name])  # Only positive contributions
    
    # Compute EWC importance (Fisher)
    ewc_fisher = {}
    model.eval()
    for name, param in model.named_parameters():
        ewc_fisher[name] = torch.zeros_like(param)
    
    for i in range(100):
        model.zero_grad()
        logits = model(x[i:i+1])
        log_p = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(log_p, y[i:i+1])
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                ewc_fisher[name] += param.grad.data ** 2
    
    for name in ewc_fisher:
        ewc_fisher[name] /= 100
    
    # Compute MAS importance (output sensitivity)
    mas_omega = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    
    x_unlabeled = torch.randn(100, 6)  # No labels needed!
    for i in range(100):
        model.zero_grad()
        output = model(x_unlabeled[i:i+1])
        # L2 norm of output as proxy for "output magnitude"
        output_norm = output.norm(2)
        output_norm.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                mas_omega[name] += param.grad.data.abs()
    
    for name in mas_omega:
        mas_omega[name] /= 100
    
    model.train()
    
    # Compare importance estimates
    print(f"\n  ── Importance Estimates Comparison ──\n")
    print(f"  {'Parameter':>12} │ {'EWC (Fisher)':>12} │ {'SI':>12} │ {'MAS':>12}")
    print(f"  {'─'*12}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*12}")
    
    for name in ewc_fisher:
        ewc_val = ewc_fisher[name].mean().item()
        si_val = si_omega[name].mean().item()
        mas_val = mas_omega[name].mean().item()
        print(f"  {name:>12} │ {ewc_val:>12.6f} │ {si_val:>12.6f} │ {mas_val:>12.6f}")
    
    # Correlation between methods
    all_ewc = torch.cat([f.flatten() for f in ewc_fisher.values()])
    all_si = torch.cat([f.flatten() for f in si_omega.values()])
    all_mas = torch.cat([f.flatten() for f in mas_omega.values()])
    
    def rank_correlation(a, b):
        """Spearman-like rank correlation."""
        rank_a = a.argsort().argsort().float()
        rank_b = b.argsort().argsort().float()
        n = len(a)
        d = rank_a - rank_b
        return 1 - 6 * (d ** 2).sum() / (n * (n**2 - 1))
    
    print(f"\n  ── Rank Correlations ──")
    print(f"  EWC vs SI:  {rank_correlation(all_ewc, all_si):.3f}")
    print(f"  EWC vs MAS: {rank_correlation(all_ewc, all_mas):.3f}")
    print(f"  SI vs MAS:  {rank_correlation(all_si, all_mas):.3f}")
    
    del model
    
    print(f"""
  ═══ Comparison Summary ═══
  
  ┌──────────┬────────────┬────────────┬──────────────────────┐
  │ Method   │ When       │ Needs      │ Pros/Cons            │
  │          │ Computed   │ Labels?    │                      │
  ├──────────┼────────────┼────────────┼──────────────────────┤
  │ EWC      │ After task │ Yes        │ Principled, costly   │
  │ SI       │ During     │ Yes        │ Online, approximate  │
  │ MAS      │ After task │ No         │ Flexible, output-dep │
  └──────────┴────────────┴────────────┴──────────────────────┘
  
  For LLMs:
  • EWC Fisher is most common (compute once after each task)
  • SI is attractive (no extra computation pass)
  • MAS is useful when you don't have labeled data
""")


# ============================================================================
# SECTION 4: STABILITY-PLASTICITY DILEMMA
# ============================================================================

def stability_plasticity():
    """The fundamental tradeoff in continual learning."""
    print("\n\n" + "=" * 70)
    print("  SECTION 4: STABILITY-PLASTICITY DILEMMA")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ The Core Tradeoff ═══
  
  STABILITY: Ability to retain previously learned knowledge
  PLASTICITY: Ability to learn new information
  
  These are fundamentally in TENSION:
  
  λ = regularization strength in EWC:
  
    L = L_new + (λ/2) · Σ F_i · (θ_i - θ*_i)²
    
    λ too small → high plasticity, low stability (forgetting)
    λ too large → high stability, low plasticity (can't learn new)
    
    ┌───────────────────────────────────────────────┐
    │           Stability-Plasticity Tradeoff        │
    │                                                │
    │ Plasticity ▲                                   │
    │            │ ★                                  │
    │            │   ★                                │
    │            │     ★ ← sweet spot                 │
    │            │       ★                            │
    │            │         ★                          │
    │            └──────────────→ Stability           │
    │              Low λ → → → High λ                │
    └───────────────────────────────────────────────┘
""")
    
    # Demonstrate the tradeoff
    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(6, 16)
            self.fc2 = nn.Linear(16, 4)
        
        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))
    
    def make_data(task_id, n=300):
        x = torch.randn(n, 6)
        if task_id == 0:
            y = (x[:, :3].sum(1) > 0).long() + 2 * (x[:, 0] > 0.5).long()
        else:
            y = (x[:, 3:].sum(1) > 0).long() + 2 * (x[:, 5] > 0.5).long()
        return x, y % 4
    
    def evaluate_model(model, test_data):
        x, y = test_data
        with torch.no_grad():
            return (model(x).argmax(1) == y).float().mean().item()
    
    test_a = make_data(0, 200)
    test_b = make_data(1, 200)
    
    lambda_values = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    
    print(f"\n  ── EWC with different λ values ──\n")
    print(f"  {'λ':>10} │ {'Task A Acc':>10} │ {'Task B Acc':>10} │ "
          f"{'Average':>8} │ {'Assessment':>20}")
    print(f"  {'─'*10}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*20}")
    
    for lam in lambda_values:
        # Train on Task A
        torch.manual_seed(42)
        model = SmallModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        train_a_x, train_a_y = make_data(0)
        
        for _ in range(60):
            loss = F.cross_entropy(model(train_a_x), train_a_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Compute Fisher
        fisher = {}
        star_params = {}
        for name, param in model.named_parameters():
            fisher[name] = torch.zeros_like(param)
            star_params[name] = param.data.clone()
        
        model.eval()
        for i in range(50):
            model.zero_grad()
            lp = F.log_softmax(model(train_a_x[i:i+1]), dim=-1)
            loss = F.nll_loss(lp, train_a_y[i:i+1])
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        for name in fisher:
            fisher[name] /= 50
        model.train()
        
        # Train on Task B with EWC
        train_b_x, train_b_y = make_data(1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        
        for _ in range(60):
            logits = model(train_b_x)
            loss_b = F.cross_entropy(logits, train_b_y)
            
            # EWC penalty
            ewc_loss = 0.0
            for name, param in model.named_parameters():
                ewc_loss += (fisher[name] * (param - star_params[name]) ** 2).sum()
            
            total_loss = loss_b + (lam / 2) * ewc_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        acc_a = evaluate_model(model, test_a)
        acc_b = evaluate_model(model, test_b)
        avg = (acc_a + acc_b) / 2
        
        if lam == 0:
            assessment = "No protection"
        elif acc_a < 0.35:
            assessment = "Too plastic"
        elif acc_b < 0.35:
            assessment = "Too stable"
        else:
            assessment = "← Good balance!"
        
        print(f"  {lam:>10.0f} │ {acc_a:>9.1%} │ {acc_b:>9.1%} │ "
              f"{avg:>7.1%} │ {assessment:>20}")
        
        del model
    
    print(f"""
  INTERPRETATION:
  • λ=0:     No EWC → full forgetting of Task A
  • λ=small: Some retention, good plasticity  
  • λ=right: Best average performance ← SWEET SPOT
  • λ=huge:  Perfect retention, but can't learn Task B
  
  
  ═══ Practical Guidelines for λ ═══
  
  • Start with λ = 1000-5000 for small models
  • For LLMs: λ = 0.1-10 (they have more capacity)
  • Scale λ by 1/learning_rate for consistency
  • Adaptive λ: increase for important tasks, decrease for similar tasks
  • When using LoRA: λ can be much smaller (fewer params change)
""")


# ============================================================================
# SECTION 5: FORGETTING MEASURES
# ============================================================================

def forgetting_measures():
    """Quantifying forgetting in continual learning."""
    print("\n\n" + "=" * 70)
    print("  SECTION 5: FORGETTING MEASURES")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ Quantifying Forgetting ═══
  
  Given T tasks trained sequentially, let A_{{i,j}} be
  accuracy on task j after training through task i.
  
  
  1. BACKWARD TRANSFER (BWT):
     Average effect on old tasks after learning new ones.
     
     BWT = (1/(T-1)) · Σ_{{j=1..T-1}} (A_{{T,j}} - A_{{j,j}})
     
     BWT < 0 → forgetting (bad)
     BWT ≈ 0 → no forgetting (good)
     BWT > 0 → old tasks IMPROVED (positive transfer!)
  
  
  2. FORWARD TRANSFER (FWT):  
     How much previous learning helps new tasks.
     
     FWT = (1/(T-1)) · Σ_{{j=2..T}} (A_{{j-1,j}} - baseline_j)
     
     FWT > 0 → positive transfer (learning A helps learn B)
     FWT < 0 → negative transfer (A hurts B)
  
  
  3. AVERAGE ACCURACY (AA):
     Overall performance across all tasks.
     
     AA = (1/T) · Σ_{{j=1..T}} A_{{T,j}}
  
  
  4. FORGETTING MEASURE (FM):
     Maximum loss per task across training.
     
     FM_j = max_{{k=1..T}} A_{{k,j}} - A_{{T,j}}
     FM = (1/(T-1)) · Σ_{{j=1..T-1}} FM_j
""")
    
    # Simulate a 4-task continual learning scenario
    n_tasks = 4
    
    # Accuracy matrix A[i][j] = accuracy on task j after training through task i
    # Rows: training stage, Columns: task evaluation
    accuracy_matrix = {
        "No Protection": [
            [0.85, 0.25, 0.25, 0.25],  # After task 1
            [0.35, 0.82, 0.25, 0.25],  # After task 2
            [0.28, 0.40, 0.88, 0.25],  # After task 3
            [0.27, 0.30, 0.42, 0.86],  # After task 4
        ],
        "EWC": [
            [0.85, 0.25, 0.25, 0.25],
            [0.72, 0.78, 0.25, 0.25],
            [0.65, 0.68, 0.82, 0.25],
            [0.60, 0.62, 0.72, 0.80],
        ],
        "Experience Replay": [
            [0.85, 0.25, 0.25, 0.25],
            [0.78, 0.80, 0.25, 0.25],
            [0.72, 0.74, 0.84, 0.25],
            [0.68, 0.70, 0.76, 0.82],
        ],
        "Task Adapters": [
            [0.85, 0.25, 0.25, 0.25],
            [0.84, 0.80, 0.25, 0.25],
            [0.83, 0.79, 0.85, 0.25],
            [0.82, 0.78, 0.84, 0.83],
        ],
    }
    
    def compute_metrics(A, T=4):
        """Compute continual learning metrics from accuracy matrix."""
        # Average Accuracy
        aa = sum(A[T-1]) / T
        
        # Backward Transfer
        bwt = sum(A[T-1][j] - A[j][j] for j in range(T-1)) / (T-1)
        
        # Forgetting Measure
        fm_per_task = []
        for j in range(T-1):
            max_acc = max(A[k][j] for k in range(T))
            fm_per_task.append(max_acc - A[T-1][j])
        fm = sum(fm_per_task) / (T-1)
        
        return aa, bwt, fm
    
    print(f"\n  ── 4-Task Continual Learning Metrics ──\n")
    
    # Print accuracy matrices
    for method, A in accuracy_matrix.items():
        print(f"  {method}:")
        print(f"  {'Stage':>8} │ {'Task 1':>8} {'Task 2':>8} {'Task 3':>8} {'Task 4':>8}")
        print(f"  {'─'*8}─┼─{'─'*35}")
        for i in range(n_tasks):
            row = f"  {'After '+str(i+1):>8} │ "
            for j in range(n_tasks):
                if j <= i:
                    val = A[i][j]
                    # Highlight diagonal (just learned) and <0.5 (forgotten)
                    row += f"{val:>7.1%} "
                else:
                    row += f"{'---':>8}"
            print(row)
        
        aa, bwt, fm = compute_metrics(A)
        print(f"  AA={aa:.1%}  BWT={bwt:+.1%}  FM={fm:.1%}")
        print()
    
    # Summary table
    print(f"\n  ── Summary ──\n")
    print(f"  {'Method':>20} │ {'Avg Acc':>8} │ {'BWT':>8} │ {'Forgetting':>10}")
    print(f"  {'─'*20}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*10}")
    
    for method, A in accuracy_matrix.items():
        aa, bwt, fm = compute_metrics(A)
        print(f"  {method:>20} │ {aa:>7.1%} │ {bwt:>+7.1%} │ {fm:>9.1%}")
    
    print(f"""
  INTERPRETATION:
  • No Protection: Severe forgetting (BWT very negative)
  • EWC: Moderate protection, some forgetting remains
  • Experience Replay: Good balance with data mixing
  • Task Adapters: Near-zero forgetting (separate params per task)
  
  BEST APPROACH DEPENDS ON:
  • How different the tasks are (more different → more forgetting)
  • Memory budget (replay needs stored data)
  • Task count (adapters scale linearly with tasks)
  • Whether tasks share useful features (forward transfer)
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  CONTINUAL FINE-TUNING THEORY — FORGETTING & IMPORTANCE          ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    catastrophic_forgetting()
    fisher_information()
    importance_estimation()
    stability_plasticity()
    forgetting_measures()
    
    print("\n" + "=" * 70)
    print("  THEORY MODULE COMPLETE")
    print("=" * 70)
    print("""
    Covered:
    ✓ Catastrophic forgetting — why it happens, empirical demo
    ✓ Fisher Information Matrix — parameter importance
    ✓ Importance estimation — EWC vs SI vs MAS compared
    ✓ Stability-plasticity dilemma — λ tradeoff analysis
    ✓ Forgetting measures — BWT, FWT, AA, FM metrics
    """)


if __name__ == "__main__":
    main()
