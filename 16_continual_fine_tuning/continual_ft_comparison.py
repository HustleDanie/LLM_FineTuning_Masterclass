"""
Continual Fine-Tuning Comparison — Regularization vs Replay vs Architecture
=============================================================================

Comprehensive comparison and decision guide:

1. Method Family Comparison — all three families head-to-head
2. Memory & Compute Analysis — costs of each approach
3. Model Merging for Continual Learning — TIES-Merging, Task Arithmetic
4. Scalability Analysis — how methods scale with number of tasks
5. Decision Framework — when to use which method

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional


# ============================================================================
# SECTION 1: METHOD FAMILY COMPARISON
# ============================================================================

def method_family_comparison():
    """Compare all three families of continual learning."""
    print("=" * 70)
    print("  SECTION 1: METHOD FAMILY COMPARISON")
    print("=" * 70)
    
    print(f"""
  ═══ Three Families of Continual Learning ═══
  
  ┌─────────────────────────────────────────────────────────────────┐
  │  REGULARIZATION-BASED                                          │
  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐             │
  │  │  EWC    │ │   SI    │ │  MAS    │ │  L2-SP  │             │
  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘             │
  │  Penalty on changing important parameters                      │
  │  + No data storage needed                                      │
  │  − Approximates true constraint                                │
  │  − Importance may shift across many tasks                      │
  ├─────────────────────────────────────────────────────────────────┤
  │  REPLAY-BASED                                                  │
  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │
  │  │ Exp.     │ │ Gen.     │ │ Dark     │ │  GEM     │         │
  │  │ Replay   │ │ Replay   │ │ Replay   │ │          │         │
  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘         │
  │  Re-train on old task examples                                 │
  │  + Simple & effective                                          │
  │  + Works well with small buffers                               │
  │  − Requires data storage (privacy concerns)                    │
  ├─────────────────────────────────────────────────────────────────┤
  │  ARCHITECTURE-BASED                                            │
  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │
  │  │ Prog.    │ │ PackNet  │ │ Task     │ │ MoLoRA   │         │
  │  │ Networks │ │          │ │ Adapters │ │          │         │
  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘         │
  │  Dedicated parameters per task                                 │
  │  + Zero forgetting possible                                    │
  │  + Clean task separation                                       │
  │  − Model grows with tasks                                      │
  │  − Needs task ID at inference                                  │
  └─────────────────────────────────────────────────────────────────┘
""")
    
    # === Quantitative comparison ===
    torch.manual_seed(42)
    
    in_dim, hidden, n_classes = 8, 64, 4
    n_tasks = 5
    
    # Generate harder benchmark (5 tasks with interference)
    tasks = []
    for t in range(n_tasks):
        torch.manual_seed(t * 17)
        W = torch.randn(in_dim, 1) * 2
        offset = (t * 2) % in_dim
        
        x = torch.randn(300, in_dim)
        # Mix features to create cross-task interference
        features = x[:, [offset % in_dim, (offset+1) % in_dim, 
                         (offset+2) % in_dim, (offset+3) % in_dim]]
        scores = features @ W[:4]
        y = torch.zeros(len(x), dtype=torch.long)
        q = torch.quantile(scores.squeeze(), torch.tensor([0.25, 0.5, 0.75]))
        y[scores.squeeze() > q[0]] = 1
        y[scores.squeeze() > q[1]] = 2
        y[scores.squeeze() > q[2]] = 3
        
        x_test = torch.randn(80, in_dim)
        feat_test = x_test[:, [offset % in_dim, (offset+1) % in_dim,
                               (offset+2) % in_dim, (offset+3) % in_dim]]
        s_test = feat_test @ W[:4]
        y_test = torch.zeros(len(x_test), dtype=torch.long)
        y_test[s_test.squeeze() > q[0]] = 1
        y_test[s_test.squeeze() > q[1]] = 2  
        y_test[s_test.squeeze() > q[2]] = 3
        
        tasks.append({'train': (x, y), 'test': (x_test, y_test)})
    
    def evaluate_all(model) -> List[float]:
        model.eval()
        accs = []
        for t in tasks:
            ex, ey = t['test']
            with torch.no_grad():
                preds = model(ex).argmax(1)
                acc = (preds == ey).float().mean().item()
            accs.append(acc)
        model.train()
        return accs
    
    results = {}
    
    # --- Naive ---
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden), nn.ReLU(),
                          nn.Linear(hidden, n_classes))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    naive_matrix = []
    for t in range(n_tasks):
        tx, ty = tasks[t]['train']
        for _ in range(80):
            loss = F.cross_entropy(model(tx), ty)
            opt.zero_grad(); loss.backward(); opt.step()
        naive_matrix.append(evaluate_all(model))
    results['Naive'] = naive_matrix
    del model, opt
    
    # --- EWC ---
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden), nn.ReLU(),
                          nn.Linear(hidden, n_classes))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    fisher_list, params_list = [], []
    ewc_matrix = []
    
    for t in range(n_tasks):
        tx, ty = tasks[t]['train']
        for _ in range(80):
            loss = F.cross_entropy(model(tx), ty)
            # EWC penalty
            ewc_pen = 0.0
            for f, p_star in zip(fisher_list, params_list):
                for (n1, p1), (n2, v) in zip(model.named_parameters(), p_star.items()):
                    if n1 in f:
                        ewc_pen += (f[n1] * (p1 - v) ** 2).sum()
            total = loss + 2500 * ewc_pen if fisher_list else loss
            opt.zero_grad(); total.backward(); opt.step()
        
        # Compute Fisher
        fisher = {}
        model.eval()
        for i in range(min(100, len(tx))):
            model.zero_grad()
            l = F.cross_entropy(model(tx[i:i+1]), ty[i:i+1])
            l.backward()
            for n, p in model.named_parameters():
                if n not in fisher:
                    fisher[n] = torch.zeros_like(p)
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2
        for n in fisher:
            fisher[n] /= 100
        model.train()
        
        fisher_list.append(fisher)
        params_list.append({n: p.data.clone() for n, p in model.named_parameters()})
        ewc_matrix.append(evaluate_all(model))
    
    results['EWC'] = ewc_matrix
    del model, opt
    
    # --- Experience Replay ---
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden), nn.ReLU(),
                          nn.Linear(hidden, n_classes))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    replay_x, replay_y = [], []
    er_matrix = []
    
    for t in range(n_tasks):
        tx, ty = tasks[t]['train']
        for _ in range(80):
            loss = F.cross_entropy(model(tx), ty)
            if replay_x:
                all_rx = torch.cat(replay_x)
                all_ry = torch.cat(replay_y)
                idx = torch.randperm(len(all_rx))[:32]
                loss += F.cross_entropy(model(all_rx[idx]), all_ry[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        
        idx = torch.randperm(len(tx))[:40]
        replay_x.append(tx[idx])
        replay_y.append(ty[idx])
        er_matrix.append(evaluate_all(model))
    
    results['Exp. Replay'] = er_matrix
    del model, opt
    
    # --- Per-Task Heads (Architecture) ---
    torch.manual_seed(42)
    
    class MultiHeadModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU())
            self.heads = nn.ModuleList()
        
        def add_head(self):
            self.heads.append(nn.Linear(hidden, n_classes))
        
        def forward(self, x, task_id=0):
            features = self.shared(x)
            return self.heads[task_id](features)
    
    multi_model = MultiHeadModel()
    arch_matrix = []
    
    for t in range(n_tasks):
        multi_model.add_head()
        # Only train new head + shared features
        opt = torch.optim.Adam(multi_model.parameters(), lr=1e-3)
        tx, ty = tasks[t]['train']
        
        for _ in range(80):
            loss = F.cross_entropy(multi_model(tx, task_id=t), ty)
            # Add replay for shared features
            for j in range(t):
                jx, jy = tasks[j]['train']
                idx = torch.randperm(len(jx))[:20]
                loss += 0.3 * F.cross_entropy(
                    multi_model(jx[idx], task_id=j), jy[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        
        multi_model.eval()
        accs = []
        for j in range(n_tasks):
            if j <= t:
                ex, ey = tasks[j]['test']
                with torch.no_grad():
                    preds = multi_model(ex, task_id=j).argmax(1)
                    acc = (preds == ey).float().mean().item()
            else:
                acc = 0.0
            accs.append(acc)
        multi_model.train()
        arch_matrix.append(accs)
    
    results['Task Heads'] = arch_matrix
    del multi_model
    
    # --- EWC + Replay (Combined) ---
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden), nn.ReLU(),
                          nn.Linear(hidden, n_classes))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    fisher_list2, params_list2 = [], []
    replay_x2, replay_y2 = [], []
    combined_matrix = []
    
    for t in range(n_tasks):
        tx, ty = tasks[t]['train']
        for _ in range(80):
            loss = F.cross_entropy(model(tx), ty)
            # EWC
            ewc_pen = 0.0
            for f, p_star in zip(fisher_list2, params_list2):
                for (n1, p1), (n2, v) in zip(model.named_parameters(), p_star.items()):
                    if n1 in f:
                        ewc_pen += (f[n1] * (p1 - v) ** 2).sum()
            if fisher_list2:
                loss += 1000 * ewc_pen
            # Replay
            if replay_x2:
                all_rx = torch.cat(replay_x2)
                all_ry = torch.cat(replay_y2)
                idx = torch.randperm(len(all_rx))[:32]
                loss += 0.5 * F.cross_entropy(model(all_rx[idx]), all_ry[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        
        # Fisher
        fisher = {}
        model.eval()
        for i in range(min(100, len(tx))):
            model.zero_grad()
            l = F.cross_entropy(model(tx[i:i+1]), ty[i:i+1])
            l.backward()
            for n, p in model.named_parameters():
                if n not in fisher:
                    fisher[n] = torch.zeros_like(p)
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2
        for n in fisher:
            fisher[n] /= 100
        model.train()
        fisher_list2.append(fisher)
        params_list2.append({n: p.data.clone() for n, p in model.named_parameters()})
        
        # Replay buffer
        idx = torch.randperm(len(tx))[:40]
        replay_x2.append(tx[idx])
        replay_y2.append(ty[idx])
        combined_matrix.append(evaluate_all(model))
    
    results['EWC+Replay'] = combined_matrix
    del model, opt
    
    # --- Print results ---
    T = n_tasks
    print(f"\n  ── Results After {T} Sequential Tasks ──\n")
    print(f"  {'Method':>14} │ {'T1':>5} {'T2':>5} {'T3':>5} {'T4':>5} {'T5':>5} │ "
          f"{'AA':>5} │ {'BWT':>6}")
    print(f"  {'─'*14}─┼─{'─'*30}─┼─{'─'*5}─┼─{'─'*6}")
    
    for name, matrix in results.items():
        final = matrix[-1]
        aa = sum(final[j] for j in range(T) if matrix[j][j] > 0) / T
        bwt_vals = [final[j] - matrix[j][j] for j in range(T-1)]
        bwt = sum(bwt_vals) / len(bwt_vals)
        
        print(f"  {name:>14} │ ", end="")
        for j in range(T):
            print(f"{final[j]:>4.0%} ", end="")
        print(f"│ {aa:>4.0%} │ {bwt:>+5.0%}")
    
    return results


# ============================================================================
# SECTION 2: MEMORY & COMPUTE ANALYSIS
# ============================================================================

def memory_compute_analysis():
    """Analyze memory and compute costs of each method."""
    print("\n\n" + "=" * 70)
    print("  SECTION 2: MEMORY & COMPUTE ANALYSIS")
    print("=" * 70)
    
    print(f"""
  ═══ Cost Analysis for 7B Parameter LLM ═══
  
  Assume:
  • Base model: 7B params × 2 bytes (FP16) = 14 GB
  • LoRA adapter (r=16): ~20M params = 40 MB
  • Training: 100K examples per domain
""")
    
    # Model sizes for different LLMs
    model_params = {
        'GPT-2': 124_000_000,
        'LLaMA-7B': 7_000_000_000,
        'LLaMA-13B': 13_000_000_000,
        'LLaMA-70B': 70_000_000_000,
    }
    
    n_tasks_list = [2, 5, 10, 20, 50]
    
    print(f"\n  ── Storage Cost (per task, for LLaMA-7B) ──\n")
    
    methods = {
        'EWC': {
            'per_task_bytes': lambda params, t: params * 4 * 2,  # Fisher + params (FP32)
            'desc': 'Fisher matrix + checkpoint per task',
        },
        'Exp. Replay': {
            'per_task_bytes': lambda params, t: 1000 * 512 * 2,  # 1000 tokens × 512 dim × FP16  
            'desc': '1000 examples stored per task',
        },
        'LoRA Adapter': {
            'per_task_bytes': lambda params, t: int(params * 0.003) * 2,  # ~0.3% params
            'desc': 'Small adapter per task (~0.3% of model)',
        },
        'Full Checkpoint': {
            'per_task_bytes': lambda params, t: params * 2,  # Full FP16 model
            'desc': 'Save entire model per task',
        },
    }
    
    model_name = 'LLaMA-7B'
    params = model_params[model_name]
    
    print(f"  {'Method':>16} │ {'Per Task':>10} │", end="")
    for nt in n_tasks_list:
        print(f" {nt:>3} tasks", end="")
    print()
    print(f"  {'─'*16}─┼─{'─'*10}─┼─{'─'*45}")
    
    for name, method in methods.items():
        per_task = method['per_task_bytes'](params, 1)
        
        def fmt_bytes(b):
            if b >= 1e12:
                return f"{b/1e12:.0f} TB"
            elif b >= 1e9:
                return f"{b/1e9:.1f} GB"
            elif b >= 1e6:
                return f"{b/1e6:.0f} MB"
            else:
                return f"{b/1e3:.0f} KB"
        
        print(f"  {name:>16} │ {fmt_bytes(per_task):>10} │", end="")
        for nt in n_tasks_list:
            total = per_task * nt
            print(f" {fmt_bytes(total):>8}", end="")
        print()
    
    print(f"""
  ═══ Training Compute Overhead ═══
  
  ┌──────────────────┬────────────────────────────────────────┐
  │ Method           │ Extra Compute per Training Step        │
  ├──────────────────┼────────────────────────────────────────┤
  │ EWC              │ +1 forward pass (Fisher) per epoch end │
  │                  │ +N param penalty computations          │
  │                  │ Overhead: ~5-10% per task              │
  ├──────────────────┼────────────────────────────────────────┤
  │ SI               │ +1 param update tracking per step      │
  │                  │ Overhead: ~2-5% per task               │
  ├──────────────────┼────────────────────────────────────────┤
  │ Experience Replay│ +1 forward pass on replay batch        │
  │                  │ Overhead: ~10-25% (scales with buffer) │
  ├──────────────────┼────────────────────────────────────────┤
  │ GEM              │ +K forward passes (per memory set)     │
  │                  │ +QP solver for gradient projection     │
  │                  │ Overhead: ~50-200% (expensive!)        │
  ├──────────────────┼────────────────────────────────────────┤
  │ LoRA per Task    │ 0% overhead on current task            │
  │                  │ But: can't share knowledge across tasks│
  ├──────────────────┼────────────────────────────────────────┤
  │ Knowledge Dist.  │ +1 forward pass with teacher model     │
  │                  │ Overhead: ~100% (runs two models)      │
  └──────────────────┴────────────────────────────────────────┘
""")
    
    # Compute overhead simulation
    print(f"  ── Simulated Training Time (relative to naive) ──\n")
    
    import time
    
    in_dim, hidden, n_classes = 8, 64, 4
    model = nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, n_classes))
    
    x = torch.randn(200, in_dim)
    y = torch.randint(0, n_classes, (200,))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    n_steps = 200
    
    # Naive
    t0 = time.perf_counter()
    for _ in range(n_steps):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    naive_time = time.perf_counter() - t0
    
    # EWC overhead
    fisher = {n: torch.randn_like(p) for n, p in model.named_parameters()}
    params_star = {n: p.data.clone() for n, p in model.named_parameters()}
    
    t0 = time.perf_counter()
    for _ in range(n_steps):
        loss = F.cross_entropy(model(x), y)
        ewc_pen = sum((fisher[n] * (p - params_star[n])**2).sum() 
                      for n, p in model.named_parameters() if n in fisher)
        total = loss + 100 * ewc_pen
        opt.zero_grad(); total.backward(); opt.step()
    ewc_time = time.perf_counter() - t0
    
    # Replay overhead
    rx = torch.randn(50, in_dim)
    ry = torch.randint(0, n_classes, (50,))
    
    t0 = time.perf_counter()
    for _ in range(n_steps):
        loss = F.cross_entropy(model(x), y)
        loss += F.cross_entropy(model(rx), ry)
        opt.zero_grad(); loss.backward(); opt.step()
    replay_time = time.perf_counter() - t0
    
    print(f"  {'Method':>18} │ {'Time':>8} │ {'Overhead':>8}")
    print(f"  {'─'*18}─┼─{'─'*8}─┼─{'─'*8}")
    print(f"  {'Naive':>18} │ {naive_time*1000:>6.1f}ms │ {'baseline':>8}")
    print(f"  {'EWC':>18} │ {ewc_time*1000:>6.1f}ms │ {(ewc_time/naive_time - 1)*100:>+6.0f}%")
    print(f"  {'Replay':>18} │ {replay_time*1000:>6.1f}ms │ {(replay_time/naive_time - 1)*100:>+6.0f}%")
    
    del model, opt


# ============================================================================
# SECTION 3: MODEL MERGING FOR CONTINUAL LEARNING
# ============================================================================

def model_merging():
    """Model merging as an alternative to sequential continual learning."""
    print("\n\n" + "=" * 70)
    print("  SECTION 3: MODEL MERGING FOR CONTINUAL LEARNING")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ Model Merging: An Alternative Paradigm ═══
  
  Instead of sequential training, train INDEPENDENTLY and MERGE:
  
  1. Task Arithmetic (Ilharco et al., 2023):
     θ_merged = θ_base + Σ α_k · (θ_k - θ_base)
  
  2. TIES-Merging (Yadav et al., 2024):
     - Trim: Remove low-magnitude changes
     - Elect: Resolve sign conflicts via majority vote
     - Sum: Merge surviving changes
  
  3. DARE (Yu et al., 2024):
     - Randomly drop delta params (90-99%)
     - Scale surviving ones to compensate
     - Average the sparse deltas
  
  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ Base     │   │ Base     │   │ Base     │
  │ + LoRA A │   │ + LoRA B │   │ + LoRA C │
  └────┬─────┘   └────┬─────┘   └────┬─────┘
       └──────────────┼──────────────┘
                      ▼
              ┌──────────────┐
              │   MERGED     │
              │   MODEL      │
              └──────────────┘
""")
    
    in_dim, hidden, n_classes = 8, 64, 4
    
    # Train separate models from same base
    torch.manual_seed(42)
    base_model = nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, n_classes))
    base_state = {k: v.clone() for k, v in base_model.state_dict().items()}
    
    # Generate tasks
    tasks = []
    for t in range(3):
        torch.manual_seed(t * 42 + 7)
        W = torch.randn(in_dim, 1) * 2
        x = torch.randn(300, in_dim)
        y = ((x @ W).squeeze() > 0).long()
        x_test = torch.randn(80, in_dim)
        y_test = ((x_test @ W).squeeze() > 0).long()
        tasks.append({'train': (x, y), 'test': (x_test, y_test)})
    
    # Train independent models
    task_states = []
    for t in range(3):
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes))
        model.load_state_dict({k: v.clone() for k, v in base_state.items()})
        
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        tx, ty = tasks[t]['train']
        for _ in range(100):
            loss = F.cross_entropy(model(tx), ty)
            opt.zero_grad(); loss.backward(); opt.step()
        
        task_states.append({k: v.clone() for k, v in model.state_dict().items()})
    
    def evaluate(state_dict, task_idx):
        model = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes))
        model.load_state_dict(state_dict)
        model.eval()
        ex, ey = tasks[task_idx]['test']
        with torch.no_grad():
            return (model(ex).argmax(1) == ey).float().mean().item()
    
    # === Task Arithmetic ===
    print(f"\n  ── Task Arithmetic: θ_merged = θ_base + α·Σ(θ_k - θ_base) ──\n")
    
    for alpha in [0.3, 0.5, 0.7, 1.0]:
        merged = {}
        for k in base_state:
            merged[k] = base_state[k].clone()
            for ts in task_states:
                delta = ts[k] - base_state[k]
                merged[k] += (alpha / len(task_states)) * delta
        
        accs = [evaluate(merged, t) for t in range(3)]
        avg = sum(accs) / len(accs)
        print(f"  α={alpha:.1f}: T1={accs[0]:.0%}  T2={accs[1]:.0%}  "
              f"T3={accs[2]:.0%}  │ Avg={avg:.0%}")
    
    # === TIES-Merging ===
    print(f"\n  ── TIES-Merging (Trim + Elect + Sum) ──\n")
    
    def ties_merge(base_state, task_states, density=0.2, alpha=1.0):
        """TIES-Merging implementation."""
        merged = {}
        
        for k in base_state:
            # Compute task vectors (deltas)
            deltas = [ts[k] - base_state[k] for ts in task_states]
            
            # Step 1: TRIM — keep only top-k% by magnitude
            trimmed = []
            for d in deltas:
                threshold = torch.quantile(d.abs().float(), 1.0 - density)
                mask = d.abs() >= threshold
                trimmed.append(d * mask.float())
            
            # Step 2: ELECT — resolve sign conflicts by majority vote
            signs = torch.stack([torch.sign(t) for t in trimmed])
            # Count positive vs negative votes (ignoring zeros)
            pos_votes = (signs > 0).float().sum(dim=0)
            neg_votes = (signs < 0).float().sum(dim=0)
            elected_sign = torch.where(pos_votes >= neg_votes, 
                                       torch.ones_like(pos_votes),
                                       -torch.ones_like(pos_votes))
            
            # Step 3: SUM — average only values matching elected sign
            merged_delta = torch.zeros_like(base_state[k])
            counts = torch.zeros_like(base_state[k])
            
            for t in trimmed:
                agree = (torch.sign(t) == elected_sign) & (t != 0)
                merged_delta += t * agree.float()
                counts += agree.float()
            
            counts = counts.clamp(min=1)
            merged_delta = merged_delta / counts
            
            merged[k] = base_state[k] + alpha * merged_delta
        
        return merged
    
    for density in [0.1, 0.2, 0.5, 1.0]:
        merged = ties_merge(base_state, task_states, density=density)
        accs = [evaluate(merged, t) for t in range(3)]
        avg = sum(accs) / len(accs)
        print(f"  density={density:.1f}: T1={accs[0]:.0%}  T2={accs[1]:.0%}  "
              f"T3={accs[2]:.0%}  │ Avg={avg:.0%}")
    
    # === DARE (Drop And REscale) ===
    print(f"\n  ── DARE (Drop And REscale) ──\n")
    
    def dare_merge(base_state, task_states, drop_rate=0.9, alpha=1.0):
        """DARE: randomly drop most delta params, rescale survivors."""
        merged = {}
        
        for k in base_state:
            deltas = [ts[k] - base_state[k] for ts in task_states]
            
            sparse_deltas = []
            for d in deltas:
                # Random mask: keep (1-drop_rate) fraction
                mask = (torch.rand_like(d.float()) > drop_rate).float()
                # Rescale to compensate for dropped values
                sparse = d * mask / (1.0 - drop_rate + 1e-8)
                sparse_deltas.append(sparse)
            
            # Average sparse deltas
            avg_delta = torch.stack(sparse_deltas).mean(dim=0)
            merged[k] = base_state[k] + alpha * avg_delta
        
        return merged
    
    for drop_rate in [0.5, 0.8, 0.9, 0.95]:
        # Average over multiple random seeds for stability
        all_accs = []
        for seed in range(3):
            torch.manual_seed(seed)
            merged = dare_merge(base_state, task_states, drop_rate=drop_rate)
            accs = [evaluate(merged, t) for t in range(3)]
            all_accs.append(accs)
        
        avg_accs = [sum(a[t] for a in all_accs) / 3 for t in range(3)]
        avg = sum(avg_accs) / 3
        print(f"  drop={drop_rate:.2f}: T1={avg_accs[0]:.0%}  T2={avg_accs[1]:.0%}  "
              f"T3={avg_accs[2]:.0%}  │ Avg={avg:.0%}")
    
    print(f"""
  ═══ Model Merging vs Sequential Training ═══
  
  ┌───────────────┬──────────────────┬──────────────────┐
  │ Aspect        │ Sequential       │ Merging          │
  ├───────────────┼──────────────────┼──────────────────┤
  │ Forgetting    │ High risk        │ Low (independent)│
  │ Data sharing  │ Required         │ Not required     │
  │ Parallelism   │ Must be serial   │ Fully parallel   │
  │ Quality       │ Can be higher    │ May lose nuance  │
  │ Complexity    │ Need CL methods  │ Just merge       │
  │ Scalability   │ Degrades         │ Stays stable     │
  └───────────────┴──────────────────┴──────────────────┘
  
  Model merging is especially good for LoRA adapters:
  • Train LoRA A on medical data, LoRA B on legal data
  • Merge: merged_LoRA = α·LoRA_A + β·LoRA_B
  • Get multi-domain model without any continual learning!
""")


# ============================================================================
# SECTION 4: SCALABILITY ANALYSIS
# ============================================================================

def scalability_analysis():
    """How methods scale with increasing number of tasks."""
    print("\n\n" + "=" * 70)
    print("  SECTION 4: SCALABILITY ANALYSIS")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  ═══ Scalability: What Happens With Many Tasks? ═══
  
  Testing: 2, 4, 6, 8 sequential tasks
  Measuring: Average accuracy after all tasks complete
""")
    
    in_dim, hidden, n_classes = 8, 32, 4
    
    def generate_tasks(n: int):
        tasks = []
        for t in range(n):
            torch.manual_seed(t * 13 + 7)
            W = torch.randn(in_dim, 1) * 2
            x = torch.randn(200, in_dim)
            y = ((x @ W).squeeze() > 0).long() * 2 + (t % 2)
            y = y % n_classes
            x_test = torch.randn(50, in_dim)
            y_test = ((x_test @ W).squeeze() > 0).long() * 2 + (t % 2)
            y_test = y_test % n_classes
            tasks.append({'train': (x, y), 'test': (x_test, y_test)})
        return tasks
    
    def run_naive(tasks):
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for t in tasks:
            tx, ty = t['train']
            for _ in range(60):
                loss = F.cross_entropy(model(tx), ty)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        accs = []
        for t in tasks:
            ex, ey = t['test']
            with torch.no_grad():
                accs.append((model(ex).argmax(1) == ey).float().mean().item())
        return sum(accs) / len(accs)
    
    def run_replay(tasks, buf=30):
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        buffer_x, buffer_y = [], []
        for i, t in enumerate(tasks):
            tx, ty = t['train']
            for _ in range(60):
                loss = F.cross_entropy(model(tx), ty)
                if buffer_x:
                    all_bx = torch.cat(buffer_x)
                    all_by = torch.cat(buffer_y)
                    idx = torch.randperm(len(all_bx))[:32]
                    loss += F.cross_entropy(model(all_bx[idx]), all_by[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            idx = torch.randperm(len(tx))[:buf]
            buffer_x.append(tx[idx])
            buffer_y.append(ty[idx])
        model.eval()
        accs = []
        for t in tasks:
            ex, ey = t['test']
            with torch.no_grad():
                accs.append((model(ex).argmax(1) == ey).float().mean().item())
        return sum(accs) / len(accs)
    
    def run_ewc(tasks, lambda_ewc=1000):
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        fishers, saved_params = [], []
        for t in tasks:
            tx, ty = t['train']
            for _ in range(60):
                loss = F.cross_entropy(model(tx), ty)
                ewc_pen = 0.0
                for f, ps in zip(fishers, saved_params):
                    for (n, p), (_, v) in zip(model.named_parameters(), ps.items()):
                        if n in f:
                            ewc_pen += (f[n] * (p - v)**2).sum()
                if fishers:
                    loss += lambda_ewc * ewc_pen
                opt.zero_grad(); loss.backward(); opt.step()
            # Fisher
            fisher = {}
            model.eval()
            for i in range(min(50, len(tx))):
                model.zero_grad()
                l = F.cross_entropy(model(tx[i:i+1]), ty[i:i+1])
                l.backward()
                for n, p in model.named_parameters():
                    if n not in fisher: fisher[n] = torch.zeros_like(p)
                    if p.grad is not None: fisher[n] += p.grad.data**2
            for n in fisher: fisher[n] /= 50
            model.train()
            fishers.append(fisher)
            saved_params.append({n: p.data.clone() for n, p in model.named_parameters()})
        model.eval()
        accs = []
        for t in tasks:
            ex, ey = t['test']
            with torch.no_grad():
                accs.append((model(ex).argmax(1) == ey).float().mean().item())
        return sum(accs) / len(accs)
    
    task_counts = [2, 4, 6, 8]
    
    print(f"\n  {'#Tasks':>6} │ {'Naive':>6} │ {'EWC':>6} │ {'Replay':>6} │ {'Degradation':>11}")
    print(f"  {'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*11}")
    
    for nt in task_counts:
        tasks = generate_tasks(nt)
        naive_aa = run_naive(tasks)
        ewc_aa = run_ewc(tasks)
        replay_aa = run_replay(tasks)
        
        bars = "▼" * int(max(0, (1 - naive_aa) * 20))
        print(f"  {nt:>6} │ {naive_aa:>5.0%} │ {ewc_aa:>5.0%} │ {replay_aa:>5.0%} │ {bars:>11}")
    
    print(f"""
  ═══ Scalability Observations ═══
  
  • Naive: Degrades rapidly as task count increases
  • EWC: Moderate degradation (accumulated constraints conflict)
  • Replay: Most stable (direct data access helps)
  
  For 50+ tasks, recommended approaches:
  1. Per-task LoRA adapters (no degradation)
  2. Model merging (parallel training)
  3. Replay with fixed-size reservoir (bounded memory)
  
  Regularization methods (EWC, SI) struggle beyond ~10 tasks
  because importance scores conflict across many tasks.
""")


# ============================================================================
# SECTION 5: DECISION FRAMEWORK
# ============================================================================

def decision_framework():
    """When to use which continual learning method."""
    print("\n\n" + "=" * 70)
    print("  SECTION 5: DECISION FRAMEWORK")
    print("=" * 70)
    
    print(f"""
  ═══ Continual Fine-Tuning Decision Tree ═══
  
  START: Do you need to fine-tune on sequential domains?
    │
    ├─ Can you store old domain data?
    │   ├─ YES: Can you afford extra training compute?
    │   │   ├─ YES → Experience Replay + LoRA
    │   │   │       (Best overall performance)
    │   │   └─ NO  → Per-Task LoRA Adapters
    │   │           (Zero forgetting, minimal overhead)
    │   └─ NO (privacy/legal constraints):
    │       ├─ Can you generate synthetic old data?
    │       │   ├─ YES → Generative Replay
    │       │   └─ NO  → EWC or SI regularization
    │       └─ Can domains be trained independently?
    │           └─ YES → Model Merging (TIES / DARE)
    │
    ├─ How many sequential domains?
    │   ├─ 2-5 domains  → Any method works well
    │   ├─ 5-20 domains → Replay or Per-Task Adapters
    │   └─ 20+ domains  → Per-Task Adapters or Model Merging
    │
    └─ Do you have task labels at inference?
        ├─ YES → Per-Task Adapters (route to correct adapter)
        └─ NO  → Replay or Regularization (single model)

  ═══ Method Selection Matrix ═══
  
  ┌──────────────────┬───────┬───────┬────────┬────────┬────────┬────────┐
  │ Consideration    │  EWC  │  SI   │ Replay │  GEM   │ LoRA/  │ Model  │
  │                  │       │       │        │        │ Adapter│ Merge  │
  ├──────────────────┼───────┼───────┼────────┼────────┼────────┼────────┤
  │ Forgetting       │  ◐    │  ◐    │  ●     │  ●     │  ●●    │  ●     │
  │ protection       │       │       │        │        │        │        │
  ├──────────────────┼───────┼───────┼────────┼────────┼────────┼────────┤
  │ Data privacy     │  ●●   │  ●●   │  ○     │  ○     │  ●●    │  ●●    │
  │ friendly         │       │       │        │        │        │        │
  ├──────────────────┼───────┼───────┼────────┼────────┼────────┼────────┤
  │ Compute cost     │  ●    │  ●●   │  ◐     │  ○     │  ●●    │  ●●    │
  │ (low = good)     │       │       │        │        │        │        │
  ├──────────────────┼───────┼───────┼────────┼────────┼────────┼────────┤
  │ Storage cost     │  ◐    │  ●    │  ●     │  ◐     │  ◐     │  ●     │
  │ (low = good)     │       │       │        │        │        │        │
  ├──────────────────┼───────┼───────┼────────┼────────┼────────┼────────┤
  │ Scalability      │  ◐    │  ◐    │  ●     │  ○     │  ●●    │  ●●    │
  │ (many tasks)     │       │       │        │        │        │        │
  ├──────────────────┼───────┼───────┼────────┼────────┼────────┼────────┤
  │ Implementation   │  ●    │  ◐    │  ●●    │  ○     │  ●     │  ●     │
  │ simplicity       │       │       │        │        │        │        │
  ├──────────────────┼───────┼───────┼────────┼────────┼────────┼────────┤
  │ No task labels   │  ●●   │  ●●   │  ●●    │  ●     │  ○     │  ●●    │
  │ at inference     │       │       │        │        │        │        │
  └──────────────────┴───────┴───────┴────────┴────────┴────────┴────────┘
  
  ●● = Excellent   ● = Good   ◐ = Fair   ○ = Poor
  
  ═══ Recommended Approach for LLMs ═══
  
  For MOST practical LLM fine-tuning scenarios:
  
  ┌─────────────────────────────────────────────────────────┐
  │  1. Use QLoRA/LoRA (implicit regularization)            │
  │  2. Include 5-10% replay data from previous domains     │
  │  3. Use progressive learning rate schedule               │
  │  4. Evaluate on all domains after each training stage    │
  │  5. Keep adapter checkpoints for rollback                │
  │                                                          │
  │  This combination handles 90% of real-world scenarios!   │
  └─────────────────────────────────────────────────────────┘
  
  For remaining 10% (extreme data privacy or 50+ domains):
  • Use per-task adapters with adapter routing
  • Or model merging for parallel training
  
  ═══ Common Pitfalls ═══
  
  1. Too high learning rate → immediate forgetting
  2. Too many epochs → overfitting to new domain
  3. No evaluation on old domains → silent degradation
  4. No replay data → relying solely on regularization
  5. No gradient clipping → sudden parameter jumps
  6. Ignoring replay data quality → garbage in, garbage out
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  CONTINUAL FT COMPARISON — METHODS, MERGING, DECISION GUIDE     ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    method_family_comparison()
    memory_compute_analysis()
    model_merging()
    scalability_analysis()
    decision_framework()
    
    print("\n" + "=" * 70)
    print("  COMPARISON MODULE COMPLETE")
    print("=" * 70)
    print("""
    Covered:
    ✓ 5-method head-to-head comparison on sequential tasks
    ✓ Memory and compute cost analysis for LLM scale
    ✓ Model merging: Task Arithmetic, TIES-Merging, DARE
    ✓ Scalability analysis (2-8 tasks)
    ✓ Complete decision framework for method selection
    """)


if __name__ == "__main__":
    main()
