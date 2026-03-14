"""
Continual Fine-Tuning From Scratch — EWC, SI, Experience Replay, Generative Replay
=====================================================================================

Implementations from scratch for deep understanding:

1. Elastic Weight Consolidation (EWC) — full implementation
2. Synaptic Intelligence (SI) — online importance tracking
3. Experience Replay — data mixing strategy
4. Generative Replay — model generates its own replay data
5. Gradient Episodic Memory (GEM) — gradient projection
6. Multi-task sequential training benchmark

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Dict, List, Tuple, Optional


# ============================================================================
# SHARED: Models and Utilities
# ============================================================================

class ContinualModel(nn.Module):
    """Shared model for continual learning experiments."""
    
    def __init__(self, in_dim: int = 8, hidden: int = 64, n_classes: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_classes)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)
    
    def features(self, x):
        """Extract intermediate features."""
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return h


class SequentialTaskBenchmark:
    """Generate a sequence of classification tasks."""
    
    def __init__(self, n_tasks: int = 4, in_dim: int = 8, 
                 n_classes: int = 4, n_train: int = 400, n_test: int = 100):
        self.n_tasks = n_tasks
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.tasks = []
        
        torch.manual_seed(0)
        for t in range(n_tasks):
            # Each task uses different feature dimensions
            # This creates INTERFERENCE between tasks
            offset = t * 2
            dims = [(offset + i) % in_dim for i in range(4)]
            
            W = torch.randn(4, 1) * 2
            
            def make_data(n, dims=dims, W=W):
                x = torch.randn(n, in_dim)
                features = x[:, dims]
                scores = features @ W
                y = torch.zeros(n, dtype=torch.long)
                thresholds = torch.tensor([-0.5, 0.0, 0.5])
                for i, thresh in enumerate(thresholds):
                    y[scores.squeeze() > thresh] = i + 1
                return x, y % n_classes
            
            train_x, train_y = make_data(n_train)
            test_x, test_y = make_data(n_test)
            
            self.tasks.append({
                'train': (train_x, train_y),
                'test': (test_x, test_y),
                'name': f'Task {t+1}'
            })
    
    def evaluate_all(self, model) -> List[float]:
        """Evaluate model on all tasks."""
        model.eval()
        accs = []
        for task in self.tasks:
            x, y = task['test']
            with torch.no_grad():
                preds = model(x).argmax(dim=1)
                acc = (preds == y).float().mean().item()
            accs.append(acc)
        model.train()
        return accs


# ============================================================================
# SECTION 1: ELASTIC WEIGHT CONSOLIDATION (EWC)
# ============================================================================

class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).
    
    L = L_task + (λ/2) · Σ_i F_i · (θ_i - θ*_i)²
    
    Protects parameters important for previous tasks by penalizing
    changes proportional to their Fisher Information.
    """
    
    def __init__(self, model: nn.Module, lambda_ewc: float = 1000.0):
        self.model = model
        self.lambda_ewc = lambda_ewc
        
        # Store per-task: (Fisher, optimal params)
        self.task_fisher: List[Dict[str, torch.Tensor]] = []
        self.task_params: List[Dict[str, torch.Tensor]] = []
    
    def compute_fisher(self, data_x: torch.Tensor, data_y: torch.Tensor,
                       n_samples: int = 200):
        """Compute diagonal Fisher Information Matrix."""
        fisher = {n: torch.zeros_like(p) 
                  for n, p in self.model.named_parameters()}
        
        self.model.eval()
        n = min(n_samples, len(data_x))
        
        for i in range(n):
            self.model.zero_grad()
            logits = self.model(data_x[i:i+1])
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Use true labels (empirical Fisher)
            loss = F.nll_loss(log_probs, data_y[i:i+1])
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        
        for name in fisher:
            fisher[name] /= n
        
        self.model.train()
        return fisher
    
    def register_task(self, data_x: torch.Tensor, data_y: torch.Tensor):
        """Register a completed task (compute Fisher, save params)."""
        fisher = self.compute_fisher(data_x, data_y)
        self.task_fisher.append(fisher)
        
        params = {n: p.data.clone() for n, p in self.model.named_parameters()}
        self.task_params.append(params)
    
    def penalty(self) -> torch.Tensor:
        """Compute EWC penalty across all registered tasks."""
        if not self.task_fisher:
            return torch.tensor(0.0)
        
        loss = 0.0
        for fisher, params in zip(self.task_fisher, self.task_params):
            for name, param in self.model.named_parameters():
                loss += (fisher[name] * (param - params[name]) ** 2).sum()
        
        return (self.lambda_ewc / 2) * loss


def ewc_training():
    """Full EWC training demonstration."""
    print("=" * 70)
    print("  SECTION 1: ELASTIC WEIGHT CONSOLIDATION (EWC)")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  EWC Algorithm:
  1. Train on Task k
  2. Compute Fisher Information on Task k data
  3. Save optimal parameters θ*_k
  4. When training on Task k+1:
     L = L_{{k+1}} + (λ/2) · Σ_{{j=1..k}} Σ_i F^j_i · (θ_i - θ*^j_i)²
""")
    
    benchmark = SequentialTaskBenchmark(n_tasks=4, in_dim=8)
    model = ContinualModel(in_dim=8, hidden=64, n_classes=4)
    ewc = EWC(model, lambda_ewc=5000.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Track accuracy matrix
    acc_matrix = []
    
    print(f"\n  ── EWC Sequential Training (λ=5000) ──\n")
    
    for task_idx, task in enumerate(benchmark.tasks):
        train_x, train_y = task['train']
        
        # Train on current task
        for epoch in range(80):
            logits = model(train_x)
            task_loss = F.cross_entropy(logits, train_y)
            ewc_loss = ewc.penalty()
            total_loss = task_loss + ewc_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # Register completed task
        ewc.register_task(train_x, train_y)
        
        # Evaluate on all tasks
        accs = benchmark.evaluate_all(model)
        acc_matrix.append(accs)
        
        print(f"  After {task['name']:>8}: ", end="")
        for j, acc in enumerate(accs):
            marker = "✓" if j <= task_idx and acc > 0.5 else " "
            print(f"T{j+1}={acc:.1%}{marker}  ", end="")
        print()
    
    # Compute metrics
    T = len(acc_matrix)
    bwt = sum(acc_matrix[T-1][j] - acc_matrix[j][j] for j in range(T-1)) / (T-1)
    aa = sum(acc_matrix[T-1]) / T
    
    print(f"\n  Average Accuracy: {aa:.1%}")
    print(f"  Backward Transfer: {bwt:+.1%}")
    
    del model, ewc
    return acc_matrix


# ============================================================================
# SECTION 2: SYNAPTIC INTELLIGENCE (SI)
# ============================================================================

class SynapticIntelligence:
    """
    Synaptic Intelligence (Zenke et al., 2017).
    
    Computes importance ONLINE during training by tracking
    how much each parameter contributed to loss decrease.
    """
    
    def __init__(self, model: nn.Module, lambda_si: float = 1.0, 
                 epsilon: float = 0.1):
        self.model = model
        self.lambda_si = lambda_si
        self.epsilon = epsilon  # Damping term
        
        # Running importance sum
        self.omega = {n: torch.zeros_like(p) 
                      for n, p in model.named_parameters()}
        
        # Per-task tracking
        self.prev_params = {n: p.data.clone() 
                            for n, p in model.named_parameters()}
        self.running_product = {n: torch.zeros_like(p) 
                                for n, p in model.named_parameters()}
        
        # Task checkpoints
        self.task_params: List[Dict[str, torch.Tensor]] = []
    
    def update_running_sum(self):
        """Call AFTER each optimizer step to track parameter changes."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Contribution = -gradient * parameter_change
                delta = param.data - self.prev_params[name]
                self.running_product[name] += -param.grad.data * delta
                self.prev_params[name] = param.data.clone()
    
    def register_task(self):
        """Call AFTER finishing a task to consolidate importance."""
        for name, param in self.model.named_parameters():
            # Importance = contribution / (total change²)
            if self.task_params:
                delta = param.data - self.task_params[-1][name]
            else:
                delta = param.data - self.prev_params[name]
            
            importance = self.running_product[name] / (delta ** 2 + self.epsilon)
            self.omega[name] += F.relu(importance)  # Only positive contributions
        
        # Save task params
        self.task_params.append(
            {n: p.data.clone() for n, p in self.model.named_parameters()})
        
        # Reset running sum
        self.running_product = {n: torch.zeros_like(p) 
                                for n, p in self.model.named_parameters()}
    
    def penalty(self) -> torch.Tensor:
        """Compute SI penalty using accumulated importance."""
        if not self.task_params:
            return torch.tensor(0.0)
        
        loss = 0.0
        last_params = self.task_params[-1]
        for name, param in self.model.named_parameters():
            loss += (self.omega[name] * (param - last_params[name]) ** 2).sum()
        
        return self.lambda_si * loss


def si_training():
    """Full SI training demonstration."""
    print("\n\n" + "=" * 70)
    print("  SECTION 2: SYNAPTIC INTELLIGENCE (SI)")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  SI tracks importance ONLINE during training:
  1. After each gradient step, record -grad × Δθ
  2. After task completes, normalize by Δθ²
  3. Accumulate importance across tasks
  4. Penalize changes to important parameters
""")
    
    benchmark = SequentialTaskBenchmark(n_tasks=4, in_dim=8)
    model = ContinualModel(in_dim=8, hidden=64, n_classes=4)
    si = SynapticIntelligence(model, lambda_si=100.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    acc_matrix = []
    
    print(f"\n  ── SI Sequential Training (λ=100) ──\n")
    
    for task_idx, task in enumerate(benchmark.tasks):
        train_x, train_y = task['train']
        
        for epoch in range(80):
            logits = model(train_x)
            task_loss = F.cross_entropy(logits, train_y)
            si_loss = si.penalty()
            total_loss = task_loss + si_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # SI: track parameter changes after each step
            si.update_running_sum()
        
        # Register completed task
        si.register_task()
        
        accs = benchmark.evaluate_all(model)
        acc_matrix.append(accs)
        
        print(f"  After {task['name']:>8}: ", end="")
        for j, acc in enumerate(accs):
            marker = "✓" if j <= task_idx and acc > 0.5 else " "
            print(f"T{j+1}={acc:.1%}{marker}  ", end="")
        print()
    
    T = len(acc_matrix)
    bwt = sum(acc_matrix[T-1][j] - acc_matrix[j][j] for j in range(T-1)) / (T-1)
    aa = sum(acc_matrix[T-1]) / T
    
    print(f"\n  Average Accuracy: {aa:.1%}")
    print(f"  Backward Transfer: {bwt:+.1%}")
    
    del model, si
    return acc_matrix


# ============================================================================
# SECTION 3: EXPERIENCE REPLAY
# ============================================================================

class ExperienceReplay:
    """
    Experience Replay: Store and replay examples from previous tasks.
    
    Strategies:
    - Random: Store random subset from each task
    - Herding: Store examples closest to class centroids
    - Reservoir: Maintain fixed-size buffer with reservoir sampling
    """
    
    def __init__(self, buffer_size_per_task: int = 50):
        self.buffer_size = buffer_size_per_task
        self.buffer_x: List[torch.Tensor] = []
        self.buffer_y: List[torch.Tensor] = []
    
    def add_task_data(self, x: torch.Tensor, y: torch.Tensor,
                      strategy: str = "random"):
        """Add data from completed task to replay buffer."""
        n = min(self.buffer_size, len(x))
        
        if strategy == "random":
            indices = torch.randperm(len(x))[:n]
            self.buffer_x.append(x[indices])
            self.buffer_y.append(y[indices])
        
        elif strategy == "balanced":
            # Equal samples per class
            classes = y.unique()
            per_class = max(1, n // len(classes))
            selected_x, selected_y = [], []
            
            for c in classes:
                mask = y == c
                c_indices = torch.where(mask)[0]
                sel = c_indices[torch.randperm(len(c_indices))[:per_class]]
                selected_x.append(x[sel])
                selected_y.append(y[sel])
            
            self.buffer_x.append(torch.cat(selected_x))
            self.buffer_y.append(torch.cat(selected_y))
    
    def get_replay_batch(self, batch_size: int = 32) -> Optional[Tuple]:
        """Sample a batch from the replay buffer."""
        if not self.buffer_x:
            return None
        
        all_x = torch.cat(self.buffer_x)
        all_y = torch.cat(self.buffer_y)
        
        n = min(batch_size, len(all_x))
        indices = torch.randperm(len(all_x))[:n]
        
        return all_x[indices], all_y[indices]
    
    @property
    def total_stored(self) -> int:
        return sum(len(x) for x in self.buffer_x)


def experience_replay_training():
    """Full experience replay training."""
    print("\n\n" + "=" * 70)
    print("  SECTION 3: EXPERIENCE REPLAY")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  Experience Replay:
  1. Train on Task k
  2. Store a subset of Task k data in buffer
  3. When training on Task k+1:
     - Mix new task data with replay buffer
     - Combine losses: L = L_new + α·L_replay
  4. Repeat
  
  Simple but very effective! The key insight is that even a SMALL
  amount of old data prevents catastrophic forgetting.
""")
    
    benchmark = SequentialTaskBenchmark(n_tasks=4, in_dim=8)
    model = ContinualModel(in_dim=8, hidden=64, n_classes=4)
    replay = ExperienceReplay(buffer_size_per_task=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    replay_weight = 1.0  # Weight for replay loss
    
    acc_matrix = []
    
    print(f"\n  ── Experience Replay Training (buffer=50/task, α={replay_weight}) ──\n")
    
    for task_idx, task in enumerate(benchmark.tasks):
        train_x, train_y = task['train']
        
        for epoch in range(80):
            # Current task loss
            logits = model(train_x)
            task_loss = F.cross_entropy(logits, train_y)
            
            # Replay loss
            replay_batch = replay.get_replay_batch(batch_size=32)
            if replay_batch is not None:
                replay_x, replay_y = replay_batch
                replay_logits = model(replay_x)
                replay_loss = F.cross_entropy(replay_logits, replay_y)
                total_loss = task_loss + replay_weight * replay_loss
            else:
                total_loss = task_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # Add task data to buffer
        replay.add_task_data(train_x, train_y, strategy="balanced")
        
        accs = benchmark.evaluate_all(model)
        acc_matrix.append(accs)
        
        print(f"  After {task['name']:>8} (buffer: {replay.total_stored:>4} samples): ", 
              end="")
        for j, acc in enumerate(accs):
            marker = "✓" if j <= task_idx and acc > 0.5 else " "
            print(f"T{j+1}={acc:.1%}{marker}  ", end="")
        print()
    
    T = len(acc_matrix)
    bwt = sum(acc_matrix[T-1][j] - acc_matrix[j][j] for j in range(T-1)) / (T-1)
    aa = sum(acc_matrix[T-1]) / T
    
    print(f"\n  Average Accuracy: {aa:.1%}")
    print(f"  Backward Transfer: {bwt:+.1%}")
    print(f"  Total buffer size: {replay.total_stored} samples")
    
    del model, replay
    return acc_matrix


# ============================================================================
# SECTION 4: GENERATIVE REPLAY
# ============================================================================

def generative_replay_training():
    """Generative replay — model generates its own old-task data."""
    print("\n\n" + "=" * 70)
    print("  SECTION 4: GENERATIVE REPLAY")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  Generative Replay (Shin et al., 2017):
  Instead of storing real data, use a GENERATOR to create
  pseudo-examples from previous tasks.
  
  For LLMs, generative replay is natural:
  • The LLM IS the generator
  • Generate prompts + completions from previous tasks
  • Mix generated data with new task data during training
  
  Advantages: No data storage needed!
  Disadvantages: Generated data may not perfectly represent old tasks.
  
  Simplified version (Dark Experience Replay):
  • Store model LOGITS instead of raw data
  • Train to match old logits (knowledge distillation)
""")
    
    # Dark Experience Replay: store logits from teacher
    class DarkExperienceReplay:
        """Store input-logit pairs from previous model states."""
        
        def __init__(self, buffer_size_per_task: int = 50, temperature: float = 2.0):
            self.buffer_size = buffer_size_per_task
            self.temperature = temperature
            self.buffer_x: List[torch.Tensor] = []
            self.buffer_logits: List[torch.Tensor] = []
        
        def add_task_data(self, model: nn.Module, x: torch.Tensor):
            """Store input-logit pairs from current model."""
            n = min(self.buffer_size, len(x))
            indices = torch.randperm(len(x))[:n]
            
            model.eval()
            with torch.no_grad():
                logits = model(x[indices])
            model.train()
            
            self.buffer_x.append(x[indices])
            self.buffer_logits.append(logits)
        
        def distillation_loss(self, model: nn.Module, 
                               batch_size: int = 32) -> torch.Tensor:
            """KD loss: match current model's output to stored logits."""
            if not self.buffer_x:
                return torch.tensor(0.0)
            
            all_x = torch.cat(self.buffer_x)
            all_logits = torch.cat(self.buffer_logits)
            
            n = min(batch_size, len(all_x))
            indices = torch.randperm(len(all_x))[:n]
            
            current_logits = model(all_x[indices])
            
            # KL divergence with temperature
            T = self.temperature
            p = F.log_softmax(current_logits / T, dim=-1)
            q = F.softmax(all_logits[indices] / T, dim=-1)
            
            return F.kl_div(p, q, reduction='batchmean') * (T ** 2)
    
    benchmark = SequentialTaskBenchmark(n_tasks=4, in_dim=8)
    model = ContinualModel(in_dim=8, hidden=64, n_classes=4)
    dark_replay = DarkExperienceReplay(buffer_size_per_task=50, temperature=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    alpha = 0.5  # Distillation weight
    
    acc_matrix = []
    
    print(f"\n  ── Dark Experience Replay (logit distillation, α={alpha}) ──\n")
    
    for task_idx, task in enumerate(benchmark.tasks):
        train_x, train_y = task['train']
        
        for epoch in range(80):
            logits = model(train_x)
            task_loss = F.cross_entropy(logits, train_y)
            distill_loss = dark_replay.distillation_loss(model)
            total_loss = task_loss + alpha * distill_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # Store current model's logits
        dark_replay.add_task_data(model, train_x)
        
        accs = benchmark.evaluate_all(model)
        acc_matrix.append(accs)
        
        print(f"  After {task['name']:>8}: ", end="")
        for j, acc in enumerate(accs):
            marker = "✓" if j <= task_idx and acc > 0.5 else " "
            print(f"T{j+1}={acc:.1%}{marker}  ", end="")
        print()
    
    T = len(acc_matrix)
    bwt = sum(acc_matrix[T-1][j] - acc_matrix[j][j] for j in range(T-1)) / (T-1)
    aa = sum(acc_matrix[T-1]) / T
    
    print(f"\n  Average Accuracy: {aa:.1%}")
    print(f"  Backward Transfer: {bwt:+.1%}")
    
    del model, dark_replay
    
    print(f"""
  ═══ Generative Replay for LLMs ═══
  
  For real LLMs, generative replay looks like:
  
  1. Before fine-tuning on new domain:
     - Generate N examples using current model
     - Save: (prompt_i, completion_i) pairs
  
  2. During fine-tuning:
     - Mix real new-domain data with generated old-domain data
     - Ratio: 80% new data, 20% generated replay data
  
  3. Alternative: Knowledge Distillation approach
     - Save model logits on sample of old data
     - Add KD loss: KL(new_model || old_logits)
     - No generation needed, but needs stored data
  
  Code pattern:
    # Before new domain FT
    replay_data = generate_from_model(model, old_prompts, n=1000)
    
    # During FT
    for batch in new_domain_data:
        loss_new = compute_loss(model, batch)
        replay_batch = sample(replay_data)
        loss_replay = compute_loss(model, replay_batch)
        loss = loss_new + 0.2 * loss_replay
""")
    
    return acc_matrix


# ============================================================================
# SECTION 5: GRADIENT EPISODIC MEMORY (GEM)
# ============================================================================

def gem_training():
    """Gradient Episodic Memory — project gradients to avoid interference."""
    print("\n\n" + "=" * 70)
    print("  SECTION 5: GRADIENT EPISODIC MEMORY (GEM)")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print(f"""
  GEM (Lopez-Paz & Ranzato, 2017):
  
  Key idea: Before applying gradient for new task, CHECK if it
  would increase loss on old tasks. If so, PROJECT the gradient
  to the closest vector that doesn't harm old tasks.
  
  Algorithm:
  1. Store a small memory M_k for each past task k
  2. Compute gradient g for new task
  3. Compute gradients g_k for each memory M_k
  4. If g · g_k < 0 for any k (conflict!):
     Project g to closest vector where g · g_k ≥ 0 for all k
  5. Apply projected gradient
  
  This GUARANTEES no increase in old task losses (in theory).
""")
    
    class GEM:
        def __init__(self, model: nn.Module, memory_size: int = 50):
            self.model = model
            self.memory_size = memory_size
            self.memory_x: List[torch.Tensor] = []
            self.memory_y: List[torch.Tensor] = []
        
        def add_memory(self, x: torch.Tensor, y: torch.Tensor):
            n = min(self.memory_size, len(x))
            idx = torch.randperm(len(x))[:n]
            self.memory_x.append(x[idx])
            self.memory_y.append(y[idx])
        
        def _flatten_grads(self) -> torch.Tensor:
            grads = []
            for p in self.model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.data.flatten())
                else:
                    grads.append(torch.zeros(p.numel()))
            return torch.cat(grads)
        
        def _set_grads(self, flat_grad: torch.Tensor):
            offset = 0
            for p in self.model.parameters():
                n = p.numel()
                if p.grad is not None:
                    p.grad.data = flat_grad[offset:offset+n].view(p.shape)
                offset += n
        
        def project_gradient(self):
            """Project current gradient to not conflict with memories."""
            if not self.memory_x:
                return
            
            # Get current gradient (for new task)
            g = self._flatten_grads().clone()
            
            # Get gradients for each memory
            memory_grads = []
            for mx, my in zip(self.memory_x, self.memory_y):
                self.model.zero_grad()
                logits = self.model(mx)
                loss = F.cross_entropy(logits, my)
                loss.backward()
                memory_grads.append(self._flatten_grads().clone())
            
            # Check for conflicts and project
            projected = g.clone()
            for mg in memory_grads:
                dot = projected.dot(mg)
                if dot < 0:
                    # Project: remove component along mg
                    projected -= (dot / (mg.dot(mg) + 1e-8)) * mg
            
            self._set_grads(projected)
    
    benchmark = SequentialTaskBenchmark(n_tasks=4, in_dim=8)
    model = ContinualModel(in_dim=8, hidden=64, n_classes=4)
    gem = GEM(model, memory_size=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    acc_matrix = []
    
    print(f"\n  ── GEM Sequential Training (memory=50/task) ──\n")
    
    for task_idx, task in enumerate(benchmark.tasks):
        train_x, train_y = task['train']
        
        for epoch in range(80):
            # Forward pass for current task
            model.zero_grad()
            logits = model(train_x)
            task_loss = F.cross_entropy(logits, train_y)
            task_loss.backward()
            
            # GEM: project gradient if it conflicts with old tasks
            gem.project_gradient()
            
            optimizer.step()
        
        gem.add_memory(train_x, train_y)
        
        accs = benchmark.evaluate_all(model)
        acc_matrix.append(accs)
        
        print(f"  After {task['name']:>8}: ", end="")
        for j, acc in enumerate(accs):
            marker = "✓" if j <= task_idx and acc > 0.5 else " "
            print(f"T{j+1}={acc:.1%}{marker}  ", end="")
        print()
    
    T = len(acc_matrix)
    bwt = sum(acc_matrix[T-1][j] - acc_matrix[j][j] for j in range(T-1)) / (T-1)
    aa = sum(acc_matrix[T-1]) / T
    
    print(f"\n  Average Accuracy: {aa:.1%}")
    print(f"  Backward Transfer: {bwt:+.1%}")
    
    del model, gem
    return acc_matrix


# ============================================================================
# SECTION 6: COMPREHENSIVE COMPARISON
# ============================================================================

def comprehensive_comparison():
    """Compare all methods side by side."""
    print("\n\n" + "=" * 70)
    print("  SECTION 6: COMPREHENSIVE COMPARISON")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Run naive baseline
    benchmark = SequentialTaskBenchmark(n_tasks=4, in_dim=8)
    
    # Naive (no protection)
    model = ContinualModel(in_dim=8, hidden=64, n_classes=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    naive_matrix = []
    
    for task_idx, task in enumerate(benchmark.tasks):
        tx, ty = task['train']
        for _ in range(80):
            loss = F.cross_entropy(model(tx), ty)
            opt.zero_grad(); loss.backward(); opt.step()
        naive_matrix.append(benchmark.evaluate_all(model))
    
    del model, opt
    
    # Joint training (upper bound)
    torch.manual_seed(42)
    model = ContinualModel(in_dim=8, hidden=64, n_classes=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    all_x = torch.cat([t['train'][0] for t in benchmark.tasks])
    all_y = torch.cat([t['train'][1] for t in benchmark.tasks])
    
    for _ in range(200):
        idx = torch.randperm(len(all_x))[:128]
        loss = F.cross_entropy(model(all_x[idx]), all_y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    
    joint_accs = benchmark.evaluate_all(model)
    del model, opt
    
    # Collect all results (rerun methods to get consistent comparison)
    print(f"\n  Running all methods on same benchmark...\n")
    
    methods = {
        "Naive (no prot.)": naive_matrix,
    }
    
    # EWC
    torch.manual_seed(42)
    model = ContinualModel(in_dim=8, hidden=64, n_classes=4)
    ewc = EWC(model, lambda_ewc=5000.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ewc_matrix = []
    for task_idx, task in enumerate(benchmark.tasks):
        tx, ty = task['train']
        for _ in range(80):
            loss = F.cross_entropy(model(tx), ty) + ewc.penalty()
            opt.zero_grad(); loss.backward(); opt.step()
        ewc.register_task(tx, ty)
        ewc_matrix.append(benchmark.evaluate_all(model))
    methods["EWC (λ=5000)"] = ewc_matrix
    del model, ewc, opt
    
    # SI
    torch.manual_seed(42)
    model = ContinualModel(in_dim=8, hidden=64, n_classes=4)
    si = SynapticIntelligence(model, lambda_si=100.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    si_matrix = []
    for task_idx, task in enumerate(benchmark.tasks):
        tx, ty = task['train']
        for _ in range(80):
            loss = F.cross_entropy(model(tx), ty) + si.penalty()
            opt.zero_grad(); loss.backward(); opt.step()
            si.update_running_sum()
        si.register_task()
        si_matrix.append(benchmark.evaluate_all(model))
    methods["SI (λ=100)"] = si_matrix
    del model, si, opt
    
    # Experience Replay
    torch.manual_seed(42)
    model = ContinualModel(in_dim=8, hidden=64, n_classes=4)
    replay = ExperienceReplay(buffer_size_per_task=50)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    er_matrix = []
    for task_idx, task in enumerate(benchmark.tasks):
        tx, ty = task['train']
        for _ in range(80):
            loss = F.cross_entropy(model(tx), ty)
            rb = replay.get_replay_batch(32)
            if rb:
                rx, ry = rb
                loss += F.cross_entropy(model(rx), ry)
            opt.zero_grad(); loss.backward(); opt.step()
        replay.add_task_data(tx, ty, "balanced")
        er_matrix.append(benchmark.evaluate_all(model))
    methods["Exp. Replay (50)"] = er_matrix
    del model, replay, opt
    
    # Summary table
    print(f"  ── Final Results After All 4 Tasks ──\n")
    print(f"  {'Method':>18} │ {'T1':>6} {'T2':>6} {'T3':>6} {'T4':>6} │ "
          f"{'AA':>6} │ {'BWT':>7}")
    print(f"  {'─'*18}─┼─{'─'*26}─┼─{'─'*6}─┼─{'─'*7}")
    
    # Joint training baseline
    jt_aa = sum(joint_accs) / 4
    print(f"  {'Joint (upper bnd)':>18} │ ", end="")
    for acc in joint_accs:
        print(f"{acc:>5.1%} ", end="")
    print(f"│ {jt_aa:>5.1%} │ {'N/A':>7}")
    
    for name, matrix in methods.items():
        T = 4
        accs = matrix[-1]
        aa = sum(accs) / T
        bwt = sum(accs[j] - matrix[j][j] for j in range(T-1)) / (T-1)
        
        print(f"  {name:>18} │ ", end="")
        for acc in accs:
            print(f"{acc:>5.1%} ", end="")
        print(f"│ {aa:>5.1%} │ {bwt:>+6.1%}")
    
    print(f"""
  ═══ Observations ═══
  
  • Joint training is the UPPER BOUND (sees all data simultaneously)
  • Naive training suffers severe forgetting on early tasks
  • EWC and SI provide moderate protection via regularization
  • Experience Replay is simple and effective with small buffer
  • All methods trade off stability vs plasticity
  
  For LLMs, Experience Replay + LoRA is often the most practical:
  • LoRA limits parameter changes (implicit regularization)
  • Small replay buffer of old prompts prevents forgetting
  • No complex importance computation needed
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  CONTINUAL FT FROM SCRATCH — EWC, SI, REPLAY, GEM               ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    ewc_training()
    si_training()
    experience_replay_training()
    generative_replay_training()
    gem_training()
    comprehensive_comparison()
    
    print("\n" + "=" * 70)
    print("  FROM-SCRATCH MODULE COMPLETE")
    print("=" * 70)
    print("""
    Implemented:
    ✓ EWC — Elastic Weight Consolidation with Fisher
    ✓ SI — Synaptic Intelligence with online tracking
    ✓ Experience Replay — data buffer mixing
    ✓ Dark Experience Replay — logit distillation
    ✓ GEM — Gradient Episodic Memory with projection
    ✓ Comprehensive comparison on 4-task benchmark
    """)


if __name__ == "__main__":
    main()
