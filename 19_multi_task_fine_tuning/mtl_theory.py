"""
Multi-Task Fine-Tuning - Theoretical Foundations
================================================

Deep dive into the theory behind training one model on multiple tasks
simultaneously: why it works, when it fails, and mathematical frameworks
for understanding task relationships and gradient dynamics.

Sections:
    1. Multi-Task Learning Fundamentals
    2. Task Relatedness and Transfer Theory
    3. Negative Transfer and Task Conflicts
    4. Gradient Dynamics in Multi-Task Optimization
    5. Task Balancing Theory and Loss Weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


# =============================================================================
# SECTION 1: Multi-Task Learning Fundamentals
# =============================================================================

class MultiTaskLearningTheory:
    """
    Theoretical foundations of Multi-Task Learning (MTL).
    
    Core Idea:
        Instead of training K separate models for K tasks:
            θ*_k = argmin_θ L_k(θ; D_k)  for k = 1,...,K
        
        We train ONE shared model:
            θ* = argmin_θ Σ_k w_k · L_k(θ; D_k)
        
        where w_k are task weights controlling relative importance.
    
    Why MTL Works — Multiple Theoretical Perspectives:
    
    1. Implicit Data Augmentation:
       - Each task provides additional training signal
       - Task A's data acts as a regularizer for Task B
       - More diverse gradients → better generalization
    
    2. Inductive Bias / Representation Bias:
       - MTL biases the model toward representations useful for multiple tasks
       - Shared features are more likely to generalize
       - Features that help ONLY one task are suppressed
    
    3. Eavesdropping:
       - Some features are easy to learn for Task A but hard for Task B
       - Through shared representations, Task B "eavesdrops" on Task A
       - Example: POS tagging helps dependency parsing
    
    4. Attention Focusing:
       - Multiple tasks help the model focus on relevant features
       - Reduces the risk of overfitting to task-specific noise
    """
    
    def __init__(self):
        self.concepts = {
            "hard_parameter_sharing": {
                "description": "Shared hidden layers + task-specific output layers",
                "diagram": """
                    Input
                      │
                    ┌─▼─┐
                    │   │  Shared layers (most parameters)
                    │   │  - These learn task-agnostic representations
                    └─┬─┘
                      │
                    ┌─┴─┐
                    │Split│
                    └┬─┬─┘
                     │ │
                   ┌─▼┐▼─┐
                   │H₁│H₂│  Task-specific heads (few parameters)
                   └──┘──┘  - These adapt shared repr. to each task
                """,
                "parameter_sharing": "~95% shared, ~5% task-specific",
                "overfitting_risk": "Low (regularization from multiple tasks)",
                "negative_transfer_risk": "High (forced to share all features)"
            },
            "soft_parameter_sharing": {
                "description": "Separate models regularized to be similar",
                "diagram": """
                    Input       Input
                      │           │
                    ┌─▼─┐      ┌─▼─┐
                    │M₁ │←reg→ │M₂ │  Separate but regularized models
                    └─┬─┘      └─┬─┘
                      │          │
                    ┌─▼─┐      ┌─▼─┐
                    │H₁ │      │H₂ │  Task-specific outputs
                    └───┘      └───┘
                """,
                "parameter_sharing": "0% shared + L2/trace regularization",
                "overfitting_risk": "Higher (more parameters per task)",
                "negative_transfer_risk": "Lower (each task can specialize)"
            }
        }
    
    def demonstrate_mtl_objective(self):
        """Show the standard multi-task objective function."""
        print("=" * 70)
        print("MULTI-TASK LEARNING OBJECTIVE")
        print("=" * 70)
        
        print("""
        Standard Single-Task Objective:
        ───────────────────────────────
            θ*_k = argmin_θ  L_k(θ; D_k) + λ‖θ‖²
        
        Multi-Task Objective:
        ─────────────────────
            θ* = argmin_θ  Σ_{k=1}^{K} w_k · L_k(θ_shared, θ_k; D_k)
        
        Where:
            θ_shared = Parameters shared across all tasks
            θ_k      = Task-specific parameters for task k
            w_k      = Weight for task k (controls importance)
            L_k      = Loss function for task k
            D_k      = Dataset for task k
        
        Decomposition of Parameters:
        ──────────────────────────────
            θ = {θ_shared, θ_1, θ_2, ..., θ_K}
        
        In Hard Parameter Sharing:
            θ_shared = Encoder parameters (large, ~95% of model)
            θ_k     = Head parameters (small, ~5% of model)
        
        The shared parameters act as an inductive bias:
        ─────────────────────────────────────────────
            By requiring θ_shared to minimize ALL task losses,
            we constrain the hypothesis space to representations
            that are useful across tasks → better generalization.
        """)
    
    def demonstrate_mtl_generalization_bound(self):
        """
        Show the theoretical generalization advantage of MTL.
        
        Baxter (2000) showed that MTL has tighter generalization bounds:
        
        Single-task bound:
            ε_single ≤ O(√(H/n))     where H = hypothesis complexity, n = samples
        
        Multi-task bound:
            ε_mtl ≤ O(√(H/nK))       where K = number of tasks
        
        The bound scales with 1/√(K) — more tasks = better generalization.
        This assumes tasks share a common representation.
        """
        print("\n" + "=" * 70)
        print("GENERALIZATION BOUNDS: SINGLE-TASK vs MULTI-TASK")
        print("=" * 70)
        
        H = 1000  # Hypothesis complexity
        n = 500   # Samples per task
        
        for K in [1, 2, 5, 10, 20]:
            single_bound = math.sqrt(H / n)
            mtl_bound = math.sqrt(H / (n * K))
            improvement = (1 - mtl_bound / single_bound) * 100
            
            print(f"\n  K={K:2d} tasks:")
            print(f"    Single-task bound: {single_bound:.4f}")
            print(f"    Multi-task bound:  {mtl_bound:.4f}")
            print(f"    Improvement:       {improvement:.1f}%")
        
        print("\n  Key insight: More related tasks → tighter bounds → better generalization")
        print("  Caveat: This assumes tasks share a common representation!")


# =============================================================================
# SECTION 2: Task Relatedness and Transfer Theory
# =============================================================================

class TaskRelatednessTheory:
    """
    Theory for understanding when tasks help or hurt each other.
    
    Task Similarity Measures:
    
    1. Feature Overlap:
       - Tasks sharing features benefit from joint training
       - Measure: Cosine similarity of learned representations
    
    2. Gradient Alignment:
       - Tasks with aligned gradients reinforce each other
       - Tasks with opposing gradients conflict
       - Measure: Cosine similarity of task gradients
    
    3. Task Taxonomy:
       - NLP hierarchy: syntax → semantics → pragmatics
       - Lower-level tasks help higher-level tasks
       - POS → NER → RE → QA → Summarization
    
    4. Information-Theoretic View:
       - Tasks share mutual information I(T_A; T_B)
       - More shared information → more positive transfer
       - Transfer ∝ I(T_A; T_B) / H(T_B)
    """
    
    @staticmethod
    def compute_task_gradient_similarity(
        model: nn.Module,
        task_a_batch: Dict[str, torch.Tensor],
        task_b_batch: Dict[str, torch.Tensor],
        task_a_head: nn.Module,
        task_b_head: nn.Module
    ) -> float:
        """
        Measure task relatedness via gradient cosine similarity.
        
        High positive similarity → tasks reinforce each other
        Near zero → tasks are independent  
        Negative similarity → tasks conflict (negative transfer risk)
        
        Gradient similarity is THE most reliable predictor of
        whether MTL will help or hurt performance.
        """
        # Get gradients from task A
        model.zero_grad()
        shared_output = model(task_a_batch["input"])
        loss_a = F.cross_entropy(task_a_head(shared_output), task_a_batch["labels"])
        loss_a.backward()
        grad_a = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        
        # Get gradients from task B
        model.zero_grad()
        shared_output = model(task_b_batch["input"])
        loss_b = F.cross_entropy(task_b_head(shared_output), task_b_batch["labels"])
        loss_b.backward()
        grad_b = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        
        # Cosine similarity
        similarity = F.cosine_similarity(grad_a.unsqueeze(0), grad_b.unsqueeze(0)).item()
        
        return similarity
    
    @staticmethod
    def analyze_task_relationships():
        """Demonstrate common NLP task relationships."""
        print("=" * 70)
        print("TASK RELATIONSHIP ANALYSIS")
        print("=" * 70)
        
        # Known task relationships from literature
        relationships = {
            ("Sentiment Analysis", "Emotion Detection"): {
                "similarity": 0.85,
                "transfer": "Strong Positive",
                "reason": "Both analyze affective content, shared sentiment features"
            },
            ("NER", "POS Tagging"): {
                "similarity": 0.72,
                "transfer": "Positive",
                "reason": "POS informs entity boundaries; syntactic features shared"
            },
            ("Summarization", "Translation"): {
                "similarity": 0.45,
                "transfer": "Weak Positive",
                "reason": "Both are seq2seq, share attention mechanisms"
            },
            ("Sentiment Analysis", "NER"): {
                "similarity": 0.25,
                "transfer": "Neutral/Weak",
                "reason": "Different goals; some shared contextual features"
            },
            ("Question Answering", "NLI"): {
                "similarity": 0.68,
                "transfer": "Positive",
                "reason": "Both require reasoning; NLI teaches entailment for QA"
            },
            ("Code Generation", "Sentiment Analysis"): {
                "similarity": -0.1,
                "transfer": "Negative",
                "reason": "Conflicting representations; code ≠ natural language patterns"
            }
        }
        
        for (task_a, task_b), info in relationships.items():
            sim = info["similarity"]
            bar = "█" * int(abs(sim) * 20) + "░" * (20 - int(abs(sim) * 20))
            sign = "+" if sim > 0 else "-"
            print(f"\n  {task_a} ↔ {task_b}")
            print(f"    Similarity: {sign}{bar} {sim:.2f}")
            print(f"    Transfer:   {info['transfer']}")
            print(f"    Reason:     {info['reason']}")

    @staticmethod
    def task_affinity_grouping_theory():
        """
        Theory behind grouping tasks for optimal MTL.
        
        The task grouping problem:
            Given K tasks, find the optimal partition into groups
            such that tasks within each group benefit from MTL.
        
        This is combinatorial (Bell number of partitions).
        
        Heuristics:
        1. Greedy grouping by gradient similarity
        2. Hierarchical clustering on task embeddings
        3. TAG (Task Affinity Grouping) — train pairwise, measure improvement
        """
        print("\n" + "=" * 70)
        print("TASK AFFINITY GROUPING")
        print("=" * 70)
        
        print("""
        Problem: Given K tasks, find optimal groups for MTL.
        
        Approach 1 — Pairwise Affinity:
        ─────────────────────────────────
            For each pair (i, j):
                Train MTL model on tasks i and j
                Measure: a(i→j) = perf_MTL(j) - perf_single(j)
                This gives asymmetric affinity matrix
            
            Then cluster using spectral/hierarchical methods.
        
        Approach 2 — Gradient-Based Grouping:
        ──────────────────────────────────────
            Compute gradient similarity matrix:
                S(i,j) = cos(∇L_i, ∇L_j)
            
            Group tasks with S(i,j) > threshold
            Tasks with S(i,j) < 0 should NOT be grouped
        
        Approach 3 — TAG (Fifty et al., 2021):
        ─────────────────────────────────────
            Use inter-task affinity score:
                a(i→j) = Z_j^{i,j} / Z_j^{j}
            
            Where Z is validation performance.
            Group tasks using higher-order affinities.
        
        Example grouping for NLP tasks:
        ─────────────────────────────────
            Group A: [Sentiment, Emotion, Sarcasm, Stance]
                → All analyze subjective content
            
            Group B: [NER, POS, Chunking, Dependency Parsing]
                → All analyze syntactic/structural features
            
            Group C: [QA, NLI, Paraphrase Detection]
                → All require semantic reasoning
            
            Training 3 MTL models (one per group) typically outperforms
            training 1 MTL model on all tasks together.
        """)


# =============================================================================
# SECTION 3: Negative Transfer and Task Conflicts
# =============================================================================

class NegativeTransferTheory:
    """
    Theory behind when and why Multi-Task Learning HURTS performance.
    
    Negative Transfer occurs when jointly training task A with task B
    results in WORSE performance on task A than training A alone.
    
    Sources of Negative Transfer:
    
    1. Representational Conflict:
       - Tasks require mutually exclusive features
       - Shared encoder forced into suboptimal compromise
    
    2. Gradient Interference:
       - Task gradients point in opposite directions
       - Updates for one task undo progress on another
    
    3. Capacity Bottleneck:
       - Shared model too small to learn all tasks well
       - Tasks compete for limited representational capacity
    
    4. Optimization Conflict:
       - Different tasks converge at different rates
       - Fast-learning task dominates early training
       - Slow-learning task never catches up
    
    5. Label Space Conflict:
       - Contradictory training signals for similar inputs
       - E.g., "bank" labeled LOCATION vs ORGANIZATION
    """
    
    @staticmethod
    def demonstrate_gradient_conflict():
        """
        Simulate gradient conflict between two tasks.
        
        When gradients conflict, the model must choose between:
        1. Average gradient (compromise — may not help either task)
        2. Projected gradient (PCGrad — remove conflicting component)
        3. Weighted gradient (favor the more important task)
        """
        print("=" * 70)
        print("GRADIENT CONFLICT DEMONSTRATION")
        print("=" * 70)
        
        torch.manual_seed(42)
        
        # Simulate gradients for different task relationships
        scenarios = {
            "Aligned Tasks (Positive Transfer)": {
                "grad_a": torch.tensor([0.5, 0.3, 0.8, 0.2]),
                "grad_b": torch.tensor([0.4, 0.35, 0.7, 0.15]),
                "expected": "Both benefit"
            },
            "Orthogonal Tasks (No Transfer)": {
                "grad_a": torch.tensor([0.5, 0.0, 0.3, 0.0]),
                "grad_b": torch.tensor([0.0, 0.4, 0.0, 0.6]),
                "expected": "Independent, no interference"
            },
            "Conflicting Tasks (Negative Transfer)": {
                "grad_a": torch.tensor([0.5, -0.3, 0.8, -0.2]),
                "grad_b": torch.tensor([-0.4, 0.35, -0.7, 0.15]),
                "expected": "Mutual interference"
            },
            "Partially Conflicting (Mixed)": {
                "grad_a": torch.tensor([0.5, 0.3, -0.4, 0.2]),
                "grad_b": torch.tensor([0.4, 0.35, 0.6, -0.15]),
                "expected": "Some dimensions conflict"
            }
        }
        
        for name, data in scenarios.items():
            g_a = data["grad_a"]
            g_b = data["grad_b"]
            
            # Naive average
            naive_avg = (g_a + g_b) / 2
            
            # Cosine similarity
            cos_sim = F.cosine_similarity(g_a.unsqueeze(0), g_b.unsqueeze(0)).item()
            
            # Inner product (how much does average gradient help each task)
            help_a = torch.dot(naive_avg, g_a).item()  # >0 means helpful for A
            help_b = torch.dot(naive_avg, g_b).item()  # >0 means helpful for B
            
            print(f"\n  {name}:")
            print(f"    grad_A:       {g_a.tolist()}")
            print(f"    grad_B:       {g_b.tolist()}")
            print(f"    Cosine sim:   {cos_sim:.3f}")
            print(f"    Naive avg:    {naive_avg.tolist()}")
            print(f"    Helps task A: {'Yes' if help_a > 0 else 'NO — NEGATIVE TRANSFER'} ({help_a:.3f})")
            print(f"    Helps task B: {'Yes' if help_b > 0 else 'NO — NEGATIVE TRANSFER'} ({help_b:.3f})")
            print(f"    Expected:     {data['expected']}")
    
    @staticmethod
    def demonstrate_capacity_bottleneck():
        """
        Show how model capacity affects negative transfer.
        When the model is too small, tasks compete for capacity.
        """
        print("\n" + "=" * 70)
        print("CAPACITY BOTTLENECK EFFECT")
        print("=" * 70)
        
        print("""
        Scenario: 2 tasks, each requiring ~C features to solve well.
        
        Model with capacity 2C (sufficient):
        ──────────────────────────────────
            ┌─────────────────────────────┐
            │  Features for A  │ Features for B  │  ← Plenty of room
            └─────────────────┴─────────────────┘
            Result: Both tasks learned well, positive transfer possible
        
        Model with capacity C (bottleneck):
        ────────────────────────────────
            ┌───────────────────────────┐
            │ A features │ B features   │  ← Capacity competition!
            └────────────┴──────────────┘
            Result: Tasks compete, negative transfer likely
        
        Empirical findings:
        ─────────────────────
            • Larger models → less negative transfer
            • LoRA-based MTL reduces negative transfer
              (each task gets separate adapter capacity)
            • Task-specific heads reduce head competition
        
        Capacity vs Transfer relationship:
        ──────────────────────────────────
            Capacity    Transfer Risk    Solution
            ─────────   ─────────────    ────────
            Very Low    High (-)         Don't use MTL
            Low         Moderate (-)     Reduce task count
            Medium      Low (±)          Use gradient surgery
            High        Very Low (+)     Standard MTL works
            Very High   Minimal (+)      Full positive transfer
        """)


# =============================================================================
# SECTION 4: Gradient Dynamics in Multi-Task Optimization
# =============================================================================

class GradientDynamicsTheory:
    """
    Advanced theory of gradient-based multi-task optimization.
    
    The key challenge: how to combine gradients from K tasks into
    a single update direction that helps ALL tasks.
    
    Methods:
    1. Linear scalarization (weighted sum)
    2. PCGrad (Projecting Conflicting Gradients)
    3. GradNorm (Gradient Normalization)
    4. CAGrad (Conflict-Averse Gradient)
    5. Nash-MTL (Nash bargaining for MTL)
    """
    
    @staticmethod
    def pcgrad_algorithm(
        task_gradients: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        PCGrad: Projecting Conflicting Gradients (Yu et al., 2020).
        
        Key idea: When two task gradients conflict (negative cosine similarity),
        project one onto the normal plane of the other to remove the
        conflicting component.
        
        Algorithm:
            For each task i:
                For each other task j:
                    If cos(g_i, g_j) < 0:  # Conflicting
                        g_i = g_i - (g_i · g_j / ‖g_j‖²) · g_j
                        # Remove the component of g_i that conflicts with g_j
            
            Return average of modified gradients
        
        This is one of the most important algorithms in MTL optimization.
        """
        num_tasks = len(task_gradients)
        # Clone to avoid modifying originals
        modified_grads = [g.clone() for g in task_gradients]
        
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i == j:
                    continue
                    
                g_i = modified_grads[i]
                g_j = task_gradients[j]  # Use original (not modified) for j
                
                # Check for conflict
                dot_product = torch.dot(g_i, g_j)
                
                if dot_product < 0:  # Conflicting gradients
                    # Project g_i onto the normal plane of g_j
                    # This removes the component of g_i that opposes g_j
                    projection = (dot_product / (torch.dot(g_j, g_j) + 1e-8)) * g_j
                    modified_grads[i] = g_i - projection
        
        # Average modified gradients
        combined = torch.stack(modified_grads).mean(dim=0)
        return combined
    
    @staticmethod
    def demonstrate_pcgrad():
        """Demonstrate PCGrad on a concrete example."""
        print("=" * 70)
        print("PCGrad: PROJECTING CONFLICTING GRADIENTS")
        print("=" * 70)
        
        torch.manual_seed(42)
        
        # Create conflicting gradients
        g1 = torch.tensor([1.0, 0.5, -0.3])
        g2 = torch.tensor([-0.5, 0.8, 0.4])
        
        cos_sim = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()
        print(f"\n  Task 1 gradient: {g1.tolist()}")
        print(f"  Task 2 gradient: {g2.tolist()}")
        print(f"  Cosine similarity: {cos_sim:.3f}")
        
        # Naive average
        naive = (g1 + g2) / 2
        print(f"\n  Naive average: {naive.tolist()}")
        print(f"    Dot with g1: {torch.dot(naive, g1).item():.3f}")
        print(f"    Dot with g2: {torch.dot(naive, g2).item():.3f}")
        
        # PCGrad
        dynamics = GradientDynamicsTheory()
        pcgrad_result = dynamics.pcgrad_algorithm([g1, g2])
        print(f"\n  PCGrad result: {pcgrad_result.tolist()}")
        print(f"    Dot with g1: {torch.dot(pcgrad_result, g1).item():.3f}")
        print(f"    Dot with g2: {torch.dot(pcgrad_result, g2).item():.3f}")
        
        print("""
  Interpretation:
    PCGrad removes conflicting components so the combined gradient
    does not actively HURT any task (positive dot product with both).
    The naive average may have negative dot product with one task,
    causing it to move AWAY from that task's optimum.
        """)
    
    @staticmethod
    def gradnorm_theory():
        """
        GradNorm: Gradient Normalization (Chen et al., 2018).
        
        Key insight: Balance task learning rates by normalizing gradient norms.
        
        If task A has 10× larger gradients than task B,
        task A dominates training and task B is neglected.
        
        GradNorm dynamically adjusts task weights so all tasks
        train at similar rates relative to their difficulty.
        
        Algorithm:
            1. Compute task-specific gradient norms: G_i = ‖∇_W L_i‖
            2. Compute average gradient norm: G_avg = mean(G_i)
            3. Compute relative inverse training rate: r_i = L_i(t) / L_i(0)
            4. Compute target norm: target_i = G_avg × [r_i / mean(r)]^α
            5. Update weights w_i to drive G_i toward target_i
        """
        print("\n" + "=" * 70)
        print("GradNorm: GRADIENT NORMALIZATION THEORY")
        print("=" * 70)
        
        print("""
        Problem: Gradient Magnitude Imbalance
        ──────────────────────────────────────
            Task A loss: 0.01 → small gradients
            Task B loss: 5.00 → large gradients
            
            Naive sum: ∇L = ∇L_A + ∇L_B ≈ ∇L_B  (A is essentially ignored!)
        
        GradNorm Solution:
        ──────────────────
            Step 1: Measure gradient norms per task
                G_A = ‖∇_W L_A‖ = 0.05
                G_B = ‖∇_W L_B‖ = 2.50
                G_avg = 1.275
            
            Step 2: Measure training rates (how fast each task is learning)
                r_A = L_A(t) / L_A(0) = 0.01 / 1.0 = 0.01  (fast learner)
                r_B = L_B(t) / L_B(0) = 5.0 / 6.0 = 0.83   (slow learner)
            
            Step 3: Slow tasks should get MORE gradient focus
                target_A = G_avg × (r_A / r_avg)^α  (small — task almost done)
                target_B = G_avg × (r_B / r_avg)^α  (large — task needs help)
            
            Step 4: Adjust task weights to match targets
                w_A ← w_A such that G_A → target_A
                w_B ← w_B such that G_B → target_B
        
        α controls sensitivity:
            α = 0: Equal gradients (ignore training rate)
            α = 1: Proportional to training rate (recommended)
            α → ∞: Focus entirely on slowest task
        """)


# =============================================================================
# SECTION 5: Task Balancing Theory and Loss Weighting
# =============================================================================

class TaskBalancingTheory:
    """
    Theory and mathematics of task balancing strategies.
    
    The fundamental question: How to set task weights w_k in
        L_total = Σ_k w_k · L_k
    
    Approaches:
    1. Static Weights: Fixed throughout training
    2. Dynamic Weights: Adapt during training
    3. Uncertainty-Based: Weight by task uncertainty
    4. Loss-Based: Weight by current loss values
    5. Data-Based: Weight by dataset characteristics
    """
    
    @staticmethod
    def uncertainty_weighting_theory():
        """
        Uncertainty Weighting (Kendall et al., 2018).
        
        Key idea: Use homoscedastic (task-dependent) uncertainty
        to automatically learn task weights.
        
        For each task k, introduce a learnable noise parameter σ_k.
        The multi-task loss becomes:
        
            L_total = Σ_k (1/(2σ_k²)) · L_k + log(σ_k)
        
        The 1/(2σ_k²) term acts as the task weight:
            - High uncertainty σ_k → lower weight (less confident, penalize less)
            - Low uncertainty σ_k → higher weight (more confident, penalize more)
        
        The log(σ_k) term prevents all σ → ∞ (which would set all weights to 0).
        
        This elegantly learns relative task importance without manual tuning.
        """
        print("=" * 70)
        print("UNCERTAINTY-BASED TASK WEIGHTING")
        print("=" * 70)
        
        print("""
        Derivation from Maximum Likelihood:
        ─────────────────────────────────────
        
        For regression task with Gaussian likelihood:
            p(y|x, θ) = N(f_θ(x), σ²)
            
            -log p(y|x) = (1/2σ²)·‖y - f_θ(x)‖² + (1/2)·log(σ²) + const
            
            L = (1/2σ²) · MSE + log(σ)
        
        For classification task with softmax:
            L = (1/σ²) · CE + log(σ)
        
        Multi-task with K tasks:
            L_total = Σ_k [(1/σ_k²) · L_k + log(σ_k)]
        
        Properties:
        ───────────
            ✓ Automatic weight learning via σ_k
            ✓ No manual tuning of task weights
            ✓ Tasks with high aleatoric noise get lower weight
            ✓ Principled (derived from probabilistic model)
            
            ✗ Adds K learnable parameters (negligible)
            ✗ Weight depends on noise, not task importance
            ✗ May not handle task imbalance well
        """)
        
        # Demonstrate how σ affects weights
        print("  Effect of σ on task weight:")
        print("  " + "-" * 40)
        for sigma in [0.1, 0.5, 1.0, 2.0, 5.0]:
            weight = 1.0 / (2 * sigma ** 2)
            regularizer = math.log(sigma)
            print(f"    σ = {sigma:.1f}  →  weight = {weight:.4f}, reg = {regularizer:.4f}")
    
    @staticmethod
    def demonstrate_sampling_strategies():
        """
        Data sampling strategies for multi-task training.
        
        When tasks have different dataset sizes, how do we sample batches?
        """
        print("\n" + "=" * 70)
        print("TASK SAMPLING STRATEGIES")
        print("=" * 70)
        
        # Example: 4 tasks with different dataset sizes
        tasks = {
            "Sentiment": 100000,
            "NER": 20000,
            "QA": 50000,
            "Summarization": 5000
        }
        total = sum(tasks.values())
        
        print(f"\n  Dataset sizes:")
        for name, size in tasks.items():
            bar = "█" * int(size / 5000)
            print(f"    {name:20s}: {size:>7d} samples  {bar}")
        
        # Strategy 1: Proportional
        print("\n  Strategy 1: Proportional Sampling")
        print("  " + "-" * 50)
        for name, size in tasks.items():
            prob = size / total
            print(f"    {name:20s}: p = {prob:.3f}  (= {size}/{total})")
        print("  → Large tasks dominate; small tasks barely seen")
        
        # Strategy 2: Equal
        K = len(tasks)
        print(f"\n  Strategy 2: Equal Sampling")
        print("  " + "-" * 50)
        for name in tasks:
            prob = 1.0 / K
            print(f"    {name:20s}: p = {prob:.3f}  (= 1/{K})")
        print("  → Fair but large tasks waste data; small tasks overfit")
        
        # Strategy 3: Temperature-scaled (T=2)
        T = 2.0
        print(f"\n  Strategy 3: Temperature Sampling (T={T})")
        print("  " + "-" * 50)
        sizes = list(tasks.values())
        scaled = [s ** (1.0 / T) for s in sizes]
        scaled_sum = sum(scaled)
        for (name, size), sc in zip(tasks.items(), scaled):
            prob = sc / scaled_sum
            print(f"    {name:20s}: p = {prob:.3f}  (= {size}^(1/{T}) / Z)")
        print("  → Balanced compromise; T→1 = proportional, T→∞ = equal")
        
        # Strategy 4: Square root
        print(f"\n  Strategy 4: Square Root Sampling")
        print("  " + "-" * 50)
        sqrt_sizes = [math.sqrt(s) for s in sizes]
        sqrt_sum = sum(sqrt_sizes)
        for (name, size), sq in zip(tasks.items(), sqrt_sizes):
            prob = sq / sqrt_sum
            print(f"    {name:20s}: p = {prob:.3f}  (= √{size} / Z)")
        print("  → Good default; similar to T=2")
        
        print("""
  Recommendations:
    ─────────────
    • Start with square root or T=2 sampling
    • If small tasks underfit, try T=3 or equal
    • If large tasks underfit, try T=1.5 or proportional
    • Monitor per-task validation metrics to adjust
        """)
    
    @staticmethod
    def multi_task_pareto_optimality():
        """
        Theory of Pareto optimality in multi-task learning.
        
        A solution θ* is Pareto optimal if no other θ can improve
        one task without hurting another.
        
        The set of all Pareto optimal solutions forms the Pareto front.
        Different task weight configurations explore different points
        on the Pareto front.
        """
        print("\n" + "=" * 70)
        print("PARETO OPTIMALITY IN MTL")
        print("=" * 70)
        
        print("""
        Definition:
        ──────────
            θ* is Pareto optimal if ∄ θ such that:
                L_k(θ) ≤ L_k(θ*) for all tasks k
                AND L_j(θ) < L_j(θ*) for at least one task j
        
        Visualization (2 tasks):
        ──────────────────────
            Task B Loss
            ▲
            │   x          
            │    x         Single-task B optimum
            │     x x
            │       x x ← Pareto Front
            │         x x
            │           x
            │            x  Single-task A optimum
            └────────────────► Task A Loss
        
        Each point on the Pareto front corresponds to a different
        weighting of tasks. There is no single "best" solution —
        the choice depends on which tasks matter more.
        
        Key insight:
        ────────────
            Linear scalarization (L = w₁L₁ + w₂L₂) can only find
            points on the CONVEX part of the Pareto front.
            
            For non-convex Pareto fronts, we need more sophisticated
            methods like MGDA (Multiple Gradient Descent Algorithm)
            or CAGrad (Conflict-Averse Gradient descent).
        
        Practical implication:
        ──────────────────────
            If you care about Task A 2× more than Task B,
            setting w_A = 2 × w_B does NOT guarantee the
            right point on the Pareto front!
            
            The relationship between weights and Pareto front
            position is highly non-linear.
        """)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 70)
    print("MULTI-TASK FINE-TUNING — THEORETICAL FOUNDATIONS")
    print("=" * 70)
    
    # Section 1: Fundamentals
    print("\n\n📖 SECTION 1: Multi-Task Learning Fundamentals")
    mtl_theory = MultiTaskLearningTheory()
    mtl_theory.demonstrate_mtl_objective()
    mtl_theory.demonstrate_mtl_generalization_bound()
    
    # Section 2: Task Relatedness
    print("\n\n📖 SECTION 2: Task Relatedness and Transfer Theory")
    TaskRelatednessTheory.analyze_task_relationships()
    TaskRelatednessTheory.task_affinity_grouping_theory()
    
    # Section 3: Negative Transfer
    print("\n\n📖 SECTION 3: Negative Transfer and Task Conflicts")
    NegativeTransferTheory.demonstrate_gradient_conflict()
    NegativeTransferTheory.demonstrate_capacity_bottleneck()
    
    # Section 4: Gradient Dynamics
    print("\n\n📖 SECTION 4: Gradient Dynamics in Multi-Task Optimization")
    GradientDynamicsTheory.demonstrate_pcgrad()
    GradientDynamicsTheory.gradnorm_theory()
    
    # Section 5: Task Balancing
    print("\n\n📖 SECTION 5: Task Balancing Theory and Loss Weighting")
    TaskBalancingTheory.uncertainty_weighting_theory()
    TaskBalancingTheory.demonstrate_sampling_strategies()
    TaskBalancingTheory.multi_task_pareto_optimality()
    
    print("\n" + "=" * 70)
    print("THEORY SUMMARY")
    print("=" * 70)
    print("""
    Key Takeaways:
    
    1. MTL trains one model on K tasks: L = Σ w_k · L_k
       → Shared representations → better generalization
    
    2. Task relatedness determines MTL success:
       → Measure via gradient cosine similarity
       → Related tasks help; unrelated tasks hurt
    
    3. Negative transfer is the main risk:
       → Gradient conflict, capacity bottleneck, optimization mismatch
       → Solutions: PCGrad, task grouping, larger models
    
    4. Gradient surgery (PCGrad) removes conflicting components:
       → Projects gradients to avoid negative transfer
       → Preserves beneficial gradient components
    
    5. Task balancing is critical:
       → Uncertainty weighting: automatic via homoscedastic noise
       → GradNorm: balance gradient norms across tasks
       → Temperature sampling: control dataset representation
    
    6. Pareto optimality: no single "best" MTL solution
       → Different weight configs explore Pareto front
       → Choose based on which tasks matter most
    """)


if __name__ == "__main__":
    main()
