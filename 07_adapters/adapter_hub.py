"""
Adapter Hub & Composition
===========================

Adapter sharing, composition, and hub ecosystem:

1. AdapterHub Ecosystem
   - Sharing adapters on AdapterHub
   - Loading community adapters
   - Adapter cards and metadata

2. Adapter Composition Patterns
   - Sequential stacking
   - Parallel composition
   - AdapterFusion recap
   - Attention-based routing

3. Adapter Arithmetic
   - Adding/subtracting adapter weights
   - Task vector operations
   - Negation and combination

4. Building a Multi-Adapter Serving System
   - Dynamic adapter loading
   - Efficient adapter management
   - Production patterns

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import copy
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


# ============================================================================
# SECTION 1: ADAPTERHUB ECOSYSTEM
# ============================================================================

class AdapterHubGuide:
    """
    Guide to the AdapterHub ecosystem for sharing and reusing adapters.
    
    AdapterHub (adapterhub.ml) is a centralized repository for sharing
    pre-trained adapters, similar to what the HuggingFace Hub does for
    full model weights but specifically designed for adapters.
    
    Key benefits:
    - Share adapter weights independently of the base model
    - Community-contributed adapters for many tasks
    - Standardized adapter format and metadata
    """
    
    @staticmethod
    def overview():
        """Overview of the AdapterHub ecosystem."""
        print("=" * 65)
        print("  ADAPTERHUB ECOSYSTEM")
        print("=" * 65)
        
        info = """
  AdapterHub Architecture:
  ═══════════════════════════════════════════════════════════════
  
  ┌─────────────────────────────────────────────────────────────┐
  │                    AdapterHub.ml                             │
  │                                                             │
  │   ┌──────────┐  ┌──────────┐  ┌──────────┐                │
  │   │ Sentiment│  │   NLI    │  │    QA    │  ...adapters   │
  │   │ Adapter  │  │ Adapter  │  │ Adapter  │                │
  │   │  ~2MB    │  │  ~2MB    │  │  ~3MB    │                │
  │   └────┬─────┘  └────┬─────┘  └────┬─────┘                │
  │        │              │              │                      │
  │        └──────────────┼──────────────┘                      │
  │                       │                                     │
  │              ┌────────▼────────┐                            │
  │              │  Base Model     │                            │
  │              │  (e.g., BERT)   │                            │
  │              │  ~440MB         │                            │
  │              └─────────────────┘                            │
  │                                                             │
  │  One base model + many small adapters = many tasks!        │
  └─────────────────────────────────────────────────────────────┘
  
  How It Works:
  ─────────────────────────────────────────────────────────────
  1. Train an adapter on your task
  2. Upload adapter weights + metadata (adapter card)
  3. Others download your adapter (~2MB instead of ~440MB)
  4. Plug into matching base model → instant task capability
  
  Adapter Card Metadata:
  ─────────────────────────────────────────────────────────────
  - Base model (e.g., "bert-base-uncased")
  - Task type (e.g., "text-classification")
  - Dataset used for training
  - Performance metrics
  - Adapter architecture (Pfeiffer/Houlsby/LoRA/etc.)
  - Hyperparameters used
  - License
"""
        print(info)
    
    @staticmethod
    def show_hub_api():
        """Show AdapterHub API usage patterns."""
        print("\n" + "=" * 65)
        print("  ADAPTERHUB API PATTERNS")
        print("=" * 65)
        
        code = '''
# ═══════════════════════════════════════════════════════════
# Using adapter-transformers library with AdapterHub
# pip install adapter-transformers
# ═══════════════════════════════════════════════════════════

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load base model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ── Load Adapter from Hub ────────────────────────────────
# Load a pre-trained sentiment adapter from AdapterHub
model.load_adapter("sentiment/sst-2@ukp", source="ah")
#                   └── task/dataset@author ──┘  └── source: adapterhub

# Activate the adapter
model.set_active_adapters("sentiment/sst-2@ukp")

# Now the model performs sentiment analysis!
inputs = tokenizer("This movie was amazing!", return_tensors="pt")
outputs = model(**inputs)
prediction = outputs.logits.argmax(dim=-1)

# ── Load Multiple Adapters ───────────────────────────────
# Load NLI adapter
model.load_adapter("nli/multinli@ukp", source="ah")

# Load QA adapter  
model.load_adapter("qa/squad2@ukp", source="ah")

# Switch between tasks by changing active adapter
model.set_active_adapters("sentiment/sst-2@ukp")   # Sentiment mode
model.set_active_adapters("nli/multinli@ukp")       # NLI mode
model.set_active_adapters("qa/squad2@ukp")          # QA mode

# ── Upload Your Adapter ─────────────────────────────────
# After training your adapter
model.save_adapter("./my_adapter", "my_task")

# Push to AdapterHub (requires account)
model.push_adapter_to_hub(
    "my_adapter",
    "my_task",
    adapterhub_tag="my_username/my_adapter",
    datasets_tag="my_dataset",
)

# ── From HuggingFace Hub (alternative) ──────────────────
# Adapters can also be shared on HuggingFace Hub
from peft import PeftModel, PeftConfig

# Load adapter config
config = PeftConfig.from_pretrained("username/my-adapter-repo")

# Load base model 
from transformers import AutoModelForCausalLM
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

# Apply adapter
model = PeftModel.from_pretrained(base_model, "username/my-adapter-repo")
'''
        print(code)
        return code


# ============================================================================
# SECTION 2: ADAPTER COMPOSITION PATTERNS
# ============================================================================

class AdapterComposition:
    """
    Patterns for composing multiple adapters together.
    
    Composition approaches:
    1. Sequential (Stack): Output of adapter A feeds into adapter B
    2. Parallel: Both adapters process the same input, outputs merged
    3. Fusion: Learned attention over adapter outputs
    4. Split: Different adapters for different layers
    """
    
    @staticmethod
    def sequential_stacking():
        """Stack adapters sequentially."""
        print("\n" + "=" * 65)
        print("  SEQUENTIAL ADAPTER STACKING")
        print("=" * 65)
        
        diagram = """
  Sequential: h → [Adapter_A] → [Adapter_B] → output
  
  Use case: Transfer learning chain
    e.g., English → Domain → Task
  
  ┌─────────────────────────────────────────────────────────┐
  │  Input: h                                                │
  │    │                                                     │
  │    ▼                                                     │
  │  ┌──────────────┐                                       │
  │  │  Adapter A   │  ← Language adapter (English NLU)     │
  │  │  (frozen)    │                                       │
  │  └──────┬───────┘                                       │
  │         │                                                │
  │         ▼                                                │
  │  ┌──────────────┐                                       │
  │  │  Adapter B   │  ← Domain adapter (Medical)           │
  │  │  (frozen)    │                                       │
  │  └──────┬───────┘                                       │
  │         │                                                │
  │         ▼                                                │
  │  ┌──────────────┐                                       │
  │  │  Adapter C   │  ← Task adapter (Diagnosis classify)  │
  │  │  (trainable) │                                       │
  │  └──────┬───────┘                                       │
  │         │                                                │
  │         ▼                                                │
  │  Output                                                  │
  └─────────────────────────────────────────────────────────┘
"""
        print(diagram)
        
        # Implementation
        print("  Implementation:")
        
        class SequentialAdapterStack(nn.Module):
            """Stack multiple adapters sequentially."""
            
            def __init__(self, adapters: List[nn.Module]):
                super().__init__()
                self.adapters = nn.ModuleList(adapters)
            
            def forward(self, x):
                for adapter in self.adapters:
                    x = adapter(x)
                return x
        
        # Demo
        d_model = 768
        adapters = []
        for i, name in enumerate(["language", "domain", "task"]):
            adapter = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Linear(64, d_model),
            )
            adapters.append(adapter)
        
        stack = SequentialAdapterStack(adapters)
        x = torch.randn(2, 10, d_model)
        output = stack(x)
        print(f"    Input:  {x.shape}")
        print(f"    Output: {output.shape}")
        n_params = sum(p.numel() for p in stack.parameters())
        print(f"    Total adapter params: {n_params:,}")
    
    @staticmethod
    def parallel_composition():
        """Process through multiple adapters in parallel."""
        print("\n" + "=" * 65)
        print("  PARALLEL ADAPTER COMPOSITION")
        print("=" * 65)
        
        diagram = """
  Parallel: h → [Adapter_A] ──┐
            h → [Adapter_B] ──┼─→ Merge → output
            h → [Adapter_C] ──┘
  
  Merge strategies:
    • Average:  output = (A(h) + B(h) + C(h)) / 3
    • Weighted: output = w1·A(h) + w2·B(h) + w3·C(h)
    • Concat:   output = MLP([A(h); B(h); C(h)])
    • Attention: output = softmax(Q·K^T)·V  (AdapterFusion)
"""
        print(diagram)
        
        class ParallelAdapters(nn.Module):
            """Run multiple adapters in parallel and merge."""
            
            def __init__(self, adapters: List[nn.Module], merge: str = "weighted"):
                super().__init__()
                self.adapters = nn.ModuleList(adapters)
                self.merge = merge
                n = len(adapters)
                
                if merge == "weighted":
                    self.weights = nn.Parameter(torch.ones(n) / n)
                elif merge == "attention":
                    d_model = 768
                    self.query = nn.Linear(d_model, d_model)
                    self.key = nn.Linear(d_model, d_model)
            
            def forward(self, x):
                outputs = [adapter(x) for adapter in self.adapters]
                
                if self.merge == "average":
                    return torch.stack(outputs).mean(dim=0)
                
                elif self.merge == "weighted":
                    w = F.softmax(self.weights, dim=0)
                    result = torch.zeros_like(outputs[0])
                    for i, out in enumerate(outputs):
                        result += w[i] * out
                    return result
                
                elif self.merge == "attention":
                    # Stack: (n_adapters, batch, seq, d)
                    stacked = torch.stack(outputs, dim=2)  # (B, S, N, D)
                    q = self.query(x).unsqueeze(2)         # (B, S, 1, D)
                    k = self.key(stacked)                  # (B, S, N, D)
                    
                    attn = (q * k).sum(dim=-1, keepdim=True)  # (B, S, N, 1)
                    attn = F.softmax(attn / math.sqrt(x.size(-1)), dim=2)
                    
                    return (attn * stacked).sum(dim=2)     # (B, S, D)
        
        # Demo
        d_model = 768
        adapters = []
        for _ in range(3):
            adapters.append(nn.Sequential(
                nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, d_model)
            ))
        
        for merge in ["average", "weighted", "attention"]:
            parallel = ParallelAdapters(adapters, merge=merge)
            x = torch.randn(2, 10, d_model)
            output = parallel(x)
            n_extra = sum(p.numel() for n, p in parallel.named_parameters()
                         if "adapter" not in n)
            print(f"    {merge:>10}: output={output.shape}, "
                  f"merge params={n_extra:,}")
    
    @staticmethod
    def split_composition():
        """Different adapters for different layers."""
        print("\n" + "=" * 65)
        print("  SPLIT ADAPTER COMPOSITION")
        print("=" * 65)
        
        diagram = """
  Split: Assign different adapters to different layers
  
  ┌──────────┐
  │ Layer 0  │ → [Adapter A]  ← General language
  │ Layer 1  │ → [Adapter A]
  │ Layer 2  │ → [Adapter A]
  │ Layer 3  │ → [Adapter A]
  ├──────────┤
  │ Layer 4  │ → [Adapter B]  ← Domain knowledge
  │ Layer 5  │ → [Adapter B]
  │ Layer 6  │ → [Adapter B]
  │ Layer 7  │ → [Adapter B]
  ├──────────┤
  │ Layer 8  │ → [Adapter C]  ← Task-specific
  │ Layer 9  │ → [Adapter C]
  │ Layer 10 │ → [Adapter C]
  │ Layer 11 │ → [Adapter C]
  └──────────┘
  
  Intuition: Different layers encode different levels of abstraction.
  Lower layers: syntax/morphology
  Middle layers: semantics
  Upper layers: task-specific reasoning
  
  This allows specializing adapters for each level.
"""
        print(diagram)
        
        # Implementation
        class SplitAdapterConfig:
            """Configure which adapter goes where."""
            def __init__(self):
                self.layer_mapping = {
                    range(0, 4):  "language",
                    range(4, 8):  "domain",
                    range(8, 12): "task",
                }
            
            def get_adapter_for_layer(self, layer_idx: int) -> str:
                for layer_range, adapter_name in self.layer_mapping.items():
                    if layer_idx in layer_range:
                        return adapter_name
                return "default"
        
        config = SplitAdapterConfig()
        print("  Layer assignment:")
        for i in range(12):
            adapter = config.get_adapter_for_layer(i)
            print(f"    Layer {i:>2} → {adapter}")


# ============================================================================
# SECTION 3: ADAPTER ARITHMETIC
# ============================================================================

class AdapterArithmetic:
    """
    Adapter Arithmetic: Manipulating adapter weights algebraically.
    
    Inspired by word2vec arithmetic (king - man + woman = queen),
    we can perform similar operations on adapter weights:
    
    - Add: Combine capabilities from two adapters
    - Subtract: Remove a capability
    - Scale: Control the strength of an adapter
    - Negate: Reverse an adapter's effect
    
    This is related to "Task Vectors" (Ilharco et al., 2023).
    """
    
    @staticmethod
    def compute_task_vector(base_state: Dict, adapted_state: Dict) -> Dict:
        """
        Compute the task vector τ = θ_adapted - θ_base.
        
        The task vector captures what the adapter "learned".
        """
        task_vector = {}
        for key in adapted_state:
            if key in base_state:
                task_vector[key] = adapted_state[key] - base_state[key]
            else:
                task_vector[key] = adapted_state[key]
        return task_vector
    
    @staticmethod
    def apply_task_vector(
        base_state: Dict,
        task_vector: Dict,
        scaling: float = 1.0,
    ) -> Dict:
        """
        Apply a task vector to the base model:
        θ_new = θ_base + α · τ
        """
        new_state = {}
        for key in base_state:
            if key in task_vector:
                new_state[key] = base_state[key] + scaling * task_vector[key]
            else:
                new_state[key] = base_state[key]
        return new_state
    
    @staticmethod
    def demonstrate_arithmetic():
        """Demonstrate adapter arithmetic operations."""
        print("\n" + "=" * 65)
        print("  ADAPTER ARITHMETIC & TASK VECTORS")
        print("=" * 65)
        
        # Create mock adapters
        d_model = 256
        bottleneck = 32
        
        # "Base" adapter (untrained)
        base_down = torch.zeros(bottleneck, d_model)
        base_up = torch.zeros(d_model, bottleneck)
        
        # "Sentiment" adapter
        torch.manual_seed(42)
        sent_down = torch.randn(bottleneck, d_model) * 0.01
        sent_up = torch.randn(d_model, bottleneck) * 0.01
        
        # "Toxicity" adapter
        torch.manual_seed(123)
        tox_down = torch.randn(bottleneck, d_model) * 0.01
        tox_up = torch.randn(d_model, bottleneck) * 0.01
        
        print(f"\n  Base adapter:      ||W|| = {base_down.norm():.4f}")
        print(f"  Sentiment adapter: ||W|| = {sent_down.norm():.4f}")
        print(f"  Toxicity adapter:  ||W|| = {tox_down.norm():.4f}")
        
        # ── Operation 1: Addition ────────────────────────────────
        print(f"\n  ── Operation 1: Addition ──")
        print(f"  sentiment + toxicity → multi-task adapter")
        combined_down = sent_down + tox_down
        combined_up = sent_up + tox_up
        print(f"  Combined ||W|| = {combined_down.norm():.4f}")
        
        # ── Operation 2: Scaling ─────────────────────────────────
        print(f"\n  ── Operation 2: Scaling ──")
        for alpha in [0.5, 1.0, 1.5, 2.0]:
            scaled = sent_down * alpha
            print(f"  α={alpha:.1f}: ||W|| = {scaled.norm():.4f}")
        
        # ── Operation 3: Negation ────────────────────────────────
        print(f"\n  ── Operation 3: Negation ──")
        print(f"  -toxicity → removes toxic generation patterns")
        negated_down = -tox_down
        print(f"  Original:  ||W|| = {tox_down.norm():.4f}")
        print(f"  Negated:   ||W|| = {negated_down.norm():.4f}")
        print(f"  Sum (should be 0): ||W|| = {(tox_down + negated_down).norm():.6f}")
        
        # ── Operation 4: Subtraction ────────────────────────────
        print(f"\n  ── Operation 4: Subtraction ──")
        print(f"  sentiment - toxicity → sentiment without toxicity")
        diff_down = sent_down - tox_down
        print(f"  Result ||W|| = {diff_down.norm():.4f}")
        
        # ── Operation 5: Interpolation ──────────────────────────
        print(f"\n  ── Operation 5: Interpolation ──")
        print(f"  lerp(sentiment, toxicity, t) → smooth blend")
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            interp = (1 - t) * sent_down + t * tox_down
            print(f"    t={t:.2f}: ||W|| = {interp.norm():.4f}")
    
    @staticmethod
    def demonstrate_task_vectors():
        """
        Task Vectors (Ilharco et al., 2023):
        
        A unified framework for:
        - Forgetting: subtract task vector to remove capability
        - Learning: add task vector to gain capability
        - Analogies: combine task vectors from different domains
        """
        print("\n" + "=" * 65)
        print("  TASK VECTORS FRAMEWORK")
        print("=" * 65)
        
        diagram = """
  Task Vector: τ = θ_finetuned - θ_pretrained
  
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  θ_pretrained ─────────── + α·τ ─────────→ θ_new            │
  │       │                                       │              │
  │       │ fine-tune                              │ has new      │
  │       │ on task                                │ capability   │
  │       ▼                                                      │
  │  θ_finetuned                                                 │
  │                                                              │
  │  τ = θ_finetuned - θ_pretrained                              │
  │                                                              │
  │  Operations on τ:                                            │
  │  ─────────────────────────────────────────────────────────── │
  │  Negation:     θ_pretrained - α·τ → forget task              │
  │  Addition:     θ_pretrained + α·τ_A + β·τ_B → multi-task    │
  │  Analogy:      θ_pretrained + τ_A - τ_B + τ_C               │
  │                                                              │
  │  Example:                                                    │
  │    τ_sentiment = θ_sentiment - θ_base                        │
  │    τ_french = θ_french - θ_base                              │
  │                                                              │
  │    θ_french_sentiment = θ_base + τ_sentiment + τ_french      │
  │    → Model that does sentiment analysis in French!           │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
"""
        print(diagram)
        
        print("  Task Vector Code:")
        code = '''
import torch
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
base_state = base_model.state_dict()

# Load fine-tuned models
sentiment_model = AutoModelForCausalLM.from_pretrained("sentiment-gpt2")
french_model = AutoModelForCausalLM.from_pretrained("french-gpt2")

# Compute task vectors
tau_sentiment = {k: sentiment_model.state_dict()[k] - base_state[k]
                 for k in base_state}
tau_french = {k: french_model.state_dict()[k] - base_state[k]
              for k in base_state}

# Combine: French sentiment analysis
alpha = 0.5  # Scaling to prevent interference
new_state = {k: base_state[k] + alpha * tau_sentiment[k] + alpha * tau_french[k]
             for k in base_state}

base_model.load_state_dict(new_state)
# Now base_model can do French sentiment analysis!
'''
        print(code)


# ============================================================================
# SECTION 4: MULTI-ADAPTER SERVING SYSTEM
# ============================================================================

class AdapterServingSystem:
    """
    Production patterns for serving models with multiple adapters.
    
    Key challenge: Serve many tasks from a single base model
    with dynamic adapter loading and efficient memory management.
    """
    
    def __init__(self, base_model_name: str = "distilgpt2"):
        """Initialize the serving system."""
        self.base_model_name = base_model_name
        self.loaded_adapters: Dict[str, Dict] = {}
        self.adapter_cache_limit = 10
        self.access_counts: Dict[str, int] = {}
    
    def register_adapter(self, name: str, path: str, metadata: Dict = None):
        """Register an adapter for serving."""
        self.loaded_adapters[name] = {
            "path": path,
            "metadata": metadata or {},
            "loaded": False,
            "weights": None,
        }
        self.access_counts[name] = 0
        print(f"  Registered adapter: {name}")
    
    def load_adapter(self, name: str):
        """Load adapter weights into memory."""
        if name not in self.loaded_adapters:
            raise ValueError(f"Adapter '{name}' not registered")
        
        adapter = self.loaded_adapters[name]
        if adapter["loaded"]:
            self.access_counts[name] += 1
            return  # Already loaded
        
        # Check cache limit
        loaded_count = sum(
            1 for a in self.loaded_adapters.values() if a["loaded"]
        )
        if loaded_count >= self.adapter_cache_limit:
            self._evict_least_used()
        
        # Simulate loading
        adapter["loaded"] = True
        self.access_counts[name] += 1
        print(f"  Loaded adapter: {name}")
    
    def _evict_least_used(self):
        """Evict the least recently/frequently used adapter."""
        loaded = {
            name: count for name, count in self.access_counts.items()
            if self.loaded_adapters[name]["loaded"]
        }
        if not loaded:
            return
        
        victim = min(loaded, key=loaded.get)
        self.loaded_adapters[victim]["loaded"] = False
        self.loaded_adapters[victim]["weights"] = None
        print(f"  Evicted adapter: {victim}")
    
    @staticmethod
    def demonstrate_serving_architecture():
        """Show production adapter serving patterns."""
        print("\n" + "=" * 65)
        print("  MULTI-ADAPTER SERVING ARCHITECTURE")
        print("=" * 65)
        
        architecture = """
  ┌──────────────────────────────────────────────────────────────┐
  │                   Adapter Serving System                      │
  │                                                              │
  │  ┌─────────────────────────────────────────────────────────┐ │
  │  │ Request Router                                          │ │
  │  │ Examines request → determines which adapter(s) to use   │ │
  │  └───────────┬──────────────────────────────┬──────────────┘ │
  │              │                              │                │
  │  ┌───────────▼──────────┐    ┌──────────────▼─────────────┐ │
  │  │ Adapter Cache (GPU)  │    │  Adapter Store (Disk/S3)   │ │
  │  │ ┌──────┐ ┌──────┐   │    │  ┌──────┐ ┌──────┐        │ │
  │  │ │Sent. │ │ QA   │   │◄───│  │ NLI  │ │Summ. │ ...    │ │
  │  │ └──────┘ └──────┘   │    │  └──────┘ └──────┘        │ │
  │  │ Hot adapters (~10)   │    │  Cold adapters (100s)      │ │
  │  └───────────┬──────────┘    └────────────────────────────┘ │
  │              │                                               │
  │  ┌───────────▼─────────────────────────────────────────────┐ │
  │  │ Base Model (GPU, shared)                                │ │
  │  │ Frozen weights, loaded once                             │ │
  │  │                                                         │ │
  │  │  Forward: base_output + adapter(base_output)            │ │
  │  └─────────────────────────────────────────────────────────┘ │
  └──────────────────────────────────────────────────────────────┘
  
  Key Design Decisions:
  ─────────────────────────────────────────────────────────────
  • Base model shared across ALL adapters (one GPU copy)
  • Hot adapters kept in GPU memory (~2MB each → 10 = ~20MB)
  • Cold adapters on disk, loaded on demand (~50ms latency)
  • LRU/LFU eviction for adapter cache
  • Batch requests by adapter for throughput
  
  Scaling Strategy:
  ─────────────────────────────────────────────────────────────
  1. Single GPU: 1 base model + 10-50 adapters in GPU memory
  2. Multi-GPU:  Replicate base model, shard adapter cache
  3. Multi-node: Each node has base model + subset of adapters
     Route requests to node with matching adapter
"""
        print(architecture)
        
        code = '''
# Production Adapter Server (FastAPI example)
from fastapi import FastAPI, Request
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from functools import lru_cache

app = FastAPI()

# Load base model ONCE
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Adapter registry
ADAPTER_REGISTRY = {
    "sentiment": "adapters/sentiment",
    "summarize": "adapters/summarize",
    "translate": "adapters/translate",
}

# Cache loaded adapters
@lru_cache(maxsize=10)
def get_adapter_model(adapter_name: str):
    """Load and cache adapter model."""
    path = ADAPTER_REGISTRY[adapter_name]
    model = PeftModel.from_pretrained(base_model, path)
    model.eval()
    return model

@app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    adapter_name = body["adapter"]
    prompt = body["prompt"]
    
    # Get model with the right adapter
    model = get_adapter_model(adapter_name)
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response, "adapter": adapter_name}

# Usage:
# POST /generate {"adapter": "sentiment", "prompt": "Review: Great movie!"}
# POST /generate {"adapter": "summarize", "prompt": "Summarize: ..."}
'''
        print(code)
    
    @staticmethod
    def demonstrate_dynamic_loading():
        """Demonstrate dynamic adapter loading at runtime."""
        print("\n" + "=" * 65)
        print("  DYNAMIC ADAPTER LOADING DEMO")
        print("=" * 65)
        
        system = AdapterServingSystem()
        
        # Register adapters
        adapters = [
            ("sentiment", "/models/adapters/sentiment"),
            ("nli", "/models/adapters/nli"),
            ("qa", "/models/adapters/qa"),
            ("summarize", "/models/adapters/summarize"),
            ("translate", "/models/adapters/translate"),
        ]
        
        print("\n  Registering adapters:")
        for name, path in adapters:
            system.register_adapter(name, path, {
                "task": name,
                "base_model": "distilgpt2",
            })
        
        # Simulate request traffic
        print("\n  Simulating request traffic:")
        requests = [
            "sentiment", "sentiment", "qa", "nli",
            "sentiment", "summarize", "qa", "translate",
            "sentiment", "nli",
        ]
        
        for req in requests:
            system.load_adapter(req)
        
        # Show stats
        print(f"\n  Access statistics:")
        for name, count in sorted(
            system.access_counts.items(),
            key=lambda x: -x[1]
        ):
            loaded = "✓" if system.loaded_adapters[name]["loaded"] else "✗"
            print(f"    [{loaded}] {name:>12}: {count} requests")


# ============================================================================
# SECTION 5: COMPLETE ADAPTER COMPOSITION EXAMPLE
# ============================================================================

def complete_composition_example():
    """End-to-end adapter composition workflow."""
    print("\n" + "=" * 65)
    print("  COMPLETE COMPOSITION WORKFLOW")
    print("=" * 65)
    
    workflow = """
  Real-World Scenario: Multi-language Customer Support Bot
  ═══════════════════════════════════════════════════════════
  
  STEP 1: Train Language Adapters
  ────────────────────────────────
  • adapter_english (trained on English corpus)
  • adapter_french (trained on French corpus)
  • adapter_german (trained on German corpus)
  
  STEP 2: Train Task Adapters
  ────────────────────────────────
  • adapter_faq (FAQ answering)
  • adapter_complaint (complaint handling)
  • adapter_product (product information)
  
  STEP 3: Compose for Deployment
  ────────────────────────────────
  For French FAQ bot:
    model.set_active_adapters(Stack("adapter_french", "adapter_faq"))
  
  For German complaints:
    model.set_active_adapters(Stack("adapter_german", "adapter_complaint"))
  
  STEP 4: AdapterFusion (optional, requires training)
  ────────────────────────────────
  For a general-purpose bot that handles all tasks:
    model.add_adapter_fusion(Fuse("adapter_faq", "adapter_complaint", 
                                   "adapter_product"))
    model.train_adapter_fusion(...)
  
  COST ANALYSIS:
  ════════════════════════════════════════════════════════════
  Without adapters:
    3 languages × 3 tasks = 9 full models
    9 × 500MB = 4.5 GB of model weights
  
  With adapters:
    1 base model + 3 language + 3 task adapters  
    500MB + 6 × 2MB = 512 MB total
    
    Savings: 88% reduction in storage!
  
  DEPLOYMENT:
  ════════════════════════════════════════════════════════════
  All 9 combinations served from a single GPU:
    1 × base model in GPU RAM  = 500 MB
    6 × adapters in GPU RAM    =  12 MB
    Total GPU memory            = 512 MB
  
  vs. 9 separate models = 4.5 GB GPU memory
"""
    print(workflow)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all adapter hub demonstrations."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║              ADAPTER HUB & COMPOSITION                       ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: AdapterHub
    hub = AdapterHubGuide()
    hub.overview()
    hub.show_hub_api()
    
    # Section 2: Composition patterns
    comp = AdapterComposition()
    comp.sequential_stacking()
    comp.parallel_composition()
    comp.split_composition()
    
    # Section 3: Adapter arithmetic
    arith = AdapterArithmetic()
    arith.demonstrate_arithmetic()
    arith.demonstrate_task_vectors()
    
    # Section 4: Serving system
    AdapterServingSystem.demonstrate_serving_architecture()
    AdapterServingSystem.demonstrate_dynamic_loading()
    
    # Section 5: Complete workflow
    complete_composition_example()
    
    print("\n" + "=" * 65)
    print("  MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered in this module:
    ✓ AdapterHub ecosystem and API
    ✓ Sequential, parallel, and split composition
    ✓ Adapter arithmetic and task vectors
    ✓ Multi-adapter serving architecture
    ✓ Dynamic adapter loading and caching
    ✓ Complete multi-language multi-task workflow
    """)


if __name__ == "__main__":
    main()
