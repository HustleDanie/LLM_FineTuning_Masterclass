"""
Retrieval-Augmented Fine-Tuning - Comparison & Analysis
========================================================

Comprehensive benchmarks comparing RAG, RAFT, standard fine-tuning,
and other retrieval-augmented approaches with decision frameworks.

Sections:
    1. RAG vs RAFT vs Standard Fine-Tuning
    2. Retrieval Strategy Comparison
    3. Context Window and Chunk Size Analysis
    4. Scaling and Efficiency Analysis
    5. Decision Framework
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict


# =============================================================================
# SECTION 1: RAG vs RAFT vs Standard Fine-Tuning
# =============================================================================

class RAGvsRAFTvsFTBenchmark:
    """
    Compare three paradigms:
    - RAG: Retrieve + Generate (no fine-tuning on retrieval context)
    - RAFT: Retrieve + Fine-tune with oracle/distractor mixing
    - Standard FT: Fine-tune without retrieval augmentation
    
    Key finding from RAFT paper:
    RAFT significantly outperforms RAG on domain-specific QA because
    the model learns to find relevant info among noise during training.
    """
    
    def __init__(self):
        self.approaches = {
            "standard_ft": {
                "name": "Standard Fine-Tuning",
                "description": "Fine-tune on QA pairs without retrieval context",
                "uses_retrieval_at_train": False,
                "uses_retrieval_at_inference": False,
                "trains_generator": True,
                "trains_retriever": False,
            },
            "rag": {
                "name": "Retrieval-Augmented Generation (RAG)",
                "description": "Retrieve docs at inference, no training on retrieval",
                "uses_retrieval_at_train": False,
                "uses_retrieval_at_inference": True,
                "trains_generator": False,
                "trains_retriever": False,
            },
            "rag_ft": {
                "name": "RAG + Fine-Tuning",
                "description": "Fine-tune on QA, then add retrieval at inference",
                "uses_retrieval_at_train": False,
                "uses_retrieval_at_inference": True,
                "trains_generator": True,
                "trains_retriever": False,
            },
            "raft": {
                "name": "RAFT (Retrieval-Augmented Fine-Tuning)",
                "description": "Fine-tune with oracle + distractor document mixing",
                "uses_retrieval_at_train": True,
                "uses_retrieval_at_inference": True,
                "trains_generator": True,
                "trains_retriever": False,
            },
            "self_rag": {
                "name": "Self-RAG",
                "description": "Model learns when to retrieve and how to evaluate",
                "uses_retrieval_at_train": True,
                "uses_retrieval_at_inference": True,
                "trains_generator": True,
                "trains_retriever": False,
            },
            "ra_dit": {
                "name": "RA-DIT (Retrieval-Augmented Dual Instruction Tuning)",
                "description": "Joint retriever + generator instruction tuning",
                "uses_retrieval_at_train": True,
                "uses_retrieval_at_inference": True,
                "trains_generator": True,
                "trains_retriever": True,
            },
        }
    
    def simulate_benchmark(self, num_domains: int = 3) -> Dict[str, Dict]:
        """
        Simulate benchmark results across different conditions.
        
        Based on findings from RAFT, Self-RAG, and RA-DIT papers.
        """
        domains = ["Medical QA", "Legal QA", "Scientific QA"][:num_domains]
        
        # Simulated performance (based on paper trends)
        performance = {
            "standard_ft": {
                "in_domain": {"em": 0.52, "f1": 0.68},
                "out_domain": {"em": 0.25, "f1": 0.40},
                "with_noise": {"em": 0.48, "f1": 0.63},
                "knowledge_update": {"em": 0.15, "f1": 0.28},
            },
            "rag": {
                "in_domain": {"em": 0.58, "f1": 0.73},
                "out_domain": {"em": 0.42, "f1": 0.58},
                "with_noise": {"em": 0.35, "f1": 0.50},
                "knowledge_update": {"em": 0.55, "f1": 0.70},
            },
            "rag_ft": {
                "in_domain": {"em": 0.62, "f1": 0.76},
                "out_domain": {"em": 0.38, "f1": 0.55},
                "with_noise": {"em": 0.40, "f1": 0.58},
                "knowledge_update": {"em": 0.52, "f1": 0.68},
            },
            "raft": {
                "in_domain": {"em": 0.72, "f1": 0.85},
                "out_domain": {"em": 0.45, "f1": 0.62},
                "with_noise": {"em": 0.65, "f1": 0.78},
                "knowledge_update": {"em": 0.60, "f1": 0.75},
            },
            "self_rag": {
                "in_domain": {"em": 0.70, "f1": 0.83},
                "out_domain": {"em": 0.50, "f1": 0.66},
                "with_noise": {"em": 0.62, "f1": 0.76},
                "knowledge_update": {"em": 0.58, "f1": 0.73},
            },
            "ra_dit": {
                "in_domain": {"em": 0.74, "f1": 0.87},
                "out_domain": {"em": 0.52, "f1": 0.68},
                "with_noise": {"em": 0.68, "f1": 0.82},
                "knowledge_update": {"em": 0.65, "f1": 0.80},
            },
        }
        
        return performance
    
    def analyze_results(self, results: Dict) -> None:
        """Print comprehensive comparison analysis."""
        print("\n  Performance Comparison (Simulated based on paper results):")
        print(f"  {'Approach':<20} {'In-Domain EM':>13} {'Out-Domain EM':>14} "
              f"{'Noisy EM':>10} {'Updated EM':>11}")
        print("  " + "-" * 72)
        
        for approach_key, metrics in results.items():
            name = self.approaches[approach_key]["name"][:19]
            in_d = metrics["in_domain"]["em"]
            out_d = metrics["out_domain"]["em"]
            noise = metrics["with_noise"]["em"]
            update = metrics["knowledge_update"]["em"]
            print(f"  {name:<20} {in_d:>13.3f} {out_d:>14.3f} "
                  f"{noise:>10.3f} {update:>11.3f}")
        
        print("""
  Key Findings:
  ┌────────────────────────────────────────────────────────────────┐
  │ 1. RAFT > RAG on domain-specific QA (trained on noisy context)│
  │ 2. RAFT > Standard FT for knowledge-intensive tasks           │
  │ 3. RAG > Standard FT for knowledge updates (no retraining)    │
  │ 4. RAFT most robust to noisy/irrelevant documents             │
  │ 5. RA-DIT best overall but most expensive to train            │
  │ 6. Self-RAG flexible: knows WHEN to retrieve                  │
  │ 7. Standard FT fails on knowledge updates (parametric only)   │
  └────────────────────────────────────────────────────────────────┘
        """)
    
    def cost_analysis(self) -> None:
        """Analyze training and inference costs."""
        costs = {
            "Standard FT": {
                "train_gpu_hours": 4, "train_data_prep": "Low",
                "inference_latency": "Low", "storage": "Model only",
                "retraining_for_update": True
            },
            "RAG": {
                "train_gpu_hours": 0, "train_data_prep": "Low",
                "inference_latency": "Medium", "storage": "Model + Index",
                "retraining_for_update": False
            },
            "RAFT": {
                "train_gpu_hours": 8, "train_data_prep": "High",
                "inference_latency": "Medium", "storage": "Model + Index",
                "retraining_for_update": True
            },
            "Self-RAG": {
                "train_gpu_hours": 12, "train_data_prep": "Very High",
                "inference_latency": "High", "storage": "Model + Index",
                "retraining_for_update": True
            },
            "RA-DIT": {
                "train_gpu_hours": 20, "train_data_prep": "Very High",
                "inference_latency": "Medium", "storage": "Model + Index + Retriever",
                "retraining_for_update": True
            },
        }
        
        print("\n  Cost Comparison:")
        print(f"  {'Approach':<14} {'GPU Hours':>10} {'Data Prep':>10} "
              f"{'Latency':>10} {'Re-train?':>10}")
        print("  " + "-" * 58)
        
        for name, cost in costs.items():
            print(f"  {name:<14} {cost['train_gpu_hours']:>10} {cost['train_data_prep']:>10} "
                  f"{cost['inference_latency']:>10} {'Yes' if cost['retraining_for_update'] else 'No':>10}")


def demonstrate_rag_vs_raft():
    """Demonstrate RAG vs RAFT comparison."""
    print("=" * 60)
    print("RAG vs RAFT vs STANDARD FINE-TUNING")
    print("=" * 60)
    
    benchmark = RAGvsRAFTvsFTBenchmark()
    results = benchmark.simulate_benchmark()
    benchmark.analyze_results(results)
    benchmark.cost_analysis()


# =============================================================================
# SECTION 2: Retrieval Strategy Comparison
# =============================================================================

class RetrievalStrategyComparison:
    """
    Compare different retrieval approaches for RAG/RAFT:
    - BM25 (sparse, keyword-based)
    - Dense Retrieval (DPR, Contriever)
    - Hybrid (BM25 + Dense)
    - Cross-Encoder Re-ranking
    - ColBERT (late interaction)
    """
    
    def __init__(self):
        self.strategies = {
            "bm25": {
                "name": "BM25 (Sparse)",
                "type": "sparse",
                "training_required": False,
                "index_type": "inverted",
                "pros": ["No training needed", "Fast indexing", "Good keyword match"],
                "cons": ["No semantic understanding", "Vocabulary mismatch"],
                "best_for": "Exact keyword matching, initial retrieval"
            },
            "dpr": {
                "name": "DPR (Dense Passage Retrieval)",
                "type": "dense",
                "training_required": True,
                "index_type": "FAISS/vector",
                "pros": ["Semantic understanding", "Handles paraphrases"],
                "cons": ["Requires training data", "Slower indexing"],
                "best_for": "Semantic similarity, open-domain QA"
            },
            "contriever": {
                "name": "Contriever (Unsupervised Dense)",
                "type": "dense",
                "training_required": False,
                "index_type": "FAISS/vector",
                "pros": ["No labeled data needed", "Good zero-shot"],
                "cons": ["May underperform supervised DPR"],
                "best_for": "Low-resource domains, zero-shot retrieval"
            },
            "hybrid": {
                "name": "Hybrid (BM25 + Dense)",
                "type": "hybrid",
                "training_required": True,
                "index_type": "Both",
                "pros": ["Best of both worlds", "Robust"],
                "cons": ["More complex", "Two indices needed"],
                "best_for": "Production systems, maximum recall"
            },
            "colbert": {
                "name": "ColBERT (Late Interaction)",
                "type": "dense",
                "training_required": True,
                "index_type": "Token-level vectors",
                "pros": ["Fine-grained matching", "Efficient with precomputation"],
                "cons": ["Large index size", "Complex implementation"],
                "best_for": "High-precision retrieval, passage ranking"
            },
            "cross_encoder": {
                "name": "Cross-Encoder Re-ranker",
                "type": "reranker",
                "training_required": True,
                "index_type": "N/A (reranks candidates)",
                "pros": ["Most accurate scoring", "Captures query-doc interaction"],
                "cons": ["O(n) per query (slow)", "Only for reranking, not retrieval"],
                "best_for": "Re-ranking top-K from first-stage retriever"
            },
        }
    
    def simulate_performance(self) -> Dict:
        """Simulate retrieval performance metrics."""
        performance = {
            "bm25":          {"recall@5": 0.65, "recall@20": 0.82, "mrr": 0.58, "latency_ms": 5},
            "dpr":           {"recall@5": 0.72, "recall@20": 0.88, "mrr": 0.65, "latency_ms": 15},
            "contriever":    {"recall@5": 0.68, "recall@20": 0.85, "mrr": 0.60, "latency_ms": 15},
            "hybrid":        {"recall@5": 0.78, "recall@20": 0.92, "mrr": 0.72, "latency_ms": 20},
            "colbert":       {"recall@5": 0.76, "recall@20": 0.90, "mrr": 0.70, "latency_ms": 25},
            "cross_encoder": {"recall@5": 0.82, "recall@20": 0.82, "mrr": 0.78, "latency_ms": 200},
        }
        return performance
    
    def print_comparison(self) -> None:
        """Print detailed retrieval strategy comparison."""
        perf = self.simulate_performance()
        
        print("\n  Retrieval Performance Comparison:")
        print(f"  {'Strategy':<25} {'R@5':>6} {'R@20':>6} {'MRR':>6} {'Latency':>10}")
        print("  " + "-" * 57)
        
        for key in self.strategies:
            name = self.strategies[key]["name"][:24]
            p = perf[key]
            print(f"  {name:<25} {p['recall@5']:>6.3f} {p['recall@20']:>6.3f} "
                  f"{p['mrr']:>6.3f} {p['latency_ms']:>8}ms")
        
        print("""
  Recommended Pipeline:
  ┌──────────────────────────────────────────────────────────────────┐
  │ Stage 1: BM25 or Hybrid → Retrieve top-100 candidates          │
  │          (Fast, high recall, no GPU needed)                     │
  │                                                                  │
  │ Stage 2: Dense Retriever → Re-rank to top-20                   │
  │          (Semantic matching, moderate compute)                   │
  │                                                                  │
  │ Stage 3: Cross-Encoder → Re-rank to top-5                     │
  │          (Most accurate, only processes 20 candidates)          │
  │                                                                  │
  │ Stage 4: RAFT Generator → Generate answer from top-5           │
  │          (Trained to handle noisy context)                      │
  └──────────────────────────────────────────────────────────────────┘
        """)


def demonstrate_retrieval_comparison():
    """Demonstrate retrieval strategy comparison."""
    print("\n" + "=" * 60)
    print("RETRIEVAL STRATEGY COMPARISON")
    print("=" * 60)
    
    comparison = RetrievalStrategyComparison()
    comparison.print_comparison()
    
    # Detailed strategy analysis
    print("  Strategy Details:")
    for key, info in comparison.strategies.items():
        print(f"\n  {info['name']}:")
        print(f"    Type: {info['type']}")
        print(f"    Best for: {info['best_for']}")
        print(f"    Pros: {', '.join(info['pros'][:2])}")
        print(f"    Cons: {', '.join(info['cons'][:2])}")


# =============================================================================
# SECTION 3: Context Window and Chunk Size Analysis
# =============================================================================

class ContextChunkAnalysis:
    """
    Analyze the impact of context window size, chunk size,
    number of retrieved documents, and chunk overlap on RAFT performance.
    """
    
    def __init__(self):
        pass
    
    def chunk_size_analysis(self) -> Dict:
        """
        Analyze how chunk size affects retrieval and generation.
        
        Trade-offs:
        - Small chunks: precise retrieval, may miss context
        - Large chunks: more context, may dilute relevance
        """
        chunk_sizes = [64, 128, 256, 512, 1024]
        
        # Simulated results based on typical findings
        results = {}
        for size in chunk_sizes:
            # Retrieval quality peaks at medium chunk sizes
            retrieval_quality = 0.5 + 0.3 * np.exp(-((size - 256) / 200) ** 2)
            # Generation quality benefits from more context (up to a point)
            gen_quality = 0.4 + 0.35 * (1 - np.exp(-size / 300))
            # Too large chunks hurt generation
            if size > 512:
                gen_quality -= 0.05 * (size - 512) / 512
            
            results[size] = {
                "retrieval_recall": min(retrieval_quality, 0.95),
                "generation_f1": min(gen_quality, 0.90),
                "combined_score": min(
                    (retrieval_quality * 0.4 + gen_quality * 0.6), 0.90
                ),
                "chunks_per_doc": max(1, 2000 // size),
                "context_utilization": min(size / 512, 1.0)
            }
        
        return results
    
    def top_k_analysis(self) -> Dict:
        """
        Analyze how the number of retrieved documents affects performance.
        
        More documents = more likely to include the oracle
        But also = more noise for the generator to handle
        """
        k_values = [1, 3, 5, 10, 20]
        
        results = {}
        for k in k_values:
            # Recall improves with more docs
            recall = 1.0 - 0.5 * np.exp(-k / 3)
            # F1 peaks then decreases (too much noise)
            f1 = 0.5 + 0.25 * np.log(k + 1) - 0.08 * k
            f1 = max(f1, 0.35)
            # RAFT is more robust to noise than RAG
            raft_f1 = f1 + 0.08
            rag_f1 = f1 - 0.05 * max(0, k - 5)
            
            results[k] = {
                "recall@k": min(recall, 0.99),
                "raft_f1": min(raft_f1, 0.90),
                "rag_f1": max(min(rag_f1, 0.85), 0.30),
                "context_tokens": k * 256,  # Assuming 256 tokens per chunk
            }
        
        return results
    
    def overlap_analysis(self) -> Dict:
        """Analyze chunk overlap impact."""
        overlaps = [0, 0.1, 0.2, 0.3, 0.5]
        
        results = {}
        for overlap in overlaps:
            # Moderate overlap improves boundary coverage
            boundary_coverage = 0.7 + 0.25 * overlap
            # But increases index size
            index_multiplier = 1.0 / (1.0 - overlap) if overlap < 1.0 else 10.0
            # Retrieval quality
            retrieval = 0.72 + 0.1 * overlap - 0.05 * overlap ** 2
            
            results[overlap] = {
                "boundary_coverage": min(boundary_coverage, 0.95),
                "index_size_multiplier": round(index_multiplier, 2),
                "retrieval_recall": min(retrieval, 0.90),
            }
        
        return results
    
    def print_analysis(self):
        """Print all analysis results."""
        # Chunk size
        print("\n  Chunk Size Analysis:")
        chunk_results = self.chunk_size_analysis()
        print(f"  {'Size':>6} {'Retrieval':>10} {'Gen F1':>8} {'Combined':>10} {'Chunks/Doc':>11}")
        print("  " + "-" * 49)
        for size, m in chunk_results.items():
            print(f"  {size:>6} {m['retrieval_recall']:>10.3f} {m['generation_f1']:>8.3f} "
                  f"{m['combined_score']:>10.3f} {m['chunks_per_doc']:>11}")
        
        # Top-K
        print("\n  Top-K Documents Analysis:")
        k_results = self.top_k_analysis()
        print(f"  {'K':>4} {'Recall@K':>9} {'RAFT F1':>9} {'RAG F1':>8} {'Tokens':>8}")
        print("  " + "-" * 42)
        for k, m in k_results.items():
            print(f"  {k:>4} {m['recall@k']:>9.3f} {m['raft_f1']:>9.3f} "
                  f"{m['rag_f1']:>8.3f} {m['context_tokens']:>8}")
        
        # Overlap
        print("\n  Chunk Overlap Analysis:")
        overlap_results = self.overlap_analysis()
        print(f"  {'Overlap':>8} {'Boundary':>10} {'Index×':>8} {'Recall':>8}")
        print("  " + "-" * 38)
        for overlap, m in overlap_results.items():
            print(f"  {overlap:>8.0%} {m['boundary_coverage']:>10.3f} "
                  f"{m['index_size_multiplier']:>8.2f} {m['retrieval_recall']:>8.3f}")
        
        print("""
  Recommendations:
  ┌────────────────────────────────────────────────────────────┐
  │ Chunk size: 256-512 tokens (balance precision & context)   │
  │ Top-K:      5-10 documents (RAFT handles noise well)       │
  │ Overlap:    10-20% (good boundary coverage, moderate cost) │
  │ Note: RAFT maintains quality at higher K better than RAG   │
  └────────────────────────────────────────────────────────────┘
        """)


def demonstrate_context_analysis():
    """Demonstrate context and chunk analysis."""
    print("\n" + "=" * 60)
    print("CONTEXT WINDOW AND CHUNK SIZE ANALYSIS")
    print("=" * 60)
    
    analysis = ContextChunkAnalysis()
    analysis.print_analysis()


# =============================================================================
# SECTION 4: Scaling and Efficiency Analysis
# =============================================================================

class ScalingAnalysis:
    """
    Analyze how RAFT performance scales with:
    - Model size
    - Training data size
    - Corpus size
    - Number of distractors during training
    """
    
    def __init__(self):
        pass
    
    def model_size_scaling(self) -> Dict:
        """How does RAFT benefit scale with model size?"""
        model_sizes = {
            "125M": 125_000_000,
            "350M": 350_000_000,
            "1.3B": 1_300_000_000,
            "7B": 7_000_000_000,
            "70B": 70_000_000_000,
        }
        
        results = {}
        for name, params in model_sizes.items():
            log_params = np.log10(params)
            
            # Larger models benefit more from RAFT
            base_ft = 0.3 + 0.06 * log_params
            rag = base_ft + 0.08
            raft = base_ft + 0.15 + 0.02 * max(0, log_params - 8)
            raft_gap = raft - rag
            
            results[name] = {
                "params": params,
                "standard_ft_f1": min(base_ft, 0.88),
                "rag_f1": min(rag, 0.90),
                "raft_f1": min(raft, 0.93),
                "raft_advantage": round(raft_gap, 3),
            }
        
        return results
    
    def training_data_scaling(self) -> Dict:
        """How does RAFT performance scale with training data size?"""
        data_sizes = [100, 500, 1000, 5000, 10000, 50000]
        
        results = {}
        for n in data_sizes:
            log_n = np.log10(n)
            
            raft = 0.3 + 0.15 * log_n - 0.01 * log_n ** 2
            rag = 0.45 + 0.05 * log_n  # RAG less affected by training data
            ft = 0.2 + 0.17 * log_n - 0.015 * log_n ** 2
            
            results[n] = {
                "raft_f1": min(raft, 0.90),
                "rag_f1": min(rag, 0.82),
                "standard_ft_f1": min(ft, 0.85),
            }
        
        return results
    
    def distractor_count_scaling(self) -> Dict:
        """Effect of number of distractors during RAFT training."""
        distractor_counts = [0, 1, 2, 3, 5, 8, 10]
        
        results = {}
        for d in distractor_counts:
            # More distractors during training → more robust model
            train_quality = 0.55 + 0.12 * np.log(d + 1) - 0.01 * d
            noise_robustness = 0.40 + 0.15 * np.log(d + 1)
            training_cost = 1.0 + 0.3 * d  # Longer context → more compute
            
            results[d] = {
                "clean_f1": min(train_quality, 0.88),
                "noisy_f1": min(noise_robustness, 0.85),
                "relative_training_cost": round(training_cost, 1),
            }
        
        return results
    
    def lora_vs_full_ft(self) -> Dict:
        """Compare LoRA vs full fine-tuning for RAFT."""
        configs = {
            "Full FT": {
                "trainable_pct": 100.0, "f1": 0.85, "gpu_memory_gb": 40,
                "train_time_hrs": 12, "switch_domains": "Full retrain"
            },
            "LoRA r=4": {
                "trainable_pct": 0.3, "f1": 0.79, "gpu_memory_gb": 10,
                "train_time_hrs": 3, "switch_domains": "Swap adapter"
            },
            "LoRA r=16": {
                "trainable_pct": 1.2, "f1": 0.83, "gpu_memory_gb": 12,
                "train_time_hrs": 4, "switch_domains": "Swap adapter"
            },
            "LoRA r=64": {
                "trainable_pct": 4.8, "f1": 0.84, "gpu_memory_gb": 16,
                "train_time_hrs": 6, "switch_domains": "Swap adapter"
            },
            "QLoRA r=16": {
                "trainable_pct": 1.2, "f1": 0.81, "gpu_memory_gb": 6,
                "train_time_hrs": 5, "switch_domains": "Swap adapter"
            },
        }
        return configs
    
    def print_analysis(self):
        """Print scaling analysis."""
        # Model size
        print("\n  Model Size Scaling:")
        model_results = self.model_size_scaling()
        print(f"  {'Model':>6} {'Std FT':>8} {'RAG':>6} {'RAFT':>6} {'RAFT Adv':>10}")
        print("  " + "-" * 40)
        for name, m in model_results.items():
            print(f"  {name:>6} {m['standard_ft_f1']:>8.3f} {m['rag_f1']:>6.3f} "
                  f"{m['raft_f1']:>6.3f} {m['raft_advantage']:>+10.3f}")
        
        # Training data
        print("\n  Training Data Scaling:")
        data_results = self.training_data_scaling()
        print(f"  {'N':>8} {'RAFT':>6} {'RAG':>6} {'Std FT':>8}")
        print("  " + "-" * 32)
        for n, m in data_results.items():
            print(f"  {n:>8} {m['raft_f1']:>6.3f} {m['rag_f1']:>6.3f} "
                  f"{m['standard_ft_f1']:>8.3f}")
        
        # Distractor count
        print("\n  Training Distractor Count:")
        dist_results = self.distractor_count_scaling()
        print(f"  {'Dists':>6} {'Clean F1':>9} {'Noisy F1':>9} {'Cost ×':>8}")
        print("  " + "-" * 36)
        for d, m in dist_results.items():
            print(f"  {d:>6} {m['clean_f1']:>9.3f} {m['noisy_f1']:>9.3f} "
                  f"{m['relative_training_cost']:>8.1f}")
        
        # LoRA comparison
        print("\n  LoRA vs Full Fine-Tuning for RAFT:")
        lora_results = self.lora_vs_full_ft()
        print(f"  {'Config':<14} {'Trainable':>10} {'F1':>5} {'GPU GB':>7} {'Hours':>6}")
        print("  " + "-" * 46)
        for config, m in lora_results.items():
            print(f"  {config:<14} {m['trainable_pct']:>9.1f}% {m['f1']:>5.2f} "
                  f"{m['gpu_memory_gb']:>7} {m['train_time_hrs']:>6}")
        
        print("""
  Scaling Insights:
  ┌──────────────────────────────────────────────────────────────┐
  │ • RAFT advantage grows with model size (more capacity)      │
  │ • RAFT needs 1K+ examples for clear benefit over RAG        │
  │ • 3-5 distractors during training is the sweet spot         │
  │ • LoRA r=16 captures ~97% of full FT quality at 10% cost   │
  │ • QLoRA enables RAFT on consumer GPUs (6GB VRAM)            │
  └──────────────────────────────────────────────────────────────┘
        """)


def demonstrate_scaling():
    """Demonstrate scaling analysis."""
    print("\n" + "=" * 60)
    print("SCALING AND EFFICIENCY ANALYSIS")
    print("=" * 60)
    
    analysis = ScalingAnalysis()
    analysis.print_analysis()


# =============================================================================
# SECTION 5: Decision Framework
# =============================================================================

class RAFTDecisionFramework:
    """
    Decision framework for choosing between RAG, RAFT, and alternatives.
    """
    
    def __init__(self):
        self.decision_tree = {
            "root": {
                "question": "Is the knowledge base static or frequently updated?",
                "options": {
                    "Frequently updated (weekly+)": "update_freq",
                    "Mostly static": "accuracy_req"
                }
            },
            "update_freq": {
                "question": "Is high accuracy on domain-specific QA critical?",
                "options": {
                    "Yes, accuracy is paramount": "recommend_raft_with_refresh",
                    "Good enough accuracy is fine": "recommend_rag"
                }
            },
            "accuracy_req": {
                "question": "Do you have labeled QA pairs for your domain?",
                "options": {
                    "Yes, 1000+ QA pairs": "recommend_raft",
                    "Limited data (<500)": "recommend_rag_or_selfrag",
                    "No labeled data": "recommend_rag"
                }
            },
            "recommend_raft": {
                "recommendation": "RAFT",
                "reason": "Static domain + labeled data = ideal for RAFT training"
            },
            "recommend_raft_with_refresh": {
                "recommendation": "RAFT + Periodic Retraining",
                "reason": "High accuracy needed → RAFT; updates → periodic retrain + index refresh"
            },
            "recommend_rag": {
                "recommendation": "RAG (no fine-tuning)",
                "reason": "Frequent updates or no labeled data → pure RAG is most practical"
            },
            "recommend_rag_or_selfrag": {
                "recommendation": "Self-RAG or RAG + Few-shot",
                "reason": "Limited data → Self-RAG learns when to retrieve; few-shot prompting helps too"
            },
        }
    
    def print_decision_tree(self):
        """Print the decision tree."""
        print("""
  RAFT Decision Tree:
  
  Q1: Knowledge base update frequency?
  ├── Frequent (weekly+)
  │   ├── High accuracy critical?
  │   │   ├── Yes → RAFT + Periodic Retraining
  │   │   └── No  → Pure RAG (easiest to maintain)
  │   └── 
  └── Mostly static
      ├── Labeled QA pairs available?
      │   ├── 1000+ pairs → RAFT (best quality)
      │   ├── <500 pairs  → Self-RAG or RAG + Few-shot
      │   └── No data     → Pure RAG
      └──
        """)
    
    def print_production_checklist(self):
        """Print production deployment checklist."""
        print("""
  Production RAFT Deployment Checklist:
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  Data Preparation:                                               │
  │  □ Collect domain-specific QA pairs (target: 1K-10K)            │
  │  □ Prepare document corpus (clean, deduplicate)                  │
  │  □ Design chunking strategy (test 256/512 token sizes)           │
  │  □ Match QA pairs to oracle documents                            │
  │  □ Set oracle/distractor mixing ratios (60/20/20)                │
  │  □ Generate chain-of-thought annotations                         │
  │                                                                  │
  │  Retriever Setup:                                                │
  │  □ Choose retriever (BM25 for MVP, dense for production)        │
  │  □ Build document index (FAISS for scale)                        │
  │  □ Evaluate retrieval quality (Recall@K ≥ 0.85)                 │
  │  □ Consider hybrid retrieval (BM25 + Dense)                      │
  │                                                                  │
  │  Generator Training:                                             │
  │  □ Select base model (7B+ recommended for quality)               │
  │  □ Apply LoRA (r=16 for efficiency, r=64 for quality)           │
  │  □ Train with RAFT data (3-5 epochs, lr=2e-5)                   │
  │  □ Evaluate on held-out test set (EM, F1, faithfulness)         │
  │  □ Test robustness with varying distractor counts                │
  │                                                                  │
  │  Deployment:                                                     │
  │  □ Set up retrieval infrastructure (FAISS, Elasticsearch)       │
  │  □ Configure top-K (5 for accuracy, 3 for speed)                │
  │  □ Add monitoring for retrieval quality degradation              │
  │  □ Plan index refresh schedule (if corpus changes)               │
  │  □ Implement fallback to parametric knowledge                    │
  │  □ A/B test against pure RAG baseline                            │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
        """)
    
    def print_common_pitfalls(self):
        """Print common pitfalls and solutions."""
        pitfalls = [
            {
                "problem": "Generator ignores retrieved documents",
                "cause": "Not trained on retrieval context (using RAG, not RAFT)",
                "solution": "Switch to RAFT training with oracle/distractor mixing"
            },
            {
                "problem": "Hallucination despite retrieval",
                "cause": "Model trusts parametric memory over context",
                "solution": "Increase distractor-only ratio to teach context reliance; "
                           "use chain-of-thought extraction"
            },
            {
                "problem": "Poor performance with many documents",
                "cause": "Context too long, oracle info diluted",
                "solution": "Reduce top-K, improve retrieval quality, use re-ranking"
            },
            {
                "problem": "Retriever returns irrelevant documents",
                "cause": "Weak retriever or domain mismatch",
                "solution": "Domain-adapt retriever, add hard negative mining, try hybrid"
            },
            {
                "problem": "High latency at inference",
                "cause": "Dense retrieval + re-ranking + generation pipeline",
                "solution": "Cache common queries, use approximate search, reduce top-K"
            },
            {
                "problem": "Performance drops on new topics",
                "cause": "RAFT overfit to training domain subdistributions",
                "solution": "Diversify training data, use Self-RAG for adaptive retrieval"
            },
        ]
        
        print("\n  Common Pitfalls and Solutions:")
        for i, p in enumerate(pitfalls, 1):
            print(f"\n  {i}. {p['problem']}")
            print(f"     Cause:    {p['cause']}")
            print(f"     Solution: {p['solution']}")
    
    def print_when_not_to_use_raft(self):
        """Cases where RAFT is NOT recommended."""
        print("""
  When NOT to Use RAFT:
  ┌──────────────────────────────────────────────────────────────────┐
  │ 1. Rapidly changing knowledge (daily): Pure RAG is better       │
  │ 2. No domain QA data available: RAG or Self-RAG instead         │
  │ 3. Open-ended creative tasks: RAFT adds unnecessary rigidity    │
  │ 4. Very small corpus (<100 docs): Standard FT may suffice       │
  │ 5. Extreme latency requirements (<50ms): FT only, no retrieval  │
  │ 6. Multi-lingual with limited per-language data: RAG + prompting│
  └──────────────────────────────────────────────────────────────────┘
        """)


def demonstrate_decision_framework():
    """Demonstrate decision framework."""
    print("\n" + "=" * 60)
    print("DECISION FRAMEWORK")
    print("=" * 60)
    
    framework = RAFTDecisionFramework()
    framework.print_decision_tree()
    framework.print_production_checklist()
    framework.print_common_pitfalls()
    framework.print_when_not_to_use_raft()


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 70)
    print("RETRIEVAL-AUGMENTED FINE-TUNING — COMPARISON & ANALYSIS")
    print("=" * 70)
    
    # Section 1: RAG vs RAFT
    print("\n\n📊 SECTION 1: RAG vs RAFT vs Standard Fine-Tuning")
    demonstrate_rag_vs_raft()
    
    # Section 2: Retrieval Strategies
    print("\n\n📊 SECTION 2: Retrieval Strategy Comparison")
    demonstrate_retrieval_comparison()
    
    # Section 3: Context Analysis
    print("\n\n📊 SECTION 3: Context Window and Chunk Size Analysis")
    demonstrate_context_analysis()
    
    # Section 4: Scaling
    print("\n\n📊 SECTION 4: Scaling and Efficiency Analysis")
    demonstrate_scaling()
    
    # Section 5: Decision Framework
    print("\n\n📊 SECTION 5: Decision Framework")
    demonstrate_decision_framework()
    
    print("\n" + "=" * 70)
    print("COMPARISON & ANALYSIS COMPLETE")
    print("=" * 70)
    print("""
    Analyzed:
    1. RAG vs RAFT vs FT — performance, cost, knowledge update trade-offs
    2. Retrieval Strategies — BM25, DPR, hybrid, ColBERT, cross-encoder
    3. Context Analysis — chunk size, top-K, overlap optimization
    4. Scaling — model size, data size, distractor count, LoRA efficiency
    5. Decision Framework — when to use RAFT, checklist, pitfalls
    """)


if __name__ == "__main__":
    main()
