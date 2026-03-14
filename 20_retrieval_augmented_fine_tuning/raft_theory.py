"""
Retrieval-Augmented Fine-Tuning - Theoretical Foundations
==========================================================

Deep dive into the theory behind combining retrieval systems with
fine-tuning: why parametric + non-parametric memory works, how to
train models to use retrieved context, and the mathematics of
retrieval-generation coupling.

Sections:
    1. Parametric vs Non-Parametric Knowledge
    2. Retrieval-Augmented Generation (RAG) Theory
    3. RAFT: Training to Use Retrieved Context
    4. Joint Retriever-Generator Optimization
    5. Self-Reflective Retrieval Theory (Self-RAG)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


# =============================================================================
# SECTION 1: Parametric vs Non-Parametric Knowledge
# =============================================================================

class KnowledgeStorageTheory:
    """
    Theory of how LLMs store and access knowledge.
    
    Two types of memory in neural systems:
    
    1. Parametric Memory (Model Weights):
       - Knowledge encoded in billions of parameters
       - Learned during pre-training/fine-tuning
       - Fixed after training (stale knowledge)
       - Access: implicit via forward pass
       - Capacity: limited by model size
    
    2. Non-Parametric Memory (External Retrieval):
       - Knowledge stored in external database/index
       - Updatable without retraining
       - Access: explicit via retrieval query
       - Capacity: unlimited (just add more documents)
    
    The Key Insight:
        Hybrid systems (parametric + non-parametric) outperform
        either approach alone. The model learns REASONING in its
        weights and FACTS from the retrieval system.
    """
    
    @staticmethod
    def demonstrate_knowledge_types():
        """Compare parametric vs non-parametric knowledge storage."""
        print("=" * 70)
        print("PARAMETRIC vs NON-PARAMETRIC KNOWLEDGE")
        print("=" * 70)
        
        print("""
    Parametric Memory (Model Weights):
    ──────────────────────────────────
        ┌─────────────────────────────────────┐
        │  Model Weights (billions of params)  │
        │                                      │
        │  "Paris is the capital of France"    │  ← Learned during pre-training
        │  "Water boils at 100°C"             │  ← Compressed into weights
        │  "Shakespeare wrote Hamlet"          │  ← Cannot be updated easily
        │                                      │
        │  Capacity: ~10B parameters           │
        │  Update: Requires retraining         │
        │  Access: Forward pass                │
        └──────────────────────────────────────┘
    
    Non-Parametric Memory (Retrieval Index):
    ────────────────────────────────────────
        ┌──────────────────────────────────────────┐
        │  Document Index (millions of documents)    │
        │                                            │
        │  Doc 1: "Paris, the capital of France..."  │
        │  Doc 2: "Water has boiling point 100°C..." │
        │  Doc 3: "Latest news: Election results..." │  ← Updated daily!
        │  ...                                       │
        │  Doc N: "New research findings..."          │
        │                                            │
        │  Capacity: Unlimited (add more docs)        │
        │  Update: Add/remove documents               │
        │  Access: Similarity search                  │
        └────────────────────────────────────────────┘
    
    Hybrid (RAFT Approach):
    ──────────────────────
        Model learns:  HOW to reason, HOW to use context (parametric)
        Index stores:  WHAT the facts are (non-parametric)
        
        Result: Accurate, updatable, citable answers
        """)
    
    @staticmethod
    def knowledge_capacity_analysis():
        """Analyze how much knowledge fits in model weights vs external index."""
        print("\n" + "=" * 70)
        print("KNOWLEDGE CAPACITY ANALYSIS")
        print("=" * 70)
        
        # Approximate knowledge storage
        model_sizes = {
            "GPT-2 (1.5B)": {"params_b": 1.5, "est_facts_m": 5},
            "LLaMA-7B": {"params_b": 7, "est_facts_m": 25},
            "LLaMA-70B": {"params_b": 70, "est_facts_m": 200},
            "GPT-4 (~1.7T)": {"params_b": 1700, "est_facts_m": 3000},
        }
        
        # Retrieval index costs
        index_configs = {
            "Wikipedia (6M docs)": {"docs_m": 6, "storage_gb": 20},
            "PubMed (35M abstracts)": {"docs_m": 35, "storage_gb": 50},
            "Common Crawl (1B pages)": {"docs_m": 1000, "storage_gb": 500},
            "Custom Domain (100K docs)": {"docs_m": 0.1, "storage_gb": 0.5},
        }
        
        print("\n  Parametric Storage (Model Weights):")
        print(f"  {'Model':<20} {'Parameters':>12} {'~Facts Stored':>15} {'GPU Memory':>12}")
        print("  " + "-" * 65)
        for name, info in model_sizes.items():
            gpu_gb = info["params_b"] * 2  # ~2 bytes per param (fp16)
            print(f"  {name:<20} {info['params_b']:>10.1f}B {info['est_facts_m']:>13.0f}M "
                  f"{gpu_gb:>10.1f} GB")
        
        print("\n  Non-Parametric Storage (Retrieval Index):")
        print(f"  {'Index':<30} {'Documents':>12} {'Storage':>10}")
        print("  " + "-" * 55)
        for name, info in index_configs.items():
            print(f"  {name:<30} {info['docs_m']:>10.1f}M {info['storage_gb']:>8.1f} GB")
        
        print("""
  Key Insight:
    A 20GB retrieval index (Wikipedia) stores more factual knowledge
    than a 1.5TB model (GPT-4 scale), at 0.01× the storage cost.
    
    The model should focus on REASONING, not MEMORIZATION.
    External retrieval handles the knowledge storage.
        """)


# =============================================================================
# SECTION 2: Retrieval-Augmented Generation (RAG) Theory
# =============================================================================

class RAGTheory:
    """
    Theoretical foundations of Retrieval-Augmented Generation.
    
    RAG models the probability of an output y given input x as:
    
        P(y|x) = Σ_z P(y|x, z) · P(z|x)
    
    Where:
        x = input query
        z = retrieved document(s) 
        P(z|x) = retriever probability (how relevant is z for x)
        P(y|x,z) = generator probability (answer given query + context)
    
    This marginalizes over all possible retrieved documents z,
    weighting each by its retrieval probability.
    
    Two variants:
    1. RAG-Sequence: retrieve once, generate full sequence
    2. RAG-Token: retrieve per token (more expensive, more flexible)
    """
    
    @staticmethod
    def demonstrate_rag_formulation():
        """Show the mathematical formulation of RAG."""
        print("=" * 70)
        print("RAG: MATHEMATICAL FORMULATION")
        print("=" * 70)
        
        print("""
    Standard Language Model:
    ────────────────────────
        P(y|x) = Π_t P(y_t | y_{<t}, x)
        
        The model generates each token based only on the input
        and previously generated tokens. All knowledge must be
        stored in model parameters.
    
    RAG-Sequence:
    ─────────────
        P(y|x) = Σ_{z ∈ top-k} P(z|x) · Π_t P(y_t | y_{<t}, x, z)
        
        1. Retrieve top-k documents z based on query x
        2. For each document z:
           - Generate full answer conditioned on (x, z)
           - Weight by retrieval probability P(z|x)
        3. Marginalize (sum) over all documents
        
        The same document is used throughout generation.
    
    RAG-Token:
    ──────────
        P(y|x) = Π_t Σ_{z ∈ top-k} P(z|x) · P(y_t | y_{<t}, x, z)
        
        At EACH token position:
        1. Retrieve relevant documents
        2. Score each document's contribution to each token
        3. Marginalize per-token
        
        Different documents can influence different parts of the answer.
    
    Retriever Scoring:
    ──────────────────
        P(z|x) = exp(sim(q(x), d(z))) / Σ_{z'} exp(sim(q(x), d(z')))
        
        Where:
            q(x) = query encoder (embeds the question)
            d(z) = document encoder (embeds the document)
            sim() = dot product or cosine similarity
        
        This is a softmax over retrieval scores.
        """)
    
    @staticmethod
    def demonstrate_retrieval_methods():
        """Compare different retrieval approaches."""
        print("\n" + "=" * 70)
        print("RETRIEVAL METHODS COMPARISON")
        print("=" * 70)
        
        methods = {
            "BM25 (Sparse)": {
                "description": "TF-IDF based keyword matching",
                "strengths": "Fast, interpretable, good for exact matches",
                "weaknesses": "No semantic understanding, vocabulary mismatch",
                "latency": "~1ms per query",
                "quality": "Good for keyword-heavy queries",
                "training_needed": False
            },
            "Dense Retrieval (DPR)": {
                "description": "Dual-encoder: query + document embeddings",
                "strengths": "Semantic matching, handles paraphrases",
                "weaknesses": "Needs training data, embedding index storage",
                "latency": "~10ms per query (ANN search)",
                "quality": "Best for semantic/conceptual queries",
                "training_needed": True
            },
            "Hybrid (BM25 + Dense)": {
                "description": "Combine sparse and dense retrieval scores",
                "strengths": "Best of both worlds, robust",
                "weaknesses": "Two-system complexity, score calibration",
                "latency": "~15ms per query",
                "quality": "Best overall, recommended default",
                "training_needed": True
            },
            "ColBERT (Late Interaction)": {
                "description": "Per-token interaction between query and doc",
                "strengths": "Fine-grained matching, high quality",
                "weaknesses": "Larger index, slower search",
                "latency": "~50ms per query",
                "quality": "Highest quality, especially for long documents",
                "training_needed": True
            }
        }
        
        for name, info in methods.items():
            print(f"\n  {name}:")
            print(f"    {info['description']}")
            print(f"    Quality:  {info['quality']}")
            print(f"    Latency:  {info['latency']}")
            print(f"    Training: {'Required' if info['training_needed'] else 'Not needed'}")
    
    @staticmethod
    def chunking_strategy_theory():
        """Theory behind document chunking for retrieval."""
        print("\n" + "=" * 70)
        print("DOCUMENT CHUNKING THEORY")
        print("=" * 70)
        
        print("""
    Why Chunking Matters:
    ─────────────────────
        Documents are often too long to embed or feed to the LM.
        We must split them into chunks that:
        1. Fit within the retriever's context window
        2. Contain self-contained information units
        3. Are small enough for precise retrieval
        4. Are large enough to preserve context
    
    Chunking Strategies:
    ────────────────────
    
    1. Fixed-Size Chunks (256-512 tokens):
       ┌────────┐┌────────┐┌────────┐
       │Chunk 1 ││Chunk 2 ││Chunk 3 │
       └────────┘└────────┘└────────┘
       + Simple, predictable
       - May split mid-sentence or mid-paragraph
    
    2. Overlapping Chunks (stride < window):
       ┌────────────┐
       │  Chunk 1   │
       └────┬───────┘
            └┬──────────┐
             │  Chunk 2  │
             └────┬──────┘
                  └┬──────────┐
                   │  Chunk 3  │
                   └───────────┘
       + No information lost at boundaries
       - More chunks, larger index
    
    3. Semantic Chunks (by paragraph/section):
       ┌─── Section 1 ───┐
       │ Introduction     │ → Chunk 1
       ├─── Section 2 ───┤
       │ Methods          │ → Chunk 2
       │ (long section)   │ → Chunk 3
       ├─── Section 3 ───┤
       │ Results          │ → Chunk 4
       └─────────────────┘
       + Semantically coherent
       - Variable sizes, complex parsing
    
    4. Recursive Chunking:
       Document → Sections → Paragraphs → Sentences
       Start with large chunks; if too big, split recursively.
       + Balances coherence and size
       - Multiple retrieval granularities needed
    
    Optimal Chunk Size (empirical):
    ───────────────────────────────
        Task                    Optimal Chunk Size
        ─────────────           ──────────────────
        Factual QA              128-256 tokens (precise facts)
        Open-ended QA           256-512 tokens (more context)
        Summarization           512-1024 tokens (full passages)
        Code assistance         Function-level (variable)
        Legal/Medical           Paragraph-level (section context)
        """)


# =============================================================================
# SECTION 3: RAFT: Training to Use Retrieved Context
# =============================================================================

class RAFTTheory:
    """
    RAFT (Retrieval Augmented Fine-Tuning) — Zhang et al., 2024.
    
    Core Idea:
        Standard RAG just prepends retrieved documents to the prompt.
        The LLM wasn't specifically trained to handle this context.
        
        RAFT trains the model to:
        1. Identify relevant documents among retrieved results
        2. Extract the correct answer from the relevant document
        3. Ignore "distractor" documents that are irrelevant
        4. Fall back to parametric knowledge when retrieval fails
    
    This is like training a student to take an open-book exam:
    - Sometimes the answer IS in the book (oracle doc present)
    - Sometimes the book has irrelevant pages (distractor docs)
    - Sometimes the book doesn't help (need prior knowledge)
    """
    
    @staticmethod
    def demonstrate_raft_training_strategy():
        """Show the RAFT training data construction strategy."""
        print("=" * 70)
        print("RAFT TRAINING STRATEGY")
        print("=" * 70)
        
        print("""
    RAFT Training Data Construction:
    ────────────────────────────────
    
    Given: Question Q, Answer A, Oracle Document D*, Distractor Docs {D_1..D_k}
    
    Training Mix (the key innovation):
    
    ┌──────────────────────────────────────────────────────────────────┐
    │ Type 1: Oracle + Distractors (P=0.6)                           │
    │                                                                  │
    │ Input:  [Q] + [D_1] + [D*] + [D_3] + [D_4]                    │
    │ Output: <ANSWER> ... extracted from D* ... </ANSWER>            │
    │                                                                  │
    │ Teaches: Find the needle in the haystack                        │
    │          Model must identify D* among distractors               │
    ├──────────────────────────────────────────────────────────────────┤
    │ Type 2: Oracle Only (P=0.2)                                     │
    │                                                                  │
    │ Input:  [Q] + [D*]                                              │
    │ Output: <ANSWER> ... extracted from D* ... </ANSWER>            │
    │                                                                  │
    │ Teaches: Use good context when available                        │
    │          Builds reading comprehension ability                    │
    ├──────────────────────────────────────────────────────────────────┤
    │ Type 3: Distractors Only (P=0.2)                                │
    │                                                                  │
    │ Input:  [Q] + [D_1] + [D_2] + [D_3]  (no D*)                  │
    │ Output: <ANSWER> ... from parametric knowledge ... </ANSWER>    │
    │                                                                  │
    │ Teaches: Fall back to parametric knowledge                      │
    │          Don't trust retrieval blindly                           │
    └──────────────────────────────────────────────────────────────────┘
    
    Chain-of-Thought Enhancement:
    ─────────────────────────────
    RAFT also trains the model to generate a reasoning chain:
    
    Input:  "In the context of [D*], [D_1], ..., answer: Q"
    Output: "The relevant information is in document [D*], which states
             that '...'. Based on this, the answer is: A"
    
    The CoT helps the model:
    1. Explicitly select the oracle document
    2. Quote relevant passages
    3. Reason to the answer
    → More interpretable AND more accurate
        """)
    
    @staticmethod
    def raft_vs_standard_rag():
        """Compare RAFT with standard RAG approaches."""
        print("\n" + "=" * 70)
        print("RAFT vs STANDARD RAG vs FINE-TUNING")
        print("=" * 70)
        
        print("""
    Approach 1: Standard Fine-Tuning (No Retrieval)
    ────────────────────────────────────────────────
        Training: Fine-tune on (question, answer) pairs
        Inference: Generate answer from parametric knowledge only
        
        ✓ Simple, fast inference
        ✗ Hallucination on facts not seen during training
        ✗ Cannot handle knowledge updates
    
    Approach 2: RAG (No Fine-Tuning for Retrieval)
    ───────────────────────────────────────────────
        Training: Standard fine-tuning or no fine-tuning
        Inference: Retrieve docs → prepend to prompt → generate
        
        ✓ Access to external knowledge
        ✗ Model not trained to handle retrieved noise
        ✗ May ignore or misuse retrieved context
        ✗ Distractor documents confuse the model
    
    Approach 3: RAFT (Fine-Tuned for Retrieval)
    ────────────────────────────────────────────
        Training: Fine-tune with oracle + distractor mixing
        Inference: Retrieve docs → prepend to prompt → generate
        
        ✓ Trained to identify relevant documents
        ✓ Handles noisy retrieval gracefully
        ✓ Falls back to parametric knowledge when needed
        ✓ CoT reasoning improves interpretability
    
    Performance Comparison (from RAFT paper):
    ─────────────────────────────────────────
                            Standard FT     RAG      RAFT
        HotpotQA            28.4           41.2     52.8
        Natural Questions   32.1           45.7     54.3
        Domain-Specific     35.2           48.1     61.4
        
    RAFT consistently outperforms both approaches,
    especially on domain-specific tasks where retrieval
    quality varies significantly.
        """)
    
    @staticmethod
    def distractor_design_theory():
        """Theory behind designing effective distractor documents."""
        print("\n" + "=" * 70)
        print("DISTRACTOR DOCUMENT DESIGN")
        print("=" * 70)
        
        print("""
    Why Distractors Matter:
    ───────────────────────
        In real retrieval, not all retrieved docs are relevant.
        Top-k results often include:
        - Partially relevant documents (right topic, wrong answer)
        - Semantically similar but factually different documents
        - Lexically matching but contextually irrelevant documents
        
        The model MUST learn to handle this noise.
    
    Distractor Generation Strategies:
    ─────────────────────────────────
    
    1. Random Documents:
       Pick random docs from the corpus.
       → Easy to distinguish, not very useful for training
    
    2. BM25 Top-k Retrieval (Excluding Oracle):
       Use the question to retrieve top-k docs, remove the oracle.
       → Realistic: these are the actual distractors at inference
       → RECOMMENDED for production
    
    3. Semantic Nearest Neighbors:
       Find docs with high embedding similarity to the oracle.
       → Hardest distractors: model must learn fine distinctions
       → Good for advanced training
    
    4. Entity-Based Distractors:
       Find docs mentioning the same entities but different facts.
       → Tests entity disambiguation ability
       → "Other facts about the same topic"
    
    Difficulty Curriculum:
    ──────────────────────
        Epoch 1: Random distractors (easy)
        Epoch 2: BM25 distractors (medium)
        Epoch 3: Semantic nearest neighbors (hard)
        
        Progressive difficulty helps the model learn robustly.
    
    Number of Distractors:
    ──────────────────────
        k=1:  Easy task, limited noise tolerance training
        k=3:  Good balance (recommended default)
        k=5:  Harder, more realistic for large knowledge bases
        k=10: Very challenging, tests model's focusing ability
        
        At inference, models trained with k=3-5 distractors
        generalize well to any number of retrieved documents.
        """)


# =============================================================================
# SECTION 4: Joint Retriever-Generator Optimization
# =============================================================================

class JointOptimizationTheory:
    """
    Theory behind jointly training the retriever and generator.
    
    The retriever and generator form a coupled system:
    - Better retrieval → better generation
    - Better generation signal → better retrieval training
    
    This creates an opportunity for joint optimization,
    but also risk of degenerate solutions.
    """
    
    @staticmethod
    def demonstrate_joint_training():
        """Show joint retriever-generator optimization approaches."""
        print("=" * 70)
        print("JOINT RETRIEVER-GENERATOR OPTIMIZATION")
        print("=" * 70)
        
        print("""
    The Coupling Problem:
    ─────────────────────
        Retriever and generator are interdependent:
        
        P(y|x) = Σ_z P(y|x,z) · P(z|x)
                  ┬────────────   ┬──────
                  Generator       Retriever
        
        Optimal retriever depends on what the generator needs.
        Optimal generator depends on what the retriever provides.
        → Chicken-and-egg problem!
    
    Approach 1: Frozen Retriever, Fine-Tune Generator (RAFT-style)
    ──────────────────────────────────────────────────────────────
        1. Use off-the-shelf retriever (BM25, DPR, Contriever)
        2. Fine-tune LLM to handle retrieved context
        3. Retriever stays fixed
        
        ✓ Simpler, no retriever training needed
        ✓ Works well with strong pre-trained retrievers
        ✗ Retriever not optimized for this specific generator
    
    Approach 2: Frozen Generator, Fine-Tune Retriever (REPLUG LSR)
    ──────────────────────────────────────────────────────────────
        1. Use fixed LLM as "reward model" for retriever
        2. Score retrieved docs by how much they help the LLM
        3. Train retriever to maximize LLM performance
        
        Loss: KL(P_retriever(z|x) ‖ P_LM(z|x,y))
        
        Aligns retriever distribution with what the LLM actually needs.
    
    Approach 3: Joint Training (RA-DIT, REALM)
    ──────────────────────────────────────────
        1. Train both retriever and generator simultaneously
        2. Generator loss flows back to retriever via REINFORCE
        3. Both systems co-adapt
        
        ∇_retriever = E_z[∇ log P(z|x) · R(x, z)]
        where R(x, z) = log P(y|x, z) (generator's score)
        
        ✓ Globally optimal (in theory)
        ✗ Unstable training, mode collapse risk
        ✗ Requires periodically re-indexing (expensive)
    
    Approach 4: Alternating Training
    ────────────────────────────────
        Phase 1: Fix retriever, train generator (several epochs)
        Phase 2: Fix generator, train retriever (several epochs)
        Phase 3: Repeat
        
        ✓ More stable than fully joint training
        ✓ Each component gets focused optimization
        ✗ May not converge to global optimum
        ✗ Expensive (multiple training phases)
    
    Practical Recommendation:
    ─────────────────────────
        Start with Approach 1 (RAFT: frozen retriever, fine-tune generator).
        This gives 80% of the benefit with 20% of the complexity.
        Only move to joint training if the retriever is the bottleneck.
        """)
    
    @staticmethod
    def retrieval_training_signal():
        """Theory of training the retriever using generator feedback."""
        print("\n" + "=" * 70)
        print("RETRIEVER TRAINING SIGNAL FROM GENERATOR")
        print("=" * 70)
        
        print("""
    How can the generator help train the retriever?
    
    Key Idea: A "good" document is one that helps the generator
    produce the correct answer. We can quantify this:
    
    Document Quality Score:
    ──────────────────────
        R(x, z) = log P_generator(y* | x, z) - log P_generator(y* | x)
                  ┬───────────────────────────   ┬────────────────────
                  Log-likelihood WITH document    Log-likelihood WITHOUT
        
        If R > 0: Document z HELPS the generator → good retrieval
        If R ≈ 0: Document z is irrelevant → neutral
        If R < 0: Document z HURTS the generator → bad retrieval
    
    This score can be used to:
    1. Filter training retrievals (keep only helpful docs)
    2. Re-rank retrieved documents at inference
    3. Train the retriever via distillation:
       P_retriever(z|x) should match the "ideal" distribution:
       P_ideal(z|x) ∝ exp(R(x, z))
    
    REPLUG LSR Loss:
    ────────────────
        L = KL(P_retriever(z|x) ‖ P_ideal(z|x))
        
        This directly trains the retriever to provide documents
        that maximize the generator's answer quality.
        """)


# =============================================================================
# SECTION 5: Self-Reflective Retrieval Theory (Self-RAG)
# =============================================================================

class SelfRAGTheory:
    """
    Self-RAG (Asai et al., 2023): Teaching LLMs to decide WHEN 
    to retrieve, WHAT to retrieve, and WHETHER retrieved content
    is actually useful.
    
    Key Innovation: Special reflection tokens that let the model
    reason about its own retrieval needs and the quality of
    retrieved content.
    """
    
    @staticmethod
    def demonstrate_self_rag():
        """Show the Self-RAG framework."""
        print("=" * 70)
        print("SELF-RAG: SELF-REFLECTIVE RETRIEVAL")
        print("=" * 70)
        
        print("""
    Standard RAG: Always retrieves, always uses what's retrieved.
    Self-RAG:     Learns WHEN to retrieve and WHETHER to use results.
    
    Reflection Tokens:
    ──────────────────
    
    1. [Retrieve]: Should I retrieve additional information?
       Values: {Yes, No, Continue}
       
       The model decides if it needs external knowledge:
       - Factual question → [Retrieve: Yes]
       - Creative writing → [Retrieve: No]
       - Already has context → [Retrieve: Continue]
    
    2. [IsRelevant]: Is the retrieved document relevant?
       Values: {Relevant, Partially, Irrelevant}
       
       After retrieval, model evaluates each document:
       - Contains the answer → [IsRelevant: Relevant]
       - Related but not helpful → [IsRelevant: Partially]
       - Off-topic → [IsRelevant: Irrelevant]
    
    3. [IsSupported]: Is my answer supported by the evidence?
       Values: {Fully, Partially, No}
       
       Self-checks if generated answer matches retrieved evidence:
       - Answer directly stated in docs → [IsSupported: Fully]
       - Answer inferred from docs → [IsSupported: Partially]
       - Answer not in docs → [IsSupported: No]
    
    4. [IsUseful]: Is my answer actually useful to the user?
       Values: {Useful, Somewhat, Not Useful}
    
    Example Self-RAG Generation:
    ────────────────────────────
    Query: "What year was the Eiffel Tower built?"
    
    Model: [Retrieve: Yes]
    → Retrieves documents about Eiffel Tower
    
    Model: [IsRelevant: Relevant] (for doc about Eiffel Tower history)
    
    Model: "The Eiffel Tower was built in 1889."
    
    Model: [IsSupported: Fully] (answer found in doc)
    Model: [IsUseful: Useful]
    
    Training Self-RAG:
    ──────────────────
    1. Use a critic model to label training data with reflection tokens
    2. Fine-tune the LLM on data with inline reflection tokens
    3. At inference, reflection tokens guide retrieval decisions
    4. Tree-based decoding: generate multiple candidates, select best
        """)
    
    @staticmethod
    def self_rag_training_pipeline():
        """Theory behind training a Self-RAG model."""
        print("\n" + "=" * 70)
        print("SELF-RAG TRAINING PIPELINE")
        print("=" * 70)
        
        print("""
    Phase 1: Critic Training
    ────────────────────────
        Train a separate critic model to generate reflection tokens:
        
        Input:  (query, document, response)
        Output: (retrieve_needed, is_relevant, is_supported, is_useful)
        
        Use GPT-4 or human annotations to create critic training data.
        Train a smaller model (e.g., LLaMA-7B) as the critic.
    
    Phase 2: Training Data Augmentation
    ────────────────────────────────────
        Use the critic to annotate the full training corpus:
        
        For each (query, response) pair:
        1. Run critic: Should retrieval have been used?
        2. If yes, retrieve documents
        3. Run critic: Are retrieved docs relevant?
        4. Run critic: Is response supported by docs?
        5. Insert reflection tokens into training data:
           
           "Q: {query} [Retrieve: Yes] [Context: {doc}]
            [IsRelevant: Relevant] {response}
            [IsSupported: Fully] [IsUseful: Useful]"
    
    Phase 3: Fine-Tuning
    ────────────────────
        Fine-tune the LLM on augmented data with reflection tokens.
        The model learns to:
        - Generate reflection tokens as part of output
        - Use them to guide retrieval decisions
        - Self-assess answer quality
    
    Phase 4: Inference with Tree Decoding
    ─────────────────────────────────────
        1. Generate with beam search / sampling
        2. At each [Retrieve] token:
           - If "Yes": fetch documents, branch into multiple candidates
           - If "No": continue generating from parametric knowledge
        3. Score candidates using reflection token probabilities
        4. Select best candidate based on composite score:
           
           Score = P(IsRelevant=Rel) × P(IsSupported=Full) × P(IsUseful=Yes)
    
    Benefits over Standard RAG:
    ───────────────────────────
    ✓ Selective retrieval → faster (skips when not needed)
    ✓ Self-assessment → more reliable outputs
    ✓ Interpretable → reflection tokens show reasoning
    ✓ Adaptive → adjusts retrieval frequency per query
        """)
    
    @staticmethod
    def theoretical_comparison():
        """Compare RAG variants theoretically."""
        print("\n" + "=" * 70)
        print("THEORETICAL COMPARISON OF RAG VARIANTS")
        print("=" * 70)
        
        print("""
    ┌──────────────────┬──────────────┬───────────────┬────────────────┐
    │ Aspect           │ Standard RAG │ RAFT          │ Self-RAG       │
    ├──────────────────┼──────────────┼───────────────┼────────────────┤
    │ When to retrieve │ Always       │ Always        │ Model decides  │
    │ Retriever FT     │ No           │ No            │ No (optional)  │
    │ Generator FT     │ No/Standard  │ Yes (oracle+  │ Yes (with      │
    │                  │              │ distractor)   │ reflection)    │
    │ Noise handling   │ Implicit     │ Trained       │ Self-assessed  │
    │ Citing sources   │ No           │ CoT citation  │ Via tokens     │
    │ Inference cost   │ Retrieve +   │ Retrieve +    │ Selective      │
    │                  │ generate     │ generate      │ retrieve       │
    │ Training data    │ Standard     │ Q,A,Oracle,   │ Q,A + critic   │
    │                  │              │ Distractors   │ annotations    │
    │ Complexity       │ Low          │ Medium        │ High           │
    └──────────────────┴──────────────┴───────────────┴────────────────┘
    
    When to choose each:
    ─────────────────────
    Standard RAG:  Quick deployment, decent retriever, no FT budget
    RAFT:          Domain-specific QA, training data available, need accuracy
    Self-RAG:      Mixed query types, latency-sensitive, need reliability
        """)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 70)
    print("RETRIEVAL-AUGMENTED FINE-TUNING — THEORETICAL FOUNDATIONS")
    print("=" * 70)
    
    # Section 1: Knowledge Storage
    print("\n\n📖 SECTION 1: Parametric vs Non-Parametric Knowledge")
    KnowledgeStorageTheory.demonstrate_knowledge_types()
    KnowledgeStorageTheory.knowledge_capacity_analysis()
    
    # Section 2: RAG Theory
    print("\n\n📖 SECTION 2: RAG Theory")
    RAGTheory.demonstrate_rag_formulation()
    RAGTheory.demonstrate_retrieval_methods()
    RAGTheory.chunking_strategy_theory()
    
    # Section 3: RAFT
    print("\n\n📖 SECTION 3: RAFT Training Strategy")
    RAFTTheory.demonstrate_raft_training_strategy()
    RAFTTheory.raft_vs_standard_rag()
    RAFTTheory.distractor_design_theory()
    
    # Section 4: Joint Optimization
    print("\n\n📖 SECTION 4: Joint Retriever-Generator Optimization")
    JointOptimizationTheory.demonstrate_joint_training()
    JointOptimizationTheory.retrieval_training_signal()
    
    # Section 5: Self-RAG
    print("\n\n📖 SECTION 5: Self-RAG")
    SelfRAGTheory.demonstrate_self_rag()
    SelfRAGTheory.self_rag_training_pipeline()
    SelfRAGTheory.theoretical_comparison()
    
    print("\n" + "=" * 70)
    print("THEORY SUMMARY")
    print("=" * 70)
    print("""
    Key Takeaways:
    
    1. Hybrid memory (parametric + non-parametric) is superior
       → Models store reasoning in weights, facts in retrieval index
    
    2. RAG marginalizes over retrieved documents:
       P(y|x) = Σ_z P(y|x,z) · P(z|x)
    
    3. RAFT trains models to handle retrieval noise:
       → Oracle + distractor mixing teaches discrimination
       → Chain-of-thought improves citation and accuracy
    
    4. Joint retriever-generator training is powerful but complex:
       → Start with frozen retriever (RAFT-style)
       → Move to joint training only if retriever is bottleneck
    
    5. Self-RAG adds meta-reasoning about retrieval:
       → Selective retrieval (when needed)
       → Self-assessment (is the answer supported?)
       → Tree decoding for best-candidate selection
    """)


if __name__ == "__main__":
    main()
