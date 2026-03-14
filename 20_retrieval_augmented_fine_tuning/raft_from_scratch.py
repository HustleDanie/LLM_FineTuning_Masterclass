"""
Retrieval-Augmented Fine-Tuning - From Scratch Implementation
==============================================================

Pure PyTorch implementations of retrieval-augmented systems,
RAFT training data construction, dense retrieval, and
Self-RAG reflection mechanisms.

Sections:
    1. Dense Retriever (Dual Encoder)
    2. Document Store and Chunking
    3. RAFT Training Data Constructor
    4. Retrieval-Augmented Generator
    5. Self-RAG Reflection Mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
import math
import random
from dataclasses import dataclass, field
from collections import defaultdict


# =============================================================================
# SECTION 1: Dense Retriever (Dual Encoder)
# =============================================================================

class DualEncoder(nn.Module):
    """
    Dense retriever using dual encoder architecture (DPR-style).
    
    Architecture:
        Query Encoder:    q(x) → q_vec ∈ R^d
        Document Encoder: d(z) → d_vec ∈ R^d
        Score:           sim(q_vec, d_vec) = q_vec · d_vec
    
    The two encoders map queries and documents into the same
    embedding space. Relevant documents have high dot-product
    similarity with the query.
    
    Training: Contrastive learning with in-batch negatives.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 256,
        d_embedding: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        max_seq_len: int = 256,
        shared_encoder: bool = False
    ):
        super().__init__()
        
        self.d_embedding = d_embedding
        self.shared_encoder = shared_encoder
        
        # Token + position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Query encoder
        q_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, batch_first=True
        )
        self.query_encoder = nn.TransformerEncoder(q_layer, num_layers=n_layers)
        self.query_projection = nn.Linear(d_model, d_embedding)
        
        # Document encoder (separate or shared)
        if shared_encoder:
            self.doc_encoder = self.query_encoder
            self.doc_projection = self.query_projection
        else:
            d_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4, batch_first=True
            )
            self.doc_encoder = nn.TransformerEncoder(d_layer, num_layers=n_layers)
            self.doc_projection = nn.Linear(d_model, d_embedding)
        
        params = sum(p.numel() for p in self.parameters())
        print(f"  [DualEncoder] Parameters: {params:,}, d_embedding={d_embedding}")
        print(f"    Shared encoder: {shared_encoder}")
    
    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create token + position embeddings."""
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        return self.token_embedding(input_ids) + self.position_embedding(positions)
    
    def encode_query(self, query_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode query to dense vector.
        Returns: [batch, d_embedding]
        """
        x = self._embed(query_ids)
        hidden = self.query_encoder(x)
        # Mean pooling over sequence
        pooled = hidden.mean(dim=1)
        return F.normalize(self.query_projection(pooled), dim=-1)
    
    def encode_document(self, doc_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode document to dense vector.
        Returns: [batch, d_embedding]
        """
        x = self._embed(doc_ids)
        hidden = self.doc_encoder(x)
        pooled = hidden.mean(dim=1)
        return F.normalize(self.doc_projection(pooled), dim=-1)
    
    def forward(
        self,
        query_ids: torch.Tensor,
        doc_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity scores between queries and documents.
        
        Args:
            query_ids: [batch_q, seq_len_q]
            doc_ids: [batch_d, seq_len_d]
        
        Returns:
            scores: [batch_q, batch_d] similarity matrix
        """
        q_vecs = self.encode_query(query_ids)    # [batch_q, d_emb]
        d_vecs = self.encode_document(doc_ids)    # [batch_d, d_emb]
        
        # Dot product similarity
        scores = torch.matmul(q_vecs, d_vecs.T)  # [batch_q, batch_d]
        return scores
    
    def contrastive_loss(
        self,
        query_ids: torch.Tensor,
        pos_doc_ids: torch.Tensor,
        temperature: float = 0.05
    ) -> torch.Tensor:
        """
        In-batch contrastive loss for training the retriever.
        
        Each query's positive document is the diagonal element.
        All other documents in the batch serve as negatives.
        
        loss = -log(exp(sim(q, d+)/τ) / Σ exp(sim(q, d)/τ))
        """
        scores = self.forward(query_ids, pos_doc_ids) / temperature
        
        # Labels: diagonal elements are positives
        labels = torch.arange(scores.size(0), device=scores.device)
        
        loss = F.cross_entropy(scores, labels)
        return loss


class DocumentIndex:
    """
    Simple in-memory document index for nearest neighbor search.
    
    In production, use FAISS, Milvus, or similar vector databases.
    This implementation demonstrates the core concept.
    """
    
    def __init__(self, encoder: DualEncoder):
        self.encoder = encoder
        self.doc_embeddings: Optional[torch.Tensor] = None
        self.documents: List[Dict] = []
    
    @torch.no_grad()
    def index_documents(
        self,
        documents: List[Dict],
        batch_size: int = 32
    ):
        """
        Build the document index by encoding all documents.
        
        Args:
            documents: List of {"text": str, "doc_id": str, "token_ids": tensor}
        """
        self.documents = documents
        all_embeddings = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            doc_ids = torch.stack([d["token_ids"] for d in batch])
            embeddings = self.encoder.encode_document(doc_ids)
            all_embeddings.append(embeddings)
        
        self.doc_embeddings = torch.cat(all_embeddings, dim=0)
        print(f"  Indexed {len(documents)} documents ({self.doc_embeddings.shape})")
    
    @torch.no_grad()
    def search(
        self,
        query_ids: torch.Tensor,
        top_k: int = 5
    ) -> List[List[Tuple[int, float]]]:
        """
        Retrieve top-k documents for each query.
        
        Returns: List of [(doc_idx, score)] for each query
        """
        query_vecs = self.encoder.encode_query(query_ids)  # [batch, d_emb]
        
        # Compute similarities
        scores = torch.matmul(query_vecs, self.doc_embeddings.T)  # [batch, num_docs]
        
        # Get top-k
        top_scores, top_indices = scores.topk(top_k, dim=1)
        
        results = []
        for batch_idx in range(query_ids.size(0)):
            query_results = []
            for k in range(top_k):
                doc_idx = top_indices[batch_idx, k].item()
                score = top_scores[batch_idx, k].item()
                query_results.append((doc_idx, score))
            results.append(query_results)
        
        return results


def demonstrate_dense_retriever():
    """Demonstrate the dual encoder retriever."""
    print("=" * 60)
    print("DENSE RETRIEVER DEMONSTRATION")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    retriever = DualEncoder(
        vocab_size=5000,
        d_model=128,
        d_embedding=64,
        n_layers=1,
        n_heads=4
    )
    
    # Synthetic documents
    num_docs = 50
    doc_seq_len = 64
    documents = [
        {
            "text": f"Document {i} about topic {i % 5}",
            "doc_id": f"doc_{i}",
            "token_ids": torch.randint(0, 5000, (doc_seq_len,))
        }
        for i in range(num_docs)
    ]
    
    # Build index
    index = DocumentIndex(retriever)
    index.index_documents(documents)
    
    # Search
    query = torch.randint(0, 5000, (2, 32))  # 2 queries
    results = index.search(query, top_k=3)
    
    print(f"\n  Query 1 top-3 results:")
    for doc_idx, score in results[0]:
        print(f"    Doc {doc_idx}: score={score:.4f}")
    
    print(f"\n  Query 2 top-3 results:")
    for doc_idx, score in results[1]:
        print(f"    Doc {doc_idx}: score={score:.4f}")
    
    # Training step
    pos_docs = torch.randint(0, 5000, (2, doc_seq_len))
    loss = retriever.contrastive_loss(query, pos_docs)
    print(f"\n  Contrastive training loss: {loss.item():.4f}")


# =============================================================================
# SECTION 2: Document Store and Chunking
# =============================================================================

class DocumentChunker:
    """
    Document chunking implementations for building retrieval indices.
    
    Transforms long documents into retrievable chunks that:
    1. Fit within model context windows
    2. Contain self-contained information
    3. Preserve enough context for understanding
    """
    
    @staticmethod
    def fixed_size_chunking(
        text: str,
        chunk_size: int = 256,
        overlap: int = 50
    ) -> List[Dict]:
        """
        Split text into fixed-size overlapping chunks.
        
        Args:
            text: Full document text
            chunk_size: Target chunk size in characters
            overlap: Number of overlapping characters between chunks
        """
        chunks = []
        words = text.split()
        
        # Approximate word-based chunking
        words_per_chunk = chunk_size // 5  # ~5 chars per word
        overlap_words = overlap // 5
        stride = max(words_per_chunk - overlap_words, 1)
        
        for i in range(0, len(words), stride):
            chunk_words = words[i:i + words_per_chunk]
            if len(chunk_words) < words_per_chunk // 4:
                break  # Skip tiny final chunks
            
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "text": chunk_text,
                "start_idx": i,
                "end_idx": min(i + words_per_chunk, len(words)),
                "chunk_type": "fixed_overlap"
            })
        
        return chunks
    
    @staticmethod
    def semantic_chunking(
        text: str,
        delimiters: List[str] = None
    ) -> List[Dict]:
        """
        Split text at semantic boundaries (paragraphs, sections).
        
        This preserves the natural structure of the document.
        """
        if delimiters is None:
            delimiters = ["\n\n", "\n# ", "\n## ", "\n### "]
        
        chunks = []
        # Split by double newline (paragraph boundaries)
        paragraphs = text.split("\n\n")
        
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) < 20:  # Skip very short paragraphs
                continue
            
            chunks.append({
                "text": para,
                "paragraph_idx": i,
                "chunk_type": "semantic"
            })
        
        return chunks
    
    @staticmethod
    def recursive_chunking(
        text: str,
        max_chunk_size: int = 512,
        min_chunk_size: int = 100
    ) -> List[Dict]:
        """
        Recursively split text: first by sections, then paragraphs,
        then sentences, until chunks are within size limits.
        """
        # Level 1: Try section splits
        separators = ["\n\n\n", "\n\n", "\n", ". "]
        
        chunks = []
        
        def split_recursive(text_piece, level=0):
            if len(text_piece) <= max_chunk_size:
                if len(text_piece) >= min_chunk_size:
                    chunks.append({"text": text_piece.strip(), "chunk_type": "recursive"})
                return
            
            if level >= len(separators):
                # Force split at max_chunk_size
                chunks.append({
                    "text": text_piece[:max_chunk_size].strip(),
                    "chunk_type": "recursive_forced"
                })
                if len(text_piece) > max_chunk_size:
                    split_recursive(text_piece[max_chunk_size:], level)
                return
            
            parts = text_piece.split(separators[level])
            for part in parts:
                if part.strip():
                    split_recursive(part, level + 1)
        
        split_recursive(text)
        return chunks


def demonstrate_chunking():
    """Demonstrate different chunking strategies."""
    print("\n" + "=" * 60)
    print("DOCUMENT CHUNKING DEMONSTRATION")
    print("=" * 60)
    
    # Sample document
    document = """
    Introduction to Machine Learning

    Machine learning is a subset of artificial intelligence that enables systems
    to learn and improve from experience. It focuses on developing programs that
    can access data and use it to learn for themselves.

    Supervised Learning

    In supervised learning, algorithms learn from labeled training data. The algorithm
    makes predictions on data and is corrected when those predictions are wrong.
    Common examples include classification and regression tasks. Popular algorithms
    include linear regression, decision trees, and neural networks.

    Unsupervised Learning

    Unsupervised learning algorithms learn from unlabeled data. They discover hidden
    patterns and structures in the data without explicit guidance. Clustering and
    dimensionality reduction are key tasks. K-means, DBSCAN, and PCA are widely used.

    Deep Learning

    Deep learning uses neural networks with many layers to learn representations
    of data. It has achieved breakthrough results in image recognition, natural
    language processing, and game playing. Convolutional neural networks handle
    images while recurrent networks process sequences.
    """
    
    chunker = DocumentChunker()
    
    # Fixed-size chunking
    fixed_chunks = chunker.fixed_size_chunking(document, chunk_size=200, overlap=40)
    print(f"\n  Fixed-size chunks (200 chars, 40 overlap): {len(fixed_chunks)} chunks")
    for i, chunk in enumerate(fixed_chunks[:3]):
        print(f"    Chunk {i}: '{chunk['text'][:60]}...' ({len(chunk['text'])} chars)")
    
    # Semantic chunking
    semantic_chunks = chunker.semantic_chunking(document)
    print(f"\n  Semantic chunks: {len(semantic_chunks)} chunks")
    for i, chunk in enumerate(semantic_chunks[:3]):
        print(f"    Chunk {i}: '{chunk['text'][:60]}...' ({len(chunk['text'])} chars)")
    
    # Recursive chunking
    recursive_chunks = chunker.recursive_chunking(document, max_chunk_size=300)
    print(f"\n  Recursive chunks: {len(recursive_chunks)} chunks")
    for i, chunk in enumerate(recursive_chunks[:3]):
        print(f"    Chunk {i}: '{chunk['text'][:60]}...' ({len(chunk['text'])} chars)")


# =============================================================================
# SECTION 3: RAFT Training Data Constructor
# =============================================================================

@dataclass
class RAFTExample:
    """A single RAFT training example."""
    question: str
    answer: str
    oracle_doc: str
    distractor_docs: List[str]
    example_type: str  # "oracle_with_distractors", "oracle_only", "distractors_only"
    chain_of_thought: Optional[str] = None


class RAFTDataConstructor:
    """
    Constructs RAFT training data from a QA dataset and document corpus.
    
    The key innovation: mixing oracle documents with distractors
    at specific ratios to teach the model to:
    1. Find relevant info among noise (60% oracle + distractors)
    2. Use good context when available (20% oracle only)
    3. Rely on parametric knowledge (20% distractors only)
    """
    
    def __init__(
        self,
        oracle_with_distractor_ratio: float = 0.6,
        oracle_only_ratio: float = 0.2,
        distractor_only_ratio: float = 0.2,
        num_distractors: int = 3
    ):
        assert abs(oracle_with_distractor_ratio + oracle_only_ratio + 
                   distractor_only_ratio - 1.0) < 1e-6
        
        self.oracle_distractor_ratio = oracle_with_distractor_ratio
        self.oracle_only_ratio = oracle_only_ratio
        self.distractor_only_ratio = distractor_only_ratio
        self.num_distractors = num_distractors
        
        print(f"  RAFT Data Constructor:")
        print(f"    Oracle + Distractors: {oracle_with_distractor_ratio:.0%}")
        print(f"    Oracle only:          {oracle_only_ratio:.0%}")
        print(f"    Distractors only:     {distractor_only_ratio:.0%}")
        print(f"    Num distractors:      {num_distractors}")
    
    def construct_examples(
        self,
        qa_pairs: List[Dict],
        corpus: List[str],
        seed: int = 42
    ) -> List[RAFTExample]:
        """
        Construct RAFT training examples from QA pairs.
        
        Args:
            qa_pairs: List of {"question": str, "answer": str, "oracle_doc": str}
            corpus: List of all documents (for sampling distractors)
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        examples = []
        
        for qa in qa_pairs:
            question = qa["question"]
            answer = qa["answer"]
            oracle = qa["oracle_doc"]
            
            # Sample distractors (documents that are NOT the oracle)
            available_distractors = [d for d in corpus if d != oracle]
            distractors = random.sample(
                available_distractors,
                min(self.num_distractors, len(available_distractors))
            )
            
            # Determine example type based on ratios
            r = random.random()
            
            if r < self.oracle_distractor_ratio:
                # Type 1: Oracle + Distractors
                # Shuffle oracle position among distractors
                all_docs = distractors + [oracle]
                random.shuffle(all_docs)
                
                cot = f"Looking at the provided documents, the relevant " \
                      f"information is found in the passage that states: " \
                      f"'{oracle[:80]}...'. Based on this, the answer is: {answer}"
                
                examples.append(RAFTExample(
                    question=question,
                    answer=answer,
                    oracle_doc=oracle,
                    distractor_docs=distractors,
                    example_type="oracle_with_distractors",
                    chain_of_thought=cot
                ))
            
            elif r < self.oracle_distractor_ratio + self.oracle_only_ratio:
                # Type 2: Oracle Only
                examples.append(RAFTExample(
                    question=question,
                    answer=answer,
                    oracle_doc=oracle,
                    distractor_docs=[],
                    example_type="oracle_only",
                    chain_of_thought=f"The provided document contains the answer: {answer}"
                ))
            
            else:
                # Type 3: Distractors Only (no oracle!)
                examples.append(RAFTExample(
                    question=question,
                    answer=answer,
                    oracle_doc="",  # No oracle provided
                    distractor_docs=distractors,
                    example_type="distractors_only",
                    chain_of_thought=f"The provided documents don't contain "
                                     f"the answer. From general knowledge: {answer}"
                ))
        
        # Print statistics
        type_counts = defaultdict(int)
        for ex in examples:
            type_counts[ex.example_type] += 1
        
        print(f"\n  Constructed {len(examples)} RAFT examples:")
        for etype, count in type_counts.items():
            print(f"    {etype}: {count} ({count/len(examples)*100:.1f}%)")
        
        return examples
    
    def format_for_training(
        self,
        example: RAFTExample,
        include_cot: bool = True
    ) -> Dict[str, str]:
        """
        Format a RAFT example into text for LLM training.
        
        Returns:
            {"prompt": str, "completion": str}
        """
        # Build context from documents
        docs = []
        if example.oracle_doc:
            docs.append(example.oracle_doc)
        docs.extend(example.distractor_docs)
        random.shuffle(docs)
        
        context = ""
        for i, doc in enumerate(docs):
            context += f"\n[Document {i+1}]: {doc}\n"
        
        prompt = f"### Question: {example.question}\n"
        if context:
            prompt += f"\n### Context:{context}\n"
        prompt += "\n### Answer:"
        
        if include_cot and example.chain_of_thought:
            completion = f" {example.chain_of_thought}"
        else:
            completion = f" {example.answer}"
        
        return {"prompt": prompt, "completion": completion}


def demonstrate_raft_constructor():
    """Demonstrate RAFT training data construction."""
    print("\n" + "=" * 60)
    print("RAFT TRAINING DATA CONSTRUCTION")
    print("=" * 60)
    
    constructor = RAFTDataConstructor(
        oracle_with_distractor_ratio=0.6,
        oracle_only_ratio=0.2,
        distractor_only_ratio=0.2,
        num_distractors=3
    )
    
    # Sample QA pairs
    qa_pairs = [
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "oracle_doc": "Paris is the capital and most populous city of France."
        },
        {
            "question": "When was Python created?",
            "answer": "1991",
            "oracle_doc": "Python was created by Guido van Rossum and first released in 1991."
        },
        {
            "question": "What is the speed of light?",
            "answer": "approximately 299,792 km/s",
            "oracle_doc": "Light travels at approximately 299,792 kilometers per second in vacuum."
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "answer": "William Shakespeare",
            "oracle_doc": "Romeo and Juliet is a tragedy written by William Shakespeare around 1597."
        },
        {
            "question": "What is the atomic number of carbon?",
            "answer": "6",
            "oracle_doc": "Carbon has atomic number 6 and is a nonmetallic element."
        },
    ]
    
    corpus = [
        "Paris is the capital and most populous city of France.",
        "London is the capital of the United Kingdom.",
        "Berlin is the capital of Germany.",
        "Python was created by Guido van Rossum and first released in 1991.",
        "Java was released by Sun Microsystems in 1995.",
        "Light travels at approximately 299,792 kilometers per second in vacuum.",
        "Sound travels at 343 meters per second in air.",
        "Romeo and Juliet is a tragedy written by William Shakespeare around 1597.",
        "Carbon has atomic number 6 and is a nonmetallic element.",
        "The Earth orbits the Sun at an average distance of 150 million km.",
    ]
    
    examples = constructor.construct_examples(qa_pairs, corpus)
    
    # Show formatted examples
    print("\n  Formatted RAFT examples:")
    for i, ex in enumerate(examples[:3]):
        formatted = constructor.format_for_training(ex, include_cot=True)
        print(f"\n  --- Example {i+1} ({ex.example_type}) ---")
        print(f"  Prompt: {formatted['prompt'][:150]}...")
        print(f"  Completion: {formatted['completion'][:100]}...")


# =============================================================================
# SECTION 4: Retrieval-Augmented Generator
# =============================================================================

class RetrievalAugmentedGenerator(nn.Module):
    """
    Full retrieval-augmented generator that combines:
    1. Dense retrieval for finding relevant documents
    2. Context prepending for augmenting the LM input
    3. Answer generation conditioned on query + retrieved context
    
    This implements the RAG-Sequence approach where retrieved
    documents are prepended to the input.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        max_seq_len: int = 512
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # LM components
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Layer norm
        self.ln = nn.LayerNorm(d_model)
        
        params = sum(p.numel() for p in self.parameters())
        print(f"  [RAGenerator] Parameters: {params:,}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass over context + query sequence.
        
        The input should be formatted as:
            [context_tokens] [SEP] [query_tokens] [SEP] [answer_tokens]
        
        This is the standard approach: prepend retrieved context
        to the input sequence and let the model attend to it.
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.ln(x)
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1
        ).bool()
        
        # Decode (self-attention with causal mask)
        # Using decoder as autoregressive model
        hidden = self.decoder(x, x, tgt_mask=causal_mask)
        logits = self.lm_head(hidden)
        
        result = {"logits": logits}
        
        if labels is not None:
            # Compute loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            result["loss"] = loss
        
        return result
    
    @staticmethod
    def prepare_rag_input(
        query_tokens: torch.Tensor,
        doc_tokens_list: List[torch.Tensor],
        sep_token_id: int = 2,
        max_length: int = 512
    ) -> torch.Tensor:
        """
        Prepare RAG input by concatenating retrieved docs + query.
        
        Format: [DOC_1] [SEP] [DOC_2] [SEP] ... [DOC_K] [SEP] [QUERY]
        
        This is the standard way to feed retrieved context to an LM.
        """
        sep = torch.tensor([sep_token_id])
        
        # Concatenate: docs + separators + query
        parts = []
        for doc_tokens in doc_tokens_list:
            parts.append(doc_tokens)
            parts.append(sep)
        parts.append(query_tokens)
        
        combined = torch.cat(parts)
        
        # Truncate from the left (keep query, trim context)
        if combined.size(0) > max_length:
            combined = combined[-max_length:]
        
        return combined


def demonstrate_rag_generator():
    """Demonstrate the retrieval-augmented generator."""
    print("\n" + "=" * 60)
    print("RETRIEVAL-AUGMENTED GENERATOR")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    generator = RetrievalAugmentedGenerator(
        vocab_size=5000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_seq_len=256
    )
    
    # Simulate retrieval + generation
    query_tokens = torch.randint(0, 5000, (32,))
    doc1_tokens = torch.randint(0, 5000, (64,))
    doc2_tokens = torch.randint(0, 5000, (64,))
    
    # Prepare RAG input
    rag_input = RetrievalAugmentedGenerator.prepare_rag_input(
        query_tokens,
        [doc1_tokens, doc2_tokens],
        sep_token_id=2,
        max_length=256
    )
    
    print(f"\n  Query tokens: {query_tokens.shape}")
    print(f"  Doc 1 tokens: {doc1_tokens.shape}")
    print(f"  Doc 2 tokens: {doc2_tokens.shape}")
    print(f"  RAG input (concatenated): {rag_input.shape}")
    
    # Forward pass
    rag_input_batch = rag_input.unsqueeze(0)  # Add batch dim
    labels = rag_input_batch.clone()
    
    outputs = generator(rag_input_batch, labels=labels)
    print(f"\n  Output logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")


# =============================================================================
# SECTION 5: Self-RAG Reflection Mechanism
# =============================================================================

class SelfRAGReflector(nn.Module):
    """
    Self-RAG reflection mechanism implementation.
    
    Adds special reflection tokens that the model generates
    to reason about retrieval quality and answer reliability.
    
    Reflection tokens:
    - [RETRIEVE]: Should I retrieve? {Yes, No, Continue}
    - [RELEVANT]: Is this doc relevant? {Relevant, Partially, Irrelevant}
    - [SUPPORTED]: Is answer supported? {Fully, Partially, No}
    - [USEFUL]: Is answer useful? {Useful, Somewhat, Not}
    """
    
    def __init__(self, d_model: int = 256):
        super().__init__()
        
        # Reflection classifiers
        self.retrieve_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3)  # Yes, No, Continue
        )
        
        self.relevance_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3)  # Relevant, Partially, Irrelevant
        )
        
        self.support_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3)  # Fully, Partially, No
        )
        
        self.usefulness_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3)  # Useful, Somewhat, Not
        )
        
        self.reflection_names = {
            "retrieve": ["Yes", "No", "Continue"],
            "relevant": ["Relevant", "Partially", "Irrelevant"],
            "supported": ["Fully", "Partially", "No"],
            "useful": ["Useful", "Somewhat", "Not"]
        }
    
    def forward(
        self,
        hidden_state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Generate all reflection scores from hidden state.
        
        Args:
            hidden_state: [batch, d_model] from LM's last hidden state
        
        Returns:
            Dict of reflection logits
        """
        return {
            "retrieve": self.retrieve_classifier(hidden_state),
            "relevant": self.relevance_classifier(hidden_state),
            "supported": self.support_classifier(hidden_state),
            "useful": self.usefulness_classifier(hidden_state),
        }
    
    def get_reflection_labels(
        self,
        reflection_logits: Dict[str, torch.Tensor]
    ) -> Dict[str, str]:
        """Convert logits to human-readable reflection labels."""
        labels = {}
        for key, logits in reflection_logits.items():
            idx = logits.argmax(dim=-1).item()
            labels[key] = self.reflection_names[key][idx]
        return labels
    
    def compute_composite_score(
        self,
        reflection_logits: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute composite quality score from reflection tokens.
        
        Score = P(Relevant) × P(Supported=Fully) × P(Useful)
        
        Used for re-ranking candidates in tree decoding.
        """
        probs = {k: F.softmax(v, dim=-1) for k, v in reflection_logits.items()}
        
        # Score: probability of positive reflections
        score = (
            probs["relevant"][..., 0] *   # P(Relevant)
            probs["supported"][..., 0] *   # P(Fully supported)
            probs["useful"][..., 0]        # P(Useful)
        )
        
        return score.item()


def demonstrate_self_rag():
    """Demonstrate Self-RAG reflection mechanism."""
    print("\n" + "=" * 60)
    print("SELF-RAG REFLECTION DEMONSTRATION")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    reflector = SelfRAGReflector(d_model=128)
    
    # Simulate hidden states from different scenarios
    scenarios = {
        "Factual question (should retrieve)": torch.randn(1, 128) + 0.5,
        "Creative writing (no retrieval)": torch.randn(1, 128) - 0.3,
        "Well-supported answer": torch.randn(1, 128) + 0.8,
    }
    
    for scenario_name, hidden in scenarios.items():
        logits = reflector(hidden)
        labels = reflector.get_reflection_labels(logits)
        score = reflector.compute_composite_score(logits)
        
        print(f"\n  Scenario: {scenario_name}")
        print(f"    Retrieve?: {labels['retrieve']}")
        print(f"    Relevant?: {labels['relevant']}")
        print(f"    Supported?: {labels['supported']}")
        print(f"    Useful?: {labels['useful']}")
        print(f"    Composite score: {score:.4f}")
    
    print("""
  In a full Self-RAG system:
    1. Model generates text, periodically emitting [RETRIEVE] tokens
    2. If [RETRIEVE: Yes], pause generation and fetch documents
    3. For each doc, model emits [RELEVANT] assessment
    4. After answering, model emits [SUPPORTED] and [USEFUL]
    5. Composite score used to select best candidate from beam search
    """)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 70)
    print("RETRIEVAL-AUGMENTED FINE-TUNING — FROM SCRATCH IMPLEMENTATION")
    print("=" * 70)
    
    # Section 1: Dense Retriever
    print("\n\n🔧 SECTION 1: Dense Retriever")
    demonstrate_dense_retriever()
    
    # Section 2: Document Chunking
    print("\n\n🔧 SECTION 2: Document Store and Chunking")
    demonstrate_chunking()
    
    # Section 3: RAFT Data Construction
    print("\n\n🔧 SECTION 3: RAFT Training Data Construction")
    demonstrate_raft_constructor()
    
    # Section 4: RAG Generator
    print("\n\n🔧 SECTION 4: Retrieval-Augmented Generator")
    demonstrate_rag_generator()
    
    # Section 5: Self-RAG
    print("\n\n🔧 SECTION 5: Self-RAG Reflection")
    demonstrate_self_rag()
    
    print("\n" + "=" * 70)
    print("FROM-SCRATCH IMPLEMENTATION COMPLETE")
    print("=" * 70)
    print("""
    Implemented:
    1. Dense Retriever — Dual encoder with contrastive training
    2. Document Chunking — Fixed, semantic, and recursive chunking
    3. RAFT Data Constructor — Oracle + distractor mixing with CoT
    4. RAG Generator — Context prepending + autoregressive generation
    5. Self-RAG — Reflection tokens for retrieval decision-making
    """)


if __name__ == "__main__":
    main()
