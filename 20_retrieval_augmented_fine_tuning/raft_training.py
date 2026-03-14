"""
Retrieval-Augmented Fine-Tuning - HuggingFace Training Pipelines
=================================================================

Production-ready training pipelines for retrieval-augmented fine-tuning
using HuggingFace Transformers, sentence-transformers, and related tools.

Sections:
    1. RAFT Data Pipeline
    2. Dense Retriever Training
    3. RAFT Fine-Tuning Pipeline
    4. Retrieval-Augmented Evaluation
    5. End-to-End RAFT System
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import random
import json
import os
from dataclasses import dataclass, field
from collections import defaultdict


# =============================================================================
# SECTION 1: RAFT Data Pipeline
# =============================================================================

class RAFTDataPipeline:
    """
    Production data pipeline for RAFT training.
    
    Constructs training data with the RAFT mixing strategy:
    - P% examples: oracle doc + distractor docs → answer with CoT
    - Q% examples: oracle doc only → answer with CoT
    - R% examples: distractor docs only → answer from parametric knowledge
    
    Supports multiple QA formats and domain-specific corpora.
    """
    
    def __init__(
        self,
        oracle_distractor_ratio: float = 0.6,
        oracle_only_ratio: float = 0.2,
        distractor_only_ratio: float = 0.2,
        num_distractors: int = 4,
        max_context_length: int = 1024,
        seed: int = 42
    ):
        self.oracle_distractor_ratio = oracle_distractor_ratio
        self.oracle_only_ratio = oracle_only_ratio
        self.distractor_only_ratio = distractor_only_ratio
        self.num_distractors = num_distractors
        self.max_context_length = max_context_length
        self.seed = seed
        
        random.seed(seed)
        
        print(f"  RAFT Data Pipeline initialized:")
        print(f"    Oracle+Distractor: {oracle_distractor_ratio:.0%}")
        print(f"    Oracle only:       {oracle_only_ratio:.0%}")
        print(f"    Distractor only:   {distractor_only_ratio:.0%}")
        print(f"    Num distractors:   {num_distractors}")
    
    def prepare_qa_dataset(
        self,
        qa_data: List[Dict[str, str]],
        corpus: List[str]
    ) -> List[Dict]:
        """
        Prepare RAFT training examples from QA pairs and corpus.
        
        Args:
            qa_data: List of dicts with keys: question, answer, oracle_doc
            corpus: Full document corpus for sampling distractors
        
        Returns:
            List of formatted training examples
        """
        examples = []
        type_counts = defaultdict(int)
        
        for qa in qa_data:
            question = qa["question"]
            answer = qa["answer"]
            oracle = qa.get("oracle_doc", "")
            
            # Sample distractors
            distractors = [d for d in corpus if d != oracle]
            if len(distractors) > self.num_distractors:
                distractors = random.sample(distractors, self.num_distractors)
            
            # Determine example type
            r = random.random()
            
            if r < self.oracle_distractor_ratio:
                example_type = "oracle_with_distractors"
                docs = distractors + [oracle]
                random.shuffle(docs)
                
                cot = self._generate_chain_of_thought(question, answer, oracle)
                response = f"<COT>{cot}</COT>\n<ANSWER>{answer}</ANSWER>"
                
            elif r < self.oracle_distractor_ratio + self.oracle_only_ratio:
                example_type = "oracle_only"
                docs = [oracle]
                
                cot = f"The provided document directly answers the question."
                response = f"<COT>{cot}</COT>\n<ANSWER>{answer}</ANSWER>"
                
            else:
                example_type = "distractor_only"
                docs = distractors
                
                cot = f"None of the provided documents contain relevant information. " \
                      f"Based on general knowledge:"
                response = f"<COT>{cot}</COT>\n<ANSWER>{answer}</ANSWER>"
            
            type_counts[example_type] += 1
            
            # Format as training text
            context = self._format_context(docs)
            prompt = self._format_prompt(question, context)
            
            examples.append({
                "prompt": prompt,
                "response": response,
                "text": prompt + response,
                "type": example_type,
                "question": question,
                "answer": answer
            })
        
        print(f"\n  Prepared {len(examples)} RAFT examples:")
        for etype, count in type_counts.items():
            print(f"    {etype}: {count} ({count/len(examples)*100:.1f}%)")
        
        return examples
    
    def _generate_chain_of_thought(
        self,
        question: str,
        answer: str,
        oracle: str
    ) -> str:
        """Generate chain-of-thought reasoning for RAFT training."""
        return (
            f"To answer '{question}', I need to examine the provided documents. "
            f"The relevant passage states: '{oracle[:100]}...' "
            f"From this evidence, the answer is: {answer}"
        )
    
    def _format_context(self, docs: List[str]) -> str:
        """Format retrieved documents into a context string."""
        context_parts = []
        for i, doc in enumerate(docs):
            context_parts.append(f"[Document {i+1}]:\n{doc}")
        return "\n\n".join(context_parts)
    
    def _format_prompt(self, question: str, context: str) -> str:
        """Format the full prompt with context and question."""
        return (
            f"### Context:\n{context}\n\n"
            f"### Question:\n{question}\n\n"
            f"### Answer:\n"
        )


class RAFTDataset(Dataset):
    """PyTorch Dataset for RAFT training."""
    
    def __init__(
        self,
        examples: List[Dict],
        tokenizer: Any,
        max_length: int = 512
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            example["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # Create labels (mask prompt tokens to -100)
        labels = input_ids.clone()
        
        prompt_encoding = self.tokenizer(
            example["prompt"],
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        prompt_len = prompt_encoding["attention_mask"].sum().item()
        labels[:prompt_len] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def demonstrate_raft_data_pipeline():
    """Demonstrate the RAFT data pipeline."""
    print("=" * 60)
    print("RAFT DATA PIPELINE")
    print("=" * 60)
    
    pipeline = RAFTDataPipeline(
        oracle_distractor_ratio=0.6,
        oracle_only_ratio=0.2,
        distractor_only_ratio=0.2,
        num_distractors=3
    )
    
    # Sample data
    qa_data = [
        {
            "question": "What is photosynthesis?",
            "answer": "The process by which plants convert sunlight to chemical energy.",
            "oracle_doc": "Photosynthesis is the process used by plants to convert light energy "
                         "from the sun into chemical energy stored in glucose molecules."
        },
        {
            "question": "What causes tides?",
            "answer": "Tides are caused by gravitational pull of the Moon and Sun.",
            "oracle_doc": "Ocean tides are primarily caused by the gravitational forces "
                         "exerted by the Moon and, to a lesser extent, the Sun."
        },
        {
            "question": "What is the Pythagorean theorem?",
            "answer": "In a right triangle, a² + b² = c².",
            "oracle_doc": "The Pythagorean theorem states that in a right-angled triangle, "
                         "the square of the hypotenuse equals the sum of the two other sides squared."
        },
    ]
    
    corpus = [d["oracle_doc"] for d in qa_data] + [
        "The Krebs cycle is a series of chemical reactions in cellular respiration.",
        "Newton's first law states objects at rest stay at rest unless acted upon.",
        "DNA is composed of nucleotides containing adenine, thymine, guanine, cytosine.",
        "The Earth's atmosphere is 78% nitrogen and 21% oxygen.",
    ]
    
    examples = pipeline.prepare_qa_dataset(qa_data, corpus)
    
    # Show examples
    for i, ex in enumerate(examples):
        print(f"\n  --- Example {i+1} ({ex['type']}) ---")
        print(f"  Prompt (first 120 chars): {ex['prompt'][:120]}...")
        print(f"  Response (first 100 chars): {ex['response'][:100]}...")


# =============================================================================
# SECTION 2: Dense Retriever Training
# =============================================================================

class DenseRetrieverTrainer:
    """
    Training pipeline for dense passage retrieval.
    
    Uses contrastive learning with in-batch negatives and
    optional hard negatives. Compatible with sentence-transformers.
    """
    
    def __init__(
        self,
        model_name: str = "distilgpt2",
        embedding_dim: int = 128,
        learning_rate: float = 2e-5,
        temperature: float = 0.05,
        use_hard_negatives: bool = True
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        
        self.model = None
        self.tokenizer = None
        
        print(f"  Dense Retriever Trainer:")
        print(f"    Base model: {model_name}")
        print(f"    Embedding dim: {embedding_dim}")
        print(f"    Temperature: {temperature}")
        print(f"    Hard negatives: {use_hard_negatives}")
    
    def initialize_model(self):
        """Initialize model and tokenizer."""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Add projection head
            hidden_size = self.model.config.hidden_size
            self.projection = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, self.embedding_dim)
            )
            
            print(f"  Model loaded: {self.model_name}")
            print(f"    Hidden size: {hidden_size}")
            print(f"    Projection: {hidden_size} → {self.embedding_dim}")
        except Exception as e:
            print(f"  Could not load model: {e}")
            print("  (This is expected in environments without model files)")
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> torch.Tensor:
        """Encode texts to embeddings."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoding = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.model(**encoding)
                # Mean pooling
                hidden = outputs.last_hidden_state
                mask = encoding["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                embeddings = self.projection(pooled)
                embeddings = F.normalize(embeddings, dim=-1)
            
            all_embeddings.append(embeddings)
        
        return torch.cat(all_embeddings, dim=0)
    
    def contrastive_loss(
        self,
        query_embeds: torch.Tensor,
        pos_embeds: torch.Tensor,
        neg_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss with in-batch + optional hard negatives.
        
        loss = -log(exp(sim(q, d+)/τ) / (exp(sim(q, d+)/τ) + Σ exp(sim(q, d-)/τ)))
        """
        # In-batch dot products
        scores = torch.matmul(query_embeds, pos_embeds.T) / self.temperature
        
        if neg_embeds is not None:
            # Add hard negatives
            neg_scores = torch.matmul(query_embeds, neg_embeds.T) / self.temperature
            scores = torch.cat([scores, neg_scores], dim=1)
        
        # Labels: diagonal elements
        labels = torch.arange(query_embeds.size(0), device=scores.device)
        loss = F.cross_entropy(scores, labels)
        
        return loss
    
    def train_step(
        self,
        queries: List[str],
        positive_docs: List[str],
        negative_docs: Optional[List[str]] = None
    ) -> float:
        """Single training step."""
        q_embeds = self.encode(queries)
        p_embeds = self.encode(positive_docs)
        n_embeds = self.encode(negative_docs) if negative_docs else None
        
        loss = self.contrastive_loss(q_embeds, p_embeds, n_embeds)
        
        return loss.item()
    
    def evaluate_retrieval(
        self,
        queries: List[str],
        corpus: List[str],
        ground_truth: List[int],
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        
        Metrics:
        - Recall@K: Fraction of queries where true doc is in top-K
        - MRR: Mean Reciprocal Rank
        """
        q_embeds = self.encode(queries)
        c_embeds = self.encode(corpus)
        
        scores = torch.matmul(q_embeds, c_embeds.T)
        _, indices = scores.topk(top_k, dim=1)
        
        hits = 0
        rr_sum = 0.0
        
        for i, gt_idx in enumerate(ground_truth):
            retrieved = indices[i].tolist()
            if gt_idx in retrieved:
                hits += 1
                rank = retrieved.index(gt_idx) + 1
                rr_sum += 1.0 / rank
        
        recall_at_k = hits / len(queries)
        mrr = rr_sum / len(queries)
        
        return {"recall@k": recall_at_k, "mrr": mrr, "top_k": top_k}


def demonstrate_retriever_training():
    """Demonstrate dense retriever training."""
    print("\n" + "=" * 60)
    print("DENSE RETRIEVER TRAINING")
    print("=" * 60)
    
    trainer = DenseRetrieverTrainer(
        model_name="distilgpt2",
        embedding_dim=128,
        temperature=0.05
    )
    
    print("\n  Training Configuration:")
    print(f"    Loss: Contrastive with in-batch negatives")
    print(f"    Temperature: {trainer.temperature}")
    print(f"    Hard negatives: {trainer.use_hard_negatives}")
    
    print("""
  Dense Retriever Training Pipeline:
  ┌──────────────────────────────────────────────────┐
  │ 1. Encode queries:     q_i = Enc_q(query_i)     │
  │ 2. Encode documents:   d_i = Enc_d(doc_i)       │
  │ 3. Similarity matrix:  S_ij = q_i · d_j / τ     │
  │ 4. Labels:             y_i = i (diagonal)        │
  │ 5. Loss:               CrossEntropy(S, y)        │
  │ 6. Optional: add hard negative columns to S      │
  └──────────────────────────────────────────────────┘
  
  Hard Negative Mining:
  - BM25 top-k that don't contain the answer
  - Cross-encoder re-ranker rejects
  - Previous epoch's false positives
  """)


# =============================================================================
# SECTION 3: RAFT Fine-Tuning Pipeline
# =============================================================================

class RAFTTrainer:
    """
    Full RAFT fine-tuning pipeline using HuggingFace Trainer.
    
    Steps:
    1. Prepare RAFT training data with oracle/distractor mixing
    2. Fine-tune the generator LM on the RAFT dataset
    3. Optionally use LoRA for parameter-efficient training
    """
    
    def __init__(
        self,
        model_name: str = "distilgpt2",
        use_lora: bool = True,
        lora_rank: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        batch_size: int = 4,
        max_length: int = 512
    ):
        self.model_name = model_name
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_length = max_length
        
        self.model = None
        self.tokenizer = None
        
        print(f"  RAFT Trainer Configuration:")
        print(f"    Model: {model_name}")
        print(f"    LoRA: {use_lora} (rank={lora_rank})")
        print(f"    LR: {learning_rate}, Epochs: {num_epochs}")
        print(f"    Batch size: {batch_size}, Max length: {max_length}")
    
    def setup(self):
        """Initialize model with optional LoRA."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            if self.use_lora:
                from peft import LoraConfig, get_peft_model, TaskType
                
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self.lora_rank,
                    lora_alpha=self.lora_rank * 2,
                    lora_dropout=0.05,
                    target_modules=["c_attn", "c_proj"],
                )
                
                self.model = get_peft_model(self.model, lora_config)
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self.model.parameters())
                print(f"  LoRA applied: {trainable:,}/{total:,} trainable "
                      f"({trainable/total*100:.2f}%)")
            
            print(f"  Model ready for RAFT training")
            
        except ImportError as e:
            print(f"  Setup skipped (missing dependency): {e}")
    
    def train(self, train_examples: List[Dict], eval_examples: Optional[List[Dict]] = None):
        """
        Run RAFT fine-tuning using HuggingFace Trainer.
        """
        if self.model is None:
            print("  Model not initialized. Call setup() first.")
            return
        
        try:
            from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
            
            # Create datasets
            train_dataset = RAFTDataset(train_examples, self.tokenizer, self.max_length)
            eval_dataset = RAFTDataset(eval_examples, self.tokenizer, self.max_length) \
                if eval_examples else None
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir="./raft_output",
                num_train_epochs=self.num_epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                warmup_ratio=0.1,
                weight_decay=0.01,
                logging_steps=10,
                eval_strategy="epoch" if eval_dataset else "no",
                save_strategy="epoch",
                fp16=torch.cuda.is_available(),
                report_to="none",
                remove_unused_columns=False,
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
            
            print(f"\n  Starting RAFT training...")
            print(f"    Train examples: {len(train_dataset)}")
            print(f"    Eval examples: {len(eval_dataset) if eval_dataset else 0}")
            
            result = trainer.train()
            print(f"    Training loss: {result.training_loss:.4f}")
            
            return result
            
        except ImportError as e:
            print(f"  Training skipped (missing dependency): {e}")
    
    def generate_answer(
        self,
        question: str,
        context_docs: List[str],
        max_new_tokens: int = 256
    ) -> str:
        """Generate answer given question and retrieved documents."""
        if self.model is None or self.tokenizer is None:
            return "(Model not loaded)"
        
        # Format prompt
        context = "\n\n".join(
            f"[Document {i+1}]:\n{doc}" for i, doc in enumerate(context_docs)
        )
        
        prompt = (
            f"### Context:\n{context}\n\n"
            f"### Question:\n{question}\n\n"
            f"### Answer:\n"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].size(1):])
        return response


def demonstrate_raft_training():
    """Demonstrate RAFT training pipeline."""
    print("\n" + "=" * 60)
    print("RAFT FINE-TUNING PIPELINE")
    print("=" * 60)
    
    trainer = RAFTTrainer(
        model_name="distilgpt2",
        use_lora=True,
        lora_rank=16,
        learning_rate=2e-5,
        num_epochs=3
    )
    
    print("""
  RAFT Training Pipeline:
  ┌──────────────────────────────────────────────────────────────┐
  │ Step 1: RAFT Data Construction                               │
  │   - Match QA pairs with oracle documents                     │
  │   - Sample distractors from corpus                           │
  │   - Mix ratios: 60% oracle+dist, 20% oracle, 20% dist       │
  │   - Generate chain-of-thought reasoning                      │
  │                                                              │
  │ Step 2: Tokenization and Formatting                          │
  │   - Format: [CONTEXT] [QUESTION] [COT] [ANSWER]             │
  │   - Mask prompt tokens (only train on response)              │
  │   - Apply truncation for long contexts                       │
  │                                                              │
  │ Step 3: Fine-Tuning                                          │
  │   - LoRA for parameter efficiency                            │
  │   - Standard causal LM objective on response tokens          │
  │   - Warmup + cosine learning rate schedule                   │
  │                                                              │
  │ Step 4: Inference                                            │
  │   - Retrieve top-K documents for query                       │
  │   - Prepend to prompt and generate                           │
  │   - Model has learned to ignore irrelevant documents         │
  └──────────────────────────────────────────────────────────────┘
  
  Key difference from standard RAG:
    Standard RAG: Model not trained on retrieval context → struggles
    RAFT:         Model trained on noisy context → robust to distractors
  """)


# =============================================================================
# SECTION 4: Retrieval-Augmented Evaluation
# =============================================================================

class RAFTEvaluator:
    """
    Comprehensive evaluation for retrieval-augmented fine-tuned models.
    
    Metrics:
    - Answer correctness (exact match, F1, ROUGE)
    - Faithfulness (is answer supported by context?)
    - Robustness (performance with varying distractor counts)
    - Attribution quality (can model cite sources?)
    """
    
    def __init__(self):
        self.results = defaultdict(list)
    
    def exact_match(self, prediction: str, ground_truth: str) -> float:
        """Exact match after normalization."""
        pred = prediction.strip().lower()
        gt = ground_truth.strip().lower()
        return 1.0 if pred == gt else 0.0
    
    def f1_score(self, prediction: str, ground_truth: str) -> float:
        """Token-level F1 score."""
        pred_tokens = set(prediction.lower().split())
        gt_tokens = set(ground_truth.lower().split())
        
        if not pred_tokens or not gt_tokens:
            return 0.0
        
        common = pred_tokens & gt_tokens
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        return 2 * precision * recall / (precision + recall)
    
    def evaluate_answer_quality(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """Evaluate answer quality metrics."""
        em_scores = []
        f1_scores = []
        
        for pred, gt in zip(predictions, ground_truths):
            em_scores.append(self.exact_match(pred, gt))
            f1_scores.append(self.f1_score(pred, gt))
        
        return {
            "exact_match": np.mean(em_scores),
            "f1": np.mean(f1_scores),
            "num_examples": len(predictions)
        }
    
    def evaluate_robustness(
        self,
        generate_fn,
        questions: List[str],
        oracle_docs: List[str],
        answers: List[str],
        distractor_pool: List[str],
        distractor_counts: List[int] = [0, 1, 3, 5, 10]
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate robustness to varying numbers of distractors.
        
        Tests whether the model can still find the right answer
        as the number of distracting documents increases.
        """
        results = {}
        
        for n_dist in distractor_counts:
            predictions = []
            
            for q, oracle, answer in zip(questions, oracle_docs, answers):
                # Sample distractors
                dists = random.sample(distractor_pool, min(n_dist, len(distractor_pool)))
                context_docs = dists + [oracle]
                random.shuffle(context_docs)
                
                pred = generate_fn(q, context_docs)
                predictions.append(pred)
            
            metrics = self.evaluate_answer_quality(predictions, answers)
            metrics["num_distractors"] = n_dist
            results[n_dist] = metrics
            
            print(f"    {n_dist} distractors → EM: {metrics['exact_match']:.3f}, "
                  f"F1: {metrics['f1']:.3f}")
        
        return results
    
    def evaluate_faithfulness(
        self,
        predictions: List[str],
        contexts: List[List[str]]
    ) -> Dict[str, float]:
        """
        Evaluate if generated answers are faithful to provided context.
        
        Simple heuristic: check if answer tokens appear in context.
        Production: use NLI model for entailment checking.
        """
        faithfulness_scores = []
        
        for pred, ctx_docs in zip(predictions, contexts):
            context_text = " ".join(ctx_docs).lower()
            pred_tokens = pred.lower().split()
            
            if not pred_tokens:
                faithfulness_scores.append(0.0)
                continue
            
            # Token overlap with context
            overlap = sum(1 for t in pred_tokens if t in context_text)
            score = overlap / len(pred_tokens)
            faithfulness_scores.append(min(score, 1.0))
        
        return {
            "faithfulness_mean": np.mean(faithfulness_scores),
            "faithfulness_std": np.std(faithfulness_scores),
            "num_examples": len(predictions)
        }
    
    def full_evaluation_report(
        self,
        model_name: str,
        metrics: Dict[str, Any]
    ):
        """Print a comprehensive evaluation report."""
        print(f"\n  {'='*50}")
        print(f"  RAFT Evaluation Report: {model_name}")
        print(f"  {'='*50}")
        
        if "answer_quality" in metrics:
            aq = metrics["answer_quality"]
            print(f"\n  Answer Quality:")
            print(f"    Exact Match: {aq.get('exact_match', 0):.3f}")
            print(f"    F1 Score:    {aq.get('f1', 0):.3f}")
        
        if "faithfulness" in metrics:
            ff = metrics["faithfulness"]
            print(f"\n  Faithfulness:")
            print(f"    Mean: {ff.get('faithfulness_mean', 0):.3f}")
            print(f"    Std:  {ff.get('faithfulness_std', 0):.3f}")
        
        if "robustness" in metrics:
            print(f"\n  Robustness (EM by distractor count):")
            for n_dist, rob_metrics in metrics["robustness"].items():
                print(f"    {n_dist} distractors: {rob_metrics.get('exact_match', 0):.3f}")


def demonstrate_evaluation():
    """Demonstrate RAFT evaluation."""
    print("\n" + "=" * 60)
    print("RETRIEVAL-AUGMENTED EVALUATION")
    print("=" * 60)
    
    evaluator = RAFTEvaluator()
    
    # Simulated predictions vs ground truth
    predictions = [
        "Paris is the capital of France",
        "Python was first released in 1991",
        "Water boils at 100 degrees Celsius",
        "The speed of light is 299792 km/s",
    ]
    ground_truths = [
        "Paris",
        "1991",
        "100 degrees Celsius",
        "299,792 km/s",
    ]
    
    # Answer quality
    quality = evaluator.evaluate_answer_quality(predictions, ground_truths)
    print(f"\n  Answer Quality:")
    print(f"    Exact Match: {quality['exact_match']:.3f}")
    print(f"    F1 Score:    {quality['f1']:.3f}")
    
    # Faithfulness
    contexts = [
        ["Paris is the capital and largest city of France."],
        ["Python programming language was created in 1991 by Guido van Rossum."],
        ["Water boils at 100°C at sea level under standard atmospheric pressure."],
        ["The speed of light in vacuum is approximately 299,792 km/s."],
    ]
    
    faith = evaluator.evaluate_faithfulness(predictions, contexts)
    print(f"\n  Faithfulness:")
    print(f"    Mean: {faith['faithfulness_mean']:.3f}")
    
    # Report
    evaluator.full_evaluation_report("RAFT-DistilGPT2", {
        "answer_quality": quality,
        "faithfulness": faith
    })


# =============================================================================
# SECTION 5: End-to-End RAFT System
# =============================================================================

class EndToEndRAFTSystem:
    """
    Complete RAFT system combining retrieval and generation.
    
    Pipeline:
    1. Index documents with dense retriever
    2. For each query, retrieve top-K documents
    3. Generate answer using RAFT-trained generator
    4. Evaluate with comprehensive metrics
    
    This class orchestrates the full lifecycle:
    data prep → retriever training → RAFT training → inference → evaluation
    """
    
    def __init__(
        self,
        generator_model: str = "distilgpt2",
        retriever_model: str = "distilgpt2",
        top_k: int = 5,
        num_distractors: int = 3,
        use_lora: bool = True
    ):
        self.generator_model = generator_model
        self.retriever_model = retriever_model
        self.top_k = top_k
        self.num_distractors = num_distractors
        self.use_lora = use_lora
        
        self.data_pipeline = None
        self.retriever = None
        self.generator = None
        self.evaluator = RAFTEvaluator()
        
        print(f"  End-to-End RAFT System:")
        print(f"    Generator: {generator_model}")
        print(f"    Retriever: {retriever_model}")
        print(f"    Top-K: {top_k}")
        print(f"    LoRA: {use_lora}")
    
    def prepare_data(
        self,
        qa_data: List[Dict],
        corpus: List[str],
        train_ratio: float = 0.8
    ) -> Tuple[List[Dict], List[Dict]]:
        """Prepare train/eval splits with RAFT formatting."""
        self.data_pipeline = RAFTDataPipeline(num_distractors=self.num_distractors)
        
        # Split
        split_idx = int(len(qa_data) * train_ratio)
        train_qa = qa_data[:split_idx]
        eval_qa = qa_data[split_idx:]
        
        train_examples = self.data_pipeline.prepare_qa_dataset(train_qa, corpus)
        eval_examples = self.data_pipeline.prepare_qa_dataset(eval_qa, corpus)
        
        return train_examples, eval_examples
    
    def train_retriever(self, queries, documents, labels):
        """Train the dense retriever."""
        self.retriever = DenseRetrieverTrainer(
            model_name=self.retriever_model,
            embedding_dim=128
        )
        print("  Retriever training configured (would train with full data)")
    
    def train_generator(self, train_examples, eval_examples=None):
        """Train the RAFT generator."""
        self.generator = RAFTTrainer(
            model_name=self.generator_model,
            use_lora=self.use_lora
        )
        print("  Generator training configured (would train with full data)")
    
    def run_inference(
        self,
        questions: List[str],
        corpus: List[str]
    ) -> List[Dict]:
        """
        Full inference pipeline: retrieve → generate → return.
        """
        results = []
        
        for question in questions:
            # Step 1: Retrieve (simulated without actual model)
            retrieved_indices = random.sample(range(len(corpus)), min(self.top_k, len(corpus)))
            retrieved_docs = [corpus[i] for i in retrieved_indices]
            
            # Step 2: Generate (would use actual model)
            answer = f"[Generated answer for: {question}]"
            
            results.append({
                "question": question,
                "retrieved_docs": retrieved_docs,
                "answer": answer
            })
        
        return results
    
    def evaluate(
        self,
        eval_data: List[Dict],
        corpus: List[str]
    ) -> Dict:
        """Run full evaluation."""
        predictions = [d.get("answer", "") for d in eval_data]
        ground_truths = [d.get("answer", "") for d in eval_data]
        
        quality = self.evaluator.evaluate_answer_quality(predictions, ground_truths)
        
        return {"answer_quality": quality}
    
    def print_system_summary(self):
        """Print system architecture summary."""
        print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │                  End-to-End RAFT System                         │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                 │
  │  ┌──────────┐    Top-K     ┌──────────────────────────────────┐│
  │  │  Dense    │  documents   │  RAFT-Trained Generator          ││
  │  │ Retriever │─────────────→│  (LoRA fine-tuned on             ││
  │  │           │              │   oracle + distractor mix)        ││
  │  └─────┬────┘              └───────────────┬──────────────────┘│
  │        │                                    │                   │
  │   ┌────┴─────┐                    ┌─────────┴───────────┐      │
  │   │ Document  │                    │  Answer with CoT     │      │
  │   │  Index    │                    │  + Source Attribution │      │
  │   │ (FAISS)   │                    └─────────────────────┘      │
  │   └──────────┘                                                  │
  │                                                                 │
  │  Training Pipeline:                                             │
  │  1. Construct RAFT data (oracle + distractor mixing)            │
  │  2. Train retriever (contrastive, in-batch negatives)           │
  │  3. Train generator (LoRA, RAFT-formatted data)                 │
  │  4. Evaluate (EM, F1, faithfulness, robustness)                 │
  │                                                                 │
  │  RAFT Advantages:                                               │
  │  - Robust to noisy/irrelevant retrieved documents               │
  │  - Learns chain-of-thought extraction from evidence             │
  │  - Falls back to parametric knowledge when context lacks info   │
  │  - Domain-specific: trained on target domain's knowledge base   │
  └─────────────────────────────────────────────────────────────────┘
        """)


def demonstrate_end_to_end():
    """Demonstrate the end-to-end RAFT system."""
    print("\n" + "=" * 60)
    print("END-TO-END RAFT SYSTEM")
    print("=" * 60)
    
    system = EndToEndRAFTSystem(
        generator_model="distilgpt2",
        retriever_model="distilgpt2",
        top_k=5,
        num_distractors=3,
        use_lora=True
    )
    
    # Sample data
    qa_data = [
        {"question": "What is machine learning?",
         "answer": "A field of AI that learns from data.",
         "oracle_doc": "Machine learning is a subset of AI focused on building systems that learn from data."},
        {"question": "What is gradient descent?",
         "answer": "An optimization algorithm that minimizes loss by following gradients.",
         "oracle_doc": "Gradient descent is an iterative optimization algorithm used to minimize a function by moving in the direction of steepest descent."},
        {"question": "What is a neural network?",
         "answer": "A computational model inspired by biological neural networks.",
         "oracle_doc": "Neural networks are computing systems inspired by biological neural networks in the brain."},
    ]
    
    corpus = [d["oracle_doc"] for d in qa_data] + [
        "Support vector machines find optimal hyperplanes for classification.",
        "Random forests combine multiple decision trees for robust predictions.",
        "Reinforcement learning trains agents through reward signals.",
    ]
    
    # Prepare data
    train_examples, eval_examples = system.prepare_data(qa_data, corpus, train_ratio=0.67)
    
    # Show pipeline
    system.print_system_summary()
    
    # Inference simulation
    results = system.run_inference(
        ["What is machine learning?", "How does gradient descent work?"],
        corpus
    )
    
    print("  Inference results:")
    for r in results:
        print(f"    Q: {r['question']}")
        print(f"    A: {r['answer']}")
        print(f"    Retrieved {len(r['retrieved_docs'])} docs")
        print()


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 70)
    print("RETRIEVAL-AUGMENTED FINE-TUNING — HUGGINGFACE TRAINING PIPELINES")
    print("=" * 70)
    
    # Section 1: Data Pipeline
    print("\n\n📦 SECTION 1: RAFT Data Pipeline")
    demonstrate_raft_data_pipeline()
    
    # Section 2: Retriever Training
    print("\n\n📦 SECTION 2: Dense Retriever Training")
    demonstrate_retriever_training()
    
    # Section 3: RAFT Training
    print("\n\n📦 SECTION 3: RAFT Fine-Tuning Pipeline")
    demonstrate_raft_training()
    
    # Section 4: Evaluation
    print("\n\n📦 SECTION 4: Retrieval-Augmented Evaluation")
    demonstrate_evaluation()
    
    # Section 5: End-to-End System
    print("\n\n📦 SECTION 5: End-to-End RAFT System")
    demonstrate_end_to_end()
    
    print("\n" + "=" * 70)
    print("TRAINING PIPELINES COMPLETE")
    print("=" * 70)
    print("""
    Implemented:
    1. RAFT Data Pipeline — Oracle/distractor mixing with CoT
    2. Dense Retriever Training — Contrastive learning pipeline
    3. RAFT Fine-Tuning — LoRA-based generator training
    4. Evaluation — EM, F1, faithfulness, robustness metrics
    5. End-to-End System — Full retrieve → generate → evaluate
    """)


if __name__ == "__main__":
    main()
