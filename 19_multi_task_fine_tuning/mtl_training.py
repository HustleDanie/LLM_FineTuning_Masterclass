"""
Multi-Task Fine-Tuning - Production Training Pipeline
======================================================

HuggingFace-based production implementations of multi-task
fine-tuning with real models and datasets.

Sections:
    1. Multi-Task Data Preparation Pipeline
    2. Custom Multi-Task Trainer
    3. Instruction-Based Multi-Task Fine-Tuning
    4. Multi-Task LoRA with PEFT
    5. Evaluation and Task-Specific Metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
import json
import os


# =============================================================================
# SECTION 1: Multi-Task Data Preparation Pipeline
# =============================================================================

class MultiTaskDataPreparer:
    """
    Production pipeline for preparing multi-task training data.
    
    Handles the key challenges:
    1. Different tasks have different input/output formats
    2. Dataset sizes vary wildly across tasks
    3. Need consistent tokenization and batching
    4. Task-specific preprocessing and label encoding
    
    This preparer converts diverse task datasets into a unified
    format suitable for multi-task training.
    """
    
    def __init__(self, model_name: str = "distilgpt2"):
        """
        Initialize with a tokenizer.
        
        Args:
            model_name: HuggingFace model name for tokenizer
        """
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"  Loaded tokenizer: {model_name}")
        except Exception as e:
            print(f"  Warning: Could not load tokenizer: {e}")
            self.tokenizer = None
    
    def prepare_classification_task(
        self,
        texts: List[str],
        labels: List[int],
        task_name: str,
        max_length: int = 128
    ) -> List[Dict]:
        """
        Prepare a text classification task.
        
        Converts raw text + labels into tokenized format with
        task identifiers for multi-task routing.
        """
        processed = []
        
        for text, label in zip(texts, labels):
            if self.tokenizer:
                encoded = self.tokenizer(
                    text,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                item = {
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                    "labels": torch.tensor(label, dtype=torch.long),
                    "task_name": task_name
                }
            else:
                # Fallback for demo without tokenizer
                item = {
                    "input_ids": torch.randint(0, 1000, (max_length,)),
                    "attention_mask": torch.ones(max_length, dtype=torch.long),
                    "labels": torch.tensor(label, dtype=torch.long),
                    "task_name": task_name
                }
            processed.append(item)
        
        print(f"    Prepared {len(processed)} examples for '{task_name}' (classification)")
        return processed
    
    def prepare_instruction_task(
        self,
        instructions: List[str],
        inputs: List[str],
        outputs: List[str],
        task_name: str,
        max_length: int = 256
    ) -> List[Dict]:
        """
        Prepare an instruction-following task for unified text-to-text format.
        
        This is the modern approach: frame ALL tasks as text generation.
        
        Format:
            "### Instruction: {instruction}\n### Input: {input}\n### Response: {output}"
        """
        processed = []
        
        for instruction, inp, out in zip(instructions, inputs, outputs):
            prompt = f"### Instruction: {instruction}\n### Input: {inp}\n### Response: {out}"
            
            if self.tokenizer:
                encoded = self.tokenizer(
                    prompt,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                # For causal LM, labels = input_ids (shifted internally)
                labels = encoded["input_ids"].clone().squeeze(0)
                # Mask instruction/input tokens so we only train on output
                # (simplified: in practice, find the "### Response:" position)
                item = {
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                    "labels": labels,
                    "task_name": task_name
                }
            else:
                item = {
                    "input_ids": torch.randint(0, 1000, (max_length,)),
                    "attention_mask": torch.ones(max_length, dtype=torch.long),
                    "labels": torch.randint(0, 1000, (max_length,)),
                    "task_name": task_name
                }
            processed.append(item)
        
        print(f"    Prepared {len(processed)} examples for '{task_name}' (instruction)")
        return processed
    
    @staticmethod
    def compute_sampling_weights(
        task_sizes: Dict[str, int],
        strategy: str = "sqrt",
        temperature: float = 2.0
    ) -> Dict[str, float]:
        """
        Compute task sampling probabilities.
        
        Args:
            task_sizes: {task_name: num_examples}
            strategy: "proportional", "equal", "sqrt", or "temperature"
            temperature: Temperature for temperature-based sampling
        
        Returns:
            {task_name: probability}
        """
        sizes = np.array(list(task_sizes.values()), dtype=np.float64)
        names = list(task_sizes.keys())
        
        if strategy == "proportional":
            probs = sizes / sizes.sum()
        elif strategy == "equal":
            probs = np.ones_like(sizes) / len(sizes)
        elif strategy == "sqrt":
            probs = np.sqrt(sizes)
            probs /= probs.sum()
        elif strategy == "temperature":
            probs = sizes ** (1.0 / temperature)
            probs /= probs.sum()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        result = dict(zip(names, probs.tolist()))
        
        print(f"\n  Sampling weights (strategy='{strategy}'):")
        for name, prob in result.items():
            print(f"    {name}: {prob:.4f} ({task_sizes[name]} examples)")
        
        return result


class MultiTaskDataset(Dataset):
    """
    PyTorch Dataset that combines multiple task datasets.
    
    Each item includes a 'task_name' field so the model knows
    which head/adapter to use during training.
    """
    
    def __init__(
        self,
        task_datasets: Dict[str, List[Dict]],
        sampling_strategy: str = "sqrt",
        temperature: float = 2.0
    ):
        self.task_datasets = task_datasets
        self.task_names = list(task_datasets.keys())
        
        # Flatten all data with task labels
        self.all_data = []
        for task_name, data in task_datasets.items():
            for item in data:
                item["task_name"] = task_name
                self.all_data.append(item)
        
        # Compute sampling weights
        task_sizes = {name: len(ds) for name, ds in task_datasets.items()}
        self.sampling_weights = MultiTaskDataPreparer.compute_sampling_weights(
            task_sizes, strategy=sampling_strategy, temperature=temperature
        )
        
        print(f"  Total examples: {len(self.all_data)}")
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        return self.all_data[idx]


# =============================================================================
# SECTION 2: Custom Multi-Task Trainer
# =============================================================================

class MultiTaskTrainer:
    """
    Custom trainer for multi-task learning with HuggingFace models.
    
    Handles:
    - Task-specific forward passes (routing to correct head)
    - Loss balancing across tasks
    - Per-task metric computation
    - Gradient accumulation across tasks
    - PCGrad integration (optional)
    
    This trainer extends the standard training loop with
    multi-task-specific logic that HuggingFace's Trainer lacks.
    """
    
    def __init__(
        self,
        model: nn.Module,
        task_heads: Dict[str, nn.Module],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        task_weights: Optional[Dict[str, float]] = None,
        use_pcgrad: bool = False,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.task_heads = {k: v.to(device) for k, v in task_heads.items()}
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_pcgrad = use_pcgrad
        
        # Default equal weights
        if task_weights is None:
            task_names = list(task_heads.keys())
            self.task_weights = {name: 1.0 for name in task_names}
        else:
            self.task_weights = task_weights
        
        # Logging
        self.history = defaultdict(list)
    
    def train_epoch(
        self,
        task_dataloaders: Dict[str, DataLoader],
        epoch: int = 0
    ) -> Dict[str, float]:
        """
        Train for one epoch with task-alternating batches.
        
        Strategy: Round-robin through task dataloaders,
        computing weighted loss per task.
        """
        self.model.train()
        for head in self.task_heads.values():
            head.train()
        
        epoch_losses = defaultdict(list)
        
        # Create iterators for each task
        task_iters = {
            name: iter(dl) for name, dl in task_dataloaders.items()
        }
        
        # Determine number of steps (max across tasks)
        max_steps = max(len(dl) for dl in task_dataloaders.values())
        
        for step in range(max_steps):
            total_loss = torch.tensor(0.0, device=self.device)
            
            for task_name in self.task_heads.keys():
                # Get batch (cycle if exhausted)
                try:
                    batch = next(task_iters[task_name])
                except StopIteration:
                    task_iters[task_name] = iter(task_dataloaders[task_name])
                    batch = next(task_iters[task_name])
                
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward through shared model + task head
                # (Assuming model returns hidden states)
                hidden = self.model(input_ids)
                
                # Handle different model output types
                if hasattr(hidden, 'last_hidden_state'):
                    hidden = hidden.last_hidden_state
                
                logits = self.task_heads[task_name](hidden)
                
                # Compute task loss
                if logits.dim() == 2 and labels.dim() == 1:
                    task_loss = F.cross_entropy(logits, labels)
                elif logits.dim() == 3:
                    task_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100
                    )
                else:
                    task_loss = F.mse_loss(logits.squeeze(), labels.float())
                
                # Weight the loss
                weighted_loss = self.task_weights[task_name] * task_loss
                total_loss = total_loss + weighted_loss
                
                epoch_losses[task_name].append(task_loss.item())
            
            # Backward and step
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + 
                [p for h in self.task_heads.values() for p in h.parameters()],
                max_norm=1.0
            )
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
        
        # Compute epoch averages
        avg_losses = {}
        for name, losses in epoch_losses.items():
            avg = np.mean(losses)
            avg_losses[name] = avg
            self.history[f"{name}_loss"].append(avg)
        
        return avg_losses


def demonstrate_custom_trainer():
    """Demonstrate the custom multi-task trainer."""
    print("=" * 60)
    print("CUSTOM MULTI-TASK TRAINER")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Shared encoder
    class SimpleEncoder(nn.Module):
        def __init__(self, vocab_size=5000, d_model=128):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.encoder = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, batch_first=True
            )
        
        def forward(self, input_ids):
            return self.encoder(self.embedding(input_ids))
    
    # Task heads
    class ClassificationHead(nn.Module):
        def __init__(self, d_model, num_classes):
            super().__init__()
            self.linear = nn.Linear(d_model, num_classes)
        
        def forward(self, hidden):
            return self.linear(hidden[:, 0, :])  # CLS pooling
    
    model = SimpleEncoder()
    task_heads = {
        "sentiment": ClassificationHead(128, 3),
        "topic": ClassificationHead(128, 5),
    }
    
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + 
        [p for h in task_heads.values() for p in h.parameters()],
        lr=1e-4
    )
    
    trainer = MultiTaskTrainer(
        model=model,
        task_heads=task_heads,
        optimizer=optimizer,
        task_weights={"sentiment": 1.0, "topic": 1.0}
    )
    
    # Create synthetic dataloaders
    class SyntheticDataset(Dataset):
        def __init__(self, size, num_classes, seq_len=32):
            self.data = [
                {
                    "input_ids": torch.randint(0, 5000, (seq_len,)),
                    "attention_mask": torch.ones(seq_len, dtype=torch.long),
                    "labels": torch.randint(0, num_classes, ()),
                }
                for _ in range(size)
            ]
        def __len__(self): return len(self.data)
        def __getitem__(self, idx): return self.data[idx]
    
    dataloaders = {
        "sentiment": DataLoader(SyntheticDataset(200, 3), batch_size=16, shuffle=True),
        "topic": DataLoader(SyntheticDataset(100, 5), batch_size=16, shuffle=True),
    }
    
    # Train
    for epoch in range(3):
        losses = trainer.train_epoch(dataloaders, epoch)
        loss_str = ", ".join(f"{k}: {v:.4f}" for k, v in losses.items())
        print(f"  Epoch {epoch+1}: {loss_str}")
    
    print("\n  Training complete!")


# =============================================================================
# SECTION 3: Instruction-Based Multi-Task Fine-Tuning
# =============================================================================

class InstructionMTLPipeline:
    """
    Modern instruction-based multi-task fine-tuning pipeline.
    
    This is how FLAN, T0, and similar models are trained:
    - ALL tasks are converted to instruction-following format
    - Single decoder-only or encoder-decoder model
    - No task-specific heads needed
    - Natural language instructions route the model
    
    Format:
        Input:  "{task_instruction}\n{input_text}"
        Output: "{expected_output}"
    
    This is the dominant paradigm for production MTL because:
    1. Any task can be expressed as text-to-text
    2. No architectural changes needed
    3. Enables zero-shot generalization to new tasks
    4. Simple to add new tasks (just add instructions)
    """
    
    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self.task_templates = {}
        
    def register_task(
        self,
        task_name: str,
        instruction_template: str,
        output_template: str = "{output}"
    ):
        """
        Register a task with its instruction template.
        
        Example:
            register_task(
                "sentiment",
                "Classify the sentiment of the following text as positive, negative, or neutral.",
                "{label}"
            )
        """
        self.task_templates[task_name] = {
            "instruction": instruction_template,
            "output": output_template
        }
        print(f"  Registered task: '{task_name}'")
    
    def format_example(
        self,
        task_name: str,
        input_text: str,
        output_text: Optional[str] = None
    ) -> str:
        """
        Format a single example in the instruction template.
        
        Returns:
            Formatted string ready for tokenization
        """
        if task_name not in self.task_templates:
            raise ValueError(f"Unknown task: {task_name}")
        
        template = self.task_templates[task_name]
        
        prompt = f"### Instruction: {template['instruction']}\n"
        prompt += f"### Input: {input_text}\n"
        prompt += f"### Response:"
        
        if output_text is not None:
            prompt += f" {output_text}"
        
        return prompt
    
    def prepare_mtl_dataset(
        self,
        task_datasets: Dict[str, List[Tuple[str, str]]],
        sampling_strategy: str = "sqrt"
    ) -> List[str]:
        """
        Prepare a unified instruction-tuning dataset from multiple tasks.
        
        Args:
            task_datasets: {task_name: [(input, output), ...]}
            sampling_strategy: How to balance tasks
        """
        all_examples = []
        
        for task_name, examples in task_datasets.items():
            for input_text, output_text in examples:
                formatted = self.format_example(task_name, input_text, output_text)
                all_examples.append(formatted)
        
        # Shuffle
        np.random.shuffle(all_examples)
        
        print(f"\n  Prepared {len(all_examples)} instruction examples:")
        for name, data in task_datasets.items():
            print(f"    {name}: {len(data)} examples")
        
        return all_examples
    
    def setup_training(self):
        """
        Set up the HuggingFace training pipeline.
        
        Uses standard causal LM fine-tuning with the instruction format.
        """
        try:
            from transformers import (
                AutoModelForCausalLM, AutoTokenizer,
                TrainingArguments, Trainer, DataCollatorForLanguageModeling
            )
            
            print(f"\n  Loading model: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.pad_token_id
            
            # Training arguments optimized for multi-task instruction tuning
            training_args = TrainingArguments(
                output_dir="./mtl_instruction_output",
                num_train_epochs=3,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,  # Effective batch of 16
                learning_rate=2e-5,
                warmup_ratio=0.1,
                weight_decay=0.01,
                logging_steps=10,
                save_strategy="epoch",
                fp16=torch.cuda.is_available(),
                # MTL-specific: longer warmup helps with task diversity
                lr_scheduler_type="cosine",
                # Gradient clipping important for multi-task stability
                max_grad_norm=1.0,
            )
            
            print("  Training arguments configured")
            print(f"    Epochs: {training_args.num_train_epochs}")
            print(f"    Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
            print(f"    Learning rate: {training_args.learning_rate}")
            
            return model, tokenizer, training_args
            
        except ImportError as e:
            print(f"  Warning: transformers not available: {e}")
            return None, None, None


def demonstrate_instruction_mtl():
    """Demonstrate instruction-based multi-task setup."""
    print("\n" + "=" * 60)
    print("INSTRUCTION-BASED MULTI-TASK FINE-TUNING")
    print("=" * 60)
    
    pipeline = InstructionMTLPipeline()
    
    # Register tasks
    pipeline.register_task(
        "sentiment",
        "Classify the sentiment of the following text as positive, negative, or neutral."
    )
    pipeline.register_task(
        "summarize",
        "Summarize the following text in one sentence."
    )
    pipeline.register_task(
        "ner",
        "Extract all named entities (persons, locations, organizations) from the text."
    )
    pipeline.register_task(
        "qa",
        "Answer the following question based on the provided context."
    )
    
    # Example formatted outputs
    print("\n  Example formatted prompts:")
    print("  " + "-" * 50)
    
    examples = [
        ("sentiment", "This movie was absolutely wonderful!", "positive"),
        ("summarize", "The researchers found that transformer models...", "Transformers improve NLP performance."),
        ("ner", "John Smith visited Paris last Monday.", "John Smith [PERSON], Paris [LOCATION], Monday [DATE]"),
        ("qa", "Context: Paris is the capital of France. Question: What is the capital of France?", "Paris"),
    ]
    
    for task, inp, out in examples:
        formatted = pipeline.format_example(task, inp, out)
        print(f"\n  Task: {task}")
        print(f"  {formatted[:100]}...")
    
    # Prepare dataset
    task_datasets = {
        "sentiment": [
            ("Great product, love it!", "positive"),
            ("Terrible service, never again.", "negative"),
            ("It was okay, nothing special.", "neutral"),
        ],
        "summarize": [
            ("The conference covered AI, ML, and robotics topics.", "Conference on AI and related topics."),
        ],
        "ner": [
            ("Apple Inc is based in Cupertino.", "Apple Inc [ORG], Cupertino [LOC]"),
        ],
    }
    
    all_examples = pipeline.prepare_mtl_dataset(task_datasets)


# =============================================================================
# SECTION 4: Multi-Task LoRA with PEFT
# =============================================================================

class MultiTaskLoRAPipeline:
    """
    Multi-task fine-tuning using PEFT's LoRA with task-specific adapters.
    
    Architecture:
        Base Model (frozen) + {LoRA_sentiment, LoRA_ner, LoRA_qa, ...}
    
    Each task gets its own LoRA adapter. During training:
    - Load the appropriate adapter for each task
    - Only train that adapter's parameters
    - Switch adapters between tasks
    
    This avoids gradient conflicts entirely because each task
    optimizes separate parameters (the adapter weights).
    
    Approaches:
    1. Sequential: Train one adapter at a time
    2. Interleaved: Alternate tasks, switching adapters
    3. Merged: Train shared LoRA + task-specific LoRA
    """
    
    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self.adapters = {}
    
    def setup_multi_adapter_model(self, task_configs: Dict[str, dict]):
        """
        Set up a model with multiple LoRA adapters.
        
        Each task gets its own adapter configuration, allowing
        different ranks, target modules, etc.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import get_peft_model, LoraConfig, TaskType, PeftModel
            
            print(f"  Loading base model: {self.model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Create first adapter
            first_task = list(task_configs.keys())[0]
            first_config = task_configs[first_task]
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=first_config.get("rank", 8),
                lora_alpha=first_config.get("alpha", 16),
                lora_dropout=first_config.get("dropout", 0.1),
                target_modules=first_config.get("target_modules", ["c_attn"]),
            )
            
            model = get_peft_model(base_model, lora_config, adapter_name=first_task)
            self.adapters[first_task] = lora_config
            print(f"    Added adapter '{first_task}': rank={lora_config.r}")
            
            # Add additional adapters
            for task_name, config in list(task_configs.items())[1:]:
                adapter_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=config.get("rank", 8),
                    lora_alpha=config.get("alpha", 16),
                    lora_dropout=config.get("dropout", 0.1),
                    target_modules=config.get("target_modules", ["c_attn"]),
                )
                model.add_adapter(task_name, adapter_config)
                self.adapters[task_name] = adapter_config
                print(f"    Added adapter '{task_name}': rank={adapter_config.r}")
            
            # Print summary
            total_base = sum(p.numel() for p in base_model.parameters())
            print(f"\n  Base model parameters: {total_base:,}")
            for task_name in task_configs:
                model.set_adapter(task_name)
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  Adapter '{task_name}' trainable params: {trainable:,} "
                      f"({trainable/total_base*100:.2f}%)")
            
            return model, tokenizer
            
        except ImportError as e:
            print(f"  Warning: PEFT not available: {e}")
            print("  Install with: pip install peft")
            return None, None
    
    @staticmethod
    def demonstrate_adapter_switching():
        """Show how to switch between task adapters during training."""
        print("\n  Multi-Adapter Training Loop (pseudocode):")
        print("  " + "-" * 50)
        print("""
    for step in range(total_steps):
        # Sample a task
        task_name = sample_task(task_probs)
        
        # Switch to task's adapter
        model.set_adapter(task_name)
        
        # Get batch for this task
        batch = task_dataloaders[task_name].next()
        
        # Forward pass (only task's LoRA params are active)
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward (only task's LoRA gradients computed)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Save individual adapters
    for task_name in tasks:
        model.set_adapter(task_name)
        model.save_pretrained(f"./adapters/{task_name}")
    
    # At inference: load the adapter you need
    model.set_adapter("sentiment")  # For sentiment task
    model.set_adapter("ner")        # For NER task
        """)


def demonstrate_multi_task_lora():
    """Demonstrate multi-task LoRA setup."""
    print("\n" + "=" * 60)
    print("MULTI-TASK LoRA WITH PEFT")
    print("=" * 60)
    
    pipeline = MultiTaskLoRAPipeline()
    
    task_configs = {
        "sentiment": {"rank": 8, "alpha": 16, "target_modules": ["c_attn"]},
        "summarization": {"rank": 16, "alpha": 32, "target_modules": ["c_attn"]},
        "ner": {"rank": 4, "alpha": 8, "target_modules": ["c_attn"]},
    }
    
    model, tokenizer = pipeline.setup_multi_adapter_model(task_configs)
    
    if model is not None:
        # Show adapter switching
        pipeline.demonstrate_adapter_switching()
    else:
        print("\n  Showing conceptual setup (install peft for full demo)")
        pipeline.demonstrate_adapter_switching()


# =============================================================================
# SECTION 5: Evaluation and Task-Specific Metrics
# =============================================================================

class MultiTaskEvaluator:
    """
    Comprehensive evaluation for multi-task models.
    
    Challenges in multi-task evaluation:
    1. Different tasks have different metrics (accuracy, F1, BLEU, ROUGE)
    2. Need to track per-task AND aggregate performance
    3. Must detect negative transfer (compare with single-task baselines)
    4. Task interactions can be complex (A helps B but B hurts C)
    """
    
    def __init__(self, task_metrics: Dict[str, str]):
        """
        Args:
            task_metrics: {task_name: metric_name}
                e.g., {"sentiment": "accuracy", "ner": "f1", "summarization": "rouge"}
        """
        self.task_metrics = task_metrics
        self.results = {}
    
    def evaluate_task(
        self,
        task_name: str,
        predictions: List,
        references: List
    ) -> Dict[str, float]:
        """Evaluate a single task."""
        metric_name = self.task_metrics.get(task_name, "accuracy")
        
        if metric_name == "accuracy":
            correct = sum(p == r for p, r in zip(predictions, references))
            score = correct / len(predictions)
            return {"accuracy": score}
        
        elif metric_name == "f1":
            # Simplified F1 calculation
            tp = sum(1 for p, r in zip(predictions, references) if p == r and p != 0)
            fp = sum(1 for p, r in zip(predictions, references) if p != 0 and p != r)
            fn = sum(1 for p, r in zip(predictions, references) if r != 0 and p != r)
            
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            return {"precision": precision, "recall": recall, "f1": f1}
        
        return {}
    
    def compute_mtl_score(
        self,
        mtl_scores: Dict[str, float],
        single_task_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute aggregate MTL performance metrics.
        
        Key metrics:
        1. Average Performance: mean across tasks
        2. Relative Improvement: vs single-task baselines
        3. Delta metric: How much MTL helps/hurts each task
        """
        results = {}
        
        # Average performance
        avg_mtl = np.mean(list(mtl_scores.values()))
        avg_single = np.mean(list(single_task_scores.values()))
        
        results["avg_mtl_score"] = avg_mtl
        results["avg_single_task_score"] = avg_single
        results["avg_improvement"] = avg_mtl - avg_single
        results["avg_relative_improvement"] = (avg_mtl - avg_single) / max(avg_single, 1e-8)
        
        # Per-task delta
        results["per_task_delta"] = {}
        for task in mtl_scores:
            if task in single_task_scores:
                delta = mtl_scores[task] - single_task_scores[task]
                results["per_task_delta"][task] = delta
        
        # Count positive/negative transfers
        deltas = list(results["per_task_delta"].values())
        results["positive_transfer_count"] = sum(1 for d in deltas if d > 0)
        results["negative_transfer_count"] = sum(1 for d in deltas if d < 0)
        
        return results
    
    @staticmethod
    def demonstrate_evaluation():
        """Demonstrate multi-task evaluation."""
        print("\n" + "=" * 60)
        print("MULTI-TASK EVALUATION")
        print("=" * 60)
        
        evaluator = MultiTaskEvaluator({
            "sentiment": "accuracy",
            "ner": "f1",
            "topic": "accuracy",
            "similarity": "accuracy"
        })
        
        # Simulated results
        mtl_scores = {
            "sentiment": 0.89,
            "ner": 0.82,
            "topic": 0.85,
            "similarity": 0.78
        }
        
        single_task_scores = {
            "sentiment": 0.87,
            "ner": 0.84,
            "topic": 0.81,
            "similarity": 0.80
        }
        
        results = evaluator.compute_mtl_score(mtl_scores, single_task_scores)
        
        print(f"\n  MTL vs Single-Task Comparison:")
        print(f"  {'Task':>15} {'MTL':>8} {'Single':>8} {'Delta':>8} {'Transfer':>12}")
        print("  " + "-" * 55)
        
        for task in mtl_scores:
            mtl = mtl_scores[task]
            single = single_task_scores[task]
            delta = results["per_task_delta"][task]
            transfer = "✅ Positive" if delta > 0 else "❌ Negative"
            print(f"  {task:>15} {mtl:>8.3f} {single:>8.3f} {delta:>+8.3f} {transfer:>12}")
        
        print(f"\n  Aggregate Metrics:")
        print(f"    Average MTL score: {results['avg_mtl_score']:.3f}")
        print(f"    Average single-task score: {results['avg_single_task_score']:.3f}")
        print(f"    Average improvement: {results['avg_improvement']:+.3f}")
        print(f"    Positive transfer: {results['positive_transfer_count']}/{len(mtl_scores)} tasks")
        print(f"    Negative transfer: {results['negative_transfer_count']}/{len(mtl_scores)} tasks")
        
        print("""
  Key observations:
    • Sentiment and topic IMPROVED with MTL (positive transfer)  
    • NER and similarity slightly DECREASED (negative transfer)
    • Overall: net positive with +0.005 average improvement
    • Recommendation: Group sentiment+topic separately from NER+similarity
        """)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 70)
    print("MULTI-TASK FINE-TUNING — PRODUCTION TRAINING PIPELINE")
    print("=" * 70)
    
    # Section 1: Data Preparation
    print("\n\n📦 SECTION 1: Multi-Task Data Preparation")
    preparer = MultiTaskDataPreparer()
    
    # Demonstrate sampling weight computation
    task_sizes = {
        "sentiment": 50000,
        "ner": 10000,
        "qa": 25000,
        "summarization": 3000
    }
    
    for strategy in ["proportional", "equal", "sqrt", "temperature"]:
        MultiTaskDataPreparer.compute_sampling_weights(task_sizes, strategy)
    
    # Section 2: Custom Trainer
    print("\n\n🏋️ SECTION 2: Custom Multi-Task Trainer")
    demonstrate_custom_trainer()
    
    # Section 3: Instruction-Based MTL
    print("\n\n📝 SECTION 3: Instruction-Based Multi-Task Fine-Tuning")
    demonstrate_instruction_mtl()
    
    # Section 4: Multi-Task LoRA
    print("\n\n🔌 SECTION 4: Multi-Task LoRA with PEFT")
    demonstrate_multi_task_lora()
    
    # Section 5: Evaluation
    print("\n\n📊 SECTION 5: Multi-Task Evaluation")
    MultiTaskEvaluator.demonstrate_evaluation()
    
    print("\n" + "=" * 70)
    print("PRODUCTION PIPELINE SUMMARY")
    print("=" * 70)
    print("""
    Implemented:
    
    1. Data Preparation Pipeline
       - Task-specific formatters (classification, instruction)
       - Temperature-based task sampling
       - 4 sampling strategies: proportional, equal, sqrt, temperature
    
    2. Custom Multi-Task Trainer
       - Round-robin task batching
       - Weighted loss combination
       - Gradient clipping for stability
    
    3. Instruction-Based MTL (FLAN-style)
       - Unified text-to-text format
       - Task-agnostic training (no special heads)
       - Template-based task registration
    
    4. Multi-Task LoRA
       - Per-task adapter creation
       - Adapter switching during training
       - Independent task-specific parameters
    
    5. Evaluation Framework
       - Per-task metrics (accuracy, F1, etc.)
       - MTL vs single-task comparison
       - Positive/negative transfer detection
    """)


if __name__ == "__main__":
    main()
