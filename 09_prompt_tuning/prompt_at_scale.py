"""
Prompt Tuning — Scaling & Production
======================================

Scaling behavior and production deployment patterns:

1. Scaling Laws for Prompt Tuning
   - Performance vs model size
   - Performance vs prompt length
   - Crossover point with full FT

2. Multi-Task Serving Architecture
   - One model, many prompts
   - Batching different tasks together
   - Prompt routing and management

3. Production Deployment Patterns
   - Serving infrastructure
   - Prompt versioning
   - A/B testing with prompts

4. Comparison: Prompt Tuning vs Full FT at Scale
   - When prompt tuning wins
   - When to switch methods
   - Cost analysis

5. End-to-End Production Example
   - Complete pipeline from data to deployment

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================================
# SECTION 1: SCALING LAWS
# ============================================================================

def scaling_laws():
    """
    Analyze how prompt tuning performance scales with model size.
    """
    print("=" * 65)
    print("  SECTION 1: SCALING LAWS FOR PROMPT TUNING")
    print("=" * 65)
    
    # Data based on Lester et al. (2021) findings with T5
    print(f"""
  ═══ The Central Finding ═══
  
  "As models exceed ~10B parameters, prompt tuning matches
   model tuning (full fine-tuning) despite training 10,000×
   fewer parameters."  — Lester et al., 2021
  
  
  ═══ Performance vs Model Size (SuperGLUE) ═══
  
  Model Params  │ Full FT │Prompt T.│  Gap  │ Prompt/Full
  ──────────────┼─────────┼─────────┼───────┼───────────
      60M       │  74.0   │  61.0   │  13.0 │  82.4%
     220M       │  82.4   │  72.0   │  10.4 │  87.4%
     770M       │  86.5   │  80.5   │   6.0 │  93.1%
      3B        │  89.5   │  87.0   │   2.5 │  97.2%
     11B        │  91.5   │  91.0   │   0.5 │  99.5%  ★
  
  The gap narrows exponentially:
  
  100M: ████████████████████████████████     13.0 pt gap
  200M: █████████████████████████            10.4
  800M: ████████████████                      6.0
    3B: ██████                                2.5
   11B: █                                     0.5 ← near zero!
  
  
  ═══ Performance vs Prompt Length ═══
  
  Using T5-XXL (11B):
  
  Length │Score│ Params  │ Observation
  ───────┼─────┼─────────┼──────────────────────
     1   │ 86.5│  1,024  │ Single token — works!
     5   │ 89.8│  5,120  │ Good improvement
    10   │ 90.5│ 10,240  │ Diminishing returns
    20   │ 91.0│ 20,480  │ Near optimal
    50   │ 90.8│ 51,200  │ No further gain
   100   │ 90.5│102,400  │ Slight degradation!
   150   │ 90.0│153,600  │ Context cost > benefit
  
  Sweet spots:
  • Small models: 20-50 tokens (need more expressiveness)
  • Large models: 5-20 tokens (few tokens suffice)
  • Very large (10B+): even 1 token works reasonably!
""")
    
    # Simulate the scaling curve
    print(f"  ─── Simulated Scaling Curve ───")
    
    model_sizes = [60, 220, 770, 3000, 11000]  # millions
    full_ft_scores = [74.0, 82.4, 86.5, 89.5, 91.5]
    prompt_scores = [61.0, 72.0, 80.5, 87.0, 91.0]
    
    print(f"\n  Gap convergence rate:")
    for i, (size, ft, pt) in enumerate(zip(model_sizes, full_ft_scores, prompt_scores)):
        gap = ft - pt
        ratio = pt / ft * 100
        bar = "█" * max(1, int(gap * 2))
        print(f"  {size:>6}M: gap={gap:>5.1f}  ratio={ratio:>5.1f}%  {bar}")
    
    # Estimate crossover
    print(f"""
  ═══ The Crossover Point ═══
  
  Extrapolating the trend:
  • At ~11B params: Gap ≈ 0.5 points (negligible)
  • At ~20B params: Gap ≈ 0 (complete convergence)
  • At  100B+: Prompt tuning may even EXCEED full FT
    (less overfitting due to fewer trainable params)
  
  This means for modern large models (7B, 13B, 70B),
  prompt tuning is a viable REPLACEMENT for full FT!
""")


# ============================================================================
# SECTION 2: MULTI-TASK SERVING
# ============================================================================

@dataclass
class TaskPrompt:
    """Represents a trained soft prompt for a specific task."""
    task_name: str
    prompt_tensor: torch.Tensor
    num_tokens: int
    trained_epochs: int = 0
    accuracy: float = 0.0
    version: str = "1.0"
    metadata: dict = field(default_factory=dict)


class PromptServingEngine:
    """
    Production serving engine for multi-task prompt tuning.
    
    Architecture:
    ┌──────────────────────────────────────────────────┐
    │              Prompt Serving Engine                │
    │                                                  │
    │  ┌──────────┐  ┌──────────────────────────────┐  │
    │  │  Prompt   │  │     Frozen Base Model         │  │
    │  │  Registry │→ │     (loaded once, shared)     │  │
    │  │           │  │                               │  │
    │  │ sentiment │  │  [prompt ; input] → output    │  │
    │  │ nli       │  │                               │  │
    │  │ summary   │  │  Same model serves ALL tasks  │  │
    │  │ translate │  │  by swapping the prompt!       │  │
    │  │ qa        │  │                               │  │
    │  └──────────┘  └──────────────────────────────┘  │
    │                                                  │
    │  Benefits:                                       │
    │  • One GPU for ALL tasks                         │
    │  • Add new tasks = add ~60KB prompt              │
    │  • Switch tasks in < 1ms                         │
    │  • Different tasks in same batch!                │
    └──────────────────────────────────────────────────┘
    """
    
    def __init__(self, base_model_name: str = "distilgpt2"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"  Loading base model: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.model.eval()
        
        self.d_model = self.model.config.n_embd
        self.prompt_registry: Dict[str, TaskPrompt] = {}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model loaded: {total_params:,} params")
        print(f"  Ready to serve multiple tasks!")
    
    def register_task(self, task_name: str, prompt: torch.Tensor, **metadata):
        """Register a trained prompt for a task."""
        task_prompt = TaskPrompt(
            task_name=task_name,
            prompt_tensor=prompt.detach(),
            num_tokens=prompt.shape[0],
            metadata=metadata,
        )
        self.prompt_registry[task_name] = task_prompt
        
        size_kb = prompt.numel() * 4 / 1024
        print(f"  Registered '{task_name}': {prompt.shape[0]} tokens, "
              f"{size_kb:.1f} KB")
    
    def serve(
        self,
        task_name: str,
        text: str,
        max_new_tokens: int = 50,
    ) -> str:
        """Serve a single request with the specified task prompt."""
        if task_name not in self.prompt_registry:
            raise ValueError(f"Unknown task: {task_name}")
        
        task = self.prompt_registry[task_name]
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        # Get embeddings
        embed_layer = self.model.transformer.wte
        input_embeds = embed_layer(input_ids)
        
        # Prepend task prompt
        prompt = task.prompt_tensor.unsqueeze(0)  # [1, N, D]
        combined = torch.cat([prompt, input_embeds], dim=1)
        
        # Create attention mask
        attention_mask = torch.ones(1, combined.shape[1])
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=combined,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def serve_batch(
        self,
        requests: List[Tuple[str, str]],  # [(task_name, text), ...]
        max_new_tokens: int = 30,
    ) -> List[str]:
        """
        Serve multiple requests from DIFFERENT tasks in one batch!
        
        This is prompt tuning's superpower: different tasks can be
        batched together because they share the same model.
        """
        embed_layer = self.model.transformer.wte
        
        all_embeds = []
        max_len = 0
        
        for task_name, text in requests:
            task = self.prompt_registry[task_name]
            tokens = self.tokenizer(text, return_tensors="pt")
            input_embeds = embed_layer(tokens["input_ids"])
            
            prompt = task.prompt_tensor.unsqueeze(0)
            combined = torch.cat([prompt, input_embeds], dim=1)
            
            all_embeds.append(combined)
            max_len = max(max_len, combined.shape[1])
        
        # Pad to same length for batching
        padded_embeds = []
        attention_masks = []
        
        for embeds in all_embeds:
            seq_len = embeds.shape[1]
            pad_len = max_len - seq_len
            
            if pad_len > 0:
                padding = torch.zeros(1, pad_len, self.d_model)
                embeds = torch.cat([padding, embeds], dim=1)
                mask = torch.cat([
                    torch.zeros(1, pad_len),
                    torch.ones(1, seq_len),
                ], dim=1)
            else:
                mask = torch.ones(1, seq_len)
            
            padded_embeds.append(embeds)
            attention_masks.append(mask)
        
        batch_embeds = torch.cat(padded_embeds, dim=0)
        batch_masks = torch.cat(attention_masks, dim=0)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=batch_embeds,
                attention_mask=batch_masks,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        return [
            self.tokenizer.decode(out, skip_special_tokens=True)
            for out in outputs
        ]
    
    def list_tasks(self):
        """Show all registered tasks."""
        print(f"\n  Registered Tasks:")
        print(f"  {'Task':>15} {'Tokens':>8} {'Size (KB)':>10}")
        print(f"  {'─'*15}─{'─'*8}─{'─'*10}")
        total_kb = 0
        for name, task in self.prompt_registry.items():
            kb = task.prompt_tensor.numel() * 4 / 1024
            total_kb += kb
            print(f"  {name:>15}  {task.num_tokens:>6}  {kb:>8.1f}")
        print(f"  {'─'*15}─{'─'*8}─{'─'*10}")
        print(f"  {'TOTAL':>15}  {'':>6}  {total_kb:>8.1f}")


def demonstrate_multi_task_serving():
    """Show multi-task serving with prompt tuning."""
    print("\n" + "=" * 65)
    print("  SECTION 2: MULTI-TASK SERVING")
    print("=" * 65)
    
    torch.manual_seed(42)
    
    engine = PromptServingEngine("distilgpt2")
    d = engine.d_model
    
    # Register multiple task prompts (simulated as random — 
    # in production these would be trained)
    tasks = {
        "sentiment": "Classify the sentiment of this text:",
        "summarize": "Summarize the following text concisely:",
        "translate": "Translate the following from English to French:",
        "qa": "Answer the following question based on the context:",
        "creative": "Continue the story in a creative way:",
    }
    
    for task_name, init_text in tasks.items():
        # Simulate a trained prompt (in practice: load from checkpoint)
        prompt = torch.randn(20, d) * 0.02
        engine.register_task(task_name, prompt)
    
    engine.list_tasks()
    
    # Serve individual requests
    print(f"\n  ─── Individual Serving ───")
    
    test_requests = [
        ("sentiment", "This movie was a complete waste of time."),
        ("creative", "Once upon a time in a faraway land,"),
        ("qa", "What is the capital of France?"),
    ]
    
    for task_name, text in test_requests:
        start = time.time()
        result = engine.serve(task_name, text, max_new_tokens=30)
        elapsed = time.time() - start
        print(f"\n  Task: {task_name}")
        print(f"  Input: '{text}'")
        print(f"  Output: '{result[:100]}...'")
        print(f"  Time: {elapsed*1000:.1f}ms")
    
    # Batch serving (different tasks in same batch!)
    print(f"\n\n  ─── Batch Serving (Mixed Tasks!) ───")
    batch_requests = [
        ("sentiment", "I love this product!"),
        ("creative", "The robot opened its eyes for the first time"),
        ("qa", "Who wrote Romeo and Juliet?"),
    ]
    
    start = time.time()
    results = engine.serve_batch(batch_requests, max_new_tokens=25)
    elapsed = time.time() - start
    
    for (task, text), result in zip(batch_requests, results):
        print(f"  [{task:>10}] '{text}' → '{result[:80]}...'")
    
    print(f"\n  Batch time: {elapsed*1000:.1f}ms for {len(batch_requests)} requests")
    print(f"  (Different tasks processed in a single forward pass!)")
    
    del engine
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================================
# SECTION 3: PRODUCTION DEPLOYMENT PATTERNS
# ============================================================================

def production_patterns():
    """Production deployment patterns for prompt tuning."""
    print("\n\n" + "=" * 65)
    print("  SECTION 3: PRODUCTION DEPLOYMENT PATTERNS")
    print("=" * 65)
    
    patterns = """
  ═══════════════════════════════════════════════════════════════
  PATTERN 1: PROMPT VERSIONING & A/B TESTING
  ═══════════════════════════════════════════════════════════════
  
  prompt_store/
  ├── sentiment/
  │   ├── v1.0/
  │   │   ├── prompt.pt          # 60 KB
  │   │   └── config.json        # training config
  │   ├── v1.1/                   # improved version
  │   │   ├── prompt.pt
  │   │   └── config.json
  │   └── v2.0/                   # major update
  │       ├── prompt.pt
  │       └── config.json
  ├── summarize/
  │   └── v1.0/
  │       └── prompt.pt
  └── translate_en_fr/
      └── v1.0/
          └── prompt.pt
  
  Total storage for 10 tasks × 5 versions = ~3 MB!
  (Compare: LoRA × 50 versions = ~300 MB)
  (Compare: Full FT × 50 = ~250 GB)
  
  
  ═══════════════════════════════════════════════════════════════
  PATTERN 2: PERSONALIZED MODELS
  ═══════════════════════════════════════════════════════════════
  
  Each user gets their own soft prompt (< 100 KB each):
  
  ┌──────────────┐
  │  Shared LLM  │  ← One large model (7B+ params)
  │  (frozen)    │
  └──────┬───────┘
         │
    ┌────┼────────────┬────────────────┐
    │    │            │                │
  [User A's]    [User B's]      [User C's]
  [prompt  ]    [prompt  ]      [prompt  ]
  [60 KB   ]    [60 KB   ]      [60 KB   ]
  
  1 million users × 60 KB = only 60 GB of prompts!
  All sharing one base model in memory.
  
  
  ═══════════════════════════════════════════════════════════════
  PATTERN 3: DYNAMIC TASK ROUTING
  ═══════════════════════════════════════════════════════════════
  
  Request → Task Classifier → Select Prompt → Generate
  
  ```python
  class PromptRouter:
      def __init__(self, prompts_dict):
          self.prompts = prompts_dict
          self.classifier = TaskClassifier()
      
      def route(self, text):
          task = self.classifier.predict(text)
          prompt = self.prompts[task]
          return generate_with_prompt(text, prompt)
  ```
  
  The classifier can be a tiny model or even keyword-based,
  since the heavy lifting is done by prompt selection.
  
  
  ═══════════════════════════════════════════════════════════════
  PATTERN 4: PROMPT CACHING
  ═══════════════════════════════════════════════════════════════
  
  Since soft prompts are tiny tensors, cache aggressively:
  
  [Request] → Check cache → [Cached prompt] → Forward
                  ↓ miss
           Load from disk (< 1ms) → Cache → Forward
  
  Cache hit rate approaches 100% for known tasks.
  Miss penalty is negligible (loading 60 KB).
"""
    print(patterns)


# ============================================================================
# SECTION 4: COST ANALYSIS
# ============================================================================

def cost_analysis():
    """Cost comparison: prompt tuning vs other methods at scale."""
    print("\n" + "=" * 65)
    print("  SECTION 4: COST ANALYSIS")
    print("=" * 65)
    
    analysis = """
  ═══════════════════════════════════════════════════════════════
  COST OF SUPPORTING 100 TASKS
  ═══════════════════════════════════════════════════════════════
  
  Assumptions:
  • Base model: 7B parameters (Llama-2)
  • GPU: A100 80GB ($2/hr spot)
  • Storage: S3 ($0.02/GB/month)
  
  ┌─────────────────┬──────────┬──────────┬──────────┬─────────┐
  │ Item            │ Full FT  │ LoRA     │ Prompt   │ Winner  │
  ├─────────────────┼──────────┼──────────┼──────────┼─────────┤
  │ Training time   │ 100×40h  │ 100×8h   │ 100×4h   │ Prompt  │
  │ Training cost   │ $8,000   │ $1,600   │ $800     │ Prompt  │
  │ GPU for train   │ 4× A100  │ 1× A100  │ 1× A100  │ Tie     │
  │ Storage (ckpts) │ 1.4 TB   │ 6 GB     │ 6 MB     │ Prompt  │
  │ Storage cost/mo │ $28      │ $0.12    │ $0.0001  │ Prompt  │
  │ Serving GPUs    │ 100×1    │ 1 (swap) │ 1 (swap) │ Tie     │
  │ Serving cost/mo │ $144K    │ $1,440   │ $1,440   │ Tie     │
  │ Task switch     │ minutes  │ ~50ms    │ < 1ms    │ Prompt  │
  │ Quality (7B)    │ ★★★★★    │ ★★★★     │ ★★★      │ Full FT │
  │ Quality (70B)   │ ★★★★★    │ ★★★★★    │ ★★★★★    │ Tie     │
  └─────────────────┴──────────┴──────────┴──────────┴─────────┘
  
  KEY INSIGHT: At large scale (70B+), prompt tuning wins on
  nearly every metric while matching quality!
  
  
  ═══════════════════════════════════════════════════════════════
  BREAK-EVEN ANALYSIS
  ═══════════════════════════════════════════════════════════════
  
  When does prompt tuning become the best choice?
  
  Decision factors:
  ┌────────────────────┬───────────────────────────────────────┐
  │ Factor             │ Recommendation                       │
  ├────────────────────┼───────────────────────────────────────┤
  │ Model < 1B         │ Use LoRA (prompt tuning too weak)     │
  │ Model 1-7B         │ LoRA default, prompt if #tasks > 10  │
  │ Model 7B+          │ Prompt tuning viable for most tasks   │
  │ Model 13B+         │ Prompt tuning excellent choice        │
  │ Many tasks (20+)   │ Prompt tuning for storage/switching   │
  │ Single task        │ LoRA (best quality per param)         │
  │ Real-time switch   │ Prompt tuning (< 1ms swap)            │
  │ Batch mixed tasks  │ Prompt tuning (unique capability!)    │
  └────────────────────┴───────────────────────────────────────┘
"""
    print(analysis)


# ============================================================================
# SECTION 5: END-TO-END PRODUCTION EXAMPLE
# ============================================================================

def end_to_end_example():
    """Complete production example: train → save → deploy."""
    print("\n" + "=" * 65)
    print("  SECTION 5: END-TO-END PRODUCTION EXAMPLE")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import get_peft_model, PeftModel, PromptTuningConfig, PromptTuningInit, TaskType
    from datasets import load_dataset
    from trl import SFTTrainer
    
    model_name = "distilgpt2"
    
    # ─── Phase 1: Train ───
    print(f"\n  PHASE 1: TRAINING")
    print(f"  ─────────────────")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
        prompt_tuning_init=PromptTuningInit.TEXT,
        prompt_tuning_init_text="Generate an inspirational and motivational quote:",
        tokenizer_name_or_path=model_name,
    )
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    # Dataset
    dataset = load_dataset("Abirate/english_quotes", split="train")
    dataset = dataset.map(
        lambda x: {"text": f"Quote: \"{x.get('quote', '')}\""},
        remove_columns=dataset.column_names,
    )
    split = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Train
    training_args = TrainingArguments(
        output_dir="./prod_prompt_tuning",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=3e-2,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
    )
    
    result = trainer.train()
    print(f"  Train loss: {result.training_loss:.4f}")
    
    eval_result = trainer.evaluate()
    print(f"  Eval loss: {eval_result['eval_loss']:.4f}")
    
    # ─── Phase 2: Save ───
    print(f"\n  PHASE 2: SAVING")
    print(f"  ────────────────")
    
    save_dir = "./prod_prompt_checkpoint"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    import os
    total_size = sum(
        os.path.getsize(os.path.join(save_dir, f))
        for f in os.listdir(save_dir)
        if os.path.isfile(os.path.join(save_dir, f))
    )
    print(f"  Saved to: {save_dir}")
    print(f"  Checkpoint size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    
    # ─── Phase 3: Deploy ───
    print(f"\n  PHASE 3: DEPLOYMENT")
    print(f"  ────────────────────")
    
    # Simulate production: load fresh model + saved prompt
    del model, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    prod_model = AutoModelForCausalLM.from_pretrained(model_name)
    prod_model.config.pad_token_id = tokenizer.pad_token_id
    prod_model = PeftModel.from_pretrained(prod_model, save_dir)
    prod_model.eval()
    
    print(f"  Production model loaded!")
    
    # Serve requests
    test_prompts = [
        "Life is",
        "The key to success",
        "In difficult times",
    ]
    
    print(f"\n  ─── Serving Requests ───")
    
    total_time = 0
    for prompt_text in test_prompts:
        inputs = tokenizer(prompt_text, return_tensors="pt")
        
        start = time.time()
        with torch.no_grad():
            outputs = prod_model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.time() - start
        total_time += elapsed
        
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n  Input: '{prompt_text}'")
        print(f"  Output: '{text[:120]}'")
        print(f"  Latency: {elapsed*1000:.1f}ms")
    
    print(f"\n  Average latency: {total_time/len(test_prompts)*1000:.1f}ms")
    
    # Clean up
    import shutil
    for dir_name in ["./prod_prompt_tuning", "./prod_prompt_checkpoint"]:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
    
    print(f"""
  ═══ Production Pipeline Summary ═══
  
  1. TRAIN:    ~4 hours for 7B model (1 A100)
  2. SAVE:     ~60 KB checkpoint  
  3. DEPLOY:   Load base model + attach prompt
  4. SERVE:    Standard inference + tiny prompt prepend
  5. SWITCH:   Swap prompt tensor (< 1ms)
  6. SCALE:    Add new tasks = train new prompts
  
  Zero model copies. Zero infrastructure changes.
  Just tiny prompt files that steer one shared model.
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all scaling and production demonstrations."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     PROMPT TUNING — SCALING & PRODUCTION                     ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: Scaling laws
    scaling_laws()
    
    # Section 2: Multi-task serving
    demonstrate_multi_task_serving()
    
    # Section 3: Production patterns
    production_patterns()
    
    # Section 4: Cost analysis
    cost_analysis()
    
    # Section 5: End-to-end
    end_to_end_example()
    
    print("\n" + "=" * 65)
    print("  MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered:
    ✓ Scaling laws (model size vs prompt tuning quality)
    ✓ Multi-task serving engine (one model, many prompts)
    ✓ Production deployment patterns
    ✓ Cost analysis (prompt vs LoRA vs full FT)
    ✓ End-to-end production pipeline
    """)


if __name__ == "__main__":
    main()
