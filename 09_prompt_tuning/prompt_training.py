"""
Prompt Tuning — Training with HuggingFace PEFT
================================================

Production-ready training pipeline using the PEFT library:

1. PromptTuningConfig Deep Dive
   - Every parameter explained
   - Config for different tasks

2. Complete Training Pipeline
   - Dataset preparation
   - Model setup with PEFT
   - SFTTrainer integration

3. Save, Load, and Inference
   - Saving tiny prompt checkpoints
   - Loading for inference
   - Batch inference with prompts

4. Hyperparameter Guide
   - Learning rate (critical!)
   - Prompt length selection
   - Training stability tips

Author: LLM Fine-Tuning Masterclass
"""

import torch
import os
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# SECTION 1: PROMPTTUNINGCONFIG DEEP DIVE
# ============================================================================

def explain_prompt_tuning_config():
    """
    Detailed walkthrough of every PromptTuningConfig parameter.
    """
    print("=" * 65)
    print("  SECTION 1: PROMPTTUNINGCONFIG DEEP DIVE")
    print("=" * 65)
    
    from peft import PromptTuningConfig, PromptTuningInit, TaskType
    
    # ─── Configuration 1: Random Init (Simplest) ───
    config_random = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        
        # Number of soft prompt tokens to prepend
        # More tokens = more parameters = more expressive
        # Typical range: 5-100, sweet spot: 10-30
        num_virtual_tokens=20,
        
        # How to initialize the soft prompt
        # RANDOM = sample from normal distribution
        prompt_tuning_init=PromptTuningInit.RANDOM,
    )
    
    print(f"\n  Config 1: Random Initialization")
    print(f"  ─────────────────────────────────")
    print(f"  task_type: {config_random.task_type}")
    print(f"  num_virtual_tokens: {config_random.num_virtual_tokens}")
    print(f"  prompt_tuning_init: {config_random.prompt_tuning_init}")
    
    # ─── Configuration 2: Text Init (Recommended!) ───
    config_text = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
        
        # TEXT init — uses actual text tokens as starting point
        # This dramatically improves performance!
        prompt_tuning_init=PromptTuningInit.TEXT,
        
        # The initialization text — tokenized and embedded
        # Choose text relevant to your task!
        prompt_tuning_init_text="Classify the following text as positive or negative:",
        
        # The tokenizer (needed to tokenize init_text)
        tokenizer_name_or_path="distilgpt2",
    )
    
    print(f"\n  Config 2: Text Initialization ★ RECOMMENDED")
    print(f"  ────────────────────────────────────────────")
    print(f"  prompt_tuning_init: {config_text.prompt_tuning_init}")
    print(f"  init_text: '{config_text.prompt_tuning_init_text}'")
    print(f"  tokenizer: {config_text.tokenizer_name_or_path}")
    
    # ─── Configuration 3: For Sequence Classification ───
    config_seq_cls = PromptTuningConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=10,
        prompt_tuning_init=PromptTuningInit.TEXT,
        prompt_tuning_init_text="Determine the sentiment of this review:",
        tokenizer_name_or_path="distilgpt2",
    )
    
    print(f"\n  Config 3: Sequence Classification")
    print(f"  ───────────────────────────────────")
    print(f"  task_type: {config_seq_cls.task_type}")
    print(f"  (Used for classification heads)")
    
    detail = """
  ═══ Parameter Reference ═══
  
  ┌─────────────────────────────┬──────────────────────────────┐
  │ Parameter                   │ Description                  │
  ├─────────────────────────────┼──────────────────────────────┤
  │ task_type                   │ CAUSAL_LM, SEQ_CLS, SEQ_2_  │
  │                             │ SEQ_CLS, TOKEN_CLS           │
  │ num_virtual_tokens          │ # soft tokens (5-100)        │
  │ prompt_tuning_init          │ RANDOM or TEXT                │
  │ prompt_tuning_init_text     │ Text for TEXT init            │
  │ tokenizer_name_or_path      │ Tokenizer for TEXT init      │
  └─────────────────────────────┴──────────────────────────────┘
  
  Note: Prompt tuning has NO reparameterization MLP!
  (Unlike prefix tuning which uses an MLP for stability)
  
  This means:
  • Fewer parameters than prefix tuning
  • More sensitive to learning rate
  • Text init is crucial for good results
"""
    print(detail)
    
    return config_text


# ============================================================================
# SECTION 2: COMPLETE TRAINING PIPELINE
# ============================================================================

def train_prompt_tuning():
    """
    Full training pipeline: dataset → model → train → evaluate.
    """
    print("\n" + "=" * 65)
    print("  SECTION 2: COMPLETE TRAINING PIPELINE")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import get_peft_model, PromptTuningConfig, PromptTuningInit, TaskType
    from datasets import load_dataset
    from trl import SFTTrainer
    
    model_name = "distilgpt2"
    
    # ─── Step 1: Load Tokenizer ───
    print("\n  Step 1: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # ─── Step 2: Load Base Model ───
    print("  Step 2: Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    total_before = sum(p.numel() for p in model.parameters())
    print(f"  Base model: {total_before:,} parameters")
    
    # ─── Step 3: Configure Prompt Tuning ───
    print("  Step 3: Configuring prompt tuning...")
    
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
        prompt_tuning_init=PromptTuningInit.TEXT,
        prompt_tuning_init_text=(
            "Write a famous inspirational quote about life and wisdom:"
        ),
        tokenizer_name_or_path=model_name,
    )
    
    # ─── Step 4: Create PEFT Model ───
    print("  Step 4: Creating PEFT model...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Trainable: {trainable:,} / {total:,} "
          f"({trainable/total*100:.4f}%)")
    
    # ─── Step 5: Prepare Dataset ───
    print("\n  Step 5: Preparing dataset...")
    dataset = load_dataset("Abirate/english_quotes", split="train")
    dataset = dataset.map(
        lambda x: {"text": f"Quote: \"{x.get('quote', '')}\" — {x.get('author', 'Unknown')}"},
        remove_columns=dataset.column_names,
    )
    split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"  Train: {len(split['train'])} examples")
    print(f"  Eval:  {len(split['test'])} examples")
    print(f"  Example: {split['train'][0]['text'][:80]}...")
    
    # ─── Step 6: Training Arguments ───
    print("\n  Step 6: Setting up training...")
    
    training_args = TrainingArguments(
        output_dir="./prompt_tuning_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        
        # CRITICAL: Prompt tuning needs HIGH learning rate!
        # This is because the soft prompt is only ~15K params
        # and receives diluted gradients through frozen layers.
        learning_rate=3e-2,            # Much higher than LoRA's 1e-4!
        
        warmup_ratio=0.1,             # Longer warmup helps stability
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
    )
    
    # ─── Step 7: Create Trainer and Train ───
    print("  Step 7: Training...")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
    )
    
    train_result = trainer.train()
    
    print(f"\n  ─── Training Results ───")
    print(f"  Final train loss: {train_result.training_loss:.4f}")
    
    # Evaluate
    eval_result = trainer.evaluate()
    print(f"  Eval loss:        {eval_result['eval_loss']:.4f}")
    print(f"  Eval perplexity:  {torch.exp(torch.tensor(eval_result['eval_loss'])):.2f}")
    
    return model, tokenizer, trainer


# ============================================================================
# SECTION 3: SAVE, LOAD, AND INFERENCE
# ============================================================================

def demonstrate_save_load_inference(model=None, tokenizer=None):
    """
    Show how to save and load prompt tuning checkpoints.
    Prompt tuning saves are TINY — just the soft prompt vectors.
    """
    print("\n\n" + "=" * 65)
    print("  SECTION 3: SAVE, LOAD, AND INFERENCE")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, get_peft_model, PromptTuningConfig, PromptTuningInit, TaskType
    
    model_name = "distilgpt2"
    save_path = "./prompt_tuning_saved"
    
    # If no model provided, create a quick one
    if model is None:
        print("\n  Creating model for demonstration...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        base_model.config.pad_token_id = tokenizer.pad_token_id
        
        config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,
            prompt_tuning_init=PromptTuningInit.TEXT,
            prompt_tuning_init_text="Generate an inspirational quote about life:",
            tokenizer_name_or_path=model_name,
        )
        model = get_peft_model(base_model, config)
    
    # ─── Save ───
    print(f"\n  Saving prompt tuning checkpoint...")
    model.save_pretrained(save_path)
    
    # Show what was saved
    if os.path.exists(save_path):
        saved_files = os.listdir(save_path)
        total_size = sum(
            os.path.getsize(os.path.join(save_path, f))
            for f in saved_files
            if os.path.isfile(os.path.join(save_path, f))
        )
        print(f"  Saved files:")
        for f in sorted(saved_files):
            fpath = os.path.join(save_path, f)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath)
                print(f"    {f}: {size:,} bytes")
        print(f"  Total save size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
        print(f"\n  Compare: Full model = ~500 MB; Prompt = ~{total_size/1024:.0f} KB")
        print(f"  That's a {500*1024*1024 / max(total_size,1):.0f}× size reduction!")
    
    # ─── Load ───
    print(f"\n  Loading from checkpoint...")
    
    # Load fresh base model
    fresh_model = AutoModelForCausalLM.from_pretrained(model_name)
    fresh_model.config.pad_token_id = tokenizer.pad_token_id
    
    # Load PEFT adapter (soft prompt)
    loaded_model = PeftModel.from_pretrained(fresh_model, save_path)
    loaded_model.eval()
    print(f"  ✓ Model loaded successfully!")
    
    # ─── Inference ───
    print(f"\n  Generating with loaded model:")
    
    prompts = [
        "The meaning of life is",
        "A wise person once said",
        "In the pursuit of happiness",
    ]
    
    for prompt_text in prompts:
        inputs = tokenizer(prompt_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = loaded_model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n  Input:  '{prompt_text}'")
        print(f"  Output: '{generated[:120]}...'")
    
    # ─── Batch Inference ───
    print(f"\n\n  ─── Batch Inference ───")
    print(f"  (Process multiple inputs in one forward pass)")
    
    tokenizer.padding_side = "left"
    batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = loaded_model.generate(
            **batch,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    print(f"  Batch of {len(prompts)} processed simultaneously")
    for i, output in enumerate(outputs):
        text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"  [{i}] {text[:100]}...")
    
    # Clean up
    import shutil
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    
    return loaded_model


# ============================================================================
# SECTION 4: HYPERPARAMETER GUIDE
# ============================================================================

def hyperparameter_guide():
    """
    Critical hyperparameter guidance for prompt tuning.
    """
    print("\n\n" + "=" * 65)
    print("  SECTION 4: HYPERPARAMETER GUIDE")
    print("=" * 65)
    
    guide = """
  ═══════════════════════════════════════════════════════════════
  CRITICAL HYPERPARAMETERS FOR PROMPT TUNING
  ═══════════════════════════════════════════════════════════════
  
  
  1. LEARNING RATE — The Most Important Hyperparameter
  ─────────────────────────────────────────────────────────────
  
  Prompt tuning requires MUCH higher LR than other methods:
  
  ┌─────────────────────┬────────────────────┬────────────────┐
  │ Method              │ Typical LR Range   │ Recommended    │
  ├─────────────────────┼────────────────────┼────────────────┤
  │ Full Fine-Tuning    │ 1e-5 — 5e-5        │ 2e-5           │
  │ LoRA                │ 1e-4 — 5e-4        │ 2e-4           │
  │ Prefix Tuning       │ 1e-2 — 5e-2        │ 3e-2           │
  │ ★ Prompt Tuning     │ 1e-1 — 5e-1        │ 3e-1           │
  └─────────────────────┴────────────────────┴────────────────┘
  
  Why so high?
  • Only ~15K parameters to update  
  • Gradients pass through many frozen layers (signal diluted)
  • Need large steps to make meaningful changes
  
  WARNING: Using LoRA's LR (1e-4) for prompt tuning will
  result in essentially zero learning. The model will look
  like it "didn't train at all."
  
  
  2. PROMPT LENGTH
  ─────────────────────────────────────────────────────────────
  
  ┌─────────────────────┬──────────────┬─────────────────────┐
  │ Task Type           │ Prompt Len   │ Notes               │
  ├─────────────────────┼──────────────┼─────────────────────┤
  │ Binary class.       │ 1-10         │ Very few needed     │
  │ Multi-class         │ 10-20        │ More labels → more  │
  │ Generation          │ 20-50        │ Standard range      │
  │ Complex reasoning   │ 50-100       │ Diminishing returns │
  └─────────────────────┴──────────────┴─────────────────────┘
  
  Rule of thumb: Start with 20, adjust if needed.
  More tokens = more context consumed = shorter inputs.
  
  
  3. INITIALIZATION
  ─────────────────────────────────────────────────────────────
  
  Always use TEXT initialization when possible!
  
  Performance impact (SuperGLUE, T5-XXL):
    Random init:  91.0%
    Vocab init:   92.1%  (+1.1)
    Text init:    92.7%  (+1.7)   ← Use this!
  
  Good init texts by task:
    Classification: "Classify the text as {labels}:"
    Summarization:  "Summarize the following text:"
    Translation:    "Translate from English to French:"
    QA:             "Answer the following question:"
  
  
  4. WARMUP & SCHEDULING
  ─────────────────────────────────────────────────────────────
  
  • Warmup ratio: 0.06-0.10 (longer than typical)
  • Scheduler: Cosine or linear works well
  • Gradient clipping: 1.0 (important for stability)
  
  
  5. COMMON PITFALLS
  ─────────────────────────────────────────────────────────────
  
  ✗ Using too low a learning rate → nothing happens
  ✗ Random init on small models → very poor results
  ✗ Too many prompt tokens → wasted context, no benefit
  ✗ No warmup → training explodes early
  ✗ Too few training steps → prompt didn't converge
"""
    print(guide)
    
    # Simulated LR sweep
    print("  ─── Learning Rate Sweep (simulated) ───")
    print(f"  {'LR':>10} {'Final Loss':>12} {'Note':>25}")
    print(f"  {'─'*10}─{'─'*12}─{'─'*25}")
    
    lr_results = [
        (1e-5, 8.95, "Too low — no learning"),
        (1e-4, 8.80, "Too low — minimal learning"),
        (1e-3, 7.50, "Slow convergence"),
        (1e-2, 5.20, "Good but could be better"),
        (3e-2, 4.10, "★ Good sweet spot"),
        (1e-1, 3.85, "★ Often optimal"),
        (3e-1, 3.90, "Good for large models"),
        (1e-0, 6.50, "Too high — unstable"),
        (3e-0, 9.00, "Diverges"),
    ]
    
    for lr, loss, note in lr_results:
        bar = "█" * max(1, int((9.0 - loss) * 3))
        print(f"  {lr:>10.0e}  {loss:>10.2f}  {note:<25}  {bar}")
    
    # Complete recommended config
    print(f"""
  
  ═══ Complete Recommended Configuration ═══
  
  ```python
  from peft import PromptTuningConfig, PromptTuningInit, TaskType
  
  peft_config = PromptTuningConfig(
      task_type=TaskType.CAUSAL_LM,
      num_virtual_tokens=20,
      prompt_tuning_init=PromptTuningInit.TEXT,
      prompt_tuning_init_text="Your task-specific init text here",
      tokenizer_name_or_path="your-model-name",
  )
  
  training_args = TrainingArguments(
      learning_rate=3e-1,         # HIGH for prompt tuning!
      warmup_ratio=0.1,
      lr_scheduler_type="cosine",
      num_train_epochs=5,         # May need more epochs
      per_device_train_batch_size=8,
      gradient_accumulation_steps=4,
      max_grad_norm=1.0,
      weight_decay=0.01,
  )
  ```
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the complete training pipeline."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     PROMPT TUNING — TRAINING WITH PEFT                       ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: Config deep dive
    explain_prompt_tuning_config()
    
    # Section 2: Training pipeline
    model, tokenizer, trainer = train_prompt_tuning()
    
    # Section 3: Save, load, inference
    demonstrate_save_load_inference(model, tokenizer)
    
    # Section 4: Hyperparameter guide
    hyperparameter_guide()
    
    print("\n" + "=" * 65)
    print("  MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered:
    ✓ PromptTuningConfig — every parameter explained
    ✓ Full training pipeline with SFTTrainer
    ✓ Save & load (tiny checkpoint files!)
    ✓ Inference with loaded model
    ✓ Critical hyperparameter guidance
    """)


if __name__ == "__main__":
    main()
