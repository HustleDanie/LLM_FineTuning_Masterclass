"""
Prompt Tuning — Advanced Techniques
=====================================

Advanced prompt tuning concepts and techniques:

1. Prompt Ensembling
   - Average multiple prompts for robustness
   - Majority voting with prompt ensembles
   - Diversity in prompt initialization

2. Prompt Transfer
   - Task-to-task transfer of soft prompts
   - Cross-domain transfer
   - SPoT: Soft Prompt Transfer

3. Prompt Decomposition
   - Factored soft prompts (low-rank)
   - Shared + task-specific components
   - Parameter efficiency at the extreme

4. Prompt Tuning + Quantization
   - 8-bit/4-bit base model with soft prompt
   - Memory-efficient inference
   - Practical deployment pattern

5. Analysis & Interpretability
   - What do soft prompts "mean"?
   - Nearest-token analysis
   - Attention pattern visualization

Author: LLM Fine-Tuning Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple


# ============================================================================
# SECTION 1: PROMPT ENSEMBLING
# ============================================================================

class PromptEnsemble(nn.Module):
    """
    Train multiple soft prompts and ensemble them for better performance.
    
    Ensembling reduces variance and improves robustness.
    Because soft prompts are SO tiny, ensembling N prompts
    still uses far fewer parameters than a single LoRA adapter.
    
    Strategies:
    1. Average: Mean of prompt embeddings
    2. Weighted: Learned weights per prompt
    3. Vote: Each prompt predicts independently, take majority
    """
    
    def __init__(
        self,
        num_prompts: int = 5,
        num_tokens: int = 20,
        d_model: int = 768,
        ensemble_strategy: str = "weighted",
    ):
        super().__init__()
        
        self.num_prompts = num_prompts
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.strategy = ensemble_strategy
        
        # Multiple independent soft prompts
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.randn(num_tokens, d_model) * 0.02)
            for _ in range(num_prompts)
        ])
        
        # Learned weights for weighted ensemble
        if ensemble_strategy == "weighted":
            self.weights = nn.Parameter(torch.ones(num_prompts) / num_prompts)
        
        total_params = num_prompts * num_tokens * d_model
        if ensemble_strategy == "weighted":
            total_params += num_prompts
        print(f"  PromptEnsemble: {num_prompts} prompts × {num_tokens} tokens")
        print(f"  Total params: {total_params:,}")
        print(f"  Strategy: {ensemble_strategy}")
    
    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Combine multiple prompts and prepend to input.
        
        For "average" and "weighted": produce a single combined prompt.
        For "vote": returns list of (prompt + input) for each prompt.
        """
        batch_size = input_embeddings.shape[0]
        
        if self.strategy == "average":
            # Simple mean of all prompts
            combined = torch.stack(list(self.prompts), dim=0).mean(dim=0)
            combined = combined.unsqueeze(0).expand(batch_size, -1, -1)
            return torch.cat([combined, input_embeddings], dim=1)
        
        elif self.strategy == "weighted":
            # Learned weighted combination
            weights = F.softmax(self.weights, dim=0)
            stacked = torch.stack(list(self.prompts), dim=0)  # [K, N, D]
            combined = (weights.view(-1, 1, 1) * stacked).sum(dim=0)  # [N, D]
            combined = combined.unsqueeze(0).expand(batch_size, -1, -1)
            return torch.cat([combined, input_embeddings], dim=1)
        
        elif self.strategy == "vote":
            # Return separate (prompt + input) for each prompt
            outputs = []
            for prompt in self.prompts:
                p = prompt.unsqueeze(0).expand(batch_size, -1, -1)
                outputs.append(torch.cat([p, input_embeddings], dim=1))
            return outputs
        
        raise ValueError(f"Unknown strategy: {self.strategy}")


def demonstrate_ensembling():
    """Show prompt ensembling in action."""
    print("=" * 65)
    print("  SECTION 1: PROMPT ENSEMBLING")
    print("=" * 65)
    
    torch.manual_seed(42)
    d_model = 128
    
    explanation = """
  ═══ Why Ensemble Soft Prompts? ═══
  
  Single prompt for task A:     15,360 params (20 × 768)
  5 prompts ensembled:          76,800 params (5 × 20 × 768)
  LoRA r=8 for same model:    294,912 params
  
  Even 5 ensembled prompts use fewer params than one LoRA!
  But ensembling provides:
  ✓ Reduced variance in predictions
  ✓ Better calibration
  ✓ Robustness to initialization
"""
    print(explanation)
    
    # Create ensembles with different strategies
    for strategy in ["average", "weighted"]:
        print(f"\n  ─── Strategy: {strategy} ───")
        ensemble = PromptEnsemble(
            num_prompts=5,
            num_tokens=10,
            d_model=d_model,
            ensemble_strategy=strategy,
        )
        
        # Forward pass
        input_embeds = torch.randn(2, 20, d_model)
        output = ensemble(input_embeds)
        print(f"  Input: {input_embeds.shape} → Output: {output.shape}")
        
        if strategy == "weighted":
            weights = F.softmax(ensemble.weights, dim=0)
            print(f"  Learned weights: {weights.detach().tolist()}")
    
    # Diversity analysis
    print(f"\n  ═══ Prompt Diversity (Important for Good Ensembles) ═══")
    ensemble = PromptEnsemble(5, 10, d_model, "average")
    
    prompts = torch.stack(list(ensemble.prompts))  # [5, 10, d]
    flat = prompts.view(5, -1)  # [5, 10*d]
    
    # Cosine similarity between prompts
    norm_flat = F.normalize(flat, dim=1)
    similarity = norm_flat @ norm_flat.T
    
    print(f"  Cosine similarity between 5 random-init prompts:")
    for i in range(5):
        row = "  "
        for j in range(5):
            row += f"  {similarity[i,j]:.3f}"
        print(row)
    
    print(f"\n  Random init produces diverse prompts (low similarity)")
    print(f"  This diversity is what makes ensembling effective!")


# ============================================================================
# SECTION 2: PROMPT TRANSFER
# ============================================================================

class PromptTransferManager:
    """
    Transfer soft prompts between tasks.
    
    Key idea (SPoT — Soft Prompt Transfer, Vu et al. 2022):
    1. Train a soft prompt on a source task
    2. Use it to initialize the prompt for a target task
    3. Fine-tune on the target task
    
    This dramatically improves performance, especially
    when the target task has limited data.
    """
    
    def __init__(self, d_model: int = 768, num_tokens: int = 20):
        self.d_model = d_model
        self.num_tokens = num_tokens
        self.prompt_bank: Dict[str, torch.Tensor] = {}
    
    def register_prompt(self, task_name: str, prompt: torch.Tensor):
        """Store a trained prompt in the bank."""
        assert prompt.shape == (self.num_tokens, self.d_model), \
            f"Expected ({self.num_tokens}, {self.d_model}), got {prompt.shape}"
        self.prompt_bank[task_name] = prompt.clone().detach()
        print(f"  Registered prompt for '{task_name}'")
    
    def transfer_direct(self, source_task: str) -> nn.Parameter:
        """
        Direct transfer: use source prompt as-is for initialization.
        Best when source and target tasks are similar.
        """
        if source_task not in self.prompt_bank:
            raise ValueError(f"No prompt for task '{source_task}'")
        
        prompt = self.prompt_bank[source_task].clone()
        return nn.Parameter(prompt)
    
    def transfer_interpolate(
        self, tasks: List[str], weights: Optional[List[float]] = None,
    ) -> nn.Parameter:
        """
        Interpolation transfer: weighted average of multiple source prompts.
        Good when target task combines aspects of multiple source tasks.
        """
        if weights is None:
            weights = [1.0 / len(tasks)] * len(tasks)
        
        assert len(tasks) == len(weights)
        assert abs(sum(weights) - 1.0) < 1e-6
        
        combined = torch.zeros(self.num_tokens, self.d_model)
        for task, w in zip(tasks, weights):
            combined += w * self.prompt_bank[task]
        
        return nn.Parameter(combined)
    
    def transfer_selective(
        self, source_task: str, target_tokens: int,
    ) -> nn.Parameter:
        """
        Selective transfer: use a subset of source prompt tokens.
        Useful when target task is simpler than source.
        """
        source = self.prompt_bank[source_task]
        # Take first target_tokens (they tend to be most important)
        selected = source[:target_tokens].clone()
        return nn.Parameter(selected)


def demonstrate_prompt_transfer():
    """Show prompt transfer techniques."""
    print("\n\n" + "=" * 65)
    print("  SECTION 2: PROMPT TRANSFER")
    print("=" * 65)
    
    torch.manual_seed(42)
    d_model = 128
    num_tokens = 20
    
    manager = PromptTransferManager(d_model=d_model, num_tokens=num_tokens)
    
    # Simulate trained prompts for various tasks
    sentiment_prompt = torch.randn(num_tokens, d_model) * 0.1
    nli_prompt = torch.randn(num_tokens, d_model) * 0.1
    qa_prompt = torch.randn(num_tokens, d_model) * 0.1
    
    manager.register_prompt("sentiment", sentiment_prompt)
    manager.register_prompt("nli", nli_prompt)
    manager.register_prompt("qa", qa_prompt)
    
    # Transfer strategies
    print(f"\n  ─── Transfer Strategy 1: Direct ───")
    direct = manager.transfer_direct("sentiment")
    print(f"  Transferred from sentiment: {direct.shape}")
    
    print(f"\n  ─── Transfer Strategy 2: Interpolation ───")
    interpolated = manager.transfer_interpolate(
        ["sentiment", "nli", "qa"], 
        [0.5, 0.3, 0.2]
    )
    print(f"  Interpolated (0.5×sent + 0.3×nli + 0.2×qa): {interpolated.shape}")
    
    print(f"\n  ─── Transfer Strategy 3: Selective ───")
    selective = manager.transfer_selective("qa", target_tokens=10)
    print(f"  Selective (first 10 tokens from qa): {selective.shape}")
    
    explanation = """
  ═══ SPoT: Soft Prompt Transfer (Vu et al., 2022) ═══
  
  Key findings:
  ┌───────────────────────────────────────────────────────────┐
  │ 1. Prompts trained on ONE task transfer to similar tasks  │
  │ 2. Transfer improves performance by 2-5 points            │
  │ 3. Especially helpful for low-resource target tasks       │
  │ 4. Best source tasks are "general" NLI-like tasks         │
  └───────────────────────────────────────────────────────────┘
  
  Transfer effectiveness (simulated):
  
    Source → Target            Random Init   Transfer Init
    ─────────────────────      ──────────    ─────────────
    NLI → Sentiment            88.5%         91.2% (+2.7)
    NLI → Paraphrase           85.0%         88.8% (+3.8)
    Sentiment → NLI            82.0%         83.5% (+1.5)
    QA → Summarization         78.0%         82.1% (+4.1)
    
  NLI (Natural Language Inference) is the best "universal
  source task" — it teaches text understanding broadly.
"""
    print(explanation)


# ============================================================================
# SECTION 3: PROMPT DECOMPOSITION
# ============================================================================

class FactoredSoftPrompt(nn.Module):
    """
    Low-rank factored soft prompt for extreme parameter efficiency.
    
    Instead of learning a full [num_tokens × d_model] matrix,
    factorize it as a product of two smaller matrices:
    
    Standard: P ∈ R^{N×D}           params = N × D
    Factored: P = A × B             params = N × r + r × D
              A ∈ R^{N×r}, B ∈ R^{r×D}
    
    When r << min(N, D), this is much more efficient.
    """
    
    def __init__(
        self,
        num_tokens: int = 20,
        d_model: int = 768,
        rank: int = 4,
    ):
        super().__init__()
        
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.rank = rank
        
        # Factored representation
        self.factor_a = nn.Parameter(torch.randn(num_tokens, rank) * 0.02)
        self.factor_b = nn.Parameter(torch.randn(rank, d_model) * 0.02)
        
        full_params = num_tokens * d_model
        factored_params = num_tokens * rank + rank * d_model
        
        print(f"  FactoredSoftPrompt:")
        print(f"    Full params:     {full_params:,}")
        print(f"    Factored params: {factored_params:,}")
        print(f"    Reduction: {full_params / factored_params:.1f}×")
    
    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute prompt via matrix multiplication, then prepend."""
        batch_size = input_embeddings.shape[0]
        
        # Compute full prompt: [N, r] × [r, D] = [N, D]
        prompt = self.factor_a @ self.factor_b
        prompt = prompt.unsqueeze(0).expand(batch_size, -1, -1)
        
        return torch.cat([prompt, input_embeddings], dim=1)


class SharedTaskPrompt(nn.Module):
    """
    Shared + task-specific prompt decomposition.
    
    Architecture:
    ┌────────────────────┬────────────────────────┐
    │   Shared Prompt    │   Task-Specific Prompt  │
    │   (frozen after    │   (trained per task)    │
    │   pretraining)     │                         │
    │   [s₁...sₖ]       │   [t₁...tₘ]            │
    └────────────────────┴────────────────────────┘
    
    Combined prompt = [shared; task_specific; input]
    
    The shared portion captures general "understand this task" 
    knowledge, while task-specific tokens handle each task.
    """
    
    def __init__(
        self,
        num_shared: int = 10,
        num_task_specific: int = 10,
        d_model: int = 768,
        num_tasks: int = 3,
    ):
        super().__init__()
        
        self.num_shared = num_shared
        self.num_task = num_task_specific
        
        # Shared prompt (pretrained, could be frozen later)
        self.shared_prompt = nn.Parameter(
            torch.randn(num_shared, d_model) * 0.02
        )
        
        # Task-specific prompts
        self.task_prompts = nn.ParameterDict({
            f"task_{i}": nn.Parameter(
                torch.randn(num_task_specific, d_model) * 0.02
            )
            for i in range(num_tasks)
        })
        
        shared_params = num_shared * d_model
        per_task_params = num_task_specific * d_model
        total = shared_params + num_tasks * per_task_params
        
        print(f"\n  SharedTaskPrompt:")
        print(f"    Shared: {num_shared} tokens ({shared_params:,} params)")
        print(f"    Per task: {num_task_specific} tokens ({per_task_params:,} params)")
        print(f"    Total for {num_tasks} tasks: {total:,} params")
    
    def forward(
        self, input_embeddings: torch.Tensor, task_id: int = 0,
    ) -> torch.Tensor:
        """Combine shared + task-specific prompt with input."""
        batch_size = input_embeddings.shape[0]
        
        shared = self.shared_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        task = self.task_prompts[f"task_{task_id}"].unsqueeze(0).expand(batch_size, -1, -1)
        
        # [shared; task_specific; input]
        return torch.cat([shared, task, input_embeddings], dim=1)


def demonstrate_decomposition():
    """Show prompt decomposition techniques."""
    print("\n\n" + "=" * 65)
    print("  SECTION 3: PROMPT DECOMPOSITION")
    print("=" * 65)
    
    torch.manual_seed(42)
    d_model = 768
    
    # Factored prompt
    print(f"\n  ─── Low-Rank Factored Prompt ───")
    for rank in [1, 2, 4, 8]:
        factored = FactoredSoftPrompt(
            num_tokens=20, d_model=d_model, rank=rank
        )
        params = sum(p.numel() for p in factored.parameters())
        full = 20 * d_model
        print(f"    Rank {rank}: {params:,} params ({params/full*100:.1f}% of full)")
    
    # Shared + task-specific
    print(f"\n  ─── Shared + Task-Specific ───")
    shared_task = SharedTaskPrompt(
        num_shared=10, num_task_specific=5,
        d_model=d_model, num_tasks=5
    )
    
    input_embeds = torch.randn(2, 20, d_model)
    for task_id in range(3):
        output = shared_task(input_embeds, task_id=task_id)
        print(f"  Task {task_id}: input {input_embeds.shape} → {output.shape}")


# ============================================================================
# SECTION 4: PROMPT TUNING + QUANTIZATION
# ============================================================================

def prompt_tuning_quantized():
    """
    Combining prompt tuning with model quantization.
    This is the most memory-efficient fine-tuning possible.
    """
    print("\n\n" + "=" * 65)
    print("  SECTION 4: PROMPT TUNING + QUANTIZATION")
    print("=" * 65)
    
    explanation = """
  ═══ The Ultimate Efficiency Combo ═══
  
  Prompt tuning trains the FEWEST parameters.
  Quantization uses the LEAST memory for base model.
  Combined = maximum efficiency!
  
  Memory comparison (7B model):
  ┌──────────────────────┬────────────┬──────────┬──────────┐
  │ Configuration        │ Model Mem  │ Trainable│ Total    │
  ├──────────────────────┼────────────┼──────────┼──────────┤
  │ Full FT (fp32)       │ 28 GB      │ 7B       │ ~84 GB   │
  │ Full FT (fp16)       │ 14 GB      │ 7B       │ ~42 GB   │
  │ LoRA (fp16)          │ 14 GB      │ ~4M      │ ~15 GB   │
  │ QLoRA (4-bit)        │ 3.5 GB     │ ~4M      │ ~5 GB    │
  │ Prompt (fp16)        │ 14 GB      │ ~60K     │ ~14 GB   │
  │ ★ Prompt (4-bit)     │ 3.5 GB     │ ~60K     │ ~3.5 GB  │
  └──────────────────────┴────────────┴──────────┴──────────┘
  
  4-bit prompt tuning uses ~3.5 GB for a 7B model!
  That's less than QLoRA because prompt has fewer trainable params.
"""
    print(explanation)
    
    # Show the implementation pattern
    code = """
  ═══ Implementation Pattern ═══
  
  ```python
  from transformers import AutoModelForCausalLM, BitsAndBytesConfig
  from peft import get_peft_model, PromptTuningConfig, PromptTuningInit
  
  # Step 1: Load model in 4-bit
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.float16,
  )
  
  model = AutoModelForCausalLM.from_pretrained(
      "meta-llama/Llama-2-7b-hf",
      quantization_config=bnb_config,
      device_map="auto",
  )
  
  # Step 2: Add prompt tuning (fp32 soft prompt on 4-bit model)
  peft_config = PromptTuningConfig(
      task_type="CAUSAL_LM",
      num_virtual_tokens=20,
      prompt_tuning_init=PromptTuningInit.TEXT,
      prompt_tuning_init_text="Summarize the text:",
      tokenizer_name_or_path="meta-llama/Llama-2-7b-hf",
  )
  
  model = get_peft_model(model, peft_config)
  # Now: 4-bit base (3.5 GB) + fp32 prompt (~80 KB)
  # Total trainable: 20 × 4096 = 81,920 params!
  ```
  
  The soft prompt is kept in fp32 for training precision,
  while the base model is quantized to 4-bit.
  This is the most memory-efficient fine-tuning setup possible.
"""
    print(code)


# ============================================================================
# SECTION 5: ANALYSIS & INTERPRETABILITY
# ============================================================================

class PromptAnalyzer:
    """
    Tools for understanding what soft prompts have learned.
    """
    
    @staticmethod
    def nearest_tokens(
        soft_prompt: torch.Tensor,
        embedding_matrix: torch.Tensor,
        tokenizer,
        top_k: int = 5,
    ):
        """
        Find the nearest vocabulary tokens to each soft prompt vector.
        
        This gives us a "human-readable" approximation of what
        the soft prompt represents, though soft prompts typically
        settle in between tokens (no exact match).
        """
        num_tokens = soft_prompt.shape[0]
        
        results = []
        for i in range(num_tokens):
            # Cosine similarity to all vocab tokens
            prompt_vec = soft_prompt[i]  # [d_model]
            similarity = F.cosine_similarity(
                prompt_vec.unsqueeze(0),  # [1, d]
                embedding_matrix,          # [V, d]
                dim=1,
            )
            
            # Top-k nearest
            values, indices = similarity.topk(top_k)
            
            tokens_info = []
            for val, idx in zip(values, indices):
                word = tokenizer.decode([idx.item()])
                tokens_info.append((word, val.item()))
            
            results.append(tokens_info)
        
        return results
    
    @staticmethod
    def prompt_statistics(soft_prompt: torch.Tensor) -> Dict:
        """Compute statistical properties of the soft prompt."""
        return {
            "shape": list(soft_prompt.shape),
            "mean": soft_prompt.mean().item(),
            "std": soft_prompt.std().item(),
            "min": soft_prompt.min().item(),
            "max": soft_prompt.max().item(),
            "norm_per_token": soft_prompt.norm(dim=1).tolist(),
            "total_norm": soft_prompt.norm().item(),
        }
    
    @staticmethod
    def inter_token_similarity(soft_prompt: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between soft prompt tokens."""
        normalized = F.normalize(soft_prompt, dim=1)
        return normalized @ normalized.T


def demonstrate_analysis():
    """Show prompt analysis and interpretability tools."""
    print("\n\n" + "=" * 65)
    print("  SECTION 5: ANALYSIS & INTERPRETABILITY")
    print("=" * 65)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    embed_matrix = model.transformer.wte.weight.detach()
    d_model = model.config.n_embd
    
    # Create a "trained" soft prompt (simulated)
    torch.manual_seed(42)
    
    # Initialize from text, then add random perturbation (simulating training)
    init_text = "Classify this text as positive or negative"
    tokens = tokenizer.encode(init_text, add_special_tokens=False)[:10]
    with torch.no_grad():
        soft_prompt = model.transformer.wte(torch.tensor(tokens)).clone()
    soft_prompt += torch.randn_like(soft_prompt) * 0.3  # Simulate training drift
    
    # Nearest token analysis
    print(f"\n  ─── Nearest Vocabulary Tokens ───")
    print(f"  (What each soft prompt token is closest to)")
    
    results = PromptAnalyzer.nearest_tokens(
        soft_prompt, embed_matrix, tokenizer, top_k=3
    )
    
    for i, token_info in enumerate(results):
        original = tokenizer.decode([tokens[i]]) if i < len(tokens) else "?"
        nearest = ", ".join([f"'{w}'({s:.3f})" for w, s in token_info])
        print(f"  Token {i:2d} (init: '{original:>12}'): nearest = {nearest}")
    
    # Statistics
    print(f"\n  ─── Prompt Statistics ───")
    stats = PromptAnalyzer.prompt_statistics(soft_prompt)
    print(f"  Shape: {stats['shape']}")
    print(f"  Mean:  {stats['mean']:.4f}")
    print(f"  Std:   {stats['std']:.4f}")
    print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"  Total norm: {stats['total_norm']:.4f}")
    
    norms = stats['norm_per_token']
    print(f"\n  Per-token norms:")
    for i, n in enumerate(norms):
        bar = "█" * int(n * 3)
        print(f"    Token {i:2d}: {n:>6.3f} {bar}")
    
    # Inter-token similarity
    print(f"\n  ─── Inter-Token Similarity ───")
    sim = PromptAnalyzer.inter_token_similarity(soft_prompt)
    print(f"  (Cosine similarity between prompt tokens)")
    
    print(f"         ", end="")
    for j in range(min(10, len(tokens))):
        print(f"  T{j:d}", end="   ")
    print()
    
    for i in range(min(10, len(tokens))):
        print(f"  T{i:d}  ", end="")
        for j in range(min(10, len(tokens))):
            val = sim[i, j].item()
            print(f"  {val:>5.2f}", end="")
        print()
    
    print(f"""
  ═══ Interpretability Insights ═══
  
  1. Nearest tokens show what "concept" each prompt position
     encodes, but trained prompts are BETWEEN tokens
     
  2. High inter-token similarity → redundant prompt tokens
     (could reduce prompt length without quality loss)
     
  3. Token norms indicate "importance" — higher norm tokens
     exert more influence on attention
     
  4. After training, prompts typically:
     • Move away from their initialization tokens
     • Develop higher norms than vocabulary tokens
     • Become somewhat orthogonal to each other
     • Encode task-specific "instructions" in vector space
""")
    
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all advanced prompt tuning demonstrations."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     PROMPT TUNING — ADVANCED TECHNIQUES                      ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Section 1: Ensembling
    demonstrate_ensembling()
    
    # Section 2: Transfer
    demonstrate_prompt_transfer()
    
    # Section 3: Decomposition
    demonstrate_decomposition()
    
    # Section 4: Quantization combo
    prompt_tuning_quantized()
    
    # Section 5: Analysis
    demonstrate_analysis()
    
    print("\n" + "=" * 65)
    print("  MODULE COMPLETE")
    print("=" * 65)
    print("""
    Covered:
    ✓ Prompt ensembling (average, weighted, voting)
    ✓ Prompt transfer (SPoT, interpolation, selective)
    ✓ Prompt decomposition (factored, shared+task-specific)
    ✓ Prompt tuning + 4-bit quantization
    ✓ Interpretability (nearest tokens, statistics, similarity)
    """)


if __name__ == "__main__":
    main()
