"""
═══════════════════════════════════════════════════════════════════════════
PEFT ADVANCED — Merging, Multi-Adapter, and Serving Strategies
═══════════════════════════════════════════════════════════════════════════

Beyond basic PEFT training, there are several advanced techniques
for deploying and combining PEFT adapters in production.

TOPICS COVERED:
───────────────
1. Weight Merging — Combine adapter weights into base model
2. Multi-Adapter — Load multiple adapters for different tasks
3. Adapter Composition — Combine adapters with arithmetic
4. Adapter Serving — Efficient multi-tenant serving
5. Adapter Transfer — Move adapters between compatible models
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import copy

try:
    from peft import (
        PeftModel,
        PeftConfig,
        get_peft_model,
        LoraConfig,
        TaskType,
        set_peft_model_state_dict,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
# 1. WEIGHT MERGING STRATEGIES
# ═══════════════════════════════════════════════════════════════════════

class WeightMerger:
    """
    Strategies for merging LoRA weights into the base model.

    WHY MERGE?
    ──────────
    - Inference speed: No adapter overhead
    - Deployment simplicity: Single model file
    - Compatibility: Works with any serving framework

    WHEN NOT TO MERGE:
    ──────────────────
    - Need to swap adapters at runtime
    - Multi-task serving with shared base model
    - Want to continue training the adapter
    """

    @staticmethod
    def merge_lora_weights(
        base_weight: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        scaling: float = 1.0,
    ) -> torch.Tensor:
        """
        Merge LoRA weights: W_merged = W + scaling * B @ A

        Args:
            base_weight: Original weight matrix (d_out, d_in)
            lora_A: LoRA down-projection (rank, d_in)
            lora_B: LoRA up-projection (d_out, rank)
            scaling: alpha/rank scaling factor

        Returns:
            Merged weight matrix
        """
        delta_w = lora_B @ lora_A * scaling
        return base_weight + delta_w

    @staticmethod
    def unmerge_lora_weights(
        merged_weight: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        scaling: float = 1.0,
    ) -> torch.Tensor:
        """
        Reverse the merge: W_base = W_merged - scaling * B @ A

        Useful when you need to swap adapters.
        """
        delta_w = lora_B @ lora_A * scaling
        return merged_weight - delta_w

    @staticmethod
    def merge_peft_model(model, adapter_name: str = "default"):
        """
        Merge a PEFT model using the library's built-in method.

        This handles all the complexity of:
        - Finding the right layers
        - Computing the correct scaling
        - Updating the weight matrices
        - Removing the PEFT wrapper
        """
        if PEFT_AVAILABLE and hasattr(model, 'merge_and_unload'):
            return model.merge_and_unload()
        else:
            print("Cannot merge — PEFT model not detected")
            return model


# ═══════════════════════════════════════════════════════════════════════
# 2. MULTI-ADAPTER MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════

class MultiAdapterManager:
    """
    Manage multiple LoRA adapters on a single base model.

    USE CASE:
    ─────────
    You have one base LLaMA model and want to serve:
    - Adapter A: Fine-tuned for code generation
    - Adapter B: Fine-tuned for medical Q&A
    - Adapter C: Fine-tuned for creative writing

    Instead of loading 3 full models (3 × 7B params = 21B params in memory),
    you load 1 base model + 3 tiny adapters (<100 MB each).

    SWITCH ADAPTERS AT RUNTIME:
    ──────────────────────────
    model.set_adapter("code_gen")    # Use code adapter
    model.set_adapter("medical_qa")  # Switch to medical adapter
    model.set_adapter("creative")    # Switch to creative adapter
    """

    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name
        self.loaded_adapters: Dict[str, str] = {}  # name -> path

    def load_adapter(self, model, adapter_path: str, adapter_name: str):
        """
        Load an additional adapter onto the model.

        After loading, you can switch between adapters using:
            model.set_adapter(adapter_name)
        """
        if PEFT_AVAILABLE and hasattr(model, 'load_adapter'):
            model.load_adapter(adapter_path, adapter_name)
            self.loaded_adapters[adapter_name] = adapter_path
            print(f"  Loaded adapter '{adapter_name}' from {adapter_path}")
            return model
        else:
            print("  Multi-adapter requires PEFT library")
            return model

    def switch_adapter(self, model, adapter_name: str):
        """Switch the active adapter."""
        if PEFT_AVAILABLE and hasattr(model, 'set_adapter'):
            model.set_adapter(adapter_name)
            print(f"  Active adapter: {adapter_name}")
        else:
            print(f"  Cannot switch adapter (PEFT not available)")

    def disable_adapter(self, model):
        """Disable all adapters (use base model only)."""
        if PEFT_AVAILABLE and hasattr(model, 'disable_adapter'):
            model.disable_adapter()
            print("  All adapters disabled — using base model")

    def list_adapters(self) -> Dict:
        """List all loaded adapters."""
        return self.loaded_adapters

    @staticmethod
    def print_multi_adapter_example():
        """Print example code for multi-adapter usage."""
        print("""
MULTI-ADAPTER USAGE EXAMPLE:
─────────────────────────────
from peft import PeftModel

# Load base model with first adapter
model = PeftModel.from_pretrained(base_model, "path/to/code_adapter", adapter_name="code")

# Load additional adapters
model.load_adapter("path/to/medical_adapter", adapter_name="medical")
model.load_adapter("path/to/creative_adapter", adapter_name="creative")

# Switch between adapters (instant, no model reloading!)
model.set_adapter("code")
code_response = model.generate(...)

model.set_adapter("medical")
medical_response = model.generate(...)

model.set_adapter("creative")
creative_response = model.generate(...)

# Use base model without any adapter
with model.disable_adapter():
    base_response = model.generate(...)
""")


# ═══════════════════════════════════════════════════════════════════════
# 3. ADAPTER ARITHMETIC (Model Merging)
# ═══════════════════════════════════════════════════════════════════════

class AdapterArithmetic:
    """
    Combine multiple adapters using weight arithmetic.

    ADAPTER ARITHMETIC:
    ───────────────────
    Just like word embeddings can be combined (king - man + woman = queen),
    adapter weights can be combined:

    1. ADDITION: adapter_combined = adapter_A + adapter_B
       (combines capabilities of both)

    2. SCALING: adapter_scaled = α × adapter_A
       (control the strength of adaptation)

    3. INTERPOLATION: adapter_interp = α × adapter_A + (1-α) × adapter_B
       (blend between two tasks)

    4. NEGATION: adapter_neg = adapter_A - adapter_B
       (remove capability B from A)

    PAPERS:
    - "Editing Models with Task Arithmetic" (Ilharco et al., 2023)
    - "Model Soups" (Wortsman et al., 2022)
    """

    @staticmethod
    def add_adapters(
        state_dict_a: Dict[str, torch.Tensor],
        state_dict_b: Dict[str, torch.Tensor],
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Combine two adapter state dicts: result = α·A + β·B

        Used for combining task capabilities.

        Example: code_adapter + math_adapter = code_and_math_adapter
        """
        combined = {}
        for key in state_dict_a:
            if key in state_dict_b:
                combined[key] = alpha * state_dict_a[key] + beta * state_dict_b[key]
            else:
                combined[key] = alpha * state_dict_a[key]

        # Include keys only in B
        for key in state_dict_b:
            if key not in combined:
                combined[key] = beta * state_dict_b[key]

        return combined

    @staticmethod
    def scale_adapter(
        state_dict: Dict[str, torch.Tensor],
        scale: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Scale adapter weights: result = scale × adapter

        Useful for controlling adaptation strength:
        - scale > 1: Stronger task-specific behavior
        - scale < 1: Weaker adaptation, more like base model
        - scale = 0: Base model behavior
        """
        return {key: scale * value for key, value in state_dict.items()}

    @staticmethod
    def interpolate_adapters(
        state_dict_a: Dict[str, torch.Tensor],
        state_dict_b: Dict[str, torch.Tensor],
        t: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Linear interpolation: result = (1-t)·A + t·B

        t=0.0: Pure adapter A
        t=0.5: Equal blend
        t=1.0: Pure adapter B

        "Model Soups" showed that interpolating fine-tuned models
        often improves generalization!
        """
        return AdapterArithmetic.add_adapters(
            state_dict_a, state_dict_b,
            alpha=(1 - t), beta=t,
        )

    @staticmethod
    def task_vector_negation(
        base_state_dict: Dict[str, torch.Tensor],
        adapted_state_dict: Dict[str, torch.Tensor],
        remove_state_dict: Dict[str, torch.Tensor],
        alpha: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Task vector negation: remove a capability.

        task_vector = adapted - base
        remove_vector = remove_adapted - base
        result = base + task_vector - α × remove_vector

        Example: Remove toxic behavior while keeping helpfulness.
        """
        result = {}
        for key in adapted_state_dict:
            if key in base_state_dict and key in remove_state_dict:
                task_vector = adapted_state_dict[key] - base_state_dict[key]
                remove_vector = remove_state_dict[key] - base_state_dict[key]
                result[key] = base_state_dict[key] + task_vector - alpha * remove_vector
            elif key in adapted_state_dict:
                result[key] = adapted_state_dict[key]
        return result


# ═══════════════════════════════════════════════════════════════════════
# 4. EFFICIENT MULTI-TENANT SERVING
# ═══════════════════════════════════════════════════════════════════════

class AdapterServingStrategy:
    """
    Strategies for serving multiple PEFT adapters efficiently.

    ARCHITECTURE OPTIONS:
    ─────────────────────

    Option 1: SHARED BASE + ADAPTER SWAPPING
    ─────────────────────────────────────────
    Base Model (frozen, in GPU) ──→ Adapter A ──→ Response A
                                └──→ Adapter B ──→ Response B
                                └──→ Adapter C ──→ Response C

    Pros: Minimal memory (1 base model + N tiny adapters)
    Cons: Cannot batch across different adapters

    Option 2: PRE-MERGED MODELS
    ────────────────────────────
    Merged Model A ──→ Response A
    Merged Model B ──→ Response B
    Merged Model C ──→ Response C

    Pros: Full speed, can be served independently
    Cons: N × model_size memory

    Option 3: BATCHED MULTI-ADAPTER (S-LoRA / Punica)
    ──────────────────────────────────────────────────
    Base Model (shared) + LoRA A applied to batch items for user A
                        + LoRA B applied to batch items for user B
                        + LoRA C applied to batch items for user C

    Pros: Can batch across adapters, minimal memory
    Cons: Requires custom CUDA kernels, complex implementation
    Frameworks: S-LoRA, Punica, vLLM with LoRA support
    """

    @staticmethod
    def print_serving_comparison():
        """Print serving strategy comparison."""
        print(f"""
{'═' * 70}
ADAPTER SERVING STRATEGIES COMPARISON
{'═' * 70}

Scenario: Serve 10 different LoRA adapters for a 7B model

{'Strategy':<30} {'GPU Memory':>12} {'Throughput':>12} {'Complexity':>12}
{'─' * 68}
{'Shared + Swap':<30} {'~14 GB':>12} {'Low':>12} {'Simple':>12}
{'Pre-merged (10 copies)':<30} {'~140 GB':>12} {'Highest':>12} {'Simple':>12}
{'Batched Multi-LoRA':<30} {'~14 GB':>12} {'High':>12} {'Complex':>12}
{'Offload (CPU←→GPU)':<30} {'~3 GB':>12} {'Lowest':>12} {'Medium':>12}

RECOMMENDATION BY SCALE:
─────────────────────────
• 1-3 adapters:    Pre-merge or swap (simplest)
• 4-20 adapters:   Shared base + swap
• 20-100 adapters: Batched multi-LoRA (S-LoRA/vLLM)
• 100+ adapters:   Batched multi-LoRA + adapter caching

TOOLS:
──────
• vLLM: Supports LoRA serving with --enable-lora flag
• S-LoRA: Research framework for batched LoRA serving
• Text Generation Inference (TGI): HuggingFace's serving with adapter support
• LitServe / LitLLM: Lightweight adapter serving
""")


# ═══════════════════════════════════════════════════════════════════════
# 5. ADAPTER TRANSFER & COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════

class AdapterTransfer:
    """
    Transfer adapters between compatible models.

    KEY INSIGHT:
    ────────────
    LoRA adapters modify specific weight matrices by shape.
    If two models have the SAME architecture (same d_model, same layer
    structure), their LoRA adapters may be transferable.

    COMPATIBLE TRANSFERS:
    ─────────────────────
    ✓ Same model, different quantization (fp16 → int8)
    ✓ Same model family, different fine-tune (LLaMA-7B chat → LLaMA-7B code)
    ✓ Different training data, same model (adapter trained on English → same model)

    INCOMPATIBLE:
    ─────────────
    ✗ Different d_model dimensions
    ✗ Different number of attention heads
    ✗ Different architecture families (LLaMA → Mistral has subtle differences)
    """

    @staticmethod
    def check_compatibility(
        source_config: Dict,
        target_config: Dict,
    ) -> Tuple[bool, str]:
        """Check if an adapter is compatible between two models."""
        checks = []

        # Check hidden size
        if source_config.get("hidden_size") != target_config.get("hidden_size"):
            return False, f"Hidden size mismatch: {source_config.get('hidden_size')} vs {target_config.get('hidden_size')}"

        # Check num layers
        if source_config.get("num_hidden_layers") != target_config.get("num_hidden_layers"):
            return False, f"Layer count mismatch: {source_config.get('num_hidden_layers')} vs {target_config.get('num_hidden_layers')}"

        # Check num attention heads
        if source_config.get("num_attention_heads") != target_config.get("num_attention_heads"):
            return False, f"Attention heads mismatch"

        # Check intermediate size
        if source_config.get("intermediate_size") != target_config.get("intermediate_size"):
            return False, f"FFN size mismatch"

        return True, "Models are compatible for adapter transfer"

    @staticmethod
    def transfer_adapter_weights(
        source_state_dict: Dict[str, torch.Tensor],
        target_model_name: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Attempt to transfer adapter weights to a different base model.

        In practice, this is as simple as loading the adapter onto
        the new base model — the PEFT library handles the mapping.
        """
        # The key insight: adapter weights are stored with layer-specific names
        # If architectures match, the names will map correctly
        print(f"  Transferring {len(source_state_dict)} adapter parameters")
        print(f"  Target model: {target_model_name}")
        print(f"  Adapter keys: {list(source_state_dict.keys())[:5]}...")
        return source_state_dict  # Same weights, different base model


# ═══════════════════════════════════════════════════════════════════════
# 6. BEST PRACTICES SUMMARY
# ═══════════════════════════════════════════════════════════════════════

def print_peft_best_practices():
    """Print comprehensive PEFT best practices."""
    print(f"""
{'═' * 70}
PEFT BEST PRACTICES
{'═' * 70}

1. CHOOSING A METHOD:
   ─────────────────
   • Start with LoRA (r=16, alpha=32) — best balance of quality and efficiency
   • Use QLoRA for 13B+ models on consumer GPUs
   • Use full FT only if you have unlimited compute AND data

2. HYPERPARAMETER TUNING:
   ──────────────────────
   • Learning rate: 1e-4 to 3e-4 (higher than full FT!)
   • LoRA rank: Start with 16, try 8 (efficient) or 64 (more capacity)
   • Target modules: Start with Q+V, try all linear for max quality
   • Alpha: Set to 2×rank as starting point
   • Epochs: 1-3 (monitor for overfitting)
   • Batch size: As large as memory allows (gradient accumulation helps)

3. TARGET MODULE SELECTION:
   ────────────────────────
   • Q + V only: Most efficient, works well for most tasks
   • All attention (Q, K, V, O): Better for complex adaptations
   • All linear (attention + MLP): Maximum quality, most parameters
   • Rule: More target modules = more capacity but more memory

4. AVOIDING COMMON PITFALLS:
   ─────────────────────────
   • DON'T use the same learning rate as full FT (too low for PEFT)
   • DON'T forget to set padding token
   • DON'T train for too many epochs (PEFT overfits faster)
   • DO use gradient checkpointing for memory savings
   • DO monitor eval loss closely
   • DO compare against base model to verify improvement

5. SAVING & DEPLOYMENT:
   ─────────────────────
   • Save adapter weights only (model.save_pretrained(path))
   • For serving: merge weights if single-adapter, keep separate if multi-adapter
   • Version your adapters independently of the base model
   • Store adapter configs for reproducibility

6. COMBINING WITH OTHER TECHNIQUES:
   ─────────────────────────────────
   • PEFT + Quantization (QLoRA) = massive memory savings
   • PEFT + Gradient Checkpointing = even more memory savings
   • PEFT + Flash Attention = faster training
   • PEFT + DeepSpeed ZeRO = distributed PEFT training
""")


# ═══════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 70)
    print("  PEFT ADVANCED TOPICS")
    print("═" * 70)

    # 1. Weight merging demo
    print("\n1. WEIGHT MERGING")
    print("─" * 40)
    d_model, rank = 768, 16
    base_weight = torch.randn(d_model, d_model)
    lora_A = torch.randn(rank, d_model) * 0.01
    lora_B = torch.zeros(d_model, rank)

    merged = WeightMerger.merge_lora_weights(base_weight, lora_A, lora_B, scaling=2.0)
    print(f"  Base weight shape: {base_weight.shape}")
    print(f"  LoRA A shape: {lora_A.shape}")
    print(f"  LoRA B shape: {lora_B.shape}")
    print(f"  Merged shape: {merged.shape}")
    print(f"  Max change: {(merged - base_weight).abs().max():.6f}")
    print(f"  (B is zero-initialized, so no change at init)")

    # Non-zero B
    lora_B = torch.randn(d_model, rank) * 0.01
    merged = WeightMerger.merge_lora_weights(base_weight, lora_A, lora_B, scaling=2.0)
    unmerged = WeightMerger.unmerge_lora_weights(merged, lora_A, lora_B, scaling=2.0)
    print(f"  After training (B non-zero), max change: {(merged - base_weight).abs().max():.4f}")
    print(f"  Unmerge recovers base: {torch.allclose(base_weight, unmerged, atol=1e-5)}")

    # 2. Adapter arithmetic demo
    print("\n2. ADAPTER ARITHMETIC")
    print("─" * 40)
    adapter_a = {"layer.0.lora_A": torch.randn(16, 768), "layer.0.lora_B": torch.randn(768, 16)}
    adapter_b = {"layer.0.lora_A": torch.randn(16, 768), "layer.0.lora_B": torch.randn(768, 16)}

    combined = AdapterArithmetic.add_adapters(adapter_a, adapter_b, alpha=0.5, beta=0.5)
    scaled = AdapterArithmetic.scale_adapter(adapter_a, scale=0.5)
    interpolated = AdapterArithmetic.interpolate_adapters(adapter_a, adapter_b, t=0.3)

    print(f"  Adapter A keys: {list(adapter_a.keys())}")
    print(f"  Combined (0.5*A + 0.5*B): {list(combined.keys())}")
    print(f"  Scaled (0.5*A): norm = {scaled['layer.0.lora_A'].norm():.2f}")
    print(f"  Interpolated (0.7*A + 0.3*B): norm = {interpolated['layer.0.lora_A'].norm():.2f}")

    # 3. Multi-adapter
    print("\n3. MULTI-ADAPTER SERVING")
    print("─" * 40)
    MultiAdapterManager.print_multi_adapter_example()

    # 4. Serving strategy
    AdapterServingStrategy.print_serving_comparison()

    # 5. Best practices
    print_peft_best_practices()
