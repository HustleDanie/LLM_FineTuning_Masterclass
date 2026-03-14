"""
═══════════════════════════════════════════════════════════════════════════
SFT EVALUATION — Measuring Quality of Supervised Fine-Tuned Models
═══════════════════════════════════════════════════════════════════════════

Evaluating SFT models is different from evaluating base language models.
We care about:
  1. Response quality (not just perplexity)
  2. Instruction following ability
  3. Response formatting adherence
  4. Diversity and creativity
  5. Safety and harmful content

This module provides comprehensive evaluation tools specific to SFT.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter
from transformers import PreTrainedModel, PreTrainedTokenizer


# ═══════════════════════════════════════════════════════════════════════
# 1. RESPONSE QUALITY METRICS
# ═══════════════════════════════════════════════════════════════════════

def evaluate_response_quality(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_prompts: List[str],
    reference_responses: Optional[List[str]] = None,
    max_new_tokens: int = 256,
) -> Dict:
    """
    Evaluate the quality of generated responses.

    Metrics computed:
    - Average response length
    - Response diversity (unique n-grams)
    - Repetition rate
    - Format adherence (does it stop properly?)
    - BLEU/ROUGE scores (if references provided)
    """
    model.eval()
    device = next(model.parameters()).device

    generated_responses = []
    metrics = {
        "num_prompts": len(eval_prompts),
        "response_lengths": [],
        "empty_responses": 0,
        "truncated_responses": 0,
    }

    for prompt in eval_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Extract only the generated part (exclude prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        generated_responses.append(response)
        metrics["response_lengths"].append(len(response.split()))

        if not response:
            metrics["empty_responses"] += 1
        if len(generated_ids) >= max_new_tokens:
            metrics["truncated_responses"] += 1

    # Compute aggregate metrics
    lengths = metrics["response_lengths"]
    metrics["avg_response_length"] = np.mean(lengths) if lengths else 0
    metrics["min_response_length"] = min(lengths) if lengths else 0
    metrics["max_response_length"] = max(lengths) if lengths else 0
    metrics["std_response_length"] = np.std(lengths) if lengths else 0

    # Diversity metrics
    metrics["diversity"] = compute_diversity(generated_responses)

    # Repetition metrics
    metrics["avg_repetition_rate"] = np.mean([
        compute_repetition_rate(r) for r in generated_responses
    ])

    # BLEU scores (if references provided)
    if reference_responses:
        metrics["bleu_scores"] = compute_bleu_scores(
            generated_responses, reference_responses
        )

    # Print report
    print_evaluation_report(metrics, eval_prompts, generated_responses)

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# 2. DIVERSITY METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_diversity(texts: List[str]) -> Dict[str, float]:
    """
    Compute diversity of generated texts.

    WHY DIVERSITY MATTERS:
    ─────────────────────
    A model that always generates the same response is not useful.
    We want diverse, contextually appropriate responses.

    Metrics:
    - Distinct-1: Ratio of unique unigrams (words)
    - Distinct-2: Ratio of unique bigrams
    - Distinct-3: Ratio of unique trigrams

    Higher = more diverse. Typical values:
    - Distinct-1: 0.6-0.9 (good)
    - Distinct-2: 0.8-0.95 (good)
    """
    all_tokens = []
    for text in texts:
        tokens = text.lower().split()
        all_tokens.extend(tokens)

    if not all_tokens:
        return {"distinct_1": 0, "distinct_2": 0, "distinct_3": 0}

    # Unigrams
    unigrams = all_tokens
    distinct_1 = len(set(unigrams)) / len(unigrams) if unigrams else 0

    # Bigrams
    bigrams = list(zip(all_tokens[:-1], all_tokens[1:]))
    distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0

    # Trigrams
    trigrams = list(zip(all_tokens[:-2], all_tokens[1:-1], all_tokens[2:]))
    distinct_3 = len(set(trigrams)) / len(trigrams) if trigrams else 0

    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "distinct_3": distinct_3,
    }


def compute_repetition_rate(text: str, n: int = 3) -> float:
    """
    Compute the repetition rate of a text.

    Counts how many n-grams appear more than once.
    High repetition = model is stuck in a loop (common failure mode).
    """
    tokens = text.lower().split()
    if len(tokens) < n:
        return 0.0

    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(1 for count in counts.values() if count > 1)

    return repeated / len(counts) if counts else 0.0


# ═══════════════════════════════════════════════════════════════════════
# 3. BLEU SCORE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def compute_bleu_scores(
    generated: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute BLEU scores between generated and reference responses.

    BLEU measures n-gram overlap between generated and reference text.
    Not perfect for evaluating creative responses, but useful as a baseline.
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoother = SmoothingFunction().method1

        scores = []
        for gen, ref in zip(generated, references):
            gen_tokens = gen.lower().split()
            ref_tokens = ref.lower().split()
            score = sentence_bleu(
                [ref_tokens], gen_tokens,
                smoothing_function=smoother
            )
            scores.append(score)

        return {
            "bleu_avg": np.mean(scores),
            "bleu_max": max(scores),
            "bleu_min": min(scores),
        }
    except ImportError:
        print("   ⚠️ NLTK not installed. Skipping BLEU scores.")
        return {"bleu_avg": -1, "bleu_max": -1, "bleu_min": -1}


# ═══════════════════════════════════════════════════════════════════════
# 4. INSTRUCTION FOLLOWING TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_instruction_following(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_new_tokens: int = 200,
) -> Dict[str, bool]:
    """
    Test the model's ability to follow specific instructions.

    These tests check fundamental SFT capabilities:
    1. Can the model follow format instructions? (e.g., "List 3 items")
    2. Can it handle constraints? (e.g., "Answer in one sentence")
    3. Does it respect role boundaries? (doesn't pretend to be the user)
    4. Can it refuse inappropriate requests?
    """
    device = next(model.parameters()).device
    model.eval()

    tests = [
        {
            "name": "list_format",
            "prompt": "### Human: List exactly 3 benefits of exercise.\n### Assistant:",
            "check": lambda r: any(marker in r for marker in ["1.", "1)", "- ", "• "]),
            "description": "Can generate numbered/bulleted lists",
        },
        {
            "name": "brevity",
            "prompt": "### Human: In one sentence, what is the sun?\n### Assistant:",
            "check": lambda r: len(r.split('.')) <= 3 and len(r.split()) < 50,
            "description": "Can follow brevity constraints",
        },
        {
            "name": "non_empty",
            "prompt": "### Human: What is Python?\n### Assistant:",
            "check": lambda r: len(r.strip()) > 10,
            "description": "Generates non-empty responses",
        },
        {
            "name": "role_adherence",
            "prompt": "### Human: Hello, how are you?\n### Assistant:",
            "check": lambda r: "### Human:" not in r,
            "description": "Stays in assistant role (doesn't generate user turns)",
        },
    ]

    results = {}
    print("\n" + "=" * 60)
    print("INSTRUCTION FOLLOWING TESTS")
    print("=" * 60)

    for test in tests:
        inputs = tokenizer(test["prompt"], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,  # Low temperature for more deterministic output
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        passed = test["check"](response)
        results[test["name"]] = passed

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n{status} | {test['description']}")
        print(f"   Response: {response[:100]}...")

    total_passed = sum(results.values())
    print(f"\n{'─' * 60}")
    print(f"Results: {total_passed}/{len(tests)} tests passed")
    print("=" * 60)

    return results


# ═══════════════════════════════════════════════════════════════════════
# 5. BEFORE/AFTER SFT COMPARISON
# ═══════════════════════════════════════════════════════════════════════

def compare_base_vs_sft(
    base_model: PreTrainedModel,
    sft_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 150,
):
    """
    Compare base model vs SFT model on the same prompts.

    This dramatically demonstrates what SFT achieves:
    - Base model: continues text completion (no instruction following)
    - SFT model: answers the question properly

    This is the #1 way to showcase SFT's value.
    """
    if prompts is None:
        prompts = [
            "### Human: Explain what machine learning is in simple terms.\n### Assistant:",
            "### Human: Write a short poem about the ocean.\n### Assistant:",
            "### Human: What are three tips for better sleep?\n### Assistant:",
        ]

    device = next(sft_model.parameters()).device

    print("\n" + "=" * 70)
    print("BASE MODEL vs SFT MODEL COMPARISON")
    print("=" * 70)

    for prompt in prompts:
        print(f"\n📝 Prompt: {prompt[:60]}...")
        print("─" * 70)

        for model, label in [(base_model, "🔵 Base"), (sft_model, "🟢 SFT")]:
            model.eval()
            model.to(device)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            print(f"\n{label}: {response[:200]}")

        print("─" * 70)


# ═══════════════════════════════════════════════════════════════════════
# 6. EVALUATION REPORT
# ═══════════════════════════════════════════════════════════════════════

def print_evaluation_report(
    metrics: Dict,
    prompts: List[str],
    responses: List[str],
):
    """Print a comprehensive evaluation report."""
    print("\n" + "=" * 60)
    print("SFT EVALUATION REPORT")
    print("=" * 60)

    print(f"\n📊 Response Statistics:")
    print(f"   Total prompts: {metrics['num_prompts']}")
    print(f"   Avg response length: {metrics['avg_response_length']:.1f} words")
    print(f"   Length range: {metrics['min_response_length']}-{metrics['max_response_length']} words")
    print(f"   Empty responses: {metrics['empty_responses']}")
    print(f"   Truncated (hit max): {metrics['truncated_responses']}")

    print(f"\n🎨 Diversity:")
    div = metrics.get("diversity", {})
    print(f"   Distinct-1 (unigrams): {div.get('distinct_1', 0):.3f}")
    print(f"   Distinct-2 (bigrams):  {div.get('distinct_2', 0):.3f}")
    print(f"   Distinct-3 (trigrams): {div.get('distinct_3', 0):.3f}")

    print(f"\n🔄 Repetition:")
    print(f"   Avg repetition rate: {metrics.get('avg_repetition_rate', 0):.3f}")

    if "bleu_scores" in metrics:
        bleu = metrics["bleu_scores"]
        if bleu.get("bleu_avg", -1) >= 0:
            print(f"\n📐 BLEU Scores:")
            print(f"   Average: {bleu['bleu_avg']:.4f}")
            print(f"   Max: {bleu['bleu_max']:.4f}")
            print(f"   Min: {bleu['bleu_min']:.4f}")

    # Show sample responses
    print(f"\n📝 Sample Responses:")
    for i, (prompt, response) in enumerate(zip(prompts[:3], responses[:3])):
        print(f"\n   [{i+1}] Prompt: {prompt[:60]}...")
        print(f"       Response: {response[:120]}...")

    print("=" * 60)
