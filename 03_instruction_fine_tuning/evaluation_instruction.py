"""
═══════════════════════════════════════════════════════════════════════════
EVALUATION FOR INSTRUCTION-TUNED MODELS
═══════════════════════════════════════════════════════════════════════════

Evaluating instruction-tuned models is fundamentally different from
evaluating standard language models. Perplexity alone doesn't capture
whether the model FOLLOWS INSTRUCTIONS correctly.

KEY EVALUATION DIMENSIONS:
──────────────────────────
1. Instruction Following — Does the model do what was asked?
2. Zero-Shot Generalization — Can it handle unseen task types?
3. Response Quality — Helpful, accurate, harmless?
4. Consistency — Same quality across different instruction phrasings?
5. Task-Specific Metrics — BLEU, ROUGE, accuracy per task category

EVALUATION APPROACHES:
──────────────────────
A. Automatic Metrics (fast, scalable, but imperfect)
B. LLM-as-Judge (GPT-4, Claude evaluate outputs — correlates well with humans)
C. Human Evaluation (gold standard, but expensive and slow)
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict


# ═══════════════════════════════════════════════════════════════════════
# 1. INSTRUCTION-FOLLOWING EVALUATION
# ═══════════════════════════════════════════════════════════════════════

class InstructionFollowingEvaluator:
    """
    Evaluate whether a model properly follows instruction constraints.

    Tests specific, verifiable aspects of instruction following:
    - Format constraints ("respond in bullet points")
    - Length constraints ("in under 50 words")
    - Language constraints ("respond in French")
    - Content constraints ("include at least 3 examples")
    """

    def __init__(self):
        self.test_cases = self._build_test_cases()

    def _build_test_cases(self) -> List[Dict]:
        """
        Build a suite of instruction-following tests.

        Each test has:
        - instruction: The prompt
        - constraint_type: What kind of constraint
        - check_fn: Function to verify if constraint is met
        """
        return [
            # FORMAT CONSTRAINTS
            {
                "instruction": "List 5 programming languages. Format your answer as a numbered list.",
                "constraint_type": "format_numbered_list",
                "check_fn": self._check_numbered_list,
                "expected_items": 5,
            },
            {
                "instruction": "Describe three benefits of exercise using bullet points.",
                "constraint_type": "format_bullet_points",
                "check_fn": self._check_bullet_points,
                "expected_items": 3,
            },
            {
                "instruction": "Write a JSON object with keys 'name', 'age', and 'city' for a fictional person.",
                "constraint_type": "format_json",
                "check_fn": self._check_json_format,
                "expected_keys": ["name", "age", "city"],
            },
            # LENGTH CONSTRAINTS
            {
                "instruction": "Explain gravity in exactly one sentence.",
                "constraint_type": "length_one_sentence",
                "check_fn": self._check_one_sentence,
            },
            {
                "instruction": "Describe the internet in under 50 words.",
                "constraint_type": "length_under_words",
                "check_fn": lambda resp: self._check_word_count(resp, max_words=50),
            },
            {
                "instruction": "Write a paragraph of at least 100 words about renewable energy.",
                "constraint_type": "length_min_words",
                "check_fn": lambda resp: self._check_word_count(
                    resp, min_words=100
                ),
            },
            # CONTENT CONSTRAINTS
            {
                "instruction": "Name three European countries and their capitals.",
                "constraint_type": "content_count",
                "check_fn": lambda resp: self._check_item_count(resp, min_items=3),
            },
            {
                "instruction": "Explain machine learning without using the word 'data'.",
                "constraint_type": "content_exclusion",
                "check_fn": lambda resp: self._check_word_exclusion(resp, "data"),
            },
            # STYLE CONSTRAINTS
            {
                "instruction": "Explain quantum physics as if you were talking to a 5-year-old.",
                "constraint_type": "style_simple",
                "check_fn": self._check_simple_language,
            },
        ]

    # ─── Check Functions ─────────────────────────────────────────────

    @staticmethod
    def _check_numbered_list(response: str, expected_items: int = 5, **kw) -> Dict:
        """Check if response contains a numbered list."""
        pattern = r'^\s*\d+[\.\)]\s+'
        lines = response.strip().split('\n')
        numbered_lines = [l for l in lines if re.match(pattern, l.strip())]
        passed = len(numbered_lines) >= expected_items
        return {
            "passed": passed,
            "detail": f"Found {len(numbered_lines)} numbered items (expected {expected_items})",
        }

    @staticmethod
    def _check_bullet_points(response: str, expected_items: int = 3, **kw) -> Dict:
        """Check if response uses bullet points."""
        pattern = r'^\s*[-•\*]\s+'
        lines = response.strip().split('\n')
        bullet_lines = [l for l in lines if re.match(pattern, l.strip())]
        passed = len(bullet_lines) >= expected_items
        return {
            "passed": passed,
            "detail": f"Found {len(bullet_lines)} bullet points (expected {expected_items})",
        }

    @staticmethod
    def _check_json_format(response: str, expected_keys: List[str] = None, **kw) -> Dict:
        """Check if response contains valid JSON with expected keys."""
        # Extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response)
        if not json_match:
            return {"passed": False, "detail": "No JSON object found"}

        try:
            obj = json.loads(json_match.group())
            if expected_keys:
                missing = [k for k in expected_keys if k not in obj]
                passed = len(missing) == 0
                detail = f"Valid JSON. Missing keys: {missing}" if missing else "Valid JSON with all expected keys"
            else:
                passed = True
                detail = "Valid JSON"
            return {"passed": passed, "detail": detail}
        except json.JSONDecodeError:
            return {"passed": False, "detail": "Invalid JSON format"}

    @staticmethod
    def _check_one_sentence(response: str, **kw) -> Dict:
        """Check if response is exactly one sentence."""
        # Count sentence-ending punctuation
        sentences = re.split(r'[.!?]+', response.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        passed = len(sentences) == 1
        return {
            "passed": passed,
            "detail": f"Found {len(sentences)} sentences (expected 1)",
        }

    @staticmethod
    def _check_word_count(response: str, min_words: int = 0, max_words: int = float('inf'), **kw) -> Dict:
        """Check if response stays within word count bounds."""
        word_count = len(response.split())
        passed = min_words <= word_count <= max_words
        return {
            "passed": passed,
            "detail": f"Word count: {word_count} (bounds: {min_words}-{max_words})",
        }

    @staticmethod
    def _check_item_count(response: str, min_items: int = 3, **kw) -> Dict:
        """Check if response contains enough distinct items."""
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        # Also check for comma-separated items
        if len(lines) < min_items:
            items = response.split(',')
            count = max(len(lines), len(items))
        else:
            count = len(lines)
        passed = count >= min_items
        return {
            "passed": passed,
            "detail": f"Found approximately {count} items (expected {min_items}+)",
        }

    @staticmethod
    def _check_word_exclusion(response: str, excluded_word: str, **kw) -> Dict:
        """Check if response avoids a specific word."""
        found = excluded_word.lower() in response.lower()
        return {
            "passed": not found,
            "detail": f"Word '{excluded_word}' {'found' if found else 'not found'} in response",
        }

    @staticmethod
    def _check_simple_language(response: str, **kw) -> Dict:
        """Check if language is simple (heuristic: short words, short sentences)."""
        words = response.split()
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        simple = avg_word_len < 6  # Simple language tends to have shorter words
        return {
            "passed": simple,
            "detail": f"Avg word length: {avg_word_len:.1f} ({'simple' if simple else 'complex'})",
        }

    def evaluate_model(
        self,
        generate_fn,
        verbose: bool = True,
    ) -> Dict:
        """
        Run all instruction-following tests on a model.

        Args:
            generate_fn: Function that takes instruction str and returns response str
            verbose: Print detailed results

        Returns:
            Summary dict with scores per constraint type
        """
        results = []

        for test in self.test_cases:
            response = generate_fn(test["instruction"])

            # Build kwargs from test case (exclude non-param keys)
            check_kwargs = {
                k: v for k, v in test.items()
                if k not in {"instruction", "constraint_type", "check_fn"}
            }
            result = test["check_fn"](response, **check_kwargs)
            result["constraint_type"] = test["constraint_type"]
            result["instruction"] = test["instruction"]
            result["response"] = response
            results.append(result)

            if verbose:
                status = "✓" if result["passed"] else "✗"
                print(f"  [{status}] {test['constraint_type']}: {result['detail']}")

        # Summary
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        by_type = defaultdict(list)
        for r in results:
            by_type[r["constraint_type"]].append(r["passed"])

        return {
            "total_tests": total,
            "passed": passed,
            "score": passed / max(total, 1),
            "by_type": {k: sum(v) / len(v) for k, v in by_type.items()},
            "detailed_results": results,
        }


# ═══════════════════════════════════════════════════════════════════════
# 2. ZERO-SHOT GENERALIZATION EVALUATION
# ═══════════════════════════════════════════════════════════════════════

# Tasks the model was NOT trained on — tests generalization
HELD_OUT_TASKS = [
    {
        "instruction": "Convert this Python code to JavaScript.",
        "input": "for i in range(10):\n    print(i)",
        "expected_contains": ["for", "console.log", "let", "var", "const"],
        "category": "code_translation",
    },
    {
        "instruction": "What is the sentiment of this review?",
        "input": "This product completely exceeded my expectations! Absolutely love it.",
        "expected_answer": "positive",
        "category": "sentiment_analysis",
    },
    {
        "instruction": "Extract all the named entities from this text.",
        "input": "Apple CEO Tim Cook announced new products at the event in Cupertino, California.",
        "expected_contains": ["Apple", "Tim Cook", "Cupertino", "California"],
        "category": "ner",
    },
    {
        "instruction": "Rewrite this sentence in passive voice.",
        "input": "The cat chased the mouse.",
        "expected_contains": ["mouse", "chased", "by", "cat"],
        "category": "grammar_transform",
    },
    {
        "instruction": "What logical fallacy is present in this argument?",
        "input": "Everyone is buying this product, so it must be good.",
        "expected_contains": ["bandwagon", "ad populum", "appeal to popularity"],
        "category": "critical_thinking",
    },
]


def evaluate_zero_shot(
    generate_fn,
    tasks: List[Dict] = None,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate zero-shot generalization on held-out task types.

    A well-instruction-tuned model should handle task types it
    was never explicitly trained on, if the instructions are clear.
    """
    if tasks is None:
        tasks = HELD_OUT_TASKS

    results = []
    for task in tasks:
        instruction = task["instruction"]
        if task.get("input"):
            instruction += f"\n\n{task['input']}"

        response = generate_fn(instruction)

        # Check if response contains expected elements
        if "expected_contains" in task:
            found = []
            for expected in task["expected_contains"]:
                if expected.lower() in response.lower():
                    found.append(expected)
            score = len(found) / len(task["expected_contains"])
        elif "expected_answer" in task:
            score = 1.0 if task["expected_answer"].lower() in response.lower() else 0.0
        else:
            score = 0.5  # Cannot auto-evaluate

        result = {
            "category": task["category"],
            "score": score,
            "response": response[:200],
        }
        results.append(result)

        if verbose:
            status = "✓" if score >= 0.5 else "✗"
            print(f"  [{status}] {task['category']}: score={score:.2f}")

    # Aggregate
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r["score"])

    return {
        "overall_score": sum(r["score"] for r in results) / max(len(results), 1),
        "by_category": {k: sum(v) / len(v) for k, v in by_category.items()},
        "detailed_results": results,
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. LLM-AS-JUDGE EVALUATION
# ═══════════════════════════════════════════════════════════════════════

LLM_JUDGE_PROMPT = """You are an expert evaluator. Rate the following AI assistant's response on a scale of 1-5 for each criterion.

### Instruction:
{instruction}

### Response:
{response}

### Evaluation Criteria:
1. **Helpfulness** (1-5): Does the response fully address the instruction?
2. **Accuracy** (1-5): Is the information factually correct?
3. **Clarity** (1-5): Is the response well-written and easy to understand?
4. **Completeness** (1-5): Does it cover all aspects of the instruction?

### Output Format:
Helpfulness: [score]
Accuracy: [score]
Clarity: [score]
Completeness: [score]
Overall: [average score]
Reasoning: [brief explanation]
"""

# Pairwise comparison prompt for A/B testing
PAIRWISE_JUDGE_PROMPT = """You are an expert evaluator comparing two AI assistant responses. Which response better follows the instruction?

### Instruction:
{instruction}

### Response A:
{response_a}

### Response B:
{response_b}

### Evaluation:
Which response is better? Answer "A", "B", or "Tie".
Provide a brief justification.

Winner:"""


def build_judge_prompt(
    instruction: str,
    response: str,
    template: str = "single",
) -> str:
    """
    Build an LLM-as-Judge evaluation prompt.

    LLM-AS-JUDGE:
    ──────────────
    Using a strong LLM (GPT-4, Claude) to evaluate outputs of a
    weaker/fine-tuned model. This correlates well with human judgment
    and is much cheaper than human evaluation.

    Common frameworks:
    - MT-Bench: Multi-turn benchmark with GPT-4 judge
    - AlpacaEval: GPT-4 compares against text-davinci-003
    - Arena ELO: Head-to-head comparisons
    """
    return LLM_JUDGE_PROMPT.format(instruction=instruction, response=response)


def parse_judge_scores(judge_output: str) -> Dict:
    """Parse scores from LLM judge output."""
    scores = {}
    patterns = {
        "helpfulness": r"Helpfulness:\s*(\d)",
        "accuracy": r"Accuracy:\s*(\d)",
        "clarity": r"Clarity:\s*(\d)",
        "completeness": r"Completeness:\s*(\d)",
        "overall": r"Overall:\s*([\d.]+)",
    }

    for metric, pattern in patterns.items():
        match = re.search(pattern, judge_output, re.IGNORECASE)
        if match:
            scores[metric] = float(match.group(1))

    return scores


# ═══════════════════════════════════════════════════════════════════════
# 4. RESPONSE QUALITY METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_response_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict:
    """
    Compute automatic quality metrics for instruction-tuned outputs.

    NOTE: These metrics have known limitations for instruction-tuned models:
    - BLEU/ROUGE assume one correct answer, but many responses are valid
    - Perplexity doesn't capture instruction following
    - Best used ALONGSIDE human/LLM evaluation, not as sole metrics
    """
    metrics = {}

    # 1. Average response length
    pred_lengths = [len(p.split()) for p in predictions]
    ref_lengths = [len(r.split()) for r in references]
    metrics["avg_pred_length"] = sum(pred_lengths) / max(len(pred_lengths), 1)
    metrics["avg_ref_length"] = sum(ref_lengths) / max(len(ref_lengths), 1)
    metrics["length_ratio"] = metrics["avg_pred_length"] / max(metrics["avg_ref_length"], 1)

    # 2. Exact match rate (strict)
    exact_matches = sum(
        1 for p, r in zip(predictions, references)
        if p.strip().lower() == r.strip().lower()
    )
    metrics["exact_match"] = exact_matches / max(len(predictions), 1)

    # 3. Token overlap (F1)
    f1_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())

        if not pred_tokens or not ref_tokens:
            f1_scores.append(0.0)
            continue

        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(ref_tokens) if ref_tokens else 0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    metrics["token_f1"] = sum(f1_scores) / max(len(f1_scores), 1)

    # 4. Diversity (distinct n-grams across all responses)
    all_unigrams = []
    all_bigrams = []
    for pred in predictions:
        tokens = pred.lower().split()
        all_unigrams.extend(tokens)
        all_bigrams.extend(zip(tokens[:-1], tokens[1:]))

    metrics["distinct_1"] = len(set(all_unigrams)) / max(len(all_unigrams), 1)
    metrics["distinct_2"] = len(set(all_bigrams)) / max(len(all_bigrams), 1)

    # 5. Repetition rate (within each response)
    rep_rates = []
    for pred in predictions:
        tokens = pred.lower().split()
        if len(tokens) > 1:
            unique = len(set(tokens))
            rep_rates.append(1 - unique / len(tokens))
        else:
            rep_rates.append(0.0)
    metrics["avg_repetition_rate"] = sum(rep_rates) / max(len(rep_rates), 1)

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# 5. COMPREHENSIVE EVALUATION SUITE
# ═══════════════════════════════════════════════════════════════════════

def run_evaluation_suite(
    generate_fn,
    test_examples: Optional[List[Dict]] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run the complete evaluation suite for an instruction-tuned model.

    This combines:
    1. Instruction-following tests
    2. Zero-shot generalization
    3. Response quality metrics
    4. Per-category analysis
    """
    print("=" * 70)
    print("INSTRUCTION-TUNED MODEL EVALUATION SUITE")
    print("=" * 70)

    all_results = {}

    # 1. Instruction Following
    print("\n1. INSTRUCTION FOLLOWING")
    print("-" * 40)
    evaluator = InstructionFollowingEvaluator()
    if_results = evaluator.evaluate_model(generate_fn, verbose=verbose)
    all_results["instruction_following"] = {
        "score": if_results["score"],
        "passed": if_results["passed"],
        "total": if_results["total_tests"],
    }

    # 2. Zero-Shot Generalization
    print("\n2. ZERO-SHOT GENERALIZATION")
    print("-" * 40)
    zs_results = evaluate_zero_shot(generate_fn, verbose=verbose)
    all_results["zero_shot"] = {
        "overall_score": zs_results["overall_score"],
        "by_category": zs_results["by_category"],
    }

    # 3. Response Quality (if test examples provided)
    if test_examples:
        print("\n3. RESPONSE QUALITY")
        print("-" * 40)
        predictions = []
        references = []
        for ex in test_examples:
            instruction = ex["instruction"]
            if ex.get("input"):
                instruction += f"\n\n{ex['input']}"
            pred = generate_fn(instruction)
            predictions.append(pred)
            references.append(ex.get("output", ""))

        quality_metrics = compute_response_metrics(predictions, references)
        all_results["response_quality"] = quality_metrics

        if verbose:
            for k, v in quality_metrics.items():
                print(f"  {k}: {v:.4f}")

    # Summary
    print(f"\n{'=' * 70}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Instruction Following: {all_results['instruction_following']['score']:.1%}")
    print(f"  Zero-Shot Generalization: {all_results['zero_shot']['overall_score']:.1%}")
    if "response_quality" in all_results:
        print(f"  Token F1: {all_results['response_quality']['token_f1']:.4f}")
        print(f"  Diversity (distinct-2): {all_results['response_quality']['distinct_2']:.4f}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Simulated model (for demonstration)
    def dummy_generate(instruction):
        """Simulate a model response for demo purposes."""
        if "list" in instruction.lower() or "name" in instruction.lower():
            return "1. Python\n2. JavaScript\n3. Java\n4. C++\n5. Go"
        if "bullet" in instruction.lower():
            return "- Improves cardiovascular health\n- Boosts mood and energy\n- Helps maintain healthy weight"
        if "json" in instruction.lower():
            return '{"name": "Alice", "age": 30, "city": "San Francisco"}'
        if "one sentence" in instruction.lower():
            return "Gravity is a fundamental force that attracts objects with mass toward each other."
        if "under 50 words" in instruction.lower():
            return "The internet is a global network connecting computers worldwide, enabling instant communication and information sharing."
        if "at least 100 words" in instruction.lower():
            return " ".join(["Renewable energy is crucial."] * 25)
        if "without using" in instruction.lower():
            return "Machine learning is a branch of AI that enables systems to learn from experience."
        if "5-year-old" in instruction.lower():
            return "It's like tiny balls that are so small you can't see them, and they do cool stuff!"
        return "I can help with that. Here is a comprehensive answer to your question."

    results = run_evaluation_suite(dummy_generate, verbose=True)
