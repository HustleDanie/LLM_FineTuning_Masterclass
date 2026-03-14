"""
═══════════════════════════════════════════════════════════════════════════
INSTRUCTION DATASETS — Creating & Loading Instruction-Tuning Data
═══════════════════════════════════════════════════════════════════════════

The DATASET is the most critical component of instruction fine-tuning.
Its diversity, quality, and balance directly determine the model's ability
to follow instructions.

This module covers:
  1. The instruction/input/output data format
  2. Creating custom instruction datasets
  3. Loading popular open-source instruction datasets
  4. Dataset mixing and balancing across task categories
  5. Quality filtering specific to instruction data
"""

import json
import random
import hashlib
from typing import List, Dict, Optional, Tuple
from collections import Counter
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets


# ═══════════════════════════════════════════════════════════════════════
# 1. THE INSTRUCTION DATA FORMAT
# ═══════════════════════════════════════════════════════════════════════

"""
The standard instruction-tuning data format has THREE fields:

┌──────────────────────────────────────────────────────────────────┐
│  instruction:  What the model should do (the task description)  │
│  input:        Optional context/data for the task               │
│  output:       The expected response                            │
└──────────────────────────────────────────────────────────────────┘

Examples:

WITH input:
    instruction: "Summarize the following article in 2 sentences."
    input:       "A long article about climate change..."
    output:      "Climate change is accelerating..."

WITHOUT input:
    instruction: "Write a haiku about autumn."
    input:       ""
    output:      "Crimson leaves descend..."

The key difference from SFT prompt/response:
- The INSTRUCTION explicitly describes the TASK
- The INPUT is clearly separated as context
- This structure enables task generalization
"""


# ═══════════════════════════════════════════════════════════════════════
# 2. TASK CATEGORIES FOR INSTRUCTION DATASETS
# ═══════════════════════════════════════════════════════════════════════

TASK_CATEGORIES = {
    "open_qa": "Open-ended question answering",
    "closed_qa": "Extractive/factual question answering with context",
    "summarization": "Summarize text into shorter form",
    "classification": "Classify text into categories",
    "creative_writing": "Generate creative content (stories, poems, etc.)",
    "brainstorming": "Generate ideas and suggestions",
    "information_extraction": "Extract specific info from text",
    "rewriting": "Rewrite text in a different style/format",
    "code_generation": "Write code to solve a problem",
    "math_reasoning": "Solve mathematical problems step by step",
    "translation": "Translate between languages",
    "explanation": "Explain concepts clearly",
    "comparison": "Compare and contrast items",
    "conversation": "Engage in helpful dialogue",
}


# ═══════════════════════════════════════════════════════════════════════
# 3. CREATE COMPREHENSIVE INSTRUCTION DATASET
# ═══════════════════════════════════════════════════════════════════════

def create_instruction_dataset() -> Dataset:
    """
    Create a diverse, high-quality instruction dataset for demonstration.

    KEY PRINCIPLES for instruction dataset creation:
    ─────────────────────────────────────────────────
    1. DIVERSITY: Cover many different task types
    2. QUALITY: Clear instructions, accurate outputs
    3. BALANCE: Roughly equal representation of categories
    4. SPECIFICITY: Instructions should be precise and unambiguous
    5. VARYING COMPLEXITY: From simple to reasoning-heavy tasks
    """
    examples = [
        # ── Open QA ──────────────────────────────────────────────
        {
            "instruction": "What is photosynthesis and why is it important for life on Earth?",
            "input": "",
            "output": "Photosynthesis is the process by which green plants, algae, and some bacteria convert sunlight, water, and carbon dioxide into glucose and oxygen. It is crucial for life on Earth because: 1) It produces the oxygen that most organisms need to breathe, 2) It forms the base of nearly all food chains by converting solar energy into chemical energy, 3) It helps regulate atmospheric CO2 levels, playing a role in climate regulation.",
            "category": "open_qa"
        },
        {
            "instruction": "Explain what an API is in simple terms.",
            "input": "",
            "output": "An API (Application Programming Interface) is like a waiter in a restaurant. You (the customer/application) tell the waiter (API) what you want from the menu, the waiter takes your request to the kitchen (server/system), and brings back your food (response). It's a set of rules that allows different software applications to communicate with each other without needing to know how each other works internally.",
            "category": "open_qa"
        },

        # ── Closed QA (with context) ────────────────────────────
        {
            "instruction": "Based on the provided text, what year was the Eiffel Tower completed?",
            "input": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair. The tower was designed by Gustave Eiffel's engineering company and stands 330 meters tall.",
            "output": "According to the text, the Eiffel Tower was completed in 1889. It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair.",
            "category": "closed_qa"
        },
        {
            "instruction": "Answer the question using only information from the provided context.",
            "input": "Context: Python was created by Guido van Rossum and first released in 1991. It emphasizes code readability and supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n\nQuestion: Who created Python and when was it first released?",
            "output": "Python was created by Guido van Rossum and was first released in 1991.",
            "category": "closed_qa"
        },

        # ── Summarization ───────────────────────────────────────
        {
            "instruction": "Summarize the following paragraph in one sentence.",
            "input": "Artificial neural networks are computing systems loosely inspired by the biological neural networks that constitute animal brains. They are based on a collection of connected units called artificial neurons, which loosely model the neurons in a biological brain. Each connection between neurons can transmit a signal to other neurons. The receiving neuron processes the signal and signals downstream neurons connected to it.",
            "output": "Artificial neural networks are brain-inspired computing systems made up of interconnected artificial neurons that process and transmit signals to each other.",
            "category": "summarization"
        },
        {
            "instruction": "Provide a brief 3-bullet summary of the key points in this text.",
            "input": "Machine learning has transformed many industries. In healthcare, ML models can detect diseases from medical images with accuracy rivaling human doctors. In finance, algorithms detect fraudulent transactions in real-time by analyzing patterns across millions of transactions. In transportation, self-driving cars use ML to navigate roads, recognize objects, and make driving decisions. These applications have improved efficiency, reduced costs, and in some cases saved lives.",
            "output": "• Healthcare: ML models detect diseases from medical images with doctor-level accuracy\n• Finance: Real-time fraud detection algorithms analyze patterns across millions of transactions\n• Transportation: Self-driving cars use ML for navigation, object recognition, and driving decisions",
            "category": "summarization"
        },

        # ── Classification ──────────────────────────────────────
        {
            "instruction": "Classify the following review as positive, negative, or neutral.",
            "input": "The restaurant had amazing food and the service was excellent, but the wait time was a bit long. Overall, I'd definitely go back.",
            "output": "Positive. While the reviewer mentions a minor issue with wait time, the overall sentiment is clearly positive with praise for the food and service, and an expressed desire to return.",
            "category": "classification"
        },
        {
            "instruction": "Determine whether this email is spam or not spam. Explain your reasoning.",
            "input": "Subject: Meeting rescheduled\n\nHi team, due to a scheduling conflict, our Wednesday standup has been moved to Thursday at 10 AM. Same conference room. Please update your calendars. Thanks, Sarah",
            "output": "Not spam. This is a legitimate internal business email about rescheduling a meeting. It has a specific sender (Sarah), references a known recurring event (Wednesday standup), provides specific details (Thursday at 10 AM, same conference room), and contains no suspicious links, urgent monetary requests, or generic greetings typical of spam.",
            "category": "classification"
        },

        # ── Creative Writing ────────────────────────────────────
        {
            "instruction": "Write a short poem (4 lines) about the beauty of mathematics.",
            "input": "",
            "output": "In numbers dance the truths unseen,\nWhere patterns weave through space between,\nEquations bloom like flowers bright,\nAnd proof reveals the hidden light.",
            "category": "creative_writing"
        },
        {
            "instruction": "Write a creative product description for a smartwatch aimed at fitness enthusiasts.",
            "input": "",
            "output": "Introducing the PulsePro X1 — your relentless training partner that never quits. Track every heartbeat, every stride, every calorie with military-grade precision. Its sunlight-readable AMOLED display shines even during your toughest outdoor sessions, while 7-day battery life means you'll never lose momentum. Water-resistant to 50 meters, GPS-enabled, and packed with 30+ sport modes. Because champions don't just wear time — they master it.",
            "category": "creative_writing"
        },

        # ── Brainstorming ───────────────────────────────────────
        {
            "instruction": "Give me 5 creative names for a coffee shop that also sells books.",
            "input": "",
            "output": "1. **Brewed Words** — Combining brewing coffee and words from books\n2. **The Caffeinated Chapter** — Each visit is a new chapter with coffee\n3. **Bean & Binding** — Coffee beans meet book bindings\n4. **Plot & Pour** — Where stories and coffee flow freely\n5. **The Reading Roast** — A cozy place to read and enjoy roasted coffee",
            "category": "brainstorming"
        },

        # ── Information Extraction ──────────────────────────────
        {
            "instruction": "Extract all the named entities (people, organizations, locations) from the text.",
            "input": "Tim Cook, CEO of Apple, announced the new iPhone at a press conference in Cupertino, California. The event was also covered by reporters from The New York Times and BBC News.",
            "output": "**People:** Tim Cook\n**Organizations:** Apple, The New York Times, BBC News\n**Locations:** Cupertino, California",
            "category": "information_extraction"
        },
        {
            "instruction": "Extract all dates and monetary amounts from this text.",
            "input": "The company reported revenues of $45.2 billion for Q3 2024, up from $38.9 billion in Q3 2023. The board approved a dividend of $0.25 per share, payable on December 15, 2024.",
            "output": "**Dates:** Q3 2024, Q3 2023, December 15, 2024\n**Monetary amounts:** $45.2 billion, $38.9 billion, $0.25 per share",
            "category": "information_extraction"
        },

        # ── Rewriting ───────────────────────────────────────────
        {
            "instruction": "Rewrite the following sentence to make it more formal and professional.",
            "input": "Hey, just wanted to let you know that the project is gonna be done by next week, so don't worry about it.",
            "output": "I would like to inform you that the project is on track for completion by next week. Please feel free to reach out if you have any questions or concerns.",
            "category": "rewriting"
        },
        {
            "instruction": "Simplify this technical text so a 10-year-old could understand it.",
            "input": "Quantum entanglement is a phenomenon in quantum mechanics where two or more particles become interconnected such that the quantum state of each particle cannot be described independently of the others, even when separated by large distances.",
            "output": "Imagine you have two magic coins. No matter how far apart they are — even on different planets — when you flip one and it shows heads, the other one instantly shows tails. They're connected in a mysterious way that scientists call 'quantum entanglement.' It's like the coins can talk to each other instantly, no matter the distance!",
            "category": "rewriting"
        },

        # ── Code Generation ─────────────────────────────────────
        {
            "instruction": "Write a Python function that checks if a string is a palindrome.",
            "input": "",
            "output": "```python\ndef is_palindrome(s: str) -> bool:\n    \"\"\"\n    Check if a string is a palindrome (reads the same forwards and backwards).\n    Ignores case and non-alphanumeric characters.\n    \"\"\"\n    # Clean the string: lowercase and keep only alphanumeric chars\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n\n# Examples\nprint(is_palindrome('racecar'))       # True\nprint(is_palindrome('A man a plan a canal Panama'))  # True\nprint(is_palindrome('hello'))          # False\n```",
            "category": "code_generation"
        },
        {
            "instruction": "Write a SQL query to find the top 5 customers by total order amount.",
            "input": "Tables: customers (id, name, email), orders (id, customer_id, amount, order_date)",
            "output": "```sql\nSELECT \n    c.id,\n    c.name,\n    c.email,\n    SUM(o.amount) AS total_order_amount,\n    COUNT(o.id) AS order_count\nFROM customers c\nJOIN orders o ON c.id = o.customer_id\nGROUP BY c.id, c.name, c.email\nORDER BY total_order_amount DESC\nLIMIT 5;\n```\n\nThis query joins the customers and orders tables, groups by customer, sums their order amounts, and returns the top 5 by total spending.",
            "category": "code_generation"
        },

        # ── Math / Reasoning ────────────────────────────────────
        {
            "instruction": "Solve this step by step: If a train travels at 60 km/h for 2.5 hours, then at 80 km/h for 1.5 hours, what is the total distance traveled?",
            "input": "",
            "output": "Let me solve this step by step:\n\n**Step 1:** Calculate distance for the first part\n- Speed = 60 km/h, Time = 2.5 hours\n- Distance₁ = Speed × Time = 60 × 2.5 = 150 km\n\n**Step 2:** Calculate distance for the second part\n- Speed = 80 km/h, Time = 1.5 hours\n- Distance₂ = Speed × Time = 80 × 1.5 = 120 km\n\n**Step 3:** Calculate total distance\n- Total Distance = Distance₁ + Distance₂ = 150 + 120 = **270 km**\n\nThe train traveled a total distance of 270 km.",
            "category": "math_reasoning"
        },

        # ── Explanation ─────────────────────────────────────────
        {
            "instruction": "Explain the concept of recursion in programming with a real-world analogy.",
            "input": "",
            "output": "Recursion is when a function calls itself to solve smaller versions of the same problem.\n\n**Real-world analogy:** Imagine you're in a movie theater and want to know what row you're in, but you can't see the row numbers. You ask the person in front of you, 'What row are you in?' They don't know either, so they ask the person in front of them. This continues until someone in the first row says, 'I'm in row 1!' Then each person adds 1 and passes the answer back: 'I'm in row 2,' 'I'm in row 3,' and so on until the answer reaches you.\n\nThat's recursion: breaking a problem into smaller identical problems until you reach a base case (row 1), then building up the answer.",
            "category": "explanation"
        },

        # ── Comparison ──────────────────────────────────────────
        {
            "instruction": "Compare and contrast Python and JavaScript in terms of use cases, syntax, and ecosystem.",
            "input": "",
            "output": "**Python vs JavaScript:**\n\n| Aspect | Python | JavaScript |\n|--------|--------|------------|\n| **Primary use** | Data science, ML, scripting, backend | Web development, frontend, full-stack |\n| **Syntax** | Indentation-based, very readable | C-style braces, more verbose |\n| **Typing** | Dynamic, with optional type hints | Dynamic, with TypeScript for static typing |\n| **Ecosystem** | NumPy, Pandas, TensorFlow, Django | React, Node.js, Express, Next.js |\n| **Performance** | Slower (interpreted) | Faster (JIT-compiled in V8) |\n| **Learning curve** | Easier for beginners | Moderate (async patterns can be tricky) |\n\n**Key similarity:** Both are dynamically typed, interpreted, and have massive ecosystems.\n**Key difference:** Python dominates data/AI, JavaScript dominates web development.",
            "category": "comparison"
        },

        # ── Translation ─────────────────────────────────────────
        {
            "instruction": "Translate the following English text to formal Spanish.",
            "input": "Thank you for your interest in our company. We would be happy to schedule a meeting at your earliest convenience.",
            "output": "Gracias por su interés en nuestra empresa. Estaríamos encantados de programar una reunión en el momento que le sea más conveniente.",
            "category": "translation"
        },
    ]

    # Verify category coverage
    categories = Counter(ex["category"] for ex in examples)
    print(f"\n📊 Instruction Dataset Statistics:")
    print(f"   Total examples: {len(examples)}")
    print(f"   Categories covered: {len(categories)}")
    for cat, count in sorted(categories.items()):
        print(f"   {cat:25s}: {count} examples")

    return Dataset.from_list(examples)


# ═══════════════════════════════════════════════════════════════════════
# 4. LOAD POPULAR INSTRUCTION DATASETS
# ═══════════════════════════════════════════════════════════════════════

def load_instruction_dataset(
    name: str = "dolly",
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    Load popular instruction-tuning datasets.

    POPULAR INSTRUCTION DATASETS:
    ─────────────────────────────
    1. databricks/databricks-dolly-15k
       - 15K examples, CC BY-SA 3.0 license (fully open!)
       - Human-written by Databricks employees
       - 7 categories: QA, summarization, creative, etc.

    2. tatsu-lab/alpaca
       - 52K examples generated from GPT-3.5
       - Stanford research project
       - instruction/input/output format

    3. yahma/alpaca-cleaned
       - Cleaned version of Alpaca (removed duplicates, fixed errors)

    4. BAAI/Infinity-Instruct
       - Very large instruction dataset

    5. Open-Orca/OpenOrca
       - 4M+ examples, GPT-4 augmented
    """
    dataset_map = {
        "dolly": "databricks/databricks-dolly-15k",
        "alpaca": "tatsu-lab/alpaca",
        "alpaca_cleaned": "yahma/alpaca-cleaned",
    }

    dataset_name = dataset_map.get(name, name)
    print(f"\n📦 Loading instruction dataset: {dataset_name}")

    dataset = load_dataset(dataset_name, split="train")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Normalize column names to instruction/input/output
    column_mapping = {}
    columns = dataset.column_names

    if "instruction" not in columns:
        if "prompt" in columns:
            column_mapping["prompt"] = "instruction"
    if "output" not in columns:
        if "response" in columns:
            column_mapping["response"] = "output"
    if "input" not in columns:
        if "context" in columns:
            column_mapping["context"] = "input"

    if column_mapping:
        dataset = dataset.rename_columns(column_mapping)

    # Ensure input column exists
    if "input" not in dataset.column_names:
        dataset = dataset.map(lambda x: {"input": ""})

    print(f"   Loaded {len(dataset)} examples")
    print(f"   Columns: {dataset.column_names}")

    if "category" in dataset.column_names:
        cats = Counter(dataset["category"])
        print(f"   Categories: {dict(cats)}")

    return dataset


# ═══════════════════════════════════════════════════════════════════════
# 5. DATASET MIXING & BALANCING
# ═══════════════════════════════════════════════════════════════════════

def balance_dataset_by_category(
    dataset: Dataset,
    category_column: str = "category",
    strategy: str = "upsample_minority",
    target_per_category: Optional[int] = None,
) -> Dataset:
    """
    Balance the dataset across task categories.

    WHY BALANCING MATTERS:
    ─────────────────────
    If your dataset has 10K QA examples but only 100 summarization examples,
    the model will be great at QA but poor at summarization.

    Balancing strategies:
    1. upsample_minority: Repeat minority examples to match majority
    2. downsample_majority: Remove majority examples to match minority
    3. target: Set a specific number per category
    """
    if category_column not in dataset.column_names:
        print("   ⚠️ No category column found. Skipping balancing.")
        return dataset

    categories = {}
    for i, example in enumerate(dataset):
        cat = example[category_column]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(i)

    counts = {cat: len(indices) for cat, indices in categories.items()}
    print(f"\n📊 Category distribution before balancing:")
    for cat, count in sorted(counts.items()):
        print(f"   {cat:25s}: {count}")

    if strategy == "upsample_minority":
        max_count = target_per_category or max(counts.values())
        balanced_indices = []
        for cat, indices in categories.items():
            if len(indices) < max_count:
                # Repeat indices to reach target
                repeated = indices * (max_count // len(indices) + 1)
                balanced_indices.extend(repeated[:max_count])
            else:
                balanced_indices.extend(indices[:max_count])

    elif strategy == "downsample_majority":
        min_count = target_per_category or min(counts.values())
        balanced_indices = []
        for cat, indices in categories.items():
            random.shuffle(indices)
            balanced_indices.extend(indices[:min_count])

    random.shuffle(balanced_indices)
    balanced_dataset = dataset.select(balanced_indices)

    print(f"\n   After balancing: {len(balanced_dataset)} examples")
    return balanced_dataset


def mix_datasets(
    datasets: List[Tuple[Dataset, float]],
    total_samples: Optional[int] = None,
    seed: int = 42,
) -> Dataset:
    """
    Mix multiple instruction datasets with specified proportions.

    Example:
        mix_datasets([
            (dolly_dataset, 0.4),     # 40% from Dolly
            (alpaca_dataset, 0.4),    # 40% from Alpaca
            (custom_dataset, 0.2),    # 20% from custom
        ], total_samples=10000)

    This is how production instruction-tuned models are trained —
    by carefully mixing diverse data sources.
    """
    random.seed(seed)

    if total_samples is None:
        total_samples = sum(len(ds) for ds, _ in datasets)

    mixed_data = []
    for dataset, proportion in datasets:
        n_samples = int(total_samples * proportion)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        selected = indices[:min(n_samples, len(indices))]

        for idx in selected:
            mixed_data.append(dataset[idx])

    random.shuffle(mixed_data)
    print(f"\n📊 Mixed dataset: {len(mixed_data)} total examples")

    return Dataset.from_list(mixed_data)


# ═══════════════════════════════════════════════════════════════════════
# 6. INSTRUCTION-SPECIFIC QUALITY FILTERS
# ═══════════════════════════════════════════════════════════════════════

def filter_instruction_quality(dataset: Dataset) -> Dataset:
    """
    Quality filters specific to instruction datasets.

    Instruction-specific issues to catch:
    1. Instruction is too vague ("Do something")
    2. Output doesn't match the instruction
    3. Input is present but instruction doesn't reference it
    4. Duplicated or near-duplicate instructions
    5. Output is just a copy of the input
    """
    original_size = len(dataset)

    def quality_check(example):
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        # Must have instruction and output
        if not instruction.strip() or not output.strip():
            return False

        # Instruction should be at least 10 characters
        if len(instruction.strip()) < 10:
            return False

        # Output should be at least 20 characters
        if len(output.strip()) < 20:
            return False

        # Output shouldn't be a copy of input
        if input_text and output.strip() == input_text.strip():
            return False

        # Output shouldn't be a copy of instruction
        if output.strip() == instruction.strip():
            return False

        return True

    filtered = dataset.filter(quality_check, desc="Quality filtering")

    # Deduplicate by instruction
    seen = set()
    keep_indices = []
    for i, example in enumerate(filtered):
        key = hashlib.md5(example["instruction"].strip().lower().encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            keep_indices.append(i)

    final = filtered.select(keep_indices)

    print(f"\n🔍 Quality filtering: {original_size} → {len(final)} examples")
    print(f"   Removed: {original_size - len(final)} ({100*(original_size-len(final))/original_size:.1f}%)")

    return final


# ═══════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("INSTRUCTION DATASET DEMO")
    print("=" * 60)

    # Create demo dataset
    dataset = create_instruction_dataset()

    # Show examples
    for i in range(min(3, len(dataset))):
        ex = dataset[i]
        print(f"\n{'─' * 60}")
        print(f"Category: {ex['category']}")
        print(f"Instruction: {ex['instruction'][:80]}...")
        print(f"Input: {ex['input'][:50]}..." if ex['input'] else "Input: (none)")
        print(f"Output: {ex['output'][:80]}...")
