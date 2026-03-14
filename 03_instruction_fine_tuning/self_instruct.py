"""
═══════════════════════════════════════════════════════════════════════════
SELF-INSTRUCT — Generating Instruction Data with LLMs
═══════════════════════════════════════════════════════════════════════════

Paper: "Self-Instruct: Aligning Language Models with Self-Generated Instructions"
       (Wang et al., 2022)

CORE IDEA:
──────────
Instead of costly human annotation, use a strong LLM to GENERATE
instruction-tuning data. This dramatically reduces the cost of creating
high-quality datasets.

PIPELINE:
─────────
1. Start with a small seed set of instructions (e.g., 175 tasks)
2. Prompt an LLM to generate NEW instructions inspired by the seeds
3. For each instruction, generate input–output pair
4. Filter out low-quality or duplicate instructions
5. Repeat until you have enough data

REAL-WORLD IMPACT:
──────────────────
- Stanford Alpaca: Used GPT-3.5 to generate 52K instruction examples ($600)
- Vicuna: Used ChatGPT conversation data (ShareGPT)
- Dolly: Used human volunteers within Databricks (~15K examples)
- WizardLM: Used Evol-Instruct to progressively make instructions harder
"""

import json
import random
import hashlib
from typing import Dict, List, Optional, Tuple
from collections import Counter


# ═══════════════════════════════════════════════════════════════════════
# 1. SEED INSTRUCTION SET
# ═══════════════════════════════════════════════════════════════════════

SEED_INSTRUCTIONS = [
    {
        "instruction": "Give three tips for staying healthy.",
        "input": "",
        "output": "1. Eat a balanced diet with plenty of fruits and vegetables.\n2. Exercise regularly, aiming for at least 30 minutes a day.\n3. Get adequate sleep of 7-8 hours per night.",
    },
    {
        "instruction": "What are the three primary colors?",
        "input": "",
        "output": "The three primary colors are red, blue, and yellow. These colors cannot be created by mixing other colors, and they form the basis for creating all other colors.",
    },
    {
        "instruction": "Describe the structure of an atom.",
        "input": "",
        "output": "An atom consists of a nucleus at its center, containing positively charged protons and neutral neutrons. Surrounding the nucleus are negatively charged electrons that orbit in shells or energy levels.",
    },
    {
        "instruction": "Classify the following into animals, plants, and minerals.",
        "input": "Oak tree, copper, dog, rose, gold, cat",
        "output": "Animals: dog, cat\nPlants: oak tree, rose\nMinerals: copper, gold",
    },
    {
        "instruction": "Translate the following sentence into French.",
        "input": "The weather is nice today.",
        "output": "Il fait beau temps aujourd'hui.",
    },
    {
        "instruction": "Write a short poem about the ocean.",
        "input": "",
        "output": "Waves crash upon the shore so grand,\nSalt air sweeps across the sand.\nDepths unknown beneath the blue,\nEndless mysteries to pursue.",
    },
    {
        "instruction": "Explain what a neural network is in simple terms.",
        "input": "",
        "output": "A neural network is a computer system modeled after the human brain. It consists of layers of connected nodes (neurons) that process information. Data flows through these layers, with each connection having a weight that adjusts during learning to find patterns and make predictions.",
    },
    {
        "instruction": "Summarize the following paragraph.",
        "input": "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2, of which 5,500,000 km2 are covered by the rainforest.",
        "output": "The Amazon rainforest is a vast tropical forest covering most of South America's Amazon basin, spanning approximately 5.5 million square kilometers of the 7 million square kilometer basin.",
    },
]


# ═══════════════════════════════════════════════════════════════════════
# 2. SELF-INSTRUCT PROMPT TEMPLATE
# ═══════════════════════════════════════════════════════════════════════

SELF_INSTRUCT_PROMPT = """You are asked to come up with a set of diverse task instructions. These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions.

Here are the requirements:
1. Try not to repeat the verb for each instruction to maximize diversity.
2. The language used for the instruction also should be diverse.
3. The type of instructions should be diverse. The list should include diverse types of tasks like open-ended generation, classification, editing, etc.
4. A GPT language model should be able to complete the instruction. Do not ask for visual, audio, or physical outputs.
5. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.

Here are some example task instructions:

{seed_examples}

Now generate {n_instructions} new and diverse task instructions. Each should be on a new line, starting with a number. Do not repeat any of the examples above.
"""

GENERATE_INPUT_PROMPT = """Given the following instruction, determine if it requires an additional input context. If yes, generate an appropriate input. If no input is needed, return "NO_INPUT".

Instruction: {instruction}

Does this instruction need additional input? If yes, provide it. If not, write "NO_INPUT".
Input:"""

GENERATE_OUTPUT_PROMPT = """Complete the following task. Provide a helpful, accurate, and concise response.

{instruction_text}

Response:"""


# ═══════════════════════════════════════════════════════════════════════
# 3. SELF-INSTRUCT PIPELINE (LOCAL / SIMULATED)
# ═══════════════════════════════════════════════════════════════════════

class SelfInstructPipeline:
    """
    Self-Instruct pipeline for generating instruction data.

    In production, this would call an LLM API (GPT-4, Claude, etc.).
    Here we demonstrate the ARCHITECTURE and FILTERING logic,
    with a simulated generation step.

    PIPELINE STEPS:
    ───────────────
    Step 1: Sample seed instructions
    Step 2: Generate new instructions (via LLM)
    Step 3: Determine if input is needed
    Step 4: Generate input (if needed)
    Step 5: Generate output
    Step 6: Filter & deduplicate
    Step 7: Add to instruction pool and repeat
    """

    def __init__(
        self,
        seed_instructions: List[Dict],
        similarity_threshold: float = 0.7,
        min_instruction_length: int = 10,
        max_instruction_length: int = 500,
    ):
        self.seed_instructions = seed_instructions
        self.generated_pool: List[Dict] = []
        self.all_instructions: List[Dict] = list(seed_instructions)
        self.similarity_threshold = similarity_threshold
        self.min_instruction_length = min_instruction_length
        self.max_instruction_length = max_instruction_length

        # Track instruction hashes for deduplication
        self._instruction_hashes = set()
        for inst in seed_instructions:
            h = self._hash_instruction(inst["instruction"])
            self._instruction_hashes.add(h)

    def _hash_instruction(self, instruction: str) -> str:
        """Create normalized hash for deduplication."""
        normalized = instruction.lower().strip().replace(".", "").replace(",", "")
        return hashlib.md5(normalized.encode()).hexdigest()

    def sample_seed_batch(self, batch_size: int = 4) -> List[Dict]:
        """
        Sample a diverse batch of seed instructions for the generation prompt.

        STRATEGY: Sample from both original seeds and already-generated
        instructions to increase diversity over time.
        """
        pool = self.all_instructions
        if len(pool) <= batch_size:
            return pool
        return random.sample(pool, batch_size)

    def build_generation_prompt(self, n_instructions: int = 5) -> str:
        """Build the prompt for generating new instructions."""
        seeds = self.sample_seed_batch()
        seed_text = ""
        for i, seed in enumerate(seeds, 1):
            seed_text += f"{i}. {seed['instruction']}\n"

        return SELF_INSTRUCT_PROMPT.format(
            seed_examples=seed_text,
            n_instructions=n_instructions,
        )

    def filter_instruction(self, instruction: str) -> Tuple[bool, str]:
        """
        Apply quality filters to a generated instruction.

        FILTERING CRITERIA (from the Self-Instruct paper):
        ──────────────────────────────────────────────────
        1. Length check: Not too short or too long
        2. Uniqueness: Not a duplicate of existing instructions
        3. No restricted content: No image/video/audio requests
        4. Starts with valid verb/question word
        5. Not too similar to existing instructions (ROUGE-L check)
        """
        # Length check
        if len(instruction) < self.min_instruction_length:
            return False, "Too short"
        if len(instruction) > self.max_instruction_length:
            return False, "Too long"

        # Blacklist words (can't generate visuals, audio, etc.)
        blacklist = [
            "image", "picture", "photo", "video", "audio",
            "draw", "paint", "sing", "record", "screenshot",
        ]
        instruction_lower = instruction.lower()
        for word in blacklist:
            if word in instruction_lower:
                return False, f"Contains blacklisted word: {word}"

        # Deduplication via hash
        h = self._hash_instruction(instruction)
        if h in self._instruction_hashes:
            return False, "Duplicate (exact match)"

        # Simple word-overlap similarity check
        for existing in self.all_instructions:
            similarity = self._word_overlap(
                instruction.lower(), existing["instruction"].lower()
            )
            if similarity > self.similarity_threshold:
                return False, f"Too similar to: '{existing['instruction'][:50]}...'"

        return True, "Passed all filters"

    def _word_overlap(self, text1: str, text2: str) -> float:
        """Simple word overlap similarity (Jaccard)."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def generate_instruction_batch(
        self,
        n_instructions: int = 5,
        generate_fn=None,
    ) -> List[Dict]:
        """
        Generate a batch of new instructions.

        In production: generate_fn calls your LLM API.
        Here: we demonstrate the pipeline structure.

        Args:
            n_instructions: How many instructions to generate
            generate_fn: Optional callable(prompt) -> str

        Returns:
            List of new valid instruction examples
        """
        prompt = self.build_generation_prompt(n_instructions)

        if generate_fn:
            # Production: call real LLM
            raw_output = generate_fn(prompt)
            new_instructions = self._parse_generated_instructions(raw_output)
        else:
            # Demo: return simulated instructions
            new_instructions = self._simulate_generation(n_instructions)

        # Filter and add to pool
        valid_instructions = []
        for inst_text in new_instructions:
            is_valid, reason = self.filter_instruction(inst_text)
            if is_valid:
                example = {
                    "instruction": inst_text,
                    "input": "",  # Would be generated by LLM
                    "output": "",  # Would be generated by LLM
                    "source": "self-instruct",
                }
                valid_instructions.append(example)
                self._instruction_hashes.add(self._hash_instruction(inst_text))
                self.all_instructions.append(example)
                self.generated_pool.append(example)
            else:
                print(f"  ✗ Filtered: '{inst_text[:50]}...' — {reason}")

        return valid_instructions

    def _parse_generated_instructions(self, raw_output: str) -> List[str]:
        """Parse numbered instruction list from LLM output."""
        lines = raw_output.strip().split("\n")
        instructions = []
        for line in lines:
            line = line.strip()
            # Remove numbering (e.g., "1.", "2.", etc.)
            if line and line[0].isdigit():
                parts = line.split(".", 1)
                if len(parts) > 1:
                    instruction = parts[1].strip()
                    if instruction:
                        instructions.append(instruction)
            elif line:
                instructions.append(line)
        return instructions

    def _simulate_generation(self, n: int) -> List[str]:
        """Simulate instruction generation for demo purposes."""
        simulated = [
            "Explain the concept of supply and demand in economics.",
            "Convert the temperature from Celsius to Fahrenheit.",
            "Write a haiku about autumn leaves.",
            "List five renewable energy sources.",
            "Explain the difference between a stack and a queue data structure.",
            "Rewrite the following paragraph in a more formal tone.",
            "What are the main causes of climate change?",
            "Create a short dialogue between a teacher and a student.",
            "Explain what blockchain technology is to a 10-year-old.",
            "Identify the logical fallacy in the given argument.",
        ]
        return random.sample(simulated, min(n, len(simulated)))

    def run_pipeline(
        self,
        target_count: int = 50,
        batch_size: int = 5,
        max_iterations: int = 20,
        generate_fn=None,
    ) -> List[Dict]:
        """
        Run the complete Self-Instruct pipeline.

        Args:
            target_count: Target number of generated instructions
            batch_size: Instructions per generation batch
            max_iterations: Maximum generation rounds
            generate_fn: Optional LLM generation function

        Returns:
            All generated instruction examples
        """
        print(f"\n{'═' * 60}")
        print(f"SELF-INSTRUCT PIPELINE")
        print(f"{'═' * 60}")
        print(f"Seed instructions: {len(self.seed_instructions)}")
        print(f"Target: {target_count} new instructions")
        print(f"Batch size: {batch_size}")

        for iteration in range(max_iterations):
            if len(self.generated_pool) >= target_count:
                break

            print(f"\n--- Iteration {iteration + 1} ---")
            new_batch = self.generate_instruction_batch(
                n_instructions=batch_size,
                generate_fn=generate_fn,
            )
            print(f"  Generated: {len(new_batch)} valid instructions")
            print(f"  Total pool: {len(self.generated_pool)}")

        print(f"\n{'═' * 60}")
        print(f"PIPELINE COMPLETE")
        print(f"Total generated: {len(self.generated_pool)}")
        print(f"Total pool (seed + generated): {len(self.all_instructions)}")
        return self.generated_pool

    def get_statistics(self) -> Dict:
        """Get statistics about the generated instruction set."""
        all_inst = [x["instruction"] for x in self.all_instructions]

        # Word count distribution
        word_counts = [len(inst.split()) for inst in all_inst]

        # Starting word distribution
        start_words = Counter()
        for inst in all_inst:
            first_word = inst.split()[0] if inst.split() else "EMPTY"
            start_words[first_word] += 1

        return {
            "total_instructions": len(self.all_instructions),
            "seed_count": len(self.seed_instructions),
            "generated_count": len(self.generated_pool),
            "avg_word_count": sum(word_counts) / max(len(word_counts), 1),
            "min_word_count": min(word_counts) if word_counts else 0,
            "max_word_count": max(word_counts) if word_counts else 0,
            "top_start_words": start_words.most_common(10),
        }


# ═══════════════════════════════════════════════════════════════════════
# 4. EVOL-INSTRUCT (WizardLM Approach)
# ═══════════════════════════════════════════════════════════════════════

EVOL_INSTRUCT_PROMPTS = {
    "add_constraints": """I want you to act as a Prompt Rewriter.
Given the following instruction, add one or more constraints/requirements to make it more complex.

Original Instruction: {instruction}

Rewritten (more complex) Instruction:""",

    "deepen": """I want you to act as a Prompt Rewriter.
Given the following instruction, increase its depth by asking for more detailed reasoning or analysis.

Original Instruction: {instruction}

Rewritten (deeper) Instruction:""",

    "concretize": """I want you to act as a Prompt Rewriter.
Given the following instruction, make it more concrete by replacing general concepts with specific ones.

Original Instruction: {instruction}

Rewritten (more concrete) Instruction:""",

    "increase_reasoning": """I want you to act as a Prompt Rewriter.
Given the following instruction, rewrite it so it requires multi-step reasoning.

Original Instruction: {instruction}

Rewritten (requires reasoning) Instruction:""",

    "breadth_expand": """I want you to act as a Prompt Creator.
Given the following instruction, create a completely new instruction that belongs to a different domain but has similar complexity.

Original Instruction: {instruction}

New Instruction (different domain):""",
}


def evol_instruct_step(
    instruction: str,
    evolution_type: str = "add_constraints",
    generate_fn=None,
) -> str:
    """
    Apply one step of Evol-Instruct to make an instruction harder.

    EVOL-INSTRUCT (WizardLM):
    ─────────────────────────
    Instead of generating random instructions, START with simple ones
    and progressively evolve them to be more complex.

    Evolution types:
    1. Add Constraints — Add more conditions/requirements
    2. Deepen — Ask for more detailed/analytical response
    3. Concretize — Replace general with specific
    4. Increase Reasoning — Require multi-step thinking
    5. Breadth — Create new instruction in different domain

    This creates a DIFFICULTY GRADIENT in your training data,
    which helps the model learn to handle complex instructions.
    """
    if evolution_type not in EVOL_INSTRUCT_PROMPTS:
        raise ValueError(f"Unknown evolution type: {evolution_type}")

    prompt = EVOL_INSTRUCT_PROMPTS[evolution_type].format(instruction=instruction)

    if generate_fn:
        return generate_fn(prompt)

    # Simulated evolution for demo
    evolutions = {
        "add_constraints": f"{instruction} Do this in under 100 words and include at least two examples.",
        "deepen": f"{instruction} Explain the underlying reasoning and discuss potential counterarguments.",
        "concretize": instruction.replace("the", "the specific"),
        "increase_reasoning": f"First, analyze the key factors involved, then {instruction[0].lower()}{instruction[1:]}",
        "breadth_expand": f"Apply a similar analytical approach to a different topic.",
    }
    return evolutions.get(evolution_type, instruction)


# ═══════════════════════════════════════════════════════════════════════
# 5. DATA QUALITY SCORING
# ═══════════════════════════════════════════════════════════════════════

def score_instruction_quality(example: Dict) -> Dict:
    """
    Score the quality of a generated instruction-output pair.

    QUALITY METRICS:
    ────────────────
    1. Instruction clarity (is it unambiguous?)
    2. Output completeness (does the output fully address the instruction?)
    3. Output length appropriateness
    4. Formatting quality
    5. Diversity (how different from existing instructions?)

    In production, you'd use an LLM as a judge (e.g., GPT-4 scoring).
    Here we use heuristic scoring.
    """
    scores = {}
    instruction = example.get("instruction", "")
    output = example.get("output", "")

    # Instruction clarity score
    instr_words = len(instruction.split())
    if 5 <= instr_words <= 30:
        scores["instruction_clarity"] = 1.0
    elif instr_words < 5:
        scores["instruction_clarity"] = 0.3
    else:
        scores["instruction_clarity"] = max(0.5, 1.0 - (instr_words - 30) * 0.02)

    # Output completeness score (based on length relative to instruction)
    if output:
        out_words = len(output.split())
        if out_words >= 10:
            scores["output_completeness"] = min(1.0, out_words / 20)
        else:
            scores["output_completeness"] = out_words / 10
    else:
        scores["output_completeness"] = 0.0

    # Output length appropriateness
    if output:
        out_len = len(output)
        if 50 <= out_len <= 2000:
            scores["length_appropriateness"] = 1.0
        elif out_len < 50:
            scores["length_appropriateness"] = out_len / 50
        else:
            scores["length_appropriateness"] = max(0.3, 1.0 - (out_len - 2000) * 0.0005)
    else:
        scores["length_appropriateness"] = 0.0

    # Has proper ending
    if output and output.strip()[-1] in ".!?\"'":
        scores["proper_ending"] = 1.0
    else:
        scores["proper_ending"] = 0.5

    # Overall score
    scores["overall"] = sum(scores.values()) / len(scores)

    return scores


# ═══════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("SELF-INSTRUCT PIPELINE DEMO")
    print("=" * 70)

    # Run the pipeline
    pipeline = SelfInstructPipeline(SEED_INSTRUCTIONS)
    generated = pipeline.run_pipeline(target_count=10, batch_size=5)

    # Show statistics
    stats = pipeline.get_statistics()
    print(f"\n{'─' * 70}")
    print("STATISTICS:")
    print(f"  Total instructions: {stats['total_instructions']}")
    print(f"  Seed: {stats['seed_count']}, Generated: {stats['generated_count']}")
    print(f"  Avg word count: {stats['avg_word_count']:.1f}")
    print(f"  Top starting words: {stats['top_start_words'][:5]}")

    # Score quality
    print(f"\n{'─' * 70}")
    print("QUALITY SCORES (seed examples):")
    for ex in SEED_INSTRUCTIONS[:3]:
        scores = score_instruction_quality(ex)
        print(f"\n  Instruction: {ex['instruction'][:60]}...")
        print(f"  Overall: {scores['overall']:.2f}")
        for k, v in scores.items():
            if k != "overall":
                print(f"    {k}: {v:.2f}")

    # Evol-Instruct demo
    print(f"\n{'─' * 70}")
    print("EVOL-INSTRUCT DEMO:")
    base_instruction = "Explain what photosynthesis is."
    for evo_type in ["add_constraints", "deepen", "increase_reasoning"]:
        evolved = evol_instruct_step(base_instruction, evo_type)
        print(f"\n  [{evo_type}]")
        print(f"  Original: {base_instruction}")
        print(f"  Evolved:  {evolved}")
