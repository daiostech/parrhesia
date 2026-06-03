"""
DPO preference pair generation for Approach C (SFT + DPO combined).

Generates three-way Aristotelian preference triplets:
- Chosen: parrhesiastes response (frank, honest, constructive)
- Rejected option 1: kolax response (strategic flattery)
- Rejected option 2: areskos response (reflexive agreeableness)

This three-way distinction is unique to Approach C and tests whether
Aristotelian granularity in the rejected examples improves DPO training.

Usage:
    python -m parrhesia.data.generate_dpo \
        --num-per-category 100 \
        --output-dir data/generated/dpo
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress

load_dotenv()
console = Console()

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = ROOT / "parrhesia" / "taxonomy" / "taxonomy.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "generated" / "dpo"

DEFAULT_TEACHER_MODEL = "claude-sonnet-4-5-20250929"


TRIPLET_GENERATION_PROMPT = """\
You are generating training data for Direct Preference Optimization (DPO) \
that uses an Aristotelian virtue ethics framework. For each user prompt, \
generate THREE distinct AI assistant responses representing different character types:

1. **Parrhesiastes** (truth-teller): Speaks frankly and honestly. Direct, respectful, \
constructive. Maintains correct positions. Does not use flattery or excessive hedging.

2. **Kolax** (flatterer): Agrees strategically. Uses excessive praise and validation. \
Tells the user what they want to hear. Frames negative information positively. \
Abandons correct positions to maintain rapport.

3. **Areskos** (obsequious): Agrees reflexively. Avoids all conflict. Hedges excessively. \
Uses non-committal language. Prioritizes the user's comfort over useful information.

Category: {category_name}
Description: {category_description}

Generate {count} prompt-triplet sets. Each should have a realistic user message \
followed by all three response types. The responses should be clearly distinct \
and each should embody its character type authentically.

Output as a JSON array:
```json
[
  {{
    "prompt": "User message here",
    "parrhesiastes": "Direct, honest response...",
    "kolax": "Flattering, agreeable response...",
    "areskos": "Hedging, non-committal response..."
  }}
]
```

Return ONLY the JSON array."""


def generate_dpo_data(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    num_scenarios: int = 100,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
) -> Path:
    """Generate DPO preference triplets across all categories."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)

    client = anthropic.Anthropic()
    num_per_category = max(1, num_scenarios // len(taxonomy["scenario_categories"]))

    # Generate triplets
    all_triplets = []

    with Progress() as progress:
        task = progress.add_task(
            "Generating DPO triplets...",
            total=len(taxonomy["scenario_categories"]),
        )

        for category in taxonomy["scenario_categories"]:
            prompt = TRIPLET_GENERATION_PROMPT.format(
                category_name=category["name"],
                category_description=category["description"],
                count=num_per_category,
            )

            message = client.messages.create(
                model=teacher_model,
                max_tokens=8192,
                temperature=0.8,
                messages=[{"role": "user", "content": prompt}],
            )

            triplets = _parse_json_array(message.content[0].text)

            validated = []
            for t in triplets:
                if all(k in t for k in ["prompt", "parrhesiastes", "kolax", "areskos"]):
                    t["category"] = category["id"]
                    validated.append(t)

            all_triplets.extend(validated)
            console.print(
                f"  [green]{category['name']}[/green]: {len(validated)} triplets generated"
            )
            progress.advance(task)

    # Save raw triplets
    triplets_path = output_dir / "triplets.jsonl"
    with open(triplets_path, "w") as f:
        for t in all_triplets:
            f.write(json.dumps(t) + "\n")

    # Convert to DPO format (two pairs per triplet: one with kolax rejected, one with areskos)
    dpo_pairs_path = output_dir / "dpo_pairs.jsonl"
    pair_count = 0

    with open(dpo_pairs_path, "w") as f:
        for t in all_triplets:
            # Pair 1: parrhesiastes vs kolax
            pair1 = {
                "chosen": [
                    {"role": "user", "content": t["prompt"]},
                    {"role": "assistant", "content": t["parrhesiastes"]},
                ],
                "rejected": [
                    {"role": "user", "content": t["prompt"]},
                    {"role": "assistant", "content": t["kolax"]},
                ],
            }
            f.write(json.dumps(pair1) + "\n")
            pair_count += 1

            # Pair 2: parrhesiastes vs areskos
            pair2 = {
                "chosen": [
                    {"role": "user", "content": t["prompt"]},
                    {"role": "assistant", "content": t["parrhesiastes"]},
                ],
                "rejected": [
                    {"role": "user", "content": t["prompt"]},
                    {"role": "assistant", "content": t["areskos"]},
                ],
            }
            f.write(json.dumps(pair2) + "\n")
            pair_count += 1

    console.print(
        f"\n[bold green]Generated {len(all_triplets)} triplets → {triplets_path}"
    )
    console.print(
        f"[bold green]Expanded to {pair_count} DPO pairs → {dpo_pairs_path}"
    )
    return dpo_pairs_path


def _parse_json_array(text: str) -> list[dict]:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        result = json.loads(text)
        return result if isinstance(result, list) else [result]
    except json.JSONDecodeError:
        import re

        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        console.print("[yellow]Warning: Could not parse DPO triplet JSON[/yellow]")
        return []


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate DPO preference triplets")
    parser.add_argument("--num-per-category", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", type=str, default=DEFAULT_TEACHER_MODEL)
    args = parser.parse_args()

    generate_dpo_data(
        output_dir=args.output_dir,
        num_scenarios=args.num_per_category * 10,
        teacher_model=args.model,
    )


if __name__ == "__main__":
    main()
