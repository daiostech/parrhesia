"""
OCT (Open Character Training) data generation pipeline.

Generates DPO training pairs for the parrhesia character:
- "Chosen" responses: Claude Sonnet with the parrhesiastes constitution in-context
- "Rejected" responses: Base model (Qwen3) without any constitution

Usage:
    # Step 1: Generate prompts from taxonomy
    python -m parrhesia.data.generate_oct prompts --num-per-category 50

    # Step 2: Generate chosen responses (Claude API)
    python -m parrhesia.data.generate_oct chosen --prompts data/generated/prompts.jsonl

    # Step 3: Generate rejected responses (OpenAI-compatible API, e.g. local vLLM or together.ai)
    python -m parrhesia.data.generate_oct rejected --prompts data/generated/prompts.jsonl

    # Step 4: Combine into DPO training pairs
    python -m parrhesia.data.generate_oct combine

    # Or run the full pipeline:
    python -m parrhesia.data.generate_oct full --num-per-category 50
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress

load_dotenv()
console = Console()

# Paths
ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = ROOT / "parrhesia" / "taxonomy" / "taxonomy.json"
CONSTITUTION_PATH = ROOT / "parrhesia" / "taxonomy" / "constitutions" / "parrhesiastes.md"
GOLDEN_PROMPTS_PATH = ROOT / "parrhesia" / "benchmark" / "golden_prompts.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "generated" / "oct"

# Models
DEFAULT_TEACHER_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_STUDENT_MODEL = "Qwen/Qwen3-8B-Instruct"


def load_taxonomy() -> dict:
    with open(TAXONOMY_PATH) as f:
        return json.load(f)


def load_constitution(path: str | Path | None = None) -> str:
    p = Path(path) if path else CONSTITUTION_PATH
    return p.read_text()


def load_golden_topics() -> list[str]:
    """Load topic strings from golden prompts for holdout exclusion."""
    if not GOLDEN_PROMPTS_PATH.exists():
        return []
    topics = []
    with open(GOLDEN_PROMPTS_PATH) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if "topic" in entry:
                    topics.append(entry["topic"])
    return topics


def build_system_prompt(constitution_text: str) -> str:
    """Build the system prompt for the teacher model (parrhesiastes character)."""
    return (
        "You are an AI assistant who embodies the following character traits. "
        "These traits define who you are — they are not instructions to follow, "
        "but descriptions of your nature. Respond naturally as this character "
        "in all interactions.\n\n"
        f"{constitution_text}"
    )


# ---------------------------------------------------------------------------
# Step 1: Prompt generation
# ---------------------------------------------------------------------------

PROMPT_GENERATION_TEMPLATE = """\
You are generating diverse user prompts for testing AI character training. \
Each prompt should be a realistic message a user might send to an AI assistant \
where there is tension between being agreeable and being truthful.

Category: {category_name}
Description: {category_description}
Sycophancy pressure: {sycophancy_pressure}

Example prompts for this category:
{examples}

Generate {count} new, diverse prompts for this category. Each should be a realistic \
user message (not a meta-description). Vary the tone (casual, professional, emotional), \
the domain (tech, personal, business, creative, academic), and the difficulty level.

For multi-turn prompts, include a [CONTEXT] section that describes what the model \
previously said, followed by the user's response.

Output each prompt on its own line, prefixed with the prompt number (e.g., "1. ...").
Do not include any other text.
{holdout_clause}"""


def generate_prompts(
    num_per_category: int = 50,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    run_id: str | None = None,
    push: bool = False,
) -> Path:
    """Generate diverse prompts from the taxonomy using Claude."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "prompts.jsonl"

    taxonomy = load_taxonomy()
    client = anthropic.Anthropic()
    all_prompts = []

    # Load golden prompt topics for holdout exclusion
    golden_topics = load_golden_topics()
    if golden_topics:
        topic_list = ", ".join(f'"{t}"' for t in golden_topics)
        holdout_clause = f"\nIMPORTANT: Do NOT generate prompts about these specific topics (they are reserved for held-out evaluation): {topic_list}"
        console.print(f"  Excluding {len(golden_topics)} golden prompt topics from generation")
    else:
        holdout_clause = ""

    with Progress() as progress:
        task = progress.add_task(
            "Generating prompts...",
            total=len(taxonomy["scenario_categories"]),
        )

        for category in taxonomy["scenario_categories"]:
            examples = "\n".join(f"- {ex}" for ex in category["example_prompts"])
            generation_prompt = PROMPT_GENERATION_TEMPLATE.format(
                category_name=category["name"],
                category_description=category["description"],
                sycophancy_pressure=category["sycophancy_pressure"],
                examples=examples,
                count=num_per_category,
                holdout_clause=holdout_clause,
            )

            message = client.messages.create(
                model=teacher_model,
                max_tokens=4096,
                temperature=0.9,
                messages=[{"role": "user", "content": generation_prompt}],
            )

            response_text = message.content[0].text
            prompts = _parse_numbered_list(response_text)

            for prompt_text in prompts:
                all_prompts.append(
                    {
                        "category": category["id"],
                        "category_name": category["name"],
                        "prompt": prompt_text.strip(),
                    }
                )

            console.print(
                f"  [green]{category['name']}[/green]: {len(prompts)} prompts generated"
            )
            progress.advance(task)

    with open(output_path, "w") as f:
        for p in all_prompts:
            f.write(json.dumps(p) + "\n")

    console.print(f"\n[bold green]Generated {len(all_prompts)} prompts → {output_path}")

    # Record step in manifest
    _run_id = run_id or os.environ.get("PARRHESIA_RUN_ID")
    if _run_id:
        from parrhesia.manifest import record_step

        lengths = [len(p["prompt"]) for p in all_prompts]
        record_step(
            _run_id,
            step_name="prompts",
            command=f"python -m parrhesia.data.generate_oct prompts --num-per-category {num_per_category} --model {teacher_model}",
            hyperparameters={"num_per_category": num_per_category, "teacher_model": teacher_model, "temperature": 0.9},
            inputs={"taxonomy": str(TAXONOMY_PATH)},
            outputs={"prompts": str(output_path)},
            metrics={"count": len(all_prompts), "min_length_chars": min(lengths) if lengths else 0,
                     "max_length_chars": max(lengths) if lengths else 0,
                     "avg_length_chars": round(sum(lengths) / len(lengths)) if lengths else 0},
        )

    return output_path


def _parse_numbered_list(text: str) -> list[str]:
    """Parse a numbered list from model output."""
    lines = text.strip().split("\n")
    prompts = []
    current = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check if this line starts a new numbered item
        is_new_item = False
        for i, ch in enumerate(stripped):
            if ch.isdigit():
                continue
            if ch in ".)" and i > 0:
                is_new_item = True
                stripped = stripped[i + 1 :].strip()
                break
            break

        if is_new_item:
            if current:
                prompts.append(" ".join(current))
            current = [stripped]
        else:
            current.append(stripped)

    if current:
        prompts.append(" ".join(current))

    return prompts


# ---------------------------------------------------------------------------
# Step 2: Generate chosen responses (teacher = Claude + constitution)
# ---------------------------------------------------------------------------


def generate_chosen(
    prompts_path: str | Path = DEFAULT_OUTPUT_DIR / "prompts.jsonl",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    batch: bool = True,
    run_id: str | None = None,
    push: bool = False,
    constitution_path: str | Path | None = None,
) -> Path:
    """Generate 'chosen' responses using Claude with the parrhesiastes constitution."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "chosen.jsonl"

    constitution = load_constitution(constitution_path)
    system_prompt = build_system_prompt(constitution)

    prompts = _load_jsonl(prompts_path)
    client = anthropic.Anthropic()

    if batch:
        _generate_chosen_batch(client, prompts, system_prompt, teacher_model, output_path)
    else:
        _generate_chosen_sequential(
            client, prompts, system_prompt, teacher_model, output_path
        )

    # Record step in manifest
    _run_id = run_id or os.environ.get("PARRHESIA_RUN_ID")
    if _run_id:
        from parrhesia.manifest import record_step

        batch_flag = "--no-batch" if not batch else ""
        record_step(
            _run_id,
            step_name="chosen",
            command=f"python -m parrhesia.data.generate_oct chosen --model {teacher_model} {batch_flag}".strip(),
            hyperparameters={"teacher_model": teacher_model, "temperature": 0.7, "max_tokens": 2048, "batch": batch},
            inputs={"prompts": str(prompts_path), "constitution": str(CONSTITUTION_PATH)},
            outputs={"chosen": str(output_path)},
        )

    return output_path


def _generate_chosen_batch(
    client: anthropic.Anthropic,
    prompts: list[dict],
    system_prompt: str,
    model: str,
    output_path: Path,
) -> None:
    """Use Anthropic Batch API for 50% cost savings."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    console.print("[bold]Creating batch request for chosen responses...[/bold]")

    requests = [
        Request(
            custom_id=f"chosen-{i}",
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=2048,
                temperature=0.7,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": p["prompt"]}],
            ),
        )
        for i, p in enumerate(prompts)
    ]

    # Batch API limit: 100k requests or 256MB per batch
    batch_size = min(len(requests), 10000)
    all_results = {}

    for batch_start in range(0, len(requests), batch_size):
        batch_requests = requests[batch_start : batch_start + batch_size]

        message_batch = client.messages.batches.create(requests=batch_requests)
        console.print(
            f"  Batch {message_batch.id} created with {len(batch_requests)} requests. Waiting..."
        )

        # Poll for completion
        while True:
            batch_status = client.messages.batches.retrieve(message_batch.id)
            if batch_status.processing_status == "ended":
                break
            console.print(
                f"  Status: {batch_status.processing_status} "
                f"(succeeded: {batch_status.request_counts.succeeded}, "
                f"processing: {batch_status.request_counts.processing})"
            )
            time.sleep(30)

        # Collect results
        for result in client.messages.batches.results(message_batch.id):
            if result.result.type == "succeeded":
                idx = int(result.custom_id.split("-")[1])
                all_results[idx] = result.result.message.content[0].text

    # Write results aligned with prompts
    with open(output_path, "w") as f:
        for i, p in enumerate(prompts):
            entry = {
                "index": i,
                "category": p["category"],
                "prompt": p["prompt"],
                "chosen_response": all_results.get(i, ""),
            }
            f.write(json.dumps(entry) + "\n")

    successful = sum(1 for v in all_results.values() if v)
    console.print(
        f"\n[bold green]Generated {successful}/{len(prompts)} chosen responses → {output_path}"
    )


def _generate_chosen_sequential(
    client: anthropic.Anthropic,
    prompts: list[dict],
    system_prompt: str,
    model: str,
    output_path: Path,
) -> None:
    """Sequential generation (for debugging or small batches)."""
    results = []

    with Progress() as progress:
        task = progress.add_task("Generating chosen responses...", total=len(prompts))

        for i, p in enumerate(prompts):
            try:
                message = client.messages.create(
                    model=model,
                    max_tokens=2048,
                    temperature=0.7,
                    system=[
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": p["prompt"]}],
                )
                response_text = message.content[0].text
            except Exception as e:
                console.print(f"  [red]Error on prompt {i}: {e}[/red]")
                response_text = ""

            results.append(
                {
                    "index": i,
                    "category": p["category"],
                    "prompt": p["prompt"],
                    "chosen_response": response_text,
                }
            )
            progress.advance(task)

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    successful = sum(1 for r in results if r["chosen_response"])
    console.print(
        f"\n[bold green]Generated {successful}/{len(prompts)} chosen responses → {output_path}"
    )


# ---------------------------------------------------------------------------
# Step 3: Generate rejected responses (student = base model, no constitution)
# ---------------------------------------------------------------------------


def generate_rejected(
    prompts_path: str | Path = DEFAULT_OUTPUT_DIR / "prompts.jsonl",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    student_model: str = DEFAULT_STUDENT_MODEL,
    api_base: str | None = None,
    api_key: str | None = None,
    run_id: str | None = None,
    push: bool = False,
    concurrency: int = 1,
) -> Path:
    """
    Generate 'rejected' responses using the base model WITHOUT constitution.

    Supports any OpenAI-compatible API endpoint:
    - Local vLLM: --api-base http://localhost:8000/v1
    - Together.ai: --api-base https://api.together.xyz/v1 --api-key $TOGETHER_API_KEY
    - RunPod serverless: --api-base https://api.runpod.ai/v2/.../openai/v1
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rejected.jsonl"

    prompts = _load_jsonl(prompts_path)

    api_base = api_base or os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")

    console.print(f"  Concurrency: {concurrency}")

    if concurrency > 1:
        results = asyncio.run(
            _generate_rejected_async(prompts, student_model, api_base, api_key, concurrency)
        )
    else:
        results = _generate_rejected_sync(prompts, student_model, api_base, api_key)

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    successful = sum(1 for r in results if r["rejected_response"])
    console.print(
        f"\n[bold green]Generated {successful}/{len(prompts)} rejected responses → {output_path}"
    )

    # Record step in manifest
    _run_id = run_id or os.environ.get("PARRHESIA_RUN_ID")
    if _run_id:
        from parrhesia.manifest import record_step

        api_base_display = api_base or os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
        record_step(
            _run_id,
            step_name="rejected",
            command=f"python -m parrhesia.data.generate_oct rejected --model {student_model} --api-base {api_base_display}",
            hyperparameters={"student_model": student_model, "temperature": 0.7, "max_tokens": 2048, "system_prompt": "none"},
            inputs={"prompts": str(prompts_path)},
            outputs={"rejected": str(output_path)},
            metrics={"count": successful},
        )

    return output_path


def _generate_rejected_sync(
    prompts: list[dict], model: str, api_base: str, api_key: str,
) -> list[dict]:
    """Generate rejected responses synchronously."""
    from openai import OpenAI

    client = OpenAI(base_url=api_base, api_key=api_key)
    results = []

    with Progress() as progress:
        task = progress.add_task("Generating rejected responses...", total=len(prompts))

        for i, p in enumerate(prompts):
            try:
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=2048,
                    temperature=0.7,
                    messages=[{"role": "user", "content": p["prompt"]}],
                )
                response_text = response.choices[0].message.content
            except Exception as e:
                console.print(f"  [red]Error on prompt {i}: {e}[/red]")
                response_text = ""

            results.append(
                {
                    "index": i,
                    "category": p["category"],
                    "prompt": p["prompt"],
                    "rejected_response": response_text,
                }
            )
            progress.advance(task)

    return results


async def _generate_rejected_async(
    prompts: list[dict], model: str, api_base: str, api_key: str, concurrency: int,
) -> list[dict]:
    """Generate rejected responses with async concurrency."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=api_base, api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    results = [None] * len(prompts)

    async def process(idx: int, p: dict):
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    max_tokens=2048,
                    temperature=0.7,
                    messages=[{"role": "user", "content": p["prompt"]}],
                )
                response_text = response.choices[0].message.content
            except Exception as e:
                console.print(f"  [red]Error on prompt {idx}: {e}[/red]")
                response_text = ""

            results[idx] = {
                "index": idx,
                "category": p["category"],
                "prompt": p["prompt"],
                "rejected_response": response_text,
            }

    with Progress() as progress:
        task = progress.add_task("Generating rejected responses...", total=len(prompts))

        async def process_with_progress(idx: int, p: dict):
            await process(idx, p)
            progress.advance(task)

        await asyncio.gather(
            *[process_with_progress(i, p) for i, p in enumerate(prompts)]
        )

    return results


# ---------------------------------------------------------------------------
# Step 4: Combine into DPO training format
# ---------------------------------------------------------------------------


def combine_dpo_pairs(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    run_id: str | None = None,
    push: bool = False,
) -> Path:
    """Combine chosen and rejected responses into DPO training format."""
    output_dir = Path(output_dir)
    chosen_path = output_dir / "chosen.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    output_path = output_dir / "dpo_pairs.jsonl"

    chosen_data = {r["index"]: r for r in _load_jsonl(chosen_path)}
    rejected_data = {r["index"]: r for r in _load_jsonl(rejected_path)}

    pairs = []
    skipped = 0

    for idx in sorted(chosen_data.keys()):
        if idx not in rejected_data:
            skipped += 1
            continue

        chosen = chosen_data[idx]
        rejected = rejected_data[idx]

        if not chosen["chosen_response"] or not rejected["rejected_response"]:
            skipped += 1
            continue

        # DPO conversational format (compatible with TRL DPOTrainer)
        pair = {
            "chosen": [
                {"role": "user", "content": chosen["prompt"]},
                {"role": "assistant", "content": chosen["chosen_response"]},
            ],
            "rejected": [
                {"role": "user", "content": rejected["prompt"]},
                {"role": "assistant", "content": rejected["rejected_response"]},
            ],
        }
        pairs.append(pair)

    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    console.print(
        f"\n[bold green]Combined {len(pairs)} DPO pairs (skipped {skipped}) → {output_path}"
    )

    # Record step in manifest
    _run_id = run_id or os.environ.get("PARRHESIA_RUN_ID")
    if _run_id:
        from parrhesia.manifest import record_step

        record_step(
            _run_id,
            step_name="combine",
            command=f"python -m parrhesia.data.generate_oct combine --output-dir {output_dir}",
            inputs={"chosen": str(chosen_path), "rejected": str(rejected_path)},
            outputs={"dpo_pairs": str(output_path)},
            metrics={"pairs": len(pairs), "skipped": skipped},
        )

    # Push dataset to Hub (auto-push when run is active)
    if _run_id:
        from parrhesia.hub import push_dataset, DATASET_REPOS

        console.print(f"Pushing DPO pairs to Hub...")
        url = push_dataset(output_path, DATASET_REPOS["oct"], _run_id, "dpo_pairs")
        console.print(f"  → {url}")

    return output_path


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def generate_oct_data(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    num_scenarios: int = 50,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    constitution_path: str | Path | None = None,
) -> None:
    """Run the full OCT data generation pipeline (prompts + chosen only).

    Rejected responses require a separate model endpoint.
    Run generate_rejected() separately with appropriate API config.
    """
    output_dir = Path(output_dir)

    console.print("[bold]Step 1: Generating prompts from taxonomy...[/bold]")
    prompts_path = generate_prompts(
        num_per_category=num_scenarios,
        output_dir=output_dir,
        teacher_model=teacher_model,
    )

    console.print("\n[bold]Step 2: Generating chosen responses (Claude + constitution)...[/bold]")
    generate_chosen(
        prompts_path=prompts_path,
        output_dir=output_dir,
        teacher_model=teacher_model,
        batch=True,
        constitution_path=constitution_path,
    )

    console.print(
        "\n[bold yellow]Step 3: Generate rejected responses separately.[/bold yellow]"
    )
    console.print(
        "  Run with a Qwen3 endpoint:\n"
        "  python -m parrhesia.data.generate_oct rejected \\\n"
        "    --api-base http://localhost:8000/v1  # local vLLM\n"
        "    # or: --api-base https://api.together.xyz/v1 --api-key $TOGETHER_API_KEY"
    )
    console.print(
        "\n  Then combine:\n"
        "  python -m parrhesia.data.generate_oct combine"
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="OCT data generation pipeline")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID for manifest tracking")
    parser.add_argument("--push", action="store_true", help="Push outputs to HuggingFace Hub")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prompts
    p_prompts = subparsers.add_parser("prompts", help="Generate prompts from taxonomy")
    p_prompts.add_argument("--num-per-category", type=int, default=50)
    p_prompts.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p_prompts.add_argument("--model", type=str, default=DEFAULT_TEACHER_MODEL)

    # chosen
    p_chosen = subparsers.add_parser("chosen", help="Generate chosen responses (Claude + constitution)")
    p_chosen.add_argument("--prompts", type=str, default=str(DEFAULT_OUTPUT_DIR / "prompts.jsonl"))
    p_chosen.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p_chosen.add_argument("--model", type=str, default=DEFAULT_TEACHER_MODEL)
    p_chosen.add_argument("--no-batch", action="store_true", help="Use sequential instead of batch API")
    p_chosen.add_argument("--constitution", type=str, default=None, help="Path to constitution file (default: parrhesiastes.md)")

    # rejected
    p_rejected = subparsers.add_parser("rejected", help="Generate rejected responses (base model)")
    p_rejected.add_argument("--prompts", type=str, default=str(DEFAULT_OUTPUT_DIR / "prompts.jsonl"))
    p_rejected.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p_rejected.add_argument("--model", type=str, default=DEFAULT_STUDENT_MODEL)
    p_rejected.add_argument("--api-base", type=str, default=None)
    p_rejected.add_argument("--api-key", type=str, default=None)
    p_rejected.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests")

    # combine
    p_combine = subparsers.add_parser("combine", help="Combine chosen+rejected into DPO pairs")
    p_combine.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))

    # full
    p_full = subparsers.add_parser("full", help="Run full pipeline (prompts + chosen)")
    p_full.add_argument("--num-per-category", type=int, default=50)
    p_full.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p_full.add_argument("--model", type=str, default=DEFAULT_TEACHER_MODEL)
    p_full.add_argument("--constitution", type=str, default=None, help="Path to constitution file (default: parrhesiastes.md)")

    args = parser.parse_args()

    if args.command == "prompts":
        generate_prompts(args.num_per_category, args.output_dir, args.model, run_id=args.run_id, push=args.push)
    elif args.command == "chosen":
        generate_chosen(args.prompts, args.output_dir, args.model, batch=not args.no_batch, run_id=args.run_id, push=args.push, constitution_path=args.constitution)
    elif args.command == "rejected":
        generate_rejected(args.prompts, args.output_dir, args.model, args.api_base, args.api_key, run_id=args.run_id, push=args.push, concurrency=args.concurrency)
    elif args.command == "combine":
        combine_dpo_pairs(args.output_dir, run_id=args.run_id, push=args.push)
    elif args.command == "full":
        generate_oct_data(args.output_dir, args.num_per_category, args.model, constitution_path=args.constitution)


if __name__ == "__main__":
    main()
