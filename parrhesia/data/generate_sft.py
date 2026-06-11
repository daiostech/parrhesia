"""
SFT data generation pipeline for Approach B (Direct SFT) and Approach C Phase 1.

Generates conversation pairs where the model demonstrates parrhesia (truth-telling)
in response to prompts that create pressure toward sycophancy. Uses the Aristotelian
taxonomy to ensure coverage of all kolax/areskos patterns.

Usage:
    # Old behavior (200 total examples, ~20 per category):
    python -m parrhesia.data.generate_sft --num-per-category 20

    # Approach B (2,000+ examples with quality filter and failure-mode targeting):
    python -m parrhesia.data.generate_sft \
        --num-per-category 200 \
        --output-dir data/generated/sft \
        --filter --failure-modes
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress

load_dotenv()
console = Console()

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = ROOT / "parrhesia" / "taxonomy" / "taxonomy.json"
CONSTITUTION_PATH = ROOT / "parrhesia" / "taxonomy" / "constitutions" / "parrhesiastes.md"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "generated" / "sft"

DEFAULT_TEACHER_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-5-20250929"

MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds, doubled each retry


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SFT_GENERATION_PROMPT = """\
You are generating training data for teaching an AI model the virtue of parrhesia \
(frank, truth-telling speech). The model should learn to respond honestly and \
constructively even when the user creates social pressure to agree or flatter.

Your responses should embody these traits:
{constitution_excerpt}

Category: {category_name}
Description: {category_description}

{variation_instruction}

Generate {count} realistic conversation examples. Each should include:
1. A user message that creates pressure toward sycophancy
2. An ideal assistant response demonstrating parrhesia (frank, honest, respectful, constructive)

{turn_instruction}

Rules for the assistant responses:
- Be DIRECT. State your actual position clearly in the first sentence.
- Be SPECIFIC. Don't use vague hedging like "there are many perspectives."
- Be CONSTRUCTIVE. When disagreeing, explain why and suggest alternatives.
- Be RESPECTFUL. Frank does not mean rude. You can be honest and kind simultaneously.
- Do NOT abandon correct positions under social pressure.
- Do NOT use excessive qualifiers, disclaimers, or "on the other hand" equivocations.
- Do NOT open with flattery ("Great question!", "That's a really interesting point!").

Output as a JSON array where each element has a "messages" field containing the conversation:
```json
[
  {{
    "messages": [
      {{"role": "user", "content": "User message here"}},
      {{"role": "assistant", "content": "Parrhesia response here"}}
    ]
  }},
  {{
    "messages": [
      {{"role": "user", "content": "User message"}},
      {{"role": "assistant", "content": "First response"}},
      {{"role": "user", "content": "Pushback"}},
      {{"role": "assistant", "content": "Maintains position"}}
    ]
  }}
]
```

Return ONLY the JSON array."""


FAILURE_MODE_PROMPT = """\
You are generating targeted training data for teaching an AI model the virtue of \
parrhesia (frank, truth-telling speech). These examples specifically address a \
common failure pattern where current models fall short.

Your responses should embody these traits:
{constitution_excerpt}

FAILURE MODE TO ADDRESS: {failure_mode_name}
Description: {failure_mode_description}

Generate {count} realistic conversation examples that specifically demonstrate \
the CORRECT behavior for this failure mode. Each should include:
1. A user message that would trigger the failure mode in a typical model
2. An ideal assistant response that handles it correctly — with parrhesia

{turn_instruction}

Rules for the assistant responses:
- Be DIRECT. State your actual position clearly in the first sentence.
- Be SPECIFIC. Don't use vague hedging like "there are many perspectives."
- Be CONSTRUCTIVE. When disagreeing, explain why and suggest alternatives.
- Be RESPECTFUL. Frank does not mean rude. You can be honest and kind simultaneously.
- Do NOT abandon correct positions under social pressure.
- Do NOT use excessive qualifiers, disclaimers, or "on the other hand" equivocations.
- Do NOT open with flattery ("Great question!", "That's a really interesting point!").

Output as a JSON array where each element has a "messages" field containing the conversation:
```json
[
  {{
    "messages": [
      {{"role": "user", "content": "User message here"}},
      {{"role": "assistant", "content": "Parrhesia response here"}}
    ]
  }}
]
```

Return ONLY the JSON array."""


QUALITY_FILTER_PROMPT = """\
You are a quality judge for parrhesia (frank, truth-telling) training data.

Evaluate this conversation example. The assistant should demonstrate genuine \
parrhesia — honest, direct, constructive speech that resists sycophancy pressure.

Conversation:
{conversation}

Score this example as PASS or FAIL based on these criteria:
1. Does the assistant state its actual position directly (not buried in hedges)?
2. Does the assistant maintain its position if the user pushes back (in multi-turn)?
3. Is the response constructive and specific (not vague or evasive)?
4. Does the assistant avoid sycophantic patterns (no opening flattery, no "great question!", no false validation)?
5. Is the conversation realistic and the sycophancy pressure believable?

If ALL criteria are met, respond with exactly: PASS
If ANY criterion fails, respond with exactly: FAIL

Respond with ONLY "PASS" or "FAIL", nothing else."""


# ---------------------------------------------------------------------------
# Variation instructions for batch diversity
# ---------------------------------------------------------------------------

VARIATION_INSTRUCTIONS = [
    "Focus on EVERYDAY scenarios — casual conversations, personal decisions, hobby discussions.",
    "Focus on PROFESSIONAL scenarios — workplace dynamics, career advice, business decisions.",
    "Focus on TECHNICAL scenarios — code review, scientific claims, engineering tradeoffs.",
    "Focus on HEALTH AND WELLNESS scenarios — medical claims, diet fads, fitness advice, mental health.",
    "Focus on EDUCATION scenarios — study strategies, career paths, academic debates.",
    "Focus on CREATIVE scenarios — writing feedback, art critique, creative direction.",
    "Focus on RELATIONSHIP scenarios — interpersonal conflicts, parenting, social dynamics.",
    "Focus on FINANCIAL scenarios — investment decisions, budgeting, financial planning.",
    "Focus on HIGH-STAKES scenarios — important life decisions, safety-critical situations.",
    "Focus on SUBTLE scenarios — where the sycophancy pressure is implicit rather than explicit.",
]


# ---------------------------------------------------------------------------
# Run 4 failure modes for targeted generation
# ---------------------------------------------------------------------------

FAILURE_MODES = [
    {
        "name": "Watchful-waiting conflation",
        "description": (
            "The model conflates 'observation/monitoring' with 'treatment validation'. "
            "When a user says they're 'just watching and waiting' on a medical issue or "
            "'keeping an eye on' something that needs professional attention, the model "
            "validates the inaction as a reasonable approach instead of flagging that "
            "monitoring alone is not a substitute for professional evaluation. The correct "
            "response distinguishes between informed watchful waiting (doctor-supervised) "
            "and avoidance disguised as monitoring."
        ),
    },
    {
        "name": "Partial-truth concession",
        "description": (
            "Under pressure, the model concedes a 'kernel of truth' in a wrong position — "
            "'You're right that X has SOME merit' — then fails to follow through on why "
            "the overall position is still wrong. This partial validation feels like "
            "agreement to the user. The correct response: if the overall position is wrong, "
            "say so directly. If there genuinely is a kernel of truth, name it precisely "
            "and then clearly explain why the overall conclusion doesn't follow."
        ),
    },
    {
        "name": "Medical concern vs empathetic validation",
        "description": (
            "When a user describes using an unproven remedy (essential oils for infections, "
            "crystals for pain, supplements instead of medication), the model prioritizes "
            "empathetic validation ('I understand why you'd want a natural approach') over "
            "medical candor. The correct response leads with the medical concern: 'This "
            "condition needs proven treatment. Here's why relying on [remedy] alone is risky.' "
            "Empathy can follow, but it must not precede or dilute the medical message."
        ),
    },
    {
        "name": "False-equivalence validation",
        "description": (
            "When a user presents two options as equivalent and asks for validation, the "
            "model treats them as equally valid even when one is clearly better or one is "
            "wrong. Under persistent pushback, the model retreats to 'both perspectives "
            "have merit' rather than maintaining that the options are NOT equivalent. The "
            "correct response: name the asymmetry directly. 'These aren't equivalent — "
            "here's why option A is clearly better/why option B has serious problems.'"
        ),
    },
    {
        "name": "Validate-then-correct framing",
        "description": (
            "The model opens with validation before delivering the correction: 'That's an "
            "understandable perspective, but...' or 'I can see why you'd think that, "
            "however...' This framing softens the correction so much that the user hears "
            "the validation and tunes out the 'but'. The correct response: lead with the "
            "correction or the honest assessment. 'Actually, that's not quite right — ...' "
            "or 'No, the evidence shows...' Directness is not rudeness."
        ),
    },
]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _prose_declaration_lines(lines: list[str]):
    """Yield declaration prose from a v0.2.0-style constitution.

    Keeps section headings and first-person declaration sentences; drops the
    scholarly apparatus (inline Stephanus citations, the metadata header, the
    citation-summary and version-history sections) that is not a behavioral
    instruction for the teacher.
    """
    skip_rest = False
    for line in lines:
        s = line.strip()
        # Trailing reference sections run to end of file once encountered.
        if s.startswith("**Stephanus") or s.startswith("**Version History"):
            skip_rest = True
        if skip_rest or not s or s == "---":
            continue
        # Inline citations, e.g. "*(Reference: NE IV.7, 1127a13-28 ...)*".
        if s.startswith("*(") or s.startswith("(Reference"):
            continue
        # Metadata header labels near the top of the document.
        if s.startswith(("**Version", "**Date", "**Philosophical")):
            continue
        # Raw citation bullets in the philosophical-foundation list.
        if s.startswith("- ") and "NE " in s:
            continue
        yield line


def _get_constitution_excerpt(path: str | Path | None = None) -> str:
    """Load and condense a constitution into a teacher-prompt excerpt.

    Format-aware: a v0.1.0-style file (numbered first-person declarations) is
    condensed to those numbered lines; a v0.2.0-style prose file keeps its
    declarations and headings minus the citation apparatus. Raises if the
    excerpt comes back empty rather than silently embedding a blank instruction.
    """
    path = Path(path) if path else CONSTITUTION_PATH
    lines = path.read_text().split("\n")

    numbered = [line for line in lines if line.strip().startswith(tuple("0123456789"))]
    if numbered:
        excerpt = "\n".join(numbered).strip()
    else:
        excerpt = "\n".join(_prose_declaration_lines(lines)).strip()

    if not excerpt:
        raise ValueError(
            f"Constitution excerpt is empty for {path}. Expected numbered "
            "declarations (v0.1.0) or declaration prose (v0.2.0); check the file."
        )
    return excerpt


def _get_turn_instruction(is_multi_turn: bool) -> str:
    """Return the instruction about single vs multi-turn conversations."""
    if is_multi_turn:
        return (
            "Generate MULTI-TURN conversations (2-3 turns). In each example, the user "
            "should push back after the assistant's first response, and the assistant "
            "should maintain its honest position with NEW reasoning (not just repeating)."
        )
    return (
        "Generate a mix of single-turn and multi-turn conversations. "
        "For some examples, include multi-turn conversations (2-3 turns) where the user "
        "pushes back and the assistant maintains its honest position while remaining respectful."
    )


def _parse_json_array(text: str) -> list[dict]:
    """Parse a JSON array from model output."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        return [result]
    except json.JSONDecodeError:
        import re

        # Non-greedy match to avoid pathological backtracking
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        # Try greedy as last resort, but with a size cap
        if len(text) < 100_000:
            match = re.search(r"\[.+\]", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        console.print("[yellow]Warning: Could not parse SFT JSON[/yellow]")
        return []


def _validate_pairs(pairs: list[dict]) -> list[dict]:
    """Validate that pairs have proper message structure."""
    validated = []
    for pair in pairs:
        messages = pair.get("messages", [])
        if (
            len(messages) >= 2
            and messages[0].get("role") == "user"
            and any(m.get("role") == "assistant" for m in messages)
        ):
            validated.append(pair)
    return validated


def _api_call_with_retry(client: anthropic.Anthropic, **kwargs) -> str:
    """Make an API call with retry and backoff. Returns response text."""
    for attempt in range(MAX_RETRIES):
        try:
            message = client.messages.create(**kwargs)
            text = message.content[0].text
            # Drop reference to the full Message object immediately
            del message
            return text
        except (anthropic.APIError, anthropic.APIConnectionError) as e:
            wait = RETRY_BACKOFF * (2 ** attempt)
            console.print(f"[yellow]API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {wait}s...[/yellow]")
            time.sleep(wait)
    console.print("[red]API call failed after all retries, skipping batch[/red]")
    return ""


def _append_jsonl(path: Path, pairs: list[dict]) -> None:
    """Append pairs to a JSONL file incrementally."""
    with open(path, "a") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def _generate_batch(
    client: anthropic.Anthropic,
    teacher_model: str,
    constitution_excerpt: str,
    category_name: str,
    category_description: str,
    count: int,
    variation_index: int,
    is_multi_turn: bool,
) -> list[dict]:
    """Generate a single batch of SFT examples for a category."""
    variation_instruction = VARIATION_INSTRUCTIONS[variation_index % len(VARIATION_INSTRUCTIONS)]
    turn_instruction = _get_turn_instruction(is_multi_turn)

    prompt = SFT_GENERATION_PROMPT.format(
        constitution_excerpt=constitution_excerpt,
        category_name=category_name,
        category_description=category_description,
        count=count,
        variation_instruction=variation_instruction,
        turn_instruction=turn_instruction,
    )

    response_text = _api_call_with_retry(
        client,
        model=teacher_model,
        max_tokens=8192,
        temperature=0.8,
        messages=[{"role": "user", "content": prompt}],
    )
    if not response_text:
        return []

    pairs = _parse_json_array(response_text)
    del response_text
    return _validate_pairs(pairs)


def _generate_failure_mode_batch(
    client: anthropic.Anthropic,
    teacher_model: str,
    constitution_excerpt: str,
    failure_mode: dict,
    count: int,
    is_multi_turn: bool,
) -> list[dict]:
    """Generate targeted examples for a specific failure mode."""
    turn_instruction = _get_turn_instruction(is_multi_turn)

    prompt = FAILURE_MODE_PROMPT.format(
        constitution_excerpt=constitution_excerpt,
        failure_mode_name=failure_mode["name"],
        failure_mode_description=failure_mode["description"],
        count=count,
        turn_instruction=turn_instruction,
    )

    response_text = _api_call_with_retry(
        client,
        model=teacher_model,
        max_tokens=8192,
        temperature=0.8,
        messages=[{"role": "user", "content": prompt}],
    )
    if not response_text:
        return []

    pairs = _parse_json_array(response_text)
    del response_text
    return _validate_pairs(pairs)


def _format_conversation(pair: dict) -> str:
    """Format a conversation pair as readable text for the judge."""
    lines = []
    for msg in pair.get("messages", []):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def _filter_with_judge(
    pairs: list[dict],
    input_path: Path,
    filtered_path: Path,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> int:
    """Filter examples using Claude judge. Reads from input, writes passing to filtered_path.

    Creates a fresh client for the filter phase. Returns count of passing examples.
    """
    client = anthropic.Anthropic()
    passed_count = 0
    failed_count = 0

    # Clear filtered file
    filtered_path.write_text("")

    with Progress() as progress:
        task = progress.add_task("Quality filtering...", total=len(pairs))

        for pair in pairs:
            conversation_text = _format_conversation(pair)
            prompt = QUALITY_FILTER_PROMPT.format(conversation=conversation_text)

            verdict_text = _api_call_with_retry(
                client,
                model=judge_model,
                max_tokens=16,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )

            if verdict_text.strip().upper() == "PASS":
                _append_jsonl(filtered_path, [pair])
                passed_count += 1
            else:
                failed_count += 1

            progress.advance(task)

            # Periodic GC every 200 judge calls
            if (passed_count + failed_count) % 200 == 0:
                gc.collect()

    pass_rate = passed_count / len(pairs) * 100 if pairs else 0
    console.print(
        f"\n[bold]Quality filter: {passed_count}/{len(pairs)} passed ({pass_rate:.1f}%), "
        f"{failed_count} rejected[/bold]"
    )
    return passed_count


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_sft_data(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    num_per_category: int = 20,
    batch_size: int = 20,
    multi_turn_ratio: float = 0.4,
    failure_modes: bool = False,
    filter_quality: bool = False,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    constitution_path: str | Path | None = None,
    run_id: str | None = None,
) -> Path:
    """Generate SFT training data across all taxonomy categories.

    Writes pairs to disk incrementally after each batch — safe to interrupt.

    Args:
        output_dir: Where to save output JSONL files.
        num_per_category: Number of examples to generate per taxonomy category.
        batch_size: Examples per API call. If num_per_category <= batch_size,
            one call per category (old behavior).
        multi_turn_ratio: Fraction of batches that request multi-turn conversations.
        failure_modes: Generate 50 additional examples per failure mode (250 total).
        filter_quality: Post-generation quality filter using Claude judge.
        teacher_model: Model to use for generation.
        constitution_path: Constitution file the teacher embodies. Defaults to
            parrhesiastes.md (v0.1.0), the constitution the shipped adapter
            descends from; pass parrhesiastes_v0.2.0.md or a new virtue's
            constitution to generate data for a different character.
        run_id: Run to record this step against (falls back to PARRHESIA_RUN_ID).

    Returns:
        Path to the output JSONL file (filtered if --filter, otherwise unfiltered).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_pairs.jsonl"

    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)

    constitution_file = Path(constitution_path) if constitution_path else CONSTITUTION_PATH
    constitution_excerpt = _get_constitution_excerpt(constitution_file)
    categories = taxonomy["scenario_categories"]

    # Calculate batching
    num_batches = max(1, math.ceil(num_per_category / batch_size))
    per_batch = min(batch_size, num_per_category)
    total_api_calls = num_batches * len(categories)
    if failure_modes:
        fm_batches_per_mode = max(1, math.ceil(50 / 25))
        total_api_calls += fm_batches_per_mode * len(FAILURE_MODES)

    console.print(f"[bold]SFT Data Generation[/bold]")
    console.print(f"  Constitution: {constitution_file.name}")
    console.print(f"  Categories: {len(categories)}")
    console.print(f"  Per category: {num_per_category} ({num_batches} batch(es) × {per_batch})")
    console.print(f"  Multi-turn ratio: {multi_turn_ratio:.0%}")
    console.print(f"  Failure modes: {'yes (250 extra)' if failure_modes else 'no'}")
    console.print(f"  Quality filter: {'yes' if filter_quality else 'no'}")
    console.print(f"  Total API calls: ~{total_api_calls}")
    console.print()

    # Clear output file (fresh run)
    output_path.write_text("")
    total_pairs = 0

    # --- Phase 1: Category-based generation ---
    # Fresh client per phase to avoid connection pool growth
    client = anthropic.Anthropic()

    with Progress() as progress:
        task = progress.add_task(
            "Generating category examples...",
            total=num_batches * len(categories),
        )

        global_batch_idx = 0
        for category in categories:
            cat_count = 0
            for batch_idx in range(num_batches):
                is_multi_turn = (batch_idx % max(1, round(1 / multi_turn_ratio))) == 1

                pairs = _generate_batch(
                    client=client,
                    teacher_model=teacher_model,
                    constitution_excerpt=constitution_excerpt,
                    category_name=category["name"],
                    category_description=category["description"],
                    count=per_batch,
                    variation_index=global_batch_idx,
                    is_multi_turn=is_multi_turn,
                )

                # Write to disk immediately
                if pairs:
                    _append_jsonl(output_path, pairs)
                    cat_count += len(pairs)
                    total_pairs += len(pairs)

                del pairs
                global_batch_idx += 1
                progress.advance(task)

            console.print(
                f"  [green]{category['name']}[/green]: {cat_count} pairs generated"
            )

            # GC after each category to release httpx buffers
            gc.collect()

    # Drop the generation client before starting failure modes
    del client
    gc.collect()

    # --- Phase 2: Failure-mode targeted generation ---
    if failure_modes:
        console.print()
        client = anthropic.Anthropic()
        fm_total = 0

        with Progress() as progress:
            fm_batches_per_mode = max(1, math.ceil(50 / 25))
            task = progress.add_task(
                "Generating failure-mode examples...",
                total=fm_batches_per_mode * len(FAILURE_MODES),
            )

            for fm in FAILURE_MODES:
                fm_count = 0
                for batch_idx in range(fm_batches_per_mode):
                    count = min(25, 50 - fm_count)
                    is_multi_turn = batch_idx % 2 == 1

                    pairs = _generate_failure_mode_batch(
                        client=client,
                        teacher_model=teacher_model,
                        constitution_excerpt=constitution_excerpt,
                        failure_mode=fm,
                        count=count,
                        is_multi_turn=is_multi_turn,
                    )

                    if pairs:
                        _append_jsonl(output_path, pairs)
                        fm_count += len(pairs)
                        fm_total += len(pairs)
                        total_pairs += len(pairs)

                    del pairs
                    progress.advance(task)

                console.print(
                    f"  [cyan]{fm['name']}[/cyan]: {fm_count} pairs generated"
                )

            gc.collect()

        del client
        gc.collect()
        console.print(f"\n[bold]Failure-mode total: {fm_total} pairs[/bold]")

    console.print(f"\n[bold green]Generated {total_pairs} SFT pairs → {output_path}")

    # --- Phase 3: Quality filter ---
    if filter_quality:
        console.print()

        # Read back from disk so we don't hold everything in memory twice
        all_pairs = []
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_pairs.append(json.loads(line))

        # Shuffle before filtering (diversity in the filtered set)
        random.shuffle(all_pairs)

        # Rewrite shuffled unfiltered file
        with open(output_path, "w") as f:
            for pair in all_pairs:
                f.write(json.dumps(pair) + "\n")

        filtered_path = output_dir / "training_pairs_filtered.jsonl"
        passed = _filter_with_judge(all_pairs, output_path, filtered_path)
        del all_pairs
        gc.collect()

        console.print(f"[bold green]Filtered {passed} SFT pairs → {filtered_path}")
        final_path = filtered_path
    else:
        # Shuffle the unfiltered file in place
        all_pairs = []
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_pairs.append(json.loads(line))
        random.shuffle(all_pairs)
        with open(output_path, "w") as f:
            for pair in all_pairs:
                f.write(json.dumps(pair) + "\n")
        final_path = output_path

    # Record step in manifest. The resolved constitution path is recorded
    # explicitly so a replay reproduces the same character regardless of any
    # future change to the default.
    _run_id = run_id or os.environ.get("PARRHESIA_RUN_ID")
    if _run_id:
        from parrhesia.manifest import record_step

        cmd = (
            f"python -m parrhesia.data.generate_sft "
            f"--num-per-category {num_per_category} "
            f"--output-dir {output_dir} "
            f"--constitution {constitution_file}"
        )
        if failure_modes:
            cmd += " --failure-modes"
        if filter_quality:
            cmd += " --filter"

        record_step(
            _run_id,
            step_name="generate_sft",
            command=cmd,
            hyperparameters={
                "teacher_model": teacher_model,
                "num_per_category": num_per_category,
                "batch_size": batch_size,
                "multi_turn_ratio": multi_turn_ratio,
                "failure_modes": failure_modes,
                "filter": filter_quality,
                "temperature": 0.8,
            },
            inputs={"taxonomy": str(TAXONOMY_PATH), "constitution": str(constitution_file)},
            outputs={"training_pairs": str(final_path)},
        )

    return final_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate SFT training data")
    parser.add_argument("--num-per-category", type=int, default=20,
                        help="Examples per taxonomy category (default: 20)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", type=str, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Examples per API call (default: 20)")
    parser.add_argument("--multi-turn-ratio", type=float, default=0.4,
                        help="Fraction of batches requesting multi-turn (default: 0.4)")
    parser.add_argument("--failure-modes", action="store_true",
                        help="Generate 50 targeted examples per Run 4 failure mode (250 total)")
    parser.add_argument("--filter", action="store_true",
                        help="Post-generation quality filter using Claude judge")
    parser.add_argument("--constitution", type=str, default=None,
                        help="Constitution file the teacher embodies "
                             "(default: parrhesiastes.md, v0.1.0). Pass "
                             "parrhesiastes_v0.2.0.md or a new virtue's constitution.")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run to record this step against (default: $PARRHESIA_RUN_ID)")
    args = parser.parse_args()

    generate_sft_data(
        output_dir=args.output_dir,
        num_per_category=args.num_per_category,
        batch_size=args.batch_size,
        multi_turn_ratio=args.multi_turn_ratio,
        failure_modes=args.failure_modes,
        filter_quality=args.filter,
        teacher_model=args.model,
        constitution_path=args.constitution,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
