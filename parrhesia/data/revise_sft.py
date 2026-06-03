"""
Sweep SFT training data through a Claude judge, classify quality on a rubric,
and revise examples that fall below a threshold.

Reusable: swap --classify-prompt, --revise-prompt, and --examples for any
quality dimension (phronesis, specificity, tone, etc.).

Usage:
    # Dry run — classify only, inspect scores before revising
    python -m parrhesia.data.revise_sft \
        --input data/generated/sft/training_pairs_filtered.jsonl \
        --scores data/generated/sft/scores.jsonl \
        --dry-run

    # Full run — classify and revise in one pass
    python -m parrhesia.data.revise_sft \
        --input data/generated/sft/training_pairs_filtered.jsonl \
        --output data/generated/sft/training_pairs_revised.jsonl \
        --scores data/generated/sft/scores.jsonl \
        --threshold 1

    # Resume revision using existing scores (skip classification)
    python -m parrhesia.data.revise_sft \
        --input data/generated/sft/training_pairs_filtered.jsonl \
        --output data/generated/sft/training_pairs_revised.jsonl \
        --scores data/generated/sft/scores.jsonl \
        --threshold 2 \
        --skip-classify

    # Custom rubric and examples for a different quality dimension
    python -m parrhesia.data.revise_sft \
        --input data.jsonl \
        --output revised.jsonl \
        --classify-prompt "Rate specificity on 1-3..." \
        --revise-prompt "Make responses more specific..." \
        --examples my_examples.json
"""

from __future__ import annotations

import asyncio
import gc
import json
import re
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress

load_dotenv()
console = Console()

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_CONCURRENCY = 10
MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds, doubled each retry


# ---------------------------------------------------------------------------
# Default prompts (phronesis)
# ---------------------------------------------------------------------------

DEFAULT_CLASSIFY_PROMPT = """\
You are evaluating an AI assistant's response for delivery quality — specifically \
whether it demonstrates phronesis (practical wisdom about *how* to deliver frank truths).

Rate the assistant's response on a 1-3 scale:

3 = GOOD — Frank AND appropriately delivered. Truth is spoken with suitable \
compassion and timing. Directness is proportional to the situation.

2 = MINOR ISSUE — Correct content, but delivery could be slightly more sensitive \
or the opening is unnecessarily blunt given the emotional context. Would benefit \
from acknowledging the person's situation before correcting.

1 = NEEDS REVISION — Correct content but insensitive, harsh, or needlessly \
aggressive delivery. The response would make the person defensive rather than \
receptive. Lacks acknowledgment of emotional context.

Key considerations:
- Does the response acknowledge the person's emotional state or motivation \
before delivering hard truths (when the context calls for it)?
- Is the level of directness proportional to the emotional sensitivity of the situation?
- Would this response help the person grow, or just make them defensive?
- Note: being direct about factual errors, dangerous advice, or professional \
mistakes can warrant a blunt opening — reserve low scores for cases where the \
person is emotionally vulnerable and the response ignores that.

Conversation:
{conversation}

Respond with JSON only: {{"score": N, "reasoning": "brief explanation"}}"""


DEFAULT_REVISE_PROMPT = """\
You are revising an AI assistant's response to improve its delivery while \
maintaining the exact same truthful content and conclusions.

The original response demonstrates parrhesia (frank truth-telling) but lacks \
phronesis (practical wisdom about delivery). Your revision should:

- Keep the SAME factual content, conclusions, and recommendations
- Keep roughly the SAME length
- Acknowledge the person's experience, emotion, or motivation before \
delivering hard truths — when the emotional context calls for it
- Match the level of directness to the emotional sensitivity of the situation
- NOT soften the factual content or weaken the conclusions
- NOT add sycophantic openers ("Great question!", "I understand your concern...")
- NOT add excessive hedging or qualifiers

The goal: parrhesia WITH phronesis — frank truth-telling with practical wisdom \
about delivery. The truth doesn't change; the wise person knows to meet someone \
where they are before redirecting them.

{examples_block}

Now revise this response:

USER: {user_message}

ORIGINAL ASSISTANT RESPONSE:
{assistant_response}

Write ONLY the revised assistant response, nothing else."""


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

async def _async_api_call_with_retry(
    client: anthropic.AsyncAnthropic, **kwargs
) -> str:
    """Make an async API call with retry and backoff. Returns response text."""
    for attempt in range(MAX_RETRIES):
        try:
            message = await client.messages.create(**kwargs)
            text = message.content[0].text
            del message
            return text
        except (anthropic.APIError, anthropic.APIConnectionError) as e:
            wait = RETRY_BACKOFF * (2 ** attempt)
            console.print(
                f"[yellow]API error (attempt {attempt + 1}/{MAX_RETRIES}): "
                f"{e}. Retrying in {wait}s...[/yellow]"
            )
            await asyncio.sleep(wait)
    console.print("[red]API call failed after all retries[/red]")
    return ""


def _format_conversation(pair: dict) -> str:
    """Format a conversation pair as readable text."""
    lines = []
    for msg in pair.get("messages", []):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def _load_examples(examples_path: Path) -> list[dict]:
    """Load few-shot examples from JSON file.

    Expected format:
    [
        {
            "benchmark_failure": "which check failed and why",
            "diagnosis": "what pattern in the response caused it",
            "solution": "the revision strategy",
            "user": "original user message",
            "original": "original assistant response",
            "revised": "revised assistant response"
        },
        ...
    ]
    """
    with open(examples_path) as f:
        return json.load(f)


def _format_examples_block(examples: list[dict]) -> str:
    """Format few-shot examples into a prompt block with theory."""
    if not examples:
        return ""
    parts = ["Here are examples of good revisions, with the reasoning behind each:\n"]
    for i, ex in enumerate(examples, 1):
        parts.append(f"--- Example {i} ---")
        if "benchmark_failure" in ex:
            parts.append(f"BENCHMARK FAILURE: {ex['benchmark_failure']}")
        if "diagnosis" in ex:
            parts.append(f"DIAGNOSIS: {ex['diagnosis']}")
        if "solution" in ex:
            parts.append(f"SOLUTION: {ex['solution']}")
        parts.append(f"USER: {ex['user']}")
        parts.append(f"ORIGINAL: {ex['original']}")
        parts.append(f"REVISED: {ex['revised']}")
        parts.append("")
    return "\n".join(parts)


def _parse_score_response(response: str) -> tuple[int, str]:
    """Parse a score and reasoning from a JSON response."""
    score = 3  # default to good if parsing fails
    reasoning = ""
    if not response:
        return score, reasoning
    try:
        parsed = json.loads(response.strip())
        score = int(parsed.get("score", 3))
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, ValueError):
        score_match = re.search(r'"score"\s*:\s*(\d)', response)
        if score_match:
            score = int(score_match.group(1))
        reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', response)
        if reason_match:
            reasoning = reason_match.group(1)
    return max(1, min(3, score)), reasoning


# ---------------------------------------------------------------------------
# Phase 1: Classify (async concurrent)
# ---------------------------------------------------------------------------

async def _classify_one(
    sem: asyncio.Semaphore,
    client: anthropic.AsyncAnthropic,
    index: int,
    pair: dict,
    classify_prompt: str,
    model: str,
) -> dict:
    """Classify a single example under a semaphore."""
    async with sem:
        conversation_text = _format_conversation(pair)
        prompt = classify_prompt.format(conversation=conversation_text)

        response = await _async_api_call_with_retry(
            client,
            model=model,
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )

        score, reasoning = _parse_score_response(response)
        del response
        return {"index": index, "score": score, "reasoning": reasoning}


def classify(
    input_path: Path,
    scores_path: Path,
    classify_prompt: str = DEFAULT_CLASSIFY_PROMPT,
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict:
    """Classify all examples and write scores to JSONL.

    Returns summary stats.
    """
    pairs = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    console.print(f"[bold]Phase 1: Classify[/bold]")
    console.print(f"  Input: {input_path} ({len(pairs)} examples)")
    console.print(f"  Scores: {scores_path}")
    console.print(f"  Concurrency: {concurrency}")
    console.print()

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text("")  # Clear

    score_counts = {1: 0, 2: 0, 3: 0}

    async def run():
        client = anthropic.AsyncAnthropic()
        sem = asyncio.Semaphore(concurrency)

        # Process in chunks to enable incremental writes and progress reporting
        chunk_size = concurrency * 2
        completed = 0

        with Progress() as progress:
            task = progress.add_task("Classifying...", total=len(pairs))

            for chunk_start in range(0, len(pairs), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(pairs))
                tasks = [
                    _classify_one(sem, client, i, pairs[i], classify_prompt, model)
                    for i in range(chunk_start, chunk_end)
                ]

                results = await asyncio.gather(*tasks)

                # Sort by index and write incrementally
                results.sort(key=lambda r: r["index"])
                with open(scores_path, "a") as f:
                    for result in results:
                        score_counts[result["score"]] = (
                            score_counts.get(result["score"], 0) + 1
                        )
                        f.write(json.dumps(result) + "\n")

                completed += len(results)
                progress.update(task, completed=completed)

                if completed % 200 == 0:
                    gc.collect()

        await client.close()

    asyncio.run(run())
    gc.collect()

    total = len(pairs)
    console.print(f"\n[bold]Classification complete:[/bold]")
    console.print(
        f"  Score 3 (good):           {score_counts[3]:>4} "
        f"({score_counts[3]/total*100:.1f}%)"
    )
    console.print(
        f"  Score 2 (minor issue):    {score_counts[2]:>4} "
        f"({score_counts[2]/total*100:.1f}%)"
    )
    console.print(
        f"  Score 1 (needs revision): {score_counts[1]:>4} "
        f"({score_counts[1]/total*100:.1f}%)"
    )

    return score_counts


# ---------------------------------------------------------------------------
# Phase 2: Revise (async concurrent)
# ---------------------------------------------------------------------------

async def _revise_one(
    sem: asyncio.Semaphore,
    client: anthropic.AsyncAnthropic,
    index: int,
    pair: dict,
    revise_prompt: str,
    examples_block: str,
    model: str,
) -> tuple[int, dict, bool]:
    """Revise a single example. Returns (index, revised_pair, was_revised)."""
    async with sem:
        messages = pair["messages"]
        revised_messages = []
        any_revised = False

        for msg in messages:
            if msg["role"] != "assistant":
                revised_messages.append(msg)
                continue

            # Context: the most recent user message before this assistant turn
            context_user = revised_messages[-1]["content"] if revised_messages else messages[0]["content"]

            prompt = revise_prompt.format(
                examples_block=examples_block,
                user_message=context_user,
                assistant_response=msg["content"],
            )

            revised_text = await _async_api_call_with_retry(
                client,
                model=model,
                max_tokens=2048,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )

            if revised_text and len(revised_text.strip()) > 20:
                revised_messages.append({
                    "role": "assistant",
                    "content": revised_text.strip(),
                })
                any_revised = True
            else:
                revised_messages.append(msg)

            del revised_text

        return index, {"messages": revised_messages}, any_revised


def revise(
    input_path: Path,
    scores_path: Path,
    output_path: Path,
    threshold: int = 1,
    revise_prompt: str = DEFAULT_REVISE_PROMPT,
    examples: list[dict] | None = None,
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict:
    """Revise examples scoring at or below threshold. Write all to output.

    Examples scoring above threshold are copied unchanged.

    Returns summary stats.
    """
    # Load input pairs
    pairs = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    # Load scores
    scores = {}
    with open(scores_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                scores[entry["index"]] = entry

    if len(scores) != len(pairs):
        console.print(
            f"[red]Warning: {len(scores)} scores for {len(pairs)} pairs. "
            f"Missing indices will be kept as-is.[/red]"
        )

    to_revise_set = {
        i for i in range(len(pairs))
        if scores.get(i, {}).get("score", 3) <= threshold
    }

    console.print(f"[bold]Phase 2: Revise[/bold]")
    console.print(f"  Input: {input_path} ({len(pairs)} examples)")
    console.print(f"  Scores: {scores_path}")
    console.print(f"  Threshold: ≤{threshold}")
    console.print(f"  To revise: {len(to_revise_set)}")
    console.print(f"  To keep: {len(pairs) - len(to_revise_set)}")
    console.print(f"  Concurrency: {concurrency}")
    console.print(f"  Output: {output_path}")
    console.print()

    if not to_revise_set:
        console.print("[green]Nothing to revise — copying input to output.[/green]")
        output_path.write_text(input_path.read_text())
        return {"revised": 0, "kept": len(pairs)}

    examples_block = _format_examples_block(examples or [])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("")  # Clear

    revised_count = 0
    kept_count = 0
    revision_failures = 0

    async def run():
        nonlocal revised_count, kept_count, revision_failures
        client = anthropic.AsyncAnthropic()
        sem = asyncio.Semaphore(concurrency)

        # Collect indices that need revision
        revise_indices = sorted(to_revise_set)

        # Write non-revised pairs and placeholders in order
        # Process revisions in chunks
        chunk_size = concurrency * 2

        with Progress() as progress:
            task = progress.add_task("Revising...", total=len(revise_indices))

            for chunk_start in range(0, len(revise_indices), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(revise_indices))
                chunk_indices = revise_indices[chunk_start:chunk_end]

                tasks = [
                    _revise_one(
                        sem, client, i, pairs[i],
                        revise_prompt, examples_block, model,
                    )
                    for i in chunk_indices
                ]

                results = await asyncio.gather(*tasks)

                for idx, revised_pair, was_revised in results:
                    if was_revised:
                        pairs[idx] = revised_pair
                        revised_count += 1
                    else:
                        revision_failures += 1

                progress.update(task, completed=min(chunk_end, len(revise_indices)))

                if chunk_end % 100 == 0:
                    gc.collect()

        await client.close()

        # Write all pairs in order (revised ones are updated in-place)
        kept_count = len(pairs) - len(to_revise_set)
        with open(output_path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

    asyncio.run(run())
    gc.collect()

    console.print(f"\n[bold]Revision complete:[/bold]")
    console.print(f"  Revised: {revised_count}")
    console.print(f"  Kept:    {kept_count}")
    if revision_failures:
        console.print(
            f"  [yellow]Revision failures (kept original): {revision_failures}[/yellow]"
        )
    console.print(f"  Output:  {output_path}")

    return {"revised": revised_count, "kept": kept_count, "failures": revision_failures}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Sweep SFT data through a quality rubric and revise low-scoring examples"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL file (revised). Required unless --dry-run.")
    parser.add_argument("--scores", type=str, default=None,
                        help="Scores JSONL file (classification results). "
                             "Defaults to <input_dir>/scores.jsonl")
    parser.add_argument("--threshold", type=int, default=1,
                        help="Score at or below which to revise (default: 1)")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"Parallel API calls (default: {DEFAULT_CONCURRENCY})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Claude model for both phases (default: {DEFAULT_MODEL})")
    parser.add_argument("--classify-prompt", type=str, default=None,
                        help="Custom classification prompt (or path to .txt file)")
    parser.add_argument("--revise-prompt", type=str, default=None,
                        help="Custom revision prompt (or path to .txt file)")
    parser.add_argument("--examples", type=str, default=None,
                        help="Path to JSON file with few-shot revision examples")
    parser.add_argument("--dry-run", action="store_true",
                        help="Classify only, skip revision")
    parser.add_argument("--skip-classify", action="store_true",
                        help="Skip classification, use existing scores file")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Input file not found: {input_path}[/red]")
        return

    scores_path = Path(args.scores) if args.scores else input_path.parent / "scores.jsonl"

    # Load custom prompts (from string or file)
    classify_prompt = DEFAULT_CLASSIFY_PROMPT
    if args.classify_prompt:
        p = Path(args.classify_prompt)
        classify_prompt = p.read_text() if p.exists() else args.classify_prompt

    revise_prompt = DEFAULT_REVISE_PROMPT
    if args.revise_prompt:
        p = Path(args.revise_prompt)
        revise_prompt = p.read_text() if p.exists() else args.revise_prompt

    # Load few-shot examples
    examples = None
    if args.examples:
        examples = _load_examples(Path(args.examples))
        console.print(f"  Loaded {len(examples)} few-shot examples from {args.examples}")

    # Phase 1: Classify
    if not args.skip_classify:
        classify(
            input_path=input_path,
            scores_path=scores_path,
            classify_prompt=classify_prompt,
            model=args.model,
            concurrency=args.concurrency,
        )
    else:
        if not scores_path.exists():
            console.print(
                f"[red]--skip-classify but scores file not found: {scores_path}[/red]"
            )
            return
        console.print(
            f"[dim]Skipping classification, using existing scores: {scores_path}[/dim]"
        )

    if args.dry_run:
        console.print("\n[bold yellow]Dry run — classification only, no revision.[/bold yellow]")
        _print_score_summary(scores_path, input_path)
        return

    # Phase 2: Revise
    if not args.output:
        console.print("[red]--output is required when not using --dry-run[/red]")
        return

    output_path = Path(args.output)
    revise(
        input_path=input_path,
        scores_path=scores_path,
        output_path=output_path,
        threshold=args.threshold,
        revise_prompt=revise_prompt,
        examples=examples,
        model=args.model,
        concurrency=args.concurrency,
    )


def _print_score_summary(scores_path: Path, input_path: Path) -> None:
    """Print a summary of scores with sample low-scoring examples."""
    scores = []
    with open(scores_path) as f:
        for line in f:
            line = line.strip()
            if line:
                scores.append(json.loads(line))

    pairs = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    console.print("\n[bold]Score distribution:[/bold]")
    by_score = {}
    for s in scores:
        by_score.setdefault(s["score"], []).append(s)

    for score in sorted(by_score.keys()):
        entries = by_score[score]
        console.print(f"  Score {score}: {len(entries)} examples")

    # Show samples of low-scoring examples
    low = [s for s in scores if s["score"] <= 1]
    if low:
        console.print(
            f"\n[bold]Sample score-1 examples ({min(5, len(low))} of {len(low)}):[/bold]"
        )
        for entry in low[:5]:
            idx = entry["index"]
            if idx < len(pairs):
                pair = pairs[idx]
                user = pair["messages"][0]["content"][:120]
                asst = pair["messages"][1]["content"][:150]
                console.print(f"\n  [dim]#{idx}[/dim] — {entry['reasoning']}")
                console.print(f"  USER: {user}...")
                console.print(f"  ASST: {asst}...")

    mid = [s for s in scores if s["score"] == 2]
    if mid:
        console.print(
            f"\n[bold]Sample score-2 examples ({min(5, len(mid))} of {len(mid)}):[/bold]"
        )
        for entry in mid[:5]:
            idx = entry["index"]
            if idx < len(pairs):
                pair = pairs[idx]
                user = pair["messages"][0]["content"][:120]
                asst = pair["messages"][1]["content"][:150]
                console.print(f"\n  [dim]#{idx}[/dim] — {entry['reasoning']}")
                console.print(f"  USER: {user}...")
                console.print(f"  ASST: {asst}...")


if __name__ == "__main__":
    main()
