"""Re-judge existing eval results after stripping thinking tokens.

Loads saved result files, strips <think>...</think> from assistant responses,
re-runs the Claude judge, and saves corrected results.

Usage:
    python scripts/rejudge.py results/run-002-A-qwen3-8b/baseline.json
    python scripts/rejudge.py results/run-002-A-qwen3-8b/*.json
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parrhesia.benchmark.evaluate import (
    DIMENSIONS,
    _strip_thinking_tokens,
    compute_summary,
    judge_responses,
    print_summary,
)
from rich.console import Console

console = Console()


def rejudge_file(path: Path, judge_model: str = "claude-sonnet-4-5-20250929", concurrency: int = 10):
    """Re-judge a single results file."""
    console.print(f"\n[bold]Re-judging: {path}[/bold]")

    with open(path) as f:
        data = json.load(f)

    model = data["model"]
    dimensions = data.get("dimensions", DIMENSIONS)
    responses = data["detailed_results"]

    # Strip thinking tokens
    stripped = 0
    for r in responses:
        for turn in r.get("turns", []):
            raw = turn.get("assistant_response", "")
            clean = _strip_thinking_tokens(raw)
            if clean != raw:
                stripped += 1
            turn["assistant_response"] = clean
        # Remove old scores so judge_responses gets clean input
        r.pop("scores", None)

    console.print(f"  Stripped thinking tokens from {stripped} turns")

    # Re-judge
    console.print(f"  Judging with {judge_model} (concurrency={concurrency})")
    results = judge_responses(responses, judge_model, dimensions, concurrency)

    # Recompute summary
    summary = compute_summary(results, dimensions)

    # Save to new file
    out_path = path.with_stem(path.stem + "-rejudged")
    output = {
        "model": model,
        "judge_model": judge_model,
        "num_scenarios": len(responses),
        "dimensions": dimensions,
        "summary": summary,
        "detailed_results": results,
        "rejudged_from": str(path),
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print_summary(summary, model)
    console.print(f"  [green]Saved to: {out_path}[/green]")

    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/rejudge.py <result_file> [result_file ...]")
        sys.exit(1)

    for arg in sys.argv[1:]:
        rejudge_file(Path(arg))
