"""
Sycophancy benchmark evaluation runner.

Runs test scenarios against a model, then scores responses using an LLM judge
across the 5 Aristotelian dimensions.

Usage:
    parrhesia eval --model Qwen/Qwen3-8B --output results.json
    parrhesia eval --model ./adapters/parrhesia-oct-8b --output results_oct.json
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

load_dotenv()
console = Console()

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIOS_DIR = ROOT / "parrhesia" / "benchmark" / "scenarios"

DIMENSIONS = [
    "premature_agreement",
    "flattery_classification",
    "question_raising",
    "truth_telling_quality",
    "persistence",
]

# Sentinel written into a dimension's score when the judge fails to return a
# parseable 0-3 value (see judge.py). It marks missing data, not a real score,
# so it is excluded from aggregate means and counted separately.
JUDGE_ERROR = -1

DEFAULT_RESPONSE_CONCURRENCY = 16
DEFAULT_JUDGE_CONCURRENCY = 10


def run_evaluation(
    model: str,
    scenarios_path: str | None = None,
    output_path: str = "results.json",
    judge_model: str = "claude-sonnet-4-5-20250929",
    dimensions: list[str] | None = None,
    concurrency: int = DEFAULT_RESPONSE_CONCURRENCY,
    run_id: str | None = None,
    push: bool = False,
) -> dict:
    """Run the full evaluation pipeline."""
    dimensions = dimensions or DIMENSIONS

    # Load scenarios
    scenarios = load_scenarios(scenarios_path)
    console.print(f"Loaded {len(scenarios)} scenarios")

    # Generate model responses
    console.print(f"\n[bold]Generating responses from: {model}[/bold]")
    responses = generate_responses(model, scenarios, concurrency)

    # Strip thinking tokens before judging
    responses = _strip_thinking_from_responses(responses)

    # Judge responses
    judge_concurrency = min(concurrency, DEFAULT_JUDGE_CONCURRENCY)
    console.print(
        f"\n[bold]Judging responses with: {judge_model} "
        f"(concurrency={judge_concurrency})[/bold]"
    )
    results = judge_responses(responses, judge_model, dimensions, judge_concurrency)

    # Compute aggregate scores
    summary = compute_summary(results, dimensions)

    # Save
    output = {
        "model": model,
        "judge_model": judge_model,
        "num_scenarios": len(scenarios),
        "dimensions": dimensions,
        "summary": summary,
        "detailed_results": results,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print_summary(summary, model)

    console.print(f"\n[green]Full results saved to: {output_path}[/green]")

    # Record step in manifest
    import os
    _run_id = run_id or os.environ.get("PARRHESIA_RUN_ID")
    if _run_id:
        from parrhesia.manifest import record_step

        record_step(
            _run_id,
            step_name="evaluation",
            command=f"python -m parrhesia.benchmark.evaluate {model} --output {output_path} --judge-model {judge_model} --concurrency {concurrency}",
            hyperparameters={"model": model, "judge_model": judge_model, "concurrency": concurrency},
            outputs={"results": str(output_path)},
            metrics={dim: summary[dim]["mean"] for dim in dimensions if dim in summary},
        )

    # Push eval results to Hub (auto-push when run is active, or explicit --push)
    if push or _run_id:
        from parrhesia.hub import push_eval_results

        console.print("Pushing eval results to Hub...")
        url = push_eval_results(output_path, _run_id, model)
        console.print(f"  → {url}")

    return output


def load_scenarios(path: str | None = None) -> list[dict]:
    """Load benchmark scenarios from JSONL files."""
    if path:
        return _load_jsonl(Path(path))

    # Load all scenarios from the default directory
    scenarios = []
    scenarios_dir = DEFAULT_SCENARIOS_DIR
    if not scenarios_dir.exists():
        console.print(
            f"[red]No scenarios found at {scenarios_dir}. "
            "Generate scenarios first or provide a path.[/red]"
        )
        raise SystemExit(1)

    for f in sorted(scenarios_dir.glob("*.jsonl")):
        scenarios.extend(_load_jsonl(f))

    return scenarios


def generate_responses(
    model: str, scenarios: list[dict], concurrency: int = DEFAULT_RESPONSE_CONCURRENCY,
) -> list[dict]:
    """
    Generate model responses for each scenario.

    Supports:
    - Local PEFT adapter path (loads with transformers + peft)
    - HuggingFace model name (loads directly)
    - OpenAI-compatible API endpoint (set OPENAI_API_BASE env var)
    """
    adapter_path = Path(model)

    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        return _generate_with_peft_adapter(model, scenarios)
    else:
        console.print(f"  Concurrency: {concurrency} scenarios in parallel")
        return asyncio.run(_generate_with_api_async(model, scenarios, concurrency))


# ---------------------------------------------------------------------------
# Async API-based response generation
# ---------------------------------------------------------------------------


async def _generate_with_api_async(
    model: str, scenarios: list[dict], concurrency: int,
) -> list[dict]:
    """Generate responses using an OpenAI-compatible API with concurrent scenarios."""
    import os

    try:
        from openai import AsyncOpenAI
    except ImportError:
        console.print("[red]Install openai: pip install openai[/red]")
        raise SystemExit(1)

    api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
    client = AsyncOpenAI(base_url=api_base, api_key=api_key)

    semaphore = asyncio.Semaphore(concurrency)
    results = [None] * len(scenarios)

    async def process_scenario(idx: int, scenario: dict):
        async with semaphore:
            turns = _build_conversation_turns(scenario)
            all_responses = []
            messages = []

            for turn in turns:
                messages.append({"role": "user", "content": turn["user_message"]})

                try:
                    response = await client.chat.completions.create(
                        model=model,
                        max_tokens=1024,
                        temperature=0.0,
                        messages=messages,
                    )
                    assistant_msg = response.choices[0].message.content
                except Exception as e:
                    assistant_msg = f"[ERROR: {e}]"

                messages.append({"role": "assistant", "content": assistant_msg})
                all_responses.append(
                    {
                        "turn": turn["turn_number"],
                        "user_message": turn["user_message"],
                        "assistant_response": assistant_msg,
                    }
                )

            results[idx] = {
                "scenario_id": scenario["id"],
                "category": scenario["category"],
                "turns": all_responses,
                "ground_truth": scenario.get("ground_truth"),
            }

    with Progress() as progress:
        task = progress.add_task("Generating responses...", total=len(scenarios))

        async def process_with_progress(idx: int, scenario: dict):
            await process_scenario(idx, scenario)
            progress.advance(task)

        await asyncio.gather(
            *[process_with_progress(i, s) for i, s in enumerate(scenarios)]
        )

    return results


# ---------------------------------------------------------------------------
# PEFT adapter response generation (synchronous, single GPU)
# ---------------------------------------------------------------------------


def _generate_with_peft_adapter(adapter_path: str, scenarios: list[dict]) -> list[dict]:
    """Generate responses using a local PEFT adapter."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        console.print("[red]Install transformers and peft for local adapter evaluation[/red]")
        raise SystemExit(1)

    adapter_path = Path(adapter_path)

    # Load adapter config to find base model
    with open(adapter_path / "adapter_config.json") as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config.get("base_model_name_or_path", "")

    console.print(f"  Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    console.print(f"  Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    results = []
    with Progress() as progress:
        task = progress.add_task("Generating responses...", total=len(scenarios))

        for scenario in scenarios:
            turns = _build_conversation_turns(scenario)
            all_responses = []
            messages = []

            for turn in turns:
                messages.append({"role": "user", "content": turn["user_message"]})

                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        temperature=0.01,
                        do_sample=True,
                    )

                response_tokens = outputs[0][inputs["input_ids"].shape[1] :]
                assistant_msg = tokenizer.decode(response_tokens, skip_special_tokens=True)

                messages.append({"role": "assistant", "content": assistant_msg})
                all_responses.append(
                    {
                        "turn": turn["turn_number"],
                        "user_message": turn["user_message"],
                        "assistant_response": assistant_msg,
                    }
                )

            results.append(
                {
                    "scenario_id": scenario["id"],
                    "category": scenario["category"],
                    "turns": all_responses,
                    "ground_truth": scenario.get("ground_truth"),
                }
            )
            progress.advance(task)

    return results


def _build_conversation_turns(scenario: dict) -> list[dict]:
    """Extract conversation turns from a scenario."""
    turns = [{"turn_number": 1, "user_message": scenario["initial_prompt"]}]

    for i, followup in enumerate(scenario.get("followup_pushbacks", []), start=2):
        turns.append({"turn_number": i, "user_message": followup})

    return turns


# ---------------------------------------------------------------------------
# Judging (async)
# ---------------------------------------------------------------------------


def judge_responses(
    responses: list[dict],
    judge_model: str,
    dimensions: list[str],
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
) -> list[dict]:
    """Score each response using an LLM judge (concurrently)."""
    return asyncio.run(
        _judge_responses_async(responses, judge_model, dimensions, concurrency)
    )


async def _judge_responses_async(
    responses: list[dict],
    judge_model: str,
    dimensions: list[str],
    concurrency: int,
) -> list[dict]:
    """Score responses concurrently using asyncio.to_thread."""
    from parrhesia.benchmark.judge import score_response

    semaphore = asyncio.Semaphore(concurrency)
    results = [None] * len(responses)

    async def judge_one(idx: int, response: dict):
        async with semaphore:
            scores = await asyncio.to_thread(
                score_response, response, judge_model, dimensions
            )
            results[idx] = {**response, "scores": scores}

    with Progress() as progress:
        task = progress.add_task("Judging responses...", total=len(responses))

        async def judge_with_progress(idx: int, response: dict):
            await judge_one(idx, response)
            progress.advance(task)

        await asyncio.gather(
            *[judge_with_progress(i, r) for i, r in enumerate(responses)]
        )

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def compute_summary(results: list[dict], dimensions: list[str]) -> dict:
    """Compute aggregate scores across all scenarios."""
    summary = {}

    for dim in dimensions:
        scores = []
        errors = 0
        for r in results:
            score = r.get("scores", {}).get(dim)
            if isinstance(score, (int, float)):
                if score == JUDGE_ERROR:
                    errors += 1  # judge failed to score — missing, not a real 0
                else:
                    scores.append(score)

        if scores:
            summary[dim] = {
                "mean": round(sum(scores) / len(scores), 3),
                "count": len(scores),
                "errors": errors,
                "min": min(scores),
                "max": max(scores),
            }

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {dim: [] for dim in dimensions}
        for dim in dimensions:
            score = r.get("scores", {}).get(dim)
            if isinstance(score, (int, float)) and score != JUDGE_ERROR:
                categories[cat][dim].append(score)

    summary["by_category"] = {
        cat: {
            dim: round(sum(scores) / len(scores), 3) if scores else None
            for dim, scores in dims.items()
        }
        for cat, dims in categories.items()
    }

    return summary


def print_summary(summary: dict, model_name: str) -> None:
    """Print a formatted summary table."""
    table = Table(title=f"Parrhesia Benchmark: {model_name}")
    table.add_column("Dimension", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Count", justify="right")
    table.add_column("Range", justify="right")

    for dim in DIMENSIONS:
        if dim in summary:
            s = summary[dim]
            score_str = f"{s['mean']:.2f}/3.00"
            range_str = f"{s['min']}-{s['max']}"
            table.add_row(
                dim.replace("_", " ").title(),
                score_str,
                str(s["count"]),
                range_str,
            )

    console.print(table)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _strip_thinking_tokens(text: str) -> str:
    """Remove <think>...</think> blocks from model responses.

    Qwen3 and other models emit thinking tokens that contain internal
    deliberation. If passed to the judge, hedging and uncertainty in
    thinking content gets scored as sycophantic behavior, adding noise.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_thinking_from_responses(responses: list[dict]) -> list[dict]:
    """Strip thinking tokens from all assistant responses before judging."""
    for response in responses:
        for turn in response.get("turns", []):
            raw = turn.get("assistant_response", "")
            turn["assistant_response"] = _strip_thinking_tokens(raw)
    return responses


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run parrhesia sycophancy benchmark")
    parser.add_argument("model", help="Model name or path to adapter")
    parser.add_argument("--scenarios", type=str, default=None, help="Path to scenarios JSONL")
    parser.add_argument("--output", type=str, default="results.json", help="Output path")
    parser.add_argument(
        "--judge-model", type=str, default="claude-sonnet-4-5-20250929",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_RESPONSE_CONCURRENCY,
        help="Number of scenarios to evaluate concurrently (default: 16)",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Run ID for manifest tracking")
    parser.add_argument("--push", action="store_true", help="Push results to HuggingFace Hub")
    args = parser.parse_args()

    run_evaluation(
        model=args.model,
        scenarios_path=args.scenarios,
        output_path=args.output,
        judge_model=args.judge_model,
        concurrency=args.concurrency,
        run_id=args.run_id,
        push=args.push,
    )


if __name__ == "__main__":
    main()
