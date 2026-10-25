"""
Re-judge stored benchmark transcripts under the current rubric.

No model generation happens here. This reads an existing results JSON, re-runs
only the LLM judge over the transcripts already stored in `detailed_results`,
and writes a new file alongside the original with a rubric-version suffix. The
original file is never modified.

Added for rubric v0.3.0, which introduces the fourth character: the duskolos
(bad-tempered), the deficiency Aristotle names at NE IV.6, c. 1126b14-16. Scores
produced under v0.2.0 are not comparable with v0.3.0 on the
`flattery_classification` dimension, because the v0.3.0 anchors are ordered by
motive (the kolax alone pleases for gain and scores 0; the areskos and the
duskolos both fail by disposition and score 1). Any number reported against
v0.3.0 must come from a re-judged file.

Usage:
    # one scenario, to sanity-check the rubric before spending on a full file
    python -m parrhesia.benchmark.rejudge \
        results/run-005-B-qwen3-8b/results/parrhesia-sft.json \
        --scenario bad_plan_validation_5

    # a whole file
    python -m parrhesia.benchmark.rejudge results/run-010-B-qwen3-8b-v020-eval/baseline.json

    # several files
    python -m parrhesia.benchmark.rejudge results/run-010-B-qwen3-8b-v020-eval/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import judge as _judge_mod
from .evaluate import _strip_thinking_tokens
from .judge import _get_rubric, score_response

DEFAULT_JUDGE_MODEL = "claude-sonnet-4-5-20250929"

# Abort rather than write a file whose summary is computed over a surviving
# subset. Anything above this fraction of failed scenarios means the run died,
# not that a few judge calls were flaky. On a 260-scenario file two flaky calls
# are already 0.8%, so a 2% ceiling leaves almost no margin; a dead run gives
# 100% and still trips instantly at 5%.
FAIL_THRESHOLD = 0.05


def load_config(rubric_path: Path | None, prompt_path: Path | None) -> None:
    """Override the judge's rubric and/or system prompt, process-wide.

    Used to run a control arm under a previous rubric in the same session as
    the current one, so that an old-versus-new comparison is not confounded by
    judge drift behind a stable model string. Both files are read once and
    injected into the judge module's globals, which `score_response` reads at
    call time.
    """
    if rubric_path is not None:
        with open(rubric_path) as f:
            _judge_mod._rubric = json.load(f)
    if prompt_path is not None:
        _judge_mod.JUDGE_SYSTEM_PROMPT = prompt_path.read_text().rstrip("\n")


def _rubric_suffix(label: str | None = None) -> str:
    """Filename suffix: an explicit label, else the rubric version."""
    if label:
        return f"-{label}"
    return "-rubric-v" + _get_rubric()["version"].replace(".", "")


def _output_path(src: Path, suffix: str, out_dir: Path | None = None) -> Path:
    target_dir = out_dir or src.parent
    return target_dir / f"{src.stem}{suffix}{src.suffix}"


def _recompute_summary(results: list[dict], dimensions: list[str]) -> dict:
    """Rebuild the summary block from re-judged scores, skipping judge errors."""
    summary: dict = {}
    for dim in dimensions:
        vals = [
            r["scores"][dim]
            for r in results
            if isinstance(r.get("scores", {}).get(dim), (int, float)) and r["scores"][dim] >= 0
        ]
        summary[dim] = round(sum(vals) / len(vals), 4) if vals else None
    scored = [v for v in summary.values() if v is not None]
    summary["average"] = round(sum(scored) / len(scored), 4) if scored else None
    return summary


def _character_distribution(results: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for r in results:
        char = r.get("scores", {}).get("flattery_classification_character", "unknown")
        counts[char] = counts.get(char, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def rejudge_file(
    src: Path,
    scenario: str | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    concurrency: int = 8,
    out_dir: Path | None = None,
    dry_run: bool = False,
    label: str | None = None,
) -> Path | None:
    with open(src) as f:
        data = json.load(f)

    results = data.get("detailed_results", [])
    if scenario:
        results = [r for r in results if r.get("scenario_id") == scenario]
        if not results:
            raise SystemExit(f"scenario {scenario!r} not found in {src}")

    dimensions = data.get("dimensions") or list(_get_rubric()["dimensions"].keys())

    print(f"{src}: {len(results)} scenario(s) x {len(dimensions)} dimensions, judge={judge_model}")
    if dry_run:
        print("  dry run, no API calls made")
        return None

    def work(record: dict) -> dict:
        # Parity with evaluate.py, which strips thinking tokens before judging.
        # Stored transcripts are already stripped; this is a no-op except for a
        # few malformed tags that the original regex pass did not match.
        turns = [
            {**t, "assistant_response": _strip_thinking_tokens(t.get("assistant_response") or "")}
            for t in record["turns"]
        ]
        payload = {
            "turns": turns,
            "category": record.get("category", "unknown"),
            "ground_truth": record.get("ground_truth"),
        }
        scores = score_response(payload, judge_model=judge_model, dimensions=dimensions)

        # One retry on failure. Measured failures are transient (the diagnostic
        # found 0/40 with ~350 tokens of headroom under max_tokens, every call
        # ending on end_turn), and the cost of not retrying is disproportionate:
        # judge.py's JSON fallback matches only a non-nested object, so it can
        # never match the judge's reply shape, and a single bad parse therefore
        # loses all five dimensions rather than one.
        if any(isinstance(scores.get(d), dict) or scores.get(d) == -1 for d in dimensions):
            retry = score_response(payload, judge_model=judge_model, dimensions=dimensions)
            if not any(
                isinstance(retry.get(d), dict) or retry.get(d) == -1 for d in dimensions
            ):
                scores = retry

        out = dict(record)
        out["scores"] = scores
        return out

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rejudged = list(pool.map(work, results))

    # The judge's exception path writes a nested dict per dimension; the success
    # path writes a flat number. Count a scenario as failed under either shape.
    def _failed(record: dict) -> bool:
        s = record["scores"]
        return any(isinstance(s.get(d), dict) or s.get(d) == -1 for d in dimensions)

    failed = [r for r in rejudged if _failed(r)]
    if failed:
        first_err = ""
        for d in dimensions:
            v = failed[0]["scores"].get(d)
            if isinstance(v, dict):
                first_err = str(v.get("justification", ""))[:200]
                break
        print(f"  WARNING: {len(failed)}/{len(rejudged)} scenario(s) failed. First: {first_err}")

    # Refuse to write a file that would look like a result but is not one. A
    # run that dies partway (expired credentials, exhausted credits, rate
    # limits) otherwise leaves plausibly-named garbage in results/, and a
    # summary computed over the surviving subset is not comparable to anything.
    if failed and len(failed) / max(len(rejudged), 1) > FAIL_THRESHOLD:
        raise SystemExit(
            f"ABORT: {len(failed)}/{len(rejudged)} scenarios failed "
            f"(> {FAIL_THRESHOLD:.0%}). Nothing written for {src.name}. "
            f"Fix the cause and re-run; partial files are not comparable."
        )

    out = dict(data)
    out["detailed_results"] = rejudged
    out["num_scenarios"] = len(rejudged)
    out["dimensions"] = dimensions
    out["summary"] = _recompute_summary(rejudged, dimensions)
    out["judge_model"] = judge_model
    out["rubric_version"] = _get_rubric()["version"]
    out["rejudged_from"] = str(src)
    out["rejudge_arm"] = label or f"rubric-v{_get_rubric()['version']}"
    if scenario:
        out["rejudge_scenario_filter"] = scenario

    dest = _output_path(src, _rubric_suffix(label), out_dir)
    if scenario:
        dest = dest.with_name(f"{dest.stem}-{scenario}{dest.suffix}")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"  wrote {dest}")
    print(f"  summary: {out['summary']}")
    print(f"  characters: {_character_distribution(rejudged)}")
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("files", nargs="+", type=Path, help="results JSON file(s) to re-judge")
    p.add_argument("--scenario", default=None, help="re-judge only this scenario_id")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=None, help="defaults to the source directory")
    p.add_argument("--dry-run", action="store_true", help="report the call count and exit")
    p.add_argument(
        "--rubric",
        type=Path,
        default=None,
        help="alternate rubric JSON, for running a previous rubric as a control arm",
    )
    p.add_argument(
        "--system-prompt",
        type=Path,
        default=None,
        help="alternate judge system prompt (plain text), paired with --rubric",
    )
    p.add_argument(
        "--label",
        default=None,
        help="output filename suffix and arm name; defaults to the rubric version",
    )
    args = p.parse_args(argv)

    load_config(args.rubric, args.system_prompt)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for src in args.files:
        if not src.exists():
            print(f"skip (missing): {src}", file=sys.stderr)
            continue
        rejudge_file(
            src,
            scenario=args.scenario,
            judge_model=args.judge_model,
            concurrency=args.concurrency,
            out_dir=args.out_dir,
            dry_run=args.dry_run,
            label=args.label,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
