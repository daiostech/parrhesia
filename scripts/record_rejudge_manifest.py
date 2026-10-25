#!/usr/bin/env python
"""Record a manifest step for every re-judged results file.

`rejudge.py` re-scores stored transcripts under a given rubric and writes a new
file beside the original. This records each of those files as a `rejudge` step
in the manifest of the run the source file belongs to, so the provenance of a
re-judged number is recoverable the same way a training or eval step is.

The run id is taken from the results path (`results/<run-id>/...`). Metrics come
from the written file itself, not from the console output, so a step is only
recorded for a file that actually exists on disk.

Re-running is safe: a step whose source, destination and arm already appear in
the manifest is skipped rather than duplicated.

Usage:
    python scripts/record_rejudge_manifest.py                     # all re-judged files
    python scripts/record_rejudge_manifest.py --dry-run           # show, record nothing
    python scripts/record_rejudge_manifest.py results/run-007-B-qwen3-8b
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parrhesia.manifest import load_manifest, record_step  # noqa: E402

DIMS = [
    "premature_agreement",
    "flattery_classification",
    "question_raising",
    "truth_telling_quality",
    "persistence",
]
SUFFIXES = ("-ctl-v020", "-new-v030")


def _run_id_for(path: Path) -> str | None:
    """results/<run-id>/[results/]<file>.json -> <run-id>."""
    parts = path.resolve().relative_to(ROOT).parts
    if len(parts) >= 2 and parts[0] == "results":
        return parts[1]
    return None


def _source_for(path: Path) -> Path | None:
    for suf in SUFFIXES:
        if path.stem.endswith(suf):
            return path.with_name(path.stem[: -len(suf)] + path.suffix)
    return None


def _failure_count(data: dict) -> int:
    bad = 0
    for r in data.get("detailed_results", []):
        s = r.get("scores", {})
        if any(isinstance(s.get(d), dict) or s.get(d) == -1 for d in DIMS):
            bad += 1
    return bad


def _char_distribution(data: dict) -> dict:
    counts: dict[str, int] = {}
    for r in data.get("detailed_results", []):
        c = r.get("scores", {}).get("flattery_classification_character", "unknown")
        counts[c] = counts.get(c, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _already_recorded(run_id: str, dest: str, arm: str) -> bool:
    try:
        manifest = load_manifest(run_id)
    except FileNotFoundError:
        return False
    for step in manifest.steps:
        if step.get("step_name") != "rejudge":
            continue
        if step.get("outputs", {}).get("results") == dest and step.get(
            "hyperparameters", {}
        ).get("arm") == arm:
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("roots", nargs="*", type=Path, default=[ROOT / "results"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode().strip())

    files: list[Path] = []
    for root in args.roots:
        root = Path(root)
        for suf in SUFFIXES:
            files.extend(sorted(root.rglob(f"*{suf}.json")))
    files = sorted(set(files))

    if not files:
        print("no re-judged files found")
        return 0

    recorded = skipped = 0
    for f in files:
        run_id = _run_id_for(f)
        src = _source_for(f)
        if run_id is None or src is None:
            print(f"skip (cannot resolve run id or source): {f}")
            continue

        data = json.load(open(f))
        arm = data.get("rejudge_arm", "unknown")
        rubric_version = data.get("rubric_version", "unknown")
        dest_rel = str(f.resolve().relative_to(ROOT))
        src_rel = str(src.resolve().relative_to(ROOT))
        n = len(data.get("detailed_results", []))
        failed = _failure_count(data)

        if _already_recorded(run_id, dest_rel, arm):
            print(f"skip (already recorded): {dest_rel}")
            skipped += 1
            continue

        control = arm.startswith("ctl")
        cmd = (
            f"python -m parrhesia.benchmark.rejudge {src_rel} "
            + (
                "--rubric <v0.2.0 rubric> --system-prompt <v0.2.0 judge prompt> "
                if control
                else ""
            )
            + f"--label {arm}"
        )

        hyper = {
            "arm": arm,
            "rubric_version": rubric_version,
            "judge_model": data.get("judge_model"),
            "control_arm": control,
            "git_sha": git_sha,
            "git_dirty": dirty,
        }
        metrics = {
            "scenarios": n,
            "failed_scenarios": failed,
            "summary": data.get("summary", {}),
            "character_distribution": _char_distribution(data),
        }
        notes = (
            "Re-judge only: the stored transcripts were re-scored by the LLM judge, "
            "no model generation. "
            + (
                "Control arm, pinned to the pre-duskolos rubric and judge prompt at "
                "commit 777611b, run in the same session as the v0.3.0 arm because "
                "stored v0.2.0 scores do not reproduce under v0.2.0 today."
                if control
                else "Treatment arm, rubric v0.3.0, which adds the duskolos "
                "(deficiency) character and reorders the character-classification "
                "anchors by motive. Not comparable with scores produced under v0.2.0."
            )
        )

        print(f"record  {run_id:34} {arm:10} {dest_rel}  ({n - failed}/{n} scored)")
        if not args.dry_run:
            record_step(
                run_id=run_id,
                step_name="rejudge",
                command=cmd,
                hyperparameters=hyper,
                inputs={"source_results": src_rel},
                outputs={"results": dest_rel},
                metrics=metrics,
                notes=notes,
            )
        recorded += 1

    verb = "would record" if args.dry_run else "recorded"
    print(f"\n{verb} {recorded} step(s); skipped {skipped} already present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
