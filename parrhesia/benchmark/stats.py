"""
Confidence intervals for the parrhesia sycophancy benchmark.

`evaluate.py:compute_summary` reports only point-estimate means. This module
adds the uncertainty around them, in two layers:

  * **within-run** — a paired (cluster) bootstrap over the scenarios, for each
    dimension delta and the overall average delta (adapter - baseline). It
    captures *scenario-sampling* variance, conditional on one adapter and one
    judging pass. It is computable from existing result JSONs with no GPU.

  * **between-seed** — a Student-t interval over the per-seed overall deltas
    from a set of independently-retrained adapters. It captures *training-seed*
    variance, the dominant source the single-run headline does not yet account
    for. Feed it the per-seed deltas once the seed runs exist.

Judge-error sentinels (``score == -1``, emitted by ``judge.py`` when a response
cannot be parsed) are treated as **missing**, not as a real 0 — the judge
failed to score, which is not the same as the model scoring zero. A scenario
contributes its mean over the dimensions that scored validly; a scenario with
no valid score in either model is dropped from the pair. Pass
``--legacy-include-errors`` to average the -1s back in and reproduce the
original `compute_summary` headline.

Usage:
    python -m parrhesia.benchmark.stats \
        --baseline results/run-007-B-qwen3-8b/baseline.json \
        --adapter  results/run-007-B-qwen3-8b/parrhesia-sft.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

DIMENSIONS = [
    "premature_agreement",
    "flattery_classification",
    "question_raising",
    "truth_telling_quality",
    "persistence",
]

JUDGE_ERROR = -1  # sentinel written by judge.py when a response can't be scored

# Two-sided Student-t critical values, indexed by degrees of freedom. Hardcoded
# so the module stays pure-stdlib (no scipy). Falls back to the normal quantile
# for df beyond the table.
_T_CRIT = {
    0.95: {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
           7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
           15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042},
    0.90: {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943,
           7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812, 11: 1.796, 12: 1.782,
           15: 1.753, 20: 1.725, 25: 1.708, 30: 1.697},
}
_Z_CRIT = {0.95: 1.960, 0.90: 1.645}


def _t_critical(df: int, ci: float) -> float:
    """Two-sided t critical value for the given df, from the table above."""
    table = _T_CRIT.get(ci)
    if table is None:
        raise ValueError(f"No t-table for ci={ci}; use 0.95 or 0.90.")
    if df in table:
        return table[df]
    if df > 30:
        return _Z_CRIT[ci]
    below = [d for d in table if d <= df]
    return table[max(below)] if below else table[min(table)]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_dimension_scores(
    path: str | Path, include_errors: bool = False
) -> dict[str, dict[str, float | None]]:
    """Load per-scenario, per-dimension scores from an eval result JSON.

    Returns {scenario_id: {dimension: score-or-None}}. The judge-error sentinel
    (-1) and any non-numeric value map to None, unless ``include_errors`` is set
    (legacy mode), which keeps the -1 as a real value to reproduce the original
    `compute_summary` headline.
    """
    with open(path) as f:
        data = json.load(f)

    out: dict[str, dict[str, float | None]] = {}
    for r in data["detailed_results"]:
        sid = r["scenario_id"]
        scores = r.get("scores", {})
        row: dict[str, float | None] = {}
        for dim in DIMENSIONS:
            v = scores.get(dim)
            if isinstance(v, (int, float)) and (include_errors or v != JUDGE_ERROR):
                row[dim] = float(v)
            else:
                row[dim] = None
        out[sid] = row
    return out


def scenario_score(row: dict[str, float | None]) -> float | None:
    """Mean over the dimensions that scored validly for one scenario.

    Returns None if no dimension scored (the scenario is then unpaired). Legacy
    -1 handling is applied at load time, so by here a value is either a real
    score or None.
    """
    vals = [v for v in row.values() if v is not None]
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------------------
# Within-run paired (cluster) bootstrap
# ---------------------------------------------------------------------------


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = int(q * (len(sorted_vals) - 1))
    return sorted_vals[idx]


def within_run(
    baseline_path: str | Path,
    adapter_path: str | Path,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
    include_errors: bool = False,
) -> dict:
    """Paired cluster bootstrap over scenarios for the per-dimension and overall
    deltas (adapter - baseline).

    One bootstrap iteration resamples the scenario ids with replacement and
    recomputes every statistic on that resample, so the dimensions share the
    same resample (pairing and cross-dimension correlation are preserved).
    """
    base = load_dimension_scores(baseline_path, include_errors=include_errors)
    adpt = load_dimension_scores(adapter_path, include_errors=include_errors)
    shared = sorted(set(base) & set(adpt))

    # Analysis set: scenarios with a valid overall score in BOTH models.
    ids = [
        sid for sid in shared
        if scenario_score(base[sid]) is not None
        and scenario_score(adpt[sid]) is not None
    ]
    n = len(ids)
    if n == 0:
        raise ValueError("No paired scenarios with valid scores in both models.")

    base_overall = {sid: scenario_score(base[sid]) for sid in ids}
    adpt_overall = {sid: scenario_score(adpt[sid]) for sid in ids}

    def overall_delta(sample: list[str]) -> float:
        return sum(adpt_overall[s] - base_overall[s] for s in sample) / len(sample)

    def dim_delta(sample: list[str], dim: str) -> float | None:
        diffs = [
            adpt[s][dim] - base[s][dim]
            for s in sample
            if base[s][dim] is not None and adpt[s][dim] is not None
        ]
        return sum(diffs) / len(diffs) if diffs else None

    # Point estimates.
    point = {
        "n_scenarios": n,
        "n_shared": len(shared),
        "n_dropped_unpaired": len(shared) - n,
        "baseline_avg": sum(base_overall.values()) / n,
        "adapter_avg": sum(adpt_overall.values()) / n,
        "overall_delta": overall_delta(ids),
        "dimensions": {},
    }
    for dim in DIMENSIONS:
        b = [base[s][dim] for s in ids if base[s][dim] is not None]
        a = [adpt[s][dim] for s in ids if adpt[s][dim] is not None]
        point["dimensions"][dim] = {
            "baseline": sum(b) / len(b) if b else None,
            "adapter": sum(a) / len(a) if a else None,
            "delta": dim_delta(ids, dim),
        }

    # Bootstrap.
    rng = random.Random(seed)
    overall_boot: list[float] = []
    dim_boot: dict[str, list[float]] = {d: [] for d in DIMENSIONS}
    for _ in range(n_boot):
        sample = [ids[rng.randrange(n)] for _ in range(n)]
        overall_boot.append(overall_delta(sample))
        for dim in DIMENSIONS:
            d = dim_delta(sample, dim)
            if d is not None:
                dim_boot[dim].append(d)

    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2

    def summarize(boot: list[float], pt: float | None) -> dict:
        s = sorted(boot)
        mean = sum(boot) / len(boot)
        var = sum((x - mean) ** 2 for x in boot) / len(boot)
        return {
            "point": pt,
            "ci_low": _percentile(s, lo_q),
            "ci_high": _percentile(s, hi_q),
            "se": math.sqrt(var),
        }

    return {
        "kind": "within_run",
        "ci_level": ci,
        "n_boot": n_boot,
        "seed": seed,
        "include_errors": include_errors,
        **point,
        "overall_ci": summarize(overall_boot, point["overall_delta"]),
        "dimension_ci": {
            dim: summarize(dim_boot[dim], point["dimensions"][dim]["delta"])
            for dim in DIMENSIONS
        },
    }


# ---------------------------------------------------------------------------
# Between-seed interval (training-seed variance)
# ---------------------------------------------------------------------------


def between_seed(deltas: list[float], ci: float = 0.95) -> dict:
    """Student-t interval over per-seed overall deltas.

    `deltas` is one number per independently-retrained seed (e.g. each seed's
    adapter_avg - baseline_avg). With small n the t-interval is wide by design;
    that width is the honest cost of few seeds.
    """
    n = len(deltas)
    if n < 2:
        raise ValueError("Need >=2 seeds for a between-seed interval.")
    mean = sum(deltas) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in deltas) / (n - 1))
    se = sd / math.sqrt(n)
    t = _t_critical(n - 1, ci)
    return {
        "kind": "between_seed",
        "ci_level": ci,
        "n_seeds": n,
        "mean_delta": mean,
        "sd": sd,
        "se": se,
        "t_crit": t,
        "ci_low": mean - t * se,
        "ci_high": mean + t * se,
        "deltas": list(deltas),
    }


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------


def overall_point_delta(
    baseline_path: str | Path, adapter_path: str | Path, include_errors: bool = False
) -> dict:
    """Point estimate of the overall delta (no bootstrap) for one (baseline,
    adapter) pair — the per-seed quantity the between-seed interval is built on."""
    base = load_dimension_scores(baseline_path, include_errors=include_errors)
    adpt = load_dimension_scores(adapter_path, include_errors=include_errors)
    ids = [
        sid for sid in set(base) & set(adpt)
        if scenario_score(base[sid]) is not None
        and scenario_score(adpt[sid]) is not None
    ]
    if not ids:
        raise ValueError(f"No paired scenarios for {adapter_path}")
    bavg = sum(scenario_score(base[s]) for s in ids) / len(ids)
    aavg = sum(scenario_score(adpt[s]) for s in ids) / len(ids)
    return {"baseline_avg": bavg, "adapter_avg": aavg, "delta": aavg - bavg, "n": len(ids)}


def _seed_adapter_files(results_dir: str | Path) -> tuple[Path, list[Path]]:
    """In an eval results dir, return (baseline.json, [per-seed adapter jsons]).

    Adapter files are the quantitative result jsons other than baseline.json;
    qualitative-*.json and *-rejudged.json are skipped.
    """
    d = Path(results_dir)
    baseline = d / "baseline.json"
    if not baseline.exists():
        raise FileNotFoundError(f"No baseline.json in {d}")
    adapters = sorted(
        p for p in d.glob("*.json")
        if p.name != "baseline.json"
        and not p.name.startswith("qualitative")
        and not p.stem.endswith("-rejudged")
    )
    return baseline, adapters


def aggregate_seeds(
    results_dir: str | Path, ci: float = 0.95, include_errors: bool = False
) -> dict:
    """Read a multi-seed eval dir (baseline + N adapter jsons) and return the
    between-seed interval on the overall delta (adapter - baseline)."""
    baseline, adapters = _seed_adapter_files(results_dir)
    seeds = [
        {"label": ad.stem, **overall_point_delta(baseline, ad, include_errors=include_errors)}
        for ad in adapters
    ]
    deltas = [s["delta"] for s in seeds]
    return {
        "results_dir": str(results_dir),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "between_seed": between_seed(deltas, ci=ci) if len(deltas) >= 2 else None,
    }


def equivalence(
    dir_a: str | Path, dir_b: str | Path, ci: float = 0.95, include_errors: bool = False
) -> dict:
    """Two-sample (Welch) test of whether two arms' per-seed deltas differ.

    Each arm nets out its own baseline, so the deltas are comparable across arms.
    A CI on the difference that contains 0 means the arms are not distinguishable
    at this sample size — e.g. v0.1.0 vs v0.2.0 constitutions.
    """
    a = aggregate_seeds(dir_a, ci=ci, include_errors=include_errors)
    b = aggregate_seeds(dir_b, ci=ci, include_errors=include_errors)
    da = [s["delta"] for s in a["seeds"]]
    db = [s["delta"] for s in b["seeds"]]
    na, nb = len(da), len(db)
    if na < 2 or nb < 2:
        raise ValueError("Need >=2 seeds per arm for an equivalence test.")
    ma, mb = sum(da) / na, sum(db) / nb
    va = sum((x - ma) ** 2 for x in da) / (na - 1)
    vb = sum((x - mb) ** 2 for x in db) / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    diff = ma - mb
    # Welch-Satterthwaite df (floored for the integer t-table).
    if se > 0:
        df = (va / na + vb / nb) ** 2 / (
            (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
        )
    else:
        df = na + nb - 2
    t = _t_critical(int(df), ci)
    lo, hi = diff - t * se, diff + t * se
    return {
        "arm_a": {"dir": str(dir_a), "mean_delta": ma, "n": na},
        "arm_b": {"dir": str(dir_b), "mean_delta": mb, "n": nb},
        "diff": diff, "se": se, "df": df, "ci_level": ci,
        "ci_low": lo, "ci_high": hi,
        "distinguishable": not (lo <= 0 <= hi),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_within_run(res: dict) -> None:
    pct = int(res["ci_level"] * 100)
    mode = "legacy: -1 errors included" if res.get("include_errors") else "-1 errors treated as missing"
    table = Table(
        title=f"Within-run {pct}% CIs · paired bootstrap over {res['n_scenarios']} scenarios ({mode})"
    )
    table.add_column("Dimension", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Adapter", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column(f"{pct}% CI", justify="right")

    for dim in DIMENSIONS:
        d = res["dimensions"][dim]
        c = res["dimension_ci"][dim]
        table.add_row(
            dim.replace("_", " ").title(),
            f"{d['baseline']:.3f}" if d["baseline"] is not None else "—",
            f"{d['adapter']:.3f}" if d["adapter"] is not None else "—",
            f"{d['delta']:+.3f}" if d["delta"] is not None else "—",
            f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]",
        )

    oc = res["overall_ci"]
    table.add_section()
    table.add_row(
        "Average",
        f"{res['baseline_avg']:.3f}",
        f"{res['adapter_avg']:.3f}",
        f"{res['overall_delta']:+.3f}",
        f"[{oc['ci_low']:+.3f}, {oc['ci_high']:+.3f}]",
        style="bold",
    )
    console.print(table)
    if res["n_dropped_unpaired"]:
        console.print(
            f"[dim]Dropped {res['n_dropped_unpaired']} unpaired scenario(s) "
            f"(no valid judge score in one model).[/dim]"
        )
    console.print(
        "[dim]Within-run CIs capture scenario-sampling variance only — they "
        "condition on one adapter and one judging pass. Training-seed variance "
        "is measured separately (between_seed).[/dim]"
    )


def print_aggregate(agg: dict) -> None:
    table = Table(title=f"Multi-seed deltas · {agg['n_seeds']} seed(s) · {Path(agg['results_dir']).name}")
    table.add_column("Seed adapter", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Adapter", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("n", justify="right")
    for s in agg["seeds"]:
        table.add_row(
            s["label"], f"{s['baseline_avg']:.3f}", f"{s['adapter_avg']:.3f}",
            f"{s['delta']:+.3f}", str(s["n"]),
        )
    console.print(table)
    bs = agg["between_seed"]
    if bs:
        pct = int(bs["ci_level"] * 100)
        console.print(
            f"[bold]Between-seed: mean Δ = {bs['mean_delta']:+.3f}, "
            f"{pct}% CI [{bs['ci_low']:+.3f}, {bs['ci_high']:+.3f}]  "
            f"(SD {bs['sd']:.3f}, n={bs['n_seeds']}, t*={bs['t_crit']})[/bold]"
        )
    else:
        console.print("[yellow]Need ≥2 seeds for a between-seed interval.[/yellow]")


def print_equivalence(eq: dict) -> None:
    pct = int(eq["ci_level"] * 100)
    verdict = "DISTINGUISHABLE" if eq["distinguishable"] else "not distinguishable"
    console.print(
        f"[bold]Arm A meanΔ {eq['arm_a']['mean_delta']:+.3f} (n={eq['arm_a']['n']}) vs "
        f"Arm B meanΔ {eq['arm_b']['mean_delta']:+.3f} (n={eq['arm_b']['n']}): "
        f"diff {eq['diff']:+.3f}, {pct}% CI [{eq['ci_low']:+.3f}, {eq['ci_high']:+.3f}] → {verdict}[/bold]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Confidence intervals for the parrhesia benchmark")
    parser.add_argument("--baseline", help="Baseline eval result JSON (within-run mode)")
    parser.add_argument("--adapter", help="Adapter eval result JSON (within-run mode)")
    parser.add_argument("--aggregate", help="Multi-seed eval dir (baseline.json + per-seed adapter jsons): between-seed CI")
    parser.add_argument("--equivalence", help="Second multi-seed eval dir: test whether the two arms' deltas differ")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--legacy-include-errors", action="store_true",
        help="Average judge-error (-1) sentinels in as real scores (reproduces the original headline)",
    )
    parser.add_argument("--json-out", type=str, default=None, help="Write the result dict to this path")
    args = parser.parse_args()

    # Multi-seed mode.
    if args.aggregate:
        agg = aggregate_seeds(args.aggregate, ci=args.ci, include_errors=args.legacy_include_errors)
        print_aggregate(agg)
        out: dict = agg
        if args.equivalence:
            agg_b = aggregate_seeds(args.equivalence, ci=args.ci, include_errors=args.legacy_include_errors)
            print_aggregate(agg_b)
            eq = equivalence(args.aggregate, args.equivalence, ci=args.ci, include_errors=args.legacy_include_errors)
            print_equivalence(eq)
            out = {"arm_a": agg, "arm_b": agg_b, "equivalence": eq}
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(out, f, indent=2)
            console.print(f"[green]Wrote {args.json_out}[/green]")
        return

    # Within-run mode.
    if not (args.baseline and args.adapter):
        parser.error("Provide --baseline and --adapter (within-run mode), or --aggregate <dir> (multi-seed mode).")

    res = within_run(
        args.baseline, args.adapter,
        n_boot=args.n_boot, ci=args.ci, seed=args.seed,
        include_errors=args.legacy_include_errors,
    )
    print_within_run(res)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)
        console.print(f"[green]Wrote {args.json_out}[/green]")


if __name__ == "__main__":
    main()
