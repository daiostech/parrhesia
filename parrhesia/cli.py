"""CLI entry point for parrhesia."""

import typer

app = typer.Typer(
    name="parrhesia",
    help="Aristotelian virtue training for truth-telling in AI models.",
)


@app.command()
def eval(
    model: str = typer.Argument(help="Model name or path to evaluate"),
    scenarios: str = typer.Option(
        None, help="Path to scenarios JSONL file. Uses built-in scenarios if not provided."
    ),
    output: str = typer.Option("results.json", help="Output path for results"),
    judge_model: str = typer.Option(
        "claude-sonnet-4-5-20250929", help="Model to use as judge"
    ),
    dimensions: list[str] = typer.Option(
        None, help="Specific dimensions to evaluate (default: all 5)"
    ),
):
    """Run the sycophancy benchmark against a model."""
    from parrhesia.benchmark.evaluate import run_evaluation

    run_evaluation(
        model=model,
        scenarios_path=scenarios,
        output_path=output,
        judge_model=judge_model,
        dimensions=dimensions,
    )


@app.command()
def generate(
    approach: str = typer.Argument(help="Approach to generate data for: oct, sft, dpo, introspection"),
    output_dir: str = typer.Option("data/generated", help="Output directory"),
    num_scenarios: int = typer.Option(100, help="Number of scenarios to generate per category"),
    teacher_model: str = typer.Option(
        "claude-sonnet-4-5-20250929", help="Teacher model for data generation"
    ),
    student_model: str = typer.Option(
        None, help="Student/DPO-trained model (introspection only)"
    ),
    api_base: str = typer.Option(
        None, help="OpenAI-compatible API base URL (introspection only)"
    ),
    api_key: str = typer.Option(
        None, help="API key for student model endpoint (introspection only)"
    ),
):
    """Generate training data for a specific approach."""
    if approach == "oct":
        from parrhesia.data.generate_oct import generate_oct_data

        generate_oct_data(output_dir=output_dir, num_scenarios=num_scenarios, teacher_model=teacher_model)
    elif approach == "sft":
        from parrhesia.data.generate_sft import generate_sft_data

        generate_sft_data(output_dir=output_dir, num_scenarios=num_scenarios, teacher_model=teacher_model)
    elif approach == "dpo":
        from parrhesia.data.generate_dpo import generate_dpo_data

        generate_dpo_data(output_dir=output_dir, num_scenarios=num_scenarios, teacher_model=teacher_model)
    elif approach == "introspection":
        from parrhesia.data.generate_introspection import run_full_pipeline

        run_full_pipeline(
            model=student_model or "Qwen/Qwen3-8B",
            api_base=api_base,
            api_key=api_key,
            output_dir=output_dir,
        )
    else:
        typer.echo(f"Unknown approach: {approach}. Use 'oct', 'sft', 'dpo', or 'introspection'.")
        raise typer.Exit(1)


@app.command()
def report(
    results: str = typer.Argument(help="Path to results JSON file"),
    format: str = typer.Option("terminal", help="Output format: terminal, html, json"),
):
    """Generate a scorecard from benchmark results."""
    from parrhesia.benchmark.report import generate_report

    generate_report(results_path=results, format=format)


# ---------------------------------------------------------------------------
# Run manifest commands
# ---------------------------------------------------------------------------


@app.command("run-new")
def run_new(
    approach: str = typer.Argument(help="Training approach: A, B, or C"),
    base_model: str = typer.Option("Qwen/Qwen3-8B", "--base-model", help="Base model name"),
    description: str = typer.Option("", "--description", help="Run description"),
):
    """Create a new run manifest and print the run ID."""
    from parrhesia.manifest import generate_run_id, create_manifest

    run_id = generate_run_id(approach, base_model)
    manifest = create_manifest(run_id, approach, base_model, description)

    typer.echo(f"Created run: {run_id}")
    typer.echo(f"  Manifest: runs/{run_id}/manifest.yaml")
    typer.echo(f"")
    typer.echo(f"Set the run ID in your shell:")
    typer.echo(f"  export PARRHESIA_RUN_ID={run_id}")


@app.command("run-show")
def run_show(
    run_id: str = typer.Argument(help="Run ID to display"),
):
    """Display a run manifest as formatted markdown."""
    from parrhesia.manifest import load_manifest, format_manifest_markdown
    from rich.console import Console
    from rich.markdown import Markdown

    manifest = load_manifest(run_id)
    md = format_manifest_markdown(manifest)
    Console().print(Markdown(md))


@app.command("run-reproduce")
def run_reproduce(
    run_id: str = typer.Argument(help="Run ID to reproduce"),
):
    """Print ordered shell commands to reproduce a run."""
    from parrhesia.manifest import reconstruct_commands

    commands = reconstruct_commands(run_id)
    if not commands:
        typer.echo(f"No commands recorded for {run_id}")
        raise typer.Exit(1)

    typer.echo(f"# Commands to reproduce {run_id}")
    typer.echo(f"export PARRHESIA_RUN_ID={run_id}")
    typer.echo("")
    for i, cmd in enumerate(commands, 1):
        typer.echo(f"# Step {i}")
        typer.echo(cmd)
        typer.echo("")


# ---------------------------------------------------------------------------
# Qualitative benchmark commands
# ---------------------------------------------------------------------------


@app.command()
def qualitative(
    model: str = typer.Argument(help="Model name to evaluate"),
    api_base: str = typer.Option("http://localhost:8000/v1", "--api-base", help="OpenAI-compatible API base URL"),
    output: str = typer.Option(None, help="Output path for results (default: results/<run-id>/qualitative.json)"),
    judge_model: str = typer.Option("claude-sonnet-4-5-20250929", "--judge-model", help="Model to use as behavior judge"),
    run_id: str = typer.Option(None, "--run-id", help="Run ID for manifest tracking"),
    training_data: str = typer.Option("data/generated/oct/prompts.jsonl", "--training-data", help="Path to training data JSONL for contamination check"),
    prompt_key: str = typer.Option("prompt", "--prompt-key", help="Dot-notation key to extract prompt text (e.g. 'prompt', 'messages.0.content')"),
    golden_prompts: str = typer.Option(None, "--golden-prompts", help="Path to golden prompts JSONL (default: built-in golden_prompts.jsonl)"),
    skip_contamination: bool = typer.Option(
        True,
        "--skip-contamination/--no-skip-contamination",
        help="Skip the LLM contamination check. Default true; use --no-skip-contamination for trained adapters whose training data may topically overlap golden prompts.",
    ),
):
    """Run golden prompt qualitative benchmark."""
    from parrhesia.benchmark.qualitative import run_qualitative

    import os
    _run_id = run_id or os.environ.get("PARRHESIA_RUN_ID")
    if output is None:
        if _run_id:
            output = f"results/{_run_id}/qualitative.json"
        else:
            output = "qualitative.json"

    kwargs = dict(
        model=model,
        api_base=api_base,
        output_path=output,
        judge_model=judge_model,
        run_id=_run_id,
        training_data=training_data,
        prompt_key=prompt_key,
        skip_contamination=skip_contamination,
    )
    if golden_prompts:
        kwargs["golden_path"] = golden_prompts

    run_qualitative(**kwargs)


@app.command("qualitative-diff")
def qualitative_diff(
    result_a: str = typer.Argument(help="Path to first qualitative results JSON"),
    result_b: str = typer.Argument(help="Path to second qualitative results JSON"),
):
    """Compare qualitative results between two runs."""
    from parrhesia.benchmark.qualitative import diff_qualitative

    diff_qualitative(result_a, result_b)


# ---------------------------------------------------------------------------
# Adapter merge commands
# ---------------------------------------------------------------------------


@app.command("merge-adapters")
def merge_adapters(
    base_model: str = typer.Option("Qwen/Qwen3-8B", "--base-model", help="Base model name"),
    dpo_adapter: str = typer.Option(..., "--dpo-adapter", help="Path to DPO LoRA adapter"),
    sft_adapter: str = typer.Option(..., "--sft-adapter", help="Path to SFT/introspection LoRA adapter"),
    output: str = typer.Option(..., "--output", help="Output directory for merged adapter"),
    dpo_weight: float = typer.Option(1.0, "--dpo-weight", help="Weight for DPO adapter"),
    sft_weight: float = typer.Option(0.25, "--sft-weight", help="Weight for SFT adapter"),
):
    """Weighted merge of DPO and SFT LoRA adapters (OCT-style)."""
    from scripts.merge_adapters import merge_adapters as _merge

    _merge(
        base_model=base_model,
        dpo_adapter=dpo_adapter,
        sft_adapter=sft_adapter,
        output_dir=output,
        dpo_weight=dpo_weight,
        sft_weight=sft_weight,
    )


# ---------------------------------------------------------------------------
# Hub commands
# ---------------------------------------------------------------------------


@app.command("hub-push")
def hub_push(
    run_id: str = typer.Argument(help="Run ID to push artifacts for"),
    what: str = typer.Option("all", help="What to push: all, data, adapters, results"),
):
    """Push artifacts for a run to HuggingFace Hub."""
    from parrhesia.hub import push_all

    typer.echo(f"Pushing {what} for {run_id}...")
    urls = push_all(run_id, what)

    if urls:
        typer.echo(f"\nPushed {len(urls)} artifacts:")
        for name, url in urls.items():
            typer.echo(f"  {name}: {url}")
    else:
        typer.echo("No artifacts found to push.")


@app.command("hub-pull")
def hub_pull(
    run_id: str = typer.Argument(help="Run ID to pull artifacts for"),
    what: str = typer.Option("all", help="What to pull: all, data, adapters, results"),
):
    """Pull artifacts for a run from HuggingFace Hub."""
    from parrhesia.hub import pull_all

    typer.echo(f"Pulling {what} for {run_id}...")
    paths = pull_all(run_id, what)

    if paths:
        typer.echo(f"\nPulled {len(paths)} artifacts:")
        for name, path in paths.items():
            typer.echo(f"  {name}: {path}")
    else:
        typer.echo("No artifacts found to pull.")


if __name__ == "__main__":
    app()
