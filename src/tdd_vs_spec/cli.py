import asyncio
from pathlib import Path

import typer
from rich.console import Console

from .conditions import (
    Condition,
    load_instances,
    load_llm_specs,
    write_instances,
)
from .spec_gen import generate_all_specs
from .runner import (
    gather_patches,
    run_condition,
    write_patches_json,
)
from .analysis import cost_analysis, per_repo_breakdown, print_summary, load_results

app = typer.Typer(help="Is the spec additive? TDD vs spec-driven for coding agents.")
console = Console()

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


@app.command()
def generate_specs(
    output: Path = typer.Option(
        DATA_DIR / "llm_specs.jsonl", help="Output path for generated specs"
    ),
    limit: int | None = typer.Option(None, help="Limit number of instances"),
    concurrency: int = typer.Option(10, help="Concurrent API calls"),
) -> None:
    """Generate LLM specs from test patches using pydantic-ai."""
    assert output is not None, "output path must not be None"
    assert concurrency > 0, "concurrency must be positive"
    console.print(f"Generating LLM specs -> {output}")
    asyncio.run(generate_all_specs(output, limit=limit, concurrency=concurrency))
    console.print(f"[green]Done[/green] — {output}")


@app.command()
def prepare(
    output: Path = typer.Option(
        DATA_DIR / "instances.jsonl", help="Output instances file"
    ),
    llm_specs: Path = typer.Option(
        DATA_DIR / "llm_specs.jsonl", help="Generated LLM specs"
    ),
    limit: int | None = typer.Option(None, help="Limit number of instances"),
) -> None:
    """Prepare all three condition instances and write to JSONL."""
    assert output is not None, "output path must not be None"
    assert llm_specs is not None, "llm_specs path must not be None"
    specs = load_llm_specs(llm_specs) if llm_specs.exists() else None
    if specs is None:
        console.print(
            "[yellow]No LLM specs found — skipping TESTS_PLUS_LLM_SPEC. "
            "Run generate-specs first.[/yellow]"
        )

    all_instances = []
    for condition in Condition:
        if condition == Condition.TESTS_PLUS_LLM_SPEC and specs is None:
            continue
        instances = load_instances(condition, llm_specs=specs, limit=limit)
        all_instances.extend(instances)
        console.print(f"  {condition}: {len(instances)} instances")

    write_instances(all_instances, output)
    console.print(f"[green]Wrote {len(all_instances)} instances -> {output}[/green]")


@app.command()
def run(
    instances: Path = typer.Option(
        DATA_DIR / "instances.jsonl", help="Instances file"
    ),
    output_dir: Path = typer.Option(RESULTS_DIR / "preds", help="Output dir for .pred files"),
    mini_swe_agent: Path = typer.Option(
        Path("SWE-bench_Pro-os/mini-swe-agent"),
        help="Path to mini-swe-agent directory",
    ),
    model: str = typer.Option("claude-sonnet-4-6", help="Model to use"),
    conditions: list[Condition] = typer.Option(
        list(Condition), help="Conditions to run"
    ),
    max_workers: int = typer.Option(4, help="Parallel workers"),
) -> None:
    """Run the agent on all instances for each condition."""
    assert max_workers > 0, "max_workers must be positive"
    assert instances is not None, "instances path must not be None"
    for condition in conditions:
        pred_dir = run_condition(
            instances,
            output_dir,
            condition,
            mini_swe_agent_dir=mini_swe_agent,
            model=model,
            max_workers=max_workers,
        )
        patches = gather_patches(pred_dir, condition)
        patches_path = output_dir / f"{condition}_patches.json"
        write_patches_json(patches, patches_path)
        console.print(
            f"[green]{condition}[/green]: {len(patches)} patches -> {patches_path}"
        )


@app.command()
def evaluate(
    swe_bench_pro_dir: Path = typer.Argument(
        help="Path to SWE-bench_Pro-os checkout"
    ),
    preds_dir: Path = typer.Option(RESULTS_DIR / "preds", help="Predictions dir"),
    output_dir: Path = typer.Option(RESULTS_DIR / "eval", help="Evaluation output"),
    num_workers: int = typer.Option(8, help="Docker workers"),
    dockerhub_username: str = typer.Option("", help="Docker Hub username"),
    conditions: list[Condition] = typer.Option(
        list(Condition), help="Conditions to evaluate"
    ),
) -> None:
    """Print the swe_bench_pro_eval.py commands to run for each condition."""
    assert dockerhub_username is not None, "dockerhub_username must not be None"
    assert swe_bench_pro_dir is not None, "swe_bench_pro_dir must not be None"
    console.print(
        "[bold]Run the following commands to evaluate each condition:[/bold]\n"
    )
    dataset_csv = swe_bench_pro_dir / "swe_bench_pro_full.csv"
    eval_script = swe_bench_pro_dir / "swe_bench_pro_eval.py"

    for condition in conditions:
        patches_path = preds_dir / f"{condition}_patches.json"
        cond_output = output_dir / condition
        console.print(
            f"python {eval_script} \\\n"
            f"  --raw_sample_path={dataset_csv} \\\n"
            f"  --patch_path={patches_path} \\\n"
            f"  --output_dir={cond_output} \\\n"
            f"  --scripts_dir={swe_bench_pro_dir}/run_scripts \\\n"
            f"  --num_workers={num_workers} \\\n"
            f"  --dockerhub_username={dockerhub_username}\n"
        )


@app.command()
def analyse(
    results_dir: Path = typer.Option(
        RESULTS_DIR / "eval", help="Evaluation results dir"
    ),
    breakdown: bool = typer.Option(False, help="Show per-repo breakdown"),
    costs: bool = typer.Option(False, help="Show cost analysis"),
) -> None:
    """Analyse evaluation results and print pass rate comparison."""
    assert results_dir is not None, "results_dir must not be None"
    assert isinstance(breakdown, bool), "breakdown must be a bool"
    db = load_results(results_dir)
    print_summary(db)
    if breakdown:
        per_repo_breakdown(db)
    if costs:
        cost_analysis(db)
