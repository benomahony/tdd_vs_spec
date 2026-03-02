from pathlib import Path

import duckdb
from rich.console import Console
from rich.table import Table

from .conditions import Condition

console = Console()


def load_results(results_dir: Path) -> duckdb.DuckDBPyConnection:
    assert results_dir is not None, "results_dir must not be None"
    assert results_dir.is_dir(), f"results_dir does not exist: {results_dir}"

    result_files = list(results_dir.glob("**/*.json"))
    if not result_files:
        raise FileNotFoundError(f"No result JSON files found in {results_dir}")

    db = duckdb.connect()
    db.execute("""
        CREATE TABLE results AS
        SELECT * FROM read_json_auto($files, union_by_name=true)
    """, {"files": [str(f) for f in result_files]})

    return db


def pass_rates(db: duckdb.DuckDBPyConnection) -> dict[str, float]:
    assert db is not None, "db must not be None"

    rows = db.execute("""
        SELECT
            prefix AS condition,
            COUNT(*) AS total,
            SUM(CASE WHEN resolved THEN 1 ELSE 0 END) AS passed,
            ROUND(
                100.0 * SUM(CASE WHEN resolved THEN 1 ELSE 0 END) / COUNT(*),
                1
            ) AS pass_rate
        FROM results
        GROUP BY prefix
        ORDER BY prefix
    """).fetchall()

    assert rows is not None, "query must return rows"
    return {row[0]: row[3] for row in rows}


def print_summary(db: duckdb.DuckDBPyConnection) -> None:
    assert db is not None, "db must not be None"

    rates = pass_rates(db)
    rows = db.execute("""
        SELECT
            prefix AS condition,
            COUNT(*) AS total,
            SUM(CASE WHEN resolved THEN 1 ELSE 0 END) AS passed
        FROM results
        GROUP BY prefix
        ORDER BY prefix
    """).fetchall()

    table = Table(title="Pass rates by condition")
    table.add_column("Condition", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Passed", justify="right")
    table.add_column("Pass rate (%)", justify="right", style="bold")

    condition_labels = {
        Condition.TESTS_ONLY: "Tests only",
        Condition.TESTS_PLUS_HUMAN_SPEC: "Tests + human spec",
        Condition.TESTS_PLUS_LLM_SPEC: "Tests + LLM spec",
    }

    for row in rows:
        label = condition_labels.get(row[0], row[0])
        table.add_row(label, str(row[1]), str(row[2]), str(rates.get(row[0], 0.0)))

    console.print(table)
    _print_delta(rates)


def _print_delta(rates: dict[str, float]) -> None:
    assert rates is not None, "rates must not be None"

    baseline = rates.get(Condition.TESTS_ONLY)
    if baseline is None:
        return

    console.print("\n[bold]Delta vs tests-only baseline[/bold]")
    for condition, rate in rates.items():
        if condition == Condition.TESTS_ONLY:
            continue
        delta = rate - baseline
        colour = "green" if delta > 0 else "red"
        console.print(
            f"  {condition}: [{colour}]{delta:+.1f}pp[/{colour}]"
        )


def per_repo_breakdown(db: duckdb.DuckDBPyConnection) -> None:
    assert db is not None, "db must not be None"

    rows = db.execute("""
        SELECT
            split_part(instance_id, '__', 1) AS repo,
            prefix AS condition,
            COUNT(*) AS total,
            ROUND(
                100.0 * SUM(CASE WHEN resolved THEN 1 ELSE 0 END) / COUNT(*),
                1
            ) AS pass_rate
        FROM results
        GROUP BY repo, prefix
        ORDER BY repo, prefix
    """).fetchall()

    table = Table(title="Per-repo breakdown")
    table.add_column("Repo", style="cyan")
    table.add_column("Condition")
    table.add_column("N", justify="right")
    table.add_column("Pass rate (%)", justify="right")

    for row in rows:
        table.add_row(row[0], row[1], str(row[2]), str(row[3]))

    console.print(table)


def cost_analysis(db: duckdb.DuckDBPyConnection) -> None:
    assert db is not None, "db must not be None"

    try:
        rows = db.execute("""
            SELECT
                prefix AS condition,
                ROUND(AVG(total_cost), 4) AS avg_cost,
                ROUND(AVG(steps), 1) AS avg_steps
            FROM results
            WHERE total_cost IS NOT NULL
            GROUP BY prefix
            ORDER BY prefix
        """).fetchall()
    except duckdb.Error:
        console.print("[yellow]No cost/steps data available in results[/yellow]")
        return

    table = Table(title="Cost and steps by condition")
    table.add_column("Condition", style="cyan")
    table.add_column("Avg cost (USD)", justify="right")
    table.add_column("Avg steps", justify="right")

    for row in rows:
        table.add_row(row[0], f"${row[1]:.4f}", str(row[2]))

    console.print(table)
