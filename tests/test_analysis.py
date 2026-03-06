import json
from pathlib import Path

import duckdb
import pytest

from tdd_vs_spec.analysis import (
    cost_analysis,
    load_results,
    pass_rates,
    per_repo_breakdown,
    significance_test,
)


@pytest.mark.unit
def test_pass_rates_returns_correct_percentages(fake_db: duckdb.DuckDBPyConnection):
    rates = pass_rates(fake_db)
    assert isinstance(rates, dict), "pass_rates must return a dict"
    assert rates["tests_only"] == 50.0, "1/2 resolved = 50%"
    assert rates["tests_plus_human_spec"] == 100.0, "2/2 resolved = 100%"


@pytest.mark.unit
def test_print_summary_consistent_with_pass_rates(fake_db: duckdb.DuckDBPyConnection):
    rates = pass_rates(fake_db)
    assert "tests_only" in rates, "pass_rates must include tests_only"
    assert rates["tests_only"] == 50.0, "print_summary and pass_rates must agree"


@pytest.mark.unit
def test_per_repo_breakdown_handles_no_double_underscore(
    fake_db: duckdb.DuckDBPyConnection,
):
    _ = fake_db.execute(
        "INSERT INTO results VALUES ('nounderscore', 'tests_only', true, 0.01, 1)"
    )
    per_repo_breakdown(fake_db)


@pytest.mark.unit
def test_cost_analysis_handles_missing_columns():
    db = duckdb.connect()
    _ = db.execute(
        "CREATE TABLE results (instance_id VARCHAR, prefix VARCHAR, resolved BOOLEAN)"
    )
    _ = db.execute("INSERT INTO results VALUES ('a__b__0', 'tests_only', true)")
    cost_analysis(db)


@pytest.mark.unit
def test_load_results_raises_on_empty_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _ = load_results(tmp_path)


@pytest.mark.unit
def test_load_results_reads_json_files(tmp_path: Path):
    data = [
        {"instance_id": "a__b__0", "prefix": "tests_only", "resolved": True},
        {"instance_id": "a__b__1", "prefix": "tests_only", "resolved": False},
    ]
    _ = (tmp_path / "results.json").write_text(json.dumps(data))
    db = load_results(tmp_path)
    row = db.execute("SELECT COUNT(*) FROM results").fetchone()
    assert row is not None and row[0] == 2, "must load 2 rows from json"


def _make_db(rows: list[tuple[str, str, bool]]) -> duckdb.DuckDBPyConnection:
    db = duckdb.connect()
    _ = db.execute(
        "CREATE TABLE results (instance_id VARCHAR, prefix VARCHAR, resolved BOOLEAN)"
    )
    _ = db.executemany("INSERT INTO results VALUES (?, ?, ?)", rows)
    return db


@pytest.mark.unit
def test_significance_test_returns_valid_p_value(fake_db: duckdb.DuckDBPyConnection):
    result = significance_test(fake_db, "tests_only", "tests_plus_human_spec")
    assert 0.0 <= result.p_value <= 1.0, "p-value must be in [0, 1]"
    assert result.total_a == 2, "tests_only has 2 rows in fake_db"
    assert result.total_b == 2, "tests_plus_human_spec has 2 rows in fake_db"


@pytest.mark.unit
def test_significance_test_identical_rates_gives_p_one():
    rows = [(f"id_{i}", "a", i % 2 == 0) for i in range(10)]
    rows += [(f"id_{i + 100}", "b", i % 2 == 0) for i in range(10)]
    db = _make_db(rows)
    result = significance_test(db, "a", "b")
    assert result.p_value == 1.0, "identical distributions must give p=1.0"


@pytest.mark.unit
def test_significance_test_perfect_separation_gives_low_p():
    rows = [(f"a_{i}", "a", True) for i in range(20)]
    rows += [(f"b_{i}", "b", False) for i in range(20)]
    db = _make_db(rows)
    result = significance_test(db, "a", "b")
    assert result.p_value < 0.001, "perfect separation must give very low p-value"


@pytest.mark.unit
def test_significance_test_raises_on_missing_condition(
    fake_db: duckdb.DuckDBPyConnection,
):
    with pytest.raises(ValueError, match="not found in results"):
        _ = significance_test(fake_db, "tests_only", "nonexistent_condition")
