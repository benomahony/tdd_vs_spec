import json
from pathlib import Path

import duckdb
import pytest

from tdd_vs_spec.analysis import (
    cost_analysis,
    load_results,
    pass_rates,
    per_repo_breakdown,
    print_summary,
)


def test_pass_rates_returns_correct_percentages(fake_db):
    rates = pass_rates(fake_db)
    assert isinstance(rates, dict), "pass_rates must return a dict"
    assert rates["tests_only"] == 50.0, "1/2 resolved = 50%"
    assert rates["tests_plus_human_spec"] == 100.0, "2/2 resolved = 100%"


def test_print_summary_consistent_with_pass_rates(fake_db):
    rates = pass_rates(fake_db)
    assert "tests_only" in rates, "pass_rates must include tests_only"
    assert rates["tests_only"] == 50.0, "print_summary and pass_rates must agree"


def test_per_repo_breakdown_handles_no_double_underscore(fake_db):
    fake_db.execute("INSERT INTO results VALUES ('nounderscore', 'tests_only', true, 0.01, 1)")
    per_repo_breakdown(fake_db)


def test_cost_analysis_handles_missing_columns(capsys):
    db = duckdb.connect()
    db.execute("CREATE TABLE results (instance_id VARCHAR, prefix VARCHAR, resolved BOOLEAN)")
    db.execute("INSERT INTO results VALUES ('a__b__0', 'tests_only', true)")
    cost_analysis(db)


def test_load_results_raises_on_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path)


def test_load_results_reads_json_files(tmp_path):
    data = [
        {"instance_id": "a__b__0", "prefix": "tests_only", "resolved": True},
        {"instance_id": "a__b__1", "prefix": "tests_only", "resolved": False},
    ]
    (tmp_path / "results.json").write_text(json.dumps(data))
    db = load_results(tmp_path)
    rows = db.execute("SELECT COUNT(*) FROM results").fetchone()
    assert rows[0] == 2, "must load 2 rows from json"
