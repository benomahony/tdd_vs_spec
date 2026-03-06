import csv
import pytest
from datasets import Dataset

from tdd_vs_spec.eval_data import ensure_swe_bench_pro_raw_csv


@pytest.mark.unit
def test_ensure_swe_bench_pro_raw_csv_skips_if_exists(tmp_path):
    path = tmp_path / "swe_bench_pro_full.csv"
    path.write_text("existing")
    ensure_swe_bench_pro_raw_csv(path)
    assert path.read_text() == "existing"


@pytest.mark.unit
def test_ensure_swe_bench_pro_raw_csv_creates_csv_with_lowercase_columns(tmp_path):
    path = tmp_path / "swe_bench_pro_full.csv"
    ds = Dataset.from_dict(
        {
            "instance_id": ["id1"],
            "FAIL_TO_PASS": ["['t1', 't2']"],
            "repo": ["org/repo"],
        }
    )
    ensure_swe_bench_pro_raw_csv(path, _dataset=ds)
    assert path.exists()
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["instance_id"] == "id1"
    assert rows[0]["fail_to_pass"] == "['t1', 't2']"
    assert rows[0]["repo"] == "org/repo"


@pytest.mark.unit
def test_ensure_swe_bench_pro_raw_csv_serializes_list_for_eval(tmp_path):
    path = tmp_path / "out.csv"
    # Dataset may store list columns as actual lists
    ds = Dataset.from_dict(
        {
            "instance_id": ["id1"],
            "fail_to_pass": [["t1", "t2"]],
            "pass_to_pass": [["p1"]],
        }
    )
    ensure_swe_bench_pro_raw_csv(path, _dataset=ds)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["fail_to_pass"] == "['t1', 't2']"
    assert rows[0]["pass_to_pass"] == "['p1']"
    # Eval script should be able to eval these
    assert eval(rows[0]["fail_to_pass"]) == ["t1", "t2"]
    assert eval(rows[0]["pass_to_pass"]) == ["p1"]
