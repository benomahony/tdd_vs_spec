"""Ensure eval script input files exist (e.g. raw sample CSV from Hugging Face)."""

import csv
from pathlib import Path
from typing import Any, cast


def _serialize_cell(value: object) -> str:
    """Serialize a cell for CSV so eval() works for list columns (fail_to_pass, etc.)."""
    assert not callable(value), (
        f"callable values cannot be serialized to CSV: {type(value)}"
    )
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    result = repr(value)
    assert isinstance(result, str), "repr must return a string"
    return result


def ensure_swe_bench_pro_raw_csv(
    path: Path,
    *,
    _dataset: Any = None,  # Optional for testing; when set, skip load_dataset
) -> None:
    """Create swe_bench_pro_full.csv from the Hugging Face dataset if it does not exist.

    The eval script expects lowercase column names (e.g. fail_to_pass, pass_to_pass,
    before_repo_set_cmd, selected_test_files_to_run) and list columns as strings
    that eval() can parse.
    """
    assert path is not None, "path must not be None"
    assert isinstance(path, Path), "path must be a Path object"
    if path.exists():
        return
    if _dataset is not None:
        ds = _dataset
    else:
        from datasets import load_dataset

        ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")  # nosec B615
    path.parent.mkdir(parents=True, exist_ok=True)
    column_names = list(ds.column_names)
    # Normalize to lowercase so eval script's sample["fail_to_pass"] etc. work
    fieldnames = [c.lower() for c in column_names]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in ds:
            # Row keys match column_names; normalize to lowercase for writer
            out = {
                k.lower(): _serialize_cell(v)
                for k, v in cast(dict[str, object], row).items()
            }
            writer.writerow(out)
