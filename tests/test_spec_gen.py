import asyncio
import json
from pathlib import Path

import pytest

from tdd_vs_spec.spec_gen import generate_all_specs


def _fake_rows(n: int = 10) -> list[dict]:
    return [
        {"instance_id": f"org__repo__{i}", "test_patch": f"def test_{i}(): pass"}
        for i in range(n)
    ]


async def _fast_generate(test_patch: str) -> str:
    await asyncio.sleep(0.01)
    return f"spec: {test_patch}"


async def test_generate_all_specs_writes_all_rows(tmp_path):
    out = tmp_path / "specs.jsonl"
    await generate_all_specs(out, _generate=_fast_generate, _dataset=_fake_rows(10))
    lines = [json.loads(l) for l in out.read_text().strip().splitlines()]
    assert len(lines) == 10, "must write one line per input row"
    assert all("instance_id" in l and "spec" in l for l in lines), "each line needs instance_id and spec"


async def test_generate_all_specs_resumes_without_duplicates(tmp_path):
    out = tmp_path / "specs.jsonl"
    rows = _fake_rows(10)
    first_five = rows[:5]
    with out.open("w") as f:
        for row in first_five:
            f.write(json.dumps({"instance_id": row["instance_id"], "spec": "existing"}) + "\n")

    await generate_all_specs(out, _generate=_fast_generate, _dataset=rows)
    lines = [json.loads(l) for l in out.read_text().strip().splitlines()]
    ids = [l["instance_id"] for l in lines]
    assert len(ids) == len(set(ids)), "must not generate duplicates on resume"
    assert len(lines) == 10, "must have all 10 rows after resume"


async def test_generate_all_specs_concurrent_output_valid(tmp_path):
    out = tmp_path / "specs.jsonl"
    await generate_all_specs(
        out, _generate=_fast_generate, _dataset=_fake_rows(20), concurrency=5
    )
    lines = [json.loads(l) for l in out.read_text().strip().splitlines()]
    assert len(lines) == 20, "all 20 rows must be written even under concurrency"
    assert all(json.dumps(l) for l in lines), "all lines must be valid JSON"


def test_generate_all_specs_rejects_zero_concurrency(tmp_path):
    with pytest.raises(AssertionError, match="concurrency must be positive"):
        asyncio.run(
            generate_all_specs(tmp_path / "out.jsonl", _dataset=[], concurrency=0)
        )
