import asyncio
import json
import logging
from pathlib import Path
from typing import cast

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from tdd_vs_spec.spec_gen import (
    GeneratedSpec,
    SpecResult,
    read_done_ids,
    generate_all_specs,
    generate_spec,
)


def _fake_rows(n: int = 10) -> list[dict[str, str]]:
    return [
        {"instance_id": f"org__repo__{i}", "test_patch": f"def test_{i}(): pass"}
        for i in range(n)
    ]


async def _fast_generate(test_patch: str) -> SpecResult:
    await asyncio.sleep(0.01)
    return SpecResult(spec=f"spec: {test_patch}", input_tokens=5, output_tokens=3)


@pytest.mark.unit
async def test_generate_all_specs_writes_all_rows(tmp_path: Path):
    out = tmp_path / "specs.jsonl"
    await generate_all_specs(out, _generate=_fast_generate, _dataset=_fake_rows(10))
    lines = [
        cast(dict[str, str], json.loads(ln))
        for ln in out.read_text().strip().splitlines()
    ]
    assert len(lines) == 10, "must write one line per input row"
    assert all("instance_id" in ln and "spec" in ln for ln in lines), (
        "each line needs instance_id and spec"
    )
    assert all("input_tokens" in ln and "output_tokens" in ln for ln in lines), (
        "each line must include token counts"
    )


@pytest.mark.unit
async def test_generate_all_specs_resumes_without_duplicates(tmp_path: Path):
    out = tmp_path / "specs.jsonl"
    rows = _fake_rows(10)
    first_five = rows[:5]
    with out.open("w") as f:
        for row in first_five:
            _ = f.write(
                json.dumps({"instance_id": row["instance_id"], "spec": "existing"})
                + "\n"
            )

    await generate_all_specs(out, _generate=_fast_generate, _dataset=rows)
    lines = [
        cast(dict[str, str], json.loads(ln))
        for ln in out.read_text().strip().splitlines()
    ]
    ids = [ln["instance_id"] for ln in lines]
    assert len(ids) == len(set(ids)), "must not generate duplicates on resume"
    assert len(lines) == 10, "must have all 10 rows after resume"


@pytest.mark.unit
async def test_generate_all_specs_concurrent_output_valid(tmp_path: Path):
    out = tmp_path / "specs.jsonl"
    await generate_all_specs(
        out, _generate=_fast_generate, _dataset=_fake_rows(20), concurrency=5
    )
    lines = [
        cast(dict[str, str], json.loads(ln))
        for ln in out.read_text().strip().splitlines()
    ]
    assert len(lines) == 20, "all 20 rows must be written even under concurrency"
    assert all(json.dumps(ln) for ln in lines), "all lines must be valid JSON"


@pytest.mark.unit
async def test_generate_spec_raises_on_none_patch():
    with pytest.raises(AssertionError, match="must not be None"):
        _ = await generate_spec(cast(str, cast(object, None)))


@pytest.mark.unit
async def test_generate_spec_raises_on_empty_patch():
    with pytest.raises(AssertionError, match="must not be empty"):
        _ = await generate_spec("")


@pytest.mark.unit
async def test_generate_spec_with_test_model():
    agent = Agent(
        TestModel(custom_output_args={"spec": "the code must do X"}),
        output_type=GeneratedSpec,
    )
    result = await generate_spec("def test_foo(): assert foo() == 1", agent=agent)
    assert isinstance(result, SpecResult), "must return SpecResult"
    assert isinstance(result.spec, str) and len(result.spec) > 0, (
        "spec must be non-empty string"
    )
    assert result.input_tokens >= 0 and result.output_tokens >= 0, (
        "token counts must be non-negative"
    )


@pytest.mark.unit
def test_generate_all_specs_rejects_zero_concurrency(tmp_path: Path):
    with pytest.raises(AssertionError, match="concurrency must be positive"):
        asyncio.run(
            generate_all_specs(tmp_path / "out.jsonl", _dataset=[], concurrency=0)
        )


@pytest.mark.unit
async def test_generate_all_specs_warns_on_permanent_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    async def always_fails(_: str) -> SpecResult:
        raise RuntimeError("permanent failure")

    out = tmp_path / "specs.jsonl"
    rows = [{"instance_id": "org__repo__0", "test_patch": "patch_0"}]
    with caplog.at_level(logging.WARNING, logger="tdd_vs_spec.spec_gen"):
        await generate_all_specs(
            out, _generate=always_fails, _dataset=rows, max_retries=2
        )
    assert any("org__repo__0" in r.message for r in caplog.records), (
        "must log warning with instance_id when all retries exhausted"
    )


@pytest.mark.unit
def test_read_done_ids_warns_on_corrupted_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    path = tmp_path / "specs.jsonl"
    _ = path.write_text('{"instance_id": "good"}\n{"bad":\n')
    with caplog.at_level(logging.WARNING, logger="tdd_vs_spec.spec_gen"):
        done = read_done_ids(path)
    assert "good" in done, "valid line must still be parsed"
    assert any(caplog.records), "must log at least one warning for corrupted line"


@pytest.mark.unit
async def test_generate_all_specs_gives_up_after_max_retries(tmp_path: Path):
    async def always_fails(_: str) -> SpecResult:
        raise RuntimeError("permanent failure")

    out = tmp_path / "specs.jsonl"
    rows = [{"instance_id": "org__repo__0", "test_patch": "patch_0"}]
    await generate_all_specs(out, _generate=always_fails, _dataset=rows, max_retries=2)
    assert not out.exists() or out.read_text().strip() == "", (
        "nothing written for permanent failure"
    )


@pytest.mark.unit
async def test_generate_all_specs_retries_transient_failure(tmp_path: Path):
    call_counts: dict[str, int] = {}

    async def flaky(test_patch: str) -> SpecResult:
        call_counts[test_patch] = call_counts.get(test_patch, 0) + 1
        if call_counts[test_patch] == 1:
            raise RuntimeError("transient network error")
        return SpecResult(spec=f"spec: {test_patch}", input_tokens=5, output_tokens=3)

    out = tmp_path / "specs.jsonl"
    rows = [{"instance_id": "org__repo__0", "test_patch": "patch_0"}]
    await generate_all_specs(out, _generate=flaky, _dataset=rows)

    lines = [
        cast(dict[str, str], json.loads(ln))
        for ln in out.read_text().strip().splitlines()
    ]
    assert len(lines) == 1, "must succeed after retry"
    assert call_counts["patch_0"] == 2, "must have retried once"


@pytest.mark.unit
async def test_generate_all_specs_accepts_model_param(tmp_path: Path):
    out = tmp_path / "specs.jsonl"
    await generate_all_specs(
        out,
        model="anthropic:claude-opus-4-6",
        _generate=_fast_generate,
        _dataset=_fake_rows(2),
    )
    lines = [
        cast(dict[str, str], json.loads(ln))
        for ln in out.read_text().strip().splitlines()
    ]
    assert len(lines) == 2, "must write 2 rows when model param is passed"


@pytest.mark.unit
async def test_generate_spec_accepts_model_kwarg():
    agent = Agent(
        TestModel(custom_output_args={"spec": "the code must do X"}),
        output_type=GeneratedSpec,
    )
    result = await generate_spec(
        "def test_foo(): pass", agent=agent, model="anthropic:claude-opus-4-6"
    )
    assert isinstance(result, SpecResult), (
        "must return SpecResult when model kwarg passed"
    )


@pytest.mark.unit
async def test_generate_all_specs_handles_corrupted_resume_file(tmp_path: Path):
    out = tmp_path / "specs.jsonl"
    _ = out.write_text('{"instance_id": "org__repo__0", "spec": "ok"}\n{"partial":')
    rows = [
        {"instance_id": "org__repo__0", "test_patch": "p0"},
        {"instance_id": "org__repo__1", "test_patch": "p1"},
    ]
    await generate_all_specs(out, _generate=_fast_generate, _dataset=rows)
    valid_lines = [
        cast(dict[str, str], json.loads(ln))
        for ln in out.read_text().strip().splitlines()
        if ln.strip() and not ln.startswith('{"partial"')
    ]
    ids = {ln["instance_id"] for ln in valid_lines}
    assert "org__repo__1" in ids, (
        "must process non-corrupted rows after corrupted resume file"
    )
