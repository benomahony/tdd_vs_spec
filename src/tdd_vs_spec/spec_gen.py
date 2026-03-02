from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, NamedTuple, cast
import json
import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent


SPEC_SYSTEM = """You are a senior software engineer writing an issue description.

Given a set of failing tests, write a concise natural language specification that
describes what behaviour the code should implement — without revealing the
implementation. The spec should read like a GitHub issue: clear, actionable,
and focused on observable behaviour.

Do not reference test function names, test file paths, or any implementation details.
Do not say 'make the tests pass'. Describe the desired behaviour as a user or
developer would experience it."""


class GeneratedSpec(BaseModel):
    spec: str


class SpecResult(NamedTuple):
    spec: str
    input_tokens: int
    output_tokens: int


def _default_agent() -> Agent[None, GeneratedSpec]:
    agent = Agent(
        "anthropic:claude-sonnet-4-6",
        output_type=GeneratedSpec,
        instructions=SPEC_SYSTEM,
    )
    assert agent is not None, "agent must not be None"
    assert agent.output_type is not None, "agent must have output_type"
    return agent


async def generate_spec(
    test_patch: str,
    agent: Agent[None, GeneratedSpec] | None = None,
) -> SpecResult:
    assert test_patch is not None, "test_patch must not be None"
    assert len(test_patch) > 0, "test_patch must not be empty"
    if agent is None:
        agent = _default_agent()
    prompt = f"Generate a spec for the following failing tests:\n\n{test_patch}"
    result = await agent.run(prompt)
    usage = result.usage()
    return SpecResult(
        spec=result.output.spec,
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
    )


def _load_dataset_rows(
    dataset: Iterable[dict[str, Any]] | None, limit: int | None
) -> list[dict[str, Any]]:
    assert limit is None or limit > 0, "limit must be positive"
    if dataset is None:
        from datasets import load_dataset

        ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
        rows = cast(
            list[dict[str, Any]], list(ds.select(range(limit)) if limit else ds)
        )
    else:
        rows = list(dataset)
    assert isinstance(rows, list), "rows must be a list"
    return rows[:limit] if limit else rows


def _read_done_ids(output_path: Path) -> set[str]:
    assert output_path is not None, "output_path must not be None"
    assert isinstance(output_path, Path), "output_path must be a Path"
    done: set[str] = set()
    if output_path.exists():
        with output_path.open() as f:
            for line in f:
                if line.strip():
                    try:
                        done.add(json.loads(line)["instance_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


async def generate_all_specs(
    output_path: Path,
    limit: int | None = None,
    concurrency: int = 10,
    max_retries: int = 3,
    *,
    _generate: Callable[..., Any] = generate_spec,
    _dataset: Iterable[dict[str, Any]] | None = None,
) -> None:
    assert output_path is not None, "output_path must not be None"
    assert concurrency > 0, "concurrency must be positive"

    rows = _load_dataset_rows(_dataset, limit)
    pending = [r for r in rows if r["instance_id"] not in _read_done_ids(output_path)]
    semaphore = asyncio.Semaphore(concurrency)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async def process(row: dict[str, Any]) -> dict[str, Any] | None:
        assert row is not None, "row must not be None"
        assert "test_patch" in row, "row must contain test_patch"
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    result = await _generate(row["test_patch"])
                    sr = (
                        result
                        if isinstance(result, SpecResult)
                        else SpecResult(result, 0, 0)
                    )
                    return {
                        "instance_id": row["instance_id"],
                        "spec": sr.spec,
                        "input_tokens": sr.input_tokens,
                        "output_tokens": sr.output_tokens,
                    }
                except Exception:
                    if attempt == max_retries - 1:
                        return None
            return None

    results = await asyncio.gather(*[process(row) for row in pending])

    with output_path.open("a") as f:
        if output_path.stat().st_size > 0:
            with output_path.open("rb") as rb:
                rb.seek(-1, 2)
                if rb.read(1) != b"\n":
                    f.write("\n")
        for item in results:
            if item is not None:
                f.write(json.dumps(item) + "\n")
