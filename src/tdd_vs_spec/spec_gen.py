import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import NamedTuple, cast

from pydantic import BaseModel
from pydantic_ai import Agent

logger = logging.getLogger(__name__)


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


def _default_agent(
    model: str = "anthropic:claude-sonnet-4-6",
) -> Agent[None, GeneratedSpec]:
    assert model is not None, "model must not be None"
    assert len(model) > 0, "model must not be empty"
    agent = Agent(
        model,
        output_type=GeneratedSpec,
        instructions=SPEC_SYSTEM,
    )
    assert agent is not None, "agent must not be None"
    assert agent.output_type is not None, "agent must have output_type"
    return agent


async def generate_spec(
    test_patch: str,
    agent: Agent[None, GeneratedSpec] | None = None,
    model: str = "anthropic:claude-sonnet-4-6",
) -> SpecResult:
    assert test_patch is not None, "test_patch must not be None"
    assert len(test_patch) > 0, "test_patch must not be empty"
    if agent is None:
        agent = _default_agent(model)
    prompt = f"Generate a spec for the following failing tests:\n\n{test_patch}"
    result = await agent.run(prompt)
    usage = result.usage()
    return SpecResult(
        spec=result.output.spec,
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
    )


def _load_dataset_rows(
    dataset: Iterable[dict[str, str]] | None, limit: int | None
) -> list[dict[str, str]]:
    assert limit is None or limit > 0, "limit must be positive"
    if dataset is None:
        from datasets import load_dataset

        ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")  # nosec B615
        rows = cast(
            list[dict[str, str]], list(ds.select(range(limit)) if limit else ds)
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
                        done.add(cast(dict[str, str], json.loads(line))["instance_id"])
                    except (json.JSONDecodeError, KeyError) as exc:
                        logger.warning(
                            "Skipping corrupted line in %s: %s", output_path, exc
                        )
    return done


def _append_results(
    output_path: Path, results: list[dict[str, str | int] | None]
) -> None:
    assert output_path is not None, "output_path must not be None"
    assert results is not None, "results must not be None"
    with output_path.open("a") as f:
        if output_path.stat().st_size > 0:
            with output_path.open("rb") as rb:
                _ = rb.seek(-1, 2)
                if rb.read(1) != b"\n":
                    _ = f.write("\n")
        for item in results:
            if item is not None:
                _ = f.write(json.dumps(item) + "\n")


async def generate_all_specs(
    output_path: Path,
    limit: int | None = None,
    concurrency: int = 10,
    max_retries: int = 3,
    model: str = "anthropic:claude-sonnet-4-6",
    *,
    _generate: Callable[..., Awaitable[SpecResult]] | None = None,
    _dataset: Iterable[dict[str, str]] | None = None,
) -> None:
    assert output_path is not None, "output_path must not be None"
    assert concurrency > 0, "concurrency must be positive"

    rows = _load_dataset_rows(_dataset, limit)
    pending = [r for r in rows if r["instance_id"] not in _read_done_ids(output_path)]
    semaphore = asyncio.Semaphore(concurrency)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async def process(row: dict[str, str]) -> None:
        assert row is not None, "row must not be None"
        assert "test_patch" in row, "row must contain test_patch"
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    if _generate is not None:
                        result = await _generate(row["test_patch"])
                    else:
                        result = await generate_spec(row["test_patch"], model=model)
                    sr = result
                    item = {
                        "instance_id": row["instance_id"],
                        "spec": sr.spec,
                        "input_tokens": sr.input_tokens,
                        "output_tokens": sr.output_tokens,
                    }
                    _append_results(output_path, [item])
                    return
                except Exception as exc:
                    if attempt == max_retries - 1:
                        logger.warning(
                            "Giving up on %s after %d attempts: %s",
                            row["instance_id"],
                            max_retries,
                            exc,
                        )

    _ = await asyncio.gather(*[process(row) for row in pending])
