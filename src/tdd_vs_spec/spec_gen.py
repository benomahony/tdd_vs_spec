from collections.abc import Callable, Iterable
from pathlib import Path
from typing import NamedTuple
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
    return Agent(
        "anthropic:claude-sonnet-4-6",
        output_type=GeneratedSpec,
        instructions=SPEC_SYSTEM,
    )


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


async def generate_all_specs(
    output_path: Path,
    limit: int | None = None,
    concurrency: int = 10,
    max_retries: int = 3,
    *,
    _generate: Callable[[str], asyncio.Future[str]] | None = None,
    _dataset: Iterable[dict] | None = None,
) -> None:
    assert output_path is not None, "output_path must not be None"
    assert concurrency > 0, "concurrency must be positive"

    if _dataset is None:
        from datasets import load_dataset
        dataset = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
        if limit:
            dataset = dataset.select(range(limit))
        rows: list[dict] = list(dataset)
    else:
        rows = list(_dataset)
        if limit:
            rows = rows[:limit]

    if _generate is None:
        async def _generate(test_patch: str) -> SpecResult:  # type: ignore[misc]
            return await generate_spec(test_patch)

    already_done: set[str] = set()
    if output_path.exists():
        with output_path.open() as f:
            for line in f:
                if line.strip():
                    try:
                        already_done.add(json.loads(line)["instance_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass

    pending = [row for row in rows if row["instance_id"] not in already_done]

    semaphore = asyncio.Semaphore(concurrency)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async def process(row: dict) -> dict | None:
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    result = await _generate(row["test_patch"])
                    item: dict = {"instance_id": row["instance_id"]}
                    if isinstance(result, SpecResult):
                        item["spec"] = result.spec
                        item["input_tokens"] = result.input_tokens
                        item["output_tokens"] = result.output_tokens
                    else:
                        item["spec"] = result
                    return item
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
