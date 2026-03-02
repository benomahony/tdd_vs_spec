from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
import json

from pydantic import BaseModel


TESTS_ONLY_PROMPT = (
    "There are failing tests in this repository. Your task is to identify the "
    "failing tests and fix the source code so all tests pass. Do not modify any "
    "test files."
)


class Condition(StrEnum):
    TESTS_ONLY = "tests_only"
    TESTS_PLUS_HUMAN_SPEC = "tests_plus_human_spec"
    TESTS_PLUS_LLM_SPEC = "tests_plus_llm_spec"


class Instance(BaseModel):
    instance_id: str
    condition: Condition
    problem_statement: str
    test_patch: str
    patch: str
    dockerhub_tag: str
    repo: str
    base_commit: str


def load_instances(
    condition: Condition,
    llm_specs: dict[str, str] | None = None,
    limit: int | None = None,
    *,
    dataset: Iterable[dict] | None = None,
) -> list[Instance]:
    assert condition is not None, "condition must not be None"
    if dataset is None:
        from datasets import load_dataset
        dataset = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
        if limit:
            dataset = dataset.select(range(limit))
        rows: list[dict] = list(dataset)
    else:
        rows = list(dataset)
        if limit:
            rows = rows[:limit]
    dataset = rows  # type: ignore[assignment]

    instances = []
    for row in rows:
        match condition:
            case Condition.TESTS_ONLY:
                problem_statement = TESTS_ONLY_PROMPT
            case Condition.TESTS_PLUS_HUMAN_SPEC:
                problem_statement = row["problem_statement"]
            case Condition.TESTS_PLUS_LLM_SPEC:
                if llm_specs is None:
                    raise ValueError("llm_specs required for TESTS_PLUS_LLM_SPEC")
                spec = llm_specs.get(row["instance_id"])
                if spec is None:
                    continue
                problem_statement = spec

        instances.append(
            Instance(
                instance_id=row["instance_id"],
                condition=condition,
                problem_statement=problem_statement,
                test_patch=row["test_patch"],
                patch=row["patch"],
                dockerhub_tag=row["dockerhub_tag"],
                repo=row["repo"],
                base_commit=row["base_commit"],
            )
        )
    return instances


def write_instances(instances: list[Instance], path: Path) -> None:
    assert instances is not None, "instances must not be None"
    assert path is not None, "path must not be None"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for instance in instances:
            f.write(instance.model_dump_json() + "\n")


def read_instances(path: Path) -> list[Instance]:
    assert path is not None, "path must not be None"
    assert path.exists(), f"instances file not found: {path}"
    instances = []
    with path.open() as f:
        for line in f:
            if line.strip():
                instances.append(Instance.model_validate_json(line))
    return instances


def load_llm_specs(path: Path) -> dict[str, str]:
    assert path is not None, "path must not be None"
    assert path.exists(), f"specs file not found: {path}"
    specs: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                specs[row["instance_id"]] = row["spec"]
    return specs
