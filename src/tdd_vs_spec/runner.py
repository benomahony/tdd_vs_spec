import json
import logging
import subprocess  # nosec B404
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import cast

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from ._console import console
from .conditions import Condition, Instance, read_instances

logger = logging.getLogger(__name__)


def _write_instances_json(instances: list[Instance], path: Path) -> None:
    assert instances is not None, "instances must not be None"
    assert path is not None, "path must not be None"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "instance_id": i.instance_id,
            "image_name": i.dockerhub_tag,
            "problem_statement": i.problem_statement,
            "test_patch": i.test_patch,
        }
        for i in instances
    ]
    _ = path.write_text(json.dumps(data, indent=2))


def run_condition(
    instances_path: Path,
    output_dir: Path,
    condition: Condition,
    mini_swe_agent_dir: Path,
    model: str = "claude-sonnet-4-6",
    max_workers: int = 4,
    timeout: int = 3600,
) -> Path:
    assert max_workers > 0, "max_workers must be positive"
    assert instances_path.exists(), f"instances file not found: {instances_path}"
    instances = [i for i in read_instances(instances_path) if i.condition == condition]
    pred_dir = output_dir / condition
    pred_dir.mkdir(parents=True, exist_ok=True)

    existing = {p.stem for p in pred_dir.glob("*.pred")}
    pending = [i for i in instances if i.instance_id not in existing]
    _write_instances_json(pending, pred_dir / "batch_instances.json")

    console.print(
        f"[bold]{condition}[/bold]: {len(pending)} pending / {len(instances)} total"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Running {condition}", total=len(pending))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_single,
                    instance,
                    pred_dir / f"{instance.instance_id}.pred",
                    mini_swe_agent_dir,
                    model,
                    timeout,
                ): instance
                for instance in pending
            }
            for future in as_completed(futures):
                instance = futures[future]
                try:
                    future.result()
                except Exception as e:
                    console.print(f"[red]Error {instance.instance_id}[/red]: {e}")
                progress.advance(task)

    return pred_dir


def _run_single(
    instance: Instance,
    pred_file: Path,
    mini_swe_agent_dir: Path,
    model: str,
    timeout: int,
) -> None:
    assert instance is not None, "instance must not be None"
    assert mini_swe_agent_dir.exists(), "mini_swe_agent_dir must exist"
    assert timeout > 0, "timeout must be positive"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(
            {
                "instance_id": instance.instance_id,
                "problem_statement": instance.problem_statement,
                "repo": instance.repo,
                "base_commit": instance.base_commit,
                "dockerhub_tag": instance.dockerhub_tag,
            },
            f,
        )
        task_file = Path(f.name)

    try:
        result = subprocess.run(  # nosec B603
            [
                sys.executable,
                "run.py",
                "--task-file",
                str(task_file),
                "--model",
                model,
                "--output",
                str(pred_file),
            ],
            cwd=mini_swe_agent_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            console.print(
                f"[red]Failed {instance.instance_id}[/red]: {result.stderr[-500:]}"
            )
            _ = pred_file.write_text(
                json.dumps(
                    {
                        "instance_id": instance.instance_id,
                        "patch": "",
                        "error": result.stderr[-500:],
                    }
                )
            )
    finally:
        task_file.unlink(missing_ok=True)


def gather_patches(pred_dir: Path, condition: Condition) -> list[dict[str, str]]:
    assert pred_dir is not None, "pred_dir must not be None"
    assert condition is not None, "condition must not be None"
    patches: list[dict[str, str]] = []
    for pred_file in pred_dir.glob("*.pred"):
        data = cast(dict[str, str], json.loads(pred_file.read_text()))
        patches.append(
            {
                "instance_id": data.get("instance_id", pred_file.stem),
                "patch": data.get("patch", ""),
                "prefix": condition,
            }
        )
    return patches


def write_patches_json(patches: list[dict[str, str]], path: Path) -> None:
    assert patches is not None, "patches must not be None"
    assert path is not None, "path must not be None"
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(patches, indent=2))
