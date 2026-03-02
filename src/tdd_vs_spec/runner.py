import json
import subprocess
import sys
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .conditions import Condition, Instance, read_instances

console = Console()


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
    instances = [
        i for i in read_instances(instances_path) if i.condition == condition
    ]
    pred_dir = output_dir / condition
    pred_dir.mkdir(parents=True, exist_ok=True)

    existing = {p.stem for p in pred_dir.glob("*.pred")}
    pending = [i for i in instances if i.instance_id not in existing]

    console.print(
        f"[bold]{condition}[/bold]: {len(pending)} pending / "
        f"{len(instances)} total"
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
    assert timeout > 0, "timeout must be positive"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
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
        result = subprocess.run(
            [
                sys.executable,
                "run.py",
                "--task-file", str(task_file),
                "--model", model,
                "--output", str(pred_file),
            ],
            cwd=mini_swe_agent_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            console.print(
                f"[red]Failed {instance.instance_id}[/red]: "
                f"{result.stderr[-500:]}"
            )
            pred_file.write_text(
                json.dumps({
                    "instance_id": instance.instance_id,
                    "patch": "",
                    "error": result.stderr[-500:],
                })
            )
    finally:
        task_file.unlink(missing_ok=True)


def gather_patches(pred_dir: Path, condition: Condition) -> list[dict]:
    assert pred_dir is not None, "pred_dir must not be None"
    assert condition is not None, "condition must not be None"
    patches = []
    for pred_file in pred_dir.glob("*.pred"):
        data = json.loads(pred_file.read_text())
        patches.append(
            {
                "instance_id": data.get("instance_id", pred_file.stem),
                "patch": data.get("patch", ""),
                "prefix": condition,
            }
        )
    return patches


def write_patches_json(patches: list[dict], path: Path) -> None:
    assert patches is not None, "patches must not be None"
    assert path is not None, "path must not be None"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(patches, indent=2))
