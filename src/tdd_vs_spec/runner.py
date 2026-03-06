import json
import logging
import subprocess  # nosec B404
from pathlib import Path
from typing import cast

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from ._console import console as console
from .conditions import Condition, Instance, read_instances

logger = logging.getLogger(__name__)


def _write_instances_json(
    instances: list[Instance],
    path: Path,
    dockerhub_username: str = "jefzda",
) -> None:
    """Write batch instances JSON. image_name is full Docker URI for SWE-bench Pro (registry/repo:tag)."""
    assert instances is not None, "instances must not be None"
    assert path is not None, "path must not be None"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "instance_id": i.instance_id,
            "image_name": f"{dockerhub_username}/sweap-images:{i.dockerhub_tag}",
            "problem_statement": i.problem_statement,
            "test_patch": i.test_patch,
        }
        for i in instances
    ]
    _ = path.write_text(json.dumps(data, indent=2))


def _load_preds(preds_file: Path) -> dict[str, dict[str, str]]:
    assert preds_file is not None, "preds_file must not be None"
    assert isinstance(preds_file, Path), "preds_file must be a Path"
    if not preds_file.exists():
        return {}
    try:
        return cast(dict[str, dict[str, str]], json.loads(preds_file.read_text()))
    except json.JSONDecodeError:
        logger.warning("Corrupt preds.json in %s, starting fresh", preds_file.parent)
        return {}


def _invoke_mini_extra(
    batch_file: Path,
    pred_dir: Path,
    model: str,
    max_workers: int,
    mini_swe_agent_dir: Path,
    timeout: int,
) -> "subprocess.CompletedProcess[str] | None":
    assert batch_file.exists(), "batch_file must exist"
    assert mini_swe_agent_dir.exists(), "mini_swe_agent_dir must exist"
    try:
        return subprocess.run(  # nosec B603, B607
            [
                "mini-extra",
                "run-batch",
                "--instances-path",
                str(batch_file),
                "--output-dir",
                str(pred_dir),
                "--model",
                model,
                "--num-workers",
                str(max_workers),
                "--no-shuffle",
            ],
            cwd=mini_swe_agent_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None


def _merge_preds(preds_file: Path, existing_preds: dict[str, dict[str, str]]) -> None:
    assert preds_file is not None, "preds_file must not be None"
    assert existing_preds is not None, "existing_preds must not be None"
    if not preds_file.exists() or not existing_preds:
        return
    try:
        new_preds = cast(dict[str, dict[str, str]], json.loads(preds_file.read_text()))
        merged = {**existing_preds, **new_preds}
        _ = preds_file.write_text(json.dumps(merged, indent=2))
    except json.JSONDecodeError:
        logger.warning("Could not merge preds.json for %s", preds_file.parent)


def run_condition(
    instances_path: Path,
    output_dir: Path,
    condition: Condition,
    mini_swe_agent_dir: Path,
    model: str = "claude-sonnet-4-6",
    max_workers: int = 4,
    timeout: int = 3600,
    limit: int | None = None,
    dockerhub_username: str = "jefzda",
) -> Path:
    assert max_workers > 0, "max_workers must be positive"
    assert instances_path.exists(), f"instances file not found: {instances_path}"
    assert mini_swe_agent_dir.exists(), "mini_swe_agent_dir must exist"
    instances = [i for i in read_instances(instances_path) if i.condition == condition]
    instances = instances[:limit]
    pred_dir = output_dir / condition
    pred_dir.mkdir(parents=True, exist_ok=True)

    preds_file = pred_dir / "preds.json"
    existing_preds = _load_preds(preds_file)
    pending = [i for i in instances if i.instance_id not in existing_preds]
    console.print(
        f"[bold]{condition}[/bold]: {len(pending)} pending / {len(instances)} total"
    )

    if not pending:
        return pred_dir

    batch_instances_file = (pred_dir / "batch_instances.json").resolve()
    pred_dir_abs = pred_dir.resolve()
    _write_instances_json(pending, batch_instances_file, dockerhub_username)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        _ = progress.add_task(
            f"Running {condition} ({len(pending)} instances)", total=None
        )
        try:
            from ._minibatch import run_batch_in_process

            run_batch_in_process(
                batch_instances_file,
                pred_dir_abs,
                model,
                max_workers,
            )
        except ImportError:
            result = _invoke_mini_extra(
                batch_instances_file,
                pred_dir_abs,
                model,
                max_workers,
                mini_swe_agent_dir,
                timeout,
            )
            if result is None:
                console.print(f"[yellow]Timeout for {condition}[/yellow]")
                return pred_dir
            if result.returncode != 0:
                console.print(
                    f"[red]run-batch failed for {condition}[/red]: {result.stderr[-500:]}"
                )
    _merge_preds(preds_file, existing_preds)
    return pred_dir


def gather_patches(pred_dir: Path, condition: Condition) -> list[dict[str, str]]:
    assert pred_dir is not None, "pred_dir must not be None"
    assert condition is not None, "condition must not be None"

    preds_file = pred_dir / "preds.json"
    if not preds_file.exists():
        logger.warning("No preds.json found in %s", pred_dir)
        return []
    try:
        preds = cast(dict[str, dict[str, str]], json.loads(preds_file.read_text()))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read preds.json from %s: %s", pred_dir, exc)
        return []

    assert isinstance(preds, dict), "preds.json must be a dict keyed by instance_id"
    return [
        {
            "instance_id": instance_id,
            "patch": pred.get("model_patch", ""),
            "prefix": condition,
        }
        for instance_id, pred in preds.items()
    ]


def write_patches_json(patches: list[dict[str, str]], path: Path) -> None:
    assert patches is not None, "patches must not be None"
    assert path is not None, "path must not be None"
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(patches, indent=2))
