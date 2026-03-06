"""Run mini-swe-agent batch in-process (no subprocess). Requires mini-swe-agent to be installed."""

from __future__ import annotations

import concurrent.futures
import json
import time
from pathlib import Path

from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.run.extra.swebench import process_instance
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger
import yaml
from rich.live import Live


def _load_instances(instances_path: Path) -> list[dict[str, str]]:
    assert instances_path is not None, "instances_path must not be None"
    assert isinstance(instances_path, Path), "instances_path must be a Path"
    if not instances_path.exists():
        raise FileNotFoundError(f"Instances file not found: {instances_path}")
    data = json.loads(instances_path.read_text())
    if not isinstance(data, list):
        raise ValueError("Instances file must contain a JSON array")
    return data


def _build_config(
    output_dir: Path, model: str
) -> tuple[dict[str, object], RunBatchProgressManager]:
    assert output_dir is not None, "output_dir must not be None"
    assert isinstance(model, str) and model, "model must be a non-empty string"
    add_file_handler(output_dir / "minisweagent.log")
    config_spec = builtin_config_dir / "extra" / "swebench.yaml"
    config_path = get_config_path(config_spec)
    logger.info("Loading agent config from '%s'", config_path)
    config = yaml.safe_load(config_path.read_text())
    config.setdefault("model", {})["model_name"] = model
    progress_manager = RunBatchProgressManager(
        0, output_dir / f"exit_statuses_{time.time()}.yaml"
    )
    return config, progress_manager


def _process_futures(
    futures: dict[concurrent.futures.Future, str],
    progress_manager: RunBatchProgressManager,
) -> None:
    assert futures is not None, "futures must not be None"
    assert progress_manager is not None, "progress_manager must not be None"
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except concurrent.futures.CancelledError:
            pass
        except Exception as e:
            instance_id = futures[future]
            logger.error(
                "Error in future for instance %s: %s", instance_id, e, exc_info=True
            )
            progress_manager.on_uncaught_exception(instance_id, e)


def run_batch_in_process(
    instances_path: Path,
    output_dir: Path,
    model: str,
    max_workers: int,
) -> None:
    """Run batch of instances using minisweagent API; writes preds.json to output_dir."""
    assert max_workers > 0, "max_workers must be positive"
    assert isinstance(model, str) and model, "model must be a non-empty string"
    instances_path = Path(instances_path).resolve()
    output_dir = Path(output_dir).resolve()
    instances = _load_instances(instances_path)
    if not instances:
        logger.info("No instances to run")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Results will be saved to %s", output_dir)
    config, progress_manager = _build_config(output_dir, model)
    progress_manager = RunBatchProgressManager(
        len(instances), output_dir / f"exit_statuses_{time.time()}.yaml"
    )
    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(
                    process_instance, instance, output_dir, config, progress_manager
                ): instance["instance_id"]
                for instance in instances
            }
            try:
                _process_futures(future_to_id, progress_manager)
            except KeyboardInterrupt:
                logger.info(
                    "Cancelling all pending jobs. Press ^C again to exit immediately."
                )
                for future in future_to_id:
                    if not future.running() and not future.done():
                        future.cancel()
                _process_futures(future_to_id, progress_manager)
