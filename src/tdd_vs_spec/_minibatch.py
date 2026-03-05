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


def run_batch_in_process(
    instances_path: Path,
    output_dir: Path,
    model: str,
    max_workers: int,
) -> None:
    """Run batch of instances using minisweagent API; writes preds.json to output_dir."""
    instances_path = Path(instances_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not instances_path.exists():
        raise FileNotFoundError(f"Instances file not found: {instances_path}")
    data = json.loads(instances_path.read_text())
    if not isinstance(data, list):
        raise ValueError("Instances file must contain a JSON array")
    instances = data
    if not instances:
        logger.info("No instances to run")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Results will be saved to %s", output_dir)
    add_file_handler(output_dir / "minisweagent.log")
    config_spec = builtin_config_dir / "extra" / "swebench.yaml"
    config_path = get_config_path(config_spec)
    logger.info("Loading agent config from '%s'", config_path)
    config = yaml.safe_load(config_path.read_text())
    config.setdefault("model", {})["model_name"] = model
    progress_manager = RunBatchProgressManager(
        len(instances), output_dir / f"exit_statuses_{time.time()}.yaml"
    )

    def process_futures(
        futures: dict[concurrent.futures.Future, str],
    ) -> None:
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(
                    "Error in future for instance %s: %s",
                    instance_id,
                    e,
                    exc_info=True,
                )
                progress_manager.on_uncaught_exception(instance_id, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(
                    process_instance,
                    instance,
                    output_dir,
                    config,
                    progress_manager,
                ): instance["instance_id"]
                for instance in instances
            }
            try:
                process_futures(future_to_id)
            except KeyboardInterrupt:
                logger.info(
                    "Cancelling all pending jobs. Press ^C again to exit immediately."
                )
                for future in future_to_id:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(future_to_id)
