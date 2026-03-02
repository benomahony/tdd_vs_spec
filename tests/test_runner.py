import json
import sys
from pathlib import Path

import pytest

from tdd_vs_spec.conditions import Condition
from tdd_vs_spec.runner import gather_patches, write_patches_json
from tests.conftest import fake_instance


def test_gather_patches_reads_pred_files(tmp_path):
    pred_dir = tmp_path / "preds" / "tests_only"
    pred_dir.mkdir(parents=True)
    (pred_dir / "inst_0.pred").write_text(
        json.dumps({"instance_id": "inst_0", "patch": "diff --git a/f.py"})
    )
    (pred_dir / "inst_1.pred").write_text(
        json.dumps({"instance_id": "inst_1", "patch": ""})
    )

    patches = gather_patches(pred_dir, Condition.TESTS_ONLY)
    assert len(patches) == 2, "must read both .pred files"
    ids = {p["instance_id"] for p in patches}
    assert ids == {"inst_0", "inst_1"}, "instance_ids must match filenames"


def test_write_patches_json_round_trips(tmp_path):
    patches = [{"instance_id": "a", "patch": "x", "prefix": "tests_only"}]
    out = tmp_path / "patches.json"
    write_patches_json(patches, out)
    loaded = json.loads(out.read_text())
    assert loaded == patches, "round-trip must preserve patches"


def test_gather_patches_handles_missing_instance_id(tmp_path):
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    (pred_dir / "fallback.pred").write_text(json.dumps({"patch": "something"}))
    patches = gather_patches(pred_dir, Condition.TESTS_ONLY)
    assert patches[0]["instance_id"] == "fallback", "must fall back to stem when no instance_id"


def test_run_single_uses_sys_executable(tmp_path):
    """runner._run_single must invoke sys.executable, not bare 'python'."""
    from tdd_vs_spec import runner as runner_module
    import inspect

    src = inspect.getsource(runner_module)
    assert "sys.executable" in src, "_run_single must use sys.executable not bare 'python'"
    assert '"python"' not in src or src.index("sys.executable") < src.index('"python"'), \
        "sys.executable must replace bare 'python' in subprocess call"
