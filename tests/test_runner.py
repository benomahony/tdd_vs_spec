import json

import pytest

from tdd_vs_spec.conditions import Condition, Instance
from tdd_vs_spec.runner import _run_single, gather_patches, write_patches_json


def _fake_instance() -> Instance:
    return Instance(
        instance_id="test__repo__0",
        condition=Condition.TESTS_ONLY,
        problem_statement="fix it",
        test_patch="def test_x(): pass",
        patch="",
        dockerhub_tag="tag",
        repo="test/repo",
        base_commit="abc123",
    )


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
    assert patches[0]["instance_id"] == "fallback", (
        "must fall back to stem when no instance_id"
    )


def test_run_single_missing_dir_raises(tmp_path):
    with pytest.raises(AssertionError, match="mini_swe_agent_dir"):
        _run_single(
            _fake_instance(),
            tmp_path / "out.pred",
            tmp_path / "nonexistent",
            "model",
            30,
        )


def test_run_single_writes_error_pred_on_failure(tmp_path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "run.py").write_text("import sys; sys.exit(1)\n")
    pred_file = tmp_path / "out.pred"
    _run_single(_fake_instance(), pred_file, agent_dir, "model", 30)
    assert pred_file.exists(), "error pred must be written on failure"
    data = json.loads(pred_file.read_text())
    assert data["patch"] == "", "error pred must have empty patch"
    assert "error" in data, "error pred must contain error field"


def test_run_single_uses_sys_executable(tmp_path):
    """runner._run_single must invoke sys.executable, not bare 'python'."""
    from tdd_vs_spec import runner as runner_module
    import inspect

    src = inspect.getsource(runner_module)
    assert "sys.executable" in src, (
        "_run_single must use sys.executable not bare 'python'"
    )
    assert '"python"' not in src or src.index("sys.executable") < src.index(
        '"python"'
    ), "sys.executable must replace bare 'python' in subprocess call"
