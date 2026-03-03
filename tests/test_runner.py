import json

import pytest

from tdd_vs_spec.conditions import Condition, Instance
from tdd_vs_spec.runner import (
    _run_single,
    _write_instances_json,
    gather_patches,
    write_patches_json,
)


def _fake_instance(i: int = 0) -> Instance:
    return Instance(
        instance_id=f"test__repo__{i}",
        condition=Condition.TESTS_ONLY,
        problem_statement="fix it",
        test_patch="def test_x(): pass",
        patch="",
        dockerhub_tag=f"org/img:tag{i}",
        repo="test/repo",
        base_commit="abc123",
    )


@pytest.mark.unit
def test_write_instances_json_includes_required_fields(tmp_path):
    instances = [_fake_instance(0), _fake_instance(1)]
    out = tmp_path / "instances.json"
    _write_instances_json(instances, out)
    data = json.loads(out.read_text())
    assert len(data) == 2, "must write one entry per instance"
    assert data[0]["image_name"] == "org/img:tag0", (
        "must use dockerhub_tag as image_name"
    )
    assert "problem_statement" in data[0], "must include problem_statement"
    assert "test_patch" in data[0], "must include test_patch"
    assert "instance_id" in data[0], "must include instance_id"


@pytest.mark.unit
def test_gather_patches_reads_preds_json(tmp_path):
    pred_dir = tmp_path / "tests_only"
    pred_dir.mkdir()
    preds = {
        "inst_0": {"model_patch": "diff --git a/f.py", "model_name_or_path": "claude"},
        "inst_1": {"model_patch": "", "model_name_or_path": "claude"},
    }
    (pred_dir / "preds.json").write_text(json.dumps(preds))

    patches = gather_patches(pred_dir, Condition.TESTS_ONLY)
    assert len(patches) == 2, "must read both entries from preds.json"
    ids = {p["instance_id"] for p in patches}
    assert ids == {"inst_0", "inst_1"}, "instance_ids must match preds.json keys"
    assert all(p["prefix"] == Condition.TESTS_ONLY for p in patches), (
        "prefix must be condition"
    )


@pytest.mark.unit
def test_gather_patches_uses_model_patch_field(tmp_path):
    pred_dir = tmp_path / "tests_only"
    pred_dir.mkdir()
    preds = {
        "inst_0": {"model_patch": "the patch content", "model_name_or_path": "claude"}
    }
    (pred_dir / "preds.json").write_text(json.dumps(preds))

    patches = gather_patches(pred_dir, Condition.TESTS_ONLY)
    assert patches[0]["patch"] == "the patch content", (
        "must extract model_patch as patch"
    )


@pytest.mark.unit
def test_write_patches_json_round_trips(tmp_path):
    patches = [{"instance_id": "a", "patch": "x", "prefix": "tests_only"}]
    out = tmp_path / "patches.json"
    write_patches_json(patches, out)
    loaded = json.loads(out.read_text())
    assert loaded == patches, "round-trip must preserve patches"


@pytest.mark.unit
def test_run_single_missing_dir_raises(tmp_path):
    with pytest.raises(AssertionError, match="mini_swe_agent_dir"):
        _run_single(
            _fake_instance(),
            tmp_path / "out.pred",
            tmp_path / "nonexistent",
            "model",
            30,
        )


@pytest.mark.unit
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


@pytest.mark.unit
def test_run_single_uses_sys_executable(tmp_path):
    """runner._run_single must invoke sys.executable, not bare 'python'."""
    import inspect

    from tdd_vs_spec import runner as runner_module

    src = inspect.getsource(runner_module)
    assert "sys.executable" in src, (
        "_run_single must use sys.executable not bare 'python'"
    )
    assert '"python"' not in src or src.index("sys.executable") < src.index(
        '"python"'
    ), "sys.executable must replace bare 'python' in subprocess call"
