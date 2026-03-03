import json

import pytest

from tdd_vs_spec.conditions import Condition, Instance
from tdd_vs_spec.runner import (
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
def test_gather_patches_returns_empty_when_no_preds_file(tmp_path):
    pred_dir = tmp_path / "tests_only"
    pred_dir.mkdir()
    patches = gather_patches(pred_dir, Condition.TESTS_ONLY)
    assert patches == [], "must return empty list when preds.json is missing"


@pytest.mark.unit
def test_gather_patches_returns_empty_on_corrupt_preds_json(tmp_path):
    pred_dir = tmp_path / "tests_only"
    pred_dir.mkdir()
    (pred_dir / "preds.json").write_text("{not valid json")
    patches = gather_patches(pred_dir, Condition.TESTS_ONLY)
    assert patches == [], "must return empty list on corrupt preds.json"


@pytest.mark.unit
def test_write_patches_json_round_trips(tmp_path):
    patches = [{"instance_id": "a", "patch": "x", "prefix": "tests_only"}]
    out = tmp_path / "patches.json"
    write_patches_json(patches, out)
    loaded = json.loads(out.read_text())
    assert loaded == patches, "round-trip must preserve patches"
