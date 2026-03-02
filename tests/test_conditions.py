from tdd_vs_spec.conditions import Condition, Instance, load_instances, read_instances, write_instances
from tests.conftest import fake_row, fake_instance


def _dataset(n: int = 5) -> list[dict]:
    return [fake_row(i) for i in range(n)]


def test_load_instances_tests_only_uses_fixed_prompt(tmp_path):
    instances = load_instances(Condition.TESTS_ONLY, dataset=_dataset(3))
    assert len(instances) == 3, "must load all rows"
    for inst in instances:
        assert "failing tests" in inst.problem_statement, "TESTS_ONLY must use the fixed prompt"


def test_load_instances_human_spec_uses_problem_statement(tmp_path):
    instances = load_instances(Condition.TESTS_PLUS_HUMAN_SPEC, dataset=_dataset(3))
    assert len(instances) == 3
    for i, inst in enumerate(instances):
        assert inst.problem_statement == f"Fix issue {i}", "human spec must use the row's problem_statement"


def test_load_instances_llm_spec_skips_missing(tmp_path):
    specs = {"org__repo__0": "spec for 0", "org__repo__2": "spec for 2"}
    instances = load_instances(Condition.TESTS_PLUS_LLM_SPEC, llm_specs=specs, dataset=_dataset(3))
    assert len(instances) == 2, "must skip rows with no LLM spec"
    assert instances[0].problem_statement == "spec for 0"


def test_load_instances_llm_spec_raises_without_specs():
    import pytest
    with pytest.raises(ValueError, match="llm_specs required"):
        load_instances(Condition.TESTS_PLUS_LLM_SPEC, dataset=_dataset(2))


def test_load_instances_respects_limit():
    instances = load_instances(Condition.TESTS_ONLY, limit=2, dataset=_dataset(5))
    assert len(instances) == 2, "limit must be respected"


def test_write_read_roundtrip(tmp_path):
    original = [fake_instance(i) for i in range(5)]
    path = tmp_path / "instances.jsonl"
    write_instances(original, path)
    loaded = read_instances(path)
    assert len(loaded) == 5, "must round-trip all instances"
    for orig, read in zip(original, loaded):
        assert orig.instance_id == read.instance_id
        assert orig.condition == read.condition
