"""Tests for public API surface and shared console singleton."""


def test_condition_importable_from_package():
    from tdd_vs_spec import Condition

    assert Condition.TESTS_ONLY == "tests_only"
    assert len(list(Condition)) == 3


def test_instance_importable_from_package():
    from tdd_vs_spec import Instance, Condition

    inst = Instance(
        instance_id="a__b__0",
        condition=Condition.TESTS_ONLY,
        problem_statement="fix it",
        test_patch="def test(): pass",
        patch="",
        dockerhub_tag="tag",
        repo="a/b",
        base_commit="abc",
    )
    assert inst.instance_id == "a__b__0"
    assert isinstance(inst, Instance)


def test_all_modules_share_same_console():
    from tdd_vs_spec import analysis, runner, cli

    assert analysis.console is runner.console, (
        "analysis and runner must share one Console"
    )
    assert analysis.console is cli.console, "analysis and cli must share one Console"
