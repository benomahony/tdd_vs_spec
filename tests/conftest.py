import duckdb
import pytest

from tdd_vs_spec.conditions import Condition, Instance


def fake_row(i: int = 0) -> dict:
    return {
        "instance_id": f"org__repo__{i}",
        "problem_statement": f"Fix issue {i}",
        "test_patch": f"def test_{i}(): assert True",
        "patch": f"# fix {i}",
        "dockerhub_tag": f"tag_{i}",
        "repo": "org/repo",
        "base_commit": "abc123",
    }


def fake_instance(i: int = 0, condition: Condition = Condition.TESTS_ONLY) -> Instance:
    return Instance(
        instance_id=f"org__repo__{i}",
        condition=condition,
        problem_statement=f"Fix issue {i}",
        test_patch=f"def test_{i}(): assert True",
        patch=f"# fix {i}",
        dockerhub_tag=f"tag_{i}",
        repo="org/repo",
        base_commit="abc123",
    )


@pytest.fixture
def fake_db() -> duckdb.DuckDBPyConnection:
    db = duckdb.connect()
    db.execute("""
        CREATE TABLE results (
            instance_id VARCHAR,
            prefix VARCHAR,
            resolved BOOLEAN,
            total_cost DOUBLE,
            steps INTEGER
        )
    """)
    rows = [
        ("org__repo__0", "tests_only", True, 0.05, 3),
        ("org__repo__1", "tests_only", False, 0.03, 2),
        ("org__repo__2", "tests_plus_human_spec", True, 0.06, 4),
        ("org__repo__3", "tests_plus_human_spec", True, 0.04, 3),
        ("org__repo__4", "tests_plus_llm_spec", False, 0.02, 1),
    ]
    db.executemany("INSERT INTO results VALUES (?, ?, ?, ?, ?)", rows)
    return db
