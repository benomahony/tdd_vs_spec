from pathlib import Path

from typer.testing import CliRunner

from tdd_vs_spec.cli import app

runner = CliRunner()


def test_evaluate_no_hardcoded_username(tmp_path):
    result = runner.invoke(app, ["evaluate", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "jefzda" not in result.output, "dockerhub username must not be hardcoded"


def test_evaluate_accepts_custom_username(tmp_path):
    result = runner.invoke(
        app, ["evaluate", str(tmp_path), "--dockerhub-username", "myuser"]
    )
    assert result.exit_code == 0, result.output
    assert "myuser" in result.output, "--dockerhub-username must appear in output"
    assert "jefzda" not in result.output, "hardcoded username must not appear"
