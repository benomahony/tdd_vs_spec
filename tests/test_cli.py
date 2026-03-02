import pytest
from typer.testing import CliRunner

from tdd_vs_spec.cli import app

runner = CliRunner()


@pytest.mark.unit
def test_evaluate_no_hardcoded_username(tmp_path):
    result = runner.invoke(app, ["evaluate", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "jefzda" not in result.output, "dockerhub username must not be hardcoded"


@pytest.mark.unit
def test_generate_specs_cli_accepts_model_option():
    result = runner.invoke(app, ["generate-specs", "--help"])
    assert result.exit_code == 0, result.output
    assert "--model" in result.output, (
        "--model option must appear in generate-specs help"
    )


@pytest.mark.unit
def test_evaluate_accepts_custom_username(tmp_path):
    result = runner.invoke(
        app, ["evaluate", str(tmp_path), "--dockerhub-username", "myuser"]
    )
    assert result.exit_code == 0, result.output
    assert "myuser" in result.output, "--dockerhub-username must appear in output"
    assert "jefzda" not in result.output, "hardcoded username must not appear"
