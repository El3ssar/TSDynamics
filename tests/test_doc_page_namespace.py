"""Documentation pages must supply the same imports a reader needs."""

import _doctest_select as docs_gate
import pytest


@pytest.mark.parametrize("expression", ["np.arange(3)", "ts.systems.Lorenz()", "Lorenz()"])
def test_page_runner_does_not_inject_library_names(tmp_path, monkeypatch, expression):
    (tmp_path / "example.md").write_text(f"```python\n{expression}\n```\n")
    monkeypatch.setattr(docs_gate, "DOCS_DIR", tmp_path)
    with pytest.raises(AssertionError, match="NameError"):
        docs_gate.run_page_fences("example.md")


def test_page_runner_preserves_explicit_imports_between_fences(tmp_path, monkeypatch):
    (tmp_path / "example.md").write_text(
        "```python\nimport numpy as np\nfrom tsdynamics.systems import Lorenz\n```\n"
        "```python\nassert np.arange(3).size == Lorenz().dim\n```\n"
    )
    monkeypatch.setattr(docs_gate, "DOCS_DIR", tmp_path)
    docs_gate.run_page_fences("example.md")
