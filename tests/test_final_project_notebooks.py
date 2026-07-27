import json
from pathlib import Path


def test_final_project_notebooks_are_valid_and_documented() -> None:
    notebook_dir = Path(__file__).resolve().parents[1] / "notebooks" / "final_project"
    notebooks = sorted(notebook_dir.glob("*.ipynb"))

    assert len(notebooks) == 4
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"][0]["cell_type"] == "markdown"
        assert "**Task.**" in "".join(notebook["cells"][0]["source"])
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile(
                    "".join(cell["source"]),
                    f"{path.name}:cell-{index}",
                    "exec",
                )
