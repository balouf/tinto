"""The three sources of the companion checks stay independent.

Closed forms (:mod:`tinto.theory`), exact enumeration (:mod:`tinto.exact`)
and Monte Carlo (every other module) are compared against each other in the
companion notebooks; a comparison proves something only if no source reuses
the code of another. The rule, checked on the import statements:

- :mod:`tinto.theory` imports from tinto only itself and :mod:`tinto.graphs`
  (the definition of the problem);
- :mod:`tinto.exact` imports from tinto only :mod:`tinto.graphs`;
- the simulation side imports neither :mod:`tinto.theory` nor :mod:`tinto.exact`.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "tinto"


def _source(path):
    rel = path.relative_to(PACKAGE).with_suffix("").parts
    if rel[0] == "theory":
        return "theory"
    if rel[0] == "exact":
        return "exact"
    return "simulation"


def _tinto_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return {n for n in names if n == "tinto" or n.startswith("tinto.")}


def _allowed(source, name):
    if source == "theory":
        return name.startswith(("tinto.theory", "tinto.graphs"))
    if source == "exact":
        return name.startswith(("tinto.exact", "tinto.graphs"))
    return not name.startswith(("tinto.theory", "tinto.exact"))


@pytest.mark.parametrize(
    "path", sorted(PACKAGE.rglob("*.py")), ids=lambda p: str(p.relative_to(PACKAGE))
)
def test_sources_do_not_import_each_other(path):
    source = _source(path)
    bad = sorted(n for n in _tinto_imports(path) if not _allowed(source, n))
    assert not bad, f"{source} module imports {bad}"
