"""Enforce the dependency rule: data -> features -> model -> draft -> server.

Each subpackage may import only the subpackages listed for it. yahoo is a
low-level source like data, but only server talks to Yahoo.
"""

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "statline"

ALLOWED: dict[str, set[str]] = {
    "data": set(),
    "yahoo": set(),
    "features": {"data"},
    "model": {"data", "features"},
    "draft": {"data", "features", "model"},
    "server": {"data", "features", "model", "draft", "yahoo"},
}


def imported_subpackages(path: Path) -> set[str]:
    """Names of statline subpackages imported by one file, absolute or relative."""
    own = path.relative_to(PACKAGE_ROOT).parts[:-1]  # e.g. ("draft",)
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative: resolve against this file's package
                base = list(own[: len(own) - node.level + 1])
                modules = [".".join(["statline", *base, *([node.module] if node.module else [])])]
            else:
                modules = [node.module or ""]
        else:
            continue
        for module in modules:
            parts = module.split(".")
            if parts[0] == "statline" and len(parts) > 1:
                found.add(parts[1])
    return found


def test_every_subpackage_has_a_rule() -> None:
    actual = {p.name for p in PACKAGE_ROOT.iterdir() if (p / "__init__.py").exists()}
    assert actual == set(ALLOWED), "add new subpackages to ALLOWED"


def test_imports_follow_the_dependency_rule() -> None:
    violations = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        parts = path.relative_to(PACKAGE_ROOT).parts
        if len(parts) < 2:  # statline/__init__.py belongs to no layer
            continue
        package = parts[0]
        for target in imported_subpackages(path) - {package} - ALLOWED[package]:
            violations.append(f"{path.relative_to(PACKAGE_ROOT)} imports statline.{target}")
    assert not violations, "\n".join(violations)
