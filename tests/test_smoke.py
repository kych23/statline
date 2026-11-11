import importlib

import pytest

SUBPACKAGES = ["data", "features", "model", "draft", "yahoo", "server"]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    importlib.import_module(f"statline.{name}")
