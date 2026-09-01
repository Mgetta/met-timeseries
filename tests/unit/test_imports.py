import importlib
import pytest
from pathlib import Path

SRC_PATH = Path(__file__).parent.parent.parent / "src/met_timeseries"
EXCLUDE = ['narr']
def _find_modules(src_path,modules=None, prefix="met_timeseries"):
    if modules is None:
        modules = []
    # walk pacakge to find all submodules dynamically
    for path in Path(src_path).iterdir():
        if path.is_dir() and (path / "__init__.py").exists():
            _find_modules(path, modules, prefix + "." + path.name)
        elif path.suffix == ".py" and path.name != "__init__.py" and path.stem not in EXCLUDE:
            modules.append(prefix + "." + path.stem)
    return modules


def test_module_imports():
    """Dynamically tests each module import in isolation."""
    for module_name in _find_modules(SRC_PATH):
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}. Error: {e}")