from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import List, Any


def discover_plugins(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    return [p for p in dir_path.glob("*.py") if p.name not in {"__init__.py"}]


def load_plugin(py_path: Path):  # noqa: ANN001
    spec = importlib.util.spec_from_file_location(py_path.stem, py_path)
    if not spec or not spec.loader:  # type: ignore[truthy-function]
        raise ImportError(f"Cannot load plugin: {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def before_command_all(plugins: List[Any], args: Any) -> None:  # noqa: ANN401
    for mod in plugins:
        fn = getattr(mod, "before_command", None)
        if callable(fn):
            try:
                fn(args)
            except Exception:
                # plugins should not break core execution
                continue


def after_command_all(plugins: List[Any], args: Any, result: Any = None) -> None:  # noqa: ANN401
    for mod in plugins:
        fn = getattr(mod, "after_command", None)
        if callable(fn):
            try:
                fn(args, result)
            except Exception:
                continue
