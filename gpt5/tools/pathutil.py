from __future__ import annotations

import re
from pathlib import Path


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[:\\/\n\r\t]", "_", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._-]", "", name)
    return name[:200] if len(name) > 200 else name

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

