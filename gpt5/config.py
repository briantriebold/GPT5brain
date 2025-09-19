from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


DEFAULTS: Dict[str, Any] = {
    "plugins": {"enabled": True},
    "autoExport": {
        "enabled": True,
        "prefixes": ["execution:", "mission:", "deficiency:dashboard", "Claude-Flow", "Specify7", "GPT5 Implementation Plan"],
        "dir": "reports",
    },
}


def load_settings(cwd: Path | None = None) -> Dict[str, Any]:
    cwd = cwd or Path.cwd()
    cfg_path = cwd / "gpt5.settings.json"
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            return _merge(DEFAULTS, data)
        except Exception:
            return DEFAULTS
    return DEFAULTS


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out

