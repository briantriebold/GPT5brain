from __future__ import annotations

import json
from urllib.parse import urlparse


def validate_url(url: str) -> None:
    p = urlparse(url)
    if p.scheme not in {"http", "https"} or not p.netloc:
        raise ValueError(f"Invalid URL: {url}. Expected http(s)://host[/path]")


def ensure_json_or_hint(name: str, value: str) -> dict:
    try:
        return json.loads(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Option {name} must be valid JSON (e.g., {{\"x\":1}}). "
            f"Hint: use --var key=value for simple inputs. Error: {exc}"
        )


def require_non_empty(name: str, seq) -> None:  # noqa: ANN001
    if not seq:
        raise ValueError(f"{name} must not be empty")

