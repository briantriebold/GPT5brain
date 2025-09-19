from __future__ import annotations

from pathlib import Path
from typing import List


def build_index(reports_dir: Path) -> str:
    reports_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in reports_dir.glob("*.html") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    items: List[str] = []
    for p in files:
        name = p.name
        items.append(f'<li><a href="{name}">{name}</a></li>')
    body = "\n".join(items) if items else "<li>No reports yet</li>"
    return f"""
<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>GPT5 Snapshot Index</title>
    <style>
      body {{ font-family: Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 1.5rem; }}
      h1 {{ margin-bottom: 0.5rem; }}
      ul {{ line-height: 1.8; }}
    </style>
  </head>
  <body>
    <h1>GPT5 Snapshot Index</h1>
    <p>Latest exports and reports in {reports_dir.as_posix()}</p>
    <ul>
      {body}
    </ul>
  </body>
</html>
"""

