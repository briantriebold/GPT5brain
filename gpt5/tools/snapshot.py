from __future__ import annotations

from pathlib import Path
from typing import List


def build_index(reports_dir: Path) -> str:
    reports_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in reports_dir.glob("*.html") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    # Latest (top 5)
    latest_items: List[str] = []
    for p in files[:5]:
        latest_items.append(f'<li><a href="{p.name}">{p.name}</a> <small>(updated)</small></li>')
    latest_body = "\n".join(latest_items) if latest_items else "<li>No recent reports</li>"
    # All files
    items: List[str] = []
    for p in files:
        items.append(f'<li><a href="{p.name}">{p.name}</a></li>')
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
      .cards {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(240px,1fr)); gap: 1rem; margin: 1rem 0; }}
      .card {{ border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; background: #fafafa; }}
      .card h3 {{ margin-top: 0; }}
    </style>
  </head>
  <body>
    <h1>GPT5 Snapshot Index</h1>
    <div class=\"cards\"> 
      <div class=\"card\">
        <h3>Latest Reports</h3>
        <ul>
          {latest_body}
        </ul>
      </div>
      <div class=\"card\">
        <h3>Status</h3>
        <p><a href=\"status.html\">Open PR Status</a></p>
      </div>
    </div>
    <h2>All Reports</h2>
    <ul>
      {body}
    </ul>
  </body>
</html>
"""
