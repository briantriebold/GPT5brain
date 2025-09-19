from __future__ import annotations

import html
from typing import List


MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"


def _extract_mermaid_blocks(markdown: str) -> List[str]:
    blocks: List[str] = []
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\r\n")
        if line.strip().startswith("```mermaid"):
            i += 1
            buf: List[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            blocks.append("\n".join(buf))
        i += 1
    return blocks


def render_html(markdown: str, title: str = "Report") -> str:
    mer_blocks = _extract_mermaid_blocks(markdown)
    escaped = html.escape(markdown)
    mermaid_divs = "\n".join(f'<div class="mermaid">\n{b}\n</div>' for b in mer_blocks)
    return f"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{html.escape(title)}</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 1.5rem; }}
      pre {{ background: #f6f8fa; padding: 1rem; border-radius: 6px; overflow-x: auto; }}
      .mermaid {{ margin: 1rem 0; }}
      h1, h2, h3 {{ margin-top: 1.2rem; }}
    </style>
  </head>
  <body>
    <h1>{html.escape(title)}</h1>
    <h2>Markdown (raw)</h2>
    <pre>{escaped}</pre>
    <h2>Mermaid Diagrams</h2>
    {mermaid_divs}
    <script src="{MERMAID_CDN}"></script>
    <script>mermaid.initialize({{ startOnLoad: true }});</script>
  </body>
</html>
"""
