# GPT5 Orchestration System (CLI)

Local, modular orchestration for planning, PRD generation, process mapping,
checklists, swarm execution, evergreen memory, and a simple web fetcher.

Quick start (PowerShell):
- python -m venv .venv
- .\.venv\Scripts\python -m pip install -e .
- .\.venv\Scripts\gpt5 --help


## Live Reports (GitHub Pages)
- A workflow publishes `reports/` to GitHub Pages on pushes to `main`.
- Once the first deployment completes, browse: https://briantriebold.github.io/GPT5brain/
- Index page: `reports/index.html` lists all exported reports.
