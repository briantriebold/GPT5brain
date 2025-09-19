# GPT5 Orchestration System (CLI)

Local, modular orchestration for planning, PRD generation, process mapping,
checklists, swarm execution, evergreen memory, and a simple web fetcher.

Quick start (PowerShell):
- python -m venv .venv
- .\.venv\Scripts\python -m pip install -e .
- .\.venv\Scripts\gpt5 --help


## Reports & Snapshot Index
- HTML reports auto-export to `reports/` when enabled in `gpt5.settings.json`.
- Build/update index: `gpt5 report index --dir reports`
- Open `reports/index.html` in your browser to navigate the latest reports.
