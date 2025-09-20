from __future__ import annotations

import datetime as _dt
import subprocess
from pathlib import Path
from typing import Dict, List


def _run(cmd: list[str]) -> str:
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr or cp.stdout)
    return cp.stdout.strip()


def _latest_tag() -> str | None:
    try:
        return _run(["git", "describe", "--tags", "--abbrev=0"]).strip()
    except Exception:
        return None


def generate_changelog(output: Path | None = None) -> str:
    """Generate a simple conventional-changelog style markdown from git log."""
    tag = _latest_tag()
    rng = f"{tag}..HEAD" if tag else "HEAD"
    try:
        lines = _run(["git", "log", rng, "--pretty=format:%s"]).splitlines()
    except Exception:
        lines = []
    cats: Dict[str, List[str]] = {
        "Features": [],
        "Fixes": [],
        "Docs": [],
        "CI": [],
        "Chore": [],
        "Refactor": [],
        "Perf": [],
        "Tests": [],
        "Build": [],
        "Other": [],
    }
    for s in lines:
        ls = s.strip()
        key = "Other"
        low = ls.lower()
        if low.startswith("feat"): key = "Features"
        elif low.startswith("fix"): key = "Fixes"
        elif low.startswith("docs"): key = "Docs"
        elif low.startswith("ci"): key = "CI"
        elif low.startswith("chore"): key = "Chore"
        elif low.startswith("refactor"): key = "Refactor"
        elif low.startswith("perf"): key = "Perf"
        elif low.startswith("test"): key = "Tests"
        elif low.startswith("build"): key = "Build"
        cats[key].append(ls)
    date = _dt.datetime.utcnow().strftime("%Y-%m-%d")
    parts = [f"# Changelog", "", f"## {date}"]
    if tag:
        parts.append(f"Changes since {tag}")
    for k, items in cats.items():
        if not items: continue
        parts.append("")
        parts.append(f"### {k}")
        for it in items:
            parts.append(f"- {it}")
    md = "\n".join(parts).strip() + "\n"
    if output:
        output.write_text(md, encoding="utf-8")
    return md

