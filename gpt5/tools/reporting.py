from __future__ import annotations

from typing import Iterable

from gpt5.core.tasks import Task


def format_task_report(tasks: Iterable[Task]) -> str:
    lines = ["# Task Report", "", "Title | Status | Result", "------|--------|-------"]
    for task in tasks:
        lines.append(f"{task.title} | {task.status} | {task.result or ''}")
    return "\n".join(lines)

