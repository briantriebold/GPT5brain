from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from gpt5.core.tasks import Task


@dataclass
class ProcessStep:
    actor: str
    action: str
    outcome: str


def create_process_map(steps: Iterable[ProcessStep]) -> str:
    lines = ["Actor | Action | Outcome", "------|--------|--------"]
    for step in steps:
        lines.append(f"{step.actor} | {step.action} | {step.outcome}")
    return "\n".join(lines)


def default_process_for_plan(tasks: List[Task]) -> str:
    steps = []
    for task in tasks:
        actor = task.assigned_agent or "orchestrator"
        text = task.description[:60] + ("..." if len(task.description) > 60 else "")
        steps.append(ProcessStep(actor=actor, action=task.title, outcome=text))
    return create_process_map(steps)

