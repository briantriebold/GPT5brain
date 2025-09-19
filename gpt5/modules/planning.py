from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from gpt5.core.tasks import Task


@dataclass
class PlanItem:
    title: str
    description: str
    tasks: List[Task] = field(default_factory=list)


def generate_plan(objective: str, constraints: Sequence[str] | None = None) -> List[PlanItem]:
    constraints = constraints or []
    phases = [
        PlanItem(
            title="Discovery",
            description="Clarify scope, stakeholders, and success metrics.",
            tasks=[
                Task(title="Gather requirements", description=f"Detail objectives for: {objective}", capabilities=["analysis", "planning"]),
                Task(title="Identify constraints", description="List constraints and assumptions.", capabilities=["analysis"]),
            ],
        ),
        PlanItem(
            title="Design",
            description="Shape solution architecture and processes.",
            tasks=[
                Task(title="Draft architecture", description="Outline system components, interactions, and data flows.", capabilities=["design"]),
                Task(title="Define workflows", description="Map user and agent workflows for execution.", capabilities=["design", "process"]),
            ],
        ),
        PlanItem(
            title="Implementation",
            description="Execute build, validation, and rollout steps.",
            tasks=[
                Task(title="Implement modules", description="Develop prioritized modules supporting the objective.", capabilities=["build"]),
                Task(title="Validate and iterate", description="Test outputs and adjust based on feedback.", capabilities=["qa", "planning"]),
            ],
        ),
    ]
    if constraints:
        phases.append(
            PlanItem(
                title="Constraint Review",
                description="Ensure solution respects declared constraints.",
                tasks=[Task(title="Validate constraints", description="Inspect deliverables for compliance.", capabilities=["analysis"], tags=list(constraints))],
            )
        )
    return phases

