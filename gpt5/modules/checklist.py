from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ChecklistItem:
    name: str
    description: str
    completed: bool = False


@dataclass
class Checklist:
    name: str
    items: List[ChecklistItem] = field(default_factory=list)

    def mark_complete(self, item_name: str) -> None:
        for item in self.items:
            if item.name == item_name:
                item.completed = True
                return
        raise KeyError(f"Checklist item '{item_name}' not found")

    def progress(self) -> float:
        total = len(self.items)
        if not total:
            return 0.0
        completed = sum(1 for item in self.items if item.completed)
        return completed / total


def create_execution_checklist(phase: str) -> Checklist:
    templates: Dict[str, List[ChecklistItem]] = {
        "planning": [
            ChecklistItem("capture-objectives", "Document objectives and success metrics."),
            ChecklistItem("identify-risks", "List risks, mitigations, and owners."),
        ],
        "build": [
            ChecklistItem("scaffold", "Generate code scaffolding and configs."),
            ChecklistItem("tests", "Add or update automated tests."),
            ChecklistItem("review", "Perform code review or quality checks."),
        ],
        "launch": [
            ChecklistItem("docs", "Update documentation and runbooks."),
            ChecklistItem("handoff", "Communicate changes to stakeholders."),
        ],
    }
    return Checklist(name=f"{phase}-execution", items=list(templates.get(phase, [])))

