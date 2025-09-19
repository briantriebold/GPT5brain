from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable


@dataclass
class PRDSection:
    title: str
    content: str


@dataclass
class ProductRequirementsDocument:
    name: str
    author: str
    created_at: datetime
    sections: Iterable[PRDSection]

    def to_markdown(self) -> str:
        lines = [f"# {self.name}", f"_Author: {self.author}_", f"_Created: {self.created_at.isoformat()}_", ""]
        for section in self.sections:
            lines.extend([f"## {section.title}", section.content.strip(), ""])
        return "\n".join(lines).strip()


DEFAULT_SECTIONS = [
    ("Overview", "Summarize the problem space, target users, and desired outcome."),
    ("Goals", "Enumerate primary goals and measurable success metrics."),
    ("Non-Goals", "Clarify what is deliberately out of scope."),
    ("User Stories", "Document key user stories or scenarios."),
    ("Requirements", "List functional and non-functional requirements."),
    ("Risks", "Capture risks, mitigations, and assumptions."),
    ("Launch Plan", "Outline milestones, rollout plan, and validation steps."),
]


def create_prd(name: str, author: str, prompts: Dict[str, str] | None = None) -> ProductRequirementsDocument:
    prompts = prompts or {}
    sections = []
    now = datetime.utcnow()
    for section_name, guidance in DEFAULT_SECTIONS:
        base = prompts.get(section_name, guidance)
        sections.append(PRDSection(title=section_name, content=base))
    return ProductRequirementsDocument(name=name, author=author, created_at=now, sections=sections)

