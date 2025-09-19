from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Task:
    title: str
    description: str
    capabilities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    result: Optional[str] = None

    def mark_in_progress(self) -> None:
        self.status = "in_progress"
        self.updated_at = datetime.utcnow()

    def mark_completed(self, result: Optional[str] = None) -> None:
        self.status = "completed"
        self.result = result
        self.updated_at = datetime.utcnow()

    def to_record(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "capabilities": ",".join(self.capabilities),
            "tags": ",".join(self.tags),
            "dependencies": ",".join(self.dependencies),
            "assigned_agent": self.assigned_agent,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "result": self.result or "",
        }

