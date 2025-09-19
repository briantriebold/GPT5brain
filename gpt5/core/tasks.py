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
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_ms: Optional[int] = None

    def mark_in_progress(self) -> None:
        self.status = "in_progress"
        now = datetime.utcnow()
        self.updated_at = now
        if self.started_at is None:
            self.started_at = now

    def mark_completed(self, result: Optional[str] = None) -> None:
        self.status = "completed"
        self.result = result
        now = datetime.utcnow()
        self.updated_at = now
        if self.started_at and not self.finished_at:
            self.finished_at = now
            self.duration_ms = int((self.finished_at - self.started_at).total_seconds() * 1000)

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
            "started_at": self.started_at.isoformat() if self.started_at else "",
            "finished_at": self.finished_at.isoformat() if self.finished_at else "",
            "duration_ms": self.duration_ms if self.duration_ms is not None else None,
        }
