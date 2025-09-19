from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional

from .tasks import Task


@dataclass
class AgentProfile:
    name: str
    description: str
    capabilities: Iterable[str] = field(default_factory=list)


class Agent:
    """Base class for specialized agents participating in orchestrated workflows."""

    def __init__(self, profile: AgentProfile, handler: Callable[[Task, "Agent", dict], str]) -> None:
        self.profile = profile
        self._handler = handler
        self.is_active = True

    @property
    def name(self) -> str:
        return self.profile.name

    def can_handle(self, task: Task) -> bool:
        return not task.capabilities or any(cap in self.profile.capabilities for cap in task.capabilities)

    def execute(self, task: Task, orchestrator: Optional[object] = None, context: Optional[dict] = None) -> str:
        return self._handler(task, self, context or {})

    def handle_event(self, event: str, payload: Dict[str, object]) -> None:
        return None

