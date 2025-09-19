from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from datetime import datetime
from typing import Dict, List, Optional

from .agent import Agent
from .tasks import Task
from .hooks import HookManager
from .memory import MemoryStore


@dataclass
class OrchestratorConfig:
    name: str = "gpt5-orchestrator"
    enable_hooks: bool = True
    enable_memory: bool = True


class Orchestrator:
    """Coordinates specialized agents, hooks, and memory for task execution."""

    def __init__(
        self,
        agents: Optional[List[Agent]] = None,
        memory: Optional[MemoryStore] = None,
        hook_manager: Optional[HookManager] = None,
        config: Optional[OrchestratorConfig] = None,
    ) -> None:
        self.config = config or OrchestratorConfig()
        self.agents: Dict[str, Agent] = {agent.name: agent for agent in agents or []}
        self.memory = memory
        self.hook_manager = hook_manager or HookManager()
        self.task_log: List[Task] = []

    def register_agent(self, agent: Agent) -> None:
        if agent.name in self.agents:
            raise ValueError(f"Agent '{agent.name}' already registered")
        self.agents[agent.name] = agent

    def run_task(self, task: Task) -> Task:
        if self.config.enable_hooks:
            self.hook_manager.run_pre_hooks(task)

        agent = self._select_agent(task)
        if not agent:
            raise RuntimeError(f"No agent available for task '{task.title}'")
        task.mark_in_progress()
        start = perf_counter()
        result = agent.execute(task, orchestrator=self)
        end = perf_counter()
        task.result = result
        task.status = "completed"
        if task.started_at is None:
            task.started_at = datetime.utcnow()
        task.finished_at = datetime.utcnow()
        task.duration_ms = int((end - start) * 1000)
        self.task_log.append(task)

        if self.config.enable_hooks:
            self.hook_manager.run_post_hooks(task)

        if self.config.enable_memory and self.memory:
            self.memory.persist_task(task)

        return task

    def run_tasks(self, tasks: List[Task]) -> List[Task]:
        completed = []
        for task in tasks:
            completed.append(self.run_task(task))
        return completed

    def _select_agent(self, task: Task) -> Optional[Agent]:
        if task.assigned_agent:
            return self.agents.get(task.assigned_agent)
        for agent in self.agents.values():
            if agent.can_handle(task):
                return agent
        return None

    def summary(self) -> Dict[str, int]:
        return {
            "tasks_completed": len(self.task_log),
            "agents_active": sum(1 for a in self.agents.values() if a.is_active),
        }
