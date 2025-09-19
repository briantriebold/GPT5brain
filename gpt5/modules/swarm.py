from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from gpt5.core.agent import Agent, AgentProfile
from gpt5.core.orchestrator import Orchestrator
from gpt5.core.tasks import Task


@dataclass
class SwarmConfig:
    name: str
    agent_profiles: Iterable[AgentProfile]


def spawn_swarm(config: SwarmConfig, handler_factory) -> Orchestrator:  # noqa: ANN001
    agents: List[Agent] = []
    for profile in config.agent_profiles:
        handler = handler_factory(profile)
        agents.append(Agent(profile=profile, handler=handler))
    return Orchestrator(agents=agents)


def allocate_tasks(orchestrator: Orchestrator, tasks: Iterable[Task]) -> List[Task]:
    return orchestrator.run_tasks(list(tasks))

