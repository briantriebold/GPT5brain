from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from gpt5.core.memory import MemoryStore


@dataclass
class Lesson:
    topic: str
    summary: str
    actions: Iterable[str]


def record_lesson(memory: MemoryStore, lesson: Lesson) -> None:
    memory.save_lesson(lesson.topic, lesson.summary, lesson.actions)


def lesson_template(event: str, impact: str, mitigation: str) -> Lesson:
    summary = f"Event: {event}\nImpact: {impact}\nMitigation: {mitigation}"
    actions = [mitigation]
    return Lesson(topic=event, summary=summary, actions=actions)

