from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from .tasks import Task


HookFn = Callable[[Task], None]


@dataclass
class HookManager:
    pre_hooks: List[HookFn] = None
    post_hooks: List[HookFn] = None

    def __post_init__(self) -> None:
        self.pre_hooks = list(self.pre_hooks or [])
        self.post_hooks = list(self.post_hooks or [])

    def register_pre_hook(self, hook: HookFn) -> None:
        self.pre_hooks.append(hook)

    def register_post_hook(self, hook: HookFn) -> None:
        self.post_hooks.append(hook)

    def run_pre_hooks(self, task: Task) -> None:
        for hook in self.pre_hooks:
            hook(task)

    def run_post_hooks(self, task: Task) -> None:
        for hook in self.post_hooks:
            hook(task)

