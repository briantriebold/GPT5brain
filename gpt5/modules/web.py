from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gpt5.core.memory import MemoryStore
from gpt5.tools.web_fetch import WebFetcher


@dataclass
class WebToolConfig:
    cache: bool = True


def create_web_tool(memory: Optional[MemoryStore] = None, config: Optional[WebToolConfig] = None) -> WebFetcher:
    config = config or WebToolConfig()
    return WebFetcher(memory=memory if config.cache else None)

