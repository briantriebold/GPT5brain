from __future__ import annotations

import contextlib
import urllib.request
from dataclasses import dataclass
import time

from gpt5.core.memory import MemoryStore


@dataclass
class WebFetcher:
    memory: MemoryStore | None = None

    def fetch(self, url: str, timeout: int = 10, refresh: bool = False, retries: int = 0, backoff: float = 0.5) -> str:
        if self.memory and not refresh:
            cached = self.memory.get_cached_web_content(url)
            if cached:
                return cached

        attempt = 0
        last_exc: Exception | None = None
        while True:
            try:
                with contextlib.closing(urllib.request.urlopen(url, timeout=timeout)) as response:
                    content = response.read().decode("utf-8", errors="ignore")
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= retries:
                    content = f"Unable to fetch '{url}': {exc}"
                    break
                time.sleep(backoff * (2 ** attempt))
                attempt += 1

        if self.memory:
            self.memory.cache_web_content(url, content)
        return content
