from __future__ import annotations

def before_command(args):  # noqa: ANN001
    # Elevate network retry policy by monkeypatching WebFetcher.fetch defaults.
    try:
        from gpt5.tools.web_fetch import WebFetcher as _WF
    except Exception:
        return

    _orig = _WF.fetch

    def _wrap(self, url: str, timeout: int = 10, refresh: bool = False, retries: int = 0, backoff: float = 0.5):  # noqa: ANN001
        if retries < 3:
            retries = 3
        if backoff < 0.75:
            backoff = 0.75
        return _orig(self, url, timeout=timeout, refresh=refresh, retries=retries, backoff=backoff)

    try:
        _WF.fetch = _wrap  # type: ignore[assignment]
    except Exception:
        pass

