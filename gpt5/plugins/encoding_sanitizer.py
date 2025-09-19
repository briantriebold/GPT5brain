from __future__ import annotations

def before_command(args):  # noqa: ANN001
    try:
        import sys
        # Prefer reconfigure when available (Py3.7+)
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        try:
            # Fallback: wrap using codecs writer
            import codecs, sys as _sys
            _sys.stdout = codecs.getwriter("utf-8")(_sys.stdout.buffer, errors="replace")  # type: ignore[attr-defined]
            _sys.stderr = codecs.getwriter("utf-8")(_sys.stderr.buffer, errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass

