from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from gpt5.core.memory import MemoryStore
from gpt5.modules import wisdom as wisdom_module
from pathlib import Path
import json as _json


def _normalize_text(s: str) -> str:
    return " ".join(s.replace("\n", " ").replace("\r", " ").split())[:500]


def _signature_from_exception(exc: Exception) -> Tuple[str, str]:
    etype = exc.__class__.__name__
    msg = str(exc)
    # coarse signature: exception type + first 80 chars of message
    sig = f"{etype}:{_normalize_text(msg)[:80]}"
    return etype, sig


@dataclass
class DeficiencyDetector:
    memory: MemoryStore
    threshold: int = 2  # occurrences before proposing countermeasure

    def on_exception(self, where: str, args: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
        etype, signature = _signature_from_exception(exc)
        tb = traceback.format_exc(limit=5)
        context = {"where": where, "args": args}
        did, count = self.memory.update_deficiency(signature, etype, _normalize_text(str(exc)), _normalize_text(str(context)))
        self.memory.emit_event("error", {"signature": signature, "where": where, "etype": etype, "count": count})

        plan = None
        if count >= self.threshold:
            plan = self._propose_countermeasure(signature, etype, str(exc), tb)
            # Attempt auto-apply for known patterns
            applied = self._auto_apply(signature, args)
            status = "mitigated" if applied else "proposed"
            self.memory.set_deficiency_countermeasure(signature, plan if not applied else f"auto:{applied}", status=status)
            # Record wisdom entry
            lesson = wisdom_module.lesson_template(
                event=f"deficiency:{signature}",
                impact=f"Recurring error ({count}x) at {where}",
                mitigation="Countermeasure proposed; apply or implement plugin/hook to prevent recurrence.",
            )
            wisdom_module.record_lesson(self.memory, lesson)

        return {"id": did, "signature": signature, "count": count, "proposed": bool(plan)}

    def _propose_countermeasure(self, signature: str, etype: str, message: str, tb: str) -> str:
        lines = [
            f"# Countermeasure Proposal",
            f"Signature: {signature}",
            f"Type: {etype}",
            "",
            "## Root Cause Hypothesis",
            "- Analyze error pattern and context; identify missing capability or fragile assumption.",
            "",
            "## Mitigation Options",
            "- Add input adapters or alternative flags to handle varied user inputs",
            "- Sanitize output to avoid encoding issues",
            "- Add retries/caching for network flakiness",
            "- Extend CLI with guardrails or pre-flight validation",
            "- Implement plugin/hook to intercept and normalize problematic flows",
            "",
            "## Immediate Action",
            "- Log lesson learned, index error signature",
            "- Add a checklist item to validate fix",
            "",
            "## Implementation Sketch",
            "- Update CLI or module based on the specific pattern",
            "- Write regression test or example to verify resolution",
        ]
        return "\n".join(lines)

    def _auto_apply(self, signature: str, args: Dict[str, Any]) -> str | None:
        # Known mitigation: JSON vars parsing for math expr
        if signature.startswith("JSONDecodeError") and args.get("command") == "math":
            return self.ensure_json_vars_adapter_plugin()
        return None

    def ensure_json_vars_adapter_plugin(self) -> str:
        plugin_dir = Path(__file__).resolve().parents[1] / "plugins"
        plugin_dir.mkdir(parents=True, exist_ok=True)
        plugin_path = plugin_dir / "json_vars_adapter.py"
        if not plugin_path.exists():
            code = """
from __future__ import annotations

import json as _json

def _kv_to_json(s: str) -> str:
    s = s.strip()
    if s.startswith('{') and s.endswith('}'):
        # fix single quotes to double quotes
        return s.replace("'", '"')
    # parse key=value pairs separated by comma or whitespace
    parts = []
    tmp = s.replace('\\n', ' ')
    for token in tmp.split(','):
        token = token.strip()
        if not token:
            continue
        parts.extend(token.split())
    kv = {}
    for p in parts:
        if '=' in p:
            k, v = p.split('=', 1)
            try:
                kv[k.strip()] = float(v)
            except Exception:
                pass
    if kv:
        return _json.dumps(kv)
    return s

def before_command(args):
    if getattr(args, 'command', None) == 'math' and getattr(args, 'math_cmd', None) == 'expr':
        s = getattr(args, 'vars', None)
        if isinstance(s, str) and s:
            try:
                _json.loads(s)
            except Exception:
                fixed = _kv_to_json(s)
                setattr(args, 'vars', fixed)
"""
            plugin_path.write_text(code, encoding="utf-8")
        # Seed a regression for this signature if none exists
        try:
            argv = ["math", "expr", "sin(x)", "--vars", "x=1.2", "--json"]
            self.memory.add_regression(
                signature="JSONDecodeError:Expecting property name enclosed in double quotes: line 1 column 2 (char 1)",
                argv_json=_json.dumps(argv),
            )
        except Exception:
            pass
        return str(plugin_path)
