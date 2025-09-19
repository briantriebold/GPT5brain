from __future__ import annotations

import json as _json

def _kv_to_json(s: str) -> str:
    s = s.strip()
    if s.startswith('{') and s.endswith('}'):
        # fix single quotes to double quotes
        return s.replace("'", '"')
    # parse key=value pairs separated by comma or whitespace
    parts = []
    tmp = s.replace('\n', ' ')
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
    # Adapt math expr --vars when it's not valid JSON
    if getattr(args, 'command', None) == 'math' and getattr(args, 'math_cmd', None) == 'expr':
        s = getattr(args, 'vars', None)
        if isinstance(s, str) and s:
            try:
                _json.loads(s)
            except Exception:
                fixed = _kv_to_json(s)
                setattr(args, 'vars', fixed)
