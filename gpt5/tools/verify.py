from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr


@dataclass
class EquivalenceResult:
    equivalent: bool
    method: str
    counterexample: Optional[Dict[str, float]] = None
    message: str = ""


def _symbols_from_exprs(exprs: Iterable[str]) -> List[sp.Symbol]:
    names = set()
    for s in exprs:
        for ch in s:
            # heuristic: collect ascii letters for now
            pass
    # More robust: parse and extract symbols
    syms = set()
    for s in exprs:
        try:
            e = parse_expr(s, evaluate=False)
            syms.update(map(str, e.free_symbols))
        except Exception:
            continue
    return [sp.Symbol(n) for n in sorted(syms)]


def simplify_expr(expr: str) -> str:
    e = parse_expr(expr, evaluate=True)
    s = sp.simplify(e)
    return sp.sstr(s)


def equivalent_exprs(
    a_str: str,
    b_str: str,
    var_names: Optional[Iterable[str]] = None,
    samples: int = 50,
    tol: float = 1e-9,
) -> EquivalenceResult:
    try:
        a = parse_expr(a_str, evaluate=True)
        b = parse_expr(b_str, evaluate=True)
    except Exception as exc:  # noqa: BLE001
        return EquivalenceResult(False, method="parse_error", message=str(exc))

    # Try symbolic equality when possible (assume real domain)
    try:
        diff = sp.simplify(a - b)
        if diff == 0:
            return EquivalenceResult(True, method="symbolic")
    except Exception:
        # Could be non-arithmetic expressions; fall back to numeric
        pass

    # Numeric testing
    if var_names is None:
        vars_syms = _symbols_from_exprs([a_str, b_str])
    else:
        vars_syms = [sp.Symbol(v) for v in var_names]

    if not vars_syms:
        # Constant expressions
        try:
            av = float(a.evalf())
            bv = float(b.evalf())
            if math.isclose(av, bv, rel_tol=tol, abs_tol=tol):
                return EquivalenceResult(True, method="constant_numeric")
            return EquivalenceResult(False, method="constant_numeric", counterexample={}, message=f"{av} != {bv}")
        except Exception as exc:  # noqa: BLE001
            return EquivalenceResult(False, method="constant_error", message=str(exc))

    # Sample random points in [-10, 10], avoid singularities
    for _ in range(max(5, samples)):
        env: Dict[sp.Symbol, float] = {}
        for s in vars_syms:
            env[s] = random.uniform(-10, 10)
        try:
            av = float(a.evalf(subs=env))
            bv = float(b.evalf(subs=env))
        except Exception:
            # Skip invalid points (e.g., division by zero)
            continue
        if not math.isfinite(av) or not math.isfinite(bv):
            continue
        if not math.isclose(av, bv, rel_tol=tol, abs_tol=tol):
            return EquivalenceResult(
                False,
                method="random_testing",
                counterexample={str(k): float(v) for k, v in env.items()},
                message=f"{av} != {bv}",
            )
    return EquivalenceResult(True, method="random_testing")

