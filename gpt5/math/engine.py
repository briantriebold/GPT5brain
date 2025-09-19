from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from numpy.typing import ArrayLike
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from scipy import optimize, integrate


def evaluate_expression(expr: str, variables: Dict[str, float] | None = None) -> float:
    variables = variables or {}
    sym_vars = {sp.Symbol(k): float(v) for k, v in variables.items()}
    e = parse_expr(expr, evaluate=True)
    return float(e.evalf(subs=sym_vars))


def solve_linear(A: ArrayLike, b: ArrayLike) -> np.ndarray:
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    return np.linalg.solve(A, b)


def integrate_definite(expr: str, var: str, a: float, b: float) -> float:
    x = sp.Symbol(var)
    e = parse_expr(expr, evaluate=True)
    f = sp.lambdify(x, e, modules=["numpy"])
    val, _ = integrate.quad(lambda t: float(f(t)), a, b)
    return float(val)


def solve_ode(rhs_expr: str, t_span: Tuple[float, float], y0: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    t = sp.Symbol('t')
    y = sp.Symbol('y')
    e = parse_expr(rhs_expr, evaluate=True)
    f = sp.lambdify((t, y), e, modules=["numpy"])

    def fun(_t, _y):  # noqa: ANN001
        return np.asarray([f(_t, _y[0])])

    sol = integrate.solve_ivp(fun, t_span, y0, dense_output=True)
    return sol.t, sol.y


def minimize_expression(expr: str, vars_order: List[str], x0: List[float]) -> Tuple[np.ndarray, float]:
    syms = sp.symbols(vars_order)
    e = parse_expr(expr, evaluate=True)
    f = sp.lambdify(syms, e, modules=["numpy"])

    def fun(x):  # noqa: ANN001
        return float(f(*x))

    res = optimize.minimize(fun, np.asarray(x0, dtype=float), method="L-BFGS-B")
    return res.x, float(res.fun)

