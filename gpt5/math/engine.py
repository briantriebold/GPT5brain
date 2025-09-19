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


def fft_real(data: ArrayLike, sample_rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(data, dtype=float)
    n = x.size
    yf = np.fft.rfft(x)
    xf = np.fft.rfftfreq(n, d=1.0 / sample_rate if sample_rate > 0 else 1.0)
    mag = np.abs(yf) / max(1, n)
    return xf, mag


def polyfit_fit(x: ArrayLike, y: ArrayLike, deg: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    coeffs = np.polyfit(x, y, deg)
    return coeffs


def solve_root(expr: str, var: str, x0: float | None = None, a: float | None = None, b: float | None = None) -> float:
    # Try root finding via scipy with either bracket or initial guess
    sym = sp.Symbol(var)
    e = parse_expr(expr, evaluate=True)
    f = sp.lambdify(sym, e, modules=["numpy"])
    if a is not None and b is not None:
        res = optimize.root_scalar(lambda t: float(f(t)), bracket=(a, b), method="brentq")
        if not res.converged:
            raise RuntimeError("root finding did not converge with bracket")
        return float(res.root)
    if x0 is None:
        x0 = 0.0
    # Try Newton with derivative if possible
    try:
        de = sp.diff(e, sym)
        df = sp.lambdify(sym, de, modules=["numpy"])
        res = optimize.root_scalar(lambda t: float(f(t)), fprime=lambda t: float(df(t)), x0=x0, method="newton")
        if res.converged:
            return float(res.root)
    except Exception:
        pass
    # Fallback secant
    res = optimize.root_scalar(lambda t: float(f(t)), x0=x0, x1=x0 + 1.0, method="secant")
    if not res.converged:
        raise RuntimeError("root finding did not converge")
    return float(res.root)
