"""Iterative solvers for convex bilevel problems (see problems.BilevelProblem).

Every solver has the signature
    solver(problem, x0, n_iter, alpha, monitor, **params) -> History
where ``alpha(n)`` (n = 1, 2, ...) is the vanishing outer-step / viscosity sequence and
``monitor(x)`` returns a dict of scalar diagnostics recorded at every iteration.
Iterates are never modified in place.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .problems import BilevelProblem

Array = np.ndarray
Monitor = Callable[[Array], dict]
StepSeq = Callable[[int], float]


@dataclass
class History:
    name: str
    x: Array
    records: list = field(default_factory=list)
    times: list = field(default_factory=list)

    def series(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.records])


def _norm(x: Array) -> float:
    return float(np.linalg.norm(x.ravel()))


def _inertial_coeff(cap: float, budget: float, diff_norm: float) -> float:
    """min{cap, budget/||diff||} (cap if diff = 0): the standard online inertial rule."""
    if diff_norm <= 0.0:
        return cap
    return min(cap, budget / diff_norm)


def _fb(problem: BilevelProblem, x: Array, c: float) -> Array:
    """Forward-backward (proximal gradient) operator T x = prox_{cg}(x - c grad f(x))."""
    return problem.prox_g(x - c * problem.grad_f(x), c)


def _check_inner_step(problem: BilevelProblem, c: float) -> None:
    if not 0.0 < c < 2.0 / problem.Lf:
        raise ValueError(f"inner step c={c} must lie in (0, 2/Lf) = (0, {2.0 / problem.Lf})")


def _run(name: str, step: Callable, x0: Array, n_iter: int, monitor: Monitor) -> History:
    hist = History(name=name, x=x0)
    state = {"x": x0, "x_prev": x0, "x_prev2": x0}
    elapsed = 0.0  # accumulated solver time; monitoring cost is excluded
    for n in range(1, n_iter + 1):
        t0 = time.perf_counter()
        state = step(n, state)
        elapsed += time.perf_counter() - t0
        rec = monitor(state["x"])
        rec["n"] = n
        hist.records.append(rec)
        hist.times.append(elapsed)
    hist.x = state["x"]
    return hist


# --------------------------------------------------------------------------------------
# IPVA: inertial proximal gradient step + proximal gradient step on the outer function
# --------------------------------------------------------------------------------------
def nesterov_cap(n: int) -> float:
    """theta_bar_n = (n - 1)/(n + 2)."""
    return (n - 1.0) / (n + 2.0)


def default_tau(alpha: StepSeq, K: float = 1e4) -> StepSeq:
    """Inertial budget tau_n = K alpha_n / (n+1)^0.01, so that tau_n/alpha_n -> 0."""
    return lambda n: K * alpha(n) / (n + 1) ** 0.01


def dipg_pv(problem: BilevelProblem, x0: Array, n_iter: int, alpha: StepSeq,
            monitor: Monitor, c: float, theta_cap: float | StepSeq = nesterov_cap,
            delta_cap: float | StepSeq = 0.0,
            delta_sign: float = -1.0, tau: StepSeq | None = None,
            name: str = "IPVA") -> History:
    """Algorithm 3.1 of the paper (IPVA); delta_cap > 0 gives a two-step inertial variant.

    v_n     = x_n + theta_n (x_n - x_{n-1}) + delta_n (x_{n-1} - x_{n-2})
    y_n     = prox_{c g}(v_n - c grad f(v_n))
    x_{n+1} = prox_{alpha_n phi}(y_n - alpha_n grad h(y_n))
    with theta_n = min{theta_cap_n, tau_n/||x_n - x_{n-1}||},
         delta_n = delta_sign * min{delta_cap_n, tau_n/||x_{n-1} - x_{n-2}||}.
    """
    _check_inner_step(problem, c)
    tau = tau or default_tau(alpha)
    theta_seq = theta_cap if callable(theta_cap) else (lambda n: theta_cap)
    delta_seq = delta_cap if callable(delta_cap) else (lambda n: delta_cap)

    def step(n: int, s: dict) -> dict:
        x, x1, x2 = s["x"], s["x_prev"], s["x_prev2"]
        d1, d2 = x - x1, x1 - x2
        theta = _inertial_coeff(theta_seq(n), tau(n), _norm(d1))
        delta = delta_sign * _inertial_coeff(delta_seq(n), tau(n), _norm(d2))
        v = x + theta * d1 + delta * d2
        y = _fb(problem, v, c)
        a = alpha(n)
        x_new = problem.prox_phi(y - a * problem.grad_h(y), a)
        return {"x": x_new, "x_prev": x, "x_prev2": x1}

    return _run(name, step, x0, n_iter, monitor)


def bisg(problem: BilevelProblem, x0: Array, n_iter: int, alpha: StepSeq,
         monitor: Monitor, c: float, name: str = "Bi-SG") -> History:
    """Bi-SG, Version II (Merchav and Sabach, 2023): no inertia."""
    return dipg_pv(problem, x0, n_iter, alpha, monitor, c, theta_cap=0.0,
                   delta_cap=0.0, name=name)


ipva = dipg_pv


def fixed_prox_viscosity(problem: BilevelProblem, x0: Array, n_iter: int,
                         alpha: StepSeq, monitor: Monitor, c: float, s: float,
                         name: str = "fixed-step prox-viscosity") -> History:
    """Naive viscosity scheme x_{n+1} = a_n F_s(x_n) + (1 - a_n) T x_n with the
    fixed-parameter proximal contraction F_s = prox_{s phi}(I - s grad h)."""
    _check_inner_step(problem, c)

    def step(n: int, st: dict) -> dict:
        x = st["x"]
        fs = problem.prox_phi(x - s * problem.grad_h(x), s)
        a = alpha(n)
        x_new = a * fs + (1.0 - a) * _fb(problem, x, c)
        return {"x": x_new, "x_prev": x, "x_prev2": st["x_prev"]}

    return _run(name, step, x0, n_iter, monitor)


# --------------------------------------------------------------------------------------
# Methods designed for smooth outer objectives, applied to h + env_{mu phi}
# --------------------------------------------------------------------------------------
def bigsam(problem: BilevelProblem, x0: Array, n_iter: int, alpha: StepSeq,
           monitor: Monitor, c: float, mu: float, lam: float | None = None,
           name: str = "BiG-SAM") -> History:
    """BiG-SAM (Sabach and Shtern, 2017) on the Moreau-smoothed outer objective."""
    _check_inner_step(problem, c)
    grad_w, Lw = problem.smoothed_grad_outer(mu)
    lam = lam if lam is not None else 2.0 / (Lw + problem.sigma)

    def step(n: int, st: dict) -> dict:
        x = st["x"]
        a = alpha(n)
        x_new = a * (x - lam * grad_w(x)) + (1.0 - a) * _fb(problem, x, c)
        return {"x": x_new, "x_prev": x, "x_prev2": st["x_prev"]}

    return _run(name, step, x0, n_iter, monitor)


def ibigsam(problem: BilevelProblem, x0: Array, n_iter: int, alpha: StepSeq,
            monitor: Monitor, c: float, mu: float, lam: float | None = None,
            a_param: float = 3.0, alternated: bool = False,
            name: str = "iBiG-SAM") -> History:
    """iBiG-SAM (Shehu, Vuong and Zemkoho, 2021) on the Moreau-smoothed outer objective.

    With ``alternated=True`` the inertial step is applied only at odd iterations
    (aiBiG-SAM of Duan and Zhang)."""
    _check_inner_step(problem, c)
    grad_w, Lw = problem.smoothed_grad_outer(mu)
    lam = lam if lam is not None else 2.0 / (Lw + problem.sigma)

    def step(n: int, st: dict) -> dict:
        x, x1 = st["x"], st["x_prev"]
        eps = alpha(n) / n ** 0.01
        theta = _inertial_coeff((n - 1.0) / (n + a_param - 1.0), eps, _norm(x - x1))
        if alternated and n % 2 == 0:
            theta = 0.0
        y = x + theta * (x - x1)
        a = alpha(n)
        x_new = a * (y - lam * grad_w(y)) + (1.0 - a) * _fb(problem, y, c)
        return {"x": x_new, "x_prev": x, "x_prev2": x1}

    return _run(name, step, x0, n_iter, monitor)


def dipgm_va(problem: BilevelProblem, x0: Array, n_iter: int, alpha: StepSeq,
             monitor: Monitor, c: float, mu: float, s: float | None = None,
             name: str = "DIPGM-VA") -> History:
    """DIPGM-VA (Suantai and Janngam, 2026) on the Moreau-smoothed outer objective,
    with the parameter rules of their Table 3 (gamma_n = 0.9 n/(n+1),
    mu_n = n/(n+1), rho_n = 1/n^2, tau_n = 1e14/n^2)."""
    _check_inner_step(problem, c)
    grad_w, Lw = problem.smoothed_grad_outer(mu)
    s = s if s is not None else 1.0 / (Lw + problem.sigma)

    def step(n: int, st: dict) -> dict:
        x, z1, z2 = st["x"], st["z_prev"], st["z_prev2"]
        a = alpha(n)
        w = x - a * s * grad_w(x)
        y = _fb(problem, w, c)
        gam = 0.9 * n / (n + 1.0)
        z = (1.0 - gam) * x + gam * _fb(problem, y, c)
        tau_n = 1e14 / n ** 2
        theta = _inertial_coeff(n / (n + 1.0), a * tau_n, _norm(z - z1))
        delta = -_inertial_coeff(1.0 / n ** 2, a * tau_n, _norm(z1 - z2))
        x_new = z + theta * (z - z1) + delta * (z1 - z2)
        return {"x": x_new, "x_prev": x, "x_prev2": st["x_prev"], "z_prev": z, "z_prev2": z1}

    def wrapped_step(n: int, st: dict) -> dict:
        if "z_prev" not in st:
            st = {**st, "z_prev": st["x"], "z_prev2": st["x"]}
        return step(n, st)

    return _run(name, wrapped_step, x0, n_iter, monitor)


def fbipg(problem: BilevelProblem, x0: Array, n_iter: int, monitor: Monitor,
          a0: float = 1.0, delta: float = 1.0, a: float = 2.0,
          name: str = "FBi-PG", theory_step: bool = False, step_scale: float = 1.0) -> History:
    """FBi-PG (Merchav, Sabach and Teboulle): FISTA applied to f + s_k h + g + s_k phi with
    s_k = a0 (k + a)^(-delta), step 1/(L_f + L_h). The scale a0 > 0 is equivalent to applying
    the method to a0*omega (same bilevel solution). Requires g = 0, so that
    prox of g + s_k phi is prox_{s_k phi}; this is checked on the problem name."""
    if problem.name not in ("sparse_recovery", "toy", "elm_group"):
        raise ValueError("this FBi-PG implementation requires g = 0 (explicit prox of g + s_k phi)")
    # theory_step: Lipschitz constant of grad(f + s_0 h), s_0 = a0 a^-delta (the largest s_k),
    # i.e. the step size required when FBi-PG is applied to the scaled outer objective a0*omega.
    ell = (problem.Lf + (a0 * a ** (-delta) if theory_step else 1.0) * problem.Lh) / step_scale

    def step(n: int, st: dict) -> dict:
        k = n - 1  # FBi-PG counts k = 0, 1, ...
        x, x1 = st["x"], st["x_prev"]
        t_prev, t = st.get("t_prev", 0.0), st.get("t", 1.0)
        s_k = a0 * (k + a) ** (-delta)
        y = x + ((t_prev - 1.0) / t) * (x - x1)
        grad = problem.grad_f(y) + s_k * problem.grad_h(y)
        x_new = problem.prox_phi(y - grad / ell, s_k / ell)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        return {"x": x_new, "x_prev": x, "x_prev2": x1, "t_prev": t, "t": t_new}

    return _run(name, step, x0, n_iter, monitor)


def fbipg_lifted(problem: BilevelProblem, x0: Array, n_iter: int, monitor: Monitor,
                 a0: float = 1.0, delta: float = 1.0, a: float = 2.0,
                 name: str = "FBi-PG", theory_step: bool = False, step_scale: float = 1.0) -> History:
    """FBi-PG applied to the lifted reformulation of Merchav, Sabach and Teboulle
    (Remark 2): variables w = (x, z), inner f(x) + g(x) + ||x - z||^2/2, outer h(x) + phi(z),
    so that only the separate proximal maps of g and phi are needed. Step 1/(L_f + 2 + L_h).
    The monitor and the returned iterate refer to the x-component."""
    ell = (problem.Lf + 2.0 + (a0 * a ** (-delta) if theory_step else 1.0) * problem.Lh) / step_scale
    hist = History(name=name, x=x0)
    x, xp, z, zp = x0, x0, x0, x0
    t_prev, t = 0.0, 1.0
    elapsed = 0.0
    for n in range(1, n_iter + 1):
        t0 = time.perf_counter()
        k = n - 1
        s_k = a0 * (k + a) ** (-delta)
        beta = (t_prev - 1.0) / t
        yx, yz = x + beta * (x - xp), z + beta * (z - zp)
        gx = problem.grad_f(yx) + (yx - yz) + s_k * problem.grad_h(yx)
        gz = yz - yx
        x_new = problem.prox_g(yx - gx / ell, 1.0 / ell)
        z_new = problem.prox_phi(yz - gz / ell, s_k / ell)
        xp, x, zp, z = x, x_new, z, z_new
        t_prev, t = t, (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        elapsed += time.perf_counter() - t0
        rec = monitor(x)
        rec["n"] = n
        hist.records.append(rec)
        hist.times.append(elapsed)
    hist.x = x
    return hist
