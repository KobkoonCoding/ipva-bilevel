"""IPVA and the five compared methods, and the coordinate search that selected their parameters.

Every method is evaluated only with parameters inside the range covered by the convergence theory of its own paper.
The inner function is nonsmooth (g != 0, an indicator), so the compared methods use smoothing (Moreau envelope of phi) or lifting.

PARAMETER RANGES  (n = 1, 2, ... is the iteration counter; Lf, Lh Lipschitz constants of grad f, grad h)
--------------------------------------------------------------------------------------------------------
IPVA-1   (IPVA under Theorem 3.5 of the paper; any c in (0, 2/Lf), alpha_n -> 0, sum alpha_n = infinity)
    alpha_n = a0/(n+1)^delta,  delta in {0.6, 0.7, 0.8, 0.9, 1.0};  c in {0.5, 1, 1.5, 1.9}/Lf;
    inertia theta_n = min{ (n-1)/(n+2), tau_n/||x_n - x_{n-1}|| },  tau_n = K alpha_n (n+1)^(-0.01)
    (so tau_n/alpha_n -> 0, condition (C2)),  K in {1e2, 1e4}.  a0 is the continuous scale.
Bi-SG    (Merchav-Sabach 2023, Version II = IPVA without inertia)
    eta_k = c (k+1)^(-alpha), alpha in {0.55, 0.65, 0.75, 0.85, 0.95, 0.99} (subset of (1/2, 1)), c in (0, min{1,1/Lh}]
    (c is the continuous scale, hard cap c <= 1); inner step t_k = 1/Lf (its constant-step rule).
BiG-SAM, iBiG-SAM, DIPGM-VA   (applied to h + Moreau envelope of phi with fixed mu)
    mu in {1e-1, 1e-2, 1e-3, 1e-4, 1e-5};  alpha_n = a0/(n+1)^delta, alpha_n -> 0, sum = infinity, delta in
    {0.55, 0.7, 0.85, 1.0} (exponent in (1/2, 1]); alpha_n <= 1 is required by the convex-combination form, hard cap
    alpha_1 <= 0.95 (alpha_1 is the continuous scale); inner step 1/Lf; outer step 2/(Lw+sigma) for BiG-SAM and iBiG-SAM,
    1/(Lw+sigma) for DIPGM-VA, Lw = Lh + 1/mu; inertial and relaxation rules of DIPGM-VA as in Table 3 of its paper.
FBi-PG   (Merchav-Sabach-Teboulle, lifted reformulation for g != 0; Theorems 4 and 5)
    beta_j = a0 (j+a)^(-gamma), j = 0, 1, ..., gamma in {1.1, 1.25, 1.5, 1.75} (range 1 < gamma < 2 of Thms 4, 5),
    integer a in {3, 4, 5} (Thm 5 needs a > 2); extrapolation (k-1)/(k+a) = (t_{k-1}-1)/t_k with the paper's
    t_k = (k+a)/a (eq. 3.12); step 1/(Lf + 2 + Lh) (the lifted smooth part has Lipschitz constant Lf + 2 + beta_j Lh;
    when a0 a^(-gamma) > 1 we use 1/(Lf + 2 + a0 a^(-gamma) Lh) so that the step is always admissible).

SEARCH (identical for every method; method-specific only through the sets above)
------------------------------------------------------------------------------
* A configuration is a point of (discrete sets) x (continuous scale 'level').  The scale is anchored so that the grid is
  meaningful for every exponent: level = alpha_100 (IPVA-1), beta_99 (FBi-PG), eta-scale c (Bi-SG), alpha_1 (smoothing
  methods); a0 is a bijective function of (level, discrete parameters).
* Budget: every method gets exactly B = 60 evaluations per iteration budget N (one evaluation = one complete run of N
  iterations on the tuning image).  Boundary-extension evaluations are part of the 60.
* Cyclic coordinate search: (1) scale scan at the middle element of every discrete set, 13 values on a decade grid
  (10^-10 ... 10^2, or cap*10^-j, j = 0..12 for capped scales); (2) one-at-a-time sweep of every discrete set at the
  incumbent (all values of the set, level clipped to the cap if the cap depends on the discrete values);
  (3) level refinement incumbent * s^(+-1) with s = sqrt(10), 10^(1/4), ... down to s = 1.02; (2)-(3) repeat;
  when s = 1.02 is reached, all unvisited combinations of the discrete sets are tried at the incumbent level
  (nearest to the incumbent first; an improvement restarts the refinement), and the remaining budget is spent on a dense
  local level scan (ratio 1.02) around the incumbent, until exactly 60 evaluations are used
  (ties broken by first evaluation; diverged runs have error inf).
* Boundary rule: an uncapped scale whose best value sits at the end of the scanned decade range is extended
  by two decades on that side (up to 3 times; evaluations counted in the 60).  A value at the end of a discrete set
  or at a hard cap (c <= 1, alpha_1 <= 0.95) is never extended; it is reported as a 'hard' boundary hit.
* Tuning criterion: final-iterate relative error ||x_N - x*|| / ||x*|| on one separate tuning image (Shepp-Logan phantom).
"""
from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass
from typing import Callable

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

BUDGET = 60                     # evaluations per method and iteration budget N
N_REF = 100                     # anchor iteration of the 'level' of IPVA / FBi-PG
STEP0 = math.sqrt(10.0)         # first level-refinement ratio
STEP_FLOOR = 1.02               # finest level-refinement ratio
SCAN_UNCAPPED = tuple(10.0 ** j for j in range(-10, 3))   # 13 values
EXT_MAX = 3                     # boundary extensions of the scan (two decades each)


# ----------------------------------------------------------------------------------------
# Method specifications
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Method:
    name: str
    family: str          # ipva1 | bisg | bigsam | ibigsam | dipgm | fbipg
    discrete: tuple      # ((axis, values), ...) in sweep order
    capped: bool         # True: the scale has a hard theory cap

    def axes(self) -> dict:
        return dict(self.discrete)


def method_specs() -> dict[str, Method]:
    ms = [
        Method("IPVA-1", "ipva1", (("delta", (0.6, 0.7, 0.8, 0.9, 1.0)), ("c_mult", (0.5, 1.0, 1.5, 1.9)),
                                   ("K", (1e2, 1e4))), False),
        Method("Bi-SG", "bisg", (("alpha", (0.55, 0.65, 0.75, 0.85, 0.95, 0.99)),), True),
        Method("BiG-SAM", "bigsam", (("delta", (0.55, 0.7, 0.85, 1.0)), ("mu", (1e-1, 1e-2, 1e-3, 1e-4, 1e-5))), True),
        Method("iBiG-SAM", "ibigsam", (("delta", (0.55, 0.7, 0.85, 1.0)), ("mu", (1e-1, 1e-2, 1e-3, 1e-4, 1e-5))), True),
        Method("DIPGM-VA", "dipgm", (("delta", (0.55, 0.7, 0.85, 1.0)), ("mu", (1e-1, 1e-2, 1e-3, 1e-4, 1e-5))), True),
        Method("FBi-PG", "fbipg", (("gamma", (1.1, 1.25, 1.5, 1.75)), ("a", (3.0, 4.0, 5.0))), False),
    ]
    return {m.name: m for m in ms}


def level_cap(m: Method, prm: dict, prob) -> float | None:
    """Hard theory cap of the scale (None if the scale is free)."""
    if m.family == "bisg":
        return min(1.0, 1.0 / prob.Lh)                                       # c <= min{1, 1/Lh}
    if m.family in ("bigsam", "ibigsam", "dipgm"):
        return 0.95                                                          # alpha_1 <= 0.95 (< 1)
    return None


def a0_of(m: Method, prm: dict) -> float:
    """Scale a0 of the sequence from the anchored level."""
    L = prm["level"]
    f = m.family
    if f == "ipva1":
        return L * (N_REF + 1.0) ** prm["delta"]
    if f == "bisg":
        return L
    if f in ("bigsam", "ibigsam", "dipgm"):
        return L * 2.0 ** prm["delta"]
    if f == "fbipg":
        return L * (N_REF - 1.0 + prm["a"]) ** prm["gamma"]
    raise ValueError(f)


# ----------------------------------------------------------------------------------------
# Solvers (own implementations; they also expose the feasible inner iterate y_n)
# ----------------------------------------------------------------------------------------
def _nrm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x.ravel()))


def _inertia(cap: float, budget: float, diff: float) -> float:
    return cap if diff <= 0.0 else min(cap, budget / diff)


def run_method(prob, m: Method, prm: dict, n_iter: int, x0: np.ndarray, monitor: Callable | None = None,
               at=frozenset()) -> dict:
    """Run n_iter iterations.  monitor(n, x, y, elapsed) is called at n in ``at`` (not timed).

    Returns {'x': x_N, 'y': y_N, 'time': solver seconds, 'diverged': bool}."""
    fam = m.family
    a0 = a0_of(m, prm)
    fgrad, pg, ph, gh = prob.grad_f, prob.prox_g, prob.prox_phi, prob.grad_h
    elapsed = 0.0
    x, x1, y = x0, x0, x0
    diverged = False
    clock = time.perf_counter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            if fam in ("ipva1", "bisg"):
                c = (prm.get("c_mult", 1.0)) / prob.Lf if fam != "bisg" else 1.0 / prob.Lf
                if fam == "ipva1":
                    d_, K = prm["delta"], prm["K"]

                    def alpha(n):
                        return a0 / (n + 1.0) ** d_
                else:
                    al_ = prm["alpha"]

                    def alpha(n):
                        return a0 / (n + 1.0) ** al_
                for n in range(1, n_iter + 1):
                    t0 = clock()
                    d = x - x1
                    if fam == "ipva1":
                        a = alpha(n)
                        th = _inertia((n - 1.0) / (n + 2.0), K * a / (n + 1.0) ** 0.01, _nrm(d))
                    else:
                        th = 0.0
                    v = x + th * d if th != 0.0 else x
                    y = pg(v - c * fgrad(v), c)
                    a = alpha(n)
                    xn = ph(y - a * gh(y), a)
                    x1, x = x, xn
                    elapsed += clock() - t0
                    if n in at:
                        monitor(n, x, y, elapsed)
                    if n % 50 == 0 and not np.isfinite(x).all():
                        diverged = True
                        break
            elif fam in ("bigsam", "ibigsam"):
                mu, d_ = prm["mu"], prm["delta"]
                Lw = prob.Lh + 1.0 / mu
                lam = 2.0 / (Lw + prob.sigma)
                c = 1.0 / prob.Lf
                for n in range(1, n_iter + 1):
                    t0 = clock()
                    a = a0 / (n + 1.0) ** d_
                    if fam == "ibigsam":
                        th = _inertia((n - 1.0) / (n + 2.0), a / n ** 0.01, _nrm(x - x1))
                        z = x + th * (x - x1)
                    else:
                        z = x
                    y = pg(z - c * fgrad(z), c)
                    gw = gh(z) + (z - ph(z, mu)) / mu
                    xn = a * (z - lam * gw) + (1.0 - a) * y
                    x1, x = x, xn
                    elapsed += clock() - t0
                    if n in at:
                        monitor(n, x, y, elapsed)
                    if n % 50 == 0 and not np.isfinite(x).all():
                        diverged = True
                        break
            elif fam == "dipgm":
                mu, d_ = prm["mu"], prm["delta"]
                Lw = prob.Lh + 1.0 / mu
                s = 1.0 / (Lw + prob.sigma)
                c = 1.0 / prob.Lf
                z1, z2 = x0, x0
                for n in range(1, n_iter + 1):
                    t0 = clock()
                    a = a0 / (n + 1.0) ** d_
                    gw = gh(x) + (x - ph(x, mu)) / mu
                    w = x - a * s * gw
                    yy = pg(w - c * fgrad(w), c)
                    gam = 0.9 * n / (n + 1.0)
                    y = pg(yy - c * fgrad(yy), c)
                    z = (1.0 - gam) * x + gam * y
                    tau_n = 1e14 / n ** 2
                    th = _inertia(n / (n + 1.0), a * tau_n, _nrm(z - z1))
                    de = -_inertia(1.0 / n ** 2, a * tau_n, _nrm(z1 - z2))
                    xn = z + th * (z - z1) + de * (z1 - z2)
                    x1, x = x, xn
                    z2, z1 = z1, z
                    elapsed += clock() - t0
                    if n in at:
                        monitor(n, x, y, elapsed)
                    if n % 50 == 0 and not np.isfinite(x).all():
                        diverged = True
                        break
            elif fam == "fbipg":
                gm, a_ = prm["gamma"], prm["a"]
                s0 = a0 * a_ ** (-gm)
                ell = prob.Lf + 2.0 + max(1.0, s0) * prob.Lh
                z, zp = x0, x0
                xp = x0
                for n in range(1, n_iter + 1):
                    t0 = clock()
                    k = n - 1
                    s_k = a0 * (k + a_) ** (-gm)
                    beta = (k - 1.0) / (k + a_) if k >= 1 else 0.0
                    yx = x + beta * (x - xp)
                    yz = z + beta * (z - zp)
                    gx = fgrad(yx) + (yx - yz) + s_k * gh(yx)
                    xn = pg(yx - gx / ell, 1.0 / ell)
                    zn = ph(yz - (yz - yx) / ell, s_k / ell)
                    xp, x = x, xn
                    zp, z = z, zn
                    y = x
                    elapsed += clock() - t0
                    if n in at:
                        monitor(n, x, y, elapsed)
                    if n % 50 == 0 and not np.isfinite(x).all():
                        diverged = True
                        break
            else:
                raise ValueError(fam)
    return {"x": x, "y": y, "time": elapsed, "diverged": diverged}


# ----------------------------------------------------------------------------------------
# Search: cyclic coordinate search with a fixed evaluation budget
# ----------------------------------------------------------------------------------------
class _Exhausted(Exception):
    pass


def _key(prm: dict) -> tuple:
    return tuple(sorted((k, float("%.10g" % v) if k == "level" else float(v)) for k, v in prm.items()))


def search(m: Method, prob, evaluate: Callable[[dict], float], budget: int = BUDGET) -> dict:
    """Tune ``m`` by cyclic coordinate search with exactly ``budget`` evaluations.

    evaluate(prm) -> final relative error (inf if diverged).  Returns trials, incumbent and boundary hits."""
    axes = m.axes()
    trials: list[dict] = []
    seen: dict[tuple, float] = {}

    def clip_level(prm: dict, level: float) -> tuple[float, bool]:
        cap = level_cap(m, prm, prob)
        if cap is not None and level > cap * (1 + 1e-12):
            return cap, True
        return level, False

    def ev(prm: dict) -> float:
        k = _key(prm)
        if k in seen:
            return seen[k]
        if len(trials) >= budget:
            raise _Exhausted
        e = float(evaluate(prm))
        if not math.isfinite(e):
            e = float("inf")
        seen[k] = e
        trials.append({**prm, "err": e})
        return e

    def incumbent() -> dict:
        b = min(trials, key=lambda r: r["err"])     # first minimum on ties
        return {k: v for k, v in b.items() if k != "err"}

    start = {ax: vals[len(vals) // 2] for ax, vals in axes.items()}
    cap0 = level_cap(m, start, prob)
    scan = ([cap0 * 10.0 ** (-j) for j in range(13)] if cap0 is not None else list(SCAN_UNCAPPED))
    ext_log = []
    try:
        for L in scan:
            ev({**start, "level": L})
        if cap0 is None:                                           # boundary extension of the free scale
            for _ in range(EXT_MAX):
                lv = [t["level"] for t in trials]
                b = incumbent()
                if math.isclose(b["level"], min(lv), rel_tol=1e-9):
                    new = [min(lv) * 10.0 ** (-j) for j in (1, 2)]
                    side = "lo"
                elif math.isclose(b["level"], max(lv), rel_tol=1e-9):
                    new = [max(lv) * 10.0 ** j for j in (1, 2)]
                    side = "hi"
                else:
                    break
                ext_log.append({"side": side, "values": new})
                for L in new:
                    ev({**start, "level": L})
        step = STEP0
        stall = 0
        tried_pairs = False
        while True:
            n0 = len(trials)
            best_before = min(t["err"] for t in trials)
            for ax, vals in axes.items():
                base = incumbent()
                for v in vals:
                    if v == base[ax]:
                        continue
                    cand = {**base, ax: v}
                    cand["level"], _ = clip_level(cand, base["level"])
                    ev(cand)
            base = incumbent()
            if step > STEP_FLOOR:
                for sgn in (1, -1):                      # level refinement at ratio `step`
                    cand = dict(base)
                    cand["level"], hit = clip_level(cand, base["level"] * step ** sgn)
                    if not hit:
                        ev(cand)
                step = max(step ** 0.5, STEP_FLOOR)
            else:
                if not tried_pairs:                      # resolution floor reached: explore discrete interactions
                    tried_pairs = True
                    import itertools
                    combos = [dict(zip(axes, c)) for c in itertools.product(*axes.values())]
                    combos.sort(key=lambda c: sum(c[ax] != base[ax] for ax in axes))
                    for c in combos:
                        cand = {**base, **c}
                        cand["level"], _ = clip_level(cand, base["level"])
                        if _key(cand) in seen:
                            continue
                        ev(cand)
                        if min(t["err"] for t in trials) < best_before:
                            tried_pairs = False          # improved: refine again at the new incumbent
                            step = STEP0
                            break
                else:                                    # dense local refinement around the incumbent (ratio STEP_FLOOR)
                    for sgn in (1, -1):
                        j = 1
                        while j < 200:
                            cand = dict(base)
                            cand["level"], hit = clip_level(cand, base["level"] * STEP_FLOOR ** (sgn * j))
                            if hit:
                                break
                            if _key(cand) not in seen:
                                ev(cand)
                                break
                            j += 1
            stall = stall + 1 if len(trials) == n0 else 0
            if stall >= 5:
                break
    except _Exhausted:
        pass
    best = incumbent()
    hits = []
    for ax, vals in axes.items():
        if len(vals) > 1 and best[ax] in (min(vals), max(vals)):
            hits.append({"axis": ax, "side": "lo" if best[ax] == min(vals) else "hi", "kind": "hard (end of the set)"})
    cap = level_cap(m, best, prob)
    lv = [t["level"] for t in trials]
    if cap is not None and math.isclose(best["level"], cap, rel_tol=1e-9):
        hits.append({"axis": "level", "side": "hi", "kind": "hard (cap)"})
    elif cap is None and math.isclose(best["level"], max(lv), rel_tol=1e-9) and len(set(lv)) > 1:
        hits.append({"axis": "level", "side": "hi", "kind": "extendable-unresolved"})
    if math.isclose(best["level"], min(lv), rel_tol=1e-9) and len(set(lv)) > 1:
        hits.append({"axis": "level", "side": "lo", "kind": "extendable-unresolved"})
    return {"trials": trials, "best": best, "best_err": min(t["err"] for t in trials), "n_evals": len(trials),
            "hits": hits, "extensions": ext_log}
