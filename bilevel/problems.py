"""Convex bilevel problems  min { h(x) + phi(x) : x in argmin (f + g) }.

A problem is an immutable bundle of oracles:
  f, grad_f, Lf        smooth inner part (gradient Lf-Lipschitz)
  g, prox_g            nonsmooth inner part (prox_g(x, t) = prox_{t g}(x))
  h, grad_h, sigma, Lh smooth, sigma-strongly convex outer part
  phi, prox_phi        nonsmooth convex outer part (prox_phi(x, t) = prox_{t phi}(x))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from scipy.fft import dct, idct

from .operators import (
    dct2,
    group_soft_threshold_rows,
    project_box,
    prox_l1_orthonormal,
    soft_threshold,
)

Array = np.ndarray


@dataclass(frozen=True)
class BilevelProblem:
    name: str
    f: Callable[[Array], float]
    grad_f: Callable[[Array], Array]
    Lf: float
    g: Callable[[Array], float]
    prox_g: Callable[[Array, float], Array]
    h: Callable[[Array], float]
    grad_h: Callable[[Array], Array]
    sigma: float
    Lh: float
    phi: Callable[[Array], float]
    prox_phi: Callable[[Array, float], Array]

    def inner(self, x: Array) -> float:
        return self.f(x) + self.g(x)

    def outer(self, x: Array) -> float:
        return self.h(x) + self.phi(x)

    def smoothed_grad_outer(self, mu: float) -> tuple[Callable[[Array], Array], float]:
        """Gradient of h + env_{mu phi} (Moreau envelope) and its Lipschitz constant."""
        if mu <= 0:
            raise ValueError("smoothing parameter must be positive")

        def grad(x: Array) -> Array:
            return self.grad_h(x) + (x - self.prox_phi(x, mu)) / mu

        return grad, self.Lh + 1.0 / mu


def _zero(_: Array) -> float:
    return 0.0


def _identity_prox(x: Array, _: float) -> Array:
    return x


def _box_indicator(lo: float, hi: float) -> Callable[[Array], float]:
    def val(x: Array) -> float:
        inside = np.all(x >= lo - 1e-12) and np.all(x <= hi + 1e-12)
        return 0.0 if inside else np.inf

    return val


def toy_problem(tau: float) -> BilevelProblem:
    """A small example in R^2.

    Inner: f(x) = 0.5 (x1 + x2 - 2)^2, g = 0, so S = {x : x1 + x2 = 2}.
    Outer: h(x) = 0.5 ||x||^2, phi(x) = tau |x1|.
    For tau >= 2 the bilevel solution is x* = (0, 2).
    """
    a = np.array([1.0, 1.0])
    return BilevelProblem(
        name="toy",
        f=lambda x: 0.5 * float(a @ x - 2.0) ** 2,
        grad_f=lambda x: (a @ x - 2.0) * a,
        Lf=2.0,
        g=_zero,
        prox_g=_identity_prox,
        h=lambda x: 0.5 * float(x @ x),
        grad_h=lambda x: x.copy(),
        sigma=1.0,
        Lh=1.0,
        phi=lambda x: tau * abs(float(x[0])),
        prox_phi=lambda x, t: np.array([soft_threshold(x[:1], tau * t)[0], x[1]]),
    )


def sparse_ls_problem(A: Array, b: Array, sigma: float, tau: float) -> BilevelProblem:
    """Inner: min 0.5||Ax - b||^2 + indicator(x >= 0);  outer: sigma/2 ||x||^2 + tau ||x||_1."""
    Lf = float(np.linalg.norm(A, 2) ** 2)
    return BilevelProblem(
        name="sparse_ls",
        f=lambda x: 0.5 * float(np.sum((A @ x - b) ** 2)),
        grad_f=lambda x: A.T @ (A @ x - b),
        Lf=Lf,
        g=lambda x: 0.0 if np.all(x >= -1e-12) else np.inf,
        prox_g=lambda x, t: np.maximum(x, 0.0),
        h=lambda x: 0.5 * sigma * float(x @ x),
        grad_h=lambda x: sigma * x,
        sigma=sigma,
        Lh=sigma,
        phi=lambda x: tau * float(np.sum(np.abs(x))),
        prox_phi=lambda x, t: soft_threshold(x, tau * t),
    )


def inpainting_problem(mask: Array, observed: Array, sigma: float, tau: float) -> BilevelProblem:
    """Inner: min 0.5||M(x) - b||^2 + indicator([0,1]^n);  outer: sigma/2||x||^2 + tau||W x||_1."""
    mask = mask.astype(float)
    b = observed * mask
    return BilevelProblem(
        name="inpainting",
        f=lambda x: 0.5 * float(np.sum((mask * x - b) ** 2)),
        grad_f=lambda x: mask * (mask * x - b),
        Lf=1.0,
        g=_box_indicator(0.0, 1.0),
        prox_g=lambda x, t: project_box(x, 0.0, 1.0),
        h=lambda x: 0.5 * sigma * float(np.sum(x * x)),
        grad_h=lambda x: sigma * x,
        sigma=sigma,
        Lh=sigma,
        phi=lambda x: tau * float(np.sum(np.abs(dct2(x)))),
        prox_phi=lambda x, t: prox_l1_orthonormal(x, tau * t),
    )


def elm_group_problem(H: Array, T: Array, sigma: float, tau: float) -> BilevelProblem:
    """Inner: min 0.5||H B - T||_F^2;  outer: sigma/2 ||B||_F^2 + tau sum_i ||B[i,:]||_2."""
    Lf = float(np.linalg.norm(H, 2) ** 2)
    return BilevelProblem(
        name="elm_group",
        f=lambda B: 0.5 * float(np.sum((H @ B - T) ** 2)),
        grad_f=lambda B: H.T @ (H @ B - T),
        Lf=Lf,
        g=_zero,
        prox_g=_identity_prox,
        h=lambda B: 0.5 * sigma * float(np.sum(B * B)),
        grad_h=lambda B: sigma * B,
        sigma=sigma,
        Lh=sigma,
        phi=lambda B: tau * float(np.sum(np.linalg.norm(B, axis=1))),
        prox_phi=lambda B, t: group_soft_threshold_rows(B, tau * t),
    )


def sparse_recovery_problem(A: Array, b: Array, sigma: float, tau: float = 1.0) -> BilevelProblem:
    """Inner: min 0.5||Ax - b||^2 (g = 0);  outer: sigma/2 ||x||^2 + tau ||x||_1.

    For consistent underdetermined systems S = {x : Ax = b}; for small sigma the
    bilevel solution is a basis-pursuit solution (exact regularization)."""
    Lf = float(np.linalg.norm(A, 2) ** 2)
    return BilevelProblem(
        name="sparse_recovery",
        f=lambda x: 0.5 * float(np.sum((A @ x - b) ** 2)),
        grad_f=lambda x: A.T @ (A @ x - b),
        Lf=Lf,
        g=_zero,
        prox_g=_identity_prox,
        h=lambda x: 0.5 * sigma * float(x @ x),
        grad_h=lambda x: sigma * x,
        sigma=sigma,
        Lh=sigma,
        phi=lambda x: tau * float(np.sum(np.abs(x))),
        prox_phi=lambda x, t: soft_threshold(x, tau * t),
    )


def nonneg_dct_problem(A: Array, b: Array, sigma: float, tau: float = 1.0) -> BilevelProblem:
    """Inner: min 0.5||Ax - b||^2 + indicator(x >= 0);  outer: sigma/2 ||x||^2 + tau ||W x||_1, W the orthonormal DCT.

    prox_g is the projection onto the nonnegative orthant and prox_phi is W^T soft(W x, tau t). The proximal
    mapping of g + t phi has no closed form, so FBi-PG needs the lifted reformulation."""
    Lf = float(np.linalg.norm(A, 2) ** 2)
    return BilevelProblem(
        name="nonneg_dct",
        f=lambda x: 0.5 * float(np.sum((A @ x - b) ** 2)),
        grad_f=lambda x: A.T @ (A @ x - b),
        Lf=Lf,
        g=lambda x: 0.0 if bool(np.all(x >= 0.0)) else float("inf"),
        prox_g=lambda x, t: np.maximum(x, 0.0),
        h=lambda x: 0.5 * sigma * float(x @ x),
        grad_h=lambda x: sigma * x,
        sigma=sigma,
        Lh=sigma,
        phi=lambda x: tau * float(np.sum(np.abs(dct(x, norm="ortho")))),
        prox_phi=lambda x, t: idct(soft_threshold(dct(x, norm="ortho"), tau * t), norm="ortho"),
    )


def project_simplex(x: Array) -> Array:
    """Euclidean projection onto {x >= 0, sum(x) = 1} (sorting algorithm of Duchi et al., 2008)."""
    u = np.sort(x)[::-1]
    css = np.cumsum(u) - 1.0
    k = np.arange(1, x.size + 1)
    rho = int(np.nonzero(u - css / k > 0)[0][-1])
    return np.maximum(x - css[rho] / (rho + 1.0), 0.0)


def _ls_dct_problem(name: str, A: Array, b: Array, sigma: float, tau: float, g, prox_g) -> BilevelProblem:
    return BilevelProblem(
        name=name,
        f=lambda x: 0.5 * float(np.sum((A @ x - b) ** 2)),
        grad_f=lambda x: A.T @ (A @ x - b),
        Lf=float(np.linalg.norm(A, 2) ** 2),
        g=g,
        prox_g=prox_g,
        h=lambda x: 0.5 * sigma * float(x @ x),
        grad_h=lambda x: sigma * x,
        sigma=sigma,
        Lh=sigma,
        phi=lambda x: tau * float(np.sum(np.abs(dct(x, norm="ortho")))),
        prox_phi=lambda x, t: idct(soft_threshold(dct(x, norm="ortho"), tau * t), norm="ortho"),
    )


def box_dct_problem(A: Array, b: Array, sigma: float, tau: float = 1.0) -> BilevelProblem:
    """Inner: min 0.5||Ax - b||^2 + indicator(0 <= x <= 1);  outer: sigma/2 ||x||^2 + tau ||W x||_1 (W: DCT)."""
    return _ls_dct_problem("box_dct", A, b, sigma, tau, _box_indicator(0.0, 1.0), lambda x, t: np.clip(x, 0.0, 1.0))


def simplex_dct_problem(A: Array, b: Array, sigma: float, tau: float = 1.0) -> BilevelProblem:
    """Inner: min 0.5||Ax - b||^2 + indicator(simplex);  outer: sigma/2 ||x||^2 + tau ||W x||_1 (W: DCT)."""
    def ind(x: Array) -> float:
        return 0.0 if (np.all(x >= -1e-12) and abs(float(np.sum(x)) - 1.0) <= 1e-9) else float("inf")

    return _ls_dct_problem("simplex_dct", A, b, sigma, tau, ind, lambda x, t: project_simplex(x))
