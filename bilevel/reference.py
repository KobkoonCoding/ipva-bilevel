"""Independent high-accuracy reference solver for the inpainting bilevel problem.

The bilevel problem  min { sigma/2||x||^2 + tau||Wx||_1 : x in S },
S = {x in [0,1]^n : x_Omega = b_Omega}, is written as
    min_{x,u} G(x) + tau||u||_1  s.t.  Wx = u,   G = sigma/2||.||^2 + indicator_S,
and solved by ADMM (scaled form). Because W is orthonormal, both subproblems
have closed forms: x = prox_{G/rho}(W^T(u - lam)) and u = soft(Wx + lam, tau/rho).
"""
from __future__ import annotations

import numpy as np

from .operators import dct2, idct2, soft_threshold


def inpainting_reference(mask: np.ndarray, observed: np.ndarray, sigma: float, tau: float,
                         rho: float = 1.0, max_iter: int = 100000,
                         tol: float = 1e-12) -> tuple[np.ndarray, dict]:
    if sigma <= 0 or tau <= 0 or rho <= 0:
        raise ValueError("sigma, tau and rho must be positive")
    m = mask.astype(bool)
    b = np.where(m, observed, 0.0)

    def prox_G(z: np.ndarray) -> np.ndarray:
        x = np.clip(z / (1.0 + sigma / rho), 0.0, 1.0)
        return np.where(m, b, x)

    x = prox_G(np.zeros_like(b))
    u = dct2(x)
    lam = np.zeros_like(u)
    info = {"iterations": max_iter, "primal_res": np.inf, "dual_res": np.inf}
    for k in range(1, max_iter + 1):
        x = prox_G(idct2(u - lam))
        wx = dct2(x)
        u_old = u
        u = soft_threshold(wx + lam, tau / rho)
        lam = lam + wx - u
        scale = max(float(np.linalg.norm(wx)), 1e-300)
        r_p = float(np.linalg.norm(wx - u)) / scale
        r_d = rho * float(np.linalg.norm(u - u_old)) / scale
        if r_p < tol and r_d < tol:
            info = {"iterations": k, "primal_res": r_p, "dual_res": r_d}
            break
        info = {"iterations": k, "primal_res": r_p, "dual_res": r_d}
    return x, info
