import numpy as np
import pytest
from scipy.optimize import minimize

from bilevel.operators import (
    dct2,
    group_soft_threshold_rows,
    idct2,
    prox_l1_orthonormal,
    soft_threshold,
)
from bilevel.problems import elm_group_problem, sparse_ls_problem, toy_problem
from bilevel.solvers import bigsam, bisg, dipg_pv, dipgm_va, fixed_prox_viscosity, ibigsam

RNG = np.random.default_rng(0)


def _numeric_prox(fun, z):
    res = minimize(lambda u: fun(u) + 0.5 * np.sum((u - z) ** 2), z, method="Powell",
                   options={"xtol": 1e-10, "ftol": 1e-14, "maxiter": 20000})
    return res.x


def test_soft_threshold_matches_numeric_prox():
    z = RNG.normal(size=5)
    num = _numeric_prox(lambda u: 0.7 * np.sum(np.abs(u)), z)
    assert np.allclose(soft_threshold(z, 0.7), num, atol=1e-5)


def test_group_soft_threshold_matches_numeric_prox():
    z = RNG.normal(size=(3, 2))
    num = _numeric_prox(lambda u: 0.8 * np.sum(np.linalg.norm(u.reshape(3, 2), axis=1)), z.ravel())
    assert np.allclose(group_soft_threshold_rows(z, 0.8).ravel(), num, atol=1e-4)


def test_negative_threshold_rejected():
    with pytest.raises(ValueError):
        soft_threshold(np.ones(2), -1.0)


def test_dct_is_orthonormal_and_prox_is_firmly_nonexpansive():
    x = RNG.normal(size=(8, 8))
    assert np.allclose(idct2(dct2(x)), x)
    assert np.isclose(np.linalg.norm(dct2(x)), np.linalg.norm(x))
    y = RNG.normal(size=(8, 8))
    px, py = prox_l1_orthonormal(x, 0.3), prox_l1_orthonormal(y, 0.3)
    assert np.sum((px - py) ** 2) <= np.sum((px - py) * (x - y)) + 1e-12


def _err_monitor(xstar):
    return lambda x: {"err": float(np.linalg.norm(x - xstar))}


def test_fixed_step_prox_viscosity_is_biased_on_toy_example():
    """The limit is (2s/(1+s), 2/(1+s)) instead of x* = (0, 2)."""
    prob = toy_problem(tau=3.0)
    s = 0.5
    hist = fixed_prox_viscosity(prob, np.array([3.0, -1.0]), 50000, lambda n: 1.0 / (n + 1),
                                _err_monitor(np.zeros(2)), c=0.5, s=s)
    expected = np.array([2 * s / (1 + s), 2 / (1 + s)])
    assert np.linalg.norm(hist.x - expected) < 5e-3
    assert np.linalg.norm(hist.x - np.array([0.0, 2.0])) > 0.5


def test_proposed_method_converges_to_toy_solution():
    prob = toy_problem(tau=3.0)
    xstar = np.array([0.0, 2.0])
    hist = dipg_pv(prob, np.array([3.0, -1.0]), 5000, lambda n: 1.0 / (n + 1),
                   _err_monitor(xstar), c=0.5)
    assert hist.records[-1]["err"] < 1e-2


def test_smoothed_methods_converge_to_smoothed_solution_on_toy_example():
    """With Huber smoothing the limit is x_mu = (a, 2 - a), a = 2mu/(1+2mu) (tau >= 2)."""
    prob = toy_problem(tau=3.0)
    mu = 0.1
    a = 2 * mu / (1 + 2 * mu)
    x_mu = np.array([a, 2 - a])
    hist = bigsam(prob, np.array([3.0, -1.0]), 300000, lambda n: 1.0 / (n + 1),
                  _err_monitor(x_mu), c=0.5, mu=mu)
    assert hist.records[-1]["err"] < 2e-2
    assert np.linalg.norm(hist.x - np.array([0.0, 2.0])) > 0.2


def _cvx_reference(A, b, sigma, tau):
    cp = pytest.importorskip("cvxpy")
    x = cp.Variable(A.shape[1])
    prob = cp.Problem(cp.Minimize(0.5 * sigma * cp.sum_squares(x) + tau * cp.norm1(x)),
                      [A @ x == b, x >= 0])
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
    return np.asarray(x.value)


@pytest.mark.parametrize("solver", [dipg_pv, bisg])
def test_sparse_ls_converges_to_cvx_reference(solver):
    m, n = 20, 50
    A = RNG.normal(size=(m, n)) / np.sqrt(m)
    x_true = np.maximum(RNG.normal(size=n), 0) * (RNG.random(n) < 0.3)
    b = A @ x_true
    sigma, tau = 1.0, 1.0
    xstar = _cvx_reference(A, b, sigma, tau)
    prob = sparse_ls_problem(A, b, sigma, tau)
    hist = solver(prob, np.zeros(n), 100000, lambda k: 1.0 / (k + 1),
                  _err_monitor(xstar), c=1.0 / prob.Lf)
    assert hist.records[-1]["err"] / np.linalg.norm(xstar) < 2e-2


def test_inner_step_validation():
    prob = toy_problem(tau=1.0)
    with pytest.raises(ValueError):
        dipg_pv(prob, np.zeros(2), 1, lambda n: 0.1, lambda x: {}, c=10.0)


def test_all_solvers_run_on_matrix_variables():
    H = RNG.normal(size=(10, 15))
    T = RNG.normal(size=(10, 3))
    prob = elm_group_problem(H, T, sigma=1.0, tau=0.1)
    x0 = np.zeros((15, 3))
    mon = lambda B: {"inner": prob.inner(B)}
    for solver, kw in [(dipg_pv, {}), (bisg, {}), (bigsam, {"mu": 0.1}),
                       (ibigsam, {"mu": 0.1}), (dipgm_va, {"mu": 0.1})]:
        hist = solver(prob, x0, 5, lambda n: 1.0 / (n + 1), mon, c=1.0 / prob.Lf, **kw)
        assert hist.x.shape == (15, 3)
        assert np.isfinite(hist.records[-1]["inner"])


def test_inpainting_reference_matches_cvxpy_on_small_image():
    cp = pytest.importorskip("cvxpy")
    from scipy.fft import dct
    from bilevel.reference import inpainting_reference
    rng = np.random.default_rng(3)
    img = rng.random((8, 8))
    mask = rng.random((8, 8)) > 0.5
    sigma, tau = 0.1, 0.05
    x_ref, info = inpainting_reference(mask, img, sigma, tau, max_iter=100000, tol=1e-12)
    D = dct(np.eye(8), norm="ortho", axis=0)
    X = cp.Variable((8, 8))
    obj = 0.5 * sigma * cp.sum_squares(X) + tau * cp.sum(cp.abs(D @ X @ D.T))
    cons = [X >= 0, X <= 1, cp.multiply(mask.astype(float), X) == mask * img]
    cp.Problem(cp.Minimize(obj), cons).solve(solver=cp.CLARABEL)
    assert np.allclose(x_ref, X.value, atol=1e-5)


def test_ipva_matches_reference_on_small_inpainting():
    from bilevel.problems import inpainting_problem
    from bilevel.reference import inpainting_reference
    rng = np.random.default_rng(4)
    img = rng.random((16, 16))
    mask = rng.random((16, 16)) > 0.5
    x_ref, _ = inpainting_reference(mask, img, 0.1, 0.05)
    prob = inpainting_problem(mask, img, 0.1, 0.05)
    hist = dipg_pv(prob, mask * img, 20000, lambda n: 0.1 / (n + 1),
                   lambda x: {}, c=1.0)
    assert np.linalg.norm(hist.x - x_ref) / np.linalg.norm(x_ref) < 1e-3


def test_sparse_recovery_exact():
    from bilevel.problems import sparse_recovery_problem
    rng = np.random.default_rng(7)
    m, n, k = 40, 120, 4
    A = rng.normal(size=(m, n)) / np.sqrt(m)
    xt = np.zeros(n)
    xt[rng.choice(n, k, replace=False)] = rng.normal(size=k)
    prob = sparse_recovery_problem(A, A @ xt, sigma=0.01)
    hist = dipg_pv(prob, np.zeros(n), 20000, lambda j: 0.01 / (j + 1),
                   lambda x: {}, c=1.0 / prob.Lf)
    assert np.linalg.norm(hist.x - xt) / np.linalg.norm(xt) < 1e-3


def test_ipva_matches_hand_computed_iterations():
    """Two IPVA steps reproduced by hand (inertial rule, FB step, outer proximal gradient step)."""
    from bilevel.problems import sparse_recovery_problem
    from bilevel.operators import soft_threshold as st
    rng = np.random.default_rng(11)
    A = rng.normal(size=(6, 10))
    b = rng.normal(size=6)
    prob = sparse_recovery_problem(A, b, sigma=0.5, tau=0.3)
    c = 1.0 / prob.Lf
    al = lambda n: 0.7 / (n + 1)
    tau = lambda n: 1e-3 * al(n)
    x0 = rng.normal(size=10)
    hist = dipg_pv(prob, x0, 2, al, lambda x: {"x": x.copy()}, c=c,
                   theta_cap=lambda n: (n - 1) / (n + 2), tau=tau)
    # n = 1: x_0 = x_1, so w_1 = x_1
    y = x0 - c * A.T @ (A @ x0 - b)
    x2 = st(y - al(1) * 0.5 * y, 0.3 * al(1))
    # n = 2: theta_2 = min{1/4, tau_2/||x_2 - x_1||}
    theta = min(0.25, tau(2) / np.linalg.norm(x2 - x0))
    w = x2 + theta * (x2 - x0)
    y = w - c * A.T @ (A @ w - b)
    x3 = st(y - al(2) * 0.5 * y, 0.3 * al(2))
    assert np.allclose(hist.records[0]["x"], x2, atol=1e-13)
    assert np.allclose(hist.records[1]["x"], x3, atol=1e-13)


def test_dipgm_va_first_step_matches_definition():
    """First DIPGM-VA step: z_{-1} = z_0 = x_1, theta_1 = mu_1 = 1/2 (budget inactive), delta term = 0."""
    from bilevel.problems import sparse_recovery_problem
    rng = np.random.default_rng(12)
    A = rng.normal(size=(5, 8))
    b = rng.normal(size=5)
    prob = sparse_recovery_problem(A, b, sigma=0.5, tau=0.3)
    c, mu = 1.0 / prob.Lf, 0.1
    grad_w, Lw = prob.smoothed_grad_outer(mu)
    s = 1.0 / (Lw + prob.sigma)
    x1 = rng.normal(size=8)
    hist = dipgm_va(prob, x1, 1, lambda n: 0.5 / (n + 1), lambda x: {"x": x.copy()}, c=c, mu=mu)
    T = lambda v: v - c * A.T @ (A @ v - b)
    w = x1 - 0.25 * s * grad_w(x1)
    z = (1 - 0.45) * x1 + 0.45 * T(T(w))
    x2 = z + 0.5 * (z - x1)
    assert np.allclose(hist.records[0]["x"], x2, atol=1e-13)


def test_project_simplex_is_the_projection():
    from bilevel.problems import project_simplex

    z = RNG.normal(size=6)
    p = project_simplex(z)
    assert np.all(p >= 0.0) and abs(p.sum() - 1.0) < 1e-12
    num = minimize(lambda u: 0.5 * np.sum((u - z) ** 2), np.full(6, 1 / 6), method="SLSQP",
                   bounds=[(0, None)] * 6, constraints=[{"type": "eq", "fun": lambda u: u.sum() - 1.0}],
                   options={"ftol": 1e-14, "maxiter": 500}).x
    assert np.allclose(p, num, atol=1e-6)


@pytest.mark.parametrize("variant", ["box", "simplex"])
def test_ipva_matches_cvxpy_on_small_constrained_dct_problem(variant):
    cp = pytest.importorskip("cvxpy")
    from scipy.fft import dct, idct

    from bilevel.problems import box_dct_problem, simplex_dct_problem

    rng = np.random.default_rng(3)
    m, d, sigma = 10, 24, 0.05
    A = rng.normal(size=(m, d)) / np.sqrt(m)
    theta = np.zeros(d)
    theta[rng.choice(d, 3, replace=False)] = rng.normal(size=3)
    s = idct(theta, norm="ortho")
    if variant == "box":
        x_true = np.clip(0.5 + 2.0 * s, 0.0, 1.0)
        prob, cons_of = box_dct_problem(A, A @ x_true, sigma), lambda x: [x >= 0, x <= 1]
        x0 = np.full(d, 0.5)
    else:
        x_true = np.maximum(s, 0.0) + 1e-3
        x_true = x_true / x_true.sum()
        prob, cons_of = simplex_dct_problem(A, A @ x_true, sigma), lambda x: [x >= 0, cp.sum(x) == 1]
        x0 = np.full(d, 1.0 / d)
    W = dct(np.eye(d), norm="ortho", axis=0)
    x = cp.Variable(d)
    cp.Problem(cp.Minimize(0.5 * sigma * cp.sum_squares(x) + cp.norm1(W @ x)), [A @ x == A @ x_true] + cons_of(x)).solve(
        solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
    xs = np.asarray(x.value)
    al = lambda n: 1.0 / (n + 1.0)
    res = dipg_pv(prob, x0, 30000, al, lambda z: {}, c=1.0 / prob.Lf, tau=lambda n: 1e4 * al(n) ** 2)
    assert np.linalg.norm(res.x - xs) / np.linalg.norm(xs) < 5e-2
