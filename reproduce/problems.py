"""The inpainting problem of section 4 of the paper, built for one case key.

Inner function f + g: f = 0.5 ||M x - b||^2, g = indicator of [0, 1]^n.  Outer function h + phi: h = sigma/2 ||x||^2,
phi = tau ||W x||_1, W the orthonormal 2-D DCT.
"""
from __future__ import annotations

import data as D
from bilevel.problems import BilevelProblem, inpainting_problem


def build(key: str) -> tuple[BilevelProblem, object, object]:
    """Case key -> (problem, reference solution x*, starting point b)."""
    _, _, img, mask = D.make_case(key)
    xs, _ = D.load_ref(key)
    return inpainting_problem(mask, img, D.SIGMA, D.TAU), xs, mask * img
