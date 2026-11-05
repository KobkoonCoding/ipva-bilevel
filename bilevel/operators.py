"""Proximal operators and orthonormal transforms used in the experiments.

All functions are pure: they never modify their inputs.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn


def soft_threshold(x: np.ndarray, t: float) -> np.ndarray:
    """Proximal map of t*||.||_1 (componentwise soft thresholding)."""
    if t < 0:
        raise ValueError("threshold must be nonnegative")
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def group_soft_threshold_rows(x: np.ndarray, t: float) -> np.ndarray:
    """Proximal map of t * sum_i ||x[i, :]||_2 (row-wise group soft thresholding)."""
    if t < 0:
        raise ValueError("threshold must be nonnegative")
    if x.ndim != 2:
        raise ValueError("group soft thresholding expects a 2-D array")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    scale = np.maximum(1.0 - t / np.maximum(norms, 1e-300), 0.0)
    return x * scale


def project_box(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Projection onto the box [lo, hi]^n (proximal map of its indicator)."""
    return np.clip(x, lo, hi)


def dct2(x: np.ndarray) -> np.ndarray:
    """Orthonormal 2-D DCT-II (W in the paper); W^T W = W W^T = I."""
    return dctn(x, norm="ortho")


def idct2(c: np.ndarray) -> np.ndarray:
    """Inverse (= adjoint) of the orthonormal 2-D DCT-II."""
    return idctn(c, norm="ortho")


def prox_l1_orthonormal(x: np.ndarray, t: float) -> np.ndarray:
    """Proximal map of t*||W x||_1 for the orthonormal DCT W: W^T soft(W x, t)."""
    return idct2(soft_threshold(dct2(x), t))
