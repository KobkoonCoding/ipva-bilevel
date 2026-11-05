"""Test images, masks and reference solutions of the inpainting experiment.

Images: every image is a 256 x 256 float array in [0, 1].
  fundus : skimage.data.retina(), 1411x1411 RGB -> gray (skimage rgb2gray) -> the whole image is resized to 256x256 with
           anti-aliasing.
  cxr    : data/cxr_wikimedia_cc0.jpg, 1929 wide x 2207 high -> gray -> resize so that the short side (width) is 256
           (height 293, anti-aliased) -> centred crop of 256 rows (rows 18..273 of 293).
  brain  : data/brain_unc_slice.npy (256x256 slice of skimage.data.brain()), min-max normalised to [0, 1].
  shepp  : tuning image only: skimage.data.shepp_logan_phantom() (400x400) resized to 256x256.
Masks (True = observed): random masks with the missing rate 0.5 or 0.7 and 3 seeds each.
Reference solutions x*: results/ref_exp18_<case>.npz, computed by ADMM (bilevel/reference.py).
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
from pathlib import Path

import numpy as np
from skimage import color, transform
from skimage import data as skimage_data

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"

SIZE = 256
SIGMA, TAU = 1e-2, 1.0
TEST_IMAGES = ("fundus", "cxr", "brain")
TUNE_IMAGE = "shepp"
IMAGE_INDEX = {"shepp": 20, "fundus": 21, "cxr": 22, "brain": 23}
RATES = (0.5, 0.7)
N_MASKS = 3


def load_image(name: str) -> np.ndarray:
    if name == "fundus":
        gray = color.rgb2gray(skimage_data.retina())
        img = transform.resize(gray, (SIZE, SIZE), anti_aliasing=True)
    elif name == "cxr":
        from PIL import Image
        rgb = np.asarray(Image.open(ROOT / "data" / "cxr_wikimedia_cc0.jpg").convert("RGB"), dtype=float) / 255.0
        gray = color.rgb2gray(rgb)
        h, w = gray.shape
        new_h = int(round(h * SIZE / w))
        small = transform.resize(gray, (new_h, SIZE), anti_aliasing=True)
        r0 = (new_h - SIZE) // 2
        img = small[r0:r0 + SIZE]
    elif name == "brain":
        raw = np.load(ROOT / "data" / "brain_unc_slice.npy").astype(float)
        img = (raw - raw.min()) / (raw.max() - raw.min())
    elif name == "shepp":
        img = transform.resize(skimage_data.shepp_logan_phantom(), (SIZE, SIZE), anti_aliasing=True)
    else:
        raise ValueError(name)
    return np.clip(img, 0.0, 1.0)


def random_mask(name: str, rate: float, v: int) -> np.ndarray:
    seed = 100 * IMAGE_INDEX[name] + int(round(100 * rate)) + 7919 * v
    return np.random.default_rng(seed).random((SIZE, SIZE)) >= rate


def case_key(name: str, rate_percent: str, v: int = 0) -> str:
    return f"{name}_r{rate_percent}_m{v}"


def make_case(key: str):
    """key -> (image name, kind, image, mask)."""
    name, rest = key.split("_", 1)
    rate_s, v_s = rest.split("_")
    return name, "random", load_image(name), random_mask(name, int(rate_s[1:]) / 100.0, int(v_s[1:]))


def tune_case() -> str:
    return case_key(TUNE_IMAGE, "50", 0)


def ref_path(key: str) -> Path:
    return RESULTS / f"ref_exp18_{key}.npz"


def load_ref(key: str):
    z = np.load(ref_path(key), allow_pickle=True)
    return z["x"], dict(z["info"].item())


def compute_reference(key: str, max_iter: int = 30000, rho: float = 30.0) -> dict:
    """Compute and store x* of a case by ADMM, unless the file exists.

    The stored files use the defaults, except fundus_r50_m0 (rho = 10, max_iter = 150000) and the tuning image
    shepp_r50_m0 (stopped by the tolerance after 78653 iterations)."""
    from bilevel.reference import inpainting_reference
    path = ref_path(key)
    if path.exists():
        return load_ref(key)[1]
    _, _, img, mask = make_case(key)
    x, info = inpainting_reference(mask, img, SIGMA, TAU, rho=rho, max_iter=max_iter, tol=1e-10)
    info = {**info, "rho": rho, "max_iter": max_iter}
    np.savez(path, x=x, info=np.array(info, dtype=object))
    return info
