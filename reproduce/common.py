"""Constants and loaders shared by the scripts of this folder.

Section 4 of the paper reports image inpainting at one cost C = 5000 evaluations of grad f, on 18 random-mask cases.
All numbers come from results/exp21_P4.json (stored runs and stored selections); nothing is re-tuned.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
from pathlib import Path

import numpy as np

import methods as mt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
OUT = ROOT / "outputs"                                 # regenerated tables, figures and values (not tracked by git)
OUT.mkdir(exist_ok=True)
COST = "5000"
N_FULL = int(COST)                                     # iterations of every method except DIPGM-VA
N_HALF = N_FULL // 2                                   # iterations of DIPGM-VA (two evaluations of grad f per iteration)
CASES = ["fundus_r50_m0", "cxr_r50_m0", "brain_r50_m0"]  # cases of the two figures (50 % missing, first mask of each image)

# label in exp21_P4.json -> name in the paper; order of the rows of the tables and of the legend
LABELS = ["IPVA-1", "Bi-SG", "BiG-SAM", "iBiG-SAM", "DIPGM-VA", "FBi-PG"]
NAME = {"IPVA-1": "IPVA"}
IMAGES = ("fundus", "cxr", "brain")
IMG_LABEL = {"fundus": "Retina", "cxr": "Chest radiograph", "brain": "Brain slice"}
RATES = ("50", "70")


def name(label: str) -> str:
    return NAME.get(label, label)


def load(fname: str) -> dict:
    return json.loads((RES / fname).read_text(encoding="utf-8"))


def exp21() -> dict:
    return load("exp21_P4.json")


def random_trials(d: dict) -> list[dict]:
    """The 18 random-mask cases, in file order (the stored file also holds three central-block cases '*_b25', not used here)."""
    tr = [t for t in d["trials"] if not t["key"].endswith("_b25")]
    assert len(tr) == 18 and all("_r50_" in t["key"] or "_r70_" in t["key"] for t in tr), [t["key"] for t in tr]
    return tr


# Relative tolerance of a re-run against the stored M1 of exp21_P4.json.  iBiG-SAM uses mu = 1e-4, so its outer gradient
# (z - prox_phi(z, mu)) / mu amplifies rounding differences of the order 1e-15 (the other methods reproduce the stored
# values to about 6e-15); measured deviations of the re-run on CASES: 7.3e-4, 1.0e-10, 6.4e-8.
RERUN_TOL = {"iBiG-SAM": 1e-3}
RERUN_TOL_DEFAULT = 1e-12


def rerun_tol(label: str) -> float:
    return RERUN_TOL.get(label, RERUN_TOL_DEFAULT)


def n_iter(label: str) -> int:
    return N_HALF if label == "DIPGM-VA" else N_FULL


def m1(t: dict, label: str) -> float:
    r = t["results"][label][COST]
    assert not r["diverged"] and int(r["N"]) == n_iter(label), (t["key"], label, r["N"])
    return float(r["M1"])


def selection(d: dict) -> dict:
    """Stored selections at C = COST, with the method spec and a0 (bijective function of the stored scale 'level')."""
    specs = mt.method_specs()
    out = {}
    for label in LABELS:
        s = d["selected"][COST][label]
        m = specs[s["method"]]
        out[label] = {"method": s["method"], "spec": m, "prm": dict(s["prm"]), "N": int(s["N"]), "a0": mt.a0_of(m, s["prm"])}
    assert all(out[k]["N"] == n_iter(k) for k in LABELS)
    return out


def sci_tab(x: float, digits: int = 2) -> str:
    """Table notation, e.g. 4.9e-3."""
    mnt, e = f"{x:.{digits - 1}e}".split("e")
    return f"{mnt}e{int(e)}"


def num2(x: float) -> str:
    """Two significant digits for parameter values: 0.34, 1.4, 9.3, 4.1e-2."""
    if 0.1 <= abs(x) < 10:
        return f"{x:.2g}"
    if 10 <= abs(x) < 100:
        return f"{x:.0f}"
    return sci_tab(x)


def median(xs) -> float:
    return float(np.median(np.asarray(xs, float)))
