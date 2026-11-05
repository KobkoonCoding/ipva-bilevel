"""Figure 2 of the paper: reconstructions on the cases common.CASES (first random mask with 50 % missing pixels of each test image).

It reads the last iterates written by make_figure1.py (outputs/recon_<case>.npz) and computes PSNR and SSIM: the iterate is clipped to
[0, 1] and compared with the ground truth by scikit-image (data_range = 1, default 7 x 7 uniform window for the SSIM); the relative
error is ||x_N - x*|| / ||x*|| of the unclipped iterate.
Checks (asserted): the relative errors equal M1 of results/exp21_P4.json; for Bi-SG, BiG-SAM, iBiG-SAM, DIPGM-VA and FBi-PG the PSNR,
SSIM and error equal the values of results/exp21_p4_quality.json (IPVA-1 is not in that file and is compared with the stored M1 only);
PSNR and SSIM of x* and of b equal the stored ones.
Writes
    outputs/figure2_values.json   PSNR, SSIM, relative error of all six methods, of x* and of b, per case
    outputs/figs/inpaint_P4.pdf   three rows (cases) x six columns: ground truth, degraded image b, IPVA, FBi-PG, BiG-SAM, DIPGM-VA
Usage:  python reproduce/make_figure2.py
"""
from __future__ import annotations

import json

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

import common as E
import data as D
import problems as P

PANEL_METHODS = ["IPVA-1", "FBi-PG", "BiG-SAM", "DIPGM-VA"]
OUT_JSON = E.OUT / "figure2_values.json"
OUT_PDF = "inpaint_P4.pdf"
ROW_LABEL = {"fundus_r50_m0": "Retina", "cxr_r50_m0": "Chest radiograph", "brain_r50_m0": "Brain slice"}
TOL = 1e-12


def recon(case: str):
    return np.load(E.OUT / f"recon_{case}.npz")          # written by make_figure1.py


def quality(img: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """(PSNR, SSIM) of clip(x, 0, 1) against img."""
    z = np.clip(x, 0.0, 1.0)
    assert np.isfinite(z).all()
    return float(psnr_fn(img, z, data_range=1.0)), float(ssim_fn(img, z, data_range=1.0))


def close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol * max(abs(b), 1.0)


def compute_case(case: str, d21: dict, q21all: dict) -> dict:
    _, _, img, mask = D.make_case(case)
    _, xs, x0 = P.build(case)
    z = recon(case)
    trial = next(t for t in d21["trials"] if t["key"] == case)
    q21 = next(t for t in q21all["trials"] if t["key"] == case)
    nxs = float(np.linalg.norm(xs.ravel()))
    out: dict = {"methods": {}}
    for tag, vec in (("xstar", xs), ("degraded", x0)):
        ps, ss = quality(img, vec)
        out[tag] = {"psnr": ps, "ssim": ss}
    assert np.array_equal(x0, mask * img)
    assert close(out["xstar"]["psnr"], q21["ref_psnr"]) and close(out["xstar"]["ssim"], q21["ref_ssim"])
    assert close(out["degraded"]["psnr"], q21["obs_psnr"]) and close(out["degraded"]["ssim"], q21["obs_ssim"])
    for label in E.LABELS:
        x = z[label]
        ps, ss = quality(img, x)
        err = float(np.linalg.norm((x - xs).ravel()) / nxs)
        tol = E.rerun_tol(label)
        assert close(err, E.m1(trial, label), tol), (case, label, err)
        if label != "IPVA-1":                                   # same run as in exp21_p4_quality.json
            r = q21["results"][label][E.COST]
            assert close(ps, r["psnr"], tol) and close(ss, r["ssim"], tol) and close(err, r["err"], tol), (case, label, ps, ss, err, r)
            out["methods"][label] = {"psnr": ps, "ssim": ss, "err": err, "stored": {k: r[k] for k in ("psnr", "ssim", "err")}}
        else:
            out["methods"][label] = {"psnr": ps, "ssim": ss, "err": err}
    return out


def compute() -> dict:
    d21, q21all = E.exp21(), E.load("exp21_p4_quality.json")
    out = {"cases": E.CASES, "cost": int(E.COST), "per_case": {c: compute_case(c, d21, q21all) for c in E.CASES}}
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", OUT_JSON)
    return out


def plot(q: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.pad_inches": 0.02,
    })
    ncol = 2 + len(PANEL_METHODS)
    fig, axes = plt.subplots(len(E.CASES), ncol, figsize=(5.1, 3 * (5.1 / ncol + 0.36) + 0.2))
    for i, case in enumerate(E.CASES):
        qc = q["per_case"][case]
        _, _, img, mask = D.make_case(case)
        z = recon(case)
        panels = [(img, "ground truth", None),
                  (mask * img, "degraded image $b$", f"PSNR {qc['degraded']['psnr']:.2f}\nSSIM {qc['degraded']['ssim']:.3f}\n ")]
        for label in PANEL_METHODS:
            r = qc["methods"][label]
            panels.append((np.clip(z[label], 0.0, 1.0), E.name(label),
                           f"PSNR {r['psnr']:.2f}\nSSIM {r['ssim']:.3f}\nerror {r['err']:.1e}"))
        for j, (a, title, note) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(a, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.3)
            if i == 0:
                ax.set_title(title, pad=2, fontsize=8)
            if j == 0:
                ax.set_ylabel(ROW_LABEL[case], labelpad=2, fontsize=8)
            if note:
                ax.set_xlabel(note, labelpad=1.5, fontsize=6.5, linespacing=1.0)
    fig.subplots_adjust(left=0.03, right=0.995, bottom=0.06, top=0.95, wspace=0.04, hspace=0.36)
    (E.OUT / "figs").mkdir(parents=True, exist_ok=True)
    fig.savefig(E.OUT / "figs" / OUT_PDF, bbox_inches="tight")
    fig.savefig(E.OUT / "figs" / "inpaint_P4.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", E.OUT / "figs" / OUT_PDF)


def main() -> None:
    q = compute()
    for case in E.CASES:
        qc = q["per_case"][case]
        for k in ("xstar", "degraded"):
            print(f"{case:14s} {k:9s} PSNR {qc[k]['psnr']:.4f}  SSIM {qc[k]['ssim']:.4f}")
        for label in E.LABELS:
            r = qc["methods"][label]
            print(f"{case:14s} {E.name(label):9s} PSNR {r['psnr']:.4f}  SSIM {r['ssim']:.4f}  error {r['err']:.4e}")
    plot(q)


if __name__ == "__main__":
    main()
