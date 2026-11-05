"""Figure 1 of the paper: relative error ||x_n - x*|| / ||x*|| against the number of evaluations of grad f.

It re-runs the six methods with the stored selections of results/exp21_P4.json ('selected'; IPVA = label 'IPVA-1', DIPGM-VA with
N = C/2 iterations, the other methods with N = C iterations) from the starting point b, on the first random mask with 50 % missing
pixels of each test image (common.CASES), and records x_n after every iteration.  The cost of iteration n is n evaluations of grad f,
and 2n for DIPGM-VA.
Checks (asserted): the final value of every curve equals M1 of exp21_P4.json for this case and method (relative deviation <= 1e-12;
iBiG-SAM <= 1e-3, see common.RERUN_TOL; the deviations are written to the json as rel_dev_final).
Writes
    outputs/curves.json          the curves (cost, M1) of the six methods, per case
    outputs/recon_<case>.npz     the last iterates x_N of the six methods (float64, not clipped), one file per case
    outputs/figs/curves_P4.pdf   three panels side by side (one per case), shared y axis, log-log axes, legend in one row above the panels
Usage:  python reproduce/make_figure1.py [--plot-only]
"""
from __future__ import annotations

import json
import sys

import numpy as np

import common as E
import methods as mt
import problems as P

OUT_JSON = E.OUT / "curves.json"
TITLE = {"fundus_r50_m0": "Retina", "cxr_r50_m0": "Chest radiograph", "brain_r50_m0": "Brain slice"}

# color, line style, line width, marker
STYLE = {
    "IPVA-1": ("#04224f", "-", 1.8, None),
    "Bi-SG": ("#8f1d14", "--", 1.3, "s"),
    "BiG-SAM": ("#3aa14b", "-.", 1.3, "^"),
    "iBiG-SAM": ("#d4a0f2", ":", 1.3, "v"),
    "DIPGM-VA": ("#d98a1f", (0, (5, 1, 1, 1)), 1.3, "D"),
    "FBi-PG": ("#0f7f86", (0, (1, 1)), 1.3, "o"),
}


def recon_path(case: str):
    return E.OUT / f"recon_{case}.npz"


def run_case(d: dict, sel: dict, case: str) -> dict:
    trial = next(t for t in d["trials"] if t["key"] == case)
    prob, xs, x0 = P.build(case)
    nxs = float(np.linalg.norm(xs.ravel()))
    methods: dict = {}
    finals = {}
    for label in E.LABELS:
        s = sel[label]
        n_iter = s["N"]
        errs: dict[int, float] = {}
        last: dict[str, np.ndarray] = {}

        def mon(n, x, y, elapsed, errs=errs, last=last, n_iter=n_iter):
            errs[n] = float(np.linalg.norm((x - xs).ravel()) / nxs)
            if n == n_iter:
                last["x"] = np.array(x, copy=True)

        res = mt.run_method(prob, s["spec"], s["prm"], n_iter, x0, mon, frozenset(range(1, n_iter + 1)))
        assert not res["diverged"] and len(errs) == n_iter, label
        its = sorted(errs)
        per = 2 if label == "DIPGM-VA" else 1
        stored = E.m1(trial, label)
        dev = abs(errs[n_iter] - stored) / stored
        assert dev <= E.rerun_tol(label), (case, label, errs[n_iter], stored)
        methods[label] = {"N": n_iter, "cost": [per * n for n in its], "M1": [errs[n] for n in its], "prm": s["prm"],
                          "final_M1": errs[n_iter], "stored_M1": stored, "rel_dev_final": dev}
        finals[label] = last["x"]
        print(f"{case:14s} {E.name(label):9s} N = {n_iter:4d}  final M1 = {errs[n_iter]:.6e}  "
              f"(exp21_P4.json: {stored:.6e}, rel. dev. {dev:.1e})")
    np.savez_compressed(recon_path(case), **finals)
    return {"methods": methods}


def run_curves() -> dict:
    d = E.exp21()
    sel = E.selection(d)
    out = {"cases": E.CASES, "cost": int(E.COST), "source": f"results/exp21_P4.json selected['{E.COST}']",
           "curves": {case: run_case(d, sel, case) for case in E.CASES}}
    OUT_JSON.write_text(json.dumps(out), encoding="utf-8")
    print("wrote", OUT_JSON, *(recon_path(c) for c in E.CASES))
    return out


def plot(cur: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10, "legend.fontsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": 0.5, "lines.linewidth": 1.0, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.pad_inches": 0.02,
    })
    fig, axes = plt.subplots(1, len(E.CASES), figsize=(6.6, 2.1), sharey=True)
    for ax, case in zip(axes, E.CASES):
        for label in E.LABELS:
            col, ls, lw, mk = STYLE[label]
            cv = cur["curves"][case]["methods"][label]
            kw = dict(color=col, ls=ls, lw=lw, label=E.name(label), marker=mk, ms=4, markevery=0.1, mew=0.5,
                      mfc="none" if mk else None)
            ax.loglog(np.array(cv["cost"], float), np.array(cv["M1"], float), **kw)
        ax.set_xlim(1, int(E.COST))
        ax.set_title(TITLE[case], pad=3)
        ax.set_xlabel(r"evaluations of $\nabla f$")
        ax.grid(True, which="major", color="#dddddd", lw=0.4)
    axes[0].set_ylabel(r"$\|x_n-x^\ast\|/\|x^\ast\|$")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.02), columnspacing=0.8,
               handlelength=2.2, handletextpad=0.4)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.2, top=0.8, wspace=0.08)
    (E.OUT / "figs").mkdir(parents=True, exist_ok=True)
    fig.savefig(E.OUT / "figs" / "curves_P4.pdf", bbox_inches="tight")
    fig.savefig(E.OUT / "figs" / "curves_P4.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", E.OUT / "figs" / "curves_P4.pdf")


def main() -> None:
    cur = json.loads(OUT_JSON.read_text(encoding="utf-8")) if "--plot-only" in sys.argv else run_curves()
    plot(cur)


if __name__ == "__main__":
    main()
