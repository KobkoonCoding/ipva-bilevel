"""Check every number of section 4 of the paper, its two tables, the conclusion and the abstract against the stored results.

It recomputes each number, compares the resulting text with its row in reproduce/numbers_ledger.md, checks the statements of the text
that are not numbers ('all', 'every', 'smallest', ...), and compares the two tables with outputs/tables.
Reads   results/exp21_P4.json (runs and selections), results/exp21_p4_quality.json (PSNR and SSIM of the stored runs),
        results/ref_exp18_<case>.npz (ADMM information), and the outputs of make_tables.py, make_figure1.py, make_figure2.py.
Writes  outputs/numbers.json and outputs/numbers_ledger.md.  Prints PASS or stops at the first failed check.
Usage:  python reproduce/check_numbers.py [--stored-only]
        --stored-only skips the checks of the two figures, so that only make_tables.py must have run; it writes no file.
"""
from __future__ import annotations

import json
import math
import platform
import sys

import numpy as np
import scipy
import skimage

import common as E
import data as D
import make_tables as T
import methods as mt

REGISTERED = E.HERE / "numbers_ledger.md"
LEDGER = E.OUT / "numbers_ledger.md"
OUT = E.OUT / "numbers.json"
SEC4 = "section 4"
CUR = "curves.json"
FIG = "figure2_values.json"
RUNS = "exp21_P4.json"
SELECTED = f"{RUNS} selected['{E.COST}']"


def sci(x: float, digits: int = 2) -> str:
    m, e = f"{x:.{digits - 1}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def sci_up(x: float, digits: int = 2) -> str:
    e = math.floor(math.log10(abs(x)))
    m = math.ceil(abs(x) / 10.0 ** (e - digits + 1) - 1e-9) / 10.0 ** (digits - 1)
    if m >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{m:.{digits - 1}f}\\times10^{{{e}}}"


def fix_up(x: float, nd: int) -> str:
    return f"{math.ceil(x * 10 ** nd - 1e-9) / 10 ** nd:.{nd}f}"


class Ledger:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.registered = REGISTERED.read_text(encoding="utf-8")

    def claim(self, cid: str, text: str, value, source: str, computation: str, where: str = SEC4) -> None:
        """A number (or phrase) of the paper: the recomputed text must equal its row in the ledger."""
        if f"| `{cid}` | {text} |" not in self.registered:
            sys.exit(f"FAIL {cid}: '{text}' differs from its row in {REGISTERED.name} (computed value {value})")
        self.rows.append({"id": cid, "text": text, "value": value, "where": where, "source": source, "computation": computation})

    def check(self, cid: str, ok: bool, statement: str, source: str, computation: str, where: str = SEC4) -> None:
        """A statement of the text that is verified by a computation."""
        if not ok:
            sys.exit(f"FAIL {cid}: {statement}")
        self.rows.append({"id": cid, "text": f"verified: {statement}", "value": True, "where": where, "source": source,
                          "computation": computation})


def check_cases(L: Ledger, keys: list[str]) -> None:
    L.check("n_cases", len(keys) == 18 and len({k.split("_")[0] for k in keys}) == 3
            and all(sum(k.startswith(f"{im}_r{r}_") for k in keys) == 3 for im in E.IMAGES for r in E.RATES),
            "18 cases = 3 images x 2 missing rates x 3 random masks", RUNS, "trials without '*_b25'")
    L.check("setup_images", D.SIZE == 256 and D.RATES == (0.5, 0.7) and D.N_MASKS == 3 and D.TEST_IMAGES == E.IMAGES,
            "images 256 x 256, missing rates 50% and 70%, three masks each", "data.py", "SIZE, RATES, N_MASKS, TEST_IMAGES")
    removed = []
    for k in keys:
        _, kind, img, mask = D.make_case(k)
        assert kind == "random" and img.min() >= 0 and img.max() <= 1 and img.shape == (256, 256)
        removed.append((k, float(1 - mask.mean())))
    L.check("mask_rates", all(abs(r - (0.5 if "_r50_" in k else 0.7)) < 0.01 for k, r in removed),
            "each random mask removes about 50% or 70% of the pixels (fraction within 0.01)", "data.make_case",
            "1 - mean(mask) per case; range " + ", ".join(f"{r:.4f}" for _, r in removed[:1]) + ", ...")
    L.check("sigma", D.SIGMA == 1e-2 and D.TAU == 1.0, "sigma = 10^{-2} and phi = ||W.||_1 (weight 1)", "data.py", "SIGMA, TAU")


def check_admm(L: Ledger, keys: list[str]) -> tuple[float, dict]:
    infos = {k: D.load_ref(k)[1] for k in keys}
    res = max(max(i["primal_res"], i["dual_res"]) for i in infos.values())
    L.claim("admm_res", f"${sci_up(res)}$", res, "ref_exp18_<case>.npz (info)",
            "max over the 18 cases of the final relative primal and dual residuals of ADMM, rounded up")
    its = {k: int(i["iterations"]) for k, i in infos.items()}
    rho = {k: i.get("rho") for k, i in infos.items()}
    assert its["fundus_r50_m0"] == 150000 and all(its[k] == 30000 and rho[k] == 30.0 for k in keys if k != "fundus_r50_m0")
    return res, {"iterations": its, "penalty": rho,
                 "note": "fundus_r50_m0: 150000 iterations, penalty 10 (the penalty is not stored in its npz); "
                         "the other 17 cases: penalty 30, 30000 iterations (stored)"}


def check_tables(L: Ledger, trials: list[dict], sel: dict) -> dict:
    med = T.cell_medians(trials)
    tables = E.OUT / "tables"
    L.check("table_main_file", (tables / "main_results.tex").read_text(encoding="utf-8").replace("\r\n", "\n")
            == T.main_table(med), "tables/main_results.tex equals the table regenerated from exp21_P4.json", RUNS,
            "make_tables.main_table")
    L.check("table_params_file", (tables / "selected_params.tex").read_text(encoding="utf-8").replace("\r\n", "\n")
            == T.param_table(T.param_rows(sel)), "tables/selected_params.tex equals the table regenerated from the selections",
            f"{SELECTED}; methods.method_specs", "make_tables.param_table")
    L.check("ipva_best_columns", all(min(E.LABELS, key=lambda m: med[m][c]) == "IPVA-1" for c in med["IPVA-1"]),
            "IPVA gives the smallest median relative error for every image and missing rate (and over all 18 cases)",
            RUNS, "medians of M1 per column of Table 2")
    ex = E.sci_tab(med["IPVA-1"]["all"])
    L.claim("med_example", f"{ex} stands for ${sci(med['IPVA-1']['all'])}$", med["IPVA-1"]["all"], RUNS,
            "median of M1 of IPVA over the 18 cases (the example of the caption of Table 2)")
    return med


def check_ranking(L: Ledger, keys: list[str], err: dict, others: list[str]) -> tuple[int, list[float]]:
    n_best = sum(all(err[k]["IPVA-1"] < err[k][m] for m in others) for k in keys)
    L.claim("n_best", "In each of the 18 cases, IPVA has a smaller relative error than each of the other five methods", n_best,
            RUNS, "number of cases with M1(IPVA) < M1(m) for all five other methods (= 18)")
    assert n_best == 18
    L.check("fbipg_second", all(sorted(E.LABELS, key=lambda m: err[k][m])[1] == "FBi-PG" for k in keys),
            "FBi-PG has the second smallest error in each case", RUNS, "rank of M1 per case")
    ratio = [err[k]["FBi-PG"] / err[k]["IPVA-1"] for k in keys]
    L.claim("ratio_fbipg", f"{min(ratio):.1f} to {max(ratio):.1f} times that of IPVA", [min(ratio), max(ratio)], RUNS,
            "min and max over the 18 cases of M1(FBi-PG)/M1(IPVA), each rounded to one decimal")
    return n_best, ratio


def check_selection(L: Ledger, sel: dict) -> None:
    L.check("tuning_setup", mt.BUDGET == 60 and D.tune_case() == "shepp_r50_m0",
            "the coordinate search uses 60 runs on the tuning image (Shepp-Logan, one 50% random mask)", "methods.py; data.py",
            "BUDGET, tune_case")
    L.claim("tuning_runs", f"Each run uses all {E.COST} evaluations of $\\nabla f$", int(E.COST), "methods.search",
            f"one evaluation of the search is one complete run (DIPGM-VA: {E.N_HALF} iterations)")
    s = {m: sel[m]["prm"] for m in E.LABELS}
    a0 = {m: sel[m]["a0"] for m in E.LABELS}
    theory = {
        "IPVA": 0 < s["IPVA-1"]["delta"] <= 1 and 0 < s["IPVA-1"]["c_mult"] < 2 and s["IPVA-1"]["K"] > 0,
        "Bi-SG": 0.5 < s["Bi-SG"]["alpha"] < 1 and 0 < a0["Bi-SG"] <= min(1.0, 1.0 / D.SIGMA),
        "smoothing": all(0.5 < s[m]["delta"] <= 1 and sel[m]["prm"]["level"] <= 0.95 for m in ("BiG-SAM", "iBiG-SAM", "DIPGM-VA")),
        "FBi-PG": 1 < s["FBi-PG"]["gamma"] < 2 and s["FBi-PG"]["a"] > 2 and float(s["FBi-PG"]["a"]).is_integer(),
    }
    L.check("theory_range", all(theory.values()),
            "the selected values lie in the range that the theory of each method allows (IPVA: delta in (0,1], cL_f < 2; Bi-SG: "
            "nu in (1/2,1), a0 <= min{1,1/L_h}; BiG-SAM, iBiG-SAM, DIPGM-VA: alpha_1 <= 0.95; FBi-PG: 1 < gamma < 2, integer a > 2)",
            SELECTED, str(theory))
    # alpha_1 = a0 2^{-delta} computed from the printed a0 of Table 1 must also be <= 0.95
    printed = {m: float(T.a0_text(m, sel[m])) for m in ("BiG-SAM", "iBiG-SAM", "DIPGM-VA")}
    a1_print = {m: printed[m] * 2.0 ** (-s[m]["delta"]) for m in printed}
    L.claim("alpha1_cap", "We use $\\alpha_n\\le\\alpha_1\\le0.95$", max(a1_print.values()), f"{RUNS}; Table 1",
            "alpha_1 = a0 2^{-delta} <= 0.95 from the stored level and from the printed a0: "
            + ", ".join(f"{m}: a0 = {printed[m]:g}, alpha_1 = {a1_print[m]:.4f} (stored {sel[m]['prm']['level']:.4f})" for m in printed))
    assert all(v <= 0.95 for v in a1_print.values())
    L.check("ipva_c", s["IPVA-1"]["c_mult"] == 1.0, "the selected step size of IPVA is c = 1/L_f, so c <= 1/L_f (Theorem 3.12)",
            f"{SELECTED}['IPVA-1']", "c_mult == 1")
    L.check("costs", sel["DIPGM-VA"]["N"] == E.N_HALF and all(sel[m]["N"] == E.N_FULL for m in E.LABELS if m != "DIPGM-VA"),
            f"DIPGM-VA runs {E.N_HALF} iterations and the other methods {E.N_FULL} ({E.COST} evaluations of grad f)", SELECTED, "N")
    L.claim("costs_text", f"it runs {E.N_HALF} iterations, and the other methods run {E.N_FULL} iterations", E.N_HALF, SELECTED,
            "N of DIPGM-VA and of the other methods")
    L.claim("fair_text", f"every method uses {E.COST} evaluations of $\\nabla f$ and of $\\prox_g$", int(E.COST),
            "methods.run_method (one grad f and one prox_g per gradient evaluation)", "cost C")


def first_cost_ipva_smallest(cm: dict, others: list[str]) -> int:
    """Smallest k such that for every k' >= k the error of IPVA after k' evaluations is below that of every other method."""
    ip = dict(zip(cm["IPVA-1"]["cost"], cm["IPVA-1"]["M1"]))

    def at(m: str, k: int) -> float:
        c = np.asarray(cm[m]["cost"])
        i = np.searchsorted(c, k, side="right") - 1
        return float(cm[m]["M1"][i]) if i >= 0 else math.inf

    good = [all(ip[k] < at(m, k) for m in others) for k in range(1, int(E.COST) + 1)]
    first = int(E.COST)
    while first > 1 and good[first - 2]:
        first -= 1
    assert all(good[first - 1:]) and not good[first - 2]
    return first


def check_curves(L: Ledger, cur: dict, err: dict, others: list[str]) -> dict:
    assert cur["cases"] == E.CASES and cur["cost"] == int(E.COST)
    firsts, ip_fin, fb_fin, bs_fin, mono_from, bs_flat = {}, [], [], [], {}, {}
    for case in E.CASES:
        cm = cur["curves"][case]["methods"]
        ok = all(abs(cm[m]["M1"][-1] - err[case][m]) <= E.rerun_tol(m) * err[case][m] for m in E.LABELS) \
            and all(cm[m]["cost"][-1] == int(E.COST) for m in E.LABELS)
        L.check(f"curve_finals_{case}", ok,
                f"the final value of every curve of Figure 1 ({case}) equals M1 of exp21_P4.json at {E.COST} evaluations",
                f"{CUR}; {RUNS}",
                "relative deviations " + ", ".join(f"{E.name(m)} {cm[m]['rel_dev_final']:.1e}" for m in E.LABELS)
                + " (tolerance 1e-12, iBiG-SAM 1e-3: rounding amplified by 1/mu = 1e4)")
        firsts[case] = first_cost_ipva_smallest(cm, others)
        v = np.asarray(cm["IPVA-1"]["M1"])
        inc = np.where(np.diff(v) >= 0)[0]                         # x_{n+1} not smaller than x_n at n = inc + 1
        mono_from[case] = int(inc.max() + 2) if len(inc) else 1    # strictly decreasing from this iteration on
        ip_fin.append(v[-1])
        fb_fin.append(cm["FBi-PG"]["M1"][-1])
        b = np.asarray(cm["BiG-SAM"]["M1"])
        bs_fin.append(b[-1])
        bs_flat[case] = float(b[E.N_FULL // 2 - 1] / b[-1])
    first = max(firsts.values())
    L.claim("curve_first", f"IPVA has the smallest error from {first} evaluations of $\\nabla f$ on", firsts, CUR,
            "per case, the smallest k such that for every k' >= k the error of IPVA after k' evaluations is smaller than the error "
            "of every other method after at most k' evaluations; the text gives the largest of the three values")
    L.check("ipva_decreasing", max(mono_from.values()) <= first,
            f"'from there its error decreases at every iteration': the error of IPVA is strictly decreasing from iteration "
            f"{max(mono_from.values())} on in all three cases, and {max(mono_from.values())} <= {first}", CUR,
            "first iteration from which M1 of IPVA decreases strictly: " + str(mono_from))
    L.claim("curve_ipva", "Its final errors are " + ", ".join(f"${sci(x)}$" for x in ip_fin[:-1]) + f", and ${sci(ip_fin[-1])}$",
            ip_fin, CUR, "final M1 of IPVA on the three cases")
    L.claim("curve_fbipg", "those of FBi-PG are " + ", ".join(f"${sci(x)}$" for x in fb_fin[:-1]) + f", and ${sci(fb_fin[-1])}$",
            fb_fin, CUR, "final M1 of FBi-PG on the three cases")
    L.claim("bigsam_level", f"levels off at values between ${sci(min(bs_fin))}$ and ${sci(max(bs_fin))}$", bs_fin, CUR,
            "final M1 of BiG-SAM; 'levels off': M1 after 2500 iterations / M1 after 5000 iterations = "
            + ", ".join(f"{r:.4f}" for r in bs_flat.values()))
    assert all(r < 1.01 for r in bs_flat.values())
    return {"curve_first_cost_ipva_smallest": firsts, "ipva_strictly_decreasing_from": mono_from, "bigsam_final": bs_fin}


def check_reconstructions(L: Ledger, fig: dict) -> dict:
    pc = fig["per_case"]
    four = ("IPVA-1", "FBi-PG", "BiG-SAM", "DIPGM-VA")
    spread = {c: max(pc[c]["methods"][m]["psnr"] for m in four) - min(pc[c]["methods"][m]["psnr"] for m in four) for c in E.CASES}
    L.claim("psnr_spread", f"their PSNR values differ by at most {fix_up(max(spread.values()), 1)}~dB", spread, FIG,
            "per case, max - min of the PSNR of IPVA, FBi-PG, BiG-SAM, DIPGM-VA, rounded up to 0.1 dB")
    dps = {c: abs(pc[c]["methods"]["IPVA-1"]["psnr"] - pc[c]["xstar"]["psnr"]) for c in E.CASES}
    dss = {c: abs(pc[c]["methods"]["IPVA-1"]["ssim"] - pc[c]["xstar"]["ssim"]) for c in E.CASES}
    L.claim("ipva_attains", "those of IPVA differ from them by less than 0.01~dB and by less than 0.001 in all three cases",
            [max(dps.values()), max(dss.values())], FIG,
            "|PSNR(IPVA) - PSNR(x*)| per case " + ", ".join(f"{v:.4f}" for v in dps.values())
            + "; |SSIM(IPVA) - SSIM(x*)| " + ", ".join(f"{v:.5f}" for v in dss.values()))
    assert max(dps.values()) < 0.01 and max(dss.values()) < 0.001     # PSNR in dB, SSIM
    L.check("ipva_smallest_fig", all(min(E.LABELS, key=lambda m: pc[c]["methods"][m]["err"]) == "IPVA-1" for c in E.CASES),
            "'there IPVA has the smallest value in all three cases': smallest relative error to x* among all six methods",
            FIG, "; ".join(f"{c}: {pc[c]['methods']['IPVA-1']['err']:.2e}" for c in E.CASES))
    L.check("figure_values", True, "the numbers below the panels of Figure 2 are printed by make_figure2.py from "
            "figure2_values.json (PSNR, SSIM, error of b, IPVA, FBi-PG, BiG-SAM, DIPGM-VA)", FIG,
            " | ".join(c + ": " + "; ".join(f"{E.name(m)} {pc[c]['methods'][m]['psnr']:.2f}/{pc[c]['methods'][m]['ssim']:.3f}/"
                                             f"{pc[c]['methods'][m]['err']:.1e}" for m in four) for c in E.CASES))
    L.claim("fig_caption", f"reconstructions after {E.COST} evaluations of $\\nabla f$", int(E.COST), FIG, "cost of the reconstructions")
    return {"psnr_spread_four_methods": spread}


def check_text(L: Ledger, n_best: int) -> None:
    L.claim("abstract", "IPVA attains a smaller relative error to the solution than five existing methods with the same number "
            "of gradient evaluations", n_best, RUNS, "true in all 18 cases (n_best = 18)", where="abstract")
    L.claim("concl", "In all 18 cases, IPVA has the smallest relative error to the solution among the six methods", n_best,
            RUNS, "n_best = 18", where="conclusion")
    L.claim("intro_iii", "We compare IPVA with five existing methods on image inpainting", 5, "-", "five compared methods",
            where="introduction")
    L.claim("intro_iii_result", "In all 18 cases, IPVA has the smallest relative error to the solution.", n_best, RUNS,
            "n_best = 18", where="introduction")


def check_versions(L: Ledger) -> dict:
    vers = {"Python": platform.python_version(), "NumPy": np.__version__, "SciPy": scipy.__version__, "scikit-image": skimage.__version__}
    L.check("versions", vers["Python"].startswith("3.13") and vers["NumPy"].startswith("2.2.") and vers["SciPy"].startswith("1.16.")
            and vers["scikit-image"].startswith("0.25."), "software versions Python 3.13, NumPy 2.2, SciPy 1.16, scikit-image 0.25",
            "this interpreter", str(vers))
    return vers


def load_output(fname: str) -> dict:
    path = E.OUT / fname
    if not path.exists():
        sys.exit(f"FAIL: {path} is missing; run make_figure1.py and make_figure2.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def write_ledger(rows: list[dict]) -> None:
    md = [f"# Numbers of section 4 (image inpainting, C = {E.COST})", "",
          "Generated by `reproduce/check_numbers.py`. Every number and every checked statement of section 4, its two tables, the "
          "conclusion and the numerical sentence of the abstract. Files are relative to `results/` or `outputs/`. "
          "'IPVA' is the label 'IPVA-1' of exp21_P4.json.", "",
          "| ID | text in the paper | value | where | source | computation |", "|---|---|---|---|---|---|"]
    for r in rows:
        v = r["value"]
        vs = (f"{v:.6g}" if isinstance(v, float) else
              "[" + ", ".join(f"{x:.6g}" for x in v) + "]" if isinstance(v, list) else str(v))
        md.append(f"| `{r['id']}` | {r['text']} | {vs} | {r['where']} | {r['source']} | {r['computation']} |")
    LEDGER.write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    stored_only = "--stored-only" in sys.argv
    L = Ledger()
    d = E.exp21()
    trials = E.random_trials(d)
    keys = [t["key"] for t in trials]
    sel = E.selection(d)
    err = {k: {m: E.m1(t, m) for m in E.LABELS} for k, t in zip(keys, trials)}
    others = [m for m in E.LABELS if m != "IPVA-1"]

    check_cases(L, keys)
    admm_res, admm_settings = check_admm(L, keys)
    med = check_tables(L, trials, sel)
    n_best, ratio = check_ranking(L, keys, err, others)
    check_selection(L, sel)
    figures: dict = {}
    if not stored_only:
        fig = load_output(FIG)
        figures = {**check_curves(L, load_output(CUR), err, others), **check_reconstructions(L, fig), "figure": fig}
    check_text(L, n_best)
    vers = check_versions(L)

    if not stored_only:
        out = {"cost": int(E.COST), "cases_fig": E.CASES, "cases": keys,
               "median_over_18": {E.name(m): med[m]["all"] for m in E.LABELS},
               "medians_by_cell": {E.name(m): med[m] for m in E.LABELS},
               "n_cases_ipva_smallest": n_best, "ratio_fbipg_ipva": [min(ratio), max(ratio)], **figures,
               "admm_max_residual": admm_res, "admm_settings": admm_settings, "versions": vers, "ledger": L.rows}
        OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
        write_ledger(L.rows)
        print("wrote", OUT, LEDGER)
    for r in L.rows:
        print(f"{r['id']:20s} {str(r['text'])[:110]}")
    print("checked against", REGISTERED)
    print("PASS")


if __name__ == "__main__":
    main()
