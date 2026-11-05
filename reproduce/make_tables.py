"""Tables 1 and 2 of the paper from the stored results; nothing is re-run.

Reads results/exp21_P4.json.
Writes
    outputs/tables/main_results.tex     median relative error ||x_N - x*|| / ||x*|| (metric M1 of exp21_P4.json) of each method over the
                                        three random masks of each image and missing rate, and over all 18 random-mask cases; bold = smallest
                                        value of each column (asserted unique)
    outputs/tables/selected_params.tex  parameters of the compared methods: sequences and step sizes, the sets of the coordinate search
                                        (taken from methods.method_specs) and the values selected at C = 5000 (exp21_P4.json 'selected')
    outputs/table_values.json           all values of both tables
Usage:  python reproduce/make_tables.py
"""
from __future__ import annotations

import json
import math

import common as E

# ----------------------------------------------------------------------------------------
# Table: median relative errors
# ----------------------------------------------------------------------------------------


def cell_medians(trials: list[dict]) -> dict:
    out: dict = {}
    for label in E.LABELS:
        row = {}
        for im in E.IMAGES:
            for rt in E.RATES:
                vals = [E.m1(t, label) for t in trials if t["key"].startswith(f"{im}_r{rt}_")]
                assert len(vals) == 3, (label, im, rt)
                row[f"{im}_r{rt}"] = E.median(vals)
        row["all"] = E.median([E.m1(t, label) for t in trials])
        out[label] = row
    return out


def main_table(med: dict) -> str:
    cols = [f"{im}_r{rt}" for im in E.IMAGES for rt in E.RATES] + ["all"]
    best = {}
    for c in cols:
        vals = sorted((med[m][c], m) for m in E.LABELS)
        assert E.sci_tab(vals[0][0]) != E.sci_tab(vals[1][0]), (c, vals[:2])     # bold is unique after rounding
        best[c] = vals[0][1]
    lines = [r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{E.IMG_LABEL[im]}}}" for im in E.IMAGES) + r" & All \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             "Method & " + " & ".join(rf"{rt}\%" for _ in E.IMAGES for rt in E.RATES) + r" & 18 cases \\", r"\midrule"]
    for m in E.LABELS:
        cells = []
        for c in cols:
            s = E.sci_tab(med[m][c])
            cells.append(rf"\textbf{{{s}}}" if best[c] == m else s)
        lines.append(f"{E.name(m)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------
# Table: parameters
# ----------------------------------------------------------------------------------------
SYM = {"delta": r"\delta", "c_mult": r"cL_f", "K": "K", "alpha": r"\nu", "mu": r"\mu", "gamma": r"\gamma", "a": "a"}

# sequences and step sizes (IPVA: label 'IPVA-1', see methods.run_method, family 'ipva1')
FORMULAS = {
    "IPVA-1": r"$\alpha_n=a_0(n+1)^{-\delta}$, $\bar\theta_n=\frac{n-1}{n+2}$, $\tau_n=K\alpha_n(n+1)^{-0.01}$",
    "Bi-SG": r"$\eta_k=a_0(k+1)^{-\nu}$, $c=1/L_f$",
    "BiG-SAM": r"$\alpha_n=a_0(n+1)^{-\delta}$, $c=1/L_f$, $s=2/(L_\mu+\sigma)$",
    "iBiG-SAM": r"$\alpha_n=a_0(n+1)^{-\delta}$, $c=1/L_f$, $s=2/(L_\mu+\sigma)$",
    "DIPGM-VA": r"$\alpha_n=a_0(n+1)^{-\delta}$, $c=1/L_f$, $s=1/(L_\mu+\sigma)$",
    "FBi-PG": r"$\beta_k=a_0(k+a)^{-\gamma}$ ($k\ge0$), step~size\newline $1/(L_f+2+\max\{1,\beta_0\}L_h)$",
}
# column widths of the parameter table (p columns, so that LaTeX breaks the lines inside the cells)
WIDTHS = ("1.7in", "1.45in", "1.05in")
# hard caps of the scale (methods.level_cap); IPVA-1 and FBi-PG have none
CAPS = {"Bi-SG": r"$a_0\le\min\{1,1/L_h\}$", "BiG-SAM": r"$\alpha_1\le0.95$"}
SAME_AS = {"iBiG-SAM": "BiG-SAM", "DIPGM-VA": "BiG-SAM"}


def fmt_val(axis: str, v: float) -> str:
    if axis in ("mu", "K"):
        e = round(math.log10(v))
        assert abs(v - 10.0 ** e) <= 1e-12 * v
        return f"10^{{{e}}}"
    return f"{v:g}"


def fmt_set(axis: str, vals: tuple) -> str:
    if axis == "mu":
        assert tuple(vals) == tuple(10.0 ** -k for k in range(1, 6))
        return r"$\mu\in\{10^{-k}:k=1,\dots,5\}$"
    return rf"${SYM[axis]}\in\{{" + r",\allowbreak ".join(fmt_val(axis, v) for v in vals) + r"\}$"


def sets_of(label: str, sel: dict) -> str:
    if label in SAME_AS:
        ref, mine = sel[SAME_AS[label]]["spec"].discrete, sel[label]["spec"].discrete
        assert ref == mine, (label, ref, mine)
        return f"the same sets as {SAME_AS[label]}"
    parts = [fmt_set(ax, vals) for ax, vals in sel[label]["spec"].discrete]
    if label in CAPS:
        parts.append(CAPS[label])
    return ", ".join(parts)


ALPHA1_CAP = 0.95          # alpha_1 = a0 2^{-delta} <= 0.95 for BiG-SAM, iBiG-SAM, DIPGM-VA (methods.level_cap)
CAPPED = ("BiG-SAM", "iBiG-SAM", "DIPGM-VA")


def a0_text(label: str, s: dict) -> str:
    """a0 with two significant digits, or three where two would give a printed alpha_1 = a0 2^{-delta} above the cap 0.95."""
    a0 = E.num2(s["a0"])
    if label in CAPPED and float(a0) * 2.0 ** (-s["prm"]["delta"]) > ALPHA1_CAP:
        a0 = f"{s['a0']:.3g}"
        assert float(a0) * 2.0 ** (-s["prm"]["delta"]) <= ALPHA1_CAP, (label, a0)
    return a0


def selected_of(label: str, sel: dict) -> str:
    s = sel[label]
    parts = [f"${SYM[ax]}={fmt_val(ax, s['prm'][ax])}$" for ax, _ in s["spec"].discrete]
    a0 = a0_text(label, s)
    if "e" in a0:
        mnt, ex = a0.split("e")
        a0 = rf"{mnt}\times10^{{{int(ex)}}}"
    parts.append(f"$a_0={a0}$")
    return ", ".join(parts)


def param_rows(sel: dict) -> dict:
    return {label: {"formulas": FORMULAS[label], "sets": sets_of(label, sel), "selected": selected_of(label, sel),
                    "prm": sel[label]["prm"], "a0": sel[label]["a0"], "N": sel[label]["N"]} for label in E.LABELS}


def param_table(rows: dict) -> str:
    cols = "".join(rf">{{\raggedright\arraybackslash}}p{{{w}}}" for w in WIDTHS)
    lines = [rf"\begin{{tabular}}{{@{{}}l{cols}@{{}}}}", r"\toprule",
             r"Method & Sequences and step sizes & Values that the tuning may select & Selected values \\", r"\midrule"]
    for i, label in enumerate(E.LABELS):
        r = rows[label]
        lines.append(f"{E.name(label)} & {r['formulas']} & {r['sets']} & {r['selected']}" + r" \\")
        if i < len(E.LABELS) - 1:
            lines.append(r"\addlinespace[3pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    d = E.exp21()
    trials = E.random_trials(d)
    med = cell_medians(trials)
    sel = E.selection(d)
    rows = param_rows(sel)
    (E.OUT / "tables").mkdir(parents=True, exist_ok=True)
    (E.OUT / "tables" / "main_results.tex").write_text(main_table(med), encoding="utf-8", newline="\r\n")     # CRLF, as the .tex files of the paper
    (E.OUT / "tables" / "selected_params.tex").write_text(param_table(rows), encoding="utf-8", newline="\r\n")
    out = {"source": "results/exp21_P4.json", "cost": int(E.COST), "cases": [t["key"] for t in trials], "medians": med,
           "params": {k: {kk: vv for kk, vv in v.items()} for k, v in rows.items()}}
    (E.OUT / "table_values.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for m in E.LABELS:
        print(f"{E.name(m):9s}", " ".join(E.sci_tab(med[m][c]) for c in med[m]), "|", rows[m]["selected"])
    print("wrote", E.OUT / "tables" / "main_results.tex", E.OUT / "tables" / "selected_params.tex", E.OUT / "table_values.json")


if __name__ == "__main__":
    main()
