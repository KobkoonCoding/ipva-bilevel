# ipva-bilevel

Code and stored results for the paper

> Kobkoon Janngam, Sayaphat Suksai, and Suthep Suantai, *Strong convergence of an inertial proximal gradient method for convex bilevel optimization with nonsmooth objectives*.

The paper proposes IPVA, the Inertial Proximal Gradient Algorithm with Vanishing Step Sizes. IPVA and the five compared methods are implemented in `reproduce/methods.py`. The package `bilevel/` holds the operators, the problem definition, and the ADMM solver for the reference solutions.

## What this repository reproduces

Section 4 of the paper (image inpainting, 18 random-mask cases, 5000 evaluations of the gradient of the inner smooth function for every method):

| Item in the paper | Script | Output |
|---|---|---|
| Table 1 (parameters of the methods) | `reproduce/make_tables.py` | `outputs/tables/selected_params.tex` |
| Table 2 (median relative errors) | `reproduce/make_tables.py` | `outputs/tables/main_results.tex` |
| Figure 1 (error curves of the three first-mask 50% cases) | `reproduce/make_figure1.py` | `outputs/figs/curves_P4.pdf` |
| Figure 2 (reconstructions) | `reproduce/make_figure2.py` | `outputs/figs/inpaint_P4.pdf` |
| Every number quoted in section 4 | `reproduce/check_numbers.py` | prints `PASS`; ledger in `outputs/numbers_ledger.md` |

## Quick start

Run all commands from the repository root, in this order.

```
pip install -r requirements.txt
python -m pytest -q
python reproduce/make_tables.py
python reproduce/make_figure1.py
python reproduce/make_figure2.py
python reproduce/check_numbers.py
```

The last command must print `PASS`. All outputs are written to `outputs/`, which git ignores.

- `make_tables.py` takes seconds and reads only the stored results.
- `make_figure1.py` re-runs the six methods on three cases and takes about 5 minutes on one core. It stops with an error if a final error differs from the stored value by more than the tolerance in `reproduce/common.py` (relative 1e-12; 1e-3 for iBiG-SAM, whose smoothing parameter 1e-4 amplifies rounding errors).
- `make_figure2.py` takes seconds. It needs the last iterates that `make_figure1.py` writes.
- `check_numbers.py` takes seconds. It recomputes each number and each checked statement and stops at the first failure. It compares each text with its row in `reproduce/numbers_ledger.md`, which lists the text printed in the paper. It also asserts the versions of Python, NumPy, SciPy, and scikit-image listed below.
- `tests/test_reproduce.py` runs `make_tables.py` and `check_numbers.py --stored-only`, which skips the checks of the two figures.
- Figures are drawn with Times New Roman when this font is installed. With another font, the numbers are the same but the figures are not pixel-identical to those of the paper.

## The stored data

The experiment has 18 random-mask cases: three test images, two missing rates (50% and 70%), and three random masks for each pair. Every method uses 5000 evaluations of the gradient of the inner smooth function. The parameters of each method were selected by a coordinate search of 60 runs on the Shepp-Logan tuning image (`search` in `reproduce/methods.py`). The solutions x* were computed by ADMM (`bilevel/reference.py`, `compute_reference` in `reproduce/data.py`).

- `results/exp21_P4.json` holds the runs and the selected parameters. The paper uses `selected["5000"]` and the values at cost `"5000"`.
- `results/exp21_p4_quality.json` holds the PSNR and SSIM values of the stored runs.
- `results/ref_exp18_<case>.npz` holds x* for the 18 cases and the tuning image.

The scripts of this repository read these files and re-run the methods only for Figure 1. They do not repeat the coordinate search or the runs on all 18 cases. The stored files also hold entries that the paper does not use: the cost `"1000"`, three central-block cases (`*_b25`), and the labels listed below.

## Names of the methods in the result files

`IPVA-1` is the IPVA setting of the paper. `Bi-SG`, `BiG-SAM`, `iBiG-SAM`, `DIPGM-VA`, and `FBi-PG` are the compared methods of the same names. The other entries of the stored files (`IPVA (delta<2)`, `DIPGM-VA (N=C)`) are not used. The metric `M1` is the relative error ||x_N - x*|| / ||x*||.

## Data sources

| Image | Where it comes from | Source and license |
|---|---|---|
| Retina | `skimage.data.retina()`, included in scikit-image | M. Häggström, Wikimedia Commons, CC0 1.0 |
| Brain slice | `data/brain_unc_slice.npy`, one slice of `skimage.data.brain()` | University of North Carolina Volume Rendering Test Data Set, public domain; the University of North Carolina asks that its origin be acknowledged |
| Chest radiograph | `data/cxr_wikimedia_cc0.jpg` | M. Häggström, Wikimedia Commons, CC0 1.0 |
| Shepp-Logan phantom (tuning image) | `skimage.data.shepp_logan_phantom()`, included in scikit-image | synthetic |

The MIT license of this repository covers the code. The images keep the licenses in this table.

## Software

Python 3.13, NumPy 2.2, SciPy 1.16, scikit-image 0.25, Matplotlib 3.10, Pillow 11. The exact versions are in `requirements.txt`.

CVXPY and Clarabel are not needed for the experiment, and they are not in `requirements.txt`. Five test cases of `tests/test_bilevel.py` use them and are skipped if CVXPY is not installed.

## How to cite

Please cite the paper and the archived code:

- K. Janngam, S. Suksai, and S. Suantai, *Strong convergence of an inertial proximal gradient method for convex bilevel optimization with nonsmooth objectives*, submitted to SIAM Journal on Optimization.
- K. Janngam, S. Suksai, and S. Suantai, *ipva-bilevel*, version 1.0.0, Zenodo, 2026, DOI: 10.5281/zenodo.23972283.

## License

MIT, see `LICENSE`.
