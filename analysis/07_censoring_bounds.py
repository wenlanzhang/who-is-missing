#!/usr/bin/env python3
"""
A7 — What if we fill in the missing cells? Bounds, imputation, and a placebo.

The worry this answers: the headline result uses Meta's *published* values, so is
it just an artefact of trusting Meta's own product? And what happens if we put
numbers into the suppressed cells instead of dropping them?

Filling suppressed cells with a random draw is a reasonable instinct but an
arbitrary answer — it gives one number with no way to know how much of it is the
imputation. There is a better version, because the censoring here is *known*, not
mysterious: every published baseline value in the PDC extracts is >= 10, so a
suppressed tile is not "unknown", it is known to lie in [0, 10). That converts an
imputation problem into a partial-identification problem with hard bounds.

  1. BOUNDS         re-estimate the intensive-margin tau with every censored cell
                    at its lower bound (0) and at its upper bound (10). The truth
                    is inside. If the sign is the same at both ends, the
                    conclusion holds for *every* possible value of the hidden data.
  2. IMPUTATION     the literal "put a number in" idea, at several fill values
                    including a random draw, shown as points inside those bounds.
  3. TOBIT          a likelihood estimate that uses the censoring rather than
                    guessing, as an interior point estimate.
  4. PLACEBO        randomly reassign which tiles are suppressed, holding the
                    number of suppressed tiles and the population gradient fixed.
                    If the deprivation gradient survives that, it is mechanical.

Outputs
  outputs/analysis/A7_censoring_bounds.csv
  outputs/analysis/A7_imputation_grid.csv
  outputs/analysis/A7_placebo_permutation.csv

Usage:
  python analysis/07_censoring_bounds.py [--draws 2000]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "analysis" / "panel" / "tile_panel.parquet"
OUT = ROOT / "outputs" / "analysis"

# Meta suppresses tiles below this many users; the minimum published value across
# the PDC baseline extracts is exactly 10.0, which pins the censoring point.
THRESHOLD = 10.0
SEED = 20260829


def parse_args():
    p = argparse.ArgumentParser(description="Censoring bounds, imputation, placebo")
    # 2,000 is the number quoted in analysis/README.md §5.4. Keep the two in step:
    # the empirical p-value floor is 1/draws, so changing this changes the result.
    p.add_argument("--draws", type=int, default=2000,
                   help="Permutation draws for the placebo test (default 2000).")
    return p.parse_args()


def load() -> pd.DataFrame:
    d = pd.read_parquet(PANEL)
    d = d[d.in_eligible].dropna(subset=["z_poverty_mean", "z_log_wp"]).reset_index(drop=True)
    return d


def tau_from_meta(d: pd.DataFrame, meta: np.ndarray) -> dict:
    """Intensive-margin tau on a *filled* grid.

    Shares are renormalised within city over the full eligible grid, so the
    estimand is the same log allocation ratio the pipeline uses, just defined on
    every tile rather than only the published ones. Tiles with a filled value of
    exactly zero have no defined log ratio and are dropped, which is why the
    lower bound is evaluated at a small epsilon as well.
    """
    g = d.copy()
    g["meta_fill"] = meta
    g["meta_share"] = g.meta_fill / g.groupby("city")["meta_fill"].transform("sum")
    g["wp_share"] = g.worldpop_count / g.groupby("city")["worldpop_count"].transform("sum")
    g = g[(g.meta_share > 0) & (g.wp_share > 0)]
    if g.empty or g.city.nunique() == 0:
        return {"tau": np.nan, "se": np.nan, "p": np.nan, "n": 0}
    g["y"] = np.log(g.meta_share / g.wp_share)
    fe = pd.get_dummies(g["city"], drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([g[["z_poverty_mean"]].reset_index(drop=True),
                                   fe.reset_index(drop=True)], axis=1))
    res = sm.OLS(g["y"].to_numpy(), X.to_numpy()).fit(
        cov_type="cluster", cov_kwds={"groups": g["blk10"].to_numpy()})
    i = X.columns.tolist().index("z_poverty_mean")
    return {"tau": res.params[i], "se": res.bse[i], "p": res.pvalues[i], "n": len(g)}


def fill(d: pd.DataFrame, value) -> np.ndarray:
    """Published tiles keep their value; suppressed tiles take `value`."""
    m = d.meta_baseline.to_numpy(dtype=float).copy()
    cens = d.published.to_numpy() == 0
    m[cens] = value if np.isscalar(value) else value[cens]
    return m


def bounds(d: pd.DataFrame) -> pd.DataFrame:
    """The identified interval: every censored cell at 0+eps, then at 10."""
    rng = np.random.default_rng(SEED)
    n_cens = int((d.published == 0).sum())
    specs = [
        ("Published only (pipeline)", d.meta_baseline.where(d.published == 1).to_numpy()),
        ("Lower bound: censored = 0.01", fill(d, 0.01)),
        ("Lower-ish: censored = 1", fill(d, 1.0)),
        ("Midpoint: censored = 5", fill(d, 5.0)),
        ("Upper bound: censored = 10", fill(d, THRESHOLD)),
        ("Random draw U(0,10)", fill(d, rng.uniform(0, THRESHOLD, len(d)))),
        ("Random draw U(0,10), seed 2", fill(d, np.random.default_rng(SEED + 1)
                                             .uniform(0, THRESHOLD, len(d)))),
    ]
    rows = []
    for label, m in specs:
        r = tau_from_meta(d, m)
        r["spec"] = label
        r["n_censored_filled"] = 0 if "Published only" in label else n_cens
        rows.append(r)
        print(f"  {label:<30} tau={r['tau']:+.4f}  se={r['se']:.4f}  "
              f"p={r['p']:.3g}  n={r['n']}")
    return pd.DataFrame(rows)[["spec", "n", "n_censored_filled", "tau", "se", "p"]]


def imputation_grid(d: pd.DataFrame) -> pd.DataFrame:
    """tau as a continuous function of the fill value, across the whole interval."""
    rows = []
    for v in [0.01, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        r = tau_from_meta(d, fill(d, float(v)))
        r["fill_value"] = v
        rows.append(r)
    out = pd.DataFrame(rows)[["fill_value", "n", "tau", "se", "p"]]
    print(f"  tau ranges from {out.tau.min():+.4f} to {out.tau.max():+.4f} "
          f"across fill values 0.01 to {THRESHOLD:g}")
    print(f"  all {'NEGATIVE' if (out.tau < 0).all() else 'MIXED SIGN'}; "
          f"max p-value {out.p.max():.3g}")
    return out


def tobit(d: pd.DataFrame) -> pd.DataFrame:
    """Left-censored regression of log(meta/worldpop) using the known threshold.

    Rather than filling the censored cells, this uses the fact that for a
    suppressed tile the latent log ratio is below log(10/worldpop_i), and
    integrates that region of the likelihood out.
    """
    from scipy import stats as st
    from scipy.optimize import minimize

    g = d.copy()
    g["cut"] = np.log(THRESHOLD / g.worldpop_count)
    g["y"] = np.log(g.meta_baseline / g.worldpop_count)
    obs = g.published == 1
    g.loc[~obs, "y"] = np.nan

    fe = pd.get_dummies(g["city"], drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([g[["z_poverty_mean"]].reset_index(drop=True),
                                   fe.reset_index(drop=True)], axis=1)).to_numpy(float)
    y = g["y"].to_numpy(float)
    cut = g["cut"].to_numpy(float)
    o = obs.to_numpy()

    def nll(theta):
        b, ls = theta[:-1], theta[-1]
        s = np.exp(np.clip(ls, -6, 6))
        xb = X @ b
        ll = np.zeros(len(y))
        ll[o] = st.norm.logpdf(y[o], xb[o], s)
        ll[~o] = st.norm.logcdf((cut[~o] - xb[~o]) / s)
        return -np.sum(ll)

    start = np.append(np.linalg.lstsq(X[o], y[o], rcond=None)[0], 0.0)
    res = minimize(nll, start, method="BFGS")
    b = res.x[0:len(start) - 1]
    try:
        se = np.sqrt(np.diag(res.hess_inv))[1]
    except Exception:
        se = np.nan
    tau = b[1]
    z = tau / se if np.isfinite(se) and se > 0 else np.nan
    p = 2 * st.norm.sf(abs(z)) if np.isfinite(z) else np.nan
    print(f"  Tobit tau={tau:+.4f}  se={se:.4f}  p={p:.3g}  "
          f"(censored {int((~o).sum())} of {len(o)})")
    print("  ! SE is from the BFGS inverse Hessian and is NOT cluster-robust. Given the\n"
          "    spatial autocorrelation in publication it is certainly too small; treat the\n"
          "    point estimate as informative and the p-value as an upper bound on precision.")
    if not res.success:
        # Reported rather than swallowed: the estimate sits comfortably inside
        # the bounds in section 1, which is the reason it is still worth quoting,
        # but "the optimiser did not declare success" has to travel with it.
        print(f"  ! BFGS did NOT report convergence ({res.message.strip()}).\n"
              "    Quote this as a supporting interior point only, never as a headline\n"
              "    estimate, and always alongside the bounds it falls inside.")
    return pd.DataFrame([{"model": "Left-censored (Tobit) on log(meta/worldpop)",
                          "n": len(o), "n_censored": int((~o).sum()),
                          "tau": tau, "se": se, "p": p, "converged": bool(res.success),
                          "se_note": "not cluster-robust; understated"}])


def placebo(d: pd.DataFrame, draws: int) -> pd.DataFrame:
    """Reassign suppression at random within city, keeping the count fixed.

    Two nulls. 'uniform' ignores population entirely. 'population-matched'
    reassigns within population deciles, so the fake suppression has the same
    relationship to density as the real thing and only the deprivation link is
    broken. The second is the one that matters.
    """
    rng = np.random.default_rng(SEED)
    # The design matrix never changes across draws — only the outcome does — so
    # build it once and reuse the pseudo-inverse.
    fe = pd.get_dummies(d["city"], drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([d[["z_poverty_mean", "z_log_wp"]].reset_index(drop=True),
                                   fe.reset_index(drop=True)], axis=1)).to_numpy(float)
    pinv_row = np.linalg.pinv(X)[1]          # row that yields the GRDI coefficient

    groups = {mode: list(d.groupby(["city"] if mode == "uniform"
                                   else ["city", "wp_decile"]).indices.values())
              for mode in ("uniform", "population-matched")}

    real = float(pinv_row @ d.published.to_numpy(float))
    rows = []
    for mode in ("uniform", "population-matched"):
        idx_sets = groups[mode]
        arr = np.empty(draws)
        base = d.published.to_numpy(float)
        for k in range(draws):
            pub = base.copy()
            for idx in idx_sets:
                pub[idx] = rng.permutation(pub[idx])
            arr[k] = pinv_row @ pub
        p_emp = float(np.mean(arr <= real))
        rows.append({"null": mode, "draws": draws, "observed_coef": real,
                     "null_mean": arr.mean(), "null_sd": arr.std(),
                     "null_q025": np.quantile(arr, 0.025),
                     "null_q975": np.quantile(arr, 0.975),
                     "excess_over_null": real - arr.mean(),
                     "empirical_p": max(p_emp, 1.0 / draws)})
        print(f"  {mode:<20} observed={real:+.4f}  null mean={arr.mean():+.4f} "
              f"[{np.quantile(arr,0.025):+.4f},{np.quantile(arr,0.975):+.4f}]  "
              f"p<={max(p_emp, 1.0/draws):.4f}")

    pm = rows[1]
    share = pm["null_mean"] / real
    print(f"\n  The population-matched null is not zero ({pm['null_mean']:+.4f}): coarse "
          f"density matching\n  alone reproduces {share:.0%} of the observed gradient. "
          f"The remaining {1-share:.0%}\n  ({real - pm['null_mean']:+.4f}) is deprivation "
          f"structure beyond population.")
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    d["wp_decile"] = d.groupby("city")["worldpop_count"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
    n_cens = int((d.published == 0).sum())
    print(f"Eligible tiles {len(d):,}; censored (no Meta value) {n_cens:,} "
          f"({n_cens/len(d):.1%})")
    print(f"Censoring point: published values are all >= {THRESHOLD:g}\n")

    print("=== 1-2. Bounds and imputation: intensive-margin tau on the filled grid ===")
    b = bounds(d)
    b.to_csv(OUT / "A7_censoring_bounds.csv", index=False)

    print("\n=== tau as a function of the fill value ===")
    grid = imputation_grid(d)
    grid.to_csv(OUT / "A7_imputation_grid.csv", index=False)

    print("\n=== 3. Tobit: use the censoring instead of guessing ===")
    tb = tobit(d)
    tb.to_csv(OUT / "A7_tobit.csv", index=False)

    print(f"\n=== 4. Placebo: reshuffle which tiles are suppressed ({args.draws} draws) ===")
    pl = placebo(d, args.draws)
    pl.to_csv(OUT / "A7_placebo_permutation.csv", index=False)

    print(f"\nWrote 4 tables to {OUT}")


if __name__ == "__main__":
    main()
