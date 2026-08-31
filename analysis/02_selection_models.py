#!/usr/bin/env python3
"""
A2 — The extensive margin: does deprivation predict whether Meta publishes a tile?

The existing 03a–03f models ask "among tiles Meta published, is the Meta/WorldPop
allocation residual related to deprivation?" That question conditions on
publication, which is itself the outcome of Meta's user-density and privacy
thresholds. This script asks the prior question:

    P(published_i = 1) = f(GRDI_i | population, urbanicity, neighbourhood, city)

Everything is a *within-city* comparison (city fixed effects, city-demeaned
predictors), so no cross-city difference in deprivation level or city size can
drive the estimate. Standard errors are clustered on coarse quadkey blocks
because publication is strongly spatially autocorrelated.

Models
  M1  GRDI only                          — raw association
  M2  + log WorldPop                     — same population, different deprivation
  M3  + log WorldPop + squared           — flexible density control
  M4  + GHSL settlement class            — urban centre vs town vs rural held fixed
  M5  + spatial lag of GRDI              — SLX; neighbourhood deprivation held fixed
  M6  + spatial lag of log WorldPop      — full SLX, the spatial-confounding guard
  M7  GRDI orthogonalised to density     — circularity check (see analysis/README.md)

Outputs
  outputs/analysis/A2_extensive_margin_ladder.csv
  outputs/analysis/A2_extensive_margin_by_city.csv
  outputs/analysis/A2_cluster_bandwidth.csv
  outputs/analysis/A2_dose_response_grdi_decile.csv

Usage:
  python analysis/02_selection_models.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "analysis" / "panel" / "tile_panel.parquet"
OUT = ROOT / "outputs" / "analysis"

# Zoom-14 tiles are ~2.4 km at the equator; k=8 is the queen-contiguity analogue
# on a grid with holes (water, out-of-clip tiles) where true contiguity is ragged.
KNN_K = 8


def load_panel() -> tuple[pd.DataFrame, list[str]]:
    """Eligible tiles, minus cities where publication has no within-city variation.

    Four cities (Medan, Banda Aceh, Colombo, Barranquilla) publish 100% of their
    eligible tiles. Their city fixed effect perfectly predicts the outcome, so
    they contribute nothing to a within-city likelihood and only cause complete
    separation. Dropping them is what a conditional FE logit does implicitly;
    doing it explicitly keeps the Hessian invertible and the sample honest.
    """
    if not PANEL.exists():
        raise SystemExit(f"Missing {PANEL}. Run: python analysis/01_build_panel.py")
    d = pd.read_parquet(PANEL)
    d = d[d.in_eligible].dropna(subset=["z_poverty_mean", "z_log_wp"])
    varies = d.groupby("city")["published"].transform("nunique") > 1
    dropped = sorted(d.loc[~varies, "city"].unique())
    return d[varies].reset_index(drop=True), dropped


def spatial_lags(d: pd.DataFrame) -> pd.DataFrame:
    """Row-standardised KNN spatial lag of GRDI and log population, within city.

    Weights are built per city because tiles in different cities are never
    neighbours, and a pooled KNN would otherwise link distant urban areas.
    """
    from libpysal.weights import KNN

    d = d.copy()
    d["w_grdi"] = np.nan
    d["w_logwp"] = np.nan
    for city, g in d.groupby("city"):
        if len(g) <= KNN_K:
            continue
        w = KNN.from_array(g[["lon", "lat"]].to_numpy(), k=KNN_K)
        w.transform = "r"
        from libpysal.weights.spatial_lag import lag_spatial
        d.loc[g.index, "w_grdi"] = lag_spatial(w, g["z_poverty_mean"].to_numpy())
        d.loc[g.index, "w_logwp"] = lag_spatial(w, g["z_log_wp"].to_numpy())
    return d


def design(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Covariates + city fixed effects + intercept."""
    fe = pd.get_dummies(d["city"], prefix="city", drop_first=True).astype(float)
    X = pd.concat([d[cols].astype(float).reset_index(drop=True),
                   fe.reset_index(drop=True)], axis=1)
    return sm.add_constant(X)


def fit_logit(d: pd.DataFrame, cols: list[str], cluster: str = "blk10"):
    X = design(d, cols)
    return sm.Logit(d["published"].to_numpy(), X.to_numpy()).fit(
        disp=0, method="bfgs", maxiter=500,
        cov_type="cluster", cov_kwds={"groups": d[cluster].to_numpy()},
    ), X.columns.tolist()


def report(res, names, term="z_poverty_mean") -> dict:
    i = names.index(term)
    b, se = res.params[i], res.bse[i]
    return {
        "coef": b, "se": se, "p": res.pvalues[i],
        "OR": np.exp(b), "OR_lo": np.exp(b - 1.96 * se), "OR_hi": np.exp(b + 1.96 * se),
    }


def ladder(d: pd.DataFrame) -> pd.DataFrame:
    """M1–M7: add one confounder family at a time and watch the estimate move."""
    smod = pd.get_dummies(d["smod_class"], prefix="smod", drop_first=True).astype(float)
    d = pd.concat([d, smod], axis=1)
    smod_cols = list(smod.columns)

    # M7: strip out everything GRDI shares with population density and settlement
    # class, keeping only the orthogonal deprivation signal.
    d["z_log_wp2"] = d["z_log_wp"] ** 2
    Z = design(d, ["z_log_wp", "z_log_wp2"] + smod_cols)
    resid = sm.OLS(d["z_poverty_mean"].to_numpy(), Z.to_numpy()).fit().resid
    d["grdi_orth"] = resid / resid.std()

    specs = [
        ("M1  GRDI only", ["z_poverty_mean"], "z_poverty_mean"),
        ("M2  + log WorldPop", ["z_poverty_mean", "z_log_wp"], "z_poverty_mean"),
        ("M3  + log WP squared", ["z_poverty_mean", "z_log_wp", "z_log_wp2"], "z_poverty_mean"),
        ("M4  + GHSL settlement class",
         ["z_poverty_mean", "z_log_wp", "z_log_wp2"] + smod_cols, "z_poverty_mean"),
        ("M5  + spatial lag of GRDI",
         ["z_poverty_mean", "z_log_wp", "z_log_wp2"] + smod_cols + ["w_grdi"], "z_poverty_mean"),
        ("M6  + spatial lag of log WP",
         ["z_poverty_mean", "z_log_wp", "z_log_wp2"] + smod_cols + ["w_grdi", "w_logwp"],
         "z_poverty_mean"),
        ("M7  GRDI orthogonal to density",
         ["grdi_orth", "z_log_wp", "z_log_wp2"] + smod_cols, "grdi_orth"),
    ]

    rows = []
    for label, cols, term in specs:
        sub = d.dropna(subset=cols).reset_index(drop=True)
        res, names = fit_logit(sub, cols)
        r = report(res, names, term)
        r.update(model=label, n=len(sub), n_clusters=sub["blk10"].nunique())
        rows.append(r)
        print(f"  {label:<32} OR={r['OR']:.3f}  [{r['OR_lo']:.3f},{r['OR_hi']:.3f}]  p={r['p']:.2e}")

    # Linear probability model as a functional-form check: the logit OR is a
    # ratio, the LPM coefficient is directly readable as percentage points.
    sub = d.dropna(subset=["z_poverty_mean", "z_log_wp"]).reset_index(drop=True)
    X = design(sub, ["z_poverty_mean", "z_log_wp"])
    lpm = sm.OLS(sub["published"].to_numpy(), X.to_numpy()).fit(
        cov_type="cluster", cov_kwds={"groups": sub["blk10"].to_numpy()})
    i = X.columns.tolist().index("z_poverty_mean")
    rows.append({"model": "LPM  GRDI + log WP (pp)", "n": len(sub),
                 "n_clusters": sub["blk10"].nunique(),
                 "coef": lpm.params[i], "se": lpm.bse[i], "p": lpm.pvalues[i],
                 "OR": np.nan, "OR_lo": np.nan, "OR_hi": np.nan})
    print(f"  {'LPM  GRDI + log WP':<32} {100*lpm.params[i]:+.1f} pp per SD  p={lpm.pvalues[i]:.2e}")

    cols = ["model", "n", "n_clusters", "coef", "se", "p", "OR", "OR_lo", "OR_hi"]
    return pd.DataFrame(rows)[cols]


def slx_total(d: pd.DataFrame) -> pd.DataFrame:
    """Total SLX effect of deprivation = own-tile + neighbourhood.

    GRDI is a 1 km raster and deprivation is spatially smooth, so a tile's own
    GRDI and its neighbours' mean GRDI are near-collinear at zoom 14. The split
    of the effect between them is therefore not separately identified, and the
    *direct* coefficient in M5/M6 collapses toward zero for that reason rather
    than because deprivation stops mattering. The identified quantity is the sum
    beta_own + beta_lag: the effect of shifting a whole neighbourhood up 1 SD of
    deprivation. Variance comes from the delta method on the full covariance.
    """
    smod = pd.get_dummies(d["smod_class"], prefix="smod", drop_first=True).astype(float)
    d = pd.concat([d.reset_index(drop=True), smod.reset_index(drop=True)], axis=1)
    d["z_log_wp2"] = d["z_log_wp"] ** 2
    smod_cols = list(smod.columns)

    rows = []
    specs = [
        ("SLX (GRDI lag)", ["z_poverty_mean", "z_log_wp", "z_log_wp2"] + smod_cols + ["w_grdi"]),
        ("SLX (GRDI + WP lags)",
         ["z_poverty_mean", "z_log_wp", "z_log_wp2"] + smod_cols + ["w_grdi", "w_logwp"]),
    ]
    corr = d[["z_poverty_mean", "w_grdi"]].corr().iloc[0, 1]
    print(f"  corr(own GRDI, neighbour mean GRDI) = {corr:.3f}"
          "  <- why the direct effect alone is not identified")
    for label, cols in specs:
        sub = d.dropna(subset=cols).reset_index(drop=True)
        res, names = fit_logit(sub, cols)
        i, j = names.index("z_poverty_mean"), names.index("w_grdi")
        total = res.params[i] + res.params[j]
        V = res.cov_params()
        se = float(np.sqrt(V[i, i] + V[j, j] + 2 * V[i, j]))
        z = total / se
        rows.append({
            "model": label, "n": len(sub),
            "direct": res.params[i], "indirect": res.params[j], "total": total, "se": se,
            "OR_total": np.exp(total), "OR_lo": np.exp(total - 1.96 * se),
            "OR_hi": np.exp(total + 1.96 * se),
            "p": 2 * (1 - stats.norm.cdf(abs(z))),
        })
        r = rows[-1]
        print(f"  {label:<24} direct={r['direct']:+.2f} indirect={r['indirect']:+.2f} "
              f"TOTAL OR={r['OR_total']:.3f} [{r['OR_lo']:.3f},{r['OR_hi']:.3f}] p={r['p']:.2e}")
    return pd.DataFrame(rows)


def adjusted_dose_response(d: pd.DataFrame) -> pd.DataFrame:
    """Publication rate by GRDI decile, raw and holding population fixed.

    The raw gradient partly reflects that deprived tiles hold fewer people, and
    Meta's privacy threshold is a count threshold. The adjusted column is the
    model-predicted publication probability from M3 with log WorldPop set to the
    overall mean, so only deprivation varies.

    The squared term is set to the *square of the held value*, not to its own
    mean. Setting every covariate to its sample mean is the usual recipe but it
    is wrong for a polynomial: z_log_wp is standardised, so its mean is 0 while
    the mean of its square is 1.0, and holding both at their means describes a
    tile whose population is average and whose squared deviation is 1 — no tile
    at all. It is worth 10 points at the most deprived decile. This is also what
    makes the curve agree with pooled_curve() in 05_city_models.py, which holds
    both at 0.

    Settlement class is deliberately not in this model. Holding it at observed
    values while calling the line "settlement type held fixed" would have been
    wrong, and its "Urban centre" category is 100% published, which is the same
    complete-separation problem that excludes four cities from the sample. It
    changes the deprivation odds ratio by 0.002. Dropping it also makes this
    curve agree with the per-city curves in 05_city_models.py, which use the
    same specification.
    """
    d = d.reset_index(drop=True).copy()
    d["z_log_wp2"] = d["z_log_wp"] ** 2
    cols = ["z_poverty_mean", "z_log_wp", "z_log_wp2"]
    sub = d.dropna(subset=cols + ["grdi_decile"]).reset_index(drop=True)
    res, names = fit_logit(sub, cols)

    X = design(sub, cols)
    Xc = X.copy()
    held = sub["z_log_wp"].mean()
    Xc["z_log_wp"] = held
    Xc["z_log_wp2"] = held ** 2
    sub["p_adj"] = res.predict(Xc.to_numpy())

    g = sub.groupby("grdi_decile").agg(
        n_tiles=("published", "size"),
        pub_rate_raw=("published", "mean"),
        pub_rate_adj=("p_adj", "mean"),
        median_grdi=("poverty_mean", "median"),
        median_wp=("worldpop_count", "median"),
    ).reset_index()
    print("\n  decile  raw    adjusted  median_GRDI  median_WP")
    for _, r in g.iterrows():
        print(f"    {int(r.grdi_decile):>2}    {r.pub_rate_raw:.3f}   {r.pub_rate_adj:.3f}"
              f"     {r.median_grdi:6.2f}  {r.median_wp:9.0f}")
    return g


def cluster_bandwidths(d: pd.DataFrame) -> pd.DataFrame:
    """Same point estimate, four inference assumptions.

    If publication is spatially clustered, naive SEs are too small. Widening the
    cluster from zoom-12 (~10 km) to zoom-8 (~150 km) or to the whole city shows
    how much of the significance survives conservative inference.
    """
    cols = ["z_poverty_mean", "z_log_wp"]
    rows = []
    for lab, g in [("none (naive)", None), ("quadkey z12 (~10km)", "blk12"),
                   ("quadkey z10 (~39km)", "blk10"), ("quadkey z8 (~157km)", "blk8"),
                   ("city", "city")]:
        X = design(d, cols)
        kw = {} if g is None else {"cov_type": "cluster", "cov_kwds": {"groups": d[g].to_numpy()}}
        res = sm.Logit(d["published"].to_numpy(), X.to_numpy()).fit(
            disp=0, method="bfgs", maxiter=500, **kw)
        r = report(res, X.columns.tolist())
        r.update(cluster=lab, n_clusters=(len(d) if g is None else d[g].nunique()))
        rows.append(r)
        print(f"  cluster={lab:<22} G={r['n_clusters']:<5} OR={r['OR']:.3f} "
              f"[{r['OR_lo']:.3f},{r['OR_hi']:.3f}] p={r['p']:.2e}")
    return pd.DataFrame(rows)[["cluster", "n_clusters", "coef", "se", "p", "OR", "OR_lo", "OR_hi"]]


def by_city(d: pd.DataFrame) -> pd.DataFrame:
    """One logit per city. The sign test across cities is the robustness claim.

    Uses the headline M3 right-hand side minus the city fixed effect, which a
    single-city fit cannot have, so this matches the per-city curves drawn in
    analysis/05_city_models.py. They were previously on different
    specifications, which meant F2 and F1b reported different odds ratios for
    the same city.
    """
    rows = []
    d = d.copy()
    d["z_log_wp2"] = d["z_log_wp"] ** 2
    for city, g in d.groupby("city"):
        row = {"city": city, "country": g.country.iloc[0], "n": len(g),
               "C_c": g.published.mean(), "OR": np.nan, "OR_lo": np.nan,
               "OR_hi": np.nan, "p": np.nan, "note": ""}
        if g.published.nunique() < 2:
            row["note"] = "no variation (100% published)"
        elif len(g) < 30:
            row["note"] = "too few tiles"
        else:
            try:
                X = sm.add_constant(
                    g[["z_poverty_mean", "z_log_wp", "z_log_wp2"]].astype(float))
                res = sm.Logit(g.published.to_numpy(), X.to_numpy()).fit(
                    disp=0, method="bfgs", maxiter=500)
                b, se = res.params[1], res.bse[1]
                row.update(OR=np.exp(b), OR_lo=np.exp(b - 1.96 * se),
                           OR_hi=np.exp(b + 1.96 * se), p=res.pvalues[1])
            except Exception as exc:                      # separation / singular Hessian
                row["note"] = f"did not converge: {type(exc).__name__}"
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("OR")
    est = out.dropna(subset=["OR"])
    n_neg = int((est.OR < 1).sum())
    sign_p = stats.binomtest(n_neg, len(est), 0.5, alternative="greater").pvalue
    print(f"\n  {n_neg}/{len(est)} estimable cities have OR<1 "
          f"(sign test p={sign_p:.2e}); {int((est.p < 0.05).sum())} significant at 5%")
    return out


def dose_response(d: pd.DataFrame) -> pd.DataFrame:
    """Publication rate by within-city GRDI decile — the shape of the selection."""
    g = d.groupby("grdi_decile").agg(
        n_tiles=("published", "size"),
        pub_rate=("published", "mean"),
        median_grdi=("poverty_mean", "median"),
        median_wp=("worldpop_count", "median"),
        wp_total=("worldpop_count", "sum"),
    ).reset_index()
    g["se"] = np.sqrt(g.pub_rate * (1 - g.pub_rate) / g.n_tiles)
    print("\n  decile  pub_rate  median_GRDI  median_WP")
    for _, r in g.iterrows():
        print(f"    {int(r.grdi_decile):>2}     {r.pub_rate:.3f}      {r.median_grdi:6.2f}"
              f"    {r.median_wp:9.0f}")
    return g


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d, dropped = load_panel()
    full = pd.read_parquet(PANEL)
    full = full[full.in_eligible].dropna(subset=["z_poverty_mean", "z_log_wp"])

    print(f"Eligible tiles: {len(full):,} in {full.city.nunique()} cities "
          f"(published {full.published.mean():.1%})")
    if dropped:
        print(f"Excluded from pooled models (100% published, no within-city "
              f"variation): {', '.join(dropped)}")
    print(f"Estimation sample: {len(d):,} tiles in {d.city.nunique()} cities\n")

    print("Building within-city KNN spatial lags...")
    d = spatial_lags(d)

    print("\n=== Extensive margin: P(Meta publishes tile), logit, city FE, SE clustered z10 ===")
    print("    OR is per +1 SD of within-city GRDI (higher = more deprived)\n")
    lad = ladder(d)
    lad.to_csv(OUT / "A2_extensive_margin_ladder.csv", index=False)

    print("\n=== SLX: total (own + neighbourhood) deprivation effect ===")
    slx = slx_total(d)
    slx.to_csv(OUT / "A2_slx_total_effect.csv", index=False)

    print("\n=== Inference: cluster bandwidth ladder (M2 spec) ===")
    cb = cluster_bandwidths(d)
    cb.to_csv(OUT / "A2_cluster_bandwidth.csv", index=False)

    print("\n=== Per-city extensive margin (GRDI | log WorldPop) ===")
    bc = by_city(full)
    bc.to_csv(OUT / "A2_extensive_margin_by_city.csv", index=False)
    print(bc[["city", "n", "C_c", "OR", "p", "note"]].to_string(index=False,
          float_format=lambda v: f"{v:.4f}"))

    print("\n=== Dose-response by within-city GRDI decile ===")
    dr = dose_response(full)
    dr.to_csv(OUT / "A2_dose_response_grdi_decile.csv", index=False)

    print("\n=== Dose-response holding population fixed ===")
    adj = adjusted_dose_response(d)
    adj.to_csv(OUT / "A2_dose_response_adjusted.csv", index=False)

    print(f"\nWrote 6 tables to {OUT}")


if __name__ == "__main__":
    main()
