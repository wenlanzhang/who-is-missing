#!/usr/bin/env python3
"""
A5 — Per-city coverage models, and the bridge from one city to all of them.

All modelling in this project happens in Python; R only draws. This script fits
the single-city version of the coverage model and writes everything the figures
need, so no estimation happens inside a figure script.

The specification deliberately matches the pooled model in 02_selection_models.py
(M3) minus the city fixed effects, which a single-city fit cannot have:

    city level   logit P(published) = a + b*z_GRDI + g1*z_logWP + g2*z_logWP^2
    pooled       logit P(published) = a_c + b*z_GRDI + g1*z_logWP + g2*z_logWP^2

Same right-hand side, one extra term when cities are pooled. That makes the
single-city figure and the cross-city figure comparable rather than two unrelated
models, which is otherwise a real gap in the story: the talk moves from one city
to eighteen, and the audience should be looking at the same object twice.

Two predictions are produced for every tile:

  p_actual     using the tile's own population — what the model says about the
               data as it is, and it should track the observed rates
  p_equalpop   with log population held at the city mean — the counterfactual
               "what if every neighbourhood had the same number of residents?"

The gap between them is the share of the coverage gradient that population
explains, drawn rather than asserted.

Outputs
  outputs/analysis/A5_city_tiles.csv     one row per tile, with both predictions
  outputs/analysis/A5_city_curves.csv    fitted curve per city over z_GRDI
  outputs/analysis/A5_city_deciles.csv   observed publication rate per decile
  outputs/analysis/A5_pooled_curve.csv   the pooled model on the same axis

Usage:
  python analysis/05_city_models.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "analysis" / "panel" / "tile_panel.parquet"
OUT = ROOT / "outputs" / "analysis"

GRID_N = 200          # points along the fitted curve
Z_LO, Z_HI = -2.5, 2.5


def load() -> pd.DataFrame:
    d = pd.read_parquet(PANEL)
    d = d[d.in_eligible].dropna(subset=["z_poverty_mean", "z_log_wp"]).copy()
    d["z_log_wp2"] = d.z_log_wp ** 2
    return d.reset_index(drop=True)


def fit(y, X, groups):
    """Logit with cluster-robust errors, matching the inference used elsewhere."""
    return sm.Logit(y, X).fit(disp=0, method="bfgs", maxiter=500,
                              cov_type="cluster", cov_kwds={"groups": groups})


def city_models(d: pd.DataFrame):
    """One model per city, plus per-tile predictions and a fitted curve."""
    tiles, curves, deciles = [], [], []
    for city, g in d.groupby("city"):
        g = g.copy()
        if g.published.nunique() < 2 or len(g) < 30:
            continue

        cols = ["z_poverty_mean", "z_log_wp", "z_log_wp2"]
        X = sm.add_constant(g[cols].astype(float)).to_numpy()
        try:
            res = fit(g.published.to_numpy(), X, g.blk10.to_numpy())
        except Exception:
            print(f"  {city}: did not converge, skipped")
            continue

        # Prediction 1: each tile's own population.
        g["p_actual"] = res.predict(X)
        # Prediction 2: population held at the city mean (z = 0, so the squared
        # term is 0 too).
        Xc = X.copy()
        Xc[:, cols.index("z_log_wp") + 1] = 0.0
        Xc[:, cols.index("z_log_wp2") + 1] = 0.0
        g["p_equalpop"] = res.predict(Xc)
        tiles.append(g[["country", "city", "quadkey", "poverty_mean", "worldpop_count",
                        "z_poverty_mean", "published", "grdi_decile",
                        "p_actual", "p_equalpop"]])

        # Fitted curve at equal population, across the city's own GRDI range.
        mu, sd = g.poverty_mean.mean(), g.poverty_mean.std(ddof=0)
        z = np.linspace(max(Z_LO, g.z_poverty_mean.min()),
                        min(Z_HI, g.z_poverty_mean.max()), GRID_N)
        Xn = np.column_stack([np.ones_like(z), z, np.zeros_like(z), np.zeros_like(z)])
        eta = Xn @ res.params
        se = np.sqrt(np.einsum("ij,jk,ik->i", Xn, res.cov_params(), Xn))
        curves.append(pd.DataFrame({
            "city": city, "country": g.country.iloc[0], "z_grdi": z,
            "poverty_mean": mu + z * sd,
            "fit": 1 / (1 + np.exp(-eta)),
            "lo": 1 / (1 + np.exp(-(eta - 1.96 * se))),
            "hi": 1 / (1 + np.exp(-(eta + 1.96 * se))),
        }))

        dec = g.groupby("grdi_decile").agg(
            median_grdi=("poverty_mean", "median"),
            median_wp=("worldpop_count", "median"),
            observed=("published", "mean"),
            fit_actual=("p_actual", "mean"),
            fit_equalpop=("p_equalpop", "mean"),
            n=("published", "size"),
        ).reset_index()
        dec.insert(0, "city", city)
        deciles.append(dec)

        b = res.params[1]
        print(f"  {city:<20} n={len(g):<5} OR={np.exp(b):.4f}  "
              f"equal-pop D1->D10: {100*dec.fit_equalpop.iloc[0]:.0f}% -> "
              f"{100*dec.fit_equalpop.iloc[-1]:.0f}%  "
              f"(observed {100*dec.observed.iloc[0]:.0f}% -> {100*dec.observed.iloc[-1]:.0f}%)")

    return (pd.concat(tiles, ignore_index=True),
            pd.concat(curves, ignore_index=True),
            pd.concat(deciles, ignore_index=True))


def pooled_curve(d: pd.DataFrame) -> pd.DataFrame:
    """Same specification with city fixed effects, on the same z_GRDI axis.

    The intercept is taken at the average city, so this curve is the pooled
    analogue of the per-city ones and can be drawn on top of them.
    """
    cols = ["z_poverty_mean", "z_log_wp", "z_log_wp2"]
    fe = pd.get_dummies(d.city, drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([d[cols].astype(float).reset_index(drop=True),
                                   fe.reset_index(drop=True)], axis=1))
    res = fit(d.published.to_numpy(), X.to_numpy(), d.blk10.to_numpy())

    # Average adjusted prediction, not prediction-at-the-mean. For each point on
    # the grid every tile keeps its own city fixed effect, has its deprivation set
    # to that value and its population set to the mean, and the predicted
    # probabilities are averaged. Because the logit is non-linear these differ
    # (Jensen), and averaging is the convention that matches the decile curve in
    # 02_selection_models.py — evaluating at the mean tile would make the two
    # figures disagree by ~20 points at the deprived end.
    Xa = X.to_numpy().astype(float)
    V = res.cov_params()
    z = np.linspace(Z_LO, Z_HI, GRID_N)
    pred, lo, hi = [], [], []
    for zi in z:
        Xn = Xa.copy()
        Xn[:, 1] = zi        # deprivation
        Xn[:, 2] = 0.0       # log population at its mean
        Xn[:, 3] = 0.0       # and its square
        p = 1 / (1 + np.exp(-(Xn @ res.params)))
        # Delta method on the averaged probability.
        grad = ((p * (1 - p))[:, None] * Xn).mean(axis=0)
        se = float(np.sqrt(grad @ V @ grad))
        m = float(p.mean())
        pred.append(m)
        lo.append(max(0.0, m - 1.96 * se))
        hi.append(min(1.0, m + 1.96 * se))
    print(f"\n  pooled OR = {np.exp(res.params[1]):.4f} "
          f"({len(d):,} tiles, {d.city.nunique()} cities)")
    return pd.DataFrame({"z_grdi": z, "fit": pred, "lo": lo, "hi": hi})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    print(f"Fitting per-city coverage models ({d.city.nunique()} cities, {len(d):,} tiles)\n")
    tiles, curves, deciles = city_models(d)

    est = d[d.groupby("city")["published"].transform("nunique") > 1]
    pooled = pooled_curve(est)

    tiles.to_csv(OUT / "A5_city_tiles.csv", index=False)
    curves.to_csv(OUT / "A5_city_curves.csv", index=False)
    deciles.to_csv(OUT / "A5_city_deciles.csv", index=False)
    pooled.to_csv(OUT / "A5_pooled_curve.csv", index=False)
    print(f"\nWrote 4 tables to {OUT}  ({curves.city.nunique()} cities modelled)")


if __name__ == "__main__":
    main()
