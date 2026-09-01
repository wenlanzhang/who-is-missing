#!/usr/bin/env python3
"""
A9 — Five sensitivity checks a discussant is likely to ask for.

  1. POPULATION FLOOR   Meta suppresses tiles below ~10 users. A tile with 30
                        residents can almost never clear that bar whatever its
                        deprivation, so part of the raw gradient could be a
                        mechanical small-tile effect that log(WorldPop) and its
                        square do not fully absorb. Re-estimate on progressively
                        larger tiles, where publication is feasible for anyone.
                        This is the most dangerous objection to the design.
  2. JACKKNIFE          Leave one city out, then one country out. Guards against
                        the pooled estimate resting on the two Ecuadorian cities,
                        which have the sparsest coverage in the sample.
  3. OUT-OF-SAMPLE      Add the three excluded cities back in. Two were dropped
                        for sparse coverage and one (Kisumu) for AOI truncation,
                        so this is a "the exclusions were not cherry-picking"
                        check rather than a better estimate - see the docstring
                        on out_of_sample() for why the two reasons differ.
  4. SPATIAL SCALE      Re-run at zoom 13 and 12 (coarser tiles). A modifiable
                        areal unit check: if the gradient only exists at zoom 14
                        it is a small-tile artefact.
  5. FUNCTIONAL FORM    GRDI as decile dummies and as a within-city rank, not
                        just a linear z-score. Note the decile-dummy contrast is
                        perfectly separated (the least deprived decile is ~100%
                        published), so its OR collapses to zero - informative as
                        a statement, not usable as an estimate.

Outputs
  outputs/analysis/A9_population_floor.csv
  outputs/analysis/A9_jackknife_city.csv
  outputs/analysis/A9_jackknife_country.csv
  outputs/analysis/A9_out_of_sample.csv
  outputs/analysis/A9_spatial_scale.csv
  outputs/analysis/A9_functional_form.csv

Usage:
  python analysis/09_sensitivity.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "analysis" / "panel" / "tile_panel.parquet"
PANEL_ALL = ROOT / "analysis" / "panel" / "tile_panel_all.parquet"
OUT = ROOT / "outputs" / "analysis"

BASE = ["z_poverty_mean", "z_log_wp"]


def load(path=PANEL) -> pd.DataFrame:
    d = pd.read_parquet(path)
    d = d[d.in_eligible].dropna(subset=BASE)
    return d[d.groupby("city")["published"].transform("nunique") > 1].reset_index(drop=True)


def logit_or(d: pd.DataFrame, cols=BASE, term="z_poverty_mean", cluster="blk10") -> dict:
    """M2 spec: logit, city FE, clustered SE. Returns the OR on `term`."""
    d = d.dropna(subset=cols)
    d = d[d.groupby("city")["published"].transform("nunique") > 1]
    if len(d) < 100 or d.published.nunique() < 2:
        return {"OR": np.nan, "OR_lo": np.nan, "OR_hi": np.nan, "p": np.nan,
                "n": len(d), "n_cities": d.city.nunique()}
    fe = pd.get_dummies(d.city, drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([d[cols].astype(float).reset_index(drop=True),
                                   fe.reset_index(drop=True)], axis=1))
    try:
        r = sm.Logit(d.published.to_numpy(), X.to_numpy()).fit(
            disp=0, method="bfgs", maxiter=500,
            cov_type="cluster", cov_kwds={"groups": d[cluster].to_numpy()})
    except Exception:
        return {"OR": np.nan, "OR_lo": np.nan, "OR_hi": np.nan, "p": np.nan,
                "n": len(d), "n_cities": d.city.nunique()}
    i = X.columns.tolist().index(term)
    b, se = r.params[i], r.bse[i]
    return {"OR": float(np.exp(b)), "OR_lo": float(np.exp(b - 1.96 * se)),
            "OR_hi": float(np.exp(b + 1.96 * se)), "p": float(r.pvalues[i]),
            "n": len(d), "n_cities": d.city.nunique()}


def population_floor(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for floor in [0, 10, 25, 50, 100, 250, 500]:
        s = d[d.worldpop_count >= floor]
        r = logit_or(s)
        r.update(floor=floor, pub_rate=float(s.published.mean()) if len(s) else np.nan)
        rows.append(r)
        if np.isfinite(r["OR"]):
            print(f"  WorldPop >= {floor:<4} n={r['n']:<5} cities={r['n_cities']:<3} "
                  f"published={r['pub_rate']:.3f}  OR={r['OR']:.3f} "
                  f"[{r['OR_lo']:.3f},{r['OR_hi']:.3f}]  p={r['p']:.1e}")
    return pd.DataFrame(rows)[["floor", "n", "n_cities", "pub_rate",
                               "OR", "OR_lo", "OR_hi", "p"]]


def jackknife(d: pd.DataFrame, by: str) -> pd.DataFrame:
    full = logit_or(d)
    rows = [dict(dropped="(none - full sample)", **full)]
    for v in sorted(d[by].unique()):
        r = logit_or(d[d[by] != v])
        rows.append(dict(dropped=v, **r))
    out = pd.DataFrame(rows)[["dropped", "n", "n_cities", "OR", "OR_lo", "OR_hi", "p"]]
    est = out[out.dropped != "(none - full sample)"].dropna(subset=["OR"])
    print(f"  full sample OR={full['OR']:.3f}")
    print(f"  leave-one-out range: {est.OR.min():.3f} to {est.OR.max():.3f} "
          f"across {len(est)} refits; max p = {est.p.max():.1e}")
    worst = est.loc[est.OR.idxmax()]
    print(f"  least favourable exclusion: drop {worst.dropped} -> OR={worst.OR:.3f} "
          f"[{worst.OR_lo:.3f},{worst.OR_hi:.3f}]")
    return out


def out_of_sample() -> pd.DataFrame:
    """Add the three excluded cities back.

    Two were excluded for sparse coverage (Nakuru, Garden Route) and one for a
    different reason: Kisumu's absent tiles mark the edge of the published event
    AOI, not a coverage decision (see analysis/12_aoi_check.py). Calling all
    three "the sparse cities" was wrong, and it matters — adding an AOI-truncated
    city back inflates the odds ratio for a reason that has nothing to do with
    deprivation, so this row is not a clean robustness check. Read it as "the
    exclusions were not cherry-picking", not as a better estimate.
    """
    if not PANEL_ALL.exists():
        print(f"  ! {PANEL_ALL} missing. It is written by:")
        print("    python analysis/01_build_panel.py")
        return pd.DataFrame()
    a = load(PANEL_ALL)
    d = load(PANEL)
    rows = []
    for label, s in [(f"study sample ({d.city.nunique()} cities)", d),
                     (f"+ the 3 excluded cities ({a.city.nunique()} cities)", a)]:
        r = logit_or(s)
        r.update(sample=label, pub_rate=float(s.published.mean()))
        rows.append(r)
        print(f"  {label:<38} n={r['n']:<6} cities={r['n_cities']:<3} "
              f"published={r['pub_rate']:.3f}  OR={r['OR']:.3f} "
              f"[{r['OR_lo']:.3f},{r['OR_hi']:.3f}]  p={r['p']:.1e}")
    return pd.DataFrame(rows)[["sample", "n", "n_cities", "pub_rate",
                               "OR", "OR_lo", "OR_hi", "p"]]


def spatial_scale(d: pd.DataFrame) -> pd.DataFrame:
    """Coarsen the grid by truncating the quadkey, then re-estimate.

    At a coarser zoom the natural outcome is no longer binary: a parent tile
    contains several children, some published and some not. The analogue of the
    publication probability is the *share* of child tiles published, so this is
    fitted by OLS rather than logit and the coefficient is in percentage points.
    Estimating at zoom 14 by the same OLS makes the three rows comparable.
    """
    rows = []
    for zoom in (14, 13, 12):
        s = d.copy()
        s["parent"] = s.quadkey.str[:zoom]
        g = s.groupby(["city", "country", "parent"]).agg(
            coverage=("published", "mean"),
            worldpop_count=("worldpop_count", "sum"),
            poverty_mean=("poverty_mean", "mean"),
            n_children=("published", "size"),
        ).reset_index()
        g["log_wp"] = np.log(g.worldpop_count.clip(lower=1e-3))
        for c in ("poverty_mean", "log_wp"):
            g["z_" + c] = g.groupby("city")[c].transform(
                lambda x: (x - x.mean()) / x.std(ddof=0))
        g = g.dropna(subset=BASE)
        g = g[g.groupby("city")["coverage"].transform("std") > 0]
        fe = pd.get_dummies(g.city, drop_first=True).astype(float)
        X = sm.add_constant(pd.concat([g[BASE].astype(float).reset_index(drop=True),
                                       fe.reset_index(drop=True)], axis=1))
        r = sm.OLS(g.coverage.to_numpy(), X.to_numpy()).fit(
            cov_type="cluster", cov_kwds={"groups": g.parent.str[:8].to_numpy()})
        i = X.columns.tolist().index("z_poverty_mean")
        rows.append({"zoom": zoom, "n_units": len(g),
                     "median_children": float(g.n_children.median()),
                     "coef_pp": 100 * r.params[i], "se_pp": 100 * r.bse[i],
                     "p": r.pvalues[i]})
        print(f"  zoom {zoom}  units={len(g):<5} median children={g.n_children.median():.0f}"
              f"   coverage slope = {100*r.params[i]:+.1f} pp per SD  "
              f"(se {100*r.bse[i]:.1f})  p={r.pvalues[i]:.1e}")
    return pd.DataFrame(rows)


def functional_form(d: pd.DataFrame) -> pd.DataFrame:
    """Linear z-score vs within-city rank vs decile dummies."""
    rows = []
    r = logit_or(d)
    r.update(form="linear z-score (main spec)")
    rows.append(r)
    print(f"  {'linear z-score (main spec)':<30} OR={r['OR']:.3f} "
          f"[{r['OR_lo']:.3f},{r['OR_hi']:.3f}]  p={r['p']:.1e}")

    s = d.copy()
    s["z_poverty_rank"] = s.groupby("city").poverty_mean.transform(
        lambda x: (x.rank(pct=True) - 0.5) / x.rank(pct=True).std(ddof=0))
    r = logit_or(s, cols=["z_poverty_rank", "z_log_wp"], term="z_poverty_rank")
    r.update(form="within-city rank (percentile)")
    rows.append(r)
    print(f"  {'within-city rank':<30} OR={r['OR']:.3f} "
          f"[{r['OR_lo']:.3f},{r['OR_hi']:.3f}]  p={r['p']:.1e}")

    # Decile dummies: no linearity imposed at all. Report decile 10 vs decile 1.
    s = d.dropna(subset=["grdi_decile"]).copy()
    dum = pd.get_dummies(s.grdi_decile.astype(int), prefix="d").astype(float)
    dum = dum.drop(columns=[c for c in ["d_1"] if c in dum])
    s = pd.concat([s.reset_index(drop=True), dum.reset_index(drop=True)], axis=1)
    cols = [c for c in dum.columns] + ["z_log_wp"]
    r = logit_or(s, cols=cols, term="d_10")
    r.update(form="decile dummies (D10 vs D1)")
    rows.append(r)
    print(f"  {'decile dummies (D10 vs D1)':<30} OR={r['OR']:.3f} "
          f"[{r['OR_lo']:.3f},{r['OR_hi']:.3f}]  p={r['p']:.1e}")
    return pd.DataFrame(rows)[["form", "n", "n_cities", "OR", "OR_lo", "OR_hi", "p"]]


def within_settlement_class(d: pd.DataFrame) -> pd.DataFrame:
    """Stratify by GHSL settlement class instead of pooling across it.

    WorldPop is not ground truth - it is a dasymetric model that redistributes
    census counts using satellite covariates overlapping those in GRDI. It enters
    here twice, as the eligibility rule and as the density control, so a check
    that does not lean on pooling across settlement types is worth having. GHSL
    SMOD is an independent product, so estimating separately within each class
    tests whether the gradient is an artefact of mixing urban and rural tiles.
    """
    rows = []
    # Iterate over the classes actually present rather than a hard-coded three.
    # A fixed list silently dropped the "Water" tiles: a SMOD centroid can land
    # on water while WorldPop still places people in the tile, so they are in the
    # estimation sample and have to be accounted for somewhere.
    classes = [c for c in d.smod_class.dropna().unique()]
    order = {"Urban centre": 0, "Town / semi-dense": 1, "Rural": 2, "Water": 3}
    for cls in sorted(classes, key=lambda c: order.get(c, 9)):
        s = d[d.smod_class == cls]
        # n_stratum / pub_stratum describe the class itself. logit_or's own n is
        # the estimable subset after dropping cities with no within-class
        # variation, and reads as 0 for a fully published class — which looks
        # like "no tiles" when it means "805 tiles, every one of them published".
        n_stratum, pub_stratum = len(s), float(s.published.mean()) if len(s) else np.nan
        for lab, cols in [("no density control", ["z_poverty_mean"]), ("+ log WP", BASE)]:
            r = logit_or(s, cols=cols)
            r.update(smod_class=cls, spec=lab, n_stratum=n_stratum,
                     pub_rate_stratum=pub_stratum)
            rows.append(r)
            status = (f"OR={r['OR']:.3f} [{r['OR_lo']:.3f},{r['OR_hi']:.3f}] p={r['p']:.1e}"
                      if np.isfinite(r["OR"]) else
                      f"not estimable ({pub_stratum:.1%} of its tiles published)")
            print(f"  {cls:<20} {lab:<20} stratum n={n_stratum:<5} "
                  f"estimable n={r['n']:<5} {status}")
    return pd.DataFrame(rows)[["smod_class", "spec", "n_stratum", "pub_rate_stratum",
                               "n", "n_cities", "OR", "OR_lo", "OR_hi", "p"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    print(f"Estimation sample: {len(d):,} tiles, {d.city.nunique()} cities\n")

    print("=== 1. Population floor: is it a mechanical small-tile effect? ===")
    print("    Meta suppresses below ~10 users, so tiny tiles can never qualify.\n")
    pf = population_floor(d)
    pf.to_csv(OUT / "A9_population_floor.csv", index=False)

    print("\n=== 2a. Leave-one-city-out ===")
    jc = jackknife(d, "city")
    jc.to_csv(OUT / "A9_jackknife_city.csv", index=False)

    print("\n=== 2b. Leave-one-country-out ===")
    jn = jackknife(d, "country")
    jn.to_csv(OUT / "A9_jackknife_country.csv", index=False)

    print("\n=== 3. Adding the three excluded cities back ===")
    oos = out_of_sample()
    if len(oos):
        oos.to_csv(OUT / "A9_out_of_sample.csv", index=False)

    print("\n=== 4. Spatial scale (modifiable areal unit check) ===")
    ss = spatial_scale(d)
    ss.to_csv(OUT / "A9_spatial_scale.csv", index=False)

    print("\n=== 5. Functional form of deprivation ===")
    ff = functional_form(d)
    ff.to_csv(OUT / "A9_functional_form.csv", index=False)

    print("\n=== 6. Within GHSL settlement class (does not lean on pooling) ===")
    ws = within_settlement_class(d)
    ws.to_csv(OUT / "A9_within_settlement_class.csv", index=False)

    print(f"\nWrote 7 tables to {OUT}")


if __name__ == "__main__":
    main()
