#!/usr/bin/env python3
"""
A3 — Why the original hypothesis returned a null, and what the data actually say.

The pipeline's estimand is the allocation residual log(meta_share / wp_share),
estimated on the tiles Meta published. That sample is not given by nature: it is
the output of Meta's own user-density and privacy thresholds. The original
hypothesis (more deprivation -> Meta under-represents) was therefore tested on
precisely the tiles that survived a deprivation-selective filter.

This script separates the two margins that the single residual confounds:

  extensive   whether Meta publishes a tile at all       (A2: strongly
                                                          deprivation-graded)
  intensive   given publication, how Meta's share
              compares with WorldPop's                   (the original estimand)

Two results here are negative, and both matter for how the finding is stated.

  1. Section 4 shows tau is *identical* under published-tile and eligible-grid
     normalisation. Once city fixed effects are in, the two denominators differ
     only by a per-city constant. So the null is not a normalisation artefact
     and cannot be fixed by renormalising - the sample is the issue.
  2. Section 1 shows that in *population* terms Meta is close to unbiased in
     every deprivation decile (R ~ 1.0 throughout). The unpublished tiles hold
     very few people, so they barely move a population-weighted statistic.

What survives is sharper than the original hypothesis: Meta's baseline is
population-representative but geographically censored, and the censoring is
deprivation-selective. Any estimand weighted by population will find nothing;
the bias is visible only in the space of places.

The central object is the representation ratio for a group g of tiles:

    R_g = (Meta's share of the city total in g) / (WorldPop's share in g)

R_g = 1 means Meta represents group g exactly in proportion to WorldPop. The
ratio is computed two ways: normalising over published tiles only (what the
pipeline does) and over the full eligible grid.

Outputs
  outputs/analysis/A3_representation_ratio_by_decile.csv
  outputs/analysis/A3_invisible_population.csv
  outputs/analysis/A3_margin_decomposition.csv
  outputs/analysis/A3_intensive_margin_null.csv

Usage:
  python analysis/03_two_margin_decomposition.py
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


def load():
    d = pd.read_parquet(PANEL)
    d = d[d.in_eligible].copy()
    d["meta_pub"] = d["meta_obs"].fillna(0.0).clip(lower=0)   # 0 where unpublished
    return d.reset_index(drop=True)


def representation_ratio(d: pd.DataFrame) -> pd.DataFrame:
    """R by within-city GRDI decile, under both normalisations.

    Shares are formed within each city and then pooled by summing numerator and
    denominator across cities, so a large city cannot dominate through its total
    population — every city contributes exactly 1.0 of Meta share and 1.0 of
    WorldPop share.
    """
    rows = []
    for city, g in d.groupby("city"):
        pub = g.published == 1
        meta_tot = g.loc[pub, "meta_pub"].sum()
        wp_tot_elig = g["worldpop_count"].sum()
        wp_tot_pub = g.loc[pub, "worldpop_count"].sum()
        for dec, gg in g.groupby("grdi_decile"):
            rows.append({
                "city": city, "grdi_decile": dec,
                "meta_sh": gg.loc[gg.published == 1, "meta_pub"].sum() / meta_tot,
                # denominator over published tiles only = the pipeline's view
                "wp_sh_pub": gg.loc[gg.published == 1, "worldpop_count"].sum() / wp_tot_pub,
                # denominator over the whole eligible grid = the correct view
                "wp_sh_elig": gg["worldpop_count"].sum() / wp_tot_elig,
                "n_tiles": len(gg), "n_pub": int(gg.published.sum()),
            })
    r = pd.DataFrame(rows)
    agg = r.groupby("grdi_decile").agg(
        n_cities=("city", "nunique"), n_tiles=("n_tiles", "sum"), n_pub=("n_pub", "sum"),
        meta_sh=("meta_sh", "sum"), wp_sh_pub=("wp_sh_pub", "sum"),
        wp_sh_elig=("wp_sh_elig", "sum"),
    ).reset_index()
    agg["R_published_norm"] = agg.meta_sh / agg.wp_sh_pub
    agg["R_eligible_norm"] = agg.meta_sh / agg.wp_sh_elig
    agg["pub_rate"] = agg.n_pub / agg.n_tiles

    print("  decile  R (pipeline norm)  R (eligible norm)  tiles published")
    for _, x in agg.iterrows():
        print(f"    {int(x.grdi_decile):>2}        {x.R_published_norm:.3f}"
              f"              {x.R_eligible_norm:.3f}           {x.pub_rate:.1%}")
    lo = agg[agg.grdi_decile <= 2]
    hi = agg[agg.grdi_decile >= 9]
    print(f"\n  least-deprived (D1-D2) vs most-deprived (D9-D10):")
    print(f"    pipeline normalisation : {lo.R_published_norm.mean():.3f} vs "
          f"{hi.R_published_norm.mean():.3f}  (ratio {lo.R_published_norm.mean()/hi.R_published_norm.mean():.2f}x)")
    print(f"    eligible normalisation : {lo.R_eligible_norm.mean():.3f} vs "
          f"{hi.R_eligible_norm.mean():.3f}  (ratio {lo.R_eligible_norm.mean()/hi.R_eligible_norm.mean():.2f}x)")
    return agg


def invisible_population(d: pd.DataFrame) -> pd.DataFrame:
    """Who lives in the tiles Meta never publishes."""
    rows = []
    for city, g in d.groupby("city"):
        pub, unpub = g[g.published == 1], g[g.published == 0]
        wt = lambda x: (np.average(x.poverty_mean, weights=x.worldpop_count)
                        if len(x) and x.worldpop_count.sum() > 0 else np.nan)
        rows.append({
            "country": g.country.iloc[0], "city": city,
            "n_eligible": len(g), "C_c": g.published.mean(),
            "wp_total": g.worldpop_count.sum(),
            "wp_invisible": unpub.worldpop_count.sum(),
            "pct_wp_invisible": 100 * unpub.worldpop_count.sum() / g.worldpop_count.sum(),
            "grdi_visible": wt(pub), "grdi_invisible": wt(unpub),
        })
    out = pd.DataFrame(rows)
    out["grdi_gap"] = out.grdi_invisible - out.grdi_visible
    out = out.sort_values("pct_wp_invisible", ascending=False)

    pub, unpub = d[d.published == 1], d[d.published == 0]
    tot = {
        "country": "—", "city": f"ALL {d.city.nunique()} CITIES",
        "n_eligible": len(d), "C_c": d.published.mean(),
        "wp_total": d.worldpop_count.sum(), "wp_invisible": unpub.worldpop_count.sum(),
        "pct_wp_invisible": 100 * unpub.worldpop_count.sum() / d.worldpop_count.sum(),
        "grdi_visible": np.average(pub.poverty_mean, weights=pub.worldpop_count),
        "grdi_invisible": np.average(unpub.poverty_mean, weights=unpub.worldpop_count),
    }
    tot["grdi_gap"] = tot["grdi_invisible"] - tot["grdi_visible"]
    print(out.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n  POOLED: {tot['wp_invisible']:,.0f} of {tot['wp_total']:,.0f} people "
          f"({tot['pct_wp_invisible']:.1f}%) live in tiles Meta never publishes.")
    print(f"  Population-weighted GRDI  visible {tot['grdi_visible']:.1f}  vs  "
          f"invisible {tot['grdi_invisible']:.1f}  (gap {tot['grdi_gap']:+.1f} points)")
    return pd.concat([out, pd.DataFrame([tot])], ignore_index=True)


def margin_decomposition(d: pd.DataFrame) -> pd.DataFrame:
    """How much of the deprivation gradient sits on each margin?

    Meta's share of a tile can be written as the product of two pieces:
        share_i = published_i  x  (meta_i / total | published)
    Regressing log of each piece on deprivation splits the total gradient into a
    coverage component and a within-coverage intensity component. The intensive
    piece is estimated on published tiles only, which is exactly the sample the
    original pipeline used, so the comparison is like-for-like.
    """
    d = d.copy()
    rows = []
    for city, g in d.groupby("city"):
        g = g.copy()
        g["wp_sh_elig"] = g.worldpop_count / g.worldpop_count.sum()
        pub = g.published == 1
        g.loc[pub, "meta_sh"] = g.loc[pub, "meta_pub"] / g.loc[pub, "meta_pub"].sum()
        rows.append(g)
    d = pd.concat(rows, ignore_index=True)

    fe = pd.get_dummies(d.city, prefix="c", drop_first=True).astype(float)

    # Extensive: linear probability of publication (interpretable as a share of
    # the gradient, unlike a logit coefficient).
    X = sm.add_constant(pd.concat([d[["z_poverty_mean"]].reset_index(drop=True),
                                   fe.reset_index(drop=True)], axis=1)).astype(float)
    ext = sm.OLS(d.published.to_numpy(), X.to_numpy()).fit(
        cov_type="cluster", cov_kwds={"groups": d.blk10.to_numpy()})

    # Intensive: the pipeline's own estimand, on the pipeline's own sample.
    p = d[d.published == 1].dropna(subset=["meta_sh", "wp_sh_elig"]).reset_index(drop=True)
    p["log_ratio"] = np.log(p.meta_sh / p.wp_sh_elig)
    fep = pd.get_dummies(p.city, prefix="c", drop_first=True).astype(float)
    Xp = sm.add_constant(pd.concat([p[["z_poverty_mean"]], fep], axis=1)).astype(float)
    inten = sm.OLS(p.log_ratio.to_numpy(), Xp.to_numpy()).fit(
        cov_type="cluster", cov_kwds={"groups": p.blk10.to_numpy()})

    out = pd.DataFrame([
        {"margin": "Extensive: P(published) ~ GRDI  [pp per SD]",
         "sample": "all eligible tiles", "n": len(d),
         "coef": 100 * ext.params[1], "se": 100 * ext.bse[1], "p": ext.pvalues[1]},
        {"margin": "Intensive: log(meta_sh/wp_sh) ~ GRDI  [log points per SD]",
         "sample": "published tiles only", "n": len(p),
         "coef": inten.params[1], "se": inten.bse[1], "p": inten.pvalues[1]},
    ])
    for _, r in out.iterrows():
        print(f"  {r.margin:<58} {r.coef:+7.3f} (se {r.se:.3f})  p={r.p:.2e}  n={r.n:,}")
    return out


def intensive_null(d: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the original null and show renormalising does not rescue it.

    Same regression, same published-tile sample, two denominators. The estimates
    come out identical: with city fixed effects the two normalisations differ by
    a per-city constant that the fixed effect absorbs. This is the negative
    control for the obvious objection "you just normalised it away" - the null on
    published tiles is a genuine feature of that sample, not an artefact of the
    denominator. It is the sample itself that has been filtered.
    """
    rows = []
    frames = []
    for city, g in d.groupby("city"):
        g = g.copy()
        pub = g.published == 1
        g["wp_sh_elig"] = g.worldpop_count / g.worldpop_count.sum()
        g.loc[pub, "wp_sh_pub"] = (g.loc[pub, "worldpop_count"]
                                   / g.loc[pub, "worldpop_count"].sum())
        g.loc[pub, "meta_sh"] = g.loc[pub, "meta_pub"] / g.loc[pub, "meta_pub"].sum()
        frames.append(g)
    d = pd.concat(frames, ignore_index=True)
    p = d[d.published == 1].copy()
    p["resid_pipeline"] = np.log(p.meta_sh / p.wp_sh_pub)
    p["resid_eligible"] = np.log(p.meta_sh / p.wp_sh_elig)
    p = p.replace([np.inf, -np.inf], np.nan)

    fe = pd.get_dummies(p.city, prefix="c", drop_first=True).astype(float)
    for label, y in [("pipeline normalisation (published tiles)", "resid_pipeline"),
                     ("eligible-grid normalisation", "resid_eligible")]:
        s = p.dropna(subset=[y, "z_poverty_mean"]).reset_index(drop=True)
        f = pd.get_dummies(s.city, prefix="c", drop_first=True).astype(float)
        X = sm.add_constant(pd.concat([s[["z_poverty_mean"]], f], axis=1)).astype(float)
        r = sm.OLS(s[y].to_numpy(), X.to_numpy()).fit(
            cov_type="cluster", cov_kwds={"groups": s.blk10.to_numpy()})
        rows.append({"outcome": label, "n": len(s), "coef": r.params[1],
                     "se": r.bse[1], "p": r.pvalues[1]})
        print(f"  {label:<45} tau={r.params[1]:+.4f} (se {r.bse[1]:.4f}) p={r.pvalues[1]:.3f}")

    # Both normalisations also leave the population-weighted mean residual close
    # to zero: among published tiles Meta tracks WorldPop well in aggregate.
    w = p.dropna(subset=["resid_pipeline"])
    print(f"\n  WorldPop-weighted mean residual, pipeline normalisation: "
          f"{np.average(w.resid_pipeline, weights=w.worldpop_count):+.4f}")
    w2 = p.dropna(subset=["resid_eligible"])
    print(f"  WorldPop-weighted mean residual, eligible normalisation: "
          f"{np.average(w2.resid_eligible, weights=w2.worldpop_count):+.4f}")
    return pd.DataFrame(rows)


def intensive_by_city(d: pd.DataFrame) -> pd.DataFrame:
    """The original estimand, city by city.

    The pooled intensive-margin tau is a flat zero, but "flat zero" could hide
    strong effects of opposite sign. This re-estimates the pipeline's original
    regression per city so the claim "mixed and centred on nothing" is checkable
    from this repository rather than from the earlier pipeline's deleted tables.
    """
    rows = []
    for city, g in d.groupby("city"):
        pub = g[g.published == 1].copy()
        if len(pub) < 30:
            rows.append({"city": city, "n": len(pub), "tau": np.nan,
                         "se": np.nan, "p": np.nan, "note": "too few published tiles"})
            continue
        pub["meta_sh"] = pub.meta_pub / pub.meta_pub.sum()
        pub["wp_sh"] = pub.worldpop_count / pub.worldpop_count.sum()
        pub = pub[(pub.meta_sh > 0) & (pub.wp_sh > 0)]
        y = np.log(pub.meta_sh / pub.wp_sh)
        X = sm.add_constant(pub[["z_poverty_mean"]].astype(float))
        r = sm.OLS(y.to_numpy(), X.to_numpy()).fit(
            cov_type="cluster", cov_kwds={"groups": pub.blk10.to_numpy()})
        rows.append({"city": city, "n": len(pub), "tau": r.params[1],
                     "se": r.bse[1], "p": r.pvalues[1], "note": ""})
    out = pd.DataFrame(rows)
    e = out.dropna(subset=["tau"])
    neg = int(((e.tau < 0) & (e.p < 0.05)).sum())
    pos = int(((e.tau > 0) & (e.p < 0.05)).sum())
    null = int((e.p >= 0.05).sum())
    print(f"  {len(e)} cities estimable: {neg} significantly negative, "
          f"{pos} significantly positive, {null} null")
    print(f"  tau ranges {e.tau.min():+.3f} to {e.tau.max():+.3f}; "
          f"median {e.tau.median():+.3f}")
    return out.sort_values("tau")


def ppml_alternative(d: pd.DataFrame) -> pd.DataFrame:
    """A failed alternative, kept because its failure is informative.

    PPML looks like the right tool: Poisson with log(WorldPop) as offset handles
    the zeros of unpublished tiles natively, so it should let the full eligible
    grid speak without any imputation. It returns a null.

    The reason is that a Poisson likelihood is count-weighted. Suppressed tiles
    have almost no WorldPop population, so they contribute almost nothing to the
    likelihood, and the estimate is driven by the large published tiles - the
    same failure mode as share-based measures. This is the clearest single
    demonstration that *any* population-weighted estimand is null here, and hence
    why the binary publication outcome is the right one.
    """
    rows = []
    for label, mask in [("full eligible grid (suppressed = 0)", d.index == d.index),
                        ("published tiles only", d.published == 1)]:
        s = d[mask].copy()
        s["meta"] = s.meta_pub.fillna(0.0).clip(lower=0)
        fe = pd.get_dummies(s.city, prefix="c", drop_first=True).astype(float)
        X = sm.add_constant(pd.concat([s[["z_poverty_mean"]].reset_index(drop=True),
                                       fe.reset_index(drop=True)], axis=1)).astype(float)
        r = sm.GLM(s.meta.to_numpy(), X.to_numpy(), family=sm.families.Poisson(),
                   offset=np.log(s.worldpop_count.to_numpy())).fit(
            cov_type="cluster", cov_kwds={"groups": s.blk10.to_numpy()})
        rows.append({"sample": label, "n": len(s), "tau": r.params[1],
                     "se": r.bse[1], "p": r.pvalues[1],
                     "ratio_per_SD": float(np.exp(r.params[1]))})
        print(f"  {label:<38} tau={r.params[1]:+.4f} (se {r.bse[1]:.4f}) "
              f"p={r.pvalues[1]:.3f}  n={len(s)}")
    print("  -> null on both. Poisson is count-weighted, so the suppressed tiles "
          "(tiny population)\n     barely enter the likelihood. Same failure mode as shares.")
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    print(f"Eligible tiles {len(d):,} in {d.city.nunique()} cities\n")

    print("=== 1. Representation ratio by within-city GRDI decile ===")
    print("    R = Meta's share of the city / WorldPop's share. R=1 is unbiased.\n")
    rr = representation_ratio(d)
    rr.to_csv(OUT / "A3_representation_ratio_by_decile.csv", index=False)

    print("\n\n=== 2. The invisible population ===\n")
    inv = invisible_population(d)
    inv.to_csv(OUT / "A3_invisible_population.csv", index=False)

    print("\n\n=== 3. Where the deprivation gradient lives ===\n")
    dec = margin_decomposition(d)
    dec.to_csv(OUT / "A3_margin_decomposition.csv", index=False)

    print("\n\n=== 4. The original null, and its dependence on normalisation ===\n")
    nul = intensive_null(d)
    nul.to_csv(OUT / "A3_intensive_margin_null.csv", index=False)

    print("\n\n=== 4b. The original estimand, city by city ===\n")
    ibc = intensive_by_city(d)
    ibc.to_csv(OUT / "A3_intensive_margin_by_city.csv", index=False)

    print("\n\n=== 5. A failed alternative: PPML on the full grid ===\n")
    pp = ppml_alternative(d)
    pp.to_csv(OUT / "A3_ppml_failed_alternative.csv", index=False)

    print(f"\nWrote 5 tables to {OUT}")


if __name__ == "__main__":
    main()
