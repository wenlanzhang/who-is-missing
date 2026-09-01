#!/usr/bin/env python3
"""
A8 — Who and what is missing: the unreported burden in people, land and places.

A2 shows Meta's decision to publish a tile is deprivation-graded. A3 shows the
gap is invisible to any population-weighted statistic. This script answers the
question an operational reader asks first — *how many people are not in this
map?* — and shows why the pooled answer to that question is misleading.

THE TRAP THIS SCRIPT EXISTS TO AVOID
------------------------------------
Pooled, only ~0.5% of the population lives in a tile Meta never publishes. Quoted
on its own that number says Meta is fine, and it is the reason the original
population-weighted design found nothing (A3). It is true but it is the wrong
conditioning.

The informative quantity is the *conditional* one: of the people who live in the
most deprived tenth of a city, what share is unreported? That runs from 0% in the
least deprived decile to about 31% in the most. The unconditional 0.5% is small
because deprived tiles are sparse, not because the deprived are covered.

So every table here reports the same three burdens side by side —

    PLACES   tiles with no Meta value
    LAND     km^2 those tiles cover
    PEOPLE   WorldPop residents in them

— each as a count, as a share, and as a ratio against the sample-wide share, and
always both pooled and *within* deprivation decile. Reporting people without land
overstates coverage; reporting land without people overstates the harm. The
density column is included for the same reason: it is the honest explanation for
why the population share is small, and it belongs next to the finding rather than
in a footnote.

Outputs
  outputs/analysis/A8_unreported_headline.csv    pooled totals, the three burdens
  outputs/analysis/A8_unreported_by_decile.csv   the conditional gradient
  outputs/analysis/A8_unreported_by_city.csv     per city, for the appendix

Usage:
  python analysis/08_unreported_burden.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "analysis" / "panel" / "tile_panel.parquet"
OUT = ROOT / "outputs" / "analysis"

# "Most deprived 20% of tiles" throughout, matching A4's blind-spot tables.
TOP_DECILES = 9


def load() -> pd.DataFrame:
    if not PANEL.exists():
        raise SystemExit(f"Missing {PANEL}. Run: python analysis/01_build_panel.py")
    d = pd.read_parquet(PANEL)
    return d[d.in_eligible].reset_index(drop=True)


def _burden(g: pd.DataFrame) -> dict:
    """The three burdens for one group of tiles."""
    u = g.published == 0
    tiles, area, pop = len(g), g.area_km2.sum(), g.worldpop_count.sum()
    u_tiles = int(u.sum())
    u_area = float(g.loc[u, "area_km2"].sum())
    u_pop = float(g.loc[u, "worldpop_count"].sum())
    return {
        "n_tiles": tiles, "n_tiles_unreported": u_tiles,
        "pct_tiles_unreported": 100 * u_tiles / tiles if tiles else np.nan,
        "area_km2": float(area), "area_km2_unreported": u_area,
        "pct_area_unreported": 100 * u_area / area if area else np.nan,
        "pop_total": float(pop), "pop_unreported": u_pop,
        "pct_pop_unreported": 100 * u_pop / pop if pop else np.nan,
        # Why the population share is so much smaller than the land share.
        "density_reported": (g.loc[~u, "worldpop_count"].sum()
                             / g.loc[~u, "area_km2"].sum()) if (~u).any() else np.nan,
        "density_unreported": (u_pop / u_area) if u_area > 0 else np.nan,
    }


def headline(d: pd.DataFrame) -> pd.DataFrame:
    """Pooled totals, and the same restricted to the most deprived 20% of tiles."""
    rows = []
    for label, g in [("All eligible tiles", d),
                     ("Most deprived 20% of tiles (D9-D10)", d[d.grdi_decile >= TOP_DECILES]),
                     ("Least deprived 20% of tiles (D1-D2)", d[d.grdi_decile <= 2])]:
        r = _burden(g)
        r["scope"] = label
        rows.append(r)
    out = pd.DataFrame(rows)

    a = out.iloc[0]
    print("  Pooled across 18 cities, on the eligible grid:\n")
    print(f"    PLACES  {a.n_tiles_unreported:>10,.0f} of {a.n_tiles:>10,.0f} tiles "
          f"= {a.pct_tiles_unreported:5.1f}%  have no Meta value")
    print(f"    LAND    {a.area_km2_unreported:>10,.0f} of {a.area_km2:>10,.0f} km2   "
          f"= {a.pct_area_unreported:5.1f}%")
    print(f"    PEOPLE  {a.pop_unreported:>10,.0f} of {a.pop_total:>10,.0f}       "
          f"= {a.pct_pop_unreported:5.2f}%")
    print(f"\n    Density: {a.density_reported:,.0f} people/km2 where Meta reports, "
          f"{a.density_unreported:,.0f} where it does not.")
    print("    That ratio is the whole reason the population share is small: the")
    print("    missing land is sparse, not empty. Quote PEOPLE and LAND together.")

    dep = out.iloc[1]
    print(f"\n  Restricted to the most deprived 20% of each city's tiles:\n")
    print(f"    PLACES  {dep.pct_tiles_unreported:5.1f}%   LAND {dep.pct_area_unreported:5.1f}%"
          f"   PEOPLE {dep.pct_pop_unreported:5.1f}%")
    print(f"    {dep.pop_unreported:,.0f} of the {dep.pop_total:,.0f} people living in the most")
    print(f"    deprived 20% of these cities' tiles are absent from Meta's baseline.")
    return out[["scope", "n_tiles", "n_tiles_unreported", "pct_tiles_unreported",
                "area_km2", "area_km2_unreported", "pct_area_unreported",
                "pop_total", "pop_unreported", "pct_pop_unreported",
                "density_reported", "density_unreported"]]


def by_decile(d: pd.DataFrame) -> pd.DataFrame:
    """The conditional gradient — the point of this script.

    Deciles are formed within city, so pooling them compares like with like: each
    decile holds roughly a tenth of every city's tiles, and no city's deprivation
    level shifts which decile its tiles land in.
    """
    rows = []
    for dec, g in d.groupby("grdi_decile"):
        r = _burden(g)
        r["grdi_decile"] = int(dec)
        r["median_grdi"] = float(g.poverty_mean.median())
        rows.append(r)
    out = pd.DataFrame(rows)

    # Concentration ratio: how over-represented is this decile among the
    # unreported, relative to its share of the population? 1.0 = proportional.
    tot_unrep = out.pop_unreported.sum()
    tot_pop = out.pop_total.sum()
    out["share_of_unreported_pop"] = 100 * out.pop_unreported / tot_unrep
    out["share_of_all_pop"] = 100 * out.pop_total / tot_pop
    out["concentration_ratio"] = out.share_of_unreported_pop / out.share_of_all_pop

    print("\n  Within each within-city deprivation decile, the share that is unreported:\n")
    print("    decile   places%   land%   people%    people unreported        conc.")
    for _, r in out.iterrows():
        print(f"      {int(r.grdi_decile):>2}     {r.pct_tiles_unreported:6.1f}  "
              f"{r.pct_area_unreported:6.1f}  {r.pct_pop_unreported:7.1f}   "
              f"{r.pop_unreported:>14,.0f}   {r.concentration_ratio:8.1f}x")

    lo, hi = out.iloc[0], out.iloc[-1]
    print(f"\n  The people share runs {lo.pct_pop_unreported:.1f}% -> "
          f"{hi.pct_pop_unreported:.1f}% across deciles, while the pooled figure is "
          f"{100*d.loc[d.published==0,'worldpop_count'].sum()/d.worldpop_count.sum():.1f}%.")
    print("  The pooled number is an average over a gradient this steep, which is why")
    print("  it should never be quoted without the gradient beside it.")

    top = out[out.grdi_decile >= TOP_DECILES]
    print(f"\n  {top.share_of_unreported_pop.sum():.0f}% of all unreported people live in the "
          f"most deprived 20% of tiles,\n  which hold only {top.share_of_all_pop.sum():.1f}% of the "
          f"total population — a {top.share_of_unreported_pop.sum()/top.share_of_all_pop.sum():.0f}x "
          f"concentration.")
    return out[["grdi_decile", "median_grdi", "n_tiles", "n_tiles_unreported",
                "pct_tiles_unreported", "area_km2", "area_km2_unreported",
                "pct_area_unreported", "pop_total", "pop_unreported", "pct_pop_unreported",
                "share_of_unreported_pop", "share_of_all_pop", "concentration_ratio",
                "density_reported", "density_unreported"]]


def by_city(d: pd.DataFrame) -> pd.DataFrame:
    """Per city, overall and restricted to its own most deprived 20% of tiles."""
    rows = []
    for city, g in d.groupby("city"):
        r = _burden(g)
        top = g[g.grdi_decile >= TOP_DECILES]
        t = _burden(top)
        r.update(country=g.country.iloc[0], city=city,
                 pct_tiles_unreported_top=t["pct_tiles_unreported"],
                 pct_area_unreported_top=t["pct_area_unreported"],
                 pct_pop_unreported_top=t["pct_pop_unreported"],
                 pop_unreported_top=t["pop_unreported"])
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("pct_pop_unreported_top", ascending=False)

    # places% and land% are near-identical within a city because zoom-14 tiles at
    # one latitude are near-equal area. They separate only when pooling cities at
    # different latitudes, which is why both are kept in the pooled table.
    print("\n  Per city — the last column is the number that matters:\n")
    print(f"    {'city':<20} {'places%':>8} {'land%':>7} {'people%':>8}   "
          f"{'people% in its most deprived 20%':>32}")
    for _, r in out.iterrows():
        print(f"    {r.city:<20} {r.pct_tiles_unreported:8.1f} {r.pct_area_unreported:7.1f} "
              f"{r.pct_pop_unreported:8.2f}   {r.pct_pop_unreported_top:28.1f}")
    return out[["country", "city", "n_tiles", "n_tiles_unreported", "pct_tiles_unreported",
                "area_km2_unreported", "pct_area_unreported",
                "pop_total", "pop_unreported", "pct_pop_unreported",
                "pct_tiles_unreported_top", "pct_area_unreported_top",
                "pop_unreported_top", "pct_pop_unreported_top"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    print(f"Eligible tiles {len(d):,} in {d.city.nunique()} cities\n")

    print("=== 1. The unreported burden: places, land, people ===\n")
    h = headline(d)
    h.to_csv(OUT / "A8_unreported_headline.csv", index=False)

    print("\n\n=== 2. The same burden, conditioned on deprivation ===")
    dec = by_decile(d)
    dec.to_csv(OUT / "A8_unreported_by_decile.csv", index=False)

    print("\n\n=== 3. By city ===")
    bc = by_city(d)
    bc.to_csv(OUT / "A8_unreported_by_city.csv", index=False)

    print(f"\nWrote 3 tables to {OUT}")


if __name__ == "__main__":
    main()
