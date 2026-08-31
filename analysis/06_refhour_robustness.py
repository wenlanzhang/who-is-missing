#!/usr/bin/env python3
"""
A6 — Is the coverage gradient an artefact of which hour we snapshot?

Meta publishes a baseline per 8-hour window. The study sample uses one hour per
country, chosen to sit near evening locally. Publication is hour-specific: a tile
busy at 8pm may fall below the privacy threshold at 8am. If the deprivation
gradient only appears at the chosen hour, it is a story about commuting, not
about representation.

Five countries have a second baseline hour built (IDN, KEN, LKA, PHL, ZAF =
11 of 18 cities). COL, ECU and MEX only have h00, which is already their
designated hour, so they contribute no within-city contrast and are reported but
not compared.

This script also documents why Meta's RWI is *not* used as an alternative
deprivation measure. The reason is structural, not stylistic: RWI is itself
estimated from Facebook data and is therefore missing in precisely the tiles
whose publication we are trying to model. Regressing publication on a covariate
that is only observed when the outcome is 1 is not a robustness check.

Outputs
  outputs/analysis/A6_refhour_extensive_margin.csv
  outputs/analysis/A6_refhour_coverage.csv
  outputs/analysis/A6_rwi_structural_selection.csv

Usage:
  python analysis/06_refhour_robustness.py
"""

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import region_config  # noqa: E402

PANEL = ROOT / "analysis" / "panel" / "tile_panel.parquet"
BASELINES = ROOT / "data" / "baselines"
OUT = ROOT / "outputs" / "analysis"
# Resolved against the configured data_root (config/regions.json, or the
# RESIDENTIAL_DATA_ROOT environment variable) so the path is not machine-specific.
RWI_SUBDIR = "Meta_Event/RWI/relative-wealth-index-april-2021"


def rwi_dir() -> Path:
    root = region_config.load_regions().get("data_root")
    return Path(root) / RWI_SUBDIR if root else Path(RWI_SUBDIR)

# Hour used in the main analysis, from README "Study sample".
DESIGNATED = {"PHL": 8, "KEN": 16, "MEX": 0, "IDN": 8,
              "LKA": 8, "COL": 0, "ECU": 0, "ZAF": 16}


def available_hours() -> dict[str, list[int]]:
    hours = {}
    for p in sorted(BASELINES.glob("*/fb_baseline_median_h*.gpkg")):
        country = p.parts[-2]
        hours.setdefault(country, []).append(int(p.stem.split("_h")[-1]))
    return {k: sorted(v) for k, v in hours.items()}


def published_at_hour(country: str, hour: int) -> tuple[set, set]:
    """(quadkeys with a published value, quadkeys present in the AOI at all).

    A quadkey absent from the file is outside the published event AOI; a quadkey
    present with NaN is inside the AOI but suppressed. Only the second is a
    coverage decision, so the two are kept apart.
    """
    p = BASELINES / country / f"fb_baseline_median_h{hour:02d}.gpkg"
    g = gpd.read_file(p, columns=["quadkey", "fb_baseline_median"])
    qk = g["quadkey"].astype(str)
    in_aoi = set(qk)
    pub = set(qk[g["fb_baseline_median"].notna()])
    return pub, in_aoi


def fit_extensive(d: pd.DataFrame, cols=("z_poverty_mean", "z_log_wp")) -> dict:
    """Same M2 spec as A2: logit, city FE, SE clustered on zoom-10 blocks."""
    d = d.dropna(subset=list(cols))
    varies = d.groupby("city")["published"].transform("nunique") > 1
    d = d[varies]
    if d.empty or d["city"].nunique() == 0:
        return {"OR": np.nan, "OR_lo": np.nan, "OR_hi": np.nan, "p": np.nan,
                "n": 0, "n_cities": 0}
    fe = pd.get_dummies(d["city"], drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([d[list(cols)].astype(float).reset_index(drop=True),
                                   fe.reset_index(drop=True)], axis=1))
    res = sm.Logit(d["published"].to_numpy(), X.to_numpy()).fit(
        disp=0, method="bfgs", maxiter=500,
        cov_type="cluster", cov_kwds={"groups": d["blk10"].to_numpy()})
    i = X.columns.tolist().index("z_poverty_mean")
    b, se = res.params[i], res.bse[i]
    return {"OR": np.exp(b), "OR_lo": np.exp(b - 1.96 * se),
            "OR_hi": np.exp(b + 1.96 * se), "p": res.pvalues[i],
            "n": len(d), "n_cities": d["city"].nunique()}


def refhour_analysis(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hours = available_hours()
    cov_rows, fit_rows = [], []

    for country, hrs in sorted(hours.items()):
        sub = panel[(panel.country == country) & panel.in_eligible]
        if sub.empty:
            continue
        for h in hrs:
            pub, in_aoi = published_at_hour(country, h)
            s = sub.copy()
            s["published"] = s.quadkey.isin(pub).astype(int)
            s["in_aoi"] = s.quadkey.isin(in_aoi)
            for city, g in s.groupby("city"):
                cov_rows.append({
                    "country": country, "city": city, "hour": h,
                    "designated": h == DESIGNATED.get(country),
                    "n_eligible": len(g), "n_in_aoi": int(g.in_aoi.sum()),
                    "C_c": g.published.mean(),
                    "cov_least_deprived": g.loc[g.grdi_decile <= 2, "published"].mean(),
                    "cov_most_deprived": g.loc[g.grdi_decile >= 9, "published"].mean(),
                })
            r = fit_extensive(s)
            r.update(country=country, hour=h, designated=h == DESIGNATED.get(country),
                     C_c=s.published.mean())
            fit_rows.append(r)

    cov = pd.DataFrame(cov_rows)
    fits = pd.DataFrame(fit_rows)[
        ["country", "hour", "designated", "n", "n_cities", "C_c",
         "OR", "OR_lo", "OR_hi", "p"]]
    return cov, fits


def pooled_by_hour(panel: pd.DataFrame) -> pd.DataFrame:
    """Pool the five two-hour countries and estimate once per hour set.

    'designated' uses each country's chosen near-evening hour. For all five of
    these countries the only other built hour is h00, so 'alternate' and 'h00'
    would be the same regression; only the two distinct sets are reported.
    """
    hours = available_hours()
    two_hour = [c for c, h in hours.items() if len(h) > 1]
    rows = []
    for label in ("designated", "alternate (h00)"):
        frames = []
        for country in two_hour:
            hrs = hours[country]
            des = DESIGNATED.get(country)
            h = des if label == "designated" else [x for x in hrs if x != des][0]
            if h not in hrs:
                continue
            pub, _ = published_at_hour(country, h)
            s = panel[(panel.country == country) & panel.in_eligible].copy()
            s["published"] = s.quadkey.isin(pub).astype(int)
            s["hour_used"] = h
            frames.append(s)
        d = pd.concat(frames, ignore_index=True)
        r = fit_extensive(d)
        r.update(hour_set=label, countries=len(two_hour), C_c=d.published.mean())
        rows.append(r)
        print(f"  {label:<12} C_c={r['C_c']:.3f}  OR={r['OR']:.3f} "
              f"[{r['OR_lo']:.3f},{r['OR_hi']:.3f}]  p={r['p']:.2e}  "
              f"n={r['n']} in {r['n_cities']} cities")
    return pd.DataFrame(rows)[["hour_set", "countries", "n", "n_cities", "C_c",
                               "OR", "OR_lo", "OR_hi", "p"]]


def rwi_selection(panel: pd.DataFrame) -> pd.DataFrame:
    """Why RWI cannot serve as an alternative deprivation measure here."""
    rows = []
    for country, g in panel[panel.in_eligible].groupby("country"):
        f = rwi_dir() / f"{country}_relative_wealth_index.csv"
        if not f.exists():
            print(f"  {country}: no RWI file")
            continue
        r = pd.read_csv(f, usecols=["quadkey", "rwi"])
        have = set(r.quadkey.astype(str).str.zfill(14))
        g = g.assign(has_rwi=g.quadkey.astype(str).isin(have))
        rows.append({
            "country": country, "n_eligible": len(g),
            "rwi_cov_published": g.loc[g.published == 1, "has_rwi"].mean(),
            "rwi_cov_unpublished": (g.loc[g.published == 0, "has_rwi"].mean()
                                    if (g.published == 0).any() else np.nan),
            "n_unpublished": int((g.published == 0).sum()),
        })
    if not rows:
        # The RWI CSVs live outside the repository. Without them this section is
        # simply skipped rather than failing the whole script.
        print("  No RWI files found. Set RESIDENTIAL_DATA_ROOT (or data_root in\n"
              "  config/regions.json) to the directory holding "
              f"{RWI_SUBDIR} to run this check.")
        return pd.DataFrame(columns=["country", "n_eligible", "rwi_cov_published",
                                     "rwi_cov_unpublished", "n_unpublished", "gap_pp"])
    out = pd.DataFrame(rows)
    out["gap_pp"] = 100 * (out.rwi_cov_published - out.rwi_cov_unpublished)
    return out.sort_values("gap_pp", ascending=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PANEL)
    panel["quadkey"] = panel["quadkey"].astype(str)

    print("Baseline hours available:")
    for c, h in sorted(available_hours().items()):
        star = DESIGNATED.get(c)
        print(f"  {c}: {h}   (designated h{star:02d})")

    print("\n=== Per-country extensive margin by baseline hour ===")
    cov, fits = refhour_analysis(panel)
    for _, r in fits.iterrows():
        mark = "*" if r.designated else " "
        or_s = "n/a" if not np.isfinite(r.OR) else f"{r.OR:.3f} [{r.OR_lo:.3f},{r.OR_hi:.3f}]"
        print(f" {mark}{r.country} h{int(r.hour):02d}  C_c={r.C_c:.3f}  "
              f"n={int(r.n):>4}  OR={or_s}")
    cov.to_csv(OUT / "A6_refhour_coverage.csv", index=False)
    fits.to_csv(OUT / "A6_refhour_extensive_margin.csv", index=False)

    print("\n=== Pooled across the five two-hour countries ===")
    pooled = pooled_by_hour(panel)
    pooled.to_csv(OUT / "A6_refhour_pooled.csv", index=False)

    print("\n=== Why RWI is not used: it is missing where the outcome is 0 ===")
    rwi = rwi_selection(panel)
    rwi.to_csv(OUT / "A6_rwi_structural_selection.csv", index=False)
    if len(rwi):
        print(rwi.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\n  RWI covers {rwi.rwi_cov_published.mean():.1%} of published tiles but "
              f"only {rwi.rwi_cov_unpublished.mean():.1%} of unpublished ones "
              f"(unweighted mean).")

    print(f"\nWrote 4 tables to {OUT}")


if __name__ == "__main__":
    main()
