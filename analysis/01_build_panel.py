#!/usr/bin/env python3
"""
A1 — Build the pooled tile panel for the selection-margin analysis.

The city pipeline (01 → 02 → 03x) starts from *published* Meta tiles, so a tile
Meta never published cannot appear in any of its tables. Step 01b already builds
the independent city grid that does contain those tiles. This script pools those
grids into a single tile-level panel so every downstream model can be estimated
on the full eligible grid rather than on the published subset.

One row = one zoom-14 quadkey tile inside one city's clip boundary.

Key columns
  published        1 if Meta published a baseline value for the tile
  worldpop_count   WorldPop persons in the tile (the eligible-grid denominator)
  poverty_mean     GRDI (higher = more deprived)
  smod_class       GHSL settlement class (urban centre / cluster / rural)
  z_*              within-city standardised versions (city-demeaned)
  wp_share_pub     WorldPop share renormalised over published tiles only
  wp_share_elig    WorldPop share renormalised over the full eligible grid

The two share columns are what make the "null result" reproducible: the pipeline
uses wp_share_pub, which redistributes the population of unpublished tiles across
the published ones and so cannot express a coverage deficit.

Two panels are written on every run, from a single pass over the grids:

  tile_panel.parquet      the 18-city study sample  (02-07, 09 read this)
  tile_panel_all.parquet  every city including the three excluded ones
                          (09's out-of-sample check and 12's AOI check read this)

They are built together deliberately. Every derived column is computed within
city, so the study panel is exactly a row-subset of the full one — writing both
here is what guarantees the two files can never be from different builds, which
would silently invalidate the out-of-sample comparison in 09.

Usage:
  python analysis/01_build_panel.py
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

import ghsl_utils  # noqa: E402
import region_config  # noqa: E402

PANEL = ROOT / "analysis" / "panel" / "tile_panel.parquet"
PANEL_ALL = ROOT / "analysis" / "panel" / "tile_panel_all.parquet"
SMOD_RASTER = ROOT / "data" / "raw" / "ghsl" / "GHS_SMOD_E2020_GLOBE_R2023A_4326_30ss_V2_0.tif"

# Kept in config for provenance but excluded from the study sample.
#
# Nakuru, Garden Route: Meta coverage so sparse there (11% / 13% of eligible
#   tiles) that they would dominate any pooled model.
#
# Kisumu: excluded for a different and more important reason. 303 of its 434
#   eligible tiles never appear in ANY of the 64 timestamps of the Kenya Floods
#   extract, and those tiles hold a median of 1,634 WorldPop residents. A tile
#   with 1,600 people clears Meta's ~10-user threshold easily, so their absence
#   marks the edge of the published event AOI, not a coverage decision. Scoring
#   them as "suppressed" would count an AOI boundary as deprivation bias. Every
#   other city's never-present tiles have a median WorldPop under 800 and most
#   under 200, consistent with genuine sub-threshold counts.
#   Diagnostic: outputs/analysis/A10_aoi_diagnostic.csv
OUT_OF_SAMPLE = {"GardenRoute", "Nakuru", "Kisumu"}


def parse_args():
    p = argparse.ArgumentParser(description="Build the pooled tile panels")
    p.add_argument("--no-smod", action="store_true",
                   help="Skip the GHSL settlement-class join (faster; drops smod_class).")
    p.add_argument("-o", "--out", type=Path, default=PANEL,
                   help="Study-sample panel (default analysis/panel/tile_panel.parquet).")
    p.add_argument("--out-all", type=Path, default=PANEL_ALL,
                   help="All-cities panel (default analysis/panel/tile_panel_all.parquet).")
    return p.parse_args()


def load_grids() -> gpd.GeoDataFrame:
    """Concatenate every per-city independent grid written by pipeline/01b."""
    pattern = "data/processed/city/*/*/01b_coverage/independent_grid.gpkg"
    paths = sorted(ROOT.glob(pattern))
    if not paths:
        raise SystemExit(
            f"No independent grids under {pattern}.\n"
            "Run: ./run --all   (or at least pipeline/01b_meta_coverage_qa.py per city)"
        )

    frames = []
    for path in paths:
        country, city = path.parts[-4], path.parts[-3]
        grid = gpd.read_file(path)
        grid["country"] = country
        grid["city"] = city
        grid["region"] = f"{country}_{city}"
        frames.append(grid)
        mark = "  (out of sample)" if city in OUT_OF_SAMPLE else ""
        print(f"  {country}/{city:<18} {len(grid):>5} tiles{mark}")

    panel = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(panel, geometry="geometry", crs=frames[0].crs)


def attach_smod(panel: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Join the GHSL settlement class so urbanicity can be held fixed."""
    if not SMOD_RASTER.exists():
        print(f"  ! SMOD raster missing at {SMOD_RASTER} — skipping settlement class")
        panel["smod_class"] = pd.NA
        return panel
    codes = ghsl_utils.assign_smod_centroid(panel, SMOD_RASTER)
    panel["smod_code"] = codes
    panel["smod_class"] = [ghsl_utils.smod_class(c) for c in codes]
    print("  SMOD classes:", panel["smod_class"].value_counts(dropna=False).to_dict())
    return panel


def add_derived(panel: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Eligible-grid restriction, shares under both normalisations, and z-scores."""
    panel["published"] = panel["published"].astype(int)
    panel["meta_baseline"] = pd.to_numeric(panel["meta_baseline"], errors="coerce")
    # An unpublished tile is not "zero people" — it is a censored observation.
    # meta_obs keeps that distinction explicit for the two-margin decomposition.
    panel["meta_obs"] = panel["meta_baseline"].where(panel["published"] == 1)

    # The eligible grid is the estimand's support: tiles where a comparison is
    # even defined (WorldPop places people there and GRDI is observed).
    panel["in_eligible"] = (
        (panel["eligible"] == 1)
        & np.isfinite(panel["poverty_mean"])
        & (panel["worldpop_count"] > 0)
    )

    panel["log_wp"] = np.log(panel["worldpop_count"].where(panel["worldpop_count"] > 0))

    elig = panel["in_eligible"]
    grp = panel.loc[elig].groupby("city")

    # Two normalisations of the same WorldPop counts. The pipeline uses the first.
    panel.loc[elig, "wp_share_elig"] = (
        panel.loc[elig, "worldpop_count"] / grp["worldpop_count"].transform("sum")
    )
    pub = elig & (panel["published"] == 1)
    pub_tot = panel.loc[pub].groupby("city")["worldpop_count"].transform("sum")
    panel.loc[pub, "wp_share_pub"] = panel.loc[pub, "worldpop_count"] / pub_tot
    meta_tot = panel.loc[pub].groupby("city")["meta_obs"].transform("sum")
    panel.loc[pub, "meta_share_pub"] = panel.loc[pub, "meta_obs"] / meta_tot

    # City-demeaned predictors: every model below is a within-city comparison, so
    # cross-city differences in GRDI level or city size never enter the estimate.
    for col in ["poverty_mean", "log_wp"]:
        z = panel.loc[elig].groupby("city")[col].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
        panel.loc[elig, f"z_{col}"] = z

    # Within-city GRDI decile, for the dose-response curve.
    panel.loc[elig, "grdi_decile"] = (
        panel.loc[elig].groupby("city")["poverty_mean"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
    )

    # Coarser quadkey prefixes give nested spatial clusters for the SE bandwidth
    # ladder. A zoom-10 cell is ~39 km across at the equator and less at higher
    # latitude; a zoom-8 cell is ~157 km. Within a city each cell is only
    # partly filled, so a zoom-10 cluster holds a median of ~51 tiles.
    for zoom in (8, 10, 12):
        panel[f"blk{zoom}"] = panel["quadkey"].str[:zoom]
    return panel


def summarise(out: pd.DataFrame, path: Path, label: str) -> None:
    elig = out[out.in_eligible]
    print(f"\nWrote {path}  ({label})")
    print(f"  cities            {out.city.nunique()}")
    print(f"  tiles (all)       {len(out):,}")
    print(f"  tiles (eligible)  {len(elig):,}")
    print(f"  published rate    {elig.published.mean():.3f}")
    print(f"  WorldPop total    {elig.worldpop_count.sum():,.0f}")
    print(f"  WorldPop unpublished {elig.loc[elig.published == 0, 'worldpop_count'].sum():,.0f}")


def main():
    args = parse_args()
    print("Loading independent city grids...")
    panel = load_grids()

    if not args.no_smod:
        print("Attaching GHSL settlement class...")
        panel = attach_smod(panel)
    else:
        panel["smod_class"] = pd.NA

    print("Deriving shares, z-scores, spatial blocks...")
    panel = add_derived(panel)

    cent = panel.geometry.centroid
    panel["lon"] = cent.x
    panel["lat"] = cent.y

    # A zoom-14 quadkey is ~2.4 km on a side at the equator but narrows with
    # latitude, so Cape Town tiles are ~20% smaller than Medan ones. Compute the
    # real area per tile rather than assuming a constant.
    panel["area_km2"] = panel.geometry.to_crs("ESRI:54009").area / 1e6

    keep = [
        "region", "country", "city", "quadkey", "blk8", "blk10", "blk12",
        "lon", "lat", "area_km2", "eligible", "in_eligible", "published", "overlap_class",
        "meta_baseline", "meta_obs", "worldpop_count", "log_wp", "poverty_mean",
        "poverty_source", "smod_code", "smod_class", "grdi_decile",
        "wp_share_elig", "wp_share_pub", "meta_share_pub",
        "z_poverty_mean", "z_log_wp",
    ]
    out = pd.DataFrame(panel[[c for c in keep if c in panel.columns]])

    # Every derived column above is computed within city (z-scores, deciles,
    # shares) or per row (blocks, area), so dropping whole cities is exactly a
    # row-subset — the study panel is identical to what a separate in-sample-only
    # build would produce. Writing both from one pass is what stops 09's
    # out-of-sample comparison from silently contrasting two different builds.
    study = out[~out.city.isin(OUT_OF_SAMPLE)].reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out_all.parent.mkdir(parents=True, exist_ok=True)
    study.to_parquet(args.out, index=False)
    out.to_parquet(args.out_all, index=False)

    summarise(study, args.out, "study sample")
    summarise(out, args.out_all, f"all cities, incl. {', '.join(sorted(OUT_OF_SAMPLE))}")


if __name__ == "__main__":
    main()
