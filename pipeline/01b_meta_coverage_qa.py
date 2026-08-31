#!/usr/bin/env python3
"""
01b — Meta coverage QA on an independently defined city grid.

Step 01 starts from published Meta tiles, so unpublished cells never appear.
This step builds the zoom-level city grid from the clip polygon (same tiles as
03f-D), then records two related coverage views:

  1. Overlap 2×2 (Meta × WorldPop presence, independent of GRDI)
     both / worldpop_only / meta_only  (neither = empty clip tiles)
     Percentages of N_union = cells with Meta or WorldPop > 0.

  2. Eligible-grid coverage C_c (WP>0 and GRDI valid)
     C_c = N_published / N_grid

Usage:
  python pipeline/01b_meta_coverage_qa.py --region ZAF_CapeTown
  python pipeline/01b_meta_coverage_qa.py --region ZAF_CapeTown --rebuild

Outputs:
  outputs/.../01b_coverage/Table_meta_coverage.csv
  data/processed/.../01b_coverage/independent_grid.gpkg
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import independent_grid
import region_config

OUT_SUBDIR = "01b_coverage"


def parse_args():
    p = argparse.ArgumentParser(description="01b — Meta coverage QA on the independent city grid")
    p.add_argument("--region", type=str, default=None)
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the independent grid even if independent_grid.gpkg exists.",
    )
    region_config.add_footprint_arg(p)
    return p.parse_args()


def _median(series):
    v = pd.to_numeric(series, errors="coerce")
    v = v[np.isfinite(v)]
    return float(v.median()) if len(v) else np.nan


def _sum(series):
    v = pd.to_numeric(series, errors="coerce")
    v = v[np.isfinite(v)]
    return float(v.sum()) if len(v) else 0.0


def _pct(n, denom):
    return (100.0 * n / denom) if denom else np.nan


def coverage_row(grid, code):
    """One-row coverage table: overlap 2×2 plus eligible-grid C_c."""
    zoom = len(str(grid["quadkey"].iloc[0])) if len(grid) and "quadkey" in grid.columns else np.nan
    n_tiles = int(len(grid))

    both = grid["overlap_class"] == "both"
    wp_only = grid["overlap_class"] == "worldpop_only"
    meta_only = grid["overlap_class"] == "meta_only"
    neither = grid["overlap_class"] == "neither"
    n_both = int(both.sum())
    n_wp_only = int(wp_only.sum())
    n_meta_only = int(meta_only.sum())
    n_neither = int(neither.sum())
    n_union = n_both + n_wp_only + n_meta_only

    wp_both = _sum(grid.loc[both, "worldpop_count"])
    wp_wp_only = _sum(grid.loc[wp_only, "worldpop_count"])
    wp_total = wp_both + wp_wp_only
    meta_both = _sum(grid.loc[both, "meta_baseline"])
    meta_meta_only = _sum(grid.loc[meta_only, "meta_baseline"])
    meta_total = meta_both + meta_meta_only

    elig = grid.loc[grid["eligible"]].copy()
    pub = elig.loc[elig["published"]]
    miss = elig.loc[~elig["published"]]
    n_grid = int(len(elig))
    n_published = int(len(pub))
    n_missing = int(len(miss))
    c_c = (n_published / n_grid) if n_grid else np.nan
    n_both_grdi = int((both & grid["eligible"]).sum())
    n_both_no_grdi = n_both - n_both_grdi

    return {
        "region": code,
        "city": region_config.display_label(code),
        "zoom": int(zoom) if np.isfinite(zoom) else np.nan,
        "N_tiles": n_tiles,
        "N_both": n_both,
        "N_wp_only": n_wp_only,
        "N_meta_only": n_meta_only,
        "N_neither": n_neither,
        "N_union": n_union,
        "pct_both": _pct(n_both, n_union),
        "pct_wp_only": _pct(n_wp_only, n_union),
        "pct_meta_only": _pct(n_meta_only, n_union),
        "N_both_GRDI": n_both_grdi,
        "N_both_no_GRDI": n_both_no_grdi,
        "pct_both_GRDI": _pct(n_both_grdi, n_both),
        "WP_both": wp_both,
        "WP_wp_only": wp_wp_only,
        "pct_WP_unpublished": _pct(wp_wp_only, wp_total),
        "Meta_both": meta_both,
        "Meta_meta_only": meta_meta_only,
        "pct_Meta_no_WP": _pct(meta_meta_only, meta_total),
        "N_grid": n_grid,
        "N_published": n_published,
        "N_missing": n_missing,
        "C_c": c_c,
        "C_c_pct": _pct(n_published, n_grid),
        "median_WP_published": _median(pub["worldpop_count"]) if n_published else np.nan,
        "median_WP_missing": _median(miss["worldpop_count"]) if n_missing else np.nan,
        "median_GRDI_published": _median(pub["poverty_mean"]) if n_published else np.nan,
        "median_GRDI_missing": _median(miss["poverty_mean"]) if n_missing else np.nan,
    }


def _print_row(row):
    n_union = row["N_union"]
    print(
        f"  Tiles in clip: N_tiles={row['N_tiles']}, "
        f"N_union (Meta or WP>0)={n_union}, empty={row['N_neither']}"
    )
    print(
        f"  Overlap 2×2 (% of union): "
        f"both={row['N_both']} ({row['pct_both']:.1f}%), "
        f"WorldPop only={row['N_wp_only']} ({row['pct_wp_only']:.1f}%), "
        f"Meta only={row['N_meta_only']} ({row['pct_meta_only']:.1f}%)"
        if n_union
        else "  Overlap 2×2: union is empty"
    )
    if n_union:
        print(
            f"  Both-source cells with GRDI: {row['N_both_GRDI']}/{row['N_both']}"
            + (
                f" ({row['N_both_no_GRDI']} missing deprivation)"
                if row["N_both_no_GRDI"]
                else " (all have GRDI)"
            )
        )
        print(
            f"  WorldPop in unpublished Meta cells: {row['pct_WP_unpublished']:.1f}% of WP "
            f"({row['WP_wp_only']:,.0f} / {row['WP_both'] + row['WP_wp_only']:,.0f})"
        )
        print(
            f"  Meta in cells with no WorldPop: {row['pct_Meta_no_WP']:.1f}% of Meta "
            f"({row['Meta_meta_only']:,.0f} / {row['Meta_both'] + row['Meta_meta_only']:,.0f})"
        )
    n_grid = row["N_grid"]
    if n_grid:
        print(
            f"  Eligible grid (WP>0, GRDI valid): N_grid={n_grid}, "
            f"N_published={row['N_published']}, N_missing={row['N_missing']}, "
            f"C_c={row['C_c_pct']:.1f}%"
        )
        if row["N_missing"]:
            print(
                f"  median WP published={row['median_WP_published']:.1f}, "
                f"missing={row['median_WP_missing']:.1f}"
            )
            print(
                f"  median GRDI published={row['median_GRDI_published']:.2f}, "
                f"missing={row['median_GRDI_missing']:.2f}"
            )
        else:
            print(f"  median WP published={row['median_WP_published']:.1f} (no unpublished eligible cells)")
    else:
        print("  Eligible grid is empty")


def main():
    args = parse_args()
    if getattr(args, "footprint", None):
        print("01b skipped (city clip grid; not run on footprints)")
        return
    if not args.region:
        raise SystemExit("01b requires --region CITY (e.g. ZAF_CapeTown)")

    paths = independent_grid.resolve_city_grid_inputs(region=args.region)
    if paths is None:
        raise SystemExit(f"01b needs a city region, got {args.region!r}")

    out = region_config.step_paths(args.region, OUT_SUBDIR)

    print("=" * 60)
    print("01b — Meta coverage QA")
    print("=" * 60)
    print(f"  Region: {paths['code']}")

    grid = independent_grid.load_or_build_independent_grid(paths, rebuild=args.rebuild)
    grid = independent_grid.annotate_coverage(grid)
    grid = independent_grid.annotate_overlap(grid)

    row = coverage_row(grid, paths["code"])
    _print_row(row)

    tbl = pd.DataFrame([row])
    tbl_path = out / "Table_meta_coverage.csv"
    tbl.to_csv(tbl_path, index=False)
    print(f"  Saved: {tbl_path}")

    write = grid.copy()
    for col in ("meta_missing", "eligible", "published", "has_wp", "has_meta"):
        if col in write.columns:
            write[col] = write[col].astype(int)
    gpkg_path = out / "independent_grid.gpkg"
    write.to_file(gpkg_path, driver="GPKG")
    print(f"  Saved: {gpkg_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
