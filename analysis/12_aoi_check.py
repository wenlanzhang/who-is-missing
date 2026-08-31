#!/usr/bin/env python3
"""
A12 — Is a missing tile *suppressed*, or simply outside the published event AOI?

The outcome modelled everywhere else is "Meta has no value for this tile". That
can mean two different things:

  suppressed        the tile is inside the event extract but fell below Meta's
                    ~10-user threshold. This IS a coverage decision and is what
                    the analysis is about.
  outside the AOI   the tile never appears in the extract at all, because the
                    published event footprint does not reach it. This is NOT a
                    coverage decision, and scoring it as deprivation bias would
                    be a serious error.

Two checks distinguish them:

  1. Is the quadkey set fixed across timestamps? If Meta published a fixed AOI
     and flagged sub-threshold tiles as NaN, every timestamp would carry the same
     rows. If instead tiles enter and leave, absence is threshold behaviour.
  2. For tiles that never appear in ANY timestamp, how much population does
     WorldPop put there? A tile with 1,600 residents clears a 10-user threshold
     easily, so if such tiles are systematically absent that is an AOI edge.

This is what identified Kisumu: 303 of its 434 eligible tiles never appear in any
of 64 timestamps and hold a median of 1,634 residents. It is excluded in
01_build_panel.py for that reason.

NOTE ON QUADKEYS: they must be read as strings. Quadkeys in the Americas begin
with "0", and pandas will infer int64 and silently strip the leading zero, so
every join against the panel fails and the country looks 100% outside its own
AOI. That bug is why an earlier version of this check appeared to fail for
Mexico and Colombia.

Outputs
  outputs/analysis/A10_aoi_diagnostic.csv

Usage:
  python analysis/12_aoi_check.py
"""

import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import region_config  # noqa: E402

PANEL = ROOT / "analysis" / "panel" / "tile_panel_all.parquet"
OUT = ROOT / "outputs" / "analysis"

# A never-present tile holding more people than this is very unlikely to be
# below a 10-user threshold, so a high median here points at an AOI edge rather
# than suppression.
SUSPICIOUS_MEDIAN_WP = 800


def zip_for(country: str, regions: dict) -> Path | None:
    """The PDC extract the country's baselines were built from."""
    root = Path(regions["data_root"])
    for key, cfg in regions.items():
        if isinstance(cfg, dict) and key.split("_")[0] == country and cfg.get("pdc_raw_dir"):
            p = root / cfg["pdc_raw_dir"]
            if p.exists():
                return p
    return None


def aoi_union(path: Path) -> tuple[set, int]:
    """Every quadkey appearing in any timestamp, and the number of timestamps."""
    z = zipfile.ZipFile(path)
    files = [n for n in z.namelist() if n.endswith(".csv")]
    seen = set()
    for n in files:
        try:
            # dtype=str is load-bearing: see NOTE ON QUADKEYS above.
            df = pd.read_csv(z.open(n), usecols=["quadkey"], dtype={"quadkey": str})
        except Exception:
            continue
        seen |= set(df["quadkey"].str.zfill(14))
    return seen, len(files)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    regions = region_config.load_regions()
    if not PANEL.exists():
        raise SystemExit(
            f"Missing {PANEL}. Build it with:\n"
            "  python analysis/01_build_panel.py --include-out-of-sample "
            "-o analysis/panel/tile_panel_all.parquet")
    d = pd.read_parquet(PANEL)
    d = d[d.in_eligible].copy()
    d["quadkey"] = d["quadkey"].astype(str).str.zfill(14)

    rows = []
    for country in sorted(d.country.unique()):
        p = zip_for(country, regions)
        if p is None:
            print(f"{country}: no PDC extract found")
            continue
        union, n_ts = aoi_union(p)
        print(f"{country}: {n_ts} timestamps, AOI union = {len(union):,} quadkeys")
        for city, g in d[d.country == country].groupby("city"):
            inside = g.quadkey.isin(union)
            lost = g[~inside]
            rows.append({
                "country": country, "city": city, "n_eligible": len(g),
                "in_aoi": int(inside.sum()), "never_in_aoi": int((~inside).sum()),
                "pct_never_in_aoi": 100 * (~inside).mean(),
                "C_c": g.published.mean(),
                "median_wp_never_in_aoi": float(lost.worldpop_count.median())
                if len(lost) else np.nan,
                "median_grdi_never_in_aoi": float(lost.poverty_mean.median())
                if len(lost) else np.nan,
            })

    out = pd.DataFrame(rows).sort_values("median_wp_never_in_aoi", ascending=False)
    out["verdict"] = np.where(
        out.median_wp_never_in_aoi > SUSPICIOUS_MEDIAN_WP,
        "AOI EDGE - exclude", "consistent with suppression")
    out.loc[out.never_in_aoi == 0, "verdict"] = "fully inside AOI"

    print(f"\n{'city':<20} {'elig':>5} {'never in AOI':>13} {'%':>6} "
          f"{'medWP(lost)':>12}  verdict")
    for _, r in out.iterrows():
        mw = "-" if not np.isfinite(r.median_wp_never_in_aoi) else f"{r.median_wp_never_in_aoi:,.0f}"
        print(f"{r.city:<20} {r.n_eligible:>5} {r.never_in_aoi:>13} "
              f"{r.pct_never_in_aoi:>5.1f}% {mw:>12}  {r.verdict}")

    flagged = out[out.verdict == "AOI EDGE - exclude"]
    print(f"\n  {len(flagged)} city/cities flagged as AOI-truncated: "
          f"{', '.join(flagged.city) if len(flagged) else 'none'}")
    print("  Flagged cities belong in OUT_OF_SAMPLE in analysis/01_build_panel.py.")

    out.to_csv(OUT / "A10_aoi_diagnostic.csv", index=False)
    print(f"\nWrote {OUT / 'A10_aoi_diagnostic.csv'}")


if __name__ == "__main__":
    main()
