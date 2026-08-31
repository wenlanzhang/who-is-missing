#!/usr/bin/env python3
"""
Step 1 — Harmonise all rasters to the quadkey grid.

Aggregates to the exact Meta quadkey grid:
  - WorldPop (zonal sum)
  - Meta baseline (already in quadkeys)
  - Poverty (zonal mean). Default source is GRDI (GeoTIFF); Meta RWI CSV is optional.

Usage:
  python pipeline/01_harmonise_datasets.py --worldpop /path/to.tif --meta /path/to.gpkg
  python pipeline/01_harmonise_datasets.py --region KEN_Nairobi
  python pipeline/01_harmonise_datasets.py --region KEN_Nairobi --poverty-source rwi
  python pipeline/01_harmonise_datasets.py --footprint KEN
  python pipeline/01_harmonise_datasets.py --worldpop ... --meta ... --poverty /path/to/poverty.tif
  python pipeline/01_harmonise_datasets.py --filter-by meta --filter-min 50   # keep quadkeys with meta_baseline > 50
  python pipeline/01_harmonise_datasets.py --filter-by worldpop --filter-min 50   # keep quadkeys with worldpop_count > 50
  python pipeline/01_harmonise_datasets.py --region KEN_Nairobi --clip-source osm
  python pipeline/01_harmonise_datasets.py --region KEN_Nairobi --clip-source geob
  python pipeline/01_harmonise_datasets.py --filter-by both --filter-min 50   # keep quadkeys where BOTH meta and worldpop >= 50

City mode (--region) clips to a city boundary and writes the city 01 GPKG.
Footprint mode (--footprint) keeps the Meta event AOI with no extra clip,
writes outputs/footprints/{CODE}/ so it does not overwrite outputs/city/, and
caches data/processed/footprints/{CODE}_aligned.parquet. Use one or the other, not both.
"""

# python pipeline/01_harmonise_datasets.py --filter-by both --filter-min 30

import argparse
import itertools
import multiprocessing
import sys
from pathlib import Path

# Allow importing region_config / clip_utils from pipeline/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import clip_utils

import pandas as pd
import geopandas as gpd
import numpy as np
from rasterstats import zonal_stats

try:
    import mercantile
except ImportError:
    mercantile = None

# Default paths (used when --region not set)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = PROJECT_ROOT / "data" / "raw"  # WorldPop, Poverty rasters; put files here or override with CLI
DEFAULT_WORLDPOP = PROJECT_ROOT / "data" / "raw" / "worldpop" / "phl_pop_2026_CN_100m_R2025A_v1.tif"
DEFAULT_META = PROJECT_ROOT / "outputs" / "fb_baseline_median_PHL.gpkg"  # from build_fb_baseline_median.py
DEFAULT_POVERTY = PROJECT_ROOT / "data" / "raw" / "povmap-grdi-v1-10.tif"


def filter_quadkeys(gdf, by=None, min_val=50):
    """
    Keep quadkeys only where the specified variable(s) exceed the threshold.

    Args:
        gdf: GeoDataFrame with meta_baseline and worldpop_count columns
        by: "meta"/"fb" = meta_baseline; "worldpop" = worldpop_count; "both" = both >= min_val
        min_val: minimum value (default 50)

    Returns:
        Filtered GeoDataFrame. If by is None, returns gdf unchanged.
    """
    if by is None:
        return gdf
    if by == "both":
        if "meta_baseline" not in gdf.columns or "worldpop_count" not in gdf.columns:
            return gdf
        return gdf[(gdf["meta_baseline"] >= min_val) & (gdf["worldpop_count"] >= min_val)].copy()
    col = "meta_baseline" if by in ("meta", "fb") else "worldpop_count"
    if col not in gdf.columns:
        return gdf
    return gdf[gdf[col] >= min_val].copy()


def parse_args():
    p = argparse.ArgumentParser(description="Harmonise rasters to quadkey grid")
    p.add_argument("--region", type=str, default=None,
                   help="City region from config/regions.json (e.g. KEN_Nairobi, or used by ./run --region KEN).")
    p.add_argument(
        "--footprint",
        type=str,
        default=None,
        help="Event-footprint country (same countries as ./run --region). "
             "Philippines is PHL. No city clip. One country; every footprint treated equally.",
    )
    p.add_argument("--ref-hour", type=int, default=None, choices=[0, 8, 16], metavar="HOUR",
                   help="Reference hour (0, 8, or 16). Uses outputs/{region}/fb_baseline_median_h{HOUR:02d}.gpkg. Overrides config meta path.")
    p.add_argument(
        "--baseline-method",
        type=str,
        default="n_baseline",
        choices=["n_baseline", "shift"],
        help="Meta GPKG method: n_baseline (default, column in the PDC CSV) or shift (7-day lag). "
             "Applies to both --region (city) and --footprint.",
    )
    p.add_argument(
        "--worldpop-method",
        type=str,
        default=None,
        choices=["centre", "fractional", "all_touched"],
        help="WorldPop aggregation (centre, fractional, all_touched). "
             "Footprint default: centre. City default: all_touched unless this is set.",
    )
    p.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Override GeoParquet cache (footprint default: data/processed/footprints/{CODE}_aligned.parquet).",
    )
    p.add_argument("--worldpop", type=Path, default=None, help="Override: WorldPop raster path")
    p.add_argument("--meta", type=Path, default=None, help="Override: Meta baseline GPKG path")
    p.add_argument("--poverty", type=Path, default=None, help="Override: Poverty raster or RWI CSV path")
    p.add_argument(
        "--poverty-source",
        type=str,
        default=None,
        choices=["grdi", "rwi"],
        help="Poverty layer: grdi (default, global GeoTIFF) or rwi (per-region Meta RWI CSV). "
             "Ignored when --poverty is set, except to label the source.",
    )
    p.add_argument("--no-poverty", action="store_true", help="Skip poverty aggregation")
    p.add_argument("--poverty-nodata", type=float, default=None)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--filter-by", type=str, default=None, choices=["meta", "fb", "worldpop", "both"],
                   help="Keep quadkeys where variable(s) >= threshold. meta/fb = meta_baseline; worldpop = worldpop_count; both = both >= threshold")
    p.add_argument("--filter-min", type=float, default=50,
                   help="Minimum threshold when --filter-by is set (default: 50)")
    p.add_argument("--min-meta", type=float, default=None, help="(Deprecated) Use --filter-by meta --filter-min N")
    p.add_argument("--plot", action="store_true", help="Create data overview figure (R)")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers for zonal_stats (default: 1). Use 4+ for large grids.")
    p.add_argument("--clip-shape", type=Path, default=None,
                   help="City --region only: path to .shp, .gpkg, or .geojson when --clip-source local.")
    p.add_argument(
        "--clip-source",
        type=str,
        default=None,
        choices=["local", "osm", "geob"],
        help="City boundary: local (config clip_shape), osm (OSMnx/Nominatim), "
             "or geob (geoBoundaries). Default: local.",
    )
    p.add_argument("--clip-osm-place", type=str, default=None,
                   help="OSM/Nominatim place query when --clip-source osm (overrides config).")
    p.add_argument("--clip-geob-iso3", type=str, default=None,
                   help="geoBoundaries ISO3 code when --clip-source geob (e.g. KEN).")
    p.add_argument("--clip-geob-adm", type=str, default=None,
                   help="geoBoundaries ADM level when --clip-source geob (e.g. ADM1, ADM3).")
    p.add_argument("--clip-geob-name", type=str, default=None,
                   help="geoBoundaries feature name to keep (e.g. Nairobi).")
    p.add_argument("--clip-refresh", action="store_true",
                   help="Re-download OSM/geoBoundaries even if a cache file exists.")
    return p.parse_args()


def run_footprint(args):
    """
    Event-footprint 01: Meta event AOI as published, no extra clip.
    Writes outputs/footprints/{CODE}/01/ and data/processed/footprints/{CODE}_aligned.parquet.
    """
    from align_utils import (
        add_shares_and_ratios,
        aggregate_poverty,
        aggregate_worldpop,
        filter_quadkeys as _filter_quadkeys,
        print_harmonisation_checks,
        rename_meta_baseline,
        save_aligned,
    )
    import region_config

    if (
        args.clip_shape is not None
        or args.clip_source is not None
        or args.clip_refresh
        or args.clip_osm_place
        or args.clip_geob_iso3
        or args.clip_geob_adm
        or args.clip_geob_name
    ):
        raise ValueError(
            "Footprint mode keeps the Meta event AOI. City clips belong to --region, not --footprint."
        )

    cfg = region_config.get_footprint_config(args.footprint)
    code = args.footprint
    worldpop = args.worldpop or cfg["worldpop"]
    if args.meta is not None:
        meta_path = args.meta
    else:
        meta_path = region_config.get_meta_baseline_path(
            cfg, code, ref_hour=args.ref_hour, baseline_method=args.baseline_method
        )
    if args.no_poverty:
        poverty_path, poverty_source = None, None
    elif args.poverty is not None:
        poverty_path = args.poverty
        poverty_source = args.poverty_source or (
            "rwi" if Path(poverty_path).suffix.lower() == ".csv" else "grdi"
        )
    else:
        poverty_source = args.poverty_source or cfg.get("poverty_source") or "grdi"
        poverty_path = region_config.resolve_poverty_path(cfg, poverty_source)

    out_dir = region_config.get_footprint_output_dir(code, "01")
    out_gpkg = args.output or (out_dir / "harmonised_meta_worldpop.gpkg")
    out_parquet = args.parquet or region_config.get_aligned_parquet(code)

    print(f"Footprint: {code} ({cfg.get('name', code)})")
    print(f"  Meta baseline: {meta_path}")
    if args.baseline_method:
        print(f"  Baseline method: {args.baseline_method}")
    print("  Processing extent: Meta availability (no extra clip)")
    if not args.no_poverty:
        print(f"Poverty source: {poverty_source} → {poverty_path}")

    worldpop_p, meta_p = Path(worldpop), Path(meta_path)
    if not worldpop_p.exists():
        raise FileNotFoundError(f"WorldPop raster not found: {worldpop}")
    if not meta_p.exists():
        raise FileNotFoundError(
            f"Meta baseline GPKG not found: {meta_path}\n"
            f"  Build it with: python data_prep/build_fb_baseline_median.py --region {code}"
            + (f" --baseline-method {args.baseline_method}" if args.baseline_method else "")
        )
    if not args.no_poverty:
        if poverty_path is None:
            raise FileNotFoundError(
                "Poverty data is required. Pass --poverty or use --no-poverty to skip."
            )
        pov_p = Path(poverty_path)
        if "/path/to" in str(poverty_path).lower() or not pov_p.exists():
            raise FileNotFoundError(f"Poverty data required but missing: {poverty_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading datasets...")
    meta = gpd.read_file(meta_path)
    print(f"  Meta: {len(meta)} quadkeys, CRS={meta.crs}")
    if meta.crs != "EPSG:4326":
        meta = meta.to_crs("EPSG:4326")
        print("  Meta reprojected to EPSG:4326")

    workers = getattr(args, "workers", 1)
    method = args.worldpop_method or "centre"
    meta = aggregate_worldpop(meta, worldpop, workers=workers, method=method)
    meta = rename_meta_baseline(meta)
    print_harmonisation_checks(meta, region=code)

    if poverty_path and Path(poverty_path).exists():
        meta = aggregate_poverty(
            meta,
            poverty_path,
            poverty_source=poverty_source,
            poverty_nodata=args.poverty_nodata,
            workers=workers,
        )
    elif poverty_path is not None:
        print("\n*** Poverty SKIPPED (file missing) — output will NOT include poverty_mean ***")

    filter_by = args.filter_by
    if args.min_meta is not None:
        filter_by = "meta"
        args.filter_min = args.min_meta
        print("  Note: --min-meta is deprecated, use --filter-by meta --filter-min N")
    if filter_by is not None:
        before = len(meta)
        meta = _filter_quadkeys(meta, by=filter_by, min_val=args.filter_min)
        print(f"  Kept {len(meta)} / {before} quadkeys (dropped {before - len(meta)})")

    meta = add_shares_and_ratios(meta)
    print("\n--- Harmonised summary ---")
    cols = ["worldpop_count", "meta_baseline", "worldpop_share", "meta_share"]
    if "poverty_mean" in meta.columns:
        cols.append("poverty_mean")
    print(meta[[c for c in cols if c in meta.columns]].describe())
    save_aligned(meta, out_gpkg=out_gpkg, out_parquet=out_parquet)
    return meta


def main():
    args = parse_args()
    if args.region and args.footprint:
        raise ValueError("Use either --region (city) or --footprint (event Meta AOI), not both.")
    if args.footprint:
        return run_footprint(args)

    # Resolve paths from --region config or defaults
    if args.region:
        import region_config
        args.region = region_config.require_city_region(args.region)
        cfg = region_config.get_region_config(args.region)
        worldpop = args.worldpop or cfg["worldpop"]
        if args.meta is not None:
            meta_path = args.meta
        else:
            meta_path = region_config.get_meta_baseline_path(
                cfg, args.region, ref_hour=args.ref_hour, baseline_method=args.baseline_method
            )
        if args.no_poverty:
            poverty_path = None
            poverty_source = None
        elif args.poverty is not None:
            poverty_path = args.poverty
            poverty_source = args.poverty_source or (
                "rwi" if Path(poverty_path).suffix.lower() == ".csv" else "grdi"
            )
        else:
            poverty_source = args.poverty_source or cfg.get("poverty_source") or "grdi"
            poverty_path = region_config.resolve_poverty_path(cfg, poverty_source)
        if args.clip_shape is not None:
            cfg["clip_shape"] = args.clip_shape
        if args.clip_osm_place:
            cfg["clip_osm_place"] = args.clip_osm_place
        if args.clip_geob_iso3:
            cfg["clip_geob_iso3"] = args.clip_geob_iso3
        if args.clip_geob_adm:
            cfg["clip_geob_adm"] = args.clip_geob_adm
        if args.clip_geob_name:
            cfg["clip_geob_name"] = args.clip_geob_name
        clip_source = args.clip_source or cfg.get("clip_source") or "local"
        cfg["clip_source"] = clip_source
        out_dir = region_config.step_paths(args.region, "01")
        print(f"Region: {args.region} ({cfg.get('name', args.region)})")
        if not args.no_poverty:
            print(f"Poverty source: {poverty_source} → {poverty_path}")
        print(f"Clip source: {clip_source}")
    else:
        worldpop = args.worldpop or DEFAULT_WORLDPOP
        meta_path = args.meta or DEFAULT_META
        if args.no_poverty:
            poverty_path = None
            poverty_source = None
        elif args.poverty is not None:
            poverty_path = args.poverty
            poverty_source = args.poverty_source or (
                "rwi" if Path(poverty_path).suffix.lower() == ".csv" else "grdi"
            )
        else:
            poverty_source = args.poverty_source or "grdi"
            if poverty_source == "rwi":
                raise FileNotFoundError(
                    "--poverty-source rwi requires --poverty PATH or --region (per-city RWI CSV in config)."
                )
            poverty_path = DEFAULT_POVERTY
        cfg = {
            "clip_shape": args.clip_shape,
            "clip_osm_place": args.clip_osm_place,
            "clip_geob_iso3": args.clip_geob_iso3,
            "clip_geob_adm": args.clip_geob_adm,
            "clip_geob_name": args.clip_geob_name,
            "clip_source": args.clip_source or "local",
        }
        clip_source = cfg["clip_source"]
        out_dir = PROJECT_ROOT / "outputs" / "01"
        if not args.no_poverty:
            print(f"Poverty source: {poverty_source} → {poverty_path}")
        print(f"Clip source: {clip_source}")

    out_gpkg = args.output or (out_dir / "harmonised_meta_worldpop.gpkg")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 0. Validate data paths (signal missing data early)
    # -------------------------------------------------------------------------
    worldpop_p = Path(worldpop)
    meta_p = Path(meta_path)
    if not worldpop_p.exists():
        raise FileNotFoundError(f"WorldPop raster not found: {worldpop}")
    if not meta_p.exists():
        raise FileNotFoundError(f"Meta baseline GPKG not found: {meta_path}")
    if not args.no_poverty and poverty_path is None:
        raise FileNotFoundError(
            "Poverty data is required. Add poverty path to config/regions.json or pass --poverty /path/to/file, "
            "or use --no-poverty to skip."
        )
    elif not args.no_poverty:
        pov_p = Path(poverty_path)
        is_placeholder = "/path/to" in str(poverty_path).lower() or str(poverty_path).startswith("/path/")
        if is_placeholder or not pov_p.exists():
            msg = (
                f"Poverty data required but missing or placeholder: {poverty_path}\n"
                "  → Update config/regions.json with a valid path, or use --no-poverty to skip."
            )
            raise FileNotFoundError(msg)

    # Validate clip_shape only for local files; OSM/geoBoundaries are fetched later
    if clip_source == "local" and cfg.get("clip_shape") is not None:
        clip_p = Path(cfg["clip_shape"])
        if not clip_p.exists():
            raise FileNotFoundError(f"Clip shape not found: {cfg['clip_shape']}")

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    print("Loading datasets...")
    meta = gpd.read_file(meta_path)
    print(f"  Meta: {len(meta)} quadkeys, CRS={meta.crs}")

    # -------------------------------------------------------------------------
    # 2. Harmonise CRS and extent
    # -------------------------------------------------------------------------
    target_crs = "EPSG:4326"
    if meta.crs != target_crs:
        meta = meta.to_crs(target_crs)
        print("  Meta reprojected to EPSG:4326")

    # -------------------------------------------------------------------------
    # 2b. Clip to study area (city/region boundary)
    # -------------------------------------------------------------------------
    clip_gdf = clip_utils.load_clip_boundary(
        cfg,
        source=clip_source,
        region_code=args.region,
        refresh=args.clip_refresh,
    )
    if clip_gdf is not None:
        if clip_gdf.crs != meta.crs:
            clip_gdf = clip_gdf.to_crs(meta.crs)
        clip_union = clip_utils.unary_geom(clip_gdf)
        before = len(meta)
        meta = meta[meta.geometry.intersects(clip_union)].copy()
        print(f"  Clipped to study area ({clip_source}): kept {len(meta)} / {before} quadkeys")
        clip_save = gpd.GeoDataFrame(
            {"clip_source": [clip_source]},
            geometry=[clip_union],
            crs=clip_gdf.crs,
        )
        clip_out = out_dir / "clip_boundary.gpkg"
        clip_save.to_file(clip_out, driver="GPKG")
        print(f"  Saved clip polygon: {clip_out}")

    # WorldPop is already EPSG:4326; rasterstats will handle CRS alignment
    # when zones are in same CRS as raster

    # -------------------------------------------------------------------------
    # 3. Aggregate WorldPop to Meta quadkey grid (zonal sum)
    # -------------------------------------------------------------------------
    workers = getattr(args, "workers", 1)
    if args.worldpop_method:
        from align_utils import aggregate_worldpop

        meta = aggregate_worldpop(meta, worldpop, workers=workers, method=args.worldpop_method)
    else:
        print(f"Aggregating WorldPop to Meta quadkey grid... (workers={workers})")
        zs_kw = dict(
            stats=["sum", "count", "min", "max", "mean"],
            nodata=-99999.0,  # WorldPop NoData value
            all_touched=True,  # include edge pixels
        )
        if workers > 1:
            geoms = meta.geometry.tolist()
            n = len(geoms)
            chunk_size = max(1, (n + workers - 1) // workers)
            chunks = [(geoms[i : i + chunk_size], str(worldpop), zs_kw) for i in range(0, n, chunk_size)]

            def _zonal_stats_chunk(args):
                geoms_chunk, raster_path, kw = args
                return zonal_stats(geoms_chunk, raster_path, **kw)

            with multiprocessing.Pool(workers) as pool:
                stats_lists = pool.map(_zonal_stats_chunk, chunks)
            stats = list(itertools.chain.from_iterable(stats_lists))
        else:
            stats = zonal_stats(meta.geometry, str(worldpop), **zs_kw)

        meta = meta.copy()
        meta["worldpop_count"] = [s["sum"] if s["sum"] is not None else np.nan for s in stats]
        meta["worldpop_n_pixels"] = [s["count"] if s["count"] is not None else 0 for s in stats]
        meta["worldpop_min"] = [s["min"] if s["min"] is not None else np.nan for s in stats]
        meta["worldpop_max"] = [s["max"] if s["max"] is not None else np.nan for s in stats]
        meta["worldpop_mean"] = [s["mean"] if s["mean"] is not None else np.nan for s in stats]

    # Rename Meta baseline column (auto-detect first numeric non-geometry column)
    meta_col = next((c for c in meta.columns if c not in ("geometry", "quadkey") and pd.api.types.is_numeric_dtype(meta[c])), None)
    if meta_col and meta_col != "meta_baseline":
        meta = meta.rename(columns={meta_col: "meta_baseline"})

    # -------------------------------------------------------------------------
    # 5. Validation: zero-division and sparsity
    # -------------------------------------------------------------------------
    print("\n--- Harmonisation checks ---")

    # Same projection
    print(f"CRS: {meta.crs} (both datasets)")

    # Spatial extent: Meta defines extent
    meta_bounds = meta.total_bounds
    print(f"Spatial extent (Meta): {meta_bounds}")

    # Check if Meta extent overlaps expected region (avoids wrong-PDC-folder issues)
    if args.region:
        try:
            import region_config
            cfg = region_config.get_region_config(args.region)
            lon_r, lat_r = cfg.get("lon_range"), cfg.get("lat_range")
            if lon_r and lat_r and len(lon_r) == 2 and len(lat_r) == 2:
                xmin, ymin, xmax, ymax = meta_bounds
                overlaps = not (xmax < lon_r[0] or xmin > lon_r[1] or ymax < lat_r[0] or ymin > lat_r[1])
                if not overlaps:
                    print("\n*** WARNING: Meta extent does NOT overlap expected region! ***")
                    print(f"  Meta: lon [{xmin:.1f}, {xmax:.1f}], lat [{ymin:.1f}, {ymax:.1f}]")
                    print(f"  Expected ({args.region}): lon {lon_r}, lat {lat_r}")
                    print("  → Meta quadkeys may be from wrong PDC event/folder. Check pdc_raw_dir.")
        except (ImportError, ValueError, KeyError):
            pass

    # Unit: both in counts per quadkey
    print("Unit: counts per quadkey (WorldPop = sum of 100m cells; Meta = midnight baseline)")

    # Zero-division: cells with 0 in either dataset
    wp_zeros = (meta["worldpop_count"] == 0) | meta["worldpop_count"].isna()
    meta_zeros = meta["meta_baseline"] == 0
    both_valid = ~wp_zeros & ~meta_zeros
    print(f"WorldPop zero/NaN cells: {wp_zeros.sum()}")
    print(f"Meta zero cells: {meta_zeros.sum()}")
    print(f"Cells with both non-zero: {both_valid.sum()} (safe for ratio comparisons)")

    # Sparsity: cells with very few pixels
    sparse = meta["worldpop_n_pixels"] < 10
    print(f"Extreme sparsity (n_pixels < 10): {sparse.sum()} quadkeys")

    # -------------------------------------------------------------------------
    # 6. Optional: Aggregate poverty (raster or RWI CSV) to quadkey grid
    # -------------------------------------------------------------------------
    if poverty_path and Path(poverty_path).exists():
        suffix = poverty_path.suffix.lower()
        if suffix == ".csv":
            # RWI CSV: lat, lon, rwi (or latitude, longitude, rwi)
            if mercantile is None:
                raise ImportError("mercantile required for RWI CSV: pip install mercantile")
            print("\n--- Aggregating RWI CSV to quadkeys ---")
            df_rwi = pd.read_csv(poverty_path)
            # Normalize column names
            lat_col = next((c for c in df_rwi.columns if "lat" in c.lower() and "lon" not in c.lower()), "latitude")
            lon_col = next((c for c in df_rwi.columns if "lon" in c.lower() or c == "longitude"), "longitude")
            rwi_col = next((c for c in df_rwi.columns if "rwi" in c.lower() or "wealth" in c.lower()), None)
            if rwi_col is None:
                rwi_col = df_rwi.select_dtypes(include=[np.number]).columns[0]  # first numeric
            df_rwi = df_rwi[[lat_col, lon_col, rwi_col]].dropna().rename(
                columns={lat_col: "lat", lon_col: "lon", rwi_col: "rwi"}
            )
            # Get zoom from meta quadkeys (vectorized for speed)
            zoom = len(str(meta["quadkey"].iloc[0]))
            lons = df_rwi["lon"].values
            lats = df_rwi["lat"].values
            df_rwi["quadkey"] = [
                mercantile.quadkey(mercantile.tile(lon, lat, zoom))
                for lon, lat in zip(lons, lats)
            ]
            # RWI: higher = wealthier. Negate so poverty_mean: higher = poorer (same direction as GRDI).
            rwi_by_qk = df_rwi.groupby("quadkey")["rwi"].mean().reset_index()
            rwi_by_qk["poverty_mean"] = -rwi_by_qk["rwi"]
            rwi_by_qk = rwi_by_qk[["quadkey", "poverty_mean"]]
            meta = meta.merge(rwi_by_qk, on="quadkey", how="left")
            meta["poverty_n_pixels"] = meta["quadkey"].map(
                df_rwi.groupby("quadkey").size().reindex(meta["quadkey"]).fillna(0).astype(int)
            )
            poverty_source = poverty_source or "rwi"
            print(f"  RWI: mean per quadkey, valid cells: {(meta['poverty_mean'].notna()).sum()}")
        else:
            # Raster (e.g. GRDI GeoTIFF): higher = more deprived; do not negate.
            print("\n--- Aggregating poverty raster to quadkeys ---")
            zs_kw = {"stats": ["mean", "count"], "all_touched": True}
            if args.poverty_nodata is not None:
                zs_kw["nodata"] = args.poverty_nodata
            stats_pov = zonal_stats(meta.geometry, str(poverty_path), **zs_kw)
            meta["poverty_mean"] = [s["mean"] if s["mean"] is not None else np.nan for s in stats_pov]
            meta["poverty_n_pixels"] = [s["count"] if s["count"] is not None else 0 for s in stats_pov]
            poverty_source = poverty_source or "grdi"
            print(f"  Poverty: mean per quadkey, valid cells: {(meta['poverty_n_pixels'] > 0).sum()}")
        if poverty_source:
            meta["poverty_source"] = poverty_source
    else:
        if poverty_path is not None:
            print("\n*** Poverty SKIPPED (file missing or placeholder) — output will NOT include poverty_mean ***")
        elif not args.no_poverty and args.region:
            print("\n*** Poverty SKIPPED (no poverty path in config for this region) ***")

    # -------------------------------------------------------------------------
    # 6b. Optional: Filter quadkeys by minimum count
    # -------------------------------------------------------------------------
    filter_by = args.filter_by
    if args.min_meta is not None:
        filter_by = "meta"
        args.filter_min = args.min_meta
        print("  Note: --min-meta is deprecated, use --filter-by meta --filter-min N")
    if filter_by is not None:
        before = len(meta)
        meta = filter_quadkeys(meta, by=filter_by, min_val=args.filter_min)
        if filter_by == "both":
            print(f"\n--- Filter: meta_baseline >= {args.filter_min} AND worldpop_count >= {args.filter_min} ---")
        else:
            col = "meta_baseline" if filter_by in ("meta", "fb") else "worldpop_count"
            print(f"\n--- Filter: {col} >= {args.filter_min} ---")
        print(f"  Kept {len(meta)} / {before} quadkeys (dropped {before - len(meta)})")

    # -------------------------------------------------------------------------
    # 6c. Create spatial shares (for all downstream analysis)
    # -------------------------------------------------------------------------
    wp_sum = meta["worldpop_count"].fillna(0).sum()
    meta_sum = meta["meta_baseline"].fillna(0).sum()
    meta["worldpop_share"] = meta["worldpop_count"].fillna(0) / (wp_sum + 1e-10)
    meta["meta_share"] = meta["meta_baseline"].fillna(0) / (meta_sum + 1e-10)
    meta["worldpop_raw"] = meta["worldpop_count"]
    meta["meta_raw"] = meta["meta_baseline"]
    print(f"\n--- Spatial shares ---")
    print(f"  worldpop_share = worldpop / sum(worldpop), meta_share = meta / sum(meta)")
    print(f"  Sum worldpop_share: {meta['worldpop_share'].sum():.6f}, Sum meta_share: {meta['meta_share'].sum():.6f}")

    # Summary stats
    print("\n--- Harmonised summary ---")
    cols = ["worldpop_count", "meta_baseline", "worldpop_share", "meta_share"]
    if "poverty_mean" in meta.columns:
        cols.append("poverty_mean")
    print(meta[[c for c in cols if c in meta.columns]].describe())

    # -------------------------------------------------------------------------
    # 7. Save
    # -------------------------------------------------------------------------
    meta.to_file(out_gpkg, driver="GPKG")
    print(f"\nSaved: {out_gpkg}")

    # -------------------------------------------------------------------------
    # 8. Optional: Data overview figure (R)
    # -------------------------------------------------------------------------
    if args.plot:
        import subprocess
        script_dir = Path(__file__).resolve().parent
        r_script = script_dir / "01_plot_descriptive.R"
        if r_script.exists():
            out_path = out_dir / "01_descriptive_overview.png"
            result = subprocess.run(
                ["Rscript", str(r_script), "-i", str(out_gpkg)],
                cwd=script_dir.parent,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"  Figures: 01_data_overview.png, 01_bivariate_worldpop_meta.png, 01_bivariate_worldpop_meta_basemap.png")
            else:
                print(f"  R figure failed: {result.stderr[:200] if result.stderr else result.stdout}")
        else:
            print("  01_plot_descriptive.R not found, skipping figure")

    return meta


if __name__ == "__main__":
    main()
