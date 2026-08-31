"""Shared helpers for Step 01 alignment (city and country/footprint modes)."""

from __future__ import annotations

import itertools
import multiprocessing
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterstats import zonal_stats

try:
    import mercantile
except ImportError:
    mercantile = None


def filter_quadkeys(gdf, by=None, min_val=50):
    """Keep quadkeys only where the specified variable(s) exceed the threshold."""
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


def _zonal_stats_chunk(args):
    geoms_chunk, raster_path, kw = args
    return zonal_stats(geoms_chunk, raster_path, **kw)


def run_zonal_stats(geoms, raster_path, zs_kw, workers=1):
    """zonal_stats over a geometry list, optionally in parallel."""
    raster_path = str(raster_path)
    workers = max(1, int(workers))
    if workers == 1:
        return zonal_stats(geoms, raster_path, **zs_kw)
    geoms = list(geoms)
    n = len(geoms)
    chunk_size = max(1, (n + workers - 1) // workers)
    chunks = [(geoms[i : i + chunk_size], raster_path, zs_kw) for i in range(0, n, chunk_size)]
    with multiprocessing.Pool(workers) as pool:
        stats_lists = pool.map(_zonal_stats_chunk, chunks)
    return list(itertools.chain.from_iterable(stats_lists))


def _pad_window(window, src, pad=2):
    from rasterio.windows import Window

    col_off = max(0, int(window.col_off) - pad)
    row_off = max(0, int(window.row_off) - pad)
    col_end = min(src.width, int(window.col_off + window.width) + pad)
    row_end = min(src.height, int(window.row_off + window.height) + pad)
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


def _worldpop_pixel_centre(meta, raster_path, nodata=-99999.0):
    """
    Assign each WorldPop pixel to exactly one quadkey via the pixel centre.

    rasterio.features.rasterize(..., all_touched=False) burns a pixel only if
    its centre falls inside a polygon, so adjacent quadkeys cannot share a pixel.
    """
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import from_bounds

    meta = meta.copy().reset_index(drop=True)
    n = len(meta)
    print("Aggregating WorldPop (pixel-centre → unique quadkey; no double-counting)...")
    with rasterio.open(raster_path) as src:
        window = _pad_window(from_bounds(*meta.total_bounds, transform=src.transform), src)
        fill = src.nodata if src.nodata is not None else nodata
        data = src.read(1, window=window, boundless=True, fill_value=fill)
        transform = src.window_transform(window)
        nd = src.nodata if src.nodata is not None else nodata
        print(f"  Raster window: {tuple(data.shape)} ({data.nbytes / 1e6:.0f} MB), {n} quadkeys")
        shapes = ((geom, i + 1) for i, geom in enumerate(meta.geometry))
        burned = rasterize(
            shapes,
            out_shape=data.shape,
            transform=transform,
            all_touched=False,
            fill=0,
            dtype="uint32",
        )

    valid = np.isfinite(data) & (data != nd) & (data > -1e20)
    assigned = valid & (burned > 0)
    ids = burned[assigned]
    pops = data[assigned].astype(np.float64)
    n_pix = np.bincount(ids, minlength=n + 1)
    sums = np.bincount(ids, weights=pops, minlength=n + 1)

    meta["worldpop_count"] = sums[1:]
    meta["worldpop_n_pixels"] = n_pix[1:].astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        meta["worldpop_mean"] = np.where(n_pix[1:] > 0, sums[1:] / n_pix[1:], np.nan)

    assigned_sum = float(sums[1:].sum())
    unique_in_quadkeys = float(data[assigned].sum())
    window_sum = float(data[valid].sum())
    print(f"  Assigned WorldPop: {assigned_sum:,.0f}  (unique pixels in quadkeys: {unique_in_quadkeys:,.0f})")
    print(f"  WorldPop in raster window but outside Meta quadkeys: {window_sum - unique_in_quadkeys:,.0f}")
    if unique_in_quadkeys:
        rel = abs(assigned_sum - unique_in_quadkeys) / unique_in_quadkeys
        print(f"  Conservation error (should be ~0): {rel:.4%}")
    return meta


def _worldpop_fractional(meta, raster_path, nodata=-99999.0):
    """
    Area-weighted split of WorldPop pixels that straddle quadkey edges.

    Requires exactextract. Interior pixels match the centre method; only
    overlapping edge pixels are split.
    """
    try:
        from exactextract import exact_extract
    except ImportError as e:
        raise ImportError(
            "WorldPop method 'fractional' needs exactextract (pip install exactextract)."
        ) from e

    print("Aggregating WorldPop (fractional area-weighted overlap)...")
    meta = meta.copy().reset_index(drop=True)
    stats = exact_extract(str(raster_path), meta, ["sum", "count", "mean"], output="pandas")
    meta["worldpop_count"] = stats["sum"].to_numpy()
    meta["worldpop_n_pixels"] = stats["count"].fillna(0).to_numpy()
    meta["worldpop_mean"] = stats["mean"].to_numpy()
    print(f"  Fractional WorldPop sum: {meta['worldpop_count'].sum():,.0f}")
    return meta


def aggregate_worldpop(meta, worldpop_path, workers=1, method="centre"):
    """
    Sum WorldPop onto Meta quadkeys.

    method:
      centre      — each 100 m pixel belongs to exactly one quadkey (pixel centre).
                    Default. Population-conserving; no edge double-counting.
      fractional  — area-weighted split of edge pixels (exactextract).
      all_touched — legacy rasterstats all_touched=True (double-counts edges; do not use).
    """
    method = (method or "centre").strip().lower()
    if method == "centre":
        return _worldpop_pixel_centre(meta, worldpop_path)
    if method == "fractional":
        return _worldpop_fractional(meta, worldpop_path)
    if method in ("all_touched", "zonal"):
        print(f"Aggregating WorldPop with rasterstats all_touched=True (legacy; workers={workers})...")
        zs_kw = dict(
            stats=["sum", "count", "min", "max", "mean"],
            nodata=-99999.0,
            all_touched=True,
        )
        stats = run_zonal_stats(meta.geometry, worldpop_path, zs_kw, workers=workers)
        meta = meta.copy()
        meta["worldpop_count"] = [s["sum"] if s["sum"] is not None else np.nan for s in stats]
        meta["worldpop_n_pixels"] = [s["count"] if s["count"] is not None else 0 for s in stats]
        meta["worldpop_min"] = [s["min"] if s["min"] is not None else np.nan for s in stats]
        meta["worldpop_max"] = [s["max"] if s["max"] is not None else np.nan for s in stats]
        meta["worldpop_mean"] = [s["mean"] if s["mean"] is not None else np.nan for s in stats]
        return meta
    raise ValueError(f"Unknown WorldPop method {method!r}. Use centre, fractional, or all_touched.")


def rename_meta_baseline(meta):
    """Rename the first numeric non-geometry Meta column to meta_baseline."""
    meta_col = next(
        (
            c
            for c in meta.columns
            if c not in ("geometry", "quadkey") and pd.api.types.is_numeric_dtype(meta[c])
        ),
        None,
    )
    if meta_col and meta_col != "meta_baseline":
        meta = meta.rename(columns={meta_col: "meta_baseline"})
    return meta


def aggregate_poverty(meta, poverty_path, poverty_source=None, poverty_nodata=None, workers=1):
    """
    Attach poverty_mean to quadkeys.

    CSV (RWI): mean RWI per quadkey, then poverty_mean = -RWI (higher = poorer).
    Raster (GRDI): zonal mean; higher = more deprived (do not negate).
    """
    poverty_path = Path(poverty_path)
    suffix = poverty_path.suffix.lower()
    if suffix == ".csv":
        if mercantile is None:
            raise ImportError("mercantile required for RWI CSV: pip install mercantile")
        print("\n--- Aggregating RWI CSV to quadkeys ---")
        df_rwi = pd.read_csv(poverty_path)
        lat_col = next((c for c in df_rwi.columns if "lat" in c.lower() and "lon" not in c.lower()), "latitude")
        lon_col = next((c for c in df_rwi.columns if "lon" in c.lower() or c == "longitude"), "longitude")
        rwi_col = next((c for c in df_rwi.columns if "rwi" in c.lower() or "wealth" in c.lower()), None)
        if rwi_col is None:
            rwi_col = df_rwi.select_dtypes(include=[np.number]).columns[0]
        df_rwi = df_rwi[[lat_col, lon_col, rwi_col]].dropna().rename(
            columns={lat_col: "lat", lon_col: "lon", rwi_col: "rwi"}
        )
        zoom = len(str(meta["quadkey"].iloc[0]))
        lons = df_rwi["lon"].values
        lats = df_rwi["lat"].values
        df_rwi["quadkey"] = [
            mercantile.quadkey(mercantile.tile(lon, lat, zoom))
            for lon, lat in zip(lons, lats)
        ]
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
        print("\n--- Aggregating poverty raster to quadkeys ---")
        zs_kw = {"stats": ["mean", "count"], "all_touched": True}
        if poverty_nodata is not None:
            zs_kw["nodata"] = poverty_nodata
        stats_pov = run_zonal_stats(meta.geometry, poverty_path, zs_kw, workers=workers)
        meta["poverty_mean"] = [s["mean"] if s["mean"] is not None else np.nan for s in stats_pov]
        meta["poverty_n_pixels"] = [s["count"] if s["count"] is not None else 0 for s in stats_pov]
        poverty_source = poverty_source or "grdi"
        print(f"  Poverty: mean per quadkey, valid cells: {(meta['poverty_n_pixels'] > 0).sum()}")
    if poverty_source:
        meta["poverty_source"] = poverty_source
    return meta


def add_shares_and_ratios(meta, eps=1.0, universe=None):
    """
    Representation measures. Meta NA is kept as missing (never filled with 0).

    Stage-2 universe S (default): Meta observed AND WorldPop > 0.
    Pass `universe` (e.g. land GHSL mask) to intersect with S so both share
    denominators use the identical conditional sample:

      M_i* = Meta_i / sum_{j in S} Meta_j
      W_i* = WP_i   / sum_{j in S} WP_j
      R_i  = log(M_i* / W_i*)

    WorldPop in Meta-NA cells is excluded from W* — Stage 1 missingness
    does not leak into Stage 2.
    """
    wp = meta["worldpop_count"].astype(float)
    fb = meta["meta_baseline"].astype(float)
    observed = fb.notna()
    meta["meta_observed"] = observed
    wp_ok = wp.notna() & (wp > 0)
    univ = observed & wp_ok
    if universe is not None:
        univ = univ & universe.fillna(False).astype(bool)

    meta["worldpop_raw"] = wp
    meta["meta_raw"] = fb
    meta["diff_meta_wp"] = np.where(univ, fb - wp, np.nan)
    meta["ratio_meta_wp"] = np.where(univ, fb / wp, np.nan)
    meta["log_ratio_meta_wp"] = np.where(univ, np.log((fb + eps) / (wp + eps)), np.nan)

    meta["meta_share"] = np.nan
    meta["worldpop_share"] = np.nan
    meta["repr_residual"] = np.nan
    n_s = int(univ.sum())
    if univ.any():
        m_sum = float(fb[univ].sum())
        w_sum = float(wp[univ].sum())
        meta.loc[univ, "meta_share"] = fb[univ] / (m_sum + 1e-12)
        meta.loc[univ, "worldpop_share"] = wp[univ] / (w_sum + 1e-12)
        pos = univ & (fb > 0) & (wp > 0)
        n_zero = int((univ & (fb == 0)).sum())
        ms = meta.loc[pos, "meta_share"].astype(float)
        ws = meta.loc[pos, "worldpop_share"].astype(float)
        meta.loc[pos, "repr_residual"] = np.log(ms / ws)
        print(f"  Exact Meta zeros in S: {n_zero} (R not computed for zeros; no epsilon fill)")

    n_na = int((~observed).sum())
    print("\n--- Representation (Meta NA kept as missing) ---")
    print(f"  Meta observed-user baseline missing: {n_na} / {len(meta)} ({n_na / len(meta):.1%})")
    print(f"  Stage-2 universe S (observed ∩ WorldPop>0"
          f"{' ∩ extra mask' if universe is not None else ''}): {n_s} cells")
    if univ.any():
        ms_sum = float(meta.loc[univ, "meta_share"].sum())
        ws_sum = float(meta.loc[univ, "worldpop_share"].sum())
        print(f"  Sum M* on S: {ms_sum:.8f}  Sum W* on S: {ws_sum:.8f}")
        outside = ~univ
        n_wp_leak = int((outside & wp_ok & meta["worldpop_share"].notna()).sum())
        n_meta_na_with_share = int((~observed & meta["worldpop_share"].notna()).sum())
        print(f"  Cells outside S with a WorldPop share (should be 0): {n_wp_leak}")
        print(f"  Meta-NA cells with a WorldPop share (should be 0): {n_meta_na_with_share}")
        print("  R = log(M*/W*)  (same S in both denominators)")
    return meta


def print_harmonisation_checks(meta, region=None):
    print("\n--- Harmonisation checks ---")
    print(f"CRS: {meta.crs} (both datasets)")
    meta_bounds = meta.total_bounds
    print(f"Spatial extent (Meta): {meta_bounds}")

    if region:
        try:
            import region_config

            cfg = None
            try:
                cfg = region_config.get_footprint_config(region)
            except ValueError:
                try:
                    cfg = region_config.get_region_config(region)
                except ValueError:
                    cfg = None
            if cfg:
                lon_r, lat_r = cfg.get("lon_range"), cfg.get("lat_range")
                if lon_r and lat_r and len(lon_r) == 2 and len(lat_r) == 2:
                    xmin, ymin, xmax, ymax = meta_bounds
                    overlaps = not (xmax < lon_r[0] or xmin > lon_r[1] or ymax < lat_r[0] or ymin > lat_r[1])
                    if not overlaps:
                        print("\n*** WARNING: Meta extent does NOT overlap expected region! ***")
                        print(f"  Meta: lon [{xmin:.1f}, {xmax:.1f}], lat [{ymin:.1f}, {ymax:.1f}]")
                        print(f"  Expected ({region}): lon {lon_r}, lat {lat_r}")
                        print("  → Meta quadkeys may be from wrong PDC event/folder. Check pdc_raw_dir.")
        except (ImportError, ValueError, KeyError):
            pass

    print("WorldPop: resident population (pixel-centre sum). Meta: observed-user baseline (PDC n_baseline), not a census.")
    wp_na = meta["worldpop_count"].isna()
    wp_zero = meta["worldpop_count"] == 0
    meta_na = meta["meta_baseline"].isna()
    meta_zero = meta["meta_baseline"] == 0
    both_obs = meta["meta_baseline"].notna() & meta["worldpop_count"].notna() & (meta["worldpop_count"] > 0)
    print(f"WorldPop NA: {int(wp_na.sum())}  zero: {int(wp_zero.sum())}")
    print(f"Meta baseline NA (keep as missing): {int(meta_na.sum())}  zero: {int(meta_zero.sum())}")
    print(f"Cells with Meta observed and WorldPop>0: {int(both_obs.sum())}")
    if "worldpop_n_pixels" in meta.columns:
        sparse = meta["worldpop_n_pixels"] < 10
        print(f"Extreme sparsity (n_pixels < 10): {int(sparse.sum())} quadkeys")


def save_aligned(gdf, out_gpkg: Path | None = None, out_parquet: Path | None = None):
    """Write GeoPackage and/or GeoParquet cache."""
    if out_gpkg is not None:
        out_gpkg = Path(out_gpkg)
        out_gpkg.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out_gpkg, driver="GPKG")
        print(f"\nSaved: {out_gpkg}")
    if out_parquet is not None:
        out_parquet = Path(out_parquet)
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(out_parquet, index=False)
        print(f"Saved: {out_parquet}")
