"""
Independent zoom-level city grid (clip tiles, not Meta-defined).

Used by 01b (coverage QA) and 03f-D (privacy-censoring sensitivity).
Unpublished tiles get WorldPop via the same pixel-centre method as step 01,
and GRDI zonal means. Eligible cells are WorldPop > 0 with valid GRDI.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

import clip_utils
import region_config


def resolve_city_grid_inputs(region=None, input_path=None, footprint=None):
    """City clip, step-01 GPKG, WorldPop and GRDI paths. None if this is a footprint run."""
    if footprint:
        return None
    code = region or region_config.region_from_artifact_path(input_path)
    if not code:
        return None
    try:
        code = region_config.require_city_region(code)
    except ValueError:
        return None
    g01 = region_config.find_artifact(code, "01", "harmonised_meta_worldpop.gpkg")
    clip = region_config.geo_dir(code, "01") / "clip_boundary.gpkg"
    if not clip.exists():
        clip = region_config.find_artifact(code, "01", "clip_boundary.gpkg")
    cfg = region_config.get_region_config(code)
    return {
        "code": code,
        "g01": g01,
        "clip": clip,
        "worldpop": cfg.get("worldpop"),
        "poverty": cfg.get("poverty_grdi"),
        "cfg": cfg,
        "cache": region_config.geo_dir(code, "01b_coverage") / "independent_grid.gpkg",
    }


def load_city_clip(paths):
    """Prefer the saved step-01 clip polygon; otherwise rebuild from config."""
    clip_path = paths.get("clip")
    if clip_path is not None and Path(clip_path).exists():
        clip_gdf = gpd.read_file(clip_path)
        if clip_gdf.crs is None:
            clip_gdf = clip_gdf.set_crs("EPSG:4326")
        return clip_gdf.to_crs("EPSG:4326")
    return clip_utils.load_clip_boundary(paths["cfg"], region_code=paths["code"])


def _as_bool(series):
    if series is None:
        return None
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().any():
        return num.fillna(0).astype(int).astype(bool)
    return series.astype(str).str.lower().isin(("1", "true", "yes", "t"))


def is_published_meta(gdf):
    """True when the cell is in the Meta 01 layer with a positive count."""
    meta = pd.to_numeric(gdf["meta_baseline"], errors="coerce")
    missing = (
        _as_bool(gdf["meta_missing"])
        if "meta_missing" in gdf.columns
        else pd.Series(False, index=gdf.index)
    )
    return (~missing) & meta.notna() & (meta > 0)


def has_worldpop(gdf):
    """True when WorldPop zonal sum is strictly positive."""
    wp = pd.to_numeric(gdf["worldpop_count"], errors="coerce")
    return wp.notna() & (wp > 0)


def is_eligible(gdf):
    """WorldPop > 0 and poverty-valid. Independent of Meta."""
    ok = has_worldpop(gdf)
    ok &= gdf["poverty_mean"].notna()
    if "poverty_n_pixels" in gdf.columns:
        ok &= pd.to_numeric(gdf["poverty_n_pixels"], errors="coerce").fillna(0) > 0
    return ok


def build_independent_city_grid(paths):
    """
    All zoom-level tiles intersecting the city clip, left-joined to Meta (01).

    Unpublished tiles (in clip, not in Meta) get WorldPop via pixel-centre
    assignment and GRDI zonal means. No WorldPop-size filter here.
    """
    from shapely.geometry import box
    from shapely.prepared import prep

    from align_utils import aggregate_poverty, aggregate_worldpop

    try:
        import mercantile
    except ImportError as e:
        raise ImportError("mercantile is required for the independent city grid") from e

    g01_path = paths["g01"]
    worldpop = paths["worldpop"]
    poverty = paths["poverty"]
    if g01_path is None or not Path(g01_path).exists():
        raise FileNotFoundError("Independent grid needs step-01 harmonised_meta_worldpop.gpkg")
    if worldpop is None or not Path(worldpop).exists():
        raise FileNotFoundError(f"Independent grid needs WorldPop raster: {worldpop}")
    if poverty is None or not Path(poverty).exists():
        raise FileNotFoundError(f"Independent grid needs GRDI raster: {poverty}")

    clip_gdf = load_city_clip(paths)
    if clip_gdf is None or clip_gdf.empty:
        raise FileNotFoundError("Independent grid needs a city clip polygon (01/clip_boundary.gpkg)")

    g01 = gpd.read_file(g01_path)
    if g01.crs is None:
        g01 = g01.set_crs("EPSG:4326")
    g01 = g01.to_crs("EPSG:4326")
    g01["quadkey"] = g01["quadkey"].astype(str)
    keep = [
        c
        for c in (
            "quadkey",
            "geometry",
            "meta_baseline",
            "worldpop_count",
            "poverty_mean",
            "poverty_n_pixels",
        )
        if c in g01.columns
    ]
    obs = g01[keep].copy()
    obs["meta_missing"] = False

    zoom = len(str(obs["quadkey"].iloc[0]))
    have = set(obs["quadkey"].astype(str))
    clip_u = clip_utils.unary_geom(clip_gdf.to_crs("EPSG:4326"))
    clip_prep = prep(clip_u)
    west, south, east, north = clip_gdf.to_crs("EPSG:4326").total_bounds

    missing_recs = []
    n_bbox_tiles = 0
    for t in mercantile.tiles(west, south, east, north, zoom):
        n_bbox_tiles += 1
        qk = mercantile.quadkey(t)
        if qk in have:
            continue
        bb = mercantile.bounds(t)
        poly = box(bb.west, bb.south, bb.east, bb.north)
        if not clip_prep.intersects(poly):
            continue
        missing_recs.append({"quadkey": qk, "geometry": poly})

    print(
        f"  Independent grid: zoom={zoom}, clip-bbox tiles={n_bbox_tiles}, "
        f"Meta (01)={len(obs)}, unpublished intersecting clip={len(missing_recs)}"
    )

    if not missing_recs:
        grid = obs.reset_index(drop=True)
    else:
        miss = gpd.GeoDataFrame(missing_recs, crs="EPSG:4326")
        miss["meta_baseline"] = np.nan
        miss["meta_missing"] = True
        miss = aggregate_worldpop(miss, worldpop, method="centre")
        miss = aggregate_poverty(miss, poverty, poverty_source="grdi")
        grid = pd.concat([obs, miss], ignore_index=True)
        grid = gpd.GeoDataFrame(grid, geometry="geometry", crs="EPSG:4326")

    grid.attrs["zoom"] = zoom
    return grid


def load_or_build_independent_grid(paths, rebuild=False):
    """Reuse 01b GPKG when present so 03f-D matches the coverage QA grid."""
    cache = paths.get("cache")
    if cache is not None and Path(cache).exists() and not rebuild:
        print(f"  Independent grid ← {cache}")
        g = gpd.read_file(cache)
        if g.crs is None:
            g = g.set_crs("EPSG:4326")
        else:
            g = g.to_crs("EPSG:4326")
        if "meta_missing" in g.columns:
            g["meta_missing"] = _as_bool(g["meta_missing"])
        return g
    return build_independent_city_grid(paths)


def annotate_coverage(grid):
    """Add eligible / published / coverage_class on the independent grid."""
    g = grid.copy()
    g["meta_missing"] = (
        _as_bool(g["meta_missing"]) if "meta_missing" in g.columns else False
    )
    g["eligible"] = is_eligible(g)
    g["published"] = is_published_meta(g) & g["eligible"]
    g["coverage_class"] = np.where(
        ~g["eligible"],
        "ineligible",
        np.where(g["published"], "published Meta", "eligible but unpublished Meta"),
    )
    return g


def annotate_overlap(grid):
    """Add Meta × WorldPop presence class (independent of GRDI eligibility)."""
    g = grid.copy()
    has_wp = has_worldpop(g)
    has_meta = is_published_meta(g)
    g["has_wp"] = has_wp
    g["has_meta"] = has_meta
    g["overlap_class"] = np.select(
        [has_wp & has_meta, has_wp & ~has_meta, ~has_wp & has_meta],
        ["both", "worldpop_only", "meta_only"],
        default="neither",
    )
    return g
