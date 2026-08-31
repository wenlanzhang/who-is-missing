"""GHSL settlement-model helpers (GHS-SMOD class map + optional download)."""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

# Degree of Urbanisation / GHS-SMOD R2023A raster codes
SMOD_LABELS = {
    10: "Water",
    11: "Very low density rural",
    12: "Low density rural",
    13: "Rural cluster",
    21: "Suburban or peri-urban",
    22: "Semi-dense urban cluster",
    23: "Dense urban cluster",
    30: "Urban centre",
}

# Collapse to the three-way urban–rural frame used in analysis
SMOD_CLASS_COLLAPSE = {
    10: "Water",
    11: "Rural",
    12: "Rural",
    13: "Rural",
    21: "Town / semi-dense",
    22: "Town / semi-dense",
    23: "Town / semi-dense",
    30: "Urban centre",
}

# Raster codes used when taking a pixel-centre majority
SMOD_CODES = (10, 11, 12, 13, 21, 22, 23, 30)
WATER_CODE = 10

DEFAULT_SMOD_ZIP_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_SMOD_GLOBE_R2023A/GHS_SMOD_E2020_GLOBE_R2023A_4326_30ss/V2-0/"
    "GHS_SMOD_E2020_GLOBE_R2023A_4326_30ss_V2_0.zip"
)


def smod_label(code) -> str | None:
    if code is None or (isinstance(code, float) and code != code):
        return None
    try:
        return SMOD_LABELS.get(int(code))
    except (TypeError, ValueError):
        return None


def smod_class(code) -> str | None:
    if code is None or (isinstance(code, float) and code != code):
        return None
    try:
        return SMOD_CLASS_COLLAPSE.get(int(code))
    except (TypeError, ValueError):
        return None


def ensure_smod_raster(dest: Path, url: str = DEFAULT_SMOD_ZIP_URL, download: bool = False) -> Path:
    """
    Return path to GHS-SMOD GeoTIFF. If missing and download=True, fetch the 30-arc-sec
    WGS84 global mosaic (~32 MB zip) from JRC and unzip next to dest.
    """
    dest = Path(dest)
    if dest.exists():
        return dest
    if not download:
        raise FileNotFoundError(
            f"GHS-SMOD raster not found: {dest}\n"
            "  Place GHS_SMOD_E2020_GLOBE_R2023A_4326_30ss_V2_0.tif in data/raw/ghsl/,\n"
            "  or rerun with --download-smod to fetch it from JRC."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    zip_path = dest.with_suffix(".zip")
    if not zip_path.exists():
        print(f"Downloading GHS-SMOD from JRC → {zip_path}")
        urlretrieve(url, zip_path)
    print(f"Unpacking {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest.parent)
        tifs = [n for n in zf.namelist() if n.lower().endswith(".tif") and not n.lower().endswith(".ovr")]
    if not dest.exists() and tifs:
        extracted = dest.parent / Path(tifs[0]).name
        if extracted != dest and extracted.exists():
            extracted.rename(dest)
    if not dest.exists():
        raise FileNotFoundError(f"Unzipped GHS-SMOD but did not find {dest}")
    return dest


def _smod_lookup_table():
    lookup = np.full(256, -1, dtype=np.int16)
    for i, code in enumerate(SMOD_CODES):
        lookup[int(code)] = i
    return lookup


def assign_smod_centroid(gdf, raster_path):
    """One GHS-SMOD sample at each quadkey representative point."""
    import rasterio
    from rasterio.warp import transform as rio_transform

    pts = gdf.geometry.representative_point()
    xs = pts.x.to_numpy()
    ys = pts.y.to_numpy()
    with rasterio.open(raster_path) as src:
        if src.crs is not None and str(src.crs) not in ("EPSG:4326", "OGC:CRS84"):
            xs, ys = rio_transform("EPSG:4326", src.crs, xs, ys)
        nodata = src.nodata
        codes = []
        for val in src.sample(zip(xs, ys)):
            v = val[0] if val is not None else None
            if v is None or (nodata is not None and v == nodata) or v == 0:
                codes.append(np.nan)
            else:
                codes.append(int(v))
    return np.array(codes, dtype=object)


def assign_smod_composition(gdf, raster_path, all_touched=True) -> pd.DataFrame:
    """
    One GHS-SMOD pass per quadkey. Returns:

      water_fraction          — Water votes / all SMOD votes
      ghsl_smod_majority      — overall majority code (sensitivity)
      ghsl_smod               — majority among non-water pixels (main);
                                Water only if the cell has no land pixels
    """
    import pandas as pd
    import rasterio
    from rasterio.windows import from_bounds
    from rasterstats import zonal_stats

    from align_utils import _pad_window

    with rasterio.open(raster_path) as src:
        window = _pad_window(from_bounds(*gdf.total_bounds, transform=src.transform), src)
        fill = src.nodata if src.nodata is not None else 0
        data = src.read(1, window=window, boundless=True, fill_value=fill)
        transform = src.window_transform(window)
        nodata = src.nodata if src.nodata is not None else 0

    stats = zonal_stats(
        list(gdf.geometry),
        data,
        affine=transform,
        categorical=True,
        all_touched=all_touched,
        nodata=nodata,
    )
    water = WATER_CODE
    rows = []
    for d in stats:
        votes = {int(k): float(v) for k, v in (d or {}).items() if k not in (None, 0) and v}
        n = sum(votes.values())
        n_water = votes.get(water, 0.0)
        land_votes = {k: v for k, v in votes.items() if k != water}
        majority = max(votes, key=votes.get) if votes else np.nan
        land_code = max(land_votes, key=land_votes.get) if land_votes else (water if n_water else np.nan)
        rows.append(
            {
                "water_fraction": (n_water / n) if n else np.nan,
                "smod_n_pixels": n,
                "smod_n_water": n_water,
                "smod_n_land": n - n_water,
                "ghsl_smod_majority": majority,
                "ghsl_smod": land_code,
            }
        )
    return pd.DataFrame(rows, index=gdf.index)


def assign_smod_majority(gdf, raster_path, land_preferred=False, all_touched=True):
    """Back-compat: overall majority or land-preferred majority codes."""
    comp = assign_smod_composition(gdf, raster_path, all_touched=all_touched)
    col = "ghsl_smod" if land_preferred else "ghsl_smod_majority"
    return comp[col].to_numpy()


def attach_smod_columns(gdf, codes=None, composition=None):
    """Add GHSL columns. Prefer `composition` (water_fraction + both class rules)."""
    out = gdf.copy()
    if composition is not None:
        for c in composition.columns:
            out[c] = composition[c].to_numpy()
        out["ghsl_smod_label"] = out["ghsl_smod"].map(smod_label)
        out["ghsl_class"] = out["ghsl_smod"].map(smod_class)
        out["ghsl_smod_majority_label"] = out["ghsl_smod_majority"].map(smod_label)
        out["ghsl_class_majority"] = out["ghsl_smod_majority"].map(smod_class)
        return out
    out["ghsl_smod"] = codes
    out["ghsl_smod_label"] = out["ghsl_smod"].map(smod_label)
    out["ghsl_class"] = out["ghsl_smod"].map(smod_class)
    return out
