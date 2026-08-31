#!/usr/bin/env python3
"""
Study-area clip polygons: local file, OSM/Nominatim (via OSMnx), or geoBoundaries.

Sources
  local  existing clip_shape (.gpkg / .shp / .geojson)
  osm    OSMnx geocode_to_gdf (Nominatim fallback if osmnx is not installed)
  geob   geoBoundaries gbOpen API, filtered by admin name

Online results are cached under data/raw/boundaries/cache/ so later runs are offline.
"""

from __future__ import annotations

import json
import re
import tempfile
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "boundaries" / "cache"

VALID_CLIP_SOURCES = ("local", "osm", "geob")
DEFAULT_CLIP_SOURCE = "local"
USER_AGENT = "Residential_population2/1.0 (research pipeline; clip boundaries)"
GEOB_API = "https://www.geoboundaries.org/api/current/gbOpen/{iso}/{adm}/"
NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"

# Region-code prefix → ISO3 / default ADM level for geoBoundaries city units.
_PREFIX_ISO3 = {
    "PHL": "PHL",
    "KEN": "KEN",
    "MEX": "MEX",
    "IDN": "IDN",
    "LKA": "LKA",
    "COL": "COL",
    "ECU": "ECU",
    "ZAF": "ZAF",
}
_PREFIX_ADM = {
    "PHL": "ADM3",
    "KEN": "ADM1",
    "MEX": "ADM1",
    "IDN": "ADM1",
    "LKA": "ADM0",
    "COL": "ADM1",
    "ECU": "ADM1",
    "ZAF": "ADM1",
}
_PREFIX_COUNTRY = {
    "PHL": "Philippines",
    "KEN": "Kenya",
    "MEX": "Mexico",
    "IDN": "Indonesia",
    "LKA": "Sri Lanka",
    "COL": "Colombia",
    "ECU": "Ecuador",
    "ZAF": "South Africa",
}


def get_clip_source(cfg: dict | None = None, override: str | None = None) -> str:
    """CLI override → per-region clip_source → global/default `local`."""
    raw = override or (cfg or {}).get("clip_source") or DEFAULT_CLIP_SOURCE
    src = str(raw).strip().lower()
    if src not in VALID_CLIP_SOURCES:
        raise ValueError(
            f"Unknown clip_source {src!r}. Use one of: {', '.join(VALID_CLIP_SOURCES)}"
        )
    return src


def has_clip_boundary(cfg: dict, source: str | None = None) -> bool:
    """True when this region will be clipped (local file or an online source)."""
    src = get_clip_source(cfg, source)
    if src == "local":
        return bool(cfg.get("clip_shape"))
    return True


def unary_geom(gdf: gpd.GeoDataFrame):
    """Dissolved geometry (union_all, with unary_union fallback)."""
    geom = gdf.geometry
    if hasattr(geom, "union_all"):
        return geom.union_all()
    return geom.unary_union


def load_clip_boundary(
    cfg: dict,
    source: str | None = None,
    *,
    region_code: str | None = None,
    refresh: bool = False,
) -> gpd.GeoDataFrame | None:
    """
    Return a clip GeoDataFrame in EPSG:4326, or None when local clipping is unset.

    `source`: local | osm | geob. Cached online polygons are reused unless refresh=True.
    """
    src = get_clip_source(cfg, source)
    code = region_code or cfg.get("region_code") or "custom"

    if src == "local":
        return _load_local(cfg)

    cache_path = CACHE_DIR / src / f"{code}.gpkg"
    if cache_path.exists() and not refresh:
        gdf = gpd.read_file(cache_path)
        print(f"  Clip source {src}: cached {cache_path}")
        return _to_wgs84(gdf)

    if src == "osm":
        gdf = _load_osm(cfg, code)
    else:
        gdf = _load_geob(cfg, code)

    gdf = _to_wgs84(gdf)
    _save_cache(gdf, cache_path)
    print(f"  Clip source {src}: saved cache {cache_path}")
    return gdf


def _to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def _save_cache(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG")


def _load_local(cfg: dict) -> gpd.GeoDataFrame | None:
    clip_shape = cfg.get("clip_shape")
    if not clip_shape:
        return None
    path = Path(clip_shape)
    if not path.exists():
        raise FileNotFoundError(f"Clip shape not found: {clip_shape}")
    print(f"  Clip source local: {path}")
    return _to_wgs84(gpd.read_file(path))


def _prefix(region_code: str | None) -> str:
    code = (region_code or "").upper()
    return code[:3] if code else ""


def _osm_place(cfg: dict, region_code: str | None) -> str:
    place = cfg.get("clip_osm_place")
    if place:
        return str(place).strip()
    label = cfg.get("city_label") or cfg.get("map_bbox_label")
    country = _PREFIX_COUNTRY.get(_prefix(region_code))
    if label and country:
        return f"{label}, {country}"
    if label:
        return str(label)
    raise ValueError(
        "clip-source osm needs clip_osm_place in config, --clip-osm-place, "
        "or a city_label / map_bbox_label."
    )


def _load_osm(cfg: dict, region_code: str | None) -> gpd.GeoDataFrame:
    place = _osm_place(cfg, region_code)
    print(f"  Clip source osm: geocoding {place!r}")
    try:
        import osmnx as ox

        gdf = ox.geocode_to_gdf(place)
    except ImportError:
        print("  osmnx not installed; using Nominatim directly (pip install osmnx)")
        gdf = _nominatim_geocode(place)
    gdf = _polygons_only(gdf, source="osm", query=place)
    gdf["clip_source"] = "osm"
    gdf["clip_query"] = place
    return gdf


def _nominatim_geocode(place: str) -> gpd.GeoDataFrame:
    params = urllib.parse.urlencode(
        {
            "q": place,
            "format": "geojson",
            "polygon_geojson": 1,
            "limit": 1,
        }
    )
    data = _http_json(f"{NOMINATIM_SEARCH}?{params}")
    features = data.get("features") or []
    if not features:
        raise RuntimeError(f"Nominatim returned no results for {place!r}.")
    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")


def _load_geob(cfg: dict, region_code: str | None) -> gpd.GeoDataFrame:
    iso3, adm, name = _geob_query(cfg, region_code)
    print(f"  Clip source geob: {iso3} {adm} name={name!r}")
    meta = _http_json(GEOB_API.format(iso=iso3, adm=adm))
    url = meta.get("simplifiedGeometryGeoJSON") or meta.get("gjDownloadURL")
    if not url:
        raise RuntimeError(
            f"geoBoundaries API returned no GeoJSON URL for {iso3} {adm}."
        )
    print(f"  Downloading {url}")
    layer = _read_vector_url(url)
    matched = _match_admin_name(layer, name)
    matched = _polygons_only(matched, source="geob", query=name)
    matched = matched.copy()
    matched["clip_source"] = "geob"
    matched["clip_query"] = name
    matched["clip_geob_iso3"] = iso3
    matched["clip_geob_adm"] = adm
    return matched


def _geob_query(cfg: dict, region_code: str | None) -> tuple[str, str, str]:
    prefix = _prefix(region_code)
    iso3 = (cfg.get("clip_geob_iso3") or _PREFIX_ISO3.get(prefix) or "").strip().upper()
    adm = (cfg.get("clip_geob_adm") or _PREFIX_ADM.get(prefix) or "ADM1").strip().upper()
    if adm.isdigit():
        adm = f"ADM{adm}"
    name = (
        cfg.get("clip_geob_name")
        or cfg.get("city_label")
        or cfg.get("map_bbox_label")
        or ""
    )
    name = str(name).strip()
    if not iso3:
        raise ValueError(
            "clip-source geob needs clip_geob_iso3 in config or --clip-geob-iso3."
        )
    if not name:
        raise ValueError(
            "clip-source geob needs clip_geob_name in config, --clip-geob-name, "
            "or a city_label / map_bbox_label."
        )
    return iso3, adm, name


def _match_admin_name(gdf: gpd.GeoDataFrame, query: str) -> gpd.GeoDataFrame:
    name_col = _name_column(gdf)
    folded_query = _fold(query)
    names = gdf[name_col].fillna("").astype(str)
    folded_names = names.map(_fold)

    exact = gdf.loc[folded_names == folded_query]
    if len(exact) == 1:
        return exact
    if len(exact) > 1:
        return exact

    contained = gdf.loc[folded_names.map(lambda n: folded_query in n and bool(n))]
    if len(contained) == 1:
        return contained
    if len(contained) > 1:
        # Prefer the shortest name (e.g. "Nairobi" over "Nairobi Informal Settlements")
        idx = contained.assign(_nlen=contained[name_col].astype(str).str.len())["_nlen"].idxmin()
        return contained.loc[[idx]]

    sample = sorted({n for n in names if n})[:25]
    extra = " …" if names.nunique() > 25 else ""
    raise ValueError(
        f"No geoBoundaries feature matching {query!r} in column {name_col!r}. "
        f"Available names (sample): {sample}{extra}"
    )


def _name_column(gdf: gpd.GeoDataFrame) -> str:
    for col in ("shapeName", "shapeGroup", "NAME", "name", "ADM1_EN", "ADM2_EN", "ADM3_EN"):
        if col in gdf.columns:
            return col
    text_cols = [c for c in gdf.columns if gdf[c].dtype == object]
    if text_cols:
        return text_cols[0]
    raise ValueError("geoBoundaries layer has no name column to match against.")


def _polygons_only(gdf: gpd.GeoDataFrame, *, source: str, query: str) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        raise RuntimeError(f"{source} returned no geometry for {query!r}.")
    polys = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if polys.empty:
        types = sorted(set(gdf.geometry.geom_type.dropna().astype(str)))
        raise RuntimeError(
            f"{source} did not return a polygon for {query!r} (got {types}). "
            "Try a more specific place name (e.g. 'Nairobi, Kenya')."
        )
    return polys


def _fold(text: str) -> str:
    """Lowercase, strip accents, collapse punctuation for name matching."""
    nfkd = unicodedata.normalize("NFKD", str(text))
    ascii_text = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text)
    return " ".join(ascii_text.split())


def _read_vector_url(url: str) -> gpd.GeoDataFrame:
    """Read GeoJSON from a URL; download to a temp file if GeoPandas cannot open it remotely."""
    try:
        return gpd.read_file(url)
    except Exception:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = resp.read()
        suffix = ".geojson" if "json" in url.lower() else ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(payload)
            tmp_path = tmp.name
        try:
            return gpd.read_file(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


def _http_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} fetching {url}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error fetching {url}: {e.reason}") from e
