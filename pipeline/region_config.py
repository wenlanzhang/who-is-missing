#!/usr/bin/env python3
"""
Region configuration for multi-country pipeline.

Loads config/regions.json and provides paths for each region.
Paths in config can be relative (to project root) or absolute.
"""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "regions.json"

# Top-level keys in regions.json that are not region codes
GLOBAL_KEYS = (
    "data_root",
    "poverty_source",
    "poverty_grdi",
    "clip_source",
    "footprints",
    "ghsl_smod",
    "ghsl_ucdb",
)
DEFAULT_POVERTY_SOURCE = "grdi"
VALID_POVERTY_SOURCES = ("grdi", "rwi")
DEFAULT_POVERTY_GRDI = PROJECT_ROOT / "data" / "raw" / "povmap-grdi-v1-10.tif"
DEFAULT_GHSL_SMOD = (
    PROJECT_ROOT / "data" / "raw" / "ghsl" / "GHS_SMOD_E2020_GLOBE_R2023A_4326_30ss_V2_0.tif"
)
DEFAULT_CLIP_SOURCE = "local"
VALID_CLIP_SOURCES = ("local", "osm", "geob")


def load_regions():
    """Load regions.json. Returns dict region_code -> config (includes global keys)."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Region config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        data = json.load(f)
    env_root = os.environ.get("RESIDENTIAL_DATA_ROOT")
    if env_root:
        data["data_root"] = env_root
    return data


def resolve_path(p: str, base: Path | None = None) -> Path:
    """Resolve path: absolute stays, relative is relative to base or PROJECT_ROOT."""
    path = Path(p)
    if not path.is_absolute():
        path = (base or PROJECT_ROOT) / path
    return path


def get_poverty_source(regions: dict | None = None) -> str:
    """Global default poverty source from config (`grdi` or `rwi`)."""
    if regions is None:
        regions = load_regions()
    src = str(regions.get("poverty_source") or DEFAULT_POVERTY_SOURCE).strip().lower()
    if src not in VALID_POVERTY_SOURCES:
        raise ValueError(
            f"Unknown poverty_source {src!r}. Use one of: {', '.join(VALID_POVERTY_SOURCES)}"
        )
    return src


def resolve_poverty_path(cfg: dict, source: str | None = None) -> Path | None:
    """
    Active poverty file for a region config.

    grdi → top-level poverty_grdi (project-root GeoTIFF)
    rwi  → per-region poverty (RWI CSV under data_root)
    """
    src = (source or cfg.get("poverty_source") or DEFAULT_POVERTY_SOURCE)
    src = str(src).strip().lower()
    if src not in VALID_POVERTY_SOURCES:
        raise ValueError(
            f"Unknown poverty source {src!r}. Use one of: {', '.join(VALID_POVERTY_SOURCES)}"
        )
    if src == "grdi":
        path = cfg.get("poverty_grdi")
        return Path(path) if path else None
    path = cfg.get("poverty")
    return Path(path) if path else None


def get_clip_source(regions: dict | None = None, region_cfg: dict | None = None) -> str:
    """Clip boundary source: `local` (default), `osm`, or `geob`."""
    raw = None
    if region_cfg and region_cfg.get("clip_source"):
        raw = region_cfg.get("clip_source")
    elif regions is not None:
        raw = regions.get("clip_source")
    elif region_cfg is None:
        regions = load_regions()
        raw = regions.get("clip_source")
    src = str(raw or DEFAULT_CLIP_SOURCE).strip().lower()
    if src not in VALID_CLIP_SOURCES:
        raise ValueError(
            f"Unknown clip_source {src!r}. Use one of: {', '.join(VALID_CLIP_SOURCES)}"
        )
    return src


def reject_legacy_phi(code: str) -> str:
    """Philippines is PHL only. Do not silently rewrite PHI."""
    raw = str(code or "").strip()
    if raw == "PHI" or raw.startswith("PHI_"):
        raise ValueError(
            f"{raw!r} is not a valid code. Philippines is PHL "
            "(e.g. ./run --region PHL or PHL_CagayandeOroCity)."
        )
    return raw


def get_region_config(region: str) -> dict:
    """Get config for a region key (city or data-donor). Resolves paths."""
    region = reject_legacy_phi(region)
    regions = load_regions()
    data_root = regions.get("data_root")
    if data_root:
        data_root = Path(data_root)
    if region not in regions or region in GLOBAL_KEYS:
        raise ValueError(f"Unknown region: {region}. Available: {list_regions()}")
    cfg = regions[region].copy()
    path_keys = ("worldpop", "meta", "poverty", "clip_shape", "pdc_raw_dir", "pdc_processed_csv")
    data_root_keys = ("poverty", "pdc_raw_dir")
    for key in path_keys:
        if key in cfg and cfg[key]:
            base = data_root if (data_root and key in data_root_keys) else PROJECT_ROOT
            cfg[key] = resolve_path(cfg[key], base)

    cfg["region_code"] = region
    cfg["poverty_source"] = get_poverty_source(regions)
    grdi = regions.get("poverty_grdi")
    cfg["poverty_grdi"] = resolve_path(grdi, PROJECT_ROOT) if grdi else DEFAULT_POVERTY_GRDI
    cfg["clip_source"] = get_clip_source(regions, cfg)
    return cfg


def require_city_region(code: str) -> str:
    """City codes only (KEN_Nairobi, MEX_MexicoCity). Not a country and not a donor key."""
    code = reject_legacy_phi(str(code).strip())
    if is_event_region(code):
        raise ValueError(
            f"{code!r} is a data donor (PDC/WorldPop paths), not a city run. "
            f"Cities: ./run --region {code}   Footprint: ./run --footprint {code}"
        )
    if code in list_cities(in_sample_only=False):
        return code
    if "_" not in code:
        cities = list_cities(code)
        if cities:
            raise ValueError(
                f"{code!r} is a country code, not a city. "
                f"One city: --region {cities[0]}   All cities in {code}: ./run --region {code}"
            )
    raise ValueError(f"Unknown city {code!r}. Available: {list_cities()}")


def _attach_ghsl(cfg: dict, regions: dict) -> dict:
    data_root = Path(regions["data_root"]) if regions.get("data_root") else None
    smod = cfg.get("ghsl_smod") or regions.get("ghsl_smod")
    cfg["ghsl_smod"] = resolve_path(smod, PROJECT_ROOT) if smod else DEFAULT_GHSL_SMOD
    ucdb = cfg.get("ghsl_ucdb") or regions.get("ghsl_ucdb")
    if not ucdb:
        cfg["ghsl_ucdb"] = None
        return cfg
    if Path(ucdb).is_absolute():
        cfg["ghsl_ucdb"] = Path(ucdb)
    elif str(ucdb).startswith("data/") or str(ucdb).startswith("outputs/"):
        cfg["ghsl_ucdb"] = resolve_path(ucdb, PROJECT_ROOT)
    else:
        cfg["ghsl_ucdb"] = resolve_path(ucdb, data_root or PROJECT_ROOT)
    return cfg


def list_footprints() -> list:
    """
    Every country with a city run or an event-donor key.

    Not a fixed four-country set. Adding a city country is enough for
    ``./run --footprint COUNTRY`` once a donor region exists.
    """
    regions = load_regions()
    ordered: list[str] = []

    def add(code: str) -> None:
        c = str(code or "").strip()
        if c and c not in ordered and c not in GLOBAL_KEYS:
            ordered.append(c)

    for c in (regions.get("footprints") or {}):
        add(c)
    for c in list_cities():
        add(country_prefix(c))
    for c in list_event_regions():
        add(c)
    return ordered


def _city_boundary_path(city_code: str, cfg: dict) -> Path | None:
    clip = cfg.get("clip_shape")
    if clip:
        p = Path(clip)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            return p
    cache = PROJECT_ROOT / "data" / "raw" / "boundaries" / "cache" / "geob" / f"{city_code}.gpkg"
    if cache.exists():
        return cache
    return None


def _synthesize_footprint(code: str, regions: dict) -> dict | None:
    """Build a footprint entry from the unclipped extract or the first city in that country."""
    cities = list_cities(code)
    donor_key = None
    if code in regions and code not in GLOBAL_KEYS:
        donor_key = code
    elif cities:
        donor_key = cities[0]
    if donor_key is None:
        return None
    donor = regions[donor_key]
    city_boundaries = {}
    for city_code in cities:
        city_cfg = regions[city_code]
        label = city_cfg.get("city_label") or city_code
        path = _city_boundary_path(city_code, city_cfg)
        if path is not None:
            city_boundaries[label] = str(path)
    iso = donor.get("clip_geob_iso3") or code
    hh = int(donor.get("pdc_ref_hour") or 0)
    meta = donor.get("meta") or f"data/baselines/{iso}/fb_baseline_median_h{hh:02d}.gpkg"
    return {
        "name": donor.get("name") or f"{country_display_name(code)} (Meta event footprint)",
        "iso3": iso,
        "country_name": country_display_name(code),
        "pdc_ref_hour": hh,
        "worldpop": donor.get("worldpop"),
        "meta": meta,
        "poverty": donor.get("poverty"),
        "pdc_raw_dir": donor.get("pdc_raw_dir"),
        "pdc_processed_csv": donor.get("pdc_processed_csv"),
        "pdc_use_baseline_column": donor.get("pdc_use_baseline_column"),
        "city_boundaries": city_boundaries,
        "lon_range": donor.get("lon_range"),
        "lat_range": donor.get("lat_range"),
    }


def _resolve_footprint_paths(cfg: dict, regions: dict) -> dict:
    data_root = Path(regions["data_root"]) if regions.get("data_root") else None
    path_keys = (
        "worldpop",
        "meta",
        "poverty",
        "pdc_raw_dir",
        "pdc_processed_csv",
    )
    data_root_keys = ("poverty", "pdc_raw_dir")
    for key in path_keys:
        if key in cfg and cfg[key]:
            base = data_root if (data_root and key in data_root_keys) else PROJECT_ROOT
            cfg[key] = resolve_path(cfg[key], base)
    resolved_cities = {}
    for name, path in (cfg.get("city_boundaries") or {}).items():
        if path:
            resolved_cities[name] = resolve_path(path, PROJECT_ROOT)
    cfg["city_boundaries"] = resolved_cities
    cfg["poverty_source"] = get_poverty_source(regions)
    grdi = regions.get("poverty_grdi")
    cfg["poverty_grdi"] = resolve_path(grdi, PROJECT_ROOT) if grdi else DEFAULT_POVERTY_GRDI
    return _attach_ghsl(cfg, regions)


def get_footprint_config(footprint: str) -> dict:
    """Meta event AOI as published. Distinct from city --region clips."""
    footprint = reject_legacy_phi(footprint)
    regions = load_regions()
    footprints = regions.get("footprints") or {}
    if footprint in footprints:
        cfg = footprints[footprint].copy()
    else:
        cfg = _synthesize_footprint(footprint, regions)
        if cfg is None:
            raise ValueError(f"Unknown footprint: {footprint}. Available: {list_footprints()}")
    cfg["code"] = footprint
    return _resolve_footprint_paths(cfg, regions)


def get_aligned_parquet(footprint: str) -> Path:
    return _footprint_parquet(footprint, "_aligned.parquet")


def _footprint_parquet(footprint: str, suffix: str) -> Path:
    """Prefer data/processed/footprints/; fall back to a legacy file at processed root."""
    new = PROJECT_ROOT / "data" / "processed" / "footprints" / f"{footprint}{suffix}"
    old = PROJECT_ROOT / "data" / "processed" / f"{footprint}{suffix}"
    if new.exists() or not old.exists():
        return new
    return old


def get_footprint_output_dir(footprint: str, step: str) -> Path:
    """Inspection outputs; does not overwrite city folders under outputs/city/{COUNTRY}/{city}/."""
    return PROJECT_ROOT / "outputs" / "footprints" / footprint / step


def add_footprint_arg(parser) -> None:
    """Attach --footprint COUNTRY to an analysis script parser."""
    parser.add_argument(
        "--footprint",
        type=str,
        default=None,
        help="Event-footprint ISO3 (PHL, KEN, MEX, IDN, LKA, COL, ECU, ZAF). "
        "Writes CSVs/figures under outputs/figure/footprints/{CODE}/, not city folders.",
    )


def get_meta_baseline_path(cfg: dict, code: str, ref_hour=None, baseline_method=None) -> Path:
    """Resolve a Meta GPKG for a city or a footprint (optional shift / n_baseline tag)."""
    hh = int(ref_hour if ref_hour is not None else cfg.get("pdc_ref_hour") or 0)
    country_dir = cfg.get("iso3") or country_prefix(code)
    shared = PROJECT_ROOT / "data" / "baselines" / country_dir / f"fb_baseline_median_h{hh:02d}.gpkg"
    tagged_city = (
        PROJECT_ROOT / "data" / "baselines" / country_dir / f"fb_baseline_median_h{hh:02d}_{baseline_method}.gpkg"
        if baseline_method
        else None
    )
    tagged_fp = (
        PROJECT_ROOT
        / "outputs"
        / "footprints"
        / code
        / "meta"
        / f"fb_baseline_median_h{hh:02d}_{baseline_method}.gpkg"
        if baseline_method
        else None
    )
    if baseline_method in ("n_baseline", "shift"):
        for p in (tagged_fp, tagged_city):
            if p is not None and p.exists():
                return p
        if cfg.get("meta"):
            alt = Path(cfg["meta"]).with_name(f"fb_baseline_median_h{hh:02d}_{baseline_method}.gpkg")
            if alt.exists():
                return alt
        if shared.exists():
            return shared
        return tagged_fp or tagged_city or shared
    if cfg.get("meta"):
        cfg_meta = Path(cfg["meta"])
        if cfg_meta.exists():
            return cfg_meta
    if shared.exists():
        return shared
    if ref_hour is not None:
        return shared
    return Path(cfg["meta"]) if cfg.get("meta") else shared


def footprints_with_geographies() -> list:
    return [c for c in list_footprints() if get_geographies_parquet(c).exists()]


# Event extracts are not a city folder. Mexico City is MEX_MexicoCity → MexicoCity.
_CITY_PRODUCT = "city"
_RESERVED_LAYOUT = {
    "city",
    "footprints",
    "cross-city",
    "paper",
    "_archive",
    "_snapshots",
    "qa",
    "meta",
    "geographies",
}
_IMAGE_EXTS = {".png", ".pdf", ".svg", ".jpg", ".jpeg"}
_GEO_EXTS = {".gpkg", ".shp", ".geojson", ".tif", ".tiff"}


def _city_product_root(kind: str) -> Path:
    """outputs/city, figure/city, or data/processed/city."""
    if kind == "processed":
        return PROJECT_ROOT / "data" / "processed" / _CITY_PRODUCT
    return PROJECT_ROOT / kind / _CITY_PRODUCT


def layout_parts(code: str) -> tuple[str, str]:
    """(country, city) for the city-product folder layout."""
    country = country_prefix(code)
    if is_event_region(code):
        raise ValueError(
            f"{code!r} is a data donor, not a city folder. "
            f"Use ./run --region {code} for cities or ./run --footprint {code}."
        )
    if "_" in code:
        return country, code.split("_", 1)[1]
    return country, code


def region_from_layout(country: str, place: str) -> str:
    """Inverse of layout_parts."""
    if place == country:
        return country
    return f"{country}_{place}"


def csv_dir(region: str, step: str | None = None) -> Path:
    country, place = layout_parts(region)
    p = _city_product_root("outputs") / country / place
    return p / step if step else p


def figure_dir(region: str, step: str | None = None) -> Path:
    country, place = layout_parts(region)
    p = _city_product_root("figure") / country / place
    return p / step if step else p


def geo_dir(region: str, step: str | None = None) -> Path:
    country, place = layout_parts(region)
    p = _city_product_root("processed") / country / place
    return p / step if step else p


_COUNTRY_LABEL = {
    "PHL": "Philippines",
    "KEN": "Kenya",
    "MEX": "Mexico",
    "IDN": "Indonesia",
    "LKA": "Sri Lanka",
    "COL": "Colombia",
    "ECU": "Ecuador",
    "ZAF": "South Africa",
}

# The ArmyRose palette now lives in analysis/R/theme_armyrose.R, which is the only
# place that draws anything. It was duplicated here for the retired Python figure
# steps.


def country_display_name(code: str) -> str:
    return _COUNTRY_LABEL.get(country_prefix(code), country_prefix(code))


def display_label(code: str, cfg: dict | None = None) -> str:
    """City name, or '{name} (full)' for an unclipped extract."""
    if cfg is None:
        cfg = get_region_config(code)
    label = cfg.get("city_label") or cfg.get("map_bbox_label") or cfg.get("name") or code
    if is_event_region(code, cfg):
        return f"{label} (full)"
    return label


def list_event_regions() -> list:
    regions = load_regions()
    return [k for k in list_regions() if is_event_region(k, regions[k])]


def find_artifact(region: str, step: str, filename: str) -> Path | None:
    """City umbrella first, then pre-umbrella country/city, then outputs/{REGION}/{step}."""
    ext = Path(filename).suffix.lower()
    country, place = layout_parts(region)
    if ext in _GEO_EXTS:
        candidates = [
            geo_dir(region, step) / filename,
            PROJECT_ROOT / "data" / "processed" / country / place / step / filename,
        ]
    elif ext in _IMAGE_EXTS:
        candidates = [
            figure_dir(region, step) / filename,
            PROJECT_ROOT / "figure" / country / place / step / filename,
        ]
    else:
        candidates = [
            csv_dir(region, step) / filename,
            PROJECT_ROOT / "outputs" / country / place / step / filename,
        ]
    candidates.append(PROJECT_ROOT / "outputs" / region / step / filename)
    for path in candidates:
        if path.exists():
            return path
    return None


def baseline_path(region: str, hour: int, method: str | None = None) -> Path:
    """Shared Meta baseline GPKG for a country (optional _n_baseline / _shift tag)."""
    country, _ = layout_parts(region)
    name = f"fb_baseline_median_h{int(hour):02d}"
    if method in ("n_baseline", "shift"):
        name = f"{name}_{method}"
    return PROJECT_ROOT / "data" / "baselines" / country / f"{name}.gpkg"


class StepPaths:
    """One step's CSV / figure / GPKG dirs. ``paths / 'a.png'`` routes by suffix."""

    def __init__(self, region: str, step: str, *, product: str = "city"):
        self.region = region
        self.step = step
        self.product = product
        if product == "footprint":
            self.csv_dir = PROJECT_ROOT / "outputs" / "footprints" / region / step
            self.fig_dir = PROJECT_ROOT / "figure" / "footprints" / region / step
            self.geo_dir = PROJECT_ROOT / "data" / "processed" / "footprints" / region / step
        else:
            self.csv_dir = csv_dir(region, step)
            self.fig_dir = figure_dir(region, step)
            self.geo_dir = geo_dir(region, step)

    def mkdir(self, parents: bool = True, exist_ok: bool = True):
        for d in (self.csv_dir, self.fig_dir, self.geo_dir):
            d.mkdir(parents=parents, exist_ok=exist_ok)
        return self

    def __truediv__(self, name):
        n = str(name)
        ext = Path(n).suffix.lower()
        if ext in _IMAGE_EXTS:
            return self.fig_dir / n
        if ext in _GEO_EXTS:
            return self.geo_dir / n
        return self.csv_dir / n

    def __fspath__(self):
        return str(self.csv_dir)

    def __str__(self):
        return str(self.csv_dir)


def step_paths(region: str, step: str) -> StepPaths:
    paths = StepPaths(region, step)
    paths.mkdir()
    return paths


def region_from_artifact_path(path: Path | str | None) -> str | None:
    """Recover region code from outputs/city|figure/city|data/processed/city paths."""
    if path is None:
        return None
    parts = Path(path).resolve().parts
    for root in ("outputs", "figure", "processed", "baselines"):
        if root not in parts:
            continue
        i = parts.index(root)
        if root == "baselines":
            return None
        if i + 1 >= len(parts):
            continue
        nxt = parts[i + 1]
        if nxt == _CITY_PRODUCT and i + 3 < len(parts):
            country, place = parts[i + 2], parts[i + 3]
        elif nxt == "footprints" and i + 2 < len(parts):
            code = parts[i + 2]
            if code in ("qa", "meta"):
                continue
            return code
        elif nxt in _RESERVED_LAYOUT:
            continue
        elif i + 2 < len(parts):
            country, place = parts[i + 1], parts[i + 2]
        else:
            continue
        if place in ("01", "02", "cross-city") or place in _RESERVED_LAYOUT:
            continue
        return region_from_layout(country, place)
    return None


def list_regions() -> list:
    """List available region codes (cities and event extracts)."""
    return [k for k in load_regions().keys() if k not in GLOBAL_KEYS]


def country_prefix(code: str) -> str:
    """PHL_CagayandeOroCity → PHL; MEX → MEX."""
    return code.split("_", 1)[0]


def is_event_region(code: str, cfg: dict | None = None) -> bool:
    """Unclipped Meta extract: ISO3-only region key with clip_shape unset."""
    if "_" in code:
        return False
    regions = load_regions()
    if code not in regions or code in GLOBAL_KEYS:
        return False
    if cfg is None:
        cfg = regions[code]
    return not bool(cfg.get("clip_shape"))


def is_in_sample(code: str, cfg: dict | None = None) -> bool:
    """False when regions.json sets in_sample to false (kept in config, not in the study sample)."""
    if cfg is None:
        regions = load_regions()
        cfg = regions.get(code) or {}
    return cfg.get("in_sample", True) is not False


def list_cities(country: str | None = None, *, in_sample_only: bool = True) -> list:
    """Selected study cities, optionally limited to one country prefix (PHL, KEN, MEX, …)."""
    regions = load_regions()
    if country is not None:
        country = reject_legacy_phi(country)
    out = []
    for k in list_regions():
        if is_event_region(k, regions[k]):
            continue
        if country is not None and country_prefix(k) != country:
            continue
        if in_sample_only and not is_in_sample(k, regions[k]):
            continue
        out.append(k)
    return out


def expand_region_to_list(region_or_prefix: str, *, event: bool = False) -> list:
    """
    Expand a country code to the runs to execute.

    Default: all selected cities in that country
      MEX → Mexico City, Puebla, León
      IDN → Medan, Banda Aceh
      PHL → all Philippines cities

    event=True is retired (use ./run --footprint COUNTRY).

    A full city code still maps to itself so other scripts can resume one city.
    Comma-separated lists are not accepted; pass one country code.
    """
    raw = reject_legacy_phi(str(region_or_prefix).strip())
    if "," in raw:
        raise ValueError(
            "Pass one country code (e.g. MEX or IDN), not a comma-separated city list."
        )
    if not raw:
        return []

    if event:
        raise ValueError(
            "Unclipped city-pipeline extracts are retired. "
            f"Use ./run --footprint {raw} for the Meta event AOI, "
            f"or ./run --region {raw} for selected cities."
        )

    regions = load_regions()
    keys = list_regions()

    cities = list_cities(raw)
    if cities:
        return cities
    if raw in keys:
        return [raw]
    return []
