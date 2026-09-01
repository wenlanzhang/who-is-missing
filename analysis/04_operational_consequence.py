#!/usr/bin/env python3
"""
A4 — What the coverage gap costs an operational user.

A2 establishes that Meta's decision to publish a tile is deprivation-graded. A3
shows the gap is invisible to any population-weighted statistic. This script asks
the question a crisis responder actually faces: if you plan from the Meta
baseline, which places do you never see, and how much worse is your targeting?

Three exercises.

  1. Blind spots. Among the most deprived tiles in each city, what share carry no
     Meta value at all, and how much land and population do they hold?

  2. Clustering. Are blind spots scattered noise or contiguous zones? Scattered
     missingness averages out over a district; contiguous missingness removes a
     whole neighbourhood from the map. Measured with join-count / Moran's I on
     the publication indicator.

  3. Targeting. A responder can reach K tiles and wants the most deprived ones.
     Compare an oracle that ranks all eligible tiles by GRDI against a responder
     restricted to tiles Meta published. Recall of the true top-K is the loss
     attributable purely to the data, not to the decision rule - both rank on the
     same GRDI, so the only difference is which tiles are selectable.

Outputs
  outputs/analysis/A4_blind_spots.csv
  outputs/analysis/A4_spatial_clustering.csv
  outputs/analysis/A4_targeting_recall.csv

Usage:
  python analysis/04_operational_consequence.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "analysis" / "panel" / "tile_panel.parquet"
OUT = ROOT / "outputs" / "analysis"

KNN_K = 8

# esda.Moran draws its reference distribution from numpy's global RNG and takes
# no seed argument, so an unseeded run gives a different p_sim every time — the
# statistic is stable but the permutation p-value moves (General Santos flipped
# between 0.001 and 0.005 across runs). Seeding makes A4_spatial_clustering.csv
# reproducible; it does not change any Moran's I.
SEED = 20260829


def load():
    d = pd.read_parquet(PANEL)
    return d[d.in_eligible].reset_index(drop=True)


def blind_spots(d: pd.DataFrame) -> pd.DataFrame:
    """Coverage of the most deprived tiles, city by city."""
    rows = []
    for city, g in d.groupby("city"):
        top = g[g.grdi_decile >= 9]          # most deprived fifth of the city
        bot = g[g.grdi_decile <= 2]
        rows.append({
            "country": g.country.iloc[0], "city": city,
            "n_eligible": len(g), "C_c": g.published.mean(),
            "cov_least_deprived": bot.published.mean(),
            "cov_most_deprived": top.published.mean(),
            "coverage_gap_pp": 100 * (bot.published.mean() - top.published.mean()),
            "blind_tiles_top20pct": int((top.published == 0).sum()),
            "blind_km2_top20pct": float(top.loc[top.published == 0, "area_km2"].sum()),
            "blind_pop_top20pct": float(top.loc[top.published == 0, "worldpop_count"].sum()),
        })
    out = pd.DataFrame(rows).sort_values("coverage_gap_pp", ascending=False)

    top, bot = d[d.grdi_decile >= 9], d[d.grdi_decile <= 2]
    print(f"  Coverage of the least deprived 20% of tiles : {bot.published.mean():.1%}")
    print(f"  Coverage of the most  deprived 20% of tiles : {top.published.mean():.1%}")
    print(f"  Gap                                         : "
          f"{100*(bot.published.mean()-top.published.mean()):.1f} pp")
    blind = (d.published == 0).sum()
    blind_km2 = d.loc[d.published == 0, "area_km2"].sum()
    print(f"\n  Tiles with no Meta value : {blind:,} of {len(d):,} "
          f"({blind/len(d):.1%}) = {blind_km2:,.0f} km2 of mapped urban area")
    print(f"  Of those, {(d.loc[d.published==0,'grdi_decile']>=9).mean():.1%} "
          f"fall in the most deprived fifth of their city")
    print()
    print(out.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return out


def spatial_clustering(d: pd.DataFrame) -> pd.DataFrame:
    """Is the missingness contiguous? Moran's I on the publication indicator."""
    from libpysal.weights import KNN
    from esda.moran import Moran

    np.random.seed(SEED)
    rows = []
    for city, g in d.groupby("city"):
        if len(g) <= KNN_K or g.published.nunique() < 2:
            rows.append({"city": city, "n": len(g), "morans_I": np.nan,
                         "p": np.nan, "note": "no variation or too few tiles"})
            continue
        w = KNN.from_array(g[["lon", "lat"]].to_numpy(), k=KNN_K)
        w.transform = "r"
        mi = Moran(g.published.to_numpy().astype(float), w, permutations=999)
        rows.append({"city": city, "n": len(g), "morans_I": mi.I,
                     "p": mi.p_sim, "note": ""})
    out = pd.DataFrame(rows)
    est = out.dropna(subset=["morans_I"])
    print(f"  Moran's I on published (0/1), KNN k={KNN_K}, 999 permutations")
    print(f"  median I = {est.morans_I.median():.3f}; "
          f"{(est.p < 0.05).sum()}/{len(est)} cities significant at 5%")
    print("  I > 0 means unpublished tiles sit next to unpublished tiles: the")
    print("  blind spots are contiguous zones, not scattered dropout.\n")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return out


def targeting(d: pd.DataFrame) -> pd.DataFrame:
    """Recall of the truly most-deprived tiles when restricted to Meta's map.

    Both the oracle and the Meta-constrained planner rank by the same GRDI, so
    any difference is caused by tiles being absent from Meta's product rather
    than by a worse targeting rule.
    """
    rows = []
    for frac in (0.05, 0.10, 0.20, 0.30):
        rec, pop_rec, per_city = [], [], []
        for city, g in d.groupby("city"):
            k = max(1, int(round(frac * len(g))))
            truth = set(g.nlargest(k, "poverty_mean").quadkey)
            avail = g[g.published == 1]
            got = set(avail.nlargest(min(k, len(avail)), "poverty_mean").quadkey)
            r = len(truth & got) / len(truth)
            wp_truth = g[g.quadkey.isin(truth)].worldpop_count.sum()
            wp_got = g[g.quadkey.isin(truth & got)].worldpop_count.sum()
            rec.append(r)
            pop_rec.append(wp_got / wp_truth if wp_truth > 0 else np.nan)
            per_city.append((city, r))
        rows.append({
            "budget_frac_of_tiles": frac,
            "tile_recall_mean": float(np.mean(rec)),
            "tile_recall_median": float(np.median(rec)),
            "pop_recall_mean": float(np.nanmean(pop_rec)),
            "worst_city": min(per_city, key=lambda t: t[1])[0],
            "worst_recall": min(per_city, key=lambda t: t[1])[1],
        })
        r = rows[-1]
        print(f"  reach top {int(frac*100):>2}% of tiles:  place recall "
              f"{r['tile_recall_mean']:.1%} (median {r['tile_recall_median']:.1%}), "
              f"population recall {r['pop_recall_mean']:.1%}"
              f"   worst: {r['worst_city']} {r['worst_recall']:.1%}")
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    print(f"Eligible tiles {len(d):,} in {d.city.nunique()} cities\n")

    print("=== 1. Blind spots: coverage of the most deprived places ===\n")
    bs = blind_spots(d)
    bs.to_csv(OUT / "A4_blind_spots.csv", index=False)

    print("\n\n=== 2. Are the blind spots contiguous? ===\n")
    sc = spatial_clustering(d)
    sc.to_csv(OUT / "A4_spatial_clustering.csv", index=False)

    print("\n\n=== 3. Targeting the most deprived places from the Meta map ===\n")
    tg = targeting(d)
    tg.to_csv(OUT / "A4_targeting_recall.csv", index=False)

    print(f"\nWrote 3 tables to {OUT}")


if __name__ == "__main__":
    main()
