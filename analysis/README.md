# Does Meta's crisis population baseline under-represent deprived places?

Full method, results, robustness, and the things that did not work.

**Run order**

```bash
python analysis/01_build_panel.py             # both tile panels -> analysis/panel/
python analysis/02_selection_models.py        # extensive margin + robustness ladder
python analysis/03_two_margin_decomposition.py  # reconciles the null
python analysis/04_operational_consequence.py   # blind spots, clustering, targeting
python analysis/05_city_models.py             # per-city models (feeds F0b, F1b)
python analysis/06_refhour_robustness.py      # baseline-hour check, and why not RWI
python analysis/07_censoring_bounds.py        # bounds, imputation, Tobit, placebo
python analysis/08_unreported_burden.py       # how many people / km2 / places are missing
python analysis/09_sensitivity.py             # floor, jackknife, MAUP, form
python analysis/12_aoi_check.py               # suppression vs event-AOI edge
Rscript analysis/10_figures_main.R            # F0, F0b, F1, F1b, F2-F5, F10
Rscript analysis/11_figures_robustness.R      # F6-F9
```

Tables → `outputs/analysis/`, figures → `figure/analysis/`. Whole chain ≈ 1 minute.

Step 01 writes **two** panels in one pass: `tile_panel.parquet` (the 18-city study
sample, read by 02–07 and 09) and `tile_panel_all.parquet` (all 21 cities, read by 09's
out-of-sample row and by 12). Every derived column is computed within city, so the first
is exactly a row-subset of the second — building them together is what stops the
out-of-sample comparison in §5.5(c) from contrasting two different builds.

Steps 06 and 12 additionally need the raw Meta PDC / RWI files under `data_root`
(`config/regions.json`, or the `RESIDENTIAL_DATA_ROOT` environment variable). Without them
06 skips its RWI section and leaves the existing table in place, and 12 stops with a
message rather than a traceback.

---

## 1. The hypothesis

> Higher deprivation → lower Facebook penetration → Meta under-counts people relative to
> WorldPop → weaker agreement between Meta and WorldPop in deprived areas.

Both sources are normalised to within-city shares first, so the comparison is about
**spatial distribution**, not totals. Deprivation and population are spatially structured,
so spatial models are used throughout.

The original test was `log(meta_share / worldpop_share) ~ GRDI`, with a spatial error model
for spatial confounding. It returned nothing usable. Re-estimated per city on the current
sample (`A3_intensive_margin_by_city.csv`): of 17 estimable cities, τ ranges **−0.53 to
+1.59 with a median of +0.09**. Pooled, it is a flat zero.

Read the *spread*, not the significance counts. Taken at face value the 17 cities split 6
significantly negative / 4 positive / 7 null, but 10 of them have fewer than five zoom-10
clusters, which is too few for a cluster-robust variance to mean anything — Medan returns
`se = 0.004, p = 0.0` off **two** clusters, which is a degenerate fit, not a precise one.
Among the 7 cities with ≥ 5 clusters the split is 4 negative / 0 positive / 3 null. The
table now carries `n_clusters` and flags the unreliable rows. Either way the conclusion is
the same and it is the only one this table supports: **the estimand is mixed across cities
and centred on nothing.**

**The hypothesis was right. The estimand was wrong.** What follows is why, and what
replaced it.

---

## 2. Data and measures

| | |
|---|---|
| **Unit** | zoom-14 quadkey tile (~2.4 km at the equator; real area computed per tile, 4.8–5.8 km²) |
| **Meta** | PDC "Population During Crisis" baseline, `n_baseline`, median over the pre-crisis window at one hour per country (near local evening) |
| **WorldPop** | country raster, zonal sum to tile |
| **Deprivation** | GRDI v1.10, zonal mean to tile. Higher = more deprived |
| **Urbanicity** | GHSL SMOD 2020 settlement class (urban centre / town / rural / water) |
| **Sample** | 18 cities, 8 countries, **4,999 eligible tiles**, 52.0M people |

**Eligible grid.** A tile is eligible if WorldPop places people there (`> 0`) and GRDI is
observed. This is the support on which a Meta-vs-WorldPop comparison is even defined. It is
built by `pipeline/01b`, which reconstructs the city grid from the clip polygon rather than
from Meta's published tiles — so suppressed tiles are present and flagged.

**Censoring.** The minimum published Meta value across all extracts is exactly **10.0**.
Suppressed tiles are therefore not missing at random and not zero: they are known to lie in
**[0, 10)**. This fact is what makes §5.3 possible.

**Suppression vs the event AOI — and why Kisumu was dropped** (`12_aoi_check.py`). The outcome is "Meta has no
value for this tile", which could mean two different things: the tile was *suppressed*
(inside the extract, below threshold) or it was *never in the extract* (outside the
published event AOI). Only the first is a coverage decision; scoring the second as
deprivation bias would be a serious error.

Checking the raw PDC CSVs settles it for the sample as a whole. The quadkey set **varies
across timestamps** — the Kenya extract ranges from 1,908 to 3,402 rows across 64 files, and
the union already exceeds any single file — so tiles enter and leave as they cross the
threshold. Absence is suppression, not a fixed geographic AOI.

**Kisumu is the exception.** 303 of its 434 eligible tiles never appear in *any* of the 64
timestamps, and those tiles hold a **median of 1,634 WorldPop residents**. A tile with 1,600
people clears a 10-user threshold easily, so this is the edge of the event AOI, not a
coverage decision. Every other city's never-present tiles have a median WorldPop under 800
and most under 200, consistent with genuine sub-threshold counts. Kisumu is therefore
excluded, leaving **18 cities**. Dropping it does not move the result (OR 0.095 → 0.092);
it removes a contaminated observation, not an inconvenient one.

Every other city checks out. The next-highest median WorldPop among never-present tiles is
Mombasa at 755 across just 8 tiles; Mexico City's 31 absent tiles hold a median of **2
people** and Puebla's 14 hold a median of **1**. Diagnostic:
`outputs/analysis/A10_aoi_diagnostic.csv`.

*(One trap worth recording: quadkeys must be read as strings. Quadkeys in the Americas begin
with "0", and pandas infers int64 and silently strips it, which makes Mexico and Colombia
appear to be entirely outside their own event AOI.)*

**Two key correlations**, both within-city, both central to the design:

- `corr(GRDI, log WorldPop) = −0.80` — deprived tiles are also sparse tiles. Any coverage
  result must survive controlling for population, or it is a density story.
- `corr(own GRDI, neighbours' GRDI) = 0.94` — deprivation is smooth in space. This is why
  the direct/indirect split in a spatial model is not identified here (§5.1).

---

## 3. Why the original test could not have worked

Three diagnoses, in order of importance. Only the first is fatal.

### 3.1 Selection on the outcome of the process being tested

`pipeline/01` starts from **published** Meta tiles. A tile Meta suppressed cannot appear in
any downstream table. But whether Meta publishes a tile is the result of its user-density
and privacy thresholds — the very mechanism the hypothesis is about. The test ran on the
survivors of the process it was trying to detect.

At the whole-sample level:

| Margin | Sample | Per +1 SD deprivation | p |
|---|---|---|---|
| Intensive: `log(meta_share/wp_share)` | published tiles (n=3,209) | **−0.007** (se 0.081) | 0.93 |
| Extensive: `P(tile published)` | all eligible tiles (n=4,999) | **−26.2 pp** (se 1.8) | 6e-49 |

### 3.2 It is *not* a normalisation artefact — checked and ruled out

The obvious suspicion is that renormalising shares over published tiles launders the
missing mass away. `03_two_margin_decomposition.py` re-estimates τ with WorldPop shares
renormalised over the **full eligible grid** instead. The estimate is identical to four
decimal places: with city fixed effects the two denominators differ only by a per-city
constant that the fixed effect absorbs. **Renormalising cannot rescue this.**

### 3.3 It is not hiding in the population weights either

The representation ratio (Meta's share of a city ÷ WorldPop's share) is ≈1.00 in *every*
deprivation decile. Among the people Meta counts, it allocates them essentially correctly.

Which denominator matters, so both are reported in `A3_representation_ratio_by_decile.csv`
and the figure (`F3`) uses the second:

| Normalisation | Range of R across deciles |
|---|---|
| over **published** tiles (what the pipeline does) | 0.97–1.79 |
| over the **full eligible grid** (the correct view) | **0.98–1.28** |

The published-tile denominator drifts up at the deprived end because it renormalises over a
shrinking set of tiles. The eligible-grid figure is the one to quote.

**Conclusion.** Meta's baseline is **population-representative but geographically
censored**, and the censoring is deprivation-selective. Any population-weighted estimand is
null more or less by construction. The bias lives in the space of *places*, and the original
design had no estimand in that space.

---

## 4. The main analysis

**Estimand.** `P(published_i = 1 | GRDI_i, population_i, urbanicity_i, city_i)`

**Why this one.** It is the only quantity here that is fully observed. Whether a tile was
published is known for every tile — nothing imputed, nothing censored, no conditioning on a
selected sample. §5.3 shows that every alternative estimand on this data is either null or
not identified.

**Specification.** Logit, city fixed effects, standard errors clustered on **zoom-10 quadkey
cells** (~39 km across at the equator, less at higher latitude; a median of 51 city tiles
each, 75 clusters). Clustering is needed because publication is strongly spatially
autocorrelated — Moran's I on the 0/1 indicator has median 0.57 — so treating 4,700 tiles as
independent would overstate precision. Zoom 10 is chosen because it is large enough to
absorb that dependence while leaving enough clusters for the cluster-robust variance
estimator to behave; zoom-8 (27 clusters) and city-level (14) fall below the usual
rule of thumb. The choice is not load-bearing: see the bandwidth ladder below. All predictors are standardised **within city**, so every comparison is
between two tiles in the same city and no cross-city difference in wealth, size or GRDI
level can drive the estimate. Estimation sample is **4,700 tiles in 14 cities, 75 clusters**
— four cities (Colombo, Medan, Banda Aceh, Barranquilla) publish 100% of eligible tiles, so
their fixed effect perfectly predicts the outcome and they drop out exactly as a conditional
FE logit would drop them.

### Results — the confounder ladder

OR is per +1 SD of within-city GRDI.

| Model | Adds | OR | 95% CI | p |
|---|---|---|---|---|
| M1 | GRDI only | 0.023 | 0.013–0.039 | 2e-44 |
| M2 | + log WorldPop | 0.092 | 0.058–0.147 | 1e-23 |
| **M3** | **+ log WorldPop²** | **0.123** | **0.075–0.200** | **3e-17** |
| M4 | + GHSL settlement class | 0.125 | 0.076–0.206 | 2e-16 |
| M7 | GRDI orthogonalised to density & settlement | 0.362 | 0.284–0.462 | 2e-16 |
| SLX | total (own + neighbourhood) effect | 0.077 | 0.044–0.135 | <1e-9 |
| LPM | linear probability, M2 spec | −8.3 pp | | 1e-3 |

**M3 is the headline specification.** A tile one SD more deprived than its city average has
about **one-eighth the odds** of appearing in Meta's product, holding its population and
population squared fixed.

Settlement class (M4) is reported as a robustness row rather than in the headline. It is
jointly significant (χ²(3) = 33.5) but moves the deprivation odds ratio by 0.002, and its
"Urban centre" category is 100% published — the same complete-separation problem that
excludes four cities from the sample. Keeping a perfectly separated dummy while dropping
perfectly separated cities would be inconsistent. Where settlement type does real work is
§5.5(f), where the model is estimated *within* each class.

**Dose-response (`F1`).** On the 4,700-tile estimation sample, Meta publishes **100%** of
tiles in the least deprived decile and **16%** in the most deprived. Holding population
fixed: **99% → 46%**.

*(Both `F1` lines come from `A2_dose_response_adjusted.csv`, the 14-city estimation sample,
so the observed and adjusted curves describe the same tiles. Across all 18 cities the raw
gradient is 100% → **21%** (`A2_dose_response_grdi_decile.csv`, the descriptive table).
Quote 21% for the full sample, or 16% against the adjusted line — never 16% against 21%,
which was what the earlier version of this figure plotted.)*

*(A note on "holding population fixed". `z_logWP` is standardised, so its mean is 0 but the
mean of its **square** is 1.0. Setting both to their own means — the usual "predict at the
covariate means" recipe — describes a tile whose population is average and whose squared
deviation is 1, which is no tile at all, and it lifts the most deprived decile by 10 points.
The squared term is therefore set to the square of the held value, i.e. 0. This is also what
makes this line and `F1b`'s pooled curve the same object.)*

**`F1b` is the bridge from one city to all of them.** The same model fitted separately to
each of the 14 cities, drawn on a common axis, with the pooled fixed-effects curve on top.
It exists because the talk otherwise jumps from a single-city figure to a pooled estimate
with no visible connection.

It also settles a question `F2` raises: if every city agrees, why do the curves look so
different? Because **the odds ratios are homogeneous and the starting points are not.** The
median city OR is **0.132** against a pooled **0.123**, and 8 of 14 cities sit above the
pooled value — the slope is close to common. What differs is what that slope does on the
probability scale. An eight-fold reduction in odds applied to a city already publishing
99.9% of its tiles is invisible; the same reduction applied to a city at 60% collapses it.
Puebla's curve is one of the flattest on the plot and its OR is **0.069**, steeper than
pooled. Read `F1b` as *same bias everywhere, very different visible damage* — which is what
sets up the operational section.

Two things to be careful about when reading the figure:

- **Each curve spans only that city's own GRDI range**, so the lines stop at different
  places and comparing line *ends* compares cities at different deprivation levels. Mombasa
  looks like the steepest collapse only because its tiles reach +2 SD; at **+1.13 SD**, the
  last point where all 14 cities are still present, Mombasa is at **82%**. At that common
  point 6 cities are still above 90% and 4 are below 50%. The genuine collapses are Cuenca
  (1.5%), Guayaquil (4.7%) and Zamboanga (24.8%).
- **The pooled line is a tile-weighted average**, so it sits below most of the thin lines —
  at z = 0 it is at 74% while 11 of 14 city curves are above it. This is correct, not a
  drawing error: Guayaquil (806 tiles) and Cuenca (576) are 29% of the 4,700 and are two of
  the three steepest, while the flat cities are smaller. Right of +1.13 SD the pooled line
  extrapolates every city past its own support, so it is drawn dashed there.

**Every city agrees (`F2`).** 14/14 estimable cities have OR < 1 (sign test p = 6.1e-05);
10 significant at 5%. The per-city fits use the headline M3 right-hand side minus the city
fixed effect, so they match the curves in `F1b` — but note that with 73–806 tiles per city
the quadratic costs power, and per-city intervals are wide. The sign test, not the
individual p-values, is the claim here.

**Inference is not doing the work.** The M2 point estimate is 0.092 under every clustering
assumption; only the interval moves.

| Clustering | Clusters | OR | 95% CI |
|---|---|---|---|
| none (naive) | 4,700 | 0.092 | 0.068–0.124 |
| quadkey z12 (~10 km) | 481 | 0.092 | 0.057–0.149 |
| **quadkey z10 (~39 km)** | **75** | **0.092** | **0.058–0.147** |
| quadkey z8 (~157 km) | 27 | 0.092 | 0.063–0.134 |
| city | 14 | 0.092 | 0.062–0.136 |

Clustering is doing real work — it widens the CI by roughly 20% over naive — but the
conclusion is identical at every bandwidth. The two widest blocks give *narrower* intervals
than z10, which is a symptom of too few clusters rather than better precision; z10 is the
level to quote.

### What it costs an operational user (`F4`, `F5`)

| | |
|---|---|
| Coverage, least deprived fifth of tiles | 98.9% |
| Coverage, most deprived fifth | 24.3% |
| **Gap** | **74.7 pp** |
| Tiles with no Meta value | 1,790 of 4,999 (35.8%) ≈ 10,100 km² |
| People in tiles Meta never publishes | 251,293 of 52.0M (**0.5%**) |
| Population-weighted GRDI: visible vs invisible | **25.4 vs 47.0** |
| Targeting recall at a 10%-of-tiles budget | **39%** of the places you should reach |

The 0.5% is the number that makes this subtle and it should be said out loud: **the
invisible population is small but far more deprived.** Those tiles are sparse peri-urban
settlement, so they barely move a population-weighted average — which is exactly why the
original estimand found nothing.

### How many people are unreported? (`F10`, `08_unreported_burden.py`)

"0.5%" is the honest pooled answer and it is the wrong conditioning. `08` reports the same
three burdens — **places**, **land**, **people** — pooled and within deprivation decile,
because each one alone misleads: people-only overstates coverage, land-only overstates harm.

| Burden | Unreported | of total | share |
|---|---|---|---|
| Places | 1,790 tiles | 4,999 | 35.8% |
| Land | 10,106 km² | 27,897 | 36.2% |
| People | 251,293 | 52.0M | **0.48%** |

Density is the reconciliation, and it belongs next to the number rather than in a footnote:
**2,907 people/km² where Meta reports, 25 where it does not.** The missing land is sparse,
not empty.

Now condition on deprivation, which is what makes the operational claim:

| Within-city decile | Places unreported | Land | **People** |
|---|---|---|---|
| 1 (least deprived) | 0.0% | 0.0% | **0.0%** |
| 5 | 26.6% | 28.2% | **0.6%** |
| 8 | 64.4% | 64.5% | **4.8%** |
| 9 | 72.4% | 71.6% | **18.5%** |
| **10 (most deprived)** | **79.0%** | **78.0%** | **31.2%** |

**Meta reports every person in the least deprived tenth of a city and misses nearly a third
of the people in the most deprived tenth.** Across the poorest *fifth*, 90,255 of 412,248
residents — **21.9%** — have no Meta value.

The concentration is the compact version: **36% of all unreported people live in the most
deprived fifth, which holds 0.8% of the total population — a 45× concentration.** Treat that
ratio as a summary, not a headline; its denominator is small precisely because deprived
tiles are sparse, so the decile gradient above is the more robust way to say it.

Per city, the share of the poorest fifth's residents who are unreported reaches **90.6% in
Kandy, 87.2% in Mombasa, 83.4% in Cape Town, 80.1% in León and 79.6% in Guayaquil**
(`A8_unreported_by_city.csv`). Four cities publish everything and sit at 0%.

*Caveat that has to travel with these numbers:* WorldPop is the population reference, and it
is a model, not a census — see §8. These are counts of people WorldPop places in tiles Meta
does not publish, not verified residents.

**The blind spots are contiguous, not scattered** (`F5`). Moran's I on the publication
indicator has median 0.57 and is significant in **14 of the 14 cities where it is defined**
(the other four publish 100% of their tiles, so there is no variation to autocorrelate — the
denominator here is 14, not 18). Scattered dropout averages out over
a district; contiguous dropout removes a whole neighbourhood from the map.

---

## 5. Robustness and sensitivity

### 5.1 Is it a density tautology? (partly — and here is how much)

The sharpest objection. **Three of GRDI's six components are nightlights / built-up
derived**, and its survey components (subnational IHDI, infant mortality) are constant
within a city — so *within-city* GRDI variation is disproportionately the satellite part.
With `corr(GRDI, log WP) = −0.80`, "deprived tiles get dropped" could reduce to "dark tiles
get dropped."

Four pushbacks, and one honest concession:

1. M2–M4 hold population, its square, and settlement class fixed: OR moves 0.023 → 0.125.
   Density explains part of the raw association, not most of it.
2. **M7** residualises GRDI on population, population² and settlement class, keeping only
   the orthogonal signal: OR = 0.362, p = 2e-16.
3. The adjusted dose-response still falls 99% → 46%.
4. Meta's threshold is *declared* to be a count threshold. Finding a residual deprivation
   effect after conditioning on count is the interesting part, not a confound.
5. **Concession (§5.4):** the permutation test puts a number on the density channel —
   about **12%** of the observed gradient is reproducible by density structure alone.

### 5.2 Baseline hour (`F6`)

Publication is hour-specific — a tile busy at 8pm can fall below threshold at 8am — so this
could have been a commuting story. Five countries have a second hour built (11 of 18 cities).

| Hour set | Coverage | OR | 95% CI | p |
|---|---|---|---|---|
| Designated evening hour | 71.2% | 0.057 | 0.030–0.108 | 8e-19 |
| Alternate hour (h00) | 76.4% | 0.097 | 0.058–0.161 | 3e-19 |

Both rows are the **same 2,410 tiles in the same 8 cities**. That restriction matters: which
cities are estimable depends on the hour — a city can publish 100% of its tiles at the
designated hour and drop out, while varying at h00 and staying in — so the unrestricted
version compared 8 cities against 9 and let a difference in sample read as a difference in
hour. Intersecting first makes the contrast about the hour alone.

Every country × hour sits well below 1. Indonesia is excluded: ~100% coverage at both hours,
so its logit is near-separated and uninformative.

### 5.3 Filling the censored cells — bounds, not imputation (`F7`)

"Aren't you just trusting Meta's published numbers? What if we put values into the missing
cells?" Done properly: the censoring is *known*, so this is a bounds problem, not a guess.

| Assumption for each censored cell | Intensive-margin τ |
|---|---|
| Lower bound (≈0) | **−1.40** |
| = 1 | −0.19 |
| = 5 | +0.23 |
| Upper bound (= 10) | **+0.42** |
| Random draw U(0,10) | **+0.16** |

**τ is not sign-identified.** It swings from strongly negative to strongly positive across
the identified interval, flipping around a fill of ~2. A random fill gives **+0.16** — which
would say Meta *over*-represents deprived areas, the opposite conclusion, produced entirely
by the imputation. The mechanism is mechanical: adding a floor of 5–10 users to tiles with
very little WorldPop population inflates their Meta-to-WorldPop ratio.

A left-censored **Tobit**, which uses the censoring structure instead of guessing, gives
τ = **−0.28**, inside the bounds and negative. Two caveats, both of which have to travel
with the number:

- The SE is a BFGS inverse-Hessian SE, **not** cluster-robust, and given the spatial
  autocorrelation it is certainly too small. Ignore the p-value.
- **The optimiser does not report convergence** (`converged = False` in `A7_tobit.csv`).
  The estimate is worth quoting only because it lands comfortably inside the bounds above,
  which is a weak check but a real one. It is a supporting interior point, never a headline
  estimate, and `F7` labels it as such.

**This negative result is load-bearing.** The intensive margin is either null (conditioning
on published) or arbitrarily sign-flippable (imputing). It is not a well-identified estimand
on this data. The binary publication outcome is. That is the strongest argument for the
design in §4.

### 5.4 Is the selection mechanical? Permutation placebo (`F7`, right)

Randomly reassign which tiles are suppressed, 2,000 draws, holding the number suppressed
fixed per city.

| Null | Null mean slope | 95% of draws | Observed |
|---|---|---|---|
| Reshuffled at random | −0.000 | −0.021 to +0.021 | **−0.075** |
| Reshuffled within population deciles | −0.009 | −0.019 to +0.001 | **−0.075** |

Both p ≤ 0.0005, which is the floor `1/draws` imposes — so this table is only reproducible
at the script's default of 2,000 draws. (`07_censoring_bounds.py --draws` used to default to
500, which put the floor at 0.002; the default now matches what is printed here.) The second null is the informative one and it is **not zero**: coarse
density matching alone reproduces about **12%** of the observed gradient, leaving **88% as
deprivation structure beyond population**. (This is a cleaner number than the earlier
19-city version, where Kisumu's AOI truncation inflated the density-matched null.) Quote it
rather than claiming the confound is eliminated.

### 5.5 Five specification checks (`F9`)

`09_sensitivity.py`. All hold the M2 spec fixed and vary one design choice at a time.

**(a) Population floor — the most dangerous objection.** Meta suppresses below ~10 users,
so a tile with 30 residents can almost never qualify whatever its deprivation. If deprived
tiles are systematically tiny, part of the gradient is a mechanical floor that `log(WP)` and
its square may not fully absorb. Restricting to tiles where publication is plainly feasible:

| Restriction | n | Published | OR | p |
|---|---|---|---|---|
| All eligible tiles | 4,700 | 62% | 0.092 | 1e-23 |
| WorldPop ≥ 50 | 3,503 | 80% | 0.128 | 2e-18 |
| WorldPop ≥ 250 | 2,135 | 91% | 0.108 | 2e-13 |
| WorldPop ≥ 500 | 1,782 | 95% | 0.094 | 3e-08 |

The OR is flat across a sample that shrinks by more than half and a publication rate that
climbs from 62% to 95%. **The result is not a small-tile artefact.** Full ladder, including
the ≥10, ≥25 and ≥100 rows: `A9_population_floor.csv`.

**(b) Jackknife.** Leave one city out: OR ranges 0.080–0.099 across 14 refits. Leave one
country out: 0.076–0.106. Max p across all 21 refits is 5e-16. Nothing rests on any one place.

**(c) Adding the excluded cities back.** Nakuru, Garden Route and Kisumu were dropped.
Putting them back takes the sample from 4,700 to 10,443 tiles and gives OR = **0.137** —
the exclusions are not cherry-picking.

**Do not read this row as a better estimate.** Two of the three were dropped for sparse
coverage, but Kisumu was dropped because its absent tiles mark the edge of the published
event AOI, and adding an AOI-truncated city back strengthens the odds ratio for a reason
that has nothing to do with deprivation. Nakuru is a borderline case on the same test (76%
of its tiles never appear in any timestamp, median WorldPop 672 against the 800 cut-off in
`12_aoi_check.py`), so it may be a second one. This row establishes that the exclusions were
conservative, not that 0.137 is the truer number.

**(d) Spatial scale (MAUP).** Aggregating tiles to their parent quadkey and modelling the
*coverage share* by OLS:

| Scale | Units | Coverage drop per SD |
|---|---|---|
| zoom 14 (~1 tile) | 4,700 | −8.3 pp |
| zoom 13 (~4 tiles) | 1,441 | −12.4 pp |
| zoom 12 (~11 tiles) | 481 | **−16.8 pp** |

The gradient **strengthens** at coarser resolution. This matters for the operational claim:
the blindness is a neighbourhood-scale phenomenon, not tile noise.

**(e) Functional form.** Linear z-score OR = 0.092; within-city percentile rank OR = 0.263.
Decile dummies are now *perfectly separated* — the least deprived decile is 100% published —
so the D10-vs-D1 odds ratio collapses to zero. That is a statement, not an estimate, and it
is excluded from `F9`. The linear specification is the conservative one.

**(f) Within settlement class.** WorldPop enters twice — as the eligibility rule and as the
density control — and it is itself a dasymetric model built on satellite covariates that
overlap GRDI's. Stratifying by GHSL SMOD (an independent product) instead of pooling:

| Settlement class | Tiles in stratum | Published | n estimable | OR |
|---|---|---|---|---|
| Urban centre | 805 | **100%** | 0 | not estimable — no variation to explain |
| Town / semi-dense | 526 | 96.2% | 289 | 0.081 [0.012–0.544] |
| Rural | 3,124 | 48.1% | 3,124 | 0.101 [0.059–0.174] |
| Water | 245 | 38.8% | 225 | 0.131 [0.045–0.379] |

All four classes are now reported, and all three estimable strata sit around the pooled
0.092. Not a pooling artefact.

Two notes on reading the table. "Water" is a SMOD centroid class, not an empty tile — a tile
can have its centroid on water while WorldPop still places people in it, so those 245 tiles
are in the estimation sample and are shown rather than quietly dropped. And the estimable
*n* is smaller than the stratum wherever cities with no within-stratum variation fall out,
which is why Urban centre reads 0: that is **805 tiles, every one of them published**, not
805 missing tiles.

**Note what this also shows:** the entire effect lives in **peri-urban and rural** tiles —
urban centres are 100% covered. That matches the maps in `F5` and sharpens the operational
claim: Meta sees the dense core of a city and goes blind at the edge.

### 5.6 Why RWI is not used (`F8`)

Meta's Relative Wealth Index is the obvious alternative deprivation measure. It is
unusable, and for a stronger reason than circularity: RWI is estimated *from* Facebook data,
so it is missing in precisely the tiles whose publication we are modelling.

| | RWI available |
|---|---|
| Tiles Meta published | **95.4%** |
| Tiles Meta suppressed | **47.3%** |

South Africa is the extreme: 86.9% vs 11.7%. Regressing publication on a covariate that is
only observed when the outcome is 1 is not a robustness check — it is the same selection
problem one layer down. Reported as a finding instead: **Meta's poverty index, widely used
for aid targeting, inherits Meta's blind spots.**

---

## 6. What we tried that did not work

Kept here deliberately. Several of these are more informative than the things that worked.

| Attempt | What happened | What it tells us |
|---|---|---|
| **Renormalise shares over the eligible grid** to recover the missing mass | τ identical to 4 dp | City FE absorbs the per-city constant. The sample, not the denominator, is the problem |
| **PPML** (Poisson, `log(WorldPop)` offset, city FE) on the full grid | τ ≈ 0, p > 0.5 | Count-weighted, so dominated by big published tiles. Suppressed tiles have tiny population and contribute almost nothing. Same failure mode as shares |
| **Impute the censored cells** | Sign flips across the identified interval (§5.3) | Not identified. Do not report a single imputed number |
| **SLX direct/indirect split** (M5, M6) | Direct coefficient goes insignificant (OR 0.90, p = 0.78) | `corr(own, neighbour GRDI) = 0.94`. The split is not identified at this resolution; only the **total** effect is. If someone points at M5 and says "the effect vanishes once you control for space", the answer is that M5's direct term is not separately estimable — the total, OR 0.077, is what the model identifies |
| **Meta RWI as alternative deprivation** | Structurally impossible (§5.6) | Became a result instead |
| **Cross-city meta-regression**: does τ track city-level deprivation or coverage? | τ vs coverage Spearman ≈ +0.45, n = 18 | Underpowered. Suggestive only — do not lead with it |
| **Causal estimation** (`03e_causal`, removed) | — | No exogenous variation exists in this design. Removed rather than reported |

---

## 7. What we could not do

- **An independent deprivation measure.** GRDI is partly remote-sensing derived and RWI is
  unusable, so the circularity caveat in §5.1 cannot be fully closed with what is on disk.
  Census or DHS deprivation at admin level for two or three countries would settle it. This
  is the single highest-value addition to the design.
- **Footprint replication.** The city clip truncates exactly the peri-urban zone where the
  effect is largest, so these estimates are likely **conservative**. The eight harmonised
  event-footprint extracts are on disk (`data/processed/footprints/*_aligned.parquet`) and
  have the columns needed; this was not run for time.
- **Full hour coverage.** COL, ECU and MEX have only h00 built, so §5.2 draws on 11 of the
  18 cities and its pooled contrast is estimable on 8. `./run --ref-hour` builds the rest
  from the PDC zips.
- **Per-city censoring thresholds.** The Tobit assumes a common cut-off of 10. Meta's
  effective threshold may vary with its noise injection; estimating it per city would let
  you ask whether the *threshold* or the *underlying usage* differs by deprivation.
- **Baseline-method sensitivity.** Only `n_baseline` was used; the `shift` alternative is
  untested.
- **Clip-boundary sensitivity.** All cities use their default `clip_source`; `osm` and
  `geob` alternatives change which tiles count as in-city and were not re-run (each needs a
  full rebuild of 01 and 01b).
- **Conley HAC standard errors.** Inference uses nested quadkey-block clustering at three
  bandwidths (§4), which is a reasonable approximation but not distance-based Conley SEs.

---

## 8. Limitations to state out loud

- **Associational only.** No exogenous variation in deprivation. No causal language anywhere.
- **~12% of the gradient is density structure** (§5.4), not deprivation per se.
- **Kandy dominates the pooled invisible-population figure** (90k of 251k). Report the
  distribution, not just the total.
- **Four cities are not estimable** at the extensive margin (100% coverage).
- **WorldPop is a model, not ground truth**, and it enters twice: as the eligibility rule
  and as the density control. It is dasymetric, redistributing census counts with satellite
  covariates that overlap GRDI's — so the deprivation measure and the density control are
  **not independent sources of error**. §5.5(f) shows the result is stable within GHSL
  settlement strata, which helps, but does not make WorldPop exogenous.
- **18 cities is not a random sample** of anything; they are the cities with PDC crisis
  extracts.

---

## 9. Suggested slide order

Fifteen slides for a 15-minute talk. Background → data → method → one city → **bridge** →
all cities → consequences. Equations are given in full in §2 and §4; the notes below are what each one
*means* out loud.

**Background (S1–S4, ~3 min)**

1. **Title** — 18 cities, 8 countries, 4,999 tiles.
2. **Population maps decide where help goes** — no figure. Name real users (OCHA, IFRC,
   national disaster agencies). Establishes that this is an allocation decision.
3. **Two ways to build one** — census/dasymetric (WorldPop) vs digital trace (Meta PDC).
   Timeliness is the reason responders adopted digital traces; that trade is what we audit.
4. **Who a digital trace actually sees** — funnel: population → smartphone → app → location
   on → **≥10 users in the tile**. Four filters, each plausibly correlated with poverty.

**Data (S5, ~1.5 min)**

5. **Data and the eligible grid** — notation plus the two sets. `P_c` is what Meta chose to
   publish; `E_c` is every tile where a comparison is defined. **The gap between them is the
   paper.**

**Method (S6–S7, ~2 min)**

6. **The obvious test, and why it fails** — Equation 1, then τ = −0.007, p = 0.93,
   n = 3,209. The catch is the subscript `i ∈ P_c`. Put the S4 funnel alongside with a box
   round the last stage: *"our entire sample."* Title: *"We tested the survivors of the
   process we were trying to detect."*
7. **The right estimand** — Equation 2. Don't ask how much Meta counts, ask whether the tile
   exists at all. Every term observed for every tile; within-city standardisation means
   every comparison is between two tiles in the same city.

**One city (S8, ~2 min)**

8. **`F0_capetown_case_study.png`** — full bleed. Same peripheral ring in rose in both maps.
   100% of deciles 1–3, 3% of decile 10; 231 of 690 tiles missing. Name the places.
   **`F0b_capetown_fitted_probability.png`** is the same city as a fitted curve, and is the
   slide to use instead of quoting an odds ratio. It shows all 690 tiles as their raw 0/1
   outcome, the observed rate per deprivation tenth, and two model lines:

   - **dashed** — each tile's own population. This is what the model says about the data,
     and it tracks the observed rates within 2.4 pp on average.
   - **solid** — population held at the city average. A counterfactual: what if every
     neighbourhood had the same number of residents?

   The solid line sits above the dots because deprived tiles are also far sparser — median
   28,238 residents in the least deprived tenth against 13 in the most. **The gap between
   the two lines is the density channel, drawn**, and the fact that the solid line still
   falls from 100% to ~24% is the part deprivation explains on its own.

**All cities (S9–S12, ~3.5 min)**

9. **`F1b_city_curves_bridge.png`** — the bridge, and the only slide where the audience sees
   one city become fourteen. Cape Town's rose curve from S8 is still there; the thin olive
   lines are the *same model refitted* on each of the other thirteen, and the ink line is the
   pooled fit. Say: **"same model, once per city — and the odds ratio is much the same
   everywhere: median city 0.13 against a pooled 0.12."** Then the point the pooled number
   hides: what that identical slope *does* depends on where a city starts. Puebla and Nairobi
   stay pinned near 100% (Puebla's OR is 0.069 — steeper than pooled — it is just starting
   from 99.9%); Cuenca and Guayaquil fall below 5%. One beat, ~45 seconds, then move on.
   Do **not** invite endpoint comparisons — the lines stop where each city's tiles stop.
10. **`F1`** — 100% → 16% raw; 99% → 46% holding population fixed (population and its
    square, M3; settlement class is deliberately not in this line — see §4).
    exp(β) = 0.123 [0.075, 0.200]. The olive line is the defence against "fewer people,
    fewer users".
11. **`F2`** — 14/14 cities below 1.0, 10 significant. Robustness without a table. Follows
    naturally from S9: the audience has already seen the fourteen curves, this is the same
    fourteen as intervals.
12. **`F3`** — the contribution sentence: *among the people it counts, Meta allocates them
    correctly; it just counts far fewer deprived places.* Reconciles S6 with S10.

**Consequences (S13–S15, ~3 min)**

13. **`F5`** — Moran's I median 0.57, significant in all 14 cities where it is defined.
    Contiguous dropout deletes a
    neighbourhood; scattered dropout would average out.
14. **`F10` — how many people are missing.** This is the slide that answers the question the
    audience is already holding, and the order of the two panels is the whole trick.

    Left, say the honest pooled numbers first: **1,790 of 4,999 tiles, ~10,100 km², but only
    251,293 people — 0.5%.** Let that land as "so Meta is fine?", then give the density
    reconciliation: **2,907 people/km² where Meta reports, 25 where it doesn't.** The missing
    land is sparse, not empty. Then walk the gradient right: the *people* line runs 0% → 31%.

    Right, the operational version: **in the poorest fifth of Kandy, 91% of residents have no
    Meta value; Mombasa 87%; Cape Town 83%.**

    Closing line for the slide: *the 0.5% is why the first analysis found nothing, and the
    31% is why it matters anyway.* Keep 39% targeting recall at a 10% budget in reserve for
    questions.
15. **Takeaway and caveats** — population-representative but geographically censored;
    blindness is peri-urban and rural; association not causation, ~12% of the gradient is
    density structure, GRDI and WorldPop share satellite inputs.

**Closing line:** *the first analysis asked whether Meta miscounts people — it doesn't. The
right question was whether Meta misses places, and it does, systematically, in the poorest
part of every city we looked at.*

**Backup, not shown**

| Question | Slide |
|---|---|
| "Isn't it just small tiles?" | `F9` — OR flat to WorldPop ≥ 500 |
| "Why not fill in the missing cells?" | `F7` — sign flips; random fill gives +0.16 |
| "Does the hour matter?" | `F6` |
| "Which cities drive it?" | `F9` jackknife panel |
| "Why not use Meta's RWI?" | `F8` — 95.4% vs 47.3% availability |
| Per-city coverage gaps | `F4` |
| Full confounder ladder | `A2_extensive_margin_ladder.csv` |
