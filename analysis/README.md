# Does Meta's crisis population baseline under-represent deprived places?

Full method, results, robustness, and the things that did not work.

**Run order**

```bash
python analysis/01_build_panel.py             # pooled tile panel -> analysis/panel/
python analysis/02_selection_models.py        # extensive margin + robustness ladder
python analysis/03_two_margin_decomposition.py  # reconciles the null
python analysis/04_operational_consequence.py   # blind spots, clustering, targeting
python analysis/05_city_models.py             # per-city models (feeds F0b, F1b)
python analysis/06_refhour_robustness.py      # baseline-hour check, and why not RWI
python analysis/07_censoring_bounds.py        # bounds, imputation, Tobit, placebo
python analysis/09_sensitivity.py             # floor, jackknife, MAUP, form
python analysis/12_aoi_check.py               # suppression vs event-AOI edge
Rscript analysis/10_figures_main.R            # F0, F0b, F1, F1b, F2-F5
Rscript analysis/11_figures_robustness.R      # F6-F9
```

Tables → `outputs/analysis/`, figures → `figure/analysis/`. Whole chain ≈ 1 minute.

---

## 1. The hypothesis

> Higher deprivation → lower Facebook penetration → Meta under-counts people relative to
> WorldPop → weaker agreement between Meta and WorldPop in deprived areas.

Both sources are normalised to within-city shares first, so the comparison is about
**spatial distribution**, not totals. Deprivation and population are spatially structured,
so spatial models are used throughout.

The original test was `log(meta_share / worldpop_share) ~ GRDI`, with a spatial error model
for spatial confounding. It returned nothing usable. Re-estimated per city on the current
sample (`A3_intensive_margin_by_city.csv`): of 17 estimable cities, **6 significantly
negative, 4 significantly positive, 7 null**, τ ranging −0.53 to +1.59 with a median of
+0.09. Pooled, it is a flat zero.

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
deprivation decile (range 0.98–1.28). Among the people Meta counts, it allocates them
essentially correctly.

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

**Dose-response (`F1`).** Meta publishes **100%** of tiles in the least deprived decile and
**21%** in the most deprived. Holding population fixed: **100% → 56%**.

**`F1b` is the bridge from one city to all of them.** The same model fitted separately to
each of the 14 cities, drawn on a common axis, with the pooled fixed-effects curve on top.
It exists because the talk otherwise jumps from a single-city figure to a pooled estimate
with no visible connection. It also shows something the pooled number hides: **cities differ
enormously in how much of their gap survives equalising population.** In Mexico City, Puebla
and Nairobi the equal-population curve stays near 100% — their coverage gap is almost
entirely about how few people live in deprived tiles. In Cuenca, Guayaquil and Mombasa it
collapses below 10% — theirs is about deprivation. Cape Town sits in between. The pooled
line is the average of that range, not a description of any one city.

**Every city agrees (`F2`).** 14/14 estimable cities have OR < 1 (sign test p = 6.1e-05);
12 significant at 5%.

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

**The blind spots are contiguous, not scattered** (`F5`). Moran's I on the publication
indicator has median 0.57, significant in 14/18 cities. Scattered dropout averages out over
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
3. The adjusted dose-response still falls 100% → 59%.
4. Meta's threshold is *declared* to be a count threshold. Finding a residual deprivation
   effect after conditioning on count is the interesting part, not a confound.
5. **Concession (§5.4):** the permutation test puts a number on the density channel —
   about **12%** of the observed gradient is reproducible by density structure alone.

### 5.2 Baseline hour (`F6`)

Publication is hour-specific — a tile busy at 8pm can fall below threshold at 8am — so this
could have been a commuting story. Five countries have a second hour built (11 of 18 cities).

| Hour set | Coverage | OR | 95% CI | p |
|---|---|---|---|---|
| Designated evening hour | 74.0% | 0.057 | 0.030–0.108 | 8e-15 |
| Alternate hour (h00) | 78.6% | 0.096 | 0.057–0.159 | 2e-16 |

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
τ = **−0.28**, inside the bounds and negative. Treat the point estimate as informative but
**not** the standard error — it is a BFGS inverse-Hessian SE, not cluster-robust, and given
the spatial autocorrelation it is certainly too small.

**This negative result is load-bearing.** The intensive margin is either null (conditioning
on published) or arbitrarily sign-flippable (imputing). It is not a well-identified estimand
on this data. The binary publication outcome is. That is the strongest argument for the
design in §4.

### 5.4 Is the selection mechanical? Permutation placebo (`F7`, right)

Randomly reassign which tiles are suppressed, 2,000 draws, holding the number suppressed
fixed per city.

| Null | Null mean slope | 95% of draws | Observed |
|---|---|---|---|
| Reshuffled at random | −0.000 | −0.020 to +0.022 | **−0.075** |
| Reshuffled within population deciles | −0.009 | −0.018 to +0.001 | **−0.075** |

Both p ≤ 0.0005. The second null is the informative one and it is **not zero**: coarse
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
| WorldPop ≥ 50 | 3,554 | 76% | 0.111 | 3e-19 |
| WorldPop ≥ 250 | 2,146 | 90% | 0.109 | 5e-11 |
| WorldPop ≥ 500 | 1,782 | 95% | 0.094 | 3e-08 |

The OR is flat, if anything slightly *weaker* at the bottom. **The result is not a
small-tile artefact.**

**(b) Jackknife.** Leave one city out: OR ranges 0.080–0.099 across 14 refits. Leave one
country out: 0.076–0.106. Max p across all 21 refits is 5e-16. Nothing rests on any one place.

**(c) Adding the excluded cities back.** Nakuru, Garden Route and Kisumu were dropped.
Putting them back more than doubles the sample and gives OR = 0.137 — the exclusions are
not cherry-picking, and if anything they were conservative.

**(d) Spatial scale (MAUP).** Aggregating tiles to their parent quadkey and modelling the
*coverage share* by OLS:

| Scale | Units | Coverage drop per SD |
|---|---|---|
| zoom 14 (~1 tile) | 4,700 | −8.3 pp |
| zoom 13 (~4 tiles) | 1,441 | −12.4 pp |
| zoom 12 (~11 tiles) | 481 | **−16.8 pp** |

The gradient **strengthens** at coarser resolution. This matters for the operational claim:
the blindness is a neighbourhood-scale phenomenon, not tile noise.

**(e) Functional form.** Linear z-score OR = 0.092; within-city percentile rank OR = 0.24.
Decile dummies are now *perfectly separated* — the least deprived decile is 100% published —
so the D10-vs-D1 odds ratio collapses to zero. That is a statement, not an estimate, and it
is excluded from `F9`. The linear specification is the conservative one.

**(f) Within settlement class.** WorldPop enters twice — as the eligibility rule and as the
density control — and it is itself a dasymetric model built on satellite covariates that
overlap GRDI's. Stratifying by GHSL SMOD (an independent product) instead of pooling:

| Settlement class | n | OR |
|---|---|---|
| Urban centre | — | not estimable — almost every tile is published |
| Town / semi-dense | 289 | 0.081 [0.012–0.544] |
| Rural | 3,124 | 0.101 [0.059–0.174] |

Both estimable strata bracket the pooled 0.092. Not a pooling artefact.

**Note what this also shows:** the entire effect lives in **peri-urban and rural** tiles —
urban centres are essentially fully covered. That matches the maps in `F5` and sharpens the
operational claim: Meta sees the dense core of a city and goes blind at the edge.

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
- **Full hour coverage.** COL, ECU and MEX have only h00 built, so §5.2 covers 12 of 19
  cities. `./run --ref-hour` builds the rest from the PDC zips.
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

Fourteen slides for a 15-minute talk. Background → data → method → one city → all cities →
consequences. Equations are given in full in §2 and §4; the notes below are what each one
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

**All cities (S9–S11, ~3 min)**

9. **`F1`** — 100% → 21% raw; 100% → 59% holding population and settlement type fixed.
   exp(β) = 0.125 [0.076, 0.206]. The olive line is the defence against "fewer people,
   fewer users".
10. **`F2`** — 14/14 cities below 1.0, 12 significant. Robustness without a table.
11. **`F3`** — the contribution sentence: *among the people it counts, Meta allocates them
    correctly; it just counts far fewer deprived places.* Reconciles S6 with S9.

**Consequences (S12–S14, ~3 min)**

12. **`F5`** — Moran's I median 0.57, significant in 14/18. Contiguous dropout deletes a
    neighbourhood; scattered dropout would average out.
13. **What it costs** — 98.9% vs 24.3% coverage (74.7 pp gap); 1,790 of 4,999 tiles
    (~10,100 km²); 251,293 people (0.5%) at deprivation 47.0 vs 25.4; 39% targeting recall
    at a 10% budget. Say the 0.5% out loud — it is small *because* those tiles are sparse,
    which is why the population-weighted test found nothing.
14. **Takeaway and caveats** — population-representative but geographically censored;
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
