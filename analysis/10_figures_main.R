#!/usr/bin/env Rscript
# Main-story figures F0-F5, in the ArmyRose palette.
#
# Run from the repository root, after the Python analysis scripts:
#   Rscript analysis/10_figures_main.R

suppressPackageStartupMessages({
  library(sf); library(dplyr); library(tidyr); library(ggplot2)
  library(patchwork); library(forcats); library(readr); library(ggrepel)
})
source("analysis/R/theme_armyrose.R")

TAB <- "outputs/analysis"
rd  <- function(f) read_csv(file.path(TAB, f), show_col_types = FALSE)

cat("Writing figures to figure/analysis\n")

# ---------------------------------------------------------------- F0 case study
f0 <- function(country = "ZAF", city = "CapeTown") {
  p <- file.path("data/processed/city", country, city,
                 "01b_coverage/independent_grid.gpkg")
  if (!file.exists(p)) { cat("  ! missing ", p, "\n"); return(invisible()) }
  g <- st_read(p, quiet = TRUE) |>
    filter(eligible == 1, !is.na(poverty_mean), worldpop_count > 0) |>
    mutate(decile = ntile(poverty_mean, 10),
           status = factor(ifelse(published == 1, "Meta publishes", "No Meta data"),
                           levels = c("Meta publishes", "No Meta data")))

  lim <- quantile(g$poverty_mean, c(.02, .98), na.rm = TRUE)
  p1 <- ggplot(g) +
    geom_sf(aes(fill = poverty_mean), colour = "white", linewidth = 0.04) +
    scale_fill_gradientn(colours = ARMY, limits = lim, oob = scales::squish,
                         labels = NULL) +
    labs(title = "1. Where deprivation is",
         subtitle = "GRDI — rose is more deprived") +
    coord_sf(datum = NA) + theme_ar_map() + theme(legend.position = "none")

  p2 <- ggplot(g) +
    geom_sf(aes(fill = status), colour = "white", linewidth = 0.04) +
    scale_fill_manual(values = c("Meta publishes" = CREAM, "No Meta data" = ROSE)) +
    labs(title = "2. What Meta publishes",
         subtitle = sprintf("%d of %d tiles have no data",
                            sum(g$published == 0), nrow(g))) +
    coord_sf(datum = NA) + theme_ar_map()

  rate <- g |> st_drop_geometry() |> group_by(decile) |>
    summarise(pct = 100 * mean(published), .groups = "drop")
  p3 <- ggplot(rate, aes(decile, pct, fill = pct > 50)) +
    geom_col(width = 0.74) +
    scale_fill_manual(values = c(`TRUE` = OLIVE, `FALSE` = ROSE), guide = "none") +
    scale_x_continuous(breaks = 1:10) +
    scale_y_continuous(limits = c(0, 108), expand = c(0, 0)) +
    labs(title = "3. The gradient", subtitle = " ",
         x = "Deprivation decile  →  more deprived",
         y = "% of tiles with Meta data") +
    theme_ar()

  out <- (p1 | p2 | p3) + plot_layout(widths = c(1, 1, 1.15)) +
    plot_annotation(
      title = sprintf("%s: Meta's crisis baseline covers the whole city — except the townships",
                      nice_city(city)),
      caption = paste0(
        nrow(g), " eligible tiles. Meta publishes ",
        round(100 * mean(g$published[g$decile <= 2])), "% of the least deprived fifth and ",
        round(100 * mean(g$published[g$decile >= 9])), "% of the most deprived fifth.\n",
        "The missing tiles hold ", format(round(sum(g$worldpop_count[g$published == 0])),
                                          big.mark = ","),
        " people whose population-weighted deprivation is far above the city average."),
      theme = theme_ar())
  save_fig(out, "F0_capetown_case_study.png", 12.6, 5.4)
}

# ------------------------------------------------- F0b single-city fitted curve
f0b <- function(city_id = "CapeTown") {
  # All estimation happens in analysis/05_city_models.py; this only draws.
  tiles <- rd("A5_city_tiles.csv")   |> filter(city == city_id)
  curve <- rd("A5_city_curves.csv")  |> filter(city == city_id)
  dec   <- rd("A5_city_deciles.csv") |> filter(city == city_id)
  if (!nrow(curve)) { cat("  ! no model for ", city_id, "\n"); return(invisible()) }

  lim <- range(tiles$poverty_mean)
  p <- ggplot() +
    geom_jitter(data = tiles, aes(poverty_mean, 100 * published, colour = poverty_mean),
                height = 3.2, width = 0, size = 0.7, alpha = 0.35) +
    geom_ribbon(data = curve, aes(poverty_mean, ymin = 100 * lo, ymax = 100 * hi),
                fill = OLIVE, alpha = 0.16) +
    geom_line(data = dec, aes(median_grdi, 100 * fit_actual), colour = ROSE,
              linewidth = 0.9, linetype = "22") +
    geom_line(data = curve, aes(poverty_mean, 100 * fit), colour = INK, linewidth = 1.1) +
    geom_point(data = dec, aes(median_grdi, 100 * observed, fill = median_grdi),
               shape = 21, size = 3.6, colour = "white", stroke = 0.7) +
    scale_colour_gradientn(colours = ARMY, limits = lim, guide = "none") +
    scale_fill_gradientn(colours = ARMY, limits = lim, guide = "none") +
    scale_y_continuous(limits = c(-8, 108), breaks = seq(0, 100, 25),
                       expand = c(0, 0), labels = function(v) paste0(v, "%")) +
    annotate("text", x = lim[2], y = 105, hjust = 1, size = 3.0, colour = GREY,
             label = "one faint dot = one tile Meta published") +
    annotate("text", x = lim[1], y = -5, hjust = 0, size = 3.0, colour = GREY,
             label = "one faint dot = one tile Meta did not publish") +
    labs(title = sprintf("%s: how likely is Meta to publish a neighbourhood?",
                         nice_city(city_id)),
         subtitle = paste("Solid = every area given the same population ·",
                          "dashed = each area's real population · dots = observed"),
         x = "Deprivation (GRDI)  →  more deprived",
         y = "Probability Meta publishes the tile",
         caption = paste0(
           "All ", nrow(tiles), " eligible tiles shown. Logistic model of publication on ",
           "deprivation and log population (fitted in analysis/05_city_models.py);\n",
           "band is a 95% interval with errors clustered on ~39 km blocks. The dashed line ",
           "follows the real data. The solid line asks what would happen if\nevery ",
           "neighbourhood held the city's average number of residents — it sits higher ",
           "because deprived tiles are also far sparser.")) +
    theme_ar()
  save_fig(p, "F0b_capetown_fitted_probability.png", 9.0, 5.8)
}

# ------------------------------------------- F1b one city -> all cities bridge
f1b <- function(highlight = "CapeTown") {
  # Same model per city as F0b, drawn on a common axis (deprivation in
  # within-city standard deviations) so 14 cities can be compared, with the
  # pooled fixed-effects model on top. This is the step between "one city" and
  # "all cities": the same object, once per city.
  curves <- rd("A5_city_curves.csv")
  pooled <- rd("A5_pooled_curve.csv")

  # Read the odds ratios rather than hardcoding them in the subtitle. They were
  # written out by hand and would silently go stale the next time the sample or
  # the specification moves.
  ors      <- rd("A2_extensive_margin_by_city.csv") |> filter(!is.na(OR))
  or_city  <- median(ors$OR)
  or_pool  <- rd("A2_extensive_margin_ladder.csv") |>
    filter(grepl("^M3", model)) |> pull(OR)

  # Every city's curve is drawn only over the deprivation range it actually has
  # tiles in, so the lines stop at different places. z_common is the last point
  # where all 14 are still present. To its right cities leave the comparison one
  # by one, and the pooled line — an average over every tile at every grid point —
  # is extrapolating city fixed effects past their own data, so it is dashed there.
  z_common <- min(tapply(curves$z_grdi, curves$city, max))

  ends <- curves |> group_by(city) |> filter(z_grdi == max(z_grdi)) |> ungroup()

  # Where each city sits at that common point. Reading the line ends instead
  # would compare cities at different deprivation levels: Mombasa looks like the
  # steepest collapse only because its tiles reach +2 SD.
  at_common <- curves |> group_by(city) |>
    summarise(f = approx(z_grdi, fit, xout = z_common)$y, .groups = "drop")
  n_high <- sum(at_common$f >= 0.9)
  n_low  <- sum(at_common$f <= 0.5)

  # Every line is named. The panel is extended to the right to make a label
  # gutter, and ggrepel moves labels in y only, drawing a leader when it has to
  # shift one far from its line — so a label never sits on top of a curve.
  # Colours stay on the deck's convention: olive = a city, rose = the highlighted
  # city, ink = pooled.
  x_pool <- max(pooled$z_grdi)
  end_pool <- 100 * pooled$fit[nrow(pooled)]
  lab <- bind_rows(
    ends |> transmute(x = z_grdi, y = 100 * fit, nm = nice_city(city),
                      col = ifelse(city == highlight, ROSE, OLIVE),
                      fw  = ifelse(city == highlight, "bold", "plain")),
    tibble(x = x_pool, y = end_pool, nm = "pooled, 14 cities", col = INK, fw = "bold"))

  # ggrepel avoids other labels and its own points, but knows nothing about the
  # curves — which is how a label ends up sitting on a line it does not belong
  # to. Feeding it the curves as empty-label points makes them obstacles too.
  all_pts <- bind_rows(curves |> select(x = z_grdi, y = fit),
                       pooled |> select(x = z_grdi, y = fit)) |> mutate(y = 100 * y)
  obstacles <- all_pts |> slice(seq(1, n(), by = 4)) |>
    transmute(x, y, nm = "", col = NA_character_, fw = "plain")

  # Labels may only move vertically (below), so the horizontal offset has to be
  # right first time: push each one far enough to clear any curve crossing its
  # own horizontal band. Without this a label whose line ends just left of a
  # steep curve — Kandy against Cape Town — has nowhere to go.
  BAND  <- 3.6   # half-height of a label, in percentage points
  REACH <- 0.95  # how far right it is worth looking, in SD
  nudge <- vapply(seq_len(nrow(lab)), function(i) {
    hit <- all_pts[abs(all_pts$y - lab$y[i]) < BAND &
                   all_pts$x > lab$x[i] & all_pts$x < lab$x[i] + REACH, ]
    if (!nrow(hit)) 0.10 else min(REACH, max(hit$x) - lab$x[i] + 0.10)
  }, numeric(1))

  p <- ggplot() +
    geom_vline(xintercept = z_common, colour = RULE, linetype = "22", linewidth = 0.4) +
    geom_ribbon(data = pooled, aes(z_grdi, ymin = 100 * lo, ymax = 100 * hi),
                fill = INK, alpha = 0.10) +
    geom_line(data = curves |> filter(city != highlight),
              aes(z_grdi, 100 * fit, group = city), colour = OLIVE_MID,
              linewidth = 0.55, alpha = 0.75) +
    geom_point(data = ends |> filter(city != highlight),
               aes(z_grdi, 100 * fit), colour = OLIVE_MID, size = 1.15, alpha = 0.9) +
    geom_line(data = curves |> filter(city == highlight),
              aes(z_grdi, 100 * fit), colour = ROSE, linewidth = 1.3) +
    geom_point(data = ends |> filter(city == highlight),
               aes(z_grdi, 100 * fit), colour = ROSE, size = 2.1) +
    # Pooled: solid where every city has data, dashed where it is extrapolating.
    geom_line(data = pooled |> filter(z_grdi <= z_common),
              aes(z_grdi, 100 * fit), colour = INK, linewidth = 1.4) +
    geom_line(data = pooled |> filter(z_grdi >= z_common),
              aes(z_grdi, 100 * fit), colour = INK, linewidth = 1.1, linetype = "22") +
    geom_point(aes(x_pool, end_pool), colour = INK, size = 2.1) +
    annotate("text", x = z_common - 0.06, y = 105, hjust = 1, size = 2.9, colour = GREY,
             label = "all 14 cities have tiles left of here") +
    ggrepel::geom_text_repel(
      data = bind_rows(lab, obstacles),
      aes(x, y, label = nm, colour = col, fontface = fw),
      hjust = 0, direction = "y", size = 2.6, seed = 7, na.rm = TRUE,
      nudge_x = c(nudge, rep(0, nrow(obstacles))),
      segment.size = 0.25, segment.colour = GREY, segment.alpha = 0.6,
      min.segment.length = 0.2, box.padding = 0.14, point.padding = 0.12,
      max.overlaps = Inf, xlim = c(NA, 3.55), ylim = c(-2, 103),
      max.iter = 20000, max.time = 3) +
    scale_colour_identity() +
    scale_discrete_identity(aesthetics = "fontface") +
    scale_x_continuous(breaks = seq(-2, 2, 1), limits = c(-2.55, 3.6)) +
    scale_y_continuous(limits = c(-2, 108), breaks = seq(0, 100, 25),
                       expand = c(0, 0), labels = function(v) paste0(v, "%")) +
    labs(title = "The same model, once per city",
         subtitle = sprintf(paste0(
           "Each thin line is one city, with population held at that city's average.\n",
           "The odds ratio is much the same everywhere — median city %.2f, pooled %.2f — ",
           "but what it does to a map depends on where the city starts."),
           or_city, or_pool),
         x = "Deprivation, in standard deviations from the city's own average  →  more deprived",
         y = "Probability Meta publishes the tile",
         caption = paste0(
           "14 cities with enough variation to estimate; a dot marks the most deprived tile ",
           "each city actually has. Because those ranges differ, comparing line ends compares\n",
           "cities at different deprivation levels. At +", sprintf("%.2f", z_common),
           " SD, the last point where all 14 are present, ", n_high, " cities are still above ",
           "90% and ", n_low, " are below 50%. The pooled line is a tile-weighted\naverage, so ",
           "the largest cities pull it below the middle of the band; right of the rule it ",
           "extrapolates every city past its own data and is dashed.")) +
    theme_ar()
  save_fig(p, "F1b_city_curves_bridge.png", 9.8, 6.0)
}

# ------------------------------------------------------------ F1 dose-response
f1 <- function() {
  # Both lines come from the *same* table, which is the 14-city estimation
  # sample. Taking the observed line from A2_dose_response_grdi_decile.csv
  # instead would draw it on all 18 cities (4,999 tiles) against a fitted line
  # on 4,700 — two different samples on one pair of axes.
  adj <- rd("A2_dose_response_adjusted.csv")
  n_tiles <- sum(adj$n_tiles)
  d <- adj |>
    transmute(decile = grdi_decile, pct = 100 * pub_rate_raw,
              lo = 100 * (pub_rate_raw - 1.96 * se_raw),
              hi = 100 * (pub_rate_raw + 1.96 * se_raw),
              adj = 100 * pub_rate_adj)

  lab_obs <- d |> filter(decile %in% c(1, 9, 10))
  lab_adj <- d |> filter(decile %in% c(9, 10))     # decile 1 would overlap

  p <- ggplot(d, aes(decile)) +
    geom_ribbon(aes(ymin = lo, ymax = hi), fill = ROSE, alpha = 0.18) +
    geom_line(aes(y = pct, colour = "Observed"), linewidth = 1.15) +
    geom_point(aes(y = pct, colour = "Observed"), size = 2.6) +
    geom_line(aes(y = adj, colour = "Population held fixed"),
              linewidth = 1.0, linetype = "22") +
    geom_point(aes(y = adj, colour = "Population held fixed"),
               size = 2.4, shape = 22, fill = "white", stroke = 0.8) +
    scale_colour_manual(values = c("Observed" = ROSE,
                                   "Population held fixed" = OLIVE)) +
    # Label the ends so the audience can read "100% down to what?" without
    # tracing back to the axis. Decile 1 gets one label because both lines sit
    # on ~100% there and two would overlap.
    geom_text(data = lab_obs, aes(decile, pct, label = paste0(round(pct), "%")),
              colour = ROSE, fontface = "bold", size = 3.4, vjust = 2.1) +
    geom_text(data = lab_adj, aes(decile, adj, label = paste0(round(adj), "%")),
              colour = OLIVE, fontface = "bold", size = 3.4, vjust = -1.3) +
    scale_x_continuous(breaks = 1:10) +
    scale_y_continuous(limits = c(0, 104), expand = c(0, 0)) +
    labs(title = "Meta publishes almost every affluent tile and few deprived ones",
         subtitle = "Deciles are formed within each city, so no cross-city difference drives the gradient",
         x = "Within-city deprivation decile (GRDI)  →  more deprived",
         y = "% of tiles with a Meta baseline value",
         caption = sprintf(paste("Both lines on the same %s tiles in 14 cities (the four cities publishing",
                                 "100%% of tiles are not estimable).\nRibbon is the 95%% interval on the",
                                 "observed rate."),
                           format(n_tiles, big.mark = ","))) +
    theme_ar()
  save_fig(p, "F1_publication_by_deprivation_decile.png", 8.0, 5.0)
}

# --------------------------------------------------------------- F2 city forest
f2 <- function() {
  d <- rd("A2_extensive_margin_by_city.csv") |>
    filter(!is.na(OR)) |>
    mutate(city = nice_city(city),
           lab  = sprintf("%s  (%.0f%% covered)", city, 100 * C_c),
           sig  = ifelse(p < 0.05, "significant at 5%", "not significant")) |>
    arrange(OR) |> mutate(lab = fct_inorder(lab))

  n_sig <- sum(d$p < 0.05)
  p <- ggplot(d, aes(OR, lab, colour = sig)) +
    geom_vline(xintercept = 1, colour = GREY, linetype = "22") +
    geom_errorbar(aes(xmin = OR_lo, xmax = OR_hi), width = 0, linewidth = 0.7,
                  orientation = "y") +
    geom_point(size = 2.7) +
    scale_colour_manual(values = c("significant at 5%" = ROSE,
                                   "not significant" = OLIVE_LIGHT)) +
    scale_x_log10(breaks = c(0.001, 0.01, 0.1, 1),
                  labels = c("0.001", "0.01", "0.1", "1")) +
    labs(title = "Every city points the same way",
         subtitle = sprintf("%d of %d cities below 1.0  ·  %d significant at 5%%  ·  each adjusted for log WorldPop",
                            sum(d$OR < 1), nrow(d), n_sig),
         x = "Odds ratio of publication per +1 SD of deprivation  (log scale)", y = NULL,
         caption = "Cities publishing 100% of eligible tiles (Colombo, Medan, Banda Aceh, Barranquilla) are not estimable.") +
    theme_ar()
  save_fig(p, "F2_per_city_odds_ratios.png", 8.4, 5.6)
}

# --------------------------------------------------------------- F3 two margins
f3 <- function() {
  d <- rd("A3_representation_ratio_by_decile.csv")
  a <- ggplot(d, aes(grdi_decile, R_eligible_norm)) +
    geom_hline(yintercept = 1, colour = GREY, linetype = "22") +
    geom_line(colour = OLIVE, linewidth = 1.1) +
    geom_point(colour = OLIVE, size = 2.4) +
    scale_x_continuous(breaks = 1:10) + coord_cartesian(ylim = c(0.6, 1.5)) +
    labs(title = "Among the people it counts,\nMeta is unbiased",
         subtitle = "this is the null the original hypothesis kept finding",
         x = "Deprivation decile  →", y = "Meta share ÷ WorldPop share") +
    theme_ar()

  b <- ggplot(d, aes(grdi_decile, 100 * pub_rate)) +
    geom_line(colour = ROSE, linewidth = 1.1) +
    geom_point(colour = ROSE, size = 2.4) +
    scale_x_continuous(breaks = 1:10) +
    scale_y_continuous(limits = c(0, 104), expand = c(0, 0)) +
    labs(title = "But it counts far fewer\ndeprived places",
         subtitle = " ", x = "Deprivation decile  →", y = "% of tiles published") +
    theme_ar()

  out <- (a | b) + plot_annotation(
    title = "The bias is in which places exist in the data, not in how people are allocated",
    theme = theme_ar())
  save_fig(out, "F3_two_margins.png", 10.6, 4.8)
}

# ------------------------------------------------------------------ F4 dumbbell
f4 <- function() {
  d <- rd("A4_blind_spots.csv") |>
    mutate(city = nice_city(city)) |> arrange(coverage_gap_pp) |>
    mutate(city = fct_inorder(city))
  long <- d |>
    select(city, `most deprived fifth` = cov_most_deprived,
           `least deprived fifth` = cov_least_deprived) |>
    pivot_longer(-city, names_to = "grp", values_to = "cov")

  # Four cities publish 100% of both fifths, so the two points coincide exactly
  # and read as a single dot. Label them rather than leaving a bare marker.
  full <- d |> filter(coverage_gap_pp < 0.01)

  p <- ggplot(d) +
    geom_segment(aes(x = 100 * cov_most_deprived, xend = 100 * cov_least_deprived,
                     y = city, yend = city), colour = RULE, linewidth = 0.9) +
    geom_point(data = long, aes(100 * cov, city, colour = grp), size = 3) +
    geom_text(data = d |> filter(coverage_gap_pp >= 0.01),
              aes(x = 100 * cov_most_deprived, y = city,
                  label = sprintf("%.0f pp gap", coverage_gap_pp)),
              hjust = 1.35, size = 2.9, colour = GREY) +
    geom_text(data = full, aes(x = 100, y = city, label = "fully covered"),
              hjust = -0.18, size = 2.9, colour = GREY) +
    scale_colour_manual(values = c("most deprived fifth" = ROSE,
                                   "least deprived fifth" = OLIVE)) +
    scale_x_continuous(limits = c(-22, 128), breaks = c(0, 25, 50, 75, 100),
                       expand = c(0, 0)) +
    labs(title = "The poorest fifth of a city is the part Meta cannot see",
         subtitle = "Coverage of the least vs most deprived fifth of tiles, within each city",
         x = "% of tiles with a Meta baseline value", y = NULL) +
    theme_ar()
  save_fig(p, "F4_coverage_dumbbell.png", 8.0, 5.6)
}

# ----------------------------------------------------------------- F5 blind map
f5 <- function(cities = list(c("ZAF", "CapeTown"), c("LKA", "Kandy"),
                             c("ECU", "Guayaquil"))) {
  panels <- lapply(cities, function(cc) {
    p <- file.path("data/processed/city", cc[1], cc[2],
                   "01b_coverage/independent_grid.gpkg")
    if (!file.exists(p)) return(NULL)
    g <- st_read(p, quiet = TRUE) |>
      filter(eligible == 1, !is.na(poverty_mean), worldpop_count > 0)
    mis <- filter(g, published == 0)
    lim <- quantile(g$poverty_mean, c(.05, .95), na.rm = TRUE)
    ggplot() +
      geom_sf(data = filter(g, published == 1), fill = CREAM, colour = "white",
              linewidth = 0.04) +
      geom_sf(data = mis, aes(fill = poverty_mean), colour = "white", linewidth = 0.04) +
      scale_fill_gradientn(colours = ARMY[4:7], limits = lim, oob = scales::squish,
                           guide = "none") +
      labs(title = nice_city(cc[2]),
           subtitle = sprintf("%d of %d tiles missing (%.0f%%)",
                              nrow(mis), nrow(g), 100 * nrow(mis) / nrow(g))) +
      coord_sf(datum = NA) + theme_ar_map()
  })
  panels <- Filter(Negate(is.null), panels)
  out <- wrap_plots(panels, nrow = 1) +
    plot_annotation(
      title = "Missing tiles are contiguous zones on the urban edge, shaded by deprivation",
      caption = "Cream = Meta publishes a value. Rose = no Meta value, deeper is more deprived (GRDI).",
      theme = theme_ar())
  save_fig(out, "F5_blind_spot_maps.png", 12.4, 5.0)
}

# ------------------------------------------------- F10 the unreported burden
f10 <- function() {
  # Two panels answering "how many people are missing?" honestly. The left panel
  # exists to stop the pooled 0.5% being quoted alone: places and land are
  # missing at ~36% throughout, while the *people* share only takes off in the
  # deprived deciles. The right panel is the operational version of the same
  # fact, city by city.
  dec <- rd("A8_unreported_by_decile.csv")
  cty <- rd("A8_unreported_by_city.csv")
  hd  <- rd("A8_unreported_headline.csv")
  pooled_pop <- hd$pct_pop_unreported[hd$scope == "All eligible tiles"]

  # Land is deliberately not drawn. Within a city, zoom-14 tiles are near-equal
  # area, so % of tiles and % of km2 differ by under a point at every decile
  # (35.8% vs 36.2% pooled) and the two lines sit on top of each other. Plotting
  # both implied two independent facts where there is one. The km2 total goes in
  # the caption instead.
  long <- dec |>
    transmute(decile = grdi_decile,
              `Tiles (places)` = pct_tiles_unreported,
              `People living in them` = pct_pop_unreported) |>
    pivot_longer(-decile, names_to = "burden", values_to = "pct") |>
    mutate(burden = factor(burden, levels = c("Tiles (places)", "People living in them")))

  ends <- long |> filter(decile == 10)

  a <- ggplot(long, aes(decile, pct, colour = burden)) +
    geom_hline(yintercept = pooled_pop, colour = GREY, linetype = "22") +
    annotate("text", x = 1, y = pooled_pop + 4, hjust = 0, size = 2.9, colour = GREY,
             label = sprintf("all people, pooled: %.1f%%", pooled_pop)) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 2.3) +
    geom_text(data = ends, aes(label = sprintf("%.0f%%", pct)),
              hjust = -0.35, size = 3.3, fontface = "bold", show.legend = FALSE) +
    scale_colour_manual(values = c(`Tiles (places)` = OLIVE,
                                   `People living in them` = ROSE)) +
    scale_x_continuous(breaks = 1:10, limits = c(1, 10.9)) +
    scale_y_continuous(limits = c(0, 100), expand = c(0, 0)) +
    labs(title = "Missing tiles and missing people, by deprivation",
         subtitle = "Share of each within-city deprivation decile that Meta never publishes",
         x = "Within-city deprivation decile  →  more deprived",
         y = "% unreported", colour = NULL) +
    theme_ar()

  cc <- cty |>
    filter(pct_pop_unreported_top > 0) |>
    mutate(city = nice_city(city)) |>
    arrange(pct_pop_unreported_top) |>
    mutate(city = fct_inorder(city))

  b <- ggplot(cc, aes(pct_pop_unreported_top, city)) +
    geom_segment(aes(x = 0, xend = pct_pop_unreported_top, yend = city),
                 colour = RULE, linewidth = 0.9) +
    geom_point(colour = ROSE, size = 3) +
    geom_text(aes(label = sprintf("%.0f%%", pct_pop_unreported_top)),
              hjust = -0.45, size = 2.9, colour = GREY) +
    scale_x_continuous(limits = c(0, 108), expand = c(0, 0)) +
    labs(title = "Residents unreported in the most deprived 20% of tiles",
         subtitle = "% of the people living there who have no Meta value",
         x = NULL, y = NULL) +
    theme_ar()

  out <- (a | b) + plot_layout(widths = c(1.25, 1)) +
    plot_annotation(
      title = "Where Meta's missing tiles are, and who lives in them",
      caption = paste0(
        "18 cities, 4,999 eligible tiles (10,106 km² of the 27,897 km² mapped have no Meta ",
        "value). Missing land tracks missing tiles almost exactly — 36.2% against 35.8% — so ",
        "only tiles are drawn.\nLeft: both burdens rise with deprivation, but the people line ",
        "sits far below because the missing tiles are sparse (25 vs 2,907 people/km²). That ",
        "density gap, not good coverage, is why the\npooled figure is 0.5%. Right: the four ",
        "cities publishing 100% of their tiles are omitted. WorldPop is the population ",
        "reference throughout."),
      theme = theme_ar())
  save_fig(out, "F10_unreported_burden.png", 12.4, 5.4)
}

f0(); f0b(); f1(); f1b(); f2(); f3(); f4(); f5(); f10()
cat("Done.\n")
