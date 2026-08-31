#!/usr/bin/env Rscript
# Main-story figures F0-F5, in the ArmyRose palette.
#
# Run from the repository root, after the Python analysis scripts:
#   Rscript analysis/10_figures_main.R

suppressPackageStartupMessages({
  library(sf); library(dplyr); library(tidyr); library(ggplot2)
  library(patchwork); library(forcats); library(readr)
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
f0b <- function(country = "ZAF", city = "CapeTown") {
  # The odds ratio is the statistically correct summary but a hard one to read in
  # a talk. This is the same model expressed as a predicted probability: for a
  # tile of average population, how likely is Meta to publish it at each level of
  # deprivation? The observed decile rates are overlaid so the curve can be
  # checked against the raw data rather than taken on trust.
  p <- file.path("data/processed/city", country, city,
                 "01b_coverage/independent_grid.gpkg")
  if (!file.exists(p)) { cat("  ! missing ", p, "\n"); return(invisible()) }

  g <- sf::st_read(p, quiet = TRUE) |> sf::st_drop_geometry() |>
    filter(eligible == 1, !is.na(poverty_mean), worldpop_count > 0) |>
    mutate(log_wp = log(worldpop_count),
           z_grdi = as.numeric(scale(poverty_mean)),
           z_lwp  = as.numeric(scale(log_wp)),
           decile = ntile(poverty_mean, 10))

  m <- glm(published ~ z_grdi + z_lwp, data = g, family = binomial())

  mu <- mean(g$poverty_mean); sdv <- sd(g$poverty_mean)
  grid <- tibble::tibble(
    poverty_mean = seq(min(g$poverty_mean), max(g$poverty_mean), length.out = 300)) |>
    mutate(z_grdi = (poverty_mean - mu) / sdv,
           z_lwp  = mean(g$z_lwp))          # population held at the city average
  pr <- predict(m, newdata = grid, type = "link", se.fit = TRUE)
  grid <- grid |> mutate(fit = plogis(pr$fit),
                         lo  = plogis(pr$fit - 1.96 * pr$se.fit),
                         hi  = plogis(pr$fit + 1.96 * pr$se.fit))

  obs <- g |> group_by(decile) |>
    summarise(grdi = median(poverty_mean), rate = mean(published), .groups = "drop")

  lim <- range(g$poverty_mean)
  p_out <- ggplot() +
    geom_ribbon(data = grid, aes(poverty_mean, ymin = 100 * lo, ymax = 100 * hi),
                fill = OLIVE, alpha = 0.16) +
    geom_line(data = grid, aes(poverty_mean, 100 * fit), colour = INK, linewidth = 1.1) +
    geom_point(data = obs, aes(grdi, 100 * rate, fill = grdi), shape = 21,
               size = 3.4, colour = "white", stroke = 0.6) +
    scale_fill_gradientn(colours = ARMY, limits = lim, guide = "none") +
    scale_y_continuous(limits = c(-2, 104), expand = c(0, 0),
                       labels = function(v) paste0(v, "%")) +
    labs(title = sprintf("%s: how likely is Meta to publish a neighbourhood?",
                         nice_city(city)),
         subtitle = "Fitted probability for a tile of average population, with the observed decile rates overlaid",
         x = "Deprivation (GRDI)  →  more deprived",
         y = "Probability Meta publishes the tile",
         caption = paste0(
           "Logistic fit on ", nrow(g), " tiles, controlling for log population. ",
           "Line = fitted probability holding population at the city average;\n",
           "band = 95% interval; dots = the share actually published in each ",
           "deprivation tenth, coloured by deprivation.")) +
    theme_ar()
  save_fig(p_out, "F0b_capetown_fitted_probability.png", 8.0, 5.2)
}

# ------------------------------------------------------------ F1 dose-response
f1 <- function() {
  raw <- rd("A2_dose_response_grdi_decile.csv")
  adj <- rd("A2_dose_response_adjusted.csv")
  d <- raw |>
    transmute(decile = grdi_decile, pct = 100 * pub_rate,
              lo = 100 * (pub_rate - 1.96 * se), hi = 100 * (pub_rate + 1.96 * se)) |>
    left_join(adj |> transmute(decile = grdi_decile, adj = 100 * pub_rate_adj),
              by = "decile")

  p <- ggplot(d, aes(decile)) +
    geom_ribbon(aes(ymin = lo, ymax = hi), fill = ROSE, alpha = 0.18) +
    geom_line(aes(y = pct, colour = "Observed"), linewidth = 1.15) +
    geom_point(aes(y = pct, colour = "Observed"), size = 2.6) +
    geom_line(aes(y = adj, colour = "Population and settlement type held fixed"),
              linewidth = 1.0, linetype = "22") +
    geom_point(aes(y = adj, colour = "Population and settlement type held fixed"),
               size = 2.1, shape = 15) +
    scale_colour_manual(values = c("Observed" = ROSE,
                                   "Population and settlement type held fixed" = OLIVE)) +
    scale_x_continuous(breaks = 1:10) +
    scale_y_continuous(limits = c(0, 104), expand = c(0, 0)) +
    labs(title = "Meta publishes almost every affluent tile and few deprived ones",
         subtitle = "Deciles are formed within each city, so no cross-city difference drives the gradient",
         x = "Within-city deprivation decile (GRDI)  →  more deprived",
         y = "% of tiles with a Meta baseline value",
         caption = "18 cities, 8 countries. Ribbon is the 95% interval on the observed rate.") +
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

f0(); f0b(); f1(); f2(); f3(); f4(); f5()
cat("Done.\n")
