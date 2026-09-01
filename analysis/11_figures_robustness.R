#!/usr/bin/env Rscript
# Robustness figures F6-F9, in the ArmyRose palette.
#
#   Rscript analysis/11_figures_robustness.R

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(ggplot2)
  library(patchwork); library(forcats); library(readr)
})
source("analysis/R/theme_armyrose.R")

TAB <- "outputs/analysis"
rd  <- function(f) read_csv(file.path(TAB, f), show_col_types = FALSE)

cat("Writing figures to figure/analysis\n")

# ------------------------------------------------------------------- F6 ref-hour
f6 <- function() {
  # Drop degenerate fits: a country publishing ~100% of tiles at both hours is
  # near-separated, with a point estimate around 1e-11 and a CI spanning nine
  # orders of magnitude. It carries no information and would rescale the axis.
  fits <- rd("A6_refhour_extensive_margin.csv") |> filter(is.finite(OR), OR > 1e-4)
  pooled <- rd("A6_refhour_pooled.csv")

  d <- fits |>
    mutate(country_lab = ifelse(country %in% names(COUNTRY_LABEL),
                                COUNTRY_LABEL[country], country),
           lab = sprintf("%s  h%02d", country_lab, hour),
           grp = ifelse(designated, "country's designated (evening) hour",
                        "alternate hour")) |>
    arrange(country, hour) |> mutate(lab = fct_rev(fct_inorder(lab)))

  p_pool <- pooled |>
    mutate(lab = paste0("POOLED  ", hour_set),
           grp = ifelse(grepl("designated", hour_set),
                        "country's designated (evening) hour", "alternate hour"))

  p <- ggplot(d, aes(OR, lab, colour = grp)) +
    geom_vline(xintercept = 1, colour = GREY, linetype = "22") +
    geom_errorbar(aes(xmin = OR_lo, xmax = OR_hi), width = 0, linewidth = 0.7,
                  orientation = "y") +
    geom_point(aes(shape = grp), size = 2.7) +
    geom_errorbar(data = p_pool, aes(y = lab, xmin = OR_lo, xmax = OR_hi),
                  width = 0, linewidth = 1.5, orientation = "y", inherit.aes = FALSE,
                  colour = INK) +
    geom_point(data = p_pool, aes(OR, lab), size = 4, shape = 18,
               inherit.aes = FALSE, colour = INK) +
    scale_colour_manual(values = c("country's designated (evening) hour" = ROSE,
                                   "alternate hour" = OLIVE)) +
    scale_shape_manual(values = c("country's designated (evening) hour" = 16,
                                  "alternate hour" = 15)) +
    scale_x_log10() +
    labs(title = "The coverage gradient does not depend on which hour we snapshot",
         subtitle = "Five countries have a second baseline hour built (11 of 18 cities)",
         x = "Odds ratio of publication per +1 SD of deprivation  (log scale)", y = NULL,
         caption = "Indonesia is omitted: it publishes ~100% of tiles at both hours, so nothing is estimable.") +
    theme_ar()
  save_fig(p, "F6_refhour_robustness.png", 8.4, 5.6)
}

# ------------------------------------------------------------- F7 bounds/placebo
f7 <- function() {
  grid <- rd("A7_imputation_grid.csv")
  bnds <- rd("A7_censoring_bounds.csv")
  pl   <- rd("A7_placebo_permutation.csv")
  tob  <- if (file.exists(file.path(TAB, "A7_tobit.csv"))) rd("A7_tobit.csv") else NULL
  rnd  <- bnds |> filter(grepl("^Random draw", spec))

  a <- ggplot(grid, aes(fill_value, tau)) +
    geom_hline(yintercept = 0, colour = GREY, linetype = "22") +
    geom_ribbon(aes(ymin = tau - 1.96 * se, ymax = tau + 1.96 * se),
                fill = OLIVE, alpha = 0.16) +
    geom_line(colour = OLIVE, linewidth = 1.1) +
    geom_point(colour = OLIVE, size = 2.2)
  if (nrow(rnd)) a <- a + geom_point(data = rnd, aes(x = 5, y = tau),
                                     colour = ROSE, size = 4, shape = 18)
  if (!is.null(tob)) a <- a +
    geom_hline(yintercept = tob$tau[1], colour = ROSE_MID, linetype = "13",
               linewidth = 0.9)
  # The Tobit is worth drawing because it falls inside the bounds, but the
  # optimiser does not always report convergence and the figure must say so
  # rather than presenting it as a settled point estimate.
  tob_note <- if (!is.null(tob) && isFALSE(tob$converged[1]))
    "dotted = Tobit (optimiser did not converge — indicative only)" else
    "dotted = Tobit estimate"
  a <- a +
    labs(title = "Filling the gaps does not identify anything",
         subtitle = paste("rose diamond = random fill U(0,10);", tob_note),
         x = "Value assumed for each censored cell  (the truth is somewhere in 0–10)",
         y = "Intensive-margin τ") +
    theme_ar()

  pm <- pl |> mutate(lab = ifelse(null == "uniform", "reshuffled\nat random",
                                  "reshuffled within\npopulation deciles"))
  b <- ggplot(pm, aes(y = lab)) +
    geom_vline(xintercept = 0, colour = GREY, linetype = "22") +
    geom_segment(aes(x = null_q025, xend = null_q975, yend = lab),
                 colour = OLIVE_LIGHT, linewidth = 7, lineend = "butt") +
    geom_point(aes(x = null_mean), colour = INK, size = 2, shape = 124) +
    geom_point(aes(x = observed_coef), colour = ROSE, size = 4) +
    labs(title = "But the selection itself is real",
         subtitle = sprintf("rose = observed; band = 95%% of %s placebo draws",
                            format(pm$draws[1], big.mark = ",")),
         x = "Publication–deprivation slope (pp per SD)", y = NULL) +
    theme_ar()

  out <- (a | b) + plot_layout(widths = c(1.4, 1)) +
    plot_annotation(
      title = "The censored cells cannot be filled in — which is why the binary coverage outcome is the right one",
      theme = theme_ar())
  save_fig(out, "F7_censoring_bounds.png", 12.0, 5.0)
}

# ------------------------------------------------------------------- F8 RWI
f8 <- function() {
  d <- rd("A6_rwi_structural_selection.csv") |>
    filter(!is.na(rwi_cov_unpublished)) |>
    mutate(country_lab = ifelse(country %in% names(COUNTRY_LABEL),
                                COUNTRY_LABEL[country], country)) |>
    arrange(gap_pp) |> mutate(country_lab = fct_inorder(country_lab))

  long <- d |>
    select(country_lab,
           `tiles Meta published`  = rwi_cov_published,
           `tiles Meta suppressed` = rwi_cov_unpublished) |>
    pivot_longer(-country_lab, names_to = "grp", values_to = "cov")

  p <- ggplot(long, aes(100 * cov, country_lab, fill = grp)) +
    geom_col(position = position_dodge(width = 0.72), width = 0.66) +
    scale_fill_manual(values = c("tiles Meta published" = OLIVE,
                                 "tiles Meta suppressed" = ROSE)) +
    scale_x_continuous(limits = c(0, 104), expand = c(0, 0)) +
    labs(title = "Meta's poverty index inherits Meta's blind spots",
         subtitle = "RWI is estimated from Facebook data, so it is missing where Meta has no users",
         x = "% of tiles with a Meta RWI value", y = NULL,
         caption = paste("That makes RWI unusable as an alternative deprivation measure here",
                         "— and is itself a finding about downstream use.")) +
    theme_ar()
  save_fig(p, "F8_rwi_structural_selection.png", 8.2, 4.8)
}

# ----------------------------------------------------------- F9 specification
f9 <- function() {
  pf  <- rd("A9_population_floor.csv")
  jc  <- rd("A9_jackknife_city.csv")
  jn  <- rd("A9_jackknife_country.csv")
  oos <- rd("A9_out_of_sample.csv")
  ff  <- rd("A9_functional_form.csv")
  ss  <- rd("A9_spatial_scale.csv")

  main <- pf |> filter(floor == 0) |> slice(1)
  worst <- function(t) t |> filter(dropped != "(none - full sample)", !is.na(OR)) |>
    slice_max(OR, n = 1)
  wc <- worst(jc); wn <- worst(jn)

  rows <- bind_rows(
    main |> transmute(lab = "MAIN: all eligible tiles", OR, OR_lo, OR_hi, is_main = TRUE),
    pf |> filter(floor > 0) |>
      transmute(lab = sprintf("only tiles with ≥%d people", floor),
                OR, OR_lo, OR_hi, is_main = FALSE),
    wc |> transmute(lab = paste0("drop any one city (worst): ", dropped),
                    OR, OR_lo, OR_hi, is_main = FALSE),
    wn |> transmute(lab = paste0("drop any one country (worst): ", dropped),
                    OR, OR_lo, OR_hi, is_main = FALSE),
    # Not "the sparse cities": one of the three (Kisumu) is excluded for AOI
    # truncation, which is a different and stronger reason. See
    # out_of_sample() in analysis/09_sensitivity.py.
    oos |> slice_tail(n = 1) |>
      transmute(lab = "+ the 3 excluded cities", OR, OR_lo, OR_hi, is_main = FALSE),
    ff |> filter(form != "linear z-score (main spec)") |>
      transmute(lab = paste0("deprivation as ", form), OR, OR_lo, OR_hi, is_main = FALSE)
  ) |>
    # Decile dummies are perfectly separated once the least deprived decile is
    # ~100% published, so their OR collapses to ~0 with a zero-width interval.
    # That is a real finding but not a point you can put on a log axis.
    filter(OR > 1e-6) |>
    mutate(lab = fct_rev(fct_inorder(lab)))

  a <- ggplot(rows, aes(OR, lab)) +
    annotate("rect", xmin = main$OR_lo, xmax = main$OR_hi, ymin = -Inf, ymax = Inf,
             fill = ROSE, alpha = 0.10) +
    geom_vline(xintercept = 1, colour = GREY, linetype = "22") +
    geom_errorbar(aes(xmin = OR_lo, xmax = OR_hi, colour = is_main),
                  width = 0, orientation = "y", linewidth = 0.75) +
    geom_point(aes(colour = is_main, size = is_main, shape = is_main)) +
    scale_colour_manual(values = c(`TRUE` = ROSE, `FALSE` = INK), guide = "none") +
    scale_size_manual(values = c(`TRUE` = 3.6, `FALSE` = 2.3), guide = "none") +
    scale_shape_manual(values = c(`TRUE` = 18, `FALSE` = 16), guide = "none") +
    scale_x_log10() +
    labs(title = "Nothing moves the conclusion",
         subtitle = "shaded band = main 95% CI",
         x = "Odds ratio per +1 SD of deprivation  (log scale)", y = NULL) +
    theme_ar()

  b <- ggplot(ss, aes(factor(zoom), -coef_pp)) +
    geom_col(fill = OLIVE, width = 0.6) +
    geom_errorbar(aes(ymin = -coef_pp - 1.96 * se_pp, ymax = -coef_pp + 1.96 * se_pp),
                  width = 0.14, colour = INK, linewidth = 0.6) +
    scale_x_discrete(labels = function(z)
      sprintf("zoom %s\n~%s tiles", z, ss$median_children[match(z, ss$zoom)])) +
    labs(title = "Stronger at coarser scale",
         subtitle = "not a small-tile artefact",
         x = NULL, y = "Drop in coverage, pp per SD") +
    theme_ar()

  out <- (a | b) + plot_layout(widths = c(2, 1)) +
    plot_annotation(
      title = "Sensitivity: population floor, jackknife, sample, functional form, spatial scale",
      theme = theme_ar())
  save_fig(out, "F9_sensitivity_specification_curve.png", 12.4, 5.2)
}

f6(); f7(); f8(); f9()
cat("Done.\n")
