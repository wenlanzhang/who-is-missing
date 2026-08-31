# Shared palette and theme for every figure in the deck.
#
# Palette: rcartocolor::ArmyRose, a 7-step diverging ramp running olive -> cream
# -> rose. It suits this project because the substantive variable is itself
# diverging: olive is "covered / least deprived", rose is "missing / most
# deprived", and the cream midpoint is a natural neutral.
#
#   #798234  #A3AD62  #D0D3A2  #FDFBE4  #F0C6C3  #DF91A3  #D46780
#      olive ....................  cream  ....................  rose
#
# Semantic aliases below are what the figure scripts use, so the mapping stays
# consistent across the deck: OLIVE always means covered/advantaged, ROSE always
# means missing/deprived.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(scales)
})

# Rscript starts in the C locale on this machine, where R treats UTF-8 as raw
# bytes and the graphics device renders em-dashes and arrows as "...". Force a
# UTF-8 ctype so the axis labels come out right.
if (!grepl("UTF-8", Sys.getlocale("LC_CTYPE"), fixed = TRUE)) {
  for (loc in c("en_US.UTF-8", "C.UTF-8", "UTF-8")) {
    if (suppressWarnings(Sys.setlocale("LC_CTYPE", loc)) != "") break
  }
}

ARMY <- c("#798234", "#A3AD62", "#D0D3A2", "#FDFBE4", "#F0C6C3", "#DF91A3", "#D46780")

OLIVE       <- ARMY[1]   # covered, least deprived, "what Meta sees"
OLIVE_MID   <- ARMY[2]
OLIVE_LIGHT <- ARMY[3]
CREAM       <- ARMY[4]   # neutral fill / published tiles on maps
ROSE_LIGHT  <- ARMY[5]
ROSE_MID    <- ARMY[6]
ROSE        <- ARMY[7]   # missing, most deprived, "what Meta misses"

INK  <- "#2b2b26"
GREY <- "#8c8c84"
RULE <- "#c9c9bf"

# Continuous ramp for deprivation (low -> high). Reversed so that more deprived
# reads as rose, matching the categorical use above.
army_gradient <- function(...) {
  scale_fill_gradientn(colours = ARMY, ...)
}

theme_ar <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(
      text             = element_text(colour = INK),
      plot.title       = element_text(face = "bold", size = base_size + 2.5, hjust = 0,
                                      margin = margin(b = 4)),
      plot.subtitle    = element_text(colour = GREY, size = base_size - 0.5, hjust = 0,
                                      margin = margin(b = 10)),
      plot.caption     = element_text(colour = GREY, size = base_size - 2.5, hjust = 0,
                                      margin = margin(t = 10)),
      plot.title.position  = "plot",
      plot.caption.position = "plot",
      axis.title       = element_text(size = base_size - 0.5),
      axis.text        = element_text(colour = "#5a5a52", size = base_size - 1),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = RULE, linewidth = 0.3),
      legend.position  = "top",
      legend.justification = "left",
      legend.title     = element_blank(),
      legend.key.height = unit(9, "pt"),
      legend.margin    = margin(b = 2),
      plot.background  = element_rect(fill = "white", colour = NA),
      plot.margin      = margin(12, 14, 10, 12)
    )
}

theme_ar_map <- function(base_size = 11) {
  theme_ar(base_size) +
    theme(
      panel.grid  = element_blank(),
      axis.text   = element_blank(),
      axis.title  = element_blank(),
      axis.ticks  = element_blank()
    )
}

CITY_LABEL <- c(
  CagayandeOroCity = "Cagayan de Oro", DavaoCity = "Davao City",
  ZamboangaCity    = "Zamboanga City", GeneralSantosCity = "General Santos",
  MexicoCity       = "Mexico City",    BandaAceh = "Banda Aceh",
  CapeTown         = "Cape Town",      Leon = "Leon"
)
nice_city <- function(x) ifelse(x %in% names(CITY_LABEL), CITY_LABEL[x], x)

COUNTRY_LABEL <- c(PHL = "Philippines", KEN = "Kenya", MEX = "Mexico",
                   IDN = "Indonesia", LKA = "Sri Lanka", COL = "Colombia",
                   ECU = "Ecuador", ZAF = "South Africa")

ROOT    <- normalizePath(file.path(dirname(sys.frame(1)$ofile %||% "."), "..", ".."),
                         mustWork = FALSE)
`%||%`  <- function(a, b) if (is.null(a)) b else a

# The default quartz/png device drops em-dashes and arrows to "...". ragg renders
# UTF-8 correctly, so use it when available and fall back to cairo otherwise.
save_fig <- function(plot, name, width, height, dir = "figure/analysis") {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  dev <- if (requireNamespace("ragg", quietly = TRUE)) ragg::agg_png else "cairo"
  ggsave(file.path(dir, name), plot, width = width, height = height,
         dpi = 200, bg = "white", device = dev)
  cat("  ", name, "\n", sep = "")
}
