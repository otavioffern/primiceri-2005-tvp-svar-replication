#!/usr/bin/env Rscript

# Benchmark replication using Fabian Krueger's bvarsv R/C++ package.
# The package implements the Primiceri (2005) model and incorporates the
# Del Negro and Primiceri (2015) corrigendum.

parse_args <- function(args) {
  opts <- list(
    nrep = 10000L,
    nburn = 2000L,
    thinfac = 10L,
    seed = 5813L,
    itprint = 1000L,
    install = FALSE
  )

  for (arg in args) {
    if (arg == "--install") {
      opts$install <- TRUE
    } else if (grepl("^--[^=]+=", arg)) {
      key <- sub("^--([^=]+)=.*$", "\\1", arg)
      value <- sub("^--[^=]+=", "", arg)
      key <- gsub("-", "_", key)
      if (!key %in% names(opts)) {
        stop(sprintf("Unknown argument: %s", arg), call. = FALSE)
      }
      if (is.logical(opts[[key]])) {
        opts[[key]] <- tolower(value) %in% c("1", "true", "yes")
      } else {
        opts[[key]] <- as.integer(value)
      }
    } else {
      stop(sprintf("Arguments must be --name=value or --install; got: %s", arg), call. = FALSE)
    }
  }
  opts
}

ensure_package <- function(package, install) {
  if (requireNamespace(package, quietly = TRUE)) {
    return(invisible(TRUE))
  }
  if (!install) {
    stop(
      sprintf(
        "Package '%s' is not installed. Re-run with --install or install it manually.",
        package
      ),
      call. = FALSE
    )
  }
  install.packages(
    package,
    repos = c("https://fk83.r-universe.dev", "https://cloud.r-project.org")
  )
}

quarter_label <- function(x) {
  year <- floor(x)
  quarter <- round((x - year) * 4) + 1L
  sprintf("%dQ%d", year, quarter)
}

period_mean <- function(values, dates, start, end) {
  idx <- dates >= start & dates <= end
  colMeans(values[idx, , drop = FALSE])
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
ensure_package("bvarsv", args$install)

dir.create("outputs", showWarnings = FALSE, recursive = TRUE)
dir.create(file.path("outputs", "figures"), showWarnings = FALSE, recursive = TRUE)

library(bvarsv)
data(usmacro, package = "bvarsv")

set.seed(args$seed)
fit <- bvar.sv.tvp(
  usmacro,
  p = 2,
  tau = 40,
  nrep = args$nrep,
  nburn = args$nburn,
  thinfac = args$thinfac,
  itprint = args$itprint,
  save.parameters = TRUE
)

series_names <- colnames(usmacro)

h_mean <- fit$H.postmean
time_index_full <- as.numeric(time(usmacro))
time_index <- tail(time_index_full, dim(h_mean)[3])
dates <- quarter_label(time_index)
volatility <- t(vapply(
  seq_len(dim(h_mean)[3]),
  function(t) sqrt(diag(h_mean[, , t])),
  numeric(length(series_names))
))
colnames(volatility) <- paste0(series_names, "_shock_std")

volatility_df <- data.frame(
  quarter = dates,
  time = time_index,
  volatility,
  check.names = FALSE
)
write.csv(
  volatility_df,
  file.path("outputs", "r_bvarsv_volatility_paths.csv"),
  row.names = FALSE
)

periods <- list(
  pre_volcker_1963Q3_1979Q2 = c(1963.50, 1979.25),
  volcker_transition_1979Q3_1983Q4 = c(1979.50, 1983.75),
  post_1984Q1_2001Q3 = c(1984.00, 2001.50)
)

summary_rows <- lapply(names(periods), function(name) {
  bounds <- periods[[name]]
  means <- period_mean(volatility, time_index, bounds[1], bounds[2])
  data.frame(period = name, as.list(means), check.names = FALSE)
})
summary_df <- do.call(rbind, summary_rows)
write.csv(
  summary_df,
  file.path("outputs", "r_bvarsv_volatility_summary.csv"),
  row.names = FALSE
)

png(
  filename = file.path("outputs", "figures", "r_bvarsv_volatilities.png"),
  width = 1600,
  height = 1050,
  res = 180
)
old_par <- par(no.readonly = TRUE)
on.exit(par(old_par), add = TRUE)
par(mfrow = c(length(series_names), 1), mar = c(3, 4, 2, 1), oma = c(1, 0, 2, 0))
for (i in seq_along(series_names)) {
  plot(
    time_index,
    volatility[, i],
    type = "l",
    col = "#1f4e79",
    lwd = 1.5,
    xlab = "",
    ylab = "Std. dev.",
    main = sprintf("bvarsv posterior mean volatility: %s", series_names[i])
  )
  grid(col = "#dddddd")
}
mtext("Primiceri TVP-VAR-SV benchmark via bvarsv", outer = TRUE, cex = 1.1)
dev.off()

saveRDS(
  list(
    command_args = commandArgs(trailingOnly = TRUE),
    args = args,
    series_names = series_names,
    volatility_summary = summary_df,
    fit_names = names(fit)
  ),
  file.path("outputs", "r_bvarsv_benchmark_summary.rds")
)

message("Wrote:")
message("  outputs/r_bvarsv_volatility_paths.csv")
message("  outputs/r_bvarsv_volatility_summary.csv")
message("  outputs/figures/r_bvarsv_volatilities.png")
message("  outputs/r_bvarsv_benchmark_summary.rds")
