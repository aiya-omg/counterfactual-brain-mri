# 病変検出の GLMM。リポジトリのルートから実行する。
# Rscript src/stage_curve/stats/glmm.R results/stage_curve/lesions.csv
#
# 収束しなかったブートストラップは信頼区間から外し、回数だけ記録する。

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("lesions.csv のパスを渡す")
lesion_csv <- args[[1]]
out_dir <- dirname(lesion_csv)

cfg_lines <- readLines("config/stage_curve.yaml")
value_of <- function(key) {
  line <- grep(paste0("^", key, ":"), cfg_lines, value = TRUE)[[1]]
  as.numeric(sub(".*:\\s*", "", line))
}
seed <- value_of("seed")
draws <- value_of("bootstrap_draws")
set.seed(seed)

suppressPackageStartupMessages({
  library(lme4)
  library(blme)
  library(detectseparation)
  library(dplyr)
  library(readr)
})

raw <- read_csv(lesion_csv, show_col_types = FALSE)
raw$patient_id <- as.character(raw$patient_id)
raw$dataset <- as.character(raw$dataset)

usable <- function(data, ratio_col = "adc_ratio_median") {
  data <- data[is.finite(data[[ratio_col]]) & data[[ratio_col]] > 0, ]
  data$ratio <- data[[ratio_col]]
  data
}

on_original_scale <- function(fit, data, standardized) {
  if (!standardized) {
    beta <- lme4::fixef(fit)
    return(c(intercept = unname(beta[["(Intercept)"]]),
             log_adc = unname(beta[["log(ratio)"]]),
             log_vol = unname(beta[["log_vol_c"]])))
  }
  beta <- lme4::fixef(fit)
  m1 <- mean(log(data$ratio)); s1 <- sd(log(data$ratio))
  m2 <- mean(data$log_vol_c); s2 <- sd(data$log_vol_c)
  b0 <- unname(beta[["(Intercept)"]])
  b1 <- unname(beta[["z_adc"]])
  b2 <- unname(beta[["z_vol"]])
  c(intercept = b0 - b1 * m1 / s1 - b2 * m2 / s2,
    log_adc = b1 / s1,
    log_vol = b2 / s2)
}

fit_one <- function(data) {
  x <- model.matrix(~ log(ratio) + log_vol_c, data)
  sep <- tryCatch(
    detect_separation(x, data$detected, family = binomial(), intercept = FALSE),
    error = function(e) TRUE)
  separated <- isTRUE(sep) || (is.logical(sep) && any(sep, na.rm = TRUE))
  fit <- tryCatch(
    glmer(detected ~ log(ratio) + log_vol_c + (1 | patient_id),
          data = data, family = binomial,
          control = glmerControl(optimizer = "bobyqa")),
    error = function(e) NULL)
  extreme <- !is.null(fit) && any(abs(lme4::fixef(fit)) > 10)
  used <- "glmer"
  standardized <- FALSE
  if (is.null(fit) || separated || extreme) {
    scaled <- data
    scaled$z_adc <- as.numeric(scale(log(data$ratio)))
    scaled$z_vol <- as.numeric(scale(data$log_vol_c))
    fit <- bglmer(detected ~ z_adc + z_vol + (1 | patient_id),
                  data = scaled, family = binomial,
                  fixef.prior = normal(sd = c(10, 2.5, 2.5)),
                  control = glmerControl(optimizer = "bobyqa"))
    used <- "bglmer"
    standardized <- TRUE
    separated <- TRUE
  }
  list(fit = fit, used = used, separated = separated,
       coef = on_original_scale(fit, data, standardized))
}

converged <- function(fit) {
  messages <- fit@optinfo$conv$lme4$messages
  is.null(messages) || length(messages) == 0
}

ed_from <- function(coef, probability) {
  # 体積 1 mL（log_vol_c = 0）での ADC 比。
  exp((qlogis(probability) - coef[["intercept"]]) / coef[["log_adc"]])
}

bootstrap_eds <- function(data) {
  patients <- unique(data$patient_id)
  ed50 <- ed90 <- rep(NA_real_, draws)
  intercept <- log_adc <- rep(NA_real_, draws)
  failed <- 0L
  for (draw in seq_len(draws)) {
    chosen <- sample(patients, length(patients), replace = TRUE)
    parts <- lapply(seq_along(chosen), function(i) {
      part <- data[data$patient_id == chosen[[i]], ]
      part$patient_id <- paste(chosen[[i]], i, sep = "#")
      part
    })
    boot <- bind_rows(parts)
    result <- tryCatch(fit_one(boot), error = function(e) NULL)
    if (is.null(result) || !converged(result$fit) || !is.finite(result$coef[["log_adc"]]) ||
        result$coef[["log_adc"]] == 0) {
      failed <- failed + 1L
      next
    }
    ed50[[draw]] <- ed_from(result$coef, 0.5)
    ed90[[draw]] <- ed_from(result$coef, 0.9)
    intercept[[draw]] <- result$coef[["intercept"]]
    log_adc[[draw]] <- result$coef[["log_adc"]]
  }
  list(ed50 = ed50, ed90 = ed90, intercept = intercept, log_adc = log_adc, failed = failed)
}

ci <- function(samples) {
  samples <- samples[is.finite(samples)]
  if (length(samples) < 10) return(c(NA_real_, NA_real_))
  as.numeric(quantile(samples, c(0.025, 0.975)))
}

summarize <- function(name, data, ratio_col = "adc_ratio_median", outcome = "detected",
                      save_draws = FALSE) {
  data <- usable(data, ratio_col)
  data$detected <- data[[outcome]]
  point <- fit_one(data)
  boot <- bootstrap_eds(data)
  ed50_ci <- ci(boot$ed50)
  ed90_ci <- ci(boot$ed90)
  row <- tibble(
    analysis = name,
    ed50 = ed_from(point$coef, 0.5),
    ed50_ci_low = ed50_ci[[1]],
    ed50_ci_high = ed50_ci[[2]],
    ed90 = ed_from(point$coef, 0.9),
    ed90_ci_low = ed90_ci[[1]],
    ed90_ci_high = ed90_ci[[2]],
    lesions = nrow(data),
    patients = n_distinct(data$patient_id),
    separated = point$separated,
    model = point$used,
    n_bootstrap_failed = boot$failed,
    log_adc = point$coef[["log_adc"]],
    log_adc_ci_low = ci(boot$log_adc)[[1]],
    log_adc_ci_high = ci(boot$log_adc)[[2]]
  )
  draws <- NULL
  if (save_draws) {
    draws <- tibble(intercept = boot$intercept, log_adc = boot$log_adc,
                    ed50 = boot$ed50, ed90 = boot$ed90)
  }
  list(row = row, draws = draws, point = point, boot = boot)
}

main_data <- usable(raw[raw$dataset %in% c("ISLES", "SENORA_masked"), ])
main_data$detected <- main_data$detected
results <- list()
main_fit <- summarize("main", main_data, save_draws = TRUE)
results[[1]] <- main_fit$row

results[[2]] <- summarize("detected_any", main_data, outcome = "detected_any")$row
results[[3]] <- summarize("detected_half", main_data, outcome = "detected_half")$row
results[[4]] <- summarize("min_volume_0.5ml", main_data[main_data$vol_ml >= 0.5, ])$row
results[[5]] <- summarize("adc_ratio_eroded", main_data, ratio_col = "adc_ratio_eroded")$row
results[[6]] <- summarize("supratentorial", main_data[main_data$infratentorial == 0, ])$row

with_dataset <- main_data
with_dataset$dataset <- factor(with_dataset$dataset, levels = c("ISLES", "SENORA_masked"))
dataset_fit <- tryCatch(
  glmer(detected ~ log(ratio) + log_vol_c + dataset + (1 | patient_id),
        data = with_dataset, family = binomial,
        control = glmerControl(optimizer = "bobyqa")),
  error = function(e) NULL)
dataset_row <- main_fit$row
dataset_row$analysis <- "dataset_fixed_effect"
dataset_row$model <- if (is.null(dataset_fit)) "failed" else "glmer"
dataset_row$ed50_ci_low <- NA_real_
dataset_row$ed50_ci_high <- NA_real_
dataset_row$ed90_ci_low <- NA_real_
dataset_row$ed90_ci_high <- NA_real_
dataset_row$log_adc_ci_low <- NA_real_
dataset_row$log_adc_ci_high <- NA_real_
dataset_row$n_bootstrap_failed <- NA_integer_
if (!is.null(dataset_fit)) {
  beta <- lme4::fixef(dataset_fit)
  # ED50 は基準水準（ISLES）、体積 1 mL。dataset を足すと ADC 比と相関するので探索的。
  dataset_row$ed50 <- exp((qlogis(0.5) - beta[["(Intercept)"]]) / beta[["log(ratio)"]])
  dataset_row$ed90 <- exp((qlogis(0.9) - beta[["(Intercept)"]]) / beta[["log(ratio)"]])
  dataset_row$log_adc <- unname(beta[["log(ratio)"]])
  dataset_row$separated <- any(abs(beta) > 10)
}
results[[7]] <- dataset_row

cluster_data <- usable(raw[raw$dataset %in% c("ISLES", "SENORA_masked", "SENORA_cluster"), ])
results[[8]] <- summarize("plus_senora_cluster", cluster_data)$row

dwi_data <- main_data[is.finite(main_data$dwi_ratio_median) & main_data$dwi_ratio_median > 0, ]
dwi_cor <- cor(log(dwi_data$ratio), log(dwi_data$dwi_ratio_median), use = "complete.obs")
dwi_lm <- lm(log(ratio) ~ log(dwi_ratio_median), data = dwi_data)
dwi_vif <- 1 / (1 - summary(dwi_lm)$r.squared)

table <- bind_rows(results)
write_csv(table, file.path(out_dir, "ed_estimates.csv"))
write_csv(main_fit$draws, file.path(out_dir, "coef_draws.csv"))

summary_lines <- c(
  "主モデル: detected ~ log(adc_ratio) + log_vol_c + (1 | patient_id)",
  "体積 1 mL での ED50 / ED90。信頼区間は患者単位のクラスター・ブートストラップ。",
  paste("使ったモデル:", main_fit$row$model, " 分離:", main_fit$row$separated),
  paste("log(ADC 比) の係数:", signif(main_fit$row$log_adc, 4),
        " 95% CI", signif(main_fit$row$log_adc_ci_low, 4), "〜", signif(main_fit$row$log_adc_ci_high, 4)),
  paste("H1（係数が負で信頼区間の上限が 0 未満）:",
        isTRUE(main_fit$row$log_adc_ci_high < 0)),
  "H2 の予測範囲は事前登録で空欄。推定値は ed_estimates.csv。",
  paste("遷移域（ADC 比 0.8〜1.5）の病変数は run_meta.json を参照。"),
  paste("DWI 比と ADC 比の相関（log）:", signif(dwi_cor, 3), " VIF:", signif(dwi_vif, 3)),
  "dataset を固定効果に加えた ED50 は ISLES を基準にした探索的な値。ADC 比とデータセットは相関する。",
  capture.output(print(table))
)
writeLines(summary_lines, file.path(out_dir, "model_summary.txt"))
cat(paste(summary_lines, collapse = "\n"), "\n")
