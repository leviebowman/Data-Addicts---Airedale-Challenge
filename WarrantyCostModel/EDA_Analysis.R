data = read.csv("TrainingData.csv")

analyze_dataframe <- function(
    df,
    target = "Warranty Cost",
    exclude = character(),
    alpha = 0.05,
    min_complete = 20,
    min_abs_rho = 0.10,
    min_epsilon2 = 0.01,
    redundancy_cutoff = 0.90,
    max_category_levels = 30,
    top_n = 30,
    seed = 123
) {
  stopifnot(is.data.frame(df))
  
  if (anyDuplicated(names(df))) {
    stop("Column names must be unique.")
  }
  if (!target %in% names(df)) {
    stop("Target column not found: ", target)
  }
  if (!is.numeric(df[[target]])) {
    stop("Warranty Cost must be numeric.")
  }
  if (target %in% exclude) {
    stop("Do not include the target in exclude.")
  }
  
  # Preserve the caller's random-number state.
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) old_seed <- get(".Random.seed", envir = .GlobalEnv)
  
  on.exit({
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv,
                      inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  
  set.seed(seed)
  
  supported <- vapply(
    df,
    function(x) {
      is.numeric(x) || is.character(x) || is.factor(x) || is.logical(x)
    },
    logical(1)
  )
  
  usable <- names(df)[supported & !names(df) %in% exclude]
  dat <- df[usable]
  
  # Non-finite numeric values become missing.
  for (nm in names(dat)) {
    if (is.numeric(dat[[nm]])) {
      dat[[nm]][!is.finite(dat[[nm]])] <- NA_real_
    } else {
      dat[[nm]] <- factor(dat[[nm]])
    }
  }
  
  unique_counts <- vapply(
    dat,
    function(x) length(unique(x[!is.na(x)])),
    integer(1)
  )
  
  constant <- names(unique_counts)[unique_counts < 2]
  
  high_cardinality <- names(dat)[vapply(
    dat,
    function(x) {
      is.factor(x) && nlevels(droplevels(x)) > max_category_levels
    },
    logical(1)
  )]
  
  dropped <- unique(c(constant, high_cardinality))
  dat <- dat[setdiff(names(dat), dropped)]
  
  if (!target %in% names(dat)) {
    stop("Target has fewer than two distinct finite values.")
  }
  if (ncol(dat) < 2) {
    stop("No usable predictors remain.")
  }
  
  empty_results <- data.frame(
    variable_1 = character(),
    variable_2 = character(),
    test = character(),
    n = integer(),
    effect = numeric(),
    strength = numeric(),
    p_value = numeric(),
    note = character(),
    stringsAsFactors = FALSE
  )
  
  test_pair <- function(a, b) {
    x <- dat[[a]]
    y <- dat[[b]]
    
    keep <- complete.cases(x, y)
    x <- x[keep]
    y <- y[keep]
    n <- length(x)
    
    if (n < min_complete ||
        length(unique(x)) < 2 ||
        length(unique(y)) < 2) {
      return(NULL)
    }
    
    nx <- is.numeric(x)
    ny <- is.numeric(y)
    
    tryCatch({
      note <- ""
      
      if (nx && ny) {
        fit <- suppressWarnings(
          cor.test(x, y, method = "spearman", exact = FALSE)
        )
        
        test <- "Spearman"
        effect <- unname(fit$estimate)
        strength <- abs(effect)
        p <- fit$p.value
        
      } else if (xor(nx, ny)) {
        z <- if (nx) x else y
        g <- droplevels(factor(if (nx) y else x))
        k <- nlevels(g)
        
        if (n <= k) return(NULL)
        
        fit <- kruskal.test(z, g)
        
        test <- "Kruskal-Wallis"
        effect <- max(
          0,
          min(1, (unname(fit$statistic) - k + 1) / (n - k))
        )
        strength <- effect
        p <- fit$p.value
        
        if (min(table(g)) < 5) {
          note <- "Some categories have fewer than 5 observations."
        }
        
      } else {
        tab <- table(droplevels(factor(x)),
                     droplevels(factor(y)))
        
        initial <- suppressWarnings(chisq.test(tab, correct = FALSE))
        
        # Use Monte Carlo p-values when expected counts are sparse.
        sparse <- any(initial$expected < 5)
        
        fit <- if (sparse) {
          suppressWarnings(
            chisq.test(tab, simulate.p.value = TRUE, B = 9999)
          )
        } else {
          initial
        }
        
        test <- "Chi-squared"
        effect <- sqrt(
          unname(initial$statistic) /
            (n * min(nrow(tab) - 1, ncol(tab) - 1))
        )
        strength <- min(1, effect)
        p <- fit$p.value
        
        if (sparse) {
          note <- "Monte Carlo p-value; sparse expected counts."
        }
      }
      
      if (!is.finite(p) || !is.finite(strength)) return(NULL)
      
      data.frame(
        variable_1 = a,
        variable_2 = b,
        test = test,
        n = n,
        effect = effect,
        strength = strength,
        p_value = p,
        note = note,
        stringsAsFactors = FALSE
      )
    }, error = function(e) NULL)
  }
  
  pairs <- combn(names(dat), 2, simplify = FALSE)
  rows <- lapply(pairs, function(z) test_pair(z[1], z[2]))
  rows <- Filter(Negate(is.null), rows)
  
  if (!length(rows)) {
    stop("No pairs had sufficient usable data for analysis.")
  }
  
  all_results <- rbind(empty_results, do.call(rbind, rows))
  
  # FDR across every pair tested.
  all_results$q_value <- p.adjust(all_results$p_value, method = "BH")
  all_results <- all_results[
    order(all_results$q_value, -all_results$strength),
  ]
  rownames(all_results) <- NULL
  
  significant <- all_results[
    all_results$q_value <= alpha,
  ]
  
  # Analyze the target as its own multiple-testing family.
  target_results <- all_results[
    all_results$variable_1 == target |
      all_results$variable_2 == target,
  ]
  
  target_results$predictor <- ifelse(
    target_results$variable_1 == target,
    target_results$variable_2,
    target_results$variable_1
  )
  
  target_results$q_target <- p.adjust(
    target_results$p_value,
    method = "BH"
  )
  
  thresholds <- ifelse(
    target_results$test == "Spearman",
    min_abs_rho,
    min_epsilon2
  )
  
  target_results$passes_screen <- (
    target_results$q_target <= alpha &
      target_results$strength >= thresholds
  )
  
  # Effect sizes from different tests are not directly comparable.
  # Rank primarily by target-adjusted p-value.
  target_results <- target_results[
    order(target_results$q_target, -target_results$strength),
  ]
  rownames(target_results) <- NULL
  
  candidates <- target_results$predictor[
    target_results$passes_screen
  ]
  
  # Remove strongly redundant numeric-numeric or
  # categorical-categorical predictors, keeping the earlier-ranked one.
  # Mixed-type effect sizes use a different scale and are not pruned here.
  selected <- character()
  redundancy <- data.frame(
    removed = character(),
    retained = character(),
    strength = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (candidate in candidates) {
    redundant_with <- NULL
    
    for (retained in selected) {
      pair_result <- all_results[
        (all_results$variable_1 == candidate &
           all_results$variable_2 == retained) |
          (all_results$variable_1 == retained &
             all_results$variable_2 == candidate),
      ]
      
      if (nrow(pair_result) == 1 &&
          pair_result$test %in% c("Spearman", "Chi-squared") &&
          pair_result$strength >= redundancy_cutoff) {
        redundant_with <- retained
        
        redundancy <- rbind(
          redundancy,
          data.frame(
            removed = candidate,
            retained = retained,
            strength = pair_result$strength
          )
        )
        break
      }
    }
    
    if (is.null(redundant_with)) {
      selected <- c(selected, candidate)
    }
  }
  
  target_results$recommended <- (
    target_results$predictor %in% selected
  )
  
  # Missing predictor values still require handling inside each
  # training/cross-validation fold.
  model_data <- dat[
    !is.na(dat[[target]]),
    c(target, selected),
    drop = FALSE
  ]
  
  cat("\nRecommended predictors for ", target, ":\n", sep = "")
  if (length(selected)) {
    cat(paste0("  - ", selected, collapse = "\n"), "\n")
  } else {
    cat("  None passed the significance and effect-size thresholds.\n")
  }
  
  cat(
    "\nSignificant associations across all columns: ",
    nrow(significant), "\n",
    sep = ""
  )
  
  cat("\nTop associations:\n")
  print(head(significant, top_n), row.names = FALSE)
  
  cat("\nTarget predictor rankings:\n")
  print(
    target_results[
      ,
      c("predictor", "test", "n", "effect",
        "q_target", "recommended", "note")
    ],
    row.names = FALSE
  )
  
  invisible(list(
    recommended_predictors = selected,
    target_rankings = target_results,
    significant_associations = significant,
    all_associations = all_results,
    redundant_predictors = redundancy,
    excluded_columns = list(
      user_excluded = intersect(names(df), exclude),
      unsupported = names(df)[!supported],
      constant_or_empty = constant,
      high_cardinality_categorical = high_cardinality
    ),
    model_data = model_data
  ))
}

# Split your data first, then run screening on training data only.
# For time-dependent claims, use a chronological split instead.
set.seed(123)

results <- analyze_dataframe(
  data,
  target = "Warranty.Cost"
)

# Explicit list of recommended training factors:
results$recommended_predictors

# Why each predictor was selected or rejected:
results$target_rankings

# Training data containing the target and selected predictors:
training_data <- results$model_data