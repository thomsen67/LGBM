library(lightgbm)

# ── 1. Load directly from UCI ─────────────────────────────────────────────────
col_names <- c("age", "workclass", "fnlwgt", "education", "education_num",
               "marital_status", "occupation", "relationship", "race", "sex",
               "capital_gain", "capital_loss", "hours_per_week",
               "native_country", "income")

df <- read.csv(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
  header      = FALSE,
  col.names   = col_names,
  strip.white = TRUE,
  na.strings  = "?"
)

# ── 2. Clean ──────────────────────────────────────────────────────────────────
df <- na.omit(df)
df$label <- as.integer(df$income == ">50K")

# ── 3. Encode categoricals as 0-based integers ────────────────────────────────
cat_cols <- c("workclass", "education", "marital_status", "occupation",
              "relationship", "race", "sex", "native_country")

for (col in cat_cols) {
  df[[col]] <- as.integer(factor(df[[col]])) - 1L
}

# ── 4. Build feature matrix ───────────────────────────────────────────────────
feature_cols <- c("age", "workclass", "fnlwgt", "education", "education_num",
                  "marital_status", "occupation", "relationship", "race", "sex",
                  "capital_gain", "capital_loss", "hours_per_week", "native_country")

X <- as.matrix(df[, feature_cols])
storage.mode(X) <- "double"
y <- df$label

# ── 5. Train/test split ───────────────────────────────────────────────────────
set.seed(42)
idx     <- sample(nrow(X), 0.8 * nrow(X))
X_train <- X[idx,  ];  y_train <- y[idx]
X_test  <- X[-idx, ];  y_test  <- y[-idx]

# ── 6. Build datasets then explicitly set categoricals by NAME ────────────────
dtrain <- lgb.Dataset(X_train, label = y_train, free_raw_data = FALSE)
dtest  <- lgb.Dataset(X_test,  label = y_test,  free_raw_data = FALSE,
                      reference = dtrain)

# This is the correct way — explicit by column name, no heuristics involved
lgb.Dataset.set.categorical(dtrain, cat_cols)
lgb.Dataset.set.categorical(dtest,  cat_cols)

# ── 7. Parameters ─────────────────────────────────────────────────────────────
params <- list(
  objective        = "binary",
  metric           = "auc",
  num_leaves       = 31,
  learning_rate    = 0.05,
  feature_fraction = 0.9,
  bagging_fraction = 0.8,
  bagging_freq     = 5,
  verbose          = -1
)

# ── 8. Train ──────────────────────────────────────────────────────────────────
model <- lgb.train(
  params                = params,
  data                  = dtrain,
  nrounds               = 300,
  valids                = list(train = dtrain, test = dtest),
  early_stopping_rounds = 20,
  eval_freq             = 50
)

cat(sprintf("\nBest iteration : %d\n", model$best_iter))
cat(sprintf("Best AUC (test): %.4f\n", model$best_score))

# ── 9. Evaluate ───────────────────────────────────────────────────────────────
probs <- predict(model, X_test)
preds <- as.integer(probs >= 0.5)

tp <- sum(preds == 1 & y_test == 1)
fp <- sum(preds == 1 & y_test == 0)
tn <- sum(preds == 0 & y_test == 0)
fn <- sum(preds == 0 & y_test == 1)

precision <- tp / (tp + fp)
recall    <- tp / (tp + fn)
f1        <- 2 * precision * recall / (precision + recall)

cat("\n── Metrics ──────────────────────────────────────────\n")
cat(sprintf("  Accuracy  : %.4f\n", mean(preds == y_test)))
cat(sprintf("  Precision : %.4f\n", precision))
cat(sprintf("  Recall    : %.4f\n", recall))
cat(sprintf("  F1        : %.4f\n", f1))

cat("\n── Confusion Matrix ─────────────────────────────────\n")
cat(sprintf("  TP: %d  FP: %d\n", tp, fp))
cat(sprintf("  FN: %d  TN: %d\n", fn, tn))

# ── 10. Feature importance ────────────────────────────────────────────────────
imp <- lgb.importance(model, percentage = TRUE)
cat("\n── Feature Importance (top 10) ──────────────────────\n")
print(head(imp, 10))

# ── 11. Save ──────────────────────────────────────────────────────────────────
lgb.save(model, "adult_model.txt")
cat("\nSaved: adult_model.txt\n")

# ── 12. Sanity check ──────────────────────────────────────────────────────────
lines     <- readLines("adult_model.txt")
info_line <- lines[grep("^feature_infos", lines)]
feature_infos <- strsplit(info_line, " ")[[1]]

cat("\n── feature_infos per column ─────────────────────────\n")
for (i in seq_along(feature_cols)) {
  info    <- feature_infos[i]
  is_cont <- startsWith(info, "[")
  cat(sprintf("  %-20s : %s  %s\n",
              feature_cols[i],
              ifelse(is_cont, "continuous ✓", "categorical ✓"),
              substr(info, 1, 30)))
}


# ==============================================

# 1. Create the same sample vector used in C#
# These must match the exact 0-based integer encoding from your training
sample_data <- c(
  39,         # age
  6,          # workclass (e.g., 'Private' might have mapped to 6)
  77516,      # fnlwgt
  9,          # education
  13,         # education_num
  4,          # marital_status
  1,          # occupation
  1,          # relationship
  4,          # race
  1,          # sex
  2174,       # capital_gain
  0,          # capital_loss
  40,         # hours_per_week
  38          # native_country
)

# 2. Convert to a 1-row matrix (LightGBM predicts on matrices)
sample_matrix <- matrix(sample_data, nrow = 1)
colnames(sample_matrix) <- feature_cols

# 3. Get the Probability (Sigmoid output)
prob <- predict(model, sample_matrix)

# 4. Get the Raw Score (Logit/Leaf Sum)
# This is what C#'s GetOutput() returns before the sigmoid
#raw_score <- predict(model, sample_matrix, raw = TRUE)
#cat("--- R Prediction Results ---\n")
#at(sprintf("Raw Score (Logit): %f\n", raw_score))
cat(sprintf("Probability      : %f\n", prob))