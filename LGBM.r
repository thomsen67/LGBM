# Install if needed:
# install.packages("lightgbm")
# install.packages("dplyr")

library(lightgbm)
library(dplyr)


# ── 1. Load data ──────────────────────────────────────────────────────────────
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

col_names <- c("age", "workclass", "fnlwgt", "education", "education_num",
               "marital_status", "occupation", "relationship", "race", "sex",
               "capital_gain", "capital_loss", "hours_per_week",
               "native_country", "income")

df <- read.csv(url, header = FALSE, col.names = col_names,
               strip.white = TRUE, na.strings = "?")

# ── 2. Clean ──────────────────────────────────────────────────────────────────
df <- na.omit(df)

# Binary label: >50K = 1, <=50K = 0
df$label <- as.integer(df$income == ">50K")

# ── 3. Encode categoricals as integers ───────────────────────────────────────
cat_cols <- c("workclass", "education", "marital_status", "occupation",
              "relationship", "race", "sex", "native_country")

for (col in cat_cols) {
  df[[col]] <- as.integer(factor(df[[col]])) - 1L  # 0-based integer codes
}

# ── 4. Feature matrix ─────────────────────────────────────────────────────────
feature_cols <- c("age", "workclass", "fnlwgt", "education", "education_num",
                  "marital_status", "occupation", "relationship", "race", "sex",
                  "capital_gain", "capital_loss", "hours_per_week", "native_country")

X <- as.matrix(df[, feature_cols])
y <- df$label

# ── 5. Train / test split ─────────────────────────────────────────────────────
set.seed(42)
idx        <- sample(nrow(X), 0.8 * nrow(X))
X_train    <- X[idx, ];   y_train <- y[idx]
X_test     <- X[-idx, ];  y_test  <- y[-idx]

# ── 6. LightGBM datasets ──────────────────────────────────────────────────────
# categorical_feature uses 0-based column indices within X
cat_indices <- which(feature_cols %in% cat_cols) - 1L  # 0-based

dtrain <- lgb.Dataset(X_train, label = y_train,
                      categorical_feature = cat_indices)
dtest  <- lgb.Dataset(X_test,  label = y_test,
                      categorical_feature = cat_indices,
                      reference = dtrain)

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

# ── 8. Train with early stopping ──────────────────────────────────────────────
model <- lgb.train(
  params            = params,
  data              = dtrain,
  nrounds           = 300,
  valids            = list(train = dtrain, test = dtest),
  early_stopping_rounds = 20,
  eval_freq         = 50
)

cat("\nBest iteration:", model$best_iter, "\n")
cat("Best AUC (test):", model$best_score, "\n")

# ── 9. Evaluate ───────────────────────────────────────────────────────────────
probs     <- predict(model, X_test)
predicted <- as.integer(probs >= 0.5)

accuracy  <- mean(predicted == y_test)
tp        <- sum(predicted == 1 & y_test == 1)
fp        <- sum(predicted == 1 & y_test == 0)
tn        <- sum(predicted == 0 & y_test == 0)
fn        <- sum(predicted == 0 & y_test == 1)
precision <- tp / (tp + fp)
recall    <- tp / (tp + fn)
f1        <- 2 * precision * recall / (precision + recall)

cat("\n── Metrics ─────────────────────────────────────────\n")
cat(sprintf("  Accuracy  : %.4f\n", accuracy))
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

# ── 11. Save model ────────────────────────────────────────────────────────────
lgb.save(model, "adult_model.txt")
cat("\nSaved: adult_model.txt\n")