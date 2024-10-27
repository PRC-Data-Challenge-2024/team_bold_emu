install.packages(c("tidyverse", "caret", "ParBayesianOptimization", 
                   "doParallel", "foreach", "Matrix", "minioclient", "lubridate"))


# Load necessary libraries
library(tidyverse)
library(minioclient)
library(caret)
library(ParBayesianOptimization)
library(doParallel)
library(foreach)
library(Matrix)
library(lubridate)



#remove.packages("xgboost")
xgboost_url <- "https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/release_2.0.0/xgboost_r_gpu_linux_82d846bbeb83c652a0b1dff0e3519e67569c4a3d.tar.gz"
install.packages(xgboost_url, repos = NULL, type = "source")

library(xgboost)


setwd("~/R")


# ---------------------------------
# Second Model (as per original code)
# ---------------------------------

# Load data for the second model
all_data_second <- read.csv("challenge_set_v6.csv") |>
  mutate(date = as.numeric(date(date)))

# Remove flights with missing 'tow'
all_data_second <- all_data_second[!is.na(all_data_second$tow), ]

# Dividing categorical and numeric columns for second model
categorical_col_second <- c("aircraft_type", "country_code_adep", "country_code_ades", "airline", "callsign", "adep", "ades")
numeric_col_second <- c("flight_duration", "flown_distance", "ground_speed_at_lift_off", "ground_speed_delta", "taxiout_time", "time_to_lift_off", "avg_altitude", "jet_stream_coeff", "date",
                        "departure_temp", "arrival_temp", "u_wind", "v_wind", "vertical_ascend", "vertical_descend", "humidity_diff")

# Select relevant columns including the target variable
all_data_second_filtered <- all_data_second %>%
  select(all_of(c(categorical_col_second, numeric_col_second, "flight_id", "tow")))

# Prepare the target variable
target_column_second <- all_data_second_filtered$tow

# Split data into training and test sets
set.seed(123)  # Ensure reproducibility
train_index_second <- createDataPartition(target_column_second, p = 0.8, list = FALSE)

# Creating training and testing datasets
train_data_second <- all_data_second_filtered[train_index_second, ]
test_data_second <- all_data_second_filtered[-train_index_second, ]

# Feature Engineering and Preprocessing for second model

# Handling categorical data (one-hot encode) and create sparse matrix
train_sparse_categorical_second <- sparse.model.matrix(~ . - 1, data = train_data_second[categorical_col_second])
test_sparse_categorical_second <- sparse.model.matrix(~ . - 1, data = test_data_second[categorical_col_second])

# Handling numerical data (standardization)
preProcValues_second <- preProcess(train_data_second[numeric_col_second], method = c("center", "scale"))
train_numeric_standardized_second <- predict(preProcValues_second, train_data_second[numeric_col_second])
test_numeric_standardized_second <- predict(preProcValues_second, test_data_second[numeric_col_second])

# Convert numeric data to sparse matrix
train_sparse_numeric_second <- Matrix(as.matrix(train_numeric_standardized_second), sparse = TRUE)
test_sparse_numeric_second <- Matrix(as.matrix(test_numeric_standardized_second), sparse = TRUE)

# Combine both sparse matrices
X_train_sparse_second <- cbind(train_sparse_categorical_second, train_sparse_numeric_second)
X_test_sparse_second <- cbind(test_sparse_categorical_second, test_sparse_numeric_second)

# Prepare target variables
y_train_second <- train_data_second$tow
y_test_second <- test_data_second$tow

# Define the Objective Function for Bayesian Optimization for second model

bayes_opt_function_second <- function(eta, max_depth, min_child_weight, subsample,
                                      colsample_bytree, gamma, nrounds, lambda, alpha) {
  
  # Create dtrain inside the function
  dtrain <- xgb.DMatrix(data = X_train_sparse_second, label = y_train_second)
  
  # Set parameters for XGBoost
  params <- list(
    objective = "reg:squarederror",
    device = "cuda",
    tree_method = "gpu_hist",  # Enable GPU acceleration
    eta = eta,
    max_depth = as.integer(max_depth),
    min_child_weight = min_child_weight,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    gamma = gamma,
    lambda = lambda,  # L2 regularization
    alpha = alpha,    # L1 regularization
    nthread = num_cores)
  
  # Perform cross-validation to estimate performance
  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = as.integer(nrounds),
    nfold = 5,  # Adjust the number of folds as needed
    early_stopping_rounds = 7,
    maximize = FALSE,
    eval_metric = "rmse",
    verbose = FALSE  # Set verbosity to FALSE to reduce output clutter
  )
  
  # Return a named list with the Score (negative RMSE)
  return(list(Score = -min(cv$evaluation_log$test_rmse_mean)))
  
}

# Bounds should be defined outside the function for optimization
bounds <- list(
  eta = c(0.01, 0.5),
  max_depth = c(8L, 15L),  # Must be integer
  min_child_weight = c(8, 15),
  subsample = c(0.5, 0.8),
  colsample_bytree = c(0.5, 0.8),
  gamma = c(0, 10),
  nrounds = c(200L, 750L),
  lambda = c(0, 10),    # L2 regularization
  alpha = c(0, 10)      # L1 regularization
)



num_cores <- detectCores() - 10

# Set Up Parallel Backend for Second Model
if (exists("cl")) {
  try(stopCluster(cl), silent = TRUE)
  rm(cl)
}
cl <- makeCluster(num_cores)
registerDoParallel(cl)
clusterExport(cl, varlist = c("X_train_sparse_second", "y_train_second", "num_cores", "bayes_opt_function_second", "bounds"))
clusterEvalQ(cl, {
  library(xgboost)
  library(caret)
  library(Matrix)
  library(ParBayesianOptimization)
})

# Run Bayesian Optimization for Second Model
set.seed(123)
bayes_opt_results_second <- bayesOpt(
  FUN = bayes_opt_function_second,
  bounds = bounds,
  initPoints = 10,
  iters.n = 70,
  acq = "ucb",
  kappa = 2.576,
  verbose = 1,
  parallel = TRUE
)

# Shut Down Parallel Backend for Second Model
stopCluster(cl)
registerDoSEQ()
rm(cl)


# Extract Best Parameters for second model
best_params_second <- as.data.frame(bayes_opt_results_second$scoreSummary) %>% arrange(desc(Score)) %>% head(1)

final_params_second <- list(
  objective = "reg:squarederror",
  device = "cuda",
  tree_method = "hist",
  eta = best_params_second$eta,
  max_depth = as.integer(best_params_second$max_depth),
  min_child_weight = best_params_second$min_child_weight,
  subsample = best_params_second$subsample,
  colsample_bytree = best_params_second$colsample_bytree,
  gamma = best_params_second$gamma,
  lambda = best_params_second$lambda,
  alpha = best_params_second$alpha,
  nthread = num_cores
)

# Ensure alignment of test data
common_cols <- intersect(colnames(X_train_sparse_second), colnames(X_test_sparse_second))
missing_in_test <- setdiff(colnames(X_train_sparse_second), colnames(X_test_sparse_second))
if (length(missing_in_test) > 0) {
  missing_mat <- Matrix(0, nrow = nrow(X_test_sparse_second), ncol = length(missing_in_test), sparse = TRUE)
  colnames(missing_mat) <- missing_in_test
  X_test_sparse_second <- cbind(X_test_sparse_second, missing_mat)
}
X_test_sparse_second <- X_test_sparse_second[, colnames(X_train_sparse_second)]

# Create DMatrix for Training and Testing
dtrain_second <- xgb.DMatrix(data = X_train_sparse_second, label = y_train_second)
dtest_second <- xgb.DMatrix(data = X_test_sparse_second, label = y_test_second)

# Train the Final Model for second model
set.seed(123)
final_model_second <- xgb.train(
  params = final_params_second,
  data = dtrain_second,
  nrounds = as.integer(best_params_second$nrounds),
  watchlist = list(train = dtrain_second, test = dtest_second),
  eval_metric = "rmse",
  early_stopping_rounds = 10,
  verbose = 1
)

# Extract and view the importance matrix
importance_matrix_second <- xgb.importance(
  feature_names = colnames(X_train_sparse_second),
  model = final_model_second
)
print(importance_matrix_second)

# Plot the feature importance
xgb.plot.importance(importance_matrix_second)

# Alternatively, using ggplot2
ggplot(importance_matrix_second, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  xlab("Features") +
  ylab("Importance (Gain)") +
  ggtitle("Feature Importance from Second Model") +
  theme_minimal()

# Save the model to a file named "my_xgb_model.model"
xgb.save(final_model_second, 'advanced.model')

# ---------------------------------
# Third Model
# ---------------------------------

# Load data for the third model

all_data_second_cut <- read.csv("challenge_set_v6.csv") |> select("flight_id", "aircraft_type", "country_code_adep", "country_code_ades", "airline", "callsign", "adep", "ades",
                                                 "flight_duration", "flown_distance", "taxiout_time", "MSCI.Adj.Close", "Oil.Adj.Close", "date", "tow") |>
  mutate(date = as.numeric(date(date)))


# Remove flights with missing 'tow'
all_data_third <- all_data_second_cut[!is.na(all_data_second_cut$tow), ]

# Dividing categorical and numeric columns for third model
categorical_col_third <- c("aircraft_type", "country_code_adep", "country_code_ades", "airline", "callsign", "adep", "ades")
numeric_col_third <- c("flight_duration", "flown_distance", "taxiout_time", "MSCI.Adj.Close", "Oil.Adj.Close", "date")

# Select relevant columns including the target variable
all_data_third_filtered <- all_data_third %>%
  select(all_of(c(categorical_col_third, numeric_col_third, "flight_id", "tow")))

# Prepare the target variable
target_column_third <- all_data_third_filtered$tow

# Split data into training and test sets
set.seed(123)  # Ensure reproducibility
train_index_third <- createDataPartition(target_column_third, p = 0.8, list = FALSE)

# Creating training and testing datasets
train_data_third <- all_data_third_filtered[train_index_third, ]
test_data_third <- all_data_third_filtered[-train_index_third, ]

# Feature Engineering and Preprocessing for third model

# Handling categorical data (one-hot encode) and create sparse matrix
train_sparse_categorical_third <- sparse.model.matrix(~ . - 1, data = train_data_third[categorical_col_third])
test_sparse_categorical_third <- sparse.model.matrix(~ . - 1, data = test_data_third[categorical_col_third])

# Handling numerical data (standardization)
preProcValues_third <- preProcess(train_data_third[numeric_col_third], method = c("center", "scale"))
train_numeric_standardized_third <- predict(preProcValues_third, train_data_third[numeric_col_third])
test_numeric_standardized_third <- predict(preProcValues_third, test_data_third[numeric_col_third])

# Convert numeric data to sparse matrix
train_sparse_numeric_third <- Matrix(as.matrix(train_numeric_standardized_third), sparse = TRUE)
test_sparse_numeric_third <- Matrix(as.matrix(test_numeric_standardized_third), sparse = TRUE)

# Combine both sparse matrices
X_train_sparse_third <- cbind(train_sparse_categorical_third, train_sparse_numeric_third)
X_test_sparse_third <- cbind(test_sparse_categorical_third, test_sparse_numeric_third)

# Prepare target variables
y_train_third <- train_data_third$tow
y_test_third <- test_data_third$tow

# Define the Objective Function for Bayesian Optimization for third model

bayes_opt_function_third <- function(eta, max_depth, min_child_weight, subsample,
                                     colsample_bytree, gamma, nrounds, lambda, alpha) {
  
  # Create dtrain inside the function
  dtrain <- xgb.DMatrix(data = X_train_sparse_third, label = y_train_third)
  
  # Set parameters for XGBoost
  params <- list(
    objective = "reg:squarederror",
    device = "cuda",
    tree_method = "gpu_hist",  # Enable GPU acceleration
    eta = eta,
    max_depth = as.integer(max_depth),
    min_child_weight = min_child_weight,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    gamma = gamma,
    lambda = lambda,  # L2 regularization
    alpha = alpha,    # L1 regularization
    nthread = num_cores     # Prevent over-subscription
  )
  
  # Perform cross-validation to estimate performance
  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = as.integer(nrounds),
    nfold = 5,  # Adjust the number of folds as needed
    early_stopping_rounds = 7,
    maximize = FALSE,
    eval_metric = "rmse",
    verbose = 1
  )
  
  
  
  
  # Return a named list with the Score (negative RMSE)
  list(Score = -min(cv$evaluation_log$test_rmse_mean))
}


# Set Up Parallel Backend for Third Model
if (exists("cl")) {
  try(stopCluster(cl), silent = TRUE)
  rm(cl)
}
cl <- makeCluster(num_cores)
registerDoParallel(cl)
clusterExport(cl, varlist = c("X_train_sparse_third", "y_train_third", "num_cores", "bayes_opt_function_third", "bounds"))
clusterEvalQ(cl, {
  library(xgboost)
  library(caret)
  library(Matrix)
  library(ParBayesianOptimization)
})

# Run Bayesian Optimization for Third Model
set.seed(123)
bayes_opt_results_third <- bayesOpt(
  FUN = bayes_opt_function_third,
  bounds = bounds,
  initPoints = 10,
  iters.n = 50,
  acq = "ucb",
  kappa = 2.576,
  verbose = 1,
  parallel = TRUE
)

# Shut Down Parallel Backend for Third Model
stopCluster(cl)
registerDoSEQ()
rm(cl)

# Extract Best Parameters for third model
best_params_third <- as.data.frame(bayes_opt_results_third$scoreSummary) %>% arrange(desc(Score)) %>% head(1)

final_params_third <- list(
  objective = "reg:squarederror",
  device = "cuda",
  tree_method = "hist",
  eta = best_params_third$eta,
  max_depth = as.integer(best_params_third$max_depth),
  min_child_weight = best_params_third$min_child_weight,
  subsample = best_params_third$subsample,
  colsample_bytree = best_params_third$colsample_bytree,
  gamma = best_params_third$gamma,
  lambda = best_params_third$lambda,
  alpha = best_params_third$alpha,
  nthread = num_cores
)

# Ensure alignment of test data
common_cols <- intersect(colnames(X_train_sparse_third), colnames(X_test_sparse_third))
missing_in_test <- setdiff(colnames(X_train_sparse_third), colnames(X_test_sparse_third))
if (length(missing_in_test) > 0) {
  missing_mat <- Matrix(0, nrow = nrow(X_test_sparse_third), ncol = length(missing_in_test), sparse = TRUE)
  colnames(missing_mat) <- missing_in_test
  X_test_sparse_third <- cbind(X_test_sparse_third, missing_mat)
}
X_test_sparse_third <- X_test_sparse_third[, colnames(X_train_sparse_third)]

# Create DMatrix for Training and Testing
dtrain_third <- xgb.DMatrix(data = X_train_sparse_third, label = y_train_third)
dtest_third <- xgb.DMatrix(data = X_test_sparse_third, label = y_test_third)

# Train the Final Model for third model
set.seed(123)
final_model_third <- xgb.train(
  params = final_params_third,
  data = dtrain_third,
  nrounds = as.integer(best_params_third$nrounds),
  watchlist = list(train = dtrain_third, test = dtest_third),
  eval_metric = "rmse",
  early_stopping_rounds = 10,
  verbose = 1
)


# Extract and view the importance matrix
importance_matrix_third <- xgb.importance(
  feature_names = colnames(X_train_sparse_third),
  model = final_model_third
)
print(importance_matrix_third)

# Plot the feature importance
xgb.plot.importance(head(importance_matrix_third))

# Alternatively, using ggplot2
ggplot(head(importance_matrix_third), aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  xlab("Features") +
  ylab("Importance (Gain)") +
  ggtitle("Feature Importance from third Model") +
  theme_minimal()

# Save the model to a file named "my_xgb_model.model"
xgb.save(final_model_third, 'barebones.model')


# ---------------------------------
# Predicting on the Submission Set
# ---------------------------------

# Read submission set
submission_set <- read.csv("final_submission_set_v6.csv") |>
  mutate(date = as.numeric(date(date)))


no_extra_data_submission <- read.csv("final_submission_set_v6_filtered.csv") |>
  mutate(date = as.numeric(date(date)))

# -------------------------------
# Predicting with Third Model
# -------------------------------


# Prepare submission_third data
submission_third_filtered <- no_extra_data_submission %>%
  select(all_of(c(categorical_col_third, numeric_col_third, "flight_id")))

# Handle categorical data
submission_sparse_categorical_third <- sparse.model.matrix(~ . - 1, data = submission_third_filtered[categorical_col_third])

# Handle numerical data
submission_numeric_standardized_third <- predict(preProcValues_third, submission_third_filtered[numeric_col_third])
submission_sparse_numeric_third <- Matrix(as.matrix(submission_numeric_standardized_third), sparse = TRUE)

# Combine both sparse matrices
submission_sparse_processed_third <- cbind(submission_sparse_categorical_third, submission_sparse_numeric_third)

# Align features
missing_cols_third <- setdiff(colnames(X_train_sparse_third), colnames(submission_sparse_processed_third))
if (length(missing_cols_third) > 0) {
  missing_mat <- Matrix(0, nrow = nrow(submission_sparse_processed_third), ncol = length(missing_cols_third), sparse = TRUE)
  colnames(missing_mat) <- missing_cols_third
  submission_sparse_processed_third <- cbind(submission_sparse_processed_third, missing_mat)
}
submission_sparse_processed_third <- submission_sparse_processed_third[, colnames(X_train_sparse_third)]

# Create DMatrix
dsubmission_third <- xgb.DMatrix(data = submission_sparse_processed_third)

# Predict
submission_predictions_third <- predict(final_model_third, dsubmission_third)

# Prepare results
submission_results_third <- data.frame(
  flight_id = submission_third_filtered$flight_id,
  tow = submission_predictions_third
)

# -------------------------------
# Predicting with Second Model
# -------------------------------

# Prepare submission_second data
submission_second_filtered <- submission_set


# Handle categorical data
submission_sparse_categorical_second <- sparse.model.matrix(~ . - 1, data = submission_second_filtered[categorical_col_second])

# Handle numerical data
submission_numeric_standardized_second <- predict(preProcValues_second, submission_second_filtered[numeric_col_second])
submission_sparse_numeric_second <- Matrix(as.matrix(submission_numeric_standardized_second), sparse = TRUE)

# Combine both sparse matrices
submission_sparse_processed_second <- cbind(submission_sparse_categorical_second, submission_sparse_numeric_second)

# Align features
missing_cols_second <- setdiff(colnames(X_train_sparse_second), colnames(submission_sparse_processed_second))
if (length(missing_cols_second) > 0) {
  missing_mat <- Matrix(0, nrow = nrow(submission_sparse_processed_second), ncol = length(missing_cols_second), sparse = TRUE)
  colnames(missing_mat) <- missing_cols_second
  submission_sparse_processed_second <- cbind(submission_sparse_processed_second, missing_mat)
}
submission_sparse_processed_second <- submission_sparse_processed_second[, colnames(X_train_sparse_second)]

# Create DMatrix
dsubmission_second <- xgb.DMatrix(data = submission_sparse_processed_second)

# Predict
submission_predictions_second <- predict(final_model_second, dsubmission_second)

# Prepare results
submission_results_second <- data.frame(
  flight_id = submission_second_filtered$flight_id,
  tow = submission_predictions_second
)


# -------------------------------
# Combine All Submission Results
# -------------------------------

submission_results <- bind_rows(submission_results_second, submission_results_third)

# Remove duplicates if any
submission_results <- submission_results %>% distinct()

# Save the submission results
write.csv(submission_results, "team_bold_emu_v3_5dd97dd4-790b-40d3-a19f-c1471b2e4bb0.csv", row.names = FALSE)


# -------------------------------
# Uploading Submission to MinIO
# -------------------------------



# Define your access key and secret key
access_key <- "ZG58zJvKhts2bkOX"
secret_key <- "eU95azmBpK82kg96mE0TNzsWov3OvP2d"

# Set the MinIO alias using system command
minio_alias_command <- paste(
  "mc alias set dc24 https://s3.opensky-network.org/",
  access_key,
  secret_key
)

# Run the command in the system
system(minio_alias_command)

# Verify if alias has been set correctly
system("mc alias ls")




# Define the local path to your submission file
local_file_path <- "team_bold_emu_v3_5dd97dd4-790b-40d3-a19f-c1471b2e4bb0.csv"

# Define the destination bucket and path in MinIO (e.g., dc24/submissions/)
minio_destination <- "dc24/submissions/team_bold_emu_v3_5dd97dd4-790b-40d3-a19f-c1471b2e4bb0.csv"

# Construct the command to upload the file
minio_upload_command <- paste("mc cp", shQuote(local_file_path), shQuote(minio_destination))

# Execute the command to upload the file to MinIO
result <- system(minio_upload_command)

# Check if the upload was successful
if (result == 0) {
  cat("File uploaded successfully to MinIO.\n")
} else {
  warning("Failed to upload the file. Please check the settings.")
  
}



