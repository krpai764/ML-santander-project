# ML-santander-project
# Santander Customer Transaction Prediction – README

## Project Overview

This repository contains a full pipeline solution for the **Santander Customer Transaction Prediction** challenge. The primary task is to build a robust machine learning model to predict whether a customer will make a particular transaction in the future, based on anonymized tabular data provided by Santander.

- **Task:** Predict the binary `target` outcome (0 or 1) for each customer in the test set, optimizing for the ROC-AUC metric.
- **Data:** The project uses `train.csv` and `test.csv`, each with 200+ anonymized features. The train set includes the `target` variable.
- **Pipeline Steps:** End-to-end data analysis, feature engineering, hyperparameter tuning (with both RandomizedSearchCV and Optuna), model training (LightGBM), interpretability, and automated submission file creation.

## Repository Structure

```
.
├── train.csv
├── test.csv
├── output1.html         # ydata_profiling report for train.csv
├── output2.html         # ydata_profiling report for test.csv
├── submission.csv       # Submission with all features
├── submission2.csv      # Submission with top-N selected features
├── santander_pipeline.py  # (Main pipeline script; code in this README)
├── README.md
```

## Project Workflow

### 1. Data Loading

- Loads train and test datasets from CSV (with error handling for file paths).
- Displays initial data shape and previews.

### 2. Exploratory Data Analysis (EDA)

- Generates detailed data profiling reports using `ydata_profiling` for both train and test data (`output1.html`, `output2.html`).
- Examines variable/feature distributions in both datasets **side-by-side** using seaborn histograms, enabling quick identification of distribution shifts.

### 3. Distribution Statistics (Comparative EDA)

- Calculates and visualizes the distribution of:
  - Column-wise and row-wise means, min, max
  - Skewness, standard deviation, kurtosis
  - These are all compared between train and test, as well as split by the `target` value within train.

- Uses insightful KDE and histogram plots to assess similarities and differences between datasets, and between classes.

### 4. Feature Engineering

- Creates additional **row-level aggregate features** (e.g., sum, mean, min, max, std, skew, kurtosis, median) for both train and test, leveraging 200 primary variable columns.
- These engineered features can help the model capture distributional properties for each customer.

### 5. Correlation Analysis

- Computes the Pearson correlation between each feature and the `target` variable.
- Helps identify individually predictive features and inspire initial feature selection.

### 6. Model Training & Hyperparameter Tuning

#### RandomizedSearchCV

- Sets up a robust LightGBM pipeline with a comprehensive hyperparameter grid.
- Uses 5-fold Stratified cross-validation with ROC-AUC as the evaluation metric.
- Runs a randomized search to efficiently find high-performing parameter combinations.

#### Optuna

- Implements Optuna for **Bayesian hyperparameter optimization**, making parameter search faster and smarter.
- Defines the search space to mirror the RandomizedSearchCV grid.
- Optimizes for ROC-AUC using early stopping.
- Reports the best parameters and corresponding validation ROC-AUC.

### 7. Final Model Training & Prediction

- Trains the final LightGBM model on the full training set using the best hyperparameters found.
- Predicts class probabilities for the test set and creates `submission.csv` (full feature model).

### 8. Feature Importance & Feature Selection

- Extracts LightGBM feature importance to rank all variable columns.
- Selects the top N important features (N=30 or N=50 typically).
- Retrains a new LightGBM model (`model2`) using only these top features.
- Generates and saves predictions in `submission2.csv` (top-feature model).

## Quick Start

1. **Clone the repo and add train/test data to the root folder.**
2. **Install required libraries:**
   ```bash
   pip install pandas numpy matplotlib seaborn lightgbm scikit-learn ydata-profiling optuna
   ```
3. **Run the main pipeline (in a Jupyter notebook or as a script):**
   ```python
   # See 'santander_pipeline.py' for full code, or adapt snippets from this README
   ```

4. **Key outputs:**
   - `output1.html`, `output2.html`: Data profiling.
   - `submission.csv`: Submission using all features.
   - `submission2.csv`: Submission using top features (by importance).

## Highlights

- **End-to-end feature analysis** (univariate, multivariate EDA, target grouping).
- **Visual and statistical comparison of train vs. test**.
- **Automated feature engineering** for improved model capability.
- **Dual approach to hyperparameter tuning** (RandomizedSearch + Optuna).
- **Built-in feature selection and model interpretability**.
- **Ready-to-submit CSVs** for competition evaluation—optimized for ROC-AUC.

## Customization Tips

- **Change the number of top features:** Set `N = 30` or `N = 50` (or different).
- **Tune model further:** Increase `n_trials` in Optuna for tighter hyperparameter search.
- **Add more features:** Consider using feature interactions, or different aggregation rules.
- **Switch LightGBM metrics:** The code is easily adaptable to other metrics if needed.

## Licensing and Credits

- **Data:** Provided by Santander through Kaggle.
- **Libraries:** LightGBM, pandas, numpy, matplotlib, seaborn, scikit-learn, optuna, ydata-profiling.
- **Notebook/Script developed as a complete solution to the Santander Customer Transaction Prediction Challenge.
