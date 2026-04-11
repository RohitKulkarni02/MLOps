# Feature Selection (Model Development Lab)

This folder studies supervised and unsupervised feature selection on the **Breast Cancer (Wisconsin)** CSV (`data/breast_cancer_data.csv`), following the original ungraded lab flow in `Feature_Selection.ipynb`.

## How to run

- **Notebook:** Open `Feature_Selection.ipynb` and run all cells (from the top so `X`, `Y`, `df`, and `results` exist before the submission section at the end).
- **Script (same workflow):** From this directory:

  ```bash
  python run_feature_selection.py
  ```

  Figures and metric CSVs are written under `outputs/` (ignored by git via `.gitignore`).

Install dependencies if needed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Changes made

The following changes are applied in **both** `Feature_Selection.ipynb` and `run_feature_selection.py`:

1. **Second classifier for evaluation:** `fit_model` / `train_and_get_metrics` / `evaluate_model_on_features` accept `kind='rf'` (default) or `kind='lr'`. **LogisticRegression** (`max_iter=5000`, `lbfgs`) is used to score the _same_ feature subsets as **RandomForest**, so you can compare how sensitive feature-selection conclusions are to the downstream model.

2. **Extra filter method — mutual information:** Added `SelectKBest(mutual_info_classif, k=20)` on scaled training data. This captures non-linear dependencies between features and label differently from the original ANOVA **F-test** filter, and may pick a different set of 20 features.

3. **Extra wrapper — RFE with LogisticRegression:** The original lab uses **RFE** only around `RandomForestClassifier`. This submission adds **RFE** with **`LogisticRegression`** as the base estimator (still `n_features_to_select=20`), exposing how the wrapper’s internal model changes the selected subset.

4. **Aggregated comparison tables:** After all feature sets are built, the script (and the final notebook cells) produce two summary tables: one with RandomForest metrics and one with LogisticRegression metrics, each row aligned to the same feature-selection method. The script saves them as `outputs/metrics_summary_random_forest.csv` and `outputs/metrics_summary_logistic_regression.csv`.

5. **Notebook maintenance (compatibility):** Replaced deprecated `DataFrame.append` with `pd.concat`, replaced `df.drop("diagnosis_int", 1)` with `df.drop(columns=["diagnosis_int"])` or `X.columns` where appropriate, and aligned `LinearSVC(..., random_state=0)` with the script for reproducibility.
