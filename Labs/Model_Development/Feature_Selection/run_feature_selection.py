#!/usr/bin/env python3
"""
Ungraded Lab: Feature Selection (Breast Cancer dataset)
Converted from Feature_Selection.ipynb, with submission-specific extensions.

Run from this folder:
    python run_feature_selection.py

Optional:
    python run_feature_selection.py --show
    python run_feature_selection.py --output-dir ./outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ModelKind = Literal["rf", "lr"]


def _lab_dir() -> Path:
    return Path(__file__).resolve().parent


def load_and_prepare(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    print("Column dtypes:\n", df.dtypes, sep="")

    columns_to_remove = ["Unnamed: 32", "id"]
    df = df.drop(columns=columns_to_remove, errors="ignore")

    df["diagnosis_int"] = (df["diagnosis"] == "M").astype(int)
    df = df.drop(columns=["diagnosis"])

    feature_cols = [c for c in df.columns if c != "diagnosis_int"]
    X = df[feature_cols]
    Y = df["diagnosis_int"]
    return df, X, Y


def fit_model(X_train: np.ndarray, Y_train: pd.Series, kind: ModelKind = "rf"):
    if kind == "rf":
        model = RandomForestClassifier(criterion="entropy", random_state=47)
    else:
        model = LogisticRegression(max_iter=5000, random_state=47, solver="lbfgs")
    model.fit(X_train, Y_train)
    return model


def calculate_metrics(model, X_test_scaled: np.ndarray, Y_test: pd.Series):
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(Y_test, y_pred)
    roc = roc_auc_score(Y_test, y_pred)
    prec = precision_score(Y_test, y_pred)
    rec = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    return acc, roc, prec, rec, f1


def train_and_get_metrics(X: pd.DataFrame, Y: pd.Series, kind: ModelKind = "rf"):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=123
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = fit_model(X_train_scaled, Y_train, kind)
    acc, roc, prec, rec, f1 = calculate_metrics(model, X_test_scaled, Y_test)
    return acc, roc, prec, rec, f1


def evaluate_model_on_features(X: pd.DataFrame, Y: pd.Series, kind: ModelKind = "rf") -> pd.DataFrame:
    acc, roc, prec, rec, f1 = train_and_get_metrics(X, Y, kind)
    return pd.DataFrame(
        [[acc, roc, prec, rec, f1, X.shape[1]]],
        columns=["Accuracy", "ROC", "Precision", "Recall", "F1 Score", "Feature Count"],
    )


def univariate_selection_f_test(X: pd.DataFrame, Y: pd.Series, feature_columns: pd.Index):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=123
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    selector = SelectKBest(f_classif, k=20)
    selector.fit(X_train_scaled, Y_train)
    feature_idx = selector.get_support()
    for name, included in zip(feature_columns, feature_idx):
        print(f"{name}: {included}")
    return feature_columns[feature_idx]


def univariate_selection_mutual_info(X: pd.DataFrame, Y: pd.Series, feature_columns: pd.Index, k: int = 20):
    """Filter method using mutual information (captures non-linear dependence; differs from ANOVA F-test)."""
    X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=123)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X_train_scaled, Y_train)
    feature_idx = selector.get_support()
    return feature_columns[feature_idx]


def run_rfe(
    X: pd.DataFrame,
    Y: pd.Series,
    feature_columns: pd.Index,
    *,
    base_kind: ModelKind = "rf",
    n_features: int = 20,
):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=123
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    if base_kind == "rf":
        base = RandomForestClassifier(criterion="entropy", random_state=47)
    else:
        base = LogisticRegression(max_iter=5000, random_state=47, solver="lbfgs")

    rfe = RFE(base, n_features_to_select=n_features)
    rfe.fit(X_train_scaled, Y_train)
    return feature_columns[rfe.get_support()]


def feature_importances_from_tree_based_model(X: pd.DataFrame, Y: pd.Series):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=123
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    model = RandomForestClassifier()
    model.fit(X_train_scaled, Y_train)

    plt.figure(figsize=(10, 12))
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.sort_values(ascending=False).plot(kind="barh")
    plt.tight_layout()
    return model


def select_features_from_model(model: RandomForestClassifier, feature_columns: pd.Index):
    selector = SelectFromModel(model, prefit=True, threshold=0.013)
    feature_idx = selector.get_support()
    return feature_columns[feature_idx]


def run_l1_regularization(X: pd.DataFrame, Y: pd.Series, feature_columns: pd.Index):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=123
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    selection = SelectFromModel(LinearSVC(C=1, penalty="l1", dual=False, random_state=0))
    selection.fit(X_train_scaled, Y_train)
    return feature_columns[selection.get_support()]


def savefig(path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def build_metrics_table(labels: list[str], frames: list[pd.DataFrame], model_label: str) -> pd.DataFrame:
    out = pd.concat(frames, axis=0)
    out.index = labels
    out.insert(0, "Model", model_label)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature selection lab (Breast Cancer) — script version.")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to breast_cancer_data.csv (default: ./data/breast_cancer_data.csv next to this script)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for saved figures (default: ./outputs next to this script)",
    )
    parser.add_argument("--show", action="store_true", help="Display matplotlib windows after saving")
    args = parser.parse_args()

    lab = _lab_dir()
    csv_path = args.data if args.data is not None else lab / "data" / "breast_cancer_data.csv"
    out_dir = args.output_dir if args.output_dir is not None else lab / "outputs"

    if not csv_path.is_file():
        raise SystemExit(f"Data file not found: {csv_path}")

    df, X, Y = load_and_prepare(csv_path)
    feature_columns = X.columns

    runs: list[tuple[str, pd.DataFrame]] = []

    print("\n--- Baseline: all features ---")
    runs.append(("All features", X))
    print(evaluate_model_on_features(X, Y, "rf").to_string())

    plt.figure(figsize=(20, 20))
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.PuBu)
    plt.tight_layout()
    savefig(out_dir / "correlation_full.png", args.show)

    cor_target = abs(cor["diagnosis_int"])
    relevant = cor_target[cor_target > 0.2]
    names = [idx for idx in relevant.index if idx != "diagnosis_int"]
    print("\nFeatures with |corr(diagnosis_int)| > 0.2:\n", names)

    print("\n--- Strong features (correlation filter) ---")
    runs.append(("Strong features", df[names]))
    print(evaluate_model_on_features(df[names], Y, "rf").to_string())

    plt.figure(figsize=(20, 20))
    new_corr = df[names].corr()
    sns.heatmap(new_corr, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    savefig(out_dir / "correlation_strong_features.png", args.show)

    plt.figure(figsize=(12, 10))
    subset_cols = ["perimeter_mean", "radius_worst", "perimeter_worst", "area_worst", "radius_mean"]
    sns.heatmap(df[subset_cols].corr(), annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    savefig(out_dir / "correlation_radius_subset.png", args.show)

    subset_feature_corr_names = [x for x in names if x not in ["radius_worst", "perimeter_worst", "area_worst"]]
    print("\n--- Subset features (drop redundant worst features) ---")
    runs.append(("Subset features", df[subset_feature_corr_names]))
    print(evaluate_model_on_features(df[subset_feature_corr_names], Y, "rf").to_string())

    print("\n--- Univariate: SelectKBest (f_classif, k=20) ---")
    univariate_f_names = univariate_selection_f_test(X, Y, feature_columns)
    runs.append(("F-test", df[univariate_f_names]))
    print(evaluate_model_on_features(df[univariate_f_names], Y, "rf").to_string())

    print("\n--- Univariate: SelectKBest (mutual_info_classif, k=20) ---")
    mi_names = univariate_selection_mutual_info(X, Y, feature_columns, k=20)
    runs.append(("Mutual information", df[mi_names]))
    print(evaluate_model_on_features(df[mi_names], Y, "rf").to_string())

    print("\n--- Wrapper: RFE with RandomForest (n=20) ---")
    rfe_rf_names = run_rfe(X, Y, feature_columns, base_kind="rf", n_features=20)
    runs.append(("RFE", df[rfe_rf_names]))
    print(evaluate_model_on_features(df[rfe_rf_names], Y, "rf").to_string())

    print("\n--- Wrapper: RFE with LogisticRegression (n=20) ---")
    rfe_lr_names = run_rfe(X, Y, feature_columns, base_kind="lr", n_features=20)
    runs.append(("RFE (LogisticRegression)", df[rfe_lr_names]))
    print(evaluate_model_on_features(df[rfe_lr_names], Y, "rf").to_string())

    print("\n--- Embedded: RandomForest feature_importances_ + SelectFromModel ---")
    rf_model = feature_importances_from_tree_based_model(X, Y)
    savefig(out_dir / "feature_importances_rf.png", args.show)
    feat_imp_names = select_features_from_model(rf_model, feature_columns)
    runs.append(("Feature Importance", df[feat_imp_names]))
    print(evaluate_model_on_features(df[feat_imp_names], Y, "rf").to_string())

    print("\n--- Embedded: L1 (LinearSVC) + SelectFromModel ---")
    l1_names = run_l1_regularization(X, Y, feature_columns)
    runs.append(("L1 Reg", df[l1_names]))
    print(evaluate_model_on_features(df[l1_names], Y, "rf").to_string())

    labels = [label for label, _ in runs]
    rf_frames = [evaluate_model_on_features(Xsub, Y, "rf") for _, Xsub in runs]
    lr_frames = [evaluate_model_on_features(Xsub, Y, "lr") for _, Xsub in runs]

    results_rf = build_metrics_table(labels, rf_frames, "RandomForest")
    results_lr = build_metrics_table(labels, lr_frames, "LogisticRegression")

    print("\n=== Summary: RandomForest (all feature sets) ===")
    print(results_rf.to_string())
    print("\n=== Summary: LogisticRegression (same feature sets) ===")
    print(results_lr.to_string())

    out_dir.mkdir(parents=True, exist_ok=True)
    results_rf.to_csv(out_dir / "metrics_summary_random_forest.csv", index=True)
    results_lr.to_csv(out_dir / "metrics_summary_logistic_regression.csv", index=True)
    print(f"\nSaved figure outputs under: {out_dir}")
    print(f"Saved: {out_dir / 'metrics_summary_random_forest.csv'}")
    print(f"Saved: {out_dir / 'metrics_summary_logistic_regression.csv'}")


if __name__ == "__main__":
    main()
