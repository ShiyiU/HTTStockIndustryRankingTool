#!/usr/bin/env python
"""
Turn a JSON export of stock indicators + monthly return into quick exploratory plots
and baseline models.

Usage (from repo root):
    python stock_analysis.py --input Data/oil_industry_data.json --target "Monthly Return"

Outputs go to ./analysis_output by default:
    - scatter_plots/*.png  (each feature vs target)
    - metrics.json         (train/test metrics for Linear Regression + Random Forest)
    - correlations.csv     (Pearson r vs target)
    - feature_importance.csv (RF feature importances)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


def _normalize_name(name: str) -> str:
    """Lowercase and strip non-alphanumerics to allow space/underscore/slash agnostic matching."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _resolve_target_column(df: pd.DataFrame, target_col: str) -> str:
    """Return the exact column name in df that matches target_col, tolerant to spaces/underscores/slashes."""
    normalized = {_normalize_name(c): c for c in df.columns}
    key = _normalize_name(target_col)
    if key not in normalized:
        raise ValueError(
            f"Target column '{target_col}' missing from data. "
            f"Available columns: {list(df.columns)}"
        )
    return normalized[key]


def load_json_table(path: Path, target_col: str) -> pd.DataFrame:
    """Load JSON to tabular form, handling both list-of-records and ticker->date->dict shapes."""
    with path.open() as f:
        data = json.load(f)

    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        # Detect if inner values themselves contain dict payloads (ticker -> date -> payload)
        has_nested_payload = False
        for date_map in data.values():
            if isinstance(date_map, dict) and any(isinstance(v, dict) for v in date_map.values()):
                has_nested_payload = True
                break

        if has_nested_payload:
            jdf = pd.read_json(path)
            stacked = jdf.stack()
            stacked = stacked[stacked.map(lambda x: isinstance(x, dict))]
            if stacked.empty:
                raise ValueError("Nested JSON detected but no dict payloads found.")
            records_df = pd.json_normalize(stacked.values)
            records_df.index = stacked.index  # MultiIndex: (Ticker, Date)
            records_df = records_df.reset_index()
            # stack returns level order (Ticker, Date); user wants column names swapped
            records_df.columns = ["Date", "Ticker"] + list(records_df.columns[2:])
            df = records_df
        else:
            records: List[Dict] = []
            for ticker, date_map in data.items():
                if not isinstance(date_map, dict):
                    continue
                rec = {"Ticker": ticker}
                rec.update({k: v for k, v in date_map.items() if v is not None})
                records.append(rec)
            if not records:
                raise ValueError("No dict-like payloads found in JSON structure.")
            df = pd.DataFrame(records)
    else:
        raise ValueError("Unsupported JSON shape; expected list or dict.")

    target_col = _resolve_target_column(df, target_col)

    # Keep rows that have the target
    df = df.dropna(subset=[target_col])
    if df.empty:
        raise ValueError(
            f"No rows contain the target column '{target_col}' after filtering out missing values."
        )
    # Keep only numeric predictors plus target; allow ticker/date for traceability
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    keep_cols = [c for c in numeric_cols if c != target_col] + [target_col]
    non_numeric_keep = [c for c in ["Ticker", "Date"] if c in df.columns]
    df = df[non_numeric_keep + keep_cols].copy()
    return df


def generate_scatter_plots(df: pd.DataFrame, target_col: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]

    for col in numeric_cols:
        x = df[col]
        y = df[target_col]

        # Percentile-based limits to reduce outlier influence
        x_low, x_high = x.quantile(0.05), x.quantile(0.95)
        y_low, y_high = y.quantile(0.05), y.quantile(0.95)

        x_pad = (x_high - x_low) * 0.1
        y_pad = (y_high - y_low) * 0.1

        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, alpha=0.5, color="steelblue", label="Actual")
        plt.xlim(x_low - x_pad, x_high + x_pad)
        plt.ylim(y_low - y_pad, y_high + y_pad)
        plt.xlabel(col)
        plt.ylabel(target_col)
        plt.title(f"{target_col} vs {col}")
        plt.grid(True)
        plt.legend()

        safe = col.replace(" ", "_").replace("/", "_")
        plt.savefig(out_dir / f"{safe}_vs_{target_col}.png", bbox_inches="tight", dpi=150)
        plt.close()


def train_models(df: pd.DataFrame, target_col: str, test_size: float, seed: int) -> Dict:
    feature_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=300, random_state=seed, n_jobs=-1, min_samples_leaf=2
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    def metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
        # Older sklearn may not support squared=False or mean_absolute_percentage_error
        mse = mean_squared_error(actual, pred)
        rmse = float(np.sqrt(mse))
        try:
            mape_val = mean_absolute_percentage_error(actual, pred)
        except Exception:
            actual_safe = np.where(np.abs(actual) < 1e-8, 1e-8, actual)
            mape_val = np.mean(np.abs((actual - pred) / actual_safe))
        return {
            "r2": r2_score(actual, pred),
            "rmse": rmse,
            "mae": mean_absolute_error(actual, pred),
            "mape": float(mape_val),
        }

    results = {
        "feature_columns": feature_cols,
        "linear_regression": metrics(y_test, lr_preds),
        "random_forest": metrics(y_test, rf_preds),
        "linear_coefficients": dict(zip(feature_cols, lr.coef_)),
        "rf_feature_importance": dict(zip(feature_cols, rf.feature_importances_)),
    }
    return results


def compute_correlations(df: pd.DataFrame, target_col: str) -> pd.Series:
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]
    return df[numeric_cols + [target_col]].corr()[target_col].drop(labels=[target_col])


def save_corr_heatmap(df: pd.DataFrame, target_col: str, out_dir: Path) -> None:
    """Save correlation heatmap between predictors (target excluded)."""
    predictors = [c for c in df.select_dtypes(include="number").columns if c != target_col]
    # drop predictors with too few non-null points or zero variance (corr would be NaN)
    filtered = []
    for c in predictors:
        series = df[c].dropna()
        if len(series) < 3:
            continue
        if series.std() == 0:
            continue
        filtered.append(c)

    if len(filtered) < 2:
        return  # not enough usable predictors for pairwise heatmap

    corr = df[filtered].corr()
    corr = corr.fillna(0)  # guard against any remaining NaNs

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="seismic", vmin=-1, vmax=1)
    ax.set_xticks(range(len(filtered)))
    ax.set_yticks(range(len(filtered)))
    ax.set_xticklabels(filtered, rotation=90, ha="center", fontsize=8)
    ax.set_yticklabels(filtered, fontsize=8)
    ax.set_title("Predictor Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated indicator vs monthly return analysis.")
    parser.add_argument("--input", required=True, help="Path to JSON file with indicators + target.")
    parser.add_argument(
        "--target",
        default="Monthly Return",
        help="Name of the target column (default: 'Monthly Return').",
    )
    parser.add_argument(
        "--outdir",
        default="analysis_output",
        help="Where to write plots and metrics (default: analysis_output/).",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    plots_dir = outdir / "scatter_plots"

    df = load_json_table(input_path, target_col=args.target)
    target_col = _resolve_target_column(df, args.target)

    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "clean_dataset.csv", index=False)

    generate_scatter_plots(df, target_col, plots_dir)
    save_corr_heatmap(df, target_col, outdir)

    results = train_models(df, target_col, args.test_size, args.seed)
    correlations = compute_correlations(df, target_col)

    (outdir / "metrics.json").write_text(json.dumps(results, indent=2))
    correlations.to_csv(outdir / "correlations.csv", header=["pearson_r"])

    # Feature importances as CSV for convenience
    pd.Series(results["rf_feature_importance"]).sort_values(ascending=False).to_csv(
        outdir / "feature_importance.csv", header=["importance"]
    )

    print(f"Saved scatter plots -> {plots_dir}")
    print(f"Saved metrics -> {outdir / 'metrics.json'}")
    print(f"Saved correlations -> {outdir / 'correlations.csv'}")
    print(f"Saved heatmap -> {outdir / 'correlation_heatmap.png'}")
    print(f"Saved feature importance -> {outdir / 'feature_importance.csv'}")


if __name__ == "__main__":
    main()
