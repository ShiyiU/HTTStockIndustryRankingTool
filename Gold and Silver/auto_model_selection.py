# -*- coding: utf-8 -*-# -*- coding: utf1-03."
def _ensure_list(obj):
    """Internal helper: ensure `obj` is returned as a list."""
    return obj if isinstance(obj, list) else [obj]


def auto_model_selection(
    groups: List[pd.DataFrame],
    *,
    time_col: str,
    target_col: str,
    fh: int = 12,
    loss_metric: str = "SMAPE",
    analyze_both: bool = False,
    min_months_5y: int = 60,
    n_select_7y: int = 10,
    n_select_5y: int = 5,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    fold: int = 3,
    fold_strategy: str = "rolling",
    plot: bool = False,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run automated model selection + forecasting for each group using PyCaret Time Series.

    This function initializes a separate `TSForecastingExperiment` per group,
    performs model comparison, and generates out-of-sample forecasts. Optionally,
    it can re-run the pipeline on a fixed 5-year (60 months) subset when enough
    observations are available.

    Parameters
    ----------
    groups : list of pd.DataFrame
        Each DataFrame is one time series group. Must contain `time_col` and `target_col`.
    time_col : str
        Name of the column containing year–month values (e.g., 202103, "2021-03").
    target_col : str
        Name of the numeric target column to forecast.
    fh : int, default=12
        Forecast horizon (number of future periods to predict).
    loss_metric : str, default="SMAPE"
        Metric used for ranking models in `compare_models`.
    analyze_both : bool, default=False
        If True and a group has >= `min_months_5y` observations, the function runs
        a second pipeline on the last 60 months (a "5-year" window), in addition
        to the full history.
    min_months_5y : int, default=60
        Minimum observations to consider the 5-year subset analysis.
    n_select_7y : int, default=10
        Number of top models to keep from `compare_models` on the full dataset.
    n_select_5y : int, default=5
        Number of top models to keep from `compare_models` on the 5-year subset.
    include : list[str] or None, default=None
        Model IDs to include in `compare_models` (PyCaret identifiers). If None, all eligible.
    exclude : list[str] or None, default=None
        Model IDs to exclude in `compare_models`.
    fold : int, default=3
        Number of folds for rolling cross-validation.
    fold_strategy : str, default="rolling"
        Fold strategy used by PyCaret (typically "rolling" for time series).
    plot : bool, default=False
        If True, produce forecast plots via `plot_model` for the top model(s).
    verbose : bool, default=False
        If True, PyCaret will print detailed logs during setup and training.

    Returns
    -------
    list of dict
        One dictionary per input group with these keys:
        - 'group_index': int
        - 'n_observations': int
        - 'experiment_full': TSForecastingExperiment
        - 'models_full': list[BaseEstimator]
        - 'forecast_full': pd.Series (or pd.DataFrame depending on PyCaret version)
        - 'actuals': pd.Series of the last `fh` true values (if available; else empty)
        - 'experiment_5y': TSForecastingExperiment or None
        - 'models_5y': list[BaseEstimator] or None
        - 'forecast_5y': pd.Series or None

    Notes
    -----
    - The function sets a monthly start frequency ("MS") on each group's index.
    - It does **not** mutate the input `groups`.
    - If `fh` exceeds the available observations, `actuals` will be empty.

    Examples
    --------
    >>> # Example groups
    >>> df1 = pd.DataFrame({'period': [202101, 202102, 202103], 'y': [10, 12, 14]})
    >>> df2 = pd.DataFrame({'period': ['2020-11', '2020-12', '2021-01'], 'y': [8, 9, 13]})
    >>> results = auto_model_selection(
    ...     groups=[df1, df2],
    ...     time_col='period',
    ...     target_col='y',
    ...     fh=6,
    ...     analyze_both=False,
    ... )
    """
    if not isinstance(groups, list) or not all(isinstance(g, pd.DataFrame) for g in groups):
        raise TypeError("`groups` must be a list of pandas DataFrames.")

    if not time_col or not target_col:
        raise ValueError("Both `time_col` and `target_col` must be provided.")

    results: List[Dict[str, Any]] = []

    for idx, group in enumerate(groups):
        # ---- 1) Validate & prepare data (copy to avoid in-place mutation)
        df = group.copy()

        if time_col not in df.columns:
            raise KeyError(f"Time column '{time_col}' not found in group {idx}.")

        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in group {idx}.")

        # Parse year–month to a monthly Timestamp and set as index
        df["__time_index__"] = df[time_col].apply(parse_year_month)
        df = df.set_index("__time_index__").sort_index()
        df = df.asfreq("MS")

        # Ensure target is numeric
        try:
            df[target_col] = pd.to_numeric(df[target_col], errors="raise")
        except Exception as exc:
            raise ValueError(
                f"Target column '{target_col}' in group {idx} must be numeric."
            ) from exc

        n_obs = len(df)

        # Keep only columns relevant for the experiment unless user wants exogenous vars.
        # For now, we pass the full df; PyCaret will treat non-target columns as exogenous
        # if supported by the chosen models. If you want only the target, uncomment line below.
        # df = df[[target_col]]

        # Compute "actuals" as the last `fh` observations if available
        actuals = df[target_col].iloc[-fh:] if fh <= n_obs else pd.Series(dtype=float)

        # ---- 2) Full-history experiment
        exp_full = TSForecastingExperiment()
        exp_full.setup(
            data=df,
            target=target_col,
            fh=fh,
            fold=fold,
            fold_strategy=fold_strategy,
            verbose=verbose,
        )

        models_full = _ensure_list(
            exp_full.compare_models(
                n_select=n_select_7y, sort=loss_metric, include=include, exclude=exclude
            )
        )
        # Forecast using the top-ranked model
        forecast_full = exp_full.predict_model(models_full[0])

        if plot:
            exp_full.plot_model(models_full[0], plot="forecast", data_kwargs={"fh": fh})

        # ---- 3) Optional 5-year subset experiment
        exp_5y = None
        models_5y = None
        forecast_5y = None

        if analyze_both and n_obs >= min_months_5y:
            subset = df.iloc[-min_months_5y:]  # last 60 months
            exp_5y = TSForecastingExperiment()
            exp_5y.setup(
                data=subset,
                target=target_col,
                fh=fh,
                fold=fold,
                fold_strategy=fold_strategy,
                verbose=verbose,
            )
            models_5y = _ensure_list(
                exp_5y.compare_models(
                    n_select=n_select_5y, sort=loss_metric, include=include, exclude=exclude
                )
            )
            forecast_5y = exp_5y.predict_model(models_5y[0])

            if plot:
                exp_5y.plot_model(models_5y[0], plot="forecast", data_kwargs={"fh": fh})

        # ---- 4) Collect results
        results.append(
            {
                "group_index": idx,
                "n_observations": n_obs,
                "experiment_full": exp_full,
                "models_full": models_full,
                "forecast_full": forecast_full,
                "actuals": actuals,
                "experiment_5y": exp_5y,
                "models_5y": models_5y,
                "forecast_5y": forecast_5y,
            }
        )

    return results
"""
A small utility module for monthly time series forecasting using PyCaret.

It provides:
1) `parse_year_month` — converts a "YYYYMM"-like value into a pandas Timestamp (1st day of month).
2) `auto_model_selection` — runs automated model selection and forecasting per group
   using PyCaret's time series `TSForecastingExperiment`.

Assumptions
-----------
- The time column contains a year-month representation (e.g., 202103, "2021-03", "2021/03", "202103").
- Series are monthly and should be indexed at monthly start ("MS").
- Each group (DataFrame) must contain at least the time column and a numeric target column.

Dependencies
------------
- pandas, numpy
- pycaret >= 3 (time series module)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pycaret.time_series import TSForecastingExperiment


def parse_year_month(value: Any) -> pd.Timestamp:
    """
    Convert a year–month value into a pandas Timestamp at the first day of that month.

    Accepted formats (examples):
        - 202103      (int or str with 6 digits)
        - "2021-03"
        - "2021/03"
        - "2021 03"
        - "2021_03"

    Parameters
    ----------
    value : Any
        A value representing year and month.

    Returns
    -------
    pd.Timestamp
        Timestamp with day=1.

    Raises
    ------
    ValueError
        If a year–month pattern cannot be identified.

    Examples
    --------
    >>> parse_year_month(202103)
    Timestamp('2021-03-01 00:00:00')
    >>> parse_year_month("2021-03")
    Timestamp('2021-03-01 00:00:00')
    """
    s = str(value).strip()

    # Normalize common "YYYY<sep>MM" forms by removing separators, then parse YYYYMM.
    normalized = re.sub(r"[^\d]", "", s)  # keep only digits
    if re.fullmatch(r"\d{6}", normalized):
        year = int(normalized[:4])
        month = int(normalized[4:])
        return pd.Timestamp(year=year, month=month, day=1)

    # If not matched, fail fast with a helpful message.
    raise ValueError(
