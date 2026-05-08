"""
Run SARIMAX and ARIMA analysis on oil industry stock data.

Outputs to a timestamped folder under results/:
  results/YYYYMMDD_HHMMSS/
    SARIMAX/
      Backtest/   {ticker}_backtest.png
      Forecast/   {ticker}_forecast.png
      sarimax_training_results.csv
      sarimax_cumulative_returns.csv
    ARIMA/
      Backtest/   {ticker}_backtest.png
      Forecast/   {ticker}_forecast.png
      arima_training_results.csv
      arima_cumulative_returns.csv
    cumulative_returns_combined.csv
"""

import itertools
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
TEST_SIZE      = 6
WINDOW_SIZE    = 20
FORECAST_STEPS = 2        # quarters ahead for final forecast
P_RANGE        = range(0, 5)
Q_RANGE        = range(0, 5)

ROOT       = Path(__file__).parent
DATA_PATH  = ROOT / "Data" / "oil_industry_data.json"
RUN_DIR    = ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")

SX_DIR           = RUN_DIR / "SARIMAX"
SX_BACKTEST_DIR  = SX_DIR / "Backtest"
SX_FORECAST_DIR  = SX_DIR / "Forecast"
AR_DIR           = RUN_DIR / "ARIMA"
AR_BACKTEST_DIR  = AR_DIR / "Backtest"
AR_FORECAST_DIR  = AR_DIR / "Forecast"

for d in [SX_BACKTEST_DIR, SX_FORECAST_DIR, AR_BACKTEST_DIR, AR_FORECAST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data Loading ───────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    oil_industry_data = json.load(f)


def load_data_for_ticker(ticker):
    try:
        df = pd.DataFrame.from_dict(oil_industry_data.get(ticker, {}), orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().resample("QE").mean()
        df = df.dropna(axis=1, how="all")
        df = df.interpolate(method="linear", limit_direction="both")
        return df
    except Exception:
        return None


def enough_data_points(ticker_data):
    return ticker_data is not None and len(ticker_data) >= 20


def check_stationary_category(column_data):
    adf = adfuller(column_data)
    kps = kpss(column_data)
    if adf[1] < 0.05 and kps[1] > 0.05:
        return "stationary"
    if adf[1] > 0.05 and kps[1] < 0.05:
        return "unit-root"
    return "other"


def check_arima_category(ticker_data):
    columns = ["OCF/BOED", "ROA", "CostExpenses/BOED", "Stock/Oil ZScore", "PGR"]
    counts = {"stationary": 0, "unit_root": 0, "other": 0}
    for col in columns:
        t = check_stationary_category(ticker_data[col])
        if t == "stationary":
            counts["stationary"] += 1
        elif t == "unit-root":
            counts["unit_root"] += 1
        else:
            counts["other"] += 1
    return max(counts, key=counts.get)


# ── Stock Classification ───────────────────────────────────────────────────────
print("Classifying stocks...")
unviable, unit_root_tickers, stationary_tickers, other_tickers = [], [], [], []

for stock_ticker in oil_industry_data.keys():
    td = load_data_for_ticker(stock_ticker)
    if not enough_data_points(td) or len(td.columns) == 0:
        unviable.append(stock_ticker)
        continue
    if check_stationary_category(td["Monthly Return"]) == "stationary":
        cat = check_arima_category(td)
        if cat == "stationary":
            stationary_tickers.append(stock_ticker)
        elif cat == "unit_root":
            unit_root_tickers.append(stock_ticker)
        else:
            other_tickers.append(stock_ticker)
    else:
        unviable.append(stock_ticker)

print(f"  Stationary : {stationary_tickers}")
print(f"  Unit-root  : {unit_root_tickers}")
print(f"  Unviable   : {len(unviable)} tickers")

ticker_data_stationary = {x: load_data_for_ticker(x) for x in stationary_tickers}
ticker_data_unit_root  = {x: load_data_for_ticker(x) for x in unit_root_tickers}


# ══════════════════════════════════════════════════════════════════════════════
# SARIMAX
# ══════════════════════════════════════════════════════════════════════════════

def sarimax_test_model(y_train, X_train, order):
    return SARIMAX(y_train, exog=X_train, order=order,
                   enforce_stationarity=False,
                   enforce_invertibility=False).fit(disp=False)


def sarimax_create_trainingset(ticker_data, window_size=20, test_size=6, min_train_rows=4):
    target_col = "Monthly Return"
    y = ticker_data[target_col]
    X = (ticker_data.drop(columns=[target_col, "BOED"])
         if "BOED" in ticker_data.columns
         else ticker_data.drop(columns=[target_col]))
    aligned_df = pd.concat([y, X.shift(1)], axis=1).dropna()
    y_aligned = aligned_df[target_col]
    X_aligned = aligned_df.drop(columns=[target_col])

    indices = np.array_split(np.arange(len(X_aligned)), 2)
    X_blocks = [X_aligned.iloc[idx] for idx in indices]
    y_blocks = [y_aligned.iloc[idx] for idx in indices]

    if test_size >= min(len(idx) for idx in indices):
        raise ValueError(f"Not enough data: min block {min(len(i) for i in indices)} <= test_size {test_size}")

    blocks = []
    for bid in range(2):
        test_start  = int(indices[bid][-test_size])
        train_start = max(0, test_start - window_size)
        X_train = X_aligned.iloc[train_start:test_start]
        y_train = y_aligned.iloc[train_start:test_start]
        if len(X_train) < min_train_rows:
            raise ValueError(f"Block {bid}: only {len(X_train)} training rows < {min_train_rows}")
        blocks.append({
            "X_train": X_train, "X_test": X_blocks[bid].iloc[-test_size:],
            "y_train": y_train, "y_test": y_blocks[bid].iloc[-test_size:],
        })

    return {"blocks": blocks,
            "last_known_financials": X_aligned.iloc[-1:],
            "y_aligned": y_aligned,
            "X_aligned": X_aligned}


def sarimax_sharpe(model_results, y_test, X_test):
    preds = model_results.get_forecast(steps=len(y_test), exog=X_test).predicted_mean
    ret = [y_test.iloc[i] / 100 if p > 0 else -y_test.iloc[i] / 100
           for i, p in enumerate(preds)]
    std = np.std(ret)
    return np.mean(ret) / std if std > 0 else 0


def train_sarimax(ticker_data, d_val, window_size=WINDOW_SIZE, test_size=TEST_SIZE):
    split = sarimax_create_trainingset(ticker_data, window_size, test_size)
    blocks     = split["blocks"]
    y_aligned  = split["y_aligned"]
    X_aligned  = split["X_aligned"]
    last_known = split["last_known_financials"]
    weights    = np.array([0.25, 0.75])

    best_sharpe, best_order = 0, None
    for order in itertools.product(P_RANGE, [d_val], Q_RANGE):
        try:
            models  = [sarimax_test_model(b["y_train"], b["X_train"], order) for b in blocks]
            sharpes = [sarimax_sharpe(models[i], blocks[i]["y_test"], blocks[i]["X_test"])
                       for i in range(2)]
            score = float(np.dot(sharpes, weights))
            if score > best_sharpe:
                best_sharpe, best_order = score, order
        except Exception:
            continue

    if best_order is None:
        raise ValueError("No valid SARIMAX order — all scored <= 0 or failed")

    print(f"    Best order: {best_order}  Sharpe: {best_sharpe:.4f}")

    backtest_data = []
    for b in blocks:
        try:
            m    = sarimax_test_model(b["y_train"], b["X_train"], best_order)
            pred = m.get_forecast(steps=len(b["y_test"]), exog=b["X_test"]).predicted_mean
            pred.index = b["y_test"].index
        except Exception:
            pred = pd.Series([np.nan] * len(b["y_test"]), index=b["y_test"].index)
        backtest_data.append({"actual": b["y_test"], "predicted": pred})

    final_model = sarimax_test_model(y_aligned, X_aligned, best_order)
    return last_known, y_aligned, final_model, backtest_data, best_order, best_sharpe


def plot_backtest(ticker, backtest_data, save_path, model_label):
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#1f77b4", "#2ca02c"]
    actuals = pd.concat([b["actual"] for b in backtest_data])
    lo, hi  = actuals.min(), actuals.max()
    pad     = (hi - lo) * 0.3 or 5
    ax.set_ylim(lo - pad, hi + pad)
    for i, b in enumerate(backtest_data):
        ax.plot(b["actual"].index,    b["actual"].values,    color=colors[i], marker="o",  label=f"Actual block {i+1}")
        ax.plot(b["predicted"].index, b["predicted"].values, color=colors[i], marker="x",  linestyle="--", label=f"Predicted block {i+1}")
        ax.axvline(b["actual"].index[0], color="gray", linestyle=":", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"{ticker} — {model_label} Backtest: Predicted vs Actual Monthly Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly Return (%)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def sarimax_forecast_and_plot(ticker, forecast_steps, last_known, y_aligned, model, save_path):
    future_X = pd.concat([last_known] * forecast_steps, ignore_index=True)
    forecast  = model.get_forecast(steps=forecast_steps, exog=future_X)
    fc_mean   = forecast.predicted_mean
    ci        = forecast.conf_int()
    dates     = pd.date_range(start=y_aligned.index[-1], periods=forecast_steps + 1, freq="QE")[1:]
    fc_mean.index = dates
    ci.index = dates

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_aligned.index, y_aligned, label="Historical", color="#1f77b4", marker="o")
    ax.plot(y_aligned.index, model.fittedvalues, label="Fitted", color="green")
    ax.plot(fc_mean.index, fc_mean, label="Forecast", color="#ff7f0e", linestyle="--", marker="x")
    ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="#ff7f0e", alpha=0.2, label="95% CI")
    ax.set_title(f"{ticker} — SARIMAX Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly Return (%)")
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    cum = (1 + pd.Series(fc_mean) / 100).cumprod() - 1
    return float(cum.iloc[-1] * 100)


# ── SARIMAX training loop ──────────────────────────────────────────────────────
print("\n── SARIMAX Training ─────────────────────────────────────────────────────")
sarimax_train_records = []
sx_last_known = {}
sx_y_aligned  = {}
sx_models     = {}

def _run_sarimax_group(ticker_dict, d_val):
    for ticker, td in ticker_dict.items():
        print(f"  {ticker}")
        try:
            last, ya, mdl, bt, order, sharpe = train_sarimax(td, d_val)
            sx_last_known[ticker] = last
            sx_y_aligned[ticker]  = ya
            sx_models[ticker]     = mdl
            sarimax_train_records.append({"Ticker": ticker, "Best Order": str(order),
                                          "Sharpe": round(sharpe, 4), "d_val": d_val, "Status": "OK"})
            plot_backtest(ticker, bt, SX_BACKTEST_DIR / f"{ticker}_backtest.png", "SARIMAX")
        except ValueError as e:
            print(f"    Skipping: {e}")
            sarimax_train_records.append({"Ticker": ticker, "Best Order": "SKIPPED",
                                          "Sharpe": None, "d_val": d_val, "Status": str(e)})

_run_sarimax_group(ticker_data_stationary, d_val=0)
_run_sarimax_group(ticker_data_unit_root,  d_val=1)

pd.DataFrame(sarimax_train_records).to_csv(SX_DIR / "sarimax_training_results.csv", index=False)
print("  Saved SARIMAX/sarimax_training_results.csv")

# ── SARIMAX forecasting loop ───────────────────────────────────────────────────
print("\n── SARIMAX Forecasting ──────────────────────────────────────────────────")
sx_cum_returns = {}
for ticker in [x for x in stationary_tickers + unit_root_tickers if x in sx_models]:
    try:
        cr = sarimax_forecast_and_plot(
            ticker, FORECAST_STEPS,
            sx_last_known[ticker], sx_y_aligned[ticker], sx_models[ticker],
            SX_FORECAST_DIR / f"{ticker}_forecast.png",
        )
        sx_cum_returns[ticker] = cr
        print(f"  {ticker}: {cr:.2f}%")
    except Exception as e:
        print(f"  {ticker}: Error — {e}")

df_sx = (pd.DataFrame(list(sx_cum_returns.items()), columns=["Ticker", "SARIMAX Cumulative Return (%)"])
           .sort_values("SARIMAX Cumulative Return (%)", ascending=False)
           .reset_index(drop=True))
df_sx.to_csv(SX_DIR / "sarimax_cumulative_returns.csv", index=False)
print("  Saved SARIMAX/sarimax_cumulative_returns.csv")


# ══════════════════════════════════════════════════════════════════════════════
# ARIMA
# ══════════════════════════════════════════════════════════════════════════════

def arima_test_model(y_train, order):
    return ARIMA(y_train, order=order).fit()


def arima_create_trainingset(ticker_data, window_size=20, test_size=6, min_train_rows=6):
    y = ticker_data["Monthly Return"].dropna()
    indices = np.array_split(np.arange(len(y)), 2)
    y_blocks = [y.iloc[idx] for idx in indices]

    if test_size >= min(len(idx) for idx in indices):
        raise ValueError(f"Not enough data: min block {min(len(i) for i in indices)} <= test_size {test_size}")

    blocks = []
    for bid in range(2):
        test_start  = int(indices[bid][-test_size])
        train_start = max(0, test_start - window_size)
        y_train = y.iloc[train_start:test_start]
        if len(y_train) < min_train_rows:
            raise ValueError(f"Block {bid}: only {len(y_train)} training rows < {min_train_rows}")
        blocks.append({"y_train": y_train, "y_test": y_blocks[bid].iloc[-test_size:]})

    return {"blocks": blocks, "y_aligned": y}


def arima_sharpe(model_results, y_test):
    preds = model_results.get_forecast(steps=len(y_test)).predicted_mean
    ret = [y_test.iloc[i] / 100 if p > 0 else -y_test.iloc[i] / 100
           for i, p in enumerate(preds)]
    std = np.std(ret)
    return np.mean(ret) / std if std > 0 else 0


def train_arima(ticker_data, d_val, window_size=WINDOW_SIZE, test_size=TEST_SIZE):
    split   = arima_create_trainingset(ticker_data, window_size, test_size)
    blocks  = split["blocks"]
    y_aligned = split["y_aligned"]
    weights = np.array([0.25, 0.75])

    best_sharpe, best_order = 0, None
    for order in itertools.product(P_RANGE, [d_val], Q_RANGE):
        try:
            models  = [arima_test_model(b["y_train"], order) for b in blocks]
            sharpes = [arima_sharpe(models[i], blocks[i]["y_test"]) for i in range(2)]
            score   = float(np.dot(sharpes, weights))
            if score > best_sharpe:
                best_sharpe, best_order = score, order
        except Exception:
            continue

    if best_order is None:
        raise ValueError("No valid ARIMA order — all scored <= 0 or failed")

    print(f"    Best order: {best_order}  Sharpe: {best_sharpe:.4f}")

    backtest_data = []
    for b in blocks:
        try:
            m    = arima_test_model(b["y_train"], best_order)
            pred = m.get_forecast(steps=len(b["y_test"])).predicted_mean
            pred.index = b["y_test"].index
        except Exception:
            pred = pd.Series([np.nan] * len(b["y_test"]), index=b["y_test"].index)
        backtest_data.append({"actual": b["y_test"], "predicted": pred})

    final_model = arima_test_model(y_aligned, best_order)
    return y_aligned, final_model, backtest_data, best_order, best_sharpe


def arima_forecast_and_plot(ticker, forecast_steps, y_aligned, model, save_path):
    forecast = model.get_forecast(steps=forecast_steps)
    fc_mean  = forecast.predicted_mean
    ci       = forecast.conf_int()
    dates    = pd.date_range(start=y_aligned.index[-1], periods=forecast_steps + 1, freq="QE")[1:]
    fc_mean.index = dates
    ci.index = dates

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_aligned.index, y_aligned, label="Historical", color="#1f77b4", marker="o")
    ax.plot(y_aligned.index, model.fittedvalues, label="Fitted", color="green")
    ax.plot(fc_mean.index, fc_mean, label="Forecast", color="#ff7f0e", linestyle="--", marker="x")
    ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="#ff7f0e", alpha=0.2, label="95% CI")
    ax.set_title(f"{ticker} — ARIMA Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly Return (%)")
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    cum = (1 + pd.Series(fc_mean) / 100).cumprod() - 1
    return float(cum.iloc[-1] * 100)


# ── ARIMA training loop ────────────────────────────────────────────────────────
print("\n── ARIMA Training ───────────────────────────────────────────────────────")
arima_train_records = []
ar_y_aligned = {}
ar_models    = {}

def _run_arima_group(ticker_dict, d_val):
    for ticker, td in ticker_dict.items():
        print(f"  {ticker}")
        try:
            ya, mdl, bt, order, sharpe = train_arima(td, d_val)
            ar_y_aligned[ticker] = ya
            ar_models[ticker]    = mdl
            arima_train_records.append({"Ticker": ticker, "Best Order": str(order),
                                        "Sharpe": round(sharpe, 4), "d_val": d_val, "Status": "OK"})
            plot_backtest(ticker, bt, AR_BACKTEST_DIR / f"{ticker}_backtest.png", "ARIMA")
        except ValueError as e:
            print(f"    Skipping: {e}")
            arima_train_records.append({"Ticker": ticker, "Best Order": "SKIPPED",
                                        "Sharpe": None, "d_val": d_val, "Status": str(e)})

_run_arima_group(ticker_data_stationary, d_val=0)
_run_arima_group(ticker_data_unit_root,  d_val=1)

pd.DataFrame(arima_train_records).to_csv(AR_DIR / "arima_training_results.csv", index=False)
print("  Saved ARIMA/arima_training_results.csv")

# ── ARIMA forecasting loop ─────────────────────────────────────────────────────
print("\n── ARIMA Forecasting ────────────────────────────────────────────────────")
ar_cum_returns = {}
for ticker in [x for x in stationary_tickers + unit_root_tickers if x in ar_models]:
    try:
        cr = arima_forecast_and_plot(
            ticker, FORECAST_STEPS,
            ar_y_aligned[ticker], ar_models[ticker],
            AR_FORECAST_DIR / f"{ticker}_forecast.png",
        )
        ar_cum_returns[ticker] = cr
        print(f"  {ticker}: {cr:.2f}%")
    except Exception as e:
        print(f"  {ticker}: Error — {e}")

df_ar = (pd.DataFrame(list(ar_cum_returns.items()), columns=["Ticker", "ARIMA Cumulative Return (%)"])
           .sort_values("ARIMA Cumulative Return (%)", ascending=False)
           .reset_index(drop=True))
df_ar.to_csv(AR_DIR / "arima_cumulative_returns.csv", index=False)
print("  Saved ARIMA/arima_cumulative_returns.csv")


# ── Combined Rankings ──────────────────────────────────────────────────────────
print("\n── Combined Rankings ─────────────────────────────────────────────────────")
all_tickers = list(set(list(sx_cum_returns) + list(ar_cum_returns)))
df_combined = pd.DataFrame([{
    "Ticker":                        t,
    "SARIMAX Cumulative Return (%)": sx_cum_returns.get(t),
    "ARIMA Cumulative Return (%)":   ar_cum_returns.get(t),
} for t in all_tickers])
df_combined["Avg Cumulative Return (%)"] = df_combined[
    ["SARIMAX Cumulative Return (%)", "ARIMA Cumulative Return (%)"]
].mean(axis=1)
df_combined = df_combined.sort_values("Avg Cumulative Return (%)", ascending=False).reset_index(drop=True)
df_combined.to_csv(RUN_DIR / "cumulative_returns_combined.csv", index=False)
print("  Saved cumulative_returns_combined.csv")
print(df_combined.to_string(index=False))

print(f"\nAll outputs saved to: {RUN_DIR}")
