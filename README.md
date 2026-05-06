# HTT Stock Industry Ranking Tool

Ranks oil industry stocks by forecasted cumulative return using SARIMAX and ARIMA time series models trained on quarterly financial data.

---

## Setup

Install Python 3.12.2 and create a virtual environment:

```bash
winget install Python.Python.3.12
py --list
py -3.12 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To use the notebooks in VS Code, also install the Jupyter kernel:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name=venv-3.12.2 --display-name="Python 3.12.2 (Project)"
```

Then select the **Python 3.12.2 (Project)** kernel in VS Code.

---

## Running the Analysis Script

`run_analysis.py` runs both SARIMAX and ARIMA models end-to-end and saves all outputs to a timestamped results folder.

```bash
python run_analysis.py
```

### What it does

1. **Classifies stocks** — filters tickers from `Data/oil_industry_data.json` by data sufficiency (≥20 quarters) and stationarity (ADF + KPSS tests). Tickers are split into *stationary* (d=0) and *unit-root* (d=1) groups.

2. **Trains models** — for each ticker, runs 2-block sliding window cross-validation over a grid of ARIMA/SARIMAX orders (p=0–4, q=0–4). Each block trains on a fixed window of 20 quarters immediately before its 6-quarter test window. Orders are scored by a weighted Sharpe ratio (block 1: 25%, block 2: 75%). The best-scoring order is used to retrain on the full dataset.

3. **Backtests** — generates a predicted vs actual plot for each ticker's test blocks.

4. **Forecasts** — uses the final trained model to forecast 2 quarters ahead and computes the cumulative return.

### Configuration

At the top of `run_analysis.py`:

| Variable | Default | Description |
|---|---|---|
| `TEST_SIZE` | `6` | Quarters held out per test block |
| `WINDOW_SIZE` | `20` | Training window size (quarters) |
| `FORECAST_STEPS` | `2` | Quarters ahead to forecast |
| `P_RANGE` | `0–4` | Grid search range for AR order p |
| `Q_RANGE` | `0–4` | Grid search range for MA order q |

---

## Output Structure

Each run creates a timestamped folder under `results/`:

```
results/
└── YYYYMMDD_HHMMSS/
    ├── SARIMAX/
    │   ├── Backtest/
    │   │   └── {ticker}_backtest.png      Predicted vs actual for each test block
    │   ├── Forecast/
    │   │   └── {ticker}_forecast.png      Historical + 2-quarter ahead forecast
    │   ├── sarimax_training_results.csv   Best order, Sharpe score, skip reason per ticker
    │   └── sarimax_cumulative_returns.csv Tickers ranked by forecasted cumulative return
    ├── ARIMA/
    │   ├── Backtest/
    │   │   └── {ticker}_backtest.png
    │   ├── Forecast/
    │   │   └── {ticker}_forecast.png
    │   ├── arima_training_results.csv
    │   └── arima_cumulative_returns.csv
    └── cumulative_returns_combined.csv    Both models side-by-side, sorted by average
```

### CSV schemas

**`{model}_training_results.csv`**

| Column | Description |
|---|---|
| Ticker | Stock ticker |
| Best Order | Chosen (p, d, q) order |
| Sharpe | Weighted cross-validation Sharpe ratio |
| d_val | Differencing parameter used (0 = stationary, 1 = unit-root) |
| Status | `OK` or skip reason if insufficient data |

**`{model}_cumulative_returns.csv`**

| Column | Description |
|---|---|
| Ticker | Stock ticker |
| {Model} Cumulative Return (%) | Forecasted cumulative return over the forecast horizon |

**`cumulative_returns_combined.csv`**

| Column | Description |
|---|---|
| Ticker | Stock ticker |
| SARIMAX Cumulative Return (%) | SARIMAX forecast |
| ARIMA Cumulative Return (%) | ARIMA forecast |
| Avg Cumulative Return (%) | Average of both models (used for ranking) |

---

## Notebooks

| Notebook | Model | Notes |
|---|---|---|
| `Notebooks/Sarimax.ipynb` | SARIMAX | Uses lagged financial indicators as exogenous variables |
| `Notebooks/Arima.ipynb` | ARIMA | Univariate — monthly return only |

Both notebooks follow the same methodology as the script but display plots inline and allow step-by-step inspection.