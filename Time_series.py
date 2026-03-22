import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_monthly_return(df, ticker, start_date=None):
    firm_data = df[df['Ticker'] == ticker].copy()
    firm_data['Date'] = pd.to_datetime(firm_data['Date'])
    firm_data = firm_data.sort_values('Date')

    if start_date is not None:
        firm_data = firm_data[firm_data['Date'] >= pd.to_datetime(start_date)]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(firm_data['Date'], firm_data['Monthly Return'], color='steelblue', linewidth=1.8, zorder=2)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

    ax.fill_between(firm_data['Date'], firm_data['Monthly Return'], 0,
                    where=(firm_data['Monthly Return'] >= 0), alpha=0.2, color='green', label='Positive')
    ax.fill_between(firm_data['Date'], firm_data['Monthly Return'], 0,
                    where=(firm_data['Monthly Return'] < 0), alpha=0.2, color='red', label='Negative')

    ax.set_title(f'{ticker} — Monthly Return', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Monthly Return (%)')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_category_returns(df, category, start_date=None):
    cat_data = df[df['Category'] == category].copy()
    cat_data['Date'] = pd.to_datetime(cat_data['Date'])
    cat_data = cat_data.sort_values('Date')

    if start_date is not None:
        cat_data = cat_data[cat_data['Date'] >= pd.to_datetime(start_date)]

    tickers = cat_data['Ticker'].unique()
    colors = ['steelblue', 'darkorange', 'green', 'red', 'purple']

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, ticker in enumerate(tickers):
        ticker_data = cat_data[cat_data['Ticker'] == ticker]
        ax.plot(ticker_data['Date'], ticker_data['Monthly Return'],
                color=colors[i], linewidth=1.5, label=ticker, zorder=2)

    mean_return = cat_data['Monthly Return'].mean()
    std_return = cat_data['Monthly Return'].std()
    ax.set_ylim(mean_return - 3 * std_return, mean_return + 3 * std_return)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(mean_return, color='gray', linewidth=1.0, linestyle=':', alpha=0.7, label=f'Mean ({mean_return:.2f}%)')
    ax.set_title(f'Category {category} — Monthly Returns', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Monthly Return (%)')
    ax.legend(title='Ticker')
    plt.tight_layout()
    plt.show()


def plot_returns(df, ticker=None, category=None, start_date=None):
    if ticker is None and category is None:
        raise ValueError("Must provide either --ticker or --category.")
    if ticker is not None and category is not None:
        raise ValueError("Provide either --ticker or --category, not both.")

    if ticker is not None:
        plot_monthly_return(df, ticker, start_date)
    else:
        plot_category_returns(df, category, start_date)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot monthly returns by ticker or category.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker", type=str, help="Ticker symbol, e.g. REPX")
    group.add_argument("--category", type=int, help="Category number (1–5)")

    parser.add_argument("--start-date", type=str, default=None, help="Earliest date to plot, e.g. 2020-01-01")
    parser.add_argument("--data", type=str, default="data.csv", help="Path to CSV file (default: data.csv)")

    args = parser.parse_args()

    df = pd.read_csv(args.data)

    plot_returns(df, ticker=args.ticker, category=args.category, start_date=args.start_date)