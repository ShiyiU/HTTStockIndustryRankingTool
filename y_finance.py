import pandas as pd
import os
import yfinance as yf


def quarterly_price_change(ticker: str, year: int, quarter: int) -> float:

    if quarter not in [1, 2, 3, 4]:
        raise ValueError("Quarter must be 1, 2, 3, or 4")

    quarter_dates = {
        1: ("01-01", "03-31"),
        2: ("04-01", "06-30"),
        3: ("07-01", "09-30"),
        4: ("10-01", "12-31"),
    }

    start_date = f"{year}-{quarter_dates[quarter][0]}"
    end_date = f"{year}-{quarter_dates[quarter][1]}"

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker} in Q{quarter} of {year}.")

    # Extract scalar values safely
    start_price = df["Close"].iloc[0].item()
    end_price = df["Close"].iloc[-1].item()

    pct_change = (end_price - start_price) / start_price * 100
    return round(pct_change, 2)


def get_price_change_multiple_quarters(
    ticker: str,
    years: list[int], # Updated to accept a list of years
    quarters: list[int]
):
    """
    Calculates and appends quarterly price changes for a ticker and multiple years/quarters
    to the master CSV file. Creates the file if it does not exist.
    """
    CSV_PATH = "quarterly_price_change.csv"
    new_rows = []

    # New outer loop for years
    for year in years:
        for quarter in quarters:
            try:
                price_change = quarterly_price_change(ticker, year, quarter)
            except ValueError as e:
                print(f"Skipping {ticker} Q{quarter} {year}: {e}")
                continue # Skip to the next quarter if data is not found

            new_rows.append({
                "Ticker": ticker,
                "Year": year,
                "Quarter": f"Q{quarter}",
                "Price Change (%)": price_change
            })

    new_df = pd.DataFrame(new_rows)
    
    if new_df.empty:
        return pd.DataFrame() # Return empty if no data was fetched

    # Load existing or create new
    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
        existing_df = pd.read_csv(CSV_PATH)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    # Prevent duplicate rows
    combined_df = combined_df.drop_duplicates(
        subset=["Ticker", "Year", "Quarter"],
        keep="last"
    )

    combined_df.to_csv(CSV_PATH, index=False)

    return combined_df


# Example usage with multiple years: 2023, 2024, and 2025
get_price_change_multiple_quarters(
    ticker="AAPL",
    years=[2023, 2024, 2025],
    quarters=[1, 2, 3, 4]
)