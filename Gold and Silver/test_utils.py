import pandas as pd
import numpy as np
import matplotlib.pyplot as PLT

def generate_mock_series(n_months, start_date="2019-01-01"):
    """Generates a dataframe with 'Period' and 'Balance' columns."""
    dates = pd.date_range(start=start_date, periods=n_months, freq='MS')
    
    # Create a 'YYYYMM' string format to test your parse_year_month function
    periods = [d.strftime('%Y%m') for d in dates]
    
    # Generate random balance data with a slight upward trend
    balance = 1000 + np.cumsum(np.random.normal(10, 5, size=n_months))
    
    return pd.DataFrame({
        'Fiscal Year Period': periods,
        'Balance': balance
    })

# Ensuring that this script will not come into effect accidentally (~Guarding)
if __name__ == "__main__":
    # Group A: 24 months (Only full history will run)
    group_a = generate_mock_series(24)
    # Group B: 84 months (7 years - will trigger analyze_both logic)
    group_b = generate_mock_series(84)

    # Visualization of the samples
    fig, axis = PLT.subplots(1,2, figsize = (16,10))

    axis[0].plot(group_a['Fiscal Year Period'], group_a['Balance'], label=f"Group 1, {len(group_a)} months")
    axis[0].set_title("First Group hich holds only 24 months/data points")
    axis[0].set_xlabel("Fiscal Year Period")
    axis[0].set_ylabel("Balance")
    axis[0].grid()
    axis[0].legend()
    
    axis[1].plot(group_b['Fiscal Year Period'], group_b['Balance'], label=f"Group 1, {len(group_b)} months")
    axis[1].set_title("First Group hich holds only 24 months/data points")
    axis[1].set_xlabel("Fiscal Year Period")
    axis[1].set_ylabel("Balance")
    axis[1].grid()
    axis[1].legend()

    PLT.show()

