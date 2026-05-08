# from tensorflow import Tensor
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller


def check_for_Stationarity(x:pd.DataFrame, name:str, alpha:float=0.01, n_order:int = 0):
    """
    The hypothetised assumption reads as follows: H_0: adfuller is unit rot exists (non-stationary)
    We must observe significant p-value to convince ourselves that the series is stationary
    Integration order is by default 0 equivalent with testing for stationarity the RAW/AUTHENTIC series;
    
    Retrieving the `p` value to confirm whether the test is statistically meaningfull the alpha level can be fixed in place by 'cutoff' argument;
    """

    def _get_results(data):
        # Drop NaNs (crucial for differenced data)
        pvalue = adfuller(data.dropna())[1]
        if pvalue < alpha:
            print(f"P_value = {pvalue:.4f}: the series '{name}' is likely stationary")
            return True
        else:
            print(f"P_value = {pvalue:.4f}: the series '{name}' is likely non-stationary")
            return False

        # 1. Original level, I(0)
    if n_order == 0:
        return _get_results(x)

    # 2. First-order difference, I(1)
    elif n_order == 1:
        # Create a new variable to avoid modifying the original dataframe in place
        x_diff = x.diff()
        return _get_results(x_diff)
    
    return False # Default fallback

def check_for_Cointegraiton():
    pass

