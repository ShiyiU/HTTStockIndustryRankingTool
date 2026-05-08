# from tensorflow import Tensor
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller


def check_for_stationarity(X:pd.DataFrame, name:str, alpha=0.01):
    """
    The hypothetised assumption reads as follows: H_0: adfuller is unit rot exists (non-stationary)
    We must observe significant p-value to convince ourselves that the series is stationary
    """

    # 1. Retrieving the `p` value to confirm whether the test is statistically meaningfull
    # ... the alpha level can be fixed in place by 'cutoff' argument;
    pvalue = adfuller(X)[1]

    if pvalue < alpha:
        print("P_value =" + str(pvalue) + "the series " + name + " is likely staitonary")
        return True

    else:
        print("P_value =" + str(pvalue) + "the series " + name + " is likely non-staitonary")
        return False


