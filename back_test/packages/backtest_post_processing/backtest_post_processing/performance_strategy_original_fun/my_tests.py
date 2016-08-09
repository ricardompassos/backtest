import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import forecast_ts


def test_variables_random(n_points, indexes = False):
    """
    Creates test variables
    'n_points' is the number of observations
    'indexes' is an indication to include dates in the resulting dataframe or not
    """
    x = np.random.normal(0, 1, n_points)/100.0
    if indexes:
        rng = pd.date_range('1/1/1990', periods=n_points, freq='B')
        d = {'x' : x}
        df = pd.DataFrame(d, index = rng)
    else:
        d = {'x' : x}
        df = pd.DataFrame(d)

    return df


# GENERATE SIMPLE RANDOM DATA
N = 1000
df = test_variables_random(N, indexes = True)
returns = df['x']
df = test_variables_random(N, indexes = True)
benchmark_rets = df['x']


import ts_metrics
 
res_1 = ts_metrics.perf_stats(returns, factor_returns=None, period = 'daily', mode = 'per_year', window_length = int(5))


# mode = 'simple', 'per_year' or'rolling'

print type(res_1) 

print res_1

# import tears
# tears.create_full_tear_sheet(returns = returns,benchmark_rets = benchmark_rets, filename = 'ola.txt')
# plt.show()




