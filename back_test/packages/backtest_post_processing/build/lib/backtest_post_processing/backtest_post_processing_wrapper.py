import pandas as pd
import ts_metrics
import tears
import utils
import matplotlib.pyplot as plt
import numpy as np

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


def generate_tables(returns, symbol = 'SPX INDEX', period = 'daily', mode = 'simple', window_length = 250):
  
    # returns = returns.resample('B', fill_method = 'pad')
    # returns = returns.resample('D') # I had to do this because of problems with the time hours
    # we never know....
    try:
        benchmark_rets = utils.default_returns_func(symbol=symbol)

    except:
        benchmark_rets = utils.default_returns_func(symbol='SPX')
    
    returns, benchmark_rets = returns.align(benchmark_rets,  join='inner')
    
    result = ts_metrics.perf_stats(returns, factor_returns=benchmark_rets, period = 'daily', mode = 'simple', window_length = window_length)

    return pd.DataFrame(result)




def generate_plots(returns, symbol = 'SPX INDEX'):
    
    try:
        benchmark_rets = utils.default_returns_func(symbol=symbol)
     
    except:

        benchmark_rets = utils.default_returns_func(symbol='SPY')

    returns, benchmark_rets = returns.align(benchmark_rets,  join='inner')    

    tears.create_full_tear_sheet(returns = returns,benchmark_rets = benchmark_rets, filename = None)
    
    plt.show()
    
    return pd.DataFrame([[1,2,3]], columns = ['a', 'b', 'c'])


if __name__ == '__main__':
    N = 1000
    df = test_variables_random(N, indexes = True)
    returns = df['x']

    # seila = generate_tables(returns)
    # print seila
    generate_plots(returns,symbol = 'SPX INDEX')


