from __future__ import division

from functools import partial
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy as sp
from sklearn import preprocessing

import utils
from interesting_periods import PERIODS

APPROX_BDAYS_PER_MONTH = 21
APPROX_BDAYS_PER_YEAR = 250
QUARTERS_PER_YEAR = 4
MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52

DAILY = 'daily'
WEEKLY = 'weekly'
MONTHLY = 'monthly'
QUARTERLY = 'quaterly'
YEARLY = 'yearly'

ANNUALIZATION_FACTORS = {
    DAILY: APPROX_BDAYS_PER_YEAR,
    WEEKLY: WEEKS_PER_YEAR,
    MONTHLY: MONTHS_PER_YEAR,
    QUARTERLY: QUARTERS_PER_YEAR,
    YEARLY: 1
}

def infer_periodicity(df):
    """
    Given a pd.Series or pd.DataFrame whose indices are dates, infer its
    periodicity.
    Parameters
    ----------
    df : pd.Series / pd.DataFrame
        DataFrame containing the data (and dates as index) whose periodicity
        is to be inferred.
    Returns
    -------
    period : str
        Output can be: 'daily', 'weekly', 'quarterly' or 'yearly'.
    Additional Notes
    ----------------
    Correct inference is not always guaranteed.
    """
    f  = np.median(np.diff(df.index.values))
    days = f.astype('timedelta64[D]').item().days
    thresholds = np.array([1, 7, 31, 91])
    periods = [DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY]
    return periods[np.sum(days>thresholds)]

def per_year(func):
  """
  Decorator to apply functions on a year basis.
  """
  def wrapper(*args, **kw):
      kw_copy = kw.copy()
      if len(args):
          rets = args[0]
          args = args[1:]
      else:
          rets = kw['returns']
          del kw_copy['returns']
      if 'factor_returns' in func.__code__.co_varnames[:func.__code__.co_argcount]:
          if len(args):
              factor_rets = args[0]
              args = args[1:]
          else:
              factor_rets = kw_copy['factor_returns']
              del kw_copy['factor_returns']
          main_df_ = pd.concat([rets, factor_rets],axis=1)
          res = main_df_.groupby([lambda x: x.year]).apply(lambda x: func(x.ix[:,0], x.ix[:,1], *args, **kw_copy))
      else:
          res = rets.groupby([lambda x: x.year]).apply(func, *args, **kw_copy)
      return res
  return wrapper

def rolling_window(func, rolling_window):
  """
  Decorator to apply functions on sliding data.
  """
  def wrapper(*args, **kw):
      kw_copy = kw.copy()
      if len(args):
          rets = args[0]
          args = args[1:]
      else:
          rets = kw['returns']
          del kw_copy['returns']
      if 'factor_returns' in func.__code__.co_varnames[:func.__code__.co_argcount]:
          if len(args):
              factor_rets = args[0]
              args = args[1:]
          else:
              factor_rets = kw_copy['factor_returns']
              del kw_copy['factor_returns']
          main_df_ = pd.concat([rets, factor_rets], axis=1)
          res = utils.rolling_apply2d(main_df_, window=rolling_window, func = lambda x: func(x.ix[:,0], x.ix[:,1], *args, **kw_copy))
      else:
          res = utils.rolling_apply2d(rets, window=rolling_window, func = lambda rets: func(rets, *args, **kw_copy))
      return res
  return wrapper


def returns(prices):
    """
    Given a set of prices, compute the respective returns (relative change).
    Parameters
    ----------
    prices : pd.DataFrame
        Prices of a set of securities. One security per column.
    Returns
    -------
    Returns : pd.DataFrame
        DataFrame of simple returns (relative change).
    Additional Notes
    ----------------
    Rows (dates) containing invalid, non-finite values are discarded.
    """
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

def create_stats(returns):
    df_cum_rets = cum_returns(returns)
    print("Entire data start date: " + str(df_cum_rets.index[0].strftime('%Y-%m-%d')))
    print("Entire data end date: " + str(df_cum_rets.index[-1].strftime('%Y-%m-%d')))
    print('\n')

def cum_returns(returns, starting_value=None):
    """
    Compute cumulative returns from simple returns.
    Parameters
    ----------
    returns : pd.Series
        Returns of the strategy, noncumulative.
    starting_value : float, optional
       The starting returns (default 1).
    Returns
    -------
    pandas.Series
        Series of cumulative returns.
    Additional Notes
    ----------------
    For increased numerical accuracy, convert input to log returns
    where it is possible to sum instead of multiplying.
    """
    # df_price.pct_change() adds a nan in first position, we can use
    # that to have cum_returns start at the origin so that
    # df_cum.iloc[0] == starting_value
    # Note that we can't add that ourselves as we don't know which dt
    # to use.
    if pd.isnull(returns.iloc[0]):
        returns.iloc[0] = 0.

    df_cum = np.exp(np.log(1.+returns).cumsum())

    if starting_value is None:
        return df_cum - 1
    else:
        return df_cum * starting_value

def aggregate_returns(df_daily_rets, convert_to):
    """
    Aggregates returns by week, month or year.
    Parameters
    ----------
    df_daily_rets : pd.Series
       Daily returns of the strategy, noncumulative.
    convert_to : str
       Can be 'weekly', 'monthly', or 'yearly'.
    Returns
    -------
    pd.Series
        Aggregated returns.
    """
    def cumulate_returns(x):
        return cum_returns(x)[-1]

    if convert_to == WEEKLY:
        return df_daily_rets.groupby(
            [lambda x: x.year,
             lambda x: x.month,
             lambda x: x.isocalendar()[1]]).apply(cumulate_returns)
    elif convert_to == MONTHLY:
        return df_daily_rets.groupby([lambda x: x.year, lambda x: x.month]).apply(cumulate_returns)
    elif convert_to == YEARLY:
        return df_daily_rets.groupby([lambda x: x.year]).apply(cumulate_returns)
    else:
        ValueError('convert_to must be {}, {} or {}'.format(WEEKLY, MONTHLY, YEARLY))

def max_drawdown(returns):
    """
    Determines the maximum drawdown of a strategy.
    Parameters
    ----------
    returns : pd.Series
        Returns of the strategy, noncumulative.
    Returns
    -------
    float
        Maximum drawdown.
    """
    if returns.size < 1:
        return np.nan
    df_cum_rets = cum_returns(returns, starting_value=1)
    running_max = np.maximum.accumulate(df_cum_rets)
    DD = (running_max - df_cum_rets) / running_max
    MDD = np.max(DD)
    return -1 * MDD

def annual_return(returns, period=DAILY):
    """Determines the annual returns of a strategy.
    Parameters
    ----------
    returns : pd.Series
        Periodic returns of the strategy, noncumulative.
    period : str, optional
        - defines the periodicity of the 'returns' data for purposes of
        annualizing. Can be 'yearly', 'quarterly', 'monthly', 'weekly',
        or 'daily'.
        - defaults to 'daily'.
    Returns
    -------
    float
        Annual Return as CAGR (Compounded Annual Growth Rate)
    """
    if returns.size < 1:
        return np.nan
    try:
        ann_factor = ANNUALIZATION_FACTORS[period]
    except KeyError:
        raise ValueError(
            "period cannot be '{}'. "
            "Must be '{}', '{}', '{}', or '{}'".format(
                period, DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY
            )
        )
    num_years = float(len(returns)) / ann_factor
    df_cum_rets = cum_returns(returns, starting_value=100)
    start_value = 100
    end_value = df_cum_rets.iloc[-1]

    total_return = (end_value - start_value) / start_value
    annual_return = (1. + total_return) ** (1 / num_years) - 1
    return annual_return

def annual_volatility(returns, period=DAILY):
    """
    Determines the annual volatility of a strategy.
    Parameters
    ----------
    returns : pd.Series
        Periodic returns of the strategy, noncumulative.
    period : str, optional
        - defines the periodicity of the 'returns' data for purposes of
        annualizing volatility. Can be 'yearly', 'quarterly', 'monthly',
        'weekly' or 'daily'.
        - defaults to 'daily'
    Returns
    -------
    float
        Annual volatility.
    """
    if returns.size < 2:
        return np.nan
    try:
        ann_factor = ANNUALIZATION_FACTORS[period]
    except KeyError:
        raise ValueError(
            "period cannot be '{}'. "
            "Must be '{}', '{}', '{}', or '{}'".format(
                period, DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY
            )
        )
    return returns.std() * np.sqrt(ann_factor)


def tracking_error(returns, factor_returns, period = DAILY):
    """
    Determines the annualized tracking error.
    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Returns of the strategy, noncumulative.
    factor_returns: float / series
    period : str, optional
        - defines the periodicity of the 'returns' data for purposes of
        annualization. Can be 'yearly', 'quarterly', 'monthly', 'weekly'
        or 'daily'.
        - set to 'yearly' to skip annualization.
        - defaults to 'daily
    Returns
    -------
    float
        Annualized tracking error.
    """
    try:
        ann_factor = ANNUALIZATION_FACTORS[period]
    except KeyError:
        raise ValueError(
            "period cannot be '{}'. "
            "Must be '{}', '{}', '{}', or '{}'".format(
                period, DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY
            )
        )
    active_return = returns - factor_returns
    return  np.std(active_return) * np.sqrt(ann_factor)


def information_ratio(returns, factor_returns, period = DAILY):
    """
    Determines the annualized information ratio of a strategy.
    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Returns of the strategy, noncumulative.
    factor_returns: float / series
    period : str, optional
        - defines the periodicity of the 'returns' data for purposes of
        annualization. Can be 'yearly', 'quarterly', 'monthly', 'weekly'
        or 'daily'.
        - set to 'yearly' to skip annualization.
        - defaults to 'daily
    Returns
    -------
    float
        Annualized information ratio.
    """
    active_return = returns - factor_returns
    tracking_error = np.std(active_return)
    if np.isnan(tracking_error):
        return 0.0
    try:
        ann_factor = ANNUALIZATION_FACTORS[period]
    except KeyError:
        raise ValueError(
            "period cannot be '{}'. "
            "Must be '{}', '{}', '{}', or '{}'".format(
                period, DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY
            )
        )
    return (np.mean(active_return) / tracking_error)  * np.sqrt(ann_factor)




def calmar_ratio(returns, period=DAILY):
    """
    Determines the annualized Calmar ratio, or drawdown ratio, of a strategy.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    period : str, optional
        - defines the periodicity of the 'returns' data for purposes of
        annualization. Can be 'yearly', 'quarterly', 'monthly', 'weekly'
        or 'daily'.
        - defaults to 'daily'.
    Returns
    -------
    float
        Annualized Calmar ratio (drawdown ratio).
    Additional Notes
    ----------------
    Assumes that the risk-free rate is 0.
    Makes use of annual_return().
    """
    temp_max_dd = max_drawdown(returns=returns)
    if temp_max_dd < 0:
        temp = annual_return(returns=returns, period=period) / abs(temp_max_dd)
    else:
        return np.nan
    if np.isinf(temp):
        return np.nan
    return temp


def sterling_calmar_ratio(returns, period = DAILY):
    """
    Determines the anualized Sterling-Calmar ratio.
    Parameters
    ----------
    returns : pd.Series
       Returns of the strategy, noncumulative.
    period : str, optional
       - defines the periodicity of the 'returns' data for purposes of
       annualization. Can be 'yearly', 'quarterly', 'monthly', 'weekly'
       or 'daily'.
       - defaults to 'daily'.
    Returns
    -------
    float
       Sterling-Calmar ratio.
    Additional Notes
    ----------------
    Assumes that the risk-free rate is 0.
    Makes use of annual_return().
    """
    avg_annual_DD = np.nanmean(returns.groupby([lambda x: x.year]).apply(max_drawdown))
    return annual_return(returns, period) / abs(avg_annual_DD)

#r = np.array([0.3, 2.6, 1.1, -1.0,  1.5, 2.5,  1.6,  6.7, -1.4, 4.0,  -0.5, 8.1, 4.0, -3.7, -6.1, 1.7, -4.9, -2.2, 7.0, 5.8, -6.5, 2.4, -.5, -.9])/100.
#rets = pd.Series(r, index = pd.date_range(start = '2010-1-1', freq = 'M', periods = len(r)))
#sterling_calmar(rets, period = MONTHLY)

def modified_burke_ratio(returns, period = DAILY, topDD = 10):
    """
    Determines the annualized modified Burke ratio.
    Parameters
    ----------
    returns : pd.Series
       Returns of the strategy, noncumulative.
    period : str, optional
       - defines the periodicity of the 'returns' data for purposes of
       annualization. Can be 'yearly', 'quarterly', 'monthly', 'weekly'
       or 'daily'.
       - defaults to 'daily'.
    topDD : int, optional
       - defines the number of drawdowns to be considered (denominator)
       - defaults to 10.
    Returns
    -------
    float
       Modified Burke ratio
    Additional Notes
    ----------------
    Assumes that the risk-free rate is 0. Only the topDD drawdowns are considered.
    Makes use of gen_drawdown_table() and annual_return().
    """
    df_drawdowns = gen_drawdown_table(returns, topDD)
    return annual_return(returns, period) /  np.sqrt(np.nanmean((df_drawdowns['net drawdown in %'].values/100.)**2))

def hurst_index(returns):
    """
    Determines the Hurst index.
    Parameters
    ----------
    returns : pd.Series
       Returns of the strategy, noncumulative.
    Returns
    -------
    float
        Hurst index.
    """
    m = (max(returns) - min(returns)) / np.std(returns)
    n = len(returns)
    return np.log(m) / np.log(n)

def omega_ratio(returns, annual_return_threshhold=0.0):
    """Determines the Omega ratio of a strategy.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    annual_return_threshold : float, optional
        Threshold over which to consider positive vs negative
        returns. For the ratio, it will be converted to a daily return
        and compared to returns.
    Returns
    -------
    float
        Omega ratio.
    """
    daily_return_thresh = pow(1 + annual_return_threshhold, 1 /APPROX_BDAYS_PER_YEAR) - 1
    returns_less_thresh = returns - daily_return_thresh
    numer = sum(returns_less_thresh[returns_less_thresh > 0.0])
    denom = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])
    if denom > 0.0:
        return numer / denom
    else:
        return np.nan

def sortino_ratio(returns, required_return=0, period=DAILY):
    """
    Determines the Sortino ratio of a strategy.
    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
         Returns of the strategy, noncumulative.
    required_return: float / series
        minimum acceptable return
    period : str, optional
       - defines the periodicity of the 'returns' data for purposes of
       annualization. Can be 'yearly', 'quarterly', 'monthly', 'weekly'
       or 'daily'.
       - set to 'yearly' to skip annualization.
       - defaults to 'daily'.
    Returns
    -------
    depends on input type
    series ==> float
    DataFrame ==> np.array
        Annualized Sortino ratio.
    """
    try:
        ann_factor = ANNUALIZATION_FACTORS[period]
    except KeyError:
        raise ValueError(
            "period cannot be '{}'. "
            "Must be '{}', '{}', '{}', or '{}'".format(
                period, DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY
            )
        )
    mu = np.nanmean(returns - required_return, axis=0)
    sortino = mu / downside_risk(returns, required_return, YEARLY) #skip dside annualization
    if len(returns.shape) == 2:
        sortino = pd.Series(sortino, index=returns.columns)
    return sortino * np.sqrt(ann_factor)

def downside_risk(returns, required_return=0, period=DAILY):
    """
    Determines the downside deviation below a threshold
    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Returns of the strategy, noncumulative.
    required_return: float / series
        minimum acceptable return
    period : str, optional
       - defines the periodicity of the 'returns' data for purposes of
       annualization. Can be 'yearly', 'quarterly', 'monthly', 'weekly'
       or 'daily'.
       - set to 'yearly' to skip annualization.
       - defaults to 'daily'.
    Returns
    -------
    depends on input type
    series ==> float
    DataFrame ==> pd.Series in which indices are the column names of the input
        Annualized downside deviation
    """
    try:
        ann_factor = ANNUALIZATION_FACTORS[period]
    except KeyError:
        raise ValueError(
            "period cannot be '{}'. "
            "Must be '{}', '{}', '{}', or '{}'".format(
                period, DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY
            )
        )
    downside_diff = returns - required_return
    mask = downside_diff > 0
    downside_diff[mask] = 0.0
    squares = np.square(downside_diff)
    mean_squares = np.nanmean(squares, axis=0)
    dside_risk = np.sqrt(mean_squares) * np.sqrt(ann_factor)
    if len(returns.shape) == 2:
        dside_risk = pd.Series(dside_risk, index=returns.columns)
    return dside_risk

def upside_risk(returns, required_return=0, period=DAILY):
    """
    Determines the upside deviation below a threshold
    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Returns of the strategy, noncumulative.
    required_return: float / series
        minimum acceptable return
    period : str, optional
       - defines the periodicity of the 'returns' data for purposes of
       annualization. Can be 'yearly', 'quarterly', 'monthly', 'weekly'
       or 'daily'.
       - set to 'yearly' to skip annualization.
       - defaults to 'daily'.
    Returns
    -------
    depends on input type
    series ==> float
    DataFrame ==> pd.Series in which indices are the column names of the input
        Annualized upside deviation
    """
    try:
        ann_factor = ANNUALIZATION_FACTORS[period]
    except KeyError:
        raise ValueError(
            "period cannot be '{}'. "
            "Must be '{}', '{}', '{}', or '{}'".format(
                period, DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY
            )
        )
    upside_diff = returns - required_return
    mask = upside_diff < 0
    upside_diff[mask] = 0.0
    squares = np.square(upside_diff)
    mean_squares = np.nanmean(squares, axis=0)
    uside_risk = np.sqrt(mean_squares) * np.sqrt(ann_factor)
    if len(returns.shape) == 2:
        uside_risk = pd.Series(uside_risk, index=returns.columns)
    return uside_risk


def sharpe_ratio(returns, risk_free=0, period=DAILY):
    """
    Determines the annualized Sharpe ratio of a strategy.
    Parameters
    ----------
    returns : pd.Series
        Returns of the strategy, noncumulative.
    period : str, optional
       - defines the periodicity of the 'returns' data for purposes of
       annualization. Can be 'yearly', 'quarterly', 'monthly', 'weekly'
       or 'daily'.
       - set to 'yearly' to skip annualization.
       - defaults to 'daily'.
    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    try:
        ann_factor = ANNUALIZATION_FACTORS[period]
    except KeyError:
        raise ValueError(
            "period cannot be '{}'. "
            "Must be '{}', '{}', '{}', or '{}'".format(
                period, DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY
            )
        )
    returns_risk_adj = returns - risk_free
    if (len(returns_risk_adj) < 5) or np.all(returns_risk_adj == 0):
        return np.nan
    return (np.mean(returns_risk_adj) / np.std(returns_risk_adj)) * np.sqrt(ann_factor)

def adjusted_sharpe_ratio(returns, risk_free=0, period=DAILY):
    """
    Determines the adjusted Sharpe ratio of a strategy.
    Parameters
    ----------
    returns : pd.Series
        Returns of the strategy, noncumulative.
    period : str, optional
       - defines the periodicity of the 'returns' data for purposes of
       annualization. Can be 'yearly', 'quarterly', 'monthly', 'weekly'
       or 'daily'.
       - set to 'yearly' to skip annualization.
       - defaults to 'daily'.
    Returns
    -------
    float
        Sharpe ratio.
    """
    try:
        ann_factor = ANNUALIZATION_FACTORS[period]
    except KeyError:
        raise ValueError(
            "period cannot be '{}'. "
            "Must be '{}', '{}', '{}', or '{}'".format(
                period, DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY
            )
        )
    SR = sharpe_ratio(returns, risk_free, YEARLY) # skip annualization
    S = sp.stats.skew(returns)
    K = sp.stats.kurtosis(returns) # computes fisher's kurtosis (aka excess kurtosis)
    return SR * (1+ SR * S/6 - SR*SR*K/24) * np.sqrt(ann_factor)

def bera_jarque(returns):
    """
    Determines the Bera-Jarque statistic.
    Parameters
    ----------
    returns : pd.Series
        Returns of the strategy, noncumulative.
    Returns
    -------
    float
        Bera-Jarque statistic
    Additional Notes
    ----------------
    If > 5.99, reject distribution as normal with 95% confidence.
    If > 9.21,  reject distribution as normal with 99% confidence.
    Perfectly normal distributins: BJ = 0.
    """
    S = sp.stats.skew(returns)
    K = sp.stats.kurtosis(returns) # computes fisher's kurtosis (aka excess kurtosis)
    n = len(returns)
    return n/6 * (S**2 + (K**2) / 4)

def var(returns, alpha = 0.05):
    # This method computes the historical simulation var of the returns
    sorted_returns = returns.sort().values
    # Compute the index associated with alpha
    index = int(alpha * len(sorted_returns))
    # VaR should be positive
    return abs(sorted_returns[index]), alpha


def cvar(returns, alpha = 0.05):
    # This method computes the condition VaR of the returns
    sorted_returns = returns.sort().values
    # Calculate the index associated with alpha
    index = int(alpha * len(sorted_returns))
    # Calculate the total VaR beyond alpha
    sum_var = sorted_returns[0]
    for i in range(1, index):
        sum_var += sorted_returns[i]
    # Return the average VaR
    # CVaR should be positive
    return abs(sum_var / index), alpha

def rolling_sharpe(returns, rolling_sharpe_window):
    """
    Determines the rolling Sharpe ratio of a strategy.
    Parameters
    ----------
    returns : pd.Series or pd.Dataframe
        Daily returns of the strategy, noncumulative.
    rolling_sharpe_window : int
        Length of rolling window, in days, over which to compute.
    Returns
    -------
    pd.Series
        Rolling Sharpe ratio (Annualized). (First row will have NaNs)
    Additional Notes
    ----------------
    Alternatively, use sharpe_ratio() with rolling_window decorator.
    """
    return (pd.rolling_mean(returns, rolling_sharpe_window) / pd.rolling_std(returns, rolling_sharpe_window)) * np.sqrt(APPROX_BDAYS_PER_YEAR)

def stability_of_timeseries(returns):
    """Determines R-squared of a linear fit to the cumulative
    log returns. Computes an ordinary least squares linear fit,
    and returns R-squared.
    Parameters
    ----------
    returns : pd.Series
        Returns of the strategy, noncumulative.
    Returns
    -------
    float
        R-squared.
    """
    cum_log_returns = np.log1p(returns).cumsum()
    rhat = sp.stats.linregress(np.arange(len(cum_log_returns)), cum_log_returns.values)[2]
    return rhat

def calc_multifactor(returns, factors):
    """Computes multiple ordinary least squares linear fits, and returns
    fit parameters.
    Parameters
    ----------
    returns : pd.Series
       Returns of the strategy, noncumulative.
    factors : pd.Series
        Secondary sets to fit.
    Returns
    -------
    pd.DataFrame
        Fit parameters.
    """
    import statsmodels.api as sm
    factors = factors.loc[returns.index]
    factors = sm.add_constant(factors)
    factors = factors.dropna(axis=0)
    results = sm.OLS(returns[factors.index], factors).fit()
    return results.params

def alpha_beta(returns, factor_returns, period = DAILY, compute_std = False):
    """Calculates alpha, beta and standard error of estimate (optional).
    Parameters
    ----------
    returns : pd.Series
         Returns of the strategy, noncumulative.
    factor_returns : pd.Series
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.
    period : str, optional
        - defines the periodicity of the 'returns' data for purposes of
        annualizing. Can be 'monthly', 'weekly', or 'daily'
        - defaults to 'daily'.
    compute_std: bool, optional
        - If True, computes standard error of estimate.
        - defaults to False
    Returns
    -------
    float
        Annualized alpha.
    float
        Beta.
    float (optional)
        Annualized standard error of the estimate.
    Additional Notes
    ----------------
    Beta is a slope measure, therefore annualization does not change its value.
    """
    try:
        ann_factor = ANNUALIZATION_FACTORS[period]
    except KeyError:
        raise ValueError(
            "period cannot be: '{}'."
            " Must be '{}', '{}', or '{}'".format(
                period, DAILY, WEEKLY, MONTHLY
            )
        )
    ret_index = returns.index
    beta, alpha = sp.stats.linregress(factor_returns.loc[ret_index].values, returns.values)[:2] #unintuitive function, first returns slope
    if compute_std:
        stderr = np.std(returns.values - alpha - beta * factor_returns)
        return alpha * ann_factor, beta, stderr * np.sqrt(ann_factor)
    else:
        return alpha * ann_factor, beta


#def teste():
#    # p. 76, C. Bacon, "Practical Portfolio Performance: Measure and Attribution"
#    r = pd.Series(np.array([0.3, 2.6, 1.1, -1.0,  1.5, 2.5,  1.6,  6.7, -1.4, 4.0,  -0.5, 8.1, 4.0, -3.7, -6.1, 1.7, -4.9, -2.2, 7.0, 5.8, -6.5, 2.4, -.5, -.9]))
#    b = pd.Series(np.array([0.2, 2.5, 1.8, -1.1,  1.4, 1.8, 1.4,   6.5, -1.5, 4.2, -0.6, 8.3, 3.9, -3.8, -6.2, 1.5, -4.8, 2.1, 6.0, 5.6, -6.7, 1.9, -0.3, 0.0]))
#    alpha, beta, stderr = alpha_beta(r, b, period = MONTHLY, compute_std = True)
#    print alpha, beta, stderr
#    print np.std(b)*np.sqrt(12)
#teste()


def alpha(returns, factor_returns, period = DAILY):
    """Calculates annualized alpha.
    Parameters
    ----------
    returns : pd.Series
        Returns of the strategy, noncumulative.
    factor_returns : pd.Series
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.
    period : str, optional
        - defines the periodicity of the 'returns' data for purposes of
        annualizing. Can be 'monthly', 'weekly', or 'daily'
        - defaults to 'daily'.
    Returns
    -------
    float
        Alpha.
    """
    return alpha_beta(returns, factor_returns, period)[0]

def beta(returns, factor_returns):
    """Calculates beta.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    factor_returns : pd.Series
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.
    Returns
    -------
    float
        Beta.
    """
    return alpha_beta(returns, factor_returns)[1]

def bull_beta(returns, factor_returns):
    """"
    Calculates bull beta.
    Similar to beta, but only taking the positive portfolio returns.
    """
    bull_dates = returns[returns > 0].index
    return alpha_beta(returns[bull_dates], factor_returns[bull_dates])[1]

def bear_beta(returns, factor_returns):
    """
    Calculates bull beta.
    Similar to beta, but only taking the negative portfolio returns.
    """
    bear_dates = returns[returns < 0].index
    return alpha_beta(returns[bear_dates], factor_returns[bear_dates])[1]

def systematic_risk(returns, factor_returns, period = DAILY):
    """
    Calculates the annualized systematic risk.
    """
    beta_ = beta(returns, factor_returns)
    market_risk = annual_volatility(factor_returns)
    return market_risk * beta_

def specific_risk(returns, factor_returns, period = DAILY):
    """
    Calculates the annualized specific risk of a portfolio.
    """
    return alpha_beta(returns, factor_returns, period = period, compute_std = True)[-1]

def rolling_beta(returns, factor_returns, rolling_window=APPROX_BDAYS_PER_MONTH * 6):
    """Determines the rolling beta of a strategy.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    factor_returns : pd.Series or pd.DataFrame
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.
        If DataFrame is passed, computes rolling beta for each column.
    rolling_window : int, optional
        The size of the rolling window, in days, over which to compute
        beta (default 6 months).
    Returns
    -------
    pd.Series
        Rolling beta.
    Additional Notes
    ----------------
    Alternatively, use beta() with rolling_window decorator.
    This function is still used by rolling_fama_french().
    """
    if factor_returns.ndim > 1:
        # Apply column-wise
        return factor_returns.apply(partial(rolling_beta, returns), rolling_window=rolling_window)
    else:
        out = pd.Series(index=returns.index)
        for beg, end in zip(returns.index[0:-rolling_window], returns.index[rolling_window:]):
            out.loc[end] = alpha_beta(returns.loc[beg:end], factor_returns.loc[beg:end])[1]
        return out

def rolling_fama_french(returns, factor_returns=None, rolling_window=APPROX_BDAYS_PER_MONTH * 6):
    """Computes rolling Fama-French single factor betas.
    Specifically, returns SMB, HML, and UMD.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    factor_returns : pd.DataFrame, optional
        data set containing the Fama-French risk factors. See
        utils.load_portfolio_risk_factors.
    rolling_window : int, optional
        The days window over which to compute the beta.
        Default is 6 months.
    Returns
    -------
    pandas.DataFrame
        DataFrame containing rolling beta coefficients for SMB, HML
        and UMD
    """
    if factor_returns is None:
        factor_returns = utils.load_portfolio_risk_factors(start=returns.index[0], end=returns.index[-1])
        factor_returns = factor_returns.drop(['Mkt-RF', 'RF'], axis='columns')
    return rolling_beta(returns, factor_returns, rolling_window=rolling_window)

SIMPLE_STAT_FUNCS = [
    annual_return,
    annual_volatility,
    sharpe_ratio,
    max_drawdown,
    calmar_ratio,
    sterling_calmar_ratio,
    modified_burke_ratio,
    hurst_index,
    stability_of_timeseries,
    upside_risk,
    downside_risk,
    omega_ratio,
    sortino_ratio,
    sp.stats.skew,
    sp.stats.kurtosis,
    bera_jarque,
    adjusted_sharpe_ratio
]

FACTOR_STAT_FUNCS = [
    tracking_error,
    information_ratio,
    alpha,
    beta,
    bull_beta,
    bear_beta,
    systematic_risk,
    specific_risk
]

def perf_stats(returns, factor_returns=None, period = DAILY, mode = 'simple', window_length = APPROX_BDAYS_PER_YEAR):
    """Calculates various performance metrics of a strategy.
    Parameters
    ----------
    returns : pd.Series
        Returns of the strategy, noncumulative.
    factor_returns : pd.Series (optional)
        Noncumulative returns of the benchmark.
         - This is in the same style as returns.
        If None, do not compute FACTOR_STAT_FUNCS: tracking_error,
        information_ratio, alpha, beta, bull_beta, bear_beta,
        systematic_risk, specific_risk.
    period : str, optional
        - defines the periodicity of the 'returns' data for purposes of
        annualizing. Can be 'yearly', 'quarterly', 'monthly', 'weekly',
        or 'daily'.
        - defaults to 'daily'.
    mode : str, optional
        - This variable can have the following values: 'simple', 'per_year' or
        'rolling':
            - If 'simple', computes the global statistics (all period of backtest).
            The returned variable is a Series.
            - If 'per_year', groups the backtest periods into years and computes the
            statistics for each year. The returned variable is a DataFrame.
            - If 'rolling', computes the statistics with a rolling window whose length is 12
            months (default: 252 business days). This value assumes that the returns are daily;
            therefore, if this assumption does not hold, the user must set the appropriate
            length of the rolling window as an additional argument. One can also use
            infer_periodicity() to set the period (correct inference is not always guarateed).
    window_length : int, optional
        - This argument is only valid if mode='rolling'. In this case, this variable sets the
        length of the rolling window for which the statistics shall be computed.
    Returns
    -------
    pd.Series / pd.DataFrame
        Performance metrics.
    """
    def apply_wrappers(func, mode, *args, **kwargs):
         # Auxiliary Function: check mode, apply wrappers if needed
         if  mode == 'simple':
             result = func(*args, **kwargs)
         elif  mode == 'per_year':
             result = per_year(stat_func)(*args, **kwargs)
         elif mode == 'rolling':
             result = rolling_window(stat_func, window_length)(*args, **kwargs)
         return result


    stats = pd.Series() if mode == 'simple' else  pd.DataFrame()

    for stat_func in SIMPLE_STAT_FUNCS:
        if stat_func.__code__.co_argcount >= 2 and ('period' in stat_func.__code__.co_varnames):
            stats[stat_func.__name__] = apply_wrappers(stat_func, mode, returns, period = period)
        else:
            stats[stat_func.__name__] = apply_wrappers(stat_func, mode, returns)

    if factor_returns is not None:
        for stat_func in FACTOR_STAT_FUNCS:
            if stat_func.__code__.co_argcount == 2:
                stats[stat_func.__name__] = apply_wrappers(stat_func, mode, returns, factor_returns)
            elif stat_func.__code__.co_argcount == 3:
                stats[stat_func.__name__] = apply_wrappers(stat_func, mode, returns, factor_returns, period = period)
    return stats



def calc_distribution_stats(x):
    """Calculate various summary statistics of data.
    Parameters
    ----------
    x : numpy.ndarray or pandas.Series
        Array to compute summary statistics for.
    Returns
    -------
    pandas.Series
        Series containing mean, median, std, as well as 5, 25, 75 and
        95 percentiles of passed in values.
    """
    return pd.Series({'mean': np.mean(x),
                      'median': np.median(x),
                      'std': np.std(x),
                      '5%': np.percentile(x, 5),
                      '25%': np.percentile(x, 25),
                      '75%': np.percentile(x, 75),
                      '95%': np.percentile(x, 95),
                      'IQR': np.subtract.reduce(
                          np.percentile(x, [75, 25])),
                      })


def portfolio_returns_metric_weighted(holdings_returns, name_strat = 'port_ret',
                                      exclude_non_overlapping=True,
                                      weight_function=None, weight_function_args = None,
                                      weight_function_window=None,
                                      inverse_weight=False,
                                      portfolio_rebalance_rule=None,
                                      weight_func_transform=None):
    """
    Generates an equal-weight portfolio, or portfolio weighted by
    weight_function
    Parameters
    ----------
    holdings_returns : pd.DataFrame
       Frame containing each individual holding's daily returns of the
       strategy, noncumulative. Each asset per column
    name_strat : string
       Name of the strategy. Default: port_ret
    exclude_non_overlapping : boolean, optional
       (Only applicable if equal-weight portfolio, e.g. weight_function=None)
       If True, timeseries returned will include values only for dates
       available across all holdings_returns timeseries If False, 0%
       returns will be assumed for a holding until it has valid data
    weight_function : function, optional
       Function to be applied to holdings_returns timeseries
    weight_function_window : int, optional
       Rolling window over which weight_function will use as its input values
    portfolio_rebalance_rule : string, optional
       A pandas.resample valid rule. Specifies how frequently to compute
       the weighting criteria
    weight_func_transform : function, optional
       Function applied to value returned from weight_function
    Returns
    -------
    pd.Series
        pd.Series : Portfolio returns timeseries.
    Additional Notes
    ----------------
    A more general function has been developed (see utils.rolling_apply2d).
    This should be replaced by a better approach in the near future.
    """
    def f_wrapper(ii, df, f, args): # Extending rolling_apply to 2D...
        x_df = df.ix[map(int,ii), :-1] # Exclude 'ii' column
        if args is not None:
            return f(x_df, args)
        else:
            return f(x_df)

    if weight_function is None:
        if exclude_non_overlapping:
            holdings_df = pd.DataFrame(holdings_returns).dropna(axis = 1)
        else:
            holdings_df = pd.DataFrame(holdings_returns).fillna(0)
        holdings_df = holdings_df.sum(axis=1) / len(holdings_returns.columns)
    else:
        holdings_df_na = pd.DataFrame(holdings_returns)
        holdings_df = holdings_df_na.dropna(axis = 1)
        holdings_df['ii'] =   range(len(holdings_df))
        holdings_df = pd.rolling_apply(holdings_df['ii'], window=weight_function_window, func=lambda ii: f_wrapper(ii, holdings_df, weight_function, weight_function_args)).dropna()
        if portfolio_rebalance_rule is not None:
            holdings_df = holdings_df.resample(rule=portfolio_rebalance_rule, how='last')
        if weight_func_transform is not None:
            holdings_df = holdings_df.applymap(weight_func_transform)
    holdings_df.name = name_strat
    return holdings_df

#==============================================================================
#  DRAWDOWNS
#==============================================================================
def get_max_drawdown_underwater(underwater):
    """Determines peak, valley, and recovery dates given an 'underwater' DataFrame.
    An underwater DataFrame is a DataFrame that has precomputed rolling drawdown.
    Parameters
    ----------
    underwater : pd.Series
       Underwater returns (rolling drawdown) of a strategy.
    Returns
    -------
    peak : datetime
        The maximum drawdown's peak.
    valley : datetime
        The maximum drawdown's valley.
    recovery : datetime
        The maximum drawdown's recovery.
    """
    valley = np.argmax(underwater)  # end of the period
    # Find first 0
    peak = underwater[:valley][underwater[:valley] == 0].index[-1]
    # Find last 0
    try:
        recovery = underwater[valley:][underwater[valley:] == 0].index[0]
    except IndexError:
        recovery = np.nan  # drawdown not recovered
    return peak, valley, recovery

def get_max_drawdown(returns):
    """
    Finds maximum drawdown.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    Returns
    -------
    peak : datetime
        The maximum drawdown's peak.
    valley : datetime
        The maximum drawdown's valley.
    recovery : datetime
        The maximum drawdown's recovery.
    """
    returns = returns.copy()
    df_cum = cum_returns(returns, 1.0)
    running_max = np.maximum.accumulate(df_cum)
    underwater = (running_max - df_cum) / running_max
    return get_max_drawdown_underwater(underwater)

def get_top_drawdowns(returns, top=10):
    """
    Finds top drawdowns, sorted by drawdown amount.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    top : int, optional
        The amount of top drawdowns to find (default 10).
    Returns
    -------
    drawdowns : list
        List of drawdown peaks, valleys, and recoveries. See get_max_drawdown.
    """
    returns = returns.copy()
    df_cum = cum_returns(returns, 1.0)
    running_max = np.maximum.accumulate(df_cum)
    underwater = running_max - df_cum

    drawdowns = []
    for t in range(top):
        peak, valley, recovery = get_max_drawdown_underwater(underwater)
        # Slice out draw-down period
        if not pd.isnull(recovery):
            underwater.drop(underwater[peak: recovery].index[1:-1], inplace=True)
        else:
            # drawdown has not ended yet
            underwater = underwater.loc[:peak]

        drawdowns.append((peak, valley, recovery))
        if (len(returns) == 0) or (len(underwater) == 0):
            break
    return drawdowns

def gen_drawdown_table(returns, top=10):
    """
    Places top drawdowns in a table.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    top : int, optional
        The amount of top drawdowns to find (default 10).
    Returns
    -------
    df_drawdowns : pd.DataFrame
        Information about top drawdowns.
    """
    df_cum = cum_returns(returns, 1.0)
    drawdown_periods = get_top_drawdowns(returns, top=top)
    df_drawdowns = pd.DataFrame(index=list(range(top)),
                                columns=['net drawdown in %',
                                         'peak date',
                                         'valley date',
                                         'recovery date',
                                         'duration'])

    for i, (peak, valley, recovery) in enumerate(drawdown_periods):
        if pd.isnull(recovery):
            df_drawdowns.loc[i, 'duration'] = np.nan
        else:
            df_drawdowns.loc[i, 'duration'] = len(pd.date_range(peak, recovery, freq='B'))
        df_drawdowns.loc[i, 'peak date'] = (peak.to_pydatetime().strftime('%Y-%m-%d'))
        df_drawdowns.loc[i, 'valley date'] = (valley.to_pydatetime().strftime('%Y-%m-%d'))
        if isinstance(recovery, float):
            df_drawdowns.loc[i, 'recovery date'] = recovery
        else:
            df_drawdowns.loc[i, 'recovery date'] = (recovery.to_pydatetime().strftime('%Y-%m-%d'))
        df_drawdowns.loc[i, 'net drawdown in %'] = ((df_cum.loc[peak] - df_cum.loc[valley]) / df_cum.loc[peak]) * 100

    df_drawdowns['peak date'] = pd.to_datetime(df_drawdowns['peak date'], unit='D')
    df_drawdowns['valley date'] = pd.to_datetime(df_drawdowns['valley date'], unit='D')
    df_drawdowns['recovery date'] = pd.to_datetime(df_drawdowns['recovery date'], unit='D')
    return df_drawdowns


#==============================================================================
#  OTHER FUNCTIONS
#==============================================================================

def extract_interesting_date_ranges(returns):
    """Extracts returns based on interesting events. See gen_date_range_interesting.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    Returns
    -------
    ranges : OrderedDict
        Date ranges, with returns, of all valid events.
    """
    returns_dupe = returns.copy()
    returns_dupe.index = returns_dupe.index.map(pd.Timestamp)
    ranges = OrderedDict()
    for name, (start, end) in PERIODS.items():
        try:
            period = returns_dupe.loc[start:end]
            if len(period) == 0:
                continue
            ranges[name] = period
        except:
            continue
    return ranges

def out_of_sample_vs_in_sample_returns_kde(
        bt_ts, oos_ts, transform_style='scale', return_zero_if_exception=True):
    """Determines similarity between two returns timeseries.
    Typically a backtest frame (in-sample) and live frame
    (out-of-sample).
    Parameters
    ----------
    bt_ts : pd.Series
       In-sample (backtest) returns of the strategy, noncumulative.
    oos_ts : pd.Series
       Out-of-sample (live trading) returns of the strategy,
       noncumulative.
    transform_style : float, optional
        'raw', 'scale', 'Normalize_L1', 'Normalize_L2' (default
        'scale')
    return_zero_if_exception : bool, optional
        If there is an exception, return zero instead of NaN.
    Returns
    -------
    float
        Similarity between returns.
    """
    bt_ts_pct = bt_ts.dropna()
    oos_ts_pct = oos_ts.dropna()

    bt_ts_r = bt_ts_pct.reshape(len(bt_ts_pct), 1)
    oos_ts_r = oos_ts_pct.reshape(len(oos_ts_pct), 1)

    if transform_style == 'raw':
        bt_scaled = bt_ts_r
        oos_scaled = oos_ts_r
    if transform_style == 'scale':
        bt_scaled = preprocessing.scale(bt_ts_r, axis=0)
        oos_scaled = preprocessing.scale(oos_ts_r, axis=0)
    if transform_style == 'normalize_L2':
        bt_scaled = preprocessing.normalize(bt_ts_r, axis=1)
        oos_scaled = preprocessing.normalize(oos_ts_r, axis=1)
    if transform_style == 'normalize_L1':
        bt_scaled = preprocessing.normalize(bt_ts_r, axis=1, norm='l1')
        oos_scaled = preprocessing.normalize(oos_ts_r, axis=1, norm='l1')

    X_train = bt_scaled
    X_test = oos_scaled
    X_train = X_train.reshape(len(X_train))
    X_test = X_test.reshape(len(X_test))
    x_axis_dim = np.linspace(-4, 4, 100)
    kernal_method = 'scott'
    try:
        scipy_kde_train = sp.stats.gaussian_kde(
            X_train,
            bw_method=kernal_method)(x_axis_dim)
        scipy_kde_test = sp.stats.gaussian_kde(
            X_test,
            bw_method=kernal_method)(x_axis_dim)
    except:
        if return_zero_if_exception:
            return 0.0
        else:
            return np.nan

    kde_diff = sum(abs(scipy_kde_test - scipy_kde_train)) / \
        (sum(scipy_kde_train) + sum(scipy_kde_test))
    return kde_diff

def forecast_cone_bootstrap(is_returns, num_days, cone_std=(1., 1.5, 2.),
                            starting_value=1, num_samples=1000,
                            random_seed=None):
    """
    Determines the upper and lower bounds of an n standard deviation
    cone of forecasted cumulative returns. Future cumulative mean and
    standard devation are computed by repeatedly sampling from the
    in-sample daily returns (i.e. bootstrap). This cone is non-parametric,
    meaning it does not assume that returns are normally distributed.
    Parameters
    ----------
    is_returns : pd.Series
        In-sample daily returns of the strategy, noncumulative.
    num_days : int
        Number of days to project the probability cone forward.
    cone_std : int, float, or list of int/float
        Number of standard devations to use in the boundaries of
        the cone. If multiple values are passed, cone bounds will
        be generated for each value.
    starting_value : int or float
        Starting value of the out of sample period.
    num_samples : int
        Number of samples to draw from the in-sample daily returns.
        Each sample will be an array with length num_days.
        A higher number of samples will generate a more accurate
        bootstrap cone.
    random_seed : int
        Seed for the pseudorandom number generator used by the pandas
        sample method.
    Returns
    -------
    pd.DataFrame
        Contains upper and lower cone boundaries. Column names are
        strings corresponding to the number of standard devations
        above (positive) or below (negative) the projected mean
        cumulative returns.
    """
    samples = np.empty((num_samples, num_days))
    seed = np.random.RandomState(seed=random_seed)
    for i in range(num_samples):
        samples[i, :] = is_returns.sample(num_days, replace=True,
                                          random_state=seed)
    cum_samples = np.cumprod(1 + samples, axis=1) * starting_value

    cum_mean = cum_samples.mean(axis=0)
    cum_std = cum_samples.std(axis=0)

    if isinstance(cone_std, (float, int)):
        cone_std = [cone_std]

    cone_bounds = pd.DataFrame(columns=pd.Float64Index([]))
    for num_std in cone_std:
        cone_bounds.loc[:, float(num_std)] = cum_mean + cum_std * num_std
        cone_bounds.loc[:, float(-num_std)] = cum_mean - cum_std * num_std
    return cone_bounds



def bucket_std(value, bins=[0.12, 0.15, 0.18, 0.21], max_default=0.24):
    """
    Simple quantizing function. For use in binning stdevs into a "buckets"
    Parameters
    ----------
    value : float
       Value corresponding to the the stdev to be bucketed
    bins : list, optional
       Floats used to describe the buckets which the value can be placed
    max_default : float, optional
       If value is greater than all the bins, max_default will be returned
    Returns
    -------
    float
        bin which the value falls into
    """
    annual_vol = value * np.sqrt(252)
    for i in bins:
        if annual_vol <= i:
            return i
    return max_default

def min_max_vol_bounds(value, lower_bound=0.12, upper_bound=0.24):
    """
    Restrict volatility weighting of the lowest volatility asset versus the
    highest volatility asset to a certain limit.
    E.g. Never allocate more than 2x to the lowest volatility asset.
    round up all the asset volatilities that fall below a certain bound
    to a specified "lower bound" and round down all of the asset
    volatilites that fall above a certain bound to a specified "upper bound"
    Parameters
    ----------
    value : float
       Value corresponding to a daily volatility

    lower_bound : float, optional
       Lower bound for the volatility

    upper_bound : float, optional
       Upper bound for the volatility
    Returns
    -------
    float
        The value input, annualized, or the lower_bound or upper_bound
    """
    annual_vol = value * np.sqrt(252)
    if annual_vol < lower_bound:
        return lower_bound
    if annual_vol > upper_bound:
        return upper_bound
    return annual_vol



#def teste(x):
#    print x
#    return sum(x)
#returns = pd.DataFrame(np.arange(30).reshape((15,2)), columns = ['A', 'B'])
#returns.index = pd.date_range('1/1/2011', periods=15, freq='D'); returns.index.name = 'Date'
##
##
##print "##################"
#print returns
##returns = np.arange(10).reshape((5,2))
#a = portfolio_returns_metric_weighted(returns, name_strat = 'ola', weight_function=teste, weight_function_window = 3)
#print
#print a
##print returns
#print a
##a = pd.DataFrame(np.arange(10).reshape((5,2)), columns = ['A', 'B'])
##a.index = pd.date_range('1/1/2011', periods=5, freq='D'); a.index.name = 'Date'
##print a
##print downside_risk(a, 2)
##b = pd.Series(np.arange(0,9,2), name = 'C')
##b.index = pd.date_range('1/1/2011', periods=5, freq='D'); b.index.name = 'Date'
##print b
##print downside_risk(b, 2)
#
##list(map(lambda x: x + "_t", holdings_cols))