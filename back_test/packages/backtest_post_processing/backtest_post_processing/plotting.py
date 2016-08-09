from __future__ import division
#from collections import OrderedDict

import pandas as pd
import numpy as np
#import scipy as sp

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
#import matplotlib.lines as mlines

#from sklearn import preprocessing

import _seaborn as sns
import utils
import ts_metrics
#import pos
#import txn
from ts_metrics import APPROX_BDAYS_PER_MONTH

def show_perf_stats(returns, factor_returns = None, filename = None):
    """Prints some performance metrics of the strategy.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.
    filename : str, optional
        Allows to save the output in a file whose path is given by
        this variable. The data is appended.
        If None, no file is generated.
    Additional Notes
    ----------------
    It currently assumes that the returns are provided on a daily basis
    (business days).
    See ts_metrics.perf_stats.
    """
    returns_backtest = returns
    
    # print ts_metrics.perf_stats( returns_backtest, factor_returns=factor_returns)
    
    # This wasnt working - modification by Jaulino
    # perf_stats = np.round(ts_metrics.perf_stats( returns_backtest, factor_returns=factor_returns), 2)
    perf_stats = ts_metrics.perf_stats( returns_backtest, factor_returns=factor_returns)
  
    perf_stats.columns = [str(returns.name)]

    print('Backtest Months: ' + str(int(len(returns_backtest) / APPROX_BDAYS_PER_MONTH)))
    print(perf_stats.to_string())
    if filename is not None:
        with open(filename, "a") as myfile:
            myfile.write('---- '+ str(returns.name) + ' ----\n')
            myfile.write('Backtest Months: ' + str(int(len(returns_backtest) / APPROX_BDAYS_PER_MONTH)) + '\n')
            myfile.write(perf_stats.to_string())


def show_worst_drawdown_periods(returns, top=5, filename = None):
    """Prints information about the worst drawdown periods.
    Prints peak dates, valley dates, recovery dates, and net
    drawdowns.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        Amount of top drawdowns periods to plot (default 5).
    """

    drawdown_df = ts_metrics.gen_drawdown_table(returns, top=top)
    drawdown_df['net drawdown in %'] = list(map(utils.round_two_dec_places, drawdown_df['net drawdown in %']))
    print('\nWorst Drawdown Periods')
    print(drawdown_df.sort_values('net drawdown in %', ascending=False).to_string(index=False))
    print(' average drawdown: %.2f%%' % (np.mean(drawdown_df['net drawdown in %'])))
    print(' average duration: %.0f' % np.mean(drawdown_df['duration']))
    if filename is not None:
        with open(filename, "a") as myfile:
            myfile.write('\nWorst Drawdown Periods\n')
            myfile.write(drawdown_df.sort_values('net drawdown in %', ascending=False).to_string(index=False) + '\n')
            myfile.write(' average drawdown: %.2f%%\n' % (np.mean(drawdown_df['net drawdown in %'])))
            myfile.write(' average duration: %.0f\n' % np.mean(drawdown_df['duration']))
            myfile.write('\n\n')


def show_return_range(returns, df_weekly, filename = None):
    """
    Print monthly return and weekly return standard deviations.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    df_weekly : pd.Series
        Weekly returns of the strategy, noncumulative.
         - See timeseries.aggregate_returns.
    """
    two_sigma_daily = np.round(returns.mean() - 2 * returns.std(), 3)
    two_sigma_weekly = np.round(df_weekly.mean() - 2 * df_weekly.std(), 3)
    var_sigma = pd.Series([two_sigma_daily, two_sigma_weekly],
                          index=['2-sigma returns daily',
                                 '2-sigma returns weekly'])
    print('\n'+var_sigma.to_string())
    if filename is not None:
        with open(filename, "a") as myfile:
            myfile.write('\n'+var_sigma.to_string()+'\n')


def plot_annual_returns(returns, ax=None, **kwargs):
    """
    Plots a bar graph of returns by year.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        ax = plt.gca()

    x_axis_formatter = FuncFormatter(utils.percentage)
    ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
    ax.tick_params(axis='x', which='major', labelsize=10)

    ann_ret_df = pd.DataFrame(ts_metrics.aggregate_returns(returns,'yearly'))

    ax.axvline(100 * ann_ret_df.values.mean(),color='steelblue', linestyle='--',
               lw=4, alpha=0.7)
    (100 * ann_ret_df.sort_index(ascending=False)).plot(ax=ax, kind='barh', alpha=0.70, **kwargs)
    ax.axvline(0.0, color='black', linestyle='-', lw=3)
    ax.set_ylabel('Year', fontsize = 12)
    ax.set_xlabel('Returns', fontsize = 12)
    ax.set_title("Annual Returns", fontsize = 12)
    ax.legend(['mean'], fontsize = 12)
    return ax

def plot_monthly_returns_heatmap(returns, ax=None, **kwargs):
    """
    Plots a heatmap of returns by month.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        ax = plt.gca()
    monthly_ret_table = ts_metrics.aggregate_returns(returns,  'monthly')
    monthly_ret_table = monthly_ret_table.unstack()
    monthly_ret_table = np.round(monthly_ret_table, 3)
    sns.heatmap(monthly_ret_table.fillna(0) * 100.0, annot=True, linewidths=.5, annot_kws={"size": 12},
                alpha=1.0, center=0.0, cbar=False, cmap=matplotlib.cm.RdYlGn, ax=ax, **kwargs)
    ax.set_ylabel('Year', fontsize = 12)
    ax.set_xlabel('Month', fontsize = 12)
    ax.set_title("Monthly Returns (%)", fontsize = 12)
    return ax

def plot_monthly_returns_dist(returns, ax=None, **kwargs):
    """
    Plots a distribution of monthly returns.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()
    x_axis_formatter = FuncFormatter(utils.percentage)
    ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
    ax.tick_params(axis='x', which='major', labelsize=10)

    
    monthly_ret_table = ts_metrics.aggregate_returns(returns, 'monthly')
    
    
    ax.hist( 100 * monthly_ret_table, color='orangered', alpha=0.80,
            bins=20, **kwargs)
    
    ax.axvline(100 * monthly_ret_table.mean(),color='gold',linestyle='--',
               lw=4, alpha=1.0)

    ax.axvline(0.0, color='black', linestyle='-', lw=3, alpha=0.75)
    ax.legend(['mean'])
    ax.set_ylabel('Number of months', fontsize = 12)
    ax.set_xlabel('Returns', fontsize = 12)
    ax.set_title("Distribution of Monthly Returns", fontsize = 12)
    return ax

def plot_rolling_returns(returns, factor_returns=None,  live_start_date=None,
                         cone_std=None, legend_loc='best', volatility_match=False,
                         cone_function=ts_metrics.forecast_cone_bootstrap, ax=None, **kwargs):
    """
    Plots cumulative rolling returns versus some benchmarks'.
    Backtest returns are in green, and out-of-sample (live trading)
    returns are in red.
    Additionally, a non-parametric cone plot may be added to the
    out-of-sample returns region.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of a risk factor.
         - This is in the same style as returns.
    live_start_date : datetime, optional
        The date when the strategy began live trading, after
        its backtest period. This date should be normalized.
    cone_std : float, or tuple, optional
        If float, The standard deviation to use for the cone plots.
        If tuple, Tuple of standard deviation values to use for the cone plots
         - See timeseries.forecast_cone_bounds for more details.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    volatility_match : bool, optional
        Whether to normalize the volatility of the returns to those of the
        benchmark returns. This helps compare strategies with different
        volatilities. Requires passing of benchmark_rets.
    cone_function : function, optional
        Function to use when generating forecast probability cone.
        The function signiture must follow the form:
        def cone(in_sample_returns (pd.Series),
                 days_to_project_forward (int),
                 cone_std= (float, or tuple),
                 starting_value= (int, or float))
        See ts_metrics.forecast_cone_bootstrap for an example.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
"""
    if ax is None:
        ax = plt.gca()

    if volatility_match and factor_returns is None:
        raise ValueError('volatility_match requires passing of'
                         'factor_returns.')
    elif volatility_match and factor_returns is not None:
        bmark_vol = factor_returns.loc[returns.index].std()
        returns = (returns / returns.std()) * bmark_vol

    cum_rets = ts_metrics.cum_returns(returns, 1.0)


    y_axis_formatter = FuncFormatter(utils.one_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    if factor_returns is not None:
        cum_factor_returns = ts_metrics.cum_returns( factor_returns[cum_rets.index], 1.0)
        cum_factor_returns.plot(lw=2, color='gray', label=factor_returns.name, alpha=0.60,
                                ax=ax, **kwargs)

    if live_start_date is not None:
        live_start_date = utils.get_utc_timestamp(live_start_date)
        is_cum_returns = cum_rets.loc[cum_rets.index < live_start_date]
        oos_cum_returns = cum_rets.loc[cum_rets.index >= live_start_date]
    else:
        is_cum_returns = cum_rets
        oos_cum_returns = pd.Series([])

    is_cum_returns.plot(lw=3, color='forestgreen', alpha=0.6, label='Backtest', ax=ax, **kwargs)

    if len(oos_cum_returns) > 0:
        oos_cum_returns.plot(lw=4, color='red', alpha=0.6, label='Live', ax=ax, **kwargs)

        if cone_std is not None:
            if isinstance(cone_std, (float, int)):
                cone_std = [cone_std]

            is_returns = returns.loc[returns.index < live_start_date]
            cone_bounds = cone_function( is_returns,len(oos_cum_returns),
                                        cone_std=cone_std, starting_value=is_cum_returns[-1])

            cone_bounds = cone_bounds.set_index(oos_cum_returns.index)

            for std in cone_std:
                ax.fill_between(cone_bounds.index,
                                cone_bounds[float(std)],
                                cone_bounds[float(-std)],
                                color='steelblue', alpha=0.5)
    if legend_loc is not None:
        ax.legend(loc=legend_loc)
    ax.axhline(1.0, linestyle='--', color='black', lw=2)
    ax.set_ylabel('Cumulative returns')
    ax.set_xlabel('')
    return ax

def plot_rolling_sharpe(returns, rolling_window=APPROX_BDAYS_PER_MONTH * 6,
                        legend_loc='best', ax=None, **kwargs):
    """
    Plots the rolling Sharpe ratio versus date.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    rolling_window : int, optional
        The days window over which to compute the sharpe ratio.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        ax = plt.gca()
    y_axis_formatter = FuncFormatter(utils.one_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    rolling_sharpe_ts = ts_metrics.rolling_sharpe(returns, rolling_window)
    rolling_sharpe_ts.plot(alpha=.7, lw=3, color='orangered', ax=ax, **kwargs)

    #ax.set_title('Rolling Sharpe ratio (6-month)')
    ax.axhline(rolling_sharpe_ts.mean(),color='steelblue',linestyle='--', lw=3)
    ax.axhline(0.0, color='black', linestyle='-', lw=3)

    ax.set_ylim((-3.0, 6.0))
    ax.set_ylabel('Rolling Sharpe ratio (6-month)')
    ax.set_xlabel('')
    ax.legend(['Sharpe', 'Average'], loc=legend_loc)
    return ax


def plot_rolling_beta(returns, factor_returns, legend_loc='best', ax=None, **kwargs):
    """
    Plots the rolling 6-month and 12-month beta versus date.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()
    y_axis_formatter = FuncFormatter(utils.one_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    ax.set_title("Rolling Portfolio Beta to " + str(factor_returns.name))
    ax.set_ylabel('Beta')
    rb_1 = ts_metrics.rolling_beta(returns, factor_returns, rolling_window=APPROX_BDAYS_PER_MONTH * 6)
    rb_1.plot(color='steelblue', lw=3, alpha=0.6, ax=ax, **kwargs)
    rb_2 = ts_metrics.rolling_beta(returns, factor_returns, rolling_window=APPROX_BDAYS_PER_MONTH * 12)
    rb_2.plot(color='grey', lw=3, alpha=0.4, ax=ax, **kwargs)
    ax.set_ylim((-2.5, 2.5))
    ax.axhline(rb_1.mean(), color='steelblue', linestyle='--', lw=3)
    ax.axhline(0.0, color='black', linestyle='-', lw=2)
    ax.set_xlabel('')
    ax.legend(['6-mo','12-mo'],loc=legend_loc)
    return ax

def plot_rolling_fama_french(returns, factor_returns=None, rolling_window=APPROX_BDAYS_PER_MONTH * 6, legend_loc='best', ax=None, **kwargs):
    """Plots rolling Fama-French single factor betas.
    Specifically, plots SMB, HML, and UMD vs. date with a legend.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    factor_returns : pd.DataFrame, optional
        data set containing the Fama-French risk factors. See
        utils.load_portfolio_risk_factors.
    rolling_window : int, optional
        The days window over which to compute the beta.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    ax.set_title( "Rolling Fama-French Single Factor Betas (%.0f-month)" % (rolling_window / APPROX_BDAYS_PER_MONTH ))
    ax.set_ylabel('beta')

    rolling_beta = ts_metrics.rolling_fama_french( returns,factor_returns=factor_returns, rolling_window=rolling_window)
    rolling_beta.plot(alpha=0.7, ax=ax, **kwargs)
    ax.axhline(0.0, color='black')
    ax.legend(['Small-Caps (SMB)','High-Growth (HML)', 'Momentum (UMD)'], loc=legend_loc)
    ax.set_ylim((-2.0, 2.0))
    y_axis_formatter = FuncFormatter(utils.one_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    ax.axhline(0.0, color='black')
    ax.set_xlabel('')
    return ax

def plot_return_quantiles(returns, df_weekly, df_monthly, ax=None, **kwargs):
    """Creates a box plot of daily, weekly, and monthly return
    distributions.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    df_weekly : pd.Series
        Weekly returns of the strategy, noncumulative.
         - See timeseries.aggregate_returns.
    df_monthly : pd.Series
        Monthly returns of the strategy, noncumulative.
         - See timeseries.aggregate_returns.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.

    """
    if ax is None:
        ax = plt.gca()

    sns.boxplot(data=[returns, df_weekly, df_monthly], ax=ax, **kwargs)
    ax.set_xticklabels(['daily', 'weekly', 'monthly'])
    ax.set_title('Return quantiles')
    return ax

def plot_drawdown_periods(returns, top=10, ax=None, **kwargs):
    """
    Plots cumulative returns highlighting top drawdown periods.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    top : int, optional
        Amount of top drawdowns periods to plot (default 10).
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        ax = plt.gca()
    y_axis_formatter = FuncFormatter(utils.one_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    df_cum_rets = ts_metrics.cum_returns(returns, starting_value=1.0)
    df_drawdowns = ts_metrics.gen_drawdown_table(returns, top=top)

    df_cum_rets.plot(ax=ax, **kwargs)

    lim = ax.get_ylim()
    colors = sns.cubehelix_palette(len(df_drawdowns))[::-1]
    for i, (peak, recovery) in df_drawdowns[
            ['peak date', 'recovery date']].iterrows():
        if pd.isnull(recovery):
            recovery = returns.index[-1]
        ax.fill_between((peak, recovery),lim[0], lim[1],alpha=.4,color=colors[i])

    ax.set_title('Top %i Drawdown Periods' % top)
    ax.set_ylabel('Cumulative returns')
    ax.legend(['Portfolio'], loc='upper left')
    ax.set_xlabel('')
    return ax


def plot_drawdown_underwater(returns, ax=None, **kwargs):
    """Plots how far underwaterr returns are over time, or plots current
    drawdown vs. date.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.percentage)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    df_cum_rets = ts_metrics.cum_returns(returns, starting_value=1.0)
    running_max = np.maximum.accumulate(df_cum_rets)
    underwater = -100 * ((running_max - df_cum_rets) / running_max)
    (underwater).plot(ax=ax, kind='area', color='coral', alpha=0.7, **kwargs)
    ax.set_ylabel('Drawdown', fontsize = 12)
    ax.set_title('Underwater Plot', fontsize = 12)
    ax.set_xlabel('')
    return ax

def plot_holdings(returns, positions, legend_loc='best', ax=None, **kwargs):
    """Plots total amount of stocks with an active position, either short
    or long.
    Displays daily total, daily average per month, and all-time daily
    average.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    positions : pd.DataFrame, optional
        Daily net position values.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        ax = plt.gca()

    positions = positions.copy().drop('cash', axis='columns')
    df_holdings = positions.apply(lambda x: np.sum(x != 0), axis='columns')
    df_holdings_by_month = df_holdings.resample('1M', how='mean')
    df_holdings.plot(color='steelblue', alpha=0.6, lw=0.5, ax=ax, **kwargs)
    df_holdings_by_month.plot(color='orangered', alpha=0.5, lw=2, ax=ax, **kwargs)
    ax.axhline(df_holdings.values.mean(), color='steelblue', ls='--', lw=3, alpha=1.0)

    ax.set_xlim((returns.index[0], returns.index[-1]))

    ax.legend(['Daily holdings', 'Average daily holdings, by month', 'Average daily holdings, net'], loc=legend_loc)
    ax.set_title('Holdings per Day')
    ax.set_ylabel('Amount of holdings per day')
    ax.set_xlabel('')
    return ax