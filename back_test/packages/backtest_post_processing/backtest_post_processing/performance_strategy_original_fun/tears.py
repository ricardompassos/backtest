from __future__ import division

from time import time
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.stats
import pandas as pd

import ts_metrics
import utils
import pos
import plotting


def timer(msg_body, previous_time):
    current_time = time()
    run_time = current_time - previous_time
    message = "\nFinished " + msg_body + " (required {:.2f} seconds)."
    print(message.format(run_time))
    return current_time

def create_full_tear_sheet(returns, positions=None, transactions=None, benchmark_rets=None, gross_lev=None,
                           slippage=None, live_start_date=None, sector_mappings=None, round_trips=False, filename = None):
    """
    THIS FUNCTION IS YET TO BE FULLY IMPLEMENTED.
    Generate a number of tear sheets that are useful
    for analyzing a strategy's performance.
    - Fetches benchmarks if needed.
    - Creates tear sheets for returns, and significant events.
        If possible, also creates tear sheets for position analysis,
        transaction analysis, and Bayesian analysis.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - Time series with decimal returns.
         - Example:
            2015-07-16    -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902
    positions : pd.DataFrame, optional
        Daily net position values.
         - Time series of dollar amount invested in each position and cash.
         - Days where stocks are not held can be represented by 0 or NaN.
         - Non-working capital is labelled 'cash'
         - Example:
            index         'AAPL'         'MSFT'          cash
            2004-01-09    13939.3800     -14012.9930     711.5585
            2004-01-12    14492.6300     -14624.8700     27.1821
            2004-01-13    -13853.2800    13653.6400      -43.6375
    transactions : pd.DataFrame, optional
        Executed trade volumes and fill prices.
        - One row per trade.
        - Trades on different names that occur at the
          same time will have identical indicies.
        - Example:
            index                  amount   price    symbol
            2004-01-09 12:18:01    483      324.12   'AAPL'
            2004-01-09 12:18:01    122      83.10    'MSFT'
            2004-01-13 14:12:23    -75      340.43   'AAPL'
    gross_lev : pd.Series, optional
        The leverage of a strategy.
         - Time series of the sum of long and short exposure per share
            divided by net asset value.
         - Example:
            2009-12-04    0.999932
            2009-12-07    0.999783
            2009-12-08    0.999880
            2009-12-09    1.000283
    slippage : int/float, optional
        Basis points of slippage to apply to returns before generating
        tearsheet stats and plots.
        If a value is provided, slippage parameter sweep
        plots will be generated from the unadjusted returns.
        Transactions and positions must also be passed.
        - See txn.adjust_returns_for_slippage for more details.
    live_start_date : datetime, optional
        The point in time when the strategy began live trading,
        after its backtest period. This datetime should be normalized.
    round_trips: boolean, optional
        If True, causes the generation of a round trip tear sheet.
    """

    if benchmark_rets is None:
        benchmark_rets = utils.default_returns_func()

    returns, benchmark_rets = returns.align(benchmark_rets,  join='inner')

#    if slippage is not None and transactions is not None:
#        turnover = txn.get_turnover(positions, transactions, period=None, average=False)
#        unadjusted_returns = returns.copy()
#        returns = txn.adjust_returns_for_slippage(returns, turnover, slippage)
#    else:
#        unadjusted_returns = None
    create_returns_tear_sheet(returns = returns,benchmark_rets=benchmark_rets, live_start_date=live_start_date, filename = filename)
    create_interesting_times_tear_sheet(returns, benchmark_rets=benchmark_rets, filename = filename)


def create_returns_tear_sheet(returns, benchmark_rets = None, live_start_date = None, filename = None):
    if benchmark_rets is None:
        benchmark_rets = utils.default_returns_func()
    returns, benchmark_rets = returns.align(benchmark_rets,  join='inner')
#    if returns.index[0] < benchmark_rets.index[0]: # If the strategy's history is longer than the benchmark's, limit
#        returns = returns[returns.index >= benchmark_rets.index[0]] # the strategy
#    elif benchmark_rets.index[0] < returns.index[0]:
#        benchmark_rets = benchmark_rets[benchmark_rets.index >= returns.index[0]]
#    if benchmark_rets.index[-1] > returns.index[-1]:
#        benchmark_rets = benchmark_rets[benchmark_rets.index <= returns.index[-1]]
#    elif returns.index[-1] > returns.index[-1]:
#        returns = returns[returns.index <= benchmark_rets.index[-1]]

    plotting.show_perf_stats(returns, benchmark_rets, filename)
    df_weekly = ts_metrics.aggregate_returns(returns, 'weekly')
    plotting.show_return_range(returns, df_weekly, filename=filename)
    plotting.show_worst_drawdown_periods(returns, filename = filename)

    plt.figure('Statistics of returns: ' + str(returns.name))
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan = 3)
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan = 3)
    ax3 = plt.subplot2grid((3, 3), (2, 0))
    ax4 = plt.subplot2grid((3, 3), (2, 1))
    ax5 = plt.subplot2grid((3, 3), (2, 2))
    plotting.plot_rolling_returns( returns, factor_returns = benchmark_rets, ax = ax1)
    plotting.plot_rolling_sharpe(returns, ax = ax2)
    plotting.plot_monthly_returns_heatmap(returns, ax3)
    plotting.plot_annual_returns(returns, ax4)
    plotting.plot_monthly_returns_dist(returns, ax5)
    #df_weekly = ts_metrics.aggregate_returns(returns, 'weekly')
    #df_monthly = ts_metrics.aggregate_returns(returns, 'monthly')
    plt.figure('Statistics of returns(2): '+ str(returns.name))
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0))
    # plotting.plot_rolling_beta(returns, benchmark_rets, ax=ax1)
    # plotting.plot_rolling_fama_french(returns, ax=ax2)

    plt.figure('Drawdown: ' + str(returns.name))
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0))
    plotting.plot_drawdown_periods(returns, ax = ax1)
    plotting.plot_drawdown_underwater(returns, ax = ax2)

def create_interesting_times_tear_sheet(returns, benchmark_rets=None, legend_loc='best', return_fig=False, filename = None):
    """
    Generate a number of returns plots around interesting points in time,
    like the flash crash and 9/11.
    Plots: returns around the dotcom bubble burst, Lehmann Brothers' failure,
    9/11, US downgrade and EU debt crisis, Fukushima meltdown, US housing
    bubble burst, EZB IR, Great Recession (August 2007, March and September
    of 2008, Q1 & Q2 2009), flash crash, April and October 2014.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    benchmark_rets : pd.Series, optional
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.
    legend_loc : plt.legend_loc, optional
         The legend's location.
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    set_context : boolean, optional
        If True, set default plotting style context.
    """
    rets_interesting = ts_metrics.extract_interesting_date_ranges(returns)
    if len(rets_interesting) == 0:
        warnings.warn('Passed returns do not overlap with any interesting times.', UserWarning)
        return
    print('\nStress Events')
    print(np.round(pd.DataFrame(rets_interesting).describe().transpose().loc[:, ['mean', 'min', 'max']], 3))
    if filename is not None:
        with open(filename, "a") as myfile:
            myfile.write('\nStress Events\n')
            myfile.write(str(np.round(pd.DataFrame(rets_interesting).describe().transpose().loc[:, ['mean', 'min', 'max']], 3)) + '\n')

    if benchmark_rets is None:
        benchmark_rets = utils.default_returns_func()
        # If the strategy's history is longer than the benchmark's, limit
        # strategy
        if returns.index[0] < benchmark_rets.index[0]:
            returns = returns[returns.index > benchmark_rets.index[0]]

    bmark_interesting = ts_metrics.extract_interesting_date_ranges(benchmark_rets)
    num_plots = len(rets_interesting)
    # 2 plots, 1 row; 3 plots, 2 rows; 4 plots, 2 rows; etc.
    num_rows = int((num_plots + 1) / 2.0)
    fig = plt.figure('Interesting Periods: ' + str(returns.name), figsize=(14, num_rows * 6.0))
    gs = gridspec.GridSpec(num_rows, 2, wspace=0.5, hspace=0.5)

    for i, (name, rets_period) in enumerate(rets_interesting.items()):
        # i=0 -> 0, i=1 -> 0, i=2 -> 1 ;; i=0 -> 0, i=1 -> 1, i=2 -> 0
        ax = plt.subplot(gs[int(i / 2.0), i % 2])
        ts_metrics.cum_returns(rets_period).plot(ax=ax, color='forestgreen', label='algo', alpha=0.7, lw=2)
        ts_metrics.cum_returns(bmark_interesting[name]).plot(ax=ax, color='gray', label='SPY', alpha=0.6)
        ax.legend(['algo','SPY'], loc=legend_loc)
        ax.set_title(name, size=14)
        ax.set_ylabel('Returns')
        ax.set_xlabel('')
    plt.show()
    if return_fig:
        return fig

def create_position_tear_sheet(returns, positions, gross_lev=None,show_and_plot_top_pos=2, hide_positions=False, return_fig=False, sector_mappings=None):
    """
    Generate a number of plots for analyzing a
    strategy's positions and holdings.
    - Plots: gross leverage, exposures, top positions, and holdings.
    - Will also print the top positions held.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    positions : pd.DataFrame
        Daily net position values.
    gross_lev : pd.Series, optional
        The leverage of a strategy.
    show_and_plot_top_pos : int, optional
        By default, this is 2, and both prints and plots the
        top 10 positions.
        If this is 0, it will only plot; if 1, it will only print.
    hide_positions : bool, optional
        If True, will not output any symbol names.
        Overrides show_and_plot_top_pos to 0 to suppress text output.
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    sector_mappings : dict or pd.Series, optional
        Security identifier to sector mapping.
        Security ids as keys, sectors as values.
    """
    if hide_positions:
        show_and_plot_top_pos = 0
    vertical_sections = 6 if sector_mappings is not None else 5

    fig = plt.figure(figsize=(14, vertical_sections * 6))
    gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.5, hspace=0.5)
    ax_gross_leverage = plt.subplot(gs[0, :])
    ax_exposures = plt.subplot(gs[1, :], sharex=ax_gross_leverage)
    ax_top_positions = plt.subplot(gs[2, :], sharex=ax_gross_leverage)
    ax_max_median_pos = plt.subplot(gs[3, :], sharex=ax_gross_leverage)
    ax_holdings = plt.subplot(gs[4, :], sharex=ax_gross_leverage)

    positions_alloc = pos.get_percent_alloc(positions)

    if gross_lev is not None:
        plotting.plot_gross_leverage(returns, gross_lev, ax=ax_gross_leverage)

    plotting.plot_exposures(returns, positions_alloc, ax=ax_exposures)
    plotting.show_and_plot_top_positions(returns, positions_alloc, show_and_plot=show_and_plot_top_pos, hide_positions=hide_positions, ax=ax_top_positions)
    plotting.plot_max_median_position_concentration(positions, ax=ax_max_median_pos)
    plotting.plot_holdings(returns, positions_alloc, ax=ax_holdings)

    if sector_mappings is not None:
        sector_exposures = pos.get_sector_exposures(positions, sector_mappings)
        if len(sector_exposures.columns) > 1:
            sector_alloc = pos.get_percent_alloc(sector_exposures)
            sector_alloc = sector_alloc.drop('cash', axis='columns')
            ax_sector_alloc = plt.subplot(gs[5, :], sharex=ax_gross_leverage)
            plotting.plot_sector_allocations(returns, sector_alloc, ax=ax_sector_alloc)

    for ax in fig.axes:
        plt.setp(ax.get_xticklabels(), visible=True)

    plt.show()
    if return_fig:
        return fig