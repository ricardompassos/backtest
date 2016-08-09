from __future__ import division

import pandas as pd
import numpy as np
import warnings
import ts_metrics

def get_alloc_fraction(positions):
    """
    Determines a portfolio's allocations.
    Parameters
    ----------
    positions : pd.DataFrame
        Contains position values.
    Returns
    -------
    allocations : pd.DataFrame
        Positions and their fraction allocations.
    """
    return positions.divide(positions.abs().sum(axis='columns'),axis='rows')

def get_long_short_pos(positions):
    """
    Determines the long and short allocations in a portfolio.
    Parameters
    ----------
    positions : pd.DataFrame
        The positions that the strategy takes over time.
    Returns
    -------
    df_long_short : pd.DataFrame
        Long and short allocations / total net liquidation (fraction)
    """
    pos_wo_cash = positions.drop('cash', axis=1)
    longs = pos_wo_cash[pos_wo_cash > 0].sum(axis=1).fillna(0)
    shorts = pos_wo_cash[pos_wo_cash < 0].abs().sum(axis=1).fillna(0)
    cash = positions.cash
    net_liquidation = longs - shorts + cash
    df_long_short = pd.DataFrame({'long': longs, 'short': shorts})
    return df_long_short.divide(net_liquidation, axis='index')

def get_max_median_position_concentration(positions):
    """
    Finds the max and median long and short position concentrations
    in each time period specified by the index of positions.
    Parameters
    ----------
    positions : pd.DataFrame
        The positions that the strategy takes over time.
    Returns
    -------
    pd.DataFrame
        Columns are max long, max short, median long, and median short
        position concentrations. Rows are timeperiods.
    """
    expos = get_alloc_fraction(positions)
    expos = expos.drop('cash', axis=1)

    longs = expos.where(expos.applymap(lambda x: x > 0))
    shorts = expos.where(expos.applymap(lambda x: x < 0))

    alloc_summary = pd.DataFrame()
    alloc_summary['max_long'] = longs.max(axis=1)
    alloc_summary['median_long'] = longs.median(axis=1)
    alloc_summary['median_short'] = shorts.median(axis=1)
    alloc_summary['max_short'] = shorts.min(axis=1)
    return alloc_summary

def get_sector_exposures(positions, symbol_sector_map):
    """
    Sum position exposures by sector.
    Parameters
    ----------
    positions : pd.DataFrame
        Contains position values or amounts.
        - Example
            index         AAPL          MSFT          CHK           cash
            2004-01-09    13939.380     -15012.993    -403.870      1477.483
            2004-01-12    14492.630     -18624.870    142.630       3989.610
            2004-01-13    -13853.280    13653.640     -100.980      100.000
    symbol_sector_map : dict or pd.Series
        Security identifier to sector mapping.
        Security ids as keys/index, sectors as values.
        - Example:
            {'AAPL' : 'Technology'
             'MSFT' : 'Technology'
             'CHK' : 'Natural Resources'}
    Returns
    -------
    sector_exp : pd.DataFrame
        Sectors and their allocations.
        - Example:
            index         Technology      Natural Resources   cash
            2004-01-09    -1073.613       -403.870            1477.4830
            2004-01-12    -4132.240       142.630             3989.6100
            2004-01-13    -199.640        -100.980            100.0000
    """
    cash = positions['cash']
    positions = positions.drop('cash', axis=1)

    unmapped_pos = np.setdiff1d(positions.columns.values,list(symbol_sector_map.keys()))
    if len(unmapped_pos) > 0:
        warnings.warn('Warning: Symbols %s have no sector mapping.' % (", ".join(map(str, unmapped_pos))), UserWarning)

    sector_exp = positions.groupby(by=symbol_sector_map, axis=1).sum()
    sector_exp['cash'] = cash
    return sector_exp

def geo_excess_ret_attrib(df):
    """
    Computes geometric excess return attribution: geometric asset allocation and stock selection.
    Parameters
    ----------
    df : pd.DataFrame
        Must contain the following columns (notation taken from reference material):
        - w_i : porfolio weight
        - W_i : benchmark weight
        - r_i : portfolio return
        - b_i : benchmark return
        (Optional) Index is the asset/sector descriptor
    Returns
    -------
    df_out : pd.DataFrame
        Copy of df with 'Geometric Asset Allocation' and 'Geometric Stock Selection' columns appended.
    Reference:
    ----------
    C. Bacon, "Practical portfolio performance: measurement and attribution," John Wiley & Sons, 2nd Edition, 2008, pp. 129-132.
    """
    df_out = df.copy()
    b = df['W_i'].T.dot(df['b_i']) # total benchmark return
    df_out['Geometric Asset Allocation'] = (df['w_i'] - df['W_i']) * ((1+df['b_i'])/(1+b) -1)
    bs = df['w_i'].T.dot(df['b_i']) # total semi-notational return
    df_out['Geometric Stock Selection'] = df['w_i'] * (df['r_i'] - df['b_i']) / (1+bs)
    return df_out

def geo_excess_ret_attrib_mp(df, compute_combined = False):
    """
    Wrapper of geo_excess_ret_attrib(): groups by timestamp (axis 0 and level 0) and applies function.
    Parameters
    ----------
    df : pd.DataFrame
        Multi-index dataframe with the following columns:
        - w_i : porfolio weight
        - W_i : benchmark weight
        - r_i : portfolio return
        - b_i : benchmark return
        Must include a multiindex on axis 0:
        - Level 0: timestamp
        - Level 1: Asset/Sector descriptors
    compute_combined : bool
        If True, computes the combined returns, asset allocation and stock selection.
        This information is appended to df_out along axis 0 with level 0 = 'Combined'.
        This combined information is computed as follows
            - returns : cumulative returns (see : ts_metrics.cum_rets())
            - Asset Allocation/Stock Selection : sum of the respective values over all specified timestamps
    Returns
    -------
    df_out : pd.DataFrame
        Copy of df with 'Geometric Asset Allocation' and 'Geometric Stock Selection' columns appended.
    Additional Notes:
    ----------
    See geo_excess_ret_attrib()
    The user is advised to access columns by label, since column order is not enforced.
    """
    df_out = df.groupby(axis ='index', level=0).apply(geo_excess_ret_attrib)
    if compute_combined:
        df_combo = pd.DataFrame(np.full((len(df.index.levels[1]),2), np.nan), columns = ['w_i', 'W_i'], index = df.index.levels[1])
        df_combo['r_i'] = df['r_i'].groupby(level=1, axis = 0).apply(lambda rets: ts_metrics.cum_returns(rets).iloc[-1])
        df_combo['b_i'] = df['b_i'].groupby(level=1, axis = 0).apply(lambda rets: ts_metrics.cum_returns(rets).iloc[-1])
        df_combo[['Geometric Asset Allocation','Geometric Stock Selection']] = df_out[['Geometric Asset Allocation','Geometric Stock Selection']].groupby(level=1, axis=0).sum()
        df_combo.index = pd.MultiIndex.from_product([['Combined'], df_combo.index]) # add 'Combined' to multiindex
        df_out = pd.concat([df_out, df_combo], join='inner', axis = 0) # join outputs into single dataframe
    return df_out


#==============================================================================
# TEST FUNCTIONS
#==============================================================================
def test_create_positions_df():
        symbols = ['AAPL', 'MSFT', 'CHK']
        symbols_p_cash = symbols + ['cash']
        pos_vals = np.array([[13939.380, -15012.993, -403.870, 1477.483],
                             [14492.630, -18624.870, 142.630,  3989.610],
                             [-13853.280, 13653.640, -100.980,  100.000]])
        pos = pd.DataFrame(pos_vals, index = pd.to_datetime(['20040109','20040112','20040113']), columns = symbols_p_cash)
        return pos

def test_print_example(example_description, dict_str_inputs, dict_str_outputs):
    print('Example:\t%s' % example_description)
    print("#############################INPUT#####################################")
    for key, value in dict_str_inputs.iteritems():
        print("%s:\n%s\n" % (key, value))
    print("#############################OUTPUT####################################")
    for key, value in dict_str_outputs.iteritems():
        print("%s:\n%s\n" % (key, value))
    print("#######################################################################")

def test_get_alloc_fraction():
    pos = test_create_positions_df()
    test_print_example('Allocation (fraction)', {'Positions':pos}, {'Allocation (fraction)': get_alloc_fraction(pos)})

def test_get_long_short_pos():
    pos = test_create_positions_df()
    test_print_example('',  {'Positions':pos}, {'Allocation (fraction)': get_long_short_pos(pos)})

def test_get_max_median_position_concentration():
    pos = test_create_positions_df()
    test_print_example('Statistics long/short',  {'Positions':pos}, {'Allocation (fraction)': get_max_median_position_concentration(pos)})

def test_get_sector_exposures():
    pos = test_create_positions_df()
    sectors = ['Technology', 'Natural Resources']
    symbol_sector_map = {'AAPL': sectors[0], 'MSFT': sectors[0], 'CHK':sectors[1]}
    test_print_example('Section Exposure', {'Positions': pos, 'Symbol Sector Map': symbol_sector_map}, {'Positions (by sector)': get_sector_exposures(pos, symbol_sector_map)})
    # ANOTHER EXAMPLE
    #del symbol_sector_map['MSFT']
    #test_print_example('(MSFT mapping is missing)', {'Positions':pos, 'Symbol Sector Map': symbol_sector_map}, {'Positions (by sector)': get_sector_exposures(pos, symbol_sector_map)})

def test_get_excess_ret_attrib():
    # Values taken from Table 5.5: C. Bacon, "Practical portfolio performance: measurement and attribution," John Wiley & Sons, 2nd Edition, 2008, p. 132.
    w_i = np.array([.4, .3, .3]) # portfolio weight
    W_i = np.array([.4, .2, .4]) # benchmark weight
    r_i = np.array([.2, -.05, .06]) # portfolio return
    b_i = np.array([.1, -.04, .08]) # benchmark return
    index = ['UK Equities','JP Equities','US Equities']
    df = pd.DataFrame({'w_i':w_i, 'W_i':W_i, 'r_i':r_i, 'b_i':b_i}, index = index)
    df_out = geo_excess_ret_attrib(df)
    test_print_example('Geometric Excess Return Attibution ', {'df':df}, {'df_out': df_out})

    # Validate results
    asset_allocation_reference = np.array([0.0, -0.98, -0.15])/100.
    assert(np.allclose(np.round(df_out['Geometric Asset Allocation'].values,4), asset_allocation_reference))
    stock_selection_reference = np.array([3.8, -0.28, -0.57]) / 100.
    assert(np.allclose(np.round(df_out['Geometric Stock Selection'].values,4), stock_selection_reference, atol = 1e-4))

def test_geo_excess_ret_attrib_mp():
    # Values taken from Table 5.7: C. Bacon, "Practical portfolio performance: measurement and attribution," John Wiley & Sons, 2nd Edition, 2008, p. 135.
    index = ['UK Equities','JP Equities','US Equities']
    dates = pd.date_range('20100101', periods=2, freq='B') # made-up dates
    # First timestamp:
    w_i_0 = np.array([.4, .4, .2]) # portfolio weight
    W_i_0 = np.array([.4, .2, .4]) # benchmark weight
    r_i_0 = np.array([.1, .05, .1]) # portfolio return
    b_i_0 = np.array([.1, .1, .1]) # benchmark return
    df0 = pd.DataFrame({'w_i':w_i_0, 'W_i':W_i_0, 'r_i':r_i_0, 'b_i':b_i_0}, index = index)
    # Second timestamp:
    w_i_1 = np.array([.4074, .2037, .3889])
    W_i_1 = np.array([.4, .2, .4])
    r_i_1 = np.array([.091, -.159, -.005])
    b_i_1 =np.array([.0, -.127, -.018])
    df1 = pd.DataFrame({'w_i':w_i_1, 'W_i':W_i_1, 'r_i':r_i_1, 'b_i':b_i_1}, index = index)
    # Concatenate and generate a multi index dataframe
    df = pd.concat([df0, df1], join='inner', keys=dates)
    df_out = geo_excess_ret_attrib_mp(df, True)
    test_print_example('Geometric Excess Return Attibution (Multiple Timestamps)', {'df': df}, {'df_out':df_out})

    # Validate results
    asset_allocation_reference = np.zeros(9)
    assert(np.allclose(np.round(df_out['Geometric Asset Allocation'].values,3), asset_allocation_reference))
    stock_selection_reference = np.array([0.0, -1.8, 0.0, 3.8, -0.7, 0.5, 3.8, -2.5, 0.5]) / 100.
    assert(np.allclose(np.round(df_out['Geometric Stock Selection'].values,3), stock_selection_reference))
    combined_r_i = np.array([20, -11.7, 9.5])/100.
    assert(np.allclose(np.round(df_out['r_i'].loc['Combined'].values, 3), combined_r_i))
    combined_b_i = np.array([10, -4, 8])/100.
    assert(np.allclose(np.round(df_out['b_i'].loc['Combined'].values, 3), combined_b_i))


def test_get_excess_ret_attrib2():
    # Example from PASS documentation (A. Eduardo)
    # Groupby attr0
    w_i = np.array([.55, .25, .2])
    W_i = np.array([.5, .4, .1])
    r_i = np.array([.03273, .016, .01])
    b_i = np.array([.0616, .02, .0105])
    index = ['A', 'B', 'C']
    df = pd.DataFrame({'w_i':w_i,'W_i':W_i,'r_i':r_i, 'b_i':b_i}, index = index)
    df_out = geo_excess_ret_attrib(df)
    test_print_example('PASS Documentation (A. Eduardo): group by attr0', {'df':df}, {'df_out':df_out})

    # Validation
    r = sum(df['w_i']*df['r_i'])
    b = sum(df['W_i']*df['b_i'])
    alloc = sum(df_out['Geometric Asset Allocation'])
    selec = sum(df_out['Geometric Stock Selection'])
    assert(np.isclose((1+r)/(1+b) - 1, (1+alloc)*(1+selec) - 1))

# test_geo_excess_ret_attrib_mp()
