from __future__ import division

import numpy as np
import pandas as pd

from pandas.tseries.offsets import BDay
from pandas_datareader import data as web

import MySQLdb
import pandas.io.sql as psql


def rolling_apply2d(df, window=None, func=None, name = '', func_args = (), **func_kwargs):
    """
    Extends pd.rolling_apply to multiple columns, hence 2D.
    Additional Notes
    ----------------
    NaN values are dropped from the final result.
    """
    def f_wrapper(ii, df, f, args, **kwargs):
        x_df = df.ix[map(int,ii)]
        return f(x_df, *args, **kwargs)

    df_out = pd.rolling_apply(pd.Series(range(len(df)), index=df.index), window=window,
                              func=lambda ii: f_wrapper(ii, df, func, func_args, **func_kwargs)).dropna()
    df_out.name = name
    return df_out

def round_two_dec_places(x):
    """
    Rounds a number to 1/100th decimal.
    """
    return np.round(x, 2)

def one_dec_places(x, pos):
    """
    Adds 1/10th decimal to plot ticks.
    """
    return '%.1f' % x

def percentage(x, pos):
    """
    Adds percentage sign to plot ticks.
    """
    return '%.0f%%' % x

def _1_bday_ago():
    return pd.Timestamp.now().normalize() - BDay()

def default_returns_func(symbol='SPY', start=None, end=None, dbmode = True):
    """
    Gets returns for a symbol. Queries Yahoo Finance.
    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. APPL.
    start : date, optional
        Earliest date to fetch data for.
        Defaults to earliest date available.
    end : date, optional
        Latest date to fetch data for.
        Defaults to latest date available.
    Returns
    -------
    pd.Series
        Daily returns for the symbol.
    """
    if start is None:
        start = '1/1/1970'
    if end is None:
        _1_bday = BDay()
        end = pd.Timestamp.now().normalize() - _1_bday

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    if dbmode:
        if symbol == 'SPY':
            symbol = 'SPX INDEX'
        rets = get_symbol_from_db(symbol, start=start, end=end)
    else:
        rets = get_symbol_from_yahoo(symbol, start=start, end=end)
    return rets[symbol]

def get_symbol_from_db(symbol='SPX INDEX', start = None, end = None): # CURRENTLY HARDCODED TO MARKET SERIES
    db = MySQLdb.connect("192.168.51.100","PASS_DEV","Develop-2015","PASS_SYS" )
    cursor = db.cursor()
    query = "select HD_PK from PASS_SYS.V_SERIE where ST_SECURITY_CODE='" + symbol + "'"
    cursor.execute(query)
    data = cursor.fetchall()
    if symbol == 'SPX INDEX':
        descriptor_key =  data[0][0].encode('hex')
    else:
        descriptor_key = data[1][0].encode('hex')
    query = "select DT_DATE, NU_PX_LAST from PASS_SYS.V_MKTDATA where LK_SERIE = unhex('%s')" % (descriptor_key)
    df_query = psql.read_sql(query, db, index_col = 'DT_DATE'); df_query.index.name = 'Date'
    db.close()
    df_query.columns = [symbol]
    if start is not None:
        if start > df_query.index[0]:
            df_query = df_query.loc[start:]
    if end is not None:
        if end < df_query.index[-1]:
            df_query = df_query.loc[:end]
    return df_query.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

def get_currency_from_db(symbol_list):
    db = MySQLdb.connect("192.168.51.100","PASS_DEV","Develop-2015","PASS_SYS" )
    cursor = db.cursor()
    currencies = pd.Series(index = symbol_list)
    for symbol in symbol_list:
        query = "select ST_CURRENCY_ISO from PASS_SYS.V_SERIE where ST_SECURITY_CODE='" + symbol + "' and (ST_SOURCE = 'BLP' or ST_SOURCE = 'BLP_FILE')"
        cursor.execute(query)
        currencies[symbol] = cursor.fetchall()[0][0]
    db.close()
    return currencies

def get_members_from_index_db(index = 'SPX INDEX'):
    db = MySQLdb.connect("192.168.51.100","PASS_DEV","Develop-2015","PASS_SYS" )
    cursor = db.cursor()
    query = "select ST_SECURITY_CODE from PASS_SYS.V_INDEX_MEMBERS where ST_INDEX_CODE='%s' order by ST_SECURITY_CODE" % (index)
    cursor.execute(query)
    members =  [member[0] for member in cursor.fetchall()]
    db.close()
    return members

def get_xch_rate(from_ = 'USD', to = 'EUR'):
    invert = False
    db = MySQLdb.connect("192.168.51.100","PASS_DEV","Develop-2015","PASS_SYS" )
    cursor = db.cursor()
    cursor.execute("select HD_PK from PASS_SYS.V_SERIE where ST_SECURITY_CODE='%s BGN CURNCY' and (ST_SOURCE = 'BLP' or ST_SOURCE = 'BLP_FILE')" % (from_+to))
    key = cursor.fetchall()
    if len(key) == 0:
        invert = True
        cursor.execute("select HD_PK from PASS_SYS.V_SERIE where ST_SECURITY_CODE='%s BGN CURNCY' and (ST_SOURCE = 'BLP' or ST_SOURCE = 'BLP_FILE')" % (to+from_))
        key = cursor.fetchall()
    key = key[0][0].encode('hex')
    query = "select DT_DATE, NU_PX_LAST from PASS_SYS.V_MKTDATA where LK_SERIE = unhex('%s')" % (key)
    df = psql.read_sql(query, db, index_col = 'DT_DATE'); df.columns = [from_+to];  df.index.name = 'Date'
    db.close()
    if invert:
        df = 1./df
    return df

def convert_to(prices, reference_currency):
    #prices is a dataframe
    def convertbycur(prices, ref_cur):
        if prices.columns.levels[0][0] == ref_cur:
            return prices
        x_rate = get_xch_rate(from_ = prices.columns.levels[0][0], to=ref_cur)
        x_rate, prices = x_rate.align(prices,join='right', axis = 0)
        x_rate.fillna(method = 'pad', axis = 0, inplace = True)
        return np.multiply(prices, x_rate)

    symbols = list(prices.columns)
    currencies = get_currency_from_db(symbols)
    adj_prices = prices.copy()
    adj_prices.columns =  pd.MultiIndex.from_arrays([currencies.values, list(currencies.index)], names = ['Currency', 'Asset'])
    adj_prices = adj_prices.groupby(level='Currency', axis = 1).apply(convertbycur, reference_currency)
    adj_prices.columns = adj_prices.columns.droplevel(); adj_prices.index.name = 'Date'
    return adj_prices


#prices = pd.DataFrame(np.ones((5, 3)), columns = ['AAPL UW EQUITY', 'MSFT UW EQUITY', 'XOM UN EQUITY'], index = pd.date_range('20150901', periods = 5, freq = 'B'))
#for idx ,column in enumerate(prices.columns):
#    prices[column] = prices[column] * (idx+1)
#print convert_to(prices, 'EUR')


def get_symbol_from_yahoo(symbol = 'SPY', start=None, end=None):
    """Wrapper for pandas.io.data.get_data_yahoo().
    Retrieves prices for symbol from yahoo and computes returns
    based on adjusted closing prices.
    Parameters
    ----------
    symbol : str
        Symbol name to load, e.g. 'SPY'
    start : pandas.Timestamp compatible, optional
        Start date of time period to retrieve
    end : pandas.Timestamp compatible, optional
        End date of time period to retrieve
    Returns
    -------
    pandas.DataFrame
        Returns of symbol in requested period.
    """
    px = web.get_data_yahoo(symbol, start=start, end=end)
    rets = px[['Adj Close']].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    #rets.index = rets.index.tz_localize("UTC")
    rets.columns = [symbol]
    #rets = rets.fillna('pad')
    #rets = rets.interpolate(method='time') #interpolate eventual missing values
    return rets


def get_fama_french():
    """Retrieve Fama-French factors via pandas-datareader
    from http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    Returns
    -------
    pandas.DataFrame
        Percent change of Fama-French factors
    """
    start = '1/1/1970'
    research_factors = web.DataReader('F-F_Research_Data_Factors_daily',
                                      'famafrench', start=start)[0]
    momentum_factor = web.DataReader('F-F_Momentum_Factor_daily',
                                     'famafrench', start=start)[0]
    five_factors = research_factors.join(momentum_factor).dropna()
    five_factors /= 100.
    #five_factors.index = five_factors.index.tz_localize('utc')
    return five_factors

def load_portfolio_risk_factors(filepath_prefix=None, start=None, end=None):
    """
    Loads risk factors Mkt-Rf, SMB, HML, Rf, and UMD.
    Data is stored in HDF5 file. If the data is more than 2
    days old, redownload from Dartmouth.
    Returns
    -------
    five_factors : pd.DataFrame
        Risk factors timeseries.
    """
    if start is None:
        start = '1/1/1970'
    if end is None:
        end = _1_bday_ago()

    five_factors = get_fama_french()
    return five_factors.loc[start:end]

def load_data_from_passdb(fields_to_load, source = 'BLP', timezone_indication = 'UTC', start = '19900101', end = None,
                          instruments = ['open','high','low','close','volume'], set_price = 'close', align_dates = True,
                          set_volume_val = 1e6, transform_to_weights = True):
						  
    def _translate(str_instrument):
        if str_instrument == 'close':
            str_instrument = 'last'
        if str_instrument == 'market cap':
            return 'NU_CUR_MKT_CAP'
        return 'NU_PX_'+str_instrument.upper()

    if start is not None:
        start = pd.Timestamp(start, tz = timezone_indication)
    if end is not None:
        end = pd.Timestamp(end, tz = timezone_indication)

    instruments.append(set_price)
    query_instruments = [_translate(instrument) for instrument in instruments]

    names = list(instruments[:-1])
    names.append('price')

    str_query_instruments = ', '.join(list(query_instruments))
    # ====================================================
    # OPEN DATABASE CONNECTION
    db = MySQLdb.connect("192.168.51.100","PASS_DEV","Develop-2015","PASS_SYS" )
    cursor = db.cursor()
    # ====================================================

    df_dict = {}
    if align_dates:
        start_date_found = None
        end_date_found = None
    for local_field in fields_to_load:
        query = "select HD_PK, CD_SERIETYPE, ST_INSTRUMENT, ST_SECURITY_NAME, ST_FREQ, ST_NAME from PASS_SYS.V_SERIE where ST_SECURITY_CODE='" + local_field + "'"
        cursor.execute(query)
        data = cursor.fetchall()

        # select the source
        if len(data) != 1:
            for k in xrange(len(data)):
                if data[k][-1].split(' / ')[0] == source:
                    search_index = k
        else:
            search_index = 0

        query = "select DT_DATE, %s from PASS_SYS.V_MKTDATA where LK_SERIE = unhex('%s')" % (str_query_instruments, data[search_index][0].encode('hex'))
        df_local = psql.read_sql(query, db, index_col = 'DT_DATE')
        # ----------------------
        # Data Post Processing
        df_local = df_local.dropna()
        # Resampling
        df_local = df_local.resample(rule='B', how='last', fill_method='pad')
        # Change columns name
        df_local.columns = names
        # Change index name
        df_local.index.names = ['Date']
        # Convert the dates to a format that match zipline
        df_local.index = df_local.index.tz_localize(timezone_indication)
        if 'volume' in instruments:
            df_local['volume'].replace(to_replace = 0, method = 'bfill', inplace = True)
            if np.all(df_local['volume']) == False:
                df_local['volume'] = set_volume_val
        df_dict[local_field] = df_local
        if align_dates:
            if start_date_found is None:
                start_date_found = df_local.index[0]
            else:
                if df_local.index[0] > start_date_found:
                    start_date_found = df_local.index[0]
            if end_date_found is None:
                end_date_found = df_local.index[-1]
            else:
                if df_local.index[-1] < end_date_found:
                    end_date_found = df_local.index[-1]
    #print start_date_found
    if align_dates:
        if start is None:
            start = start_date_found
        else:
            if start_date_found > start:
                start = start_date_found
        if end is None:
            end = end_date_found
        else:
            if end is not None:
                if end_date_found < end:
                    end = end_date_found

    # Convert everything to a pandas Panel
    panel_data = pd.Panel(df_dict)
    panel_data = panel_data.truncate(before=start, after=end, copy = False)
    if  'market cap' in instruments and transform_to_weights:
        mkt_weights = panel_data.minor_xs('market cap').apply(lambda col, div: col/div, args = (panel_data.minor_xs('market cap').sum(axis=1).values,))
        for item in panel_data: # Could not find a pythonic way of setting values...
            panel_data[item]['market cap'] = mkt_weights[item]
    return panel_data



#members = get_members_from_index_db('INDU INDEX')
#members = members[:2]
#a =  load_data_from_passdb(members, instruments = ['open', 'high', 'low', 'close', 'volume', 'market cap'])
#print a[members[0]]

#a = a.minor_xs('market cap') + 1e10 #a.minor_xs('market cap') = 1
#print a[members[29]]
#print a[members[1]]