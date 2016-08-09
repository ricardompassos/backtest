import MySQLdb
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
import datetime
import os

def make_db_connection():
	db_host = os.environ["DB_HOST"]
	db_user = os.environ["DB_USER"]
	db_pass = os.environ["DB_PASS"]
	db_db = os.environ["DB_DB"]

	return MySQLdb.connect(db_host, db_user, db_pass, db_db)


def translate_symbols_corr(symbols, sort = False):
    if sort:
        symbols = list(symbols)
        symbols.sort()
    symbols_corr = []
    for i, symbol_i in enumerate(symbols[:-1]):
        for symbol_j in symbols[i+1:]:
            symbols_corr.append(symbol_i + ' _ ' + symbol_j)
    return symbols_corr


#==============================================================================
#==============================================================================
def get_asset_keys_db():
    db = make_db_connection()
    cursor = db.cursor()
    
    query_asset_keys = "select LK_RSECURITY, ST_CODE from PASS_SYS.V_SECURITY"
    select_asset_keys = pd.read_sql(query_asset_keys, db, index_col = 'ST_CODE')

    db.close()
    return select_asset_keys

#==============================================================================
#==============================================================================
def get_returns_db(prediction_method, symbol_list, asset_keys_df):
    db = make_db_connection()
    cursor = db.cursor()

    all_returns = pd.DataFrame()
    for asset in symbol_list:
        asset_key = asset_keys_df.loc[asset].values[0].encode('hex')
        query_1 = "select DT_DATE, NU_VALUE as '%s' from PASS_SYS.T_PASS_CALC where LK_RSECURITY = unhex('%s') and ST_PREDICTIONMETHOD = '%s' order by DT_DATE" % (asset, asset_key, prediction_method)
        select_1= pd.read_sql_query(query_1, db, index_col='DT_DATE')
        all_returns = pd.concat([all_returns, select_1], axis=1, join='outer')
    
    db.close()
    return all_returns


#==============================================================================
#==============================================================================
def get_volatilities_db(prediction_method, symbol_list, asset_keys_df):
    db = make_db_connection()
    cursor = db.cursor()

    all_volatilities = pd.DataFrame()
    for asset in symbol_list:
        asset_key = asset_keys_df.loc[asset].values[0].encode('hex')
        query_1 = "select DT_DATE, NU_VALUE as '%s' from PASS_SYS.T_PASS_CALC where LK_RSECURITY = unhex('%s') and ST_PREDICTIONMETHOD = '%s' order by DT_DATE" % (asset, asset_key, prediction_method)
        select_1= pd.read_sql_query(query_1, db, index_col='DT_DATE')
        all_volatilities = pd.concat([all_volatilities, select_1], axis=1, join='outer')
    
    db.close()
    return all_volatilities

#==============================================================================
#==============================================================================
def get_correlations_db(prediction_method, symbol_list, asset_keys_df):
    db = make_db_connection()
    cursor = db.cursor()

    some_df = pd.DataFrame(columns=symbol_list)
    ind0, ind1 = np.triu_indices(len(some_df.columns), 1)
    symbol_jobs = zip(some_df[ind0], some_df[ind1])
    
    all_correlations = pd.DataFrame()
    for asset1, asset2 in symbol_jobs:
        asset1_key = asset_keys_df.loc[asset1].values[0].encode('hex')
        asset2_key = asset_keys_df.loc[asset2].values[0].encode('hex')

        query_1 = "select DT_DATE, NU_VALUE as '%s _ %s' from PASS_SYS.T_PASS_CORRELATION where LK_RSECURITY1 = unhex('%s') and LK_RSECURITY2 = unhex('%s') and ST_PREDICTIONMETHOD = '%s' order by DT_DATE" % (asset1, asset2, asset1_key, asset2_key, prediction_method)
        select_1 = pd.read_sql_query(query_1, db, index_col='DT_DATE')
        all_correlations = pd.concat([all_correlations, select_1], axis=1, join='outer')

    #AF test
    #all_correlations = all_correlations.replace(np.nan, None)
    #all_correlations = all_correlations.dropna()
    
    db.close()
    return all_correlations

#==============================================================================
#==============================================================================

# OEX INDEX has been hardcoded, because there was a problem with it being put in the sql table (members)
def get_performance_membership(symbol_list, sep = ' / '):
    db = make_db_connection()
    df_res = None
    for symbol in symbol_list:
        query = "select ST_BENCHMARK_NAME from PASS_SYS.V_IN_BENCHMARK where ST_CODE='%s'"  % (symbol)
        df =  pd.read_sql(query, db, coerce_float = False)
        df.index = [symbol]*len(df)
        if df.values[0][0] == None:
           print "Warning: %s has no membership!" % symbol
        if len(df) > 1:
            i_max = np.argmax([len(df.values[row][0].split(sep)) for row in xrange(len(df))])
            df = df.iloc[i_max].to_frame('ST_BENCHMARK_NAME')
            df.index = [symbol]
        if df_res is None:
            df_res = df
        else:
            df_res = pd.concat((df_res, df))
    db.close()
    df_res.index = symbol_list

    if 'OEX INDEX' in symbol_list:
        df_res.loc['OEX INDEX'] =  'Global / Equity / Developed Markets / US'
    #print df_res
    #df_res = df_res.to_frame(name = 'ST_BENCHMARK_NAME') # pd.Series to pd.DataFrame whose columns is 'ST_BENCHMARK_NAME'
    df_res.name = 'Performance Membership'
    return df_res


#==============================================================================
#==============================================================================

def get_underlying(symbol_list, sep = ' / '):
    def _aux(row):
        try:
            return row.values[0].split(' / ')[1]
        except:
            return row.values[0]
    return get_performance_membership(symbol_list, sep = sep).apply(_aux, axis = 'columns')

#==============================================================================
#==============================================================================

def get_members_from_index_db(index = 'SPX INDEX'):
    db = make_db_connection()
    cursor = db.cursor()
    query = "select ST_SECURITY_CODE from PASS_SYS.V_INDEX_MEMBERS where ST_INDEX_CODE='%s' order by ST_SECURITY_CODE" % (index)
    cursor.execute(query)
    members =  [member[0] for member in cursor.fetchall()]
    db.close()
    return members

#==============================================================================
#==============================================================================
def load_market_cap(symbol_list, query_date = '2015-01-04', source =  "PASS / MARKET DATA", transform_to_weights = False, fix_symbols = False):
    """
    This function modifies symbol_list if this list is not sorted.
    """
    if not isinstance(query_date, str):
        if isinstance(query_date, datetime.date):
            query_date = str(query_date)
        else:
            query_date = str(query_date.date())

    if fix_symbols:
        symbol_list.sort()
        symbol_list = [symbol.replace('=', '/') for symbol in symbol_list]
    db = make_db_connection()
    query_symbols = ' OR '.join(["ST_SECURITY_CODE='%s'" % (symbol) for symbol in symbol_list])
    query_hdpks = "select ST_SECURITY_CODE, HD_PK from PASS_SYS.V_SERIE where (%s) and ST_NAME='%s'" %(query_symbols, source)
    df_hdpks = pd.read_sql(query_hdpks,db, index_col = 'ST_SECURITY_CODE')
    df_hdpks['HD_PK'] = df_hdpks['HD_PK'].apply(lambda key : key.encode('hex'))
    query_lkseries = ' OR '.join(["LK_SERIE=unhex('%s')" % (hdpk) for hdpk in df_hdpks.values.ravel().tolist()])
    query_mkt_cap = "select A.DT_DATE, B.ST_SECURITY_CODE, A.NU_CUR_MKT_CAP from PASS_SYS.V_MKTDATA as A LEFT JOIN PASS_SYS.V_SERIE as B on A.LK_SERIE=B.HD_PK where (%s) and A.DT_DATE<='%s' ORDER BY A.DT_DATE DESC LIMIT %d" % (query_lkseries, query_date, len(symbol_list)*10)
    mkt_caps = pd.read_sql(query_mkt_cap, db, index_col = 'ST_SECURITY_CODE')
    mkt_caps = mkt_caps.groupby(axis = 0, level=0).apply(lambda df: df.bfill()['NU_CUR_MKT_CAP'].values[0])#.apply(lambda df: df.bfill())
    db.close()
    mkt_caps.sort_index(inplace =True)
    if fix_symbols: #unfix them
        mkt_caps.index = symbol_list
    if transform_to_weights:
        mkt_caps = mkt_caps / mkt_caps.sum(skipna=True) 
    return mkt_caps.to_frame('NU_CUR_MKT_CAP')
    
# symbols = ['VWO US EQUITY', 'FOX UW EQUITY']#, 'MS UN EQUITY', 'SU FP EQUITY', 'EWJ US EQUITY', 'CS FP EQUITY', 'SAN FP EQUITY', 'PG UN EQUITY', 'CSCO UW EQUITY', 'SAN SQ EQUITY', 'GE UN EQUITY', 'COF UN EQUITY', 'CVX UN EQUITY', 'VIV FP EQUITY', 'SO UN EQUITY', 'GLE FP EQUITY', 'BLK UN EQUITY', 'UL NA EQUITY', 'ABT UN EQUITY', 'EI FP EQUITY', 'MCD UN EQUITY', 'MA UN EQUITY', 'UNA NA EQUITY', 'TGT UN EQUITY', 'DIS UN EQUITY', 'TEF SQ EQUITY', 'ENEL IM EQUITY', 'EWA US EQUITY', 'SBUX UW EQUITY', 'BIIB UW EQUITY', 'EXC UN EQUITY', 'ALV GY EQUITY', 'FDX UN EQUITY', 'CVS UN EQUITY', 'ALL UN EQUITY', 'CA FP EQUITY', 'WMT UN EQUITY', 'CELG UW EQUITY', 'DUK UN EQUITY', 'OR FP EQUITY', 'ENI IM EQUITY', 'MET UN EQUITY', 'MUV2 GY EQUITY', 'GOOGL UW EQUITY', 'SAP GY EQUITY', 'PFE UN EQUITY', 'DPW GY EQUITY', 'C UN EQUITY', 'XOM UN EQUITY', 'EWH US EQUITY', 'FP FP EQUITY', 'ISP IM EQUITY', 'DHR UN EQUITY', 'GS UN EQUITY', 'CAT UN EQUITY', 'SLB UN EQUITY', 'SGO FP EQUITY', 'LLY UN EQUITY', 'COP UN EQUITY', 'OEF US EQUITY', 'AAPL UW EQUITY', 'NOKIA FH EQUITY', 'CMCSA UW EQUITY', 'UTX UN EQUITY', 'EWS US EQUITY', 'BK UN EQUITY', 'HON UN EQUITY', 'BAC UN EQUITY', 'SAF FP EQUITY', 'DOW UN EQUITY', 'EWZ US EQUITY', 'NEE UN EQUITY', 'FRE GY EQUITY', 'LOW UN EQUITY', 'AI FP EQUITY', 'BMW GY EQUITY', 'CL UN EQUITY', 'AIR FP EQUITY', 'MSFT UW EQUITY', 'INGA NA EQUITY', 'MDT UN EQUITY', 'UNP UN EQUITY', 'AXP UN EQUITY', 'TWX UN EQUITY', 'G IM EQUITY', 'ENGI FP EQUITY', 'BAS GY EQUITY', 'RSX US EQUITY', 'COST UW EQUITY', 'OXY UN EQUITY', 'ORA FP EQUITY', 'EOAN GY EQUITY', 'BN FP EQUITY', 'EMC UN EQUITY', 'EWW US EQUITY', 'BAYN GY EQUITY', 'ASML NA EQUITY', 'MO UN EQUITY', 'MC FP EQUITY', 'NKE UN EQUITY', 'DAI GY EQUITY', 'DBK GY EQUITY', 'PCLN UW EQUITY', 'USB UN EQUITY', 'ITX SQ EQUITY', 'PHIA NA EQUITY', 'INTC UW EQUITY', 'HD UN EQUITY', 'VZ UN EQUITY', 'MMM UN EQUITY', 'DD UN EQUITY', 'SIE GY EQUITY', 'BBVA SQ EQUITY', 'BNP FP EQUITY', 'MRK UN EQUITY', 'JPM UN EQUITY', 'T UN EQUITY', 'PEP UN EQUITY', 'BRK=B UN EQUITY', 'FXI US EQUITY', 'UNH UN EQUITY', 'AIG UN EQUITY', 'SX5EEX GR EQUITY', 'BA UN EQUITY', 'UPS UN EQUITY', 'ABI BB EQUITY', 'QQQ US EQUITY', 'JNJ UN EQUITY', 'F UN EQUITY', 'DTE GY EQUITY', 'EMR UN EQUITY', 'AGN UN EQUITY', 'AMGN UW EQUITY', 'MON UN EQUITY', 'UCG IM EQUITY', 'IBE SQ EQUITY', 'GILD UW EQUITY', 'EWT US EQUITY', 'KO UN EQUITY', 'HAL UN EQUITY', 'AMZN UW EQUITY', 'EWY US EQUITY', 'LMT UN EQUITY', 'VPL US EQUITY', 'QCOM UW EQUITY', 'IBM UN EQUITY', 'VOW3 GY EQUITY', 'ACN UN EQUITY', 'SPY US EQUITY', 'BMY UN EQUITY', 'SPG UN EQUITY', 'WFC UN EQUITY', 'DG FP EQUITY', 'GD UN EQUITY', 'RTN UN EQUITY']
# query_date = '2010-01-08'

# weq = load_market_cap(symbols, query_date = query_date, transform_to_weights = True)

# weq_mask = [elem is not np.nan for elem in weq[weq.columns[0]].values.tolist()]


# print load_market_cap(['AAPL UW EQUITY','CC1 COMDTY', 'MSFT UW EQUITY'],transform_to_weights = False)
#print load_market_cap(['AAPL UW EQUITY', 'CC1 COMDTY', 'MSFT UW EQUITY'], transform_to_weights=True)

#==============================================================================
#==============================================================================
def load_data_from_passdb(fields_to_load, source = 'PASS', timezone_indication = 'UTC', start = '1990-01-01', end = None,
                          instruments = ['open','high','low','close','volume'], set_price = 'close', align_dates = True,
                          set_volume_val = 1e6, transform_to_weights = True, add_dummy_volume = False, output_asset_info = False,
                          convert_curr = None):

    # def _translate(str_instrument):
    #     if str_instrument == 'close':
    #         str_instrument = 'last'
    #     if str_instrument == 'market cap':
    #         return 'NU_CUR_MKT_CAP'
    #     return 'NU_PX_'+str_instrument.upper()
    def _translate(str_instrument):
        #if str_instrument == 'close':
        #    #str_instrument = 'last'
        #    return 'NU_CLOSE'
        if str_instrument == 'market cap':
            return 'NU_CUR_MKT_CAP'
        if str_instrument == 'volume':
            return 'NU_PX_'+str_instrument.upper()
        else:
            return 'NU_'+str_instrument.upper()

    start_dt = pd.Timestamp(start, tz = timezone_indication) if start is not None else None
    end_dt = pd.Timestamp(end, tz = timezone_indication) if end is not None else None
    
    instruments = list(instruments)

    instruments.append(set_price)
    query_instruments = [_translate(instrument) for instrument in instruments]

    names = list(instruments[:-1])
    names.append('price')
 
    str_query_instruments = ', '.join(list(query_instruments))

    # ====================================================
    # OPEN DATABASE CONNECTION
    db = make_db_connection()
    cursor = db.cursor()
    # ====================================================

    df_dict = {}
    if align_dates:
        start_date_found = None
        end_date_found = None
    if output_asset_info:
        asset_info_df = pd.DataFrame(index = ['start', 'end'], columns = fields_to_load)

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
        if start is not None:
            query += "and DT_DATE >= '%s'" % start
        if end is not None:
            query += "and DT_DATE <= '%s'" % end
        df_local = pd.read_sql(query, db, index_col = 'DT_DATE')

        # ----------------------
        # Data Post Processing
        df_local = df_local.dropna()
        # Resampling
        df_local = df_local.resample(rule='B').ffill()# fill_method='pad')
        # Change columns name
        df_local.columns = names
        # Change index name
        df_local.index.names = ['Date']
        # Convert the dates to a format that match zipline
        df_local.index = df_local.index.tz_localize(timezone_indication)
        
        if output_asset_info:
            asset_info_df[local_field] = [df_local.index[0], df_local.index[-1]]
            
        if 'volume' in instruments:
            df_local['volume'].replace(to_replace = 0, method = 'bfill', inplace = True)
            if np.all(df_local['volume']) == False:
                df_local['volume'] = set_volume_val
        elif add_dummy_volume:
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

    if align_dates:
        if start_dt is None:
            start_dt = start_date_found
        else:
            if start_date_found > start_dt:
                start_dt = start_date_found
        if end_dt is None:
            end_dt = end_date_found
        else:
            if end_dt is not None:
                if end_date_found < end_dt:
                    end_dt = end_date_found

    # Convert everything to a pandas Panel
    panel_data = pd.Panel(df_dict)
    if convert_curr is not None:
        for instrument in panel_data.minor_axis:
            if instrument not in ['volume']:
                df_adjusted = convert_to(panel_data.minor_xs(instrument), reference_currency=convert_curr, utc = True, source = source)
                for item in panel_data:
                    panel_data[item][instrument] = df_adjusted[item] 

        
    if 'GT2 GOVT' in fields_to_load: # THIS SHOULD HAVE BEEN DONE IN THE DB!
        adj_intruments = ['price', 'close', 'open']
        for adj_instrument in adj_intruments:
            if adj_instrument in panel_data.minor_axis:
                panel_data['GT2 GOVT'][adj_instrument] = 1000. + np.exp( np.log(1.+ panel_data['GT2 GOVT'][adj_instrument]/(360.*100) ).cumsum())

    panel_data = panel_data.truncate(before=start_dt, after=end_dt, copy = False)
    if  'market cap' in instruments and transform_to_weights:
        mkt_weights = panel_data.minor_xs('market cap').apply(lambda col, div: col/div, args = (panel_data.minor_xs('market cap').sum(axis=1, skipna = True).values,)) # the sum is per row (with NaNs skipped), the function is applied per column
        for item in panel_data: # Could not find a pythonic way of setting values...
            panel_data[item]['market cap'] = mkt_weights[item]

    if output_asset_info:
        return panel_data, asset_info_df
    return panel_data

#==============================================================================
#==============================================================================

def get_membership(index = '', instruments = ['name', 'weight', 'mkt_cap'], start_date = None, end_date = None):
    def _translate(str_instrument):
        if str_instrument == 'name':
            str_instrument = 'ST_SECURITY_CODE'
        if str_instrument == 'weight':
            str_instrument = 'NU_WEIGHT'
        if str_instrument == 'mkt_cap':
            str_instrument = 'NU_PCT_MKT_CAP'
        return str_instrument
    query_instruments = ', '.join([_translate(instrument) for instrument in instruments])
    db = make_db_connection()
    query = "select DT_DATE, %s from PASS_SYS.V_INDEX_MEMBERS_HIST where ST_INDEX_CODE='%s'" %(query_instruments, index)
    if start_date is not None:
        query += "and DT_DATE >= '%s'" % start_date
    if end_date is not None:
        query += "and DT_DATE <= '%s'" % end_date
    df_local = pd.read_sql(query, db, index_col = 'DT_DATE')
    df_local.columns = [instruments]
    df_local.index.name = 'Date'
    return df_local


def get_date_limits(df, date_string = '19850101'):
    
    df_res = []
    for column in df.columns.tolist():
        series = df[column].dropna()
        if len(series.index) != 0:
            df_res.append(pd.DataFrame([[series.index[0],series.index[-1]]], index = [column], columns = ['start', 'end']))
        else:
            df_res.append(pd.DataFrame([[pd.to_datetime(date_string, utc = 'True') - BDay(5),pd.to_datetime(date_string, utc = 'True')]], index = [column], columns = ['start', 'end']))
    return pd.concat(df_res)






# 
def get_currency_from_db(symbol_list):
    db = make_db_connection()
    cursor = db.cursor()
    currencies = pd.Series(index = symbol_list)
    for symbol in symbol_list:
        query = "select ST_CURRENCY_ISO from PASS_SYS.V_SERIE where ST_SECURITY_CODE='" + symbol + "' and (ST_SOURCE = 'BLP' or ST_SOURCE = 'BLP_FILE')"
        cursor.execute(query)
        currencies[symbol] = cursor.fetchall()[0][0]
    db.close()
    return currencies

def get_xch_rate(from_ = 'EUR', to = 'USD', utc = False, source = 'PASS'):
    invert = False
    db = make_db_connection()
    cursor = db.cursor()
    cursor.execute("select HD_PK from PASS_SYS.V_SERIE where ST_SECURITY_CODE='%s BGN CURNCY' and (ST_SOURCE = '%s')" % (to+from_, source))
    key = cursor.fetchall()
    if len(key) == 0:
        invert = True
        cursor.execute("select HD_PK from PASS_SYS.V_SERIE where ST_SECURITY_CODE='%s BGN CURNCY' and (ST_SOURCE = '%s')" % (from_+to, source))
        key = cursor.fetchall()
    key = key[0][0].encode('hex')
    query = "select DT_DATE, NU_CLOSE from PASS_SYS.V_MKTDATA where LK_SERIE = unhex('%s')" % (key)
    df = pd.read_sql(query, db, index_col = 'DT_DATE')
    if invert:
        df = 1./df
    df.columns = [to+from_ + ' BGN CURNCY']
    df.index.name = 'Date'
    db.close()
    if utc:
        df.index = df.index.tz_localize('UTC')
    return df

def convert_to(prices, reference_currency, utc = False, source = 'PASS'):
    # prices is a dataframe
    def convertbycur(prices, ref_cur, utc = False, source = 'PASS'):
        if prices.columns.levels[0][0] == ref_cur:
            return prices
        x_rate = get_xch_rate(from_ = prices.columns.levels[0][0], to=ref_cur, utc = utc, source = source)
        x_rate, prices = x_rate.align(prices,join='right', axis = 0)
        x_rate.fillna(method = 'pad', axis = 0, inplace = True)
        return np.multiply(prices, x_rate)
    symbols = list(prices.columns)
    currencies = get_currency_from_db(symbols)
    adj_prices = prices.copy()
    adj_prices.columns =  pd.MultiIndex.from_arrays([currencies.values, list(currencies.index)], names = ['Currency', 'Asset'])
    adj_prices = adj_prices.groupby(level='Currency', axis = 1).apply(convertbycur, reference_currency, utc = utc, source = source)
    adj_prices.columns = adj_prices.columns.droplevel(); adj_prices.index.name = 'Date'
    return adj_prices



def truncate_data(df, df_info, utc = True):
    '''
    Custom data truncation to correct time series
    '''
    df = df.drop([elem for elem in df_info.symbols[df_info.dates == 'remove'].values.ravel().tolist() if elem in df.columns.tolist()], axis = 1)
    df_info = df_info[df_info.dates != 'remove']
    if len(df_info) != 0: 
        for symbol, date in zip(df_info.symbols.values.tolist(), df_info.dates.values.tolist()):
            if symbol in df.columns.tolist():
                df[symbol].loc[:pd.to_datetime(date, utc = utc) - BDay()] = np.NaN 
    return df











