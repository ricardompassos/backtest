import sys
import os
import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)

import numpy as np
import pandas as pd
#from scipy import linalg

from pandas.tseries.offsets import BDay, BMonthBegin, BMonthEnd

from zipline.api import slippage, set_slippage
from zipline.api import commission, set_commission
from zipline.api import symbol, order_target_percent, get_open_orders, order_target
from zipline import TradingAlgorithm
from zipline.api import schedule_function, date_rules, record

#import portfolio_models as pf_m


from AbstractGroup import AbstractGroup
import utils

import glob
import time

#==============================================================================
#  GLOBAL VARIABLES
#==============================================================================
working_dir = os.getcwd()



global PARAMS
global ASSET_READER
global CORR_READER
global VOL_READER
global RETS_READER
global REBALANCE_DATES_READER
ASSET_READER = working_dir + 'Info/assets.txt'
REBALANCE_DATES_READER = working_dir + 'Info/'


CORR_READER = working_dir + 'PREDICTIONS/CORRELATION_SGARCH22' # +'/$SCHEDULER/all.csv'
VOL_READER = working_dir + 'PREDICTIONS/VOLATILITY_SGARCH22'
RETS_READER = working_dir + 'PREDICTIONS/RETURNS_CLASSIC'

#global SYMBOL_GROUPS
#SYMBOL_GROUPS = ['Equity', 'Bonds', 'Commodities']

global REFERENCE_GROUP_SYMBOL
REFERENCE_GROUP_SYMBOL = {
                            'Equity': 'MXWD INDEX',
                            'Bonds':'BNDGLB INDEX',
                            'Commodities': 'CRY INDEX',
                            'Alternatives': 'HEDGNAV INDEX',
                            'Cash':'GT2 GOVT'
                            }

global SCHEDULER
global REBALANCE_ACTION_TYPE
global START_DATE_BT_STR
global END_DATE_BT_STR
SCHEDULER = 'Weekly'
REBALANCE_ACTION_TYPE = 'close'
START_DATE_BT_STR = '20000101'
END_DATE_BT_STR = None

global REF_CURNCY
global SLIPPAGE_SPREAD
global COMMISSION_COST_PER_SHARE
global COMMISSION_MIN_TRADE_COST
REF_CURNCY = None
SLIPPAGE_SPREAD = 0.0
COMMISSION_COST_PER_SHARE = 0.1
COMMISSION_MIN_TRADE_COST = 1.0


global DATES_CHECK
DATES_CHECK = {'Daily': BDay(1), 'Weekly': BDay(10), 'Monthly': BDay(25)}

global ID
ID = 1

#==============================================================================

def store_backtest_results_db(optimization_method, returns_method, volatility_method, correlation_method, backtest_universe, results):
    db = utils.make_db_connection()
    cursor = db.cursor()

    for date in results.index:
        allocation_dict = results['allocation'].loc[date]
        if (type(allocation_dict) is float and np.isnan(allocation_dict)):
            continue
        
        allocation_str = "{"
        for key in allocation_dict.keys():
            if allocation_str != "{":
                allocation_str += ", "
            allocation_str += '"%s": %.4f' % (key, round(allocation_dict[key], 4))
        allocation_str += "}"

        return_value = results['returns'].loc[date]
        if np.isnan(return_value):
            return_value = 0.0

        query = """insert into PASS_SYS.T_PASS_BACKTEST set ST_OPTIMIZATIONMETHOD = '%s', 
                    ST_RETURNPREDICTIONMETHOD = '%s', 
                    ST_VOLATILITYPREDICTIONMETHOD = '%s', 
                    ST_CORRELATIONPREDICTIONMETHOD = '%s', 
                    DT_DATE = '%s',
                    ST_UNIVERSE = '%s', 
                    CL_ALLOCATION = '%s', 
                    NU_RETURNS = '%f'
                    """ % (optimization_method, 
                        returns_method, 
                        volatility_method, 
                        correlation_method, 
                        date, 
                        backtest_universe,
                        allocation_str,
                        return_value)
        cursor.execute(query)
        db.commit()

    db.close()
    return



def initialize(context, date_limits, verbose, symbols):
    print "in initialize"
    start_time_initialize = time.time()
    context.verbose = verbose
    # Define rebalance frequency
    context.scheduler = SCHEDULER
    if context.scheduler  == 'Daily':
        schedule_function(rebalance)
    if context.scheduler  == 'Weekly':
        schedule_function(rebalance, date_rules.week_end())
    elif context.scheduler  == 'Monthly':
        schedule_function(rebalance, date_rules.month_end())

    context.date_limits = date_limits

    # Set slippage model and commission model
    set_slippage(slippage.FixedSlippage(spread=SLIPPAGE_SPREAD))
    set_commission(commission.PerShare(cost=COMMISSION_COST_PER_SHARE, min_trade_cost=COMMISSION_MIN_TRADE_COST))

    context.symbols = symbols
    symbol_membership = utils.get_underlying(context.symbols, sep = ' / ')
    symbol_membership.index = context.symbols
    context.symbol_membership = symbol_membership
    if REFERENCE_GROUP_SYMBOL['Cash'] in context.symbol_membership.index:
        context.symbol_membership.loc[REFERENCE_GROUP_SYMBOL['Cash']] = 'Cash'
    
    asset_keys = utils.get_asset_keys_db()
    if RETS_READER is not None:
        returns = utils.get_returns_db(RETS_READER, context.symbols, asset_keys)

    else:
        returns = None
    if VOL_READER is not None and CORR_READER is not None:
        volatility = utils.get_volatilities_db(VOL_READER, context.symbols, asset_keys)
        correlations = utils.get_correlations_db(CORR_READER, context.symbols, asset_keys)

    else:
        volatility, correlations = [None]*2
    data_dict = {'returns': returns, 'volatility': volatility, 'correlations': correlations}
    context.data_dict = data_dict


    #print returns
    #print ''
    #print ''
    #print volatility
    #print ''
    #print ''
    #print correlations
    #print asfasfasf

    # Group
    group = "Top"
    if len(sys.argv) >= 3 and 'ASSETTYPE=' in sys.argv[2]:
        group = sys.argv[2].split('=')[-1]
    params_dict = {}
    params_dict.update(data_dict)
    params_dict.update(PARAMS['params'])

    ag_name = 'Top'
    if group == 'Equity':
        ag_name = PARAMS['params']['group']
    context.group = AbstractGroup(context.symbols, name = ag_name, start=START_DATE_BT_STR, end = END_DATE_BT_STR, verbose = verbose, strategy_id = ID, scheduler = SCHEDULER , **params_dict)
    #print "DEBUG [context.group.updated_symbols] : \n", context.group.updated_symbols
    context.round_weights = 4
    context.ref_group_df = pd.DataFrame(pd.Series({v:k for k,v in REFERENCE_GROUP_SYMBOL.items()}))
    context.active_orders = []

    if verbose:
        print "INFO: Initialize took %f s." % (time.time() - start_time_initialize)
    
    
def handle_data(context, data):
    pass

def deallocate_(context):
    # order_target_percent(symbol(REFERENCE_GROUP_SYMBOL['Cash']), 1.0)
    for symbol_ in context.active_orders:
        order_target_percent(symbol(symbol_), 0.0)
        # order_target(symbol(symbol_), 0.0)
    context.active_orders = []

    if context.verbose:
        print "Record Variables for Performance Analysis."
    out_dict = {}
    for symbol_ in context.symbols:
        out_dict[symbol_] = 0.0
        if context.verbose:
            print "Recording weight %s for symbol %s." %(0.0, symbol_)
    record(allocation = out_dict)
    #print "########################################"
    #print 'IN DEALLOCATE'
    #print 'Active orders: ', get_open_orders()
    #print "########################################"


def rebalance(context, data):
    start_time_rebalance = time.time()
    curr_date = context.get_datetime().date()
    # if curr_date < pd.to_datetime('2009-01-01').date():
    #     return
    # if curr_date > pd.to_datetime('2009-06-01').date():
    #     print sdfsdfsdfsdf 

    context.active_orders = []

    if context.verbose:
        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
        print "New iteration at: ", curr_date
        print context.portfolio
    # Build list of the current active symbols
    active_symbols = []
    cur_date = context.get_datetime()
    for symbol_ in context.symbols:
        if cur_date >= context.date_limits.loc[symbol_]['start'] and  cur_date + DATES_CHECK[SCHEDULER] <= context.date_limits.loc[symbol_]['end']:
            active_symbols.append(symbol_)

    if context.verbose: print "INFO: Ellapsed time : %f s." % (time.time() - start_time_rebalance)


    context.group.update(curr_date, active_symbols, round = context.round_weights, verbose = context.verbose)
    if context.group.weights is None:
        deallocate_(context)
        record(iteration_time = time.time() - start_time_rebalance)
        if context.verbose:
            print "Warning: Quitting Allocation because %s group has no weights." % context.group.name
            print "INFO: This iteration took: %f s."  % (time.time() - start_time_rebalance)
        return

    if context.verbose:
        print "Top Group Allocation:"
        print 'Weights:\n', context.group.weights
        print 'Updated Symbols: ', context.group.updated_symbols
        print "---------------------"
    if context.verbose: print "INFO: Ellapsed time : %f s." % (time.time() - start_time_rebalance)


    # Place the orders - the extended_weights of the mixed group assets
    if context.verbose:
        print "Performing Allocation"

    total_weight = 0.0 # sum of absolute weights
    for symbol_ in context.symbols:
        if symbol_ in context.group.updated_symbols:
            context.active_orders.append(symbol_)
            weight = context.group.weights.values.ravel()[context.group.updated_symbols.index(symbol_)]
            if context.verbose:
                print 'Allocating for %s in %s.' % (weight, symbol_)
            total_weight += abs(weight)
        else:
            weight = 0.0
        try:
            order_target_percent(symbol(symbol_), weight)
        except:
            pass
    if context.verbose: print "Total Weight: ", total_weight    
    if total_weight > 1.0:
        raise ValueError('Weight is larger than one!')


    # SAVE MORE VARIABLES
    if context.verbose:
        print "Record Variables for Performance Analysis."
    out_dict = {}
    for symbol_ in context.symbols:
        out_dict[symbol_] = 0
        if symbol_ in context.group.updated_symbols:
            if context.verbose:
                print "Recording weight %s for symbol %s." %(context.group.weights.values.ravel()[context.group.updated_symbols.index(symbol_)], symbol_)
            out_dict[symbol_] = context.group.weights.values.ravel()[context.group.updated_symbols.index(symbol_)]
    record(allocation = out_dict)
    record(iteration_time = time.time() - start_time_rebalance)

    if context.verbose:
        print "=================================="
    if context.verbose:
        print "INFO: This iteration took: %f s."  % (time.time() - start_time_rebalance)


    filename = os.getcwd() + '/ITERATION_TIMES_LOG_'+ SCHEDULER.upper() + '/' + ID + '.txt'
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'a') as logfile:
        logfile.write(str(time.time() - start_time_rebalance) + '\n')
    #print "INFO: This iteration took: %f s."  % (time.time() - start_time_rebalance)



def run_backtest(params, instruments = ['close'], filename = None, verbose = False, asset_type = 'Top'):
    # instruments says to wichi value(s) we look at (open, close, etc)
    # Set global variables
    global PARAMS

    try:
        thismodule = sys.modules[__name__]
        general_parameters = params['general_parameters']
        for key, value in general_parameters.iteritems():
            setattr(thismodule, key, value)

    except:
        print "Warning: General parameters loading was not completed successfully! However, going to proceed with the default parameters..."

    PARAMS = params
    
    stocks = np.unique(np.loadtxt(ASSET_READER, dtype=str, delimiter='/n'))
    stocks = [str(stocks[i]) for i in xrange(len(stocks))] # convert elements in stocks from numpy._string to python.string
    stocks.sort()


    returns_method = RETS_READER if RETS_READER != 'None' else 'NONE'
    volatility_method = VOL_READER if VOL_READER != 'None' else 'NONE'
    correlation_method = CORR_READER if CORR_READER != 'None' else 'NONE'

    start_t = time.time()

    if verbose: print "Going to read data... "
    last_date_db = START_DATE_BT_STR

    db = utils.make_db_connection()
    cursor = db.cursor()
    query_1 = """select DT_DATE 
                from PASS_SYS.T_PASS_BACKTEST 
                where ST_OPTIMIZATIONMETHOD = '%s' and 
                    ST_RETURNPREDICTIONMETHOD = '%s' and 
                    ST_VOLATILITYPREDICTIONMETHOD = '%s' and 
                    ST_CORRELATIONPREDICTIONMETHOD = '%s' and 
                    ST_UNIVERSE = '%s' 
                    order by DT_DATE
                    """ % (ID, 
                        returns_method, 
                        volatility_method, 
                        correlation_method, 
                        'Top')#TODO should be ASSET_READER instead of hardcode 'Top'
    select_1= pd.read_sql_query(query_1, db, index_col='DT_DATE')
    if len(select_1) > 0:
        last_date_db = select_1.index[-1]
    db.close()
    

    #TODO END_DATE_BT_STR should be recalculated to the last rebalance date so we don't have the last month with 0 allocations
    data = utils.load_data_from_passdb(stocks, instruments = instruments, set_price = REBALANCE_ACTION_TYPE, start = last_date_db,
                                               end = END_DATE_BT_STR, align_dates = False, transform_to_weights = False, add_dummy_volume = True,
                                               output_asset_info = False, convert_curr = REF_CURNCY)
    df_info = pd.read_csv('./Info/assets_info.csv')

    data_price_df = utils.truncate_data(data.minor_xs('price'), df_info)

    symbols = data_price_df.columns.tolist()
    data = data[symbols]
    for symbol_ in symbols:
        data[symbol_]['price'] = data_price_df[symbol_]


    if verbose: print "Time to read data: ", time.time() - start_t
    date_limits = utils.get_date_limits(data.minor_xs('price'))
    print 'get_date_limits'
    print date_limits
    print ''
    print ''
    print agdjagj

    algo = TradingAlgorithm(initialize=initialize, handle_data = handle_data, date_limits = date_limits, verbose = verbose, symbols = symbols)

    if verbose: print "Going to run..."
    results = algo.run(data)
    results.index = results.index.date

    #TODO should be ASSET_READER instead of hardcode 'Top'
    store_backtest_results_db(ID, returns_method, volatility_method, correlation_method, 'Top', results)
    
    #TODO don't send to csv unless there's some option to do it
    if filename is not None:
        results.to_csv(filename)

    return results

if __name__ == '__main__':
    import parameters
    results = run_backtest(parameters.parameters, verbose = True)
