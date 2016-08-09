
import pandas as pd
from zipline import TradingAlgorithm
from zipline.api import order, sid, symbol, get_open_orders, get_order, cancel_order, commission, set_commission, slippage, set_slippage
from zipline.data.loader import load_bars_from_yahoo


# get data from pass db
import matplotlib.pyplot as plt
import read_db
import forecast_ts
import database_wrapper_zipline



field_ticker = 'SPX INDEX' # 'USCRWTIC INDEX' # 'AAPL UW EQUITY' # 'USCRWTIC INDEX'
start_date = '2000-01-01'

input_data = database_wrapper_zipline.load_data_from_passdb([field_ticker], source = 'BLP', timezone_indication = 'UTC', start = start_date, end = None, instruments = [], set_price = 'close', align_dates = True, set_volume_val = 1e6)

print "Data is loaded"

import numpy as np

def initialize(context):
    
    context.has_ordered = False
    context.order_id = None
    context.counter = 0
    context.field_ticker = field_ticker
    context.oil_historical_data = database_wrapper_zipline.load_data_from_passdb([field_ticker], source = 'BLP', timezone_indication = 'UTC', start = None, end = None, instruments = [], set_price = 'close', align_dates = True, set_volume_val = 1e6)
    context.expert_params = {'exclude_fast': False, 'ml_method': 'Ridge Regression', 'dim': 3, 'ar_order': 2, 'k_number': 20, 'stride_size': 2, 'dim_red': None, 'wavelet_type': 'haar'}
    context.processing_params = {'returns': False, 'logarithms': True}
    context.method = 'multiscale_autoregressive'
    
    set_commission(commission.PerShare(cost=0.01, min_trade_cost=1.0)) # default
    # set_commission(commission.PerShare(cost=0.0, min_trade_cost=0.0))
    set_slippage(slippage.FixedSlippage(spread=0.0)) # remove slippage
    
    context.is_invested = False
    context.last_observed_price = 0
    context.expected_price = 0
    context.expiration_date_forecast = 0
    context.est_return = 0
    
    # probably there are better ways to do this but for now I dont mind
    context.shorted = False
    context.longed = False
    
    context.trade_quantity = 100
    
    context.verbose_here = True

def handle_data(context, data):
    
    
    if context.verbose_here:
        print context.is_invested, data[symbol(context.field_ticker)].datetime, context.portfolio.positions[symbol(context.field_ticker)].amount
    
    if not context.is_invested:
        # make a prediction
        curr_price = data[symbol(context.field_ticker)].price
        curr_date = data[symbol(context.field_ticker)].datetime
        curr_positions = context.portfolio.positions[symbol(context.field_ticker)].amount
        cash = context.portfolio.cash

        local_historical_data = context.oil_historical_data
        local_historical_data = local_historical_data[context.field_ticker][['price']]
        
    
        df_to_forecast = local_historical_data[local_historical_data.index <= curr_date] 
        result = forecast_ts.run(df = df_to_forecast, ts_list = None, freq = 'B', forecast_horizon = 15, start_date = curr_date.strftime('%Y-%m-%d'), method = context.method, processing_params = context.processing_params, expert_params = context.expert_params)
        
        context.est_return = result.iloc[-1].values[-1]
        
        context.expected_price = result.iloc[-1].values[-2]
        context.expiration_date_forecast = result.iloc[-1].values[0]
        context.last_observed_price = data[symbol(context.field_ticker)].price
        
        if context.est_return < 0:
            order(symbol(context.field_ticker), -context.trade_quantity)
            context.shorted = True
        elif context.est_return > 0:
            order(symbol(context.field_ticker), context.trade_quantity)        
            context.longed = True

        
        context.is_invested = True
        
    else:
        if context.est_return > 0:
            if data[symbol(context.field_ticker)].price > context.expected_price:
                order(symbol(context.field_ticker), -context.trade_quantity)
                context.longed = False  
                context.is_invested = False
        else:
            if data[symbol(context.field_ticker)].price < context.expected_price:
                order(symbol(context.field_ticker), context.trade_quantity)
                context.shorted = False  
                context.is_invested = False          
            
#         if data[symbol(context.field_ticker)].price*(1-1.0/100.0) < context.expected_price and data[symbol(context.field_ticker)].price*(1+1.0/100.0) > context.expected_price:
#             if context.shorted:
#                 order(symbol(context.field_ticker), context.trade_quantity)
#                 context.shorted = False
#             elif context.longed:
#                 order(symbol(context.field_ticker), -context.trade_quantity)
#                 context.longed = False
#             context.is_invested = False
        
        if context.expiration_date_forecast == data[symbol(context.field_ticker)].datetime:
            if context.shorted:
                order(symbol(context.field_ticker), context.trade_quantity)
                context.shorted = False
            elif context.longed:
                order(symbol(context.field_ticker), -context.trade_quantity)
                context.longed = False
            context.is_invested = False                

        
    
    


algo = TradingAlgorithm(initialize=initialize, handle_data=handle_data)
results = algo.run(input_data)
  
results.to_csv('results_' + field_ticker + '.csv')












# def initialize(context):
#     context.has_ordered = False

#     # get historical data to be available
#     print 'reading historical data'
#     context.oil_historical_data = read_db.load_data_from_passdb(['USCRWTIC INDEX'], start_date = None, end_date = None, source = 'BLP', timezone_indication = 'UTC')


# def handle_data(context, data):

#     # print "AQUI"
#     # print context.oil_historical_data['USCRWTIC INDEX'].iloc[-1]
#     # print data.price

#     # print data[context.security].datetime
#     # print data[context.security].price


#     if not context.has_ordered:
#         for stock in data:
#             order(sid(stock), 100)
#         context.has_ordered = True
#     print "=================="

# algo = TradingAlgorithm(initialize=initialize, handle_data=handle_data)
# results = algo.run(input_data)

# results.to_csv('results.csv')


# results.plot()
# plt.show()
