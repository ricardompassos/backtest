
import pandas as pd
from zipline import TradingAlgorithm
from zipline.api import order, sid, symbol, get_open_orders, get_order, cancel_order, commission, set_commission, slippage, set_slippage
from zipline.data.loader import load_bars_from_yahoo


# get data from pass db
import matplotlib.pyplot as plt
import read_db
import forecast_ts
import database_wrapper_zipline

input_data = database_wrapper_zipline.load_data_from_passdb(['USCRWTIC INDEX'], source = 'BLP', timezone_indication = 'UTC', start = '2000-01-01', end = None, instruments = [], set_price = 'close', align_dates = True, set_volume_val = 1e6)
# (['USCRWTIC INDEX'], source = 'BLP', timezone_indication = 'UTC', start_date = '2015-01-01')
# (fields_to_load, source = 'BLP', timezone_indication = 'UTC', start = '19900101', end = None, instruments = ['open','high','low','close','volume'], set_price = 'close', align_dates = True, set_volume_val = 1e6)

print "Data is loaded"

# print input_data['USCRWTIC INDEX']
import numpy as np

def initialize(context):
    
    context.has_ordered = False
    context.order_id = None
    context.counter = 0
    context.oil_historical_data = database_wrapper_zipline.load_data_from_passdb(['USCRWTIC INDEX'], source = 'BLP', timezone_indication = 'UTC', start = None, end = None, instruments = [], set_price = 'close', align_dates = True, set_volume_val = 1e6)
    context.expert_params = {'exclude_fast': True, 'ml_method': 'Ridge Regression', 'dim': 3, 'ar_order': 2, 'k_number': 20, 'stride_size': 2, 'dim_red': None, 'wavelet_type': 'haar'}
    context.processing_params = {'returns': False, 'logarithms': True}
    context.method = 'multiscale_autoregressive'
    
    set_commission(commission.PerShare(cost=0.01, min_trade_cost=1.0))
    set_slippage(slippage.FixedSlippage(spread=0.0)) # remove slippage
    
    context.is_invested = False
    context.last_observed_price = 0
    context.expected_price = 0
    context.expiration_date_forecast = 0
    
    # probably there are better ways to do this but for now I dont mind
    context.shorted = False
    context.longed = False

def handle_data(context, data):
    
    print context.is_invested, data[symbol('USCRWTIC INDEX')].datetime, context.portfolio.positions[symbol('USCRWTIC INDEX')].amount
    
    print data[symbol('USCRWTIC INDEX')].price, context.last_observed_price, context.expected_price
    print "=========================="
    
    if not context.is_invested:
        # make a prediction
        curr_price = data[symbol('USCRWTIC INDEX')].price
        curr_date = data[symbol('USCRWTIC INDEX')].datetime
        curr_positions = context.portfolio.positions[symbol('USCRWTIC INDEX')].amount
        cash = context.portfolio.cash

        local_historical_data = context.oil_historical_data
        local_historical_data = local_historical_data['USCRWTIC INDEX'][['price']]
        
    
        df_to_forecast = local_historical_data[local_historical_data.index <= curr_date] 
        result = forecast_ts.run(df = df_to_forecast, ts_list = None, freq = 'B', forecast_horizon = 22, start_date = curr_date.strftime('%Y-%m-%d'), method = context.method, processing_params = context.processing_params, expert_params = context.expert_params)
        
        estimated_return = result.iloc[-1].values[-1]
        
        context.expected_price = result.iloc[-1].values[-2]
        context.expiration_date_forecast = result.iloc[-1].values[0]
        context.last_observed_price = data[symbol('USCRWTIC INDEX')].price
        
        if estimated_return < 0:
            order(symbol('USCRWTIC INDEX'), -100)
            context.shorted = True
        elif estimated_return > 0:
            order(symbol('USCRWTIC INDEX'), 100)        
            context.longed = True

        
        context.is_invested = True
        
    else:
        if data[symbol('USCRWTIC INDEX')].price*(1-1.25/100.0) < context.expected_price and data[symbol('USCRWTIC INDEX')].price*(1+1.25/100.0) > context.expected_price:
            if context.shorted:
                order(symbol('USCRWTIC INDEX'), 100)
                context.shorted = False
            elif context.longed:
                order(symbol('USCRWTIC INDEX'), -100)
                context.longed = False
            
            context.is_invested = False
        
        if context.expiration_date_forecast == data[symbol('USCRWTIC INDEX')].datetime:
            if context.shorted:
                order(symbol('USCRWTIC INDEX'), 100)
                context.shorted = False
            elif context.longed:
                order(symbol('USCRWTIC INDEX'), -100)
                context.longed = False
            context.is_invested = False                

        
    
    


algo = TradingAlgorithm(initialize=initialize, handle_data=handle_data)
results = algo.run(input_data)
  
results.to_csv('results.csv')












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
