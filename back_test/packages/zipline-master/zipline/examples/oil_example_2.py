
import pandas as pd
from zipline import TradingAlgorithm
from zipline.api import order, sid, symbol, get_open_orders, get_order, cancel_order, commission, set_commission, slippage, set_slippage
from zipline.data.loader import load_bars_from_yahoo


# get data from pass db
import matplotlib.pyplot as plt
import read_db
import forecast_ts
import database_wrapper_zipline

input_data = database_wrapper_zipline.load_data_from_passdb(['USCRWTIC INDEX'], source = 'BLP', timezone_indication = 'UTC', start = '2015-01-01', end = None, instruments = [], set_price = 'close', align_dates = True, set_volume_val = 1e6)
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
    


def handle_data(context, data):
    
    
    
    
    

    curr_price = data[symbol('USCRWTIC INDEX')].price
    curr_date = data[symbol('USCRWTIC INDEX')].datetime
    curr_positions = context.portfolio.positions[symbol('USCRWTIC INDEX')].amount
    cash = context.portfolio.cash

    
    local_historical_data = context.oil_historical_data
    local_historical_data = local_historical_data['USCRWTIC INDEX'][['price']]
    
    
    df_to_forecast = local_historical_data[local_historical_data.index <= curr_date] 
    result = forecast_ts.run(df = df_to_forecast, ts_list = None, freq = 'B', forecast_horizon = 6, start_date = curr_date.strftime('%Y-%m-%d'), method = context.method, processing_params = context.processing_params, expert_params = context.expert_params)
    estimated_return = result.iloc[-1].values[-1]

    # estimated_return = np.random.rand()-1
    
    print cash, curr_positions, curr_price, curr_date, estimated_return
    
    
    if estimated_return < 0 and curr_positions == 0:
        order(symbol('USCRWTIC INDEX'), -100)
    elif estimated_return > 0 and curr_positions != 0:
        order(symbol('USCRWTIC INDEX'), 100)
        
     


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
