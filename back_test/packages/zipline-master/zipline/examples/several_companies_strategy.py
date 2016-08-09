
import pandas as pd
from zipline import TradingAlgorithm
from zipline.api import order, sid, symbol, get_open_orders, get_order, cancel_order, commission, set_commission, slippage, set_slippage
# from zipline.data.loader import load_bars_from_yahoo # this is no longer necessary


# get data from pass db
import matplotlib.pyplot as plt
import read_db
import forecast_ts
import database_wrapper_zipline
import several_companies_strategy_parameters as params

fields_tickers = params.fields_tickers # ['AAPL UW EQUITY','AXP UN EQUITY']
# field_ticker = 'SPX INDEX' # 'USCRWTIC INDEX' # 'AAPL UW EQUITY' # 'USCRWTIC INDEX'
start_date = params.start_date # '2000-01-01'


input_data = database_wrapper_zipline.load_data_from_passdb(fields_tickers, source = 'BLP', timezone_indication = 'UTC', start = start_date, end = None, instruments = ['close', 'volume'], set_price = 'close', align_dates = True, set_volume_val = 1e6)



import numpy as np
 
  
  
def initialize(context):
    
    context.strategy_verbose = params.verbose
    
    context.is_invested = [False]*len(fields_tickers) # indication if we are invested in a particular asset
    context.shorted = [False]*len(fields_tickers) # indication if we are short in a particular asset
    context.longed = [False]*len(fields_tickers) # indication if we are long in a particular asset

    context.expected_prices = [0]*len(fields_tickers) # expected price for a particular asset in a given period
    context.expected_returns = [0]*len(fields_tickers) # expected return for a particular asset in a given period
    context.expiration_date_forecasts = [0]*len(fields_tickers) # expiration date for the forecast of a particular asset
      

    # get the historical data to perform the forecasts
    context.historical_data = database_wrapper_zipline.load_data_from_passdb(fields_tickers, source = 'BLP', timezone_indication = 'UTC', start = None, end = None, instruments = [], set_price = 'close', align_dates = True, set_volume_val = 1e6)

    set_commission(commission.PerShare(cost=params.cost, min_trade_cost=params.min_trade_cost))
    set_slippage(slippage.FixedSlippage(spread=params.spread))


def handle_data(context, data):
    
    if params.verbose:
        print "============================="
        print "Going for another iteration"
        
    for i in xrange(len(fields_tickers)):

        if params.verbose:
            print fields_tickers[i], context.is_invested[i], data[symbol(fields_tickers[i])].price, data[symbol(fields_tickers[i])].datetime, context.portfolio.positions[symbol(fields_tickers[i])].amount
        
        if not context.is_invested[i]:
            curr_price = data[symbol(fields_tickers[i])].price
            curr_date = data[symbol(fields_tickers[i])].datetime
            curr_positions = context.portfolio.positions[symbol(fields_tickers[i])].amount
            cash = context.portfolio.cash 
            # print curr_price, curr_date, curr_positions, cash
            
            
            local_historical_data = context.historical_data[fields_tickers[i]][['price']]
            df_to_forecast = local_historical_data[local_historical_data.index <= curr_date] 
            result = forecast_ts.run(df = df_to_forecast, ts_list = None, freq = 'B', forecast_horizon = params.forecast_horizon, start_date = curr_date.strftime('%Y-%m-%d'), method = params.forecast_method['method'], processing_params = params.forecast_method['processing_params'], expert_params = params.forecast_method['expert_params'])
 
            context.expected_returns[i] = result.iloc[-1].values[-1]
            context.expected_prices[i] = result.iloc[-1].values[-2]
            context.expiration_date_forecasts[i] = result.iloc[-1].values[0]  
            
            if context.expected_returns[i] < 0:
                order(symbol(fields_tickers[i]), -params.trade_quantity)
                context.shorted[i] = True
            elif context.expected_returns[i] > 0:
                order(symbol(fields_tickers[i]), params.trade_quantity)        
                context.longed[i] = True
            context.is_invested[i] = True  
        
        else:
            if context.expected_returns[i] > 0:
                if data[symbol(fields_tickers[i])].price > context.expected_prices[i]:
                    order(symbol(fields_tickers[i]), -params.trade_quantity)
                    context.longed[i] = False  
                    context.is_invested[i] = False                     
            else:
                if data[symbol(fields_tickers[i])].price < context.expected_prices[i]:
                    order(symbol(fields_tickers[i]), params.trade_quantity)
                    context.shorted[i] = False  
                    context.is_invested[i] = False 

            if context.expiration_date_forecasts[i] == data[symbol(fields_tickers[i])].datetime:
                if context.shorted[i]:
                    order(symbol(fields_tickers[i]), params.trade_quantity)
                    context.shorted[i] = False
                elif context.longed[i]:
                    order(symbol(fields_tickers[i]), -params.trade_quantity)
                    context.longed[i] = False
                context.is_invested[i] = False 

 
algo = TradingAlgorithm(initialize=initialize, handle_data=handle_data)
results = algo.run(input_data)
   
results.to_csv(params.filename)








