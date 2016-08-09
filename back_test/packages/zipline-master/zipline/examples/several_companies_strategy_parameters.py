# ----------------------------------------
# Data Parameters
fields_tickers = ['SPX INDEX']# ['AAPL UW EQUITY','CSCO UW EQUITY','GE UN EQUITY' ,'IBM UN EQUITY','INTC UW EQUITY','JPM UN EQUITY','KO UN EQUITY','MSFT UW EQUITY','WMT UN EQUITY']
# ['AAPL UW EQUITY','AXP UN EQUITY','BA UN EQUITY','CAT UN EQUITY','CSCO UW EQUITY','CVX UN EQUITY','DD UN EQUITY','DIS UN EQUITY','GE UN EQUITY','GS UN EQUITY','HD UN EQUITY','IBM UN EQUITY','INTC UW EQUITY','JNJ UN EQUITY','JPM UN EQUITY','KO UN EQUITY','MCD UN EQUITY','MMM UN EQUITY','MRK UN EQUITY','MSFT UW EQUITY','NKE UN EQUITY','PFE UN EQUITY','PG UN EQUITY','TRV UN EQUITY','UNH UN EQUITY','UTX UN EQUITY','V UN EQUITY','VZ UN EQUITY','WMT UN EQUITY','XOM UN EQUITY']
start_date = '2010-01-01'
set_price = 'open'
instruments = ['open', 'close', 'volume']
# ----------------------------------------
# Forecast Parameters
forecast_method = {'method': 'multiscale_autoregressive', 'processing_params': {'returns': False, 'logarithms': False}, 'expert_params': {'exclude_fast': False, 'ml_method': 'Ridge Regression', 'dim': 3, 'ar_order': 2, 'k_number': 20, 'stride_size': 2, 'dim_red': None, 'wavelet_type': 'haar'}}
# forecast_method = {'method': 'multiscale_chaotic_simple', 'processing_params': {'returns': True, 'logarithms': True}, 'expert_params': {'exclude_fast': True, 'ml_method': 'Ridge Regression', 'dim': 6, 'ar_order': 2, 'k_number': 3, 'stride_size': 2, 'dim_red': None, 'wavelet_type': 'haar'}}
forecast_horizon = 15
# ----------------------------------------
# Trading Parameters
trade_quantity = 100
cost=0.0 # default 0.01 
min_trade_cost=0.0 # default 1.0
spread = 0.0
# ----------------------------------------
# Output Parameters
verbose = True
filename = 'results_' + '_'.join([fields_tickers[i].split(' ')[0] for i in range(len(fields_tickers))]) + '_startyear' + start_date.split('-')[0] + '_' + str(forecast_horizon) + 'daysahead' + '.csv'
# ----------------------------------------
