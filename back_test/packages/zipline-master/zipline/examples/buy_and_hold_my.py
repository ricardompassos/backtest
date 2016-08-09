#!/usr/bin/env python
#
# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pandas as pd
from zipline import TradingAlgorithm
from zipline.api import order, sid, symbol, record, commission, set_commission, slippage, set_slippage
from zipline.data.loader import load_bars_from_yahoo
from zipline.api import schedule_function, date_rules, time_rules


import database_wrapper_zipline



fields_tickers = ['AAPL UW EQUITY']
start_date = '2016-01-01'


input_data = database_wrapper_zipline.load_data_from_passdb(fields_tickers, source = 'BLP', timezone_indication = 'UTC', start = start_date, end = None, instruments = ['close', 'volume'], set_price = 'close', align_dates = True, set_volume_val = 1e6)



def initialize(context):
    context.has_ordered = False
    set_commission(commission.PerShare(cost=0.0, min_trade_cost=0.0))
    set_slippage(slippage.FixedSlippage(spread=0.0)) 
    schedule_function(func0, date_rules.week_end())
    
# time_rules.market_close(hours=0, minutes=15)

def func0(context, data):
    order(symbol('AAPL UW EQUITY'), 10)
    print 'price: ', data[symbol('AAPL UW EQUITY')].price
    # order(symbol('AAPL UW EQUITY'), -10)
    print context.portfolio.cash
    print context.get_datetime().date()
    print "==========================="   
    
    
    
def handle_data(context, data):
    print "----------------"
    print context.get_datetime().date(), data[symbol('AAPL UW EQUITY')].price
    print "----------------"
    
    pass  
    
    
    
    
    
# def handle_data(context, data):
#     
#     order(symbol('AAPL UW EQUITY'), 10)
#     print 'price: ', data[symbol('AAPL UW EQUITY')].price
#     # order(symbol('AAPL UW EQUITY'), -10)
#     print context.portfolio.cash
#     print "==========================="
    

  
algo = TradingAlgorithm(initialize=initialize, handle_data=handle_data)
results = algo.run(input_data)

