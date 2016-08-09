
import pandas as pd
from zipline import TradingAlgorithm
from zipline.api import order, sid, symbol, get_open_orders, get_order, cancel_order
from zipline.data.loader import load_bars_from_yahoo


# get data from pass db
import matplotlib.pyplot as plt
import read_db
import forecast_ts

input_data = read_db.load_data_from_passdb(['USCRWTIC INDEX'], source = 'BLP', timezone_indication = 'UTC', start_date = '2016-01-01')

print "Data is loaded"

print input_data['USCRWTIC INDEX']
import numpy as np

def initialize(context):
    context.has_ordered = False
    context.order_id = None
    context.counter = 0
    context.oil_historical_data = read_db.load_data_from_passdb(['USCRWTIC INDEX'], source = 'BLP', timezone_indication = 'UTC', start_date = None)

def handle_data(context, data):

    curr_price = data[symbol('USCRWTIC INDEX')].price
    curr_positions = context.portfolio.positions[symbol('USCRWTIC INDEX')].amount
    cash = context.portfolio.cash

    print cash, curr_positions, curr_price
    


    # context.counter += 1

    # if context.counter > 500:
    #     print "Cancelou"
    #     cancel_order(context.order_id)
    # else:
    #     print 'ola'


    random_order = np.random.rand()
 
 
    if random_order > 0.5 and curr_positions == 0:
        order(symbol('USCRWTIC INDEX'), 100)
    elif random_order < 0.5 and curr_positions != 0:
        order(symbol('USCRWTIC INDEX'), -100)

    # print data[symbol('USCRWTIC INDEX')].price
    # print get_open_orders()

    # print get_order(context.order_id)
    # print "=============================="

#     if not context.has_ordered:
#         # for stock in data:
#         #     order(sid(stock), 100)
#         context.order_id = order(symbol('USCRWTIC INDEX'), 100)
#         context.has_ordered = True


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
