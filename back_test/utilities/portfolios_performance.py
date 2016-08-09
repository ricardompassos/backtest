import perf_attr as pa
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math

import MySQLdb
import glob

freqs = ['MONTHLY'] #TODO
group = 'Top' #TODO either 'Top' or 'Equity'


def get_benchmark_from_db(type):
	db = MySQLdb.connect("192.168.51.100","PASS_DEV","Develop-2015","PASS_SYS")
	if type == 'Top':
		symbol = "PERF-GLOBAL"
	elif type == 'Equity': 
		symbol = "SPX INDEX"

	query = "select HD_PK from PASS_SYS.V_SERIE where ST_SECURITY_CODE='%s'"  % (symbol)
	select_0 = pd.read_sql(query, db, coerce_float = False)
	query1 = "select DT_DATE, NU_PX_LAST from PASS_SYS.V_MKTDATA where LK_SERIE = unhex('%s')" % (select_0.values[0][0].encode('hex'))
	bench = pd.read_sql(query1, db, index_col = 'DT_DATE')
	# calculation of benchmark standard deviation by month 
	bench_sharp = bench 
	bench_sharp['Returns'] = bench.pct_change()
	bench_sharp = bench_sharp['Returns']
	bench_st_deviation = pd.groupby(bench_sharp, by=[bench_sharp.index.year,bench_sharp.index.month]).apply(lambda rets: np.std(rets))
	# calculation of benchmark return by month 
	bench = bench.resample("M").ffill()
	bench['Returns'] = bench.pct_change()
	bench = to_multindex(bench)
	bench_returns = bench['Returns'].dropna() 
	bench_sharpe_ratio = bench_returns.to_frame(name = 'Returns') / bench_st_deviation.to_frame(name = 'Returns')
	return bench_returns, bench_sharpe_ratio

def function_ex(row):
    # print row
    aux_1 = np.array(row.index)
    aux_2 = np.array(row.values.tolist())
    return aux_1[np.argsort(aux_2)[::-1]]

def to_multindex(df):
	'''
	Creates a multiidnex from a timestamp index in a DataFrame
	'''
	dates = []
	for index, row in df.iterrows():
		date = index.strftime('%Y-%m-%d').split('-')
		tmp = [int(date[0]), int(date[1].lstrip("0"))]
		dates.append(tmp)
	multi_index = pd.MultiIndex.from_tuples(dates)
	df.set_index(multi_index, inplace = True)
	return df

def best_strategybymonth(type):
	i = 0
	periods = []
	strategies = []
	for array_periods in type:
		i += 1
		for period in array_periods:
			date = pd.to_datetime(str(period[0]) + str(period[1]).zfill(2) + '01')
			periods.append(date)
			strategies.append(i)
	return periods, strategies

def plot_beststrategies(filename, periods, strategies):
	'''
	Generates a figure with the plot of the best strategy for each month of the backtest period
	periods - list of (year,month) for each strategy 
	strategies - id's of the different strategies
	'''
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,8))  # create figure & 1 axis
	ax.set_ylabel('Strategies', fontsize = 12)
	ax.set_xlabel('Time', fontsize = 12)
	ax.set_title("Best Strategy by Month", fontsize = 12)
	plt.ylim([0,15])
	plt.yticks(np.arange(min(strategies), max(strategies)+1, 1.0))
	plt.scatter(periods, strategies, s = 60, c = strategies)
	ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
	fig.savefig(resultsdir + 'Ranking/' + filename + '.png') # save the figure to file
	plt.close(fig)    # close the figure

def get_strategies_in_crashes(resultsdir):
	'''
	Get the best strategy, in terms of return and sharpe ratio, for several stress periods 
	'''
	crashes = pd.read_csv('./Info/Crashes.csv', index_col = 0, header = None)
	new_dates = []
	for date in crashes.index:
		date = date.split("/")
		new_date = pd.to_datetime(str(date[2] + "-" + date[1] + "-" + date[0]))
		new_dates.append(new_date)
	crashes.set_index([new_dates], inplace = True)
	crashes = to_multindex(crashes)
	crashes_copy = crashes
	crashes = pd.concat([crashes, returns_sorted.ix[:,0]], axis = 1, join = "inner")
	crashes1 = pd.concat([crashes_copy, sharpe_sorted.ix[:,0]], axis = 1, join = "inner")
	crashes = pd.concat([crashes, crashes1], axis = 1)
	pd.options.display.float_format = '{:,.0f}'.format  # flot to int
	crashes.to_csv(resultsdir + "Ranking/Best_inCrashes.csv", sep = ';', decimal = ',', index_label = ['Year', 'Month'] , header = ['Best Strategy(return)', 'Best Strategy(Sharpe)'])

#===============================================================================================#
# Get benchmark cumulative returns and sharpe ratio
bench_returns = get_benchmark_from_db(group)[0]
bench_sharpe_ratio = get_benchmark_from_db(group)[1]

# Get the cumulative return/risk (sharpe ratio) of all strategies for each month
for freq in freqs:
    resultsdir = os.getcwd() + '/synology/%s/RESULTS_%s/' % (group, freq)
    if not os.path.exists(resultsdir + "Ranking/"):
        os.makedirs(resultsdir + "Ranking/")

    strategy_ids = [results.split('/')[-1].split('.')[0] for results in glob.glob(resultsdir + '*.csv')]
    i = 0
    ident = 0
    for strategy_id in strategy_ids:
       	ident += 1
       	#leg = leg.append[ident,strategy_id]
       	returns = pd.read_csv(resultsdir + strategy_id + ".csv", index_col = 'Unnamed: 0', parse_dates = True)['returns']
      	returns_copy1 = pd.groupby(returns,by=[returns.index.year,returns.index.month]).apply(lambda rets: pa.ts_metrics.cum_returns(rets).iloc[-1])
      	st_deviation = pd.groupby(returns, by=[returns.index.year,returns.index.month]).apply(lambda rets: np.std(rets))
      	sharpe_ratio = returns_copy1 / st_deviation
      	strategy_result = pd.concat([returns_copy1.to_frame(), st_deviation.to_frame(), sharpe_ratio.to_frame()], axis = 1) 
      	if (i == 0):
      		sratio = pd.DataFrame(index = strategy_result.index)
      		sratio = pd.concat([sratio, sharpe_ratio.to_frame(name = ident)], axis = 1)
      		cum_returns = pd.DataFrame(index = strategy_result.index)
      		cum_returns = pd.concat([cum_returns, returns_copy1.to_frame(name = ident)], axis = 1)
      		i = 1
      	else:
      		sratio = pd.concat([sratio, sharpe_ratio.to_frame(name = ident)], axis = 1)
      		cum_returns = pd.concat([cum_returns, returns_copy1.to_frame(name = ident)], axis = 1)

bench_returns.name = 14
bench_sharpe_ratio.columns = [14]
cum_returns = pd.concat([cum_returns, bench_returns], axis = 1, join = 'inner')
sratio = pd.concat([sratio, bench_sharpe_ratio], axis = 1, join = 'inner')
#sratio.to_csv(resultsdir + "Ranking/" + "Sharpe_Ratio.csv", sep = ';', index_label = ['Year', 'Month'], decimal = ',')
#cum_returns.to_csv(resultsdir + "Ranking/" + "Returns.csv", sep = ';', index_label = ['Year', 'Month'], decimal = ',')

# ranking for each month of the best strategies according to the sharpe ratio adn return
sharpe_sorted = sratio.apply(function_ex, axis=1)
sharpe_sorted.columns = ['Best %s'%(i+1) for i in xrange(14)]
#sharpe_sorted.to_csv(resultsdir + "Ranking/" + "Sharpe_sorted.csv", sep = ';', index_label = ['Year', 'Month'], decimal = ',') #all the results sorted
overall_sharp = sharpe_sorted.ix[:, 0:3].apply(pd.Series.value_counts) # the # of occurences in 3 best columns

returns_sorted = cum_returns.apply(function_ex, axis=1)
returns_sorted.columns = ['Best %s'%(i+1) for i in xrange(14)]
#returns_sorted.to_csv(resultsdir + "Ranking/" + "Returns_sorted.csv", sep = ';', index_label = ['Year', 'Month'], decimal = ',') #all the results sorted
overall_returns = returns_sorted.ix[:, 0:3].apply(pd.Series.value_counts) # the # of occurences in 3 best columns

get_strategies_in_crashes(resultsdir)

avg_return = []
sharpe_mean = []
return_periods = []
sharpe_periods = []
for i in range(1,15):
	# months where each strategy was Best 1 in sharpe ratio
	sharp_months = sharpe_sorted[sharpe_sorted.ix[:, 0] == i].index.values
	sharpe_periods.append(sharp_months)
	sharpe_mean.append(np.mean(sratio.ix[sharp_months][i]))
	# months where each strategy was Best 1 in return
	return_months = returns_sorted[returns_sorted.ix[:,0] == i].index.values
	return_periods.append(return_months)
	avg_return.append(np.mean(cum_returns.ix[return_months][i]))

overall_returns['Average_Return_Best1'] = avg_return
overall_sharp['Average_Sharpe_Ratio_Best1'] = sharpe_mean

#overall_sharp.to_csv(resultsdir + "Ranking/" + "Overall_Sharp.csv", index_label = ['Strategy'] ,sep = ';', decimal = ',')
#overall_returns.to_csv(resultsdir + "Ranking/" + "Overall_Returns.csv", index_label = ['Strategy'], sep = ';', decimal = ',')

#  best strategy in terms of return for each month 
periods = best_strategybymonth(sharpe_periods)[0]
strategies = best_strategybymonth(sharpe_periods)[1]
periods1 = best_strategybymonth(return_periods)[0]
strategies1 = best_strategybymonth(return_periods)[1]

# plot best strategy for every month 
plot_beststrategies('Sharpe', periods, strategies)
plot_beststrategies('Returns', periods1, strategies1)

