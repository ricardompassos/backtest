import os
import time

import sys

import subprocess
import numpy as np

sys.path.append('./predictions')
sys.path.append('./utilities')
sys.path.append('./backtests')


DEBUG = True


if len(sys.argv) != 8:
	print 'ERROR: Parameters missing, run:' 
	print sys.argv[0] + ' <Top|Equity> <Monthly|Weekly|Daily> <file with list of assets> <returns prediction method> <volatility prediction method> <correlation prediction method> <optimization method "MDP" or "MVO,bounds,delta" or "ERC">'
	quit()

DB_HOST = os.environ.get('DB_HOST')
DB_USER = os.environ.get('DB_USER')
DB_PASS = os.environ.get('DB_PASS')
DB_DB = os.environ.get('DB_DB')

if DB_HOST is None or DB_USER is None or DB_PASS is None or DB_DB is None:
	print 'ERROR: database environment variable not set up'
	exit(1)

if DEBUG:
	subprocess.call("rm -rf ./ALLOCATION_ERROR_LOGS_*", shell=True)
	subprocess.call("rm -rf ./ITERATION_TIMES_LOG_*", shell=True)
	subprocess.call("rm -rf ./MergedPredictionResults", shell=True)
	subprocess.call("rm -rf ./PARAMETERS_*", shell=True)
	subprocess.call("rm -rf ./PredictionsResults", shell=True)
	subprocess.call("rm -rf ./synology/%s" % sys.argv[1], shell=True)
	subprocess.call("rm -rf ./Info/*RebalanceDates.csv", shell=True)

#subprocess.check_call("python ./predictions/get_dates.py SCHEDULER=%s" % sys.argv[2], shell=True)
#subprocess.check_call("python ./predictions/run_backtest_forecasts.py SCHEDULER=%s ASSETTYPE=%s FILE=%s RETURNS=%s VOLATILITY=%s CORRELATION=%s" % (sys.argv[2], sys.argv[1], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]), shell=True)

ret_method =  sys.argv[4]
vol_method =  sys.argv[5]
cor_method =  sys.argv[6]

assets = list(np.unique(np.loadtxt(sys.argv[3], dtype = str, delimiter = '/n')))

##Update returns
#if 'SAMPLE' in ret_method:
#	from insertDB.update_ret import update_samplereturns_db
#	update_samplereturns_db(assets, [ret_method])
#
##Update volatility
#if 'SAMPLE' in vol_method:
#	from insertDB.update_vol import update_samplevolatilities_db
#	update_samplevolatilities_db(assets, [vol_method])
#
#if 'GARCH' in vol_method:
#	from insertDB.update_all_garch import update_garch_volatilities_db
#	update_garch_volatilities_db(assets, [vol_method])
#
#
#
##Update correlations
#if 'SAMPLE' in cor_method:
#	from insertDB.update_corr import update_samplecorrelations_db
#	update_samplecorrelations_db(assets, [cor_method])
#
#if 'TAIL' in cor_method:
#	from insertDB.update_low_tail_dep import update_low_tail_dependence_db
#	update_low_tail_dependence_db(assets, [cor_method])
#
#if 'GARCH' in cor_method:
#	from insertDB.update_dcc_all_garch import update_dcc_garch_db
#	update_dcc_garch_db(assets, [cor_method])
#

#Update lower tail dependence



subprocess.check_call("python ./backtests/parameters_generator.py SCHEDULER=%s ASSETTYPE=%s FILE=%s RETURNS=%s VOLATILITY=%s CORRELATION=%s" % (sys.argv[2], sys.argv[1], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]), shell=True)


#TODO predictions= tem que mudar
#TODO tem que receber optimization
subprocess.check_call("python ./backtests/run_parameters.py SCHEDULER=%s ASSETTYPE=%s OPTIMIZATION=%s" % (sys.argv[2], sys.argv[1], sys.argv[7]), shell=True)
subprocess.check_call("python ./utilities/generate_reports.py SCHEDULER=%s ASSETTYPE=%s" % (sys.argv[2], sys.argv[1]), shell=True)


#subprocess.check_call("python ./weights_portfolio/recomendations.py SCHEDULER=%s ASSETTYPE=%s FILE=%s PREDICTION=%s OPTIMIZATION=%s" % (sys.argv[2], sys.argv[1], sys.argv[3], sys.argv[4], sys.argv[5]), shell=True)
