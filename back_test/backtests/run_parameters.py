import os
import json
import glob

import multiprocessing
from joblib import Parallel, delayed

import time

import numpy as np
import pandas as pd
import sys
def mask_missing(arr, values_to_mask):
	"""
	Return a masking array of same size/shape as arr
	with entries equaling any member of values_to_mask set to True
	"""
	if not isinstance(values_to_mask, (list, np.ndarray)):
		values_to_mask = [values_to_mask]
	try:
		values_to_mask = np.array(values_to_mask, dtype=arr.dtype)
	except Exception:
		values_to_mask = np.array(values_to_mask, dtype=object)
	na_mask = pd.isnull(values_to_mask)
	nonna = values_to_mask[~na_mask]
	mask = None
	for x in nonna:
		if mask is None:
			mask = arr == x

			# if x is a string and arr is not, then we get False and we must
			# expand the mask to size arr.shape
			if np.isscalar(mask):
				mask = np.zeros(arr.shape, dtype=bool)
		else:
			mask |= arr == x
	if na_mask.any():
		if mask is None:
			mask = pd.isnull(arr)
		else:
			mask |= pd.isnull(arr)
	return mask
setattr(pd.core.common, 'mask_missing', mask_missing)
import backtest

# ==================================
# Auxiliary Functions
def fetch_backtest_results(pathdir, filename = '*'):
		return [elem.split('.')[0].split('/')[-1] for elem in glob.glob(pathdir + filename + '.csv')]


def fetch_backtest_params(pathdir, filename = '*'):
	return [elem.split('.')[0].split('/')[-1] for elem in glob.glob(pathdir + filename + '.json')]

def read_json(pathfilename):
	try:
		with open(pathfilename) as json_file:
			json_data = json.load(json_file)
		return json_data
	except:
		print "Error loading %s" % pathfilename
# ==================================

def func(param, *args, **kw):
    """
    Wrap of the function to be parallelized - backtest.
    """
    # try:
        # start_time = time.time()
    print param
    backtest.run_backtest(param, *args, **kw)
        # with open("Output_%s.txt" % param['general_parameters']['SCHEDULER'], "a") as text_file:
            # text_file.write("%s:%f\n" % (param['general_parameters']['ID'], time.time() - start_time))
    # except: 
    #     print "Error in %s" % param['general_parameters']['ID']
    #     with open("Output_Errors_%s.txt" % param['general_parameters']['SCHEDULER'], "a") as text_file:
    #         text_file.write("Error in %s:%f\n" % (param['general_parameters']['ID'], time.time() - start_time))


if __name__ == '__main__':
    freqs = ['Monthly']
    if len(sys.argv) >= 2 and 'SCHEDULER=' in sys.argv[1]:
        freqs = sys.argv[1].split('=')[-1].split(',')

    PARALLEL = True #TODO sould be input variables
    N_FREE_CORES = 10
    N_JOBS = 8 # max(multiprocessing.cpu_count()-N_FREE_CORES, 1)
    if len(sys.argv) >= 5 and 'JOBS=' in sys.argv[5]:
        N_JOBS = int(sys.argv[4].split('=')[-1])

    optimization_methods = ['all'] 
    if len(sys.argv) >= 4 and 'OPTIMIZATION=' in sys.argv[3]:
        optimization_methods = sys.argv[3].split('=')[-1]
        if ';' in optimization_methods:
            optimization_methods = optimization_methods.split(';')
        else:
            optimization_methods = [optimization_methods]

    asset_type = "Top"
    if len(sys.argv) >= 3 and 'ASSETTYPE=' in sys.argv[2]:
        asset_type = sys.argv[2].split('=')[-1]


    for freq in freqs:
        paramsdir = 'PARAMETERS_' + freq.upper()
        pathdir1 = os.getcwd()
        pathdir2 = pathdir1 + '/synology/' + asset_type

        if not os.path.exists(os.path.dirname(pathdir2 + '/RESULTS_' + freq.upper() + '/')):
            os.makedirs(os.path.dirname(pathdir2 + '/RESULTS_' + freq.upper() + '/')) 
		
        params_ids = fetch_backtest_params(pathdir = pathdir1 + '/' + paramsdir +'/')
        
        if len(optimization_methods) > 0 and optimization_methods[0] != 'all':
            params_ids = optimization_methods
        else:
            params_ids = list(set(params_ids))
        
        master_params = [read_json(pathdir1 + '/' + paramsdir + '/' + id_ + '.json') for id_ in params_ids]
       

        
        print  "Going to perform backtests... "

        # master_params = [{u'general_parameters': {u'REFERENCE_GROUP_SYMBOL': {u'Commodities': u'CRY INDEX', u'Bonds': u'PERF-BONDS', u'Alternatives': u'PERF-ALTERNATIVES', u'Cash': u'GT2 GOVT', u'Equity': u'MXWD INDEX'}, u'VOL_READER': u'/home/rjaulino/backtest_variables_portfolio/dev_tests/Top/PREDICTIONS_TOP/VOLATILITY_SGARCH22', u'COMMISSION_MIN_TRADE_COST': 1.0, u'SYMBOL_GROUPS': None, u'REBALANCE_ACTION_TYPE': u'close', u'START_DATE_BT_STR': u'19900101', u'RETS_READER': u'/home/rjaulino/backtest_variables_portfolio/dev_tests/Top/PREDICTIONS_TOP/RETURNS_SAMPLE1', u'REBALANCE_DATES_READER': u'/home/rjaulino/backtest_variables_portfolio/dev_tests/INFO/', u'SLIPPAGE_SPREAD': 0.0, u'END_DATE_BT_STR': None, u'REF_CURNCY': None, u'ASSET_READER': u'/home/rjaulino/backtest_variables_portfolio/dev_tests/INFO/assets_top.txt', u'CORR_READER': u'/home/rjaulino/backtest_variables_portfolio/dev_tests/Top/PREDICTIONS_TOP/CORRELATION_SGARCH22', u'SCHEDULER': u'Weekly', u'COMMISSION_COST_PER_SHARE': 0.1, u'ID': u'e9f5c8ba-eb69-4384-9b0e-4a354ad4f166'}, u'top_params': {u'framework_params': {u'model': u'None'}, u'filter_params': {u'model': u'None'}, u'pf_params': {u'max_sum_pos': 1.0, u'min_sum_neg': -1.0, u'max_sum': 1.0, u'limit_k': None, u'bounds': [-1.0, 1.0], u'mvo_delta': 256.0, u'model': u'MVO', u'mvo_util_func': u'quadratic'}}}]
        if PARALLEL:
            #, pre_dispatch = '5*n_jobs'
            Parallel(n_jobs=N_JOBS, verbose=50)(delayed(func)(param, asset_type = asset_type, filename = pathdir2 + '/RESULTS_' + freq.upper() + '/' + param['general_parameters']['ID'] + '.csv', verbose = False) for param in master_params)
        else:
            for id_, param in enumerate(master_params):
                print "================================== BACKTEST %d : %s ==================================" % (id_, param['general_parameters']['ID'])
                print param
                print 
                backtest_start_time = time.time()
                func(param, filename = pathdir2 + '/RESULTS_' + freq.upper() +'/' + param['general_parameters']['ID'] + '.csv', verbose = True, asset_type = asset_type)
                print "BACKTEST TOOK: %fs." % (time.time() - backtest_start_time)