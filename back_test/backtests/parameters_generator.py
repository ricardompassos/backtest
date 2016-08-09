import os
import itertools
import numpy as np
import copy
import uuid
import json
import sys
# =================================
# Auxiliary functions
def powerset(lst):
    return reduce(lambda result, x: result + [subset + [x] for subset in result], lst, [[]])
# =================================

def param_generator(SCHEDULERS = ['Monthly'], group = 'Top', ASSET_READER = './Info/assets.txt', ret_pred_method = 'RETURNS_SAMPLE_1D', vol_pred_method = 'VOLATILITY_SAMPLE_22D', cor_pred_method = 'CORRELATION_SAMPLE_22D'):
	SANITY_CHECK = False
	WRITE_PARAMS = True
	# NUMBER_OF_PARAMS_TO_WRITE = 1000000 # NEW, FOR TEST

	working_dir = os.getcwd() + '/'

	

	for SCHEDULER in SCHEDULERS:

		REBALANCE_DATES_READER = working_dir + 'Info/' # + SCHEDULER + 'RebalanceDates.csv'

		#SYMBOL_GROUPS = ['Equity', 'Commodities', 'Bonds'] # Inner groups to consider
		REFERENCE_GROUP_SYMBOL = {
	                                 'Equity': 'MXWD INDEX',
	                                 'Bonds':'PERF-BONDS',
	                                 'Commodities': 'CRY INDEX',
	                                 'Alternatives': 'PERF-ALTERNATIVES',
	                                 'Cash':'GT2 GOVT'
	    						}

		# =================================

		possible_returns_sources = [ret_pred_method]

		possible_cov_sources = [(vol_pred_method, cor_pred_method)] 

		fa_default_list = ['CUR_RATIO', 'NORM_NET_INC_TO_NET_INC_FO_COM', 'TOT_DEBT_TO_EBITDA', 'ASSET_TURNOVER', 'CFO_TO_AVG_CURRENT_LIABILITIES', 'FNCL_LVRG', 'ACCT_RCV_TURN', 'CAP_EXPEND_RATIO']
		buffet_fa_filter = {'indicators': ['CUR_RATIO', 'PX_TO_BOOK_RATIO', 'CUR_MKT_CAP', 'PX_TO_CASH_FLOW'], 'operators':['>=', '<=', '>=', '<='], 'thresholds': [1, 0.75, 100.0, 15.0], 'order_by': 'PE_RATIO', 'ascending_order': False}

		list_asset_types = ['Equity', 'Commodities', 'Bonds', 'Top', 'Mixed', 'Alternatives']

		possible_pf_params = [
								({'pf_params': {'model': 'EquiWeight', 'need_covar': False, 'need_returns': False}}, list_asset_types),
								
								({'pf_params': {'model': 'MVO', 'need_covar': True, 'need_returns': True, 'bounds':[0.0,1.0], 'max_sum': 1.0, 'min_sum_neg':0.0, 'max_sum_pos':1.0, 'mvo_util_func': 'quadratic', 'mvo_delta':2.0, 'limit_k': None}}, list_asset_types),
								({'pf_params': {'model': 'MVO', 'need_covar': True, 'need_returns': True, 'bounds':[0.0,1.0], 'max_sum': 1.0, 'min_sum_neg':0.0, 'max_sum_pos':1.0, 'mvo_util_func': 'quadratic', 'mvo_delta':8.0, 'limit_k': None}}, list_asset_types),
								({'pf_params': {'model': 'MVO', 'need_covar': True, 'need_returns': True, 'bounds':[0.0,1.0], 'max_sum': 1.0, 'min_sum_neg':0.0, 'max_sum_pos':1.0, 'mvo_util_func': 'quadratic', 'mvo_delta':32.0, 'limit_k': None}}, list_asset_types),
								({'pf_params': {'model': 'MVO', 'need_covar': True, 'need_returns': True, 'bounds':[0.0,1.0], 'max_sum': 1.0, 'min_sum_neg':0.0, 'max_sum_pos':1.0, 'mvo_util_func': 'quadratic', 'mvo_delta':128.0, 'limit_k': None}}, list_asset_types),
								({'pf_params': {'model': 'MVO', 'need_covar': True, 'need_returns': True, 'bounds':[0.0,1.0], 'max_sum': 1.0, 'min_sum_neg':0.0, 'max_sum_pos':1.0, 'mvo_util_func': 'quadratic', 'mvo_delta':256.0, 'limit_k': None}}, list_asset_types),
								#({'pf_params': {'model': 'MVO', 'need_covar': True, 'need_returns': True, 'bounds':[0.0,1.0], 'max_sum': 1.0, 'min_sum_neg':0.0, 'max_sum_pos':1.0, 'mvo_util_func': 'sharpe', 'limit_k': None}}, list_asset_types),
								({'pf_params': {'model': 'MVO', 'need_covar': True, 'need_returns': True, 'bounds':[-1.0,1.0], 'max_sum': 1.0, 'min_sum_neg':-1.0, 'max_sum_pos':1.0, 'mvo_util_func': 'quadratic', 'mvo_delta':2.0, 'limit_k': None}}, list_asset_types),
								({'pf_params': {'model': 'MVO', 'need_covar': True, 'need_returns': True, 'bounds':[-1.0,1.0], 'max_sum': 1.0, 'min_sum_neg':-1.0, 'max_sum_pos':1.0, 'mvo_util_func': 'quadratic', 'mvo_delta':8.0, 'limit_k': None}}, list_asset_types),
								({'pf_params': {'model': 'MVO', 'need_covar': True, 'need_returns': True, 'bounds':[-1.0,1.0], 'max_sum': 1.0, 'min_sum_neg':-1.0, 'max_sum_pos':1.0, 'mvo_util_func': 'quadratic', 'mvo_delta':32.0, 'limit_k': None}}, list_asset_types),
								({'pf_params': {'model': 'MVO', 'need_covar': True, 'need_returns': True, 'bounds':[-1.0,1.0], 'max_sum': 1.0, 'min_sum_neg':-1.0, 'max_sum_pos':1.0, 'mvo_util_func': 'quadratic', 'mvo_delta':128.0, 'limit_k': None}}, list_asset_types),
								({'pf_params': {'model': 'MVO', 'need_covar': True, 'need_returns': True, 'bounds':[-1.0,1.0], 'max_sum': 1.0, 'min_sum_neg':-1.0, 'max_sum_pos':1.0, 'mvo_util_func': 'quadratic', 'mvo_delta':256.0, 'limit_k': None}}, list_asset_types),


								({'pf_params': {'model': 'MDP', 'need_covar': True, 'need_returns': False}}, list_asset_types),
								({'pf_params': {'model': 'ERC', 'need_covar': True, 'need_returns': False}}, list_asset_types)
							]


		possible_framework_params = [
									({'framework_params':{'model': 'None', 'need_covar': False, 'need_returns': False}}, list_asset_types)
									]

		possible_filter_params =  [
									({'filter_params':{'model': 'None'}}, list_asset_types)
									]

		#possible_symbol_groups = [SYMBOL_GROUPS]#powerset(SYMBOL_GROUPS)[1:]

		# =================================

		REBALANCE_ACTION_TYPE = 'close'
		START_DATE_BT_STR = '19900101' # '20100101'
		END_DATE_BT_STR = None

		REF_CURNCY = None # 'USD'
		SLIPPAGE_SPREAD = 0.0
		COMMISSION_COST_PER_SHARE = 0.0075
		COMMISSION_MIN_TRADE_COST = 1.0

		general_parameters = {
								'ID': None,
								'ASSET_READER': ASSET_READER,
								'REBALANCE_DATES_READER': REBALANCE_DATES_READER,
								'CORR_READER': None,
								'VOL_READER': None,
								'RETS_READER': None,
								'SYMBOL_GROUPS': None,
								'REFERENCE_GROUP_SYMBOL': REFERENCE_GROUP_SYMBOL,
								'SCHEDULER': SCHEDULER,
								'REBALANCE_ACTION_TYPE':REBALANCE_ACTION_TYPE,
								'START_DATE_BT_STR':START_DATE_BT_STR,
								'END_DATE_BT_STR':END_DATE_BT_STR,
								'REF_CURNCY': REF_CURNCY,
								'SLIPPAGE_SPREAD': SLIPPAGE_SPREAD,
								'COMMISSION_COST_PER_SHARE': COMMISSION_COST_PER_SHARE,
								'COMMISSION_MIN_TRADE_COST': COMMISSION_MIN_TRADE_COST
							} # The None entries need to be set afterwards




		# Get all combinations for Equity or Top

		# Equity or Top
		params = {} #{} if Top or {'group': group} if equty 
		if group != "Top":
			params = {'group': group}
		params_confs = []
		for framework in possible_framework_params:
			if group in framework[1]:
				params.update(framework[0])
			else:
				continue
			for pf in possible_pf_params:
				if group in pf[1]:
					params.update(pf[0])
				else:
					continue
				for filter_ in possible_filter_params:
					if group in filter_[1]:
						params.update(filter_[0])
					else:
						continue
					params_confs.append(copy.deepcopy(params))
					params.pop('filter_params')
				params.pop('pf_params')
			params.pop('framework_params')


		master_params = []

		parameters = {'general_parameters': general_parameters}
		for conf in params_confs:
			need_covar = conf['pf_params']['need_covar'] | conf['framework_params']['need_covar']           
			need_returns = conf['pf_params']['need_returns'] | conf['framework_params']['need_returns']
			cov_srcs = possible_cov_sources if need_covar else [(None, None)]
			returns_srcs = possible_returns_sources if need_returns else [None]
			for rets_reader in returns_srcs:
				parameters['general_parameters']['RETS_READER'] = rets_reader
				for vol_reader, corr_reader in cov_srcs:
					parameters['general_parameters']['VOL_READER'], parameters['general_parameters']['CORR_READER'] = vol_reader, corr_reader
					params_local = copy.deepcopy(parameters)
					params_local['general_parameters']['ID'] = conf['pf_params']['model'] #str(uuid.uuid4())
					if conf['pf_params']['model'] == 'MVO':
						if conf['pf_params']['bounds'][0] == -1:
							params_local['general_parameters']['ID'] += ',short'
						else:
							params_local['general_parameters']['ID'] += ',long'
						params_local['general_parameters']['ID'] += ',%s' % int(conf['pf_params']['mvo_delta'])
					params_local.update({'params':conf})
					master_params.append(params_local)
					print params_local
					print "=========================="
	                   
		if SANITY_CHECK:
			print "Sanity Check:"
			for i in xrange(len(master_params)):
				for j in xrange(i+1, len(master_params)):
					if cmp(master_params[i], master_params[j]) == 0:
						raise ValueError('There are two equal dictionaries on indexes %d, %d'%(i,j))
			print "Dicts are OK"

		print "There were %d combinations generated"%len(master_params)

		if WRITE_PARAMS:
			print "Saving Parameters..."
			params_dir = os.getcwd() + '/PARAMETERS_' + SCHEDULER.upper() + '/'
			if not os.path.exists(os.path.dirname(params_dir)):
				os.makedirs(os.path.dirname(params_dir))
			# for params in master_params[:NUMBER_OF_PARAMS_TO_WRITE]:
			for params in master_params:
				with open(params_dir + params['general_parameters']['ID'] + '.json', 'w') as fp:
					json.dump(params, fp)

if __name__ == '__main__':
	SCHEDULERS = ['Monthly']
	if len(sys.argv) >= 2 and 'SCHEDULER=' in sys.argv[1]:
		SCHEDULERS = sys.argv[1].split('=')[-1].split(',')

	group = "Top"
	if len(sys.argv) >= 3 and 'ASSETTYPE=' in sys.argv[2]:
		group = sys.argv[2].split('=')[-1]

	ASSET_READER = './Info/assets.txt'
	if len(sys.argv) >= 4 and 'FILE=' in sys.argv[3]:
		ASSET_READER = sys.argv[3].split('=')[-1]

	ret_pred_method = 'RETURNS_SAMPLE_1D'
	if len(sys.argv) >= 5 and 'RETURNS=' in sys.argv[4]:
		ret_pred_method = sys.argv[4].split('=')[-1]

	vol_pred_method = 'VOLATILITY_SAMPLE_22D'
	if len(sys.argv) >= 6 and 'VOLATILITY=' in sys.argv[5]:
		vol_pred_method = sys.argv[5].split('=')[-1]

	cor_pred_method = 'CORRELATION_SAMPLE_22D'
	if len(sys.argv) >= 7 and 'CORRELATION=' in sys.argv[6]:
		cor_pred_method = sys.argv[6].split('=')[-1]

	param_generator(SCHEDULERS = SCHEDULERS, group = group, ASSET_READER = ASSET_READER, ret_pred_method = ret_pred_method, vol_pred_method = vol_pred_method, cor_pred_method = cor_pred_method)
