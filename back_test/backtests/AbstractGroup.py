from __future__ import division

import sys
import os
import pandas as pd
import numpy as np
from scipy import linalg

# Custom Modules
from nearest_corr import nearcorr
import portfolio_models as pf_m
import utils

# Fundamental Analysis module
import FA3 as fa

import time

class AbstractGroup(object):
    """
    """
    #POSSIBLE_FRAMEWORKS = ('None', 'Black-Litterman')
    POSSIBLE_BL_VIEWS = ('FA View', 'Dummy View', 'Returns View')
    POSSIBLE_PF_MODELS = ('Unconstrained MVO','MVO', 'MDP', 'ERC', 'EquiWeight', 'FAWeight')
    NEED_RETURNS = ('BlackLitterman','Unconstrained MVO', 'MVO')
    NEED_COV = ('BlackLitterman','Unconstrained MVO', 'MVO', 'MDP', 'ERC')

    def __init__(self, symbols, name = '', **kw):
        symbols.sort()
        self.name = name
        self.ref_symbol = kw.pop('ref_symbol', None)
        if self.ref_symbol is not None: # remove the reference symbol from the list

            self.symbols = np.array(symbols)[np.where(np.array(symbols) != self.ref_symbol)[0]].tolist()
        else:
            self.symbols = symbols

        self._set_attribs(**kw)
        self.updated_symbols = self.symbols
        self.R = None
        self.C = None
        self.check_model_prerequisites()
        start = kw['start'] if 'start' in kw.items() else None
        end = kw['end'] if 'end' in kw.items() else None
        verbose = kw['verbose'] if 'verbose' in kw.items() else False
        self.load_filters_data(start=start, end=end, verbose = verbose)
        self.load_framework_data(start=start, end=end, verbose = verbose)


    def _set_attribs(self, **kwargs):
        for key, value in kwargs.iteritems():
            if isinstance(value, pd.Series) or isinstance(value, pd.DataFrame):
                if key != 'correlations':
                    value = value[self.symbols]
                else:
                    value = value[utils.translate_symbols_corr(self.symbols)]

            setattr(self, key, value)

    def load_filters_data(self, start = None, end = None, verbose = False):

        if verbose: 
            time_counter = time.time()
            print 'In load_filters_data of %s' % self.name

        self.filtered_symbol_dates, self.filtered_symbols = [None]*2

        if self.filter_params['model'] == 'FA':
            self.filtered_symbol_dates, self.filtered_symbols = zip(*fa.filter_fa(assets_names=self.symbols, 
                                                                                   list_indicators_filters=self.filter_params['indicators'], 
                                                                                   list_operators=self.filter_params['operators'],
                                                                                   list_thresholds=self.filter_params['thresholds'],
                                                                                   indicator_orderby=self.filter_params['order_by'],
                                                                                   top_assets=self.filter_params['top_assets'], 
                                                                                   order_ascending=self.filter_params['ascending_order'], 
                                                                                   start = start, end = end, verbose = False))
            self.filtered_symbol_dates = np.array([date__.date() for date__ in self.filtered_symbol_dates])
            if verbose:
                print 'Filtered symbol dates: ', self.filtered_symbol_dates
                print 'Filtered symbol: ', self.filtered_symbols
                print 'Time to load filters [s]: ', time.time() - time_counter 
        if verbose: print 'Out of load_filters_data of %s' % self.name
        return self

    def load_framework_data(self, start = None, end = None, verbose = False):
        if verbose:
            print 'In load_framework_data of %s' % self.name
            start_time_load = time.time()
        if self.framework_params['model'] == 'BlackLitterman':

            self.framework_data = {'market cap': utils.load_data_from_passdb(self.updated_symbols, source = 'PASS', timezone_indication = 'UTC', start = start, end = end,
                                                instruments = ['market cap'], set_price = 'close', align_dates = False, set_volume_val = 1e6, transform_to_weights = True, 
                                                add_dummy_volume = False, output_asset_info = False, convert_curr = None).minor_xs('market cap') 
                                    }
            self.framework_data['market cap'].ffill(inplace = True)
            self.framework_data['market cap'] = self.framework_data['market cap'].apply(lambda col, div: col/div, args = (self.framework_data['market cap'].sum(axis=1, skipna = True).values,))
            self.framework_data['market cap'].index = [time_idx.date() for time_idx in self.framework_data['market cap'].index.tolist()]
            if verbose:
                print 'Time to load framework data of %s is %f'%(self.name, time.time()-start_time_load)

        if verbose: print 'Out of load_framework_data of %s' % self.name

    def fetch_predictions(self, verbose = False):
        """
        Get the predictions for the current date
        """

        if verbose:
            print 'In fetch_predictions of %s' % self.name
            start_time = time.time()
        if len(self.updated_symbols) < 2: return self

        if self.need_returns:
            try:
                start_time_ret = time.time()
                series_R = self.returns[self.updated_symbols].loc[self.curr_date].dropna()
                # if verbose: print "Reading returns in fetch_predictions at %s:\n%s" %(self.curr_date, series_R.to_string())
                self.updated_symbols = series_R.index.values.tolist()
                if len(self.updated_symbols) < 2: return self
                self.R = series_R.values
                if verbose: print 'Time to fetch returns of %s is %f'%(self.name, time.time()-start_time_ret)

            except KeyError:
                print "There are no return predictions at %s" % self.curr_date
                self.updated_symbols = []
                return self

        if self.need_cov:
            try:
                start_time_vol = time.time()
                series_V = self.volatility[self.updated_symbols].loc[self.curr_date].dropna()
                if verbose: print 'Time to fetch volatility of %s is %f'%(self.name, time.time()-start_time_vol)
                self.updated_symbols = series_V.index.values.tolist()
                # if verbose: print "Reading volatility in fetch_predictions at %s:\n%s" %(self.curr_date, series_V.to_string())
                if len(self.updated_symbols) < 2: 
                    self.updated_symbols = []                    
                    return self

                start_time_corr = time.time()
                print len(self.updated_symbols)
                symbols_corr = utils.translate_symbols_corr(self.updated_symbols)

                dfList = pd.DataFrame(columns=self.updated_symbols)
                dfList.to_csv('./updated_sybols.csv', sep=';')

                series_Corr = self.correlations[symbols_corr].loc[self.curr_date].dropna()
                
                #series_Corr = self.correlations[symbols_corr].loc[self.curr_date]
                #series_Corr.to_csv('./serie_corr_before.csv', sep=';')
                #series_Corr = series_Corr.dropna()
                #series_Corr.to_csv('./serie_corr_after.csv', sep=';')

                #print fghjklgfh
                #print len(symbols_corr)
                #series_Corr = self.correlations[symbols_corr] #.loc[self.curr_date].dropna()
                #print len(series_Corr)
                #series_Corr = series_Corr.loc[self.curr_date]
                #print len(series_Corr)
                #series_Corr = series_Corr.dropna()
                #print len(series_Corr)
                #print asdfaf

                if verbose: print 'Time to fetch correlations of %s is %f'%(self.name, time.time()-start_time_corr)
                # if verbose: print "Reading correlations in fetch_predictions at %s:\n%s" %(self.curr_date, series_Corr.to_string())
                self.updated_symbols = list(set([symbol for supersymbol in series_Corr.index.values.tolist() for symbol in supersymbol.split(' _ ')]))
                self.updated_symbols.sort()

                if len(self.updated_symbols) < 2: 
                    self.updated_symbols = []                    
                    return self
                series_V = series_V[self.updated_symbols]
                if self.need_returns:
                    series_R = series_R[self.updated_symbols]
                    self.R = series_R.values
				
                #print "series_Corr:\n", series_Corr, '\n'
                V = series_V.values
                Corr = np.eye(len(series_V))
                
                #print 'BEFORE:'
                #print len(np.triu_indices_from(Corr, k = 1)[0])
                #print len(series_Corr.values)
                if len(np.triu_indices_from(Corr, k = 1)[0]) != len(series_Corr.values): # try to fix, this does not fix for every case though; only when there are symbols that are missing some entries                   
                    n_counts = len(self.updated_symbols) - 1 
                    count_arr = np.array([len(np.where(np.char.find(np.array(series_Corr.index, dtype = str), symbol_) > -1)[0]) for symbol_ in self.updated_symbols])
                    self.updated_symbols = np.array(self.updated_symbols)[np.where(count_arr == n_counts)[0]].tolist()
                    if self.updated_symbols < 2:
                        raise ValueError('Not enough values for building correlation matrix.')
                    V = series_V[self.updated_symbols].values
                    Corr = np.eye(len(V))
                    #print 'AFTER:'
                    #print len(np.triu_indices_from(Corr, k = 1)[0])
                    #print len(series_Corr.values)

                #print '3'
                #print series_Corr.values
                #print Corr
                #print np.triu_indices_from(Corr, k = 1)

                try:
                    Corr[np.triu_indices_from(Corr, k = 1)] = series_Corr.values
                except Exception as e:
                    print "AAAAAAAAAAAA"
                    print e
#


                Corr[np.triu_indices_from(Corr, k = 1)] = series_Corr.values
                #print '4'
                Corr = Corr + Corr.T
                #print '5'
                np.fill_diagonal(Corr, 1.0)
                #print '6'
                start_time_psd = time.time()
                # if self.name == 'Equity':
                #     series_V.to_csv('series_V.csv')
                #     np.save('Corr', Corr)
                #     np.save('V', V)
                Corr = self._verify_psd(Corr)
                #print '7'
                if verbose: print 'Time to verify PSD of %s is %f'%(self.name, time.time()-start_time_psd)
                vols_mat = V * np.eye(V.size)
                #print '8'
                self.C = np.dot(np.dot(vols_mat, Corr), vols_mat)
                #print '9'
            except KeyError:
                print "There are no covariance predictions at %s" % self.curr_date
                self.updated_symbols = []
                return self
            except ValueError:
                print "Not enough values for building correlation matrix at %s" % self.curr_date
                self.updated_symbols = []
                return self
        if verbose:
            print 'Elapsed time for fetch_predictions is %f'%(time.time() - start_time) 
            print 'Out of fetch_predictions of %s' % self.name
        return self

    def update(self, curr_date, active_symbols = None, round = None, verbose = False):
        # Reset state
        self.updated_symbols = self.symbols
        self.weights, self.R, self.C = [None]*3

        self.curr_date = curr_date
        if active_symbols is not None:
            self.updated_symbols = np.intersect1d(self.updated_symbols, active_symbols).tolist()
        self.apply_filters_batch(verbose = verbose).fetch_predictions(verbose = verbose).apply_framework(verbose = verbose).portfolio_weights(round = round, verbose = verbose)



    def apply_filters_batch(self, verbose = False):
        if len(self.updated_symbols) == 0: return self
        if self.filter_params['model'] == 'FA':
            start_time = time.time()
            if verbose:
                print 'In apply_filters of %s' % self.name
                print 'Parameters: ', self.filter_params
                print 'Symbols before filters: ', self.updated_symbols

            index_fa = np.where(self.filtered_symbol_dates <= self.curr_date)[0]
            if len(index_fa) != 0:
                self.updated_symbols = np.intersect1d(self.updated_symbols, self.filtered_symbols[index_fa[-1]])
                self.updated_symbols.sort()
            if verbose:
                print 'Symbols after filters: ', self.updated_symbols  
                print 'Elapsed time for apply_filters is %f'%(time.time() - start_time)  
                print 'Out of apply_filters of %s' % self.name
        return self




    def apply_filters_online(self, verbose = False):
        if len(self.updated_symbols) == 0: return self
        if self.filter_params['model'] == 'FA':
            if verbose:
                print 'In apply_filters of %s' % self.name
                print 'Parameters: ', self.filter_params
                print 'Symbols before filters: ', self.updated_symbols
            
            self.updated_symbols = fa.filter_fa(self.updated_symbols, 
                                            list_indicators_filters = self.filter_params['indicators'], 
                                            list_operators = self.filter_params['operators'],
                                            list_thresholds = self.filter_params['thresholds'], 
                                            indicator_orderby = self.filter_params['order_by'], 
                                            top_assets = self.filter_params['top_assets'],
                                            order_ascending = self.filter_params['ascending_order'], 
                                            query_date = str(self.curr_date), start = None, end = None, verbose = False)[0][1].tolist()
            self.updated_symbols.sort()
            if verbose:
                print 'Symbols after filters: ', self.updated_symbols    
                print 'Out of apply_filters of %s' % self.name
        return self

    def apply_framework(self, verbose = False):


        if verbose: 
            'In apply_framework of %s' % self.name
            start_time = time.time()
        if self.framework_params['model'] != 'BlackLitterman':
            return self
        else:
            if verbose: print 'Framework Parameters: ', self.framework_params
            if len(self.updated_symbols) == 0:
                return self
            self.bl_P, self.bl_Q = [None]*2

            start_time_mkt_cap = time.time()

            self.weq = self.framework_data['market cap'][self.updated_symbols].loc[self.curr_date]

            # Load online the market cap - not in use
            #self.weq = utils.load_market_cap(self.updated_symbols, query_date = self.curr_date, source =  "PASS / MARKET DATA", transform_to_weights = True, fix_symbols = False)
            
            symbols_before = self.weq.index.tolist()
            self.weq = self.weq.dropna()
            self.weq /= self.weq.sum()
            symbols_after = self.weq.index.tolist()
            # because we need the index not just the name - R and C are arrays...
            weq_ids = [id_ for id_, symbol_ in enumerate(symbols_before) if symbol_ in symbols_after]

            self.updated_symbols = symbols_after
 
            if len(self.updated_symbols)<2:return self

            if self.R is not None:
                self.R = self.R[weq_ids]
            if self.C is not None:
                self.C = self.C[:,weq_ids][weq_ids]

            if self.framework_params['bl_view'] == 'Dummy View':
                self.bl_P = np.zeros((1,len(self.updated_symbols)))  # P must be a 2D array
                self.bl_P[0][0] = 0.5; self.bl_P[0][-1] = 0.5
                self.bl_Q = np.array([1e-6])
            elif self.framework_params['bl_view'] == 'FA View':
                if len(self.framework_params['portfolio_fa_indicators']) == 0:
                    return self
                self.bl_P = fa.portfolio_fa(assets_names = self.updated_symbols, indicators_names = self.framework_params['portfolio_fa_indicators'], query_date = self.curr_date).values
                if self.R is None:
                    return self
                self.bl_Q = np.array([np.sum(self.bl_P*self.R)])
                self.bl_P = self.bl_P.reshape((1,self.bl_P.size))
            elif self.framework_params['bl_view'] == 'Returns View':
                self.bl_P = np.eye(len(self.updated_symbols))
                self.bl_Q = self.R

            else:
                raise ValueError('%s is not a supported Black-Litterman View!\nThe supported views are: %s' % (self.framework_params['bl_view'], str(AbstractGroup.POSSIBLE_BL_VIEWS)))
            if verbose:
                print 'weq:\n', self.weq
                print 'bl_P:\n', self.bl_P
                print 'bl_Q: ', self.bl_Q
            self.R, self.C = pf_m.BL(self.weq.values.ravel(), self.C, self.bl_P, self.bl_Q, Omega = None, tau=self.framework_params['bl_tau'], delta=self.framework_params['bl_delta'], compute_unconstrained_weights = False, compute_stats = False)
            if self.R.ndim == 1:
                raise ValueError('self.R is a 1D array!!!')
            self.R = self.R.T[0]
            if verbose:
                print 'R: ', self.R
                print 'C:\n', self.C
        if verbose:
            print 'Elapsed time for apply_framework is %f'%(time.time() - start_time)
            print 'Out of apply_framework of %s' % self.name
        return self


    def portfolio_weights(self, round = None, verbose = False):
        #if self.name == 'Equity':
            #np.save('C', self.C)
            #np.save('R', self.R)
        if verbose:
            start_time = time.time()
            print 'In portfolio_weights of %s' % self.name
            print 'Entry symbols: ', self.updated_symbols
            print 'Portfolio Parameters: ', self.pf_params
        weights = None
        if len(self.updated_symbols) == 0: return self
        try:
            if self.pf_params['model'] == 'Unconstrained MVO' and self.R is not None and self.C is not None:
                weights = self.R.T.dot(linalg.inv(self.pf_params['mvo_delta'] * self.C)).T
                weights = weights.reshape(weights.size,)
            elif self.pf_params['model'] == 'MVO' and self.R is not None and self.C is not None:
                weights =  pf_m.MVO(self.R, self.C, bounds = self.pf_params['bounds'], rf=0.0, max_sum_weights = self.pf_params['max_sum'],
                                min_sum_negative_weights = self.pf_params['min_sum_neg'], max_sum_positive_weights = self.pf_params['max_sum_pos'],
                                utility_function = self.pf_params['mvo_util_func'], delta = self.pf_params['mvo_delta'] if 'mvo_delta' in self.pf_params.keys() else 2.0 , limit_k = self.pf_params['limit_k'])[0]
            elif self.pf_params['model'] == 'MDP' and self.C is not None:
                weights = pf_m.MDP(self.C)[0]
            elif self.pf_params['model'] == 'ERC' and self.C is not None:
                weights = pf_m.ERC(self.C)[0]
            elif self.pf_params['model'] == 'EquiWeight':
                n = len(self.updated_symbols)
                weights = np.ones(n)/float(n)

            elif self.pf_params['model'] == 'FAWeight':
                weights = fa.portfolio_fa(assets_names = self.updated_symbols, indicators_names = self.pf_params['indicators'], query_date = self.curr_date, 
                  number_assets = None, replace_by_zero = True, start = None, end=None, indic_weights = None, verbose = False)
                weights = weights[weights != 0.0].dropna(axis = 1)
                self.updated_symbols = weights.columns.tolist()
                weights = weights.values.ravel()

            if weights is not None:
                if round is not None:
                    print 'WEIGHTS BEFORE: ', weights
                    tmp = np.sign(weights)
                    weights = tmp * np.floor(np.abs(weights) * 10**(round))/(10**round)
                    print 'WEIGHTS AFTER: ', weights
                self.weights = pd.DataFrame(weights, index = self.updated_symbols, columns = ['weights'])
                self.weights = self.weights[self.weights['weights'] != 0.0]
                self.updated_symbols = self.weights.index.tolist()
            else:
                self.updated_symbols = []
        except:
            self.updated_symbols = []
            e = sys.exc_info()
            filename = os.getcwd() + '/ALLOCATION_ERROR_LOGS_'+ self.scheduler.upper() + '/' + self.strategy_id + '.txt'
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            with open(filename, 'a') as logfile:
                logfile.write('Allocation Error: %s\nCurrent date: %s Group: %s \n\n' % (e, self.curr_date, self.name))

        if verbose:
            print 'Weights:\n', self.weights
            print 'Elapsed time for portfolio_weights is %f'%(time.time() - start_time)
            print 'Out of portfolio_weights of %s' % self.name 
        return self

    def print_(self):
        print "-----------------------"
        print 'pf_params:\n', self.pf_params
        print "-----------------------"
        print 'symbols:\n', self.symbols
        print "-----------------------"
        print 'need returns and covariances:\n', self.need_returns, self.need_cov
        print "-----------------------"


    @staticmethod
    def _verify_psd(corr, max_iterations=10000, delta = 1e-7):

        try:
            np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            corr = nearcorr(corr, max_iterations = max_iterations, delta = delta, eps = 1e-6)#np.spacing(1e3))
        return corr

    def check_model_prerequisites(self, verbose = False):
        if verbose:
            print 'In check_model_prerequisites of %s' % self.name
        self.need_returns, self.need_cov = [False]*2
        # Define framework prerequisites
        if self.framework_params['model'] in AbstractGroup.NEED_RETURNS:
            self.need_returns = True
        if self.framework_params['model'] in AbstractGroup.NEED_COV:
            self.need_cov = True

        if self.pf_params['model'] in AbstractGroup.POSSIBLE_PF_MODELS:
            if self.pf_params['model'] in AbstractGroup.NEED_RETURNS:
                self.need_returns = True
        
            if self.pf_params['model'] in AbstractGroup.NEED_COV:
                self.need_cov = True
        else:
            raise  ValueError('ERROR: %s is not a supported portfolio model!\nThe supported portfolio models are: %s' % (self.pf_params['model'],str(AbstractGroup.POSSIBLE_PF_MODELS)))
            if verbose:
                print 'Need Returns and Covariances: ', self.need_returns, self.need_cov
                print 'Out of check_model_prerequisites of %s' % self.name
        return self

if __name__ == '__main__':
    print "OLA"