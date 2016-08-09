import os
import sys

import perf_attr as pa
import glob

import pandas as pd


def build_weights_df(csvfilename, savedir):
    savedir = savedir if '/' == savedir[-1] else savedir + '/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    df = pd.read_csv(csvfilename, index_col = 0, parse_dates = [0])
    df =  pa.utils.extract_allocation_from_zipline(df)
    filename = csvfilename.split('/')[-1].split('.')[0]
    df.to_csv(savedir + 'allocation_' + filename + '.csv')

if __name__ == '__main__':
    group = 'Top' # 'Top' or 'Equity'
    if len(sys.argv) >= 3 and 'ASSETTYPE=' in sys.argv[2]:
        group = sys.argv[2].split('=')[-1]

    freq = 'Monthly' # 'Monthly' or 'Weekly'
    if len(sys.argv) >= 2 and 'SCHEDULER=' in sys.argv[1]:
        freq = sys.argv[1].split('=')[-1]

    resultsdir = os.getcwd() + '/synology/%s/RESULTS_%s/' % (group, freq.upper())    
    savedir = os.getcwd() + '/synology/%s/RESULTS_%s/Reports/Allocation/'  % (group, freq.upper()) 
    results = glob.glob(resultsdir + '*.csv')
    for result in results:
        build_weights_df(result, savedir)
