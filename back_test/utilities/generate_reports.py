import perf_attr as pa
import pandas as pd
import os
import sys

group = 'Top'
if len(sys.argv) >= 3 and 'ASSETTYPE=' in sys.argv[2]:
    group = sys.argv[2].split('=')[-1]

freqs = ['MONTHLY']
if len(sys.argv) >= 2 and 'SCHEDULER=' in sys.argv[1]:
    freqs = sys.argv[1].split('=')[-1].split(',')

for freq in freqs:
    #try:
        freq = freq.upper()
        resultsdir = './synology/%s/RESULTS_%s/' % (group, freq)
        paramsdir = os.getcwd() + '/PARAMETERS_%s/' % (freq)

        print paramsdir
    
        paramsreportfilename = resultsdir + 'Reports/' + 'params_descriptor.csv'
        pa.report.report_params_df(paramsdir = paramsdir, savefilename = paramsreportfilename, top_params = (group == 'Top'))
        pa.report.report_strategies(pathdir = resultsdir, benchmark_str = '[Market]' if (group == 'Top') else '[Market] / %s' % group)
        df1 = pd.read_csv(paramsreportfilename, index_col = 'Unnamed: 0')
        df2 = pd.read_csv(resultsdir + 'Reports/'+ 'report_perf_metrics.csv', index_col = 'Unnamed: 0')
        pd.concat([df1,df2], axis = 1).to_csv(resultsdir + 'Reports/report_merged.csv')
        
    #except:
    #    print "ERROR: In freq %s" % freq
