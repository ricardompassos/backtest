import perf_attr as pa
import sys

group = 'Equity' # 'Top' or 'Equity'
if len(sys.argv) >= 3 and 'ASSETTYPE=' in sys.argv[2]:
	group = sys.argv[2].split('=')[-1]

freq = 'Monthly' # 'Weekly' or 'Monthly'
if len(sys.argv) >= 2 and 'SCHEDULER=' in sys.argv[1]:
	freq = sys.argv[1].split('=')[-1]
pa.pdf_reports.main(group = group, freq = freq)