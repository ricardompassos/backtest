

import MySQLdb
import pandas.io.sql as psql
from MySQLdb.converters import conversions
from MySQLdb.constants import FIELD_TYPE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dateutil.parser
import math

conversions[FIELD_TYPE.DECIMAL] = float
conversions[FIELD_TYPE.NEWDECIMAL] = float




def load_data_from_passdb(fields_to_load, start_date = None, end_date = None, source = 'BLP', timezone_indication = 'UTC'):
    
    """
    Loads data from Pass DB into a panel with the following
    column names for each indicated security:

        - open
        - high
        - low
        - close
        - volume
        - price
    
    - ONLY DONE FOR MARKET DATA TYPE
    
    -> 'fields_to_load' is a python list with the tickers to load
    -> 'source' is the data source where to get the data; default is 'BLP' (Bloomberg)
    
    """
    # ====================================================
    # OPEN DATABASE CONNECTION
    db = MySQLdb.connect("192.168.51.100","PASS_DEV","Develop-2015","PASS_SYS" )
    cursor = db.cursor()
    # ====================================================
    check_volume = True
    df_dict = {}

    for local_field in fields_to_load:
        querie = "select HD_PK, CD_SERIETYPE, ST_INSTRUMENT, ST_SECURITY_NAME, ST_FREQ, ST_NAME from PASS_SYS.V_SERIE where ST_SECURITY_CODE='" + local_field + "'"
        cursor.execute(querie)
        data = cursor.fetchall()
        
        # select the source
        if len(data) != 1:
            for k in xrange(len(data)):
                if data[k][-1].split(' / ')[0] == source:
                    search_index = k
        else:
            search_index = 0
                
        querie = "select DT_DATE, NU_PX_OPEN, NU_PX_HIGH, NU_PX_LOW, NU_PX_LAST, NU_PX_VOLUME, NU_PX_LAST from PASS_SYS.V_MKTDATA where LK_SERIE = unhex('%s')" % (data[search_index][0].encode('hex'))
        df_local = psql.read_sql(querie, db, index_col = 'DT_DATE')
        # ----------------------
        # Data Post Processing
        # Resampling
        df_local = df_local.resample(rule='B', how='last', fill_method='pad')
        # Fill NA with zeros (this may be redundant or not but its better to do)
        df_local = df_local.fillna(value=0.0)
        # Change columns name
        df_local.columns = ['open', 'high', 'low', 'close', 'volume', 'price']
        if check_volume:
            if (df_local['volume'] == np.zeros_like(df_local['volume'])).all():
                df_local['volume'] += 1000000.0
        # Change index name
        df_local.index.names = ['Date']
        # Convert the dates to a format that match zipline
        dates_list_temp = df_local.index.tolist()
        for i in xrange(len(dates_list_temp)):
            dates_list_temp[i] = pd.Timestamp(dates_list_temp[i], tz=timezone_indication)
        df_local.index = dates_list_temp
        # Cut the dataframe if requested
        if start_date is not None:
            df_local = df_local[df_local.index >= pd.Timestamp(start_date, tz=timezone_indication)] 
        if end_date is not None:
            df_local = df_local[df_local.index <= pd.Timestamp(end_date, tz=timezone_indication)] 


        # ----------------------
        df_dict[local_field] = df_local
    
    
    # Convert everything to a pandas Panel
    panel_data = pd.Panel(df_dict)
    
    return panel_data


if __name__ == "__main__":
    
    print "Begin"
    
    
    fields_to_load = ['USCRWTIC INDEX']
    
    
    data = load_data_from_passdb(fields_to_load, source = 'BLP', timezone_indication = 'UTC')
    print data
    
    
    
    