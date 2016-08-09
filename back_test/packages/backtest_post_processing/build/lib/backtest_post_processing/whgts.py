import pandas as pd

def allocation_to_df(allocation):
    """
    Allocation dict to df 
    """
    list_of_stocks = allocation.iloc[-1].keys()
    list_of_stocks.sort()
  
    columns = ['weight_'+stock for stock in list_of_stocks]
    def parse_to_dataframe(row, alloc):
        alloc.loc[row.name] = [row.values[0][key] for key in sorted(row.values[0])]
        return 0 

    alloc_aux  = pd.DataFrame(allocation).dropna()
    alloc = pd.DataFrame(index = alloc_aux.index, columns = columns)

    alloc_aux.apply(parse_to_dataframe, axis = 1, args = (alloc,))
    return list_of_stocks, alloc
    