from __future__ import division

import pandas as pd







def map_transaction(txn):
    """
    Maps a single transaction row to a dictionary.
    Parameters
    ----------
    txn : pd.DataFrame
        A single transaction object to convert to a dictionary.
    Returns
    -------
    dict
        Mapped transaction.
    """
    # sid can either be just a single value or a SID descriptor
    if isinstance(txn['sid'], dict):
        print "ENTREI!!"
        sid = txn['sid']['sid']
        symbol = txn['sid']['symbol']
    else:
        sid = txn['sid']
        symbol = txn['sid']

    return {'sid': sid,
            'symbol': symbol,
            'price': txn['price'],
            'order_id': txn['order_id'],
            'amount': txn['amount'],
            'commission': txn['commission'],
            'dt': txn['dt']}


def make_transaction_frame(transactions, list_of_stocks):
    """
    Formats a transaction DataFrame.
    Parameters
    ----------
    transactions : pd.DataFrame
        Contains improperly formatted transactional data.
    Returns
    -------
    df : pd.DataFrame
        Daily transaction volume and dollar ammount.
         - See full explanation in tears.create_full_tear_sheet.
    """

    transaction_list = []
    for dt in transactions.index:

        txns = transactions.loc[dt]

        if len(txns) == 0:
            continue

        for txn in txns:

            txn = map_transaction(txn)
            transaction_list.append(txn)

    df = pd.DataFrame(sorted(transaction_list, key=lambda x: x['dt']))
    def map_rows_to_columns_hardcoded(df):
        dict = {}
        for row in range(len(df)):
            symbol = list_of_stocks[df.iloc[row]['symbol']]
            columns = [elem+'_'+str(symbol) for elem in df.iloc[row].index if elem in ['amount', 'commission', 'price']]
            values = df.iloc[row][['amount', 'commission', 'price']].values
            for key, val in zip(columns, values):
                dict[key] = val
        return pd.DataFrame(dict, index = [df.iloc[0]['dt']])

    df =  df.groupby('dt').apply(map_rows_to_columns_hardcoded)
    df.index = df.index.levels[0]
    df.index.name = 'date'
    
    return df


def get_txn_vol(transactions):
    """Extract daily transaction data from set of transaction objects.
    Parameters
    ----------
    transactions : pd.DataFrame
        Time series containing one row per symbol (and potentially
        duplicate datetime indices) and columns for amount and
        price.
    Returns
    -------
    pd.DataFrame
        Daily transaction volume and number of shares.
         - See full explanation in tears.create_full_tear_sheet.
    """
    transactions.index = transactions.index.normalize()
    amounts = transactions.amount.abs()
    prices = transactions.price
    values = amounts * prices
    daily_amounts = amounts.groupby(amounts.index).sum()
    daily_values = values.groupby(values.index).sum()
    daily_amounts.name = "txn_shares"
    daily_values.name = "txn_volume"
    return pd.concat([daily_values, daily_amounts], axis=1)


def adjust_returns_for_slippage(returns, turnover, slippage_bps):
    """Apply a slippage penalty for every dollar traded.
    Parameters
    ----------
    returns : pd.Series
        Time series of daily returns.
    turnover: pd.Series
        Time series of daily total of buys and sells
        divided by portfolio value.
            - See txn.get_turnover.
    slippage_bps: int/float
        Basis points of slippage to apply.
    Returns
    -------
    pd.Series
        Time series of daily returns, adjusted for slippage.
    """
    slippage = 0.0001 * slippage_bps
    # Only include returns in the period where the algo traded.
    trim_returns = returns.loc[turnover.index]
    return trim_returns - turnover * slippage


def get_turnover(positions, transactions, period=None, average=True):
    """
    Portfolio Turnover Rate:
    Value of purchases and sales divided
    by the average portfolio value for the period.
    If no period is provided the period is one time step.
    Parameters
    ----------
    positions : pd.DataFrame
        Contains daily position values including cash
        - See full explanation in tears.create_full_tear_sheet
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
        - See full explanation in tears.create_full_tear_sheet
    period : str, optional
        Takes the same arguments as df.resample.
    average : bool
        if True, return the average of purchases and sales divided
        by portfolio value. If False, return the sum of
        purchases and sales divided by portfolio value.
    Returns
    -------
    turnover_rate : pd.Series
        timeseries of portfolio turnover rates.
    """
    txn_vol = get_txn_vol(transactions)
    traded_value = txn_vol.txn_volume
    portfolio_value = positions.sum(axis=1)
    if period is not None:
        traded_value = traded_value.resample(period, how='sum')
        portfolio_value = portfolio_value.resample(period, how='mean')
    # traded_value contains the summed value from buys and sells;
    # this is divided by 2.0 to get the average of the two.
    turnover = traded_value / 2.0 if average else traded_value
    turnover_rate = turnover.div(portfolio_value, axis='index')
    turnover_rate = turnover_rate.fillna(0)
    return turnover_rate