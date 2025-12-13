def add_features(df):
    """
    Adds engineered features useful for volatility prediction.
    - return: percent change of close
    - vol_7d, vol_30d: rolling std of returns
    - ma_7, ma_30: moving averages of close
    - liquidity ratio: volume / market_cap
    - price_range: (high - low) / open
    - volatility_next_1d: target (next day's vol_7d)
    """
    df = df.sort_values(['symbol','date']).reset_index(drop=True)
    df['return'] = df.groupby('symbol')['close'].pct_change()
    df['vol_7d'] = df.groupby('symbol')['return'].rolling(window=7, min_periods=1).std().reset_index(0,drop=True)
    df['vol_30d'] = df.groupby('symbol')['return'].rolling(window=30, min_periods=1).std().reset_index(0,drop=True)
    df['ma_7'] = df.groupby('symbol')['close'].rolling(window=7, min_periods=1).mean().reset_index(0,drop=True)
    df['ma_30'] = df.groupby('symbol')['close'].rolling(window=30, min_periods=1).mean().reset_index(0,drop=True)
    df['liq_ratio'] = df['volume'] / df['market_cap'].replace({0: None})
    df['price_range'] = (df['high'] - df['low']) / df['open']
    df['volatility_next_1d'] = df.groupby('symbol')['vol_7d'].shift(-1)
    return df
