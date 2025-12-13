import pandas as pd
import numpy as np

def load_and_clean(path):
    """
    Load CSV and standardize column names. Returns DataFrame.
    Expected columns (case-insensitive): date, symbol, open, high, low, close, volume, market cap/market_cap
    """
    # Parse date if present
    try:
        df = pd.read_csv(path, parse_dates=['date'])
    except Exception:
        df = pd.read_csv(path)

    # normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    # normalize market cap column name
    if 'market cap' in df.columns:
        df = df.rename(columns={'market cap': 'market_cap'})
    elif 'marketcap' in df.columns:
        df = df.rename(columns={'marketcap': 'market_cap'})

    # Ensure numeric types
    for c in ['open','high','low','close','volume','market_cap']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan

    # If symbol missing, create a generic symbol
    if 'symbol' not in df.columns:
        df['symbol'] = 'UNKNOWN'

    # Drop rows with all-NaN prices
    df = df.dropna(subset=['open','close'], how='all').reset_index(drop=True)

    return df
