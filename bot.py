import time
import numpy as np
import pandas as pd
import yfinance as yf

YEARS = 10
END_DATE = pd.Timestamp.today().normalize()
START_DATE = END_DATE - pd.DateOffset(years=YEARS)
WARMUP_START = START_DATE - pd.DateOffset(years=3)

MEMBERSHIP_URL = 'https://raw.githubusercontent.com/aditya-jha/nse-historical-membership/main/index_history/data/index_membership_history.csv'
OUTPUT_XLSX = 'nifty500_monthly_supertrend_daily_rsi_history.xlsx'


def clean_index(df):
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.get_level_values(0)
    idx = pd.to_datetime(df.index, errors='coerce')
    try:
        idx = idx.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    df.index = pd.DatetimeIndex(idx).normalize()
    df = df[~df.index.isna()]
    return df.loc[~df.index.duplicated(keep='last')].sort_index()


def supertrend(high, low, close, period=10, multiplier=1.0):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    hl2 = (high + low) / 2
    ub = hl2 + multiplier * atr
    lb = hl2 - multiplier * atr
    fu = ub.copy()
    fl = lb.copy()
    trend = pd.Series(index=close.index, dtype='object')

    for i in range(len(close)):
        if pd.isna(atr.iloc[i]):
            trend.iloc[i] = None
            continue
        if i == 0:
            trend.iloc[i] = 'GREEN' if close.iloc[i] >= hl2.iloc[i] else 'RED'
            continue
        if ub.iloc[i] < fu.iloc[i - 1] or close.iloc[i - 1] > fu.iloc[i - 1]:
            fu.iloc[i] = ub.iloc[i]
        else:
            fu.iloc[i] = fu.iloc[i - 1]
        if lb.iloc[i] > fl.iloc[i - 1] or close.iloc[i - 1] < fl.iloc[i - 1]:
            fl.iloc[i] = lb.iloc[i]
        else:
            fl.iloc[i] = fl.iloc[i - 1]
        if trend.iloc[i - 1] == 'RED':
            trend.iloc[i] = 'GREEN' if close.iloc[i] > fu.iloc[i] else 'RED'
        else:
            trend.iloc[i] = 'RED' if close.iloc[i] < fl.iloc[i] else 'GREEN'
    return trend


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.where(al != 0, 100)
    out = out.where(~((ag == 0) & (al == 0)), 50)
    return out


def extract_symbol(raw, ticker):
    if raw is None or raw.empty:
        return None
    data = raw.copy()
    if isinstance(data.columns, pd.MultiIndex):
        l0 = list(data.columns.get_level_values(0))
        l1 = list(data.columns.get_level_values(1))
        if ticker in l1:
            data = data.xs(ticker, axis=1, level=1, drop_level=True)
        elif ticker in l0:
            data = data.xs(ticker, axis=1, level=0, drop_level=True)
        elif len(set(l0)) == 1:
            data.columns = data.columns.get_level_values(1)
        elif len(set(l1)) == 1:
            data.columns = data.columns.get_level_values(0)
    needed = {'Open', 'High', 'Low', 'Close'}
    if not needed.issubset(set(data.columns)):
        return None
    return clean_index(data[['Open', 'High', 'Low', 'Close']])


def load_membership():
    m = pd.read_csv(MEMBERSHIP_URL)
    m.columns = [str(c).strip() for c in m.columns]
    def find(names):
        for c in m.columns:
            if c.lower() in names:
                return c
        raise ValueError(f'Membership column not found: {m.columns.tolist()}')
    sc = find({'symbol', 'ticker', 'stock'})
    fc = find({'valid_from', 'from', 'start_date'})
    tc = find({'valid_to', 'to', 'end_date'})
    m['Symbol'] = m[sc].astype(str).str.strip().str.upper().str.replace('.NS', '', regex=False)
    m['valid_from'] = pd.to_datetime(m[fc], errors='coerce').dt.normalize()
    m['valid_to'] = pd.to_datetime(m[tc], errors='coerce').dt.normalize()
    return m.dropna(subset=['Symbol', 'valid_from'])[['Symbol', 'valid_from', 'valid_to']]


def is_member(membership, symbol, date):
    d = pd.Timestamp(date).normalize()
    x = membership[membership['Symbol'] == symbol]
    return bool(((x['valid_from'] <= d) & (x['valid_to'].isna() | (x['valid_to'] > d))).any())


def scan_stock(symbol, daily, membership):
    daily = daily.loc[(daily.index >= WARMUP_START) & (daily.index <= END_DATE)].copy()
    if len(daily) < 250:
        return []

    daily['Daily_RSI14'] = rsi(daily['Close'], 14)

    # Monthly bars. The green state is available only after the month closes.
    monthly = daily.resample('ME').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
    monthly['Monthly_ST'] = supertrend(monthly['High'], monthly['Low'], monthly['Close'], 10, 1.0)
    monthly['Prev_ST'] = monthly['Monthly_ST'].shift(1)
    monthly['Available_Date'] = monthly.index + pd.offsets.BDay(1)

    turns = monthly[(monthly['Monthly_ST'] == 'GREEN') & (monthly['Prev_ST'] != 'GREEN')].copy()
    events = []

    for _, mt in turns.iterrows():
        green_known_date = pd.Timestamp(mt['Available_Date']).normalize()
        if green_known_date < START_DATE or green_known_date > END_DATE:
            continue
        if not is_member(membership, symbol, green_known_date):
            continue

        # Start only AFTER the monthly green state is known.
        future = daily.loc[(daily.index > green_known_date) & (daily.index >= START_DATE) & (daily.index <= END_DATE)]
        hits = future[future['Daily_RSI14'] < 31]
        for hit_date, row in hits.iterrows():
            if not is_member(membership, symbol, hit_date):
                continue
            events.append({
                'Date': hit_date.date(),
                'Stock': symbol,
                'Monthly_Supertrend_Green_Date': green_known_date.date(),
                'Daily_RSI14': round(float(row['Daily_RSI14']), 2),
                'Daily_Close': round(float(row['Close']), 2),
            })
    return events


def download_batch(symbols):
    tickers = [f'{s}.NS' for s in symbols]
    for attempt in range(3):
        try:
            raw = yf.download(
                tickers=tickers,
                start=WARMUP_START.strftime('%Y-%m-%d'),
                end=(END_DATE + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                interval='1d', auto_adjust=False, group_by='ticker',
                threads=True, progress=False,
            )
            if raw is not None and not raw.empty:
                return raw
        except Exception as e:
            print(f'Batch attempt {attempt + 1} failed: {e}')
        time.sleep(3 * (attempt + 1))
    return None


def main():
    print(f'Period: {START_DATE.date()} to {END_DATE.date()}')
    membership = load_membership()
    symbols = sorted(membership['Symbol'].dropna().unique())
    print(f'Historical symbols: {len(symbols)}')

    events = []
    errors = []

    for start in range(0, len(symbols), 25):
        batch = symbols[start:start + 25]
        print(f'Downloading {start + 1}-{start + len(batch)} of {len(symbols)}')
        raw = download_batch(batch)
        if raw is None:
            errors.extend({'Stock': s, 'Error': 'Yahoo batch download failed'} for s in batch)
            continue
        for symbol in batch:
            try:
                df = extract_symbol(raw, f'{symbol}.NS')
                if df is None:
                    errors.append({'Stock': symbol, 'Error': 'No usable Yahoo data'})
                    continue
                rows = scan_stock(symbol, df, membership)
                events.extend(rows)
                print(f'{symbol}: {len(rows)} qualifying dates')
            except Exception as e:
                errors.append({'Stock': symbol, 'Error': str(e)})

    result = pd.DataFrame(events)
    if result.empty:
        result = pd.DataFrame(columns=['Date','Stock','Monthly_Supertrend_Green_Date','Daily_RSI14','Daily_Close'])
    else:
        result['Date'] = pd.to_datetime(result['Date'])
        result = result.sort_values(['Date','Stock']).drop_duplicates(['Date','Stock']).reset_index(drop=True)

    errors_df = pd.DataFrame(errors)
    if errors_df.empty:
        errors_df = pd.DataFrame(columns=['Stock','Error'])

    criteria = pd.DataFrame({'Rule': [
        'Historical point-in-time NIFTY 500 constituents',
        'Last 10 years',
        'Monthly Supertrend (10,1)',
        'Monthly Supertrend must turn GREEN',
        'Daily scanning starts after the monthly GREEN state is known',
        'Daily RSI(14) must be strictly below 31',
        'No 0-7 day restriction',
        'No profitability/backtest calculation',
    ]})

    with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
        result.to_excel(writer, sheet_name='Qualifying Stocks', index=False)
        criteria.to_excel(writer, sheet_name='Criteria', index=False)
        errors_df.to_excel(writer, sheet_name='Data Errors', index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = 'A2'
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 12), 45)

    print(f'Created {OUTPUT_XLSX}; qualifying rows={len(result)}; errors={len(errors_df)}')


if __name__ == '__main__':
    main()
