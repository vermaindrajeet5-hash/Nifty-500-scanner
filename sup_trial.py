import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# SUP TRIAL SETTINGS
# ============================================================

YEARS = 10
RSI_PERIOD = 5
RSI_LIMIT = 31
ST_PERIOD = 10
ST_MULTIPLIER = 1.0

MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/aditya-jha/"
    "nse-historical-membership/main/index_history/data/"
    "index_membership_history.csv"
)

OUTPUT_FILE = "sup_trial_results.xlsx"

# NIFTY 500 Yahoo Finance symbols.
# The historical membership file is used so a stock is considered
# only on dates when it was a NIFTY 500 constituent.
# Yahoo symbols are converted to NSE format by adding .NS.
INDEX_SYMBOLS = [
    "360ONE", "3MINDIA", "ABB", "ACC", "ACMESOLAR", "ADANIENSOL", "ADANIENT",
    "ADANIGREEN", "ADANIPORTS", "ADANIPOWER", "ATGL", "AWL", "ABCAPITAL",
    "ABFRL", "ALKEM", "AMBER", "AMBUJACEM", "ANGELONE", "APLAPOLLO",
    "APARINDS", "APLLTD", "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY", "ASIANPAINT",
    "ASTRAL", "ASTRAZEN", "ATUL", "AUBANK", "AUROPHARMA", "AXISBANK",
    "BAJAJ-AUTO", "BAJAJFINSV", "BAJAJHLDNG", "BAJFINANCE", "BANDHANBNK",
    "BANKBARODA", "BANKINDIA", "BATAINDIA", "BDL", "BEL", "BERGEPAINT",
    "BHARATFORG", "BHARTIARTL", "BHEL", "BIOCON", "BOSCHLTD", "BPCL",
    "BRITANNIA", "BSE", "CAMS", "CANBK", "CASTROLIND", "CEATLTD", "CENTRALBK",
    "CGPOWER", "CHAMBLFERT", "CHENNPETRO", "CHOLAFIN", "CIPLA", "COALINDIA",
    "COCHINSHIP", "COFORGE", "COLPAL", "CONCOR", "COROMANDEL", "CROMPTON",
    "CUMMINSIND", "CYIENT", "DABUR", "DALBHARAT", "DBREALTY", "DCMSHRIRAM",
    "DEEPAKNTR", "DELHIVERY", "DIVISLAB", "DIXON", "DLF", "DMART", "DRREDDY",
    "EICHERMOT", "EIDPARRY", "EIHOTEL", "EMAMILTD", "ENDURANCE", "ENGINERSIN",
    "ERIS", "ESCORTS", "ETERNAL", "EXIDEIND", "FEDERALBNK", "FINCABLE",
    "FORTIS", "FSL", "GAIL", "GESHIP", "GLAND", "GLENMARK", "GODREJCP",
    "GODREJIND", "GODREJPROP", "GPPL", "GRANULES", "GRAPHITE", "GRASIM",
    "HAL", "HAVELLS", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
    "HFCL", "HINDALCO", "HINDCOPPER", "HINDPETRO", "HINDUNILVR", "HINDZINC",
    "HOMEFIRST", "HONASA", "HUDCO", "ICICIBANK", "ICICIGI", "ICICIPRULI",
    "IDBI", "IDEA", "IDFCFIRSTB", "IEX", "IFCI", "IGL", "IIFL", "INDHOTEL",
    "INDIACEM", "INDIAMART", "INDIANB", "INDIGO", "INDUSINDBK", "INDUSTOWER",
    "INFY", "INOXWIND", "INTELLECT", "IOC", "IPCALAB", "IRB", "IRCON",
    "IREDA", "IRFC", "ITC", "ITI", "JINDALSAW", "JINDALSTEL", "JIOFIN",
    "JKCEMENT", "JKTYRE", "JMFINANCIL", "JSWENERGY", "JSWHL", "JSWSTEEL",
    "JUBLFOOD", "JUBLPHARMA", "JUSTDIAL", "JYOTHYLAB", "KALYANKJIL", "KAYNES",
    "KEI", "KFINTECH", "KIRLOSBROS", "KIRLOSENG", "KOTAKBANK", "KPIL",
    "KPITTECH", "KPRMILL", "LAURUSLABS", "LEMONTREE", "LICHSGFIN", "LICI",
    "LINDEINDIA", "LLOYDSME", "LODHA", "LT", "LTIM", "LTTS", "LUPIN",
    "M&M", "M&MFIN", "MANAPPURAM", "MANKIND", "MARICO", "MARUTI", "MAXHEALTH",
    "MAZDOCK", "MCX", "MEDANTA", "MFSL", "MGL", "MOTHERSON", "MOTILALOFS",
    "MPHASIS", "MRF", "MRPL", "MUTHOOTFIN", "NATIONALUM", "NAUKRI", "NBCC",
    "NCC", "NESTLEIND", "NHPC", "NMDC", "NTPC", "NUVAMA", "NYKAA", "OBEROIRLTY",
    "OFSS", "OIL", "OLECTRA", "ONGC", "PAGEIND", "PATANJALI", "PAYTM",
    "PERSISTENT", "PETRONET", "PFC", "PFIZER", "PGEL", "PGHH", "PHOENIXLTD",
    "PIDILITIND", "PIIND", "PNB", "PNBHOUSING", "POLICYBZR", "POLYCAB",
    "POONAWALLA", "POWERGRID", "POWERINDIA", "PRESTIGE", "PPLPHARMA", "RBLBANK",
    "RECLTD", "RELIANCE", "RVNL", "SAIL", "SAMMAANCAP", "SAREGAMA", "SBICARD",
    "SBILIFE", "SBIN", "SCHAEFFLER", "SHREECEM", "SHRIRAMFIN", "SHYAMMETL",
    "SIEMENS", "SJVN", "SOLARINDS", "SONACOMS", "SRF", "STARHEALTH",
    "SUMICHEM", "SUNPHARMA", "SUNTV", "SUPREMEIND", "SUZLON", "SYNGENE",
    "TATACHEM", "TATACONSUM", "TATAELXSI", "TATAMOTORS", "TATAPOWER",
    "TATASTEEL", "TATATECH", "TBOTEK", "TCS", "TECHM", "TECHNO", "THERMAX",
    "TIINDIA", "TITAN", "TORNTPHARM", "TORNTPOWER", "TRENT", "TRIDENT",
    "TRIVENI", "TVSMOTOR", "UBL", "UCOBANK", "ULTRACEMCO", "UNOMINDA",
    "UPL", "USHAMART", "UTIAMC", "VBL", "VEDL", "VGUARD", "VIJAYA",
    "VOLTAS", "WAAREEENER", "WELCORP", "WELSPUNLIV", "WHIRLPOOL", "YESBANK",
    "ZEEL", "ZENSARTECH"
]


def rsi(series, period=5):
    """Wilder-style RSI using exponential smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))

    # If losses are zero, RSI is 100.
    out = out.where(avg_loss != 0, 100.0)
    # If both gain and loss are zero, RSI is neutral.
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)

    return out


def supertrend(df, period=10, multiplier=1.0):
    """
    Supertrend based on ATR(period), basic bands and final bands.

    Returns GREEN when the Supertrend line is below price
    and RED when it is above price.
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = pd.Series(index=df.index, dtype=float)
    final_lower = pd.Series(index=df.index, dtype=float)
    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=object)

    for i in range(len(df)):
        idx = df.index[i]

        if pd.isna(atr.iloc[i]):
            direction.iloc[i] = None
            continue

        if i == 0 or pd.isna(st.iloc[i - 1]):
            final_upper.iloc[i] = basic_upper.iloc[i]
            final_lower.iloc[i] = basic_lower.iloc[i]
            st.iloc[i] = final_lower.iloc[i]
            direction.iloc[i] = "GREEN"
            continue

        prev_idx = df.index[i - 1]

        if (
            basic_upper.iloc[i] < final_upper.loc[prev_idx]
            or close.loc[prev_idx] > final_upper.loc[prev_idx]
        ):
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.loc[prev_idx]

        if (
            basic_lower.iloc[i] > final_lower.loc[prev_idx]
            or close.loc[prev_idx] < final_lower.loc[prev_idx]
        ):
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.loc[prev_idx]

        prev_st = st.loc[prev_idx]

        if prev_st == final_upper.loc[prev_idx]:
            if close.iloc[i] <= final_upper.iloc[i]:
                st.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = "RED"
            else:
                st.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = "GREEN"
        else:
            if close.iloc[i] >= final_lower.iloc[i]:
                st.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = "GREEN"
            else:
                st.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = "RED"

    return direction


def clean_datetime_index(index):
    idx = pd.to_datetime(index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return pd.DatetimeIndex(idx).normalize()


def load_membership():
    membership = pd.read_csv(MEMBERSHIP_URL)

    # Handle likely column-name variations safely.
    cols = {c.lower().strip(): c for c in membership.columns}

    symbol_col = cols.get("symbol")
    from_col = cols.get("valid_from")
    to_col = cols.get("valid_to")

    if not symbol_col or not from_col or not to_col:
        raise ValueError(
            "Historical membership file does not contain "
            "symbol, valid_from and valid_to columns."
        )

    membership = membership[[symbol_col, from_col, to_col]].copy()
    membership.columns = ["Symbol", "ValidFrom", "ValidTo"]

    membership["Symbol"] = (
        membership["Symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    membership["ValidFrom"] = pd.to_datetime(
        membership["ValidFrom"], errors="coerce"
    )
    membership["ValidTo"] = pd.to_datetime(
        membership["ValidTo"], errors="coerce"
    )

    membership = membership.dropna(subset=["Symbol", "ValidFrom"])

    return membership


def is_member(symbol, date, membership):
    rows = membership[
        (membership["Symbol"] == symbol)
        & (membership["ValidFrom"] <= date)
        & (
            membership["ValidTo"].isna()
            | (membership["ValidTo"] > date)
        )
    ]
    return not rows.empty


def download_batch(symbols, start, end, retries=3):
    yahoo_symbols = [f"{s}.NS" for s in symbols]

    for attempt in range(1, retries + 1):
        try:
            data = yf.download(
                yahoo_symbols,
                start=start.strftime("%Y-%m-%d"),
                end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )

            if data is not None and not data.empty:
                return data

        except Exception as exc:
            print(f"Batch download attempt {attempt} failed: {exc}")

        time.sleep(3 * attempt)

    return None


def extract_symbol_data(batch_data, symbol):
    yahoo_symbol = f"{symbol}.NS"

    if batch_data is None or batch_data.empty:
        return None

    try:
        if isinstance(batch_data.columns, pd.MultiIndex):
            level0 = set(batch_data.columns.get_level_values(0))
            level1 = set(batch_data.columns.get_level_values(1))

            if yahoo_symbol in level0:
                df = batch_data[yahoo_symbol].copy()
            elif yahoo_symbol in level1:
                df = batch_data.xs(yahoo_symbol, axis=1, level=1).copy()
            else:
                return None
        else:
            df = batch_data.copy()

        needed = ["Open", "High", "Low", "Close"]
        if not all(c in df.columns for c in needed):
            return None

        df = df[needed].dropna(subset=["High", "Low", "Close"]).copy()
        df.index = clean_datetime_index(df.index)
        df = df[~df.index.duplicated(keep="last")].sort_index()

        return df

    except Exception:
        return None


def prepare_monthly_state(daily):
    monthly = daily.resample("ME").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        }
    ).dropna()

    if monthly.empty:
        return pd.Series(dtype=object)

    monthly["ST"] = supertrend(
        monthly,
        period=ST_PERIOD,
        multiplier=ST_MULTIPLIER,
    )

    return monthly["ST"]


def scan_symbol(symbol, daily, membership, start_date, end_date):
    daily = daily.copy()

    # Daily RSI(5).
    daily["Daily_RSI_5"] = rsi(
        daily["Close"],
        period=RSI_PERIOD
    )

    # Monthly Supertrend(10,1).
    monthly_st = prepare_monthly_state(daily)

    if monthly_st.empty:
        return []

    # For every trading day, use ONLY the previous completed month.
    month_start = daily.index.to_period("M").to_timestamp()
    daily["Previous_Month_End"] = (
        month_start - pd.Timedelta(days=1)
    )

    monthly_state = monthly_st.rename("Previous_Month_ST").reset_index()
    monthly_state.columns = ["Previous_Month_End", "Previous_Month_ST"]
    monthly_state["Previous_Month_End"] = clean_datetime_index(
        monthly_state["Previous_Month_End"]
    )

    daily = daily.reset_index().rename(columns={"index": "Date"})
    daily["Date"] = clean_datetime_index(daily["Date"])

    daily = pd.merge_asof(
        daily.sort_values("Date"),
        monthly_state.sort_values("Previous_Month_End"),
        left_on="Previous_Month_End",
        right_on="Previous_Month_End",
        direction="backward",
    )

    signals = daily[
        (daily["Date"] >= start_date)
        & (daily["Date"] <= end_date)
        & (daily["Daily_RSI_5"] < RSI_LIMIT)
        & (daily["Previous_Month_ST"] == "GREEN")
    ].copy()

    if signals.empty:
        return []

    results = []
    for _, row in signals.iterrows():
        signal_date = pd.Timestamp(row["Date"]).date()

        if is_member(symbol, pd.Timestamp(signal_date), membership):
            results.append(
                {
                    "Date": signal_date,
                    "Stock": symbol,
                    "Daily_RSI_5": round(float(row["Daily_RSI_5"]), 2),
                    "Previous_Month_ST": row["Previous_Month_ST"],
                    "Previous_Month_End": pd.Timestamp(
                        row["Previous_Month_End"]
                    ).date(),
                }
            )

    return results


def main():
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=YEARS)

    # Extra warm-up is needed for RSI and monthly Supertrend.
    download_start = start_date - pd.DateOffset(years=2)

    print(f"Scan period: {start_date.date()} to {end_date.date()}")
    print(f"Download period: {download_start.date()} to {end_date.date()}")

    membership = load_membership()
    print(f"Membership rows loaded: {len(membership)}")

    all_results = []
    errors = []

    # Download in batches to reduce Yahoo request failures.
    batch_size = 25

    for batch_start in range(0, len(INDEX_SYMBOLS), batch_size):
        batch_symbols = INDEX_SYMBOLS[
            batch_start: batch_start + batch_size
        ]

        print(
            f"Downloading batch "
            f"{batch_start + 1}-{batch_start + len(batch_symbols)} "
            f"of {len(INDEX_SYMBOLS)}"
        )

        batch_data = download_batch(
            batch_symbols,
            download_start,
            end_date,
        )

        if batch_data is None:
            for symbol in batch_symbols:
                errors.append(
                    {
                        "Stock": symbol,
                        "Error": "Yahoo batch download failed",
                    }
                )
            continue

        for symbol in batch_symbols:
            try:
                daily = extract_symbol_data(batch_data, symbol)

                if daily is None or daily.empty:
                    errors.append(
                        {
                            "Stock": symbol,
                            "Error": "No usable Yahoo data",
                        }
                    )
                    continue

                results = scan_symbol(
                    symbol,
                    daily,
                    membership,
                    start_date,
                    end_date,
                )

                all_results.extend(results)

                if results:
                    print(
                        f"{symbol}: {len(results)} signal(s)"
                    )

            except Exception as exc:
                errors.append(
                    {
                        "Stock": symbol,
                        "Error": str(exc),
                    }
                )

        time.sleep(1)

    signals_df = pd.DataFrame(all_results)
    errors_df = pd.DataFrame(errors)

    if signals_df.empty:
        signals_df = pd.DataFrame(
            columns=[
                "Date",
                "Stock",
                "Daily_RSI_5",
                "Previous_Month_ST",
                "Previous_Month_End",
            ]
        )
    else:
        signals_df = signals_df.sort_values(
            ["Date", "Stock"]
        ).reset_index(drop=True)

    rules_df = pd.DataFrame(
        {
            "Rule": [
                "Historical period",
                "Daily RSI",
                "Daily RSI threshold",
                "Monthly indicator",
                "Monthly Supertrend period",
                "Monthly Supertrend multiplier",
                "Monthly candle used",
                "Signal condition",
            ],
            "Value": [
                f"{start_date.date()} to {end_date.date()}",
                "RSI(5)",
                "< 31",
                "Supertrend",
                "10",
                "1",
                "Previous completed calendar month",
                "Daily RSI(5) < 31 AND previous completed month's Monthly Supertrend is GREEN",
            ],
        }
    )

    if errors_df.empty:
        errors_df = pd.DataFrame(
            columns=["Stock", "Error"]
        )

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        signals_df.to_excel(
            writer,
            sheet_name="Signals",
            index=False,
        )
        rules_df.to_excel(
            writer,
            sheet_name="Rules",
            index=False,
        )
        errors_df.to_excel(
            writer,
            sheet_name="Data Errors",
            index=False,
        )

        # Basic formatting.
        for sheet_name, worksheet in writer.sheets.items():
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

            for column_cells in worksheet.columns:
                max_length = 0
                column_letter = column_cells[0].column_letter

                for cell in column_cells:
                    value = "" if cell.value is None else str(cell.value)
                    max_length = max(max_length, len(value))

                worksheet.column_dimensions[column_letter].width = min(
                    max(max_length + 2, 12),
                    55,
                )

    print()
    print(f"Signals found: {len(signals_df)}")
    print(f"Data errors: {len(errors_df)}")
    print(f"Created: {Path(OUTPUT_FILE).resolve()}")


if __name__ == "__main__":
    main()
