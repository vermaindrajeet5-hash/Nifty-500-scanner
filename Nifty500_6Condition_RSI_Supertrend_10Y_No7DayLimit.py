import os
import smtplib
from io import StringIO
from email.message import EmailMessage

import pandas as pd
import requests
import yfinance as yf


# ============================================================
# SETTINGS
# ============================================================

YEARS = 10

END_DATE = pd.Timestamp.today().normalize()

START_DATE = END_DATE - pd.DateOffset(years=YEARS)

OUTPUT_FILE = "daily_historical_pit_signals.csv"

MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/"
    "aditya-jha/nse-historical-membership/"
    "main/index_history/data/index_membership_history.csv"
)


# ============================================================
# FORCE ALL DATETIME INDEXES TO THE SAME TYPE
# ============================================================

def clean_datetime_index(index):

    index = pd.DatetimeIndex(
        pd.to_datetime(index)
    )

    if index.tz is not None:
        index = index.tz_localize(None)

    index = index.normalize()

    # Important:
    # merge_asof requires exactly matching datetime dtypes.
    return index.astype("datetime64[ns]")


# ============================================================
# RSI WILDER
# ============================================================

def rsi_wilder(series, period=5):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


# ============================================================
# ADX WILDER
# ============================================================

def adx_wilder(high, low, close, period=14):

    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        0.0,
        index=high.index
    )

    minus_dm = pd.Series(
        0.0,
        index=high.index
    )

    plus_mask = (
        (up_move > down_move)
        &
        (up_move > 0)
    )

    minus_mask = (
        (down_move > up_move)
        &
        (down_move > 0)
    )

    plus_dm.loc[plus_mask] = up_move.loc[plus_mask]

    minus_dm.loc[minus_mask] = down_move.loc[minus_mask]

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    plus_dm_smooth = plus_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    minus_dm_smooth = minus_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    plus_di = 100 * plus_dm_smooth / atr

    minus_di = 100 * minus_dm_smooth / atr

    denominator = plus_di + minus_di

    dx = (
        100
        * (plus_di - minus_di).abs()
        / denominator
    )

    return dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


# ============================================================
# DAILY SUPERTREND
# PARAMETERS: ATR 10, FACTOR 1
# GREEN = CLOSE ABOVE SUPERTREND
# "BECOME GREEN" = TODAY GREEN, YESTERDAY NOT GREEN
# ============================================================

def supertrend_green(high, low, close, period=10, factor=1.0):

    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    hl2 = (high + low) / 2

    basic_upper = hl2 + factor * atr
    basic_lower = hl2 - factor * atr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    for i in range(1, len(close)):
        if (
            basic_upper.iloc[i] < final_upper.iloc[i - 1]
            or close.iloc[i - 1] > final_upper.iloc[i - 1]
        ):
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if (
            basic_lower.iloc[i] > final_lower.iloc[i - 1]
            or close.iloc[i - 1] < final_lower.iloc[i - 1]
        ):
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

    st = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)

    for i in range(len(close)):
        if i == 0:
            st.iloc[i] = final_upper.iloc[i]
            direction.iloc[i] = -1
            continue

        if st.iloc[i - 1] == final_upper.iloc[i - 1]:
            if close.iloc[i] <= final_upper.iloc[i]:
                st.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1
            else:
                st.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
        else:
            if close.iloc[i] >= final_lower.iloc[i]:
                st.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
            else:
                st.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1

    green = direction.eq(1)
    green_turn = green & ~green.shift(1, fill_value=False)

    return st, green, green_turn


# ============================================================
# LOAD HISTORICAL NIFTY 500 MEMBERSHIP
# ============================================================

def load_membership():

    print("Loading historical NIFTY 500 membership...")

    response = requests.get(
        MEMBERSHIP_URL,
        timeout=60
    )

    response.raise_for_status()

    membership = pd.read_csv(
        StringIO(response.text)
    )

    membership["valid_from"] = (
        pd.to_datetime(
            membership["valid_from"],
            errors="coerce",
            utc=True
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )

    membership["valid_to"] = (
        pd.to_datetime(
            membership["valid_to"],
            errors="coerce",
            utc=True
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )

    membership["index_name"] = (
        membership["index_name"]
        .astype(str)
        .str.strip()
    )

    membership["symbol"] = (
        membership["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    membership = membership[
        membership["index_name"]
        .str.upper()
        .eq("NIFTY 500")
    ].copy()

    membership = membership[
        membership["valid_from"].notna()
    ].copy()

    print(
        "Membership records:",
        len(membership)
    )

    return membership


# ============================================================
# CHECK MEMBERSHIP ON SIGNAL DATE
# ============================================================

def is_member_on_date(
    membership,
    symbol,
    date
):

    date = pd.Timestamp(date).normalize()

    rows = membership[
        (membership["symbol"] == symbol.upper())
        &
        (membership["valid_from"] <= date)
        &
        (
            membership["valid_to"].isna()
            |
            (membership["valid_to"] > date)
        )
    ]

    return not rows.empty


# ============================================================
# GET HISTORICAL SYMBOLS
# ============================================================

def get_historical_symbols(membership):

    relevant = membership[
        (
            membership["valid_to"].isna()
            |
            (membership["valid_to"] >= START_DATE)
        )
        &
        (membership["valid_from"] <= END_DATE)
    ]

    symbols = sorted(
        set(
            relevant["symbol"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
        )
    )

    print(
        "Historical NIFTY 500 symbols:",
        len(symbols)
    )

    return symbols


# ============================================================
# BUILD WEEKLY DATA
# ============================================================

def build_weekly_data(data):

    weekly = data.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    })

    weekly.dropna(inplace=True)

    weekly["Weekly_RSI5"] = rsi_wilder(
        weekly["Close"],
        5
    )

    weekly["Weekly_RSI_SMA14"] = (
        weekly["Weekly_RSI5"]
        .rolling(14)
        .mean()
    )

    weekly["Weekly_ADX14"] = adx_wilder(
        weekly["High"],
        weekly["Low"],
        weekly["Close"],
        14
    )

    weekly.index = clean_datetime_index(
        weekly.index
    )

    return weekly[
        [
            "Weekly_RSI5",
            "Weekly_RSI_SMA14",
            "Weekly_ADX14"
        ]
    ]


# ============================================================
# BUILD MONTHLY DATA
# ============================================================

def build_monthly_data(data):

    monthly = data.resample("ME").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    })

    monthly.dropna(inplace=True)

    monthly["Monthly_RSI5"] = rsi_wilder(
        monthly["Close"],
        5
    )

    monthly["Monthly_RSI_SMA14"] = (
        monthly["Monthly_RSI5"]
        .rolling(14)
        .mean()
    )

    monthly["Monthly_ADX14"] = adx_wilder(
        monthly["High"],
        monthly["Low"],
        monthly["Close"],
        14
    )

    monthly.index = clean_datetime_index(
        monthly.index
    )

    return monthly[
        [
            "Monthly_RSI5",
            "Monthly_RSI_SMA14",
            "Monthly_ADX14"
        ]
    ]



# ============================================================
# 6-CONDITION BACKTEST
# ============================================================
#
# ORIGINAL 5 CONDITIONS:
# 1. Monthly RSI(5) > Monthly RSI SMA14
# 2. Weekly RSI(5) > Weekly RSI SMA14
# 3. Monthly ADX(14) >= 25
# 4. Daily RSI(5) <= 30
# 5. Weekly ADX(14) >= 25
#
# NEW CONDITION 6:
# After the RSI trigger, Daily Supertrend(10,1) must BECOME GREEN
# within 0 to 5 trading days, inclusive.
#
# The 6-condition entry/event date is the Supertrend GREEN-TURN date.
#
# Point-in-time NIFTY 500 membership is checked on the relevant event date.
#
# Returns are measured from the signal-day CLOSE:
# +5, +10, +20 and +60 trading days.
#
# No future prices are used to decide whether the signal occurs.
# ============================================================

import os
import time
from pathlib import Path

import pandas as pd
import yfinance as yf


YEARS = 10
END_DATE = pd.Timestamp.today().normalize()
START_DATE = END_DATE - pd.DateOffset(years=YEARS)

# Extra history for monthly/weekly indicator warm-up.
DOWNLOAD_START = START_DATE - pd.DateOffset(years=2)
DOWNLOAD_END = END_DATE + pd.Timedelta(days=1)

BATCH_SIZE = 25
RETRIES = 3
RETRY_SLEEP = 4

EVENTS_FILE = "nifty500_6condition_events.csv"
SUMMARY_FILE = "nifty500_6condition_summary.csv"
ERRORS_FILE = "nifty500_6condition_errors.csv"

# Also save the original 5-condition events so the improvement can be compared.
FIVE_EVENTS_FILE = "nifty500_5condition_events.csv"
FIVE_SUMMARY_FILE = "nifty500_5condition_summary.csv"


def extract_symbol_frame(data, symbol):
    """Return a single-symbol OHLC frame from yfinance batch output."""
    if data is None or data.empty:
        return pd.DataFrame()

    try:
        if isinstance(data.columns, pd.MultiIndex):
            # yfinance commonly returns (PriceField, Ticker)
            if symbol in data.columns.get_level_values(-1):
                frame = data.xs(symbol, axis=1, level=-1, drop_level=True).copy()
            elif symbol in data.columns.get_level_values(0):
                frame = data.xs(symbol, axis=1, level=0, drop_level=True).copy()
            else:
                return pd.DataFrame()
        else:
            frame = data.copy()

        required = ["Open", "High", "Low", "Close"]
        if not all(c in frame.columns for c in required):
            return pd.DataFrame()

        frame = frame[required].copy()
        frame.dropna(inplace=True)
        frame.index = clean_datetime_index(frame.index)
        frame = frame[~frame.index.duplicated(keep="last")]
        frame.sort_index(inplace=True)
        return frame
    except Exception:
        return pd.DataFrame()


def download_batch(symbols):
    """Download a batch with retries."""
    tickers = [f"{s}.NS" for s in symbols]

    for attempt in range(1, RETRIES + 1):
        try:
            print(f"Downloading batch ({len(symbols)} stocks), attempt {attempt}/{RETRIES}")

            data = yf.download(
                tickers=tickers,
                start=DOWNLOAD_START.strftime("%Y-%m-%d"),
                end=DOWNLOAD_END.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="column",
            )

            if data is not None and not data.empty:
                return data

        except Exception as error:
            print("  Batch error:", error)

        if attempt < RETRIES:
            time.sleep(RETRY_SLEEP)

    return pd.DataFrame()


def prepare_stock(data):
    if data is None or data.empty:
        return pd.DataFrame()

    data = data.copy()
    data.index = clean_datetime_index(data.index)
    data = data[~data.index.duplicated(keep="last")]
    data.sort_index(inplace=True)

    if len(data) < 200:
        return pd.DataFrame()

    data["Daily_RSI5"] = rsi_wilder(data["Close"], 5)

    (
        data["Daily_Supertrend"],
        data["Daily_Supertrend_Green"],
        data["Daily_Supertrend_Green_Turn"]
    ) = supertrend_green(
        data["High"],
        data["Low"],
        data["Close"],
        period=10,
        factor=1.0
    )

    weekly = build_weekly_data(data)
    monthly = build_monthly_data(data)

    weekly_for_daily = weekly.copy()
    weekly_for_daily.index = clean_datetime_index(
        weekly_for_daily.index + pd.Timedelta(days=1)
    )

    data = pd.merge_asof(
        data.sort_index(),
        weekly_for_daily.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward"
    )

    monthly_for_daily = monthly.copy()
    monthly_for_daily.index = clean_datetime_index(
        monthly_for_daily.index + pd.Timedelta(days=1)
    )

    data = pd.merge_asof(
        data.sort_index(),
        monthly_for_daily.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward"
    )

    data = data[
        (data.index >= START_DATE) &
        (data.index <= END_DATE)
    ].copy()

    return data


def add_forward_returns(data, event_date):
    """Return close-to-close forward returns using trading-day positions."""
    positions = data.index.get_loc(event_date)
    close = float(data.loc[event_date, "Close"])

    result = {
        "Signal_Close": close,
        "Return_5D_pct": None,
        "Return_10D_pct": None,
        "Return_20D_pct": None,
        "Return_60D_pct": None,
    }

    horizons = [5, 10, 20, 60]

    for h in horizons:
        pos = positions + h
        if pos < len(data.index):
            future_close = float(data.iloc[pos]["Close"])
            result[f"Return_{h}D_pct"] = (future_close / close - 1.0) * 100.0

    return result


def find_six_condition_events(data, symbol, membership):
    # Original five criteria. The RSI <= 30 day is the trigger day.
    c1 = data["Monthly_RSI5"] > data["Monthly_RSI_SMA14"]
    c2 = data["Weekly_RSI5"] > data["Weekly_RSI_SMA14"]
    c3 = data["Monthly_ADX14"] >= 25
    c4 = data["Daily_RSI5"] <= 30
    c5 = data["Weekly_ADX14"] >= 25

    original_setup = c1 & c2 & c3 & c4 & c5
    trigger_dates = data.index[original_setup]

    events = []
    used_signal_dates = set()

    for trigger_date in trigger_dates:
        if not is_member_on_date(membership, symbol, trigger_date):
            continue

        trigger_position = data.index.get_loc(trigger_date)

        # Only a fresh touch/entry into RSI <= 30 starts a new sequence.
        if trigger_position > 0:
            prev_rsi = data["Daily_RSI5"].iloc[trigger_position - 1]
            if pd.notna(prev_rsi) and prev_rsi <= 30:
                continue

        # No 0..7 trading-day restriction.
        # Search all later trading days until both requirements are satisfied.
        rsi_cross_date = None
        st_green_date = None

        for j in range(trigger_position, len(data.index)):
            cur_rsi = data["Daily_RSI5"].iloc[j]
            cur_ma = data["Daily_RSI5_SMA14"].iloc[j]

            # RSI must CROSS from at/below its SMA to above it.
            if j > 0 and pd.notna(cur_rsi) and pd.notna(cur_ma):
                prev_rsi = data["Daily_RSI5"].iloc[j - 1]
                prev_ma = data["Daily_RSI5_SMA14"].iloc[j - 1]
                if (
                    pd.notna(prev_rsi)
                    and pd.notna(prev_ma)
                    and prev_rsi <= prev_ma
                    and cur_rsi > cur_ma
                ):
                    if rsi_cross_date is None:
                        rsi_cross_date = data.index[j]

            # Supertrend only needs to be GREEN after the RSI trigger;
            # it does not have to turn green on the RSI-cross day.
            if data["Daily_Supertrend_Green"].iloc[j]:
                if st_green_date is None:
                    st_green_date = data.index[j]

        if rsi_cross_date is None or st_green_date is None:
            continue

        # Signal date is the day by which BOTH new requirements are satisfied.
        signal_date = max(rsi_cross_date, st_green_date)
        signal_key = pd.Timestamp(signal_date).strftime("%Y-%m-%d")

        if signal_key in used_signal_dates:
            continue

        if not is_member_on_date(membership, symbol, signal_date):
            continue

        used_signal_dates.add(signal_key)

        events.append({
            "Date": signal_key,
            "Stock": symbol.upper(),
            "RSI_30_or_Below_Date": pd.Timestamp(trigger_date).strftime("%Y-%m-%d"),
            "RSI_Cross_Above_SMA_Date": pd.Timestamp(rsi_cross_date).strftime("%Y-%m-%d"),
            "Supertrend_Green_Date": pd.Timestamp(st_green_date).strftime("%Y-%m-%d"),
            "Days_From_RSI_to_Signal": int(data.index.get_loc(signal_date) - trigger_position),
        })

    return events

def main():
    print("=" * 72)
    print("NIFTY 500 — 10-YEAR 6-CONDITION DATE-WISE STOCK LIST")
    print("=" * 72)
    print("Period:", START_DATE.strftime("%Y-%m-%d"), "to", END_DATE.strftime("%Y-%m-%d"))
    print("RSI(5) <= 30 -> RSI(5) crosses above SMA14 + Supertrend(10,1) GREEN")
    print("Window: No time limit after RSI <= 30 trigger")

    membership = load_membership()
    symbols = get_historical_symbols(membership)

    all_events = []
    errors = []
    total = len(symbols)

    for start in range(0, total, BATCH_SIZE):
        batch_symbols = symbols[start:start + BATCH_SIZE]
        print(f"\nBATCH {start + 1}-{min(start + BATCH_SIZE, total)} / {total}")
        raw = download_batch(batch_symbols)

        for symbol in batch_symbols:
            try:
                frame = extract_symbol_frame(raw, f"{symbol}.NS")
                if frame.empty:
                    frame = extract_symbol_frame(raw, symbol)

                if frame.empty:
                    errors.append({"Stock": symbol, "Error": "No Yahoo Finance data returned"})
                    continue

                data = prepare_stock(frame)
                if data.empty:
                    errors.append({"Stock": symbol, "Error": "Insufficient/invalid OHLC data"})
                    continue

                events = find_six_condition_events(data, symbol, membership)
                all_events.extend(events)
                print(f"  {symbol}: {len(events)} qualifying signals")

            except Exception as error:
                errors.append({"Stock": symbol, "Error": str(error)})
                print("  ERROR", symbol, ":", error)

        time.sleep(1)

    columns = [
        "Date", "Stock", "RSI_30_or_Below_Date",
        "RSI_Cross_Above_SMA_Date", "Supertrend_Green_Date",
        "Days_From_RSI_to_Signal"
    ]
    result = pd.DataFrame(all_events, columns=columns)
    if not result.empty:
        result["Date"] = pd.to_datetime(result["Date"])
        result = result.sort_values(["Date", "Stock"]).reset_index(drop=True)

    errors_df = pd.DataFrame(errors, columns=["Stock", "Error"])

    with pd.ExcelWriter("nifty500_6condition_datewise.xlsx", engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="Qualifying Stocks", index=False)
        pd.DataFrame({"Criteria": [
            "1. Monthly RSI(5) > Monthly RSI SMA(14)",
            "2. Weekly RSI(5) > Weekly RSI SMA(14)",
            "3. Monthly ADX(14) >= 25",
            "4. Daily RSI(5) <= 30 (trigger day)",
            "5. Weekly ADX(14) >= 25",
            "6. After RSI <= 30, Daily RSI(5) later crosses above Daily RSI SMA(14) — no time limit",
            "7. Daily Supertrend(10,1) is GREEN after the RSI <= 30 trigger — no time limit",
            "8. RSI cross and Supertrend GREEN do NOT need to occur on the same day",
            "9. Historical NIFTY 500 membership checked on trigger and signal dates",
        ]}).to_excel(writer, sheet_name="Criteria", index=False)
        errors_df.to_excel(writer, sheet_name="Data Errors", index=False)

        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(max_len + 2, 12), 45)

    print("\nCOMPLETE")
    print("Qualifying signals:", len(result))
    print("Data/errors:", len(errors_df))
    print("Excel: nifty500_6condition_datewise.xlsx")


if __name__ == "__main__":
    main()
