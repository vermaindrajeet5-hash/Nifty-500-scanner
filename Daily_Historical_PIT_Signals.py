import time
import requests
import pandas as pd
import yfinance as yf
from io import StringIO


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
    "main/index_history/data/"
    "index_membership_history.csv"
)


# ============================================================
# RSI - WILDER
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
# ADX - WILDER
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

    membership.sort_values(
        ["valid_from", "symbol"],
        inplace=True
    )

    print(
        "Historical membership records:",
        len(membership)
    )

    return membership


# ============================================================
# CHECK WHETHER STOCK WAS IN NIFTY 500 ON THAT DATE
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
       
