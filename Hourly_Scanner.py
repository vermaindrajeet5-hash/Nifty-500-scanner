import os
import time
import smtplib
import requests
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# ============================================================
# SETTINGS
# ============================================================

IST = ZoneInfo("Asia/Kolkata")

NIFTY500_URL = (
    "https://www.niftyindices.com/IndexConstituent/"
    "ind_nifty500list.csv"
)


# ============================================================
# RSI - WILDER
# ============================================================

def rsi(series, period=5):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


# ============================================================
# ADX - WILDER
# ============================================================

def adx(data, period=14):

    high = data["High"]
    low = data["Low"]
    close = data["Close"]

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
        index=data.index
    )

    minus_dm = pd.Series(
        0.0,
        index=data.index
    )

    plus_condition = (
        (up_move > down_move)
        &
        (up_move > 0)
    )

    minus_condition = (
        (down_move > up_move)
        &
        (down_move > 0)
    )

    plus_dm.loc[plus_condition] = (
        up_move.loc[plus_condition]
    )

    minus_dm.loc[minus_condition] = (
        down_move.loc[minus_condition]
    )

    atr = true_range.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    plus_di = (
        100
        *
        plus_dm.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False
        ).mean()
        / atr
    )

    minus_di = (
        100
        *
        minus_dm.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False
        ).mean()
        / atr
    )

    denominator = plus_di + minus_di

    dx = (
        100
        *
        (plus_di - minus_di).abs()
        / denominator
    )

    adx_value = dx.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    return adx_value


# ============================================================
# GET NIFTY 500 SYMBOLS
# ============================================================

def get_nifty500_symbols():

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/120 Safari/537.36"
        )
    }

    response = requests.get(
        NIFTY500_URL,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    df = pd.read_csv(
        pd.io.common.StringIO(response.text)
    )

    symbols = []

    for symbol in df["Symbol"].dropna():

        symbols.append(
            str(symbol).strip() + ".NS"
        )

    return symbols


# ============================================================
# MAKE WEEKLY DATA
# ============================================================

def make_weekly(data):

    return data.resample(
        "W-FRI"
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()


# ============================================================
# MAKE MONTHLY DATA
# ============================================================

def make_monthly(data):

    return data.resample(
        "ME"
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()


# ============================================================
# DOWNLOAD AND CLEAN DATA
# ============================================================

def download_data(symbol, period, interval):

    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if data.empty:
        return None

    if isinstance(
        data.columns,
        pd.MultiIndex
    ):
        data.columns = (
            data.columns
            .get_level_values(0)
        )

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
