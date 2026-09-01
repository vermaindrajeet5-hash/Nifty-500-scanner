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
# NIFTY 500 HOURLY SCANNER
# ============================================================
#
# Runs after each completed 1-hour candle.
#
# Conditions:
#
# 1. Previous completed 1-hour RSI(5) < 30
#
# Higher timeframe conditions:
# 2. Monthly RSI(5) > Monthly RSI(5) SMA(14)
# 3. Weekly RSI(5) > Weekly RSI(5) SMA(14)
# 4. Daily RSI(5) > Daily RSI(5) SMA(14)
# 5. Monthly ADX(14) >= 25
# 6. Weekly ADX(14) >= 25
# 7. Daily ADX(14) >= 25
#
# Supertrend is NOT used.
# Telegram is NOT used.
# Results are sent by Gmail.
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

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat(
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
        (up_move > down_move) &
        (up_move > 0)
    )

    minus_condition = (
        (down_move > up_move) &
