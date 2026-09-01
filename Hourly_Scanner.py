import os
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


# ============================================================
# NIFTY 500 HOURLY SCANNER
# ============================================================
#
# Runs after each completed 1-hour candle.
#
# Hourly RSI condition:
# Previous completed 1-hour RSI(5) < 30
#
# Higher timeframe conditions:
# 1. Monthly RSI(5) > Monthly RSI(5) SMA(14)
# 2. Weekly RSI(5) > Weekly RSI(5) SMA(14)
# 3. Daily RSI(5) > Daily RSI(5) SMA(14)
# 4. Monthly ADX(14) >= 25
# 5. Weekly ADX(14) >= 25
# 6. Daily ADX(14) >= 25
#
# Supertrend is NOT used.
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

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

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

    plus_dm[
        (up_move > down_move) &
        (up_move > 0)
    ] = up_move[
        (up_move > down_move) &
        (up_move > 0)
    ]

    minus_dm[
        (down_move > up_move) &
        (down_move > 0)
    ] = down_move[
        (down_move > up_move) &
        (down_move > 0)
    ]

    atr = tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    plus_di = (
        100 *
        plus_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        ).mean() /
        atr
    )

    minus_di = (
        100 *
        minus_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        ).mean() /
        atr
    )

    denominator = plus_di + minus_di

    dx = (
        100 *
        (plus_di - minus_di).abs() /
        denominator
    )

    adx_value = dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    return adx_value


# ============================================================
# GET NIFTY 500 SYMBOLS
# ============================================================

def get_nifty500_symbols():

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120 Safari/537.36"
        )
    }

    response = requests.get(
        NIFTY500_URL,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    df = pd.read_csv(
    pd.io.common.StringIO(response.text),
    sep=",",
    engine="python"
    )

    symbols = []

    for symbol in df["Symbol"].dropna():
        symbols.append(str(symbol).strip() + ".NS")

    return symbols


# ============================================================
# RESAMPLE DATA
# ============================================================

def make_weekly(data):

    return data.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()


def make_monthly(data):

    return data.resample("ME").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()


# ============================================================
# CHECK ONE STOCK
# ============================================================

def check_stock(symbol):

    try:

        # ----------------------------------------------------
        # DAILY DATA
        # ----------------------------------------------------

        daily = yf.download(
            symbol,
            period="3y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if daily.empty:
            return None

        if isinstance(daily.columns, pd.MultiIndex):
            daily.columns = daily.columns.get_level_values(0)

        daily = daily[
            ["Open", "High", "Low", "Close", "Volume"]
        ].dropna()

        if len(daily) < 100:
            return None

        # ----------------------------------------------------
        # WEEKLY / MONTHLY
        # ----------------------------------------------------

        weekly = make_weekly(daily)
        monthly = make_monthly(daily)

        if len(weekly) < 30 or len(monthly) < 30:
            return None

        # ----------------------------------------------------
        # DAILY CONDITIONS
        # ----------------------------------------------------

        daily_rsi = rsi(daily["Close"], 5)
        daily_rsi_sma = daily_rsi.rolling(14).mean()
        daily_adx = adx(daily, 14)

        d_rsi = daily_rsi.iloc[-1]
        d_rsi_sma = daily_rsi_sma.iloc[-1]
        d_adx = daily_adx.iloc[-1]

        if pd.isna(d_rsi) or pd.isna(d_rsi_sma) or pd.isna(d_adx):
            return None

        daily_condition = (
            d_rsi > d_rsi_sma and
            d_adx >= 25
        )

        if not daily_condition:
            return None

        # ----------------------------------------------------
        # WEEKLY CONDITIONS
        # ----------------------------------------------------

        weekly_rsi = rsi(weekly["Close"], 5)
        weekly_rsi_sma = weekly_rsi.rolling(14).mean()
        weekly_adx = adx(weekly, 14)

        w_rsi = weekly_rsi.iloc[-1]
        w_rsi_sma = weekly_rsi_sma.iloc[-1]
        w_adx = weekly_adx.iloc[-1]

        if pd.isna(w_rsi) or pd.isna(w_rsi_sma) or pd.isna(w_adx):
            return None

        weekly_condition = (
            w_rsi > w_rsi_sma and
            w_adx >= 25
        )

        if not weekly_condition:
            return None

        # ----------------------------------------------------
        # MONTHLY CONDITIONS
        # ----------------------------------------------------

        monthly_rsi = rsi(monthly["Close"], 5)
        monthly_rsi_sma = monthly_rsi.rolling(14).mean()
        monthly_adx = adx(monthly, 14)

        m_rsi = monthly_rsi.iloc[-1]
        m_rsi_sma = monthly_rsi_sma.iloc[-1]
        m_adx = monthly_adx.iloc[-1]

        if pd.isna(m_rsi) or pd.isna(m_rsi_sma) or pd.isna(m_adx):
            return None

        monthly_condition = (
            m_rsi > m_rsi_sma and
            m_adx >= 25
        )

        if not monthly_condition:
            return None

        # ----------------------------------------------------
        # 1-HOUR DATA
        # ----------------------------------------------------

        hourly = yf.download(
            symbol,
            period="60d",
            interval="1h",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if hourly.empty:
            return None

        if isinstance(hourly.columns, pd.MultiIndex):
            hourly.columns = hourly.columns.get_level_values(0)

        hourly = hourly[
            ["Open", "High", "Low", "Close", "Volume"]
        ].dropna()

        if len(hourly) < 30:
            return None

        # ----------------------------------------------------
        # CONVERT TO IST
        # ----------------------------------------------------

        if hourly.index.tz is None:
            hourly.index = hourly.index.tz_localize("UTC")

        hourly.index = hourly.index.tz_convert(IST)

        # ----------------------------------------------------
        # IMPORTANT:
        # ONLY USE COMPLETED HOURLY CANDLES
        # ----------------------------------------------------

        now_ist = datetime.now(IST)

        completed = hourly[
            hourly.index + timedelta(hours=1) <= now_ist
        ].copy()

        if len(completed) < 10:
            return None

        hourly_rsi = rsi(
            completed["Close"],
            5
        )

        # Previous completed hourly candle
        # is deliberately used here.
        previous_hour_rsi = hourly_rsi.iloc[-1]

        if pd.isna(previous_hour_rsi):
            return None

        # ----------------------------------------------------
        # FINAL HOURLY CONDITION
        # ----------------------------------------------------

        if previous_hour_rsi >= 30:
            return None

        # ----------------------------------------------------
        # MATCH
        # ----------------------------------------------------

        return {
            "Stock": symbol.replace(".NS", ""),
            "Monthly RSI(5)": round(float(m_rsi), 2),
            "Monthly RSI SMA(14)": round(float(m_rsi_sma), 2),
            "Monthly ADX(14)": round(float(m_adx), 2),
            "Weekly RSI(5)": round(float(w_rsi), 2),
            "Weekly RSI SMA(14)": round(float(w_rsi_sma), 2),
            "Weekly ADX(14)": round(float(w_adx), 2),
            "Daily RSI(5)": round(float(d_rsi), 2),
            "Daily RSI SMA(14)": round(float(d_rsi_sma), 2),
            "Daily ADX(14)": round(float(d_adx), 2),
            "Previous 1H RSI(5)": round(float(previous_hour_rsi), 2)
        }

    except Exception as e:

        print(
            f"Error processing {symbol}: {e}"
        )

        return None


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    token = os.environ.get(
        "TELEGRAM_BOT_TOKEN"
    )

    chat_id = os.environ.get(
        "TELEGRAM_CHAT_ID"
    )

    if not token or not chat_id:

        print(
            "Telegram settings not found"
        )

        return

    url = (
        f"https://api.telegram.org/"
        f"bot{token}/sendMessage"
    )

    try:

    response = requests.post(
    url,
    data={
        "chat_id": chat_id,
        "text": message
    },
    timeout=30
)

print("Telegram HTTP status:", response.status_code)
print("Telegram response:", response.text)

response.raise_for_status()

    except Exception as e:

        print(
            f"Telegram error: {e}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NIFTY 500 HOURLY SCANNER")
    print("=" * 60)

    now = datetime.now(IST)

    print(
        "Scan time:",
        now.strftime("%Y-%m-%d %H:%M:%S IST")
    )

    print(
        "\nWaiting 10 seconds for the "
        "hourly candle data to update..."
    )

    time.sleep(10)

    print("\nGetting NIFTY 500 stocks...")

    symbols = get_nifty500_symbols()

    print(
        f"Found {len(symbols)} NIFTY 500 stocks."
    )

    results = []

    for i, symbol in enumerate(symbols, 1):

        print(
            f"[{i}/{len(symbols)}] "
            f"Checking {symbol}"
        )

        result = check_stock(symbol)

        if result:
            results.append(result)

            print(
                f"  MATCH: {symbol}"
            )

    print("\n" + "=" * 60)
    print("HOURLY SCAN COMPLETE")
    print("=" * 60)

    if not results:

        print(
            "\nNo stocks matched all conditions."
        )

        send_telegram(
            "NIFTY 500 HOURLY SCANNER\n\n"
            "No stocks matched all conditions."
        )

        return

    result_df = pd.DataFrame(results)

    print(
        f"\n{len(result_df)} stock(s) "
        "matched all conditions:\n"
    )

    print(
        result_df.to_string(index=False)
    )

    result_df.to_csv(
        "hourly_signals.csv",
        index=False
    )

    message = (
        "NIFTY 500 HOURLY SCANNER RESULTS\n\n"
        + result_df.to_string(index=False)
    )

    send_telegram(message)

    print(
        "\nSaved results to hourly_signals.csv"
    )


if __name__ == "__main__":
    main()
