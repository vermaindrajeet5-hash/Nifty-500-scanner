import os
import io
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime

# ============================================================
# NIFTY 500 SCANNER
#
# Conditions:
# 1. Monthly RSI(5) > Monthly RSI(5) SMA(14)
# 2. Weekly RSI(5)  > Weekly RSI(5) SMA(14)
# 3. Monthly ADX(14) >= 25
# 4. Weekly ADX(14)  >= 25
# 5. Daily RSI(5) < 30
#
# Supertrend is NOT used.
# ============================================================

NIFTY500_URL = (
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
)


def rsi(series, period=5):
    """Wilder RSI."""
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


def adx(data, period=14):
    """Wilder-style ADX."""
    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where(
        (up_move > down_move) & (up_move > 0), 0.0
    )

    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0), 0.0
    )

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    plus_di = 100 * (
        plus_dm.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False
        ).mean() / atr
    )

    minus_di = 100 * (
        minus_dm.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False
        ).mean() / atr
    )

    denominator = plus_di + minus_di

    dx = 100 * (
        (plus_di - minus_di).abs() /
        denominator.replace(0, pd.NA)
    )

    return dx.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()


def get_nifty500():
    """Download the current Nifty 500 constituents."""
    response = requests.get(
        NIFTY500_URL,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    response.raise_for_status()

    df = pd.read_csv(io.BytesIO(response.content))

    symbol_column = None

    for column in df.columns:
        if str(column).strip().lower() == "symbol":
            symbol_column = column
            break

    if symbol_column is None:
        raise ValueError(
            f"Could not find Symbol column. Columns: {list(df.columns)}"
        )

    symbols = (
        df[symbol_column]
        .astype(str)
        .str.strip()
        .tolist()
    )

    return [symbol + ".NS" for symbol in symbols]


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


def scan_stock(symbol, data):
    if data is None or data.empty:
        return None

    data = data.dropna(subset=["Open", "High", "Low", "Close"])

    if len(data) < 300:
        return None

    weekly = make_weekly(data)
    monthly = make_monthly(data)

    # Use completed weekly/monthly candles.
    # The current unfinished week/month is removed.
    if len(weekly) > 1:
        weekly = weekly.iloc[:-1]

    if len(monthly) > 1:
        monthly = monthly.iloc[:-1]

    # Need enough history for indicators.
    if len(weekly) < 50 or len(monthly) < 50:
        return None

    # -------------------------
    # DAILY
    # -------------------------
    daily_rsi = rsi(data["Close"], 5).iloc[-1]

    # -------------------------
    # WEEKLY
    # -------------------------
    weekly_rsi_series = rsi(weekly["Close"], 5)
    weekly_rsi = weekly_rsi_series.iloc[-1]
    weekly_rsi_ma = weekly_rsi_series.rolling(14).mean().iloc[-1]
    weekly_adx = adx(weekly, 14).iloc[-1]

    # -------------------------
    # MONTHLY
    # -------------------------
    monthly_rsi_series = rsi(monthly["Close"], 5)
    monthly_rsi = monthly_rsi_series.iloc[-1]
    monthly_rsi_ma = monthly_rsi_series.rolling(14).mean().iloc[-1]
    monthly_adx = adx(monthly, 14).iloc[-1]

    values = [
        daily_rsi,
        weekly_rsi,
        weekly_rsi_ma,
        weekly_adx,
        monthly_rsi,
        monthly_rsi_ma,
        monthly_adx
    ]

    if any(pd.isna(x) for x in values):
        return None

    # ========================================================
    # YOUR 5 CONDITIONS
    # ========================================================

    condition_1 = monthly_rsi > monthly_rsi_ma
    condition_2 = weekly_rsi > weekly_rsi_ma
    condition_3 = monthly_adx >= 25
    condition_4 = weekly_adx >= 25
    condition_5 = daily_rsi < 30

    if all([
        condition_1,
        condition_2,
        condition_3,
        condition_4,
        condition_5
    ]):
        return {
            "symbol": symbol.replace(".NS", ""),
            "daily_rsi": round(float(daily_rsi), 2),
            "weekly_rsi": round(float(weekly_rsi), 2),
            "weekly_rsi_ma": round(float(weekly_rsi_ma), 2),
            "weekly_adx": round(float(weekly_adx), 2),
            "monthly_rsi": round(float(monthly_rsi), 2),
            "monthly_rsi_ma": round(float(monthly_rsi_ma), 2),
            "monthly_adx": round(float(monthly_adx), 2),
        }

    return None


def main():

    print("=" * 60)
    print("NIFTY 500 SCANNER")
    print("=" * 60)

    print("\nDownloading Nifty 500 list...")

    symbols = get_nifty500()

    print(f"Found {len(symbols)} stocks.")

    print("\nDownloading market data...")

    # Download in chunks to reduce the chance of Yahoo throttling.
    results = []

    chunk_size = 50

    for start in range(0, len(symbols), chunk_size):

        chunk = symbols[start:start + chunk_size]

        print(
            f"Processing stocks {start + 1} "
            f"to {min(start + chunk_size, len(symbols))}..."
        )

        try:
            data = yf.download(
                chunk,
                period="5y",
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False
            )
        except Exception as e:
            print("Download error:", e)
            continue

        for symbol in chunk:

            try:

                if len(chunk) == 1:
                    stock_data = data.copy()
                else:
                    if symbol not in data.columns.get_level_values(0):
                        continue

                    stock_data = data[symbol].copy()

                if stock_data.empty:
                    continue

                result = scan_stock(symbol, stock_data)

                if result:
                    results.append(result)

            except Exception as e:
                print(f"Error processing {symbol}: {e}")

    print("\n")
    print("=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)

    if not results:
        print("\nNo stocks matched all 5 conditions.")
        return

    result_df = pd.DataFrame(results)

    print(
        f"\n{len(result_df)} stock(s) matched all 5 conditions:\n"
    )

    print(result_df.to_string(index=False))

    result_df.to_csv("signals.csv", index=False)

    print("\nSaved results to signals.csv")


if __name__ == "__main__":
    main()
