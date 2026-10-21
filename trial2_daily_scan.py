#!/usr/bin/env python3
"""
Trial 2 NIFTY 500 daily signal scanner.

Runs after the Indian market close. It scans the current NIFTY 500 constituents
and emails an Excel workbook plus the list of qualifying stocks.

Conditions preserved from the Trial 2 Python source:
1. Previous completed monthly RSI(5) > SMA(14) of monthly RSI(5)
2. Previous completed monthly ADX(14) rising, with increase > 0 and < 1
3. Latest completed daily Supertrend(10, 1) is green
4. Latest completed daily Low <= EMA(20) and Close > EMA(20)
5. Latest completed daily RSI(5) > SMA(14) of daily RSI(5)
6. Latest completed daily MACD(12,26,9) line > signal
7. Previous completed weekly RSI(5) > SMA(14) of weekly RSI(5)

This is a technical screening tool, not investment advice.
"""

from io import BytesIO
import os
import smtplib
import ssl
import time
from email.message import EmailMessage

import numpy as np
import pandas as pd
import requests
import yfinance as yf

NIFTY500_CSV = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}
LOOKBACK = "3y"


def get_universe():
    response = requests.get(NIFTY500_CSV, headers=HEADERS, timeout=30)
    response.raise_for_status()
    constituents = pd.read_csv(BytesIO(response.content))
    symbol_col = next((c for c in constituents.columns if c.strip().lower() == "symbol"), None)
    if not symbol_col:
        raise ValueError("Could not find the Symbol column in the NIFTY 500 constituent CSV.")
    symbols = sorted(set(str(s).strip().upper() + ".NS" for s in constituents[symbol_col].dropna() if str(s).strip()))
    if not symbols:
        raise ValueError("The NIFTY 500 constituent list was empty.")
    return symbols


def rma(series, length):
    # Wilder's smoothing, matching the standard Wilder RSI/ADX calculation.
    return series.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def rsi(close, length=5):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.where(avg_loss != 0, 100)
    result = result.where(~((avg_gain == 0) & (avg_loss == 0)), 50)
    return result


def adx(data, length=14):
    high, low, close = data["High"], data["Low"], data["Close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=data.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=data.index)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = rma(tr, length)
    plus_di = 100 * rma(plus_dm, length) / atr.replace(0, np.nan)
    minus_di = 100 * rma(minus_dm, length) / atr.replace(0, np.nan)
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    return rma(dx, length)


def supertrend_green(data, length=10, multiplier=1.0):
    high, low, close = data["High"], data["Low"], data["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = rma(tr, length)
    hl2 = (high + low) / 2
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr
    upper = upper_basic.copy()
    lower = lower_basic.copy()
    direction = pd.Series(index=data.index, dtype="float64")

    for i in range(len(data)):
        if pd.isna(atr.iloc[i]):
            continue
        if i == 0 or pd.isna(direction.iloc[i - 1]):
            direction.iloc[i] = 1 if close.iloc[i] >= hl2.iloc[i] else -1
            continue
        prev_u, prev_l = upper.iloc[i - 1], lower.iloc[i - 1]
        # Standard Supertrend final-band rules.
        if pd.isna(prev_u) or upper_basic.iloc[i] < prev_u or close.iloc[i - 1] > prev_u:
            upper.iloc[i] = upper_basic.iloc[i]
        else:
            upper.iloc[i] = prev_u
        if pd.isna(prev_l) or lower_basic.iloc[i] > prev_l or close.iloc[i - 1] < prev_l:
            lower.iloc[i] = lower_basic.iloc[i]
        else:
            lower.iloc[i] = prev_l

        if direction.iloc[i - 1] == 1:
            direction.iloc[i] = -1 if close.iloc[i] < lower.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if close.iloc[i] > upper.iloc[i] else -1
    return direction


def resample_ohlc(data, rule):
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    out = data.resample(rule).agg(agg).dropna(subset=["Open", "High", "Low", "Close"])
    return out


def macd(data):
    close = data["Close"]
    line = close.ewm(span=12, adjust=False, min_periods=12).mean() - close.ewm(span=26, adjust=False, min_periods=26).mean()
    signal = line.ewm(span=9, adjust=False, min_periods=9).mean()
    return line, signal


def last_completed_period(frame, period):
    """Return last fully completed week/month, excluding a currently forming period."""
    if frame.empty:
        return frame
    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    today = now_ist.tz_localize(None).normalize()
    if period == "W":
        # Only include the current week's Friday-labelled candle after the
        # Friday market close. On Mon-Thu (or before Friday close), use prior week.
        weekday = today.weekday()
        after_friday_close = weekday == 4 and (now_ist.hour, now_ist.minute) >= (15, 45)
        if weekday < 4 or (weekday == 4 and not after_friday_close):
            last_completed_friday = today - pd.Timedelta(days=weekday + 3)
        else:
            last_completed_friday = today
        frame = frame[frame.index.normalize() <= last_completed_friday]
    elif period == "M":
        current_month_start = today.replace(day=1)
        frame = frame[frame.index < current_month_start]
    return frame


def scan_symbol(symbol):
    try:
        raw = yf.download(
            symbol, period=LOOKBACK, interval="1d", auto_adjust=False,
            progress=False, threads=False, group_by="column"
        )
        if raw is None or raw.empty:
            return None, f"No price data for {symbol}"

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        needed = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in raw.columns for c in needed):
            return None, f"Missing OHLCV columns for {symbol}"
        daily = raw[needed].copy().dropna(subset=["Open", "High", "Low", "Close"])
        daily.index = pd.to_datetime(daily.index).tz_localize(None)
        daily = daily[~daily.index.duplicated(keep="last")].sort_index()
        if len(daily) < 100:
            return None, f"Not enough history for {symbol}"

        # Indicators on daily candles.
        daily_ema20 = daily["Close"].ewm(span=20, adjust=False, min_periods=20).mean()
        daily_rsi5 = rsi(daily["Close"], 5)
        daily_rsi_sma14 = daily_rsi5.rolling(14, min_periods=14).mean()
        daily_macd, daily_macd_signal = macd(daily)
        daily_st = supertrend_green(daily, 10, 1.0)

        # Before 15:45 IST, exclude today's potentially unfinished candle.
        # The scheduled 17:00 IST run can use today's completed candle if available.
        now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
        today_ist = now_ist.tz_localize(None).normalize()
        market_close_passed = (now_ist.hour, now_ist.minute) >= (15, 45)
        if daily.index[-1].normalize() >= today_ist and not market_close_passed:
            daily = daily[daily.index.normalize() < today_ist]
        if daily.empty:
            return None, f"No completed daily candle for {symbol}"
        idx = daily.index[-1]
        close = float(daily.loc[idx, "Close"])
        low = float(daily.loc[idx, "Low"])
        ema20 = float(daily_ema20.loc[idx])
        drsi = float(daily_rsi5.loc[idx])
        drsi_sma = float(daily_rsi_sma14.loc[idx])
        dmacd = float(daily_macd.loc[idx])
        dsignal = float(daily_macd_signal.loc[idx])
        st_green = bool(daily_st.loc[idx] == 1)

        # Monthly conditions: use only completed monthly candles.
        monthly = resample_ohlc(daily, "ME")
        monthly = last_completed_period(monthly, "M")
        if len(monthly) < 20:
            return None, f"Not enough monthly history for {symbol}"
        mrsi = rsi(monthly["Close"], 5)
        mrsi_sma = mrsi.rolling(14, min_periods=14).mean()
        madx = adx(monthly, 14)
        if len(monthly) < 2:
            return None, f"Not enough completed monthly candles for {symbol}"
        m_rsi = mrsi.iloc[-1]
        m_rsi_sma = mrsi_sma.iloc[-1]
        m_adx = madx.iloc[-1]
        m_adx_prev = madx.iloc[-2]
        adx_change = m_adx - m_adx_prev

        # Weekly conditions: previous completed weekly candle only.
        weekly = resample_ohlc(daily, "W-FRI")
        weekly = last_completed_period(weekly, "W")
        if len(weekly) < 20:
            return None, f"Not enough completed weekly history for {symbol}"
        wrsi = rsi(weekly["Close"], 5)
        wrsi_sma = wrsi.rolling(14, min_periods=14).mean()
        w_rsi = wrsi.iloc[-1]
        w_rsi_sma = wrsi_sma.iloc[-1]

        checks = {
            "Monthly RSI(5) > SMA(14)": bool(m_rsi > m_rsi_sma),
            "Monthly ADX(14) rising by < 1": bool(adx_change > 0 and adx_change < 1),
            "Daily Supertrend(10,1) green": st_green,
            "Daily Low <= EMA20 and Close > EMA20": bool(low <= ema20 and close > ema20),
            "Daily RSI(5) > SMA(14)": bool(drsi > drsi_sma),
            "Daily MACD(12,26,9) bullish": bool(dmacd > dsignal),
            "Weekly RSI(5) > SMA(14)": bool(w_rsi > w_rsi_sma),
        }
        row = {
            "Symbol": symbol.replace(".NS", ""),
            "Signal Date": idx.strftime("%Y-%m-%d"),
            "Close": round(close, 2),
            "EMA20": round(ema20, 2),
            "Daily RSI(5)": round(drsi, 2),
            "Daily RSI SMA(14)": round(drsi_sma, 2),
            "Daily MACD": round(dmacd, 4),
            "Daily MACD Signal": round(dsignal, 4),
            "Monthly RSI(5)": round(float(m_rsi), 2),
            "Monthly RSI SMA(14)": round(float(m_rsi_sma), 2),
            "Monthly ADX(14)": round(float(m_adx), 2),
            "Monthly ADX Change": round(float(adx_change), 4),
            "Weekly RSI(5)": round(float(w_rsi), 2),
            "Weekly RSI SMA(14)": round(float(w_rsi_sma), 2),
            "Supertrend Green": st_green,
        }
        if all(checks.values()):
            return row, None
        return None, None
    except Exception as exc:
        return None, f"{symbol}: {type(exc).__name__}: {exc}"


def build_workbook(qualified, errors, scanned_count):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(qualified).to_excel(writer, sheet_name="Qualified Stocks", index=False)
        pd.DataFrame({
            "Run Date (IST)": [pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d %H:%M:%S")],
            "Stocks scanned": [scanned_count],
            "Qualified stocks": [len(qualified)],
        }).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame({"Errors / skipped symbols": errors}).to_excel(writer, sheet_name="Scan Notes", index=False)
    output.seek(0)
    return output.read()


def send_email(qualified, workbook_bytes, errors, scanned_count):
    username = os.environ.get("GMAIL_USERNAME")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    recipient = os.environ.get("GMAIL_TO")
    if not all([username, password, recipient]):
        raise RuntimeError("Missing one or more GitHub secrets: GMAIL_USERNAME, GMAIL_APP_PASSWORD, GMAIL_TO")

    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    run_date = now_ist.strftime("%d %b %Y")
    msg = EmailMessage()
    msg["Subject"] = f"Trial 2 NIFTY 500 Scanner — {len(qualified)} qualified — {run_date}"
    msg["From"] = username
    msg["To"] = recipient

    if qualified:
        lines = ["Qualified stocks (all Trial 2 conditions passed):", ""]
        for row in qualified:
            lines.append(f"- {row['Symbol']} | Close: {row['Close']} | Signal date: {row['Signal Date']}")
        stock_list = "\n".join(lines)
    else:
        stock_list = "No stocks qualified today. The attached Excel file contains the empty results sheet and scan notes."

    msg.set_content(
        f"Trial 2 NIFTY 500 daily scanner\nRun date: {run_date}\n"
        f"Stocks scanned: {scanned_count}\nQualified: {len(qualified)}\n\n"
        f"{stock_list}\n\n"
        f"Skipped/error symbols: {len(errors)}\n"
        "The attached workbook contains the full results and scan notes.\n\n"
        "This is a technical screen, not investment advice."
    )
    msg.add_attachment(
        workbook_bytes,
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"trial_2_nifty500_signals_{now_ist.strftime('%Y%m%d')}.xlsx",
    )
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context, timeout=30) as server:
        server.login(username, password)
        server.send_message(msg)


def main():
    print("Loading current NIFTY 500 constituents...")
    symbols = get_universe()
    qualified, errors = [], []
    for i, symbol in enumerate(symbols, start=1):
        print(f"[{i}/{len(symbols)}] Scanning {symbol}", flush=True)
        row, error = scan_symbol(symbol)
        if row:
            qualified.append(row)
            print(f"  QUALIFIED: {row['Symbol']}", flush=True)
        if error:
            errors.append(error)
        time.sleep(0.08)

    qualified.sort(key=lambda r: r["Symbol"])
    workbook = build_workbook(qualified, errors, len(symbols))
    send_email(qualified, workbook, errors, len(symbols))
    print(f"Completed. {len(qualified)} stocks qualified. Email sent.", flush=True)


if __name__ == "__main__":
    main()
