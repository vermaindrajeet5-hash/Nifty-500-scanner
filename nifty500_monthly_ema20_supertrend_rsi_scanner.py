#!/usr/bin/env python3
"""Backdated monthly NIFTY 500 scanner.

For each completed month in the last 10 years, test the stocks that were
NIFTY 500 constituents on that month's final trading date.

Rules on that completed monthly candle:
  1) Monthly Close > monthly EMA(20)
  2) Monthly Supertrend(20, 1) is green (bullish; Close > Supertrend line)
  3) Monthly RSI(5) > SMA(14) of monthly RSI(5)

The scan is historical/backdated. The GitHub workflow can also be scheduled
on the first calendar day of each month to refresh the previous month's result.
"""
from __future__ import annotations

import os
import sys
import time
import smtplib
import traceback
from pathlib import Path
from email.message import EmailMessage
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf

END_DATE = pd.Timestamp.now().normalize()
START_DATE = END_DATE - pd.DateOffset(years=10)
ST_PERIOD = 20
ST_MULTIPLIER = 1.0
EMA_PERIOD = 20
RSI_PERIOD = 5
RSI_SMA_PERIOD = 14
MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/aditya-jha/nse-historical-membership/"
    "main/index_history/data/index_membership_history.csv"
)
OUTPUT_FILE = "nifty500_monthly_ema20_supertrend_rsi5_10y.xlsx"
USER_AGENT = "Mozilla/5.0 (compatible; Nifty500HistoricalScanner/1.0)"


def norm(value):
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def find_column(columns, candidates):
    mapping = {norm(c): c for c in columns}
    for candidate in candidates:
        if norm(candidate) in mapping:
            return mapping[norm(candidate)]
    for key, original in mapping.items():
        if any(norm(c) in key for c in candidates):
            return original
    return None


def load_membership():
    response = requests.get(MEMBERSHIP_URL, headers={"User-Agent": USER_AGENT}, timeout=60)
    response.raise_for_status()
    raw = pd.read_csv(BytesIO(response.content))
    if raw.empty:
        raise RuntimeError("Historical membership CSV is empty.")

    symbol_col = find_column(raw.columns, ["symbol", "ticker", "security"])
    start_col = find_column(raw.columns, ["valid_from", "start_date", "from_date", "date_from"])
    end_col = find_column(raw.columns, ["valid_to", "end_date", "to_date", "date_to"])
    index_col = find_column(raw.columns, ["index_name", "index", "indexname"])

    if not symbol_col or not start_col or not end_col:
        raise RuntimeError(f"Unrecognized membership CSV columns: {list(raw.columns)}")

    if index_col:
        idx = raw[index_col].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
        raw = raw[idx.str.contains("NIFTY500", na=False)].copy()
        if raw.empty:
            raise RuntimeError("No NIFTY 500 records found in membership CSV.")

    m = pd.DataFrame({
        "symbol": raw[symbol_col].astype(str).str.strip().str.upper(),
        "from": pd.to_datetime(raw[start_col], errors="coerce"),
        "to": pd.to_datetime(raw[end_col], errors="coerce"),
    })
    m = m[m["symbol"].ne("") & m["symbol"].ne("NAN") & m["from"].notna()].copy()
    # The referenced dataset represents membership as [from, to) intervals.
    m["from"] = m["from"].dt.normalize()
    m["to"] = m["to"].dt.normalize().fillna(pd.Timestamp.max.normalize())
    return m


def wilder(series, period):
    s = series.astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    valid_positions = np.flatnonzero(s.notna().to_numpy())
    if len(valid_positions) < period:
        return out
    seed_pos = valid_positions[period - 1]
    seed_values = s.iloc[:seed_pos + 1].dropna().iloc[:period]
    out.iloc[seed_pos] = seed_values.mean()
    for i in range(seed_pos + 1, len(s)):
        prev = out.iloc[i - 1]
        cur = s.iloc[i]
        if pd.notna(prev):
            out.iloc[i] = prev if pd.isna(cur) else ((prev * (period - 1) + cur) / period)
    return out


def rsi_wilder(close, period):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = wilder(gain, period)
    avg_loss = wilder(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.where(avg_loss.ne(0), 100.0)
    result = result.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return result


def supertrend(monthly, period=20, multiplier=1.0):
    high, low, close = monthly["High"], monthly["Low"], monthly["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = wilder(tr, period)
    midpoint = (high + low) / 2
    basic_upper = midpoint + multiplier * atr
    basic_lower = midpoint - multiplier * atr
    final_upper = pd.Series(np.nan, index=monthly.index, dtype=float)
    final_lower = pd.Series(np.nan, index=monthly.index, dtype=float)
    st = pd.Series(np.nan, index=monthly.index, dtype=float)
    bullish = pd.Series(False, index=monthly.index, dtype=bool)

    first = atr.first_valid_index()
    if first is None:
        return st, bullish

    start = monthly.index.get_loc(first)
    final_upper.iloc[start] = basic_upper.iloc[start]
    final_lower.iloc[start] = basic_lower.iloc[start]
    st.iloc[start] = final_upper.iloc[start]
    bullish.iloc[start] = False

    for i in range(start + 1, len(monthly)):
        prev_u, prev_l = final_upper.iloc[i - 1], final_lower.iloc[i - 1]
        bu, bl = basic_upper.iloc[i], basic_lower.iloc[i]
        if pd.isna(bu) or pd.isna(bl):
            continue
        final_upper.iloc[i] = bu if (bu < prev_u or close.iloc[i - 1] > prev_u) else prev_u
        final_lower.iloc[i] = bl if (bl > prev_l or close.iloc[i - 1] < prev_l) else prev_l

        if st.iloc[i - 1] == prev_u:
            if close.iloc[i] <= final_upper.iloc[i]:
                st.iloc[i] = final_upper.iloc[i]
                bullish.iloc[i] = False
            else:
                st.iloc[i] = final_lower.iloc[i]
                bullish.iloc[i] = True
        else:
            if close.iloc[i] >= final_lower.iloc[i]:
                st.iloc[i] = final_lower.iloc[i]
                bullish.iloc[i] = True
            else:
                st.iloc[i] = final_upper.iloc[i]
                bullish.iloc[i] = False
    return st, bullish


def calculate_indicators(monthly):
    x = monthly.copy()
    x["EMA20"] = x["Close"].ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean()
    x["Supertrend20_1"], x["ST_Green"] = supertrend(x, ST_PERIOD, ST_MULTIPLIER)
    x["RSI5"] = rsi_wilder(x["Close"], RSI_PERIOD)
    x["RSI5_SMA14"] = x["RSI5"].rolling(RSI_SMA_PERIOD, min_periods=RSI_SMA_PERIOD).mean()
    return x


def download_monthly(symbol):
    ticker = symbol if symbol.endswith(".NS") else symbol + ".NS"
    # Warm-up data before the requested 10-year reporting window.
    start = (START_DATE - pd.DateOffset(years=4)).strftime("%Y-%m-%d")
    end = (END_DATE + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    daily = yf.download(
        ticker, start=start, end=end, interval="1d", auto_adjust=True,
        progress=False, threads=False, timeout=35
    )
    if daily is None or daily.empty:
        return pd.DataFrame()
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close"]
    if not all(c in daily.columns for c in needed):
        return pd.DataFrame()
    daily = daily[needed].dropna(subset=["High", "Low", "Close"]).copy()
    daily.index = pd.to_datetime(daily.index)
    if daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)
    daily["LastTradingDate"] = daily.index
    monthly = daily.resample("ME").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last",
        "LastTradingDate": "last"
    }).dropna(subset=needed + ["LastTradingDate"])
    return calculate_indicators(monthly)


def was_member(membership, symbol, signal_date):
    rows = membership[membership["symbol"] == symbol]
    d = pd.Timestamp(signal_date).normalize()
    return bool(((rows["from"] <= d) & (d < rows["to"])).any())


def run_scan():
    membership = load_membership()
    symbols = sorted(membership["symbol"].unique())
    rows, errors = [], []
    print(f"Scanning {len(symbols)} historical NIFTY 500 symbols.", flush=True)

    for n, symbol in enumerate(symbols, 1):
        print(f"{n}/{len(symbols)} {symbol}", flush=True)
        try:
            monthly = download_monthly(symbol)
            if monthly.empty:
                errors.append({"Symbol": symbol, "Issue": "No price history"})
                continue

            for month_end, r in monthly.iterrows():
                signal_date = pd.Timestamp(r["LastTradingDate"]).normalize()
                if signal_date < START_DATE or signal_date > END_DATE:
                    continue
                if not was_member(membership, symbol, signal_date):
                    continue

                close = r["Close"]
                ema = r["EMA20"]
                st = r["Supertrend20_1"]
                rsi = r["RSI5"]
                rsi_sma = r["RSI5_SMA14"]
                if any(pd.isna(v) for v in [close, ema, st, rsi, rsi_sma]):
                    continue
                # Bullish/green Supertrend means monthly close is above ST line.
                if close > ema and close > st and rsi > rsi_sma:
                    rows.append({
                        "Month": month_end.strftime("%Y-%m"),
                        "Signal Date": signal_date.strftime("%Y-%m-%d"),
                        "Symbol": symbol,
                        "Monthly Close": round(float(close), 4),
                        "EMA20": round(float(ema), 4),
                        "Supertrend(20,1)": round(float(st), 4),
                        "Supertrend": "GREEN",
                        "RSI(5)": round(float(rsi), 4),
                        "SMA(14) of RSI(5)": round(float(rsi_sma), 4),
                    })
        except Exception as exc:
            errors.append({"Symbol": symbol, "Issue": str(exc)[:250]})
        time.sleep(0.05)

    signals = pd.DataFrame(rows, columns=[
        "Month", "Signal Date", "Symbol", "Monthly Close", "EMA20",
        "Supertrend(20,1)", "Supertrend", "RSI(5)", "SMA(14) of RSI(5)"
    ])
    if not signals.empty:
        signals = signals.sort_values(["Signal Date", "Symbol"]).reset_index(drop=True)

    notes = pd.DataFrame([
        {"Item": "Universe", "Details": "Historical NIFTY 500 membership on each signal date, based on configured membership dataset."},
        {"Item": "Monthly close", "Details": "Monthly OHLC aggregated from daily adjusted data; signal date is the month's last available trading date."},
        {"Item": "EMA", "Details": "Monthly EMA(20) calculated from monthly close."},
        {"Item": "Supertrend", "Details": "Monthly Supertrend(20,1); GREEN when monthly close is above the Supertrend line."},
        {"Item": "RSI", "Details": "Wilder RSI(5) on monthly close compared with simple moving average(14) of RSI(5)."},
        {"Item": "History", "Details": "Backdated scan over the trailing 10 years; results include each month meeting all criteria."},
    ])
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        signals.to_excel(writer, sheet_name="Signals", index=False)
        notes.to_excel(writer, sheet_name="Scan Notes", index=False)
        pd.DataFrame(errors, columns=["Symbol", "Issue"]).to_excel(writer, sheet_name="Skipped Symbols", index=False)
    print(f"Saved {OUTPUT_FILE}: {len(signals)} qualifying symbol-month rows.")
    return signals


def email_report():
    username = os.environ.get("GMAIL_USERNAME")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    recipient = os.environ.get("GMAIL_TO")
    if not all([username, password, recipient]):
        raise RuntimeError("Missing Gmail secrets: GMAIL_USERNAME, GMAIL_APP_PASSWORD, GMAIL_TO.")
    message = EmailMessage()
    message["Subject"] = "NIFTY 500 Monthly EMA20 Supertrend RSI(5) Historical Scan"
    message["From"] = username
    message["To"] = recipient
    message.set_content(
        "Attached is the backdated monthly NIFTY 500 scan.\n"
        "Rules on each completed month: Close > EMA20; Supertrend(20,1) GREEN; "
        "RSI(5) > SMA(14) of RSI(5).\n"
    )
    file = Path(OUTPUT_FILE)
    message.add_attachment(
        file.read_bytes(),
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=file.name
    )
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=45) as smtp:
        smtp.login(username, password)
        smtp.send_message(message)
    print(f"Report emailed to {recipient}.")


if __name__ == "__main__":
    try:
        run_scan()
        email_report()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
