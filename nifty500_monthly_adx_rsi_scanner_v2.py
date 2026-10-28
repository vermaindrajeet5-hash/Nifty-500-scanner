#!/usr/bin/env python3
"""Historical NIFTY 500 monthly ADX / RSI scanner.

Signal rule, evaluated on each completed monthly candle:
  1. Monthly ADX(14) > 20
  2. RSI(5) > EMA(14) of RSI(5)

The user's first two criteria both specify monthly ADX > 20, so they are
implemented as one condition (not two different tests).
"""
from __future__ import annotations

import os
import sys
import time
import smtplib
import traceback
from email.message import EmailMessage
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

END_DATE = pd.Timestamp.now().normalize()
START_DATE = END_DATE - pd.DateOffset(years=10)
MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/aditya-jha/nse-historical-membership/"
    "main/index_history/data/index_membership_history.csv"
)
OUTFILE = "nifty500_monthly_adx_rsi_10y.xlsx"
USER_AGENT = "Mozilla/5.0 (compatible; HistoricalScanner/1.0)"


def normalize_col(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def find_col(columns, candidates):
    norm_map = {normalize_col(c): c for c in columns}
    for candidate in candidates:
        if normalize_col(candidate) in norm_map:
            return norm_map[normalize_col(candidate)]
    for normalized, original in norm_map.items():
        if any(normalize_col(c) in normalized for c in candidates):
            return original
    return None


def load_historical_membership() -> pd.DataFrame:
    """Load and normalize historical NIFTY 500 membership intervals."""
    response = requests.get(
        MEMBERSHIP_URL, headers={"User-Agent": USER_AGENT}, timeout=60
    )
    response.raise_for_status()
    df = pd.read_csv(BytesIO(response.content))
    if df.empty:
        raise RuntimeError("Historical membership CSV downloaded but was empty.")

    symbol_col = find_col(df.columns, ["symbol", "ticker", "security"])
    start_col = find_col(df.columns, ["valid_from", "start_date", "from_date", "date_from"])
    end_col = find_col(df.columns, ["valid_to", "end_date", "to_date", "date_to"])
    index_col = find_col(df.columns, ["index_name", "index", "indexname"])

    if not symbol_col or not start_col or not end_col:
        raise RuntimeError(
            "Could not recognize membership CSV columns. Found: "
            + ", ".join(map(str, df.columns))
        )
    if index_col:
        idx = df[index_col].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
        df = df[idx.str.contains("NIFTY500", na=False)].copy()
        if df.empty:
            raise RuntimeError(
                "Membership CSV has an index column, but no NIFTY 500 rows matched."
            )

    out = pd.DataFrame({
        "symbol": df[symbol_col].astype(str).str.strip().str.upper(),
        "valid_from": pd.to_datetime(df[start_col], errors="coerce", dayfirst=False),
        "valid_to": pd.to_datetime(df[end_col], errors="coerce", dayfirst=False),
    })
    out = out[out["symbol"].ne("") & out["symbol"].ne("NAN")].copy()
    out["valid_from"] = out["valid_from"].dt.normalize()
    out["valid_to"] = out["valid_to"].dt.normalize()
    out = out[out["valid_from"].notna()]
    # This dataset uses [valid_from, valid_to) intervals; blank valid_to means current.
    out["valid_to_exclusive"] = out["valid_to"].fillna(pd.Timestamp.max.normalize())
    if out.empty:
        raise RuntimeError("No usable historical NIFTY 500 membership intervals found.")
    return out[["symbol", "valid_from", "valid_to_exclusive"]]


def true_range(high, low, close):
    prev_close = close.shift(1)
    parts = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    return parts.max(axis=1)


def wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder moving average seeded with the first full-period simple average."""
    values = series.astype(float)
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna()
    if len(valid) < period:
        return out
    first_pos = values.index.get_loc(valid.index[period - 1])
    seed = values.iloc[: first_pos + 1].dropna()
    if len(seed) < period:
        return out
    out.iloc[first_pos] = seed.iloc[:period].mean()
    for i in range(first_pos + 1, len(values)):
        prev = out.iloc[i - 1]
        current = values.iloc[i]
        if pd.isna(prev):
            continue
        if pd.isna(current):
            out.iloc[i] = prev
        else:
            out.iloc[i] = (prev * (period - 1) + current) / period
    return out


def calculate_monthly_indicators(monthly: pd.DataFrame) -> pd.DataFrame:
    """Calculate Wilder ADX(14), Wilder RSI(5), and EMA(14) of RSI(5)."""
    x = monthly.copy()
    high = x["High"].astype(float)
    low = x["Low"].astype(float)
    close = x["Close"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=x.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=x.index,
    )
    tr = true_range(high, low, close)
    atr = wilder_rma(tr, 14)
    plus_di = 100.0 * wilder_rma(plus_dm, 14) / atr.replace(0, np.nan)
    minus_di = 100.0 * wilder_rma(minus_dm, 14) / atr.replace(0, np.nan)
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    x["ADX14"] = wilder_rma(dx, 14)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = wilder_rma(gain, 5)
    avg_loss = wilder_rma(loss, 5)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss.ne(0), 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    x["RSI5"] = rsi
    x["RSI5_EMA14"] = x["RSI5"].ewm(span=14, adjust=False, min_periods=14).mean()
    return x


def download_monthly(symbol: str) -> pd.DataFrame:
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    # yfinance end is exclusive; request through one day beyond the scan end.
    data = yf.download(
        ticker,
        start=(START_DATE - pd.DateOffset(years=2)).strftime("%Y-%m-%d"),
        end=(END_DATE + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
        timeout=30,
    )
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close"]
    if not all(col in data.columns for col in required):
        return pd.DataFrame()
    daily = data[required].dropna(subset=["High", "Low", "Close"]).copy()
    daily.index = pd.to_datetime(daily.index).tz_localize(None)
    daily["TradingDate"] = daily.index
    monthly = daily.resample("ME").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "TradingDate": "last",
    }).dropna(subset=["Open", "High", "Low", "Close", "TradingDate"])

    # Include fully completed prior months. Include the current month only if
    # the latest available trading date is today and no weekdays remain in
    # this calendar month (conservative handling of a last trading day before
    # a weekend). Exchange holidays are not separately modeled.
    today = pd.Timestamp.now().normalize()
    cutoff = min(END_DATE, today)
    month_ends = monthly.index.normalize()
    complete = month_ends <= cutoff
    current_month = (monthly.index.year == today.year) & (monthly.index.month == today.month)
    remaining_weekdays = pd.bdate_range(today + pd.Timedelta(days=1), today + pd.offsets.MonthEnd(0))
    current_month_closed_today = (
        current_month
        & (monthly["TradingDate"].dt.normalize() == today)
        & (len(remaining_weekdays) == 0)
    )
    monthly = monthly[complete | current_month_closed_today]
    if monthly.empty:
        return monthly
    return calculate_monthly_indicators(monthly)


def is_member(membership: pd.DataFrame, symbol: str, month_end: pd.Timestamp) -> bool:
    rows = membership[membership["symbol"] == symbol]
    day = pd.Timestamp(month_end).normalize()
    return bool(((rows["valid_from"] <= day) & (day < rows["valid_to_exclusive"])).any())


def scan() -> pd.DataFrame:
    membership = load_historical_membership()
    symbols = sorted(membership["symbol"].dropna().unique().tolist())
    records = []
    errors = []
    total = len(symbols)

    for number, symbol in enumerate(symbols, 1):
        print(f"[{number}/{total}] {symbol}", flush=True)
        try:
            monthly = download_monthly(symbol)
            if monthly.empty:
                errors.append({"Symbol": symbol, "Reason": "No price data"})
                continue
            # Use a generous history for indicator warm-up, then retain requested 10 years.
            monthly = monthly[
                (monthly["TradingDate"].dt.normalize() >= START_DATE)
                & (monthly["TradingDate"].dt.normalize() <= END_DATE)
            ].copy()
            if monthly.empty:
                continue
            # A signal is evaluated on the actual last trading date of a completed month.
            for date, row in monthly.iterrows():
                signal_date = pd.Timestamp(row["TradingDate"]).normalize()
                if signal_date < START_DATE or signal_date > END_DATE:
                    continue
                if not is_member(membership, symbol, signal_date):
                    continue
                adx = row.get("ADX14", np.nan)
                rsi = row.get("RSI5", np.nan)
                rsi_ema = row.get("RSI5_EMA14", np.nan)
                close = row.get("Close", np.nan)
                if pd.isna(adx) or pd.isna(rsi) or pd.isna(rsi_ema) or pd.isna(close):
                    continue
                if adx > 20 and rsi > rsi_ema:
                    records.append({
                        "Signal Date": signal_date.strftime("%Y-%m-%d"),
                        "Symbol": symbol,
                        "Monthly Close": round(float(close), 4),
                        "ADX(14) Monthly": round(float(adx), 4),
                        "RSI(5) Monthly": round(float(rsi), 4),
                        "EMA(14) of RSI(5)": round(float(rsi_ema), 4),
                    })
        except Exception as exc:
            errors.append({"Symbol": symbol, "Reason": str(exc)[:250]})
        time.sleep(0.05)

    results = pd.DataFrame(records, columns=[
        "Signal Date", "Symbol", "Monthly Close", "ADX(14) Monthly",
        "RSI(5) Monthly", "EMA(14) of RSI(5)"
    ])
    if not results.empty:
        results = results.sort_values(["Signal Date", "Symbol"]).reset_index(drop=True)
    errors_df = pd.DataFrame(errors, columns=["Symbol", "Reason"])
    notes = pd.DataFrame([
        {"Item": "Rule 1 and Rule 2", "Details": "Both specify monthly ADX > 20; implemented as one condition."},
        {"Item": "Signal timing", "Details": "Only completed monthly candles are evaluated; current month is included only when its last trading session is detected."},
        {"Item": "Universe", "Details": "Historical NIFTY 500 membership on each signal month, from the configured external dataset."},
        {"Item": "ADX", "Details": "Wilder ADX(14) calculated from monthly OHLC."},
        {"Item": "RSI", "Details": "Wilder RSI(5) from monthly close; compared with EMA(14) of RSI(5)."},
    ])
    with pd.ExcelWriter(OUTFILE, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="Signals", index=False)
        notes.to_excel(writer, sheet_name="Scan Notes", index=False)
        errors_df.to_excel(writer, sheet_name="Skipped Symbols", index=False)
    print(f"Saved {OUTFILE} with {len(results)} signals.")
    return results


def send_email():
    username = os.environ.get("GMAIL_USERNAME")
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    recipient = os.environ.get("GMAIL_TO")
    if not username or not app_password or not recipient:
        raise RuntimeError("Missing GMAIL_USERNAME, GMAIL_APP_PASSWORD, or GMAIL_TO secret.")
    msg = EmailMessage()
    msg["Subject"] = "Historical NIFTY 500 Monthly ADX / RSI Scanner"
    msg["From"] = username
    msg["To"] = recipient
    msg.set_content(
        "Attached is the historical monthly NIFTY 500 scan.\n\n"
        "Criteria: monthly ADX(14) > 20 and monthly RSI(5) > EMA(14) of RSI(5), "
        "evaluated on completed monthly candles. Your first two criteria both say "
        "ADX > 20, so they are treated as the same condition.\n"
    )
    file_path = Path(OUTFILE)
    msg.add_attachment(
        file_path.read_bytes(),
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=file_path.name,
    )
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=45) as smtp:
        smtp.login(username, app_password)
        smtp.send_message(msg)
    print(f"Email sent to {recipient}.")


if __name__ == "__main__":
    try:
        scan()
        send_email()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
