#!/usr/bin/env python3
"""Historical NIFTY 500 weekly Supertrend(20,1) fresh-cross-above-EMA(50) scanner."""
from datetime import datetime
from io import BytesIO
import os, re, smtplib, ssl, time
from email.message import EmailMessage

import numpy as np
import pandas as pd
import requests
import yfinance as yf

MEMBERSHIP_URL = "https://raw.githubusercontent.com/aditya-jha/nse-historical-membership/main/index_history/data/index_membership_history.csv"
SCAN_START = pd.Timestamp("2016-10-21")
SCAN_END = pd.Timestamp("2026-10-21")
DATA_START = "2015-01-01"  # indicator warm-up before the first eligible signal
HEADERS = {"User-Agent": "Mozilla/5.0"}
BATCH_SIZE = 15


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def find_col(cols, names):
    by_norm = {norm(c): c for c in cols}
    for name in names:
        if norm(name) in by_norm:
            return by_norm[norm(name)]
    return None


def load_membership():
    response = requests.get(MEMBERSHIP_URL, headers=HEADERS, timeout=45)
    response.raise_for_status()
    df = pd.read_csv(BytesIO(response.content))
    symbol_col = find_col(df.columns, ["symbol", "ticker", "stock_symbol"])
    start_col = find_col(df.columns, ["valid_from", "from_date", "start_date", "effective_from"])
    end_col = find_col(df.columns, ["valid_to", "to_date", "end_date", "effective_to"])
    index_col = find_col(df.columns, ["index_name", "index", "indexname", "index_code"])
    if not all([symbol_col, start_col, end_col, index_col]):
        raise RuntimeError(f"Historical membership CSV columns not recognized: {list(df.columns)}")
    df["_symbol"] = df[symbol_col].astype(str).str.strip().str.upper()
    df["_start"] = pd.to_datetime(df[start_col], errors="coerce").dt.normalize()
    df["_end"] = pd.to_datetime(df[end_col], errors="coerce").dt.normalize()
    df["_index"] = df[index_col].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    df = df[df["_index"].str.contains("NIFTY500", na=False) & df["_start"].notna()]
    df = df[(df["_start"] <= SCAN_END) & (df["_end"].isna() | (df["_end"] > SCAN_START))]
    if df.empty:
        raise RuntimeError("No NIFTY 500 historical membership rows found for the scan period.")
    intervals = {}
    for _, r in df.iterrows():
        sym = r["_symbol"]
        if not sym or sym == "NAN":
            continue
        end = r["_end"] if pd.notna(r["_end"]) else pd.Timestamp("2262-04-11")
        intervals.setdefault(sym, []).append((r["_start"], end))
    return sorted(intervals), intervals


def rma(s, n):
    """Wilder's RMA, seeded with the SMA of the first n valid values.

    This matches the standard ATR smoothing used by TradingView-style
    Supertrend implementations more closely than an unseeded EWM.
    """
    values = pd.to_numeric(s, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid_positions = np.flatnonzero(values.notna().to_numpy())
    if len(valid_positions) < n:
        return out
    # Start the seed at the first contiguous run of n valid observations.
    first = int(valid_positions[0])
    seed_end = first + n
    if seed_end > len(values) or values.iloc[first:seed_end].isna().any():
        for start in range(0, len(values) - n + 1):
            if values.iloc[start:start+n].notna().all():
                first = start
                seed_end = start + n
                break
        else:
            return out
    out.iloc[seed_end - 1] = values.iloc[first:seed_end].mean()
    for i in range(seed_end, len(values)):
        if pd.isna(values.iloc[i]):
            continue
        if pd.isna(out.iloc[i - 1]):
            # Resume only after n consecutive valid values, with a fresh SMA seed.
            start = i - n + 1
            if start >= 0 and values.iloc[start:i+1].notna().all():
                out.iloc[i] = values.iloc[start:i+1].mean()
        else:
            out.iloc[i] = ((n - 1) * out.iloc[i - 1] + values.iloc[i]) / n
    return out


def supertrend(data, n=20, mult=1.0):
    h, l, c = data["High"].astype(float), data["Low"].astype(float), data["Close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    atr = rma(tr, n)
    mid = (h+l)/2
    ub, lb = mid + mult*atr, mid - mult*atr
    fu = pd.Series(np.nan, index=data.index)
    fl = pd.Series(np.nan, index=data.index)
    st = pd.Series(np.nan, index=data.index)
    direction = pd.Series(np.nan, index=data.index)
    for i in range(len(data)):
        if pd.isna(atr.iloc[i]):
            continue
        if i == 0 or pd.isna(direction.iloc[i-1]):
            fu.iloc[i], fl.iloc[i] = ub.iloc[i], lb.iloc[i]
            direction.iloc[i] = 1.0 if c.iloc[i] >= mid.iloc[i] else -1.0
        else:
            pu, pl = fu.iloc[i-1], fl.iloc[i-1]
            fu.iloc[i] = ub.iloc[i] if pd.isna(pu) or ub.iloc[i] < pu or c.iloc[i-1] > pu else pu
            fl.iloc[i] = lb.iloc[i] if pd.isna(pl) or lb.iloc[i] > pl or c.iloc[i-1] < pl else pl
            if direction.iloc[i-1] == 1.0:
                direction.iloc[i] = -1.0 if c.iloc[i] < fl.iloc[i] else 1.0
            else:
                direction.iloc[i] = 1.0 if c.iloc[i] > fu.iloc[i] else -1.0
        st.iloc[i] = fl.iloc[i] if direction.iloc[i] == 1.0 else fu.iloc[i]
    return st


def fresh_cross_above(st_now, ema_now, st_prev, ema_prev):
    """True only when weekly Supertrend freshly crosses above weekly EMA50."""
    vals = (st_now, ema_now, st_prev, ema_prev)
    if any(pd.isna(v) for v in vals):
        return False
    return (st_prev <= ema_prev) and (st_now > ema_now)


def membership_on(intervals, symbol, day):
    day = pd.Timestamp(day).normalize()
    # Source intervals are half-open [valid_from, valid_to): the end date is excluded.
    return any(start <= day < end for start, end in intervals.get(symbol.upper(), []))


def daily_to_weekly(daily):
    # Build Friday-labelled weekly OHLC candles from daily data. A candle is
    # only eligible if its Friday label is on/before SCAN_END; partial current
    # weeks therefore cannot create signals. Weekly Close is the last close
    # observed in that completed Friday-ending week.
    weekly = daily.resample("W-FRI", label="right", closed="right").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna(subset=["Open", "High", "Low", "Close"])
    # Cut at the requested historical end; labels are the Friday of each week.
    return weekly[weekly.index.normalize() <= SCAN_END]


def scan_symbol(symbol, intervals, frame):
    if frame is None or frame.empty:
        return [], f"{symbol}: no price data"
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if not {"Open", "High", "Low", "Close"}.issubset(frame.columns):
        return [], f"{symbol}: missing OHLC columns"
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index)
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
    if len(frame) < 250:
        return [], f"{symbol}: insufficient daily history"
    w = daily_to_weekly(frame)
    if len(w) < 60:
        return [], f"{symbol}: insufficient weekly history"
    w["EMA50"] = w["Close"].ewm(span=50, adjust=False, min_periods=50).mean()
    w["ST20_1"] = supertrend(w, 20, 1.0)
    w["PrevST"] = w["ST20_1"].shift(1)
    w["PrevEMA50"] = w["EMA50"].shift(1)
    # Fresh bullish crossover on completed weekly indicator values only.
    # Previous week: ST <= EMA50; current week: ST > EMA50.
    cross = (
        w["ST20_1"].notna()
        & w["EMA50"].notna()
        & w["PrevST"].notna()
        & w["PrevEMA50"].notna()
        & (w["PrevST"] <= w["PrevEMA50"])
        & (w["ST20_1"] > w["EMA50"])
    )
    # Both sides of the crossover are indicator values from completed weekly bars.
    # The weekly bars are Friday-labelled; partial weeks after SCAN_END are removed.
    cross = cross & (w.index.normalize() >= SCAN_START) & (w.index.normalize() <= SCAN_END)
    found = []
    for date, row in w.loc[cross].iterrows():
        day = pd.Timestamp(date).normalize()
        if not membership_on(intervals, symbol, day):
            continue
        found.append({
            "Symbol": symbol,
            "Signal Date (weekly close)": day.strftime("%Y-%m-%d"),
            "Weekly Close": round(float(row["Close"]), 2),
            "Weekly Supertrend (20,1)": round(float(row["ST20_1"]), 4),
            "Weekly EMA (50)": round(float(row["EMA50"]), 4),
            "Previous Weekly Supertrend": round(float(row["PrevST"]), 4),
            "Previous Weekly EMA (50)": round(float(row["PrevEMA50"]), 4),
        })
    return found, None


def make_workbook(results, errors, n_symbols):
    cols = ["Symbol", "Signal Date (weekly close)", "Weekly Close", "Weekly Supertrend (20,1)",
            "Weekly EMA (50)", "Previous Weekly Supertrend", "Previous Weekly EMA (50)"]
    df = pd.DataFrame(results, columns=cols)
    if not df.empty:
        df = df.sort_values(["Signal Date (weekly close)", "Symbol"])
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Historical Crossovers", index=False)
        pd.DataFrame([{
            "Scan start": str(SCAN_START.date()), "Scan end": str(SCAN_END.date()),
            "Rule": "Weekly Supertrend(20,1) fresh cross above weekly EMA(50)",
            "Membership": "NIFTY 500 membership on exact signal date",
            "Unique symbols": n_symbols, "Crossover rows": len(df),
            "Generated UTC": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        }]).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame({"Errors / skipped symbols": errors}).to_excel(writer, sheet_name="Scan Notes", index=False)
    out.seek(0)
    return out.read(), len(df)


def email_results(content, count, n_symbols, errors):
    user = os.environ.get("GMAIL_USERNAME")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    to = os.environ.get("GMAIL_TO")
    if not all([user, password, to]):
        raise RuntimeError("Missing GitHub secrets: GMAIL_USERNAME, GMAIL_APP_PASSWORD, GMAIL_TO")
    msg = EmailMessage()
    msg["Subject"] = f"Historical NIFTY 500 ST/EMA crossover scan: {count} signals"
    msg["From"], msg["To"] = user, to
    msg.set_content(
        f"Historical scan completed.\nPeriod: {SCAN_START.date()} to {SCAN_END.date()}\n"
        "Rule: weekly Supertrend(20,1) freshly crosses ABOVE weekly EMA(50).\n"
        "Fresh cross: current ST > current EMA50 AND previous ST <= previous EMA50.\n"
        "Membership is checked on each signal date using historical NIFTY 500 membership data.\n\n"
        f"Unique symbols attempted: {n_symbols}\nQualifying signal rows: {count}\n"
        f"Warnings/errors: {len(errors)}\n\nSee the attached Excel workbook.\n"
        "This is a technical screening tool, not investment advice."
    )
    msg.add_attachment(content, maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"historical_nifty500_crossovers_{SCAN_START:%Y%m%d}_{SCAN_END:%Y%m%d}.xlsx")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context(), timeout=45) as server:
        server.login(user, password)
        server.send_message(msg)


def main():
    print("Loading historical NIFTY 500 membership...", flush=True)
    symbols, intervals = load_membership()
    tickers = [s if s.endswith(".NS") else s + ".NS" for s in symbols]
    all_results, errors = [], []
    print(f"{len(tickers)} symbols overlap the scan period.", flush=True)
    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] {ticker}", flush=True)
        try:
            frame = yf.download(ticker, start=DATA_START,
                end=(SCAN_END + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d", auto_adjust=True, progress=False, threads=False, timeout=30)
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            symbol = ticker[:-3] if ticker.endswith(".NS") else ticker
            rows, warning = scan_symbol(symbol, intervals, frame)
            all_results.extend(rows)
            if warning:
                errors.append(warning)
        except Exception as exc:
            errors.append(f"{ticker}: {type(exc).__name__}: {exc}")
        if i % 20 == 0:
            time.sleep(1)
    # De-duplicate symbol/date rows.
    if all_results:
        all_results = list({(r["Symbol"], r["Signal Date (weekly close)"]): r for r in all_results}.values())
    content, count = make_workbook(all_results, errors, len(tickers))
    email_results(content, count, len(tickers), errors)
    print(f"Finished. {count} qualifying crossover rows emailed.", flush=True)


if __name__ == "__main__":
    main()
