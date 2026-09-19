import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

YEARS = 10
THRESHOLD = 0.85  # within 15% of ATH
BATCH_SIZE = 10
MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/aditya-jha/"
    "nse-historical-membership/main/index_history/data/"
    "index_membership_history.csv"
)
OUTPUT_FILE = "nifty500_15pct_ath_historical.xlsx"


def ns_dates(values):
    idx = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize().astype("datetime64[ns]")


def load_membership():
    m = pd.read_csv(MEMBERSHIP_URL)
    cols = {str(c).strip().lower(): c for c in m.columns}
    for name in ("symbol", "valid_from", "valid_to"):
        if name not in cols:
            raise ValueError(f"Membership file missing {name}")

    m = m[[cols["symbol"], cols["valid_from"], cols["valid_to"]]].copy()
    m.columns = ["Symbol", "ValidFrom", "ValidTo"]
    m["Symbol"] = m["Symbol"].astype(str).str.strip().str.upper()
    m["ValidFrom"] = pd.to_datetime(m["ValidFrom"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    m["ValidTo"] = pd.to_datetime(m["ValidTo"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    return m.dropna(subset=["Symbol", "ValidFrom"])


def is_member(symbol, dates, membership):
    """Return a boolean Series: stock was a NIFTY 500 member on each date."""
    periods = membership[membership["Symbol"] == symbol]
    if periods.empty:
        return pd.Series(False, index=dates)

    result = np.zeros(len(dates), dtype=bool)
    date_values = dates.values.astype("datetime64[ns]")

    for row in periods.itertuples(index=False):
        start = np.datetime64(row.ValidFrom, "ns")
        if pd.isna(row.ValidTo):
            mask = date_values >= start
        else:
            end = np.datetime64(row.ValidTo, "ns")
            mask = (date_values >= start) & (date_values < end)
        result |= mask

    return pd.Series(result, index=dates)


def download_batch(symbols):
    tickers = [f"{s}.NS" for s in symbols]
    for attempt in range(1, 4):
        try:
            data = yf.download(
                tickers=tickers,
                period="max",
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            if data is not None and not data.empty:
                return data
        except Exception as exc:
            print(f"Download attempt {attempt} failed: {exc}")
        time.sleep(3 * attempt)
    return None


def extract(data, symbol):
    ticker = f"{symbol}.NS"
    if data is None or data.empty:
        return None
    try:
        if isinstance(data.columns, pd.MultiIndex):
            l0 = set(data.columns.get_level_values(0))
            l1 = set(data.columns.get_level_values(1))
            if ticker in l0:
                df = data[ticker].copy()
            elif ticker in l1:
                df = data.xs(ticker, axis=1, level=1).copy()
            else:
                return None
        else:
            df = data.copy()

        if not {"High", "Close"}.issubset(df.columns):
            return None
        df = df[["High", "Close"]].copy()
        df.index = ns_dates(df.index)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df["High"] = pd.to_numeric(df["High"], errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        return df.dropna(subset=["High", "Close"])
    except Exception as exc:
        print(f"{symbol}: extraction error: {exc}")
        return None


def scan_symbol(symbol, df, membership, scan_start, scan_end):
    if df is None or df.empty:
        return []

    # ATH is calculated only from prices known on or before each date.
    # Using adjusted High prevents split adjustments from creating false ATH gaps.
    df = df.copy()
    df.index = ns_dates(df.index)
    df["ATH"] = df["High"].cummax()

    # "Reach within 15%" means the day's HIGH entered the 15%-from-ATH zone.
    df["DistancePct"] = (df["High"] / df["ATH"] - 1.0) * 100.0
    df["Within15"] = df["High"] >= df["ATH"] * THRESHOLD

    df = df.loc[
        (df.index >= scan_start)
        & (df.index <= scan_end)
        & df["Within15"]
    ].copy()
    if df.empty:
        return []

    member = is_member(symbol, df.index, membership)
    df = df.loc[member]
    if df.empty:
        return []

    out = []
    for date, row in df.iterrows():
        out.append({
            "Date": date.date(),
            "Stock": symbol,
            "Daily_High": round(float(row["High"]), 4),
            "All_Time_High": round(float(row["ATH"]), 4),
            "Distance_From_ATH_Pct": round(float(row["DistancePct"]), 2),
        })
    return out


def main():
    today = pd.Timestamp.today().normalize()
    scan_start = today - pd.DateOffset(years=YEARS)
    scan_end = today

    print(f"Scanning {scan_start.date()} to {scan_end.date()}")
    print("Rule: daily HIGH reaches at least 85% of the ATH known up to that day.")
    print("Universe: NIFTY 500 membership applicable on that exact date.")

    membership = load_membership()
    symbols = sorted(membership["Symbol"].dropna().unique().tolist())
    print(f"Historical NIFTY 500 symbols: {len(symbols)}")

    all_results = []
    errors = []

    for start in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[start:start + BATCH_SIZE]
        print(f"Batch {start // BATCH_SIZE + 1}: {len(batch)} stocks")
        data = download_batch(batch)

        if data is None:
            for symbol in batch:
                errors.append({"Stock": symbol, "Error": "Yahoo download failed"})
            continue

        for symbol in batch:
            try:
                df = extract(data, symbol)
                if df is None or df.empty:
                    errors.append({"Stock": symbol, "Error": "No usable Yahoo data"})
                    continue
                all_results.extend(scan_symbol(symbol, df, membership, scan_start, scan_end))
            except Exception as exc:
                errors.append({"Stock": symbol, "Error": str(exc)})
        time.sleep(2)

    signals = pd.DataFrame(all_results)
    if signals.empty:
        signals = pd.DataFrame(columns=[
            "Date", "Stock", "Daily_High", "All_Time_High", "Distance_From_ATH_Pct"
        ])
    else:
        signals = signals.sort_values(["Date", "Stock"]).reset_index(drop=True)

    errors_df = pd.DataFrame(errors)
    if errors_df.empty:
        errors_df = pd.DataFrame(columns=["Stock", "Error"])

    rules = pd.DataFrame({
        "Rule": [
            "Period", "Universe", "Timeframe", "Trigger", "ATH calculation", "Look-ahead", "Output"
        ],
        "Value": [
            f"{scan_start.date()} to {scan_end.date()}",
            "Historical NIFTY 500 constituent on the signal date",
            "Daily",
            "Daily High >= 85% of ATH (within 15% of ATH)",
            "Running maximum of daily adjusted High through that date",
            "None; future prices are never used",
            "Date-wise Stock list with High, ATH and distance from ATH",
        ],
    })

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        signals.to_excel(writer, sheet_name="Signals", index=False)
        rules.to_excel(writer, sheet_name="Rules", index=False)
        errors_df.to_excel(writer, sheet_name="Data Errors", index=False)
        for ws in writer.sheets.values():
            ws.freeze_panes = "A2"
            if ws.max_row > 1:
                ws.auto_filter.ref = ws.dimensions
            for cells in ws.columns:
                letter = cells[0].column_letter
                width = min(max(max(len(str(c.value or "")) for c in cells) + 2, 12), 50)
                ws.column_dimensions[letter].width = width

    print(f"Signal rows: {len(signals)}")
    print(f"Errors: {len(errors_df)}")
    print(f"Created: {Path(OUTPUT_FILE).resolve()}")


if __name__ == "__main__":
    main()
