import io
import os
import smtplib
from email.message import EmailMessage

import pandas as pd
import requests
import yfinance as yf

NIFTY_500_PAGE = "https://www.niftyindices.com/indices/equity/broad-based-indices/nifty-500"
NIFTY_500_CSV_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"

NEAR_ATH_PERCENT = 1.0
OUTPUT_ALL = "nifty500_all_time_high.csv"
OUTPUT_NEAR = "nifty500_at_or_near_ath.csv"


def get_nifty500_symbols():
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,text/plain,*/*",
        "Referer": NIFTY_500_PAGE,
    }
    response = requests.get(NIFTY_500_CSV_URL, headers=headers, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(io.BytesIO(response.content))

    symbol_col = next(
        (c for c in df.columns if str(c).strip().lower() == "symbol"),
        None,
    )
    if symbol_col is None:
        raise RuntimeError(
            "Could not find Symbol column in the NSE NIFTY 500 CSV. "
            f"Columns: {list(df.columns)}"
        )

    symbols = (
        df[symbol_col].astype(str).str.strip().str.upper().dropna().tolist()
    )
    symbols = [s for s in symbols if s and s != "NAN"]

    if len(symbols) < 450:
        raise RuntimeError(
            f"NSE returned only {len(symbols)} symbols. "
            "Stopping instead of scanning an incomplete NIFTY 500 list."
        )

    return sorted(set(symbols))


def scan_one_symbol(symbol):
    ticker = f"{symbol}.NS"

    try:
        data = yf.download(
            ticker,
            period="max",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if data is None or data.empty:
            return None, "No Yahoo Finance data"

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if "High" not in data.columns or "Close" not in data.columns:
            return None, "Missing High or Close columns"

        data = data[["High", "Close"]].copy()
        data["High"] = pd.to_numeric(data["High"], errors="coerce")
        data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
        data = data.dropna()

        if data.empty:
            return None, "No valid price rows"

        ath = float(data["High"].max())
        ath_date = data["High"].idxmax()
        latest_close = float(data["Close"].iloc[-1])

        if hasattr(ath_date, "date"):
            ath_date = ath_date.date().isoformat()

        distance_pct = ((ath - latest_close) / ath) * 100.0

        return {
            "Stock": symbol,
            "Yahoo_Symbol": ticker,
            "Current_Close": round(latest_close, 2),
            "All_Time_High": round(ath, 2),
            "Distance_From_ATH_Pct": round(distance_pct, 2),
            "ATH_Date": str(ath_date),
            "At_ATH": latest_close >= ath * 0.999999,
            "Within_1pct_of_ATH": distance_pct <= NEAR_ATH_PERCENT,
        }, None

    except Exception as exc:
        return None, str(exc)


def send_email(near_df):
    username = os.getenv("GMAIL_USERNAME")
    app_password = os.getenv("GMAIL_APP_PASSWORD")
    recipient = os.getenv("GMAIL_TO")

    if not username or not app_password or not recipient:
        print("Gmail secrets not configured; skipping email.")
        return

    if near_df.empty:
        body = (
            "NIFTY 500 All-Time-High Scan\n\n"
            "No NIFTY 500 stock closed within 1% of its historical ATH."
        )
    else:
        body = (
            "NIFTY 500 All-Time-High Scan\n\n"
            f"Stocks within 1% of ATH: {len(near_df)}\n\n"
            + near_df.to_string(index=False)
        )

    msg = EmailMessage()
    msg["Subject"] = "NIFTY 500 All-Time-High Scan"
    msg["From"] = username
    msg["To"] = recipient
    msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
        server.login(username, app_password)
        server.send_message(msg)

    print("Gmail sent successfully.")


def main():
    print("=" * 70)
    print("NIFTY 500 ALL-TIME-HIGH SCANNER")
    print("=" * 70)
    print("Downloading current NIFTY 500 constituents...")

    symbols = get_nifty500_symbols()
    print(f"Constituents received: {len(symbols)}")
    print("Scanning historical daily High/Close data...")
    print()

    results = []
    errors = []

    for number, symbol in enumerate(symbols, start=1):
        print(f"[{number}/{len(symbols)}] {symbol}")
        row, error = scan_one_symbol(symbol)

        if row is not None:
            results.append(row)
        else:
            errors.append((symbol, error))
            print(f"  ERROR: {error}")

    if not results:
        raise RuntimeError(
            "No stock results were produced. "
            "The scanner will not create a misleading empty ATH report."
        )

    all_df = pd.DataFrame(results)
    all_df = all_df.sort_values(
        ["Within_1pct_of_ATH", "Distance_From_ATH_Pct", "Stock"],
        ascending=[False, True, True],
    )

    near_df = all_df[all_df["Within_1pct_of_ATH"]].copy()

    all_df.to_csv(OUTPUT_ALL, index=False)
    near_df.to_csv(OUTPUT_NEAR, index=False)

    print()
    print("=" * 70)
    print(f"Stocks successfully scanned: {len(results)}")
    print(f"Stocks with errors: {len(errors)}")
    print(f"Within 1% of ATH: {len(near_df)}")
    print(f"Exact ATH closes: {int(all_df['At_ATH'].sum())}")
    print(f"Saved: {OUTPUT_ALL}")
    print(f"Saved: {OUTPUT_NEAR}")
    print("=" * 70)

    if not near_df.empty:
        print()
        print("STOCKS WITHIN 1% OF ALL-TIME HIGH:")
        print(
            near_df[
                [
                    "Stock",
                    "Current_Close",
                    "All_Time_High",
                    "Distance_From_ATH_Pct",
                    "ATH_Date",
                ]
            ].to_string(index=False)
        )

    if errors:
        pd.DataFrame(errors, columns=["Stock", "Error"]).to_csv(
            "nifty500_ath_errors.csv", index=False
        )
        print("Error details saved: nifty500_ath_errors.csv")

    send_email(near_df)


if __name__ == "__main__":
    main()
