import os
import smtplib
import requests
import pandas as pd
import yfinance as yf

from datetime import datetime
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
        & (up_move > 0)
    )

    minus_condition = (
        (down_move > up_move)
        & (down_move > 0)
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
        * plus_dm.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False
        ).mean()
        / atr
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False
        ).mean()
        / atr
    )

    denominator = plus_di + minus_di

    dx = (
        100
        * (plus_di - minus_di).abs()
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

    weekly = data.resample(
        "W-FRI"
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })

    return weekly.dropna()


# ============================================================
# MAKE MONTHLY DATA
# ============================================================

def make_monthly(data):

    monthly = data.resample(
        "ME"
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })

    return monthly.dropna()


# ============================================================
# DOWNLOAD DATA
# ============================================================

def download_data(symbol):

    try:

        data = yf.download(
            symbol,
            period="5y",
            interval="1d",
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
            "Volume"
        ]

        for column in required_columns:

            if column not in data.columns:
                return None

        data = data[required_columns].copy()

        data = data.dropna()

        if len(data) < 100:
            return None

        return data

    except Exception as e:

        print(
            f"Data error for {symbol}: {e}"
        )

        return None


# ============================================================
# CHECK THE 5 CONDITIONS
# ============================================================

def check_conditions(data):

    try:

        # ----------------------------------------------------
        # DAILY
        # ----------------------------------------------------

        daily_rsi = rsi(
            data["Close"],
            5
        )

        current_daily_rsi = (
            daily_rsi.iloc[-1]
        )

        # ----------------------------------------------------
        # WEEKLY
        # ----------------------------------------------------

        weekly = make_weekly(data)

        if len(weekly) < 40:
            return None

        weekly_rsi = rsi(
            weekly["Close"],
            5
        )

        weekly_rsi_ma = (
            weekly_rsi
            .rolling(14)
            .mean()
        )

        weekly_adx = adx(
            weekly,
            14
        )

        current_weekly_rsi = (
            weekly_rsi.iloc[-1]
        )

        current_weekly_rsi_ma = (
            weekly_rsi_ma.iloc[-1]
        )

        current_weekly_adx = (
            weekly_adx.iloc[-1]
        )

        # ----------------------------------------------------
        # MONTHLY
        # ----------------------------------------------------

        monthly = make_monthly(data)

        if len(monthly) < 40:
            return None

        monthly_rsi = rsi(
            monthly["Close"],
            5
        )

        monthly_rsi_ma = (
            monthly_rsi
            .rolling(14)
            .mean()
        )

        monthly_adx = adx(
            monthly,
            14
        )

        current_monthly_rsi = (
            monthly_rsi.iloc[-1]
        )

        current_monthly_rsi_ma = (
            monthly_rsi_ma.iloc[-1]
        )

        current_monthly_adx = (
            monthly_adx.iloc[-1]
        )

        # ----------------------------------------------------
        # CHECK ALL 5 CONDITIONS
        # ----------------------------------------------------

        condition_1 = (
            current_monthly_rsi
            > current_monthly_rsi_ma
        )

        condition_2 = (
            current_weekly_rsi
            > current_weekly_rsi_ma
        )

        condition_3 = (
            current_monthly_adx >= 25
        )

        condition_4 = (
            current_daily_rsi < 30
        )

        condition_5 = (
            current_weekly_adx >= 25
        )

        all_conditions = (
            condition_1
            and condition_2
            and condition_3
            and condition_4
            and condition_5
        )

        if not all_conditions:
            return None

        return {
            "daily_rsi": current_daily_rsi,
            "weekly_rsi": current_weekly_rsi,
            "weekly_rsi_ma": current_weekly_rsi_ma,
            "weekly_adx": current_weekly_adx,
            "monthly_rsi": current_monthly_rsi,
            "monthly_rsi_ma": current_monthly_rsi_ma,
            "monthly_adx": current_monthly_adx
        }

    except Exception as e:

        print(
            f"Indicator error: {e}"
        )

        return None


# ============================================================
# SEND EMAIL
# ============================================================

def send_email(results):

    username = os.getenv(
        "GMAIL_USERNAME"
    )

    app_password = os.getenv(
        "GMAIL_APP_PASSWORD"
    )

    recipient = os.getenv(
        "GMAIL_TO"
    )

    if not username or not app_password or not recipient:

        print(
            "Gmail settings not found"
        )

        return

    now = datetime.now(
        IST
    ).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    if not results:

        subject = (
            "NIFTY 500 DAILY SCANNER - No Matches"
        )

        body = f"""
NIFTY 500 DAILY SCANNER

No stocks matched all 5 conditions.

Scan time:
{now} IST
"""

    else:

        subject = (
            f"NIFTY 500 DAILY SCANNER - "
            f"{len(results)} Match(es)"
        )

        lines = []

        for result in results:

            lines.append(
                f"{result['symbol']}"
            )

        body = f"""
NIFTY 500 DAILY SCANNER

Stocks matching all 5 conditions:

{chr(10).join(lines)}

Total matches: {len(results)}

Scan time:
{now} IST
"""

    message = MIMEMultipart()

    message["From"] = username
    message["To"] = recipient
    message["Subject"] = subject

    message.attach(
        MIMEText(
            body,
            "plain"
        )
    )

    with smtplib.SMTP(
        "smtp.gmail.com",
        587
    ) as server:

        server.starttls()

        server.login(
            username,
            app_password
        )

        server.sendmail(
            username,
            recipient,
            message.as_string()
        )

    print(
        "Email sent successfully"
    )


# ============================================================
# MAIN SCANNER
# ============================================================

def main():

    print(
        "========================================"
    )

    print(
        "NIFTY 500 DAILY SCANNER"
    )

    print(
        "========================================"
    )

    print()

    symbols = get_nifty500_symbols()

    print(
        f"Total stocks: {len(symbols)}"
    )

    print()

    results = []

    total = len(symbols)

    for index, symbol in enumerate(
        symbols,
        start=1
    ):

        if index % 50 == 1:

            end = min(
                index + 49,
                total
            )

            print(
                f"Processing stocks "
                f"{index} to {end}..."
            )

        data = download_data(
            symbol
        )

        if data is None:
            continue

        values = check_conditions(
            data
        )

        if values is not None:

            clean_symbol = (
                symbol.replace(
                    ".NS",
                    ""
                )
            )

            values["symbol"] = (
                clean_symbol
            )

            results.append(
                values
            )

            print(
                f"MATCH: {clean_symbol}"
            )

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    print()

    print(
        "========================================"
    )

    print(
        "DAILY SCAN COMPLETE"
    )

    print(
        "========================================"
    )

    print()

    if results:

        print(
            f"{len(results)} stock(s) "
            "matched all 5 conditions:"
        )

        print()

        for result in results:

            print(
                result["symbol"]
            )

    else:

        print(
            "No stocks matched all 5 conditions."
        )

    print()

    send_email(
        results
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
