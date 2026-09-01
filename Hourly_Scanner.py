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
        &
        (up_move > 0)
    )

    minus_condition = (
        (down_move > up_move)
        &
        (down_move > 0)
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
        *
        plus_dm.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False
        ).mean()
        / atr
    )

    minus_di = (
        100
        *
        minus_dm.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False
        ).mean()
        / atr
    )

    denominator = plus_di + minus_di

    dx = (
        100
        *
        (plus_di - minus_di).abs()
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

    return data.resample(
        "W-FRI"
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()


# ============================================================
# MAKE MONTHLY DATA
# ============================================================

def make_monthly(data):

    return data.resample(
        "ME"
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()


# ============================================================
# DOWNLOAD AND CLEAN DATA
# ============================================================

def download_data(symbol, period, interval):

    data = yf.download(
        symbol,
        period=period,
        interval=interval,
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

    data = data[
        required_columns
    ].dropna()

    return data


# ============================================================
# CHECK ONE STOCK
# ============================================================

def check_stock(symbol):

    try:

        # ----------------------------------------------------
        # DAILY DATA
        # ----------------------------------------------------

        daily = download_data(
            symbol,
            "3y",
            "1d"
        )

        if daily is None:
            return None

        if len(daily) < 100:
            return None

        # ----------------------------------------------------
        # WEEKLY / MONTHLY DATA
        # ----------------------------------------------------

        weekly = make_weekly(daily)
        monthly = make_monthly(daily)

        if len(weekly) < 30:
            return None

        if len(monthly) < 30:
            return None

        # ----------------------------------------------------
        # DAILY CONDITIONS
        # ----------------------------------------------------

        daily_rsi = rsi(
            daily["Close"],
            5
        )

        daily_rsi_sma = (
            daily_rsi
            .rolling(14)
            .mean()
        )

        daily_adx = adx(
            daily,
            14
        )

        d_rsi = daily_rsi.iloc[-1]
        d_rsi_sma = daily_rsi_sma.iloc[-1]
        d_adx = daily_adx.iloc[-1]

        if (
            pd.isna(d_rsi)
            or pd.isna(d_rsi_sma)
            or pd.isna(d_adx)
        ):
            return None

        if not (
            d_rsi > d_rsi_sma
            and d_adx >= 25
        ):
            return None

        # ----------------------------------------------------
        # WEEKLY CONDITIONS
        # ----------------------------------------------------

        weekly_rsi = rsi(
            weekly["Close"],
            5
        )

        weekly_rsi_sma = (
            weekly_rsi
            .rolling(14)
            .mean()
        )

        weekly_adx = adx(
            weekly,
            14
        )

        w_rsi = weekly_rsi.iloc[-1]
        w_rsi_sma = weekly_rsi_sma.iloc[-1]
        w_adx = weekly_adx.iloc[-1]

        if (
            pd.isna(w_rsi)
            or pd.isna(w_rsi_sma)
            or pd.isna(w_adx)
        ):
            return None

        if not (
            w_rsi > w_rsi_sma
            and w_adx >= 25
        ):
            return None

        # ----------------------------------------------------
        # MONTHLY CONDITIONS
        # ----------------------------------------------------

        monthly_rsi = rsi(
            monthly["Close"],
            5
        )

        monthly_rsi_sma = (
            monthly_rsi
            .rolling(14)
            .mean()
        )

        monthly_adx = adx(
            monthly,
            14
        )

        m_rsi = monthly_rsi.iloc[-1]
        m_rsi_sma = monthly_rsi_sma.iloc[-1]
        m_adx = monthly_adx.iloc[-1]

        if (
            pd.isna(m_rsi)
            or pd.isna(m_rsi_sma)
            or pd.isna(m_adx)
        ):
            return None

        if not (
            m_rsi > m_rsi_sma
            and m_adx >= 25
        ):
            return None

        # ----------------------------------------------------
        # 1-HOUR DATA
        # ----------------------------------------------------

        hourly = download_data(
            symbol,
            "60d",
            "1h"
        )

        if hourly is None:
            return None

        if len(hourly) < 30:
            return None

        # ----------------------------------------------------
        # CONVERT HOURLY DATA TO IST
        # ----------------------------------------------------

        if hourly.index.tz is None:

            hourly.index = (
                hourly.index
                .tz_localize("UTC")
            )

        hourly.index = (
            hourly.index
            .tz_convert(IST)
        )

        # ----------------------------------------------------
        # ONLY COMPLETED HOURLY CANDLES
        # ----------------------------------------------------

        now_ist = datetime.now(IST)

        completed = hourly[
            hourly.index + timedelta(hours=1)
            <= now_ist
        ].copy()

        if len(completed) < 10:
            return None

        # ----------------------------------------------------
        # HOURLY RSI
        # ----------------------------------------------------

        hourly_rsi = rsi(
            completed["Close"],
            5
        )

        previous_hour_rsi = (
            hourly_rsi.iloc[-1]
        )

        if pd.isna(previous_hour_rsi):
            return None

        # ----------------------------------------------------
        # FINAL HOURLY CONDITION
        #
        # Previous completed 1-hour RSI(5) < 30
        # ----------------------------------------------------

        if previous_hour_rsi >= 30:
            return None

        # ----------------------------------------------------
        # MATCH
        # ----------------------------------------------------

        return {
            "Stock":
                symbol.replace(".NS", ""),

            "Monthly RSI(5)":
                round(float(m_rsi), 2),

            "Monthly RSI SMA(14)":
                round(float(m_rsi_sma), 2),

            "Monthly ADX(14)":
                round(float(m_adx), 2),

            "Weekly RSI(5)":
                round(float(w_rsi), 2),

            "Weekly RSI SMA(14)":
                round(float(w_rsi_sma), 2),

            "Weekly ADX(14)":
                round(float(w_adx), 2),

            "Daily RSI(5)":
                round(float(d_rsi), 2),

            "Daily RSI SMA(14)":
                round(float(d_rsi_sma), 2),

            "Daily ADX(14)":
                round(float(d_adx), 2),

            "Previous 1H RSI(5)":
                round(
                    float(previous_hour_rsi),
                    2
                )
        }

    except Exception as e:

        print(
            f"Error processing {symbol}: {e}"
        )

        return None


# ============================================================
# GMAIL
# ============================================================

def send_email(subject, body):

    username = os.environ.get(
        "GMAIL_USERNAME"
    )

    app_password = os.environ.get(
        "GMAIL_APP_PASSWORD"
    )

    recipient = os.environ.get(
        "GMAIL_TO"
    )

    if not username:

        print(
            "GMAIL_USERNAME not found"
        )

        return

    if not app_password:

        print(
            "GMAIL_APP_PASSWORD not found"
        )

        return

    if not recipient:

        print(
            "GMAIL_TO not found"
        )

        return

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

    try:

        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465
        ) as server:

            server.login(
                username,
                app_password
            )

            server.send_message(
                message
            )

        print(
            "GMAIL: EMAIL SENT SUCCESSFULLY"
        )

    except Exception as e:

        print(
            f"GMAIL ERROR: {e}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "NIFTY 500 HOURLY SCANNER"
    )
    print("=" * 60)

    now = datetime.now(IST)

    print(
        "Scan time:",
        now.strftime(
            "%Y-%m-%d %H:%M:%S IST"
        )
    )

    print(
        "\nWaiting 10 seconds for "
        "market data to update..."
    )

    time.sleep(10)

    print(
        "\nGetting NIFTY 500 stocks..."
    )

    symbols = get_nifty500_symbols()

    print(
        f"Found {len(symbols)} "
        "NIFTY 500 stocks."
    )

    results = []

    for i, symbol in enumerate(
        symbols,
        1
    ):

        print(
            f"[{i}/{len(symbols)}] "
            f"Checking {symbol}"
        )

        result = check_stock(
            symbol
        )

        if result:

            results.append(
                result
            )

            print(
                f"  MATCH: {symbol}"
            )

    # --------------------------------------------------------
    # SCAN COMPLETE
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "HOURLY SCAN COMPLETE"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # NO MATCHES
    # --------------------------------------------------------

    if not results:

        print(
            "\nNo stocks matched "
            "all conditions."
        )

        send_email(
            "NIFTY 500 Hourly Scanner - No Matches",
            (
                "NIFTY 500 HOURLY SCANNER\n\n"
                "No stocks matched all conditions.\n\n"
                f"Scan time: "
                f"{now.strftime('%Y-%m-%d %H:%M:%S IST')}"
            )
        )

        return

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    print(
        f"\n{len(result_df)} "
        "stock(s) matched all conditions:\n"
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    result_df.to_csv(
        "hourly_signals.csv",
        index=False
    )

    print(
        "\nSaved results to "
        "hourly_signals.csv"
    )

    # --------------------------------------------------------
    # EMAIL RESULTS
    # --------------------------------------------------------

    message = (
        "NIFTY 500 HOURLY SCANNER RESULTS\n\n"
        f"Scan time: "
        f"{now.strftime('%Y-%m-%d %H:%M:%S IST')}\n\n"
        f"{len(result_df)} stock(s) matched "
        "all conditions.\n\n"
        +
        result_df.to_string(
            index=False
        )
    )

    send_email(
        "NIFTY 500 Hourly Scanner Results",
        message
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
