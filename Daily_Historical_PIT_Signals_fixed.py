import os
import smtplib
from io import StringIO
from email.message import EmailMessage

import pandas as pd
import requests
import yfinance as yf


# ============================================================
# SETTINGS
# ============================================================

YEARS = 10

END_DATE = pd.Timestamp.today().normalize()

START_DATE = END_DATE - pd.DateOffset(years=YEARS)

OUTPUT_FILE = "daily_historical_pit_signals.csv"

MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/"
    "aditya-jha/nse-historical-membership/"
    "main/index_history/data/index_membership_history.csv"
)


# ============================================================
# FORCE ALL DATETIME INDEXES TO THE SAME TYPE
# ============================================================

def clean_datetime_index(index):

    index = pd.DatetimeIndex(
        pd.to_datetime(index)
    )

    if index.tz is not None:
        index = index.tz_localize(None)

    index = index.normalize()

    # Important:
    # merge_asof requires exactly matching datetime dtypes.
    return index.astype("datetime64[ns]")


# ============================================================
# RSI WILDER
# ============================================================

def rsi_wilder(series, period=5):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


# ============================================================
# ADX WILDER
# ============================================================

def adx_wilder(high, low, close, period=14):

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
        index=high.index
    )

    minus_dm = pd.Series(
        0.0,
        index=high.index
    )

    plus_mask = (
        (up_move > down_move)
        &
        (up_move > 0)
    )

    minus_mask = (
        (down_move > up_move)
        &
        (down_move > 0)
    )

    plus_dm.loc[plus_mask] = up_move.loc[plus_mask]

    minus_dm.loc[minus_mask] = down_move.loc[minus_mask]

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    plus_dm_smooth = plus_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    minus_dm_smooth = minus_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    plus_di = 100 * plus_dm_smooth / atr

    minus_di = 100 * minus_dm_smooth / atr

    denominator = plus_di + minus_di

    dx = (
        100
        * (plus_di - minus_di).abs()
        / denominator
    )

    return dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


# ============================================================
# LOAD HISTORICAL NIFTY 500 MEMBERSHIP
# ============================================================

def load_membership():

    print("Loading historical NIFTY 500 membership...")

    response = requests.get(
        MEMBERSHIP_URL,
        timeout=60
    )

    response.raise_for_status()

    membership = pd.read_csv(
        StringIO(response.text)
    )

    membership["valid_from"] = (
        pd.to_datetime(
            membership["valid_from"],
            errors="coerce",
            utc=True
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )

    membership["valid_to"] = (
        pd.to_datetime(
            membership["valid_to"],
            errors="coerce",
            utc=True
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )

    membership["index_name"] = (
        membership["index_name"]
        .astype(str)
        .str.strip()
    )

    membership["symbol"] = (
        membership["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    membership = membership[
        membership["index_name"]
        .str.upper()
        .eq("NIFTY 500")
    ].copy()

    membership = membership[
        membership["valid_from"].notna()
    ].copy()

    print(
        "Membership records:",
        len(membership)
    )

    return membership


# ============================================================
# CHECK MEMBERSHIP ON SIGNAL DATE
# ============================================================

def is_member_on_date(
    membership,
    symbol,
    date
):

    date = pd.Timestamp(date).normalize()

    rows = membership[
        (membership["symbol"] == symbol.upper())
        &
        (membership["valid_from"] <= date)
        &
        (
            membership["valid_to"].isna()
            |
            (membership["valid_to"] > date)
        )
    ]

    return not rows.empty


# ============================================================
# GET HISTORICAL SYMBOLS
# ============================================================

def get_historical_symbols(membership):

    relevant = membership[
        (
            membership["valid_to"].isna()
            |
            (membership["valid_to"] >= START_DATE)
        )
        &
        (membership["valid_from"] <= END_DATE)
    ]

    symbols = sorted(
        set(
            relevant["symbol"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
        )
    )

    print(
        "Historical NIFTY 500 symbols:",
        len(symbols)
    )

    return symbols


# ============================================================
# BUILD WEEKLY DATA
# ============================================================

def build_weekly_data(data):

    weekly = data.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    })

    weekly.dropna(inplace=True)

    weekly["Weekly_RSI5"] = rsi_wilder(
        weekly["Close"],
        5
    )

    weekly["Weekly_RSI_SMA14"] = (
        weekly["Weekly_RSI5"]
        .rolling(14)
        .mean()
    )

    weekly["Weekly_ADX14"] = adx_wilder(
        weekly["High"],
        weekly["Low"],
        weekly["Close"],
        14
    )

    weekly.index = clean_datetime_index(
        weekly.index
    )

    return weekly[
        [
            "Weekly_RSI5",
            "Weekly_RSI_SMA14",
            "Weekly_ADX14"
        ]
    ]


# ============================================================
# BUILD MONTHLY DATA
# ============================================================

def build_monthly_data(data):

    monthly = data.resample("ME").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    })

    monthly.dropna(inplace=True)

    monthly["Monthly_RSI5"] = rsi_wilder(
        monthly["Close"],
        5
    )

    monthly["Monthly_RSI_SMA14"] = (
        monthly["Monthly_RSI5"]
        .rolling(14)
        .mean()
    )

    monthly["Monthly_ADX14"] = adx_wilder(
        monthly["High"],
        monthly["Low"],
        monthly["Close"],
        14
    )

    monthly.index = clean_datetime_index(
        monthly.index
    )

    return monthly[
        [
            "Monthly_RSI5",
            "Monthly_RSI_SMA14",
            "Monthly_ADX14"
        ]
    ]


# ============================================================
# PROCESS ONE STOCK
# ============================================================

def process_stock(
    symbol,
    membership
):

    try:

        print("Scanning:", symbol)

        download_start = (
            START_DATE -
            pd.DateOffset(years=2)
        )

        download_end = (
            END_DATE +
            pd.Timedelta(days=1)
        )

        data = yf.download(
            f"{symbol}.NS",
            start=download_start.strftime("%Y-%m-%d"),
            end=download_end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if data is None or data.empty:

            print("  No Yahoo data")

            return []

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            data.columns = (
                data.columns
                .get_level_values(0)
            )

        required = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        if not all(
            column in data.columns
            for column in required
        ):

            print("  Missing OHLC")

            return []

        data = data[
            required
        ].copy()

        data.dropna(
            inplace=True
        )

        if len(data) < 200:

            print("  Not enough data")

            return []

        # ====================================================
        # IMPORTANT DATETIME FIX
        # ====================================================

        data.index = clean_datetime_index(
            data.index
        )

        data = data[
            ~data.index.duplicated(
                keep="last"
            )
        ]

        data.sort_index(
            inplace=True
        )

        # ====================================================
        # DAILY RSI(5)
        # ====================================================

        data["Daily_RSI5"] = rsi_wilder(
            data["Close"],
            5
        )

        # ====================================================
        # WEEKLY INDICATORS
        # ====================================================

        weekly = build_weekly_data(
            data
        )

        # ====================================================
        # MONTHLY INDICATORS
        # ====================================================

        monthly = build_monthly_data(
            data
        )

        # ====================================================
        # COMPLETED WEEK ONLY
        # ====================================================

        weekly_for_daily = weekly.copy()

        weekly_for_daily.index = clean_datetime_index(
            weekly_for_daily.index
            + pd.Timedelta(days=1)
        )

        # Explicit final dtype check
        data.index = clean_datetime_index(
            data.index
        )

        weekly_for_daily.index = clean_datetime_index(
            weekly_for_daily.index
        )

        data = pd.merge_asof(
            data.sort_index(),
            weekly_for_daily.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward"
        )

        # ====================================================
        # COMPLETED MONTH ONLY
        # ====================================================

        monthly_for_daily = monthly.copy()

        monthly_for_daily.index = clean_datetime_index(
            monthly_for_daily.index
            + pd.Timedelta(days=1)
        )

        data.index = clean_datetime_index(
            data.index
        )

        monthly_for_daily.index = clean_datetime_index(
            monthly_for_daily.index
        )

        data = pd.merge_asof(
            data.sort_index(),
            monthly_for_daily.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward"
        )

        # ====================================================
        # REQUESTED 10-YEAR PERIOD
        # ====================================================

        data = data[
            (data.index >= START_DATE)
            &
            (data.index <= END_DATE)
        ].copy()

        if data.empty:

            return []

        # ====================================================
        # CONDITION 1
        # MONTHLY RSI(5) > MONTHLY RSI SMA14
        # ====================================================

        condition_1 = (
            data["Monthly_RSI5"]
            >
            data["Monthly_RSI_SMA14"]
        )

        # ====================================================
        # CONDITION 2
        # WEEKLY RSI(5) > WEEKLY RSI SMA14
        # ====================================================

        condition_2 = (
            data["Weekly_RSI5"]
            >
            data["Weekly_RSI_SMA14"]
        )

        # ====================================================
        # CONDITION 3
        # MONTHLY ADX(14) >= 25
        # ====================================================

        condition_3 = (
            data["Monthly_ADX14"]
            >= 25
        )

        # ====================================================
        # CONDITION 4
        # DAILY RSI(5) < 30
        # ====================================================

        condition_4 = (
            data["Daily_RSI5"]
            < 30
        )

        # ====================================================
        # CONDITION 5
        # WEEKLY ADX(14) >= 25
        # ====================================================

        condition_5 = (
            data["Weekly_ADX14"]
            >= 25
        )

        # ====================================================
        # ALL 5 ORIGINAL CONDITIONS
        # ====================================================

        signal = (
            condition_1
            &
            condition_2
            &
            condition_3
            &
            condition_4
            &
            condition_5
        )

        matches = data.loc[
            signal
        ]

        if matches.empty:

            return []

        results = []

        # ====================================================
        # POINT-IN-TIME NIFTY 500 FILTER
        # ====================================================

        for date in matches.index:

            if is_member_on_date(
                membership,
                symbol,
                date
            ):

                results.append({
                    "Date": pd.Timestamp(
                        date
                    ).strftime(
                        "%Y-%m-%d"
                    ),
                    "Stock": symbol.upper()
                })

        return results

    except Exception as error:

        print(
            "ERROR",
            symbol,
            ":",
            error
        )

        return []


# ============================================================
# SEND RESULTS BY GMAIL
# ============================================================

def send_email(output):

    gmail_username = os.environ.get(
        "GMAIL_USERNAME"
    )

    gmail_password = os.environ.get(
        "GMAIL_APP_PASSWORD"
    )

    gmail_to = os.environ.get(
        "GMAIL_TO"
    )

    if not gmail_username:

        raise RuntimeError(
            "GMAIL_USERNAME secret is missing"
        )

    if not gmail_password:

        raise RuntimeError(
            "GMAIL_APP_PASSWORD secret is missing"
        )

    if not gmail_to:

        raise RuntimeError(
            "GMAIL_TO secret is missing"
        )

    message = EmailMessage()

    message["Subject"] = (
        "NIFTY 500 Historical Daily PIT Signals"
    )

    message["From"] = gmail_username

    message["To"] = gmail_to

    if output.empty:

        body = (
            "NIFTY 500 HISTORICAL DAILY PIT SCANNER\n\n"
            "No stocks matched all 5 conditions.\n\n"
            f"Period: {START_DATE:%Y-%m-%d} "
            f"to {END_DATE:%Y-%m-%d}\n\n"
            "Conditions:\n"
            "1. Monthly RSI(5) > Monthly RSI SMA14\n"
            "2. Weekly RSI(5) > Weekly RSI SMA14\n"
            "3. Monthly ADX(14) >= 25\n"
            "4. Daily RSI(5) < 30\n"
            "5. Weekly ADX(14) >= 25"
        )

    else:

        preview = output.head(50).to_string(
            index=False
        )

        body = (
            "NIFTY 500 HISTORICAL DAILY PIT SCANNER\n\n"
            f"Period: {START_DATE:%Y-%m-%d} "
            f"to {END_DATE:%Y-%m-%d}\n"
            f"Total signals: {len(output)}\n\n"
            "Conditions:\n"
            "1. Monthly RSI(5) > Monthly RSI SMA14\n"
            "2. Weekly RSI(5) > Weekly RSI SMA14\n"
            "3. Monthly ADX(14) >= 25\n"
            "4. Daily RSI(5) < 30\n"
            "5. Weekly ADX(14) >= 25\n\n"
            "First 50 signals:\n\n"
            f"{preview}\n\n"
            "The complete Date + Stock list is attached."
        )

    message.set_content(
        body
    )

    with open(
        OUTPUT_FILE,
        "rb"
    ) as file:

        message.add_attachment(
            file.read(),
            maintype="text",
            subtype="csv",
            filename=OUTPUT_FILE
        )

    print(
        "Sending email..."
    )

    with smtplib.SMTP_SSL(
        "smtp.gmail.com",
        465,
        timeout=60
    ) as smtp:

        smtp.login(
            gmail_username,
            gmail_password
        )

        smtp.send_message(
            message
        )

    print(
        "GMAIL: EMAIL SENT SUCCESSFULLY"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("NIFTY 500 HISTORICAL DAILY PIT SCANNER")
    print("=" * 60)

    print(
        "Start date:",
        START_DATE.strftime("%Y-%m-%d")
    )

    print(
        "End date:",
        END_DATE.strftime("%Y-%m-%d")
    )

    print()
    print("Conditions: 5")
    print("No Supertrend condition")
    print()

    # ========================================================
    # CREATE CSV IMMEDIATELY
    # ========================================================

    pd.DataFrame(
        columns=[
            "Date",
            "Stock"
        ]
    ).to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "CSV file created:",
        OUTPUT_FILE
    )

    # ========================================================
    # LOAD MEMBERSHIP
    # ========================================================

    membership = load_membership()

    # ========================================================
    # GET HISTORICAL SYMBOLS
    # ========================================================

    symbols = get_historical_symbols(
        membership
    )

    if not symbols:

        raise RuntimeError(
            "No historical NIFTY 500 symbols found."
        )

    # ========================================================
    # SCAN ALL SYMBOLS
    # ========================================================

    all_results = []

    total = len(symbols)

    for number, symbol in enumerate(
        symbols,
        start=1
    ):

        print(
            f"[{number}/{total}] {symbol}"
        )

        results = process_stock(
            symbol,
            membership
        )

        if results:

            all_results.extend(
                results
            )

            print(
                "  Signals found:",
                len(results)
            )

    # ========================================================
    # BUILD OUTPUT
    # ========================================================

    if all_results:

        output = pd.DataFrame(
            all_results,
            columns=[
                "Date",
                "Stock"
            ]
        )

        output.drop_duplicates(
            inplace=True
        )

        output.sort_values(
            ["Date", "Stock"],
            inplace=True
        )

        output.reset_index(
            drop=True,
            inplace=True
        )

    else:

        output = pd.DataFrame(
            columns=[
                "Date",
                "Stock"
            ]
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)

    print(
        "Total signals:",
        len(output)
    )

    print(
        "Output file:",
        OUTPUT_FILE
    )

    if output.empty:

        print(
            "No stocks matched all 5 conditions."
        )

    else:

        print()
        print(
            output.head(50).to_string(
                index=False
            )
        )

    # ========================================================
    # SEND EMAIL
    # ========================================================

    send_email(
        output
    )


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
