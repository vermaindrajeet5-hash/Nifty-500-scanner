import time
import requests
import pandas as pd
import yfinance as yf
from io import StringIO


# ============================================================
# SETTINGS
# ============================================================

YEARS = 10

START_DATE = (
    pd.Timestamp.today().normalize()
    - pd.DateOffset(years=YEARS)
)

END_DATE = pd.Timestamp.today().normalize()

OUTPUT_FILE = "daily_historical_pit_signals.csv"

MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/"
    "aditya-jha/nse-historical-membership/"
    "main/index_history/data/"
    "index_membership_history.csv"
)


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_dates(index):
    """
    Force all datetime indexes to the exact same format.

    This prevents errors such as:

    datetime64[s] vs datetime64[us]

    when using pandas merge_asof().
    """

    idx = pd.DatetimeIndex(index)

    # Remove timezone if present
    if idx.tz is not None:
        idx = idx.tz_localize(None)

    # Normalize to midnight
    idx = idx.normalize()

    # Force nanosecond precision
    idx = idx.astype("datetime64[ns]")

    return idx


# ============================================================
# RSI - WILDER
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
# ADX - WILDER
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
        (up_move > down_move) &
        (up_move > 0)
    )

    minus_mask = (
        (down_move > up_move) &
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
        100 *
        (plus_di - minus_di).abs() /
        denominator
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

    # --------------------------------------------------------
    # Convert membership dates
    # --------------------------------------------------------

    membership["valid_from"] = pd.to_datetime(
        membership["valid_from"],
        errors="coerce"
    )

    membership["valid_to"] = pd.to_datetime(
        membership["valid_to"],
        errors="coerce"
    )

    # Remove timezone if any
    if membership["valid_from"].dt.tz is not None:
        membership["valid_from"] = (
            membership["valid_from"]
            .dt.tz_localize(None)
        )

    if membership["valid_to"].dt.tz is not None:
        membership["valid_to"] = (
            membership["valid_to"]
            .dt.tz_localize(None)
        )

    # Force nanosecond precision
    membership["valid_from"] = (
        membership["valid_from"]
        .astype("datetime64[ns]")
    )

    membership["valid_to"] = (
        membership["valid_to"]
        .astype("datetime64[ns]")
    )

    # --------------------------------------------------------
    # Clean names
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Keep only NIFTY 500
    # --------------------------------------------------------

    membership = membership[
        membership["index_name"]
        .str.upper()
        .eq("NIFTY 500")
    ].copy()

    membership = membership[
        membership["valid_from"].notna()
    ].copy()

    membership.sort_values(
        [
            "valid_from",
            "symbol"
        ],
        inplace=True
    )

    print(
        f"Historical membership records: "
        f"{len(membership)}"
    )

    if not membership.empty:

        min_date = membership["valid_from"].min()

        max_valid_to = membership["valid_to"].max()

        print(
            "Membership coverage:",
            min_date.date(),
            "to",
            (
                max_valid_to
                if pd.notna(max_valid_to)
                else "current"
            )
        )

    return membership


# ============================================================
# GET MEMBERS ON A SPECIFIC DATE
# ============================================================

def members_on_date(
    membership,
    date
):

    date = pd.Timestamp(date)

    # Force same precision
    date = pd.Timestamp(
        date.to_datetime64().astype(
            "datetime64[ns]"
        )
    )

    active = membership[
        (membership["valid_from"] <= date) &
        (
            membership["valid_to"].isna() |
            (membership["valid_to"] > date)
        )
    ]

    return set(
        active["symbol"]
        .dropna()
        .tolist()
    )


# ============================================================
# GET ALL SYMBOLS REQUIRED FOR THE 10-YEAR TEST
# ============================================================

def get_historical_symbols(
    membership
):

    symbols = set()

    relevant = membership[
        (
            membership["valid_to"].isna() |
            (
                membership["valid_to"]
                >= START_DATE
            )
        ) &
        (
            membership["valid_from"]
            <= END_DATE
        )
    ]

    for symbol in relevant["symbol"]:

        symbol = str(symbol).strip().upper()

        if symbol:
            symbols.add(symbol)

    print(
        f"Unique historical NIFTY 500 symbols: "
        f"{len(symbols)}"
    )

    return sorted(symbols)


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

    weekly["RSI5"] = rsi_wilder(
        weekly["Close"],
        5
    )

    weekly["RSI_SMA14"] = (
        weekly["RSI5"]
        .rolling(14)
        .mean()
    )

    weekly["ADX14"] = adx_wilder(
        weekly["High"],
        weekly["Low"],
        weekly["Close"],
        14
    )

    weekly.index = normalize_dates(
        weekly.index
    )

    return weekly[
        [
            "RSI5",
            "RSI_SMA14",
            "ADX14"
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

    monthly["RSI5"] = rsi_wilder(
        monthly["Close"],
        5
    )

    monthly["RSI_SMA14"] = (
        monthly["RSI5"]
        .rolling(14)
        .mean()
    )

    monthly["ADX14"] = adx_wilder(
        monthly["High"],
        monthly["Low"],
        monthly["Close"],
        14
    )

    monthly.index = normalize_dates(
        monthly.index
    )

    return monthly[
        [
            "RSI5",
            "RSI_SMA14",
            "ADX14"
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

        print(
            f"Scanning {symbol}"
        )

        # ----------------------------------------------------
        # Download 11 years of daily data
        # ----------------------------------------------------

        data = yf.download(
            f"{symbol}.NS",
            period="11y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if data is None or data.empty:
            return []

        # ----------------------------------------------------
        # Flatten Yahoo MultiIndex
        # ----------------------------------------------------

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

        for column in required:

            if column not in data.columns:
                return []

        data = data[
            required
        ].copy()

        data.dropna(
            inplace=True
        )

        if len(data) < 200:
            return []

        # ----------------------------------------------------
        # IMPORTANT:
        # Normalize Yahoo dates BEFORE calculations.
        # ----------------------------------------------------

        data.index = normalize_dates(
            data.index
        )

        data.sort_index(
            inplace=True
        )

        # ----------------------------------------------------
        # Daily RSI
        # ----------------------------------------------------

        data["Daily_RSI5"] = rsi_wilder(
            data["Close"],
            5
        )

        # ----------------------------------------------------
        # Weekly indicators
        # ----------------------------------------------------

        weekly = build_weekly_data(
            data
        )

        # ----------------------------------------------------
        # Monthly indicators
        # ----------------------------------------------------

        monthly = build_monthly_data(
            data
        )

        # ----------------------------------------------------
        # Sort everything before merge_asof
        # ----------------------------------------------------

        data = data.sort_index()

        weekly = weekly.sort_index()

        monthly = monthly.sort_index()

        # ----------------------------------------------------
        # COMPLETED WEEKLY CANDLE
        #
        # Friday weekly candle becomes available only after
        # that week has completed.
        #
        # Shift by one day so Friday's incomplete/current
        # weekly candle is not accidentally used.
        # ----------------------------------------------------

        weekly_for_daily = weekly.copy()

        weekly_for_daily.index = (
            weekly_for_daily.index
            + pd.Timedelta(days=1)
        )

        weekly_for_daily.index = normalize_dates(
            weekly_for_daily.index
        )

        weekly_for_daily.sort_index(
            inplace=True
        )

        # ----------------------------------------------------
        # WEEKLY MERGE
        # ----------------------------------------------------

        data = pd.merge_asof(
            data,
            weekly_for_daily,
            left_index=True,
            right_index=True,
            direction="backward"
        )

        # ----------------------------------------------------
        # COMPLETED MONTHLY CANDLE
        # ----------------------------------------------------

        monthly_for_daily = monthly.copy()

        monthly_for_daily.index = (
            monthly_for_daily.index
            + pd.Timedelta(days=1)
        )

        monthly_for_daily.index = normalize_dates(
            monthly_for_daily.index
        )

        monthly_for_daily.sort_index(
            inplace=True
        )

        # ----------------------------------------------------
        # MONTHLY MERGE
        # ----------------------------------------------------

        data = pd.merge_asof(
            data,
            monthly_for_daily,
            left_index=True,
            right_index=True,
            direction="backward",
            suffixes=(
                "",
                "_monthly"
            )
        )

        # ----------------------------------------------------
        # Keep requested 10-year period
        # ----------------------------------------------------

        data = data[
            (
                data.index
                >= START_DATE
            ) &
            (
                data.index
                <= END_DATE
            )
        ].copy()

        if data.empty:
            return []

        # ----------------------------------------------------
        # EXACT 5 CONDITIONS
        #
        # 1. Monthly RSI(5)
        #    > Monthly RSI SMA(14)
        #
        # 2. Weekly RSI(5)
        #    > Weekly RSI SMA(14)
        #
        # 3. Monthly ADX(14) >= 25
        #
        # 4. Daily RSI(5) < 30
        #
        # 5. Weekly ADX(14) >= 25
        #
        # NO DAILY ADX CONDITION.
        # ----------------------------------------------------

        condition_1 = (
            data["RSI5_monthly"]
            >
            data["RSI_SMA14_monthly"]
        )

        condition_2 = (
            data["RSI5"]
            >
            data["RSI_SMA14"]
        )

        condition_3 = (
            data["ADX14_monthly"]
            >= 25
        )

        condition_4 = (
            data["Daily_RSI5"]
            < 30
        )

        condition_5 = (
            data["ADX14"]
            >= 25
        )

        signal = (
            condition_1 &
            condition_2 &
            condition_3 &
            condition_4 &
            condition_5
        )

        matches = data.loc[
            signal
        ]

        results = []

        # ----------------------------------------------------
        # POINT-IN-TIME NIFTY 500 MEMBERSHIP
        # ----------------------------------------------------

        for date in matches.index:

            active_members = members_on_date(
                membership,
                date
            )

            if symbol.upper() not in active_members:
                continue

            results.append({
                "Date": pd.Timestamp(
                    date
                ).strftime(
                    "%Y-%m-%d"
                ),
                "Stock": symbol.upper()
            })

        return results

    except Exception as e:

        print(
            f"ERROR {symbol}: {e}"
        )

        return []


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "NIFTY 500 POINT-IN-TIME "
        "10-YEAR DAILY SIGNAL SCANNER"
    )

    print("=" * 70)

    print(
        "Test period:",
        START_DATE.date(),
        "to",
        END_DATE.date()
    )

    print()

    # --------------------------------------------------------
    # Load historical membership
    # --------------------------------------------------------

    membership = load_membership()

    # --------------------------------------------------------
    # Get historical symbols
    # --------------------------------------------------------

    symbols = get_historical_symbols(
        membership
    )

    print()

    all_results = []

    # --------------------------------------------------------
    # Scan stocks
    # --------------------------------------------------------

    for number, symbol in enumerate(
        symbols,
        start=1
    ):

        print(
            f"[{number}/{len(symbols)}] {symbol}"
        )

        results = process_stock(
            symbol,
            membership
        )

        all_results.extend(
            results
        )

        # Conservative Yahoo request rate
        time.sleep(0.25)

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        all_results,
        columns=[
            "Date",
            "Stock"
        ]
    )

    if not result_df.empty:

        result_df.sort_values(
            [
                "Date",
                "Stock"
            ],
            inplace=True
        )

        result_df.drop_duplicates(
            inplace=True
        )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print("SCAN COMPLETE")

    print("=" * 70)

    print(
        f"Signals found: {len(result_df)}"
    )

    print(
        f"Output file: {OUTPUT_FILE}"
    )

    if not result_df.empty:

        print()

        print(
            result_df.to_string(
                index=False
            )
        )

    else:

        print()

        print(
            "No signals found."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
