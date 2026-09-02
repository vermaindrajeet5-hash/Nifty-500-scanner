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

NIFTY500_URL = (
    "https://www.niftyindices.com/IndexConstituent/"
    "ind_nifty500list.csv"
)


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

    membership["valid_from"] = pd.to_datetime(
        membership["valid_from"],
        errors="coerce"
    )

    membership["valid_to"] = pd.to_datetime(
        membership["valid_to"],
        errors="coerce"
    )

    # Handle possible naming variations
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
        membership["index_name"].str.upper()
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

        print(
            "Membership coverage:",
            membership["valid_from"].min().date(),
            "to",
            (
                membership["valid_to"]
                .max()
                if membership["valid_to"].notna().any()
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
        # Get enough history.
        #
        # We request 11 years so the monthly indicators have
        # additional warm-up data before the requested
        # 10-year test period.
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
        # Point-in-time merge
        #
        # Only completed weekly/monthly candles are used.
        # ----------------------------------------------------

        data = data.sort_index()
        weekly = weekly.sort_index()
        monthly = monthly.sort_index()

        # A daily date receives the latest COMPLETED
        # weekly candle before that date.
        weekly_for_daily = weekly.copy()

        weekly_for_daily.index = (
            weekly_for_daily.index
            + pd.Timedelta(days=1)
        )

        data = pd.merge_asof(
            data,
            weekly_for_daily,
            left_index=True,
            right_index=True,
            direction="backward"
        )

        # A daily date receives the latest COMPLETED
        # monthly candle before that date.
        monthly_for_daily = monthly.copy()

        monthly_for_daily.index = (
            monthly_for_daily.index
            + pd.Timedelta(days=1)
        )

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
        # Keep only requested 10-year period
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

        # ----------------------------------------------------
        # EXACT 5 CONDITIONS
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

        for date in matches.index:

            # Point-in-time NIFTY 500 membership
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

    membership = load_membership()

    symbols = get_historical_symbols(
        membership
    )

    print()

    all_results = []

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

        # Keep request rate conservative.
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


if __name__ == "__main__":
    main()
