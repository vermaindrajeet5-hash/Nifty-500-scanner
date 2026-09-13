import time
import requests
import pandas as pd
import yfinance as yf
from io import StringIO


# ============================================================
# SETTINGS
# ============================================================

YEARS = 10

END_DATE = pd.Timestamp.today().normalize()

START_DATE = (
    END_DATE -
    pd.DateOffset(years=YEARS)
)

OUTPUT_FILE = "daily_historical_pit_signals.csv"

MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/"
    "aditya-jha/nse-historical-membership/"
    "main/index_history/data/"
    "index_membership_history.csv"
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

    return 100 - (
        100 / (1 + rs)
    )


# ============================================================
# ADX - WILDER
# ============================================================

def adx_wilder(
    high,
    low,
    close,
    period=14
):

    previous_close = close.shift(1)

    tr1 = high - low

    tr2 = (
        high - previous_close
    ).abs()

    tr3 = (
        low - previous_close
    ).abs()

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

    plus_dm.loc[plus_mask] = (
        up_move.loc[plus_mask]
    )

    minus_dm.loc[minus_mask] = (
        down_move.loc[minus_mask]
    )

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

    plus_di = (
        100 *
        plus_dm_smooth /
        atr
    )

    minus_di = (
        100 *
        minus_dm_smooth /
        atr
    )

    denominator = (
        plus_di +
        minus_di
    )

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
# SUPERTREND
#
# ATR PERIOD = 10
# MULTIPLIER = 1
#
# 1  = GREEN
# -1 = RED
# ============================================================

def supertrend(
    high,
    low,
    close,
    period=10,
    multiplier=1
):

    hl2 = (
        high + low
    ) / 2

    tr1 = high - low

    tr2 = (
        high - close.shift(1)
    ).abs()

    tr3 = (
        low - close.shift(1)
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    basic_upper = (
        hl2 +
        multiplier * atr
    )

    basic_lower = (
        hl2 -
        multiplier * atr
    )

    final_upper = basic_upper.copy()

    final_lower = basic_lower.copy()

    direction = pd.Series(
        1,
        index=close.index,
        dtype="int64"
    )

    for i in range(1, len(close)):

        if (
            basic_upper.iloc[i]
            < final_upper.iloc[i - 1]
            or
            close.iloc[i - 1]
            > final_upper.iloc[i - 1]
        ):

            final_upper.iloc[i] = (
                basic_upper.iloc[i]
            )

        else:

            final_upper.iloc[i] = (
                final_upper.iloc[i - 1]
            )

        if (
            basic_lower.iloc[i]
            > final_lower.iloc[i - 1]
            or
            close.iloc[i - 1]
            < final_lower.iloc[i - 1]
        ):

            final_lower.iloc[i] = (
                basic_lower.iloc[i]
            )

        else:

            final_lower.iloc[i] = (
                final_lower.iloc[i - 1]
            )

        if direction.iloc[i - 1] == -1:

            if (
                close.iloc[i]
                > final_upper.iloc[i]
            ):

                direction.iloc[i] = 1

            else:

                direction.iloc[i] = -1

        else:

            if (
                close.iloc[i]
                < final_lower.iloc[i]
            ):

                direction.iloc[i] = -1

            else:

                direction.iloc[i] = 1

    return direction


# ============================================================
# LOAD HISTORICAL NIFTY 500 MEMBERSHIP
# ============================================================

def load_membership():

    print(
        "Loading historical NIFTY 500 membership..."
    )

    response = requests.get(
        MEMBERSHIP_URL,
        timeout=60
    )

    response.raise_for_status()

    membership = pd.read_csv(
        StringIO(response.text)
    )

    # --------------------------------------------------------
    # Convert membership dates to timezone-free datetime
    # --------------------------------------------------------

    membership["valid_from"] = (
        pd.to_datetime(
            membership["valid_from"],
            errors="coerce",
            utc=True
        )
        .dt.tz_localize(None)
    )

    membership["valid_to"] = (
        pd.to_datetime(
            membership["valid_to"],
            errors="coerce",
            utc=True
        )
        .dt.tz_localize(None)
    )

    membership["valid_from"] = (
        membership["valid_from"].dt.normalize()
    )

    membership["valid_to"] = (
        membership["valid_to"].dt.normalize()
    )

    # --------------------------------------------------------
    # Clean index names
    # --------------------------------------------------------

    membership["index_name"] = (
        membership["index_name"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Clean symbols
    # --------------------------------------------------------

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
        "Historical membership records:",
        len(membership)
    )

    if not membership.empty:

        print(
            "Membership coverage:",
            membership["valid_from"]
            .min()
            .date(),
            "to",
            (
                membership["valid_to"]
                .max()
                .date()
                if membership["valid_to"]
                .notna()
                .any()
                else "current"
            )
        )

    return membership


# ============================================================
# CHECK NIFTY 500 MEMBERSHIP ON A DATE
# ============================================================

def members_on_date(
    membership,
    date
):

    date = pd.Timestamp(
        date
    ).normalize()

    active = membership[
        (
            membership["valid_from"]
            <= date
        )
        &
        (
            membership["valid_to"].isna()
            |
            (
                membership["valid_to"]
                > date
            )
        )
    ]

    return set(
        active["symbol"]
        .dropna()
        .tolist()
    )


# ============================================================
# GET ALL HISTORICAL SYMBOLS
# ============================================================

def get_historical_symbols(
    membership
):

    relevant = membership[
        (
            membership["valid_to"].isna()
            |
            (
                membership["valid_to"]
                >= START_DATE
            )
        )
        &
        (
            membership["valid_from"]
            <= END_DATE
        )
    ]

    symbols = set()

    for symbol in relevant["symbol"]:

        symbol = (
            str(symbol)
            .strip()
            .upper()
        )

        if symbol:

            symbols.add(symbol)

    print(
        "Unique historical NIFTY 500 symbols:",
        len(symbols)
    )

    return sorted(symbols)


# ============================================================
# BUILD WEEKLY DATA
# ============================================================

def build_weekly_data(data):

    weekly = data.resample(
        "W-FRI"
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    })

    weekly.dropna(
        inplace=True
    )

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

    monthly = data.resample(
        "ME"
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    })

    monthly.dropna(
        inplace=True
    )

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
        # Request 11 years of data.
        #
        # One extra year provides warm-up for indicators.
        # ----------------------------------------------------

        download_start = (
            START_DATE -
            pd.DateOffset(years=1)
        )

        download_end = (
            END_DATE +
            pd.Timedelta(days=1)
        )

        data = yf.download(
            f"{symbol}.NS",
            start=download_start.strftime(
                "%Y-%m-%d"
            ),
            end=download_end.strftime(
                "%Y-%m-%d"
            ),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if data is None or data.empty:

            return []

        # ----------------------------------------------------
        # Handle Yahoo MultiIndex
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
        # Make dates timezone-free
        # ----------------------------------------------------

        index = pd.DatetimeIndex(
            data.index
        )

        if index.tz is not None:

            index = index.tz_localize(
                None
            )

        data.index = index.normalize()

        data.sort_index(
            inplace=True
        )

        # ----------------------------------------------------
        # DAILY RSI
        # ----------------------------------------------------

        data["Daily_RSI5"] = rsi_wilder(
            data["Close"],
            5
        )

        # ----------------------------------------------------
        # DAILY SUPERTREND (10,1)
        # ----------------------------------------------------

        data["Supertrend_Daily"] = (
            supertrend(
                data["High"],
                data["Low"],
                data["Close"],
                period=10,
                multiplier=1
            )
        )

        # ----------------------------------------------------
        # WEEKLY DATA
        # ----------------------------------------------------

        weekly = build_weekly_data(
            data
        )

        # ----------------------------------------------------
        # MONTHLY DATA
        # ----------------------------------------------------

        monthly = build_monthly_data(
            data
        )

        # ----------------------------------------------------
        # Normalize indicator indexes
        # ----------------------------------------------------

        weekly.index = (
            pd.DatetimeIndex(
                weekly.index
            ).normalize()
        )

        monthly.index = (
            pd.DatetimeIndex(
                monthly.index
            ).normalize()
        )

        data = data.sort_index()

        weekly = weekly.sort_index()

        monthly = monthly.sort_index()

        # ----------------------------------------------------
        # ONLY COMPLETED WEEKLY CANDLE
        #
        # Friday's completed week becomes available
        # from the following day.
        # ----------------------------------------------------

        weekly_for_daily = (
            weekly.copy()
        )

        weekly_for_daily.index = (
            weekly_for_daily.index
            + pd.Timedelta(days=1)
        )

        # ----------------------------------------------------
        # Merge weekly indicators
        # ----------------------------------------------------

        data = pd.merge_asof(
            data.sort_index(),
            weekly_for_daily.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward"
        )

        # ----------------------------------------------------
        # ONLY COMPLETED MONTHLY CANDLE
        #
        # Month-end values become available from the
        # following day.
        # ----------------------------------------------------

        monthly_for_daily = (
            monthly.copy()
        )

        monthly_for_daily.index = (
            monthly_for_daily.index
            + pd.Timedelta(days=1)
        )

        # ----------------------------------------------------
        # Merge monthly indicators
        # ----------------------------------------------------

        data = pd.merge_asof(
            data.sort_index(),
            monthly_for_daily.sort_index(),
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
            )
            &
            (
                data.index
                <= END_DATE
            )
        ].copy()

        if data.empty:

            return []

        # ====================================================
        # CONDITION 1
        #
        # MONTHLY RSI(5) > MONTHLY RSI SMA14
        # ====================================================

        condition_1 = (
            data["RSI5_monthly"]
            >
            data["RSI_SMA14_monthly"]
        )

        # ====================================================
        # CONDITION 2
        #
        # WEEKLY RSI(5) > WEEKLY RSI SMA14
        # ====================================================

        condition_2 = (
            data["RSI5"]
            >
            data["RSI_SMA14"]
        )

        # ====================================================
        # CONDITION 3
        #
        # MONTHLY ADX14 >= 25
        # ====================================================

        condition_3 = (
            data["ADX14_monthly"]
            >= 25
        )

        # ====================================================
        # CONDITION 4
        #
        # DAILY RSI(5) < 30
        # ====================================================

        condition_4 = (
            data["Daily_RSI5"]
            < 30
        )

        # ====================================================
        # CONDITION 5
        #
        # WEEKLY ADX14 >= 25
        # ====================================================

        condition_5 = (
            data["ADX14"]
            >= 25
        )

        # ====================================================
        # CONDITION 6
        #
        # DAILY SUPERTREND (10,1) = GREEN
        # ====================================================

        condition_6 = (
            data["Supertrend_Daily"]
            == 1
        )

        # ====================================================
        # ALL 6 CONDITIONS
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
            &
            condition_6
        )

        matches = data.loc[
            signal
        ]

        results = []

        # ----------------------------------------------------
        # POINT-IN-TIME NIFTY 500 CHECK
        # ----------------------------------------------------

        for date in matches.index:

            active_members = (
                members_on_date(
                    membership,
                    date
                )
            )

            if (
                symbol.upper()
                not in active_members
            ):

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


# ========================================
