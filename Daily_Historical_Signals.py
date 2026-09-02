import time
import requests
import pandas as pd
import yfinance as yf
from io import StringIO


# ============================================================
# SETTINGS
# ============================================================

YEARS = 10

NIFTY500_URL = (
    "https://www.niftyindices.com/IndexConstituent/"
    "ind_nifty500list.csv"
)

OUTPUT_FILE = "daily_historical_signals.csv"


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

    adx = dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    return adx


# ============================================================
# NIFTY 500 STOCK LIST
# ============================================================

def get_nifty500_symbols():

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        NIFTY500_URL,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    df = pd.read_csv(
        StringIO(response.text)
    )

    symbols = []

    for symbol in df["Symbol"].dropna():

        symbol = str(symbol).strip()

        if symbol:
            symbols.append(symbol + ".NS")

    return sorted(set(symbols))


# ============================================================
# BUILD COMPLETED WEEKLY DATA
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

    # Find the actual last trading day belonging
    # to each weekly candle.
    week_groups = data.groupby(
        pd.Grouper(freq="W-FRI")
    )

    completion_dates = []

    for _, group in week_groups:

        if not group.empty:
            completion_dates.append(
                group.index[-1]
            )

    completion_dates = pd.DatetimeIndex(
        completion_dates
    )

    weekly = weekly.iloc[
        -len(completion_dates):
    ].copy()

    weekly["CompletionDate"] = completion_dates

    weekly.set_index(
        "CompletionDate",
        inplace=True
    )

    return weekly[
        [
            "RSI5",
            "RSI_SMA14",
            "ADX14"
        ]
    ]


# ============================================================
# BUILD COMPLETED MONTHLY DATA
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

    # Find the actual last trading day belonging
    # to each monthly candle.
    month_groups = data.groupby(
        pd.Grouper(freq="ME")
    )

    completion_dates = []

    for _, group in month_groups:

        if not group.empty:
            completion_dates.append(
                group.index[-1]
            )

    completion_dates = pd.DatetimeIndex(
        completion_dates
    )

    monthly = monthly.iloc[
        -len(completion_dates):
    ].copy()

    monthly["CompletionDate"] = completion_dates

    monthly.set_index(
        "CompletionDate",
        inplace=True
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

def process_stock(symbol):

    try:

        print(f"Scanning {symbol}")

        data = yf.download(
            symbol,
            period=f"{YEARS}y",
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

        if len(data) < 100:
            return []

        # ----------------------------------------------------
        # DAILY RSI(5)
        # ----------------------------------------------------

        data["Daily_RSI5"] = rsi_wilder(
            data["Close"],
            5
        )

        # ----------------------------------------------------
        # WEEKLY INDICATORS
        # ----------------------------------------------------

        weekly = build_weekly_data(
            data
        )

        # ----------------------------------------------------
        # MONTHLY INDICATORS
        # ----------------------------------------------------

        monthly = build_monthly_data(
            data
        )

        # ----------------------------------------------------
        # POINT-IN-TIME MERGE
        #
        # For every daily date, use the most recent
        # COMPLETED weekly/monthly candle available
        # on that date.
        # ----------------------------------------------------

        data = data.sort_index()
        weekly = weekly.sort_index()
        monthly = monthly.sort_index()

        data = pd.merge_asof(
            data,
            weekly,
            left_index=True,
            right_index=True,
            direction="backward"
        )

        data = pd.merge_asof(
            data,
            monthly,
            left_index=True,
            right_index=True,
            direction="backward",
            suffixes=(
                "",
                "_monthly"
            )
        )

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

            results.append({
                "Date": pd.Timestamp(
                    date
                ).strftime(
                    "%Y-%m-%d"
                ),
                "Stock": symbol.replace(
                    ".NS",
                    ""
                )
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

    print("=" * 60)
    print("NIFTY 500 DAILY HISTORICAL SIGNAL SCANNER")
    print("=" * 60)

    print(
        f"Historical period: {YEARS} years"
    )

    print(
        "Conditions: 5"
    )

    print()

    symbols = get_nifty500_symbols()

    print(
        f"NIFTY 500 stocks found: {len(symbols)}"
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
            symbol
        )

        all_results.extend(
            results
        )

        time.sleep(0.2)

    # --------------------------------------------------------
    # CREATE OUTPUT
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
    print("=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)

    print(
        f"Signals found: {len(result_df)}"
    )

    print(
        f"Output file: {OUTPUT_FILE}"
    )

    print()

    if result_df.empty:

        print(
            "No historical signals found."
        )

    else:

        print(
            result_df.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
