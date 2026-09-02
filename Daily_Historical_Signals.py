import os
import time
import requests
import pandas as pd
import yfinance as yf


# ============================================================
# SETTINGS
# ============================================================

YEARS = 6

NIFTY500_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"

OUTPUT_FILE = "daily_historical_signals.csv"


# ============================================================
# INDICATORS
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

    rsi = 100 - (100 / (1 + rs))

    return rsi


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

    plus_dm[
        (up_move > down_move) &
        (up_move > 0)
    ] = up_move[
        (up_move > down_move) &
        (up_move > 0)
    ]

    minus_dm[
        (down_move > up_move) &
        (down_move > 0)
    ] = down_move[
        (down_move > up_move) &
        (down_move > 0)
    ]

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
# GET NIFTY 500 STOCK LIST
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

    from io import StringIO

    df = pd.read_csv(
        StringIO(response.text)
    )

    symbols = []

    for symbol in df["Symbol"].dropna():

        symbol = str(symbol).strip()

        if symbol:
            symbols.append(symbol + ".NS")

    return sorted(list(set(symbols)))


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

        # Handle yfinance multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        required = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        for column in required:
            if column not in data.columns:
                return []

        data = data[required].copy()

        data.dropna(inplace=True)

        if len(data) < 100:
            return []

        # ----------------------------------------------------
        # DAILY RSI(5)
        # ----------------------------------------------------

        data["Daily_RSI_5"] = rsi_wilder(
            data["Close"],
            5
        )

        # ----------------------------------------------------
        # WEEKLY DATA
        # ----------------------------------------------------

        weekly = data.resample("W-FRI").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last"
        })

        weekly.dropna(inplace=True)

        weekly["Weekly_RSI_5"] = rsi_wilder(
            weekly["Close"],
            5
        )

        weekly["Weekly_RSI_SMA14"] = (
            weekly["Weekly_RSI_5"]
            .rolling(14)
            .mean()
        )

        weekly["Weekly_ADX14"] = adx_wilder(
            weekly["High"],
            weekly["Low"],
            weekly["Close"],
            14
        )

        # Shift weekly indicators by one completed week.
        #
        # This prevents future weekly information from being
        # used in a historical daily signal.
        weekly["Weekly_RSI_5"] = weekly["Weekly_RSI_5"].shift(1)
        weekly["Weekly_RSI_SMA14"] = weekly[
            "Weekly_RSI_SMA14"
        ].shift(1)

        weekly["Weekly_ADX14"] = weekly[
            "Weekly_ADX14"
        ].shift(1)

        # ----------------------------------------------------
        # MONTHLY DATA
        # ----------------------------------------------------

        monthly = data.resample("ME").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last"
        })

        monthly.dropna(inplace=True)

        monthly["Monthly_RSI_5"] = rsi_wilder(
            monthly["Close"],
            5
        )

        monthly["Monthly_RSI_SMA14"] = (
            monthly["Monthly_RSI_5"]
            .rolling(14)
            .mean()
        )

        monthly["Monthly_ADX14"] = adx_wilder(
            monthly["High"],
            monthly["Low"],
            monthly["Close"],
            14
        )

        # Prevent future monthly information
        monthly["Monthly_RSI_5"] = monthly[
            "Monthly_RSI_5"
        ].shift(1)

        monthly["Monthly_RSI_SMA14"] = monthly[
            "Monthly_RSI_SMA14"
        ].shift(1)

        monthly["Monthly_ADX14"] = monthly[
            "Monthly_ADX14"
        ].shift(1)

        # ----------------------------------------------------
        # MAP WEEKLY/MONTHLY VALUES TO DAILY DATA
        # ----------------------------------------------------

        weekly_values = weekly[
            [
                "Weekly_RSI_5",
                "Weekly_RSI_SMA14",
                "Weekly_ADX14"
            ]
        ].reindex(
            data.index,
            method="ffill"
        )

        monthly_values = monthly[
            [
                "Monthly_RSI_5",
                "Monthly_RSI_SMA14",
                "Monthly_ADX14"
            ]
        ].reindex(
            data.index,
            method="ffill"
        )

        data = data.join(
            weekly_values
        )

        data = data.join(
            monthly_values
        )

        # ----------------------------------------------------
        # FIVE CONDITIONS
        # ----------------------------------------------------

        condition_1 = (
            data["Monthly_RSI_5"]
            >
            data["Monthly_RSI_SMA14"]
        )

        condition_2 = (
            data["Weekly_RSI_5"]
            >
            data["Weekly_RSI_SMA14"]
        )

        condition_3 = (
            data["Monthly_ADX14"]
            >= 25
        )

        condition_4 = (
            data["Daily_RSI_5"]
            < 30
        )

        condition_5 = (
            data["Weekly_ADX14"]
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
        ].copy()

        results = []

        for date in matches.index:

            results.append({
                "Date": pd.Timestamp(date).strftime(
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

    print()
    print(f"Historical period: {YEARS} years")
    print("Conditions: 5")
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

        # Small delay to reduce request pressure
        time.sleep(0.2)

    # --------------------------------------------------------
    # SAVE RESULTS
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
            ["Date", "Stock"],
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
            "No historical signals found."
        )


if __name__ == "__main__":
    main()
