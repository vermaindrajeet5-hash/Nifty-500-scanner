import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ============================================================
# SETTINGS
# ============================================================
END_DATE = pd.Timestamp.today().normalize()
START_DATE = END_DATE - pd.DateOffset(years=10)

MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/aditya-jha/nse-historical-membership/"
    "main/index_history/data/index_membership_history.csv"
)

OUTPUT_FILE = "nifty500_10yr_monthly_adx_daily_signals.xlsx"


# ============================================================
# RSI
# ============================================================
def rsi_wilder(close, period=5):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    value = 100 - (100 / (1 + rs))

    value = value.where(
        ~((avg_loss == 0) & (avg_gain > 0)),
        100,
    )

    value = value.where(
        ~((avg_gain == 0) & (avg_loss > 0)),
        0,
    )

    return value


# ============================================================
# DMI / ADX
# DMI = 14
# ADX smoothing = 14
# ============================================================
def adx_wilder(df, period=14):
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        ),
        index=df.index,
    )

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_dm_smoothed = plus_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    minus_dm_smoothed = minus_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_di = (
        100
        * plus_dm_smoothed
        / atr.replace(0, np.nan)
    )

    minus_di = (
        100
        * minus_dm_smoothed
        / atr.replace(0, np.nan)
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    # ADX smoothing = 14
    adx = dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return adx


# ============================================================
# SUPERTREND 10,1
# ============================================================
def supertrend(df, period=10, multiplier=1.0):
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    hl2 = (high + low) / 2

    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper = upper_basic.copy()
    lower = lower_basic.copy()

    direction = pd.Series(
        index=df.index,
        dtype="int64",
    )

    for i in range(len(df)):
        if i == 0:
            direction.iloc[i] = 1
            continue

        if pd.isna(atr.iloc[i]):
            direction.iloc[i] = direction.iloc[i - 1]
            continue

        if (
            upper_basic.iloc[i] < upper.iloc[i - 1]
            or close.iloc[i - 1] > upper.iloc[i - 1]
        ):
            upper.iloc[i] = upper_basic.iloc[i]
        else:
            upper.iloc[i] = upper.iloc[i - 1]

        if (
            lower_basic.iloc[i] > lower.iloc[i - 1]
            or close.iloc[i - 1] < lower.iloc[i - 1]
        ):
            lower.iloc[i] = lower_basic.iloc[i]
        else:
            lower.iloc[i] = lower.iloc[i - 1]

        if direction.iloc[i - 1] == -1:
            direction.iloc[i] = (
                1
                if close.iloc[i] > upper.iloc[i]
                else -1
            )
        else:
            direction.iloc[i] = (
                -1
                if close.iloc[i] < lower.iloc[i]
                else 1
            )

    value = pd.Series(
        np.where(
            direction == 1,
            lower,
            upper,
        ),
        index=df.index,
        name="Supertrend",
    )

    return value, direction.rename("ST_Direction")


# ============================================================
# POINT-IN-TIME NIFTY 500 MEMBERSHIP
# ============================================================
def load_membership():
    membership = pd.read_csv(
        MEMBERSHIP_URL
    )

    membership.columns = [
        str(c).strip().lower()
        for c in membership.columns
    ]

    required = {
        "index_name",
        "symbol",
        "valid_from",
        "valid_to",
    }

    missing = required - set(
        membership.columns
    )

    if missing:
        raise ValueError(
            "Membership CSV missing columns: "
            + str(sorted(missing))
        )

    membership["valid_from"] = (
        pd.to_datetime(
            membership["valid_from"],
            errors="coerce",
        ).dt.normalize()
    )

    membership["valid_to"] = (
        pd.to_datetime(
            membership["valid_to"],
            errors="coerce",
        ).dt.normalize()
    )

    membership["symbol"] = (
        membership["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    membership = membership[
        membership["index_name"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .eq("nifty 500")
    ].copy()

    membership = membership.dropna(
        subset=["symbol", "valid_from"]
    )

    membership = membership[
        (membership["valid_from"] <= END_DATE)
        & (
            membership["valid_to"].isna()
            | (
                membership["valid_to"]
                >= START_DATE
            )
        )
    ]

    return membership[
        [
            "symbol",
            "valid_from",
            "valid_to",
        ]
    ].drop_duplicates()


def is_member_on_date(
    membership,
    symbol,
    date,
):
    d = pd.Timestamp(date).normalize()

    rows = membership[
        membership["symbol"].eq(symbol)
    ]

    if rows.empty:
        return False

    return bool(
        (
            (rows["valid_from"] <= d)
            & (
                rows["valid_to"].isna()
                | (
                    rows["valid_to"] > d
                )
            )
        ).any()
    )


# ============================================================
# YAHOO DOWNLOAD
# ============================================================
def download_symbol(symbol):
    ticker = (
        symbol
        if symbol.endswith(".NS")
        else f"{symbol}.NS"
    )

    # Extra history warms up monthly RSI/ADX.
    fetch_start = (
        START_DATE
        - pd.Timedelta(days=1200)
    ).strftime("%Y-%m-%d")

    fetch_end = (
        END_DATE
        + pd.Timedelta(days=2)
    ).strftime("%Y-%m-%d")

    for attempt in range(3):
        try:
            df = yf.download(
                ticker,
                start=fetch_start,
                end=fetch_end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if df.empty:
                return pd.DataFrame()

            if isinstance(
                df.columns,
                pd.MultiIndex,
            ):
                df.columns = [
                    c[0]
                    for c in df.columns
                ]

            wanted = [
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
            ]

            df = df[
                [
                    c
                    for c in wanted
                    if c in df.columns
                ]
            ].copy()

            # Force datetime64[ns] to avoid
            # pandas datetime-unit merge errors.
            idx = pd.to_datetime(
                df.index,
                errors="coerce",
                utc=True,
            )

            df.index = (
                idx.tz_convert(None)
                .normalize()
            )

            return (
                df.dropna(
                    subset=["Close"]
                )
                .sort_index()
            )

        except Exception as exc:
            if attempt == 2:
                print(
                    f"{symbol}: {exc}",
                    flush=True,
                )
                return pd.DataFrame()

            time.sleep(2 ** attempt)

    return pd.DataFrame()


# ============================================================
# SCAN ONE STOCK
# ============================================================
def scan_symbol(
    symbol,
    membership,
):
    daily = download_symbol(symbol)

    if daily.empty:
        return (
            pd.DataFrame(),
            "No Yahoo data",
        )

    # --------------------------------------------------------
    # DAILY INDICATORS
    # --------------------------------------------------------
    daily["Daily_RSI5"] = rsi_wilder(
        daily["Close"],
        5,
    )

    daily["Daily_RSI5_SMA14"] = (
        daily["Daily_RSI5"]
        .rolling(
            14,
            min_periods=14,
        )
        .mean()
    )

    daily["EMA20"] = (
        daily["Close"]
        .ewm(
            span=20,
            adjust=False,
            min_periods=20,
        )
        .mean()
    )

    (
        daily["Supertrend"],
        daily["ST_Direction"],
    ) = supertrend(
        daily,
        10,
        1.0,
    )

    daily["ST_Green"] = (
        daily["ST_Direction"] == 1
    )

    # --------------------------------------------------------
    # MONTHLY INDICATORS
    # --------------------------------------------------------
    monthly = (
        daily["Close"]
        .resample("ME")
        .last()
        .to_frame("Close")
    )

    # Monthly RSI(5) and SMA(14)
    monthly["Monthly_RSI5"] = (
        rsi_wilder(
            monthly["Close"],
            5,
        )
    )

    monthly["Monthly_RSI5_SMA14"] = (
        monthly["Monthly_RSI5"]
        .rolling(
            14,
            min_periods=14,
        )
        .mean()
    )

    # Monthly DMI(14) / ADX(14)
    # Calculate from monthly OHLC, not monthly closes.
    monthly_ohlc = daily[
        [
            "Open",
            "High",
            "Low",
            "Close",
        ]
    ].resample("ME").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        }
    )

    monthly["Monthly_ADX14"] = (
        adx_wilder(
            monthly_ohlc,
            14,
        )
    )

    # --------------------------------------------------------
    # MONTHLY CONDITIONS
    # --------------------------------------------------------
    monthly["RSI_Above_SMA"] = (
        monthly["Monthly_RSI5"]
        > monthly["Monthly_RSI5_SMA14"]
    )

    # User requested:
    # current month ADX > previous month ADX
    monthly["ADX_Rising"] = (
        monthly["Monthly_ADX14"]
        > monthly["Monthly_ADX14"].shift(1)
    )

    # User requested:
    # difference between current and previous ADX
    # closing values is LESS THAN 1.
    #
    # Because ADX_Rising is also required, this is:
    # 0 < current ADX - previous ADX < 1
    monthly["ADX_Difference"] = (
        monthly["Monthly_ADX14"]
        - monthly["Monthly_ADX14"].shift(1)
    )

    monthly["ADX_Difference_Less_Than_1"] = (
        (monthly["ADX_Difference"] > 0)
        & (monthly["ADX_Difference"] < 1)
    )

    monthly["Monthly_Condition"] = (
        monthly["RSI_Above_SMA"]
        & monthly["ADX_Rising"]
        & monthly[
            "ADX_Difference_Less_Than_1"
        ]
    )

    # --------------------------------------------------------
    # PREVIOUS COMPLETED MONTH FOR EACH DAILY DATE
    # --------------------------------------------------------
    month_ends = monthly.index

    positions = (
        month_ends.searchsorted(
            daily.index,
            side="left",
        )
        - 1
    )

    previous_month = pd.Series(
        pd.NaT,
        index=daily.index,
        dtype="datetime64[ns]",
    )

    valid = positions >= 0

    previous_month.loc[valid] = (
        month_ends[
            positions[valid]
        ]
    )

    daily["Previous_Month_End"] = (
        previous_month
    )

    # Map completed-month values to daily dates.
    maps = {
        "Monthly_Condition": monthly[
            "Monthly_Condition"
        ].to_dict(),
        "Monthly_RSI5": monthly[
            "Monthly_RSI5"
        ].to_dict(),
        "Monthly_RSI5_SMA14": monthly[
            "Monthly_RSI5_SMA14"
        ].to_dict(),
        "Monthly_ADX14": monthly[
            "Monthly_ADX14"
        ].to_dict(),
        "Previous_Month_ADX14": monthly[
            "Monthly_ADX14"
        ].shift(1).to_dict(),
        "Monthly_ADX_Difference": monthly[
            "ADX_Difference"
        ].to_dict(),
    }

    for column, mapping in maps.items():
        daily[column] = (
            daily["Previous_Month_End"]
            .map(mapping)
        )

    # --------------------------------------------------------
    # FINAL DAILY SIGNAL
    # --------------------------------------------------------
    signal = (
        (daily.index >= START_DATE)
        & (daily.index <= END_DATE)

        # Monthly:
        & daily[
            "Monthly_Condition"
        ].fillna(False)

        # Daily Supertrend positive:
        & daily[
            "ST_Green"
        ].fillna(False)

        # Daily candle CLOSE above EMA20:
        & (
            daily["Close"]
            > daily["EMA20"]
        )

        # Daily RSI5 above its SMA14:
        & (
            daily["Daily_RSI5"]
            > daily["Daily_RSI5_SMA14"]
        )
    )

    result = daily.loc[signal].copy()

    if result.empty:
        return (
            pd.DataFrame(),
            None,
        )

    # --------------------------------------------------------
    # POINT-IN-TIME NIFTY 500 FILTER
    # --------------------------------------------------------
    keep = [
        is_member_on_date(
            membership,
            symbol,
            d,
        )
        for d in result.index
    ]

    result = result.loc[keep]

    if result.empty:
        return (
            pd.DataFrame(),
            None,
        )

    output = pd.DataFrame(
        {
            "Date": result.index.date,
            "Stock": symbol,

            "Monthly_RSI5": result[
                "Monthly_RSI5"
            ].values,

            "Monthly_RSI5_SMA14": result[
                "Monthly_RSI5_SMA14"
            ].values,

            "Monthly_ADX14": result[
                "Monthly_ADX14"
            ].values,

            "Previous_Month_ADX14": result[
                "Previous_Month_ADX14"
            ].values,

            "ADX_Difference": result[
                "Monthly_ADX_Difference"
            ].values,

            "Daily_RSI5": result[
                "Daily_RSI5"
            ].values,

            "Daily_RSI5_SMA14": result[
                "Daily_RSI5_SMA14"
            ].values,

            "Daily_Close": result[
                "Close"
            ].values,

            "EMA20": result[
                "EMA20"
            ].values,

            "Supertrend": result[
                "Supertrend"
            ].values,

            "Supertrend_Green": True,
        }
    )

    return output, None


# ============================================================
# MAIN
# ============================================================
def main():
    print(
        "==================================================",
        flush=True,
    )

    print(
        "NIFTY 500 10-YEAR HISTORICAL SCANNER",
        flush=True,
    )

    print(
        f"Start: {START_DATE.date()}",
        flush=True,
    )

    print(
        f"End:   {END_DATE.date()}",
        flush=True,
    )

    print(
        "==================================================",
        flush=True,
    )

    membership = load_membership()

    symbols = sorted(
        membership["symbol"]
        .dropna()
        .unique()
    )

    print(
        f"Point-in-time NIFTY 500 symbols: "
        f"{len(symbols)}",
        flush=True,
    )

    all_signals = []
    errors = []

    for i, symbol in enumerate(
        symbols,
        1,
    ):
        print(
            f"[{i}/{len(symbols)}] {symbol}",
            flush=True,
        )

        try:
            signals, error = (
                scan_symbol(
                    symbol,
                    membership,
                )
            )

            if not signals.empty:
                all_signals.append(
                    signals
                )

            if error:
                errors.append(
                    {
                        "Stock": symbol,
                        "Error": error,
                    }
                )

        except Exception as exc:
            errors.append(
                {
                    "Stock": symbol,
                    "Error": repr(exc),
                }
            )

    if all_signals:
        signals = pd.concat(
            all_signals,
            ignore_index=True,
        )

        signals = signals.sort_values(
            [
                "Date",
                "Stock",
            ]
        ).reset_index(drop=True)

    else:
        signals = pd.DataFrame(
            columns=[
                "Date",
                "Stock",
            ]
        )

    errors_df = pd.DataFrame(
        errors
    )

    # --------------------------------------------------------
    # RULES SHEET
    # --------------------------------------------------------
    rules = pd.DataFrame(
        {
            "Rule": [
                "Universe",
                "Period",
                "Monthly condition 1",
                "Monthly condition 2",
                "Monthly condition 3",
                "Daily condition 1",
                "Daily condition 2",
                "Daily condition 3",
                "Look-ahead control",
                "Membership control",
            ],
            "Definition": [
                "Point-in-time NIFTY 500 constituents",
                f"{START_DATE.date()} to {END_DATE.date()}",
                "Monthly RSI(5) > its SMA(14)",
                "Monthly ADX(14) > previous completed month's ADX(14)",
                "Current monthly ADX - previous monthly ADX is > 0 and < 1",
                "Daily Supertrend(10,1) is GREEN / positive",
                "Daily Close > EMA(20)",
                "Daily RSI(5) > its SMA(14)",
                "Daily signal uses the previous completed monthly candle",
                "Stock must be a NIFTY 500 member on the exact signal date",
            ],
        }
    )

    # --------------------------------------------------------
    # EXCEL
    # --------------------------------------------------------
    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl",
    ) as writer:

        signals.to_excel(
            writer,
            sheet_name="Signals",
            index=False,
        )

        rules.to_excel(
            writer,
            sheet_name="Rules",
            index=False,
        )

        errors_df.to_excel(
            writer,
            sheet_name="Data Errors",
            index=False,
        )

    print(
        "==================================================",
        flush=True,
    )

    print(
        f"Excel saved: {OUTPUT_FILE}",
        flush=True,
    )

    print(
        f"Total signals: {len(signals)}",
        flush=True,
    )

    print(
        f"Stocks with data errors: {len(errors_df)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
