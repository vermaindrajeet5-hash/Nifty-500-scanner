"""
NIFTY 500 POINT-IN-TIME — 7 YEAR HISTORICAL ATH STUDY
BATCHED / RATE-LIMIT-RESISTANT VERSION

This version:
- Uses historical NIFTY 500 membership by date.
- Downloads Yahoo Finance data in batches instead of ~951 individual requests.
- Uses ALL available Yahoo history to establish the true previous ATH.
- Counts ATH events only during the 7-year study period.
- Measures 5/10/20/60 trading-day forward returns.
- Records 20-day maximum gain/loss after the ATH.

Important:
- This is a historical research study, not investment advice.
- The historical membership file is used with:
      valid_from <= date
      AND (valid_to is null OR valid_to > date)
- Stocks with no Yahoo data are recorded as errors and do not
  invalidate the entire study.

Outputs:
    nifty500_PIT_7Y_ATH_events.csv
    nifty500_PIT_7Y_ATH_summary.csv
    nifty500_PIT_7Y_ATH_errors.csv
"""

import io
import time
from datetime import date, timedelta

import pandas as pd
import requests
import yfinance as yf


MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/aditya-jha/"
    "nse-historical-membership/main/"
    "index_history/data/index_membership_history.csv"
)

# We need history BEFORE the study period to establish the true ATH.
PRICE_START = "2000-01-01"

STUDY_START = (
    date.today() - timedelta(days=365 * 7)
).isoformat()

STUDY_END = (
    date.today() + timedelta(days=1)
).isoformat()

VOLUME_LOOKBACK = 20

# Small batches are more reliable than ~951 individual Yahoo requests.
BATCH_SIZE = 25

# Retry failed batches.
MAX_RETRIES = 3

OUTPUT_EVENTS = "nifty500_PIT_7Y_ATH_events.csv"
OUTPUT_SUMMARY = "nifty500_PIT_7Y_ATH_summary.csv"
OUTPUT_ERRORS = "nifty500_PIT_7Y_ATH_errors.csv"


def load_membership():
    print("Downloading historical NIFTY 500 membership...")

    r = requests.get(
        MEMBERSHIP_URL,
        timeout=60,
    )
    r.raise_for_status()

    df = pd.read_csv(
        io.BytesIO(r.content)
    )

    required = [
        "index_name",
        "symbol",
        "valid_from",
        "valid_to",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Membership file missing columns: {missing}"
        )

    df = df[
        df["index_name"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "nifty 500"
    ].copy()

    df["symbol"] = (
        df["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["valid_from"] = pd.to_datetime(
        df["valid_from"],
        errors="coerce",
    )

    df["valid_to"] = pd.to_datetime(
        df["valid_to"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "symbol",
            "valid_from",
        ]
    )

    if df.empty:
        raise RuntimeError(
            "No NIFTY 500 membership rows found."
        )

    print(
        f"Membership intervals loaded: {len(df)}"
    )

    return df


def download_batch(symbols):
    """
    Download a batch of NSE symbols from Yahoo.

    Returns the raw yfinance DataFrame.
    """
    tickers = [
        f"{symbol}.NS"
        for symbol in symbols
    ]

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            print(
                f"    Yahoo batch attempt "
                f"{attempt}/{MAX_RETRIES}"
            )

            data = yf.download(
                tickers=tickers,
                start=PRICE_START,
                end=STUDY_END,
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=True,
                group_by="ticker",
                repair=False,
            )

            if data is None or data.empty:
                raise RuntimeError(
                    "Yahoo returned an empty batch"
                )

            return data

        except Exception as exc:
            last_error = exc

            print(
                f"    Batch error: {exc}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(
                    5 * attempt
                )

    raise RuntimeError(
        f"Batch failed after "
        f"{MAX_RETRIES} attempts: "
        f"{last_error}"
    )


def extract_symbol_data(
    batch_data,
    symbol,
):
    """
    Extract one ticker from a batched yfinance result.
    Handles both MultiIndex layouts and single-level columns.
    """
    ticker = f"{symbol}.NS"

    if isinstance(
        batch_data.columns,
        pd.MultiIndex,
    ):
        level0 = set(
            str(x)
            for x in batch_data.columns
                .get_level_values(0)
        )

        level1 = set(
            str(x)
            for x in batch_data.columns
                .get_level_values(1)
        )

        if ticker in level0:
            data = batch_data[
                ticker
            ].copy()

        elif ticker in level1:
            data = batch_data[
                :, ticker
            ].copy()

        else:
            return None

    else:
        data = batch_data.copy()

    required = [
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing = [
        c for c in required
        if c not in data.columns
    ]

    if missing:
        return None

    data = data[
        required
    ].copy()

    for c in required:
        data[c] = pd.to_numeric(
            data[c],
            errors="coerce",
        )

    data = data.dropna(
        subset=[
            "High",
            "Low",
            "Close",
        ]
    )

    if data.empty:
        return None

    data.index = pd.to_datetime(
        data.index
    )

    if data.index.tz is not None:
        data.index = (
            data.index.tz_localize(None)
        )

    data.index = data.index.normalize()

    data = data[
        ~data.index.duplicated(
            keep="last"
        )
    ].sort_index()

    return data


def membership_dates_for_symbol(
    membership,
    symbol,
):
    """
    Return a set of calendar dates on which the
    symbol was a NIFTY 500 constituent during the
    study period.

    Uses the required half-open interval:
        valid_from <= date < valid_to
    """
    rows = membership[
        membership["symbol"]
        == symbol
    ]

    dates = set()

    study_start = pd.Timestamp(
        STUDY_START
    )

    study_end = pd.Timestamp(
        STUDY_END
    )

    for _, row in rows.iterrows():
        start = max(
            study_start,
            row["valid_from"],
        )

        if pd.isna(
            row["valid_to"]
        ):
            end = study_end
        else:
            end = min(
                study_end,
                row["valid_to"],
            )

        if start >= end:
            continue

        rng = pd.date_range(
            start=start,
            end=end
            - pd.Timedelta(days=1),
            freq="D",
        )

        dates.update(
            rng.normalize()
        )

    return dates


def forward_return(
    closes,
    i,
    days,
    entry,
):
    future = closes.iloc[
        i + 1:
    ]

    if len(future) < days:
        return None

    exit_price = float(
        future.iloc[days - 1]
    )

    return round(
        (
            exit_price
            / entry
            - 1
        ) * 100,
        2,
    )


def forward_max_gain(
    highs,
    i,
    days,
    entry,
):
    future = highs.iloc[
        i + 1:
        i + 1 + days
    ]

    if len(future) < days:
        return None

    return round(
        (
            float(future.max())
            / entry
            - 1
        ) * 100,
        2,
    )


def forward_max_loss(
    lows,
    i,
    days,
    entry,
):
    future = lows.iloc[
        i + 1:
        i + 1 + days
    ]

    if len(future) < days:
        return None

    return round(
        (
            float(future.min())
            / entry
            - 1
        ) * 100,
        2,
    )


def find_events(
    symbol,
    data,
    member_dates,
):
    """
    Find fresh ATH events.

    IMPORTANT:
    The previous ATH is calculated from ALL price
    history before the signal day, not just the
    7-year study window.
    """
    highs = data["High"]
    lows = data["Low"]
    closes = data["Close"]
    volumes = data["Volume"]

    rows = []

    for i in range(
        1,
        len(data),
    ):
        trading_date = (
            data.index[i]
        )

        # Only count the event if the stock was
        # actually a NIFTY 500 constituent that day.
        if trading_date not in member_dates:
            continue

        previous_ath = float(
            highs.iloc[:i].max()
        )

        today_high = float(
            highs.iloc[i]
        )

        today_close = float(
            closes.iloc[i]
        )

        # A fresh ATH means today's High is strictly
        # higher than every High before today.
        if today_high <= previous_ath:
            continue

        breakout_pct = (
            today_high
            / previous_ath
            - 1
        ) * 100

        close_vs_ath_pct = (
            today_close
            / previous_ath
            - 1
        ) * 100

        avg_volume = None
        volume_ratio = None

        if i >= VOLUME_LOOKBACK:
            prior_volumes = (
                volumes.iloc[
                    i - VOLUME_LOOKBACK:
                    i
                ]
                .dropna()
            )

            if not prior_volumes.empty:
                avg_volume = float(
                    prior_volumes.mean()
                )

                if (
                    avg_volume > 0
                    and pd.notna(
                        volumes.iloc[i]
                    )
                ):
                    volume_ratio = (
                        float(
                            volumes.iloc[i]
                        )
                        / avg_volume
                    )

        rows.append(
            {
                "Date": trading_date.date().isoformat(),
                "Stock": symbol,
                "ATH_High_Today": round(
                    today_high,
                    2,
                ),
                "Close": round(
                    today_close,
                    2,
                ),
                "Previous_ATH": round(
                    previous_ath,
                    2,
                ),
                "Breakout_Pct": round(
                    breakout_pct,
                    2,
                ),
                "Close_vs_Previous_ATH_Pct": round(
                    close_vs_ath_pct,
                    2,
                ),
                "Volume": (
                    int(
                        volumes.iloc[i]
                    )
                    if pd.notna(
                        volumes.iloc[i]
                    )
                    else None
                ),
                "Average_Volume_20D": (
                    round(
                        avg_volume,
                        0,
                    )
                    if avg_volume is not None
                    else None
                ),
                "Volume_Ratio": (
                    round(
                        volume_ratio,
                        2,
                    )
                    if volume_ratio is not None
                    else None
                ),
                "Return_5D_Pct": forward_return(
                    closes,
                    i,
                    5,
                    today_close,
                ),
                "Return_10D_Pct": forward_return(
                    closes,
                    i,
                    10,
                    today_close,
                ),
                "Return_20D_Pct": forward_return(
                    closes,
                    i,
                    20,
                    today_close,
                ),
                "Return_60D_Pct": forward_return(
                    closes,
                    i,
                    60,
                    today_close,
                ),
                "Max_Gain_20D_Pct": forward_max_gain(
                    highs,
                    i,
                    20,
                    today_close,
                ),
                "Max_Loss_20D_Pct": forward_max_loss(
                    lows,
                    i,
                    20,
                    today_close,
                ),
            }
        )

    return rows


def win_rate(series):
    s = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if s.empty:
        return None

    return round(
        (s > 0).mean() * 100,
        2,
    )


def make_summary(events_df):
    rows = []

    for stock, group in events_df.groupby(
        "Stock"
    ):
        rows.append(
            {
                "Stock": stock,
                "ATH_Events": len(group),

                "Avg_Return_5D_Pct": round(
                    group[
                        "Return_5D_Pct"
                    ]
                    .dropna()
                    .mean(),
                    2,
                ),

                "Win_Rate_5D_Pct": win_rate(
                    group[
                        "Return_5D_Pct"
                    ]
                ),

                "Avg_Return_10D_Pct": round(
                    group[
                        "Return_10D_Pct"
                    ]
                    .dropna()
                    .mean(),
                    2,
                ),

                "Win_Rate_10D_Pct": win_rate(
                    group[
                        "Return_10D_Pct"
                    ]
                ),

                "Avg_Return_20D_Pct": round(
                    group[
                        "Return_20D_Pct"
                    ]
                    .dropna()
                    .mean(),
                    2,
                ),

                "Win_Rate_20D_Pct": win_rate(
                    group[
                        "Return_20D_Pct"
                    ]
                ),

                "Avg_Return_60D_Pct": round(
                    group[
                        "Return_60D_Pct"
                    ]
                    .dropna()
                    .mean(),
                    2,
                ),

                "Win_Rate_60D_Pct": win_rate(
                    group[
                        "Return_60D_Pct"
                    ]
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            "Avg_Return_20D_Pct",
            ascending=False,
        )
    )


def main():
    print("=" * 70)
    print(
        "NIFTY 500 POINT-IN-TIME — "
        "7 YEAR HISTORICAL ATH STUDY"
    )
    print("=" * 70)
    print(
        f"Study start: {STUDY_START}"
    )
    print(
        f"Study end:   {STUDY_END}"
    )
    print(
        f"Price history starts: {PRICE_START}"
    )
    print()
    print(
        "True ATH is calculated using price history "
        "before each signal date."
    )
    print()

    membership = load_membership()

    symbols = sorted(
        membership[
            "symbol"
        ]
        .dropna()
        .unique()
    )

    print(
        f"Unique historical NIFTY 500 symbols: "
        f"{len(symbols)}"
    )
    print()

    all_events = []
    errors = []

    total_batches = (
        len(symbols)
        + BATCH_SIZE
        - 1
    ) // BATCH_SIZE

    for batch_number, start in enumerate(
        range(
            0,
            len(symbols),
            BATCH_SIZE,
        ),
        start=1,
    ):
        batch_symbols = symbols[
            start:
            start + BATCH_SIZE
        ]

        print(
            "=" * 70
        )
        print(
            f"BATCH {batch_number}/{total_batches} "
            f"({len(batch_symbols)} stocks)"
        )
        print(
            "=" * 70
        )

        try:
            batch_data = download_batch(
                batch_symbols
            )

        except Exception as exc:
            print(
                f"  WHOLE BATCH FAILED: {exc}"
            )

            # Record batch-level errors.
            for symbol in batch_symbols:
                errors.append(
                    {
                        "Stock": symbol,
                        "Error": (
                            "Yahoo batch failed: "
                            + str(exc)
                        ),
                    }
                )

            continue

        for symbol in batch_symbols:
            try:
                data = extract_symbol_data(
                    batch_data,
                    symbol,
                )

                if data is None or data.empty:
                    raise RuntimeError(
                        "No Yahoo Finance data"
                    )

                member_dates = (
                    membership_dates_for_symbol(
                        membership,
                        symbol,
                    )
                )

                events = find_events(
                    symbol,
                    data,
                    member_dates,
                )

                all_events.extend(
                    events
                )

                print(
                    f"  {symbol}: "
                    f"{len(events)} ATH events"
                )

            except Exception as exc:
                errors.append(
                    {
                        "Stock": symbol,
                        "Error": str(exc),
                    }
                )

                print(
                    f"  {symbol}: ERROR - {exc}"
                )

        # Give Yahoo a short break between batches.
        time.sleep(2)

    if not all_events:
        # Save errors before stopping so we can diagnose
        # a total Yahoo outage/rate-limit.
        pd.DataFrame(
            errors
        ).to_csv(
            OUTPUT_ERRORS,
            index=False,
        )

        raise RuntimeError(
            "No ATH events were produced. "
            "Check nifty500_PIT_7Y_ATH_errors.csv."
        )

    events_df = (
        pd.DataFrame(
            all_events
        )
        .sort_values(
            ["Date", "Stock"]
        )
    )

    events_df.to_csv(
        OUTPUT_EVENTS,
        index=False,
    )

    summary_df = make_summary(
        events_df
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    pd.DataFrame(
        errors
    ).to_csv(
        OUTPUT_ERRORS,
        index=False,
    )

    print()
    print("=" * 70)
    print("STUDY COMPLETE")
    print("=" * 70)
    print(
        f"Stocks attempted: {len(symbols)}"
    )
    print(
        f"Stocks with data/errors: "
        f"{len(errors)} errors"
    )
    print(
        f"Fresh PIT ATH events: "
        f"{len(events_df)}"
    )
    print()
    print(
        f"Saved: {OUTPUT_EVENTS}"
    )
    print(
        f"Saved: {OUTPUT_SUMMARY}"
    )
    print(
        f"Saved: {OUTPUT_ERRORS}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
