"""
POINT-IN-TIME NIFTY 500 — 7 YEAR HISTORICAL ATH STUDY

This version uses historical NIFTY 500 membership by date.

For each trading day:
    1. Find stocks that were actually members of NIFTY 500 on that date.
    2. Look only at price data available up to that date.
    3. Identify fresh all-time-high events.
    4. Measure what happened after the ATH:
       5, 10, 20 and 60 trading days.
    5. Record volume confirmation.

IMPORTANT:
- Membership uses half-open intervals:
      valid_from <= date
      AND (valid_to is null OR valid_to > date)
- The historical membership file has best coverage from 2017 onward.
- The 7-year window therefore fits the stronger part of the dataset.
- This is research code, not investment advice.

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

START_DATE = (date.today() - timedelta(days=365 * 7)).isoformat()
END_DATE = (date.today() + timedelta(days=1)).isoformat()

VOLUME_LOOKBACK = 20

OUTPUT_EVENTS = "nifty500_PIT_7Y_ATH_events.csv"
OUTPUT_SUMMARY = "nifty500_PIT_7Y_ATH_summary.csv"
OUTPUT_ERRORS = "nifty500_PIT_7Y_ATH_errors.csv"


def load_membership():
    print("Downloading historical NIFTY 500 membership...")

    r = requests.get(MEMBERSHIP_URL, timeout=60)
    r.raise_for_status()

    df = pd.read_csv(io.BytesIO(r.content))

    required = [
        "index_name",
        "symbol",
        "valid_from",
        "valid_to",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"Membership file missing columns: {missing}"
        )

    df = df[df["index_name"].astype(str).str.strip().str.lower() == "nifty 500"].copy()

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
        subset=["symbol", "valid_from"]
    )

    df = df.sort_values(
        ["symbol", "valid_from"]
    )

    if df.empty:
        raise RuntimeError(
            "No NIFTY 500 membership rows found."
        )

    print(
        f"Historical NIFTY 500 intervals loaded: {len(df)}"
    )

    return df


def members_on_date(membership, trading_date):
    d = pd.Timestamp(trading_date)

    mask = (
        (membership["valid_from"] <= d)
        & (
            membership["valid_to"].isna()
            | (membership["valid_to"] > d)
        )
    )

    return sorted(
        membership.loc[mask, "symbol"]
        .drop_duplicates()
        .tolist()
    )


def download_stock(symbol):
    data = yf.download(
        f"{symbol}.NS",
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
        repair=True,
    )

    if data is None or data.empty:
        raise RuntimeError(
            "No Yahoo Finance data"
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

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
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    data = data[required].copy()

    for c in required:
        data[c] = pd.to_numeric(
            data[c],
            errors="coerce",
        )

    data = data.dropna(
        subset=["High", "Low", "Close"]
    )

    data.index = pd.to_datetime(
        data.index
    )

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    data.index = data.index.normalize()

    data = data[
        ~data.index.duplicated(
            keep="last"
        )
    ].sort_index()

    if data.empty:
        raise RuntimeError(
            "No valid daily rows"
        )

    return data


def forward_return(
    closes,
    signal_index,
    days,
    entry_price,
):
    future = closes.iloc[
        signal_index + 1:
    ]

    if len(future) < days:
        return None

    exit_price = float(
        future.iloc[days - 1]
    )

    return round(
        (exit_price / entry_price - 1)
        * 100,
        2,
    )


def forward_max_gain(
    highs,
    signal_index,
    days,
    entry_price,
):
    future = highs.iloc[
        signal_index + 1:
        signal_index + 1 + days
    ]

    if len(future) < days:
        return None

    return round(
        (float(future.max()) / entry_price - 1)
        * 100,
        2,
    )


def forward_max_loss(
    lows,
    signal_index,
    days,
    entry_price,
):
    future = lows.iloc[
        signal_index + 1:
        signal_index + 1 + days
    ]

    if len(future) < days:
        return None

    return round(
        (float(future.min()) / entry_price - 1)
        * 100,
        2,
    )


def find_ath_events(
    symbol,
    data,
    membership_dates,
):
    highs = data["High"]
    lows = data["Low"]
    closes = data["Close"]
    volumes = data["Volume"]

    rows = []

    # Only dates on which this stock was actually a
    # NIFTY 500 constituent are eligible signal dates.
    eligible_dates = set(
        pd.Timestamp(d).normalize()
        for d in membership_dates
    )

    for i in range(1, len(data)):
        trading_date = data.index[i]

        if trading_date not in eligible_dates:
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

        # Fresh ATH:
        # today's High must exceed every prior High.
        if today_high <= previous_ath:
            continue

        breakout_pct = (
            today_high / previous_ath - 1
        ) * 100

        close_vs_ath_pct = (
            today_close / previous_ath - 1
        ) * 100

        avg_volume = None
        volume_ratio = None

        if i >= VOLUME_LOOKBACK:
            prior_volumes = (
                volumes.iloc[
                    i - VOLUME_LOOKBACK:i
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
                        float(volumes.iloc[i])
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
                    int(volumes.iloc[i])
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
                    group["Return_5D_Pct"]
                    .dropna()
                    .mean(),
                    2,
                ),

                "Win_Rate_5D_Pct": win_rate(
                    group["Return_5D_Pct"]
                ),

                "Avg_Return_10D_Pct": round(
                    group["Return_10D_Pct"]
                    .dropna()
                    .mean(),
                    2,
                ),

                "Win_Rate_10D_Pct": win_rate(
                    group["Return_10D_Pct"]
                ),

                "Avg_Return_20D_Pct": round(
                    group["Return_20D_Pct"]
                    .dropna()
                    .mean(),
                    2,
                ),

                "Win_Rate_20D_Pct": win_rate(
                    group["Return_20D_Pct"]
                ),

                "Avg_Return_60D_Pct": round(
                    group["Return_60D_Pct"]
                    .dropna()
                    .mean(),
                    2,
                ),

                "Win_Rate_60D_Pct": win_rate(
                    group["Return_60D_Pct"]
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
        f"Study start: {START_DATE}"
    )
    print(
        f"Study end:   {END_DATE}"
    )
    print()

    membership = load_membership()

    # Get every trading date represented in Yahoo data later.
    # We download each unique stock only once.
    symbols = sorted(
        membership["symbol"]
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

    for number, symbol in enumerate(
        symbols,
        start=1,
    ):
        print(
            f"[{number}/{len(symbols)}] {symbol}"
        )

        try:
            data = download_stock(
                symbol
            )

            # Build exact PIT membership dates
            # that overlap the 7-year study.
            member_rows = membership[
                membership["symbol"]
                == symbol
            ].copy()

            member_dates = []

            for _, row in member_rows.iterrows():
                start = max(
                    pd.Timestamp(
                        START_DATE
                    ),
                    row["valid_from"],
                )

                if pd.isna(
                    row["valid_to"]
                ):
                    end = pd.Timestamp(
                        END_DATE
                    )
                else:
                    end = min(
                        pd.Timestamp(
                            END_DATE
                        ),
                        row["valid_to"],
                    )

                # Half-open interval:
                # valid_from <= date < valid_to
                if start < end:
                    member_dates.extend(
                        pd.date_range(
                            start=start,
                            end=end
                            - pd.Timedelta(
                                days=1
                            ),
                            freq="D",
                        )
                    )

            events = find_ath_events(
                symbol,
                data,
                member_dates,
            )

            all_events.extend(
                events
            )

            print(
                f"  PIT ATH events: "
                f"{len(events)}"
            )

        except Exception as exc:
            errors.append(
                {
                    "Stock": symbol,
                    "Error": str(exc),
                }
            )

            print(
                f"  ERROR: {exc}"
            )

        time.sleep(0.15)

    if not all_events:
        raise RuntimeError(
            "No point-in-time ATH events were created."
        )

    events_df = (
        pd.DataFrame(all_events)
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

    if errors:
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
        f"ATH events: {len(events_df)}"
    )
    print(
        f"Stocks with errors: {len(errors)}"
    )
    print(
        f"Saved: {OUTPUT_EVENTS}"
    )
    print(
        f"Saved: {OUTPUT_SUMMARY}"
    )

    if errors:
        print(
            f"Saved: {OUTPUT_ERRORS}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
