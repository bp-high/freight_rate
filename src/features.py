"""Shared data cleaning and feature engineering for the freight rate model.

All transformations here use only load-level features (never the target), so
the same code path serves training, validation scoring, and the fixed
December lane rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_MILES = 3958.8

CATEGORICAL_FEATURES = ["equipment", "pickup", "delivery"]
NUMERIC_FEATURES = [
    "distance",
    "log_distance",
    "weight",
    "weight_missing",
    "market_index",
    "market_index_missing",
    "mi_daily",
    "mi_7d",
    "quote_signal",
    "qs_daily",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "bearing",
    "haversine",
    "month",
    "day_of_week",
    "day_of_year",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Rows whose rate-per-mile falls outside this band are treated as corrupted
# labels (~1.1% of training data, spread uniformly across months/equipment).
RPM_LOW, RPM_HIGH = 0.8, 6.0


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y))


def daily_market_series(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Per-date mean market_index and quote_signal pooled across the given
    frames (features only — no target involved), with a trailing 7-day mean
    of the market index so day-of-week dips don't mask the underlying level."""
    pooled = pd.concat(
        [f[["date", "market_index", "quote_signal"]] for f in frames], ignore_index=True
    )
    daily = (
        pooled.groupby("date")
        .agg(mi_daily=("market_index", "mean"), qs_daily=("quote_signal", "mean"))
        .sort_index()
    )
    daily["mi_7d"] = daily["mi_daily"].rolling(7, min_periods=1).mean()
    return daily


def build_features(df: pd.DataFrame, daily: pd.DataFrame, train_weight_median: float) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    # Weight: ~0.6% of rows carry a flipped sign (identical magnitude
    # distribution to positive rows) and ~0.6% are missing.
    out["weight_missing"] = out["weight"].isna().astype(int)
    out["weight"] = out["weight"].abs()
    out["weight"] = out["weight"].fillna(train_weight_median)

    # Market index: impute missing values with the pooled per-date mean.
    out = out.merge(daily, left_on="date", right_index=True, how="left")
    out["market_index_missing"] = out["market_index"].isna().astype(int)
    out["market_index"] = out["market_index"].fillna(out["mi_daily"])

    out["log_distance"] = np.log(out["distance"])
    out["haversine"] = haversine(
        out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"]
    )
    out["bearing"] = bearing(
        out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"]
    )

    out["month"] = out["date"].dt.month
    out["day_of_week"] = out["date"].dt.dayofweek
    out["day_of_year"] = out["date"].dt.dayofyear

    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype("category")
    return out


def clean_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose label looks corrupted (rate-per-mile far outside the
    physically plausible band). Applied to training rows only."""
    rpm = df["posted_rate"] / df["distance"]
    return df[(rpm >= RPM_LOW) & (rpm <= RPM_HIGH)].copy()
