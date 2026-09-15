from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create time and spatial features for crime-risk modelling."""
    required = {"timestamp", "latitude", "longitude"}

    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = frame.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    if result["timestamp"].isna().any():
        raise ValueError("timestamp contains invalid values.")

    result["hour"] = result["timestamp"].dt.hour
    result["day_of_week"] = result["timestamp"].dt.dayofweek
    result["day_of_month"] = result["timestamp"].dt.day
    result["month"] = result["timestamp"].dt.month
    result["year"] = result["timestamp"].dt.year
    result["is_weekend"] = (
        result["day_of_week"] >= 5
    ).astype(int)

    result["hour_sin"] = np.sin(
        2 * np.pi * result["hour"] / 24
    )
    result["hour_cos"] = np.cos(
        2 * np.pi * result["hour"] / 24
    )

    result["latitude_grid"] = (
        result["latitude"].astype(float).round(2)
    )
    result["longitude_grid"] = (
        result["longitude"].astype(float).round(2)
    )

    result["location_cell"] = (
        result["latitude_grid"].astype(str)
        + "_"
        + result["longitude_grid"].astype(str)
    )

    return result


def aggregate_risk_cells(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate incidents into geographic risk cells."""
    featured = build_features(frame)

    grouped = (
        featured.groupby(
            [
                "latitude_grid",
                "longitude_grid",
                "location_cell",
            ],
            as_index=False,
        )
        .agg(
            incident_count=("incident_id", "count"),
            unique_crime_types=("crime_type", "nunique"),
            weekend_incidents=("is_weekend", "sum"),
        )
    )

    grouped["risk_score"] = (
        grouped["incident_count"]
        + 0.5 * grouped["unique_crime_types"]
        + 0.25 * grouped["weekend_incidents"]
    )

    return grouped.sort_values(
        "risk_score",
        ascending=False,
    ).reset_index(drop=True)
