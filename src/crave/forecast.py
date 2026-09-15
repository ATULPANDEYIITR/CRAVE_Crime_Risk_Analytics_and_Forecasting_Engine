from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def create_daily_forecast(
    frame: pd.DataFrame,
    days: int = 7,
) -> pd.DataFrame:
    """Generate a simple statistical crime-risk forecast."""
    if days < 1:
        raise ValueError("days must be at least 1.")

    required = {"timestamp", "crime_type"}

    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    data = frame.copy()
    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    if data["timestamp"].isna().any():
        raise ValueError("timestamp contains invalid values.")

    daily = (
        data.assign(
            date=data["timestamp"].dt.normalize()
        )
        .groupby("date")
        .size()
        .rename("incident_count")
        .reset_index()
    )

    if daily.empty:
        raise ValueError("No historical incidents available.")

    recent = daily.tail(min(14, len(daily)))
    baseline = float(recent["incident_count"].mean())

    if len(recent) >= 2:
        x = np.arange(len(recent), dtype=float)
        y = recent["incident_count"].to_numpy(dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = 0.0

    last_date = daily["date"].max()

    rows = []

    for offset in range(1, days + 1):
        forecast_date = last_date + timedelta(days=offset)
        predicted = max(
            0.0,
            baseline + slope * offset,
        )

        rows.append(
            {
                "date": forecast_date,
                "predicted_incidents": round(
                    predicted,
                    2,
                ),
                "trend": (
                    "increasing"
                    if slope > 0.05
                    else "decreasing"
                    if slope < -0.05
                    else "stable"
                ),
            }
        )

    return pd.DataFrame(rows)


def add_risk_level(
    forecast: pd.DataFrame,
) -> pd.DataFrame:
    """Convert predicted incident counts into risk levels."""
    if "predicted_incidents" not in forecast.columns:
        raise ValueError(
            "forecast must contain predicted_incidents."
        )

    result = forecast.copy()

    values = result["predicted_incidents"]

    q1 = float(values.quantile(0.33))
    q2 = float(values.quantile(0.66))

    def classify(value: float) -> str:
        if value <= q1:
            return "LOW"
        if value <= q2:
            return "MEDIUM"
        return "HIGH"

    result["risk_level"] = values.apply(classify)

    return result
