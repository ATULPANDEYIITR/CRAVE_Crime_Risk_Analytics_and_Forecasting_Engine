from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from src.crave.data import CrimeRecord, records_to_dataframe
from src.crave.features import aggregate_risk_cells, build_features
from src.crave.forecast import add_risk_level, create_daily_forecast
from src.crave.model import CrimeRiskModel


def sample_records() -> list[CrimeRecord]:
    base = datetime(2026, 1, 1, 12, 0)

    return [
        CrimeRecord(
            incident_id=f"C{i:03d}",
            timestamp=base + timedelta(days=i),
            latitude=26.85 + (i % 3) * 0.01,
            longitude=80.95 + (i % 2) * 0.01,
            crime_type=(
                "THEFT"
                if i % 2 == 0
                else "BURGLARY"
            ),
        )
        for i in range(20)
    ]


def test_records_to_dataframe() -> None:
    frame = records_to_dataframe(sample_records())

    assert len(frame) == 20
    assert {
        "incident_id",
        "timestamp",
        "latitude",
        "longitude",
        "crime_type",
    }.issubset(frame.columns)


def test_feature_engineering() -> None:
    frame = records_to_dataframe(sample_records())
    featured = build_features(frame)

    expected = {
        "hour",
        "day_of_week",
        "month",
        "year",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "latitude_grid",
        "longitude_grid",
        "location_cell",
    }

    assert expected.issubset(featured.columns)
    assert len(featured) == len(frame)


def test_risk_cell_aggregation() -> None:
    frame = records_to_dataframe(sample_records())
    risk_cells = aggregate_risk_cells(frame)

    assert not risk_cells.empty
    assert "incident_count" in risk_cells.columns
    assert "risk_score" in risk_cells.columns
    assert risk_cells["risk_score"].is_monotonic_decreasing


def test_model_training_and_prediction() -> None:
    frame = records_to_dataframe(sample_records())
    featured = build_features(frame)

    target = (
        featured["latitude_grid"]
        + featured["longitude_grid"]
        + featured["hour"]
    )

    model = CrimeRiskModel(
        n_estimators=20,
        random_state=42,
    )

    model.fit(featured, target)

    predictions = model.predict(featured)

    assert len(predictions) == len(featured)
    assert predictions.notna().all()


def test_forecast() -> None:
    frame = records_to_dataframe(sample_records())

    forecast = create_daily_forecast(
        frame,
        days=7,
    )

    forecast = add_risk_level(forecast)

    assert len(forecast) == 7
    assert "predicted_incidents" in forecast.columns
    assert "risk_level" in forecast.columns
    assert set(forecast["risk_level"]).issubset(
        {"LOW", "MEDIUM", "HIGH"}
    )
