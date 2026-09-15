from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {
    "incident_id",
    "timestamp",
    "latitude",
    "longitude",
    "crime_type",
}


@dataclass(frozen=True)
class CrimeRecord:
    incident_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    crime_type: str


def records_to_dataframe(records: Iterable[CrimeRecord]) -> pd.DataFrame:
    """Convert crime records into a validated DataFrame."""
    frame = pd.DataFrame(
        [
            {
                "incident_id": record.incident_id,
                "timestamp": record.timestamp,
                "latitude": record.latitude,
                "longitude": record.longitude,
                "crime_type": record.crime_type,
            }
            for record in records
        ]
    )

    validate_dataframe(frame)
    return frame


def validate_dataframe(frame: pd.DataFrame) -> None:
    """Validate the minimum schema required by CRAVE."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if frame.empty:
        raise ValueError("Crime dataset cannot be empty.")

    if frame["incident_id"].isna().any():
        raise ValueError("incident_id cannot contain missing values.")

    if frame["timestamp"].isna().any():
        raise ValueError("timestamp cannot contain missing values.")

    if not pd.api.types.is_numeric_dtype(frame["latitude"]):
        raise TypeError("latitude must be numeric.")

    if not pd.api.types.is_numeric_dtype(frame["longitude"]):
        raise TypeError("longitude must be numeric.")

    if frame["crime_type"].isna().any():
        raise ValueError("crime_type cannot contain missing values.")


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate a crime dataset from a CSV file."""
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    frame = pd.read_csv(csv_path)

    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            errors="coerce",
        )

    validate_dataframe(frame)
    return frame


def save_csv(frame: pd.DataFrame, path: str | Path) -> None:
    """Validate and save a crime dataset as CSV."""
    validate_dataframe(frame)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame.to_csv(output_path, index=False)
