from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "latitude_grid",
    "longitude_grid",
]


class CrimeRiskModel:
    """Random-forest model for estimating crime-risk scores."""

    def __init__(
        self,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(
        self,
        frame: pd.DataFrame,
        target: pd.Series,
    ) -> "CrimeRiskModel":
        """Train the risk model."""
        X = frame[FEATURE_COLUMNS]
        y = pd.Series(target).astype(float)

        self.model.fit(X, y)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Predict crime-risk scores."""
        X = frame[FEATURE_COLUMNS]
        return pd.Series(
            self.model.predict(X),
            index=frame.index,
            name="predicted_risk",
        )

    def evaluate(
        self,
        frame: pd.DataFrame,
        target: pd.Series,
    ) -> dict[str, float]:
        """Evaluate predictions using common regression metrics."""
        actual = pd.Series(target).astype(float)
        predicted = self.predict(frame)

        return {
            "mae": float(
                mean_absolute_error(actual, predicted)
            ),
            "rmse": float(
                mean_squared_error(
                    actual,
                    predicted,
                ) ** 0.5
            ),
        }

    def save(self, path: str | Path) -> None:
        """Save the trained model to disk."""
        output_path = Path(path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        joblib.dump(self, output_path)

    @staticmethod
    def load(path: str | Path) -> "CrimeRiskModel":
        """Load a previously saved model."""
        loaded = joblib.load(path)

        if not isinstance(loaded, CrimeRiskModel):
            raise TypeError(
                "The saved object is not a CrimeRiskModel."
            )

        return loaded
