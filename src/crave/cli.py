from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_csv
from .features import aggregate_risk_cells
from .forecast import add_risk_level, create_daily_forecast


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crave",
        description=(
            "CRAVE: Crime Risk Analytics and "
            "Forecasting Engine"
        ),
    )

    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to a crime incident CSV file.",
    )

    parser.add_argument(
        "--forecast-days",
        type=int,
        default=7,
        help="Number of days to forecast.",
    )

    parser.add_argument(
        "--top-cells",
        type=int,
        default=10,
        help="Number of highest-risk geographic cells to display.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    frame = load_csv(args.dataset)

    print()
    print("CRAVE")
    print("=" * 60)
    print(f"Incidents loaded: {len(frame)}")

    print()
    print("Top geographic risk cells")
    print("-" * 60)

    risk_cells = aggregate_risk_cells(frame)

    print(
        risk_cells.head(args.top_cells).to_string(
            index=False
        )
    )

    print()
    print("Daily forecast")
    print("-" * 60)

    forecast = create_daily_forecast(
        frame,
        days=args.forecast_days,
    )

    forecast = add_risk_level(forecast)

    print(forecast.to_string(index=False))

    print()
    print("CRAVE analysis completed.")


if __name__ == "__main__":
    main()
