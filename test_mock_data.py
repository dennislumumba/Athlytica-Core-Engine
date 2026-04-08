from __future__ import annotations

"""Validation scenarios for Athlytica's core injury-risk engine."""

import pandas as pd

from injury_logic import InjuryRiskEngine


def build_balanced_dataset() -> pd.DataFrame:
    """Create a stable workload profile that should produce Low risk."""
    records = []
    for day in range(35):
        records.append(
            {
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
                "daily_training_load": 100 + (day % 3),
                "heart_rate_variability": 72 + ((day % 4) - 1),
                "sleep_quality_score": 82,
                "historical_injury_count": 0,
            }
        )
    return pd.DataFrame(records)


def build_overload_dataset() -> pd.DataFrame:
    """Create an overload profile with ACWR > 1.5 and depressed HRV."""
    records = []
    for day in range(35):
        load = 100 if day < 28 else 200
        hrv = 75 if day < 34 else 50
        sleep = 45 if day >= 28 else 70
        records.append(
            {
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
                "daily_training_load": load,
                "heart_rate_variability": hrv,
                "sleep_quality_score": sleep,
                "historical_injury_count": 2,
            }
        )
    return pd.DataFrame(records)


def build_deconditioning_dataset() -> pd.DataFrame:
    """Create an under-training profile with ACWR < 0.8."""
    records = []
    for day in range(35):
        load = 100 if day < 28 else 50
        records.append(
            {
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
                "daily_training_load": load,
                "heart_rate_variability": 70,
                "sleep_quality_score": 78,
                "historical_injury_count": 1,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    engine = InjuryRiskEngine()

    balanced_analysis = engine.analyze(build_balanced_dataset())
    overload_analysis = engine.analyze(build_overload_dataset())
    deconditioning_analysis = engine.analyze(build_deconditioning_dataset())

    print("\n--- Athlytica Balanced Profile ---")
    print(balanced_analysis)
    print("\n--- Athlytica Overload Profile ---")
    print(overload_analysis)
    print("\n--- Athlytica Deconditioning Profile ---")
    print(deconditioning_analysis)

    assert balanced_analysis["risk_category"] == "Low", balanced_analysis
    assert overload_analysis["risk_category"] == "High", overload_analysis
    assert overload_analysis["diagnostics"]["acwr"] > 1.5, overload_analysis
    assert overload_analysis["diagnostics"]["hrv_multiplier_applied"] is True, overload_analysis
    assert deconditioning_analysis["diagnostics"]["acwr"] < 0.8, deconditioning_analysis
    assert deconditioning_analysis["risk_category"] in {"Medium", "High"}, deconditioning_analysis

    print("\nAll Athlytica mock-data checks passed.")


if __name__ == "__main__":
    main()
