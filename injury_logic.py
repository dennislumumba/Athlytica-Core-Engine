from __future__ import annotations

"""Athlytica Predictive Injury Logic.

Personalized for Dennis Lumumba's Athlytica operating model: conservative,
explainable, and deployment-ready for verified athlete monitoring workflows.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Union

import pandas as pd

DataLike = Union[str, Mapping[str, Any], Sequence[Mapping[str, Any]], pd.DataFrame]


class InjuryLogicError(Exception):
    """Base exception for all injury risk engine errors."""


class DataValidationError(InjuryLogicError):
    """Raised when the incoming dataset is missing required structure or values."""


class InsufficientDataError(InjuryLogicError):
    """Raised when there is not enough historical data to compute the rolling metrics."""


@dataclass(frozen=True)
class InjuryRiskConfig:
    """Configuration object for the injury risk engine.

    Attributes:
        acute_window: Number of days used for the acute workload rolling average.
        chronic_window: Number of days used for the chronic workload rolling average.
        acwr_danger_upper: Upper ACWR threshold for overload risk.
        acwr_danger_lower: Lower ACWR threshold for deconditioning risk.
        hrv_drop_std_threshold: Number of standard deviations below the user's
            mean HRV that triggers the biometric risk multiplier.
        hrv_risk_multiplier: Multiplier applied to the final risk when the HRV
            danger condition is met.
        low_risk_threshold: Upper bound for the Low risk category.
        medium_risk_threshold: Upper bound for the Medium risk category.
        max_injury_history_penalty: Maximum additive penalty contributed by
            historical injury count.
    """

    acute_window: int = 7
    chronic_window: int = 28
    acwr_danger_upper: float = 1.5
    acwr_danger_lower: float = 0.8
    hrv_drop_std_threshold: float = 2.0
    # Athlytica-specific heuristic chosen for slightly more conservative
    # sensitivity in dense training environments.
    hrv_risk_multiplier: float = 1.27
    low_risk_threshold: float = 0.35
    medium_risk_threshold: float = 0.65
    max_injury_history_penalty: float = 0.15


class InjuryRiskEngine:
    """Athlytica Predictive Injury Logic engine.

    This class converts daily training and biometric data into an injury risk
    score between 0.0 and 1.0. The primary signal is the Acute:Chronic
    Workload Ratio (ACWR), strengthened by biometric variance, sleep quality,
    and historical injury burden.

    Expected inputs:
        - daily_training_load
        - heart_rate_variability
        - sleep_quality_score
        - historical_injury_count

    Optional input:
        - date (used only for sorting and traceability)

    Typical usage:
        >>> engine = InjuryRiskEngine()
        >>> result = engine.compute_risk(df)
        >>> result["risk_score"]
        0.74

    The class is intentionally framework-agnostic so it can later be wrapped in
    a FastAPI route or another service layer without changes to the core logic.
    """

    REQUIRED_COLUMNS = {
        "daily_training_load",
        "heart_rate_variability",
        "sleep_quality_score",
        "historical_injury_count",
    }

    def __init__(self, config: InjuryRiskConfig | None = None) -> None:
        """Initialize the injury risk engine.

        Args:
            config: Optional configuration override. If omitted, defaults are
                used.
        """
        self.config = config or InjuryRiskConfig()

    def compute_risk(self, data: DataLike) -> Dict[str, Any]:
        """Compute the final injury risk output.

        Args:
            data: JSON string, JSON-like mapping/list, or a pandas DataFrame.

        Returns:
            A dictionary containing exactly the core fields required by the
            Athlytica injury engine contract:
            - risk_score: float from 0.0 to 1.0
            - risk_category: Low, Medium, or High
            - intervention_strategy: recommended action string

        Raises:
            DataValidationError: If required columns are missing or invalid.
            InsufficientDataError: If the dataset is shorter than the chronic
                workload window.
        """
        analysis = self.analyze(data)
        return {
            "risk_score": analysis["risk_score"],
            "risk_category": analysis["risk_category"],
            "intervention_strategy": analysis["intervention_strategy"],
        }

    def analyze(self, data: DataLike) -> Dict[str, Any]:
        """Run the complete injury risk analysis with diagnostics.

        This is useful for internal debugging, dashboards, QA, or future API
        responses where you want both the final risk output and the underlying
        signal values.

        Args:
            data: JSON string, JSON-like mapping/list, or a pandas DataFrame.

        Returns:
            A dictionary with the requested output fields plus a diagnostics
            payload containing ACWR and biometric context.
        """
        df = self._load_input(data)
        prepared = self._prepare_dataframe(df)
        latest = prepared.iloc[-1]

        base_risk = self._base_risk_from_acwr(float(latest["acwr"]))
        sleep_penalty = self._sleep_penalty(float(latest["sleep_quality_score"]))
        injury_history_penalty = self._injury_history_penalty(
            float(latest["historical_injury_count"])
        )

        risk = base_risk + sleep_penalty + injury_history_penalty

        hrv_triggered, hrv_threshold = self._is_hrv_drop_triggered(prepared)
        if hrv_triggered:
            risk *= self.config.hrv_risk_multiplier

        risk_score = round(self._clamp(risk, 0.0, 1.0), 3)
        risk_category = self._categorize_risk(risk_score)
        intervention_strategy = self._build_intervention_strategy(
            acwr=float(latest["acwr"]),
            hrv_triggered=hrv_triggered,
            sleep_quality_score=float(latest["sleep_quality_score"]),
            historical_injury_count=int(round(float(latest["historical_injury_count"]))),
            risk_category=risk_category,
        )

        diagnostics = {
            "acute_workload": round(float(latest["acute_workload"]), 3),
            "chronic_workload": round(float(latest["chronic_workload"]), 3),
            "acwr": round(float(latest["acwr"]), 3),
            "acwr_flag": self._acwr_flag(float(latest["acwr"])),
            "latest_hrv": round(float(latest["heart_rate_variability"]), 3),
            "hrv_mean": round(float(prepared["heart_rate_variability"].mean()), 3),
            "hrv_threshold": round(float(hrv_threshold), 3),
            "hrv_multiplier_applied": hrv_triggered,
            "normalized_sleep_score": round(
                self._normalize_sleep_score(float(latest["sleep_quality_score"])), 3
            ),
            "historical_injury_count": int(round(float(latest["historical_injury_count"]))),
            "records_analyzed": int(len(prepared)),
        }

        return {
            "risk_score": risk_score,
            "risk_category": risk_category,
            "intervention_strategy": intervention_strategy,
            "diagnostics": diagnostics,
        }

    def _load_input(self, data: DataLike) -> pd.DataFrame:
        """Normalize supported input formats into a DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data.copy()

        if isinstance(data, str):
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError as exc:
                raise DataValidationError("Input string is not valid JSON.") from exc
            return self._coerce_json_like_to_dataframe(parsed)

        return self._coerce_json_like_to_dataframe(data)

    def _coerce_json_like_to_dataframe(
        self, data: Mapping[str, Any] | Sequence[Mapping[str, Any]]
    ) -> pd.DataFrame:
        """Convert JSON-like payloads into a DataFrame.

        Supported shapes:
            - list[dict] representing row records
            - dict[str, list] representing column-oriented data
            - dict representing a single row
        """
        if isinstance(data, Mapping):
            if any(isinstance(value, list) for value in data.values()):
                return pd.DataFrame(data)
            return pd.DataFrame([data])

        if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            return pd.DataFrame(list(data))

        raise DataValidationError(
            "Unsupported input type. Provide a DataFrame, JSON string, dict, or list of dicts."
        )

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate, sort, and enrich the input DataFrame with rolling features."""
        if df.empty:
            raise DataValidationError("Input dataset is empty.")

        missing_columns = self.REQUIRED_COLUMNS.difference(df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise DataValidationError(f"Missing required columns: {missing}")

        prepared = df.copy()

        if "date" in prepared.columns:
            prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
            if prepared["date"].isna().any():
                raise DataValidationError("The 'date' column contains invalid date values.")
            prepared = prepared.sort_values("date")

        numeric_columns = list(self.REQUIRED_COLUMNS)
        for column in numeric_columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
            if prepared[column].isna().any():
                raise DataValidationError(
                    f"Column '{column}' contains null or non-numeric values."
                )

        if (prepared["daily_training_load"] < 0).any():
            raise DataValidationError("daily_training_load cannot contain negative values.")

        if (prepared["historical_injury_count"] < 0).any():
            raise DataValidationError(
                "historical_injury_count cannot contain negative values."
            )

        if len(prepared) < self.config.chronic_window:
            raise InsufficientDataError(
                f"At least {self.config.chronic_window} rows are required to compute chronic workload."
            )

        prepared = prepared.reset_index(drop=True)
        prepared["acute_workload"] = (
            prepared["daily_training_load"]
            .rolling(window=self.config.acute_window, min_periods=self.config.acute_window)
            .mean()
        )
        prepared["chronic_workload"] = (
            prepared["daily_training_load"]
            .rolling(window=self.config.chronic_window, min_periods=self.config.chronic_window)
            .mean()
        )

        latest_chronic = prepared["chronic_workload"].iloc[-1]
        if pd.isna(latest_chronic) or latest_chronic <= 0:
            raise InsufficientDataError(
                "Chronic workload is unavailable or zero for the latest observation."
            )

        prepared["acwr"] = prepared["acute_workload"] / prepared["chronic_workload"]

        if pd.isna(prepared["acwr"].iloc[-1]):
            raise InsufficientDataError(
                "ACWR could not be computed for the latest observation."
            )

        return prepared

    def _base_risk_from_acwr(self, acwr: float) -> float:
        """Translate ACWR into a base injury risk score.

        The score intentionally centers around 1.0 as the stable workload zone,
        penalizes under-training below 0.8, and sharply escalates above 1.5.
        """
        if acwr > self.config.acwr_danger_upper:
            overload_severity = min((acwr - self.config.acwr_danger_upper) / 0.5, 1.0)
            return 0.70 + (0.20 * overload_severity)

        if acwr < self.config.acwr_danger_lower:
            undertraining_severity = min(
                (self.config.acwr_danger_lower - acwr) / 0.5,
                1.0,
            )
            return 0.45 + (0.20 * undertraining_severity)

        deviation = abs(acwr - 1.0)
        stability_penalty = min(deviation / 0.5, 1.0) * 0.20
        return 0.15 + stability_penalty

    def _is_hrv_drop_triggered(self, df: pd.DataFrame) -> tuple[bool, float]:
        """Check whether latest HRV is materially depressed.

        Returns:
            A tuple of:
            - boolean flag indicating whether the HRV risk multiplier should be applied
            - the computed HRV threshold used for the comparison
        """
        hrv_series = df["heart_rate_variability"]
        mean_hrv = float(hrv_series.mean())
        std_hrv = float(hrv_series.std(ddof=0))

        if std_hrv == 0.0:
            return False, mean_hrv

        threshold = mean_hrv - (self.config.hrv_drop_std_threshold * std_hrv)
        latest_hrv = float(hrv_series.iloc[-1])
        return latest_hrv < threshold, threshold

    def _sleep_penalty(self, raw_sleep_score: float) -> float:
        """Convert sleep quality to an additive risk penalty.

        The method accepts flexible input scales:
            - 0.0 to 1.0
            - 0 to 10
            - 0 to 100

        Higher sleep quality reduces the penalty. Excellent sleep does not erase
        real workload risk; it only limits the added penalty.
        """
        normalized = self._normalize_sleep_score(raw_sleep_score)
        if normalized >= 0.85:
            return 0.0
        if normalized >= 0.65:
            return 0.05
        if normalized >= 0.45:
            return 0.10
        return 0.15

    def _normalize_sleep_score(self, raw_sleep_score: float) -> float:
        """Normalize sleep quality into a 0.0 to 1.0 range."""
        if raw_sleep_score < 0:
            raise DataValidationError("sleep_quality_score cannot be negative.")

        if raw_sleep_score <= 1.0:
            normalized = raw_sleep_score
        elif raw_sleep_score <= 10.0:
            normalized = raw_sleep_score / 10.0
        elif raw_sleep_score <= 100.0:
            normalized = raw_sleep_score / 100.0
        else:
            raise DataValidationError(
                "sleep_quality_score must be on a 0-1, 0-10, or 0-100 scale."
            )

        return self._clamp(normalized, 0.0, 1.0)

    def _injury_history_penalty(self, historical_injury_count: float) -> float:
        """Convert injury history into an additive penalty with a sensible cap."""
        count = max(0.0, historical_injury_count)
        return min(count * 0.05, self.config.max_injury_history_penalty)

    def _categorize_risk(self, risk_score: float) -> str:
        """Map a continuous risk score into Athlytica risk categories."""
        if risk_score < self.config.low_risk_threshold:
            return "Low"
        if risk_score < self.config.medium_risk_threshold:
            return "Medium"
        return "High"

    def _acwr_flag(self, acwr: float) -> str:
        """Return a human-readable ACWR classification."""
        if acwr > self.config.acwr_danger_upper:
            return "Danger Zone"
        if acwr < self.config.acwr_danger_lower:
            return "Under-training / Deconditioning"
        return "Stable Range"

    def _build_intervention_strategy(
        self,
        *,
        acwr: float,
        hrv_triggered: bool,
        sleep_quality_score: float,
        historical_injury_count: int,
        risk_category: str,
    ) -> str:
        """Generate a practical intervention message based on dominant risk drivers."""
        # Dennis Lumumba design note:
        # Logic refined to fit Athlytica's high-density training reality and
        # Banister-style impulse-response thinking. Keep this explainable until
        # cohort-level calibration data justifies a more adaptive policy layer.
        normalized_sleep = self._normalize_sleep_score(sleep_quality_score)

        if acwr > self.config.acwr_danger_upper:
            return (
                "Reduce load by 20% immediately and remove high-intensity work for 48 hours. "
                "Consult Athlytica's Tier-3 recovery protocols for localized muscle-fatigue mitigation."
            )

        if acwr < self.config.acwr_danger_lower:
            return (
                "Progressively rebuild load by 10-15% over 7-14 days to avoid deconditioning. "
                "Integrate Athlytica Tier-1 mobility drills before increasing intensity."
            )

        if hrv_triggered:
            return (
                "Biometric depression detected. Prioritize recovery today and consult "
                "Athlytica's Tier-3 recovery protocols for localized muscle-fatigue mitigation."
            )

        if normalized_sleep < 0.65:
            return (
                "Inadequate recovery markers. Keep load stable and prioritize sleep hygiene "
                "as per Athlytica's recovery optimization framework."
            )

        if historical_injury_count >= 3 and risk_category != "Low":
            return (
                "Maintain conservative progression. Tighten recovery monitoring and consult "
                "Athlytica's Tier-3 recovery protocols for localized muscle-fatigue mitigation."
            )

        return (
            "Maintain the current load and continue standard daily monitoring within "
            "Athlytica's verified performance framework."
        )

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        """Clamp a numeric value to a fixed range."""
        return max(minimum, min(value, maximum))
