from __future__ import annotations

"""FastAPI service layer for Athlytica's Predictive Injury Logic engine.

This module exposes a clean API surface around the founder-tuned,
explainable injury-risk heuristic used inside Athlytica's Performance
Intelligence OS.
"""

from typing import Any, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from injury_logic import DataValidationError, InjuryRiskEngine, InsufficientDataError


class DailyMetricRecord(BaseModel):
    """Single daily athlete monitoring record.

    Notes:
        - ``date`` is optional but strongly recommended for chronological integrity.
        - ``sleep_quality_score`` supports 0-1, 0-10, or 0-100 scales.
        - The schema is intentionally narrow so Athlytica can preserve a clean,
          audit-friendly data contract across mobile, dashboard, and API clients.
    """

    date: Optional[str] = Field(
        default=None,
        description="Optional ISO date string for the observation. Strongly recommended for ordered workload analysis.",
        examples=["2026-04-08"],
    )
    daily_training_load: float = Field(
        ...,
        ge=0,
        description="Daily external or internal training load captured inside Athlytica's verified monitoring workflow.",
        examples=[125.0],
    )
    heart_rate_variability: float = Field(
        ...,
        description="Daily HRV reading used as Athlytica's biometric recovery signal.",
        examples=[72.5],
    )
    sleep_quality_score: float = Field(
        ...,
        ge=0,
        description="Sleep quality score on a 0-1, 0-10, or 0-100 scale.",
        examples=[82],
    )
    historical_injury_count: int = Field(
        ...,
        ge=0,
        description="Total number of prior injuries logged against the athlete profile.",
        examples=[1],
    )


class InjuryRiskRequest(BaseModel):
    """Request body for injury risk analysis."""

    athlete_id: Optional[str] = Field(
        default=None,
        description="Optional Athlytica athlete identifier for traceability across dashboards, reports, and verified scouting workflows.",
        examples=["ATH-NBO-ICE-001"],
    )
    include_diagnostics: bool = Field(
        default=False,
        description="When true, return ACWR and biometric diagnostics for internal QA, coach dashboards, or model-review workflows.",
    )
    records: List[DailyMetricRecord] = Field(
        ...,
        min_length=28,
        description="Ordered or orderable daily athlete monitoring records. Minimum 28 records are required to compute chronic workload.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "athlete_id": "ATH-NBO-ICE-001",
                "include_diagnostics": True,
                "records": [
                    {
                        "date": "2026-03-05",
                        "daily_training_load": 100,
                        "heart_rate_variability": 75,
                        "sleep_quality_score": 78,
                        "historical_injury_count": 1,
                    },
                    {
                        "date": "2026-03-06",
                        "daily_training_load": 105,
                        "heart_rate_variability": 74,
                        "sleep_quality_score": 80,
                        "historical_injury_count": 1,
                    },
                    {
                        "date": "2026-03-07",
                        "daily_training_load": 110,
                        "heart_rate_variability": 73,
                        "sleep_quality_score": 76,
                        "historical_injury_count": 1,
                    }
                ],
            }
        }
    }


class InjuryRiskResponse(BaseModel):
    """Response contract for injury risk analysis.

    The API returns a narrow core response by default and can optionally attach
    diagnostics so downstream systems can inspect why a given risk state was
    assigned.
    """

    athlete_id: Optional[str] = Field(default=None, description="Echoed Athlytica athlete identifier.")
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_category: str = Field(..., examples=["Low", "Medium", "High"])
    intervention_strategy: str
    diagnostics: Optional[dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Simple health-check response."""

    status: str
    service: str


app = FastAPI(
    title="Athlytica Predictive Injury Logic API",
    description=(
        "API layer for Athlytica's founder-tuned, explainable injury-risk engine. "
        "Built for verified athlete monitoring workflows using workload, HRV, sleep, and injury-history signals."
    ),
    version="1.0.0",
)

engine = InjuryRiskEngine()


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health_check() -> HealthResponse:
    """Basic liveness endpoint for deployment and orchestration checks."""
    return HealthResponse(status="ok", service="athlytica-injury-logic")


@app.post("/injury-risk", response_model=InjuryRiskResponse, tags=["injury-risk"])
def calculate_injury_risk(payload: InjuryRiskRequest) -> InjuryRiskResponse:
    """Calculate the injury risk score for an athlete.

    This endpoint is the operational edge of Athlytica's injury logic engine.
    It expects at least 28 daily records so chronic workload can be computed
    cleanly. When ``include_diagnostics`` is true, the API returns the ACWR and
    biometric context that drove the final classification.
    """
    try:
        df = pd.DataFrame([record.model_dump() for record in payload.records])

        if payload.include_diagnostics:
            result = engine.analyze(df)
        else:
            result = engine.compute_risk(df)

        return InjuryRiskResponse(athlete_id=payload.athlete_id, **result)
    except (DataValidationError, InsufficientDataError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=500,
            detail="Unexpected server error while calculating Athlytica injury risk.",
        ) from exc
