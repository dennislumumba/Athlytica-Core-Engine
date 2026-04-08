# Athlytica Predictive Injury Logic API

A focused FastAPI service for Athlytica's **Predictive Injury Logic** engine — an explainable, deployment-ready heuristic for estimating athlete injury risk from workload and recovery signals.

This repository is opinionated by design. It does **not** pretend to be a black-box AI model. It is a transparent rules engine built for operational use inside Athlytica's Performance Intelligence OS, where traceability matters more than hype.

## Why this exists

Most sports-tech injury tools fail one of two ways:

1. They are too vague to be trusted by serious operators.
2. They are too academic to be deployed in real coaching environments.

This service sits in the middle. It is built to be:

- **Explainable** enough for technical review
- **Strict** enough for clean downstream integrations
- **Simple** enough to wrap into dashboards, mobile apps, and internal tools
- **Extensible** enough for future calibration once Athlytica's verified cohort expands

## Core logic

The engine computes an **Injury Risk Score** from `0.0` to `1.0` using:

- **Acute Workload**: 7-day rolling average of `daily_training_load`
- **Chronic Workload**: 28-day rolling average of `daily_training_load`
- **ACWR flags**:
  - `> 1.5` → overload / danger zone
  - `< 0.8` → under-training / deconditioning
- **Biometric multiplier**:
  - if HRV drops `2 standard deviations` below the athlete's mean, the risk score is multiplied by `1.27`
- **Context modifiers**:
  - sleep quality
  - historical injury count

The service returns:

- `risk_score`
- `risk_category` (`Low`, `Medium`, `High`)
- `intervention_strategy`
- optional `diagnostics`

## Repository structure

```text
.
├── injury_logic.py        # Core injury-risk engine
├── main.py                # FastAPI service layer
├── example_payload.json   # Example request body
├── test_mock_data.py      # Core engine validation script
├── test_api.py            # API smoke test
├── requirements.txt       # Runtime dependencies
└── README.md              # Project documentation
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the API locally

```bash
uvicorn main:app --reload
```

### 3. Open the interactive docs

- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## API contract

### `GET /health`

Basic liveness probe.

**Example response**

```json
{
  "status": "ok",
  "service": "athlytica-injury-logic"
}
```

### `POST /injury-risk`

Calculate injury risk from daily monitoring records.

#### Request body

```json
{
  "athlete_id": "ATH-NBO-ICE-001",
  "include_diagnostics": true,
  "records": [
    {
      "date": "2026-03-01",
      "daily_training_load": 100,
      "heart_rate_variability": 75,
      "sleep_quality_score": 78,
      "historical_injury_count": 1
    }
  ]
}
```

#### Required fields per record

- `daily_training_load`
- `heart_rate_variability`
- `sleep_quality_score`
- `historical_injury_count`

#### Optional field

- `date`

> Minimum dataset length: **28 records**

#### Example response

```json
{
  "athlete_id": "ATH-NBO-ICE-001",
  "risk_score": 0.25,
  "risk_category": "Low",
  "intervention_strategy": "Maintain the current load and continue standard daily monitoring within Athlytica's verified performance framework.",
  "diagnostics": {
    "acute_workload": 100.571,
    "chronic_workload": 100.5,
    "acwr": 1.001,
    "acwr_flag": "Stable Range",
    "latest_hrv": 75.0,
    "hrv_mean": 75.0,
    "hrv_threshold": 73.586,
    "hrv_multiplier_applied": false,
    "normalized_sleep_score": 0.79,
    "historical_injury_count": 1,
    "records_analyzed": 28
  }
}
```

## Running tests

### Validate the core logic

```bash
python test_mock_data.py
```

### Smoke test the API

```bash
python test_api.py
```

## Design principles

### 1. Explainability first
This repository is designed so a coach, analyst, engineer, or auditor can understand *why* a score was returned.

### 2. Founder-tuned, not over-claimed
The coefficients are opinionated Athlytica defaults. They are meant to be inspected, challenged, and later recalibrated against real cohort outcomes.

### 3. Narrow schema, stronger system
Loose schemas create fake intelligence. This project keeps the request contract intentionally narrow so downstream clients cannot pollute the logic with inconsistent inputs.

### 4. API-ready core
The engine lives in `injury_logic.py`, independent of FastAPI. That separation makes it easy to:

- reuse in background jobs
- wrap in other services
- test independently
- evolve without breaking the API layer

## Operational notes

- Negative workloads and injury counts are rejected
- Invalid JSON or non-numeric fields raise validation errors
- Fewer than 28 records raises an insufficient-data error
- Sleep scores are accepted on `0-1`, `0-10`, or `0-100` scales
- Diagnostics are optional to keep the default response tight

## What this is not

This is **not**:

- a medical device
- a clinically validated diagnosis engine
- a substitute for professional medical judgment
- a black-box machine learning model

It is an operational injury-risk heuristic designed to support better decisions in athlete monitoring workflows.

## Next upgrades

High-leverage follow-ups for this repository:

1. Add authentication and tenant isolation
2. Persist predictions for audit and calibration history
3. Add Docker packaging and deployment manifests
4. Introduce outcome logging for future coefficient tuning
5. Add versioned model metadata to every response

## License / usage

Add your preferred license before public release.

If this repo is staying private, keep the same discipline anyway: a private repository with sloppy boundaries is still a liability.
