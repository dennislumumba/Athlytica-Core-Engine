from __future__ import annotations

"""Smoke tests for Athlytica's FastAPI injury-risk service."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from main import app


def main() -> None:
    client = TestClient(app)
    payload = json.loads(Path("example_payload.json").read_text())

    response = client.post("/injury-risk", json=payload)
    assert response.status_code == 200, response.text

    body = response.json()
    assert "risk_score" in body, body
    assert "risk_category" in body, body
    assert "intervention_strategy" in body, body
    assert body["athlete_id"] == "ATH-NBO-ICE-001", body
    assert body["diagnostics"] is not None, body

    health = client.get("/health")
    assert health.status_code == 200, health.text

    print("Athlytica API smoke test passed.")
    print(body)


if __name__ == "__main__":
    main()
