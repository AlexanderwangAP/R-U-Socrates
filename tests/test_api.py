"""API auth gate and health endpoint."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.api import main as api_main

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def client():
    # Entered as a context manager so the app's lifespan runs create_tables().
    with TestClient(api_main.app) as c:
        yield c


def test_health_is_open_and_reports_auth_state(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "auth" in body


def test_blank_database_url_falls_back_to_sqlite():
    """`DATABASE_URL=` in .env is blank, not unset.

    Following the README (copy .env.example, start the API) used to die with
    "Could not parse SQLAlchemy URL" because os.getenv(..., default) returns
    the empty string rather than the default.
    """
    result = subprocess.run(
        [sys.executable, "-c", "from services.api import database; print(database.DATABASE_URL)"],
        capture_output=True,
        text=True,
        env={**os.environ, "DATABASE_URL": ""},
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert "sqlite" in result.stdout


def test_api_is_open_when_no_token_is_configured(client, monkeypatch):
    monkeypatch.setattr(api_main, "API_TOKEN", "")
    res = client.get("/api/tasks")
    assert res.status_code == 200


def test_api_rejects_calls_without_a_token(client, monkeypatch):
    monkeypatch.setattr(api_main, "API_TOKEN", "s3cret")
    assert client.get("/api/tasks").status_code == 401
    assert client.get("/api/tasks", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_api_accepts_a_bearer_token(client, monkeypatch):
    monkeypatch.setattr(api_main, "API_TOKEN", "s3cret")
    res = client.get("/api/tasks", headers={"Authorization": "Bearer s3cret"})
    assert res.status_code == 200


def test_api_creates_a_task_on_an_empty_database(client, monkeypatch):
    """Regression: the suite used to pass only because data/rus.db existed."""
    from services.api.routes import tasks as tasks_module

    # Creating a task queues a real pipeline run. Stub it out — this test is
    # about CRUD against a fresh schema, not about the research loop.
    async def noop_run_pipeline(*args, **kwargs):
        return None

    monkeypatch.setattr(tasks_module, "_run_pipeline", noop_run_pipeline)
    monkeypatch.setattr(api_main, "API_TOKEN", "")

    res = client.post(
        "/api/tasks",
        json={"name": "t", "description": "a long enough description", "max_iterations": 1},
    )
    assert res.status_code == 201

    task_id = res.json()["id"]
    assert client.get(f"/api/tasks/{task_id}").status_code == 200
    assert client.delete(f"/api/tasks/{task_id}").status_code == 204
    assert client.get(f"/api/tasks/{task_id}").status_code == 404


def test_stream_accepts_a_query_token_for_eventsource(client, monkeypatch):
    """EventSource cannot send headers, so the stream also takes ?token="""
    monkeypatch.setattr(api_main, "API_TOKEN", "s3cret")
    assert client.get("/api/tasks/nope/stream?token=wrong").status_code in (401, 404)
    assert client.get("/api/tasks/nope/stream").status_code in (401, 404)
