from fastapi.testclient import TestClient

from supply_risk.agent import Request
from supply_risk.api import RunStatus, create_api

TOKEN = {"x-api-token": "secret"}


class FakeJobs:
    def __init__(self):
        self.started: list[Request] = []
        self.runs = {
            "r1": RunStatus(status="running", steps=["orchestrator: task(entity-resolver)"]),
            "r2": RunStatus(status="done", steps=[], report="# Report"),
            "r3": RunStatus(status="failed", error="RuntimeError: boom"),
        }

    def start(self, request):
        self.started.append(request)
        return "new"

    def status(self, run_id):
        return self.runs.get(run_id)


def client(jobs=None):
    return TestClient(create_api(jobs or FakeJobs(), "secret"))


def test_token_is_required():
    c = client()
    assert c.get("/runs/r1").status_code == 401
    assert c.get("/runs/r1", headers={"x-api-token": "wrong"}).status_code == 401
    assert c.post("/runs", json={"supplier": "X", "product": "Y"}).status_code == 401


def test_start_run_passes_the_request():
    jobs = FakeJobs()
    body = {"supplier": "Nexperia", "product": "discretes", "single_source": True}
    response = client(jobs).post("/runs", json=body, headers=TOKEN)
    assert response.json() == {"run_id": "new"}
    assert jobs.started == [Request(supplier="Nexperia", product="discretes", single_source=True)]


def test_start_run_validates_the_body():
    assert client().post("/runs", json={"supplier": "X"}, headers=TOKEN).status_code == 422


def test_run_status():
    c = client()
    assert c.get("/runs/r1", headers=TOKEN).json()["steps"] == ["orchestrator: task(entity-resolver)"]
    assert c.get("/runs/r2", headers=TOKEN).json()["report"] == "# Report"
    assert c.get("/runs/r3", headers=TOKEN).json() == {
        "status": "failed", "steps": [], "report": None, "error": "RuntimeError: boom",
    }
    assert c.get("/runs/nope", headers=TOKEN).status_code == 404


def test_cors_preflight_from_lovable():
    preflight = {"access-control-request-method": "POST", "access-control-request-headers": "x-api-token,content-type"}
    ok = client().options("/runs", headers={"origin": "https://my-app.lovable.app", **preflight})
    assert ok.headers["access-control-allow-origin"] == "https://my-app.lovable.app"
    other = client().options("/runs", headers={"origin": "https://evil.example", **preflight})
    assert "access-control-allow-origin" not in other.headers
