"""Modal deployment: `uv run modal deploy modal_app.py`. See README "Deploy on Modal"."""

import os
import uuid
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync()
    .add_local_python_source("supply_risk")
)
app = modal.App("supply-risk", image=image)
jobs_store = modal.Dict.from_name("supply-risk-jobs", create_if_missing=True)
runs_volume = modal.Volume.from_name("supply-risk-runs", create_if_missing=True)
secrets = [modal.Secret.from_name("supply-risk")]  # NEBIUS_API_KEY, TAVILY_API_KEY, API_TOKEN, OPENSANCTIONS_API_KEY?


@app.function(secrets=secrets, volumes={"/runs": runs_volume}, timeout=30 * 60)
def analyse(run_id: str, request: dict) -> str:
    from supply_risk.agent import Request, run

    steps: list[str] = []

    def on_step(step: str) -> None:
        steps.append(step)
        jobs_store[run_id] = {**jobs_store[run_id], "steps": steps}

    try:
        report = run(Request(**request), Path("/runs") / run_id, on_step=on_step)
    finally:
        runs_volume.commit()
    if not report.exists():
        raise RuntimeError("The agent finished without writing report.md")
    return report.read_text()


class ModalJobs:
    def start(self, request):
        run_id = uuid.uuid4().hex[:12]
        jobs_store[run_id] = {"call_id": None, "steps": []}
        call = analyse.spawn(run_id, request.model_dump())
        jobs_store[run_id] = {**jobs_store[run_id], "call_id": call.object_id}
        return run_id

    def status(self, run_id):
        from supply_risk.api import RunStatus

        job = jobs_store.get(run_id)
        if job is None:
            return None
        steps = job["steps"]
        if job["call_id"] is None:
            return RunStatus(status="running", steps=steps)
        try:
            report = modal.FunctionCall.from_id(job["call_id"]).get(timeout=0)
        except (TimeoutError, modal.exception.TimeoutError):
            return RunStatus(status="running", steps=steps)
        except Exception as e:
            return RunStatus(status="failed", steps=steps, error=f"{type(e).__name__}: {e}")
        return RunStatus(status="done", steps=steps, report=report)


@app.function(secrets=secrets)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def api():
    from supply_risk.api import create_api

    return create_api(ModalJobs(), os.environ["API_TOKEN"])
