"""HTTP API: start an analysis in the background and poll it. Deployed on Modal by modal_app.py."""

import secrets
from typing import Literal, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from supply_risk.agent import Request

# Lovable preview and published sites, plus a local dev server
ALLOWED_ORIGINS = r"https://([a-z0-9-]+\.)*(lovable\.app|lovableproject\.com)|http://localhost(:\d+)?"


class RunStatus(BaseModel):
    status: Literal["running", "done", "failed"]
    steps: list[str] = []
    report: str | None = None
    error: str | None = None


class Jobs(Protocol):
    def start(self, request: Request) -> str: ...

    def status(self, run_id: str) -> RunStatus | None: ...


def create_api(jobs: Jobs, token: str) -> FastAPI:
    def check_token(x_api_token: str | None = Header(None)) -> None:
        if x_api_token is None or not secrets.compare_digest(x_api_token, token):
            raise HTTPException(401, "Missing or wrong x-api-token header")

    app = FastAPI(title="Supply chain risk agent", dependencies=[Depends(check_token)])
    app.add_middleware(
        CORSMiddleware, allow_origin_regex=ALLOWED_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"]
    )

    @app.post("/runs")
    def start_run(request: Request) -> dict[str, str]:
        return {"run_id": jobs.start(request)}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> RunStatus:
        status = jobs.status(run_id)
        if status is None:
            raise HTTPException(404, "Unknown run_id")
        return status

    return app
