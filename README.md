# Supply chain risk agent

Autonomous agent that takes a supplier and the product you source from it, researches it with web search
and open data, and writes a cited supply chain risk report. It is built on LangChain
[`deepagents`](https://docs.langchain.com/oss/python/deepagents/overview) and uses only open models served by
Nebius Token Factory. See [PLAN.md](PLAN.md) for the design decisions.

## How it works

```
entity-resolver ─► /supplier_profile.md
orchestrator ─► 6 researchers in parallel ─► /findings/<dimension>.md
             ─► 6 critics in parallel ─► /reviews/<dimension>.md ─► /report.md
```

| Role | Model | Why |
|---|---|---|
| Orchestrator + report writer | `moonshotai/Kimi-K3` | Fast agentic tool use (1.1 s/call in the smoke test vs 28.5 s for GLM-5.3), 1M context holds all findings |
| Entity resolution | `Qwen/Qwen3-235B-A22B-Instruct-2507` | Fast, cheap structured extraction |
| 6 dimension researchers | `Qwen/Qwen3-235B-A22B-Instruct-2507` | Cheap, reliable tool calling, low-latency MoE for parallel runs |
| Critic / verifier | `deepseek-ai/DeepSeek-V4-Pro-0813` | Strong reasoning, from a different model family than the writers |

**Risk dimensions:** financial, geopolitical, operational, concentration, ESG/compliance, cyber. Each is scored 1–5
against a rubric. Buyer context (single source, criticality) is applied separately as impact.

**Tools:**
- Tavily web search, capped at 6 searches per researcher, and page extraction
- GLEIF (legal entity)
- OpenSanctions (optional API key)
- World Bank governance indicators
- Yahoo Finance
- ransomware.live

## Code

| File | Contents |
|---|---|
| `src/supply_risk/config.py` | Model per role and the Nebius client |
| `src/supply_risk/prompts.py` | Risk dimensions, rubrics and all system prompts |
| `src/supply_risk/tools.py` | Search and open-data tools |
| `src/supply_risk/agent.py` | deepagents wiring and `run()` |
| `src/supply_risk/smoke.py` | Tool-calling smoke test of candidate models |
| `src/supply_risk/cli.py` | Command line |
| `src/supply_risk/api.py` | HTTP API (start a run, poll its status) |
| `modal_app.py` | Modal deployment of the API and the background job |

## Usage

Setup:

```bash
uv sync
```

```bash
cp .env.example .env
```

Then fill in `NEBIUS_API_KEY` and `TAVILY_API_KEY`.

Check which candidate models handle tool calling reliably. This is the evidence for the model choice per role:

```bash
uv run supply-risk smoke
```

Analyse a supplier:

```bash
uv run supply-risk run --supplier "Nexperia" --product "automotive discrete semiconductors" --buyer "Acme Automotive" --single-source --used-in "engine control units"
```

The output goes to `runs/<supplier>-<timestamp>/`:
- `report.md`
- `supplier_profile.md`
- `findings/*.md`
- `reviews/*.md`

## Deploy on Modal (HTTP API for a website)

`modal_app.py` serves `src/supply_risk/api.py` on Modal. Each analysis runs as a background job.

- `POST /runs` takes the same fields as the CLI (`supplier`, `product`, `buyer`, `single_source`, `used_in`,
  `country`) and returns `{"run_id": ...}`.
- `GET /runs/{run_id}` returns `{status: running|done|failed, steps, report, error}`.

Every request needs an `x-api-token` header. Run folders are kept in the Modal volume `supply-risk-runs`.

```bash
uv run modal setup
```

```bash
uv run modal secret create supply-risk NEBIUS_API_KEY=... TAVILY_API_KEY=... API_TOKEN=$(openssl rand -hex 24)
```

```bash
uv run modal deploy modal_app.py
```

## Tests

```bash
uv run pytest
```

No API keys are needed. External calls are faked, and one end-to-end deepagents run uses scripted fake models.
