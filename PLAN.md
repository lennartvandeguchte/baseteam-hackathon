# Supply Chain Risk Analysis Agent — Plan

## Goal

An autonomous AI agent that takes a supplier and the product a company relies on, researches it using web search and open data, and produces a cited supply chain risk analysis report.

- Hackathon demo that runs end-to-end, architected so it can grow into a real tool
- Python, LangChain `deepagents`, open models via Nebius Token Factory, Tavily for search
- Targets per report: < 5 minutes, ~$1.70 LLM cost, ~100 Tavily credits
- Fully autonomous — no human-in-the-loop

## Input

| Field | Required | Purpose |
|---|---|---|
| Supplier name | yes | Entity to research |
| Product | yes | What is sourced from the supplier |
| Buyer company | no | Frames the risk for *us* |
| Buyer context (single-source?, share of spend, what the product goes into) | no | Adjusts impact score |
| Supplier country / website | no | Helps entity disambiguation |
| Known alternative suppliers | no | Feeds concentration analysis |

One supplier per run. Batch mode is out of scope for v1.

## Risk dimensions (v1)

1. **Financial** — credit, profitability, insolvency signals
2. **Geopolitical & country** — sanctions, export controls, conflict, political stability
3. **Operational** — plant locations, capacity, incidents, recalls, quality issues
4. **Concentration & substitutability** — single-source, alternatives, market share
5. **ESG & compliance** — forced labour, environmental violations, litigation, CSDDD/LkSG exposure
6. **Cyber** — breaches, ransomware incidents

News/reputational sentiment is folded into each dimension.

Out of v1: natural hazard / climate exposure, tier-N (supplier's suppliers) mapping.

## Architecture

```
input
  │
  ▼
Entity resolution ──► /supplier_profile.md
  │
  ▼
Orchestrator (plans via write_todos)
  │  task() × 6, in parallel
  ├─► financial-researcher   ──► /findings/financial.md
  ├─► geopolitical-researcher ──► /findings/geopolitical.md
  ├─► operational-researcher ──► /findings/operational.md
  ├─► concentration-researcher ──► /findings/concentration.md
  ├─► esg-researcher         ──► /findings/esg.md
  └─► cyber-researcher       ──► /findings/cyber.md
  │
  ▼
Critic / verifier × 6, in parallel ──► /reviews/<dimension>.md  (unsupported claims dropped)
  │
  ▼
Orchestrator writes /report.md
```

### Components

**Entity resolution**
- Resolves the supplier to a legal entity: legal name, parent chain, LEI, HQ country, listed/private (ticker), main sites.
- Sources: GLEIF LEI API, Tavily.
- Picks the most likely match autonomously. Records confidence and rejected candidates in `supplier_profile.md`; the report surfaces the assumption up top and flags low confidence as a warning.
- All downstream agents work from this shared profile.

**Orchestrator**
- Built with `create_deep_agent`, with `TodoListMiddleware()` enabled explicitly (off by default since deepagents 0.7).
- Delegates all six dimensions in a single turn so the `task` calls run in parallel.
- Aggregates findings and writes the final report after the critic pass.

**Dimension researchers (6 subagents)**
- Each has its own system prompt, tool set and scoring rubric.
- Hard cap of 6 Tavily searches each.
- Writes `findings/<dimension>.md` with findings, evidence and a 1–5 score.

**Critic / verifier**
- Checks every claim in the findings against its cited source (re-fetching the page with `extract_page`).
- Checks dates against the 24-month window and flags hallucinated or unsupported claims.
- One critic per dimension, all six in parallel (a single sequential critic took ~3 min).
- Output: `reviews/<dimension>.md` with approved / rejected per finding and a suggested score. No rework round: the orchestrator drops rejected findings and applies the suggested scores.

### Filesystem

Agents write real files into `runs/<supplier>-<timestamp>/` (deepagents `FilesystemBackend`). The output files are `supplier_profile.md`, `findings/*.md`, `reviews/*.md` and `report.md`. The final output format and UI will be decided later; for now the deliverable is `report.md` plus the findings files.

## Data sources

Tavily is used by every agent for web and news search. The core structured APIs in v1:

| Dimension | Structured sources (v1) | Via targeted Tavily queries |
|---|---|---|
| Entity | GLEIF LEI API | company registries, Wikipedia |
| Financial | yfinance | SEC EDGAR, Companies House, credit news |
| Geopolitical | OpenSanctions, World Bank WGI | export-control news, GDELT-style news |
| Operational | — | openFDA recalls, EU Safety Gate, incident news |
| Concentration | — | UN Comtrade, market share reports |
| ESG / compliance | — | US DOL forced/child labour list, UFLPA Entity List, EPA ECHO, litigation news |
| Cyber | ransomware.live | breach news |

More sources can be added later as extra tools without changing the architecture.

## Evidence & scoring rules

- **Strict citations:** every finding has a source URL, publication date and quote/snippet. No source means no finding.
- **Recency:** 24-month lookback for news. Older items are marked *historical*.
- **Absence ≠ safety:** each dimension explicitly states "no evidence found" when that is the case, never "low risk" by default.
- **Rubric-based scoring:** the LLM scores 1–5 per dimension against a written rubric per dimension. Rubrics live in prompt/skill files so they are easy to tune.
  - Example (geopolitical): 5 = entity or HQ country under comprehensive sanctions; 1 = OECD country, no sanctions hits.
- **Likelihood vs impact:** the dimension score reflects likelihood. Buyer context (single-source, criticality) adjusts impact separately.
- The report includes an overall rating and the top 3 mitigation actions.

## Model selection (open models on Nebius Token Factory)

Each model is chosen for what the task requires. The tool-calling smoke test (`supply-risk smoke`) provides the evidence that each candidate works in agent loops.

| Role | Task demands | Model | Rationale |
|---|---|---|---|
| Orchestrator + report writer | Long-horizon planning, delegation via tool calls, holding all findings, coherent long-form writing | `moonshotai/Kimi-K3` | Agentic model built for tool use; 1M context holds all findings; fast (1.1 s per call in the smoke test). $3/$15 is acceptable for a low-call-volume role |
| Entity resolution | Short, precise structured extraction | `Qwen/Qwen3-235B-A22B-Instruct-2507` | Fast non-thinking instruct model, strong structured output, cheap; the task is lookup/extraction, not deep reasoning |
| 6 dimension researchers | High-volume search→read→note loops over noisy text; ~70% of tokens; reliable tool calling | `Qwen/Qwen3-235B-A22B-Instruct-2507` | $0.20/$0.60, tool support confirmed on Nebius; MoE with 22B active parameters keeps latency low across 6 parallel agents; 262K context is ample per dimension |
| Critic / verifier | Careful claim-vs-source reasoning; must not share the writers' blind spots | `deepseek-ai/DeepSeek-V4-Pro-0813` | Strong reasoning, and a **different model family** from both the researchers (Qwen) and the writer (Kimi). This avoids self-preference bias and decorrelates errors |

**Excluded (with reasons, for the pitch):**
- `openai/gpt-oss-120b`: documented tool-calling bugs on vLLM-style serving (tool calls land in `content`, parallel calls regress). Too risky for agent loops.
- `zai-org/GLM-5.3`: the original orchestrator pick. It passed the smoke test but took 28.5 s per call (Kimi-K3: 1.1 s); with ~20 sequential orchestrator turns that breaks the 5-minute target. Kept as a cheaper fallback.
- `openai/gpt-oss-120b` (smoke test): made only 1 of 2 parallel calls and gave no correct answer after the tool result.
- `DeepSeek-V4-Flash`, `GLM-5.3-Flash`: cheaper backup candidates for the researcher role (swap via `SUPPLY_RISK_MODEL_RESEARCHER`).

**Estimated cost per report:**

| Role | Tokens (in / out) | Cost |
|---|---|---|
| Researchers | ~900K / 60K | ~$0.22 |
| Orchestrator | ~300K / 20K | ~$1.20 |
| Critic | ~200K / 10K | ~$0.24 |
| **Total** | | **~$1.70** |

**Caveats:**
- Prices and context sizes come from third-party listings. Verify them against the Nebius catalogue during setup.
- Nebius deprecates serverless models regularly. Pin model IDs in a single config file.
- Double-check licences (MIT / Apache 2.0 expected) for the exact versions used.

**Integration:** use `ChatOpenAI(base_url="https://api.tokenfactory.nebius.com/v1/", api_key=NEBIUS_API_KEY, model=...)`. The `langchain-nebius` package is stale (last release 2025-06, old Studio base URL).

## Tavily budget (8,000 free credits)

- ~60 searches per report; ~100 credits with the occasional advanced search or extract. That is roughly 80 full reports, enough for development, eval and demo.
- `search_depth="basic"` by default; `advanced` and page extraction only for key sources.
- Hard cap of 6 searches per researcher and for entity resolution.

## Evaluation

Kept minimal for the demo:
- **Model smoke test** (`supply-risk smoke`): per candidate model it checks for a valid tool call, no tool-call JSON leaking into the text, parallel calls, and an answer after the tool result. The resulting table is the evidence for the model choice per role.
- **Manual check** on a few suppliers with known, verifiable events (e.g. a ransomware incident, a bankruptcy, an export-control conflict), plus one clean supplier to catch invented risks.
- No tracing, golden-set runner or automated model bake-off; these can be added later without changing the agent.

## Build order

1. **Model smoke test:** verify tool calling on all candidate Nebius models; pin model IDs in config.
2. **Tools:** Tavily search/extract with a per-agent search cap; GLEIF, OpenSanctions, yfinance, World Bank WGI, ransomware.live.
3. **Entity resolution** producing `supplier_profile.md`.
4. **One dimension researcher end-to-end** (e.g. geopolitical), including rubric and citation format.
5. **Remaining five researchers,** run in parallel via the orchestrator.
6. **Critic / verifier** pass and revision loop.
7. **Manual check** on a few suppliers with known events (see Evaluation).
8. **Demo polish** (UI and output format, to be decided).

## Open items

- Team size and deadline, which set how the build order is split into milestones.
- Which suppliers to use for the manual check and the demo.
- UI and final report output format (deferred).
