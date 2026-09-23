"""Model choice per role, and the Nebius Token Factory client.

Model IDs are pinned here (Nebius deprecates serverless models regularly); override any role with
SUPPLY_RISK_MODEL_<ROLE>, e.g. SUPPLY_RISK_MODEL_RESEARCHER=deepseek-ai/DeepSeek-V4-Flash-0731.
"""

import os

from langchain_openai import ChatOpenAI

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

MODELS = {
    # Planning, delegation and long-context report writing: fast agentic model with 1M context.
    # (GLM-5.3 passed the smoke test too, but took ~28 s per call — too slow for ~20 sequential turns.)
    "orchestrator": "moonshotai/Kimi-K3",
    # Short structured extraction: fast, cheap instruct model.
    "entity": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    # High-volume tool-calling loops (6 in parallel): cheap, low-latency MoE with reliable tool calls.
    "researcher": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    # Verification: strong reasoning, and a different model family than the writers.
    "critic": "deepseek-ai/DeepSeek-V4-Pro-0813",
}

MAX_SEARCHES = int(os.getenv("SUPPLY_RISK_MAX_SEARCHES", "6"))
LOOKBACK_MONTHS = 24
# Wikidata/SEC-style APIs reject requests without a contact URL in the User-Agent.
USER_AGENT = os.getenv("SUPPLY_RISK_USER_AGENT", "supply-risk-agent/0.1 (https://example.org/supply-risk; hackathon prototype)")


def model_id(role: str) -> str:
    return os.getenv(f"SUPPLY_RISK_MODEL_{role.upper()}", MODELS[role])


def chat_model(model: str, temperature: float = 0.2, timeout: float = 180, max_retries: int = 3) -> ChatOpenAI:
    """An open model on Nebius Token Factory via its OpenAI-compatible API."""
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        raise RuntimeError("NEBIUS_API_KEY is empty: add your Nebius Token Factory key to .env")
    return ChatOpenAI(
        model=model,
        base_url=os.getenv("NEBIUS_BASE_URL", NEBIUS_BASE_URL),
        api_key=api_key,  # type: ignore[arg-type]
        temperature=temperature,
        max_retries=max_retries,
        timeout=timeout,
    )
