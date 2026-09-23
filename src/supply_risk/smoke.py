"""Tool-calling smoke test for candidate Nebius models: the evidence behind the model choice per role.

Per model: does it make a valid tool call, does it avoid leaking tool-call JSON into the text (a known
gpt-oss/vLLM failure), can it make parallel calls, and does it answer after getting the tool result?
"""

import time

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from supply_risk.config import chat_model

CANDIDATES = [
    "zai-org/GLM-5.3",
    "deepseek-ai/DeepSeek-V4-Pro-0813",
    "moonshotai/Kimi-K3",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "zai-org/GLM-5.3-Flash",
    "openai/gpt-oss-120b",
]


@tool
def lookup_company(name: str, country: str) -> str:
    """Look up a company's legal entity identifier by name and ISO2 country code."""
    return "LEI TEST0000000000000000"


def smoke_test(model: str) -> dict:
    result: dict = {"model": model}
    try:
        llm = chat_model(model, temperature=0).bind_tools([lookup_company])
        prompt = HumanMessage("Find the LEI of ASML (Netherlands). Use the tool.")
        start = time.monotonic()
        ai = llm.invoke([prompt])
        result["latency_s"] = round(time.monotonic() - start, 1)
        calls = ai.tool_calls  # type: ignore[attr-defined]
        result["tool_call"] = bool(calls) and {"name", "country"} <= set(calls[0]["args"])
        result["no_json_leak"] = "lookup_company" not in str(ai.content)
        if calls:
            tool_results = [ToolMessage(lookup_company.invoke(c["args"]), tool_call_id=c["id"]) for c in calls]
            result["final_answer"] = "TEST0000000000000000" in str(llm.invoke([prompt, ai, *tool_results]).content)
        parallel = llm.invoke([HumanMessage("Look up the LEIs of ASML (NL) and Infineon (DE): call the tool for both in one turn.")])
        result["parallel_calls"] = len(parallel.tool_calls)  # type: ignore[attr-defined]
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"[:200]
    return result


def run_smoke(models: list[str]) -> str:
    rows = ["| Model | Tool call | No JSON leak | Parallel calls | Final answer | Latency (s) | Error |", "|---|---|---|---|---|---|---|"]
    for model in models:
        r = smoke_test(model)
        ok = lambda k: "✅" if r.get(k) else "❌"  # noqa: E731
        rows.append(
            f"| {model} | {ok('tool_call')} | {ok('no_json_leak')} | {r.get('parallel_calls', 0)} | {ok('final_answer')} | "
            f"{r.get('latency_s', '-')} | {r.get('error', '')} |"
        )
    return "\n".join(rows)
