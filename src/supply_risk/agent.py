"""The deep agent: an orchestrator with an entity resolver, six dimension researchers and a critic."""

import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Callable

from deepagents import SubAgent, create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import AgentMiddleware, TodoListMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from supply_risk import tools
from supply_risk.config import MAX_SEARCHES, chat_model, model_id
from supply_risk.prompts import DIMENSIONS, critic_prompt, entity_prompt, orchestrator_prompt, researcher_prompt

ROLES = ("orchestrator", "entity", "researcher", "critic")
OPEN_DATA = {
    t.name: t
    for t in (tools.gleif_search, tools.opensanctions_search, tools.worldbank_governance, tools.stock_financials, tools.ransomware_victims)
}


class Request(BaseModel):
    supplier: str
    product: str
    buyer: str | None = None
    single_source: bool | None = None
    used_in: str | None = None
    country: str | None = None

    def brief(self) -> str:
        single = None if self.single_source is None else ("yes" if self.single_source else "no")
        fields = {
            "Supplier": self.supplier,
            "Product": self.product,
            "Supplier country hint": self.country,
            "Buyer": self.buyer,
            "Single source": single,
            "Product is used in": self.used_in,
        }
        return "\n".join(f"- {k}: {v or 'not provided'}" for k, v in fields.items())


class SaveReply(AgentMiddleware):
    """Saves a subagent's final reply as its output file. Models (notably Qwen) often put the document in
    their reply instead of calling write_file, so writing the file is done here, deterministically."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self._before: str | None = None

    def _read(self) -> str | None:
        return self.path.read_text() if self.path.exists() else None

    def before_agent(self, state: Any, runtime: Any) -> None:
        self._before = self._read()

    def after_agent(self, state: Any, runtime: Any) -> None:
        if self._read() != self._before:
            return None  # the model saved the file itself (its reply is then just a summary)
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.text.strip():
            text = re.sub(r"^```\w*\n|\n?```\s*$", "", last.text.strip())  # drop a wrapping code fence
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(text + "\n")


def build_subagents(models: dict[str, Any], tavily: Any, run_dir: Path, today: date | None = None) -> list[SubAgent]:
    today = today or date.today()
    agents: list[SubAgent] = [
        {
            "name": "entity-resolver",
            "description": "Resolves the supplier to a legal entity; its reply is saved as /supplier_profile.md. Run first.",
            "system_prompt": entity_prompt(today),
            "model": models["entity"],
            "tools": [tools.gleif_search, *tools.search_tools(tavily, max_searches=6, max_extracts=2)],
            "middleware": [SaveReply(run_dir / "supplier_profile.md")],
        }
    ]
    for key, (title, _, extra_tools, _) in DIMENSIONS.items():
        agents.append(
            {
                "name": f"{key}-researcher",
                "description": f"Researches {title.lower()}; its scored, cited reply is saved as /findings/{key}.md.",
                "system_prompt": researcher_prompt(key, today),
                "model": models["researcher"],
                "tools": [*tools.search_tools(tavily, MAX_SEARCHES), *(OPEN_DATA[t] for t in extra_tools)],
                "middleware": [SaveReply(run_dir / "findings" / f"{key}.md")],
            }
        )
    for key, (title, *_) in DIMENSIONS.items():
        # One critic per dimension, each with its own extract budget (a shared tool would share the cap).
        agents.append(
            {
                "name": f"{key}-critic",
                "description": f"Independent verifier of /findings/{key}.md; its reply is saved as /reviews/{key}.md.",
                "system_prompt": critic_prompt(key, today),
                "model": models["critic"],
                # extract_page only: the critic checks cited pages, it does not search for new evidence.
                "tools": tools.search_tools(tavily, max_searches=0)[1:],
                "middleware": [SaveReply(run_dir / "reviews" / f"{key}.md")],
            }
        )
    return agents


def run(
    request: Request,
    run_dir: Path,
    models: dict[str, Any] | None = None,
    tavily: Any = None,
    on_step: Callable[[str], None] = lambda _: None,
) -> Path:
    """Run the full analysis. The agents' files (/report.md, /findings/...) are written into run_dir."""
    if models is None:
        models = {role: chat_model(model_id(role)) for role in ROLES}
    if tavily is None:
        from tavily import TavilyClient

        tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    run_dir.mkdir(parents=True, exist_ok=True)
    agent = create_deep_agent(
        model=models["orchestrator"],
        system_prompt=orchestrator_prompt(date.today()),
        middleware=[TodoListMiddleware()],
        subagents=build_subagents(models, tavily, run_dir),
        backend=FilesystemBackend(root_dir=run_dir, virtual_mode=True),
    )
    task = HumanMessage("Produce the supply chain risk report for:\n" + request.brief())
    for namespace, update in agent.stream({"messages": [task]}, {"recursion_limit": 250}, stream_mode="updates", subgraphs=True):
        who = "subagent" if namespace else "orchestrator"
        for node in update.values() if isinstance(update, dict) else []:
            for msg in node.get("messages", []) if isinstance(node, dict) else []:
                for call in getattr(msg, "tool_calls", None) or []:
                    args = call["args"]
                    detail = args.get("subagent_type") or args.get("query") or args.get("file_path") or ""
                    on_step(f"{who}: {call['name']}({detail})")
    return run_dir / "report.md"
