from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from supply_risk.agent import Request, build_subagents, run
from supply_risk.prompts import DIMENSIONS
from tests.test_tools import FakeTavily


class ScriptedModel(GenericFakeChatModel):
    """Fake chat model that accepts bind_tools and replays scripted messages in order."""

    label: str = ""

    def bind_tools(self, tools: Any, **kwargs: Any):  # type: ignore[override]
        return self


def _models(**scripts: list[AIMessage]) -> dict:
    return {role: ScriptedModel(messages=iter(scripts.get(role, [])), label=role) for role in ("orchestrator", "entity", "researcher", "critic")}


def _call(name: str, args: dict, id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id, "type": "tool_call"}])


def test_subagents_roles_models_and_tools():
    specs = {s["name"]: s for s in build_subagents(_models(), FakeTavily())}
    assert list(specs) == ["entity-resolver", *(f"{k}-researcher" for k in DIMENSIONS), *(f"{k}-critic" for k in DIMENSIONS)]
    assert specs["entity-resolver"]["model"].label == "entity"  # type: ignore[union-attr]
    assert specs["cyber-critic"]["model"].label == "critic"  # type: ignore[union-attr]
    names = {n: {t.name for t in s["tools"]} for n, s in specs.items()}  # type: ignore[union-attr]
    assert "ransomware_victims" in names["cyber-researcher"]
    assert {"opensanctions_search", "worldbank_governance"} <= names["geopolitical-researcher"]
    assert "stock_financials" in names["financial-researcher"]
    assert "gleif_search" in names["entity-resolver"]
    assert names["cyber-critic"] == {"extract_page"}


def test_each_critic_has_its_own_extract_budget():
    specs = {s["name"]: s for s in build_subagents(_models(), FakeTavily())}
    def extract(name: str) -> BaseTool:
        return next(t for t in specs[name]["tools"] if isinstance(t, BaseTool) and t.name == "extract_page")  # type: ignore[union-attr]

    for _ in range(3):
        extract("cyber-critic").invoke({"urls": ["https://a.example"]})
    assert "Full article text." in extract("geopolitical-critic").invoke({"urls": ["https://a.example"]})


def test_request_brief():
    brief = Request(supplier="TSMC", product="5nm wafers", buyer="Acme", single_source=True).brief()
    assert "TSMC" in brief and "5nm wafers" in brief and "Acme" in brief and "Single source: yes" in brief


def test_end_to_end_run_writes_files(tmp_path):
    models = _models(
        orchestrator=[
            _call("task", {"description": "Resolve the supplier", "subagent_type": "entity-resolver"}, "c1"),
            _call("write_file", {"file_path": "/report.md", "content": "# Report\n"}, "c2"),
            AIMessage(content="Done: /report.md"),
        ],
        entity=[
            _call("write_file", {"file_path": "/supplier_profile.md", "content": "# Profile"}, "e1"),
            AIMessage(content="Profile written."),
        ],
    )
    report = run(Request(supplier="Acme", product="widgets"), tmp_path, models=models, tavily=FakeTavily())
    assert report.read_text() == "# Report\n"
    assert (tmp_path / "supplier_profile.md").exists()
