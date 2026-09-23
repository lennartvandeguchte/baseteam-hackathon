"""Agent tools: Tavily web search plus a few structured open-data sources.

Tools never raise: errors are returned as text so the agent can fall back to web search.
"""

import functools
import os
from typing import Any, Callable, Literal

import httpx
from langchain_core.tools import BaseTool, tool

from supply_risk.config import USER_AGENT


def _safe(fn: Callable[..., str]) -> Callable[..., str]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return f"{fn.__name__} failed ({type(e).__name__}: {e}). Fall back to web_search."

    return wrapper


def _get(url: str, params: dict | None = None, headers: dict | None = None) -> httpx.Response:
    return httpx.get(url, params=params, headers={"User-Agent": USER_AGENT, **(headers or {})}, timeout=30)


# -- web search -------------------------------------------------------------------------------


def search_tools(client: Any, max_searches: int, max_extracts: int = 3) -> list[BaseTool]:
    """web_search + extract_page for one agent, each with its own cap so no agent can loop for minutes."""
    used = 0
    extracts = 0

    @tool
    @_safe
    def web_search(
        query: str,
        topic: Literal["general", "news", "finance"] = "general",
        depth: Literal["basic", "advanced"] = "basic",
        time_range: Literal["day", "week", "month", "year"] | None = None,
        include_domains: list[str] | None = None,
    ) -> str:
        """Search the web. Returns title, URL, publication date and a snippet per result.
        Use topic="news" for recent events and include_domains to target specific sources.
        depth="advanced" costs double: use it only for your most important queries.
        You have a limited number of searches, so plan your queries."""
        nonlocal used
        if used >= max_searches:
            return f"You have used all {max_searches} searches. Work with the evidence you have."
        kwargs = {"topic": topic, "search_depth": depth, "time_range": time_range, "include_domains": include_domains}
        data = client.search(query, max_results=5, **{k: v for k, v in kwargs.items() if v is not None})
        used += 1  # only successful searches count against the cap
        results = [
            f"[{i}] {r.get('title')}\nURL: {r.get('url')}\nPublished: {r.get('published_date') or 'unknown'}\n{r.get('content', '')}"
            for i, r in enumerate(data.get("results", []), 1)
        ]
        return f"(search {used}/{max_searches})\n\n" + ("\n\n".join(results) or "No results.")

    @tool
    @_safe
    def extract_page(urls: list[str]) -> str:
        """Fetch the text of up to 5 web pages in one call, to confirm facts and copy exact quotes.
        You can only call this a few times: pass all the URLs you need at once, and do not retry failed pages."""
        nonlocal extracts
        if extracts >= max_extracts:
            return f"You have used all {max_extracts} extract calls. Finish with the evidence you have."
        extracts += 1
        data = client.extract(urls[:5], format="markdown")
        pages = [f"URL: {r.get('url')}\n{(r.get('raw_content') or '')[:4000]}" for r in data.get("results", [])]
        failed = [str(f.get("url", f)) for f in data.get("failed_results", [])]
        if failed:
            pages.append("Could not fetch (do not retry): " + ", ".join(failed))
        return "\n\n---\n\n".join(pages) or "Nothing could be extracted."

    return [web_search, extract_page]


# -- open data --------------------------------------------------------------------------------


@tool
@_safe
def gleif_search(name: str) -> str:
    """Search the GLEIF Legal Entity Identifier database by company name: legal name, LEI,
    jurisdiction and headquarters for up to 5 matches."""
    data = _get("https://api.gleif.org/api/v1/lei-records", {"filter[fulltext]": name, "page[size]": 5})
    data.raise_for_status()
    lines = []
    for r in data.json().get("data", []):
        a, e = r["attributes"], r["attributes"]["entity"]
        hq = e.get("headquartersAddress") or {}
        lines.append(f"- {e['legalName']['name']} | LEI {a['lei']} | jurisdiction {e.get('jurisdiction')} | HQ {hq.get('city')}, {hq.get('country')}")
    return "\n".join(lines) or f"No LEI records found for '{name}'."


@tool
@_safe
def opensanctions_search(name: str) -> str:
    """Search OpenSanctions (consolidated sanctions, PEP and watchlists) for an entity name."""
    key = os.getenv("OPENSANCTIONS_API_KEY")
    if not key:
        return "OpenSanctions is not configured (OPENSANCTIONS_API_KEY). Use web_search on official sanctions lists instead."
    resp = _get("https://api.opensanctions.org/search/default", {"q": name, "limit": 10}, {"Authorization": f"ApiKey {key}"})
    resp.raise_for_status()
    lines = [
        f"- {r.get('caption')} [{r.get('schema')}] countries {r.get('properties', {}).get('country', [])} | lists {r.get('datasets', [])}"
        for r in resp.json().get("results", [])
    ]
    return ("Name matches may be different entities — check country.\n" + "\n".join(lines)) if lines else f"No OpenSanctions matches for '{name}'."


WGI = {
    "GOV_WGI_PV.EST": "Political Stability",
    "GOV_WGI_RL.EST": "Rule of Law",
    "GOV_WGI_CC.EST": "Control of Corruption",
    "GOV_WGI_RQ.EST": "Regulatory Quality",
}


@tool
@_safe
def worldbank_governance(country_code: str) -> str:
    """World Bank governance indicators for a country (ISO2/ISO3 code), from about -2.5 (weak) to +2.5 (strong)."""
    lines = []
    for code, label in WGI.items():
        data = _get(f"https://api.worldbank.org/v2/country/{country_code}/indicator/{code}", {"format": "json", "mrnev": 1, "source": 3}).json()
        row = data[1][0] if isinstance(data, list) and len(data) > 1 and data[1] else None
        value = row and row.get("value")
        lines.append(f"- {row['indicator']['value']}: {value:.2f} ({row['date']})" if row and value is not None else f"- {label}: no data")
    return f"Governance indicators for {country_code}:\n" + "\n".join(lines)


@tool
@_safe
def stock_financials(ticker: str) -> str:
    """Key financial health metrics for a listed company from Yahoo Finance (Yahoo ticker, e.g. 'ASML', '2330.TW')."""
    import yfinance as yf

    info = yf.Ticker(ticker).info
    keys = [
        "longName", "country", "currency", "marketCap", "totalRevenue", "revenueGrowth", "operatingMargins",
        "profitMargins", "freeCashflow", "totalCash", "totalDebt", "debtToEquity", "currentRatio", "overallRisk",
    ]
    if not info.get("longName"):
        return f"No Yahoo Finance data for ticker {ticker}."
    return "\n".join(f"- {k}: {info[k]}" for k in keys if info.get(k) is not None)


@tool
@_safe
def ransomware_victims(keyword: str) -> str:
    """Search ransomware.live leak-site victim listings for a short, distinctive company name."""
    resp = _get(f"https://api.ransomware.live/v2/searchvictims/{keyword}")
    if resp.status_code == 404:
        return f"No ransomware victim listings found for '{keyword}'."
    resp.raise_for_status()
    lines = [
        f"- {v.get('victim')} | group {v.get('group')} | attack date {(v.get('attackdate') or '')[:10]} | country {v.get('country')}"
        for v in resp.json()[:15]
    ]
    return "Listings may be different companies with similar names:\n" + "\n".join(lines)
