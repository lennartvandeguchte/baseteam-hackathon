import respx

from supply_risk import tools


class FakeTavily:
    def __init__(self):
        self.searches = 0

    def search(self, query, **kwargs):
        self.searches += 1
        return {"results": [{"title": "Acme hit by ransomware", "url": "https://news.example/acme", "content": "Acme halted production.", "published_date": "2026-03-02"}]}

    def extract(self, urls, **kwargs):
        return {"results": [{"url": u, "raw_content": "Full article text."} for u in urls]}


def _search_tools(max_searches=2):
    client = FakeTavily()
    return {t.name: t for t in tools.search_tools(client, max_searches)}, client


def test_web_search_returns_url_date_and_snippet():
    t, _ = _search_tools()
    out = t["web_search"].invoke({"query": "Acme ransomware"})
    assert "https://news.example/acme" in out and "2026-03-02" in out and "halted production" in out


def test_web_search_is_capped_per_agent():
    t, client = _search_tools(max_searches=1)
    t["web_search"].invoke({"query": "a"})
    assert "used all 1 searches" in t["web_search"].invoke({"query": "b"})
    assert client.searches == 1


def test_extract_page():
    t, _ = _search_tools()
    assert "Full article text." in t["extract_page"].invoke({"urls": ["https://news.example/acme"]})


def test_search_errors_are_returned_to_the_agent():
    class Broken(FakeTavily):
        def search(self, query, **kwargs):
            raise RuntimeError("boom")

    t = {x.name: x for x in tools.search_tools(Broken(), 2)}
    assert "failed" in t["web_search"].invoke({"query": "q"})


@respx.mock
def test_gleif_search():
    respx.get("https://api.gleif.org/api/v1/lei-records").respond(
        json={"data": [{"attributes": {"lei": "724500Y6DUVHQD6OXN27", "entity": {"legalName": {"name": "ASML Holding N.V."}, "jurisdiction": "NL", "headquartersAddress": {"city": "Veldhoven", "country": "NL"}}}}]}
    )
    out = tools.gleif_search.invoke({"name": "ASML"})
    assert "ASML Holding N.V." in out and "724500Y6DUVHQD6OXN27" in out and "Veldhoven" in out


@respx.mock
def test_worldbank_governance():
    respx.get(url__regex=r"https://api\.worldbank\.org/v2/country/NL/indicator/.*").respond(
        json=[{"page": 1}, [{"indicator": {"value": "Political Stability"}, "date": "2024", "value": 0.43}]]
    )
    out = tools.worldbank_governance.invoke({"country_code": "NL"})
    assert "Political Stability" in out and "0.43" in out


@respx.mock
def test_ransomware_victims_found_and_not_found():
    respx.get("https://api.ransomware.live/v2/searchvictims/acme").respond(
        json=[{"victim": "Acme Corp", "group": "lockbit3", "attackdate": "2026-03-01T00:00:00", "country": "US"}]
    )
    out = tools.ransomware_victims.invoke({"keyword": "acme"})
    assert "Acme Corp" in out and "lockbit3" in out and "2026-03-01" in out

    respx.get("https://api.ransomware.live/v2/searchvictims/none").respond(404, json={"error": "No victims"})
    assert "No ransomware victim" in tools.ransomware_victims.invoke({"keyword": "none"})


def test_opensanctions_without_key_explains_fallback(monkeypatch):
    monkeypatch.delenv("OPENSANCTIONS_API_KEY", raising=False)
    assert "OPENSANCTIONS_API_KEY" in tools.opensanctions_search.invoke({"name": "Rosneft"})


@respx.mock
def test_api_errors_are_returned_to_the_agent():
    respx.get("https://api.gleif.org/api/v1/lei-records").respond(500)
    assert "failed" in tools.gleif_search.invoke({"name": "ASML"})
