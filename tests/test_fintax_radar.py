"""Mocked end-to-end checks for the FinTax candidate radar."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.fintax_radar.cli import main
from src.fintax_radar.engine import (
    RadarConfig,
    RadarError,
    build_chat_request,
    google_news_rss_url,
    load_config,
    run_radar,
)
from src.processing.profiles import ProfileRegistry
from src.scrapers.google_news import GoogleNewsScraper

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "integrations" / "fintax" / "radar.json"
PROFILES = ROOT / "profiles"
MONDAY = datetime(2026, 11, 23, 1, 0, tzinfo=timezone.utc)
TUESDAY = datetime(2026, 11, 24, 1, 0, tzinfo=timezone.utc)
SECRET = "sk-boxkeyvalue1234"


def _rss(*items: tuple[str, str, str, str]) -> bytes:
    body = "".join(
        f"<item><title>{title}</title><link>{link}</link>"
        f"<pubDate>{pub}</pubDate><description>{desc}</description></item>"
        for title, link, pub, desc in items
    )
    return f'<?xml version="1.0"?><rss version="2.0"><channel>{body}</channel></rss>'.encode()


def _config(tmp_path: Path, sources: list[dict], **limits) -> RadarConfig:
    raw = json.loads(PUBLIC.read_text(encoding="utf-8"))
    raw["sources"] = sources
    raw["output_base"] = str(tmp_path)
    raw["ledger_path"] = str(tmp_path / "ledger.json")
    raw["limits"].update(limits)
    return RadarConfig.model_validate(raw)


def _run(cfg, feeds, now, tmp_path, **kwargs):
    async def fetch(url, timeout):
        assert timeout <= 30
        return feeds[url]

    kwargs.setdefault("env", {})
    return asyncio.run(
        run_radar(
            cfg,
            now=now,
            output_base=tmp_path,
            profiles_dir=PROFILES,
            ledger_path=tmp_path / "ledger.json",
            fetch=fetch,
            commit="a" * 40,
            **kwargs,
        )
    )


def test_public_config_is_verified_and_delivery_is_off():
    config = load_config(PUBLIC)

    assert config.delivery.model_dump() == {name: False for name in config.delivery.model_dump()}
    assert config.window.monday_hours == 72
    assert config.window.default_hours == 24
    assert config.limits.max_llm_calls <= 8
    urls = [source.resolved_url() for source in config.sources]
    assert all(url.startswith(GoogleNewsScraper.BASE_URL + "?") or url in {
        "https://www.gov.uk/search/news-and-communications.atom?organisations%5B%5D=hm-revenue-customs",
        "https://taxation-customs.ec.europa.eu/node/2/rss_en",
    } for url in urls)
    assert "simonwillison" not in PUBLIC.read_text(encoding="utf-8")
    built = google_news_rss_url("CARF tax")
    assert built.startswith(GoogleNewsScraper.BASE_URL + "?")
    assert "hl=en" in built and "gl=US" in built

    raw = json.loads(PUBLIC.read_text(encoding="utf-8"))
    raw["delivery"]["email"] = True
    with pytest.raises(RadarError):
        RadarConfig.model_validate(raw).check()
    secret_file = PUBLIC.with_name("secret.json")
    try:
        secret_file.write_text(PUBLIC.read_text(encoding="utf-8").replace(
            "https://api.deepseek.com", "https://api.deepseek.com/sk-abcdefghijklmnopqrst"
        ), encoding="utf-8")
        with pytest.raises(RadarError, match="secret"):
            load_config(secret_file)
    finally:
        secret_file.unlink(missing_ok=True)


def test_fintax_profiles_have_no_search_tool():
    registry = ProfileRegistry.load(PROFILES, "fintax-timely")

    for profile_id in ("fintax-timely", "fintax-deep"):
        profile = registry.get(profile_id)
        assert profile.enrichment_prompt
        tools = [tool for block in profile.definition.enrichment.blocks for tool in block.tools]
        assert tools == []


def test_monday_uses_72h_and_other_weekdays_use_24h(tmp_path):
    source = {
        "name": "Google News 1099-DA",
        "profile": "fintax-timely",
        "kind": "google_news",
        "query": "1099-DA",
    }
    cfg = _config(tmp_path, [source])
    url = cfg.sources[0].resolved_url()
    feed = _rss(
        ("Old CARF notice", "https://news.google.com/rss/articles/old", "Fri, 20 Nov 2026 12:00:00 GMT", "CARF update"),
        ("Fresh 1099-DA", "https://news.google.com/rss/articles/fresh", "Sun, 22 Nov 2026 12:00:00 GMT", "1099-DA"),
        ("Tuesday item", "https://news.google.com/rss/articles/tue", "Tue, 24 Nov 2026 00:30:00 GMT", "1099-DA"),
    )
    monday = _run(cfg, {url: feed}, MONDAY, tmp_path / "mon")
    tuesday = _run(cfg, {url: feed}, TUESDAY, tmp_path / "tue")

    assert monday.provenance["window_hours"] == 72
    assert tuesday.provenance["window_hours"] == 24
    assert {item["url"] for item in json.loads((monday.output_dir / "candidates.json").read_text())["items"]} == {
        "https://news.google.com/rss/articles/old",
        "https://news.google.com/rss/articles/fresh",
    }
    assert [item["url"] for item in json.loads((tuesday.output_dir / "candidates.json").read_text())["items"]] == [
        "https://news.google.com/rss/articles/tue",
    ]
    assert monday.llm_calls == 0
    assert not (tmp_path / "mon" / "ledger.json").exists()


def test_execute_keeps_grounded_chinese_and_dedups(tmp_path):
    source = {
        "name": "Google News 1099-DA",
        "profile": "fintax-timely",
        "kind": "google_news",
        "query": "1099-DA",
    }
    cfg = _config(tmp_path, [source])
    url = cfg.sources[0].resolved_url()
    feed = _rss(
        (
            "IRS Form 1099-DA instructions",
            "https://news.google.com/rss/articles/da",
            "Fri, 20 Nov 2026 12:00:00 GMT",
            "Form 1099-DA instructions were updated on 2026-11-20.",
        ),
        (
            "Ungrounded rumor",
            "https://news.google.com/rss/articles/rumor",
            "Fri, 20 Nov 2026 13:00:00 GMT",
            "A rumor about Form 1099-DA with no quoted mechanism.",
        ),
    )
    seen = []

    async def complete(request):
        seen.append(request)
        assert "tools" not in request
        assert request["extra_body"] == {"thinking": {"type": "disabled"}}
        assert request["response_format"] == {"type": "json_object"}
        assert request["model"] == "deepseek-flash"
        user = request["messages"][1]["content"]
        if "rumor" in user:
            payload = {
                "keep": True,
                "title_zh": "未经证实的传闻",
                "event_date": None,
                "client_question": "需要核实吗？",
                "impact": "没有机制。",
                "relevance": "不够。",
                "reason_zh": "没有原文摘录。",
                "evidence": ["this quote was not in the feed"],
            }
        else:
            payload = {
                "keep": True,
                "title_zh": "国税局更新1099-DA说明",
                "event_date": "2026-11-20",
                "client_question": "申报机构要按新说明采集哪些数字资产字段？",
                "impact": "托管人需要核对自己的1099-DA字段。",
                "relevance": "这关系美国数字资产信息申报。",
                "reason_zh": "来源写明了表格说明的更新日期。",
                "evidence": ["Form 1099-DA instructions"],
                "url": "https://evil.example/invented",
            }
        return json.dumps(payload)

    first = _run(cfg, {url: feed}, MONDAY, tmp_path, execute=True, complete=complete)
    assert first.llm_calls == 2
    saved = json.loads((first.output_dir / "candidates.json").read_text(encoding="utf-8"))["items"]
    assert len(saved) == 1
    assert saved[0]["url"] == "https://news.google.com/rss/articles/da"
    assert saved[0]["event_date"] == "2026-11-20"
    assert "申报" in saved[0]["client_question"]
    markdown = (first.output_dir / "candidates.md").read_text(encoding="utf-8")
    assert "国税局更新1099-DA说明" in markdown
    assert "evil.example" not in markdown
    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert len(ledger["items"]) == 2

    second = _run(cfg, {url: feed}, MONDAY, tmp_path, execute=True, complete=complete, env={})
    assert second.llm_calls == 0
    assert second.candidate_count == 0
    assert len(seen) == 2


def test_official_keyword_gate_and_call_cap(tmp_path):
    sources = [
        {
            "name": "HMRC news",
            "profile": "fintax-timely",
            "kind": "official_rss",
            "url": "https://www.gov.uk/search/news-and-communications.atom?organisations%5B%5D=hm-revenue-customs",
        },
        {
            "name": "Google News DAC8",
            "profile": "fintax-timely",
            "kind": "google_news",
            "query": "DAC8",
        },
    ]
    cfg = _config(tmp_path, sources, max_llm_calls=1, max_items=8)
    hmrc = sources[0]["url"]
    dac8 = cfg.sources[1].resolved_url()
    feeds = {
        hmrc: _rss(
            ("Self Assessment scams", "https://www.gov.uk/government/news/scams", "Sun, 22 Nov 2026 12:00:00 GMT", "Report a scam"),
            ("Crypto CARF update", "https://www.gov.uk/government/news/carf", "Sun, 22 Nov 2026 13:00:00 GMT", "CARF crypto-asset reporting"),
        ),
        dac8: _rss(
            ("DAC8 one", "https://news.google.com/rss/articles/one", "Sun, 22 Nov 2026 12:00:00 GMT", "DAC8"),
            ("DAC8 two", "https://news.google.com/rss/articles/two", "Sun, 22 Nov 2026 11:00:00 GMT", "DAC8"),
        ),
    }

    async def complete(request):
        return json.dumps({
            "keep": False,
            "title_zh": "",
            "event_date": None,
            "client_question": "",
            "impact": "",
            "relevance": "",
            "reason_zh": "不是新的合规义务。",
            "evidence": [],
        })

    dry = _run(cfg, feeds, MONDAY, tmp_path / "dry")
    urls = [item["url"] for item in json.loads((dry.output_dir / "candidates.json").read_text())["items"]]
    assert "https://www.gov.uk/government/news/scams" not in urls
    assert "https://www.gov.uk/government/news/carf" in urls
    assert dry.provenance["counts"]["keyword_rejected"] == 1

    live = _run(cfg, feeds, MONDAY, tmp_path / "live", execute=True, complete=complete)
    assert live.llm_calls == 1
    assert live.provenance["counts"]["capped"] >= 1


def test_secrets_are_not_written(tmp_path):
    source = {
        "name": "Google News 1099-DA",
        "profile": "fintax-timely",
        "kind": "google_news",
        "query": "1099-DA",
    }
    cfg = _config(tmp_path, [source])
    url = cfg.sources[0].resolved_url()
    feed = _rss(
        ("Raise", "https://news.google.com/rss/articles/raise", "Sun, 22 Nov 2026 12:00:00 GMT", "1099-DA"),
        ("Echo", "https://news.google.com/rss/articles/echo", "Sun, 22 Nov 2026 11:00:00 GMT", "1099-DA"),
    )

    async def complete(request):
        user = request["messages"][1]["content"]
        if "articles/raise" in user:
            raise RuntimeError(f"Authorization: Bearer {SECRET}")
        return json.dumps({"keep": True, "title_zh": SECRET, "evidence": ["1099-DA"]})

    result = _run(
        cfg, {url: feed}, MONDAY, tmp_path, execute=True, complete=complete, env={"DEEPSEEK_API_KEY": SECRET},
    )
    blob = "".join(path.read_text(encoding="utf-8") for path in result.output_dir.iterdir() if path.is_file())
    ledger = (tmp_path / "ledger.json").read_text(encoding="utf-8") if (tmp_path / "ledger.json").exists() else ""
    assert SECRET not in blob + ledger
    assert result.provenance["counts"]["failed"] == 2


def test_chat_request_omits_web_search():
    request = build_chat_request(
        model="deepseek-flash",
        system="no tools",
        user="item",
        max_tokens=900,
        temperature=0,
    )

    assert "tools" not in request
    assert "web_search" not in request
    assert request["extra_body"]["thinking"]["type"] == "disabled"


def test_cli_execute_requires_the_env_key(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    code = main([
        "--config", str(PUBLIC),
        "--execute",
        "--output-base", str(tmp_path),
        "--ledger", str(tmp_path / "ledger.json"),
        "--now", "2026-11-25T01:00:00+00:00",
        "--profiles-dir", str(PROFILES),
    ])

    assert code == 2
    assert list(tmp_path.iterdir()) == []
