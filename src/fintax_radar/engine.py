"""Bounded FinTax candidate radar.

One Chat Completions call per selected item, no web-search tool, no retries.
Stock `horizon` is not the entrypoint: it still copies Markdown into docs/_posts.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable, Mapping, Optional
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import feedparser
import httpx
from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..ai.utils import parse_json_response
from ..scrapers.google_news import GoogleNewsScraper
from ..scrapers.rss import RSS_USER_AGENT, RSSScraper

# ponytail: ceilings keep a weekday run to a handful of feeds and one call each.
# Upgrade path: raise them only after a box smoke test shows the feeds stay sparse.
MAX_ITEMS = 12
MAX_CALLS = 12
MAX_PER_FEED = 5
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SECRET_RE = re.compile(
    r"(?i)(api[_-]?key\s*[:=]\s*\S+|bearer\s+\S+|sk-[a-z0-9_\-]{8,})"
)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

VERIFIED_OFFICIAL_RSS = {
    "https://www.gov.uk/search/news-and-communications.atom?organisations%5B%5D=hm-revenue-customs",
    "https://taxation-customs.ec.europa.eu/node/2/rss_en",
}
OFFICIAL_KEYWORDS = (
    "carf",
    "crypto-asset reporting framework",
    "dac8",
    "1099-da",
    "1099 da",
    "crs 2.0",
    "common reporting standard",
    "crypto",
    "digital asset",
    "virtual asset",
    "stablecoin",
    "ifrs",
    "iasb",
    "fasb",
)

Fetch = Callable[[str, float], Awaitable[bytes]]
Complete = Callable[[dict], Awaitable[str]]


class RadarError(ValueError):
    """Configuration or ledger error safe to print after redaction."""


class SourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    profile: str = Field(pattern=r"^fintax-(timely|deep)$")
    kind: str = Field(pattern=r"^(google_news|official_rss)$")
    query: Optional[str] = None
    url: Optional[str] = None

    def resolved_url(self) -> str:
        if self.kind == "google_news":
            if not self.query or not self.query.strip() or self.url:
                raise RadarError(f"{self.name}: google_news needs a query and no url")
            return google_news_rss_url(self.query.strip())
        if self.query or self.url not in VERIFIED_OFFICIAL_RSS:
            raise RadarError(f"{self.name}: official_rss url is not in the verified set")
        return self.url


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_items_per_feed: int = Field(ge=1, le=MAX_PER_FEED)
    max_items: int = Field(ge=1, le=MAX_ITEMS)
    max_llm_calls: int = Field(ge=1, le=MAX_CALLS)
    feed_timeout_sec: float = Field(gt=0, le=30)
    llm_timeout_sec: float = Field(gt=0, le=60)
    max_tokens: int = Field(ge=200, le=1200)
    content_chars: int = Field(ge=500, le=6000)


class AISpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(pattern=r"^deepseek$")
    base_url: str = Field(pattern=r"^https://api\.deepseek\.com$")
    model: str = Field(pattern=r"^deepseek-flash$")
    api_key_env: str = Field(pattern=r"^DEEPSEEK_API_KEY$")
    temperature: float = Field(ge=0, le=0)
    thinking: str = Field(pattern=r"^disabled$")


class DeliverySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: bool
    webhook: bool
    wechat: bool
    github_pages: bool
    x_twitter: bool
    hackernews: bool
    reddit: bool
    telegram: bool
    openbb: bool

    def assert_off(self) -> None:
        enabled = [name for name, value in self.model_dump().items() if value]
        if enabled:
            raise RadarError("delivery must stay off: " + ", ".join(enabled))


class WindowSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timezone: str
    monday_hours: int
    default_hours: int

    def hours_for(self, now: datetime, override: Optional[int]) -> int:
        if override is not None:
            if isinstance(override, bool) or not isinstance(override, int):
                raise RadarError("hours override must be an integer")
            if not 1 <= override <= 168:
                raise RadarError("hours override must be from 1 to 168")
            return override
        if self.monday_hours != 72 or self.default_hours != 24:
            raise RadarError("window is fixed at Monday 72h and other days 24h")
        try:
            local = now.astimezone(ZoneInfo(self.timezone))
        except Exception as exc:
            raise RadarError(f"unknown timezone: {self.timezone}") from exc
        return 72 if local.weekday() == 0 else 24


class RadarConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timezone: str
    window: WindowSpec
    output_base: str
    ledger_path: Optional[str] = None
    profiles_dir: Optional[str] = None
    limits: Limits
    ai: AISpec
    delivery: DeliverySpec
    sources: list[SourceSpec] = Field(min_length=1, max_length=12)

    def check(self) -> None:
        self.delivery.assert_off()
        if self.window.timezone != self.timezone:
            raise RadarError("timezone and window.timezone must match")
        ZoneInfo(self.timezone)
        names = [source.name for source in self.sources]
        if len(names) != len(set(names)):
            raise RadarError("source names must be unique")
        for source in self.sources:
            source.resolved_url()
        _safe_path(self.output_base, "output_base")
        if self.ledger_path:
            _safe_path(self.ledger_path, "ledger_path")


@dataclass
class FeedItem:
    source: str
    profile: str
    kind: str
    title: str
    url: str
    published_at: datetime
    text: str
    key: str


@dataclass
class RunResult:
    output_dir: Path
    provenance: dict
    candidate_count: int
    llm_calls: int = 0
    errors: list[str] = field(default_factory=list)


def google_news_rss_url(query: str) -> str:
    """Same endpoint the built-in Google News scraper calls."""
    params = urlencode(
        {"q": query, "hl": "en", "gl": "US", "ceid": "US:en"}
    )
    return f"{GoogleNewsScraper.BASE_URL}?{params}"


def build_chat_request(
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> dict:
    """One DeepSeek Chat Completions call.

    On deepseek-flash the Responses API ``web_search`` tool is ignored, and
    ``deepseek_web_search`` lets the model browse with no call cap. This body
    sends neither. Thinking is disabled because the API otherwise turns it on.
    """
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "extra_body": {"thinking": {"type": "disabled"}},
    }


def redact(text: str, secret: str = "") -> str:
    if not text:
        return ""
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return SECRET_RE.sub("[REDACTED]", text)


def load_config(path: Path) -> RadarConfig:
    raw = path.read_text(encoding="utf-8")
    if SECRET_RE.search(raw):
        raise RadarError("config must not contain a secret")
    if "${" in raw:
        raise RadarError("config must not expand environment variables")
    try:
        config = RadarConfig.model_validate_json(raw)
    except ValidationError as exc:
        raise RadarError(f"{exc.error_count()} config field(s) failed validation") from exc
    config.check()
    return config


def resolve_profiles_dir(config: RadarConfig, explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        root = explicit
    elif config.profiles_dir:
        root = Path(config.profiles_dir)
    else:
        packaged = Path(__file__).resolve().parents[1] / "_builtin_profiles"
        checkout = Path(__file__).resolve().parents[2] / "profiles"
        root = checkout if checkout.is_dir() else packaged
    root = root.resolve()
    for profile_id in ("fintax-timely", "fintax-deep"):
        if not (root / profile_id / "profile.json").is_file():
            raise RadarError(f"missing profile {profile_id} under {root}")
    return root


def resolve_commit(config_dir: Path) -> str:
    env_sha = os.environ.get("FINTAX_HORIZON_COMMIT", "").strip()
    if COMMIT_RE.fullmatch(env_sha):
        return env_sha
    provenance = config_dir / "PROVENANCE.json"
    if provenance.is_file():
        try:
            sha = json.loads(provenance.read_text(encoding="utf-8")).get("commit", "")
        except (OSError, json.JSONDecodeError):
            sha = ""
        if isinstance(sha, str) and COMMIT_RE.fullmatch(sha):
            return sha
    for cwd in (config_dir, Path(__file__).resolve().parents[2]):
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            continue
        if COMMIT_RE.fullmatch(sha):
            return sha
    return "unknown"


def window_hours(config: RadarConfig, now: datetime, override: Optional[int]) -> int:
    if now.tzinfo is None:
        raise RadarError("now must be timezone-aware")
    return config.window.hours_for(now, override)


async def run_radar(
    config: RadarConfig,
    *,
    now: datetime,
    output_base: Path,
    profiles_dir: Path,
    execute: bool = False,
    hours: Optional[int] = None,
    ledger_path: Optional[Path] = None,
    fetch: Optional[Fetch] = None,
    complete: Optional[Complete] = None,
    commit: str = "unknown",
    env: Optional[Mapping[str, str]] = None,
) -> RunResult:
    config.check()
    if now.tzinfo is None:
        raise RadarError("now must be timezone-aware")
    hours_used = window_hours(config, now, hours)
    since = now.astimezone(timezone.utc) - timedelta(hours=hours_used)
    run_date = now.astimezone(ZoneInfo(config.timezone)).date().isoformat()
    output_dir = _safe_path(output_base, "output_base") / run_date
    ledger_file = _safe_path(
        ledger_path or config.ledger_path or (output_base / "ledger.json"),
        "ledger",
    )
    prompts = _load_prompts(profiles_dir)
    environment = env if env is not None else os.environ
    secret = environment.get(config.ai.api_key_env, "")
    if execute and complete is None:
        if not secret:
            raise RadarError(f"{config.ai.api_key_env} is not set; dry-run does not need it")
        complete = _live_complete(config, secret)

    counts = {
        "fetched": 0,
        "outside_window": 0,
        "undated": 0,
        "keyword_rejected": 0,
        "duplicate": 0,
        "seen_skipped": 0,
        "capped": 0,
        "llm_calls": 0,
        "kept": 0,
        "rejected": 0,
        "ungrounded": 0,
        "failed": 0,
        "feed_errors": 0,
    }
    feed_errors: list[str] = []
    parsed: list[FeedItem] = []
    for source in config.sources:
        url = source.resolved_url()
        try:
            payload = await (fetch or _http_fetch)(url, config.limits.feed_timeout_sec)
            got, stats = _parse_feed(source, payload, since, now, config.limits)
        except Exception as exc:
            counts["feed_errors"] += 1
            feed_errors.append(f"{source.name}: {redact(f'{type(exc).__name__}: {exc}', secret)}")
            continue
        counts["fetched"] += stats["fetched"]
        counts["outside_window"] += stats["outside_window"]
        counts["undated"] += stats["undated"]
        counts["keyword_rejected"] += stats["keyword_rejected"]
        parsed.extend(got)

    parsed.sort(key=lambda item: (_lane_rank(item.profile), -item.published_at.timestamp(), item.url))
    unique: list[FeedItem] = []
    seen_keys: set[str] = set()
    for item in parsed:
        if item.key in seen_keys:
            counts["duplicate"] += 1
            continue
        seen_keys.add(item.key)
        unique.append(item)

    ledger_file.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(ledger_file) + ".lock")
    failures: list[dict] = []
    with lock:
        ledger = _read_ledger(ledger_file)
        fresh = []
        for item in unique:
            if item.key in ledger["items"]:
                counts["seen_skipped"] += 1
                continue
            fresh.append(item)
        selected = fresh[: config.limits.max_items]
        counts["capped"] = max(0, len(fresh) - len(selected))
        candidates: list[dict] = []
        if not execute:
            for item in selected:
                candidates.append(_dry_candidate(item))
            counts["kept"] = len(candidates)
        else:
            assert complete is not None
            for item in selected:
                if counts["llm_calls"] >= config.limits.max_llm_calls:
                    counts["capped"] += 1
                    continue
                counts["llm_calls"] += 1
                row = await _assess(item, config, prompts, complete, secret)
                status = row["status"]
                if status not in counts:
                    status = "failed"
                counts[status] += 1
                if status == "failed":
                    failures.append({"url": item.url, "error": row.get("error", "")})
                if status in {"kept", "rejected", "ungrounded"}:
                    _remember(ledger, item, run_date, status)
                if status == "kept":
                    candidates.append(row["candidate"])
            _write_ledger(ledger_file, ledger)

    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = _provenance(
        config=config,
        now=now,
        run_date=run_date,
        hours_used=hours_used,
        since=since,
        execute=execute,
        commit=commit,
        counts=counts,
        feed_errors=feed_errors,
        failures=failures,
        rows=candidates,
        secret=secret,
    )
    _write_json(output_dir / "candidates.json", {"version": 1, "items": candidates})
    (output_dir / "candidates.md").write_text(
        _markdown(run_date, hours_used, execute, candidates),
        encoding="utf-8",
    )
    _write_json(output_dir / "provenance.json", provenance)
    _assert_clean(output_dir, secret)
    return RunResult(
        output_dir=output_dir,
        provenance=provenance,
        candidate_count=len(candidates),
        llm_calls=counts["llm_calls"],
        errors=feed_errors,
    )


def _load_prompts(profiles_dir: Path) -> dict[str, str]:
    prompts = {}
    for profile_id in ("fintax-timely", "fintax-deep"):
        root = profiles_dir / profile_id
        parts = []
        for name in ("analysis.md", "enrichment.md"):
            parts.append((root / name).read_text(encoding="utf-8").strip())
        profile = json.loads((root / "profile.json").read_text(encoding="utf-8"))
        tools = [tool for block in profile["enrichment"]["blocks"] for tool in block["tools"]]
        if tools:
            raise RadarError(f"{profile_id} must not enable enrichment tools")
        prompts[profile_id] = "\n\n".join(parts)
    return prompts


def _parse_feed(source: SourceSpec, payload: bytes, since: datetime, now: datetime, limits: Limits):
    feed = feedparser.parse(payload)
    parser = RSSScraper([], None)
    stats = {"fetched": 0, "outside_window": 0, "undated": 0, "keyword_rejected": 0}
    found: list[FeedItem] = []
    for entry in feed.entries:
        stats["fetched"] += 1
        published = parser._parse_date(entry)
        if published is None:
            stats["undated"] += 1
            continue
        if published < since or published > now.astimezone(timezone.utc):
            stats["outside_window"] += 1
            continue
        title = plain_text(str(entry.get("title") or ""))
        link = str(entry.get("link") or "").strip()
        if not title or not link.startswith("https://"):
            stats["undated"] += 1
            continue
        text = plain_text(parser._extract_content(entry))
        blob = f"{title}\n{text}"
        if source.kind == "official_rss" and not any(word in blob.casefold() for word in OFFICIAL_KEYWORDS):
            stats["keyword_rejected"] += 1
            continue
        found.append(
            FeedItem(
                source=source.name,
                profile=source.profile,
                kind=source.kind,
                title=title[:300],
                url=link,
                published_at=published.astimezone(timezone.utc),
                text=blob[: limits.content_chars],
                key=item_key(link),
            )
        )
    found.sort(key=lambda item: (-item.published_at.timestamp(), item.url))
    return found[: limits.max_items_per_feed], stats


async def _assess(item: FeedItem, config: RadarConfig, prompts: dict[str, str], complete: Complete, secret: str) -> dict:
    system = (
        prompts[item.profile]
        + "\n\nReturn one JSON object and nothing else. Keys: "
        "keep (boolean), title_zh, event_date (YYYY-MM-DD or null), "
        "client_question, impact, relevance, reason_zh, evidence (array of verbatim excerpts). "
        "Write title_zh, client_question, impact, relevance, and reason_zh in Simplified Chinese. "
        "No tools are available. Do not call web_search, deepseek_web_search, or deepseek_read_url. "
        "The runner already stored the URL. Do not invent another one. "
        "Public context only: FinTax publishes institutional crypto accounting, "
        "tax-reporting preparation (CARF, DAC8, 1099-DA, TIN and self-certification, reporting XML), "
        "on-chain audit evidence, FinHub, and consulting for HK, SG, and cross-border institutions. "
        "Do not assume unstated product capabilities or private client facts."
    )
    user = (
        f"Profile: {item.profile}\nSource: {item.source}\n"
        f"Published at: {item.published_at.isoformat()}\n"
        f"Fetched URL: {item.url}\nText:\n{item.text}"
    )
    request = build_chat_request(
        model=config.ai.model,
        system=system,
        user=user,
        max_tokens=config.limits.max_tokens,
        temperature=config.ai.temperature,
    )
    try:
        raw = await complete(request)
    except Exception as exc:
        return {"status": "failed", "url": item.url, "error": redact(f"{type(exc).__name__}: {exc}", secret)}
    if secret and secret in (raw or ""):
        return {"status": "failed", "url": item.url, "error": "model output contained a secret"}
    parsed = parse_json_response(raw or "")
    if not isinstance(parsed, dict):
        return {"status": "failed", "url": item.url, "error": "model response was not a JSON object"}
    return _judge(item, parsed)


def _judge(item: FeedItem, parsed: dict) -> dict:
    keep = parsed.get("keep") is True
    reason = plain_text(str(parsed.get("reason_zh") or ""))[:500]
    if not keep:
        return {"status": "rejected", "url": item.url, "reason_zh": reason}
    fields = {
        name: plain_text(str(parsed.get(name) or ""))[:800]
        for name in ("title_zh", "client_question", "impact", "relevance")
    }
    if not all(fields.values()) or not any(CJK_RE.search(value) for value in fields.values()):
        return {"status": "failed", "url": item.url, "error": "kept item was missing Chinese assessment fields"}
    quotes = []
    for quote in parsed.get("evidence") or []:
        if isinstance(quote, str) and _quote_in_source(quote, item.text):
            quotes.append(plain_text(quote)[:240])
    if not quotes:
        return {"status": "ungrounded", "url": item.url, "reason_zh": reason}
    event_date = parsed.get("event_date")
    if not (isinstance(event_date, str) and ISO_DATE_RE.fullmatch(event_date) and event_date in item.text):
        event_date = None
    candidate = {
        "profile": item.profile,
        "lane": "timely" if item.profile == "fintax-timely" else "deep",
        "title_zh": fields["title_zh"],
        "source_title": item.title,
        "source_name": item.source,
        "url": item.url,
        "url_note": _url_note(item.kind),
        "published_at": item.published_at.isoformat(),
        "event_date": event_date,
        "client_question": fields["client_question"],
        "impact": fields["impact"],
        "relevance": fields["relevance"],
        "evidence": quotes[:3],
    }
    return {"status": "kept", "url": item.url, "candidate": candidate}


def _dry_candidate(item: FeedItem) -> dict:
    return {
        "profile": item.profile,
        "lane": "timely" if item.profile == "fintax-timely" else "deep",
        "title_zh": "",
        "source_title": item.title,
        "source_name": item.source,
        "url": item.url,
        "url_note": _url_note(item.kind),
        "published_at": item.published_at.isoformat(),
        "event_date": None,
        "client_question": "",
        "impact": "",
        "relevance": "",
        "evidence": [],
        "enrichment": "dry_run",
    }


def _live_complete(config: RadarConfig, secret: str) -> Complete:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=secret,
        base_url=config.ai.base_url,
        max_retries=0,
        timeout=config.limits.llm_timeout_sec,
    )

    async def complete(request: dict) -> str:
        response = await client.chat.completions.create(**request)
        return response.choices[0].message.content or ""

    return complete


async def _http_fetch(url: str, timeout: float) -> bytes:
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": RSS_USER_AGENT},
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


def _provenance(*, config, now, run_date, hours_used, since, execute, commit, counts, feed_errors, failures, rows, secret) -> dict:
    body = {
        "version": 1,
        "role": "candidate-research-only",
        "horizon_commit": commit,
        "generated_at": now.astimezone(timezone.utc).isoformat(),
        "run_date": run_date,
        "timezone": config.timezone,
        "window_hours": hours_used,
        "window_start": since.isoformat(),
        "dry_run": not execute,
        "python": _python_version(),
        "config_sha256": hashlib.sha256(
            json.dumps(config.model_dump(), sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "llm": {
            "provider": config.ai.provider,
            "base_url": config.ai.base_url,
            "model": config.ai.model,
            "api_key_env": config.ai.api_key_env,
            "thinking": "disabled",
            "tools": [],
            "max_retries": 0,
            "response_format": "json_object",
        },
        "sources": [
            {"name": source.name, "profile": source.profile, "kind": source.kind, "url": source.resolved_url()}
            for source in config.sources
        ],
        "limits": config.limits.model_dump(),
        "counts": counts,
        "feed_errors": feed_errors,
        "failures": failures,
        "content_id": hashlib.sha256(
            json.dumps(
                [{"url": row["url"], "profile": row["profile"]} for row in rows],
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest(),
    }
    encoded = json.dumps(body, ensure_ascii=False)
    if secret and secret in encoded:
        raise RadarError("provenance contained the API key and was not written")
    return body


def _markdown(run_date: str, hours: int, execute: bool, candidates: list[dict]) -> str:
    mode = "已做候选评估" if execute else "干跑，未调用模型"
    lines = [
        f"# FinTax 候选研究 {run_date}",
        "",
        f"窗口 {hours} 小时。{mode}。这是候选研究，不是成稿，也不会写入站点。",
        "链接和发布日期来自本次抓取的 feed。编辑流程另行核对第一方原文。",
        "",
    ]
    if not candidates:
        lines.append("本窗口没有新的候选条目。")
        return "\n".join(lines) + "\n"
    for lane, heading in (("timely", "及时"), ("deep", "深研")):
        group = [item for item in candidates if item["lane"] == lane]
        if not group:
            continue
        lines.append(f"## {heading}")
        lines.append("")
        for item in group:
            title = item["title_zh"] or item["source_title"]
            lines.extend(
                [
                    f"### {title}",
                    "",
                    f"- 分轨: {item['lane']}",
                    f"- 来源: {item['source_name']}",
                    f"- 原标题: {item['source_title']}",
                    f"- 链接: {item['url']}",
                    f"- 链接说明: {item['url_note']}",
                    f"- 发布日期: {item['published_at']}",
                    f"- 事件日期: {item['event_date'] or '来源文本未写明'}",
                    f"- 客户问题: {item['client_question'] or '（干跑未生成）'}",
                    f"- 影响: {item['impact'] or '（干跑未生成）'}",
                    f"- 相关性: {item['relevance'] or '（干跑未生成）'}",
                    "",
                ]
            )
    return "\n".join(lines)


def _read_ledger(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "items": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RadarError(f"unreadable seen-item ledger: {path.name}") from exc
    if data.get("version") != 1 or not isinstance(data.get("items"), dict):
        raise RadarError(f"unreadable seen-item ledger: {path.name}")
    return data


def _write_ledger(path: Path, ledger: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, ledger)


def _remember(ledger: dict, item: FeedItem, run_date: str, status: str) -> None:
    ledger["items"][item.key] = {
        "url": item.url,
        "title": item.title,
        "profile": item.profile,
        "source": item.source,
        "published_at": item.published_at.isoformat(),
        "status": status,
        "first_seen_run": run_date,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _assert_clean(output_dir: Path, secret: str) -> None:
    if not secret:
        return
    for path in output_dir.iterdir():
        if path.is_file() and secret in path.read_text(encoding="utf-8"):
            path.unlink()
            raise RadarError("output contained the API key and was removed")


def plain_text(value: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def item_key(url: str) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


def _quote_in_source(quote: str, source: str) -> bool:
    cleaned = plain_text(quote)
    if len(cleaned) < 8:
        return False
    return cleaned.casefold() in source.casefold()


def _url_note(kind: str) -> str:
    if kind == "google_news":
        return "Google News 跳转链接。feed 未提供出版方原始 URL，不要把它当成机关官网。"
    return "来源 feed 中的原始链接。"


def _lane_rank(profile: str) -> int:
    return 0 if profile == "fintax-timely" else 1


def _safe_path(value, label: str) -> Path:
    path = Path(value)
    if any(part == ".." for part in path.parts):
        raise RadarError(f"{label} cannot contain '..'")
    return path


def _python_version() -> str:
    import sys

    return sys.version.split()[0]
