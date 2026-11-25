# FinTax daily candidate radar

Local candidate research for a weekday morning run. It does not publish, send mail, or edit a site. The parent routine starts it. This package has no scheduler.

`horizon-fintax` is the entrypoint. The stock `horizon` command is not: a successful stock run copies Markdown into `docs/_posts` and retries model calls.

## Runtime

- Python 3.11.5 or newer (`requires-python` in `pyproject.toml`). This tree was tested on Python 3.12.3.
- Install the local wheel. Do not `pip install horizon` from PyPI; that name is not this package.
- The wheel carries the profiles. `radar.json` and `PROVENANCE.json` sit beside the install. `PROVENANCE.json` is produced with the artifact bundle and is not part of the git tree, because it records the commit that built the wheel.
- `DEEPSEEK_API_KEY` must already be in the process environment. The config names that variable. Do not put the key in a file, argument, or log.

```bash
python3 -m venv /opt/fintax-horizon/venv
/opt/fintax-horizon/venv/bin/pip install -r requirements-runtime.txt
/opt/fintax-horizon/venv/bin/pip install --no-deps /opt/fintax-horizon/horizon-0.1.0-py3-none-any.whl
```

Checked on 25 Nov 2026 against DeepSeek's API docs:

- OpenAI-compatible base URL: `https://api.deepseek.com`
- Chat Completions model: `deepseek-flash` (JSON output is documented). `deepseek-v4-pro` is the other listed model. Horizon's older default name `deepseek-chat` is not in that model list.
- Thinking defaults to on. This runner sends `thinking.type=disabled`.
- Responses API `web_search` is documented for `deepseek-v4-pro` and ignored on `deepseek-flash`. Chat Completions browsing is a different tool, `deepseek_web_search`, and can search on its own. This runner sends neither, and the FinTax profiles declare no Horizon `web_search` tool (that tool is DuckDuckGo, not DeepSeek).
- Each selected item is one call, `max_retries=0`, at most 8 calls. A failed call is not retried.

## Commands

Dry-run (no key, no ledger write):

```bash
horizon-fintax --config /opt/fintax-horizon/radar.json
```

Bounded real run, after the key is in the environment:

```bash
horizon-fintax --config /opt/fintax-horizon/radar.json --execute
```

Monday uses 72 hours and other weekdays use 24, on the Asia/Hong_Kong clock. Output goes to `/workspace/fintax-content-314cf32d/horizon-runs/YYYY-MM-DD/` with `candidates.json`, `candidates.md`, and `provenance.json`. The seen-item ledger is `/workspace/fintax-content-314cf32d/horizon-runs/ledger.json`. A second run skips URLs already assessed. Dry-run reads the ledger and does not update it.

## Sources

Google News queries use the same RSS search endpoint as Horizon's Google News scraper. HMRC and EU Taxation and Customs Union feeds were fetched and returned Atom/RSS. Official feeds are keyword-gated so unrelated tax or customs items never reach the model. IRS, IRD, IRAS, OECD, IFRS, and FASB did not expose a working public RSS from this environment; those names are scoped through Google News instead. Google News links are redirects, not agency URLs.

X, Hacker News, GitHub, Reddit, Telegram, OpenBB, email, webhooks, WeChat, and Pages delivery are not configured.

## Limits

Candidate notes are Chinese and are not articles. The editorial routine still checks first-party evidence, compares existing fintax.tech pages, and chooses what to publish. This repo must not contain run output or client material. The output directory is gitignored.
