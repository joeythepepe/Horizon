"""CLI for the FinTax candidate radar. Dry-run unless --execute is set."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from .engine import RadarError, load_config, redact, resolve_commit, resolve_profiles_dir, run_radar


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="FinTax candidate radar. Writes local research files and does not publish.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Call DeepSeek. Omit for a dry-run.")
    parser.add_argument("--hours", type=int, default=None)
    parser.add_argument("--output-base", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--profiles-dir", type=Path, default=None)
    parser.add_argument("--now", type=str, default=None, help="ISO-8601 timestamp for tests and replays.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    try:
        config = load_config(args.config)
        profiles = resolve_profiles_dir(config, args.profiles_dir)
        now = _parse_now(args.now)
        result = asyncio.run(
            run_radar(
                config,
                now=now,
                output_base=args.output_base or Path(config.output_base),
                profiles_dir=profiles,
                execute=args.execute,
                hours=args.hours,
                ledger_path=args.ledger,
                commit=resolve_commit(args.config.parent),
            )
        )
    except RadarError as exc:
        print(redact(str(exc), _secret()), file=sys.stderr)
        return 2
    except Exception as exc:
        print(redact(f"{type(exc).__name__}: {exc}", _secret()), file=sys.stderr)
        return 1

    mode = "execute" if args.execute else "dry-run"
    print(f"{mode} wrote {result.candidate_count} candidates to {result.output_dir}")
    print(f"llm_calls={result.llm_calls} horizon_commit={result.provenance['horizon_commit']}")
    if result.provenance["counts"]["feed_errors"] >= len(config.sources):
        print(f"feed_errors={len(result.errors)}", file=sys.stderr)
        return 1
    return 0


def _parse_now(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise RadarError("--now must include a timezone offset")
    return parsed


def _secret() -> str:
    import os

    return os.environ.get("DEEPSEEK_API_KEY", "")


if __name__ == "__main__":
    raise SystemExit(main())
