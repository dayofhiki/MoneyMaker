"""Fresh confirmation of unchanged rate-limited subscription policy.

Request 136 keeps request 135's policy unchanged and only aligns the
operational audit with its preregistered wording: the five-addition limit
applies after each session's mandatory initial subscription fill.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from . import rate_limited_subscription_probe as base
from .attention_flatfile_replay import FLATFILE_CACHE_DIR
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .massive_client import MassiveClient

EVAL_DAYS = [
    "2026-04-23",
    "2026-04-24",
    "2026-04-27",
    "2026-04-28",
    "2026-04-29",
]


def confirmation_gate(evaluation: dict[str, object]) -> bool:
    by_day = evaluation["by_day"]
    support_ok = all(
        int(by_day[day]["focus"]["runner_crossings"]) >= 10
        for day in EVAL_DAYS
    )
    rank40 = evaluation["rank40_hysteresis"]["prior_minute_rate"]
    active = evaluation["rate_limited_active"]["prior_minute_rate"]
    retention = evaluation["active_retention_given_focus"]
    audit = evaluation["rate_limited_audit"]
    return bool(
        support_ok
        and rank40 is not None
        and active is not None
        and float(active) > float(rank40)
        and retention is not None
        and float(retention) >= 0.95
        and int(evaluation["nonlower_capture_days"]) >= 4
        and int(audit["max_occupancy"]) <= base.SHORTLIST_BUDGET
        and audit["mean_post_initial_additions_per_decision"] is not None
        and float(audit["mean_post_initial_additions_per_decision"]) <= 5.0
        and int(audit["max_post_initial_additions_per_decision"]) <= 5
    )


def run_probe(
    store: MassiveFlatFileStore,
    rest_client: MassiveClient,
    start: date,
    end: date,
):
    previous_days = base.EVAL_DAYS
    base.EVAL_DAYS = EVAL_DAYS
    try:
        output, summary = base.run_probe(store, rest_client, start, end)
    finally:
        base.EVAL_DAYS = previous_days

    evaluation = summary["evaluation"]
    evaluation["request135_gate_definition_pass"] = bool(
        evaluation["transport_gate_pass"]
    )
    evaluation["confirmation_gate_pass"] = confirmation_gate(evaluation)
    evaluation["operational_gate_definition"] = (
        "mean and max additions per decision after mandatory session "
        "initialization are <= 5"
    )
    summary["schema_version"] = 2
    summary["eval_days"] = EVAL_DAYS
    summary["request_id"] = 136
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Confirm unchanged rate-limited transport on a fresh block."
    )
    parser.add_argument("start", type=date.fromisoformat)
    parser.add_argument("end", type=date.fromisoformat)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    settings = load_settings()
    access_key, secret_key = require_flatfile_credentials(settings)
    store = MassiveFlatFileStore(
        MassiveFlatFilesClient(access_key, secret_key), FLATFILE_CACHE_DIR
    )
    rest_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=Path("data/cache/massive"),
        request_interval_seconds=0.0,
    )
    output, summary = run_probe(store, rest_client, args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False, compression="zstd")
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
