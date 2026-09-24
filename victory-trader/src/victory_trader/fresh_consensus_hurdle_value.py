"""Request 178: fresh consensus hurdle-EV validation.

Open a new June 23/24/25/26/29 block only after freezing the candidate.
Upstream attention, HOT, economic opportunity and executability are unchanged.

For each fresh BUY anchor, build causal POSITION rows and rich one-second
features. Train the Request-176 row-weighted and Request-177 equal-day-weighted
hurdle EV heads only on the old Apr/May development partitions.

Consensus score = min(row_weighted_EV, day_weighted_EV).
Consensus-positive iff both expected values are > 0.

This request validates post-entry value observability only. It does not yet
execute a recurrent HOLD/EXIT trajectory.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .causal_entry_executability import (
    EXECUTION_FEATURES,
    _fit_exec_model,
)
from .config import load_settings, require_flatfile_credentials
from .day_balanced_hurdle_expected_value import (
    _predict_probability as predict_day_probability,
    train_day_balanced_classifier,
    train_day_balanced_magnitude,
)
from .extended_history_economic_opportunity import (
    _fit_extended_policy_selector,
    _fit_frozen_attention,
)
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .fresh_validated_attention_economic_opportunity import (
    _build_frozen_active,
)
from .hierarchical_attention_runtime import add_stage2_scores, fit_stage2
from .hot_economic_opportunity import (
    ENTRY_FEATURES,
    _first_hot_feature_rows,
)
from .market_regime_hurdle_expected_value import (
    _predict_magnitude,
    _train_magnitude_head,
)
from .market_regime_position_value import (
    MARKET_REGIME_FEATURES,
    attach_market_regime,
    build_market_regime,
)
from .market_regime_positive_value import (
    predict_probability,
    train_classifier,
)
from .massive_client import MassiveClient
from .market_calendar import is_us_equity_trading_day
from .rich_second_position_value import rich_second_features
from .second_path_attention_probe import (
    SECOND_CACHE_DIR,
    _second_frame,
)
from .selected_hot_position_value_observability import (
    apply_excess_target,
    build_position_rows,
    fit_minute_baselines,
)
from .hierarchical_active_features import HORIZON
from .targeted_second_hot_reranker import add_second_features
from .temporal_active_admission import add_cross_within_horizon_targets

REQUEST_ID = 178
FRESH_DAYS = (
    "2026-06-23",
    "2026-06-24",
    "2026-06-25",
    "2026-06-26",
    "2026-06-29",
)

MIN_PATH_COVERAGE = 0.85
MIN_CONTEXT_COVERAGE = 0.95
MIN_CONSENSUS_SPEARMAN = 0.08
MIN_SELECTED_RATE = 0.10
MIN_SELECTED_REALIZED_MEAN_PCT = 0.20
MIN_GOOD_DAYS = 4
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20261078


def _rich_enrich(
    frame: pd.DataFrame,
    client: MassiveClient,
) -> pd.DataFrame:
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    records: list[dict[str, float]] = []
    for row in frame.loc[
        :, ["trading_day", "ticker", "state_t"]
    ].to_dict("records"):
        day = str(row["trading_day"])
        ticker = str(row["ticker"]).upper()
        key = (day, ticker)
        if key not in cache:
            payload = client.second_bars_range(
                ticker,
                date.fromisoformat(day),
                date.fromisoformat(day),
                adjusted=False,
            )
            cache[key] = _second_frame(payload)
        records.append(
            rich_second_features(cache[key], int(row["state_t"]))
        )
    return pd.concat(
        [frame.reset_index(drop=True), pd.DataFrame(records)],
        axis=1,
    )


def prepare_day(
    day_text: str,
    history_first_hot_path: Path,
    stage2_candidates_path: Path,
    positions_output: Path,
    scan_output: Path,
    audit_output: Path,
) -> int:
    if day_text not in FRESH_DAYS:
        raise ValueError(f"request 178 unexpected fresh day {day_text}")
    day = date.fromisoformat(day_text)
    if not is_us_equity_trading_day(day):
        raise ValueError(f"request 178 non-trading day {day_text}")

    settings = load_settings()
    access_key, secret_key = require_flatfile_credentials(settings)
    store = MassiveFlatFileStore(
        MassiveFlatFilesClient(access_key, secret_key),
        FLATFILE_CACHE_DIR,
    )
    scan_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=Path("data/cache/massive"),
        request_interval_seconds=0.0,
    )
    second_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=SECOND_CACHE_DIR,
        request_interval_seconds=0.0,
    )

    market_model, active_model = _fit_frozen_attention(
        store,
        scan_client,
        date.fromisoformat("2026-01-02"),
        day,
    )
    scan, _ = build_flatfile_scan_day(store, scan_client, day)
    temporal = add_cross_within_horizon_targets(
        scan,
        horizons=(HORIZON,),
    )
    active = _build_frozen_active(
        temporal,
        market_model,
        active_model,
    )
    candidates, second_audit = add_second_features(
        active,
        second_client,
    )

    stage2 = pd.read_parquet(stage2_candidates_path)
    minute_model, second_model = fit_stage2(stage2)
    scored = add_stage2_scores(
        candidates,
        minute_model,
        second_model,
    )
    first_hot, runtime_audit = _first_hot_feature_rows(
        scored,
        scan,
    )

    history = pd.read_parquet(history_first_hot_path)
    economic_model, economic_gate = _fit_extended_policy_selector(history)
    exec_model, exec_cal = _fit_exec_model(history)

    econ_p = economic_model.predict_proba(
        first_hot.loc[:, ENTRY_FEATURES].replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )[:, 1]
    exec_p = exec_model.predict_proba(
        first_hot.loc[:, EXECUTION_FEATURES].replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )[:, 1]
    first_hot["economic_probability"] = econ_p
    first_hot["execution_probability"] = exec_p
    first_hot["entry_action"] = np.where(
        (econ_p >= float(economic_gate))
        & (exec_p >= float(exec_cal["threshold"])),
        "BUY",
        np.where(
            econ_p >= float(economic_gate),
            "WAIT",
            "ABSTAIN",
        ),
    )

    anchors = first_hot.loc[
        first_hot["entry_action"].astype(str).eq("BUY"),
        ["trading_day", "ticker", "t"],
    ].copy()
    positions, path_audit = build_position_rows(
        anchors,
        scan,
        second_client,
    )
    if positions.empty:
        raise ValueError(
            f"request 178 {day_text} produced no position rows"
        )
    positions = _rich_enrich(positions, second_client)

    path_anchor_count = int(
        positions.loc[:, ["trading_day", "ticker", "hot_t"]]
        .drop_duplicates()
        .shape[0]
    )
    path_coverage = (
        float(path_anchor_count / len(anchors))
        if len(anchors)
        else None
    )

    positions_output.parent.mkdir(parents=True, exist_ok=True)
    scan_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    positions.to_parquet(
        positions_output,
        index=False,
        compression="zstd",
    )
    scan.to_parquet(
        scan_output,
        index=False,
        compression="zstd",
    )

    audit = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "fresh_day": day_text,
        "first_hot_rows": int(len(first_hot)),
        "entry_actions": {
            str(key): int(value)
            for key, value in first_hot["entry_action"]
            .value_counts()
            .to_dict()
            .items()
        },
        "buy_anchors": int(len(anchors)),
        "path_anchors": path_anchor_count,
        "anchor_path_coverage": path_coverage,
        "position_rows": int(len(positions)),
        "economic_gate": float(economic_gate),
        "execution_gate": float(exec_cal["threshold"]),
        "runtime_audit": runtime_audit,
        "path_audit": path_audit,
        "active_second_audit": second_audit,
        "flatfile_stats": store.stats.to_dict(),
        "second_client_stats": second_client.stats.to_dict(),
    }
    audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


def _day_bootstrap(
    selected: pd.DataFrame,
) -> dict[str, float | int | None]:
    values = pd.to_numeric(
        selected["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    temp = selected.loc[
        values.notna(),
        ["trading_day"],
    ].copy()
    temp["value"] = values.loc[values.notna()].to_numpy()
    daily = (
        temp.groupby(
            temp["trading_day"].astype(str),
            sort=True,
        )["value"]
        .mean()
        .to_numpy(dtype=float)
    )
    if len(daily) < 2:
        return {
            "days": int(len(daily)),
            "samples": BOOTSTRAP_SAMPLES,
            "ci_low_pct": None,
            "ci_high_pct": None,
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(
        0,
        len(daily),
        size=(BOOTSTRAP_SAMPLES, len(daily)),
    )
    draws = daily[idx].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "days": int(len(daily)),
        "samples": BOOTSTRAP_SAMPLES,
        "ci_low_pct": float(low),
        "ci_high_pct": float(high),
    }


def _safe_spearman(
    target: pd.Series,
    score: pd.Series,
) -> float | None:
    y = pd.to_numeric(target, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20:
        return None
    corr = y.loc[valid].corr(
        s.loc[valid],
        method="spearman",
    )
    return None if pd.isna(corr) else float(corr)


def _score_row_ev(
    frame: pd.DataFrame,
    classifier,
    positive_head,
    nonpositive_head,
) -> np.ndarray:
    p = predict_probability(frame, classifier)
    win = _predict_magnitude(frame, positive_head)
    loss = _predict_magnitude(frame, nonpositive_head)
    return p * win + (1.0 - p) * loss


def _score_day_ev(
    frame: pd.DataFrame,
    classifier,
    positive_head,
    nonpositive_head,
) -> np.ndarray:
    p = predict_day_probability(frame, classifier)
    win = _predict_magnitude(frame, positive_head)
    loss = _predict_magnitude(frame, nonpositive_head)
    return p * win + (1.0 - p) * loss


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    history_scan_path: Path,
    fresh_dir: Path,
    output_path: Path,
    scored_output_path: Path,
) -> int:
    fit = pd.read_parquet(fit_path)
    calibration = pd.read_parquet(calibration_path)
    history_scan = pd.read_parquet(history_scan_path)

    position_paths = sorted(
        fresh_dir.glob("*-positions.parquet")
    )
    scan_paths = sorted(
        fresh_dir.glob("*-scan.parquet")
    )
    audit_paths = sorted(
        fresh_dir.glob("*-audit.json")
    )
    if (
        len(position_paths) != len(FRESH_DAYS)
        or len(scan_paths) != len(FRESH_DAYS)
        or len(audit_paths) != len(FRESH_DAYS)
    ):
        raise ValueError("request 178 fresh shards are incomplete")

    fresh = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [pd.read_parquet(path) for path in scan_paths],
        ignore_index=True,
    )
    audits = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in audit_paths
    ]
    found_days = sorted(
        fresh["trading_day"].astype(str).unique()
    )
    if found_days != list(FRESH_DAYS):
        raise ValueError(
            f"request 178 expected {FRESH_DAYS}, found {found_days}"
        )

    historical_days = set(
        fit["trading_day"].astype(str)
    ) | set(calibration["trading_day"].astype(str))
    historical_scan = history_scan.loc[
        history_scan["trading_day"].astype(str).isin(
            historical_days
        )
    ].copy()

    hist_regime = build_market_regime(historical_scan)
    fresh_regime = build_market_regime(fresh_scan)
    fit = attach_market_regime(fit, hist_regime)
    calibration = attach_market_regime(
        calibration,
        hist_regime,
    )
    fresh = attach_market_regime(
        fresh,
        fresh_regime,
    )

    baselines = fit_minute_baselines(fit)
    fit = apply_excess_target(fit, baselines)
    calibration = apply_excess_target(
        calibration,
        baselines,
    )
    fresh = apply_excess_target(fresh, baselines)

    row_classifier = train_classifier(fit, calibration)
    row_positive = _train_magnitude_head(
        fit,
        calibration,
        positive_class=True,
        seed=20261077,
    )
    row_nonpositive = _train_magnitude_head(
        fit,
        calibration,
        positive_class=False,
        seed=20261078,
    )

    day_classifier = train_day_balanced_classifier(
        fit,
        calibration,
    )
    day_positive = train_day_balanced_magnitude(
        fit,
        calibration,
        positive_class=True,
        seed=20261078,
    )
    day_nonpositive = train_day_balanced_magnitude(
        fit,
        calibration,
        positive_class=False,
        seed=20261079,
    )

    scored = fresh.copy()
    scored["row_weighted_hurdle_ev_pct"] = _score_row_ev(
        scored,
        row_classifier,
        row_positive,
        row_nonpositive,
    )
    scored["day_weighted_hurdle_ev_pct"] = _score_day_ev(
        scored,
        day_classifier,
        day_positive,
        day_nonpositive,
    )
    scored["consensus_hurdle_ev_pct"] = np.minimum(
        pd.to_numeric(
            scored["row_weighted_hurdle_ev_pct"],
            errors="coerce",
        ),
        pd.to_numeric(
            scored["day_weighted_hurdle_ev_pct"],
            errors="coerce",
        ),
    )

    target = pd.to_numeric(
        scored["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    score = pd.to_numeric(
        scored["consensus_hurdle_ev_pct"],
        errors="coerce",
    )
    valid = target.notna() & score.notna()
    selected = scored.loc[
        valid & score.gt(0),
    ].copy()
    selected_target = pd.to_numeric(
        selected["excess_remaining_option_value_pct"],
        errors="coerce",
    )

    selected_rate = (
        float(len(selected) / int(valid.sum()))
        if int(valid.sum())
        else None
    )
    selected_mean = (
        float(selected_target.mean())
        if len(selected)
        else None
    )
    daily_selected = (
        pd.DataFrame(
            {
                "day": selected["trading_day"].astype(str),
                "value": selected_target,
            }
        )
        .dropna()
        .groupby("day", sort=True)["value"]
        .mean()
    )
    day_balanced = (
        float(daily_selected.mean())
        if len(daily_selected)
        else None
    )
    bootstrap = _day_bootstrap(selected)

    by_day: dict[str, object] = {}
    good_days = 0
    for day in FRESH_DAYS:
        part = scored.loc[
            scored["trading_day"].astype(str).eq(day)
        ].copy()
        pt = pd.to_numeric(
            part["excess_remaining_option_value_pct"],
            errors="coerce",
        )
        ps = pd.to_numeric(
            part["consensus_hurdle_ev_pct"],
            errors="coerce",
        )
        pv = pt.notna() & ps.notna()
        corr = _safe_spearman(
            pt.loc[pv],
            ps.loc[pv],
        )
        psel = part.loc[pv & ps.gt(0)].copy()
        vals = pd.to_numeric(
            psel["excess_remaining_option_value_pct"],
            errors="coerce",
        )
        mean = float(vals.mean()) if len(psel) else None
        good = bool(
            corr is not None
            and corr > 0
            and mean is not None
            and mean > 0
        )
        good_days += int(good)
        by_day[day] = {
            "rows": int(len(part)),
            "evaluable_rows": int(pv.sum()),
            "spearman": corr,
            "selected_rows": int(len(psel)),
            "selected_rate": (
                float(len(psel) / int(pv.sum()))
                if int(pv.sum())
                else None
            ),
            "selected_realized_excess_mean_pct": mean,
            "both_positive": good,
        }

    path_coverage = float(
        sum(int(a["path_anchors"]) for a in audits)
        / sum(int(a["buy_anchors"]) for a in audits)
    )
    context_coverage = float(
        scored.loc[:, MARKET_REGIME_FEATURES]
        .notna()
        .any(axis=1)
        .mean()
    )
    consensus_spearman = _safe_spearman(
        target.loc[valid],
        score.loc[valid],
    )

    gate = bool(
        path_coverage >= MIN_PATH_COVERAGE
        and context_coverage >= MIN_CONTEXT_COVERAGE
        and consensus_spearman is not None
        and consensus_spearman >= MIN_CONSENSUS_SPEARMAN
        and selected_rate is not None
        and selected_rate >= MIN_SELECTED_RATE
        and selected_mean is not None
        and selected_mean >= MIN_SELECTED_REALIZED_MEAN_PCT
        and day_balanced is not None
        and day_balanced > 0
        and bootstrap["ci_low_pct"] is not None
        and float(bootstrap["ci_low_pct"]) > 0
        and good_days >= MIN_GOOD_DAYS
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "fresh_days": list(FRESH_DAYS),
        "opens_new_dates": True,
        "upstream_entry_policy_changed": False,
        "consensus_rule": (
            "min(row_weighted_hurdle_EV, day_weighted_hurdle_EV) > 0"
        ),
        "fresh_path_coverage": path_coverage,
        "market_context_coverage": context_coverage,
        "rows": int(len(scored)),
        "evaluable_rows": int(valid.sum()),
        "consensus_spearman": consensus_spearman,
        "selected_rows": int(len(selected)),
        "selected_rate": selected_rate,
        "selected_realized_excess_mean_pct": selected_mean,
        "selected_day_balanced_excess_mean_pct": day_balanced,
        "selected_day_bootstrap": bootstrap,
        "good_days": int(good_days),
        "by_day": by_day,
        "fresh_preparation_audits": audits,
        "frozen_gate": {
            "min_path_coverage": MIN_PATH_COVERAGE,
            "min_context_coverage": MIN_CONTEXT_COVERAGE,
            "min_consensus_spearman": MIN_CONSENSUS_SPEARMAN,
            "min_selected_rate": MIN_SELECTED_RATE,
            "min_selected_realized_mean_pct": (
                MIN_SELECTED_REALIZED_MEAN_PCT
            ),
            "min_good_days": MIN_GOOD_DAYS,
            "bootstrap_ci_low_must_be_positive": True,
        },
        "promotion_gate_pass": gate,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    scored.to_parquet(
        scored_output_path,
        index=False,
        compression="zstd",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare-day")
    prep.add_argument("day")
    prep.add_argument("--history-first-hot", type=Path, required=True)
    prep.add_argument("--stage2-candidates", type=Path, required=True)
    prep.add_argument("--positions-output", type=Path, required=True)
    prep.add_argument("--scan-output", type=Path, required=True)
    prep.add_argument("--audit-output", type=Path, required=True)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--fit", type=Path, required=True)
    ev.add_argument("--calibration", type=Path, required=True)
    ev.add_argument("--history-scan", type=Path, required=True)
    ev.add_argument("--fresh-dir", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--scored-output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare-day":
        return prepare_day(
            args.day,
            args.history_first_hot,
            args.stage2_candidates,
            args.positions_output,
            args.scan_output,
            args.audit_output,
        )
    return evaluate(
        args.fit,
        args.calibration,
        args.history_scan,
        args.fresh_dir,
        args.output,
        args.scored_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
