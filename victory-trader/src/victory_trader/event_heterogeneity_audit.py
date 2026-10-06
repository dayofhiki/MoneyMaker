"""R316: frozen-score event heterogeneity audit on the R311 May HOT cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .clock_momentum_increment import MOMENTUM, within_day_auc
from .config import load_settings
from .feasible_upside_observability import CEILING, truth
from .frozen_entry_second_hold_exit import KEYS
from .frozen_momentum_may_transport import ARMS, EVAL_DAYS
from .hierarchical_crack_entry_controller import _episode_day_weights
from .massive_client import MassiveClient
from .preentry_momentum_observability import metrics

REQUEST_ID = 316
PRIMARY_FLAGS = ("fresh_news_24h", "prior_8k_7d", "recent_reverse_split_20d")
SECONDARY_FLAGS = (
    "fresh_news_1h",
    "recent_news_7d",
    "prior_8k_30d",
    "recent_reverse_split_5d",
)
EVENT_CACHE_DIR = Path("data/cache/massive-event-heterogeneity")
SCORE_COLUMNS = tuple(f"{arm}_p_net_5" for arm in ARMS)


def _iso_utc(timestamp: pd.Timestamp) -> str:
    return timestamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _decision_time(value: Any) -> pd.Timestamp:
    return pd.to_datetime(int(value), unit="ms", utc=True)


def validate_frozen_input(states: pd.DataFrame, summary: dict[str, Any]) -> None:
    required = {
        "trading_day",
        "ticker",
        "decision_t",
        "observation_available",
        "label_complete",
        CEILING,
        *SCORE_COLUMNS,
        *MOMENTUM,
    }
    missing = required - set(states)
    if missing:
        raise ValueError(f"R311 frozen states missing columns: {sorted(missing)}")
    if summary.get("request_id") != 311:
        raise ValueError("official R311 summary required")
    if len(states) != int(summary.get("transport", {}).get("sampled_episodes", -1)):
        raise ValueError("R311 sampled identity count mismatch")
    if len(states) != 828 or set(states.trading_day.astype(str)) != set(EVAL_DAYS):
        raise ValueError("exact frozen R311 828-case May cohort required")
    if states.duplicated(KEYS).any():
        raise ValueError("unique R311 ticker-day-HOT identities required")
    observed = states.observation_available.astype(bool)
    if states.loc[~observed, list(SCORE_COLUMNS)].notna().any().any():
        raise ValueError("unobserved R311 cases must not carry model scores")
    complete = states.label_complete.astype(bool)
    if complete.sum() != int(summary.get("transport", {}).get("labeled_episodes", -1)):
        raise ValueError("R311 complete-label count mismatch")


def _paginate_private(client: MassiveClient, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    page = client._get(path, params)  # audit-local use of the shared secret-safe transport
    rows = list(page.get("results") or [])
    next_url = page.get("next_url")
    while next_url:
        page = client._get_next_url(str(next_url))
        rows.extend(page.get("results") or [])
        next_url = page.get("next_url")
    return rows


def market_splits(client: MassiveClient, start: date, end: date) -> list[dict[str, Any]]:
    """Fetch stock splits from Massive's current, non-deprecated endpoint."""
    return _paginate_private(
        client,
        "/stocks/v1/splits",
        {
            "execution_date.gte": start.isoformat(),
            "execution_date.lte": end.isoformat(),
            "limit": 5000,
            "sort": "execution_date.asc",
        },
    )


def _news_record(ticker: str, row: dict[str, Any]) -> dict[str, Any] | None:
    published = pd.to_datetime(row.get("published_utc"), utc=True, errors="coerce")
    if pd.isna(published):
        return None
    return {
        "source": "news",
        "ticker": ticker,
        "event_time": _iso_utc(published),
        "event_date": published.date().isoformat(),
        "event_id": str(row.get("id") or ""),
        "category": "",
    }


def _eight_k_records(row: dict[str, Any], cohort_tickers: set[str]) -> list[dict[str, Any]]:
    filing = pd.to_datetime(row.get("filing_date"), errors="coerce")
    if pd.isna(filing):
        return []
    result = []
    for raw in row.get("tickers") or []:
        ticker = str(raw).upper()
        if ticker not in cohort_tickers:
            continue
        result.append(
            {
                "source": "8k",
                "ticker": ticker,
                "event_time": "",
                "event_date": filing.date().isoformat(),
                "event_id": str(row.get("accession_number") or ""),
                "category": "|".join(
                    str(row.get(key) or "")
                    for key in ("primary_category", "secondary_category", "tertiary_category")
                ),
            }
        )
    return result


def _split_record(row: dict[str, Any], cohort_tickers: set[str]) -> dict[str, Any] | None:
    ticker = str(row.get("ticker") or "").upper()
    execution = pd.to_datetime(row.get("execution_date"), errors="coerce")
    if ticker not in cohort_tickers or pd.isna(execution):
        return None
    return {
        "source": "split",
        "ticker": ticker,
        "event_time": "",
        "event_date": execution.date().isoformat(),
        "event_id": str(row.get("id") or ""),
        "category": str(row.get("adjustment_type") or ""),
    }


def acquire_events(states: pd.DataFrame, client: MassiveClient) -> tuple[pd.DataFrame, dict[str, Any]]:
    observed = states.loc[states.observation_available.astype(bool) & states.decision_t.notna()].copy()
    tickers = set(observed.ticker.astype(str).str.upper())
    records: list[dict[str, Any]] = []
    failures: dict[str, Any] = {"news_tickers": [], "eight_k": None, "splits": None}

    for ticker, group in observed.groupby(observed.ticker.astype(str).str.upper(), sort=True):
        first = _decision_time(group.decision_t.min()) - pd.Timedelta(days=7)
        last = _decision_time(group.decision_t.max())
        try:
            rows = client.news(
                str(ticker),
                published_gte=_iso_utc(first),
                published_lte=_iso_utc(last),
                limit=1000,
            )
            for row in rows:
                normalized = _news_record(str(ticker), row)
                if normalized is not None:
                    records.append(normalized)
        except Exception as error:
            failures["news_tickers"].append({"ticker": str(ticker), "error_type": type(error).__name__})

    minimum_day = date.fromisoformat(str(observed.trading_day.min())) - timedelta(days=30)
    maximum_day = date.fromisoformat(str(observed.trading_day.max()))

    try:
        rows = client.eight_k_disclosures_market(
            filing_date_gte=minimum_day,
            filing_date_lte=maximum_day,
            limit=1000,
        )
        for row in rows:
            records.extend(_eight_k_records(row, tickers))
    except Exception as error:
        failures["eight_k"] = {"error_type": type(error).__name__}

    try:
        rows = market_splits(client, minimum_day, maximum_day)
        for row in rows:
            normalized = _split_record(row, tickers)
            if normalized is not None:
                records.append(normalized)
    except Exception as error:
        failures["splits"] = {"error_type": type(error).__name__}

    events = pd.DataFrame(
        records,
        columns=("source", "ticker", "event_time", "event_date", "event_id", "category"),
    ).drop_duplicates()
    audit = {
        "observed_tickers": len(tickers),
        "event_rows": len(events),
        "source_rows": events.source.value_counts().to_dict() if len(events) else {},
        "failures": failures,
        "client_stats": client.stats.to_dict(),
    }
    return events, audit


def _event_maps(events: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    news: dict[str, list[int]] = {}
    filings: dict[str, list[np.datetime64]] = {}
    splits: dict[str, list[np.datetime64]] = {}
    for row in events.itertuples(index=False):
        ticker = str(row.ticker).upper()
        if row.source == "news":
            stamp = pd.to_datetime(row.event_time, utc=True, errors="coerce")
            if not pd.isna(stamp):
                news.setdefault(ticker, []).append(int(stamp.value // 1_000_000))
        elif row.source == "8k":
            filings.setdefault(ticker, []).append(np.datetime64(str(row.event_date), "D"))
        elif row.source == "split" and str(row.category) == "reverse_split":
            splits.setdefault(ticker, []).append(np.datetime64(str(row.event_date), "D"))
    return (
        {k: np.sort(np.asarray(v, dtype=np.int64)) for k, v in news.items()},
        {k: np.sort(np.asarray(v, dtype="datetime64[D]")) for k, v in filings.items()},
        {k: np.sort(np.asarray(v, dtype="datetime64[D]")) for k, v in splits.items()},
    )


def _has_news(times: np.ndarray, decision_ms: int, seconds: int) -> bool:
    if len(times) == 0:
        return False
    left = decision_ms - seconds * 1000
    return bool(np.any((times >= left) & (times <= decision_ms)))


def _days_ago(dates: np.ndarray, day: np.datetime64) -> np.ndarray:
    return (day - dates).astype("timedelta64[D]").astype(int)


def annotate_events(
    states: pd.DataFrame,
    events: pd.DataFrame,
    acquisition: dict[str, Any],
) -> pd.DataFrame:
    news, filings, splits = _event_maps(events)
    failed_news = {str(row["ticker"]).upper() for row in acquisition["failures"]["news_tickers"]}
    eight_k_ok = acquisition["failures"]["eight_k"] is None
    splits_ok = acquisition["failures"]["splits"] is None

    rows = []
    for raw in states.to_dict("records"):
        ticker = str(raw["ticker"]).upper()
        result = dict(raw)
        if not bool(raw["observation_available"]) or pd.isna(raw["decision_t"]):
            for flag in (*PRIMARY_FLAGS, *SECONDARY_FLAGS, "known_event_any", "none_known"):
                result[flag] = pd.NA
            result["same_day_8k_count"] = pd.NA
            result["same_day_reverse_split_count"] = pd.NA
            result["event_sources_complete"] = False
            rows.append(result)
            continue

        decision = int(raw["decision_t"])
        trading_day = np.datetime64(str(raw["trading_day"]), "D")

        news_ok = ticker not in failed_news
        if news_ok:
            times = news.get(ticker, np.array([], dtype=np.int64))
            result["fresh_news_1h"] = _has_news(times, decision, 3600)
            result["fresh_news_24h"] = _has_news(times, decision, 24 * 3600)
            result["recent_news_7d"] = _has_news(times, decision, 7 * 24 * 3600)
        else:
            result["fresh_news_1h"] = pd.NA
            result["fresh_news_24h"] = pd.NA
            result["recent_news_7d"] = pd.NA

        if eight_k_ok:
            values = filings.get(ticker, np.array([], dtype="datetime64[D]"))
            ago = _days_ago(values, trading_day) if len(values) else np.array([], dtype=int)
            result["prior_8k_7d"] = bool(np.any((ago >= 1) & (ago <= 7)))
            result["prior_8k_30d"] = bool(np.any((ago >= 1) & (ago <= 30)))
            result["same_day_8k_count"] = int(np.sum(ago == 0))
        else:
            result["prior_8k_7d"] = pd.NA
            result["prior_8k_30d"] = pd.NA
            result["same_day_8k_count"] = pd.NA

        if splits_ok:
            values = splits.get(ticker, np.array([], dtype="datetime64[D]"))
            ago = _days_ago(values, trading_day) if len(values) else np.array([], dtype=int)
            result["recent_reverse_split_5d"] = bool(np.any((ago >= 1) & (ago <= 5)))
            result["recent_reverse_split_20d"] = bool(np.any((ago >= 1) & (ago <= 20)))
            result["same_day_reverse_split_count"] = int(np.sum(ago == 0))
        else:
            result["recent_reverse_split_5d"] = pd.NA
            result["recent_reverse_split_20d"] = pd.NA
            result["same_day_reverse_split_count"] = pd.NA

        complete = news_ok and eight_k_ok and splits_ok
        result["event_sources_complete"] = complete
        primary = [result[flag] for flag in PRIMARY_FLAGS]
        if complete:
            result["known_event_any"] = bool(any(bool(value) for value in primary))
            result["none_known"] = not result["known_event_any"]
        else:
            result["known_event_any"] = pd.NA
            result["none_known"] = pd.NA
        rows.append(result)
    return pd.DataFrame(rows)


def _labeled(frame: pd.DataFrame) -> pd.DataFrame:
    mask = (
        frame.observation_available.astype(bool)
        & frame.label_complete.astype(bool)
        & np.isfinite(frame[CEILING].to_numpy(float))
    )
    return frame.loc[mask].copy()


def _positive_days(frame: pd.DataFrame) -> int:
    return sum(bool(truth(group[CEILING], 5).any()) for _, group in frame.groupby("trading_day"))


def subgroup_report(frame: pd.DataFrame) -> dict[str, Any]:
    labeled = _labeled(frame)
    result: dict[str, Any] = {
        "rows": len(frame),
        "labeled": len(labeled),
        "positives": int(truth(labeled[CEILING], 5).sum()) if len(labeled) else 0,
        "positive_days": _positive_days(labeled) if len(labeled) else 0,
        "weighted_prevalence": None,
        "arms": {},
    }
    if not len(labeled):
        return result
    weights = _episode_day_weights(labeled)
    result["weighted_prevalence"] = float(np.average(truth(labeled[CEILING], 5), weights=weights))
    result["arms"] = {
        arm: metrics(labeled, arm, 5) | {"within_day_auc": within_day_auc(labeled, arm)}
        for arm in ARMS
    }
    return result


def _point_comparison(frame: pd.DataFrame, flag: str, weights: np.ndarray | None = None) -> dict[str, Any]:
    if weights is None:
        weights = _episode_day_weights(frame)
    mask = frame[flag].astype(bool).to_numpy()
    if mask.all() or (~mask).all():
        return {"prevalence_difference": None, "prevalence_ratio": None, "M_within_day_auc_difference": None, "MQ_increment_difference": None}
    y = truth(frame[CEILING], 5)
    true_prev = float(np.average(y[mask], weights=weights[mask]))
    false_prev = float(np.average(y[~mask], weights=weights[~mask]))
    ratio = true_prev / false_prev if false_prev > 0 else None

    def auc_difference(arm: str) -> float | None:
        left = within_day_auc(frame.loc[mask], arm, weights[mask])
        right = within_day_auc(frame.loc[~mask], arm, weights[~mask])
        return None if left is None or right is None else float(left - right)

    m = auc_difference("M")
    mq_true_m = within_day_auc(frame.loc[mask], "M", weights[mask])
    mq_true_q = within_day_auc(frame.loc[mask], "Q", weights[mask])
    mq_false_m = within_day_auc(frame.loc[~mask], "M", weights[~mask])
    mq_false_q = within_day_auc(frame.loc[~mask], "Q", weights[~mask])
    increment = None
    if None not in (mq_true_m, mq_true_q, mq_false_m, mq_false_q):
        increment = float((mq_true_m - mq_true_q) - (mq_false_m - mq_false_q))
    return {
        "prevalence_difference": true_prev - false_prev,
        "prevalence_ratio": ratio,
        "M_within_day_auc_difference": m,
        "MQ_increment_difference": increment,
    }


def _interval(values: list[float]) -> list[float] | None:
    finite = [value for value in values if np.isfinite(value)]
    return np.quantile(finite, [0.025, 0.975]).tolist() if finite else None


def bootstrap_comparison(
    frame: pd.DataFrame,
    flag: str,
    columns: list[str],
    seed: int,
    draws: int = 1000,
) -> dict[str, Any]:
    base = _labeled(frame.loc[frame[flag].notna()]).reset_index(drop=True)
    if base.empty:
        return {"clusters": 0, "draws": draws, "valid_draws": {}, "ci95": {}}
    groups = list(base.groupby(columns, sort=False).indices.values())
    base_weights = _episode_day_weights(base)
    rng = np.random.default_rng(seed)
    values = {key: [] for key in (
        "prevalence_difference",
        "prevalence_ratio",
        "M_within_day_auc_difference",
        "MQ_increment_difference",
    )}
    for _ in range(draws):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        sample = base.iloc[indices]
        report = _point_comparison(sample, flag, base_weights[indices])
        for key, value in report.items():
            if value is not None and np.isfinite(value):
                values[key].append(float(value))
    return {
        "clusters": len(groups),
        "draws": draws,
        "valid_draws": {key: len(value) for key, value in values.items()},
        "ci95": {key: _interval(value) for key, value in values.items()},
    }


def feature_directionality(frame: pd.DataFrame, flag: str) -> dict[str, Any]:
    source = _labeled(frame.loc[frame[flag].notna()])
    output: dict[str, Any] = {}
    for side, selected in (("event", source.loc[source[flag].astype(bool)]), ("complement", source.loc[~source[flag].astype(bool)])):
        features = {}
        for feature in MOMENTUM:
            usable = selected.loc[np.isfinite(pd.to_numeric(selected[feature], errors="coerce"))].copy()
            if len(usable) == 0 or len(np.unique(truth(usable[CEILING], 5))) < 2:
                features[feature] = None
                continue
            weights = _episode_day_weights(usable)
            features[feature] = float(
                roc_auc_score(
                    truth(usable[CEILING], 5),
                    usable[feature].to_numpy(float),
                    sample_weight=weights,
                )
            )
        output[side] = features
    return output


def _ci_excludes(ci: list[float] | None, value: float) -> bool:
    return ci is not None and (ci[1] < value or ci[0] > value)


def family_gate(family: dict[str, Any]) -> dict[str, bool]:
    event = family["event"]
    point = family["comparison"]
    ci = family["bootstrap"]["ticker_day"]["ci95"]
    support = event["labeled"] >= 40 and event["positives"] >= 6 and event["positive_days"] >= 3
    ratio = point["prevalence_ratio"]
    ratio_signal = (
        ratio is not None
        and (ratio >= 2.0 or ratio <= 0.5)
        and _ci_excludes(ci.get("prevalence_ratio"), 1.0)
    )
    auc = point["M_within_day_auc_difference"]
    auc_signal = auc is not None and abs(auc) >= 0.10 and _ci_excludes(ci.get("M_within_day_auc_difference"), 0.0)
    increment = point["MQ_increment_difference"]
    increment_signal = increment is not None and abs(increment) >= 0.10 and _ci_excludes(ci.get("MQ_increment_difference"), 0.0)
    return {
        "adequate_support": support,
        "prevalence_heterogeneity": support and ratio_signal,
        "M_rank_heterogeneity": support and auc_signal,
        "momentum_increment_heterogeneity": support and increment_signal,
        "materially_supported": support and (ratio_signal or auc_signal or increment_signal),
    }


def evaluate_heterogeneity(frame: pd.DataFrame, draws: int = 1000) -> dict[str, Any]:
    labeled = _labeled(frame)
    report: dict[str, Any] = {
        "overall": subgroup_report(frame),
        "same_day_8k_count": int(pd.to_numeric(frame.same_day_8k_count, errors="coerce").fillna(0).sum()),
        "same_day_reverse_split_count": int(pd.to_numeric(frame.same_day_reverse_split_count, errors="coerce").fillna(0).sum()),
        "event_source_complete_labeled": int((_labeled(frame).event_sources_complete.astype(bool)).sum()),
        "families": {},
        "event_signatures": {},
    }
    if len(labeled):
        signatures = labeled[list(PRIMARY_FLAGS)].astype("boolean").astype(str).agg("|".join, axis=1)
        report["event_signatures"] = signatures.value_counts().to_dict()

    for i, flag in enumerate(PRIMARY_FLAGS):
        eligible = frame.loc[frame[flag].notna()].copy()
        event = eligible.loc[eligible[flag].astype(bool)]
        complement = eligible.loc[~eligible[flag].astype(bool)]
        base = _labeled(eligible)
        comparison = _point_comparison(base, flag) if len(base) else {
            "prevalence_difference": None,
            "prevalence_ratio": None,
            "M_within_day_auc_difference": None,
            "MQ_increment_difference": None,
        }
        family = {
            "event": subgroup_report(event),
            "complement": subgroup_report(complement),
            "comparison": comparison,
            "bootstrap": {
                "day": bootstrap_comparison(eligible, flag, ["trading_day"], 20265160 + i * 10, draws),
                "ticker_day": bootstrap_comparison(eligible, flag, ["trading_day", "ticker"], 20265161 + i * 10, draws),
            },
            "feature_directionality": feature_directionality(eligible, flag),
        }
        family["gate"] = family_gate(family)
        report["families"][flag] = family

    gates = {flag: data["gate"] for flag, data in report["families"].items()}
    supported = [flag for flag, gate in gates.items() if gate["materially_supported"]]
    adequate = [flag for flag, gate in gates.items() if gate["adequate_support"]]
    if supported:
        status = "materially_supported"
    elif adequate:
        status = "not_supported_by_fixed_gate"
    else:
        status = "underpowered"
    report["interpretation"] = {
        "status": status,
        "supported_families": supported,
        "adequately_supported_families": adequate,
        "specialist_promotion_allowed": False,
        "next_if_supported": "test simple event-context features/interactions before any specialist or MoE routing",
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r311", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states-output", type=Path, required=True)
    parser.add_argument("--events-output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()

    summary_path = args.r311 / "request311.json"
    states_path = args.r311 / "request311-states.parquet"
    summary = json.loads(summary_path.read_text())
    states = pd.read_parquet(states_path)
    validate_frozen_input(states, summary)

    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=EVENT_CACHE_DIR,
        request_interval_seconds=0.02,
    )
    events, acquisition = acquire_events(states, client)
    annotated = annotate_events(states, events, acquisition)
    evaluation = evaluate_heterogeneity(annotated, draws=args.draws)

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "promotion_eligible": False,
        "reference_R311_run": 37223064559,
        "June_HOLD_opened": False,
        "final_July_August_opened": False,
        "model_refit": False,
        "threshold_or_policy_change": False,
        "input_sha256": {
            "request311.json": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "request311-states.parquet": hashlib.sha256(states_path.read_bytes()).hexdigest(),
        },
        "primary_flags": list(PRIMARY_FLAGS),
        "secondary_flags": list(SECONDARY_FLAGS),
        "acquisition": acquisition,
        "audit": evaluation,
        "limitations": (
            "Existing momentum universe excludes same-day split events, so this cohort cannot test the full "
            "reverse-split-day phenomenon. Same-day 8-K timing is not assumed known intraday. no-event means "
            "no event in the three acquired primary families, not no real-world catalyst. Reused May is "
            "development-only. No event taxonomy, model, feature, threshold, stop, cost, date or subgroup "
            "is selected from outcomes."
        ),
    }

    for path in (args.output, args.states_output, args.events_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    annotated.to_parquet(args.states_output, index=False, compression="zstd")
    events.to_parquet(args.events_output, index=False, compression="zstd")
    print(json.dumps({
        "acquisition": acquisition,
        "interpretation": evaluation["interpretation"],
        "same_day_reverse_split_count": evaluation["same_day_reverse_split_count"],
        "families": {key: {
            "event": value["event"],
            "comparison": value["comparison"],
            "gate": value["gate"],
        } for key, value in evaluation["families"].items()},
    }, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
