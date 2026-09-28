"""Request260: execution-aware overlay for the frozen Request258 controller."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .config import load_settings
from .full_hot_fixed_policy_value import attach_fixed_value, event_lookup
from .frozen_minute_policy_second_audit import (
    PRIMARY_LATENCY_MS,
    EXPIRY_MS,
    reconstruct,
    report as second_report,
)
from .learned_pullback_entry import (
    baseline_policy,
    build_episode_states,
    controller_columns,
    controller_x,
    policy_metrics,
)
from .massive_client import MassiveClient
from .minute_state_entry_controller import (
    MODEL_CALIBRATION_DAY,
    MODEL_FIT_DAYS,
    POSITIVE_PROBABILITY_THRESHOLD,
    SEVERE_PROBABILITY_THRESHOLD,
    TEST_DAYS,
    fit_models,
    make_policy as request258_policy,
    minute_state_columns,
    pullback_mask,
    pullback_rows,
)
from .rich_post_hot_state import (
    CURRENT_MINUTE_FEATURES,
    CURRENT_SECOND_FEATURES,
    DELTA_SECOND_FEATURES,
    SECOND_DYNAMIC_FEATURES,
    _second_frame,
    attach_dynamic_deltas,
    attach_minute_features,
    rich_second_features,
    score_and_select_candidates,
    second_path_features,
    usable,
)
from .second_execution_reconstruction import (
    first_observed_open,
    observed_opens,
)

REQUEST_ID = 260
EXECUTION_THRESHOLD = 0.60
MIN_TRAIN_EXECUTION_ROWS = 150
MIN_TEST_EXECUTION_ROWS = 100
MIN_EXECUTION_AUC = 0.65
MIN_ENTRIES = 10
MIN_MINUTE_GAIN = 0.10
MIN_SECOND_ENTRY_COVERAGE = 0.70
MIN_SECOND_RESOLUTION = 0.40
MAX_SECOND_SEVERE = 0.15


def load_second_cache(
    states: pd.DataFrame,
) -> tuple[
    dict[tuple[str, str], pd.DataFrame],
    dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    dict,
]:
    client = MassiveClient(
        load_settings().massive_api_key,
        cache_dir=Path("data/cache/request260-second-bars"),
        request_interval_seconds=0.02,
    )
    frames = {}
    paths = {}
    keys = (
        states.loc[:, ["trading_day", "ticker"]]
        .drop_duplicates()
        .sort_values(["trading_day", "ticker"])
    )
    for row in keys.itertuples(index=False):
        day_text = str(row.trading_day)
        ticker = str(row.ticker).upper()
        d = date.fromisoformat(day_text)
        payload = client.second_bars_range(
            ticker,
            d,
            d,
            adjusted=False,
        )
        frames[(day_text, ticker)] = _second_frame(payload)
        paths[(day_text, ticker)] = observed_opens(payload)
    return frames, paths, client.stats.to_dict()


def attach_causal_second_features(
    states: pd.DataFrame,
    frames: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    records = []
    for row in states.loc[
        :, ["trading_day", "ticker", "state_t"]
    ].to_dict("records"):
        key = (
            str(row["trading_day"]),
            str(row["ticker"]).upper(),
        )
        seconds = frames.get(key, pd.DataFrame())
        features = {}
        features.update(
            second_path_features(seconds, int(row["state_t"]))
        )
        features.update(
            rich_second_features(seconds, int(row["state_t"]))
        )
        records.append(
            {
                f"current_{name}": features.get(name, np.nan)
                for name in SECOND_DYNAMIC_FEATURES
            }
        )
    dynamic = pd.DataFrame(records)
    return pd.concat(
        [states.reset_index(drop=True), dynamic.reset_index(drop=True)],
        axis=1,
    )


def label_execution_support(
    states: pd.DataFrame,
    paths: dict[
        tuple[str, str],
        tuple[np.ndarray, np.ndarray],
    ],
) -> pd.DataFrame:
    result = states.copy()
    labels = []
    for row in result.itertuples(index=False):
        key = (
            str(row.trading_day),
            str(row.ticker).upper(),
        )
        times, opens = paths.get(
            key,
            (
                np.array([], dtype=np.int64),
                np.array([], dtype=float),
            ),
        )
        _, immediate = first_observed_open(
            times,
            opens,
            decision_t=int(row.state_t),
            latency_ms=PRIMARY_LATENCY_MS,
            expiry_ms=EXPIRY_MS,
        )
        continuation_hits = 0
        for minute in range(1, 6):
            anchor = int(row.state_t) + minute * 60_000
            _, ref = first_observed_open(
                times,
                opens,
                decision_t=anchor,
                latency_ms=PRIMARY_LATENCY_MS,
                expiry_ms=EXPIRY_MS,
            )
            continuation_hits += int(ref is not None)
        labels.append(
            {
                "execution_immediate_available": immediate is not None,
                "execution_continuation_hits": continuation_hits,
                "execution_supported": (
                    immediate is not None
                    and continuation_hits >= 3
                ),
            }
        )
    return pd.concat(
        [
            result.reset_index(drop=True),
            pd.DataFrame(labels),
        ],
        axis=1,
    )


def execution_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    baseline = controller_columns(frame)
    return usable(
        frame,
        [
            *baseline,
            *CURRENT_MINUTE_FEATURES,
            *CURRENT_SECOND_FEATURES,
            *DELTA_SECOND_FEATURES,
        ],
    )


def fit_execution_model(
    train: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[HistGradientBoostingClassifier, dict]:
    y = train.execution_supported.astype(int)
    if len(train) < MIN_TRAIN_EXECUTION_ROWS:
        raise ValueError(
            f"Request260 execution support only {len(train)} rows"
        )
    if y.nunique() != 2:
        raise ValueError("Request260 execution label needs both classes")
    model = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=20261380,
    )
    model.fit(controller_x(train, columns), y)
    return model, {
        "rows": int(len(train)),
        "positive_rate": float(y.mean()),
    }


def execution_signal(
    test: pd.DataFrame,
    model: HistGradientBoostingClassifier,
    columns: tuple[str, ...],
) -> dict:
    y = test.execution_supported.astype(int)
    probability = model.predict_proba(
        controller_x(test, columns)
    )[:, 1]
    auc = (
        float(roc_auc_score(y, probability))
        if len(test) >= 20 and y.nunique() == 2
        else None
    )
    admitted = probability >= EXECUTION_THRESHOLD
    return {
        "rows": int(len(test)),
        "positive_rate": float(y.mean()),
        "auc": auc,
        "admitted_rate": float(np.mean(admitted)),
        "admitted_precision": (
            float(y.loc[admitted].mean())
            if int(admitted.sum())
            else None
        ),
        "admitted_recall": (
            float(y.loc[admitted].sum() / y.sum())
            if int(y.sum())
            else None
        ),
    }


def make_overlay_policy(
    states: pd.DataFrame,
    selected: pd.DataFrame,
    economic,
    execution_model: HistGradientBoostingClassifier,
    execution_feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    work = states.copy()
    x_economic = controller_x(work, economic.columns)
    work["predicted_value_pct"] = (
        economic.value.predict(x_economic) + economic.value_offset
    )
    work["positive_probability"] = (
        economic.positive.predict_proba(x_economic)[:, 1]
    )
    work["severe_probability"] = (
        economic.severe.predict_proba(x_economic)[:, 1]
    )
    work["execution_probability"] = (
        execution_model.predict_proba(
            controller_x(work, execution_feature_columns)
        )[:, 1]
    )
    groups = {
        (str(k[0]), str(k[1]).upper(), int(k[2])): g.sort_values("state_t")
        for k, g in work.groupby(
            ["trading_day", "ticker", "hot_t"],
            sort=False,
        )
    }
    rows = []
    for raw in selected.itertuples(index=False):
        key = (
            str(raw.trading_day),
            str(raw.ticker).upper(),
            int(raw.t),
        )
        group = groups.get(key)
        chosen = None
        if group is not None:
            eligible = group.loc[pullback_mask(group)]
            for _, state in eligible.iterrows():
                economic_ok = (
                    float(state.predicted_value_pct) > 0
                    and float(state.positive_probability)
                    >= POSITIVE_PROBABILITY_THRESHOLD
                    and float(state.severe_probability)
                    < SEVERE_PROBABILITY_THRESHOLD
                )
                execution_ok = (
                    float(state.execution_probability)
                    >= EXECUTION_THRESHOLD
                )
                if economic_ok and execution_ok:
                    chosen = state
                    break
        rows.append(
            {
                "trading_day": key[0],
                "ticker": key[1],
                "hot_t": key[2],
                "entered": chosen is not None,
                "entry_t": (
                    int(chosen.state_t) if chosen is not None else None
                ),
                "entry_price": (
                    float(chosen.state_price)
                    if chosen is not None
                    else None
                ),
                "value_pct": (
                    float(chosen.enter_value_pct)
                    if chosen is not None
                    and pd.notna(chosen.enter_value_pct)
                    else np.nan
                ),
                "execution_probability": (
                    float(chosen.execution_probability)
                    if chosen is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate(
    first_hot_path: Path,
    opportunity_scan_path: Path,
    output_path: Path,
    rows_output: Path,
) -> int:
    first_hot = pd.read_parquet(first_hot_path)
    scan = pd.read_parquet(opportunity_scan_path)
    first_hot = first_hot.drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    first_hot = attach_fixed_value(first_hot, scan)
    scored, train_threshold, test_threshold = score_and_select_candidates(
        first_hot
    )
    train_selected = scored.loc[
        scored.trading_day.astype(str).isin(
            [*MODEL_FIT_DAYS, MODEL_CALIBRATION_DAY]
        )
        & scored.candidate_probability.ge(train_threshold)
    ].copy()
    test_selected = scored.loc[
        scored.trading_day.astype(str).isin(TEST_DAYS)
        & scored.candidate_probability.ge(test_threshold)
    ].copy()

    train_states = build_episode_states(train_selected, scan)
    test_states = build_episode_states(test_selected, scan)
    combined = pd.concat(
        [
            train_states.assign(_split="train"),
            test_states.assign(_split="test"),
        ],
        ignore_index=True,
    )
    combined = attach_minute_features(combined, scan)
    frames, paths, api_stats = load_second_cache(combined)
    combined = attach_causal_second_features(combined, frames)
    combined = attach_dynamic_deltas(combined)
    combined = label_execution_support(combined, paths)
    train_states = combined.loc[combined._split.eq("train")].drop(
        columns="_split"
    ).copy()
    test_states = combined.loc[combined._split.eq("test")].drop(
        columns="_split"
    ).copy()

    train_pullback = pullback_rows(train_states)
    test_pullback = pullback_rows(test_states)
    fit_pullback = train_pullback.loc[
        train_pullback.trading_day.astype(str).isin(MODEL_FIT_DAYS)
    ].copy()
    cal_pullback = train_pullback.loc[
        train_pullback.trading_day.astype(str).eq(MODEL_CALIBRATION_DAY)
    ].copy()
    economic_columns = minute_state_columns(fit_pullback)
    economic, economic_support = fit_models(
        fit_pullback,
        cal_pullback,
        economic_columns,
    )

    execution_feature_columns = execution_columns(train_pullback)
    execution_model, execution_support = fit_execution_model(
        train_pullback,
        execution_feature_columns,
    )
    signal = execution_signal(
        test_pullback,
        execution_model,
        execution_feature_columns,
    )

    overlay_rows = make_overlay_policy(
        test_states,
        test_selected,
        economic,
        execution_model,
        execution_feature_columns,
    )
    request258_rows = request258_policy(
        test_states,
        test_selected,
        economic,
    )
    fixed_rows = baseline_policy(
        test_selected,
        scan,
        "pullback_2",
    )
    candidate_count = int(len(test_selected))
    overlay_minute = policy_metrics(overlay_rows, candidate_count)
    request258_minute = policy_metrics(
        request258_rows,
        candidate_count,
    )
    fixed_minute = policy_metrics(fixed_rows, candidate_count)
    overlay_mean = overlay_minute["mean_pct"]
    fixed_mean = fixed_minute["mean_pct"]
    minute_gain = (
        overlay_mean - fixed_mean
        if overlay_mean is not None and fixed_mean is not None
        else None
    )

    second_rows = reconstruct(
        overlay_rows,
        event_lookup(scan),
        paths,
        latency_ms=PRIMARY_LATENCY_MS,
    )
    second = second_report(second_rows)

    checks = {
        "train_execution_rows": (
            execution_support["rows"] >= MIN_TRAIN_EXECUTION_ROWS
        ),
        "test_execution_rows": (
            signal["rows"] >= MIN_TEST_EXECUTION_ROWS
        ),
        "execution_auc": (
            signal["auc"] is not None
            and signal["auc"] >= MIN_EXECUTION_AUC
        ),
        "entries": overlay_minute["entries"] >= MIN_ENTRIES,
        "minute_mean_positive": (
            overlay_mean is not None and overlay_mean > 0
        ),
        "minute_gain_vs_fixed": (
            minute_gain is not None
            and minute_gain >= MIN_MINUTE_GAIN
        ),
        "second_entry_coverage": (
            second["entry_reference_coverage"]
            >= MIN_SECOND_ENTRY_COVERAGE
        ),
        "second_full_resolution": (
            second["full_trade_resolution"]
            >= MIN_SECOND_RESOLUTION
        ),
        "second_resolved_mean_positive": (
            second["resolved_mean_base_pct"] is not None
            and second["resolved_mean_base_pct"] > 0
        ),
        "second_severe_rate": (
            second["resolved_severe_loss_rate_le_minus2"] is not None
            and second["resolved_severe_loss_rate_le_minus2"]
            <= MAX_SECOND_SEVERE
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "economic_policy_source": 258,
        "execution_threshold": EXECUTION_THRESHOLD,
        "train_candidate_threshold": train_threshold,
        "test_candidate_threshold": test_threshold,
        "support": {
            "train_candidate_episodes": int(len(train_selected)),
            "test_candidate_episodes": candidate_count,
            "train_pullback_rows": int(len(train_pullback)),
            "test_pullback_rows": int(len(test_pullback)),
        },
        "economic_model_support": economic_support,
        "execution_model_support": execution_support,
        "execution_signal": signal,
        "minute_policies": {
            "fixed_2pct_pullback": fixed_minute,
            "request258": request258_minute,
            "execution_aware_overlay": overlay_minute,
        },
        "minute_gain_vs_fixed_2pct_pullback_pct": minute_gain,
        "second_reference": second,
        "checks": checks,
        "joint_gate_pass": bool(all(checks.values())),
        "api_stats": api_stats,
        "interpretation": (
            "Development-only. Execution labels and features use historical "
            "one-second aggregates; current features are causal completed-second "
            "summaries. Missing bars are not proof of market-fill impossibility."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    second_rows.to_parquet(rows_output, index=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--first-hot", type=Path, required=True)
    p.add_argument("--opportunity-scan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows-output", type=Path, required=True)
    a = p.parse_args()
    return evaluate(
        a.first_hot,
        a.opportunity_scan,
        a.output,
        a.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
