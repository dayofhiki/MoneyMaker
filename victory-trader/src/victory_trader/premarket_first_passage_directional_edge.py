"""Request 200: premarket-context first-passage directional edge.

FIT-only training, chronological calibration-only evaluation. No Request-178
fresh outcomes are opened.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .causal_first_passage_directional_edge import (
    MODEL_SEED,
    QUANTILES,
    add_first_passage,
    classifier_metrics,
)
from .causal_second_risk_compatible_entry import (
    RULES,
    SecondStore,
)
from .future_cost_cover_state_observability import _day_weights
from .market_calendar import regular_session_bounds
from .rich_second_position_value import RICH_SECOND_FEATURES
from .shadow_entry_repricing_decomposition import FEATURES

REQUEST_ID = 200
PREMARKET_START_HOUR_ET = 4
MINUTE_MS = 60_000

PREMARKET_FEATURES = (
    "pm_any_activity",
    "pm_log_active_seconds",
    "pm_log_volume",
    "pm_log_transactions",
    "pm_return_pct",
    "pm_range_pct",
    "pm_last_from_high_pct",
    "pm_close_location",
    "pm_high_age_min",
    "pm_last30_return_pct",
    "pm_last30_volume_share",
    "pm_last30_transactions_share",
    "pm_first_vs_previous_close_pct",
    "pm_last_vs_previous_close_pct",
    "pm_high_vs_previous_close_pct",
    "state_vs_pm_high_pct",
    "state_vs_pm_last_pct",
)

BASE_COLUMNS = tuple(
    column
    for column in FEATURES
    if column not in set(RICH_SECOND_FEATURES)
)

MIN_PRIMARY_AUC = 0.62
MIN_AUC_UPLIFT = 0.03
MIN_DAY_WINS = 6
MIN_P90_PROXY_UPLIFT_PCT = 0.50


@dataclass(frozen=True)
class Head:
    model: HistGradientBoostingClassifier
    columns: tuple[str, ...]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _ratio_pct(numerator: float, denominator: float) -> float:
    if not (
        np.isfinite(numerator)
        and np.isfinite(denominator)
        and denominator > 0
    ):
        return np.nan
    return float((numerator / denominator - 1.0) * 100.0)


def _premarket_day_summary(
    day: str,
    ticker: str,
    store: SecondStore,
) -> dict[str, float]:
    bounds = regular_session_bounds(date.fromisoformat(day))
    if bounds is None:
        raise ValueError(f"request 200 missing session bounds: {day}")
    market_open = bounds[0]
    pm_start = market_open.replace(
        hour=PREMARKET_START_HOUR_ET,
        minute=0,
        second=0,
        microsecond=0,
    )
    start_ms = int(pm_start.timestamp() * 1000)
    open_ms = int(market_open.timestamp() * 1000)

    seconds = store.seconds(day, ticker)
    t = _numeric(seconds["t"])
    pm = seconds.loc[t.ge(start_ms) & t.lt(open_ms)].copy()

    base = {
        "pm_any_activity": 0.0,
        "pm_log_active_seconds": 0.0,
        "pm_log_volume": 0.0,
        "pm_log_transactions": 0.0,
        "pm_return_pct": np.nan,
        "pm_range_pct": np.nan,
        "pm_last_from_high_pct": np.nan,
        "pm_close_location": np.nan,
        "pm_high_age_min": np.nan,
        "pm_last30_return_pct": np.nan,
        "pm_last30_volume_share": np.nan,
        "pm_last30_transactions_share": np.nan,
        "_pm_first": np.nan,
        "_pm_last": np.nan,
        "_pm_high": np.nan,
    }
    if pm.empty:
        return base

    pm = pm.sort_values("t", kind="stable")
    o = _numeric(pm["o"])
    h = _numeric(pm["h"])
    low = _numeric(pm["l"])
    c = _numeric(pm["c"])
    valid = (
        o.gt(0)
        & h.gt(0)
        & low.gt(0)
        & c.gt(0)
        & o.notna()
        & h.notna()
        & low.notna()
        & c.notna()
    )
    pm = pm.loc[valid].copy()
    if pm.empty:
        return base

    o = _numeric(pm["o"])
    h = _numeric(pm["h"])
    low = _numeric(pm["l"])
    c = _numeric(pm["c"])
    volume = (
        _numeric(pm["v"]).fillna(0.0).clip(lower=0.0)
        if "v" in pm.columns
        else pd.Series(0.0, index=pm.index)
    )
    transactions = (
        _numeric(pm["n"]).fillna(0.0).clip(lower=0.0)
        if "n" in pm.columns
        else pd.Series(0.0, index=pm.index)
    )

    first_price = float(o.iloc[0])
    last_price = float(c.iloc[-1])
    high_price = float(h.max())
    low_price = float(low.min())
    high_rows = pm.loc[h.eq(high_price), "t"]
    high_t = int(_numeric(high_rows).max())

    total_volume = float(volume.sum())
    total_transactions = float(transactions.sum())
    last30 = pm.loc[
        _numeric(pm["t"]).ge(open_ms - 30 * MINUTE_MS)
    ].copy()
    if last30.empty:
        last30_return = np.nan
        last30_volume_share = np.nan
        last30_transactions_share = np.nan
    else:
        last30_return = _ratio_pct(
            float(_numeric(last30["c"]).iloc[-1]),
            float(_numeric(last30["o"]).iloc[0]),
        )
        last30_volume = (
            float(_numeric(last30["v"]).fillna(0.0).clip(lower=0.0).sum())
            if "v" in last30.columns
            else 0.0
        )
        last30_transactions = (
            float(_numeric(last30["n"]).fillna(0.0).clip(lower=0.0).sum())
            if "n" in last30.columns
            else 0.0
        )
        last30_volume_share = (
            last30_volume / total_volume
            if total_volume > 0
            else np.nan
        )
        last30_transactions_share = (
            last30_transactions / total_transactions
            if total_transactions > 0
            else np.nan
        )

    close_location = (
        float((last_price - low_price) / (high_price - low_price))
        if high_price > low_price
        else np.nan
    )

    return {
        "pm_any_activity": 1.0,
        "pm_log_active_seconds": float(np.log1p(len(pm))),
        "pm_log_volume": float(np.log1p(total_volume)),
        "pm_log_transactions": float(np.log1p(total_transactions)),
        "pm_return_pct": _ratio_pct(last_price, first_price),
        "pm_range_pct": _ratio_pct(high_price, low_price),
        "pm_last_from_high_pct": _ratio_pct(last_price, high_price),
        "pm_close_location": close_location,
        "pm_high_age_min": float((open_ms - high_t) / MINUTE_MS),
        "pm_last30_return_pct": last30_return,
        "pm_last30_volume_share": last30_volume_share,
        "pm_last30_transactions_share": last30_transactions_share,
        "_pm_first": first_price,
        "_pm_last": last_price,
        "_pm_high": high_price,
    }


def add_premarket_features(
    frame: pd.DataFrame,
    store: SecondStore,
) -> pd.DataFrame:
    result = frame.copy()
    keys = (
        result.loc[:, ["trading_day", "ticker"]]
        .drop_duplicates()
        .sort_values(["trading_day", "ticker"], kind="stable")
    )
    summaries: dict[tuple[str, str], dict[str, float]] = {}
    for row in keys.itertuples(index=False):
        day = str(row.trading_day)
        ticker = str(row.ticker).upper()
        summaries[(day, ticker)] = _premarket_day_summary(
            day,
            ticker,
            store,
        )

    values: list[dict[str, float]] = []
    for row in result.to_dict("records"):
        day = str(row["trading_day"])
        ticker = str(row["ticker"]).upper()
        pm = summaries[(day, ticker)]
        current_log = pd.to_numeric(
            pd.Series([row.get("log_current_close")]),
            errors="coerce",
        ).iloc[0]
        current = (
            float(np.exp(current_log))
            if pd.notna(current_log) and np.isfinite(float(current_log))
            else np.nan
        )
        ret_prev = pd.to_numeric(
            pd.Series([row.get("return_from_previous_close_pct")]),
            errors="coerce",
        ).iloc[0]
        previous_close = (
            current / (1.0 + float(ret_prev) / 100.0)
            if (
                np.isfinite(current)
                and current > 0
                and pd.notna(ret_prev)
                and 1.0 + float(ret_prev) / 100.0 > 0
            )
            else np.nan
        )
        values.append(
            {
                **{
                    feature: float(pm.get(feature, np.nan))
                    for feature in PREMARKET_FEATURES[:12]
                },
                "pm_first_vs_previous_close_pct": _ratio_pct(
                    float(pm["_pm_first"]),
                    previous_close,
                ),
                "pm_last_vs_previous_close_pct": _ratio_pct(
                    float(pm["_pm_last"]),
                    previous_close,
                ),
                "pm_high_vs_previous_close_pct": _ratio_pct(
                    float(pm["_pm_high"]),
                    previous_close,
                ),
                "state_vs_pm_high_pct": _ratio_pct(
                    current,
                    float(pm["_pm_high"]),
                ),
                "state_vs_pm_last_pct": _ratio_pct(
                    current,
                    float(pm["_pm_last"]),
                ),
            }
        )

    features = pd.DataFrame(values, index=result.index)
    for column in PREMARKET_FEATURES:
        result[column] = features[column]
    return result


def _x(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric,
        errors="coerce",
    )


def fit_head(
    frame: pd.DataFrame,
    *,
    include_premarket: bool,
    seed: int,
) -> Head:
    y = pd.to_numeric(frame["take_before_stop"], errors="coerce")
    valid = y.notna()
    train = frame.loc[valid].copy()
    actual = y.loc[valid].astype(int)
    if len(train) < 500 or actual.nunique() < 2:
        raise ValueError(
            "request 200 insufficient first-passage training support"
        )
    columns = tuple(
        column
        for column in (
            [*BASE_COLUMNS, *PREMARKET_FEATURES]
            if include_premarket
            else BASE_COLUMNS
        )
        if column in train.columns
        and pd.to_numeric(train[column], errors="coerce").notna().any()
    )
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=seed,
    )
    model.fit(
        _x(train, columns),
        actual,
        sample_weight=_day_weights(train),
    )
    return Head(model=model, columns=columns)


def predict(frame: pd.DataFrame, head: Head) -> np.ndarray:
    return head.model.predict_proba(_x(frame, head.columns))[:, 1]


def _day_balanced_proxy(frame: pd.DataFrame) -> float | None:
    proxy = pd.to_numeric(frame["barrier_proxy_pct"], errors="coerce")
    valid = proxy.notna()
    if not int(valid.sum()):
        return None
    work = frame.loc[valid].copy()
    work["_proxy"] = proxy.loc[valid]
    daily = work.groupby(
        work["trading_day"].astype(str),
        sort=True,
    )["_proxy"].mean()
    return float(daily.mean()) if len(daily) else None


def top_groups(
    frame: pd.DataFrame,
    score_column: str,
) -> dict[str, object]:
    score = pd.to_numeric(frame[score_column], errors="coerce")
    outputs: dict[str, object] = {}
    for q in QUANTILES:
        cutoff = float(score.dropna().quantile(q))
        part = frame.loc[score.ge(cutoff)].copy()
        status = part["first_passage_status"].astype(str)
        outputs[f"p{int(q * 100)}"] = {
            "score_cutoff": cutoff,
            "states": int(len(part)),
            "episodes": int(
                part[["trading_day", "ticker", "hot_t"]]
                .drop_duplicates()
                .shape[0]
            ),
            "take_first_rate": float(status.eq("take_first").mean()),
            "stop_first_rate": float(status.eq("stop_first").mean()),
            "neither_rate": float(status.eq("neither").mean()),
            "entry_unavailable_rate": float(
                status.eq("entry_unavailable").mean()
            ),
            "barrier_proxy_mean_pct": (
                float(
                    pd.to_numeric(
                        part["barrier_proxy_pct"],
                        errors="coerce",
                    ).mean()
                )
                if len(part)
                else None
            ),
            "day_balanced_barrier_proxy_pct": _day_balanced_proxy(
                part
            ),
        }
    return outputs


def evaluate_rule(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    store: SecondStore,
    *,
    rule_name: str,
    seed_offset: int,
) -> dict[str, object]:
    take_pct = RULES[rule_name].take_pct
    fit_labeled = add_first_passage(
        fit,
        store,
        take_pct=take_pct,
    )
    cal_labeled = add_first_passage(
        calibration,
        store,
        take_pct=take_pct,
    )

    base = fit_head(
        fit_labeled,
        include_premarket=False,
        seed=MODEL_SEED + seed_offset,
    )
    pm = fit_head(
        fit_labeled,
        include_premarket=True,
        seed=MODEL_SEED + 20 + seed_offset,
    )
    cal_labeled["base_score"] = predict(cal_labeled, base)
    cal_labeled["pm_score"] = predict(cal_labeled, pm)

    base_metrics = classifier_metrics(cal_labeled, "base_score")
    pm_metrics = classifier_metrics(cal_labeled, "pm_score")
    base_auc = base_metrics["auc"]
    pm_auc = pm_metrics["auc"]
    by_day_wins = 0
    for day, pm_day in pm_metrics["by_day"].items():
        base_day = base_metrics["by_day"].get(day, {})
        if (
            pm_day.get("auc") is not None
            and base_day.get("auc") is not None
            and float(pm_day["auc"]) > float(base_day["auc"])
        ):
            by_day_wins += 1

    top = top_groups(cal_labeled, "pm_score")
    all_proxy = _day_balanced_proxy(cal_labeled)
    p90_proxy = top["p90"]["day_balanced_barrier_proxy_pct"]
    proxy_uplift = (
        float(p90_proxy) - float(all_proxy)
        if p90_proxy is not None and all_proxy is not None
        else None
    )
    auc_uplift = (
        float(pm_auc) - float(base_auc)
        if pm_auc is not None and base_auc is not None
        else None
    )
    signal_pass = bool(
        pm_auc is not None
        and float(pm_auc) >= MIN_PRIMARY_AUC
        and auc_uplift is not None
        and float(auc_uplift) >= MIN_AUC_UPLIFT
        and by_day_wins >= MIN_DAY_WINS
        and proxy_uplift is not None
        and float(proxy_uplift) >= MIN_P90_PROXY_UPLIFT_PCT
    )

    return {
        "take_pct": float(take_pct),
        "premarket_activity_rate": float(
            pd.to_numeric(
                cal_labeled["pm_any_activity"],
                errors="coerce",
            ).eq(1.0).mean()
        ),
        "base": base_metrics,
        "premarket": pm_metrics,
        "premarket_minus_base_auc": auc_uplift,
        "premarket_auc_day_wins": int(by_day_wins),
        "all_state_day_balanced_barrier_proxy_pct": all_proxy,
        "premarket_top_groups": top,
        "p90_proxy_uplift_pct": proxy_uplift,
        "development_signal_pass": signal_pass,
    }


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    history_scan_path: Path,
    output_path: Path,
) -> int:
    from .causal_first_passage_directional_edge import (
        prepare_directional_states,
    )

    fit_positions = pd.read_parquet(fit_path)
    cal_positions = pd.read_parquet(calibration_path)
    scan = pd.read_parquet(history_scan_path)

    fit_days = set(fit_positions["trading_day"].astype(str))
    cal_days = set(cal_positions["trading_day"].astype(str))
    fit_scan = scan.loc[
        scan["trading_day"].astype(str).isin(fit_days)
    ].copy()
    cal_scan = scan.loc[
        scan["trading_day"].astype(str).isin(cal_days)
    ].copy()

    store = SecondStore()
    fit = prepare_directional_states(
        fit_positions,
        fit_scan,
        store,
    )
    cal = prepare_directional_states(
        cal_positions,
        cal_scan,
        store,
    )
    fit = add_premarket_features(fit, store)
    cal = add_premarket_features(cal, store)

    diagnostics: dict[str, object] = {}
    for i, rule_name in enumerate(RULES):
        diagnostics[rule_name] = evaluate_rule(
            fit,
            cal,
            store,
            rule_name=rule_name,
            seed_offset=i,
        )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "fresh_evaluated": False,
        "promotion_eligible": False,
        "training_partition": "request171_fit_only",
        "evaluation_partition": "request171_calibration_only",
        "premarket_window": "04:00 ET to regular-session open",
        "premarket_features": list(PREMARKET_FEATURES),
        "diagnostics": diagnostics,
        "development_signal_pass": bool(
            any(
                bool(item["development_signal_pass"])
                for item in diagnostics.values()
            )
        ),
        "halt_feed_by_day": dict(sorted(store.halt_status.items())),
        "second_client_stats": store.client.stats.to_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--history-scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fit,
        args.calibration,
        args.history_scan,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
