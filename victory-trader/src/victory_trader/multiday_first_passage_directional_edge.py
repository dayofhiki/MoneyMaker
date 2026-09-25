"""Request 202: strictly-prior multi-day first-passage directional edge.

FIT-only training, chronological calibration-only evaluation. Request-178 stays
sealed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .causal_first_passage_directional_edge import (
    MODEL_SEED,
    add_first_passage,
    classifier_metrics,
    prepare_directional_states,
)
from .causal_second_risk_compatible_entry import RULES, SecondStore
from .config import load_settings
from .features import timestamp_et
from .future_cost_cover_state_observability import _day_weights
from .market_data import bars_from_massive_payload
from .massive_client import MassiveClient
from .premarket_first_passage_directional_edge import (
    BASE_COLUMNS,
    MIN_AUC_UPLIFT,
    MIN_DAY_WINS,
    MIN_P90_PROXY_UPLIFT_PCT,
    MIN_PRIMARY_AUC,
    _day_balanced_proxy,
    top_groups,
)

REQUEST_ID = 202
HISTORY_CALENDAR_LOOKBACK_DAYS = 45
MIN_HISTORY_SESSIONS = 5
FULL_HISTORY_SESSIONS = 20
MIN_SUPPORT_RATE = 0.90
HISTORY_CACHE_DIR = Path("data/cache/massive-rest-request202-daily")

MULTIDAY_FEATURES = (
    "hist_log_avg_volume_5d",
    "hist_log_avg_volume_20d",
    "hist_volume_ratio_5d_20d",
    "hist_log_avg_dollar_volume_5d",
    "hist_log_avg_dollar_volume_20d",
    "hist_realized_vol_5d_pct",
    "hist_realized_vol_20d_pct",
    "hist_avg_intraday_range_5d_pct",
    "hist_avg_intraday_range_20d_pct",
    "hist_runner_days_20d",
    "hist_big_down_days_20d",
    "hist_prior_close_vs_20d_high_pct",
    "hist_prior_close_vs_20d_low_pct",
    "state_vs_20d_high_pct",
    "state_vs_20d_low_pct",
    "hist_days_since_last_runner",
)


@dataclass(frozen=True)
class HistoryRecord:
    query_success: bool
    max_history_day: str | None
    sessions: int
    values: dict[str, float]


@dataclass(frozen=True)
class Head:
    model: HistGradientBoostingClassifier
    columns: tuple[str, ...]


def _ratio_pct(numerator: float, denominator: float) -> float:
    if not (
        np.isfinite(numerator)
        and np.isfinite(denominator)
        and denominator > 0
    ):
        return np.nan
    return float((numerator / denominator - 1.0) * 100.0)


def _positive_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.loc[numeric.ge(0)]
    if numeric.empty:
        return np.nan
    return float(numeric.mean())


class DailyHistoryStore:
    def __init__(self) -> None:
        settings = load_settings()
        self.client = MassiveClient(
            settings.massive_api_key,
            cache_dir=HISTORY_CACHE_DIR,
            request_interval_seconds=0.05,
        )
        self._records: dict[tuple[str, str], HistoryRecord] = {}

    def record(self, day: str, ticker: str) -> HistoryRecord:
        key = (str(day), str(ticker).upper())
        if key in self._records:
            return self._records[key]

        target_day = date.fromisoformat(str(day))
        start = target_day - timedelta(
            days=HISTORY_CALENDAR_LOOKBACK_DAYS
        )
        end = target_day - timedelta(days=1)
        success = True
        try:
            payload = self.client.daily_bars(
                str(ticker).upper(),
                start,
                end,
                adjusted=False,
            )
            bars = bars_from_massive_payload(payload)
        except Exception:
            success = False
            bars = pd.DataFrame(
                columns=["t", "o", "h", "l", "c", "v", "vw", "n"]
            )

        if not bars.empty:
            bars = bars.copy()
            bars["_day"] = bars["t"].map(
                lambda value: timestamp_et(int(value)).date()
            )
            bars = bars.loc[bars["_day"].lt(target_day)].copy()
            bars = (
                bars.sort_values("t", kind="stable")
                .drop_duplicates("_day", keep="last")
                .tail(FULL_HISTORY_SESSIONS + 1)
                .reset_index(drop=True)
            )

        sessions = min(int(len(bars)), FULL_HISTORY_SESSIONS)
        max_history_day = (
            str(bars["_day"].max())
            if not bars.empty
            else None
        )
        values = {
            feature: np.nan
            for feature in MULTIDAY_FEATURES
        }
        if sessions >= MIN_HISTORY_SESSIONS:
            close = pd.to_numeric(bars["c"], errors="coerce")
            high = pd.to_numeric(bars["h"], errors="coerce")
            low = pd.to_numeric(bars["l"], errors="coerce")
            volume = (
                pd.to_numeric(bars["v"], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
            )
            daily_return = close.pct_change() * 100.0
            usable = bars.iloc[-FULL_HISTORY_SESSIONS:].copy()
            usable_close = pd.to_numeric(
                usable["c"],
                errors="coerce",
            )
            usable_high = pd.to_numeric(
                usable["h"],
                errors="coerce",
            )
            usable_low = pd.to_numeric(
                usable["l"],
                errors="coerce",
            )
            usable_volume = (
                pd.to_numeric(usable["v"], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
            )
            usable_returns = daily_return.iloc[
                -len(usable):
            ].reset_index(drop=True)
            five = usable.iloc[-5:].copy()
            five_close = pd.to_numeric(
                five["c"],
                errors="coerce",
            )
            five_high = pd.to_numeric(
                five["h"],
                errors="coerce",
            )
            five_low = pd.to_numeric(
                five["l"],
                errors="coerce",
            )
            five_volume = (
                pd.to_numeric(five["v"], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
            )
            five_returns = usable_returns.iloc[
                -len(five):
            ].reset_index(drop=True)

            avg_vol_5 = _positive_mean(five_volume)
            avg_vol_20 = _positive_mean(usable_volume)
            avg_dv_5 = _positive_mean(five_close * five_volume)
            avg_dv_20 = _positive_mean(
                usable_close * usable_volume
            )

            range5 = (
                (five_high - five_low)
                / five_close.where(five_close.gt(0))
                * 100.0
            )
            range20 = (
                (usable_high - usable_low)
                / usable_close.where(usable_close.gt(0))
                * 100.0
            )
            high20 = float(usable_high.max())
            low20 = float(usable_low.min())
            prior_close = float(usable_close.iloc[-1])

            finite5 = five_returns.replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna()
            finite20 = usable_returns.replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna()
            runner_mask = finite20.ge(10.0)
            down_mask = finite20.le(-10.0)

            days_since_runner = np.nan
            runner_positions = np.flatnonzero(
                usable_returns.ge(10.0)
                .fillna(False)
                .to_numpy(dtype=bool)
            )
            if len(runner_positions):
                days_since_runner = float(
                    len(usable_returns)
                    - 1
                    - int(runner_positions[-1])
                )

            values.update(
                {
                    "hist_log_avg_volume_5d": (
                        float(np.log1p(avg_vol_5))
                        if np.isfinite(avg_vol_5)
                        else np.nan
                    ),
                    "hist_log_avg_volume_20d": (
                        float(np.log1p(avg_vol_20))
                        if np.isfinite(avg_vol_20)
                        else np.nan
                    ),
                    "hist_volume_ratio_5d_20d": (
                        float(avg_vol_5 / avg_vol_20)
                        if (
                            np.isfinite(avg_vol_5)
                            and np.isfinite(avg_vol_20)
                            and avg_vol_20 > 0
                        )
                        else np.nan
                    ),
                    "hist_log_avg_dollar_volume_5d": (
                        float(np.log1p(avg_dv_5))
                        if np.isfinite(avg_dv_5)
                        else np.nan
                    ),
                    "hist_log_avg_dollar_volume_20d": (
                        float(np.log1p(avg_dv_20))
                        if np.isfinite(avg_dv_20)
                        else np.nan
                    ),
                    "hist_realized_vol_5d_pct": (
                        float(finite5.std(ddof=0))
                        if len(finite5) >= 2
                        else np.nan
                    ),
                    "hist_realized_vol_20d_pct": (
                        float(finite20.std(ddof=0))
                        if len(finite20) >= 2
                        else np.nan
                    ),
                    "hist_avg_intraday_range_5d_pct": (
                        float(range5.replace(
                            [np.inf, -np.inf],
                            np.nan,
                        ).mean())
                    ),
                    "hist_avg_intraday_range_20d_pct": (
                        float(range20.replace(
                            [np.inf, -np.inf],
                            np.nan,
                        ).mean())
                    ),
                    "hist_runner_days_20d": float(
                        runner_mask.sum()
                    ),
                    "hist_big_down_days_20d": float(
                        down_mask.sum()
                    ),
                    "hist_prior_close_vs_20d_high_pct": (
                        _ratio_pct(prior_close, high20)
                    ),
                    "hist_prior_close_vs_20d_low_pct": (
                        _ratio_pct(prior_close, low20)
                    ),
                    "hist_days_since_last_runner": (
                        days_since_runner
                    ),
                    "_hist_high20": high20,
                    "_hist_low20": low20,
                }
            )

        record = HistoryRecord(
            query_success=success,
            max_history_day=max_history_day,
            sessions=sessions,
            values=values,
        )
        self._records[key] = record
        return record


def add_multiday_features(
    frame: pd.DataFrame,
    history: DailyHistoryStore,
) -> pd.DataFrame:
    result = frame.copy()
    rows: list[dict[str, object]] = []
    for row in result.to_dict("records"):
        day = str(row["trading_day"])
        ticker = str(row["ticker"]).upper()
        record = history.record(day, ticker)

        current_log = pd.to_numeric(
            pd.Series([row.get("log_current_close")]),
            errors="coerce",
        ).iloc[0]
        current = (
            float(np.exp(float(current_log)))
            if pd.notna(current_log)
            and np.isfinite(float(current_log))
            else np.nan
        )
        high20 = float(
            record.values.get("_hist_high20", np.nan)
        )
        low20 = float(
            record.values.get("_hist_low20", np.nan)
        )
        values = {
            feature: float(
                record.values.get(feature, np.nan)
            )
            for feature in MULTIDAY_FEATURES
            if feature not in (
                "state_vs_20d_high_pct",
                "state_vs_20d_low_pct",
            )
        }
        values.update(
            {
                "state_vs_20d_high_pct": _ratio_pct(
                    current,
                    high20,
                ),
                "state_vs_20d_low_pct": _ratio_pct(
                    current,
                    low20,
                ),
                "history_query_success": float(
                    record.query_success
                ),
                "history_sessions": float(record.sessions),
                "history_max_day": record.max_history_day,
            }
        )
        rows.append(values)

    enriched = pd.DataFrame(rows, index=result.index)
    for column in enriched.columns:
        result[column] = enriched[column]
    return result


def _x(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric,
        errors="coerce",
    )


def fit_head(
    frame: pd.DataFrame,
    *,
    include_multiday: bool,
    seed: int,
) -> Head:
    y = pd.to_numeric(
        frame["take_before_stop"],
        errors="coerce",
    )
    valid = y.notna()
    train = frame.loc[valid].copy()
    actual = y.loc[valid].astype(int)
    if len(train) < 500 or actual.nunique() < 2:
        raise ValueError(
            "request 202 insufficient first-passage training support"
        )
    candidate_columns = (
        [*BASE_COLUMNS, *MULTIDAY_FEATURES]
        if include_multiday
        else list(BASE_COLUMNS)
    )
    columns = tuple(
        column
        for column in candidate_columns
        if column in train.columns
        and pd.to_numeric(
            train[column],
            errors="coerce",
        ).notna().any()
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


def predict(
    frame: pd.DataFrame,
    head: Head,
) -> np.ndarray:
    return head.model.predict_proba(
        _x(frame, head.columns)
    )[:, 1]


def coverage(frame: pd.DataFrame) -> dict[str, object]:
    success = pd.to_numeric(
        frame["history_query_success"],
        errors="coerce",
    )
    sessions = pd.to_numeric(
        frame["history_sessions"],
        errors="coerce",
    )
    max_day = pd.to_datetime(
        frame["history_max_day"],
        errors="coerce",
    )
    trading_day = pd.to_datetime(
        frame["trading_day"],
        errors="coerce",
    )
    populated = max_day.notna()
    return {
        "rows": int(len(frame)),
        "request_success": float(
            success.eq(1.0).mean()
        ),
        "support_ge_5_sessions": float(
            sessions.ge(MIN_HISTORY_SESSIONS).mean()
        ),
        "support_ge_20_sessions": float(
            sessions.ge(FULL_HISTORY_SESSIONS).mean()
        ),
        "strict_prior_max_history_day": bool(
            (
                max_day.loc[populated]
                < trading_day.loc[populated]
            ).all()
        ),
    }


def evaluate_rule(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    second_store: SecondStore,
    *,
    rule_name: str,
    seed_offset: int,
) -> dict[str, object]:
    take_pct = RULES[rule_name].take_pct
    fit_labeled = add_first_passage(
        fit,
        second_store,
        take_pct=take_pct,
    )
    cal_labeled = add_first_passage(
        calibration,
        second_store,
        take_pct=take_pct,
    )

    base = fit_head(
        fit_labeled,
        include_multiday=False,
        seed=MODEL_SEED + seed_offset,
    )
    multiday = fit_head(
        fit_labeled,
        include_multiday=True,
        seed=MODEL_SEED + 40 + seed_offset,
    )
    cal_labeled["base_score"] = predict(
        cal_labeled,
        base,
    )
    cal_labeled["multiday_score"] = predict(
        cal_labeled,
        multiday,
    )

    base_metrics = classifier_metrics(
        cal_labeled,
        "base_score",
    )
    multi_metrics = classifier_metrics(
        cal_labeled,
        "multiday_score",
    )
    base_auc = base_metrics["auc"]
    multi_auc = multi_metrics["auc"]
    auc_uplift = (
        float(multi_auc) - float(base_auc)
        if base_auc is not None and multi_auc is not None
        else None
    )

    day_wins = 0
    for day, multi_day in multi_metrics["by_day"].items():
        base_day = base_metrics["by_day"].get(day, {})
        if (
            multi_day.get("auc") is not None
            and base_day.get("auc") is not None
            and float(multi_day["auc"])
            > float(base_day["auc"])
        ):
            day_wins += 1

    top = top_groups(
        cal_labeled,
        "multiday_score",
    )
    all_proxy = _day_balanced_proxy(cal_labeled)
    p90_proxy = top["p90"][
        "day_balanced_barrier_proxy_pct"
    ]
    proxy_uplift = (
        float(p90_proxy) - float(all_proxy)
        if p90_proxy is not None
        and all_proxy is not None
        else None
    )
    cov = coverage(cal_labeled)
    signal_pass = bool(
        multi_auc is not None
        and float(multi_auc) >= MIN_PRIMARY_AUC
        and auc_uplift is not None
        and float(auc_uplift) >= MIN_AUC_UPLIFT
        and day_wins >= MIN_DAY_WINS
        and proxy_uplift is not None
        and float(proxy_uplift)
        >= MIN_P90_PROXY_UPLIFT_PCT
        and float(cov["support_ge_5_sessions"])
        >= MIN_SUPPORT_RATE
        and bool(cov["strict_prior_max_history_day"])
    )
    return {
        "take_pct": float(take_pct),
        "base": base_metrics,
        "multiday": multi_metrics,
        "multiday_minus_base_auc": auc_uplift,
        "multiday_auc_day_wins": int(day_wins),
        "all_state_day_balanced_barrier_proxy_pct": (
            all_proxy
        ),
        "multiday_top_groups": top,
        "p90_proxy_uplift_pct": proxy_uplift,
        "development_signal_pass": signal_pass,
    }


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    history_scan_path: Path,
    output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_path)
    cal_positions = pd.read_parquet(calibration_path)
    scan = pd.read_parquet(history_scan_path)

    fit_days = set(
        fit_positions["trading_day"].astype(str)
    )
    cal_days = set(
        cal_positions["trading_day"].astype(str)
    )
    fit_scan = scan.loc[
        scan["trading_day"].astype(str).isin(fit_days)
    ].copy()
    cal_scan = scan.loc[
        scan["trading_day"].astype(str).isin(cal_days)
    ].copy()

    second_store = SecondStore()
    fit = prepare_directional_states(
        fit_positions,
        fit_scan,
        second_store,
    )
    cal = prepare_directional_states(
        cal_positions,
        cal_scan,
        second_store,
    )

    history = DailyHistoryStore()
    fit = add_multiday_features(fit, history)
    cal = add_multiday_features(cal, history)

    diagnostics: dict[str, object] = {}
    for i, rule_name in enumerate(RULES):
        diagnostics[rule_name] = evaluate_rule(
            fit,
            cal,
            second_store,
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
        "evaluation_partition": (
            "request171_calibration_only"
        ),
        "history_lookback_sessions": (
            FULL_HISTORY_SESSIONS
        ),
        "multiday_features": list(
            MULTIDAY_FEATURES
        ),
        "fit_history_coverage": coverage(fit),
        "calibration_history_coverage": coverage(cal),
        "diagnostics": diagnostics,
        "development_signal_pass": bool(
            any(
                bool(item["development_signal_pass"])
                for item in diagnostics.values()
            )
        ),
        "second_client_stats": (
            second_store.client.stats.to_dict()
        ),
        "history_client_stats": (
            history.client.stats.to_dict()
        ),
        "halt_feed_by_day": dict(
            sorted(second_store.halt_status.items())
        ),
    }
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--history-scan",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return evaluate(
        args.fit,
        args.calibration,
        args.history_scan,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
