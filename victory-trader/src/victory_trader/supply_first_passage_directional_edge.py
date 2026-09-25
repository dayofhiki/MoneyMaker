"""Request 201: point-in-time supply-turnover first-passage edge.

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
from .future_cost_cover_state_observability import _day_weights
from .market_calendar import regular_session_bounds
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

REQUEST_ID = 201
MIN_WEIGHTED_COVERAGE = 0.80
SUPPLY_CACHE_DIR = Path("data/cache/massive-rest-request201-supply")

SUPPLY_FEATURES = (
    "supply_log_weighted_shares_prior",
    "supply_log_share_class_shares_prior",
    "supply_log_implied_market_cap_prior",
    "supply_minute_weighted_turnover",
    "supply_minute_share_class_turnover",
    "supply_regular_cum_weighted_turnover",
    "supply_regular_cum_share_class_turnover",
    "supply_share_class_to_weighted_ratio",
)


@dataclass(frozen=True)
class SupplyRecord:
    query_date: str
    success: bool
    weighted_shares: float
    share_class_shares: float
    provider_market_cap: float


@dataclass(frozen=True)
class Head:
    model: HistGradientBoostingClassifier
    columns: tuple[str, ...]


def _positive(value: object) -> float:
    numeric = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]
    if pd.isna(numeric) or not np.isfinite(float(numeric)) or float(numeric) <= 0:
        return np.nan
    return float(numeric)


class SupplyStore:
    def __init__(self) -> None:
        settings = load_settings()
        self.client = MassiveClient(
            settings.massive_api_key,
            cache_dir=SUPPLY_CACHE_DIR,
            request_interval_seconds=0.05,
        )
        self._records: dict[tuple[str, str], SupplyRecord] = {}

    def record(self, day: str, ticker: str) -> SupplyRecord:
        key = (str(day), str(ticker).upper())
        if key in self._records:
            return self._records[key]

        trading_day = date.fromisoformat(str(day))
        query_day = trading_day - timedelta(days=1)
        if query_day >= trading_day:
            raise ValueError("request 201 supply query is not strictly prior")

        success = True
        payload: dict[str, object] = {}
        try:
            response = self.client.ticker_details(
                str(ticker).upper(),
                query_day,
            )
            raw = response.get("results") or {}
            if isinstance(raw, dict):
                payload = raw
            else:
                success = False
        except Exception:
            success = False

        record = SupplyRecord(
            query_date=query_day.isoformat(),
            success=success,
            weighted_shares=_positive(
                payload.get("weighted_shares_outstanding")
            ),
            share_class_shares=_positive(
                payload.get("share_class_shares_outstanding")
            ),
            provider_market_cap=_positive(
                payload.get("market_cap")
            ),
        )
        self._records[key] = record
        return record


class RegularVolumeIndex:
    def __init__(self, seconds: SecondStore) -> None:
        self.seconds = seconds
        self._index: dict[
            tuple[str, str],
            tuple[np.ndarray, np.ndarray],
        ] = {}

    def _build(
        self,
        day: str,
        ticker: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (day, ticker.upper())
        if key in self._index:
            return self._index[key]
        bounds = regular_session_bounds(date.fromisoformat(day))
        if bounds is None:
            raise ValueError(f"request 201 missing session bounds: {day}")
        open_ms = int(bounds[0].timestamp() * 1000)
        seconds = self.seconds.seconds(day, ticker)
        t = pd.to_numeric(seconds["t"], errors="coerce")
        volume = (
            pd.to_numeric(seconds["v"], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
            if "v" in seconds.columns
            else pd.Series(0.0, index=seconds.index)
        )
        valid = t.notna() & t.ge(open_ms)
        times = t.loc[valid].astype("int64").to_numpy()
        vols = volume.loc[valid].to_numpy(dtype=float)
        order = np.argsort(times, kind="stable")
        times = times[order]
        cumulative = np.cumsum(vols[order], dtype=float)
        self._index[key] = (times, cumulative)
        return times, cumulative

    def volume_before(
        self,
        day: str,
        ticker: str,
        state_t: int,
    ) -> float:
        times, cumulative = self._build(day, ticker)
        if len(times) == 0:
            return 0.0
        idx = int(np.searchsorted(times, int(state_t), side="left")) - 1
        if idx < 0:
            return 0.0
        return float(cumulative[idx])


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not (
        np.isfinite(numerator)
        and np.isfinite(denominator)
        and denominator > 0
    ):
        return np.nan
    return float(numerator / denominator)


def add_supply_features(
    frame: pd.DataFrame,
    supply: SupplyStore,
    volume_index: RegularVolumeIndex,
) -> pd.DataFrame:
    result = frame.copy()
    rows: list[dict[str, object]] = []
    for row in result.to_dict("records"):
        day = str(row["trading_day"])
        ticker = str(row["ticker"]).upper()
        record = supply.record(day, ticker)

        current_log = pd.to_numeric(
            pd.Series([row.get("log_current_close")]),
            errors="coerce",
        ).iloc[0]
        current_close = (
            float(np.exp(current_log))
            if pd.notna(current_log) and np.isfinite(float(current_log))
            else np.nan
        )
        ret_prev = pd.to_numeric(
            pd.Series([row.get("return_from_previous_close_pct")]),
            errors="coerce",
        ).iloc[0]
        previous_close = (
            current_close / (1.0 + float(ret_prev) / 100.0)
            if (
                np.isfinite(current_close)
                and current_close > 0
                and pd.notna(ret_prev)
                and 1.0 + float(ret_prev) / 100.0 > 0
            )
            else np.nan
        )

        log_minute_volume = pd.to_numeric(
            pd.Series([row.get("log_minute_volume")]),
            errors="coerce",
        ).iloc[0]
        minute_volume = (
            float(np.expm1(float(log_minute_volume)))
            if pd.notna(log_minute_volume)
            and np.isfinite(float(log_minute_volume))
            else np.nan
        )
        regular_volume = volume_index.volume_before(
            day,
            ticker,
            int(row["state_t"]),
        )

        weighted = float(record.weighted_shares)
        share_class = float(record.share_class_shares)
        implied_market_cap = (
            weighted * previous_close
            if (
                np.isfinite(weighted)
                and weighted > 0
                and np.isfinite(previous_close)
                and previous_close > 0
            )
            else np.nan
        )

        rows.append(
            {
                "supply_query_date": record.query_date,
                "supply_query_success": float(record.success),
                "supply_weighted_shares_prior": weighted,
                "supply_share_class_shares_prior": share_class,
                "supply_provider_market_cap_prior": float(
                    record.provider_market_cap
                ),
                "supply_log_weighted_shares_prior": (
                    float(np.log(weighted))
                    if np.isfinite(weighted) and weighted > 0
                    else np.nan
                ),
                "supply_log_share_class_shares_prior": (
                    float(np.log(share_class))
                    if np.isfinite(share_class) and share_class > 0
                    else np.nan
                ),
                "supply_log_implied_market_cap_prior": (
                    float(np.log(implied_market_cap))
                    if (
                        np.isfinite(implied_market_cap)
                        and implied_market_cap > 0
                    )
                    else np.nan
                ),
                "supply_minute_weighted_turnover": _safe_ratio(
                    minute_volume,
                    weighted,
                ),
                "supply_minute_share_class_turnover": _safe_ratio(
                    minute_volume,
                    share_class,
                ),
                "supply_regular_cum_weighted_turnover": _safe_ratio(
                    regular_volume,
                    weighted,
                ),
                "supply_regular_cum_share_class_turnover": _safe_ratio(
                    regular_volume,
                    share_class,
                ),
                "supply_share_class_to_weighted_ratio": _safe_ratio(
                    share_class,
                    weighted,
                ),
            }
        )

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
    include_supply: bool,
    seed: int,
) -> Head:
    y = pd.to_numeric(frame["take_before_stop"], errors="coerce")
    valid = y.notna()
    train = frame.loc[valid].copy()
    actual = y.loc[valid].astype(int)
    if len(train) < 500 or actual.nunique() < 2:
        raise ValueError(
            "request 201 insufficient first-passage training support"
        )
    candidate_columns = (
        [*BASE_COLUMNS, *SUPPLY_FEATURES]
        if include_supply
        else list(BASE_COLUMNS)
    )
    columns = tuple(
        column
        for column in candidate_columns
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


def coverage(frame: pd.DataFrame) -> dict[str, object]:
    query_date = pd.to_datetime(
        frame["supply_query_date"],
        errors="coerce",
    )
    trading_day = pd.to_datetime(
        frame["trading_day"],
        errors="coerce",
    )
    weighted = pd.to_numeric(
        frame["supply_weighted_shares_prior"],
        errors="coerce",
    )
    share_class = pd.to_numeric(
        frame["supply_share_class_shares_prior"],
        errors="coerce",
    )
    success = pd.to_numeric(
        frame["supply_query_success"],
        errors="coerce",
    )
    return {
        "rows": int(len(frame)),
        "request_success": float(success.eq(1.0).mean()),
        "weighted_shares_coverage": float(weighted.notna().mean()),
        "share_class_coverage": float(share_class.notna().mean()),
        "strict_prior_query_dates": bool(
            query_date.notna().all()
            and trading_day.notna().all()
            and query_date.lt(trading_day).all()
        ),
    }


def evaluate_rule(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    rule_name: str,
    seed_offset: int,
) -> dict[str, object]:
    take_pct = RULES[rule_name].take_pct
    fit_labeled = add_first_passage(
        fit,
        calibration.attrs["second_store"],
        take_pct=take_pct,
    )
    cal_labeled = add_first_passage(
        calibration,
        calibration.attrs["second_store"],
        take_pct=take_pct,
    )

    base = fit_head(
        fit_labeled,
        include_supply=False,
        seed=MODEL_SEED + seed_offset,
    )
    supply_head = fit_head(
        fit_labeled,
        include_supply=True,
        seed=MODEL_SEED + 30 + seed_offset,
    )
    cal_labeled["base_score"] = predict(cal_labeled, base)
    cal_labeled["supply_score"] = predict(
        cal_labeled,
        supply_head,
    )

    base_metrics = classifier_metrics(
        cal_labeled,
        "base_score",
    )
    supply_metrics = classifier_metrics(
        cal_labeled,
        "supply_score",
    )
    base_auc = base_metrics["auc"]
    supply_auc = supply_metrics["auc"]
    auc_uplift = (
        float(supply_auc) - float(base_auc)
        if base_auc is not None and supply_auc is not None
        else None
    )
    day_wins = 0
    for day, supply_day in supply_metrics["by_day"].items():
        base_day = base_metrics["by_day"].get(day, {})
        if (
            supply_day.get("auc") is not None
            and base_day.get("auc") is not None
            and float(supply_day["auc"]) > float(base_day["auc"])
        ):
            day_wins += 1

    top = top_groups(cal_labeled, "supply_score")
    all_proxy = _day_balanced_proxy(cal_labeled)
    p90_proxy = top["p90"]["day_balanced_barrier_proxy_pct"]
    proxy_uplift = (
        float(p90_proxy) - float(all_proxy)
        if p90_proxy is not None and all_proxy is not None
        else None
    )
    cov = coverage(cal_labeled)
    signal_pass = bool(
        supply_auc is not None
        and float(supply_auc) >= MIN_PRIMARY_AUC
        and auc_uplift is not None
        and float(auc_uplift) >= MIN_AUC_UPLIFT
        and day_wins >= MIN_DAY_WINS
        and proxy_uplift is not None
        and float(proxy_uplift) >= MIN_P90_PROXY_UPLIFT_PCT
        and float(cov["weighted_shares_coverage"])
        >= MIN_WEIGHTED_COVERAGE
        and bool(cov["strict_prior_query_dates"])
    )

    return {
        "take_pct": float(take_pct),
        "base": base_metrics,
        "supply": supply_metrics,
        "supply_minus_base_auc": auc_uplift,
        "supply_auc_day_wins": int(day_wins),
        "all_state_day_balanced_barrier_proxy_pct": all_proxy,
        "supply_top_groups": top,
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

    fit_days = set(fit_positions["trading_day"].astype(str))
    cal_days = set(cal_positions["trading_day"].astype(str))
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

    supply_store = SupplyStore()
    volume_index = RegularVolumeIndex(second_store)
    fit = add_supply_features(
        fit,
        supply_store,
        volume_index,
    )
    cal = add_supply_features(
        cal,
        supply_store,
        volume_index,
    )

    # Preserve the exact causal second store for first-passage labels without
    # threading another mutable service through every helper.
    fit.attrs["second_store"] = second_store
    cal.attrs["second_store"] = second_store

    diagnostics: dict[str, object] = {}
    for i, rule_name in enumerate(RULES):
        diagnostics[rule_name] = evaluate_rule(
            fit,
            cal,
            rule_name=rule_name,
            seed_offset=i,
        )

    fit_coverage = coverage(fit)
    cal_coverage = coverage(cal)
    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "fresh_evaluated": False,
        "promotion_eligible": False,
        "training_partition": "request171_fit_only",
        "evaluation_partition": "request171_calibration_only",
        "supply_features": list(SUPPLY_FEATURES),
        "fit_supply_coverage": fit_coverage,
        "calibration_supply_coverage": cal_coverage,
        "diagnostics": diagnostics,
        "development_signal_pass": bool(
            any(
                bool(item["development_signal_pass"])
                for item in diagnostics.values()
            )
        ),
        "second_client_stats": second_store.client.stats.to_dict(),
        "supply_client_stats": supply_store.client.stats.to_dict(),
        "halt_feed_by_day": dict(
            sorted(second_store.halt_status.items())
        ),
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
