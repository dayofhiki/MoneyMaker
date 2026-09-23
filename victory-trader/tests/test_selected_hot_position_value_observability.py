from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from victory_trader.execution_costs import DEFAULT_EXECUTION_SCENARIOS, net_round_trip_return_pct
from victory_trader.market_calendar import regular_session_bounds
from victory_trader.second_path_attention_probe import _annotate_scan
from victory_trader.selected_hot_position_value_observability import (
    BASE_SCENARIO,
    MODEL_FEATURES,
    _base_return,
    _selected_position_scan_features,
    _session_clock,
    apply_excess_target,
    build_position_rows,
    fit_minute_baselines,
    predict_hold,
    predict_option,
    request140b_predictive_conditions_pass,
    train_hold_model,
    train_option_model,
)


class FakeSecondClient:
    def second_bars_range(
        self,
        ticker: str,
        start: date,
        end: date,
        *,
        adjusted: bool = False,
    ) -> dict:
        del ticker, start, end, adjusted
        rows = []
        for timestamp in range(120_000, 360_000, 1_000):
            price = 1.0 + (timestamp - 120_000) / 240_000 * 0.2
            rows.append(
                {
                    "t": timestamp,
                    "o": price,
                    "h": price * 1.001,
                    "l": price * 0.999,
                    "c": price,
                    "v": 100.0,
                    "n": 5.0,
                }
            )
        return {"results": rows}


def _scan() -> pd.DataFrame:
    rows = []
    for idx, timestamp in enumerate([180_000, 240_000, 300_000, 360_000]):
        opening = [1.00, 1.05, 1.20, 1.18][idx]
        closing = [1.10, 1.20, 1.18, 1.17][idx]
        rows.append(
            {
                "trading_day": "2026-05-07",
                "ticker": "A",
                "t": timestamp,
                "o": opening,
                "h": max(opening, closing) * 1.01,
                "l": min(opening, closing) * 0.99,
                "c": closing,
                "v": 1000.0 + idx * 100,
                "n": 100.0 + idx * 10,
                "attention_score": 0.9,
                "return_from_previous_close_pct": closing * 10 - 10,
            }
        )
    return pd.DataFrame(rows)


def test_build_position_rows_uses_causal_state_and_future_only_for_labels():
    anchors = pd.DataFrame(
        [{"trading_day": "2026-05-07", "ticker": "A", "t": 120_000}]
    )

    rows, audit = build_position_rows(anchors, _scan(), FakeSecondClient())

    assert len(rows) >= 2
    first = rows.sort_values("minutes_held").iloc[0]
    assert first["minutes_held"] == 1
    assert first["entry_open"] == 1.0
    assert first["exit_reference_open"] == 1.05
    assert first["log_current_close"] == pytest.approx(np.log(1.10))
    assert first["hold_advantage_1m_pct"] > 0
    assert first["remaining_option_value_pct"] > 0
    assert first["active_seconds_60"] > 0
    assert pd.notna(first["attention_rank"])
    assert audit["position_ticker_day_requests"] == 1


def _model_panel(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            column: rng.normal(size=rows)
            for column in MODEL_FEATURES
        }
    )
    frame["minutes_held"] = rng.integers(1, 29, rows).astype(float)
    signal = (
        0.6 * frame["sec_last10_return_pct"]
        - 0.3 * frame["drawdown_from_peak_pct"]
        + 0.2 * frame["entry_to_current_close_pct"]
    )
    frame["hold_advantage_1m_pct"] = signal + rng.normal(0, 0.8, rows)
    frame["remaining_option_value_pct"] = (
        0.8 * frame["sec_accel_30_pct"]
        + 0.3 * frame["recovery_from_trough_pct"]
        + rng.normal(0, 1.0, rows)
    )
    return frame


def test_position_value_models_score_finite_outputs():
    fit = _model_panel(1800, 1)
    calibration = _model_panel(900, 2)
    evaluation = _model_panel(300, 3)

    baselines = fit_minute_baselines(fit)
    fit = apply_excess_target(fit, baselines)
    calibration = apply_excess_target(calibration, baselines)
    evaluation = apply_excess_target(evaluation, baselines)

    hold = train_hold_model(fit, calibration)
    option = train_option_model(fit, calibration)

    hold_score = predict_hold(evaluation, hold)
    option_score = predict_option(evaluation, option)

    assert np.isfinite(hold_score).all()
    assert ((hold_score >= 0) & (hold_score <= 1)).all()
    assert np.isfinite(option_score).all()


def test_request140b_predictive_conditions_allow_coverage_only_diagnostic():
    summary = {
        "evaluation": {
            "classifier_auc": 0.63,
            "auc_above_random_days": 5,
            "value_spearman": 0.27,
            "positive_spearman_days": 5,
            "selected_oracle_base_mean_pct": 1.25,
            "oracle_base_mean_pct": 0.45,
            "selected_positive_rate": 0.53,
            "opportunity_positive_rate": 0.42,
            "selected_mean_nonlower_days": 5,
            "promotion_gate_pass": False,
        }
    }
    assert request140b_predictive_conditions_pass(summary)


def test_position_model_features_exclude_future_execution_reference():
    forbidden = {
        "exit_reference_open",
        "current_open",
        "log_current_open",
        "entry_to_current_open_pct",
        "exit_now_base_return_pct",
        "next_minute_base_return_pct",
        "hold_advantage_1m_pct",
        "best_future_base_return_pct",
        "remaining_option_value_pct",
    }
    assert forbidden.isdisjoint(MODEL_FEATURES)
    assert "log_current_close" in MODEL_FEATURES
    assert "entry_to_current_close_pct" in MODEL_FEATURES


def test_position_base_return_matches_shared_execution_cost_model():
    entry = 2.0
    exit_price = 2.2
    gross = (exit_price / entry - 1.0) * 100.0
    base = next(s for s in DEFAULT_EXECUTION_SCENARIOS if s.name == "base")

    assert BASE_SCENARIO == base
    assert _base_return(entry, exit_price) == pytest.approx(
        net_round_trip_return_pct(entry, gross, base)
    )


def test_selected_position_annotation_matches_full_market_reference():
    day = "2026-05-07"
    rows = []
    for ticker, score_base in [("A", 0.9), ("B", 0.4)]:
        for idx, timestamp in enumerate([180_000, 240_000]):
            close = 1.0 + 0.1 * idx + (0.02 if ticker == "A" else 0.0)
            rows.append(
                {
                    "trading_day": day,
                    "ticker": ticker,
                    "t": timestamp,
                    "o": close - 0.02,
                    "h": close + 0.03,
                    "l": close - 0.03,
                    "c": close,
                    "v": 100.0 + idx * 10,
                    "n": 10.0 + idx,
                    "attention_score": score_base - idx * 0.01,
                    "return_from_previous_close_pct": close * 10.0 - 10.0,
                }
            )
    scan = pd.DataFrame(rows)
    anchors = pd.DataFrame(
        [{"trading_day": day, "ticker": "A", "t": 120_000}]
    )

    reference = _annotate_scan(scan)
    reference["attention_rank"] = reference.groupby(
        ["trading_day", "t"], sort=False
    )["attention_score"].rank(method="first", ascending=False)
    reference = reference.loc[reference["ticker"].eq("A")].sort_values("t")

    optimized = _selected_position_scan_features(scan, anchors).sort_values("t")

    for column in MODEL_FEATURES:
        if column in reference.columns:
            pd.testing.assert_series_equal(
                pd.to_numeric(optimized[column], errors="coerce").reset_index(drop=True),
                pd.to_numeric(reference[column], errors="coerce").reset_index(drop=True),
                check_names=False,
            )


def test_session_clock_respects_nyse_early_close():
    day = "2026-11-27"
    bounds = regular_session_bounds(date.fromisoformat(day))
    assert bounds is not None
    open_ms = int(bounds[0].timestamp() * 1000)
    close_ms = int(bounds[1].timestamp() * 1000)
    session_minutes = (close_ms - open_ms) / 60_000.0
    assert session_minutes < 390.0

    noon = open_ms + 150 * 60_000
    since, remaining = _session_clock(day, noon)

    assert since == pytest.approx(150.0)
    assert remaining == pytest.approx(session_minutes - 150.0)


def test_position_state_is_kept_when_next_exit_reference_is_missing():
    scan = _scan().loc[_scan()["t"].ne(240_000)].copy()
    anchors = pd.DataFrame(
        [{"trading_day": "2026-05-07", "ticker": "A", "t": 120_000}]
    )

    rows, audit = build_position_rows(anchors, scan, FakeSecondClient())
    first = rows.loc[rows["state_t"].eq(180_000)].iloc[0]

    assert audit["expected_wall_clock_state_slots"] > audit["observed_causal_state_slots"]
    assert audit["silent_state_slots"] > 0
    assert bool(first["exit_reference_available"]) is False
    assert pd.isna(first["exit_reference_open"])
    assert pd.isna(first["exit_now_base_return_pct"])
    assert pd.isna(first["hold_advantage_1m_pct"])
    assert pd.notna(first["log_current_close"])
    assert first["active_seconds_60"] > 0
