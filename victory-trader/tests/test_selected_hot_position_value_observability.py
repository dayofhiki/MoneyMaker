from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from victory_trader.selected_hot_position_value_observability import (
    MODEL_FEATURES,
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
