import numpy as np
import pandas as pd

from victory_trader.expanded_final_scale_hurdle_ev import (
    FinalScaleCalibration,
    apply_final_scale,
    select_policies,
)


MINUTE_MS = 60_000


def _scored_row(
    *,
    raw_ev,
    calibrated_ev=np.nan,
    base=1.0,
    gross=2.0,
    stress=0.0,
):
    return {
        "trading_day": "2026-02-02",
        "ticker": "AAA",
        "t": 0 * MINUTE_MS,
        "c": 5.0,
        "previous_close": 4.0,
        "active_minute_fraction_15m": 1.0,
        "entry_price": 5.0,
        "predicted_win_probability": 0.4,
        "predicted_win_magnitude_pct": 4.0,
        "predicted_loss_magnitude_pct": 1.0,
        "raw_hurdle_ev_pct": raw_ev,
        "calibrated_hurdle_ev_pct": calibrated_ev,
        "buy_return_15m_pct": gross,
        "buy_return_15m_base_net_return_pct": base,
        "buy_return_15m_stress_net_return_pct": stress,
    }


def test_positive_affine_scale_moves_zero_crossing_without_reordering():
    scored = pd.DataFrame(
        [
            _scored_row(raw_ev=-1.0),
            {**_scored_row(raw_ev=0.0), "ticker": "BBB"},
            {**_scored_row(raw_ev=1.0), "ticker": "CCC"},
        ]
    )
    calibration = FinalScaleCalibration(
        intercept=-0.5,
        slope=2.0,
        rows=500,
        days=8,
        valid=True,
    )

    calibrated = apply_final_scale(scored, calibration)

    assert calibrated["calibrated_hurdle_ev_pct"].tolist() == [
        -2.5,
        -0.5,
        1.5,
    ]
    assert (
        calibrated["raw_hurdle_ev_pct"].rank().tolist()
        == calibrated["calibrated_hurdle_ev_pct"].rank().tolist()
    )


def test_primary_uses_calibrated_zero_not_raw_zero():
    scored = pd.DataFrame(
        [
            _scored_row(raw_ev=0.2),
            {
                **_scored_row(raw_ev=1.0),
                "ticker": "BBB",
            },
        ]
    )
    calibration = FinalScaleCalibration(
        intercept=-0.5,
        slope=1.0,
        rows=500,
        days=8,
        valid=True,
    )
    scored = apply_final_scale(scored, calibration)

    trades, attempts, _ = select_policies(scored, calibration)

    raw_attempts = attempts.loc[
        attempts["policy"].eq("raw_hurdle_ev_15m_cap1")
    ]
    primary = trades.loc[
        trades["policy"].eq("calibrated_hurdle_ev_15m_cap1")
    ]

    assert len(raw_attempts) == 2
    assert len(primary) == 1
    assert primary.iloc[0]["ticker"] == "BBB"


def test_invalid_final_scale_disables_primary():
    scored = pd.DataFrame([_scored_row(raw_ev=1.0)])
    calibration = FinalScaleCalibration(
        intercept=0.0,
        slope=-0.2,
        rows=500,
        days=8,
        valid=False,
    )
    scored = apply_final_scale(scored, calibration)

    trades, attempts, _ = select_policies(scored, calibration)

    assert attempts.loc[
        attempts["policy"].eq("calibrated_hurdle_ev_15m_cap1")
    ].empty
    assert trades.loc[
        trades["policy"].eq("calibrated_hurdle_ev_15m_cap1")
    ].empty


def test_missing_outcome_consumes_calibrated_attempt():
    scored = pd.DataFrame(
        [
            _scored_row(
                raw_ev=1.0,
                base=np.nan,
                gross=np.nan,
                stress=np.nan,
            )
        ]
    )
    calibration = FinalScaleCalibration(
        intercept=0.0,
        slope=1.0,
        rows=500,
        days=8,
        valid=True,
    )
    scored = apply_final_scale(scored, calibration)

    trades, attempts, _ = select_policies(scored, calibration)

    primary_attempts = attempts.loc[
        attempts["policy"].eq("calibrated_hurdle_ev_15m_cap1")
    ]
    assert len(primary_attempts) == 1
    assert (
        primary_attempts.iloc[0]["evaluation_reason"]
        == "gross_label_missing"
    )
    assert trades.loc[
        trades["policy"].eq("calibrated_hurdle_ev_15m_cap1")
    ].empty
