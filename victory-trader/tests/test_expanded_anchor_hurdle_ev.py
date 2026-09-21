import numpy as np
import pandas as pd

from victory_trader.expanded_anchor_hurdle_ev import (
    HORIZON,
    MagnitudeModel,
    ProbabilityModel,
    hurdle_ev,
    score_anchors,
    select_policies,
)


MINUTE_MS = 60_000


class ConstantBinaryModel:
    def __init__(self, probability):
        self.probability = float(probability)

    def predict_proba(self, frame):
        p = np.full(len(frame), self.probability, dtype=float)
        return np.column_stack([1.0 - p, p])


class IdentityPlatt:
    def predict_proba(self, logits):
        odds = np.exp(logits[:, 0])
        p = odds / (1.0 + odds)
        return np.column_stack([1.0 - p, p])


class ConstantRegressor:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


def _row(
    minute,
    *,
    base15=1.0,
    gross15=2.0,
    stress15=0.0,
    day="2026-02-02",
    ticker="AAA",
):
    return {
        "trading_day": day,
        "ticker": ticker,
        "t": minute * MINUTE_MS,
        "c": 5.0,
        "previous_close": 4.0,
        "active_minute_fraction_15m": 1.0,
        "minutes_since_10pct_cross": float(minute),
        "entry_price": 5.0,
        f"buy_return_{HORIZON}m_pct": gross15,
        f"buy_return_{HORIZON}m_base_net_return_pct": base15,
        f"buy_return_{HORIZON}m_stress_net_return_pct": stress15,
    }


def _models(probability=0.40, win=4.0, loss=1.0):
    probability_model = ProbabilityModel(
        model=ConstantBinaryModel(probability),
        feature_columns=tuple(),
        platt=IdentityPlatt(),
    )
    win_model = MagnitudeModel(
        model=ConstantRegressor(win),
        feature_columns=tuple(),
        offset=0.0,
        label="win",
    )
    loss_model = MagnitudeModel(
        model=ConstantRegressor(loss),
        feature_columns=tuple(),
        offset=0.0,
        label="loss",
    )
    return probability_model, win_model, loss_model


def test_hurdle_ev_uses_payoff_asymmetry():
    value = hurdle_ev(
        np.asarray([0.40]),
        np.asarray([4.0]),
        np.asarray([1.0]),
    )
    assert np.allclose(value, [1.0])


def test_low_win_probability_can_still_generate_primary_buy():
    frame = pd.DataFrame([_row(0, base15=2.0)])
    scored = score_anchors(frame, *_models(probability=0.40, win=4.0, loss=1.0))

    trades, attempts, _ = select_policies(scored)

    assert attempts.loc[
        attempts["policy"].eq("probability_half_15m_cap1")
    ].empty
    primary = trades.loc[trades["policy"].eq("hurdle_ev_15m_cap1")]
    assert len(primary) == 1
    assert np.isclose(
        primary.iloc[0]["selected_hurdle_ev_pct"],
        1.0,
    )


def test_high_win_probability_can_still_be_skipped_when_payoff_is_bad():
    frame = pd.DataFrame([_row(0, base15=-2.0)])
    scored = score_anchors(
        frame,
        *_models(probability=0.70, win=0.2, loss=2.0),
    )

    trades, attempts, _ = select_policies(scored)

    assert len(
        attempts.loc[
            attempts["policy"].eq("probability_half_15m_cap1")
        ]
    ) == 1
    assert trades.loc[
        trades["policy"].eq("hurdle_ev_15m_cap1")
    ].empty


def test_anchor_is_first_eligible_even_when_later_label_is_better():
    frame = pd.DataFrame(
        [
            _row(0, base15=np.nan, gross15=np.nan, stress15=np.nan),
            _row(1, base15=10.0, gross15=11.0, stress15=9.0),
        ]
    )
    scored = score_anchors(
        frame,
        *_models(probability=0.40, win=4.0, loss=1.0),
    )

    assert len(scored) == 1
    assert scored.iloc[0]["minutes_since_10pct_cross"] == 0.0
    assert pd.isna(
        scored.iloc[0][f"buy_return_{HORIZON}m_base_net_return_pct"]
    )


def test_primary_missing_outcome_consumes_anchor_attempt():
    frame = pd.DataFrame(
        [
            _row(0, base15=np.nan, gross15=np.nan, stress15=np.nan),
            _row(1, base15=5.0, gross15=6.0, stress15=4.0),
        ]
    )
    scored = score_anchors(
        frame,
        *_models(probability=0.40, win=4.0, loss=1.0),
    )

    trades, attempts, _ = select_policies(scored)
    primary_attempts = attempts.loc[
        attempts["policy"].eq("hurdle_ev_15m_cap1")
    ]

    assert len(primary_attempts) == 1
    assert (
        primary_attempts.iloc[0]["evaluation_reason"]
        == "gross_label_missing"
    )
    assert trades.loc[
        trades["policy"].eq("hurdle_ev_15m_cap1")
    ].empty
