import numpy as np
import pandas as pd

from victory_trader.expanded_episode_viability_rank import (
    HORIZON,
    RankModel,
    ViabilityModel,
    _first_eligible_rows,
    _rank_fit_rows,
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


class MinuteRankModel:
    def predict(self, frame):
        if "minutes_since_10pct_cross" not in frame.columns:
            return np.zeros(len(frame), dtype=float)
        return (
            pd.to_numeric(
                frame["minutes_since_10pct_cross"],
                errors="coerce",
            )
            .fillna(0.0)
            .to_numpy(dtype=float)
            / 10.0
        )


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


def test_first_eligible_row_is_chronological_not_label_selected():
    frame = pd.DataFrame(
        [
            _row(0, base15=np.nan),
            _row(1, base15=10.0),
            _row(2, base15=20.0),
        ]
    )
    anchor = _first_eligible_rows(frame)
    assert len(anchor) == 1
    assert anchor.iloc[0]["minutes_since_10pct_cross"] == 0.0
    assert pd.isna(
        anchor.iloc[0][f"buy_return_{HORIZON}m_base_net_return_pct"]
    )


def test_rank_target_is_computed_before_three_minute_sampling():
    frame = pd.DataFrame(
        [_row(i, base15=float(i)) for i in range(12)]
    )
    fit = _rank_fit_rows(frame)

    assert fit["minutes_since_10pct_cross"].tolist() == [0.0, 3.0, 6.0, 9.0]
    expected = [1 / 12, 4 / 12, 7 / 12, 10 / 12]
    assert np.allclose(
        fit["_episode_rank_target"].to_numpy(dtype=float),
        expected,
    )


def test_primary_enters_first_rank_gate_after_viability_arm():
    frame = pd.DataFrame(
        [
            _row(0, base15=1.0),
            _row(1, base15=2.0),
            _row(2, base15=3.0),
            _row(3, base15=4.0),
        ]
    )
    viability = ViabilityModel(
        model=ConstantBinaryModel(0.70),
        feature_columns=tuple(),
        platt=IdentityPlatt(),
    )
    rank = RankModel(
        model=MinuteRankModel(),
        feature_columns=("minutes_since_10pct_cross",),
    )

    trades, attempts, paths = select_policies(
        frame,
        viability,
        rank,
        rank_gate=0.20,
    )

    primary = trades.loc[
        trades["policy"].eq("viability_rank_15m_cap1")
    ]
    viable = trades.loc[
        trades["policy"].eq("earliest_viable_15m_cap1")
    ]

    assert len(primary) == 1
    assert primary.iloc[0]["minutes_since_10pct_cross"] == 2.0
    assert len(viable) == 1
    assert viable.iloc[0]["minutes_since_10pct_cross"] == 0.0
    assert len(
        attempts.loc[
            attempts["policy"].eq("viability_rank_15m_cap1")
        ]
    ) == 1
    assert int(
        paths.loc[
            paths["policy"].eq("viability_rank_15m_cap1"),
            "rank_signals",
        ].iloc[0]
    ) == 1


def test_nonviable_episode_never_generates_primary_attempt():
    frame = pd.DataFrame(
        [
            _row(0, base15=10.0),
            _row(1, base15=10.0),
        ]
    )
    viability = ViabilityModel(
        model=ConstantBinaryModel(0.20),
        feature_columns=tuple(),
        platt=IdentityPlatt(),
    )
    rank = RankModel(
        model=MinuteRankModel(),
        feature_columns=("minutes_since_10pct_cross",),
    )

    trades, attempts, _ = select_policies(
        frame,
        viability,
        rank,
        rank_gate=0.0,
    )

    assert trades.loc[
        trades["policy"].eq("viability_rank_15m_cap1")
    ].empty
    assert attempts.loc[
        attempts["policy"].eq("viability_rank_15m_cap1")
    ].empty


def test_first_signal_consumes_attempt_when_outcome_missing():
    frame = pd.DataFrame(
        [
            _row(
                0,
                base15=np.nan,
                gross15=np.nan,
                stress15=np.nan,
            ),
            _row(1, base15=5.0, gross15=6.0, stress15=4.0),
        ]
    )
    viability = ViabilityModel(
        model=ConstantBinaryModel(0.80),
        feature_columns=tuple(),
        platt=IdentityPlatt(),
    )
    rank = RankModel(
        model=MinuteRankModel(),
        feature_columns=("minutes_since_10pct_cross",),
    )

    trades, attempts, _ = select_policies(
        frame,
        viability,
        rank,
        rank_gate=0.0,
    )

    primary_attempts = attempts.loc[
        attempts["policy"].eq("viability_rank_15m_cap1")
    ]
    assert len(primary_attempts) == 1
    assert primary_attempts.iloc[0]["evaluation_reason"] == "gross_label_missing"
    assert trades.loc[
        trades["policy"].eq("viability_rank_15m_cap1")
    ].empty
