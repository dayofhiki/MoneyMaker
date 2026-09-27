import numpy as np
import pandas as pd

from victory_trader import causal_tradability_admission as r245


def _state(t=10_000):
    return {
        "trading_day": "2026-05-11",
        "ticker": "TEST",
        "hot_t": 0,
        "state_t": t,
        "can_enter": True,
        "terminal": False,
    }


def test_tradable_label_requires_entry_and_continuation():
    state = pd.DataFrame([_state()])
    times = np.array(
        [
            11_000,
            70_000,
            130_000,
            190_000,
            400_000,
        ],
        dtype=np.int64,
    )
    opens = np.array([10.0, 10.1, 10.2, 10.3, 10.4])
    labeled = r245.label_tradability(
        state, {("2026-05-11", "TEST"): (times, opens)}
    )
    assert bool(labeled.loc[0, "entry_available"])
    assert labeled.loc[0, "continuation_anchor_hits"] == 3
    assert bool(labeled.loc[0, "continuation_available"])
    assert bool(labeled.loc[0, "tradable_label"])


def test_tradable_label_rejects_one_off_print():
    state = pd.DataFrame([_state()])
    labeled = r245.label_tradability(
        state,
        {
            ("2026-05-11", "TEST"): (
                np.array([11_000], dtype=np.int64),
                np.array([10.0]),
            )
        },
    )
    assert bool(labeled.loc[0, "entry_available"])
    assert not bool(labeled.loc[0, "continuation_available"])
    assert not bool(labeled.loc[0, "tradable_label"])


def test_threshold_rule_uses_lowest_qualifying_grid_point():
    rows = []
    probs = []
    # 40 positives at high scores, 60 negatives mostly low.
    for i in range(100):
        rows.append(
            {
                "trading_day": r245.THRESHOLD_DAYS[i % len(r245.THRESHOLD_DAYS)],
                "tradable_label": i < 40,
            }
        )
        probs.append(0.90 if i < 40 else (0.10 if i < 50 else 0.01))
    frame = pd.DataFrame(rows)
    chosen, table = r245.choose_threshold(frame, np.asarray(probs))
    assert chosen is not None
    chosen_row = next(row for row in table if row["threshold"] == chosen)
    assert chosen_row["precision"] >= r245.MIN_VALIDATION_PRECISION
    assert chosen_row["recall"] >= r245.MIN_VALIDATION_RECALL
    assert chosen_row["admitted"] >= r245.MIN_VALIDATION_ADMITTED


def test_gated_enter_becomes_wait_and_can_reconsider():
    scored = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 60_000,
                "can_enter": True,
                "terminal": False,
                "flat_action": "ENTER",
                "held_action": "HOLD",
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 120_000,
                "can_enter": True,
                "terminal": False,
                "flat_action": "ENTER",
                "held_action": "HOLD",
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 180_000,
                "can_enter": True,
                "terminal": False,
                "flat_action": "WAIT",
                "held_action": "EXIT",
            },
        ]
    )
    decisions = r245.gated_policy_decisions(
        scored, np.array([0.1, 0.9, 0.9]), threshold=0.5
    )
    assert bool(decisions.loc[0, "entered"])
    assert decisions.loc[0, "blocked_enter_actions"] == 1
    assert decisions.loc[0, "entry_decision_t"] == 120_000
    assert decisions.loc[0, "exit_decision_t"] == 180_000


def test_request245_contract_is_frozen():
    assert r245.REQUEST_ID == 245
    assert r245.TRAIN_DAYS == ["2026-04-30", "2026-05-01", "2026-05-04"]
    assert r245.THRESHOLD_DAYS == [
        "2026-05-05",
        "2026-05-06",
        "2026-05-07",
        "2026-05-08",
    ]
    assert r245.ANCHOR_MINUTES == (1, 2, 3, 4, 5)
    assert r245.MIN_CONTINUATION_ANCHORS == 3
