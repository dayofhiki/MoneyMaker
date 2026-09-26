import pandas as pd

from victory_trader.causal_first_passage_directional_edge import add_first_passage


class FakeStore:
    def __init__(self, seconds):
        self._seconds = seconds

    def seconds(self, day, ticker):
        return self._seconds

    def halts(self, day):
        return {}


def test_first_passage_distinguishes_take_stop_and_ambiguity():
    base = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "AAA",
                "replay_status": "closed",
                "entry_fill_t": 1_000,
                "entry_modeled_price": 10.0,
            }
        ]
    )

    take_seconds = pd.DataFrame(
        [
            {"t": 1_000, "o": 10.0, "h": 10.6, "l": 9.9, "c": 10.5},
        ]
    )
    take = add_first_passage(base, FakeStore(take_seconds), take_pct=5.0)
    assert take.iloc[0]["first_passage_status"] == "take_first"

    stop_seconds = pd.DataFrame(
        [
            {"t": 1_000, "o": 10.0, "h": 10.1, "l": 9.6, "c": 9.7},
        ]
    )
    stop = add_first_passage(base, FakeStore(stop_seconds), take_pct=5.0)
    assert stop.iloc[0]["first_passage_status"] == "stop_first"

    both_seconds = pd.DataFrame(
        [
            {"t": 1_000, "o": 10.0, "h": 10.6, "l": 9.6, "c": 10.0},
        ]
    )
    both = add_first_passage(base, FakeStore(both_seconds), take_pct=5.0)
    assert both.iloc[0]["first_passage_status"] == "ambiguous"


def test_entry_unavailable_remains_unscored():
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "AAA",
                "replay_status": "entry_unavailable",
                "entry_fill_t": None,
                "entry_modeled_price": None,
            }
        ]
    )
    empty = pd.DataFrame(columns=["t", "o", "h", "l", "c"])
    result = add_first_passage(frame, FakeStore(empty), take_pct=5.0)
    assert result.iloc[0]["first_passage_status"] == "entry_unavailable"
    assert pd.isna(result.iloc[0]["take_before_stop"])
