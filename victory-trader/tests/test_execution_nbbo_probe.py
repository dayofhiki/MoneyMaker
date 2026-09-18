import pandas as pd

from victory_trader.execution_nbbo_probe import (
    _quote_near,
    _summary,
    run_probe,
)


class FakeClient:
    def __init__(self):
        class Stats:
            def to_dict(self):
                return {"network_requests": 0}
        self.stats = Stats()

    def quotes(self, ticker, *, timestamp_gte, timestamp_lte, limit, order):
        target = (timestamp_gte + 5_000 * 1_000_000)
        return [
            {
                "bid_price": 9.98,
                "ask_price": 10.02,
                "sip_timestamp": target - 500_000_000,
            },
            {
                "bid_price": 9.99,
                "ask_price": 10.01,
                "sip_timestamp": target - 100_000_000,
            },
            {
                "bid_price": 10.00,
                "ask_price": 10.02,
                "sip_timestamp": target + 100_000_000,
            },
        ]


def test_quote_near_prefers_last_valid_quote_before_target():
    client = FakeClient()
    result = _quote_near(client, "AAA", 1_760_000_000_000)
    assert result["quote_found"] is True
    assert result["quote_side"] == "before"
    assert result["bid"] == 9.99
    assert result["ask"] == 10.01
    assert result["quote_age_ms"] == -100.0


def test_nbbo_probe_computes_ask_to_bid_return():
    trades = pd.DataFrame(
        [
            {
                "month": "2026-01",
                "trading_day": "2026-01-02",
                "ticker": "AAA",
                "entry_t": 1_760_000_000_000,
                "exit_t": 1_760_000_060_000,
                "entry_reference_price": 10.0,
                "exit_reference_price": 10.2,
                "gross_return_pct": 2.0,
                "base_net_return_pct": 1.0,
                "exit_reason": "test",
            }
        ]
    )
    details, error = run_probe(trades, FakeClient())
    assert error is None
    assert len(details) == 1
    assert details.iloc[0]["entry_quote_found"]
    assert pd.notna(details.iloc[0]["nbbo_ask_to_bid_return_pct"])


def test_nbbo_summary_has_overall_and_price_bucket():
    frame = pd.DataFrame(
        {
            "month": ["2026-01", "2026-01"],
            "entry_quote_found": [True, True],
            "exit_quote_found": [True, True],
            "entry_mid": [1.5, 6.0],
            "entry_spread_pct": [1.0, 0.2],
            "exit_spread_pct": [0.8, 0.2],
            "entry_zero_move_crossing_return_pct": [-1.0, -0.2],
            "bar_gross_return_pct": [1.0, 2.0],
            "bar_base_net_return_pct": [0.0, 1.0],
            "nbbo_ask_to_bid_return_pct": [0.5, 1.6],
            "nbbo_minus_bar_gross_pct": [-0.5, -0.4],
            "entry_quote_age_ms": [-50.0, -20.0],
            "exit_quote_age_ms": [-60.0, -30.0],
        }
    )
    summary = _summary(frame)
    assert "overall" in set(summary["group"])
    assert "price:<$2" in set(summary["group"])
    assert "price:$5-$10" in set(summary["group"])
