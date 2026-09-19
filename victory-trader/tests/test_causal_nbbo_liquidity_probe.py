import pandas as pd

from victory_trader.causal_nbbo_liquidity_probe import (
    _first_after_quote,
    _last_prior_quote,
    _sample_attempts,
    feasibility,
    summarize,
)


class Stats:
    def to_dict(self):
        return {"network_requests": 0}


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.stats = Stats()

    def quotes(
        self,
        ticker,
        *,
        timestamp_gte,
        timestamp_lte,
        limit,
        order,
    ):
        return [
            row
            for row in self.rows
            if timestamp_gte <= row["sip_timestamp"] <= timestamp_lte
        ]


def _quote(target_ms, delta_ms, *, bid=9.99, ask=10.01, size=200):
    return {
        "bid_price": bid,
        "ask_price": ask,
        "bid_size": size,
        "ask_size": size,
        "sip_timestamp": int((target_ms + delta_ms) * 1_000_000),
    }


def test_prior_quote_never_uses_after_target():
    target = 1_760_000_000_000
    client = FakeClient(
        [
            _quote(target, -1500),
            _quote(target, -100),
            _quote(target, 100),
        ]
    )
    result = _last_prior_quote(client, "AAA", target)
    assert result["found"] is True
    assert result["age_ms"] == -100.0
    assert result["bid_size"] == 200.0


def test_first_after_quote_never_uses_prior_quote():
    target = 1_760_000_000_000
    client = FakeClient(
        [
            _quote(target, -50),
            _quote(target, 50),
            _quote(target, 100),
        ]
    )
    result = _first_after_quote(client, "AAA", target)
    assert result["found"] is True
    assert result["age_ms"] == 50.0


def test_zero_or_missing_size_is_not_valid_quote():
    target = 1_760_000_000_000
    client = FakeClient(
        [
            _quote(target, -100, size=0),
        ]
    )
    result = _last_prior_quote(client, "AAA", target)
    assert result["found"] is False


def test_sampling_uses_policy_and_fixed_count():
    rows = []
    for month in ("2026-01", "2026-02", "2026-03"):
        for i in range(30):
            rows.append(
                {
                    "month": month,
                    "policy": "raw_same_fit_cap1",
                    "trading_day": f"{month}-{(i % 20) + 1:02d}",
                    "decision_t": 1_000_000 + i,
                    "ticker": f"T{i:02d}",
                    "buy_return_10m_pct": 999 - i,
                }
            )
    rows.append(
        {
            "month": "2026-01",
            "policy": "earliest_eligible_10m_cap1",
            "trading_day": "2026-01-01",
            "decision_t": 1,
            "ticker": "BAD",
            "buy_return_10m_pct": 999999,
        }
    )
    sampled = _sample_attempts(pd.DataFrame(rows))
    assert len(sampled) == 60
    assert set(sampled["policy"]) == {"raw_same_fit_cap1"}


def test_summary_and_feasibility_pass_on_complete_probe():
    rows = []
    for month in ("2026-01", "2026-02", "2026-03"):
        for i in range(5):
            rows.append(
                {
                    "month": month,
                    "decision_found": True,
                    "decision_fresh_2s": True,
                    "entry_found": True,
                    "exit_found": True,
                    "both_depth_1000_ok": True,
                    "decision_age_ms": -100.0,
                    "decision_spread_pct": 0.2,
                    "entry_age_ms": 20.0,
                    "exit_age_ms": 30.0,
                    "top_book_ask_to_bid_return_pct": 0.5,
                    "bar_gross_return_pct": 0.7,
                    "bar_base_net_return_pct": -0.2,
                }
            )
    summary = summarize(pd.DataFrame(rows))
    checks = feasibility(summary, authorization_error=None)
    assert checks["all_pass"] is True


def test_feasibility_handles_authorization_failure_without_summary():
    checks = feasibility(
        pd.DataFrame(),
        authorization_error="historical NBBO authorization failed with HTTP 403",
    )
    assert checks["historical_quote_authorization"] is False
    assert checks["all_pass"] is False
