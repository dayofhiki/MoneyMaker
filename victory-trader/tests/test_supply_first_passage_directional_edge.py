import math

import pandas as pd

from victory_trader.supply_first_passage_directional_edge import (
    SupplyRecord,
    add_supply_features,
    coverage,
)


class FakeSupply:
    def record(self, day, ticker):
        return SupplyRecord(
            query_date="2026-05-10",
            success=True,
            weighted_shares=10_000_000.0,
            share_class_shares=8_000_000.0,
            provider_market_cap=100_000_000.0,
        )


class FakeVolume:
    def volume_before(self, day, ticker, state_t):
        return 500_000.0


def test_supply_features_are_strict_prior_and_turnover_scaled():
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "state_t": 1_000,
                "log_current_close": math.log(10.0),
                "return_from_previous_close_pct": 25.0,
                "log_minute_volume": math.log1p(100_000.0),
            }
        ]
    )
    result = add_supply_features(
        frame,
        FakeSupply(),
        FakeVolume(),
    )
    row = result.iloc[0]
    assert math.isclose(
        row["supply_minute_weighted_turnover"],
        0.01,
    )
    assert math.isclose(
        row["supply_regular_cum_weighted_turnover"],
        0.05,
    )
    assert math.isclose(
        row["supply_share_class_to_weighted_ratio"],
        0.8,
    )
    previous_close = 8.0
    assert math.isclose(
        math.exp(row["supply_log_implied_market_cap_prior"]),
        10_000_000.0 * previous_close,
        rel_tol=1e-12,
    )
    cov = coverage(result)
    assert cov["request_success"] == 1.0
    assert cov["weighted_shares_coverage"] == 1.0
    assert cov["share_class_coverage"] == 1.0
    assert cov["strict_prior_query_dates"] is True


def test_missing_share_class_remains_missing_not_filtered():
    class MissingShareClass:
        def record(self, day, ticker):
            return SupplyRecord(
                query_date="2026-05-10",
                success=True,
                weighted_shares=10_000_000.0,
                share_class_shares=float("nan"),
                provider_market_cap=float("nan"),
            )

    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "state_t": 1_000,
                "log_current_close": math.log(10.0),
                "return_from_previous_close_pct": 25.0,
                "log_minute_volume": math.log1p(100_000.0),
            }
        ]
    )
    result = add_supply_features(
        frame,
        MissingShareClass(),
        FakeVolume(),
    )
    row = result.iloc[0]
    assert pd.isna(row["supply_log_share_class_shares_prior"])
    assert pd.isna(row["supply_minute_share_class_turnover"])
    assert not pd.isna(row["supply_minute_weighted_turnover"])
