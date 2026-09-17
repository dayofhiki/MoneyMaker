import pandas as pd
import pytest

from victory_trader.analytics import (
    summarize_barriers,
    summarize_concentration,
    summarize_monthly_stability,
    summarize_price_buckets,
    summarize_tail_risk,
    summarize_time_buckets,
)


def sample_frame() -> pd.DataFrame:
    rows = []
    returns = [2.0, 1.0, -4.0, 3.0, 0.5, -1.0]
    days = ["2026-01-05", "2026-01-05", "2026-01-06", "2026-02-02", "2026-02-03", "2026-02-03"]
    tickers = ["AAA", "BBB", "AAA", "CCC", "DDD", "EEE"]
    prices = [1.0, 3.0, 7.0, 15.0, 1.5, 4.0]
    minutes = [5, 45, 90, 150, 260, 20]
    for value, day, ticker, price, minute in zip(
        returns,
        days,
        tickers,
        prices,
        minutes,
        strict=True,
    ):
        rows.append(
            {
                "trading_day": day,
                "ticker": ticker,
                "threshold_pct": 20.0,
                "previous_close": price,
                "minutes_from_regular_open": minute,
                "return_5m_base_net_return_pct": value,
                "tp2_sl1_status": "session_close" if ticker == "EEE" else "take_profit",
                "tp2_sl1_exit_return_pct": value,
                "tp2_sl1_base_net_return_pct": value - 0.2,
            }
        )
    return pd.DataFrame(rows)


def test_tail_risk_reports_lower_tail_not_just_win_rate():
    result = summarize_tail_risk(sample_frame())
    row = result.iloc[0]
    assert row["worst_return_pct"] == pytest.approx(-4.0)
    assert row["p05_return_pct"] < 0
    assert row["cvar05_return_pct"] <= row["p05_return_pct"]


def test_stability_and_bucket_summaries_cover_multiple_regimes():
    frame = sample_frame()
    monthly = summarize_monthly_stability(frame)
    prices = summarize_price_buckets(frame)
    times = summarize_time_buckets(frame)
    assert set(monthly["month"]) == {"2026-01", "2026-02"}
    assert set(prices["price_bucket"]) == {"$0.50-2", "$2-5", "$5-10", "$10-20"}
    assert set(times["time_bucket"]) >= {"open-30m", "30-60m", "60-120m", "120-240m", "240m-close"}


def test_concentration_reports_cluster_dependence():
    result = summarize_concentration(sample_frame())
    row = result.iloc[0]
    assert row["ticker_day_clusters"] == 6
    assert 0 < row["top1_abs_return_share"] <= row["top5_abs_return_share"] <= 1


def test_regular_session_close_is_a_resolved_barrier_exit():
    result = summarize_barriers(sample_frame())
    row = result.iloc[0]
    assert row["resolved_n"] == 6
    assert row["session_close_rate"] == pytest.approx(1 / 6)
    assert row["unresolved_session_close_rate"] == 0.0
