import pandas as pd

from victory_trader.state_oracle_diagnostic import (
    evaluate_month,
    summarize,
)


def _panel() -> pd.DataFrame:
    rows = []
    for episode in range(40):
        ticker = f"T{episode:03d}"
        for minute in (0, 10, 20, 40, 80):
            base = -1.0
            gross = 0.0
            if minute == 20:
                base = 2.0 + episode * 0.01
                gross = base + 1.0
            rows.append(
                {
                    "trading_day": f"2026-01-{1 + episode % 10:02d}",
                    "ticker": ticker,
                    "minutes_since_10pct_cross": float(minute),
                    "entry_price": 5.0,
                    "buy_return_5m_pct": gross,
                    "buy_return_5m_base_net_return_pct": base,
                    "buy_return_10m_pct": gross,
                    "buy_return_10m_base_net_return_pct": base,
                    "buy_return_15m_pct": gross,
                    "buy_return_15m_base_net_return_pct": base,
                }
            )
    return pd.DataFrame(rows)


def test_oracle_finds_profitable_state_inside_window():
    details = evaluate_month(_panel(), month="2026-01")
    row = details.loc[
        details["horizon_min"].eq(10)
        & details["window_min"].eq(30)
    ].iloc[0]
    assert row["evaluable_episodes"] == 40
    assert row["episodes_with_any_positive_base_rate"] == 1.0
    assert row["oracle_best_base_positive_rate"] == 1.0
    assert row["oracle_entry_minute_median"] == 20.0


def test_oracle_summary_keeps_horizon_window_grid():
    details = pd.concat(
        [
            evaluate_month(_panel(), month="2026-01"),
            evaluate_month(_panel(), month="2026-02"),
            evaluate_month(_panel(), month="2026-03"),
        ],
        ignore_index=True,
    )
    summary = summarize(details)
    assert len(summary) == 9
    assert summary["months_tested"].eq(3).all()
