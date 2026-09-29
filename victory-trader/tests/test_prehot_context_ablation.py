import pandas as pd
import pytest

from victory_trader.prehot_context_ablation import (
    attach_context,
    build_ticker_context,
)


def scan():
    rows = []
    for ticker, bias in [("AAA", 0.0), ("BBB", 1.0)]:
        for i, t in enumerate([60_000, 120_000, 180_000, 240_000]):
            close = 10.0 + bias + i * 0.2
            rows.append({
                "trading_day": "2026-05-05",
                "ticker": ticker,
                "t": t,
                "o": close - 0.1,
                "h": close + 0.2,
                "l": close - 0.2,
                "c": close,
                "v": 1000 + i * 100,
                "n": 100 + i * 10,
                "attention_score": 1.0 + i,
                "attention_rank": float(i + 1),
                "return_from_previous_close_pct": float(i),
            })
    return pd.DataFrame(rows)


def candidate():
    return pd.DataFrame([
        {
            "trading_day": "2026-05-05",
            "ticker": "AAA",
            "t": 180_000,
            "hot_t": 180_000,
        }
    ])


def test_future_minute_does_not_change_hot_context():
    original, _, _ = attach_context(candidate(), scan())
    changed_scan = scan()
    future = changed_scan.t.eq(240_000)
    changed_scan.loc[future, "c"] = 9999.0
    changed_scan.loc[future, "v"] = 999999999.0
    changed, _, _ = attach_context(candidate(), changed_scan)
    context_columns = [
        c for c in original.columns
        if c.startswith("prehot_") or c.startswith("market_")
    ]
    pd.testing.assert_frame_equal(
        original[context_columns],
        changed[context_columns],
    )


def test_exact_minute_lag_is_required():
    frame = scan()
    frame = frame.loc[
        ~(
            frame.ticker.eq("AAA")
            & frame.t.eq(120_000)
        )
    ].copy()
    context, names = build_ticker_context(frame)
    row = context.loc[
        context.ticker.eq("AAA")
        & context.t.eq(180_000)
    ].iloc[0]
    assert "prehot_lag1_attention_score" in names
    assert pd.isna(row["prehot_lag1_attention_score"])


def test_market_context_uses_same_completed_timestamp():
    enriched, _, market_names = attach_context(
        candidate(),
        scan(),
    )
    assert "market_universe_size" in market_names
    assert enriched.iloc[0].market_universe_size == pytest.approx(2.0)
