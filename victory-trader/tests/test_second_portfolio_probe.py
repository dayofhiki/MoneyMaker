"""Causal episode selection for the development-only account diagnostic."""

from datetime import date

import pandas as pd
import pytest

from victory_trader.causal_second_execution import SecondBar
from victory_trader.market_calendar import regular_session_bounds
from victory_trader.second_portfolio_probe import (
    DEVELOPMENT_DAY,
    build_attempts,
    run_probe,
    summarize_scenarios,
)

T = 1_800_000_000_000


def test_exact_previous_minute_close_is_required_for_sizing():
    positions = pd.DataFrame([
        {"trading_day": DEVELOPMENT_DAY, "ticker": "A", "hot_t": T},
        {"trading_day": DEVELOPMENT_DAY, "ticker": "A", "hot_t": T},
        {"trading_day": DEVELOPMENT_DAY, "ticker": "B", "hot_t": T},
    ])
    scan = pd.DataFrame([
        {"trading_day": DEVELOPMENT_DAY, "ticker": "A", "t": T - 60_000, "c": 10},
        {"trading_day": DEVELOPMENT_DAY, "ticker": "B", "t": T - 120_000, "c": 10},
    ])
    attempts, audit = build_attempts(
        positions, scan, day=DEVELOPMENT_DAY, session_close_t=T + 30_000,
    )
    assert audit == {
        "episodes": 2, "causal_decision_refs": 1, "missing_decision_refs": 1,
        "missing_prior_minute": 1, "invalid_prior_close": 0,
        "after_session": 0,
    }
    assert attempts[0].ticker == "A"
    assert attempts[0].decision_price == 10


def test_probe_rejects_unopened_day():
    with pytest.raises(ValueError, match="already-open"):
        build_attempts(
            pd.DataFrame(columns=["trading_day", "ticker", "hot_t"]),
            pd.DataFrame(columns=["trading_day", "ticker", "t", "c"]),
            day="2026-07-01", session_close_t=T + 30_000,
        )


def test_nonfinite_prior_close_cannot_size_an_order():
    positions = pd.DataFrame([{
        "trading_day": DEVELOPMENT_DAY, "ticker": "A", "hot_t": T,
    }])
    scan = pd.DataFrame([{
        "trading_day": DEVELOPMENT_DAY, "ticker": "A",
        "t": T - 60_000, "c": float("inf"),
    }])
    attempts, audit = build_attempts(
        positions, scan, day=DEVELOPMENT_DAY, session_close_t=T + 30_000,
    )
    assert attempts == []
    assert audit["missing_decision_refs"] == 1
    assert audit["invalid_prior_close"] == 1


def test_other_preopened_day_uses_same_frozen_selection():
    day = "2026-06-24"
    attempts, audit = build_attempts(
        pd.DataFrame([{"trading_day": day, "ticker": "A", "hot_t": T}]),
        pd.DataFrame([{
            "trading_day": day, "ticker": "A", "t": T - 60_000, "c": 10,
        }]),
        day=day, session_close_t=T + 30_000,
    )
    assert len(attempts) == 1
    assert audit["causal_decision_refs"] == 1


def test_frozen_sensitivity_grid_has_twelve_reference_results():
    attempts, _ = build_attempts(
        pd.DataFrame([{"trading_day": DEVELOPMENT_DAY, "ticker": "A", "hot_t": T}]),
        pd.DataFrame([{
            "trading_day": DEVELOPMENT_DAY, "ticker": "A",
            "t": T - 60_000, "c": 10,
        }]),
        day=DEVELOPMENT_DAY, session_close_t=T + 30_000,
    )
    outputs = summarize_scenarios(
        attempts, {"A": [SecondBar(T, 10, 10, 10, 10)]},
    )
    assert len(outputs) == 12
    assert {(x["scenario"], x["latency_ms"]) for x in outputs} == {
        (scenario, latency) for scenario in ("light", "base", "stress")
        for latency in (0, 1_000, 2_000, 5_000)
    }
    assert all(not x["summary"]["promotion_allowed"] for x in outputs)


def test_offline_probe_reads_explicit_parquet_without_credentials(tmp_path, monkeypatch):
    bounds = regular_session_bounds(date.fromisoformat(DEVELOPMENT_DAY))
    assert bounds is not None
    open_t = int(bounds[0].timestamp() * 1000)
    decision_t = open_t + 60_000
    fresh = tmp_path / "fresh"
    second = tmp_path / "seconds"
    fresh.mkdir()
    second.mkdir()
    pd.DataFrame([{
        "trading_day": DEVELOPMENT_DAY, "ticker": "A", "hot_t": decision_t,
    }]).to_parquet(fresh / f"{DEVELOPMENT_DAY}-positions.parquet")
    pd.DataFrame([{
        "trading_day": DEVELOPMENT_DAY, "ticker": "A", "t": open_t, "c": 10,
    }]).to_parquet(fresh / f"{DEVELOPMENT_DAY}-scan.parquet")
    pd.DataFrame([{
        "t": decision_t, "o": 10, "h": 10, "l": 10, "c": 10,
    }]).to_parquet(second / f"{DEVELOPMENT_DAY}-A-seconds.parquet")
    monkeypatch.setattr(
        "victory_trader.second_portfolio_probe.load_settings",
        lambda: pytest.fail("offline probe accessed credentials"),
    )
    output = tmp_path / "report.json"
    report = run_probe(fresh, output, second_bars_dir=second)
    assert output.is_file()
    assert report["second_source"] == "offline_parquet"
    assert report["halt_feed"] == "unavailable:offline_no_halt_manifest"
    assert report["selection"]["causal_decision_refs"] == 1
    assert report["data_statuses"] == {"adapted": 1}
    assert report["api_stats"] is None

