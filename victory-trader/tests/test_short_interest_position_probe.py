from datetime import date

import pandas as pd

from victory_trader.short_interest_position_probe import (
    _eligible_history,
    _event_metrics,
    _prepare_ticker_history,
    success_check,
)


def _row(settlement, short_interest, avg=1000, dtc=1.0):
    return {
        "settlement_date": settlement,
        "short_interest": short_interest,
        "avg_daily_volume": avg,
        "days_to_cover": dtc,
    }


def test_publication_date_not_settlement_date_controls_availability():
    records = _prepare_ticker_history(
        [
            _row("2025-12-31", 1000),
            _row("2026-01-15", 1200),
        ]
    )

    # 2025-12-31 record was published 2026-01-12.
    eligible = _eligible_history(records, date(2026, 1, 20))
    assert len(eligible) == 1
    assert eligible[0]["_settlement_day"] == date(2025, 12, 31)

    # 2026-01-15 record is not public until 2026-01-27.
    eligible = _eligible_history(records, date(2026, 1, 27))
    assert len(eligible) == 1

    # Strictly after publication day, both records are usable.
    eligible = _eligible_history(records, date(2026, 1, 28))
    assert len(eligible) == 2


def test_unknown_settlement_cycle_is_ignored_not_guessed():
    records = _prepare_ticker_history(
        [
            _row("2026-01-10", 999),
            _row("2026-01-15", 1200),
        ]
    )
    assert len(records) == 1
    assert records[0]["_settlement_day"] == date(2026, 1, 15)


def test_event_metrics_computes_causal_position_change():
    records = _prepare_ticker_history(
        [
            _row("2025-12-15", 1000, avg=500, dtc=2.0),
            _row("2025-12-31", 1250, avg=625, dtc=2.0),
        ]
    )
    result = _event_metrics(
        month="2026-01",
        trading_day="2026-01-20",
        ticker="AAA",
        threshold_pct=10.0,
        records=records,
    )
    assert result["has_two_published_records"] is True
    assert result["latest_short_interest"] == 1250.0
    assert result["previous_short_interest"] == 1000.0
    assert result["latest_short_interest_change_pct"] == 25.0
    assert result["latest_publication_date"] == "2026-01-12"


def test_short_interest_probe_success_rules():
    rows = []
    for month in ("2026-01", "2026-02", "2026-03"):
        for i in range(10):
            rows.append(
                {
                    "month": month,
                    "has_published_record": i < 9,
                    "has_two_published_records": i < 8,
                }
            )
    frame = pd.DataFrame(rows)
    assert success_check(frame)["all_pass"] is True

    frame.loc[
        (frame["month"].eq("2026-02")) & (frame.index % 10 >= 6),
        "has_published_record",
    ] = False
    assert success_check(frame)["all_pass"] is False
