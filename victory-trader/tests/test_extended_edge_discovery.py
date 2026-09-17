from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from victory_trader.extended_edge_discovery import (
    render_extended_edge_discovery,
    summarize_outcome_missingness_reasons,
)


ET = ZoneInfo("America/New_York")


def _ms(hour: int, minute: int) -> int:
    return int(datetime(2026, 3, 2, hour, minute, tzinfo=ET).timestamp() * 1000)


def test_extended_discovery_separates_gross_net_and_availability_targets():
    rows = []
    for day in range(1, 9):
        for i in range(20):
            feature = float(i + 1)
            resolved = i % 5 != 0
            event_ts = 1_772_450_000_000 + day * 86_400_000 + i * 60_000
            rows.append(
                {
                    "trading_day": f"2026-03-{day:02d}",
                    "threshold_pct": 10.0,
                    "previous_close": feature,
                    "timestamp_ms": event_ts,
                    "entry_timestamp_ms": event_ts + 60_000,
                    "entry_price": feature + 0.1,
                    "return_5m_pct": feature * 0.1 if resolved else float("nan"),
                    "return_5m_base_net_return_pct": (
                        feature * 0.1 - 1.0 if resolved else float("nan")
                    ),
                }
            )
    report = render_extended_edge_discovery(pd.DataFrame(rows), horizon_min=5)
    assert "GROSS continuation" in report
    assert "BASE-NET continuation" in report
    assert "EXACT-HORIZON AVAILABILITY" in report
    assert "Outcome missingness reasons" in report
    assert "previous_close" in report
    assert "execution-cost mechanics" in report
    assert "unresolved-exposure stress" in report


def test_missingness_reasons_distinguish_sparse_bars_from_session_close():
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-03-02",
                "threshold_pct": 10.0,
                "timestamp_ms": _ms(15, 59),
                "entry_timestamp_ms": float("nan"),
                "entry_price": float("nan"),
                "return_5m_pct": float("nan"),
            },
            {
                "trading_day": "2026-03-02",
                "threshold_pct": 10.0,
                "timestamp_ms": _ms(10, 0),
                "entry_timestamp_ms": float("nan"),
                "entry_price": float("nan"),
                "return_5m_pct": float("nan"),
            },
            {
                "trading_day": "2026-03-02",
                "threshold_pct": 10.0,
                "timestamp_ms": _ms(15, 56),
                "entry_timestamp_ms": _ms(15, 57),
                "entry_price": 10.0,
                "return_5m_pct": float("nan"),
            },
            {
                "trading_day": "2026-03-02",
                "threshold_pct": 10.0,
                "timestamp_ms": _ms(10, 0),
                "entry_timestamp_ms": _ms(10, 1),
                "entry_price": 10.0,
                "return_5m_pct": float("nan"),
            },
            {
                "trading_day": "2026-03-02",
                "threshold_pct": 10.0,
                "timestamp_ms": _ms(10, 0),
                "entry_timestamp_ms": _ms(10, 1),
                "entry_price": 10.0,
                "return_5m_pct": 1.0,
            },
        ]
    )

    overall, by_threshold = summarize_outcome_missingness_reasons(frame, horizon_min=5)
    counts = dict(zip(overall["reason"], overall["n"], strict=True))
    assert counts == {
        "entry_session_close": 1,
        "entry_missing_bar": 1,
        "horizon_session_close": 1,
        "horizon_missing_bar": 1,
        "observed": 1,
    }
    assert by_threshold["n"].sum() == 5
