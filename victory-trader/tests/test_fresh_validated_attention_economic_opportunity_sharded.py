from __future__ import annotations

from datetime import date

from victory_trader.fresh_validated_attention_economic_opportunity import (
    OPPORTUNITY_DAYS,
)
from victory_trader.fresh_validated_attention_economic_opportunity_sharded import (
    _role_days,
)
from victory_trader.hierarchical_attention_runtime import (
    STAGE2_FIT_END,
    STAGE2_FIT_START,
)


def test_request162_shards_preserve_date_contract():
    start = date.fromisoformat("2026-01-02")
    end = date.fromisoformat("2026-06-12")
    stage2 = _role_days("stage2", start, end)
    opportunity = _role_days("opportunity", start, end)

    assert stage2
    assert stage2[0] >= STAGE2_FIT_START
    assert stage2[-1] <= STAGE2_FIT_END
    assert opportunity == OPPORTUNITY_DAYS
    assert set(stage2).isdisjoint(opportunity)


def test_request162_shards_do_not_add_june_1_to_5():
    opportunity = _role_days(
        "opportunity",
        date.fromisoformat("2026-01-02"),
        date.fromisoformat("2026-06-12"),
    )
    assert all(
        day not in opportunity
        for day in [
            "2026-06-01",
            "2026-06-02",
            "2026-06-03",
            "2026-06-04",
            "2026-06-05",
        ]
    )
