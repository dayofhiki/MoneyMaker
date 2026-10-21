"""Provider path bridge tests, including halt-boundary conservatism."""

import pandas as pd
import pytest

from victory_trader.second_path_adapter import HaltInterval, adapt_second_bars

T = 1_800_000_000_000


def frame(*seconds):
    return pd.DataFrame([
        {"t": T + second * 1_000, "o": 10, "h": 11, "l": 9, "c": 10}
        for second in seconds
    ])


def test_session_filter_and_halt_overlap_block_fills():
    result = adapt_second_bars(
        frame(-1, 0, 1, 2, 3, 4),
        session_open_t=T,
        session_close_t=T + 4_000,
        halts=[HaltInterval(T + 1_500, T + 3_000)],
    )
    assert [bar.t for bar in result.bars] == [T, T + 1_000, T + 2_000, T + 3_000]
    assert [bar.halted for bar in result.bars] == [False, True, True, False]
    assert result.provider_rows == 6
    assert result.session_rows == 4
    assert result.halt_overlap_rows == 2


def test_unknown_resume_blocks_remaining_session_bars():
    result = adapt_second_bars(
        frame(0, 1, 2), session_open_t=T, session_close_t=T + 3_000,
        halts=[HaltInterval(T + 1_000, None)],
    )
    assert [bar.halted for bar in result.bars] == [False, True, True]


def test_duplicate_provider_second_is_not_silently_kept_last():
    with pytest.raises(ValueError, match="duplicate second"):
        adapt_second_bars(
            frame(0, 0), session_open_t=T, session_close_t=T + 2_000,
        )


def test_bad_timestamp_and_active_price_are_rejected():
    bad_time = frame(0)
    bad_time.loc[0, "t"] += 500
    with pytest.raises(ValueError, match="whole-second"):
        adapt_second_bars(bad_time, session_open_t=T, session_close_t=T + 2_000)
    bad_price = frame(0)
    bad_price.loc[0, "l"] = 11
    with pytest.raises(ValueError, match="inconsistent OHLC"):
        adapt_second_bars(bad_price, session_open_t=T, session_close_t=T + 2_000)


def test_halt_intervals_cannot_overlap():
    with pytest.raises(ValueError, match="non-overlapping"):
        adapt_second_bars(
            frame(0), session_open_t=T, session_close_t=T + 3_000,
            halts=[
                HaltInterval(T, T + 2_000),
                HaltInterval(T + 1_000, T + 3_000),
            ],
        )

