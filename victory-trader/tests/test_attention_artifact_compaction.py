from pathlib import Path

import pandas as pd

from victory_trader.attention_artifact_compaction import (
    compact_attention_artifacts,
)


def test_compaction_keeps_focus_states_and_runner_crossings(tmp_path: Path):
    scan_dir = tmp_path / "scan" / "trading_day=2026-01-02"
    trace_dir = tmp_path / "trace" / "trading_day=2026-01-02"
    scan_dir.mkdir(parents=True)
    trace_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "trading_day": "2026-01-02",
                "t": 1,
                "ticker": "AAA",
                "runner_cross_now": False,
            },
            {
                "trading_day": "2026-01-02",
                "t": 2,
                "ticker": "BBB",
                "runner_cross_now": True,
            },
        ]
    ).to_parquet(scan_dir / "part-000.parquet", index=False)

    pd.DataFrame(
        [
            {
                "trading_day": "2026-01-02",
                "t": 1,
                "ticker": "AAA",
                "state": "scan",
            },
            {
                "trading_day": "2026-01-02",
                "t": 1,
                "ticker": "BBB",
                "state": "watch",
            },
            {
                "trading_day": "2026-01-02",
                "t": 2,
                "ticker": "CCC",
                "state": "hot",
            },
        ]
    ).to_parquet(trace_dir / "part-000.parquet", index=False)

    focus_output = tmp_path / "focus.parquet"
    runner_output = tmp_path / "runners.parquet"
    focus_rows, runner_rows = compact_attention_artifacts(
        tmp_path / "scan",
        tmp_path / "trace",
        focus_output=focus_output,
        runner_output=runner_output,
    )

    assert focus_rows == 2
    assert runner_rows == 1
    assert set(pd.read_parquet(focus_output)["ticker"]) == {"BBB", "CCC"}
    assert pd.read_parquet(runner_output)["ticker"].tolist() == ["BBB"]
