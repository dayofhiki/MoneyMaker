import pandas as pd

from victory_trader.executable_entry_hurdle_decomposition import (
    add_hurdle_labels,
)


def test_hurdle_labels_keep_execution_failures_explicit():
    frame = pd.DataFrame(
        [
            {
                "replay_status": "entry_unavailable",
                "partial_take_reached": False,
                "policy_net_return_pct": None,
            },
            {
                "replay_status": "unresolved",
                "partial_take_reached": False,
                "policy_net_return_pct": None,
            },
            {
                "replay_status": "closed",
                "partial_take_reached": True,
                "policy_net_return_pct": 4.0,
            },
            {
                "replay_status": "closed",
                "partial_take_reached": False,
                "policy_net_return_pct": -3.0,
            },
        ]
    )
    result = add_hurdle_labels(frame)
    assert result["entry_fillable"].tolist() == [False, True, True, True]
    assert pd.isna(result.loc[0, "terminal_closed"])
    assert result.loc[1, "terminal_closed"] == False
    assert result.loc[2, "terminal_closed"] == True
    assert result.loc[2, "partial_take_captured"] == True
    assert result.loc[3, "positive_net"] == False
    assert pd.isna(result.loc[1, "positive_net"])
