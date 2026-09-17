import re

import pandas as pd

from victory_trader.model_schema import MODEL_FEATURE_COLUMNS, build_model_frame, validate_feature_allowlist


FORWARD_RETURN_PATTERN = re.compile(r"^return_\d+m")


def test_feature_allowlist_contains_no_future_or_execution_columns():
    validate_feature_allowlist()
    forbidden_exact = {"day_high", "entry_price"}
    assert not any(column in forbidden_exact for column in MODEL_FEATURE_COLUMNS)
    assert not any(column.startswith("halt_within") for column in MODEL_FEATURE_COLUMNS)
    assert not any(FORWARD_RETURN_PATTERN.match(column) for column in MODEL_FEATURE_COLUMNS)


def test_build_model_frame_drops_non_features_and_missing_targets():
    rows = []
    for target in (1.5, None):
        row = {column: 1.0 for column in MODEL_FEATURE_COLUMNS}
        row.update(
            {
                "security_type": "CS",
                "primary_exchange": "XNAS",
                "is_premarket": False,
                "is_regular_session": True,
                "is_after_hours": False,
                "ticker": "AAA",
                "day_high": 999.0,
                "return_5m_base_net_return_pct": target,
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)

    x, y = build_model_frame(frame)
    assert list(x.columns) == list(MODEL_FEATURE_COLUMNS)
    assert len(x) == 1
    assert len(y) == 1
    assert "ticker" not in x.columns
    assert "day_high" not in x.columns
