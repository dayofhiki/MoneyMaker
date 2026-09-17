from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .analytics import load_event_dataset
from .edge_discovery import (
    DEFAULT_DISCOVERY_FEATURES,
    summarize_chronological_feature_holdout,
    summarize_univariate_discovery,
)
from .path_enrichment import ENRICHMENT_FEATURE_COLUMNS


DISCOVERY_FEATURES = tuple(
    dict.fromkeys(DEFAULT_DISCOVERY_FEATURES + ENRICHMENT_FEATURE_COLUMNS)
)


def _render_table(frame: pd.DataFrame) -> str:
    return frame.to_string(index=False) if not frame.empty else "No eligible features available."


def _gross_alias(frame: pd.DataFrame, horizon_min: int) -> pd.DataFrame:
    gross = f"return_{horizon_min}m_pct"
    if gross not in frame.columns:
        raise ValueError(f"dataset missing gross executable target: {gross}")
    result = frame.copy()
    # Reuse the leakage-aware discovery functions without changing their target API.
    result[f"return_{horizon_min}m_gross_net_return_pct"] = pd.to_numeric(
        result[gross], errors="coerce"
    )
    return result


def render_extended_edge_discovery(
    frame: pd.DataFrame,
    *,
    horizon_min: int = 5,
) -> str:
    gross_frame = _gross_alias(frame, horizon_min)
    gross_screen = summarize_univariate_discovery(
        gross_frame,
        horizon_min=horizon_min,
        scenario="gross",
        features=DISCOVERY_FEATURES,
    )
    gross_holdout = summarize_chronological_feature_holdout(
        gross_frame,
        horizon_min=horizon_min,
        scenario="gross",
        features=DISCOVERY_FEATURES,
    )
    net_screen = summarize_univariate_discovery(
        frame,
        horizon_min=horizon_min,
        scenario="base",
        features=DISCOVERY_FEATURES,
    )
    net_holdout = summarize_chronological_feature_holdout(
        frame,
        horizon_min=horizon_min,
        scenario="base",
        features=DISCOVERY_FEATURES,
    )

    return "\n".join(
        [
            "=== MoneyMaker Extended Edge Discovery ===",
            f"horizon={horizon_min}m",
            f"pre_specified_features={len(DISCOVERY_FEATURES)}",
            "",
            "=== GROSS continuation: day/threshold-controlled screen ===",
            _render_table(gross_screen),
            "",
            "=== GROSS continuation: chronological 70/30 holdout ===",
            _render_table(gross_holdout),
            "",
            "=== BASE-NET continuation: day/threshold-controlled screen ===",
            _render_table(net_screen),
            "",
            "=== BASE-NET continuation: chronological 70/30 holdout ===",
            _render_table(net_holdout),
            "",
            "Interpretation rule: a feature that is strong only in BASE-NET but weak in GROSS may be describing execution-cost mechanics rather than continuation alpha.",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.extended_edge_discovery"
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    frame = load_event_dataset(args.path)
    report = render_extended_edge_discovery(frame, horizon_min=args.horizon)
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
