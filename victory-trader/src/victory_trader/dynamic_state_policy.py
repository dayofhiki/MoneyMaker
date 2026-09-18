from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .execution_costs import (
    DEFAULT_EXECUTION_SCENARIOS,
    modeled_buy_fill,
    modeled_sell_fill,
)
from .state_value_model import (
    _eligible,
    _modeled_net_from_predicted_gross,
    feature_frame,
    train_fold_model,
)


ENTRY_HORIZON = 10
EXIT_HORIZON = 5
ENTRY_QUANTILE = 0.99
MAX_HOLD_MINUTES = 30
MINUTE_MS = 60_000


@dataclass(frozen=True)
class Trade:
    trading_day: str
    ticker: str
    entry_decision_t: int
    entry_t: int
    exit_decision_t: int | None
    exit_t: int | None
    entry_reference_price: float
    exit_reference_price: float
    exit_reason: str
    hold_minutes: float
    gross_return_pct: float
    light_net_return_pct: float
    base_net_return_pct: float
    stress_net_return_pct: float
    entry_predicted_base_10m_pct: float
    exit_predicted_gross_5m_pct: float | None

    def to_dict(self) -> dict[str, object]:
        return vars(self).copy()


def _scenario(name: str):
    return next(s for s in DEFAULT_EXECUTION_SCENARIOS if s.name == name)


def _net_return_pct(
    entry_reference_price: float,
    exit_reference_price: float,
    scenario_name: str,
) -> float:
    scenario = _scenario(scenario_name)
    buy = modeled_buy_fill(entry_reference_price, scenario)
    sell = modeled_sell_fill(exit_reference_price, scenario)
    sell *= 1.0 - scenario.sell_fee_bps / 10_000.0
    return (sell / buy - 1.0) * 100.0


def score_holdout(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    model_10 = train_fold_model(train, ENTRY_HORIZON)
    model_5 = train_fold_model(train, EXIT_HORIZON)

    scored = holdout.copy()

    eligible_10 = _eligible(scored, ENTRY_HORIZON)
    eligible_5 = _eligible(scored, EXIT_HORIZON)

    scored["predicted_gross_10m_pct"] = np.nan
    scored["predicted_base_10m_pct"] = np.nan
    scored["predicted_gross_5m_pct"] = np.nan

    if eligible_10.any():
        X10 = feature_frame(scored.loc[eligible_10]).reindex(
            columns=model_10.feature_columns
        )
        pred10 = model_10.model.predict(X10)
        pred10_base = _modeled_net_from_predicted_gross(
            scored.loc[eligible_10, "entry_price"],
            pred10,
            _scenario("base"),
        )
        scored.loc[eligible_10, "predicted_gross_10m_pct"] = pred10
        scored.loc[eligible_10, "predicted_base_10m_pct"] = pred10_base

    if eligible_5.any():
        X5 = feature_frame(scored.loc[eligible_5]).reindex(
            columns=model_5.feature_columns
        )
        pred5 = model_5.model.predict(X5)
        scored.loc[eligible_5, "predicted_gross_5m_pct"] = pred5

    cutoff = float(model_10.calibration_cutoffs[ENTRY_QUANTILE])
    return scored, cutoff


def _forced_close_reference(day_frame: pd.DataFrame, ticker: str) -> tuple[int, float] | None:
    rows = day_frame.loc[
        day_frame["ticker"].astype(str).eq(ticker)
        & pd.to_numeric(day_frame["c"], errors="coerce").notna()
    ].sort_values("t")
    if rows.empty:
        return None
    row = rows.iloc[-1]
    return int(row["t"]), float(row["c"])


def simulate_one_position(
    scored: pd.DataFrame,
    *,
    entry_cutoff: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades: list[Trade] = []
    equity_rows: list[dict[str, object]] = []
    base_equity = 1.0

    for trading_day, day_frame in scored.groupby("trading_day", sort=True):
        day_frame = day_frame.sort_values(["t", "ticker"], kind="stable").copy()
        position: dict[str, object] | None = None
        skip_entry_t: int | None = None
        day_start_equity = base_equity

        for timestamp, minute_frame in day_frame.groupby("t", sort=True):
            t = int(timestamp)

            if position is not None:
                ticker = str(position["ticker"])
                held_rows = minute_frame.loc[
                    minute_frame["ticker"].astype(str).eq(ticker)
                ]
                exit_reason = None
                exit_row = None

                if not held_rows.empty:
                    row = held_rows.iloc[-1]
                    pred5 = pd.to_numeric(
                        pd.Series([row.get("predicted_gross_5m_pct")]),
                        errors="coerce",
                    ).iloc[0]
                    hod_distance = pd.to_numeric(
                        pd.Series([row.get("hod_distance_pct")]),
                        errors="coerce",
                    ).iloc[0]
                    above_vwap = bool(row.get("above_regular_vwap"))
                    hold_minutes = (
                        t - int(position["entry_decision_t"])
                    ) / MINUTE_MS

                    if pd.notna(pred5) and float(pred5) <= 0.0:
                        exit_reason = "predicted_5m_nonpositive"
                    elif (
                        pd.notna(hod_distance)
                        and float(hod_distance) <= -5.0
                        and not above_vwap
                    ):
                        exit_reason = "failure_state"
                    elif hold_minutes >= MAX_HOLD_MINUTES:
                        exit_reason = "max_hold"

                    if exit_reason is not None:
                        exit_reference = pd.to_numeric(
                            pd.Series([row.get("entry_price")]),
                            errors="coerce",
                        ).iloc[0]
                        if pd.notna(exit_reference) and float(exit_reference) > 0:
                            exit_row = row

                if exit_reason is not None and exit_row is not None:
                    entry_reference = float(position["entry_reference_price"])
                    exit_reference = float(exit_row["entry_price"])
                    gross = (exit_reference / entry_reference - 1.0) * 100.0
                    light = _net_return_pct(entry_reference, exit_reference, "light")
                    base = _net_return_pct(entry_reference, exit_reference, "base")
                    stress = _net_return_pct(entry_reference, exit_reference, "stress")
                    base_equity *= 1.0 + base / 100.0

                    trades.append(
                        Trade(
                            trading_day=str(trading_day),
                            ticker=ticker,
                            entry_decision_t=int(position["entry_decision_t"]),
                            entry_t=int(position["entry_t"]),
                            exit_decision_t=t,
                            exit_t=t + MINUTE_MS,
                            entry_reference_price=entry_reference,
                            exit_reference_price=exit_reference,
                            exit_reason=str(exit_reason),
                            hold_minutes=(
                                t - int(position["entry_decision_t"])
                            )
                            / MINUTE_MS,
                            gross_return_pct=gross,
                            light_net_return_pct=light,
                            base_net_return_pct=base,
                            stress_net_return_pct=stress,
                            entry_predicted_base_10m_pct=float(
                                position["entry_predicted_base_10m_pct"]
                            ),
                            exit_predicted_gross_5m_pct=(
                                None
                                if pd.isna(exit_row["predicted_gross_5m_pct"])
                                else float(exit_row["predicted_gross_5m_pct"])
                            ),
                        )
                    )
                    position = None
                    skip_entry_t = t

            if position is None and skip_entry_t != t:
                candidates = minute_frame.loc[
                    pd.to_numeric(
                        minute_frame["predicted_base_10m_pct"], errors="coerce"
                    ).ge(entry_cutoff)
                    & pd.to_numeric(
                        minute_frame["entry_price"], errors="coerce"
                    ).notna()
                ].copy()
                if not candidates.empty:
                    candidates = candidates.sort_values(
                        "predicted_base_10m_pct",
                        ascending=False,
                        kind="stable",
                    )
                    row = candidates.iloc[0]
                    entry_reference = float(row["entry_price"])
                    position = {
                        "ticker": str(row["ticker"]),
                        "entry_decision_t": t,
                        "entry_t": t + MINUTE_MS,
                        "entry_reference_price": entry_reference,
                        "entry_predicted_base_10m_pct": float(
                            row["predicted_base_10m_pct"]
                        ),
                    }

        if position is not None:
            ticker = str(position["ticker"])
            forced = _forced_close_reference(day_frame, ticker)
            if forced is not None:
                exit_t, exit_reference = forced
                entry_reference = float(position["entry_reference_price"])
                gross = (exit_reference / entry_reference - 1.0) * 100.0
                light = _net_return_pct(entry_reference, exit_reference, "light")
                base = _net_return_pct(entry_reference, exit_reference, "base")
                stress = _net_return_pct(entry_reference, exit_reference, "stress")
                base_equity *= 1.0 + base / 100.0
                trades.append(
                    Trade(
                        trading_day=str(trading_day),
                        ticker=ticker,
                        entry_decision_t=int(position["entry_decision_t"]),
                        entry_t=int(position["entry_t"]),
                        exit_decision_t=None,
                        exit_t=exit_t,
                        entry_reference_price=entry_reference,
                        exit_reference_price=exit_reference,
                        exit_reason="session_close",
                        hold_minutes=(
                            exit_t - int(position["entry_t"])
                        )
                        / MINUTE_MS,
                        gross_return_pct=gross,
                        light_net_return_pct=light,
                        base_net_return_pct=base,
                        stress_net_return_pct=stress,
                        entry_predicted_base_10m_pct=float(
                            position["entry_predicted_base_10m_pct"]
                        ),
                        exit_predicted_gross_5m_pct=None,
                    )
                )

        equity_rows.append(
            {
                "trading_day": str(trading_day),
                "start_equity": day_start_equity,
                "end_equity": base_equity,
                "daily_return_pct": (
                    base_equity / day_start_equity - 1.0
                )
                * 100.0
                if day_start_equity > 0
                else np.nan,
            }
        )

    return pd.DataFrame([trade.to_dict() for trade in trades]), pd.DataFrame(equity_rows)


def _max_drawdown_pct(equity: pd.Series) -> float:
    values = pd.to_numeric(equity, errors="coerce").dropna()
    if values.empty:
        return np.nan
    running_peak = values.cummax()
    drawdown = values / running_peak - 1.0
    return float(drawdown.min() * 100.0)


def metrics(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    *,
    month: str,
    entry_cutoff: float,
) -> dict[str, object]:
    base = pd.to_numeric(trades.get("base_net_return_pct"), errors="coerce")
    gross = pd.to_numeric(trades.get("gross_return_pct"), errors="coerce")
    light = pd.to_numeric(trades.get("light_net_return_pct"), errors="coerce")
    stress = pd.to_numeric(trades.get("stress_net_return_pct"), errors="coerce")
    hold = pd.to_numeric(trades.get("hold_minutes"), errors="coerce")

    ending_equity = (
        float(equity.iloc[-1]["end_equity"]) if not equity.empty else 1.0
    )
    return {
        "month": month,
        "entry_quantile": ENTRY_QUANTILE,
        "entry_cutoff_predicted_base_pct": entry_cutoff,
        "trades": int(len(trades)),
        "days": int(equity["trading_day"].nunique()) if not equity.empty else 0,
        "trades_per_day": (
            float(len(trades) / equity["trading_day"].nunique())
            if not equity.empty and equity["trading_day"].nunique()
            else np.nan
        ),
        "median_hold_minutes": float(hold.median()) if len(trades) else np.nan,
        "gross_mean_pct": float(gross.mean()) if len(trades) else np.nan,
        "light_mean_pct": float(light.mean()) if len(trades) else np.nan,
        "base_mean_pct": float(base.mean()) if len(trades) else np.nan,
        "stress_mean_pct": float(stress.mean()) if len(trades) else np.nan,
        "base_positive_rate": float((base > 0).mean()) if len(trades) else np.nan,
        "base_p05_pct": float(base.quantile(0.05)) if len(trades) else np.nan,
        "base_worst_trade_pct": float(base.min()) if len(trades) else np.nan,
        "ending_base_equity": ending_equity,
        "base_month_return_pct": (ending_equity - 1.0) * 100.0,
        "base_max_drawdown_pct": _max_drawdown_pct(equity["end_equity"])
        if not equity.empty
        else np.nan,
        "worst_day_pct": (
            float(
                pd.to_numeric(
                    equity["daily_return_pct"], errors="coerce"
                ).min()
            )
            if not equity.empty
            else np.nan
        ),
    }


def run_lomo(
    monthly_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    trade_frames = []
    equity_frames = []

    for holdout_month, holdout in monthly_frames.items():
        train = pd.concat(
            [
                frame
                for month, frame in monthly_frames.items()
                if month != holdout_month
            ],
            ignore_index=True,
        )
        scored, cutoff = score_holdout(train, holdout)
        trades, equity = simulate_one_position(scored, entry_cutoff=cutoff)

        metric_rows.append(
            metrics(
                trades,
                equity,
                month=holdout_month,
                entry_cutoff=cutoff,
            )
        )
        if not trades.empty:
            trades["month"] = holdout_month
            trade_frames.append(trades)
        if not equity.empty:
            equity["month"] = holdout_month
            equity_frames.append(equity)

    all_trades = (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame()
    )
    all_equity = (
        pd.concat(equity_frames, ignore_index=True)
        if equity_frames
        else pd.DataFrame()
    )
    return pd.DataFrame(metric_rows), all_trades, all_equity


def render_report(metrics_frame: pd.DataFrame) -> str:
    return "\n".join(
        [
            "=== MoneyMaker Dynamic State Policy v0.1 ===",
            "policy=one market-wide long position; entry on top-1% predicted 10m base score",
            "hold=continue while predicted 5m gross > 0",
            "exit=predicted 5m gross <=0 OR failure state OR 30m max hold",
            "execution=all normal decisions at exact next-minute open; session-close forced liquidation",
            "validation=leave-one-month-out on already-seen Jan-Mar 2026",
            "NOTE=pre-registered development policy; not external proof",
            "",
            metrics_frame.to_string(index=False),
        ]
    )


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.dynamic_state_policy"
    )
    parser.add_argument(
        "--dataset", action="append", type=_parse_dataset, required=True
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--trades-csv", type=Path, required=True)
    parser.add_argument("--equity-csv", type=Path, required=True)
    args = parser.parse_args()

    monthly = {
        label: pd.read_parquet(path)
        for label, path in args.dataset
    }
    metric_frame, trades, equity = run_lomo(monthly)
    report = render_report(metric_frame)
    print(report)

    for path in (
        args.report,
        args.metrics_csv,
        args.trades_csv,
        args.equity_csv,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    metric_frame.to_csv(args.metrics_csv, index=False)
    trades.to_csv(args.trades_csv, index=False)
    equity.to_csv(args.equity_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
