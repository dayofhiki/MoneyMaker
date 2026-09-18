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
from .state_ev_risk_model import score_states, train_fold


MINUTE_MS = 60_000
ENTRY_EV_GATES_PCT = (0.00, 0.25, 0.50)
ENTRY_RISK_CEILING = 0.50
EXIT_RISK_CEILING = 0.60
MIN_HOLD_MINUTES = 5
MAX_HOLD_MINUTES = 20
HARD_STOP_GROSS_PCT = -5.0
COOLDOWN_MINUTES = 2
TARGET_ACCOUNT_RISK_PCT = 0.50
MAX_POSITION_FRACTION = 0.10


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
    position_fraction: float
    account_base_return_pct: float
    entry_predicted_base_10m_pct: float
    entry_predicted_risk: float
    exit_predicted_gross_10m_pct: float | None
    exit_predicted_risk: float | None

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


def _position_fraction() -> float:
    stop_size = abs(HARD_STOP_GROSS_PCT)
    if stop_size <= 0:
        return MAX_POSITION_FRACTION
    fraction = TARGET_ACCOUNT_RISK_PCT / stop_size
    return float(min(max(fraction, 0.0), MAX_POSITION_FRACTION))


def _forced_close_reference(
    day_frame: pd.DataFrame,
    ticker: str,
) -> tuple[int, float] | None:
    rows = day_frame.loc[
        day_frame["ticker"].astype(str).eq(ticker)
        & pd.to_numeric(day_frame["c"], errors="coerce").notna()
    ].sort_values("t")
    if rows.empty:
        return None
    row = rows.iloc[-1]
    return int(row["t"]), float(row["c"])


def score_holdout(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
) -> pd.DataFrame:
    fold = train_fold(train)
    return score_states(holdout, fold)


def _exit_signal(
    row: pd.Series,
    *,
    entry_reference: float,
    hold_minutes: float,
) -> str | None:
    close = pd.to_numeric(pd.Series([row.get("c")]), errors="coerce").iloc[0]
    hod_distance = pd.to_numeric(
        pd.Series([row.get("hod_distance_pct")]), errors="coerce"
    ).iloc[0]
    pred_gross = pd.to_numeric(
        pd.Series([row.get("predicted_mean_gross_10m_pct")]),
        errors="coerce",
    ).iloc[0]
    pred_risk = pd.to_numeric(
        pd.Series([row.get("predicted_severe_mae_prob")]),
        errors="coerce",
    ).iloc[0]
    above_vwap = bool(row.get("above_regular_vwap"))

    if pd.notna(close) and entry_reference > 0:
        unrealized_gross = (float(close) / entry_reference - 1.0) * 100.0
        if unrealized_gross <= HARD_STOP_GROSS_PCT:
            return "hard_stop"

    if hold_minutes >= MAX_HOLD_MINUTES:
        return "max_hold"

    # Only the hard stop may override the minimum hold. Structural/model
    # deterioration is deliberately hysteretic to avoid paying round-trip
    # friction on one-minute state flicker.
    if hold_minutes < MIN_HOLD_MINUTES:
        return None

    if (
        pd.notna(hod_distance)
        and float(hod_distance) <= -5.0
        and not above_vwap
    ):
        return "failure_state"

    if pd.notna(pred_gross) and float(pred_gross) <= 0.0:
        return "expected_gross_nonpositive"

    if pd.notna(pred_risk) and float(pred_risk) >= EXIT_RISK_CEILING:
        return "risk_spike"

    return None


def simulate_one_position(
    scored: pd.DataFrame,
    *,
    entry_ev_gate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades: list[Trade] = []
    equity_rows: list[dict[str, object]] = []
    base_equity = 1.0
    position_fraction = _position_fraction()

    for trading_day, day_frame in scored.groupby("trading_day", sort=True):
        day_frame = day_frame.sort_values(["t", "ticker"], kind="stable").copy()
        position: dict[str, object] | None = None
        cooldown_until_t = -1
        day_start_equity = base_equity

        for timestamp, minute_frame in day_frame.groupby("t", sort=True):
            t = int(timestamp)

            if position is not None:
                ticker = str(position["ticker"])
                held_rows = minute_frame.loc[
                    minute_frame["ticker"].astype(str).eq(ticker)
                ]
                if not held_rows.empty:
                    row = held_rows.iloc[-1]
                    hold_minutes = (
                        t - int(position["entry_decision_t"])
                    ) / MINUTE_MS
                    reason = _exit_signal(
                        row,
                        entry_reference=float(position["entry_reference_price"]),
                        hold_minutes=hold_minutes,
                    )
                    if reason is not None:
                        exit_reference = pd.to_numeric(
                            pd.Series([row.get("entry_price")]),
                            errors="coerce",
                        ).iloc[0]
                        if pd.notna(exit_reference) and float(exit_reference) > 0:
                            entry_reference = float(
                                position["entry_reference_price"]
                            )
                            exit_reference = float(exit_reference)
                            gross = (
                                exit_reference / entry_reference - 1.0
                            ) * 100.0
                            light = _net_return_pct(
                                entry_reference, exit_reference, "light"
                            )
                            base = _net_return_pct(
                                entry_reference, exit_reference, "base"
                            )
                            stress = _net_return_pct(
                                entry_reference, exit_reference, "stress"
                            )
                            account_return = position_fraction * base
                            base_equity *= 1.0 + account_return / 100.0

                            exit_pred_gross = pd.to_numeric(
                                pd.Series(
                                    [
                                        row.get(
                                            "predicted_mean_gross_10m_pct"
                                        )
                                    ]
                                ),
                                errors="coerce",
                            ).iloc[0]
                            exit_pred_risk = pd.to_numeric(
                                pd.Series(
                                    [row.get("predicted_severe_mae_prob")]
                                ),
                                errors="coerce",
                            ).iloc[0]

                            trades.append(
                                Trade(
                                    trading_day=str(trading_day),
                                    ticker=ticker,
                                    entry_decision_t=int(
                                        position["entry_decision_t"]
                                    ),
                                    entry_t=int(position["entry_t"]),
                                    exit_decision_t=t,
                                    exit_t=t + MINUTE_MS,
                                    entry_reference_price=entry_reference,
                                    exit_reference_price=exit_reference,
                                    exit_reason=reason,
                                    hold_minutes=hold_minutes,
                                    gross_return_pct=gross,
                                    light_net_return_pct=light,
                                    base_net_return_pct=base,
                                    stress_net_return_pct=stress,
                                    position_fraction=position_fraction,
                                    account_base_return_pct=account_return,
                                    entry_predicted_base_10m_pct=float(
                                        position[
                                            "entry_predicted_base_10m_pct"
                                        ]
                                    ),
                                    entry_predicted_risk=float(
                                        position["entry_predicted_risk"]
                                    ),
                                    exit_predicted_gross_10m_pct=(
                                        None
                                        if pd.isna(exit_pred_gross)
                                        else float(exit_pred_gross)
                                    ),
                                    exit_predicted_risk=(
                                        None
                                        if pd.isna(exit_pred_risk)
                                        else float(exit_pred_risk)
                                    ),
                                )
                            )
                            position = None
                            cooldown_until_t = (
                                t + COOLDOWN_MINUTES * MINUTE_MS
                            )

            if position is None and t >= cooldown_until_t:
                ev = pd.to_numeric(
                    minute_frame["predicted_mean_base_10m_pct"],
                    errors="coerce",
                )
                risk = pd.to_numeric(
                    minute_frame["predicted_severe_mae_prob"],
                    errors="coerce",
                )
                candidates = minute_frame.loc[
                    ev.ge(entry_ev_gate)
                    & risk.le(ENTRY_RISK_CEILING)
                ].copy()
                if not candidates.empty:
                    candidates = candidates.sort_values(
                        [
                            "predicted_mean_base_10m_pct",
                            "predicted_severe_mae_prob",
                        ],
                        ascending=[False, True],
                        kind="stable",
                    )
                    row = candidates.iloc[0]
                    entry_reference = pd.to_numeric(
                        pd.Series([row.get("entry_price")]),
                        errors="coerce",
                    ).iloc[0]
                    if pd.notna(entry_reference) and float(entry_reference) > 0:
                        position = {
                            "ticker": str(row["ticker"]),
                            "entry_decision_t": t,
                            "entry_t": t + MINUTE_MS,
                            "entry_reference_price": float(entry_reference),
                            "entry_predicted_base_10m_pct": float(
                                row["predicted_mean_base_10m_pct"]
                            ),
                            "entry_predicted_risk": float(
                                row["predicted_severe_mae_prob"]
                            ),
                        }

        if position is not None:
            ticker = str(position["ticker"])
            forced = _forced_close_reference(day_frame, ticker)
            if forced is not None:
                exit_t, exit_reference = forced
                entry_reference = float(position["entry_reference_price"])
                gross = (
                    exit_reference / entry_reference - 1.0
                ) * 100.0
                light = _net_return_pct(entry_reference, exit_reference, "light")
                base = _net_return_pct(entry_reference, exit_reference, "base")
                stress = _net_return_pct(
                    entry_reference, exit_reference, "stress"
                )
                account_return = position_fraction * base
                base_equity *= 1.0 + account_return / 100.0
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
                        position_fraction=position_fraction,
                        account_base_return_pct=account_return,
                        entry_predicted_base_10m_pct=float(
                            position["entry_predicted_base_10m_pct"]
                        ),
                        entry_predicted_risk=float(
                            position["entry_predicted_risk"]
                        ),
                        exit_predicted_gross_10m_pct=None,
                        exit_predicted_risk=None,
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

    return (
        pd.DataFrame([trade.to_dict() for trade in trades]),
        pd.DataFrame(equity_rows),
    )


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
    entry_ev_gate: float,
) -> dict[str, object]:
    if trades.empty:
        return {
            "month": month,
            "entry_ev_gate_pct": entry_ev_gate,
            "trades": 0,
        }

    base = pd.to_numeric(trades["base_net_return_pct"], errors="coerce")
    gross = pd.to_numeric(trades["gross_return_pct"], errors="coerce")
    light = pd.to_numeric(trades["light_net_return_pct"], errors="coerce")
    stress = pd.to_numeric(trades["stress_net_return_pct"], errors="coerce")
    hold = pd.to_numeric(trades["hold_minutes"], errors="coerce")
    account_return = pd.to_numeric(
        trades["account_base_return_pct"], errors="coerce"
    )
    ending_equity = (
        float(equity.iloc[-1]["end_equity"]) if not equity.empty else 1.0
    )

    return {
        "month": month,
        "entry_ev_gate_pct": entry_ev_gate,
        "position_fraction": _position_fraction(),
        "trades": int(len(trades)),
        "days": int(equity["trading_day"].nunique()) if not equity.empty else 0,
        "trades_per_day": (
            float(len(trades) / equity["trading_day"].nunique())
            if not equity.empty and equity["trading_day"].nunique()
            else np.nan
        ),
        "median_hold_minutes": float(hold.median()),
        "gross_mean_pct": float(gross.mean()),
        "light_mean_pct": float(light.mean()),
        "base_mean_pct": float(base.mean()),
        "stress_mean_pct": float(stress.mean()),
        "base_positive_rate": float((base > 0).mean()),
        "base_p05_pct": float(base.quantile(0.05)),
        "base_worst_trade_pct": float(base.min()),
        "account_trade_mean_pct": float(account_return.mean()),
        "ending_base_equity": ending_equity,
        "base_month_return_pct": (ending_equity - 1.0) * 100.0,
        "base_max_drawdown_pct": _max_drawdown_pct(equity["end_equity"])
        if not equity.empty
        else np.nan,
        "worst_day_pct": float(
            pd.to_numeric(equity["daily_return_pct"], errors="coerce").min()
        )
        if not equity.empty
        else np.nan,
    }


def run_lomo(
    monthly_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    trade_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []

    for holdout_month, holdout in monthly_frames.items():
        train = pd.concat(
            [
                frame
                for month, frame in monthly_frames.items()
                if month != holdout_month
            ],
            ignore_index=True,
        )
        scored = score_holdout(train, holdout)

        for gate in ENTRY_EV_GATES_PCT:
            trades, equity = simulate_one_position(
                scored,
                entry_ev_gate=gate,
            )
            metric_rows.append(
                metrics(
                    trades,
                    equity,
                    month=holdout_month,
                    entry_ev_gate=gate,
                )
            )
            if not trades.empty:
                item = trades.copy()
                item["month"] = holdout_month
                item["entry_ev_gate_pct"] = gate
                trade_frames.append(item)
            if not equity.empty:
                item = equity.copy()
                item["month"] = holdout_month
                item["entry_ev_gate_pct"] = gate
                equity_frames.append(item)

    return (
        pd.DataFrame(metric_rows),
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame(),
        pd.concat(equity_frames, ignore_index=True)
        if equity_frames
        else pd.DataFrame(),
    )


def summarize(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gate, group in metrics_frame.groupby("entry_ev_gate_pct", sort=True):
        valid = group.loc[group["trades"].ge(10)].copy()
        rows.append(
            {
                "entry_ev_gate_pct": float(gate),
                "months_tested": int(len(valid)),
                "min_trades": int(valid["trades"].min()) if len(valid) else 0,
                "median_trades_per_day": float(
                    valid["trades_per_day"].median()
                )
                if len(valid)
                else np.nan,
                "base_positive_months": int(
                    (valid["base_mean_pct"] > 0).sum()
                ),
                "account_positive_months": int(
                    (valid["base_month_return_pct"] > 0).sum()
                ),
                "median_base_mean_pct": float(
                    valid["base_mean_pct"].median()
                )
                if len(valid)
                else np.nan,
                "worst_base_mean_pct": float(
                    valid["base_mean_pct"].min()
                )
                if len(valid)
                else np.nan,
                "median_month_return_pct": float(
                    valid["base_month_return_pct"].median()
                )
                if len(valid)
                else np.nan,
                "worst_month_return_pct": float(
                    valid["base_month_return_pct"].min()
                )
                if len(valid)
                else np.nan,
                "median_max_drawdown_pct": float(
                    valid["base_max_drawdown_pct"].median()
                )
                if len(valid)
                else np.nan,
                "worst_max_drawdown_pct": float(
                    valid["base_max_drawdown_pct"].min()
                )
                if len(valid)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "account_positive_months",
            "base_positive_months",
            "median_month_return_pct",
            "worst_month_return_pct",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def render_report(
    metrics_frame: pd.DataFrame,
    summary: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "=== MoneyMaker Dynamic State Policy v0.3.1 ===",
            "entry=highest predicted 10m base EV across market, requiring EV gate and predicted severe-MAE probability <=50%",
            "positioning=one market-wide long position; 10% max allocation from 0.5% account-risk target and 5% hard stop",
            "hold=minimum 5m; after that continue while predicted 10m gross >0 and predicted severe-risk <60%",
            "exit=5% hard stop OR failure state OR post-min-hold EV deterioration/risk spike OR 20m max hold",
            "cooldown=2m after each exit before re-entry",
            "validation=leave-one-month-out on already-seen Jan-Mar 2026; exact next-minute-open execution",
            "NOTE=development policy only; no fresh month consumed",
            "",
            "=== Cross-month summary ===",
            summary.to_string(index=False),
            "",
            "=== Month-by-month ===",
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
        prog="python -m victory_trader.dynamic_state_policy_v03"
    )
    parser.add_argument(
        "--dataset", action="append", type=_parse_dataset, required=True
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--trades-csv", type=Path, required=True)
    parser.add_argument("--equity-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    args = parser.parse_args()

    monthly = {
        label: pd.read_parquet(path)
        for label, path in args.dataset
    }
    metric_frame, trades, equity = run_lomo(monthly)
    summary = summarize(metric_frame)
    report = render_report(metric_frame, summary)
    print(report)

    for path in (
        args.report,
        args.metrics_csv,
        args.trades_csv,
        args.equity_csv,
        args.summary_csv,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    metric_frame.to_csv(args.metrics_csv, index=False)
    trades.to_csv(args.trades_csv, index=False)
    equity.to_csv(args.equity_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
