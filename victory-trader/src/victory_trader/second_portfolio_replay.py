"""Chronological account bridge for frozen one-second reference trajectories.

Future bars may schedule evaluation events, but admission and order size use only
cash, reservations, and observed marks available at each decision timestamp.
This remains a synthetic research ledger, not a brokerage fill simulator.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping
from dataclasses import dataclass
from math import floor, isfinite

from .causal_second_execution import (
    RiskRule,
    SecondBar,
    replay_long,
    risk_sized_whole_shares,
)
from .execution_costs import ExecutionScenario, modeled_buy_fill


@dataclass(frozen=True)
class EntryAttempt:
    ticker: str
    decision_t: int
    decision_price: float  # already-observed causal price, never the future entry open
    session_close_t: int  # exclusive; no silent after-hours/overnight fill
    rule: RiskRule


@dataclass(frozen=True)
class PortfolioReplay:
    records: tuple[dict[str, object], ...]
    equity_curve: tuple[dict[str, float | int], ...]
    summary: dict[str, float | int | bool]


def replay_portfolio(
    attempts: list[EntryAttempt],
    bars_by_ticker: Mapping[str, list[SecondBar]],
    scenario: ExecutionScenario,
    *,
    initial_cash: float = 10_000.0,
    max_positions: int = 5,
    risk_fraction: float = 0.01,
    max_allocation_fraction: float = 0.20,
    latency_ms: int = 0,
    entry_expiry_ms: int = 5_000,
    liquidity_share_caps: Mapping[str, int] | None = None,
) -> PortfolioReplay:
    """Replay frozen entry attempts with a shared, marked cash account.

    Events at equal timestamps are ordered: expiry, completed-bar mark,
    decision, exit open, entry open. A decision therefore cannot spend proceeds
    from an exit at that same as-yet-unobserved open. Accepted orders reserve
    decision-time cash; a worse entry open may make the fixed order unaffordable.
    """
    scenario.validate()
    if not isfinite(initial_cash) or initial_cash <= 0 or max_positions < 1:
        raise ValueError("initial cash and position capacity must be positive")
    if not (0 < risk_fraction < 1 and 0 < max_allocation_fraction <= 1):
        raise ValueError("invalid risk or allocation fraction")
    if latency_ms < 0 or entry_expiry_ms < 0:
        raise ValueError("latency and expiry must be non-negative")
    keys = [(a.ticker, a.decision_t) for a in attempts]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate ticker/decision timestamp")
    for attempt in attempts:
        attempt.rule.validate()
        if not attempt.ticker or attempt.decision_t < 0:
            raise ValueError("invalid ticker or decision timestamp")
        if attempt.session_close_t <= attempt.decision_t:
            raise ValueError("session close must follow decision")
        if not isfinite(attempt.decision_price) or attempt.decision_price <= 0:
            raise ValueError("decision price must be a positive causal reference")

    events: list[tuple[int, int, int, str, int, object]] = []
    serial = 0

    def schedule(t: int, priority: int, kind: str, idx: int, data: object = None) -> None:
        nonlocal serial
        serial += 1
        heapq.heappush(events, (t, priority, serial, kind, idx, data))

    trajectories = []
    filtered_bars: list[list[SecondBar]] = []
    records: list[dict[str, object]] = []
    for idx, attempt in enumerate(attempts):
        bars = [
            bar for bar in bars_by_ticker.get(attempt.ticker, [])
            if bar.t < attempt.session_close_t
        ]
        filtered_bars.append(bars)
        path = replay_long(
            bars,
            decision_t=attempt.decision_t,
            rule=attempt.rule,
            scenario=scenario,
            latency_ms=latency_ms,
            entry_expiry_ms=entry_expiry_ms,
        )
        trajectories.append(path)
        records.append({
            "ticker": attempt.ticker,
            "decision_t": attempt.decision_t,
            "status": "not_processed",
            "path_status": path.status,
            "path_reason": path.reason,
        })
        schedule(attempt.decision_t, 2, "decision", idx)
        if path.entry is None:
            schedule(
                attempt.decision_t + latency_ms + entry_expiry_ms,
                0, "expire", idx,
            )
            continue
        schedule(path.entry.fill_t, 4, "entry", idx, path.entry)

    cash = float(initial_cash)
    reserved_cash = 0.0
    positions: dict[str, dict[str, float | int]] = {}
    pending: dict[str, tuple[int, float, int]] = {}
    curve: list[dict[str, float | int]] = []
    realized_pnl = 0.0
    turnover = 0.0
    peak_equity = cash
    max_drawdown = 0.0

    def equity() -> float:
        return cash + sum(float(p["shares"]) * float(p["mark"]) for p in positions.values())

    while events:
        t, _, _, kind, idx, data = heapq.heappop(events)
        attempt = attempts[idx]
        ticker = attempt.ticker
        record = records[idx]
        if kind == "expire":
            held = pending.get(ticker)
            if held is not None and held[0] == idx:
                reserved_cash -= held[1]
                del pending[ticker]
                record["status"] = "entry_unavailable"
        elif kind == "mark":
            position = positions.get(ticker)
            if position is not None and position["idx"] == idx:
                position["mark"] = float(data)
                position["mark_t"] = t
        elif kind == "decision":
            if ticker in positions or ticker in pending:
                record["status"] = "blocked_open_position"
            elif len(positions) + len(pending) >= max_positions:
                record["status"] = "blocked_capacity"
            elif cash - reserved_cash <= 0:
                record["status"] = "blocked_cash"
            else:
                shares = risk_sized_whole_shares(
                    equity=equity(),
                    cash=cash - reserved_cash,
                    entry_reference=attempt.decision_price,
                    rule=attempt.rule,
                    scenario=scenario,
                    risk_fraction=risk_fraction,
                    max_allocation_fraction=max_allocation_fraction,
                    max_liquidity_shares=(
                        liquidity_share_caps.get(ticker)
                        if liquidity_share_caps is not None else None
                    ),
                )
                if shares < 2:  # a partial take needs at least two whole shares
                    record["status"] = "size_below_two_shares"
                else:
                    reserve = shares * modeled_buy_fill(attempt.decision_price, scenario)
                    reserved_cash += reserve
                    pending[ticker] = (idx, reserve, shares)
                    record.update(status="pending_entry", shares=shares)
        elif kind == "entry":
            held = pending.get(ticker)
            if held is not None and held[0] == idx:
                _, reserve, shares = held
                reserved_cash -= reserve
                del pending[ticker]
                cost = shares * data.modeled_price
                if cost > cash - reserved_cash + 1e-9:
                    record["status"] = "entry_open_unaffordable"
                else:
                    cash -= cost
                    turnover += cost
                    positions[ticker] = {
                        "idx": idx,
                        "shares": shares,
                        "initial_shares": shares,
                        "entry_fill": data.modeled_price,
                        "initial_cost": cost,
                        "proceeds": 0.0,
                        "mark": data.reference_price,
                        "mark_t": t,
                    }
                    record.update(status="open_unresolved", entry_t=t, entry_cost=cost)
                    path = trajectories[idx]
                    for fill in path.exits:
                        schedule(fill.fill_t, 3, "exit", idx, fill)
                    terminal_t = (
                        path.exits[-1].fill_t if path.status == "closed" else None
                    )
                    for bar in filtered_bars[idx]:
                        mark_t = bar.t + 1_000
                        if bar.halted or bar.t < path.entry.fill_t:
                            continue
                        if terminal_t is not None and mark_t >= terminal_t:
                            continue
                        if (
                            path.status == "ambiguous"
                            and path.last_mark_t is not None
                            and mark_t > path.last_mark_t
                        ):
                            continue
                        schedule(mark_t, 1, "mark", idx, bar.c)
        elif kind == "exit":
            position = positions.get(ticker)
            if position is not None and position["idx"] == idx:
                remaining = int(position["shares"])
                if data.reason == "partial_take":
                    sold = min(remaining - 1, max(1, floor(
                        int(position["initial_shares"]) * attempt.rule.partial_fraction
                    )))
                else:
                    sold = remaining
                proceeds = sold * data.modeled_price * (
                    1 - scenario.sell_fee_bps / 10_000
                )
                cash += proceeds
                turnover += proceeds
                realized_pnl += proceeds - sold * float(position["entry_fill"])
                position["shares"] = remaining - sold
                position["proceeds"] = float(position["proceeds"]) + proceeds
                position["mark"] = data.reference_price
                position["mark_t"] = t
                if position["shares"] == 0:
                    record.update(
                        status="closed",
                        exit_t=t,
                        net_return_pct=(
                            float(position["proceeds"]) / float(position["initial_cost"]) - 1
                        ) * 100,
                    )
                    del positions[ticker]
                else:
                    record["status"] = "partial_open"

        marked = equity()
        peak_equity = max(peak_equity, marked)
        max_drawdown = max(max_drawdown, (1 - marked / peak_equity) * 100)
        curve.append({
            "t": t,
            "cash": cash,
            "reserved_cash": reserved_cash,
            "marked_equity": marked,
            "open_positions": len(positions),
        })
        if cash < -1e-7 or reserved_cash > cash + 1e-7:
            raise AssertionError("cash or reservations violated")

    for ticker, position in positions.items():
        record = records[int(position["idx"])]
        if trajectories[int(position["idx"])].status == "ambiguous":
            record["status"] = "ambiguous_open_unresolved"
        else:
            record["status"] = "open_unresolved"
        record["last_mark_t"] = int(position["mark_t"])
        record["last_mark"] = float(position["mark"])

    summary: dict[str, float | int | bool] = {
        "initial_cash": float(initial_cash),
        "ending_cash": cash,
        "ending_marked_equity": equity(),
        "realized_pnl": realized_pnl,
        "observed_mark_drawdown_pct": max_drawdown,
        "turnover": turnover,
        "attempts": len(attempts),
        "closed": sum(r["status"] == "closed" for r in records),
        "unresolved": len(positions),
        "entry_unavailable": sum(r["status"] == "entry_unavailable" for r in records),
        "blocked": sum(str(r["status"]).startswith("blocked_") for r in records),
        "complete_coverage": (
            not positions and all(r["status"] != "entry_unavailable" for r in records)
        ),
        "promotion_allowed": False,
    }
    return PortfolioReplay(tuple(records), tuple(curve), summary)

