"""Causal one-second *reference* replay for a single long position.

One-second OHLC aggregates do not reveal within-second ordering or actual fills.
Decisions use a completed second; synthetic orders use the first later eligible
open. Ambiguous paths and missing exits remain unresolved, never profitable by
assumption. This module is infrastructure, not a validated trading policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import floor, isfinite
from typing import Literal

from .execution_costs import ExecutionScenario, modeled_buy_fill, modeled_sell_fill

SECOND_MS = 1_000


@dataclass(frozen=True)
class SecondBar:
    t: int  # UTC epoch milliseconds at the start of the second
    o: float
    h: float
    l: float
    c: float
    halted: bool = False


@dataclass(frozen=True)
class RiskRule:
    stop_pct: float
    take_pct: float
    trail_retrace_pct: float
    partial_fraction: float = 0.5
    max_hold_ms: int = 30 * 60_000

    def validate(self) -> None:
        if not (0 < self.stop_pct < 100 and self.take_pct > 0):
            raise ValueError("stop and take must be positive percentages")
        if not (0 < self.trail_retrace_pct < 100):
            raise ValueError("trail must be a proportional decline from price high")
        if not (0 < self.partial_fraction < 1 and self.max_hold_ms > 0):
            raise ValueError("invalid partial fraction or holding cap")


@dataclass(frozen=True)
class Fill:
    reason: str
    trigger_t: int
    fill_t: int
    reference_price: float
    modeled_price: float
    quantity: float


@dataclass(frozen=True)
class ReplayResult:
    status: Literal["closed", "unresolved", "ambiguous", "entry_unavailable"]
    reason: str
    entry: Fill | None
    exits: tuple[Fill, ...]
    last_mark_t: int | None
    last_mark_price: float | None
    net_return_pct: float | None


def risk_sized_whole_shares(
    *,
    equity: float,
    cash: float,
    entry_reference: float,
    rule: RiskRule,
    scenario: ExecutionScenario,
    risk_fraction: float = 0.01,
    max_allocation_fraction: float = 0.20,
    max_liquidity_shares: int | None = None,
) -> int:
    """Size from modeled stop loss, capped by cash, allocation and liquidity.

    The stop is a sizing reference, not a loss guarantee across a halt or gap.
    """
    rule.validate()
    scenario.validate()
    if any(not isfinite(x) or x <= 0 for x in (equity, cash, entry_reference)):
        raise ValueError("equity, cash and entry reference must be positive")
    if not (0 < risk_fraction < 1 and 0 < max_allocation_fraction <= 1):
        raise ValueError("invalid account risk or allocation fraction")
    if max_liquidity_shares is not None and max_liquidity_shares < 0:
        raise ValueError("liquidity share cap must be non-negative")
    buy = modeled_buy_fill(entry_reference, scenario)
    stop_reference = buy * (1 - rule.stop_pct / 100)
    stop_proceeds = modeled_sell_fill(stop_reference, scenario)
    stop_proceeds *= 1 - scenario.sell_fee_bps / 10_000
    loss_per_share = buy - stop_proceeds
    if loss_per_share <= 0:
        raise ValueError("modeled stop loss must be positive")
    shares = min(
        floor(equity * risk_fraction / loss_per_share),
        floor(min(cash, equity * max_allocation_fraction) / buy),
    )
    return min(shares, max_liquidity_shares) if max_liquidity_shares is not None else shares


def replay_long(
    bars: list[SecondBar],
    *,
    decision_t: int,
    rule: RiskRule,
    scenario: ExecutionScenario,
    latency_ms: int = 0,
    entry_expiry_ms: int = 5_000,
    quantity: float = 1.0,
) -> ReplayResult:
    """Replay one long entry and its exits using only completed-bar triggers.

    A bar at t is usable for an open fill at t, but its H/L/C are first known at
    t+1000. A triggered order can fill only at an open >= trigger+latency.
    Sparse seconds do not imply a trade, a halt, or an exit. Explicit halted bars
    cannot fill; a pending stop waits for a later resumed open at its actual price.
    """
    rule.validate()
    scenario.validate()
    if latency_ms < 0 or entry_expiry_ms < 0 or not isfinite(quantity) or quantity <= 0:
        raise ValueError("invalid latency, expiry or quantity")
    if any(b.t < 0 or b.t % SECOND_MS for b in bars):
        raise ValueError("bar times must be non-negative, whole-second UTC milliseconds")
    if any(a.t >= b.t for a, b in pairwise(bars)):
        raise ValueError("bars must have unique increasing timestamps")
    for b in bars:
        if b.halted:
            continue
        if any(not isfinite(x) or x <= 0 for x in (b.o, b.h, b.l, b.c)):
            raise ValueError("active bars require finite positive OHLC")
        if b.l > min(b.o, b.c) or b.h < max(b.o, b.c):
            raise ValueError("inconsistent OHLC")

    entry: Fill | None = None
    exits: list[Fill] = []
    pending: tuple[str, int, float] | None = None  # reason, trigger time, fraction
    remaining = float(quantity)
    partial_done = False
    peak_after_partial: float | None = None
    mark_t: int | None = None
    mark: float | None = None
    proceeds = 0.0

    def finish(status: Literal["closed", "unresolved", "ambiguous", "entry_unavailable"], reason: str) -> ReplayResult:
        net = None
        if status == "closed" and entry is not None:
            net = (proceeds / (entry.modeled_price * quantity) - 1) * 100
        return ReplayResult(status, reason, entry, tuple(exits), mark_t, mark, net)

    for bar in bars:
        if bar.halted:
            continue
        if entry is None:
            if bar.t < decision_t + latency_ms:
                continue
            if bar.t > decision_t + latency_ms + entry_expiry_ms:
                return finish("entry_unavailable", "entry_order_expired")
            price = modeled_buy_fill(bar.o, scenario)
            entry = Fill("entry", decision_t, bar.t, bar.o, price, quantity)
        elif pending is not None and bar.t >= pending[1] + latency_ms:
            reason, trigger_t, fraction = pending
            sold = remaining * fraction
            price = modeled_sell_fill(bar.o, scenario)
            exits.append(Fill(reason, trigger_t, bar.t, bar.o, price, sold))
            proceeds += sold * price * (1 - scenario.sell_fee_bps / 10_000)
            remaining -= sold
            pending = None
            if remaining <= quantity * 1e-12:
                remaining = 0.0
                mark_t, mark = bar.t, bar.o
                return finish("closed", reason)
            partial_done = True
            peak_after_partial = bar.o

        # No decision from a bar until that full second is observable.
        mark_t, mark = bar.t + SECOND_MS, bar.c
        if pending is not None:
            continue
        trigger_t = bar.t + SECOND_MS
        stop_level = entry.modeled_price * (1 - rule.stop_pct / 100)
        stop_hit = bar.l <= stop_level
        if not partial_done:
            take_hit = bar.h >= entry.modeled_price * (1 + rule.take_pct / 100)
            if stop_hit and take_hit:
                return finish("ambiguous", "stop_and_take_same_second")
            if stop_hit:
                pending = ("stop", trigger_t, 1.0)
            elif take_hit:
                pending = ("partial_take", trigger_t, rule.partial_fraction)
        else:
            assert peak_after_partial is not None
            old_trail = peak_after_partial * (1 - rule.trail_retrace_pct / 100)
            new_peak = max(peak_after_partial, bar.h)
            new_trail = new_peak * (1 - rule.trail_retrace_pct / 100)
            if bar.l <= new_trail and bar.h > peak_after_partial and bar.l > old_trail:
                return finish("ambiguous", "new_peak_and_trail_same_second")
            peak_after_partial = new_peak
            if stop_hit:
                pending = ("stop", trigger_t, 1.0)
            elif bar.l <= old_trail:
                pending = ("trailing", trigger_t, 1.0)

        if pending is None and trigger_t >= entry.fill_t + rule.max_hold_ms:
            pending = ("time_cap", trigger_t, 1.0)

    if entry is None:
        return finish("entry_unavailable", "no_eligible_open")
    if pending is not None:
        return finish("unresolved", f"{pending[0]}_unfilled")
    return finish("unresolved", "no_exit_trigger_or_cap_fill")

