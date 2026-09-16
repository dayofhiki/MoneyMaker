from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionScenario:
    """A transparent execution-friction scenario, not a claim about real fills.

    Historical 1-minute bars do not contain enough order-book information to know
    the true spread or market impact. These parameters intentionally make the
    assumption explicit so a strategy must survive multiple friction levels.
    """

    name: str
    half_spread_bps: float
    slippage_bps: float
    min_half_spread_cents: float = 0.0
    sell_fee_bps: float = 0.0

    def validate(self) -> None:
        values = (
            self.half_spread_bps,
            self.slippage_bps,
            self.min_half_spread_cents,
            self.sell_fee_bps,
        )
        if any(value < 0 for value in values):
            raise ValueError("execution cost parameters must be non-negative")


DEFAULT_EXECUTION_SCENARIOS = (
    ExecutionScenario("light", half_spread_bps=10.0, slippage_bps=10.0, min_half_spread_cents=0.005),
    ExecutionScenario("base", half_spread_bps=25.0, slippage_bps=25.0, min_half_spread_cents=0.01),
    ExecutionScenario("stress", half_spread_bps=75.0, slippage_bps=75.0, min_half_spread_cents=0.02),
)


def _half_spread_dollars(reference_price: float, scenario: ExecutionScenario) -> float:
    scenario.validate()
    if reference_price <= 0:
        raise ValueError("reference_price must be positive")
    proportional = reference_price * scenario.half_spread_bps / 10_000.0
    return max(proportional, scenario.min_half_spread_cents / 100.0)


def modeled_buy_fill(signal_price: float, scenario: ExecutionScenario) -> float:
    """Model an aggressive buy: cross half-spread and pay adverse slippage."""
    half_spread = _half_spread_dollars(signal_price, scenario)
    slipped = signal_price * (1.0 + scenario.slippage_bps / 10_000.0)
    return slipped + half_spread


def modeled_sell_fill(reference_price: float, scenario: ExecutionScenario) -> float:
    """Model an aggressive sell: cross half-spread and pay adverse slippage."""
    half_spread = _half_spread_dollars(reference_price, scenario)
    slipped = reference_price * (1.0 - scenario.slippage_bps / 10_000.0)
    fill = slipped - half_spread
    return max(fill, 0.0)


def net_round_trip_return_pct(
    entry_signal_price: float,
    gross_exit_return_pct: float,
    scenario: ExecutionScenario,
) -> float:
    """Convert a theoretical signal-to-exit return into a friction-adjusted return."""
    if entry_signal_price <= 0:
        raise ValueError("entry_signal_price must be positive")
    raw_exit_price = entry_signal_price * (1.0 + gross_exit_return_pct / 100.0)
    if raw_exit_price <= 0:
        return -100.0

    entry_fill = modeled_buy_fill(entry_signal_price, scenario)
    exit_fill = modeled_sell_fill(raw_exit_price, scenario)
    proceeds_after_sell_fee = exit_fill * (1.0 - scenario.sell_fee_bps / 10_000.0)
    return (proceeds_after_sell_fee / entry_fill - 1.0) * 100.0


def add_execution_scenarios_to_record(
    entry_signal_price: float,
    gross_exit_return_pct: float | None,
    scenarios: tuple[ExecutionScenario, ...] = DEFAULT_EXECUTION_SCENARIOS,
    *,
    prefix: str = "exit",
) -> dict[str, float | None]:
    record: dict[str, float | None] = {}
    for scenario in scenarios:
        key = f"{prefix}_{scenario.name}_net_return_pct"
        if gross_exit_return_pct is None:
            record[key] = None
        else:
            record[key] = net_round_trip_return_pct(
                entry_signal_price,
                gross_exit_return_pct,
                scenario,
            )
    return record
