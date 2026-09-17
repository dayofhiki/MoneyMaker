import pytest

from victory_trader.execution_costs import (
    DEFAULT_EXECUTION_SCENARIOS,
    ExecutionScenario,
    modeled_buy_fill,
    modeled_sell_fill,
    net_round_trip_return_pct,
)


def test_costs_reduce_a_positive_gross_return():
    scenario = ExecutionScenario(
        "test", half_spread_bps=25, slippage_bps=25, min_half_spread_cents=1.0
    )
    net = net_round_trip_return_pct(2.00, 3.0, scenario)
    assert net < 3.0
    assert net > 0.0


def test_low_price_minimum_cent_spread_matters():
    scenario = ExecutionScenario(
        "test", half_spread_bps=0, slippage_bps=0, min_half_spread_cents=1.0
    )
    buy = modeled_buy_fill(1.00, scenario)
    sell = modeled_sell_fill(1.00, scenario)
    assert buy == pytest.approx(1.01)
    assert sell == pytest.approx(0.99)
    assert net_round_trip_return_pct(1.00, 0.0, scenario) < -1.9


def test_default_scenarios_use_actual_cents_not_hundredths_of_a_cent():
    light, base, stress = DEFAULT_EXECUTION_SCENARIOS
    assert light.min_half_spread_cents == pytest.approx(0.5)
    assert base.min_half_spread_cents == pytest.approx(1.0)
    assert stress.min_half_spread_cents == pytest.approx(2.0)
    assert modeled_buy_fill(1.0, base) == pytest.approx(1.0125)


def test_stress_scenario_is_harsher_than_light():
    light, _, stress = DEFAULT_EXECUTION_SCENARIOS
    light_net = net_round_trip_return_pct(3.00, 5.0, light)
    stress_net = net_round_trip_return_pct(3.00, 5.0, stress)
    assert stress_net < light_net


def test_negative_parameters_are_rejected():
    scenario = ExecutionScenario("bad", half_spread_bps=-1, slippage_bps=0)
    with pytest.raises(ValueError):
        modeled_buy_fill(2.00, scenario)
