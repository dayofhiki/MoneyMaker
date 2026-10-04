import numpy as np
import pandas as pd
import pytest

from victory_trader.feasible_upside_observability import CEILING
from victory_trader.monotone_momentum_calibration import Map, apply_map, calibrated_crossfit, fit_map


def calibration():
    return pd.DataFrame({"trading_day": "2026-05-05", "ticker": [str(i) for i in range(20)],
                         "hot_t": 0, "decision_t": 1000, "inner_probability": np.linspace(.01, .99, 20),
                         CEILING: [0.]*10+[10.]*10})


def test_calibrator_is_monotone_and_uses_only_supplied_labels():
    fitted, audit = fit_map(calibration())
    p = apply_map(np.linspace(0, 1, 100), fitted)
    assert np.diff(p).min() >= 0
    assert 0 < p.min() < p.max() < 1
    assert audit["fit_days"] == ["2026-05-05"]


def test_negative_discrimination_cannot_be_inverted():
    f = calibration()
    f[CEILING] = f[CEILING].iloc[::-1].to_numpy()
    fitted, _ = fit_map(f)
    assert fitted.slope == pytest.approx(0., abs=1e-8)
    with pytest.raises(ValueError, match="inversion"):
        apply_map([.1], Map(0., -1., .5, False))


def test_single_class_smoothed_fallback_is_recorded():
    f = calibration()
    f[CEILING] = 0.
    fitted, audit = fit_map(f)
    assert fitted.fallback and fitted.slope == 0
    assert apply_map([.01, .9], fitted) == pytest.approx([1/22, 1/22])
    assert audit["reason"] == "single_class"


@pytest.mark.parametrize("values", [[np.nan], [-.1], [1.1]])
def test_invalid_probability_rejected(values):
    with pytest.raises(ValueError, match="probability"):
        apply_map(values, Map(0., 1., .5, False))


def source():
    return pd.concat([calibration().assign(trading_day=day, C_p_net_5=.25)
                      for day in ("2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08")], ignore_index=True)


def test_outer_and_inner_days_excluded_and_held_labels_cannot_change_scores(monkeypatch):
    calls = []
    def fit(train, features, level, seed):
        calls.append(set(train.trading_day))
        return None, .25
    monkeypatch.setattr("victory_trader.monotone_momentum_calibration.fit_probability", fit)
    f = source()
    before, folds_before = calibrated_crossfit(f, ())
    f.loc[f.trading_day.eq("2026-05-05"), CEILING] = 1e12
    after, folds_after = calibrated_crossfit(f, ())
    pd.testing.assert_series_equal(before.loc[before.trading_day.eq("2026-05-05"), "E_p_net_5"], after.loc[after.trading_day.eq("2026-05-05"), "E_p_net_5"])
    assert folds_before["2026-05-05"] == folds_after["2026-05-05"]
    assert calls[0] == {"2026-05-06", "2026-05-07", "2026-05-08"}
    for day, audit in folds_before.items():
        for scored_day, inner in audit["inner_folds"].items():
            assert day not in inner["fit_days"] and scored_day not in inner["fit_days"]


def test_saved_score_mismatch_stops_research(monkeypatch):
    monkeypatch.setattr("victory_trader.monotone_momentum_calibration.fit_probability", lambda *args: (None,.25))
    f = source()
    f.loc[0, "C_p_net_5"] = .9
    with pytest.raises(ValueError, match="score mismatch"):
        calibrated_crossfit(f, ())


def test_future_day_rejected_before_fit():
    f = source()
    f.loc[0, "trading_day"] = "2026-06-15"
    with pytest.raises(ValueError, match="May5-8"):
        calibrated_crossfit(f, ())
