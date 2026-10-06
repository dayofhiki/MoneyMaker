from types import SimpleNamespace

import json
import numpy as np
import pandas as pd
import pytest

from victory_trader import canonical_population_momentum as experiment
from victory_trader.feasible_upside_observability import CEILING
from victory_trader.frozen_momentum_may_transport import EVAL_DAYS, identity_hash
from victory_trader.lagged_minute_context import CROSSFIT_DAYS
from victory_trader.pending_exit_integrity import session_limits


def population():
    rows = []
    for day in CROSSFIT_DAYS:
        opening, _ = session_limits(day)
        for i in range(20):
            rows.append({"trading_day": day, "ticker": f"X{i}", "t": opening+60000, "bar_start_t": opening,
                         "log_current_price": 1., "minutes_since_open": 1., "minutes_to_close": 389.,
                         "return_from_previous_close_pct": 2., "future_winner": i % 2})
    return pd.DataFrame(rows)


def union():
    full, _ = experiment.training_population(population())
    selected = full.groupby("trading_day", sort=False).head(2)
    return experiment.training_union(full, selected)


def bars(day, seconds=(60, 61)):
    opening, _ = session_limits(day)
    return pd.DataFrame({"t": opening+np.array(seconds)*1000, "o": 5., "c": 5., "h": 5., "l": 5., "v": 100., "n": 1.})


def test_population_is_hash_only_and_drops_targets_and_held_dates():
    source = population()
    full, audit = experiment.training_population(source)
    changed = source.copy()
    changed["future_winner"] = 99999
    changed["economic_admission_selected"] = False
    changed = pd.concat([changed, changed.iloc[:1].assign(trading_day="2026-06-15")], ignore_index=True)
    other, other_audit = experiment.training_population(changed.sample(frac=1., random_state=5))
    pd.testing.assert_frame_equal(full, other)
    assert audit == other_audit and "future_winner" not in full
    assert all(identity_hash(str(r.trading_day), str(r.ticker).lower(), int(r.hot_t)) == r.sample_hash_byte for r in full.itertuples())
    with pytest.raises(ValueError, match="completed minute"):
        experiment.training_population(source.assign(bar_start_t=source.bar_start_t+1))
    with pytest.raises(ValueError, match="four-day"):
        experiment.training_population(source.loc[source.trading_day.ne(CROSSFIT_DAYS[0])])


def test_union_membership_is_saved_independent_of_labels_and_rejects_changes():
    cohort = union()
    assert cohort.in_selected.sum() == 8
    assert (cohort.in_broad | cohort.in_selected).all()
    experiment.validate_union(cohort)
    changed = cohort.copy()
    changed.loc[changed.index[0], "in_broad"] = not changed.in_broad.iloc[0]
    with pytest.raises(ValueError, match="hash membership"):
        experiment.validate_union(changed)
    with pytest.raises(ValueError, match="scope"):
        experiment.validate_union(cohort.assign(trading_day="2026-06-15"))
    full, _ = experiment.training_population(population())
    with pytest.raises(ValueError, match="absent"):
        experiment.training_union(full, full.iloc[:1].assign(ticker="MISSING"))


def test_acquisition_reuses_raw_and_never_expands_declared_dates_or_identities():
    cohort = union().iloc[:2].copy()
    day, ticker = cohort[["trading_day", "ticker"]].iloc[0]
    raw = bars(day).assign(trading_day=day, ticker=ticker)
    calls = []

    class Client:
        stats = SimpleNamespace(to_dict=lambda: {"network_requests": 1, "cache_hits": 0, "retries": 0})

        def second_bars_range(self, symbol, start, end, adjusted):
            calls.append((symbol, str(start), str(end), adjusted))
            return {"results": bars(str(start)).to_dict("records")}

    contexts, saved, audit, failed = experiment.acquire_contexts(cohort, raw, Client())
    assert audit["inherited_pairs"] == 1 and len(calls) == 1 and not failed
    assert calls[0] == (cohort.ticker.iloc[1], day, day, False)
    assert len(contexts) == 2 and len(saved) == 4
    with pytest.raises(ValueError, match="scope"):
        experiment.acquire_contexts(cohort.assign(trading_day="2026-06-15"), raw, Client())
    with pytest.raises(ValueError, match="requires configured"):
        experiment.acquire_contexts(cohort, raw, None)


def test_fetch_failures_and_incomplete_first_labels_are_never_replaced():
    cohort = union().iloc[:2].copy()
    day, ticker = cohort[["trading_day", "ticker"]].iloc[0]
    opening, closing = session_limits(day)
    raw = bars(day)
    contexts = {(day, ticker): (raw, opening, closing)}
    failed = {(day, cohort.ticker.iloc[1])}
    states = experiment.observations(cohort, contexts, failed)
    assert len(states) == 2 and states.decision_t.iloc[0] == opening+61000
    assert states.observation_available.iloc[0] and not states.label_complete.iloc[0]
    assert states.missing_reason.iloc[0] == "incomplete_entry_or_terminal_fill_label"
    assert not states.observation_available.iloc[1] and pd.isna(states[CEILING].iloc[1])
    assert states.missing_reason.iloc[1] == "fetch_or_raw_validation_failure"
    assert experiment.coverage(states)["labeled_episodes"] == 0


def test_empty_and_rejected_fetches_keep_cohort_and_log_only_error_type():
    cohort = union().iloc[:2].copy()
    inherited = bars(CROSSFIT_DAYS[0]).iloc[:0].assign(trading_day="", ticker="")

    class Client:
        stats = SimpleNamespace(to_dict=lambda: {"network_requests": 2})

        def second_bars_range(self, ticker, start, end, adjusted):
            if ticker == cohort.ticker.iloc[0]:
                raise RuntimeError("private response details")
            return {"results": []}

    contexts, raw, audit, failed = experiment.acquire_contexts(cohort, inherited, Client())
    states = experiment.observations(cohort, contexts, failed)
    assert len(states) == len(cohort) and not states.observation_available.any()
    assert raw.empty and len(audit["failures"]) == 1
    assert audit["failures"][0]["error_type"] == "RuntimeError"
    assert "private" not in json.dumps(audit)


def labeled():
    cohort = union()
    cohort["observation_available"] = True
    cohort["label_complete"] = True
    cohort[CEILING] = np.where(np.arange(len(cohort)) % 3 == 0, 6., 0.)
    cohort["missing_reason"] = None
    return cohort


def fake_models(monkeypatch):
    seen = []

    def fit(frame, features, seed):
        seen.append((frame.copy(), features, seed))
        return None, None, float(frame[CEILING].ge(5).mean()), {"train_rows": len(frame)}

    monkeypatch.setattr(experiment, "fit_linear", fit)
    monkeypatch.setattr(experiment, "predict_linear", lambda frame, model, transform, prior: np.full(len(frame), prior))
    return seen


def test_frozen_heads_use_only_declared_training_and_ignore_eval_outcomes(monkeypatch):
    canonical = labeled()
    original = canonical.loc[canonical.in_selected].copy()
    held = canonical.iloc[:4].assign(trading_day=EVAL_DAYS[0]).copy()
    held.loc[held.index[0], "observation_available"] = False
    seen = fake_models(monkeypatch)
    first, _ = experiment.fit_heads(original, canonical, held)
    other, _ = experiment.fit_heads(original, canonical, held.assign(**{CEILING: 999999.}))
    cols = [f"{a}_p_net_5" for a in experiment.ARMS]
    pd.testing.assert_frame_equal(first[cols], other[cols])
    assert first[cols].iloc[0].isna().all()
    assert all(set(frame.trading_day).issubset(CROSSFIT_DAYS) for frame, _, _ in seen)
    pd.testing.assert_frame_equal(seen[2][0], seen[3][0])  # B/Q exactly matched training rows.
    with pytest.raises(ValueError, match="fixed training dates"):
        experiment.fit_heads(original, canonical.assign(trading_day=EVAL_DAYS[0]), held)


def test_chronological_folds_never_use_current_or_later_labels(monkeypatch):
    canonical = labeled()
    seen = fake_models(monkeypatch)
    first, folds = experiment.chronological_training(canonical)
    assert first.loc[first.trading_day.eq(CROSSFIT_DAYS[0]), "chronological_B_p_net_5"].isna().all()
    assert len(seen) == 6 and set(folds) == set(CROSSFIT_DAYS[1:])
    for day, audit in folds.items():
        assert all(d < day for d in audit["fit_days"])
    poison = canonical.copy()
    poison.loc[poison.trading_day.ge(CROSSFIT_DAYS[2]), CEILING] = 99999.
    second, _ = experiment.chronological_training(poison)
    before = canonical.trading_day.eq(CROSSFIT_DAYS[1])
    pd.testing.assert_series_equal(first.loc[before, "chronological_B_p_net_5"], second.loc[before, "chronological_B_p_net_5"])
    with pytest.raises(ValueError, match="earlier broad"):
        experiment.chronological_training(canonical.assign(label_complete=False))


def test_future_price_mutation_preserves_first_observation_and_inputs():
    cohort = union().iloc[:1]
    day, ticker = cohort[["trading_day", "ticker"]].iloc[0]
    opening, closing = session_limits(day)
    raw = bars(day, (0, 30, 60, 61, 100, 3700))
    first = experiment.observations(cohort, {(day, ticker): (raw, opening, closing)})
    changed = raw.copy()
    changed.loc[changed.t.ge(opening+61000), ["o", "c", "h", "l", "v", "n"]] = 999.
    second = experiment.observations(cohort, {(day, ticker): (changed, opening, closing)})
    fields = ["decision_t", *experiment.CLOCK_FEATURES]
    pd.testing.assert_frame_equal(first[fields], second[fields])


def test_empty_evaluation_reports_missingness_and_fails_gates(monkeypatch):
    canonical = labeled()
    original = canonical.loc[canonical.in_selected]
    held = canonical.iloc[:2].assign(trading_day=EVAL_DAYS[0], observation_available=False, label_complete=False, missing_reason="absent")
    fake_models(monkeypatch)
    scored, _ = experiment.fit_heads(original, canonical, held)
    result = experiment.report(scored, draws=5)
    assert result["sampled_episodes"] == 2 and result["labeled_episodes"] == 0
    assert not any(experiment.gate(result).values())
    json.dumps(result, allow_nan=False)


def test_parquet_nullable_flag_roundtrip_preserves_values_and_missingness(tmp_path):
    original = pd.DataFrame({"label_stop_cost_only": [True, False, np.nan], CEILING: [6., 0., np.nan]})
    path = tmp_path/"checkpoint.parquet"
    original.to_parquet(path, index=False)
    saved = pd.read_parquet(path)
    experiment.verify_checkpoint(original, saved)
    altered = saved.copy()
    altered.loc[2, "label_stop_cost_only"] = False
    with pytest.raises(AssertionError):
        experiment.verify_checkpoint(original, altered)
    altered = saved.copy()
    altered.loc[0, "label_stop_cost_only"] = False
    with pytest.raises(AssertionError):
        experiment.verify_checkpoint(original, altered)
