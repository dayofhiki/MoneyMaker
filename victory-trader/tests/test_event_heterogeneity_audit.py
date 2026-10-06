from types import SimpleNamespace

import pandas as pd
from victory_trader import event_heterogeneity_audit as experiment
from victory_trader.feasible_upside_observability import CEILING


def decision_ms(text="2026-05-11T14:30:00Z"):
    return int(pd.Timestamp(text).timestamp() * 1000)


def frozen_row():
    row = {
        "trading_day": "2026-05-11",
        "ticker": "XYZ",
        "hot_t": decision_ms("2026-05-11T14:29:00Z"),
        "decision_t": decision_ms(),
        "observation_available": True,
        "label_complete": True,
        CEILING: 6.0,
        "N_p_net_5": 0.2,
        "Q_p_net_5": 0.3,
        "M_p_net_5": 0.7,
    }
    for feature in experiment.MOMENTUM:
        row.setdefault(feature, 1.0)
    return row


def complete_acquisition():
    return {
        "failures": {
            "news_tickers": [],
            "eight_k": None,
            "splits": None,
        }
    }


def test_market_splits_uses_current_endpoint_and_paginates():
    calls = []

    class Client:
        def _get(self, path, params):
            calls.append((path, params))
            return {
                "results": [{"ticker": "A", "execution_date": "2026-05-01"}],
                "next_url": "https://api.massive.com/stocks/v1/splits?cursor=x",
            }

        def _get_next_url(self, url):
            calls.append(("next", url))
            return {"results": [{"ticker": "B", "execution_date": "2026-05-02"}]}

    rows = experiment.market_splits(
        Client(),
        pd.Timestamp("2026-04-01").date(),
        pd.Timestamp("2026-05-20").date(),
    )
    assert len(rows) == 2
    assert calls[0][0] == "/stocks/v1/splits"
    assert calls[0][1]["execution_date.gte"] == "2026-04-01"
    assert calls[0][1]["execution_date.lte"] == "2026-05-20"


def test_point_in_time_multilabel_annotation_and_same_day_audits():
    states = pd.DataFrame([frozen_row()])
    events = pd.DataFrame(
        [
            {
                "source": "news",
                "ticker": "XYZ",
                "event_time": "2026-05-11T14:00:00Z",
                "event_date": "2026-05-11",
                "event_id": "n1",
                "category": "",
            },
            {
                "source": "news",
                "ticker": "XYZ",
                "event_time": "2026-05-11T15:00:00Z",
                "event_date": "2026-05-11",
                "event_id": "future",
                "category": "",
            },
            {
                "source": "8k",
                "ticker": "XYZ",
                "event_time": "",
                "event_date": "2026-05-06",
                "event_id": "old",
                "category": "financial_results",
            },
            {
                "source": "8k",
                "ticker": "XYZ",
                "event_time": "",
                "event_date": "2026-05-11",
                "event_id": "same",
                "category": "regulatory",
            },
            {
                "source": "split",
                "ticker": "XYZ",
                "event_time": "",
                "event_date": "2026-05-01",
                "event_id": "s1",
                "category": "reverse_split",
            },
            {
                "source": "split",
                "ticker": "XYZ",
                "event_time": "",
                "event_date": "2026-05-11",
                "event_id": "same-split",
                "category": "reverse_split",
            },
        ]
    )
    annotated = experiment.annotate_events(states, events, complete_acquisition())
    row = annotated.iloc[0]
    assert row.fresh_news_1h and row.fresh_news_24h and row.recent_news_7d
    assert row.prior_8k_7d and row.prior_8k_30d
    assert row.same_day_8k_count == 1
    assert row.recent_reverse_split_20d
    assert not row.recent_reverse_split_5d
    assert row.same_day_reverse_split_count == 1
    assert row.known_event_any and not row.none_known


def test_news_after_decision_is_not_a_causal_event():
    states = pd.DataFrame([frozen_row()])
    events = pd.DataFrame(
        [{
            "source": "news",
            "ticker": "XYZ",
            "event_time": "2026-05-11T14:30:01Z",
            "event_date": "2026-05-11",
            "event_id": "future",
            "category": "",
        }],
        columns=("source", "ticker", "event_time", "event_date", "event_id", "category"),
    )
    annotated = experiment.annotate_events(states, events, complete_acquisition())
    assert not bool(annotated.fresh_news_24h.iloc[0])
    assert bool(annotated.none_known.iloc[0])


def test_failed_source_is_unknown_not_negative():
    states = pd.DataFrame([frozen_row()])
    events = pd.DataFrame(columns=("source", "ticker", "event_time", "event_date", "event_id", "category"))
    acquisition = complete_acquisition()
    acquisition["failures"]["news_tickers"] = [{"ticker": "XYZ", "error_type": "RuntimeError"}]
    annotated = experiment.annotate_events(states, events, acquisition)
    assert pd.isna(annotated.fresh_news_24h.iloc[0])
    assert pd.isna(annotated.known_event_any.iloc[0])
    assert pd.isna(annotated.none_known.iloc[0])
    assert not annotated.event_sources_complete.iloc[0]


def test_event_flags_are_independent_of_outcomes_and_frozen_scores():
    states = pd.DataFrame([frozen_row()])
    events = pd.DataFrame(
        [{
            "source": "news",
            "ticker": "XYZ",
            "event_time": "2026-05-11T14:00:00Z",
            "event_date": "2026-05-11",
            "event_id": "n1",
            "category": "",
        }],
        columns=("source", "ticker", "event_time", "event_date", "event_id", "category"),
    )
    first = experiment.annotate_events(states, events, complete_acquisition())
    changed = states.copy()
    changed[CEILING] = -999.0
    changed[["N_p_net_5", "Q_p_net_5", "M_p_net_5"]] = 0.999
    second = experiment.annotate_events(changed, events, complete_acquisition())
    columns = [
        *experiment.PRIMARY_FLAGS,
        *experiment.SECONDARY_FLAGS,
        "known_event_any",
        "none_known",
        "same_day_8k_count",
        "same_day_reverse_split_count",
    ]
    pd.testing.assert_frame_equal(first[columns], second[columns])


def test_eight_k_normalization_expands_tickers_without_manual_taxonomy():
    rows = experiment._eight_k_records(
        {
            "filing_date": "2026-05-05",
            "accession_number": "abc",
            "tickers": ["XYZ", "OTHER"],
            "primary_category": "financial_results",
            "secondary_category": "earnings_announcement",
            "tertiary_category": "quarterly_results",
        },
        {"XYZ"},
    )
    assert len(rows) == 1 and rows[0]["ticker"] == "XYZ"
    assert rows[0]["category"] == "financial_results|earnings_announcement|quarterly_results"


def test_underpowered_family_cannot_trigger_material_gate():
    family = {
        "event": {"labeled": 39, "positives": 20, "positive_days": 5},
        "comparison": {
            "prevalence_ratio": 5.0,
            "M_within_day_auc_difference": 0.4,
            "MQ_increment_difference": 0.4,
        },
        "bootstrap": {
            "ticker_day": {
                "ci95": {
                    "prevalence_ratio": [2.0, 8.0],
                    "M_within_day_auc_difference": [0.2, 0.6],
                    "MQ_increment_difference": [0.2, 0.6],
                }
            }
        },
    }
    gate = experiment.family_gate(family)
    assert not gate["adequate_support"]
    assert not gate["materially_supported"]


def test_acquisition_logs_error_types_without_leaking_details(monkeypatch):
    states = pd.DataFrame([frozen_row()])
    stats = SimpleNamespace(to_dict=lambda: {"network_requests": 3, "cache_hits": 0, "retries": 0, "cache_writes": 0})

    class Client:
        def __init__(self):
            self.stats = stats

        def news(self, *args, **kwargs):
            raise RuntimeError("private provider payload")

        def eight_k_disclosures_market(self, *args, **kwargs):
            return []

    monkeypatch.setattr(experiment, "market_splits", lambda *args, **kwargs: [])
    events, audit = experiment.acquire_events(states, Client())
    assert events.empty
    assert audit["failures"]["news_tickers"] == [{"ticker": "XYZ", "error_type": "RuntimeError"}]
    assert "private" not in str(audit)
