"""R316B: event-first cohort with explicit acceptance and execution scope limits."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
import requests

from .config import load_settings
from .event_heterogeneity_audit import market_splits
from .feasible_upside_observability import CEILING
from .market_calendar import NYSE, previous_us_equity_trading_day
from .market_dataset import grouped_daily
from .massive_client import MassiveClient
from .pending_exit_integrity import regular_bars, session_limits
from .preentry_momentum_observability import entry_labels, vector_net
from .execution_costs import DEFAULT_EXECUTION_SCENARIOS

REQUEST_ID = "316B"
OFFSETS = (-1, 0, 1, 5)
SEC_USER_AGENT = "MoneyMakerResearch dayofhiki@users.noreply.github.com"
CACHE = Path("data/cache/r316b")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def selected_accession(accession: str) -> bool:
    return bool(re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession)) and bytes.fromhex(digest("R316B|" + accession))[0] < 16


def acceptance_header(text: str, accession: str) -> pd.Timestamp:
    # SGML's unzoned clock is Eastern local; never infer UTC from JSON spelling.
    head = text.split("</SEC-HEADER>", 1)[0]
    matched = re.search(r"ACCESSION NUMBER:\s*(\d{10}-\d{2}-\d{6})", head)
    form = re.search(r"CONFORMED SUBMISSION TYPE:\s*([^\r\n]+)", head)
    stamp = re.search(r"<ACCEPTANCE-DATETIME>\s*(\d{14})", head)
    if not matched or matched[1] != accession or not form or form[1].strip() != "8-K" or not stamp:
        raise ValueError("SEC header accession/form/clock validation failed")
    return pd.to_datetime(stamp[1], format="%Y%m%d%H%M%S").tz_localize("America/New_York", ambiguous="raise", nonexistent="raise").tz_convert("UTC")


def catalyst_clock(accepted: pd.Timestamp) -> tuple[str, int, str]:
    eligible = (accepted + pd.Timedelta(seconds=180)).ceil("s")
    local_day = eligible.tz_convert("America/New_York").date()
    for shift in range(8):
        day = (local_day + timedelta(days=shift)).isoformat()
        if day < "2026-05-01" or day > "2026-05-29":
            raise ValueError("clock outside fixed May development")
        try:
            opening, closing = session_limits(day)
        except ValueError:
            continue
        clock = max(int(eligible.timestamp() * 1000), opening + 300000)
        if clock < closing:
            bucket = "intraday" if int(accepted.timestamp()*1000) >= opening and shift == 0 else "premarket_or_rolled"
            return day, clock, bucket
    raise ValueError("no allowed catalyst session")


class SecHeaders:
    """Cache only SEC header bytes; stop retries on nontransient access denial."""
    def __init__(self, directory: Path, seed_path: Path | None = Path("research/r316b-sec-header-seed.json")):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.requests = 0
        self.failures = 0
        self.last_request = 0.0
        self.blocked = False
        self.lock = Lock()
        self.seeded_headers = 0
        self.seed_sha256 = None
        self.seed = {}
        if seed_path is not None and seed_path.exists():
            self.seed_sha256 = hashlib.sha256(seed_path.read_bytes()).hexdigest()
            payload = json.loads(seed_path.read_text())
            if payload.get("version") != 1:
                raise ValueError("unsupported SEC header seed version")
            self.seed = payload["headers"]

    def get(self, cik: str, accession: str) -> str:
        path = self.directory / f"{accession}.txt"
        if path.exists():
            return path.read_text()
        if accession in self.seed:
            row = self.seed[accession]
            text = row["header"]
            if hashlib.sha256(text.encode()).hexdigest() != row["sha256"]:
                raise ValueError("SEC seed content hash mismatch")
            acceptance_header(text, accession)
            path.write_text(text)
            self.seeded_headers += 1
            return text
        if self.blocked:
            raise PermissionError("SEC source circuit open")
        if not re.fullmatch(r"\d{1,10}", str(cik)) or not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
            raise ValueError("invalid SEC identity")
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{accession}.txt"
        with self.lock:
            if self.blocked:
                raise PermissionError("SEC source circuit open")
            time.sleep(max(0., .2 - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            self.requests += 1
        try:
            with requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=30, stream=True) as response:
                if response.status_code in (403, 429):
                    self.blocked = True
                response.raise_for_status()
                data = bytearray()
                for chunk in response.iter_content(8192):
                    data.extend(chunk)
                    if b"</SEC-HEADER>" in data or len(data) >= 262144:
                        break
                text = data.decode("utf-8", errors="replace").split("</SEC-HEADER>", 1)[0]
                acceptance_header(text, accession)
                path.write_text(text)
                return text
        except Exception:
            self.failures += 1
            raise


def event_census(client: MassiveClient, output: Path) -> tuple[list, list]:
    splits = market_splits(client, date(2026, 4, 1), date(2026, 5, 29))
    filings = client.eight_k_disclosures_market(filing_date_gte="2026-04-01", filing_date_lte="2026-05-21")
    write_json(output / "event-census.json", {"splits": splits, "filings": filings})
    return splits, filings


def census_anchors(splits: list, filings: list, client: MassiveClient, sec: SecHeaders) -> tuple[pd.DataFrame, list, list]:
    sessions = [x.date().isoformat() for x in NYSE.valid_days("2026-04-01", "2026-05-29")]
    records, ledger, news_audit = [], [], []
    reverse = [s for s in splits if s.get("adjustment_type") == "reverse_split" and "2026-05-04" <= str(s.get("execution_date")) <= "2026-05-20"]
    for s in sorted(reverse, key=lambda x: (x["execution_date"], x["ticker"])):
        event_day, ticker = s["execution_date"], s["ticker"]
        root = "split|" + str(s.get("id") or digest(json.dumps(s, sort_keys=True)))
        if event_day not in sessions:
            ledger.append({"root": root, "status": "non_session_execution"})
            continue
        ratio = float(s["split_from"]) / float(s["split_to"])
        if not np.isfinite(ratio) or ratio <= 1:
            raise ValueError("inconsistent reverse-split ratio")
        for offset in OFFSETS:
            day = sessions[sessions.index(event_day) + offset]
            if not "2026-05-01" <= day <= "2026-05-29":
                continue
            opening, _ = session_limits(day)
            records.append({"anchor_id": root + f"|{offset}", "root": root, "ticker": ticker,
                            "trading_day": day, "decision_t": opening+300000,
                            "family": "split_execution" if offset == 0 else f"split_offset_{offset}",
                            "execution_day": event_day, "offset": offset, "match_price_factor": ratio if offset == 0 else 1.,
                            "accepted_utc": "", "causal_event_knowledge_verified": False,
                            "timing_bucket": "structural_retrospective"})
        try:
            rows = client.news(ticker, published_gte=(date.fromisoformat(event_day)-timedelta(days=30)).isoformat(), published_lte=event_day+"T23:59:59Z")
            candidates = [r for r in rows if re.search(r"reverse[ -](?:stock[ -])?split", str(r.get("title", "")), flags=re.I)]
            news_audit.append({"root": root, "ticker": ticker, "execution_day": event_day, "status": "acquired",
                               "candidates": candidates, "execution_link_verified": False})
        except Exception as e:
            news_audit.append({"root": root, "status": type(e).__name__, "execution_link_verified": False})
    # An accession may have multiple classified rows: fixed aggregation, not one category.
    selected = {}
    for f in filings:
        acc = str(f.get("accession_number") or "")
        if not "2026-05-11" <= str(f.get("filing_date")) <= "2026-05-20" or not selected_accession(acc):
            continue
        for ticker in f.get("tickers") or []:
            selected.setdefault((acc, ticker), []).append(f)
    for (acc, ticker), rows in sorted(selected.items()):
        root = "8k|" + acc + "|" + ticker
        item = {"root": root, "accession": acc, "ticker": ticker, "filing_date": rows[0]["filing_date"]}
        try:
            cik = str(rows[0].get("cik") or "")
            if not cik.isdigit():
                cik = str(client.ticker_details(ticker, date.fromisoformat(item["filing_date"])).get("results", {}).get("cik") or "")
            accepted = acceptance_header(sec.get(cik, acc), acc)
            # Filing date may differ after the EDGAR cutoff, but no future census dates admitted.
            if abs((accepted.tz_convert("America/New_York").date()-date.fromisoformat(item["filing_date"])).days) > 1:
                raise ValueError("acceptance date inconsistent with filing census")
            day, clock, bucket = catalyst_clock(accepted)
            categories = sorted({"|".join(str(r.get(c) or "") for c in ("primary_category", "secondary_category", "tertiary_category")) for r in rows})
            records.append({"anchor_id": root, "root": root, "ticker": ticker, "trading_day": day,
                            "decision_t": clock, "family": "exact_8k", "execution_day": "", "offset": 0,
                            "match_price_factor": 1., "accepted_utc": accepted.isoformat(),
                            "causal_event_knowledge_verified": False, "timing_bucket": bucket,
                            "categories": ";".join(categories)})
            item.update(status="exact_header", accepted_utc=accepted.isoformat(), eligible_day=day)
        except Exception as e:
            item.update(status="timing_unknown", error_type=type(e).__name__)
        ledger.append(item)
    columns = ("anchor_id", "root", "ticker", "trading_day", "decision_t", "family", "execution_day", "offset", "match_price_factor", "accepted_utc", "causal_event_knowledge_verified", "timing_bucket", "categories")
    return pd.DataFrame(records, columns=columns), ledger, news_audit


def excluded_tickers(splits: list, filings: list, day: str) -> set[str]:
    start = (date.fromisoformat(day)-timedelta(days=30)).isoformat()
    result = {s["ticker"] for s in splits if start <= str(s.get("execution_date")) <= day}
    result.update(t for f in filings if start <= str(f.get("filing_date")) <= day for t in f.get("tickers") or [])
    return result


def match_cohort(anchors: pd.DataFrame, splits: list, filings: list, client: MassiveClient) -> tuple[pd.DataFrame, list]:
    grouped_cache, details_cache = {}, {}
    manifest, ledger = [], []

    def common(ticker, prior):
        key = (ticker, prior)
        if key not in details_cache:
            try:
                result = client.ticker_details(ticker, date.fromisoformat(prior)).get("results", {})
                details_cache[key] = result.get("type") == "CS" and result.get("locale") == "us" and result.get("market") == "stocks"
            except Exception:
                details_cache[key] = None
        return details_cache[key]

    for _, anchor in anchors.sort_values(["trading_day", "anchor_id"]).iterrows():
        a = anchor.to_dict()
        prior = previous_us_equity_trading_day(date.fromisoformat(a["trading_day"])).isoformat()
        if prior not in grouped_cache:
            try:
                payload = grouped_daily(client, date.fromisoformat(prior))
                grouped_cache[prior] = {r["T"]: r for r in payload.get("results") or []}
            except Exception:
                grouped_cache[prior] = None
        market = grouped_cache[prior]
        audit = {"anchor_id": a["anchor_id"], "family": a["family"], "ticker": a["ticker"], "prior_day": prior}
        if market is None:
            audit["status"] = "prior_market_unknown"
            ledger.append(audit)
            continue
        raw = market.get(a["ticker"], {})
        price = float(raw.get("c") or 0)*float(a["match_price_factor"])
        dv = float(raw.get("c") or 0)*float(raw.get("v") or 0)
        cs = common(a["ticker"], prior)
        if cs is not True or not .5 <= price <= 20 or dv <= 0:
            audit.update(status="reference_unknown" if cs is None else "ineligible_prior_context", match_price=price, prior_dollar_volume=dv)
            ledger.append(audit)
            continue
        excluded = excluded_tickers(splits, filings, a["trading_day"]) | {a["ticker"]}
        candidates = []
        for ticker, r in market.items():
            p, d = float(r.get("c") or 0), float(r.get("c") or 0)*float(r.get("v") or 0)
            if ticker in excluded or not .5 <= p <= 20 or d <= 0:
                continue
            lp, ld = np.log(p/price), np.log(d/dv)
            if abs(lp) <= np.log(4) and abs(ld) <= np.log(4):
                candidates.append((lp*lp+ld*ld, ticker, p, d))
        controls = []
        for distance, ticker, p, d in sorted(candidates):
            if common(ticker, prior) is True:
                controls.append({**a, "ticker": ticker, "arm": "control", "prior_price": p,
                                 "prior_dollar_volume": d, "match_distance": float(distance), "prior_day": prior})
                if len(controls) == 3:
                    break
        manifest.append({**a, "arm": "event", "prior_price": price, "prior_dollar_volume": dv, "match_distance": 0., "prior_day": prior})
        manifest.extend(controls)
        audit.update(status="matched" if len(controls) == 3 else "incomplete_matching", controls=len(controls))
        ledger.append(audit)
    return pd.DataFrame(manifest), ledger


def path_labels(bars: pd.DataFrame, decision_t: int, closing: int) -> dict:
    cap = min(decision_t+3600000, closing)
    result = entry_labels(bars, decision_t, cap)
    times = bars.t.to_numpy(np.int64)
    relevant = times[(times >= decision_t) & (times <= cap)]
    gaps = np.diff(relevant)
    result["largest_interprint_gap_s"] = float(gaps.max()/1000) if len(gaps) else None
    result["gap60_observed"] = bool(len(gaps) and gaps.max() >= 60000)
    if result.get("label_complete"):
        terminal = int(result["label_stop_t"])
        index = int(np.searchsorted(times, terminal))
        result["terminal_fill_t"] = int(times[index])
        result["terminal_fill_gap_s"] = (int(times[index])-terminal)/1000
        result["terminal_base_net_pct"] = float(vector_net(result["label_entry_price"], np.array([bars.o.iloc[index]]), DEFAULT_EXECUTION_SCENARIOS[1])[0])
    return result


def label_manifest(manifest: pd.DataFrame, client: MassiveClient) -> pd.DataFrame:
    labeled, histories = [], {}
    for _, row in manifest.iterrows():
        r = row.to_dict()
        r.update(label_complete=False, **{CEILING: np.nan})
        key = (r["ticker"], r["trading_day"])
        try:
            if key not in histories:
                payload = client.second_bars_range(key[0], date.fromisoformat(key[1]), date.fromisoformat(key[1]), adjusted=False)
                bars = pd.DataFrame(payload.get("results") or [], columns=("t", "o", "h", "l", "c", "v", "n"))
                opening, closing = session_limits(key[1])
                histories[key] = (regular_bars(bars, opening, closing), closing)
            bars, closing = histories[key]
            r.update(path_labels(bars, int(r["decision_t"]), closing))
            r["path_status"] = "observed" if len(bars) else "empty"
        except Exception as e:
            r.update(label_complete=False, path_status="unknown", error_type=type(e).__name__)
        labeled.append(r)
    return pd.DataFrame(labeled)


def prevalence_bounds(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"members": 0, "complete": 0, "net5_rate": None, "all_member_bounds": None}
    complete = frame.label_complete.fillna(False).astype(bool)
    positive = int(frame.loc[complete, CEILING].ge(5).sum())
    result = {"members": len(frame), "complete": int(complete.sum()), "positives": positive,
            "net5_rate": positive/int(complete.sum()) if complete.any() else None,
            "all_member_bounds": [positive/len(frame), (positive+int((~complete).sum()))/len(frame)],
            "net10_positives": int(frame.loc[complete, CEILING].ge(10).sum()),
            "net20_positives": int(frame.loc[complete, CEILING].ge(20).sum())}
    for column in ("terminal_base_net_pct", "label_entry_gap_s", "terminal_fill_gap_s", "largest_interprint_gap_s"):
        values = pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
        result[column+"_mean"] = float(values.mean()) if len(values) else None
        result[column+"_max"] = float(values.max()) if len(values) else None
    whole = pd.to_numeric(frame.get("label_whole_max_net_pct", pd.Series(dtype=float)), errors="coerce").dropna()
    result["whole_path_complete"] = len(whole)
    result["whole_path_positives"] = {str(level): int(whole.ge(level).sum()) for level in (5, 10, 20)}
    result["fixed_cost_sensitivity_net5"] = {}
    for scenario in DEFAULT_EXECUTION_SCENARIOS:
        values = pd.to_numeric(frame.get(f"label_max_{scenario.name}_net_pct", pd.Series(dtype=float)), errors="coerce").dropna()
        result["fixed_cost_sensitivity_net5"][scenario.name] = {"complete": len(values), "positive": int(values.ge(5).sum())}
    return result


def paired_report(frame: pd.DataFrame, family: str, timing_coverage: float | None) -> dict:
    f = frame.loc[frame.family.eq(family)] if len(frame) else frame
    pairs = []
    for anchor_id, g in f.groupby("anchor_id") if len(f) else []:
        if len(g) != 4 or not g.label_complete.fillna(False).all():
            continue
        event, controls = g.loc[g.arm.eq("event")], g.loc[g.arm.eq("control")]
        if len(event) != 1 or len(controls) != 3:
            raise ValueError("invalid frozen pair")
        e = event.iloc[0]
        pairs.append({"anchor_id": anchor_id, "root": e.root, "ticker": e.ticker, "day": e.trading_day,
                      "event": float(e[CEILING] >= 5), "control": float(controls[CEILING].ge(5).mean())})
    p = pd.DataFrame(pairs)
    anchors = int(f.loc[f.arm.eq("event"), "anchor_id"].nunique()) if len(f) else 0
    report = {"anchors": anchors, "complete_pairs": len(p), "event": prevalence_bounds(f.loc[f.arm.eq("event")]) if len(f) else prevalence_bounds(f),
              "controls": prevalence_bounds(f.loc[f.arm.eq("control")]) if len(f) else prevalence_bounds(f), "ci95": {}}
    if p.empty:
        report["status"] = "timing_blocked" if family == "exact_8k" and timing_coverage == 0 else "underpowered"
        return report
    event_rate, control_rate = float(p.event.mean()), float(p.control.mean())
    ratio = event_rate/control_rate if control_rate > 0 else None
    report.update(event_rate=event_rate, control_rate=control_rate, rate_ratio=ratio, difference=event_rate-control_rate,
                  positive_days=int(p.loc[p.event.eq(1), "day"].nunique()))
    for column, seed in (("ticker", 20263161), ("day", 20263162)):
        clusters = list(p.groupby(column, sort=True).indices.values())
        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(1000):
            choices = rng.integers(0, len(clusters), len(clusters))
            positions = np.concatenate([clusters[i] for i in choices])
            sample = p.iloc[positions]
            diffs.append(float((sample.event-sample.control).mean()))
        report["ci95"][column] = np.quantile(diffs, [.025, .975]).tolist()
    support = len(p) >= 40 and p.event.sum() >= 6 and report["positive_days"] >= 3 and len(p)/anchors >= .8
    timing_ok = family != "exact_8k" or (timing_coverage is not None and timing_coverage >= .8)
    signal = ratio is not None and ratio >= 2 and all(interval[0] > 0 for interval in report["ci95"].values())
    report["status"] = "timing_blocked" if not timing_ok else ("underpowered" if not support else ("promising_development_distribution" if signal else "not_supported"))
    return report


def verify_preserved_structural(before: pd.DataFrame, after: pd.DataFrame, old_reports: dict, new_reports: dict) -> dict:
    """Exact identities/outcomes; narrowly tolerate CPU log rounding in distances."""
    keys = ["anchor_id", "ticker", "arm"]
    before = before.sort_values(keys).reset_index(drop=True)
    after = after.loc[after.family.str.startswith("split_")].sort_values(keys).reset_index(drop=True)
    columns = [c for c in before if c not in ("categories", "match_distance")]
    pd.testing.assert_frame_equal(before[columns], after[columns], check_exact=True, check_dtype=False)
    left, right = before.match_distance.to_numpy(float), after.match_distance.to_numpy(float)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("structural matching distance must be finite")
    np.testing.assert_allclose(left, right, rtol=1e-15, atol=1e-18, equal_nan=False)
    families = [family for family in old_reports if family.startswith("split_")]
    for family in families:
        if old_reports[family] != new_reports[family]:
            raise ValueError(f"original structural report changed: {family}")
    delta = np.abs(left-right)
    return {"structural_rows_exact": len(before), "structural_non_distance_columns_exact": len(columns),
            "structural_reports_exact": len(families), "matching_distance_rounding_rows": int((delta > 0).sum()),
            "max_abs_matching_distance_delta": float(delta.max(initial=0)),
            "auxiliary_distance_rtol": 1e-15, "auxiliary_distance_atol": 1e-18}


def sec_feasibility(output: Path) -> dict:
    """Independent SEC census timing check; no prices or Massive credentials."""
    sec = SecHeaders(CACHE/"sec")
    days = ("20260511", "20260512", "20260513", "20260514", "20260515", "20260518", "20260519", "20260520")
    selected, census_count, index_audit = {}, 0, []
    index_dir = output/"sec-indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    for day in days:
        path = index_dir/f"master.{day}.idx"
        try:
            if not path.exists():
                response = requests.get(f"https://www.sec.gov/Archives/edgar/daily-index/2026/QTR2/master.{day}.idx",
                                        headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
                response.raise_for_status()
                path.write_bytes(response.content)
            count = 0
            for line in path.read_text(errors="replace").splitlines():
                fields = line.split("|")
                if len(fields) != 5 or fields[2] != "8-K":
                    continue
                cik, company, form, filing_day, filename = fields
                match = re.search(r"(\d{10}-\d{2}-\d{6})\.txt$", filename)
                count += 1
                if match and selected_accession(match[1]):
                    selected[match[1]] = {"accession": match[1], "cik": cik, "company": company,
                                          "form": form, "filing_day": filing_day, "source_filename": filename}
            census_count += count
            index_audit.append({"day": day, "status": "acquired", "eight_k_count": count})
        except Exception as e:
            index_audit.append({"day": day, "status": "unknown", "error_type": type(e).__name__})
    frozen = [selected[a] for a in sorted(selected)]
    write_json(output/"sec-selected-membership.json", frozen)
    write_json(output/"sec-index-ledger.json", index_audit)
    def resolve(row):
        item = dict(row)
        try:
            accepted = acceptance_header(sec.get(row["cik"], row["accession"]), row["accession"])
            day, clock, bucket = catalyst_clock(accepted)
            item.update(status="exact_header", accepted_utc=accepted.isoformat(),
                        accepted_eastern=accepted.tz_convert("America/New_York").isoformat(),
                        eligible_day=day, decision_t=clock, timing_bucket=bucket,
                        acceptance_filing_date_difference=(date.fromisoformat(row["filing_day"])-accepted.tz_convert("America/New_York").date()).days)
        except Exception as e:
            item.update(status="timing_unknown", error_type=type(e).__name__)
        return item

    ledger = []
    # Parallel latency handling, serialized starts <=5/s; membership stays fixed.
    with ThreadPoolExecutor(max_workers=8) as executor:
        for item in executor.map(resolve, frozen):
            ledger.append(item)
            if len(ledger)%10 == 0:
                write_json(output/"sec-timing-checkpoint.json", ledger)
                print(f"SEC feasibility {len(ledger)}/{len(frozen)}", flush=True)
    write_json(output/"sec-timing-ledger.json", ledger)
    known = [r for r in ledger if r["status"] == "exact_header"]
    result = {"request_id": REQUEST_ID, "stage": "SEC_only_source_feasibility", "index_audit": index_audit,
              "eight_k_census_count": census_count, "selected_accessions": len(frozen),
              "exact_headers": len(known), "coverage": len(known)/len(frozen) if frozen else None,
              "timing_buckets": pd.Series([r["timing_bucket"] for r in known], dtype=str).value_counts().to_dict(),
              "eligible_days": pd.Series([r["eligible_day"] for r in known], dtype=str).value_counts().to_dict(),
              "acceptance_filing_date_differences": pd.Series([r["acceptance_filing_date_difference"] for r in known], dtype=int).value_counts().to_dict(),
              "failures": pd.Series([r.get("error_type") for r in ledger if r["status"] != "exact_header"], dtype=str).value_counts().to_dict(),
              "membership_sha256": hashlib.sha256((output/"sec-selected-membership.json").read_bytes()).hexdigest(),
              "SEC_header_requests": sec.requests, "SEC_circuit_open": sec.blocked,
              "market_prices_loaded": False, "promotion_eligible": False,
              "limitations": "SEC-only source feasibility is distinct from the pending Massive classified cohort. Exact acceptance plus180s is assumed eligibility, not an observed first-public catalyst timestamp."}
    write_json(output/"sec-feasibility.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("build", "label", "sec-feasibility"), required=True)
    parser.add_argument("--frozen-census", type=Path)
    parser.add_argument("--expected-census-sha256")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.phase == "sec-feasibility":
        print(json.dumps(sec_feasibility(out), indent=2), flush=True)
        return 0
    client = MassiveClient(load_settings().massive_api_key, cache_dir=CACHE/"massive", request_interval_seconds=.02)
    if args.phase == "build":
        if args.frozen_census is not None:
            census_bytes = args.frozen_census.read_bytes()
            if not args.expected_census_sha256 or hashlib.sha256(census_bytes).hexdigest() != args.expected_census_sha256:
                raise ValueError("frozen original census hash mismatch")
            payload = json.loads(census_bytes)
            splits, filings = payload["splits"], payload["filings"]
            (out/"event-census.json").write_bytes(census_bytes)
        else:
            splits, filings = event_census(client, out)
        sec = SecHeaders(CACHE/"sec")
        anchors, timings, news = census_anchors(splits, filings, client, sec)
        anchors.to_parquet(out/"event-anchors.parquet", index=False)
        write_json(out/"timing-ledger.json", timings)
        write_json(out/"announcement-audit.json", news)
        manifest, matching = match_cohort(anchors, splits, filings, client)
        if manifest.empty:
            manifest = pd.DataFrame(columns=[*anchors.columns, "arm", "prior_price", "prior_dollar_volume", "match_distance", "prior_day"])
        manifest.to_parquet(out/"frozen-manifest.parquet", index=False)
        write_json(out/"matching-ledger.json", matching)
        selected = [r for r in timings if r.get("accession")]
        coverage = sum(r["status"] == "exact_header" for r in selected)/len(selected) if selected else None
        result = {"request_id": REQUEST_ID, "stage": "membership_frozen_before_target_paths", "anchors": len(anchors),
                  "manifest_members": len(manifest), "families": anchors.family.value_counts().to_dict(),
                  "selected_catalyst_ticker_accessions": len(selected), "exact_header_coverage": coverage,
                  "timing_status": pd.Series([r["status"] for r in selected], dtype=str).value_counts().to_dict(),
                  "matching_status": pd.Series([r["status"] for r in matching], dtype=str).value_counts().to_dict(),
                  "SEC": {"network_requests": sec.requests, "failures": sec.failures, "circuit_open": sec.blocked,
                          "seeded_headers": sec.seeded_headers, "seed_sha256": sec.seed_sha256},
                  "census_sha256": hashlib.sha256((out/"event-census.json").read_bytes()).hexdigest(),
                  "original_census_reused": args.frozen_census is not None,
                  "acquisition": client.stats.to_dict(), "manifest_sha256": hashlib.sha256((out/"frozen-manifest.parquet").read_bytes()).hexdigest(),
                  "promotion_eligible": False, "June_HOLD_opened": False, "final_July_August_opened": False}
        write_json(out/"cohort-build.json", result)
        print(json.dumps(result, indent=2), flush=True)
    else:
        build = json.loads((out/"cohort-build.json").read_text())
        path = out/"frozen-manifest.parquet"
        if hashlib.sha256(path.read_bytes()).hexdigest() != build["manifest_sha256"]:
            raise ValueError("frozen membership hash changed")
        manifest = pd.read_parquet(path)
        if len(manifest) and (not manifest.trading_day.between("2026-05-01", "2026-05-29").all() or manifest.duplicated(["anchor_id", "ticker", "arm"]).any()):
            raise ValueError("invalid frozen development membership")
        labels = label_manifest(manifest, client)
        if labels.empty:
            labels = manifest.assign(label_complete=pd.Series(dtype=bool), **{CEILING: pd.Series(dtype=float)})
        labels.to_parquet(out/"outcomes.parquet", index=False)
        reports = {family: paired_report(labels, family, build["exact_header_coverage"]) for family in ("split_execution", "exact_8k", "split_offset_-1", "split_offset_1", "split_offset_5")}
        controls = manifest.loc[manifest.arm.eq("control")]
        result = {"request_id": REQUEST_ID, "stage": "development_diagnostic_complete", "build": build,
                  "manifest_sha256": build["manifest_sha256"], "reports": reports, "acquisition": client.stats.to_dict(),
                  "shared_control_extra_uses": int(controls.duplicated(["trading_day", "ticker"]).sum()),
                  "gap60_cases": int(labels.get("gap60_observed", pd.Series(dtype=bool)).fillna(False).sum()),
                  "path_status": labels.get("path_status", pd.Series(dtype=str)).value_counts().to_dict(),
                  "model_refit": False, "promotion_eligible": False,
                  "limitations": "Retrospective split membership; announcement linkage unverified. Acceptance+180s is an assumed availability clock, not first-public catalyst time. Matched controls can have unknown catalysts. Next-print modeled ceilings are not realized profits; quotes/impact/official halt labels absent. Reused controls induce dependence beyond the two reported bootstrap schemes."}
        write_json(out/"request316b.json", result)
        print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
