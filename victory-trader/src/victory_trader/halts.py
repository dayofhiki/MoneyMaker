from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from typing import Iterable
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import pandas as pd
import requests


NEW_YORK = ZoneInfo("America/New_York")
NASDAQ_HALT_RSS = "https://www.nasdaqtrader.com/rss.aspx"
VOLATILITY_HALT_CODES = {"LUDP", "LUDS", "T5", "M"}


@dataclass(frozen=True)
class HaltRecord:
    symbol: str
    halt_at_ms: int
    reason_code: str | None
    resume_at_ms: int | None
    market: str | None

    @property
    def is_volatility_pause(self) -> bool:
        return (self.reason_code or "").upper() in VOLATILITY_HALT_CODES


@dataclass(frozen=True)
class HaltAnnotation:
    halt_within_5m: bool
    halt_within_15m: bool
    halt_within_30m: bool
    halt_within_60m: bool
    first_halt_minutes_after_event: float | None
    first_halt_reason_code: str | None
    first_halt_resume_minutes: float | None
    first_halt_is_volatility_pause: bool | None

    def to_record(self) -> dict[str, bool | float | str | None]:
        return asdict(self)


def _parse_time(value: str | None) -> time | None:
    if not value:
        return None
    text = value.strip()
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            pass
    return None


def _to_ms(day: date, clock: time | None) -> int | None:
    if clock is None:
        return None
    dt = datetime.combine(day, clock, tzinfo=NEW_YORK)
    return int(dt.timestamp() * 1000)


def _text(item: ElementTree.Element, *names: str) -> str | None:
    wanted = {name.lower() for name in names}
    for child in item.iter():
        tag = child.tag.split("}")[-1].lower()
        if tag in wanted and child.text:
            return child.text.strip()
    return None


def parse_nasdaq_halt_rss(xml_text: str, requested_day: date) -> list[HaltRecord]:
    """Parse Nasdaq Trader's historical halt RSS defensively.

    Nasdaq has changed feed formatting over time, so the parser accepts several
    known tag spellings and skips incomplete rows instead of fabricating times.
    """
    root = ElementTree.fromstring(xml_text)
    records: list[HaltRecord] = []
    for item in root.findall(".//item"):
        symbol = _text(item, "issuesymbol", "symbol", "ticker")
        halt_time = _parse_time(_text(item, "halttime", "halt_time"))
        if not symbol or halt_time is None:
            continue

        halt_date_text = _text(item, "haltdate", "halt_date")
        halt_day = requested_day
        if halt_date_text:
            for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m%d%Y"):
                try:
                    halt_day = datetime.strptime(halt_date_text, fmt).date()
                    break
                except ValueError:
                    pass

        resume_date_text = _text(item, "resumptiondate", "resumedate", "resume_date")
        resume_day = halt_day
        if resume_date_text:
            for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m%d%Y"):
                try:
                    resume_day = datetime.strptime(resume_date_text, fmt).date()
                    break
                except ValueError:
                    pass

        resume_clock = _parse_time(
            _text(item, "resumptiontradetime", "resumetradetime", "resume_time")
        )
        records.append(
            HaltRecord(
                symbol=symbol.upper(),
                halt_at_ms=_to_ms(halt_day, halt_time) or 0,
                reason_code=_text(item, "reasoncode", "reason_code"),
                resume_at_ms=_to_ms(resume_day, resume_clock),
                market=_text(item, "mkt", "market"),
            )
        )
    return sorted(records, key=lambda record: (record.symbol, record.halt_at_ms))


def fetch_nasdaq_halts(day: date, timeout: float = 30.0) -> list[HaltRecord]:
    """Fetch official Nasdaq Trader halt records whose initial halt date is `day`."""
    params = {"feed": "tradehalts", "haltdate": day.strftime("%m%d%Y")}
    response = requests.get(NASDAQ_HALT_RSS, params=params, timeout=timeout)
    response.raise_for_status()
    return parse_nasdaq_halt_rss(response.text, day)


def annotate_event_halts(
    symbol: str,
    event_timestamp_ms: int,
    halt_records: Iterable[HaltRecord],
) -> HaltAnnotation:
    relevant = sorted(
        (
            record
            for record in halt_records
            if record.symbol.upper() == symbol.upper() and record.halt_at_ms >= event_timestamp_ms
        ),
        key=lambda record: record.halt_at_ms,
    )
    if not relevant:
        return HaltAnnotation(False, False, False, False, None, None, None, None)

    first = relevant[0]
    minutes = (first.halt_at_ms - event_timestamp_ms) / 60_000.0
    resume_minutes = None
    if first.resume_at_ms is not None and first.resume_at_ms >= first.halt_at_ms:
        resume_minutes = (first.resume_at_ms - first.halt_at_ms) / 60_000.0

    return HaltAnnotation(
        halt_within_5m=minutes <= 5,
        halt_within_15m=minutes <= 15,
        halt_within_30m=minutes <= 30,
        halt_within_60m=minutes <= 60,
        first_halt_minutes_after_event=minutes,
        first_halt_reason_code=first.reason_code,
        first_halt_resume_minutes=resume_minutes,
        first_halt_is_volatility_pause=first.is_volatility_pause,
    )


def annotate_frame_with_halts(frame: pd.DataFrame, halt_records: Iterable[HaltRecord]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    required = {"ticker", "timestamp_ms"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"event dataset missing halt annotation columns: {sorted(missing)}")

    records = list(halt_records)
    annotated = frame.copy()
    rows = [
        annotate_event_halts(str(row["ticker"]), int(row["timestamp_ms"]), records).to_record()
        for _, row in annotated.iterrows()
    ]
    halt_frame = pd.DataFrame(rows, index=annotated.index)
    return pd.concat([annotated, halt_frame], axis=1)
