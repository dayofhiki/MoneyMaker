from datetime import date, datetime
from zoneinfo import ZoneInfo

from victory_trader.halts import annotate_event_halts, parse_nasdaq_halt_rss


ET = ZoneInfo("America/New_York")


def ms(hour: int, minute: int) -> int:
    return int(datetime(2026, 9, 15, hour, minute, tzinfo=ET).timestamp() * 1000)


SAMPLE_XML = """<?xml version="1.0"?>
<rss xmlns:ndaq="http://www.nasdaqtrader.com/">
  <channel>
    <item>
      <ndaq:IssueSymbol>TEST</ndaq:IssueSymbol>
      <ndaq:HaltDate>09/15/2026</ndaq:HaltDate>
      <ndaq:HaltTime>10:05:00</ndaq:HaltTime>
      <ndaq:ReasonCode>LUDP</ndaq:ReasonCode>
      <ndaq:ResumptionDate>09/15/2026</ndaq:ResumptionDate>
      <ndaq:ResumptionTradeTime>10:10:00</ndaq:ResumptionTradeTime>
      <ndaq:Mkt>NASDAQ</ndaq:Mkt>
    </item>
  </channel>
</rss>
"""


def test_parse_historical_halt_feed():
    records = parse_nasdaq_halt_rss(SAMPLE_XML, date(2026, 9, 15))
    assert len(records) == 1
    record = records[0]
    assert record.symbol == "TEST"
    assert record.reason_code == "LUDP"
    assert record.is_volatility_pause
    assert record.halt_at_ms == ms(10, 5)
    assert record.resume_at_ms == ms(10, 10)


def test_event_halt_annotation_uses_future_halt_only():
    records = parse_nasdaq_halt_rss(SAMPLE_XML, date(2026, 9, 15))
    annotation = annotate_event_halts("TEST", ms(10, 0), records)
    assert annotation.halt_within_5m
    assert annotation.halt_within_15m
    assert annotation.first_halt_minutes_after_event == 5.0
    assert annotation.first_halt_resume_minutes == 5.0
    assert annotation.first_halt_reason_code == "LUDP"


def test_event_after_halt_does_not_get_backward_annotation():
    records = parse_nasdaq_halt_rss(SAMPLE_XML, date(2026, 9, 15))
    annotation = annotate_event_halts("TEST", ms(10, 15), records)
    assert not annotation.halt_within_60m
    assert annotation.first_halt_minutes_after_event is None
