"""Frozen bar-reference replay; simulated fills, never brokerage executions."""
from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .execution_costs import DEFAULT_EXECUTION_SCENARIOS, modeled_buy_fill, modeled_sell_fill

MINUTE = 60_000
INITIAL_CASH = 10_000.0
ORDER_BUDGET = 1_000.0


def positive(value):
    return pd.notna(value) and np.isfinite(value) and value > 0


def replay(attempts, panel, scenario, *, initial_cash=INITIAL_CASH, budget=ORDER_BUDGET):
    """Replay one policy/month with fixed-notional synthetic fractional orders.

    Signals become known at decision-bar close. Reserve cash immediately; next
    minute open is a zero-latency reference, not an observed execution. Missing
    entry reference expires the synthetic order. Exit at scheduled bar close;
    if absent, use first subsequent observed open without retroactive pricing.
    Unresolved positions retain capital and the last known mark indefinitely.
    """
    if not positive(initial_cash) or not positive(budget):
        raise ValueError('cash and budget must be finite and positive')
    required = {'ticker', 't', 'o', 'c'}
    if not required.issubset(panel.columns):
        raise ValueError('panel requires ticker/t/o/c')
    if panel.duplicated(['ticker', 't']).any():
        raise ValueError('duplicate observed bars')
    if attempts.duplicated(['ticker', 'decision_t']).any():
        raise ValueError('duplicate policy decisions')
    bars = {str(t): f.sort_values('t') for t, f in panel.groupby('ticker')}
    events = []
    records = []
    serial = 0

    def event(t, priority, kind, payload):
        nonlocal serial
        serial += 1
        heapq.heappush(events, (int(t), priority, serial, kind, payload))

    for r in attempts.sort_values(['decision_t', 'ticker']).to_dict('records'):
        if not positive(r['decision_t']):
            raise ValueError('invalid decision timestamp')
        event(int(r['decision_t']) + MINUTE, 2, 'signal', r)
    cash = float(initial_cash)
    positions = {}
    pending = {}
    realized = 0.0
    equity_peak = initial_cash
    max_drawdown = 0.0
    snapshots = []
    missing_entries = 0
    accepted = 0
    delayed = 0
    closed = 0
    turnover = 0.0

    while events:
        timestamp, _, _, kind, data = heapq.heappop(events)
        ticker = str(data['ticker'])
        if kind == 'signal':
            record = dict(data, scenario=scenario.name, decision_known_t=timestamp)
            records.append(record)
            if ticker in positions or ticker in pending:
                record['status'] = 'blocked_open_position'
            elif cash + 1e-9 < budget:
                record['status'] = 'blocked_cash'
            else:
                cash -= budget
                pending[ticker] = budget
                record['status'] = 'reserved'
                event(timestamp, 3, 'entry', record)
        elif kind == 'entry':
            reserved = pending.pop(ticker)
            entry = data.get('entry_price', np.nan)
            if not positive(entry):
                cash += reserved
                missing_entries += 1
                data['status'] = 'entry_reference_missing_expired_assumption'
            else:
                accepted += 1
                qty = reserved / modeled_buy_fill(float(entry), scenario)
                positions[ticker] = {'qty': qty, 'cost': reserved, 'mark': float(entry),
                                     'mark_t': timestamp, 'record': data}
                turnover += reserved
                data.update(status='open_unresolved', entry_t=timestamp, quantity=qty)
                horizon = data.get('action_horizon_min', 10)
                if not positive(horizon):
                    raise ValueError('invalid action horizon')
                target = int(data['decision_t']) + int(horizon) * MINUTE
                data['scheduled_exit_bar_t'] = target
                observed = bars.get(ticker, pd.DataFrame(columns=['t', 'o', 'c']))
                exact = observed.loc[observed.t.eq(target)]
                exit_t = None
                if not exact.empty and positive(exact.iloc[0].c):
                    exit_t, price = target + MINUTE, float(exact.iloc[0].c)
                    exit_kind = 'scheduled_close_reference'
                else:
                    later = observed.loc[observed.t.gt(target) & observed.o.map(positive)]
                    if not later.empty:
                        exit_t, price = int(later.iloc[0].t), float(later.iloc[0].o)
                        exit_kind = 'delayed_open_reference'
                # Offline event scheduling is evaluation only; no outcome is used
                # in acceptance, sizing, cash availability, or signal selection.
                mark_rows = observed.loc[observed.t.add(MINUTE).gt(timestamp)]
                if exit_t is not None:
                    mark_rows = mark_rows.loc[mark_rows.t.add(MINUTE).lt(exit_t)]
                for bar in mark_rows.itertuples():
                    if positive(bar.c):
                        event(int(bar.t) + MINUTE, 1, 'mark',
                              {'ticker': ticker, 'price': float(bar.c), 'record': data})
                if exit_t is not None:
                    event(exit_t, 0, 'exit', {'ticker': ticker, 'price': price,
                                             'exit_kind': exit_kind, 'record': data})
        elif kind == 'mark':
            pos = positions.get(ticker)
            if pos is not None and pos['record'] is data['record']:
                pos.update(mark=data['price'], mark_t=timestamp)
        elif kind == 'exit':
            pos = positions.pop(ticker)
            proceeds = pos['qty'] * modeled_sell_fill(data['price'], scenario)
            proceeds *= 1 - scenario.sell_fee_bps / 10_000
            pnl = proceeds - pos['cost']
            cash += proceeds
            realized += pnl
            turnover += proceeds
            closed += 1
            delayed += data['exit_kind'] == 'delayed_open_reference'
            data['record'].update(status='closed', exit_t=timestamp,
                                  exit_kind=data['exit_kind'], realized_pnl=pnl,
                                  net_return_pct=pnl / pos['cost'] * 100)
        marked = sum(p['qty'] * p['mark'] for p in positions.values())
        equity = cash + sum(pending.values()) + marked
        equity_peak = max(equity_peak, equity)
        max_drawdown = max(max_drawdown, (1 - equity / equity_peak) * 100)
        snapshots.append({'t': timestamp, 'cash': cash, 'marked_equity': equity,
                          'open_positions': len(positions),
                          'committed_capital': sum(p['cost'] for p in positions.values())})
        if cash < -1e-7:
            raise AssertionError('negative cash')
    for pos in positions.values():
        pos['record'].update(last_mark=pos['mark'], last_mark_t=pos['mark_t'],
                              committed_capital=pos['cost'])
    ending_marked = cash + sum(p['qty'] * p['mark'] for p in positions.values())
    summary = dict(scenario=scenario.name, attempts=len(attempts), accepted=accepted,
                   closed=closed, delayed_exits=delayed, unresolved=len(positions),
                   missing_entry_references=missing_entries,
                   blocked_cash=sum(r['status'] == 'blocked_cash' for r in records),
                   blocked_position=sum(r['status'] == 'blocked_open_position' for r in records),
                   initial_cash=initial_cash, ending_cash=cash,
                   committed_capital=sum(p['cost'] for p in positions.values()),
                   realized_pnl=realized, ending_stale_marked_equity=ending_marked,
                   marked_return_pct=(ending_marked / initial_cash - 1) * 100,
                   observed_mark_drawdown_pct=max_drawdown, turnover=turnover,
                   complete_accounting=not positions and missing_entries == 0,
                   promotion_allowed=False)
    return pd.DataFrame(records), summary, pd.DataFrame(snapshots)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempts', type=Path, required=True)
    parser.add_argument('--dataset', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    attempts = pd.read_csv(args.attempts)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for spec in args.dataset:
        month, path = spec.split('=', 1)
        panel = pd.read_parquet(path, columns=['ticker', 't', 'o', 'c'])
        selected = attempts.loc[attempts.month.astype(str).eq(month)]
        for policy in sorted(attempts.policy.unique()):
            group = selected.loc[selected.policy.eq(policy)]
            for scenario in DEFAULT_EXECUTION_SCENARIOS:
                records, summary, curve = replay(group, panel, scenario)
                summary.update(month=month, policy=policy)
                summaries.append(summary)
                prefix = args.output / f'{month}-{policy}-{scenario.name}'
                records.to_csv(str(prefix) + '-ledger.csv', index=False)
                curve.to_csv(str(prefix) + '-equity.csv', index=False)
    pd.DataFrame(summaries).to_csv(args.output / 'summary.csv', index=False)
    report = {'interpretation': 'Development-only frozen-signal bar-reference replay; '
              'monthly independent accounts; fractional fixed-notional orders; '
              'zero latency assumed; stale marks are not realized exits. Missing '
              'bars have unknown cause. No evidence of actual fills or alpha.',
              'results': summaries}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
