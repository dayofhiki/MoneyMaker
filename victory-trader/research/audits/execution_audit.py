"""Audit published v1.8 text summaries without changing policy or cost labels."""
import argparse
import json
import math
import zipfile


def audit(report):
    sections = {}
    section = None
    header = None
    for line in report.splitlines():
        if line.startswith('=== '):
            section = line.strip('= ').strip()
            header = None
        elif section in {'details', 'paths'} and line.strip():
            fields = line.split()
            if header is None:
                header = fields
            elif len(fields) == len(header):
                sections.setdefault(section, []).append(dict(zip(header, fields)))
    paths = {r['month']: r for r in sections['paths'] if r['policy'] == 'raw_same_fit_cap1'}
    rows = []
    for r in sections['details']:
        if r['policy'] != 'raw_same_fit_cap1':
            continue
        p = paths[r['month']]
        gross, net = float(r['gross_mean_pct']), float(r['base_mean_pct'])
        n, attempts, missing = int(r['trades']), int(float(p['buy_attempts'])), int(float(p['unevaluated_attempts']))
        if n + missing != attempts:
            raise ValueError('attempt/evaluation counts do not reconcile')
        friction = gross - net
        rows.append(dict(month=r['month'], evaluated=n, attempts=attempts,
                         unevaluated=missing, unevaluated_fraction=missing/attempts,
                         gross_mean_pct=gross, base_mean_pct=net,
                         modeled_friction_pp=friction,
                         descriptive_break_even_cost_pp=gross,
                         friction_reduction_for_zero_mean_fraction=1-gross/friction,
                         selected_prediction_error_pp=float(r['decision_score_mean'])-net,
                         missing_mean_needed_for_attempt_mean_zero_pct=-n*net/missing if missing else None))
    if len(rows) != 3 or not all(math.isfinite(r['modeled_friction_pp']) for r in rows):
        raise ValueError('expected three finite monthly raw-policy rows')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact_zip')
    args = parser.parse_args()
    with zipfile.ZipFile(args.artifact_zip) as archive:
        report = archive.read('state-calibrated-sequential-q-v18.txt').decode()
    print(json.dumps(audit(report), indent=2, allow_nan=False))
