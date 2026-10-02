#!/usr/bin/env python
"""Physics negative controls: run the judge on the variants from
prepare_physics_controls.py (resumable, with a spend cap).

Outputs (under --output): artifacts/, scores.jsonl, failures.jsonl, run.log.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ego2act.judge.main import PAPER_PROVIDER, Ledger, Provider, JudgeStop, judge
from ego2act.judge.tools import write_json

ROOT = Path(__file__).resolve().parents[2]
PER_VIDEO_CAP = 2.0
LOG = logging.getLogger('physics_controls')


def ledger_total(path):
    try:
        with sqlite3.connect(path, timeout=30) as db:
            return db.execute('SELECT COALESCE(SUM(COALESCE(cost, reserved)),0) FROM calls').fetchone()[0]
    except sqlite3.Error:
        return 0.0


class Budget:
    def __init__(self, cap, spent, margin):
        self.cap, self.spent, self.margin, self.lock = cap, spent, margin, threading.Lock()
        self.exhausted = False

    def check(self):
        with self.lock:
            if self.spent + self.margin >= self.cap:
                self.exhausted = True
                raise JudgeStop(f'Global controls budget reached (${self.spent:.3f} of ${self.cap})')

    def add(self, amount):
        with self.lock:
            self.spent += amount


class BudgetedProvider(Provider):
    def __init__(self, config, ledger, phase, budget):
        super().__init__(config, ledger, phase)
        self.budget = budget

    def call(self, messages, functions, directory):
        if (Path(directory) / 'response.json').exists():
            return super().call(messages, functions, directory)
        self.budget.check()
        try:
            response = super().call(messages, functions, directory)
        except Exception:
            self.budget.add(0.01)
            raise
        self.budget.add(response.get('usage', {}).get('cost') or 0.0)
        return response


def load_jobs(output):
    jobs = []
    for v in json.loads((output / 'variants.json').read_text()):
        case = v['case_id']
        goal = json.loads((ROOT / 'data' / 'ego2act' / case / 'metadata.json').read_text())['goal']
        jobs.append({**v, 'goal': goal, 'start': ROOT / 'data' / 'ego2act' / case / 'start.jpg', 'video': ROOT / v['path']})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '.outputs/physics_controls')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--budget-usd', type=float, default=6.0)
    parser.add_argument('--margin-usd', type=float, default=0.15)
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--ids', nargs='*')
    args = parser.parse_args()
    for line in (ROOT / '.env').read_text().splitlines():
        key, sep, value = line.strip().partition('=')
        if sep and not key.startswith('#'):
            os.environ.setdefault(key.removeprefix('export ').strip(), value.strip().strip('"\''))
    output = args.output
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / 'run.log')])
    jobs = load_jobs(output)
    # Recording-level shuffle keeps a recording's four variants together, so a
    # budget-truncated run still yields complete matched sets.
    recordings = sorted({j['recording_id'] for j in jobs})
    random.Random(0).shuffle(recordings)
    rank = {r: i for i, r in enumerate(recordings)}
    jobs.sort(key=lambda j: (rank[j['recording_id']], j['variant']))
    if args.ids:
        jobs = [j for j in jobs if j['video_id'] in set(args.ids)]
    scores_path, failures_path = output / 'scores.jsonl', output / 'failures.jsonl'
    done = ({json.loads(l)['video_id'] for l in scores_path.read_text().splitlines() if l.strip()}
            if scores_path.exists() else set())
    pending = [j for j in jobs if j['video_id'] not in done]
    spent = sum(ledger_total(p) for p in output.glob('artifacts*/*/costs.sqlite'))
    budget = Budget(args.budget_usd, spent, args.margin_usd)
    LOG.info('queued %d of %d videos (skipped %d done); spent so far $%.4f; cap $%.2f; workers %d',
             len(pending), len(jobs), len(jobs) - len(pending), spent, args.budget_usd, args.workers)

    def run(job):
        slug = job['video_id'].replace('::', '____')
        error = None
        for attempt in range(args.retries + 1):
            if budget.exhausted:
                return None, 'budget: global cap reached before start'
            artifact = output / 'artifacts' / slug
            started = time.monotonic()
            try:
                ledger = Ledger(artifact / 'costs.sqlite', {'judge': PER_VIDEO_CAP}, ceiling=None)
                provider = BudgetedProvider(dict(PAPER_PROVIDER), ledger, 'judge', budget)
                result = judge(job['video'], job['goal'], job['start'], output=artifact,
                               provider=provider, budget_usd=PER_VIDEO_CAP)
                write_json(artifact / 'judge_result.json', result)
                score = {'video_id': job['video_id'], 'case_id': job['case_id'],
                         'recording_id': job['recording_id'], 'variant': job['variant'],
                         'task100': result['task100'], 'physics100': result['physics100'],
                         'final100': result['final100'],
                         'task_status': result['task'].get('status'),
                         'physics_status': result['physics'].get('status'),
                         'task_levels': result['task'].get('levels'),
                         'physics_levels': result['physics'].get('levels'),
                         'cost_usd': ledger_total(artifact / 'costs.sqlite'), 'attempts': attempt + 1,
                         'elapsed_seconds': time.monotonic() - started}
                return score, None
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                if budget.exhausted or 'Global controls budget' in error:
                    return None, 'budget: ' + error
                LOG.warning('%s attempt %d failed: %s', job['video_id'], attempt + 1, error)
                if artifact.exists():
                    aside = output / 'artifacts_failed' / f'{slug}__attempt{attempt + 1}_{int(time.time())}'
                    aside.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(artifact), str(aside))
                time.sleep(5 * (attempt + 1))
        return None, error

    started = time.monotonic()
    completed = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, scores_path.open('a') as scores, \
            failures_path.open('a') as failures:
        futures = {pool.submit(run, job): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            score, error = future.result()
            if error:
                failed += 1
                failures.write(json.dumps({'video_id': job['video_id'], 'error': error,
                                           'status': 'budget' if error.startswith('budget') else 'failed',
                                           'time': time.time()}) + '\n')
                failures.flush()
                LOG.error('%s failed: %s', job['video_id'], error)
            else:
                completed += 1
                scores.write(json.dumps(score) + '\n'); scores.flush()
                LOG.info('done %s  T/P %s/%s  $%.4f  [%d ok, %d failed, %d/%d, $%.3f total, %.0fs]',
                         job['video_id'], score['task100'], score['physics100'], score['cost_usd'],
                         completed, failed, completed + failed, len(pending), budget.spent,
                         time.monotonic() - started)
    total = sum(ledger_total(p) for p in output.glob('artifacts*/*/costs.sqlite'))
    LOG.info('finished: %d completed, %d failed this session; ledger total $%.4f; %.0fs',
             completed, failed, total, time.monotonic() - started)


if __name__ == '__main__':
    main()
