#!/usr/bin/env python
"""Judge-side gate ablation on the human-rated panel (production Flash backbone).

Runs the judge with ``gating='all'`` so every Task gate (T1-T3) and every
Physics gate (P1-P4) is asked for every subgoal/window. Each video yields both
the production sequential scores (first failure ends the level) and the
independent scores (total gates passed), computed from the same answers.

Resumable (IDs already in scores.jsonl are skipped), a hard global budget cap
(USD, summed over the per-video cost ledgers), retries for transient failures
(the failed attempt's artifacts are moved aside so its spend stays counted),
and failures are recorded in failures.jsonl instead of crashing.
"""
from __future__ import annotations

import argparse
import csv
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

ROOT = Path(__file__).resolve().parents[2]
PER_VIDEO_CAP = 2.0  # same per-video safety cap as the production batch runs
LOG = logging.getLogger('gate_ablation')


def load_jobs():
    human = [row for row in csv.DictReader((ROOT / 'analysis/csv/scores_human.csv').open(newline=''))
             if row.get('human_task')]
    goals = {row['case_id']: row['description'] for row in csv.DictReader((ROOT / 'analysis/csv/metadata.csv').open(newline=''))}
    jobs = []
    for row in human:
        video_id = row['id']
        case_id, video_key = video_id.split('::', 1)
        start = ROOT / 'data' / 'ego2act' / case_id / 'start.jpg'
        if row['model'] == 'human_wrong':
            candidates = sorted((ROOT / 'data' / 'ego2act' / case_id / 'wrong').glob(f'{video_key}.*'))
        else:
            candidates = sorted((ROOT / 'data' / 'ego2act' / case_id / 'AI' / row['model']).glob(f"*{row['seed']}*.mp4"))
        if not candidates or not start.is_file():
            raise FileNotFoundError(f'No media for {video_id}')
        jobs.append((video_id, case_id, row['model'], goals[case_id], start, candidates[0]))
    return jobs


def ledger_total(path):
    """Settled cost plus still-held reservations (unknown charges count at their bound)."""
    try:
        with sqlite3.connect(path, timeout=30) as db:
            return db.execute('SELECT COALESCE(SUM(COALESCE(cost, reserved)),0) FROM calls').fetchone()[0]
    except sqlite3.Error:
        return 0.0


class Budget:
    """Global hard cap across per-video ledgers; checked before every paid call."""

    def __init__(self, cap, spent, margin):
        self.cap, self.spent, self.margin, self.lock = cap, spent, margin, threading.Lock()
        self.exhausted = False

    def check(self):
        with self.lock:
            if self.spent + self.margin >= self.cap:
                self.exhausted = True
                raise JudgeStop(f'Global ablation budget reached (${self.spent:.3f} of ${self.cap})')

    def add(self, amount):
        with self.lock:
            self.spent += amount


class BudgetedProvider(Provider):
    def __init__(self, config, ledger, phase, budget):
        super().__init__(config, ledger, phase)
        self.budget = budget

    def call(self, messages, functions, directory):
        if (Path(directory) / 'response.json').exists():  # cached replay: already counted
            return super().call(messages, functions, directory)
        self.budget.check()
        try:
            response = super().call(messages, functions, directory)
        except Exception:
            self.budget.add(0.01)  # conservative stand-in for an unknown/failed charge
            raise
        self.budget.add(response.get('usage', {}).get('cost') or 0.0)
        return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '.outputs/gate_ablation')
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--budget-usd', type=float, default=15.0)
    parser.add_argument('--margin-usd', type=float, default=0.3,
                        help='Stop dispatching paid calls this far below the cap (in-flight calls).')
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--limit', type=int, help='Only the first N videos of the shuffled order (smoke tests).')
    parser.add_argument('--ids', nargs='*', help='Explicit video IDs (smoke tests).')
    args = parser.parse_args()
    for line in (ROOT / '.env').read_text().splitlines():  # same variables run_*.sh source
        key, sep, value = line.strip().partition('=')
        if sep and not key.startswith('#'):
            os.environ.setdefault(key.removeprefix('export ').strip(), value.strip().strip('"\''))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / 'run.log')])

    jobs = load_jobs()
    random.Random(0).shuffle(jobs)  # a budget-truncated run stays a random subset of the panel
    if args.ids:
        jobs = [job for job in jobs if job[0] in set(args.ids)]
    if args.limit:
        jobs = jobs[:args.limit]
    scores_path, traces_path = output / 'scores.jsonl', output / 'traces.jsonl'
    failures_path = output / 'failures.jsonl'
    done = ({json.loads(line)['video_id'] for line in scores_path.read_text().splitlines() if line.strip()}
            if scores_path.exists() else set())
    pending = [job for job in jobs if job[0] not in done]
    spent = sum(ledger_total(p) for p in output.glob('artifacts*/*/costs.sqlite'))
    budget = Budget(args.budget_usd, spent, args.margin_usd)
    LOG.info('queued %d of %d videos (skipped %d done); spent so far $%.4f; cap $%.2f; workers %d',
             len(pending), len(jobs), len(jobs) - len(pending), spent, args.budget_usd, args.workers)

    def run(job):
        video_id, case_id, model, goal, start, video = job
        slug = video_id.replace('::', '____')
        error = None
        for attempt in range(args.retries + 1):
            if budget.exhausted:
                return None, None, 'budget: global cap reached before start'
            artifact = output / 'artifacts' / slug
            started = time.monotonic()
            try:
                ledger = Ledger(artifact / 'costs.sqlite', {'judge': PER_VIDEO_CAP}, ceiling=None)
                provider = BudgetedProvider(dict(PAPER_PROVIDER), ledger, 'judge', budget)
                result = judge(video, goal, start, output=artifact, provider=provider,
                               budget_usd=PER_VIDEO_CAP, gating='all')
                cost = ledger_total(artifact / 'costs.sqlite')
                score = {'video_id': video_id, 'case_id': case_id, 'model': model,
                         'seq_task': result['task100'], 'seq_physics': result['physics100'],
                         'ind_task': result['task100_independent'],
                         'ind_physics': result['physics100_independent'],
                         'task_status': result['task'].get('status'),
                         'physics_status': result['physics'].get('status'),
                         'cost_usd': cost, 'attempts': attempt + 1,
                         'elapsed_seconds': time.monotonic() - started}
                trace = {'video_id': video_id, 'case_id': case_id, 'model': model,
                         'judge_model': result['model'], 'plans': result['plans'],
                         **{axis: {k: result[axis].get(k) for k in
                                   ('score', 'status', 'levels', 'gate_answers', 'independent', 'reason')}
                            for axis in ('task', 'physics')}}
                return score, trace, None
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                if budget.exhausted or 'Global ablation budget' in error:
                    return None, None, 'budget: ' + error
                LOG.warning('%s attempt %d failed: %s', video_id, attempt + 1, error)
                if artifact.exists():  # keep spend on record, restart cleanly
                    aside = output / 'artifacts_failed' / f'{slug}__attempt{attempt + 1}_{int(time.time())}'
                    aside.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(artifact), str(aside))
                time.sleep(5 * (attempt + 1))
        return None, None, error

    started = time.monotonic()
    completed = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, scores_path.open('a') as scores, \
            traces_path.open('a') as traces, failures_path.open('a') as failures:
        futures = {pool.submit(run, job): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            score, trace, error = future.result()
            if error:
                failed += 1
                failures.write(json.dumps({'video_id': job[0], 'error': error, 'time': time.time()}) + '\n')
                failures.flush()
                LOG.error('%s failed: %s', job[0], error)
            else:
                completed += 1
                scores.write(json.dumps(score) + '\n'); scores.flush()
                traces.write(json.dumps(trace) + '\n'); traces.flush()
                LOG.info('done %s  seq T/P %.1f/%s  ind T/P %.1f/%s  $%.4f  [%d ok, %d failed, %d/%d, $%.3f total, %.0fs]',
                         job[0], score['seq_task'] or 0, score['seq_physics'], score['ind_task'] or 0,
                         score['ind_physics'], score['cost_usd'], completed, failed,
                         completed + failed, len(pending), budget.spent, time.monotonic() - started)
    total = sum(ledger_total(p) for p in output.glob('artifacts*/*/costs.sqlite'))
    LOG.info('finished: %d completed, %d failed this session; ledger total $%.4f; %.0fs',
             completed, failed, total, time.monotonic() - started)


if __name__ == '__main__':
    main()
