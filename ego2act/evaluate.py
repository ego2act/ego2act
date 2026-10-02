"""`ego2act judge`: score videos with Ego2ActJudge.

    ego2act judge video --video V.mp4 --goal "..." --initial-image start.jpg
    ego2act judge batch --data data/ego2act [--case laptop_open] [--workers 4]

`batch` judges every video under <data>/<case>/ (human recordings and
AI/<model>/*.mp4), reading the goal from <case>/metadata.json and the initial
scene from <case>/start.jpg. Results are appended to scores.jsonl and
traces.jsonl in --output, and finished videos are skipped on rerun.
"""
import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .data import case_metadata
from .judge.tools import write_json

ROOT = Path(__file__).resolve().parents[1]
VIDEO_SUFFIXES = {'.mp4', '.mov', '.webm', '.avi', '.mkv'}
LOG = logging.getLogger('ego2act.evaluate')


def _batch_rows(data_root, selected, excluded_models=()):
    data_root = Path(data_root)
    names = selected or sorted(p.name for p in data_root.iterdir() if p.is_dir())
    for case_id in names:
        case = data_root / case_id
        start, metadata = case / 'start.jpg', case_metadata(case)
        if not start.is_file() or not metadata:
            LOG.warning('skip %s: missing start.jpg or goal (metadata.json / prompt.txt)', case_id)
            continue
        goal = metadata['goal']
        for video in sorted(p for p in case.rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES):
            parts = video.relative_to(case).parts
            if parts[0] == 'AI' and len(parts) > 1 and parts[1] in set(excluded_models):
                continue
            yield case_id, goal, start, video


def _append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def batch(args):
    from tqdm import tqdm
    from .judge.main import judge
    output = args.output.resolve()
    traces, scores = output / 'traces.jsonl', output / 'scores.jsonl'
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / 'evaluation.log')])
    done = set()
    if scores.exists():
        done = {json.loads(line)['video_id'] for line in scores.read_text().splitlines() if line.strip()}
    jobs = []
    for case_id, goal, start, video in _batch_rows(args.data, args.case, args.exclude_model):
        video_id = str(video.relative_to(args.data))
        if video_id not in done or args.overwrite:
            jobs.append((case_id, goal, start, video, video_id))
    LOG.info('queued %d videos with %d workers; skipped %d completed', len(jobs), args.workers, len(done))

    def run(job):
        case_id, goal, start, video, video_id = job
        started = time.monotonic()
        try:
            result = judge(video, goal, start, output=output / 'artifacts' / video_id.replace('/', '__'),
                           budget_usd=args.budget_usd)
            trace = {'video_id': video_id, 'case_id': case_id, 'task': result['task'],
                     'physics': result['physics'], 'model': result['model']}
            score = {'video_id': video_id, 'case_id': case_id, 'task100': result['task100'],
                     'physics100': result['physics100'], 'final100': result['final100']}
            return trace, score, time.monotonic() - started, None
        except Exception as exc:
            return None, None, time.monotonic() - started, f'{type(exc).__name__}: {exc}'

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc='Ego2ActJudge'):
            trace, score, elapsed, error = future.result()
            if error:
                LOG.error('video failed after %.1fs: %s', elapsed, error)
                continue
            _append_jsonl(traces, trace)
            _append_jsonl(scores, score)
            LOG.info('completed %s in %.1fs', score['video_id'], elapsed)


def main(argv=None):
    from dotenv import load_dotenv
    parser = argparse.ArgumentParser(prog='ego2act judge', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['video', 'batch'])
    parser.add_argument('--video', type=Path)
    parser.add_argument('--goal')
    parser.add_argument('--initial-image', type=Path)
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'ego2act')
    parser.add_argument('--case', action='append', help='case ID (repeatable); default: all cases')
    parser.add_argument('--exclude-model', action='append', default=[])
    parser.add_argument('--output', type=Path, default=ROOT / 'judge_output')
    parser.add_argument('--budget-usd', type=float, default=2.0, help='spending cap per video')
    parser.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    load_dotenv(ROOT / '.env')
    if args.command == 'batch':
        if args.workers < 1:
            parser.error('--workers must be positive')
        batch(args)
        return 0
    if not args.video or not args.goal or not args.initial_image:
        parser.error('video requires --video, --goal, and --initial-image')
    from .judge.main import judge
    result = judge(args.video, args.goal, args.initial_image, output=args.output, budget_usd=args.budget_usd)
    write_json(args.output / 'result.json', result)
    print(json.dumps({k: result[k] for k in ('task100', 'physics100', 'final100')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
