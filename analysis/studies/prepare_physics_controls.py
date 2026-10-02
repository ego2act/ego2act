#!/usr/bin/env python
"""Physics negative controls: select successful human recordings and render clean,
teleport, swap and ghost variants of each.

Outputs (under --output): selection.csv, variants.json and videos/.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / 'analysis/csv/scores_ego2act.csv'
N_RECORDINGS = 30
MIN_S, MAX_S = 8.0, 40.0
C_MIN, C_TAIL = 4.0, 2.5
VARIANTS = ['clean', 'teleport', 'swap', 'ghost']
SCALE = "scale='if(lt(iw,ih),480,-2)':'if(lt(iw,ih),-2,480)':flags=bicubic"


def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name,pix_fmt'
                          ':format=duration', '-of', 'json', str(path)],
                         capture_output=True, check=True, text=True)
    data = json.loads(out.stdout)
    return {**data['streams'][0], 'duration': float(data['format']['duration'])}


def select(seed=0):
    rows = [r for r in csv.DictReader(SCORES.open(newline=''))
            if r['model'] == 'human_reference' and r['ego2act_physics']
            and float(r['ego2act_physics']) == 100 and (ROOT / r['video_path']).is_file()]
    by_case = defaultdict(list)
    for r in rows:
        info = probe(ROOT / r['video_path'])
        if MIN_S <= info['duration'] <= MAX_S:
            by_case[r['case_id']].append({**r, 'source_duration': info['duration']})
    rng = random.Random(seed)
    domains = defaultdict(list)
    for case in sorted(by_case):
        domains[by_case[case][0]['domain'] or 'Unassigned'].append(case)
    for d in sorted(domains):
        rng.shuffle(domains[d])
    picked, order = [], sorted(domains)
    while len(picked) < N_RECORDINGS and any(domains.values()):  # round-robin = even spread
        for d in order:
            if domains[d] and len(picked) < N_RECORDINGS:
                case = domains[d].pop(0)
                recs = sorted(by_case[case], key=lambda r: r['video_id'])
                picked.append({**rng.choice(recs), 'domain_group': d})
    return picked


def motion_centre(path, duration):
    fps = 10
    cmd = ['ffmpeg', '-v', 'error', '-i', str(path), '-an', '-vf',
           f"fps={fps},scale='if(lt(iw,ih),96,-2)':'if(lt(iw,ih),-2,96)',format=gray",
           '-f', 'rawvideo', '-']
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    w, h = gray_size(path)
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32)
    energy = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))  # energy[k] ~ time (k+1)/fps
    times = (np.arange(len(energy)) + 1) / fps
    best, best_c = -1.0, None
    for c in np.arange(C_MIN, duration - C_TAIL + 1e-9, 0.1):
        m = energy[(times >= c - 1.5) & (times < c + 1.5)].mean()
        if m > best:
            best, best_c = float(m), round(float(c), 1)
    return best_c, best, float(energy.mean())


def gray_size(path):
    out = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-frames:v', '1', '-an', '-vf',
                          "scale='if(lt(iw,ih),96,-2)':'if(lt(iw,ih),-2,96)',format=gray",
                          '-f', 'rawvideo', '-'], capture_output=True, check=True).stdout
    tmp = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-frames:v', '1', '-an', '-vf',
                          "scale='if(lt(iw,ih),96,-2)':'if(lt(iw,ih),-2,96)',format=gray",
                          '-f', 'image2pipe', '-c:v', 'pgm', '-'], capture_output=True, check=True).stdout
    w, h = map(int, tmp.split(b'\n')[1].split())
    assert w * h == len(out)
    return w, h


def decode(path, fps):
    size = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-frames:v', '1', '-an', '-vf', SCALE,
                           '-f', 'image2pipe', '-c:v', 'ppm', '-'], capture_output=True, check=True).stdout
    w, h = map(int, size.split(b'\n')[1].split())
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-an', '-vf', f'fps={fps},{SCALE}',
                          '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3), (w, h)


def encode(frames, order, blend, fps, path):
    """Stream frames[order] to libx264; blend maps output position -> overlay source index."""
    _, h, w, _ = frames.shape
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
           '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for k, i in enumerate(order):
        frame = frames[i]
        if k in blend:
            frame = ((frame.astype(np.uint16) + frames[blend[k]]) // 2).astype(np.uint8)
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(proc.stderr.read().decode()[:500])


def render(rec, out_dir):
    src = ROOT / rec['video_path']
    info = probe(src)
    fps = Fraction(info['r_frame_rate']).limit_denominator(1001)
    if fps > 60:  # guard for high-frame-rate captures; none expected
        fps = Fraction(30)
    F = float(fps)
    c, peak, mean = motion_centre(src, rec['source_duration'])
    frames, (w, h) = decode(src, fps)
    n = len(frames)
    idx = lambda t: int(round(t * F))
    edits = {
        'clean': {},
        'teleport': {'delete_source': [c - 0.75, c + 0.75]},
        'swap': {'segment_a_source': [c - 1.5, c], 'segment_b_source': [c, c + 1.5]},
        'ghost': {'blend_source': [c - 1.0, c + 1.0], 'overlay_offset_seconds': -3.0, 'alpha': 0.5},
    }
    out = {}
    for variant in VARIANTS:
        order, blend = list(range(n)), {}
        if variant == 'clean':
            ev = {'edit_start_s': None, 'edit_end_s': None}
        elif variant == 'teleport':
            a, b = idx(c - 0.75), idx(c + 0.75)
            order = order[:a] + order[b:]
            ev = {'edit_start_s': a / F, 'edit_end_s': b / F, 'deleted_frames': b - a,
                  'output_cut_at_s': a / F}
        elif variant == 'swap':
            a, m, b = idx(c - 1.5), idx(c), idx(c + 1.5)
            order = order[:a] + order[m:b] + order[a:m] + order[b:]
            ev = {'edit_start_s': a / F, 'edit_end_s': b / F, 'segment_frames': [m - a, b - m]}
        else:
            a, b, off = idx(c - 1.0), idx(c + 1.0), idx(3.0)
            blend = {k: k - off for k in range(a, b)}
            ev = {'edit_start_s': a / F, 'edit_end_s': b / F, 'overlay_offset_frames': -off}
        name = f"{rec['case_id']}__{rec['video_id'].split('::')[1]}__{variant}.mp4"
        path = out_dir / name
        try:
            reuse = int(probe(path).get('nb_frames', -1)) == len(order)
        except (subprocess.CalledProcessError, KeyError, ValueError):
            reuse = False
        if not reuse:
            encode(frames, order, blend, fps, path)
        check = probe(path)
        ok = (check['codec_name'] == 'h264' and check['pix_fmt'] == 'yuv420p'
              and int(check['nb_frames']) == len(order) and min(check['width'], check['height']) == 480
              and Fraction(check['r_frame_rate']) == fps)
        out[variant] = {'video_id': f"{rec['video_id']}::{variant}", 'case_id': rec['case_id'],
                        'recording_id': rec['video_id'], 'variant': variant, 'path': str(path.relative_to(ROOT)),
                        'fps': str(fps), 'source_frames_cfr': n, 'output_frames': len(order),
                        'output_duration_s': len(order) / F, 'activity_centre_s': c,
                        'window_motion_energy': peak, 'mean_motion_energy': mean,
                        **edits[variant], **ev, 'ffprobe': check, 'ffprobe_ok': ok}
        assert ok, (name, check)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '.outputs/physics_controls')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    videos = args.output / 'videos'
    videos.mkdir(parents=True, exist_ok=True)
    picked = select()
    fields = ['video_id', 'case_id', 'domain', 'domain_group', 'video_path', 'start_path', 'goal',
              'source_duration', 'ego2act_task', 'ego2act_physics', 'ego2act_score']
    with (args.output / 'selection.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(picked)
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda r: render(r, videos), picked))
    manifest = [v for r in results for v in r.values()]
    (args.output / 'variants.json').write_text(json.dumps(manifest, indent=2))
    print(f'{len(picked)} recordings, {len(manifest)} videos written to {videos}')


if __name__ == '__main__':
    main()
