"""Source-time visual evidence. All frames are decoded from the candidate bytes."""
import base64
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

MEDIA_COMMAND_TIMEOUT = int(os.environ.get("EGO2ACT_MEDIA_TIMEOUT_SECONDS", "120"))


def digest(value):
    if isinstance(value, Path):
        with value.open('rb') as handle:
            return hashlib.file_digest(handle, 'sha256').hexdigest()
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        temporary = Path(handle.name)
    temporary.replace(path)


def image_parts(frames):
    parts = []
    for frame in frames:
        data = Path(frame['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != frame['sha256']:
            raise ValueError('Evidence bytes changed')
        label = {k: frame[k] for k in ('id', 'time', 'bbox')}
        parts.append({'type': 'text', 'text': json.dumps(label)})
        parts.append({'type': 'image_url', 'image_url': {
            'url': 'data:image/jpeg;base64,' + base64.b64encode(data).decode(), 'detail': 'low'}})
    return parts


class VideoEvidence:
    def __init__(self, source, cache, *, long_side=512):
        self.source, self.long_side = Path(source), long_side
        self.sha256 = digest(self.source)
        self.cache = Path(cache) / self.sha256
        self.cache.mkdir(parents=True, exist_ok=True)
        timeline = self.cache / 'timeline.json'
        if not timeline.exists():
            result = subprocess.run([
                'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
                '-show_entries', 'frame=best_effort_timestamp_time,pkt_duration_time',
                '-of', 'json', str(self.source)], capture_output=True, check=True,
                timeout=MEDIA_COMMAND_TIMEOUT)
            rows = json.loads(result.stdout)['frames']
            timestamps = [float(row['best_effort_timestamp_time']) for row in rows]
            if not timestamps or any(b <= a for a, b in zip(timestamps, timestamps[1:])):
                raise ValueError('Missing or nonincreasing source presentation times')
            last_duration = float(rows[-1].get('pkt_duration_time', 0))
            if last_duration <= 0:
                last_duration = timestamps[-1] - timestamps[-2] if len(rows) > 1 else 0.04
            write_json(timeline, {'origin': timestamps[0], 'timestamps': timestamps,
                                  'duration': timestamps[-1] - timestamps[0] + last_duration})
        data = json.loads(timeline.read_text())
        self.times = np.asarray(data['timestamps']) - data['origin']
        self.duration = data['duration']

    def inspect(self, start, end, bbox=None, *, count=12):
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= self.duration):
            raise ValueError('Interval must lie within original source seconds')
        if bbox is not None:
            if len(bbox) != 4 or not all(type(v) in (int, float) and math.isfinite(v) for v in bbox):
                raise ValueError('bbox must contain four finite normalized coordinates')
            if not (0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1):
                raise ValueError('bbox must be [left, top, right, bottom] within [0, 1]')
        if not 1 <= count <= 24:
            raise ValueError('Invalid frame payload size')
        candidates = np.flatnonzero((self.times >= start) & (self.times <= end))
        if not len(candidates):
            return []
        targets = np.linspace(start, min(end, self.times[candidates[-1]]), count)
        indices = sorted({int(candidates[np.abs(self.times[candidates] - t).argmin()]) for t in targets})
        frames = []
        for index in indices:
            identifier = digest([self.sha256, index, bbox, self.long_side, 85])
            path = self.cache / f'{identifier}.jpg'
            evidence_id = f'f{index:06d}-' + digest([bbox, self.long_side, 85])[:12]
            frames.append({'id': evidence_id, 'index': index, 'time': float(self.times[index]),
                           'bbox': bbox, 'path': str(path)})
        missing = [frame for frame in frames if not Path(frame['path']).exists()]
        if missing:
            expression = '+'.join(f'eq(n\\,{frame["index"]})' for frame in missing)
            with tempfile.TemporaryDirectory(dir=self.cache) as temporary:
                subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(self.source),
                                '-map', '0:v:0', '-an', '-vf', 'select=' + expression,
                                '-fps_mode', 'passthrough', '-threads', '1',
                                str(Path(temporary) / '%06d.png')],
                               capture_output=True, check=True, timeout=MEDIA_COMMAND_TIMEOUT)
                decoded = sorted(Path(temporary).glob('*.png'))
                if len(decoded) != len(missing):
                    raise ValueError('Decoded frame count differs from source timeline selection')
                for frame, path in zip(missing, decoded):
                    with Image.open(path) as original:
                        picture = original.convert('RGB')
                        if bbox is not None:
                            w, h = picture.size
                            picture = picture.crop((math.floor(bbox[0]*w), math.floor(bbox[1]*h),
                                                    math.ceil(bbox[2]*w), math.ceil(bbox[3]*h)))
                        picture.thumbnail((self.long_side, self.long_side), Image.Resampling.LANCZOS)
                        output = Path(temporary) / (frame['id'] + '.jpg')
                        picture.save(output, quality=85)
                        output.replace(frame['path'])
        for frame in frames:
            frame['sha256'] = digest(Path(frame['path']))
        return frames


INSPECT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'start_seconds': {'type': 'number', 'minimum': 0},
        'end_seconds': {'type': 'number', 'minimum': 0},
        'question': {'type': 'string', 'minLength': 1, 'maxLength': 400},
        'expected_evidence': {'type': 'string', 'minLength': 1, 'maxLength': 400},
        'bbox': {'anyOf': [{'type': 'null'}, {'type': 'array', 'items': {'type': 'number'},
                                          'minItems': 4, 'maxItems': 4}]}},
    'required': ['start_seconds', 'end_seconds', 'question', 'expected_evidence', 'bbox']}
