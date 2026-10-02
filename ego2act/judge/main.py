"""Ego2ActJudge: plans, staged gate calls with an inspection tool, cost ledger, judge()."""
import json
import math
import os
import sqlite3
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from jsonschema import validate
from jsonschema.exceptions import ValidationError

from . import physics, prompts, task
from .tools import INSPECT_SCHEMA, VideoEvidence, digest, image_parts, write_json

class JudgeStop(RuntimeError):
    """A resource, billing or provider boundary, not a rubric verdict."""


class ContextLimit(JudgeStop):
    """This axis cannot continue within the frozen context envelope."""


class UnresolvedDecision(JudgeStop):
    """Further inspection supplies no new evidence for the current question."""


class Ledger:
    """Per-output-folder spending cap; every paid call is reserved before dispatch."""
    def __init__(self, path, caps):
        self.path, self.caps, self.ceiling = Path(path), caps, None
        if any(v < 0 for v in self.caps.values()):
            raise ValueError('Spending caps must be non-negative')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, caps TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, phase TEXT, '
                       'reserved REAL, cost REAL, state TEXT, pid INTEGER)')
            db.execute('INSERT OR IGNORE INTO settings VALUES (1, ?)', (json.dumps(self.caps, sort_keys=True),))
            if json.loads(db.execute('SELECT caps FROM settings WHERE id=1').fetchone()[0]) != self.caps:
                raise JudgeStop('Existing ledger has a different spending cap; spending cannot be reset')

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def reserve(self, id, phase, amount):
        if phase not in self.caps or not math.isfinite(amount) or amount < 0:
            raise ValueError('Invalid charge reservation')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            rows = db.execute('SELECT phase,reserved,cost,state,pid FROM calls').fetchall()
            for _, _, _, state, pid in rows:
                if state == 'unknown':
                    raise JudgeStop('Unreconciled provider charge; paid dispatch stopped')
                if state == 'reserved':
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        raise JudgeStop('Interrupted reservation needs billing reconciliation') from None
            used = sum(cost if cost is not None else held for _, held, cost, _, _ in rows)
            phase_used = sum(cost if cost is not None else held for p, held, cost, _, _ in rows if p == phase)
            if (self.ceiling is not None and used + amount > self.ceiling) or phase_used + amount > self.caps[phase]:
                raise JudgeStop(f'{phase} reservation exceeds the spending cap')
            db.execute('INSERT INTO calls VALUES (?,?,?,NULL,?,?)', (id, phase, amount, 'reserved', os.getpid()))

    def reconcile(self, id, cost):
        if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
            raise JudgeStop('Missing or invalid actual provider cost')
        with self.connect() as db:
            row = db.execute('SELECT reserved,cost FROM calls WHERE id=?', (id,)).fetchone()
            if row is None or (row[1] is not None and abs(row[1] - cost) > 1e-9):
                raise JudgeStop('Charge identity mismatch')
            db.execute('UPDATE calls SET cost=?,state=? WHERE id=?', (cost, 'settled', id))
        if cost > row[0] + 1e-8:
            raise JudgeStop('Actual charge exceeded conservative reservation; recheck billing bounds')

    def unknown(self, id):
        with self.connect() as db:
            db.execute("UPDATE calls SET state='unknown' WHERE id=? AND cost IS NULL", (id,))

    def hold_unknown(self, id):
        """Explicitly authorized deferral; keep the full bound without inventing a charge."""
        with self.connect() as db:
            changed = db.execute("UPDATE calls SET state='held_unknown' WHERE id=? "
                                 "AND state='unknown' AND cost IS NULL AND reserved>0", (id,))
            if changed.rowcount != 1:
                raise JudgeStop('Only an existing unknown charge can become a retained hold')

    def summary(self):
        with self.connect() as db:
            rows = db.execute('SELECT phase,COALESCE(SUM(cost),0),'
                              'SUM(CASE WHEN cost IS NULL THEN reserved ELSE 0 END),COUNT(*) '
                              'FROM calls GROUP BY phase').fetchall()
        return {p: {'spent': spent, 'reserved': held, 'calls': n} for p, spent, held, n in rows}


def function(name, schema):
    return {'type': 'function', 'function': {'name': name, 'parameters': schema,
            'description': 'Submit the current decision.' if name == 'submit' else
            'Inspect original source frames to resolve the stated visual observation.'}}


def media_summary(value):
    if isinstance(value, dict):
        return {k: media_summary(v) for k, v in value.items()}
    if isinstance(value, list):
        return [media_summary(v) for v in value]
    if isinstance(value, str) and value.startswith('data:image/'):
        return 'encoded-image-sha256:' + digest(value)
    return value


class Provider:
    def __init__(self, config, ledger, phase):
        self.config, self.ledger, self.phase = config, ledger, phase

    def call(self, messages, functions, directory):
        config, directory = self.config, Path(directory)
        body = {'model': config['model'], 'messages': messages, 'tools': functions,
                'tool_choice': 'required', 'max_tokens': config['max_tokens'],
                'reasoning': {'effort': config['reasoning_effort'], 'exclude': True},
                'provider': {'only': [config['endpoint']], 'allow_fallbacks': False,
                             'require_parameters': True,
                             'max_price': {'prompt': config['input_price']*1e6,
                                           'completion': config['output_price']*1e6}}}
        request_hash = digest(body)
        id = digest([str(directory.resolve()), request_hash])
        saved = directory / 'response.json'
        if saved.exists():
            record = json.loads(saved.read_text())
            if record['request_hash'] != request_hash:
                raise JudgeStop('Cached provider response has different inputs')
            self.ledger.reconcile(id, record['response']['usage'].get('cost'))
            if record['response'].get('model') not in config['accepted_model_ids']:
                raise JudgeStop('Cached response has an unpinned model identity')
            return record['response']
        if (directory / 'error.json').exists():
            raise JudgeStop('Previous provider rejection; review the recorded error before retrying')
        summary = media_summary(body)
        images = json.dumps(summary).count('encoded-image-sha256:')
        token_bound = len(json.dumps(summary).encode()) + images * config['image_token_bound']
        if token_bound + config['max_tokens'] > config['context_token_bound']:
            raise ContextLimit('Context resource boundary reached')
        bound = token_bound * config['input_price'] + config['max_tokens'] * config['output_price']
        bound = math.ceil((bound * 1.06 + 0.000001) * 1e6) / 1e6
        key = os.environ.get('OPENROUTER_API_KEY')
        if not key:
            raise JudgeStop('OPENROUTER_API_KEY is unavailable')
        self.ledger.reserve(id, self.phase, bound)
        write_json(directory / 'request.json', {'id': id, 'request_hash': request_hash, 'body': summary})
        started = time.monotonic()
        try:
            request = Request('https://openrouter.ai/api/v1/chat/completions',
                              json.dumps(body, allow_nan=False).encode(),
                              {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
            with urlopen(request, timeout=config['timeout_seconds']) as response:
                result = json.load(response)
            write_json(saved, {'request_hash': request_hash, 'response': result,
                               'elapsed_seconds': time.monotonic() - started})
            self.ledger.reconcile(id, result.get('usage', {}).get('cost'))
        except HTTPError as error:
            write_json(directory / 'error.json', {'http_status': error.code, 'message': error.read().decode()[:2000]})
            if error.code in (400, 401, 402, 403, 404, 422, 429):
                self.ledger.reconcile(id, 0)
            else:
                self.ledger.unknown(id)
            raise JudgeStop(f'Provider HTTP {error.code}; see redacted error artifact') from None
        except Exception:
            self.ledger.unknown(id)
            raise
        if result.get('model') not in config['accepted_model_ids']:
            raise JudgeStop('Provider returned an unpinned model identity')
        return result


class Session:
    def __init__(self, video, plan, goal, axis, tools_enabled, provider, directory, cache,
                 prompt_set=prompts):
        self.prompts = prompt_set
        self.video = VideoEvidence(video, cache)
        self.plan, self.goal, self.axis = plan, goal, axis
        self.tools_enabled, self.provider, self.directory = tools_enabled, provider, Path(directory)
        self.initial = self.video.inspect(0, self.video.duration, count=8 if axis == 'task' else 24)
        self.bank, self.repairs, self.trace = {}, 0, []
        self.visible = {}

    def validate(self, result, ids):
        return task.validate_rows(result, ids, self.visible, self.video.duration)

    def gate(self, gate, ids, choices):
        if self.axis == 'task':
            prompt = (self.prompts.TASK_QUESTIONS[gate] + self.prompts.TASK_COMMON
                      + self.prompts.TASK_EXAMPLES[gate])
            choices = ['yes', 'no']
        else:
            prompt = self.prompts.PHYSICS_GATE.format(gate_question=getattr(self.prompts, gate.upper()))
        result = self.ask(gate, prompt, ids, task.submission_schema(ids, choices))
        return {row['id']: row for row in result['rows']}

    def ask(self, stage, prompt, ids, schema):
        fields = ['id', 'action', 'source', 'target', 'end_state']
        if self.axis == 'task':
            fields += ['requires']
        payload = {'goal': self.goal, 'subgoals': [{k: s[k] for k in fields} for s in self.plan],
                   'eligible_ids': ids, 'source_duration_seconds': self.video.duration}
        frames = list({f['id']: f for f in self.initial + list(self.bank.values())}.values())
        self.visible = {f['id']: f for f in frames}
        messages = [{'role': 'system', 'content': prompt + '\n' + self.prompts.RESPONSE},
                    {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(payload)}] + image_parts(frames)}]
        functions = [function('submit', schema)]
        if self.tools_enabled:
            functions.append(function('inspect_video', INSPECT_SCHEMA))
        attempts, unproductive = 0, set()
        while True:
            schema['properties']['rows']['items']['properties']['evidence']['items']['enum'] = list(self.visible)
            response = self.provider.call(messages, functions, self.directory / stage / f'{attempts:04d}')
            attempts += 1
            message = response['choices'][0]['message']
            calls = message.get('tool_calls', [])
            messages.append({k: v for k, v in message.items() if k in ('role', 'content', 'tool_calls', 'reasoning_details')})
            try:
                if len(calls) != 1:
                    raise ValueError('Return exactly one native function call')
                call = calls[0]
                name = call['function']['name']
                arguments = json.loads(call['function']['arguments'])
                if name == 'submit':
                    validate(arguments, schema)
                    self.validate(arguments, ids)
                    for row in arguments['rows']:
                        if row['answer'].isdigit():
                            row['answer'] = int(row['answer'])
                    self.trace.append({'stage': stage, 'result': arguments, 'provider_responses': attempts})
                    return arguments
                if name != 'inspect_video' or not self.tools_enabled:
                    raise ValueError('Function is not available')
                validate(arguments, INSPECT_SCHEMA)
                evidence = self.video.inspect(arguments['start_seconds'], arguments['end_seconds'], arguments['bbox'])
                new = [f for f in evidence if f['id'] not in self.visible]
                signature = digest([f['id'] for f in evidence])
                if not new and signature in unproductive:
                    raise UnresolvedDecision('Repeated inspection supplies no new visual evidence')
                if not new:
                    unproductive.add(signature)
                self.visible.update({f['id']: f for f in new})
                self.bank.update({f['id']: f for f in new})
                # Keep full-scene initial evidence plus the most recent 24 tool frames between stages.
                self.bank = dict(list(self.bank.items())[-24:])
                self.trace.append({'stage': stage, 'inspection': arguments, 'frames': evidence, 'new_frames': len(new)})
                messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps({
                    'evidence_ids': [f['id'] for f in evidence], 'new_frames': len(new)})})
                if new:
                    messages.append({'role': 'user', 'content': image_parts(new)})
            except (ValueError, KeyError, TypeError, ValidationError) as error:
                if self.repairs:
                    raise JudgeStop('Axis format-repair allowance exhausted') from error
                self.repairs += 1
                if calls:
                    for call in calls:
                        messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': str(error)[:400]})
                else:
                    messages.append({'role': 'user', 'content': str(error)[:400]})


def evaluate(video, goal, plan, *, axis, provider, output, cache, tools_enabled=True,
             repeat=1, gating='sequential'):
    """Score one axis of one video against a fixed plan; results are cached by input identity."""
    if axis not in ('task', 'physics'):
        raise ValueError('Unknown axis')
    if gating not in ('sequential', 'all'):
        raise ValueError('Unknown gating mode')
    if not plan or [s['id'] for s in plan] != [f'S{i+1}' for i in range(len(plan))]:
        raise ValueError('Invalid fixed subgoal inventory')
    source_hashes = {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}
    identity = {'video': digest(Path(video)), 'goal': goal, 'plan': plan, 'axis': axis,
                'tools_enabled': tools_enabled, 'config': provider.config,
                'code': source_hashes, 'repeat': repeat, 'gating': gating}
    directory = Path(output) / digest(identity)
    result_path = directory / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    write_json(directory / 'identity.json', identity)
    session = Session(video, plan, goal, axis, tools_enabled, provider, directory, cache)
    try:
        module = task if axis == 'task' else physics
        result = module.judge(plan, session, gating=gating)
    except UnresolvedDecision as error:
        result = {'score': None, 'status': 'unresolved', 'reason': str(error)}
    except ContextLimit as error:
        result = {'score': None, 'status': 'incomplete', 'reason': str(error)}
    except JudgeStop as error:
        write_json(directory / 'incomplete.json', {'status': 'incomplete', 'reason': str(error), 'trace': session.trace})
        raise
    result.update({'identity': digest(identity), 'axis': axis, 'model': provider.config['model'],
                   'repeat': repeat, 'trace': session.trace})
    gates = ['q1', 'q2', 'q3'] if axis == 'task' else ['p1', 'p2', 'p3', 'p4']
    result['not_evaluated'] = [gate for gate in gates if gate not in {row['stage'] for row in session.trace}]
    write_json(result_path, result)
    return result


def prepare_plan(goal, starting_image, provider, output, *, prompt_set=prompts):
    """One draft plus dependency review, shared across every candidate/condition."""
    from PIL import Image
    output = Path(output)
    identity = digest([goal, digest(Path(starting_image)), prompt_set.PLAN,
                       prompt_set.DEPENDENCIES, provider.config])
    saved = output / identity / 'plan.json'
    if saved.exists():
        return json.loads(saved.read_text())
    image = output / identity / 'start.jpg'
    image.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(starting_image) as picture:
        picture = picture.convert('RGB')
        picture.thumbnail((512, 512))
        picture.save(image, quality=85)
    text = {'type': 'string', 'minLength': 1, 'maxLength': 300}
    dependency = task.object_schema({'id': text, 'requires': {'type': 'array', 'items': text}, 'reason': text})
    action = task.object_schema({**dependency['properties'], 'action': text, 'source': text,
                                 'target': {'type': ['string', 'null']}, 'end_state': text})
    frames = image_parts([{'id': 'shared_start', 'time': 0, 'bbox': None, 'path': str(image), 'sha256': digest(image)}])
    drafts = []
    for stage, prompt, row_schema in [('draft', prompt_set.PLAN, action),
                                      ('review', prompt_set.DEPENDENCIES, dependency)]:
        schema = task.object_schema({'subgoals': {'type': 'array', 'minItems': 1, 'maxItems': 16, 'items': row_schema}})
        payload = {'goal': goal, 'fixed_actions': drafts[0] if drafts else []}
        messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': [
            {'type': 'text', 'text': json.dumps(payload)}] + frames}]
        response = provider.call(messages, [function('submit', schema)], output / identity / stage)
        calls = response['choices'][0]['message'].get('tool_calls', [])
        if len(calls) != 1 or calls[0]['function']['name'] != 'submit':
            raise JudgeStop('Invalid planning function submission')
        result = json.loads(calls[0]['function']['arguments'])
        validate(result, schema)
        rows = result['subgoals']
        for index, row in enumerate(rows):
            if row['id'] != f'S{index+1}' or not set(row['requires']) <= {f'S{i+1}' for i in range(index)}:
                raise JudgeStop('Invalid subgoal IDs or dependency direction')
        drafts.append(rows)
    if [s['id'] for s in drafts[0]] != [s['id'] for s in drafts[1]]:
        raise JudgeStop('Dependency review changed the fixed action inventory')
    plan = drafts[0]
    for action, review in zip(plan, drafts[1]):
        action['requires'] = [id for id in action['requires'] if id in review['requires']]
    write_json(saved, plan)
    return plan


def prepare_requirements(goal,image_path,provider,output, *, prompt_set=prompts):
    image=Path(image_path)
    identity=digest([goal,digest(image),prompt_set.TASK_REQUIREMENTS,provider.config])
    directory=Path(output)/identity
    saved=directory/'plan.json'
    if saved.exists():return json.loads(saved.read_text())
    string={'type':'string','minLength':1}
    row=task.object_schema({'id':string,'source':string,'target':{'type':['string','null']},'action':string,'end_state':string,'requires':{'type':'array','items':string},'reason':string})
    schema=task.object_schema({'subgoals':{'type':'array','minItems':1,'items':row}})
    frame={'id':'shared_start','time':0,'bbox':None,'path':str(image),'sha256':digest(image)}
    messages=[{'role':'system','content':prompt_set.TASK_REQUIREMENTS},{'role':'user','content':[{'type':'text','text':json.dumps({'goal':goal})}]+image_parts([frame])}]
    response=provider.call(messages,[function('submit',schema)],directory/'call')
    calls=response['choices'][0]['message'].get('tool_calls',[])
    if len(calls)!=1 or calls[0]['function']['name']!='submit':raise ValueError('Invalid plan submission')
    result=json.loads(calls[0]['function']['arguments']);validate(result,schema)
    rows=result['subgoals']
    for i,r in enumerate(rows):
        if r['id']!=f'S{i+1}' or not set(r['requires'])<={f'S{j+1}' for j in range(i)}:raise ValueError('Invalid plan dependency IDs')
    write_json(saved,rows)
    return rows


PAPER_PROVIDER = {'model': 'google/gemini-3.7-flash', 'endpoint': 'google-ai-studio/flex', 'input_price': 3.75e-07, 'output_price': 1.875e-06, 'max_tokens': 4096, 'reasoning_effort': 'low', 'image_token_bound': 4096, 'context_token_bound': 220000, 'timeout_seconds': 180, 'accepted_model_ids': ['google/gemini-3.7-flash', 'google/gemini-3.7-flash', 'google/gemini-3.7-flash']}


def judge(video, goal, starting_image, *, output='judge_output', budget_usd=2.0,
          provider=None, gating='sequential'):
    """Score one generated video. Pass the initial scene given to the generator, not a reference video.

    Returns native component scores, 0–100 scores, plans and evidence traces.
    Human ratings and PDFs are never inference inputs. Missing axes stay missing.
    ``gating='all'`` (ablation only) asks every gate for every subgoal/window; the
    usual scores then apply the sequential rule to those answers and the
    ``*_independent`` scores count gates passed regardless of order.
    """
    if not goal.strip():
        raise ValueError('A nonempty goal is required')
    for source in (video, starting_image):
        if not Path(source).is_file():
            raise FileNotFoundError(source)
    prompt_set = prompts
    output = Path(output)
    if provider is None:
        provider = Provider(dict(PAPER_PROVIDER), Ledger(output/'costs.sqlite', {'judge': budget_usd}), 'judge')
    physics_plan = prepare_plan(goal, starting_image, provider, output/'plans/physics', prompt_set=prompt_set)
    image_id = digest([goal, digest(Path(starting_image)), prompt_set.PLAN,
                       prompt_set.DEPENDENCIES, provider.config])
    image = output/'plans/physics'/image_id/'start.jpg'
    task_plan = prepare_requirements(goal, image, provider, output/'plans/task', prompt_set=prompt_set)
    results = {}
    for axis, plan in [('task', task_plan), ('physics', physics_plan)]:
        results[axis] = evaluate(video, goal, plan, axis=axis, provider=provider,
                                 output=output/'predictions', cache=output/'media', gating=gating)
    t, p = results['task']['score'], results['physics']['score']
    t100 = None if t is None else 100*t/3
    p100 = None if p is None else 100*p/4
    extra = {}
    if gating == 'all':
        ti = results['task'].get('independent', {}).get('score')
        pi = results['physics'].get('independent', {}).get('score')
        extra = {'gating': gating,
                 'task100_independent': None if ti is None else 100*ti/3,
                 'physics100_independent': None if pi is None else 100*pi/4}
    return {**extra, 'model': provider.config['model'],
            'task': results['task'], 'physics': results['physics'],
            'plans': {'task': task_plan, 'physics': physics_plan},
            'task100': t100, 'physics100': p100,
            'final100': None if t100 is None or p100 is None else math.sqrt(t100*p100)}
