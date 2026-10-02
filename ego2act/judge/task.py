"""Task gates T1-T3: submission schema, validation and subgoal levels."""
from decimal import Decimal
from math import isfinite
from statistics import fmean



def object_schema(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def submission_schema(ids, answers):
    row = object_schema({
        'id': {'type': 'string', **({'enum': list(ids)} if ids else {})},
        'answer': {'type': 'string', 'enum': [str(answer) for answer in answers]},
        'window': {'type': 'array', 'items': {'type': 'number'}, 'maxItems': 2},
        'evidence': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 6,
                     'description': 'Exact supplied evidence IDs only. Put observations in note.'},
        'note': {'type': 'string'}})
    return object_schema({'rows': {'type': 'array', 'items': row, 'minItems': len(ids),
                                  'maxItems': len(ids)}})


def validate_rows(result, ids, evidence, duration):
    rows = result['rows']
    if len(rows) != len(ids) or {row['id'] for row in rows} != set(ids):
        raise ValueError('Submission must cover each eligible ID exactly once')
    for row in rows:
        submitted = row['window']
        window = []
        for value in submitted:
            if isfinite(value) and value > duration:
                precision = max(0, -Decimal(str(value)).as_tuple().exponent)
                if value == round(duration, precision):
                    value = duration
            window.append(value)
        if window and not (len(window) == 2 and 0 <= window[0] <= window[1] <= duration):
            raise ValueError('Invalid original-source window')
        if not set(row['evidence']) <= set(evidence):
            raise ValueError('Cited evidence was not supplied to this stage')
        if row['answer'] not in ('NA', 'unresolved') and not row['evidence']:
            raise ValueError('A decided judgment needs supplied visual evidence')
        if window != submitted:
            row['submitted_window'] = submitted
            row['window'] = window
    return {row['id']: row for row in rows}


def gate_levels(answers, gates, id):
    """Sequential and independent levels for one ID from a full set of gate answers.

    ``answers[gate][id]`` is 'yes', 'no', 'unresolved', 'NA' or missing (not asked).
    Sequential: gates passed before the first non-yes (the judge's rule); an
    'unresolved' reached before any failure makes the ID unresolved, matching the
    staged early exit. Independent: total gates answered yes, ignoring order; any
    'unresolved' makes the ID unresolved.
    """
    row = [answers[gate].get(id) for gate in gates]
    sequential = 0
    for answer in row:
        if answer == 'unresolved':
            sequential = 'unresolved'
        if answer != 'yes':
            break
        sequential += 1
    independent = 'unresolved' if 'unresolved' in row else sum(a == 'yes' for a in row)
    return sequential, independent


def _mean_task(levels):
    if 'unresolved' in levels.values():
        return {'score': None, 'status': 'unresolved', 'levels': levels}
    return {'score': fmean(levels.values()), 'status': 'ok', 'levels': levels}


def judge_all_gates(plan, session):
    """Gate ablation: ask T1-T3 for every subgoal regardless of earlier failures.

    The returned top-level score applies the sequential rule to the
    recorded answers; ``independent`` counts gates passed ignoring order.
    """
    ids = [s['id'] for s in plan]
    gates = ['q1', 'q2', 'q3']
    answers = {}
    for gate in gates:
        rows = session.gate(gate, ids, ['yes', 'no', 'unresolved'])
        answers[gate] = {id: rows[id]['answer'] for id in ids}
    pairs = {id: gate_levels(answers, gates, id) for id in ids}
    result = _mean_task({id: pair[0] for id, pair in pairs.items()})
    result.update({'gating': 'all', 'gate_answers': answers,
                   'independent': _mean_task({id: pair[1] for id, pair in pairs.items()})})
    return result


def judge(plan, session, gating='sequential'):
    """Task level per subgoal: gates passed before the first failure (early exit).

    ``gating='all'`` is the gate ablation of Appendix G (see judge_all_gates).
    """
    ids = [s['id'] for s in plan]
    if gating == 'all':
        return judge_all_gates(plan, session)
    if gating != 'sequential':
        raise ValueError('Unknown gating mode')
    levels, eligible = dict.fromkeys(ids, 0), ids
    for level, gate in enumerate(['q1', 'q2', 'q3'], 1):
        if not eligible:
            break
        answers = session.gate(gate, eligible, ['yes', 'no', 'unresolved'])
        if any(row['answer'] == 'unresolved' for row in answers.values()):
            return {'score': None, 'status': 'unresolved', 'stopped_at': gate}
        eligible = [id for id in eligible if answers[id]['answer'] == 'yes']
        levels.update(dict.fromkeys(eligible, level))
    if 'unresolved' in levels.values():
        return {'score': None, 'status': 'unresolved', 'levels': levels}
    return {'score': fmean(levels.values()), 'status': 'ok', 'levels': levels}
