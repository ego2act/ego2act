"""Physics gates P1-P4: the first failed gate sets the window level."""
from statistics import fmean

from .task import gate_levels


def aggregate(levels):
    if 'unresolved' in levels.values():
        return {'score': None, 'status': 'unresolved', 'levels': levels}
    observable = [value for value in levels.values() if value != 'NA']
    return {'score': fmean(observable) if observable else None,
            'status': 'ok' if observable else 'rubric_NA', 'levels': levels}


def judge_all_gates(plan, session):
    """Gate ablation: ask P1-P4 for every window regardless of earlier failures.

    Windows answered NA at P1 (no inspectable movement) stay NA/unattempted in
    both rules and are not asked P2-P4. The top-level score applies the
    sequential rule; ``independent`` counts gates passed.
    """
    ids = [s['id'] for s in plan]
    gates = ['p1', 'p2', 'p3', 'p4']
    answers = {gate: {} for gate in gates}
    rows = session.gate('p1', ids, ['yes', 'no', 'unresolved', 'NA'])
    answers['p1'] = {id: rows[id]['answer'] for id in ids}
    live = [id for id in ids if answers['p1'][id] != 'NA']
    for gate in gates[1:]:
        if not live:
            break
        rows = session.gate(gate, live, ['yes', 'no', 'unresolved'])
        answers[gate] = {id: rows[id]['answer'] for id in live}
    sequential, independent = {}, {}
    for id in ids:
        if answers['p1'][id] == 'NA':
            sequential[id] = independent[id] = 'NA'
        else:
            sequential[id], independent[id] = gate_levels(answers, gates, id)
    result = aggregate(sequential)
    result.update({'gating': 'all', 'gate_answers': answers,
                   'independent': aggregate(independent)})
    return result


def judge(plan, session, gating='sequential'):
    """Physics level per window: gates passed before the first failure; NA at P1 is excluded."""
    ids = [s['id'] for s in plan]
    if gating == 'all':
        return judge_all_gates(plan, session)
    if gating != 'sequential':
        raise ValueError('Unknown gating mode')
    levels, eligible = {}, ids
    for failure_level, gate in enumerate(['p1', 'p2', 'p3', 'p4']):
        if not eligible:
            break
        choices = ['yes', 'no', 'unresolved'] + (['NA'] if gate == 'p1' else [])
        answers = session.gate(gate, eligible, choices)
        if any(row['answer'] == 'unresolved' for row in answers.values()):
            return {'score': None, 'status': 'unresolved', 'stopped_at': gate}
        for id in eligible:
            answer = answers[id]['answer']
            if answer == 'NA':
                levels[id] = 'NA'
            elif answer == 'no':
                levels[id] = failure_level
        eligible = [id for id in eligible if answers[id]['answer'] == 'yes']
    levels.update(dict.fromkeys(eligible, 4))
    return aggregate(levels)
