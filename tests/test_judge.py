"""Ego2ActJudge routing, scoring and an offline end-to-end run; no API calls."""
import json
import math
from pathlib import Path
from ego2act.judge import prompts, task, physics
from ego2act.judge.main import Session, PAPER_PROVIDER

def test_gate_prompts_route_by_axis():
    session=object.__new__(Session);session.prompts=prompts;captured=[]
    session.ask=lambda stage,prompt,ids,schema: (captured.append((stage,prompt,schema)) or {'rows':[{'id':'S1','answer':'yes'}]})
    session.axis='task';session.gate('q2',['S1'],['yes','no','unresolved'])
    assert captured[-1][1] == prompts.TASK_QUESTIONS['q2']+prompts.TASK_COMMON+prompts.TASK_EXAMPLES['q2']
    assert captured[-1][2]['properties']['rows']['items']['properties']['answer']['enum']==['yes','no']
    session.axis='physics';session.gate('p1',['S1'],['yes','no','unresolved','NA'])
    assert captured[-1][1]==prompts.PHYSICS_GATE.format(gate_question=prompts.P1)

def test_missing_physics_stays_missing():
    assert physics.aggregate({'S1':'unresolved'})['score'] is None
    assert physics.aggregate({'S1':'NA'})['score'] is None

def test_task_score_is_mean_of_levels_without_ordering_penalty():
    plan=[{'id':'S1','requires':[]},{'id':'S2','requires':['S1']}]
    class Session:
        def gate(self, gate, ids, choices):
            return {id:{'id':id,'answer':'yes'} for id in ids}
    result=task.judge(plan, Session())
    assert result['score']==3

def test_public_judge_end_to_end_without_network(tmp_path, monkeypatch):
    from PIL import Image
    from ego2act.judge import main
    from ego2act.judge.tools import digest
    image=tmp_path/'initial.jpg';Image.new('RGB',(32,32),'white').save(image)
    video=tmp_path/'video.mp4';video.write_bytes(b'offline fixture')
    frame={'id':'f0','time':0.0,'bbox':None,'path':str(image),'sha256':digest(image)}
    class Evidence:
        def __init__(self,*args,**kwargs):self.duration=1.0
        def inspect(self,*args,**kwargs):return [frame]
    class Provider:
        config=dict(PAPER_PROVIDER)
        def __init__(self):self.calls=[]
        def call(self,messages,functions,directory):
            schema=functions[0]['function']['parameters'];self.calls.append(str(directory))
            if 'subgoals' in schema['properties']:
                properties=schema['properties']['subgoals']['items']['properties']
                row={'id':'S1','source':'spoon','target':'drawer','action':'place','end_state':'inside','requires':[],'reason':'requested outcome'}
                answer={'subgoals':[{k:row[k] for k in properties}]}
            else:
                answer={'rows':[{'id':'S1','answer':'yes','window':[0.0,1.0],'evidence':['f0'],'note':'Visible requested state.'}]}
            return {'choices':[{'message':{'role':'assistant','tool_calls':[{'id':'call','type':'function','function':{'name':'submit','arguments':json.dumps(answer)}}]}}]}
    monkeypatch.setattr(main,'VideoEvidence',Evidence)
    provider=Provider()
    result=main.judge(video,'Place spoon inside drawer.',image,output=tmp_path/'out',provider=provider)
    assert result['task']['score']==3 and result['physics']['score']==4
    assert result['final100']==100
    assert len(provider.calls)==10  # two Physics planning calls, one Task plan, seven gates
    again=main.judge(video,'Place spoon inside drawer.',image,output=tmp_path/'out',provider=provider)
    assert again['final100']==100 and len(provider.calls)==10  # cached; no new calls


def test_all_gates_ablation_records_sequential_and_independent_levels():
    answers = {'q1': {'S1': 'yes', 'S2': 'no', 'S3': 'yes'},
               'q2': {'S1': 'no', 'S2': 'yes', 'S3': 'yes'},
               'q3': {'S1': 'yes', 'S2': 'yes', 'S3': 'yes'},
               'p1': {'S1': 'yes', 'S2': 'NA'}, 'p2': {'S1': 'no'},
               'p3': {'S1': 'yes'}, 'p4': {'S1': 'yes'}}
    asked = []
    class Session:
        def gate(self, gate, ids, choices):
            asked.append((gate, list(ids)))
            return {id: {'id': id, 'answer': answers[gate][id]} for id in ids}
    plan = [{'id': f'S{i}', 'requires': []} for i in (1, 2, 3)]
    result = task.judge(plan, Session(), gating='all')
    assert asked == [(g, ['S1', 'S2', 'S3']) for g in ('q1', 'q2', 'q3')]
    assert result['levels'] == {'S1': 1, 'S2': 0, 'S3': 3}
    assert result['independent']['levels'] == {'S1': 2, 'S2': 2, 'S3': 3}
    result = physics.judge(plan[:2], Session(), gating='all')
    assert asked[3:] == [('p1', ['S1', 'S2'])] + [(g, ['S1']) for g in ('p2', 'p3', 'p4')]
    assert result['levels'] == {'S1': 1, 'S2': 'NA'} and result['score'] == 1
    assert result['independent']['levels'] == {'S1': 3, 'S2': 'NA'}
    assert task.gate_levels({'p1': {'S1': 'yes'}, 'p2': {'S1': 'unresolved'}}, ['p1', 'p2'], 'S1') == ('unresolved', 'unresolved')
    assert task.gate_levels({'p1': {'S1': 'no'}, 'p2': {'S1': 'unresolved'}}, ['p1', 'p2'], 'S1') == (0, 'unresolved')
