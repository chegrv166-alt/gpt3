import json

import pytest

from cognitive_helpers import episode, school
from social_sim.models import WorldState
from social_sim.storage import load_snapshot, save_snapshot


@pytest.mark.parametrize('during_discussion', [False, True])
def test_snapshot_resume_matches_continuous_run_with_active_situation(tmp_path, during_discussion):
    s = school(explanation=True)
    episode(s)
    if during_discussion:
        s.act('n3','ask','n1',defer_reply=True)
    s.run_days(9)
    path = tmp_path/'snapshot.json'
    save_snapshot(s, path)
    b, c = load_snapshot(path), load_snapshot(path)
    assert b.world == c.world == s.world
    s.run_days(21)
    b.run_days(21)
    assert b.world == s.world
    assert b.rng.getstate() == s.rng.getstate()


def test_snapshot_during_discussion_can_resume_conversation(tmp_path):
    s = school(explanation=True)
    e = episode(s)
    s.act('n3','ask','n1',defer_reply=True)
    save_snapshot(s, tmp_path/'state.json')
    loaded = load_snapshot(tmp_path/'state.json')
    for x in (s,loaded):
        x.step()
    assert loaded.world == s.world
    assert loaded.rng.getstate() == s.rng.getstate()


@pytest.mark.parametrize('damage', ['situation', 'revision', 'memory', 'reaction', 'trace_memory', 'trace_revision', 'evidence', 'future_revision'])
def test_corrupt_situation_memory_and_revision_references_rejected(tmp_path, damage):
    s = school()
    episode(s)
    s.act('n3','distance','n1')
    path = tmp_path/'snapshot.json'
    save_snapshot(s,path)
    data = json.loads(path.read_text())
    state = data['world']['states']['n3']
    situation = next(iter(state['situations'].values()))
    if damage == 'situation':
        situation['target'] = 'ghost'
    elif damage == 'revision':
        situation['current_revision'] = 'missing'
    elif damage == 'memory':
        state['memories'][0]['source'] = 'ghost'
    elif damage == 'reaction':
        situation['reactions'][0]['revision_id'] = 'missing'
    elif damage == 'evidence':
        situation['evidence'] = ['e999999']
    elif damage == 'future_revision':
        state['interpretations'][0]['tick'] = 100
    elif damage == 'trace_memory':
        data['world']['traces'][-1]['candidates'][0]['memory_ids'] = ['missing']
    else:
        data['world']['traces'][-1]['interpretation_ids'] = ['missing']
    path.write_text(json.dumps(data))
    before = s.world.model_dump_json()
    with pytest.raises(ValueError):
        load_snapshot(path)
    assert s.world.model_dump_json() == before


def test_unsupported_snapshot_version_has_clear_error(tmp_path):
    path = tmp_path/'old.json'
    path.write_text(json.dumps({'world': {'schema_version': 1}, 'rng_state': []}))
    with pytest.raises(ValueError, match='версия.*1.*4'):
        load_snapshot(path)


def test_real_deferred_question_snapshot_resumes_identically(tmp_path):
    s = school(explanation=True)
    episode(s)
    s.act('n3','ask','n1',defer_reply=True)
    assert s.world.states['n3'].situations['s:n3:n1:exclusion'].status == 'discussing'
    assert s.world.pending_questions
    save_snapshot(s,tmp_path/'state.json')
    restored = load_snapshot(tmp_path/'state.json')
    for x in (s,restored):
        x.step()
    assert restored.world == s.world
    assert restored.rng.getstate() == s.rng.getstate()
    assert not s.world.pending_questions


def test_trace_cannot_acquire_later_evidence_in_same_tick(tmp_path):
    from social_sim.decision import decide
    s = school(explanation=True)
    episode(s)
    trace = decide(s.view('n3'), s.rng, s.world.scenario.weights, f't{len(s.world.traces)+1:06d}')
    question = s.act('n3','ask','n1',trace=trace)
    raw = s.world.model_dump()
    t = next(t for t in raw['traces'] if t['id'] == question.trace_id)
    latest = s.world.states['n3'].interpretations[-1]
    t['known_events'] = [*t['known_events'], s.world.events[-1].id]
    t['interpretation_ids'] = [latest.id]
    t['situations'][0]['revision_id'] = latest.id
    with pytest.raises(ValueError):
        WorldState.model_validate(raw)


def test_discussing_situation_requires_pending_question():
    s = school()
    episode(s)
    raw = s.world.model_dump()
    next(iter(raw['states']['n3']['situations'].values()))['status'] = 'discussing'
    with pytest.raises(ValueError):
        WorldState.model_validate(raw)


def test_combined_group_introduction_and_pending_question_roundtrip(tmp_path):
    from pathlib import Path
    from social_sim.authoring import load_scenario
    from social_sim.engine import Simulation
    s=Simulation(load_scenario(Path('data/fixtures/school.yaml')))
    episode(s)
    s.act('n3','ask','n1',defer_reply=True)
    plan=s.invite_group('n4',['n1','n9','player'])
    s.respond(plan.id,True,invitee='n9')
    s.introduce('n5','n6','n9',answers={'n6':True,'n9':True})
    assert 'n6:n9' in s.world.relationships and s.world.pending_questions
    assert plan.replies['n1'].status=='pending'
    save_snapshot(s,tmp_path/'combined.json');b=load_snapshot(tmp_path/'combined.json')
    for x in(s,b):x.run_days(5)
    assert s.world==b.world and s.rng.getstate()==b.rng.getstate()
