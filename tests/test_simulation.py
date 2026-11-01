from pathlib import Path

import pytest
from pydantic import ValidationError

from social_sim.authoring import load_scenario
from social_sim.engine import Simulation
from social_sim.models import Scenario
from social_sim.storage import load_snapshot, save_snapshot

FIXTURE = Path('data/fixtures/school.yaml')


def sim(seed=42):
    return Simulation(load_scenario(FIXTURE), seed)


def test_ten_explicit_profiles_and_strict_traits():
    scenario = load_scenario(FIXTURE)
    assert len(scenario.people) == 10
    assert sum(p.is_player for p in scenario.people) == 1
    for p in scenario.people:
        assert len(p.personality.traits) == 24
    for bad in [True, 6, -1, 2.5, '3']:
        data = scenario.model_dump()
        data['people'][0]['personality']['traits']['initiative'] = bad
        with pytest.raises(ValidationError):
            Scenario.model_validate(data)


def test_invalid_reference_and_missing_trait():
    data = load_scenario(FIXTURE).model_dump()
    data['relationships'][0]['target'] = 'ghost'
    with pytest.raises(ValueError):
        Scenario.model_validate(data)
    data = load_scenario(FIXTURE).model_dump()
    del data['people'][0]['personality']['traits']['tact']
    with pytest.raises(ValueError):
        Scenario.model_validate(data)


def test_determinism_and_autonomy():
    a, b = sim(), sim()
    a.run_days(30)
    b.run_days(30)
    assert a.world == b.world
    meetings = [e for e in a.world.events if e.kind == 'meeting_occurred']
    assert len(meetings) > 5
    assert any(e.kind == 'invitation_declined' for e in a.world.events)
    assert not any(t.actor == 'player' for t in a.world.traces)
    assert any(r.closeness > 40 for r in a.world.relationships.values())
    assert all(t.candidates and t.selected in t.candidates for t in a.world.traces)


def test_save_continue_exactly(tmp_path):
    a = sim()
    a.run_days(9)
    path = tmp_path / 'snapshot.json'
    save_snapshot(a, path)
    b = load_snapshot(path)
    a.run_days(21)
    b.run_days(21)
    assert a.world == b.world
    assert a.rng.getstate() == b.rng.getstate()


def test_no_double_booking_and_no_invite_spam():
    s = sim()
    s.run_days(30)
    occupied = set()
    last = {}
    for e in s.world.events:
        if e.kind == 'meeting_occurred':
            for person in e.participants:
                assert (e.tick, person) not in occupied
                occupied.add((e.tick, person))
        if e.kind == 'invitation_sent':
            pair = e.participants
            assert e.tick - last.get(pair, -100) >= 8
            last[pair] = e.tick


def test_failed_action_is_atomic():
    s = sim()
    before = s.world.model_copy(deep=True)
    with pytest.raises(ValueError):
        s.act('player', 'invite', 'ghost')
    assert s.world == before


def test_private_meeting_and_subjective_reaction():
    s = sim()
    s.world.tick = 2
    s.act('n1', 'invite', 'n2')
    plan = s.world.plans[0]
    s.respond(plan.id, accept=True)
    s.step()
    s.step()
    meeting = next(e for e in s.world.events if e.kind == 'meeting_occurred' and set(e.participants) == {'n1','n2'})
    assert meeting.id not in s.world.states['n3'].knowledge
    reverse = s.world.relationships['n1:n3'].model_copy(deep=True)
    s.world.tick = 6
    s.share_information('n1', 'n3', meeting.id)
    state = s.world.states['n3']
    assert state.knowledge[meeting.id].method == 'message'
    assert state.interpretations[-1].kind == 'excluded'
    assert state.hurt['n1'] > 0
    assert s.world.relationships['n1:n3'] == reverse
    before = state.model_copy(deep=True)
    s.share_information('n1', 'n3', meeting.id)
    assert state == before
    with pytest.raises(ValueError):
        s.share_information('n8', 'n7', meeting.id)


def test_inertia_without_events():
    s = sim()
    before = s.world.relationships['n3:n1'].model_copy(deep=True)
    for _ in range(30):
        s.recover()
    assert s.world.relationships['n3:n1'] == before


def test_same_fact_different_personality_and_deep_friendship_inertia():
    from social_sim.cognition import perceive
    from social_sim.models import WorldEvent
    for sensitive, expected in [(True, 'excluded'), (False, 'neutral')]:
        data = load_scenario(FIXTURE).model_dump()
        traits = data['people'][3]['personality']['traits']
        if not sensitive:
            for key in ['rejection_sensitivity', 'inclusion_expectation', 'suspiciousness']:
                traits[key] = 0
        s = Simulation(Scenario.model_validate(data))
        rel = s.world.relationships['n3:n1']
        rel.closeness, rel.trust = 90, 90
        fact = WorldEvent(id='e000001', tick=0, kind='meeting_occurred', participants=('n1','n2'), location='park', outcome='Встреча')
        s.world.events.append(fact)
        perceive(s.world, 'n3', fact, 'n1', fact.id, message=True)
        assert s.world.states['n3'].interpretations[-1].kind == expected
        assert rel.closeness == 90
        assert rel.trust > 80


def test_decision_reacts_to_hurt_and_strategy_not_id():
    from social_sim.decision import decide
    s = sim()
    s.world.tick = 6
    state = s.world.states['n3']
    state.hurt['n1'] = 80
    t = decide(s.view('n3'), s.rng, s.world.scenario.weights, 't000001')
    assert t.selected.action == 'distance'
    data = s.world.scenario.model_dump()
    data['people'][3]['personality']['traits']['directness'] = 5
    data['people'][3]['personality']['traits']['reconciliation_readiness'] = 5
    data['people'][3]['personality']['strategies'] = ['ask']
    s2 = Simulation(Scenario.model_validate(data))
    s2.world.tick = 6
    s2.world.states['n3'].hurt['n1'] = 80
    t2 = decide(s2.view('n3'), s2.rng, s2.world.scenario.weights, 't000001')
    assert t2.selected.action == 'ask'


def test_snapshot_rejects_broken_reference_and_schema(tmp_path):
    import json
    s = sim()
    s.run_days(2)
    path = tmp_path / 'state.json'
    save_snapshot(s, path)
    data = json.loads(path.read_text())
    data['world']['schema_version'] = 99
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_snapshot(path)
    data['world']['schema_version'] = 1
    data['world']['events'][0]['participants'] = ['ghost']
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_snapshot(path)


def test_player_action_changes_trajectory_and_npc_can_refuse():
    a, b = sim(), sim()
    a.world.tick = b.world.tick = 2
    a.act('player', 'invite', 'n1')
    a.run_days(5)
    b.run_days(5)
    assert a.world != b.world
    assert any(t.actor == 'n1' and t.selected.action in ('accept', 'decline') for t in a.world.traces)
    c = sim()
    c.world.tick = 2
    c.act('player', 'invite', 'n1')
    c.respond(c.world.plans[0].id, accept=False)
    assert c.world.plans[0].status == 'declined'
    assert not any(e.kind == 'meeting_occurred' for e in c.world.events)


def test_occupied_invitation_cannot_be_accepted():
    s = sim()
    s.world.tick = 2
    s.act('n1', 'invite', 'n2')
    s.respond('p000001', accept=True)
    s.act('n3', 'invite', 'n2')
    before = s.world.model_copy(deep=True)
    with pytest.raises(ValueError):
        s.respond('p000002', accept=True)
    assert s.world == before
    e = s.respond('p000002')
    assert e.kind == 'invitation_declined'


def test_cli_run_explain_resume_and_errors(tmp_path):
    import json
    import subprocess
    import sys
    def cli(*args):
        return subprocess.run([sys.executable, '-m', 'social_sim', *map(str,args)], text=True, capture_output=True)
    assert cli('validate', 'data/fixtures').returncode == 0
    out = tmp_path / 'run'
    run = cli('run', '--scenario', FIXTURE, '--days', 3, '--seed', 42, '--no-player', '--output', out)
    assert run.returncode == 0, run.stderr
    events = [json.loads(line) for line in (out/'events.jsonl').read_text().splitlines()]
    invite = next(e for e in events if e['kind'] == 'invitation_sent')
    explain = cli('explain', '--run', out, '--event-id', invite['id'])
    assert explain.returncode == 0
    assert 'components' in explain.stdout and 'selected' in explain.stdout
    assert cli('explain', '--run', out, '--event-id', 'missing').returncode == 2
    assert cli('resume', '--snapshot', out/'snapshot.json', '--days', 2, '--output', tmp_path/'resume').returncode == 0
    assert cli('run', '--scenario', FIXTURE, '--days', -1).returncode == 2
    assert cli('validate', tmp_path/'empty').returncode == 2


def test_one_initiative_per_slot_including_rest():
    s = sim()
    s.world.tick = 2
    s.act('n1', 'rest')
    before = s.world.model_copy(deep=True)
    with pytest.raises(ValueError):
        s.act('n1', 'invite', 'n2')
    assert s.world == before


def test_question_without_informed_answer_does_not_relieve_hurt():
    s = sim()
    s.world.tick = 2
    s.world.states['n3'].hurt['n1'] = 60
    # There is no observed meeting and the interlocutor cannot explain any fact.
    s.act('n3', 'ask', 'n1')
    assert s.world.states['n3'].hurt['n1'] == 60
    assert any(e.kind == 'question_declined' for e in s.world.events)


@pytest.mark.parametrize('damage', ['empty_history', 'past_plan', 'unknown_target', 'unknown_interpretation', 'rng_negative', 'rng_huge'])
def test_corrupt_snapshot_rejected_before_resume(tmp_path, damage):
    import json
    s = sim()
    s.world.tick = 2
    s.act('n1', 'invite', 'n2')
    path = tmp_path/'bad.json'
    save_snapshot(s, path)
    data = json.loads(path.read_text())
    if damage == 'empty_history':
        data['world']['plans'][0]['history'] = []
    elif damage == 'past_plan':
        data['world']['tick'] = 6
    elif damage == 'unknown_target':
        data['world']['traces'][0]['selected']['target'] = 'ghost'
        data['world']['traces'][0]['candidates'][0]['target'] = 'ghost'
    elif damage == 'unknown_interpretation':
        data['world']['traces'][0]['interpretation_events'] = ['e999999']
    else:
        data['rng_state'][1][0] = -1 if damage == 'rng_negative' else 2**100
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_snapshot(path)


@pytest.mark.parametrize('field', ['trust', 'strength', 'priority', 'need'])
def test_authoring_rejects_boolean_continuous_scales(field):
    data = load_scenario(FIXTURE).model_dump()
    if field == 'trust':
        data['relationships'][0][field] = True
    elif field == 'strength':
        data['people'][3]['personality']['beliefs'][0][field] = True
    elif field == 'priority':
        data['people'][3]['personality']['goals'][0][field] = True
    else:
        data['weights'][field] = True
    with pytest.raises(ValueError):
        Scenario.model_validate(data)


def test_external_trace_cannot_forge_actor_or_action():
    from social_sim.decision import decide
    s = sim()
    s.world.tick = 2
    trace = decide(s.view('n1'), s.rng, s.world.scenario.weights, 't000001')
    before = s.world.model_copy(deep=True)
    with pytest.raises(ValueError):
        s.act('n2', 'rest', trace=trace)
    assert s.world == before


def test_autonomy_goal_makes_solitude_more_likely():
    from random import Random
    from social_sim.decision import decide
    choices = []
    for priority in (0.0, 1.0):
        data = load_scenario(FIXTURE).model_dump()
        data['people'][1]['personality']['goals'] = [{'kind': 'preserve_autonomy', 'priority': priority}]
        data['people'][1]['personality']['strategies'] = []
        s = Simulation(Scenario.model_validate(data))
        s.world.tick = 2
        choices.append(sum(decide(s.view('n1'), Random(seed), s.world.scenario.weights, 't000001').selected.action == 'rest' for seed in range(30)))
    assert choices[1] > choices[0]


def test_informed_question_receives_separate_explainable_answer():
    data = load_scenario(FIXTURE).model_dump()
    data['people'] = [p for p in data['people'] if p['id'] in ('n1', 'n2', 'n3')]
    for person in data['people']:
        person['is_player'] = person['id'] == 'n3'
    data['relationships'] = [r for r in data['relationships'] if r['source'] in ('n1','n2','n3') and r['target'] in ('n1','n2','n3')]
    s = Simulation(Scenario.model_validate(data))
    s.world.tick = 2
    s.act('n1', 'invite', 'n2')
    s.respond('p000001', accept=True)
    s.step()
    s.step()
    meeting = next(e for e in s.world.events if e.kind == 'meeting_occurred' and set(e.participants) == {'n1','n2'})
    s.world.tick = 6
    s.share_information('n1', 'n3', meeting.id)
    before = s.world.states['n3'].hurt['n1']
    question = s.act('n3', 'ask', 'n1')
    answer = next(e for e in s.world.events if e.kind == 'question_answered')
    assert answer.causes == (question.id,)
    assert answer.trace_id is not None
    assert s.world.states['n3'].hurt['n1'] < before
