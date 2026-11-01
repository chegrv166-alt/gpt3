from pathlib import Path
from itertools import combinations

from social_sim.authoring import load_scenario
from social_sim.engine import Simulation
from social_sim.models import Scenario
from social_sim.groups import detect_groups, update_groups
from social_sim.storage import save_snapshot,load_snapshot


def world():
    data=load_scenario(Path('data/fixtures/school.yaml')).model_dump()
    ids=[p['id'] for p in data['people']]
    data['relationships']=[{'source':a,'target':b,'closeness':60} for a in ids for b in ids if a!=b]
    return Simulation(Scenario.model_validate(data))


def experience(s, groups):
    for i,g in enumerate(groups):
        s.world.tick=2+i
        s.emit('meeting_occurred',tuple(g),'Контролируемый совместный опыт',location='park',organizer=g[0])
    s.world.tick+=1


def confirm(s):
    update_groups(s.world);s.world.tick+=4;update_groups(s.world)


def test_repeated_mutual_interactions_form_detectable_group():
    s=world();experience(s,[('n1','n2','n3')]*2);confirm(s)
    assert [g.members for g in s.world.groups.groups if g.active]==[('n1','n2','n3')]
    assert s.world.groups.history[0].evidence


def test_single_bridge_does_not_merge_distinct_groups():
    s=world();experience(s,[('n1','n2','n3'),('n4','n5','n6')]*2+[('n3','n4')]*2)
    assert set(detect_groups(s.world))=={('n1','n2','n3'),('n4','n5','n6')}


def test_person_can_belong_to_overlapping_groups():
    s=world();experience(s,[('n1','n2','n3'),('n1','n4','n5')]*2)
    assert set(detect_groups(s.world))=={('n1','n2','n3'),('n1','n4','n5')}


def test_brief_pause_does_not_destroy_group():
    s=world();experience(s,[('n1','n2','n3')]*2);confirm(s)
    before=s.world.groups.groups[0].id
    s.world.tick+=4;update_groups(s.world)
    assert s.world.groups.groups[0].active and s.world.groups.groups[0].id==before


def test_group_composition_changes_after_sustained_relevant_history():
    s=world();experience(s,[('n1','n2','n3')]*2);confirm(s);old=s.world.groups.groups[0].id
    s.world.tick+=32
    for _ in range(2):
        s.emit('meeting_occurred',('n1','n2','n4'),'Новый совместный опыт',location='park',organizer='n1')
        s.world.tick+=1
    confirm(s)
    assert any(g.id==old and g.members==('n1','n2','n4') and g.active for g in s.world.groups.groups)
    assert s.world.groups.history[-1].kind=='changed'


def test_group_detection_is_deterministic_and_order_independent():
    a=world();experience(a,[('n1','n2','n3')]*2)
    b=world();experience(b,[('n3','n2','n1')]*2)
    assert detect_groups(a.world)==detect_groups(b.world)
    confirm(a);confirm(b)
    assert [g.model_dump() for g in a.world.groups.groups]==[g.model_dump() for g in b.world.groups.groups]


def test_group_diagnostics_do_not_leak_into_actor_view():
    s=world();experience(s,[('n1','n2','n3')]*2)
    before=s.view('n4');confirm(s);after=s.view('n4')
    assert before.state==after.state
    assert not hasattr(after,'groups') and not hasattr(after,'world')


def test_group_history_survives_save_resume(tmp_path):
    s=world();experience(s,[('n1','n2','n3')]*2);confirm(s)
    save_snapshot(s,tmp_path/'g.json');b=load_snapshot(tmp_path/'g.json')
    s.run_days(2);b.run_days(2)
    assert s.world==b.world and s.rng.getstate()==b.rng.getstate()


def test_unchanged_overlapping_group_keeps_id_when_new_group_appears():
    s=world();experience(s,[('n2','n3','n4')]*2);confirm(s)
    old=s.world.groups.groups[0].id
    for _ in range(2):
        s.emit('meeting_occurred',('n1','n2','n3'),'Новый совместный опыт',location='park',organizer='n1')
        s.world.tick+=1
    confirm(s)
    assert next(g.id for g in s.world.groups.groups if g.members==('n2','n3','n4') and g.active)==old


import pytest
@pytest.mark.parametrize('damage',['group_id','proposed','duplicate','future','counter','history_members','history_future','candidate'])
def test_corrupt_group_snapshot_is_rejected(damage):
    from social_sim.models import WorldState
    s=world();experience(s,[('n1','n2','n3')]*2);confirm(s)
    raw=s.world.model_dump();groups=raw['groups']
    if damage=='group_id':groups['history'][0]['group_id']='missing'
    elif damage=='proposed':groups['groups'][0]['proposed_members']=['ghost']
    elif damage=='duplicate':groups['groups'].append(groups['groups'][0].copy())
    elif damage=='future':groups['last_tick']=99999
    elif damage=='counter':groups['groups'][0]['missing_checks']=-1
    elif damage=='history_members':groups['history'][0]['members']=['ghost']
    elif damage=='candidate':groups['candidates']={'n1:n2:ghost':1}
    else:groups['history'][0]['tick']=99999
    with pytest.raises(ValueError):WorldState.model_validate(raw)
