import json
from random import Random
import pytest
from test_apologies import reported
from test_social_interactions import ready
from cognitive_helpers import episode,school
from social_sim.engine import Simulation
from social_sim.models import Scenario,WorldState,Weights
from social_sim.storage import save_snapshot,load_snapshot
from social_sim.diagnostics import explain


@pytest.mark.parametrize('kind',['support','apology'])
def test_snapshot_during_support_or_apology_resumes_identically(tmp_path,kind):
    s,root=reported();x=s.interact('n1',kind,'n3',root,defer_reply=True)
    save_snapshot(s,tmp_path/'state.json');b=load_snapshot(tmp_path/'state.json')
    for world in(s,b):world.run_days(9)
    save_snapshot(b,tmp_path/'later.json');c=load_snapshot(tmp_path/'later.json')
    for world in(s,c):world.run_days(21)
    assert s.world==c.world and s.rng.getstate()==c.rng.getstate()


def test_save_after_new_episode_during_repair_remains_valid(tmp_path):
    s,root=reported();x=s.interact('n1','apology','n3',root,defer_reply=True)
    episode(s,tick=10)
    save_snapshot(s,tmp_path/'state.json');b=load_snapshot(tmp_path/'state.json')
    assert b.world==s.world and b.world.interactions[x.id].status=='pending'


@pytest.mark.parametrize('kind',['support','apology'])
def test_noop_player_never_accepts_support_or_apology_automatically(kind):
    data=school().world.scenario.model_dump()
    for p in data['people']:p['is_player']=p['id']=='n3'
    s=Simulation(Scenario.model_validate(data));episode(s)
    root=s.world.states['n3'].interpretations[-1].event_id
    s.interact('n3','discontent','n1',root,defer_reply=True)
    s.world.tick=7;x=s.interact('n1',kind,'n3',root)
    assert x.status=='pending' and x.response_event is None
    with pytest.raises(ValueError):s.respond_interaction(x.id,'n3')
    s.respond_interaction(x.id,'n3','defer')
    assert x.status=='deferred'
    response=next(e for e in s.world.events if e.id==x.response_event)
    assert next(t for t in s.world.traces if t.id==response.trace_id).random_draw is None


@pytest.mark.parametrize('response,actor',[('invalid','n3'),('accept','n2')])
def test_invalid_response_preserves_world_and_rng(response,actor):
    s,root=reported();x=s.interact('n1','apology','n3',root,defer_reply=True)
    before,rng=s.world.model_dump_json(),s.rng.getstate()
    with pytest.raises(ValueError):s.respond_interaction(x.id,actor,response)
    assert (s.world.model_dump_json(),s.rng.getstate())==(before,rng)


def test_explain_preserves_pre_response_knowledge_and_decision():
    s,root=reported();x=s.interact('n1','apology','n3',root,defer_reply=True)
    before=explain(s.world,x.offered_event)
    s.respond_interaction(x.id,'n3','accept')
    after=explain(s.world,x.offered_event)
    assert before['decision']==after['decision'] and before['knowledge_at_decision']==after['knowledge_at_decision']
    assert after['interaction']['message']['subject_event']==root
    assert after['interaction_effects']


def test_private_repair_does_not_leak_through_group_diagnostics():
    from social_sim.groups import update_groups
    s,root=reported();x=s.interact('n1','apology','n3',root)
    before=s.view('n2').state.model_dump_json();update_groups(s.world)
    assert s.view('n2').state.model_dump_json()==before
    assert x.offered_event not in s.world.states['n2'].knowledge


def test_new_interactions_preserve_existing_group_plan_invariants(tmp_path):
    from test_support import third_ready
    s,root=third_ready();s.interact('n3','discontent','n1',root,defer_reply=True)
    p=s.invite_group('n4',['n9','player']);s.respond(p.id,True,invitee='n9')
    save_snapshot(s,tmp_path/'state.json');b=load_snapshot(tmp_path/'state.json')
    for x in(s,b):x.run_days(3)
    assert s.world==b.world


@pytest.mark.parametrize('damage',['recipient','subject','offer','reply','status','future','knowledge','effect','budget'])
def test_corrupt_interactions_are_rejected(damage):
    s,root=reported();x=s.interact('n1','apology','n3',root,defer_reply=True);s.respond_interaction(x.id,'n3','accept')
    raw=s.world.model_dump();item=raw['interactions'][x.id]
    if damage=='recipient':item['recipient']='n2'
    elif damage=='subject':item['message']['subject_event']='missing'
    elif damage=='offer':item['offered_event']='missing'
    elif damage=='reply':item['response_event']='missing'
    elif damage=='status':item['status']='declined'
    elif damage=='future':item['created_tick']=9999
    elif damage=='knowledge':raw['states']['n3']['knowledge'][x.offered_event]['social_message']['about']='n2'
    elif damage=='budget':raw['states']['n3']['reply_tick']=9999
    else:raw['effects'][-1]['trust_delta']=50
    with pytest.raises(ValueError):WorldState.model_validate(raw)


def test_short_response_budget_prevents_free_infinite_chain():
    from test_support import third_ready
    s,root=third_ready()
    s.interact('n3','support_request','n4',root,defer_reply=True)
    s.world.tick=7;s.interact('n3','discontent','n1',root,defer_reply=True)
    a=s.interact('n1','apology','n3',root,defer_reply=True)
    b=s.interact('n4','support','n3',root,defer_reply=True)
    s.world.tick=10;s.interact('n3','support_request','n2',root,defer_reply=True)
    s.world.tick=11;c=s.interact('n2','support','n3',root,defer_reply=True)
    s.respond_interaction(a.id,'n3','accept');s.respond_interaction(b.id,'n3','accept')
    assert s.world.states['n3'].reply_count==2
    before,rng=s.world.model_dump_json(),s.rng.getstate()
    with pytest.raises(ValueError):s.respond_interaction(c.id,'n3','accept')
    assert (s.world.model_dump_json(),s.rng.getstate())==(before,rng)


def test_interaction_policies_obey_zero_weights():
    from social_sim.interactions import response_candidates
    from social_sim.decision import choose
    s,root=reported();message=s.interact('n1','apology','n3',root,defer_reply=True).message
    t=choose(s.view('n3'),response_candidates(s.view('n3'),message,'n1'),[],Random(0),
             Weights(need=0,personality=0,goal=0,relationship=0,emotion=0),'preview')
    assert all(v==0 for c in t.candidates for k,v in c.components.items() if k!='effort')


def test_pending_apology_stays_out_of_candidates_after_new_shared_experience():
    from social_sim.interactions import initiative_candidates
    data=school().world.scenario.model_dump()
    for p in data['people']:p['is_player']=p['id']=='n3'
    s=Simulation(Scenario.model_validate(data));episode(s)
    root=s.world.states['n3'].interpretations[-1].event_id
    s.interact('n3','discontent','n1',root,defer_reply=True)
    s.world.tick=7;pending=s.interact('n1','apology','n3',root)
    s.world.tick=10;s.act('n1','invite','n3');s.respond(s.world.plans[-1].id,True)
    s.world.tick=11;s.meet(s.world.plans[-1]);s.world.tick=14
    assert pending.status=='pending'
    candidates=initiative_candidates(s.view('n1'))
    assert not any(c.action=='apology' and c.social_message.subject_event==root for c in candidates)


@pytest.mark.parametrize('damage',['direction','token','response_trace'])
def test_repair_effect_direction_and_response_trace_are_validated(damage):
    s,root=reported();x=s.interact('n1','apology','n3',root,defer_reply=True);s.respond_interaction(x.id,'n3','accept')
    raw=s.world.model_dump()
    if damage=='direction':
        raw['effects'][-1]['actor'],raw['effects'][-1]['target']=raw['effects'][-1]['target'],raw['effects'][-1]['actor']
    elif damage=='token':raw['effects'][-1]['token']='unrelated'
    else:
        raw['interactions'][x.id]['status']='declined'
        raw['events'][-1]['social_response']='decline'
        raw['effects']=[]
    with pytest.raises(ValueError):WorldState.model_validate(raw)


def test_repair_fixture_autonomy_and_30_day_resume(tmp_path):
    from pathlib import Path
    from social_sim.authoring import load_scenario
    fixture=load_scenario(Path('data/fixtures/relationship_repair.yaml'))
    a,b=Simulation(fixture,42),Simulation(fixture,42)
    a.run_days(30);b.run_days(9)
    save_snapshot(b,tmp_path/'state.json');b=load_snapshot(tmp_path/'state.json');b.run_days(21)
    assert a.world==b.world and a.rng.getstate()==b.rng.getstate()
    chosen={t.selected.action for t in a.world.traces if t.random_draw is not None}
    assert {'discontent','support','apology'} <= chosen
    assert all(t.selected.action!='apology' or t.selected.social_message is not None for t in a.world.traces)
