from pathlib import Path
import pytest
from social_sim.authoring import load_scenario
from social_sim.engine import Simulation
from social_sim.models import WorldState
from social_sim.storage import save_snapshot,load_snapshot


def sim():
    s=Simulation(load_scenario(Path('data/fixtures/school.yaml')));s.world.tick=2
    return s


def test_group_plan_supports_partial_acceptance_and_decline():
    s=sim();p=s.invite_group('n1',['n2','n3','n4'])
    s.respond(p.id,True,invitee='n2');s.respond(p.id,False,invitee='n3')
    assert p.replies['n2'].status=='accepted' and p.replies['n3'].status=='declined'
    WorldState.model_validate(s.world.model_dump())


def test_group_meeting_includes_only_confirmed_available_participants():
    s=sim();p=s.invite_group('n1',['n2','n3','n4'])
    s.respond(p.id,True,invitee='n2');s.respond(p.id,False,invitee='n3')
    s.world.tick=p.tick;s.meet(p)
    e=next(e for e in s.world.events if e.kind=='meeting_occurred')
    assert set(e.participants)=={'n1','n2'}
    assert p.replies['n4'].status=='expired'
    WorldState.model_validate(s.world.model_dump())


def test_no_accepted_invitees_means_no_meeting():
    s=sim();p=s.invite_group('n1',['n2','n3'])
    s.respond(p.id,False,invitee='n2');s.respond(p.id,False,invitee='n3')
    s.world.tick=p.tick;s.meet(p)
    assert not any(e.kind=='meeting_occurred' for e in s.world.events)
    assert not s.busy('n1',p.tick)


def test_group_plan_prevents_double_booking_for_every_member():
    s=sim();p=s.invite_group('n1',['n2','n3']);s.respond(p.id,True,invitee='n2')
    q=s.invite_group('n4',['n2','n3'])
    before,rng=s.world.model_dump_json(),s.rng.getstate()
    with pytest.raises(ValueError):s.respond(q.id,True,invitee='n2')
    assert (s.world.model_dump_json(),s.rng.getstate())==(before,rng)
    s.respond(q.id,True,invitee='n3')


def test_group_plan_responses_and_meeting_are_idempotent_or_rejected_atomically():
    s=sim();p=s.invite_group('n1',['n2','n3']);s.respond(p.id,True,invitee='n2')
    before=s.world.model_dump_json()
    with pytest.raises(ValueError):s.respond(p.id,True,invitee='n2')
    assert s.world.model_dump_json()==before
    s.world.tick=p.tick;s.meet(p);before=s.world.model_dump_json()
    with pytest.raises(ValueError):s.meet(p)
    assert s.world.model_dump_json()==before


def test_private_group_invitation_does_not_leak_other_responses():
    s=sim();p=s.invite_group('n1',['n2','n3'])
    reply=s.respond(p.id,False,invitee='n2')
    assert reply.id not in s.view('n3').state.knowledge
    assert all('n2' not in k.participants for k in s.view('n3').state.knowledge.values())


def test_group_meeting_with_unknown_person_preserves_subjective_knowledge():
    from social_sim.cognition import perceive
    s=sim();p=s.invite_group('n1',['n2','n5'])
    s.respond(p.id,True,invitee='n2');s.respond(p.id,True,invitee='n5')
    s.world.tick=p.tick;s.meet(p)
    e=s.world.events[-1];s.world.tick=6;s.share_information('n1','n3',e.id)
    assert s.world.states['n3'].hurt.get('n1',0)>0
    assert 'n3:n5' not in s.world.relationships
    assert p.replies['n5'].history[-1] not in s.world.states['n3'].knowledge


def test_group_plan_save_resume_matches_uninterrupted_run(tmp_path):
    s=sim();p=s.invite_group('n1',['n2','n3','player']);s.respond(p.id,True,invitee='n2')
    save_snapshot(s,tmp_path/'x.json');b=load_snapshot(tmp_path/'x.json')
    for x in(s,b):x.run_days(5)
    assert s.world==b.world and s.rng.getstate()==b.rng.getstate()


def test_single_invitee_api_remains_supported():
    s=sim();s.act('n1','invite','n2');p=s.world.plans[0];s.respond(p.id,True)
    assert p.invitee=='n2' and p.status=='accepted'


def test_group_attendance_does_not_bypass_introduction_consent():
    s=sim();p=s.invite_group('n1',['n2','n5'])
    s.respond(p.id,True,invitee='n2');s.respond(p.id,True,invitee='n5')
    s.world.tick=p.tick;s.meet(p)
    assert 'n2:n5' not in s.world.relationships
    assert 'n5:n2' not in s.world.relationships


def test_controlled_example_script_handles_new_utility_actions(tmp_path):
    import subprocess,sys,json
    result=subprocess.run([sys.executable,'-m','scripts.inspect_cognitive_examples','--output',str(tmp_path/'examples')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert len(json.loads((tmp_path/'examples'/'summary.json').read_text()))==6
