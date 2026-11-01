from random import Random
import pytest
from test_social_interactions import ready
from social_sim.decision import decide


def reported():
    s=ready();root=s.world.states['n3'].interpretations[-1].event_id
    s.interact('n3','discontent','n1',root,defer_reply=True)
    s.world.tick=7  # n1 уже рассказывал о встрече в слоте 6; новая инициатива занимает следующий слот.
    return s,root


def test_apology_requires_known_subject():
    s=ready()
    with pytest.raises(ValueError):s.interact('n1','apology','n3','missing')
    root=s.world.states['n3'].interpretations[-1].event_id
    with pytest.raises(ValueError):s.interact('n1','apology','n3',root)


def test_apology_can_acknowledge_hurt_without_proving_malicious_intent():
    s,root=reported();x=s.interact('n1','apology','n3',root,defer_reply=True)
    assert x.message.content=='regret_hurt'
    assert next(e for e in s.world.events if e.id==x.offered_event).claim is None


@pytest.mark.parametrize('reply',['decline','defer'])
def test_apology_can_be_rejected_or_deferred(reply):
    s,root=reported();x=s.interact('n1','apology','n3',root,defer_reply=True)
    old=s.world.relationships['n3:n1'].model_copy(deep=True)
    s.respond_interaction(x.id,'n3',reply)
    assert x.status==('declined' if reply=='decline' else 'deferred')
    assert s.world.relationships['n3:n1']==old


def test_accepting_apology_does_not_force_full_trust_or_contact():
    s,root=reported()
    situation=next(iter(s.world.states['n3'].situations.values()));situation.distance_active=True
    x=s.interact('n1','apology','n3',root,defer_reply=True);s.respond_interaction(x.id,'n3','accept')
    assert s.world.relationships['n3:n1'].trust<100
    assert s.world.states['n3'].avoiding('n1')
    assert situation.episodes[root].confidence>0


def test_repeated_apology_does_not_farm_trust():
    s,root=reported();x=s.interact('n1','apology','n3',root,defer_reply=True);s.respond_interaction(x.id,'n3','accept')
    s.world.tick+=8
    with pytest.raises(ValueError):s.interact('n1','apology','n3',root)


def test_apology_effect_is_specific_to_its_episode():
    from cognitive_helpers import episode
    s,root=reported();x=s.interact('n1','apology','n3',root,defer_reply=True)
    later=episode(s,tick=10)
    situation=next(iter(s.world.states['n3'].situations.values()))
    before=situation.episodes[later.id].model_copy(deep=True)
    s.respond_interaction(x.id,'n3','accept')
    assert situation.episodes[later.id]==before


def test_repair_memory_changes_later_candidate_scores():
    s,root=reported();x=s.interact('n1','apology','n3',root,defer_reply=True)
    view=s.view('n3');a=decide(view,Random(1),s.world.scenario.weights,'preview')
    view.state.memories=[m for m in view.state.memories if m.topic!='repair_attempt']
    b=decide(view,Random(1),s.world.scenario.weights,'preview')
    assert [c.utility for c in a.candidates]!=[c.utility for c in b.candidates]


def test_forgiveness_does_not_erase_negative_history():
    s,root=reported();old=[m for m in s.world.states['n3'].memories if m.valence=='negative']
    x=s.interact('n1','apology','n3',root,defer_reply=True);s.respond_interaction(x.id,'n3','accept')
    assert all(m in s.world.states['n3'].memories for m in old)
    assert any(m.valence=='positive' for m in s.world.states['n3'].memories)


def test_acknowledging_known_act_does_not_assert_unknown_intent():
    s,root=reported()
    x=s.interact('n1','apology','n3',root,content='acknowledge_act',defer_reply=True)
    assert x.message.content=='acknowledge_act'
    assert 'n1' in s.world.states['n1'].knowledge[root].participants
    assert next(e for e in s.world.events if e.id==x.offered_event).claim is None


@pytest.mark.parametrize('first_reply',['accept','defer'])
def test_later_shared_experience_allows_retry_without_farming_effect(first_reply):
    s,root=reported();x=s.interact('n1','apology','n3',root,defer_reply=True)
    s.respond_interaction(x.id,'n3',first_reply)
    s.world.tick=10;s.act('n1','invite','n3');s.respond(s.world.plans[-1].id,True)
    s.world.tick=11;s.meet(s.world.plans[-1]);s.world.tick=14
    trust=s.world.relationships['n3:n1'].trust
    y=s.interact('n1','apology','n3',root,defer_reply=True);s.respond_interaction(y.id,'n3','accept')
    if first_reply=='accept':assert s.world.relationships['n3:n1'].trust==trust
    else:assert s.world.relationships['n3:n1'].trust>trust
    assert len([e for e in s.world.effects if e.token and ':apology:' in e.token])==1
